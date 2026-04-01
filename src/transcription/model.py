"""Voxtral Realtime transcription via mlx-audio.

Model:   mistralai/Voxtral-Mini-4B-Realtime-2602
Weights: mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit (~2–2.5GB)
Library: mlx-audio (pip install mlx-audio[stt])

mlx-audio provides a unified STT loader (mlx_audio.stt.load) that detects the
model architecture and returns a model with a .generate() method. For Voxtral
Realtime, generate() accepts numpy audio, runs the causal encoder + LM decoder
on MLX (Metal GPU / Neural Engine), and returns an STTOutput with .text.

Streaming (Phase 2): pass stream=True to generate() to get a generator that
yields text deltas token-by-token as decoding progresses.
"""

import logging
import time
from collections.abc import Generator
from pathlib import Path
from typing import Optional

import numpy as np

from config.defaults import MODEL_LOCAL_DIR, MODEL_REPO_ID, SAMPLE_RATE

logger = logging.getLogger(__name__)

# Voxtral Realtime's mlx-audio generate() silently returns empty text for
# audio longer than ~12s in batch mode (stream=False) — the observed failure
# threshold is lower than the theoretical 15s limit, likely content-dependent.
# 10s gives a safe margin; do not increase without thorough testing.
_MAX_SEGMENT_SECONDS: float = 10.0
_MAX_SEGMENT_SAMPLES: int = int(_MAX_SEGMENT_SECONDS * SAMPLE_RATE)


class VoxtralModel:
    """Wraps Voxtral-Mini-4B-Realtime via mlx-audio for MLX-native transcription.

    The model is loaded lazily. Call ``load()`` once at startup — it downloads
    weights from HuggingFace on first run (~2.5GB) and caches them locally.
    Subsequent loads are fast (weights already on disk, MLX kernels cached).

    Args:
        model_path: HuggingFace repo ID or local path to MLX model weights.
        language: Default BCP-47 language code. Can be overridden per call.
        transcription_delay_ms: Audio buffered before decoding starts (ms).
            160ms is optimised for batch mode (stream=False) where the full audio
            is already available before generate() is called — the model does not
            need to wait for more audio to arrive, so the 480ms "realtime" default
            only adds unnecessary latency. 160ms preserves enough acoustic context
            for the encoder without stalling the decoder.
    """

    def __init__(
        self,
        model_path: str | Path = MODEL_REPO_ID,
        language: str = "en",
        transcription_delay_ms: int = 160,
    ) -> None:
        self.model_path = str(model_path)
        self.language = language
        self.transcription_delay_ms = transcription_delay_ms
        self._model: Optional[object] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Download (first run) and load Voxtral weights into MLX.

        On first run this downloads ~2.5GB from HuggingFace and compiles Metal
        kernels. Subsequent calls return immediately (model already loaded).

        Raises:
            ImportError: If mlx-audio is not installed.
        """
        if self._model is not None:
            return

        try:
            from mlx_audio.stt import load as stt_load
        except ImportError as exc:
            raise ImportError(
                "mlx-audio is not installed. Run: pip install 'mlx-audio[stt]'"
            ) from exc

        logger.info("Loading Voxtral Realtime from %s…", self.model_path)
        t0 = time.perf_counter()

        # stt_load auto-detects model type from config.json and returns the
        # appropriate mlx_audio.stt model (VoxtralRealtime in this case).
        self._model = stt_load(self.model_path)

        elapsed = time.perf_counter() - t0
        logger.info("Voxtral loaded in %.1fs", elapsed)

        # Pin model weights in GPU-accessible (wired) memory for the session.
        # Without this, Metal may page weights to system RAM between inference
        # calls, causing each decoder step to re-fetch weights over the memory
        # bus and dropping throughput from ~30 tok/s to ~9 tok/s.
        # mlx-audio's own generate_transcription() does this via wired_limit();
        # we set it once globally since the model stays loaded.
        try:
            import mlx.core as mx
            if mx.metal.is_available():
                max_rec = mx.device_info()["max_recommended_working_set_size"]
                mx.set_wired_limit(max_rec)
                logger.info(
                    "Metal wired limit set to %.1f GB",
                    max_rec / 1024 ** 3,
                )
        except Exception as exc:
            logger.warning("Could not set Metal wired limit: %s", exc)

    def is_loaded(self) -> bool:
        """Return True if model weights are in memory."""
        return self._model is not None

    # ── Transcription ─────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
    ) -> str:
        """Transcribe a complete audio buffer to text.

        Args:
            audio: 1-D float32 numpy array at 16kHz.
            language: BCP-47 language code. Defaults to self.language.

        Returns:
            Transcribed text, stripped of leading/trailing whitespace.

        Raises:
            RuntimeError: If ``load()`` has not been called.
        """
        self._assert_loaded()

        duration = len(audio) / SAMPLE_RATE
        t0 = time.perf_counter()

        if len(audio) > _MAX_SEGMENT_SAMPLES:
            n_segments = -(-len(audio) // _MAX_SEGMENT_SAMPLES)  # ceiling div
            logger.info(
                "Transcribing %.1fs of audio → %d segment(s) of ≤%.0fs",
                duration, n_segments, _MAX_SEGMENT_SECONDS,
            )
            text = self._transcribe_segmented(audio, language)
        else:
            logger.info("Transcribing %.1fs of audio", duration)
            text = self._transcribe_chunk(audio)

        elapsed = time.perf_counter() - t0
        logger.info("Transcription done in %.2fs (%.1fx realtime): %r", elapsed, elapsed / duration, text[:80])
        return text

    def _transcribe_chunk(self, audio: np.ndarray) -> str:
        """Transcribe a single audio segment (must be ≤ _MAX_SEGMENT_SECONDS).

        Args:
            audio: 1-D float32 numpy array at 16kHz.

        Returns:
            Stripped transcription text, or empty string.
        """
        # generate() with stream=False returns an STTOutput dataclass.
        # temperature=0.0 → greedy decoding (deterministic, best for dictation).
        result = self._model.generate(
            audio,
            temperature=0.0,
            stream=False,
            transcription_delay_ms=self.transcription_delay_ms,
        )
        return result.text.strip()

    def _transcribe_segmented(
        self, audio: np.ndarray, language: Optional[str] = None
    ) -> str:
        """Transcribe long audio by splitting into segments and joining results.

        Voxtral Realtime's mlx-audio generate() silently returns empty text for
        audio longer than ~15s in batch mode. We split at fixed sample boundaries
        and join segment transcriptions with a space.

        Args:
            audio: 1-D float32 numpy array at 16kHz, longer than _MAX_SEGMENT_SECONDS.
            language: Unused — kept for signature consistency.

        Returns:
            Concatenated transcription text across all segments.
        """
        segments = [
            audio[i : i + _MAX_SEGMENT_SAMPLES]
            for i in range(0, len(audio), _MAX_SEGMENT_SAMPLES)
        ]
        logger.debug(
            "Audio %.1fs → %d segments of ≤%.0fs each",
            len(audio) / SAMPLE_RATE,
            len(segments),
            _MAX_SEGMENT_SECONDS,
        )

        parts: list[str] = []
        for idx, segment in enumerate(segments):
            t_seg = time.perf_counter()
            text = self._transcribe_chunk(segment)
            seg_elapsed = time.perf_counter() - t_seg
            logger.info(
                "Segment %d/%d (%.1fs audio): %.2fs → %r",
                idx + 1, len(segments), len(segment) / SAMPLE_RATE, seg_elapsed, text[:60],
            )
            if text:
                parts.append(text)

        return " ".join(parts)

    def transcribe_stream(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Transcribe audio and yield text deltas as they are decoded.

        Used in Phase 2 for low-latency text injection — each token is
        injected into the target application as soon as it arrives.

        Args:
            audio: 1-D float32 numpy array at 16kHz.
            language: BCP-47 language code. Defaults to self.language.

        Yields:
            str: Text delta strings as the model decodes them.

        Raises:
            RuntimeError: If ``load()`` has not been called.
        """
        self._assert_loaded()

        # generate() with stream=True returns a generator of text delta strings.
        yield from self._model.generate(
            audio,
            temperature=0.0,
            stream=True,
            transcription_delay_ms=self.transcription_delay_ms,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _assert_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "Model is not loaded. Call VoxtralModel.load() first."
            )
