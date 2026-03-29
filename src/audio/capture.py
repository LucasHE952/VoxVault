"""Microphone audio capture for VoxVault.

Uses sounddevice (PortAudio bindings) for low-latency, reliable audio capture.

Why sounddevice over PyAudio:
- Cleaner Python API with numpy integration (no manual buffer casting)
- Actively maintained; PyAudio's last release was 2017
- No complex C extension build steps on macOS
- Built-in support for non-blocking callback streams
"""

import logging
import queue
import threading
from collections.abc import Generator
from typing import Optional

import numpy as np
import sounddevice as sd

from config.defaults import CHANNELS, CHUNK_SIZE, SAMPLE_RATE

logger = logging.getLogger(__name__)


class AudioCapture:
    """Captures microphone audio as a continuous stream of numpy chunks.

    Audio is collected in a background callback thread and placed on a thread-safe
    queue. Consumers call ``stream()`` to iterate over chunks.

    Args:
        sample_rate: Audio sample rate in Hz. Must match the model's expectation.
        chunk_size: Number of samples per chunk delivered to the consumer.
        device: PortAudio device index or name. None = system default.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_size: int = CHUNK_SIZE,
        device: Optional[int | str] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device = device

        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

    # ── Public API ───────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open and start the microphone input stream.

        Raises:
            sd.PortAudioError: If the microphone cannot be opened (e.g. permission denied).
            RuntimeError: If the stream is already running.
        """
        with self._lock:
            if self._stream is not None and self._stream.active:
                raise RuntimeError("AudioCapture is already running")

            logger.debug(
                "Opening microphone: rate=%d Hz, chunk=%d samples, device=%s",
                self.sample_rate,
                self.chunk_size,
                self.device or "default",
            )
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=CHANNELS,
                dtype="float32",
                blocksize=self.chunk_size,
                device=self.device,
                callback=self._audio_callback,
            )
            self._stream.start()
            logger.info("Microphone capture started")

    def stop(self) -> None:
        """Stop and close the microphone input stream."""
        with self._lock:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
                logger.info("Microphone capture stopped")
        # Unblock any stream() generator waiting on queue.get()
        self._queue.put_nowait(None)

    def stream(self) -> Generator[np.ndarray, None, None]:
        """Yield audio chunks as float32 numpy arrays of shape (chunk_size,).

        This is a blocking generator. It will block between chunks until the
        next chunk arrives from the microphone callback. Call ``stop()`` from
        another thread to unblock and end iteration (sentinel None is enqueued).

        Yields:
            numpy.ndarray: 1-D float32 array of ``chunk_size`` samples,
                normalised to the range [-1.0, 1.0].
        """
        while True:
            chunk = self._queue.get()
            if chunk is None:
                break
            yield chunk

    def drain(self) -> None:
        """Discard all queued audio chunks (e.g. between recording sessions)."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    # ── Internal ─────────────────────────────────────────────────────────────

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """PortAudio callback — runs in a high-priority audio thread.

        Args:
            indata: Raw audio buffer, shape (frames, channels), float32.
            frames: Number of frames in this callback.
            time_info: Timing information from PortAudio (unused).
            status: Overflow/underflow flags.
        """
        if status:
            logger.warning("Audio stream status: %s", status)

        # Flatten to 1-D mono and copy (indata is a view into PortAudio's buffer)
        self._queue.put_nowait(indata[:, 0].copy())

    # ── Context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
        # Unblock any waiting stream() generator
        self._queue.put_nowait(None)


def list_input_devices() -> list[dict]:
    """Return a list of available audio input devices.

    Returns:
        List of dicts with keys: index, name, max_input_channels, default_samplerate.
    """
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append({
                "index": idx,
                "name": dev["name"],
                "max_input_channels": dev["max_input_channels"],
                "default_samplerate": dev["default_samplerate"],
            })
    return devices
