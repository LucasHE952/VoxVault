"""Measure Voxtral inference speed across delay / length / warmup configs.

Records a 10s speech clip once, then runs model.generate() against slices of
it under varied transcription_delay_ms values. Prints 3-trial medians so we
can see what actually moves the needle on realtime throughput.

Usage:
    .venv/bin/python scripts/benchmark_inference.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audio.capture import AudioCapture
from config.defaults import MODEL_LOCAL_DIR, SAMPLE_RATE
from transcription.model import VoxtralModel

LENGTHS_SEC = (5.0, 8.0, 10.0)
DELAY_MS_VALUES = (160, 80, 40)
TRIALS = 3


def record_clip(seconds: float) -> np.ndarray:
    """Record `seconds` of mono 16kHz audio from the default mic."""
    print(f"\nRecording {seconds:.0f}s — start speaking now.")
    chunks: list[np.ndarray] = []
    capture = AudioCapture(sample_rate=SAMPLE_RATE)
    capture.start()
    try:
        deadline = time.perf_counter() + seconds
        for chunk in capture.stream():
            chunks.append(chunk)
            elapsed = seconds - (deadline - time.perf_counter())
            bar = "#" * int(elapsed * 10 / seconds)
            print(f"\r[{bar:<10}] {elapsed:.1f}s", end="", flush=True)
            if time.perf_counter() >= deadline:
                break
    finally:
        capture.stop()
    print()
    return np.concatenate(chunks)


def _one_call(raw_model, clip: np.ndarray, delay_ms: int) -> tuple[float, str]:
    t0 = time.perf_counter()
    result = raw_model.generate(
        clip,
        temperature=0.0,
        stream=False,
        transcription_delay_ms=delay_ms,
    )
    elapsed = time.perf_counter() - t0
    text = getattr(result, "text", str(result)) or ""
    return elapsed, text.strip()


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    print("Loading Voxtral…")
    model = VoxtralModel(model_path=MODEL_LOCAL_DIR)
    model.load()
    raw = model._model  # benchmark talks directly to mlx-audio
    print("Model ready.")

    input("Press Enter, then read ~10s of continuous speech after the countdown.")
    audio = record_clip(seconds=10.5)

    # Use tight slices of the recorded clip for each length
    clips = {f"{sec:>4.1f}s": audio[: int(sec * SAMPLE_RATE)] for sec in LENGTHS_SEC}

    # ── Real speech warmup (exercises decoder kernels unlike quiet noise) ────
    # The production warmup is 0.5s of amplitude-0.01 white noise — likely
    # short-circuits inside the model before the decoder runs any tokens.
    print("\nWarming up with 2s of recorded speech…")
    warmup_clip = audio[: 2 * SAMPLE_RATE]
    for i in range(2):
        t = time.perf_counter()
        raw.generate(warmup_clip, temperature=0.0, stream=False, transcription_delay_ms=160)
        print(f"  warmup {i + 1}: {time.perf_counter() - t:.2f}s")

    # ── Benchmark matrix ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'length':>8}  {'delay_ms':>9}  {'trial':>6}  {'wall_s':>8}  {'xRT':>6}  text")
    print("-" * 78)

    results: dict[tuple[str, int], list[float]] = {}
    for name, clip in clips.items():
        duration = len(clip) / SAMPLE_RATE
        for delay_ms in DELAY_MS_VALUES:
            times: list[float] = []
            last_text = ""
            for trial in range(TRIALS):
                try:
                    elapsed, text = _one_call(raw, clip, delay_ms)
                except Exception as exc:  # noqa: BLE001
                    print(f"{name:>8}  {delay_ms:>9}  {trial + 1:>6}  ERROR: {exc}")
                    continue
                times.append(elapsed)
                last_text = text
                print(
                    f"{name:>8}  {delay_ms:>9}  {trial + 1:>6}  "
                    f"{elapsed:>7.2f}s  {elapsed / duration:>5.2f}x  {text[:40]!r}"
                )
            results[(name, delay_ms)] = times
            del last_text

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("MEDIAN ACROSS TRIALS")
    print(f"{'length':>8}  {'delay_ms':>9}  {'median_s':>10}  {'xRT':>6}")
    for (name, delay_ms), times in results.items():
        if not times:
            print(f"{name:>8}  {delay_ms:>9}  {'N/A':>10}")
            continue
        median = sorted(times)[len(times) // 2]
        duration = len(clips[name]) / SAMPLE_RATE
        print(f"{name:>8}  {delay_ms:>9}  {median:>9.2f}s  {median / duration:>5.2f}x")


if __name__ == "__main__":
    main()
