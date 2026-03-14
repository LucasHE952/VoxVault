"""Entry point for ondevice-dictation.

Phase 1 goal: confirm the model loads and produces accurate output.
Run with: python src/main.py --phase1

Full push-to-talk dictation (Phase 2+) will be added in subsequent phases.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure src/ is on the path when running as `python src/main.py`
sys.path.insert(0, str(Path(__file__).parent))

from config.defaults import APP_NAME, APP_VERSION, LOG_FILE, SAMPLE_RATE
from config.settings import Settings


def _configure_logging(verbose: bool = False) -> None:
    """Set up logging to both stderr and the log file."""
    level = logging.DEBUG if verbose else logging.INFO
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def phase1_smoke_test(settings: Settings, duration_seconds: float = 5.0) -> None:
    """Phase 1: Record audio from the mic, transcribe with Voxtral, print result.

    This is the minimal end-to-end test. No hotkey, no injection — just the
    audio → model → text pipeline.

    Args:
        settings: User settings instance.
        duration_seconds: How many seconds of audio to record.
    """
    logger = logging.getLogger(__name__)

    # ── Lazy imports (heavy; keep startup fast) ───────────────────────────────
    from audio.capture import AudioCapture
    from transcription.model import VoxtralModel

    model = VoxtralModel(language=settings["language"])

    print(f"\n{APP_NAME} v{APP_VERSION} — Phase 1 smoke test")
    print("=" * 50)

    # Load model (downloads ~2.5GB on first run via HuggingFace, then cached)
    print("Loading Voxtral Realtime (downloads ~2.5GB on first run, then cached) …")
    try:
        model.load()
    except ImportError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Model ready. Recording {duration_seconds:.0f}s of audio …\n")
    print("Speak now!")
    print("-" * 50)

    # Record audio
    chunks: list[np.ndarray] = []
    capture = AudioCapture(sample_rate=SAMPLE_RATE)

    try:
        capture.start()
        deadline = time.perf_counter() + duration_seconds
        for chunk in capture.stream():
            chunks.append(chunk)
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            # Progress indicator
            elapsed = duration_seconds - remaining
            bar = "#" * int(elapsed * 10 / duration_seconds)
            print(f"\r[{bar:<10}] {elapsed:.1f}s / {duration_seconds:.0f}s", end="", flush=True)
    finally:
        capture.stop()

    print("\n" + "-" * 50)
    print("Recording complete. Transcribing …\n")

    if not chunks:
        print("No audio captured. Check your microphone.", file=sys.stderr)
        sys.exit(1)

    audio = np.concatenate(chunks)
    logger.info("Captured %.1fs of audio (%d samples)", len(audio) / SAMPLE_RATE, len(audio))

    # Transcribe
    text = model.transcribe(audio, language=settings["language"])

    print("Transcription:")
    print("=" * 50)
    print(text or "(empty — no speech detected)")
    print("=" * 50)


def main() -> None:
    """Parse arguments and dispatch to the appropriate mode."""
    parser = argparse.ArgumentParser(
        prog=APP_NAME,
        description="On-device dictation powered by Voxtral Realtime on Apple Silicon.",
    )
    parser.add_argument(
        "--phase1",
        action="store_true",
        help="Run Phase 1 smoke test: record audio and print transcription.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Recording duration for --phase1 (default: 5s).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{APP_NAME} {APP_VERSION}",
    )
    args = parser.parse_args()

    _configure_logging(verbose=args.verbose)
    settings = Settings.load()

    if args.phase1:
        phase1_smoke_test(settings, duration_seconds=args.duration)
    else:
        # Phase 2+ entry point placeholder
        parser.print_help()
        print(
            "\nNote: Full push-to-talk mode (Phase 2) is not yet implemented.\n"
            "Run with --phase1 to test the transcription pipeline."
        )


if __name__ == "__main__":
    main()
