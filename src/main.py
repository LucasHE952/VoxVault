"""Entry point for VoxVault.

Phase 1 goal: confirm the model loads and produces accurate output.
Run with: python src/main.py --phase1

Full push-to-talk dictation with menu bar UI (Phase 3):
Run with: python src/main.py
"""

import argparse
import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure src/ is on the path when running as `python src/main.py` (no-op in bundle)
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).parent))

from config.defaults import APP_NAME, APP_VERSION, LOG_FILE, MODEL_LOCAL_DIR, SAMPLE_RATE
from config.settings import Settings


def _show_alert(title: str, message: str) -> None:
    """Show a native NSAlert dialog (visible even when running as a .app bundle)."""
    try:
        from AppKit import NSAlert, NSWarningAlertStyle
        alert = NSAlert.alloc().init()
        alert.setMessageText_(title)
        alert.setInformativeText_(message)
        alert.setAlertStyle_(NSWarningAlertStyle)
        alert.runModal()
    except Exception:
        # Fallback for environments where AppKit isn't available
        print(f"\n{title}\n{message}", file=sys.stderr)


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

    model = VoxtralModel(
        model_path=MODEL_LOCAL_DIR,
        language=settings["language"],
    )

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


def run_dictation_app(settings: Settings) -> None:
    """Phase 3: Full dictation loop with menu bar UI.

    Architecture:
      - rumps.App owns the main thread / NSRunLoop
      - Transcription runs in a daemon thread
      - ui_queue bridges background state changes to the main-thread overlay

    Args:
        settings: User settings instance.
    """
    logger = logging.getLogger(__name__)

    from audio.capture import AudioCapture
    from audio.vad import VoiceActivityDetector
    from hotkey.listener import HotkeyListener
    from injection.text_injector import TextInjector, check_accessibility_permission
    from transcription.model import VoxtralModel
    from transcription.streaming import AudioBuffer
    from ui.menu_bar import DictationMenuBarApp, UIEvent

    # ── Accessibility permission check ────────────────────────────────────────
    if not check_accessibility_permission():
        print(
            "\nERROR: Accessibility permission is required for text injection.\n"
            "\nTo grant access:\n"
            "  1. Open System Settings -> Privacy & Security -> Accessibility\n"
            "  2. Enable 'Terminal' (or whichever app you launch this from)\n"
            "  3. Re-run this app\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Load models (blocking — do before rumps takes the main thread) ────────
    model = VoxtralModel(model_path=MODEL_LOCAL_DIR, language=settings["language"])
    vad = VoiceActivityDetector(sensitivity=settings["vad_sensitivity"])
    injector = TextInjector()

    print(f"\n{APP_NAME} v{APP_VERSION} — dictation mode")
    print("=" * 50)
    print("Loading Voxtral Realtime...")
    try:
        model.load()
    except Exception as exc:
        _show_alert(
            "Model not found",
            f"Could not load Voxtral weights from:\n{MODEL_LOCAL_DIR}\n\n"
            "Run setup.sh to download the model (~2.9 GB).\n\n"
            f"Error: {exc}",
        )
        sys.exit(1)
    print("Loading VAD...")
    vad.load()

    # Pre-compile MLX Metal kernels with a speech-like dummy pass so the first
    # real transcription doesn't pay the compilation cost (can add 2-3s otherwise).
    # White noise (non-zero signal) exercises both the encoder and the LM decoder's
    # token generation kernels, unlike silence which short-circuits the decoder path.
    print("Warming up inference engine...")
    rng = np.random.default_rng(seed=0)
    _warmup = rng.standard_normal(int(SAMPLE_RATE * 0.5)).astype(np.float32) * 0.01
    model.transcribe(_warmup)
    print("Ready. Starting menu bar app…\n")

    # ── Shared state ──────────────────────────────────────────────────────────
    buffer = AudioBuffer()
    _recording = threading.Event()
    _transcribing = threading.Event()
    _cancel = threading.Event()
    _stop_event = threading.Event()
    _transcribe_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()
    ui_queue: queue.Queue[UIEvent] = queue.Queue()
    # Single-element list so the audio thread can write amplitude and the
    # main thread can read it without locks (GIL makes float writes atomic).
    _amplitude: list[float] = [0.0]
    # MLX Metal command buffers are not thread-safe. Both the transcription
    # thread (model.transcribe) and the audio-collector thread (vad.is_speech)
    # dispatch Metal work. Without serialisation, concurrent calls crash with:
    #   "encodeSignalEvent:value: with uncommitted encoder"
    # This lock ensures only one MLX caller runs Metal work at a time.
    _mlx_lock = threading.Lock()

    # ── Hotkey callbacks (run on main thread via CGEventTap) ─────────────────
    def on_press() -> None:
        _cancel.clear()
        buffer.clear()
        vad.reset_state()
        capture.drain()
        _recording.set()
        ui_queue.put(UIEvent("recording"))

    def on_release() -> None:
        _recording.clear()
        if _cancel.is_set():
            # Escape was pressed while hotkey was held — discard everything.
            buffer.clear()
            _cancel.clear()
            ui_queue.put(UIEvent("idle"))
            return
        audio = buffer.flush()
        if audio is not None and len(audio) > 0:
            ui_queue.put(UIEvent("transcribing"))
            _transcribe_queue.put(audio)
        else:
            ui_queue.put(UIEvent("idle"))

    # ── Escape cancels recording or pending transcription ─────────────────────
    def on_escape() -> None:
        if _recording.is_set() or _transcribing.is_set():
            logger.debug("Cancel requested via Escape")
            _cancel.set()
            _recording.clear()
            buffer.clear()
            ui_queue.put(UIEvent("idle"))

    # ── Audio collection thread ───────────────────────────────────────────────
    capture = AudioCapture(sample_rate=SAMPLE_RATE)

    def _collect_audio() -> None:
        try:
            for chunk in capture.stream():
                if _stop_event.is_set():
                    break
                if _recording.is_set():
                    # Exponential moving average — smooths out per-chunk spikes
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    _amplitude[0] = _amplitude[0] * 0.4 + rms * 0.6
                    with _mlx_lock:
                        speech = vad.is_speech(chunk)
                    if speech:
                        buffer.append_speech(chunk)
                    else:
                        buffer.append_silence(chunk)
                else:
                    _amplitude[0] = 0.0
        except Exception:
            logger.exception("Audio-collector thread crashed — microphone may have stopped")

    # Restart capture stream if AUHAL initialization fails (err=-50 on macOS).
    # This can happen when the device's native sample rate conflicts with the
    # requested 16kHz; a brief pause and retry lets AUHAL settle.
    import time as _time
    for _attempt in range(3):
        try:
            capture.start()
            break
        except Exception as exc:
            if _attempt < 2:
                logger.warning("Audio capture start failed (attempt %d): %s — retrying", _attempt + 1, exc)
                _time.sleep(0.5)
            else:
                logger.error("Audio capture failed after 3 attempts: %s", exc)
                raise

    collect_thread = threading.Thread(
        target=_collect_audio, daemon=True, name="audio-collector"
    )
    collect_thread.start()

    # ── Hotkey listener ───────────────────────────────────────────────────────
    hotkey = HotkeyListener(
        hotkey=settings["hotkey"],
        on_press=on_press,
        on_release=on_release,
        on_escape=on_escape,
    )
    hotkey.start()

    # ── Transcription loop (daemon thread — main thread belongs to rumps) ─────
    def _transcription_loop() -> None:
        while not _stop_event.is_set():
            try:
                audio = _transcribe_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if audio is None:
                break
            logger.info("Transcribing %.1fs of audio…", len(audio) / SAMPLE_RATE)
            _transcribing.set()
            try:
                # Voxtral Realtime is a corrective streaming model: it emits a
                # draft transcription first, then re-emits corrected tokens as
                # context improves. Streaming injection would type both versions.
                # We use stream=False to wait for the final corrected output only.
                # _mlx_lock serialises Metal work between this thread and the
                # VAD calls in the audio-collector thread.
                with _mlx_lock:
                    text = model.transcribe(audio, language=settings["language"])
                if _cancel.is_set():
                    logger.debug("Transcription result discarded (cancelled)")
                elif text:
                    logger.info("Transcription: %r", text)
                    injector.type(text)
                else:
                    logger.debug("Empty transcription — no speech detected")
            except PermissionError as exc:
                logger.error("%s", exc)
            except Exception as exc:
                logger.error("Transcription failed: %s", exc, exc_info=True)
            finally:
                _transcribing.clear()
                cancelled = _cancel.is_set()
                _cancel.clear()
                ui_queue.put(UIEvent("idle" if cancelled else "done"))

    transcription_thread = threading.Thread(
        target=_transcription_loop, daemon=True, name="transcription-loop"
    )
    transcription_thread.start()

    # ── Stop / restart callbacks passed to the menu bar app ──────────────────

    def _stop_all() -> None:
        """Called by the menu bar Quit action."""
        _stop_event.set()
        hotkey.stop()
        capture.stop()
        _transcribe_queue.put(None)  # unblock transcription thread

    def _restart_hotkey(new_key: str) -> None:
        """Called after the user selects a new hotkey in Settings.

        Uses set_hotkey() to swap the key in-place rather than stopping
        and recreating the listener. Recreating would start a new pynput
        thread that calls TSMGetInputSourceProperty off the main queue,
        which macOS 15+ kills with SIGTRAP.
        """
        hotkey.set_hotkey(new_key)

    # ── Menu bar app (takes over the main thread) ─────────────────────────────
    app = DictationMenuBarApp(
        settings=settings,
        ui_queue=ui_queue,
        amplitude_ref=_amplitude,
        stop_callback=_stop_all,
        hotkey_restart_callback=_restart_hotkey,
    )
    # Provide the live VAD so settings can update its threshold immediately
    # (must be done before app.run() since set_vad defers to AppKit startup)
    app._vad_for_settings = vad  # picked up in _on_startup

    app.run()  # blocks until Quit


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
        run_dictation_app(settings)


if __name__ == "__main__":
    main()
