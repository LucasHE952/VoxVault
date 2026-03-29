"""Menu bar app for VoxVault (Phase 3).

DictationMenuBarApp subclasses rumps.App to:
- Show a menu bar icon that reflects idle / recording / transcribing state
- Drain the ui_queue every 50ms on the main thread to update the overlay
- Provide a Preferences menu item and a Quit action

Threading model:
    rumps.App.run() owns the main thread / NSRunLoop. All AppKit calls
    (overlay, icon updates) happen inside the 50ms rumps.Timer callback,
    which runs on the main thread.

Icon approach:
    SF Symbols are loaded via NSImage.imageWithSystemSymbolName_ and set
    directly on the NSStatusItem's button so the system handles dark/light
    mode tinting automatically.
"""

import logging
import queue
from collections.abc import Callable
from typing import Optional, NamedTuple

import rumps

from config.defaults import APP_NAME, APP_VERSION

logger = logging.getLogger(__name__)

# SF Symbol names for each state
_SYMBOL_IDLE = "mic"
_SYMBOL_RECORDING = "mic.fill"
_SYMBOL_PROCESSING = "waveform"


class UIEvent(NamedTuple):
    """State transition event produced by background threads."""
    state: str  # "recording" | "transcribing" | "done" | "idle"


class DictationMenuBarApp(rumps.App):
    """Rumps menu bar application that coordinates the overlay and settings.

    Args:
        settings: Loaded user settings instance.
        ui_queue: Thread-safe queue of UIEvent from background threads.
        stop_callback: Called when the user clicks Quit.
        hotkey_restart_callback: Called with new key name when hotkey changes.
    """

    def __init__(
        self,
        settings,
        ui_queue: "queue.Queue[UIEvent]",
        amplitude_ref: "list[float] | None" = None,
        stop_callback: Optional[Callable[[], None]] = None,
        hotkey_restart_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(
            name=APP_NAME,
            title=None,
            icon=None,
            quit_button=None,  # We add our own Quit with a clean shutdown
        )

        self._settings = settings
        self._ui_queue = ui_queue
        self._amplitude_ref = amplitude_ref or [0.0]
        self._stop_callback = stop_callback
        self._hotkey_restart_callback = hotkey_restart_callback

        self._overlay = None   # created on main thread in _on_startup
        self._settings_win = None  # ditto

        # ── Menu ─────────────────────────────────────────────────────────────
        self._status_item_label = rumps.MenuItem("Status: Idle")
        self._status_item_label.set_callback(None)  # non-clickable

        self.menu = [
            self._status_item_label,
            None,  # separator
            rumps.MenuItem("Preferences…", callback=self._open_preferences),
            None,
            rumps.MenuItem("Quit", callback=self._quit),
        ]

        # ── Timers ───────────────────────────────────────────────────────────
        # One-shot: initialise AppKit objects after the run loop is live.
        self._startup_timer = rumps.Timer(self._on_startup, 0)
        self._startup_timer.start()

        # Recurring: drain ui_queue and advance overlay animation.
        self._ui_timer = rumps.Timer(self._drain_ui_queue, 0.05)
        self._ui_timer.start()

    # ── Startup ───────────────────────────────────────────────────────────────

    def _on_startup(self, sender) -> None:
        """One-shot timer: runs on main thread after NSRunLoop is live."""
        sender.stop()

        # Set SF Symbol icon (requires AppKit, so must run after app.run())
        self._set_symbol(_SYMBOL_IDLE)

        # Build overlay and settings window (both AppKit — main thread only)
        try:
            from ui.overlay import RecordingOverlay
            self._overlay = RecordingOverlay()
        except Exception:
            logger.exception("Failed to create RecordingOverlay")

        # _vad_for_settings is set by run_dictation_app before app.run()
        vad = getattr(self, "_vad_for_settings", None)

        try:
            from ui.settings_window import SettingsWindow
            self._settings_win = SettingsWindow(
                settings=self._settings,
                vad=vad,
                on_hotkey_changed=self._hotkey_restart_callback,
            )
        except Exception:
            logger.exception("Failed to create SettingsWindow")

        logger.info("%s %s running in menu bar", APP_NAME, APP_VERSION)

    def set_vad(self, vad) -> None:
        """Provide the live VoiceActivityDetector so settings can update its threshold."""
        if self._settings_win is not None:
            self._settings_win.set_vad(vad)

    # ── UI queue drain (main thread, every 50ms) ──────────────────────────────

    def _drain_ui_queue(self, _sender) -> None:
        """Process all pending UIEvents and advance overlay animation."""
        # Drain all queued events
        while True:
            try:
                event: UIEvent = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            self._apply_event(event)

        # Advance overlay animation, passing current mic amplitude
        if self._overlay is not None:
            try:
                self._overlay.tick(self._amplitude_ref[0])
            except Exception:
                logger.exception("Overlay tick error")

    def _apply_event(self, event: UIEvent) -> None:
        from ui.overlay import OverlayState

        if event.state == "recording":
            self._set_symbol(_SYMBOL_RECORDING)
            self._status_item_label.title = "Status: Recording…"
            if self._overlay:
                self._overlay.set_state(OverlayState.RECORDING)

        elif event.state == "transcribing":
            self._set_symbol(_SYMBOL_PROCESSING)
            self._status_item_label.title = "Status: Transcribing…"
            if self._overlay:
                self._overlay.set_state(OverlayState.TRANSCRIBING)

        elif event.state == "done":
            self._set_symbol(_SYMBOL_IDLE)
            self._status_item_label.title = "Status: Idle"
            if self._overlay:
                self._overlay.set_state(OverlayState.DONE)

        elif event.state == "idle":
            self._set_symbol(_SYMBOL_IDLE)
            self._status_item_label.title = "Status: Idle"
            if self._overlay:
                self._overlay.set_state(OverlayState.HIDDEN)

    # ── Menu callbacks ────────────────────────────────────────────────────────

    def _open_preferences(self, _sender) -> None:
        if self._settings_win is not None:
            self._settings_win.show()
        else:
            logger.warning("Settings window not available")

    def _quit(self, _sender) -> None:
        try:
            if self._stop_callback:
                self._stop_callback()
        finally:
            rumps.quit_application()

    # ── Icon helpers ──────────────────────────────────────────────────────────

    def _set_symbol(self, symbol_name: str) -> None:
        """Set the menu bar icon to an SF Symbol. No-op if AppKit is unavailable."""
        try:
            from AppKit import NSImage
            img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
                symbol_name, None
            )
            if img is None:
                return
            img.setTemplate_(True)
            # rumps 0.4.0 stores the NSStatusItem as self.nsstatusitem
            self.nsstatusitem.button().setImage_(img)
        except Exception as exc:
            # Degrade gracefully — status label still conveys state
            logger.debug("Could not set SF Symbol icon %s: %s", symbol_name, exc)
