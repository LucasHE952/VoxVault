"""Native NSPanel settings window for ondevice-dictation (Phase 3).

Sections:
  - Hotkey: click to capture next key press
  - VAD Sensitivity: Low / Medium / High segmented control (live update)
  - Custom Vocabulary: editable list with +/- buttons
  - About: version info and GitHub link

Layout is frame-based (not Auto Layout) — simpler and more reliable in pyobjc
for a fixed-size, non-resizable panel.

Threading note:
  - show() must be called on the main thread (from menu bar timer or callback)
  - Hotkey capture uses a temporary pynput Listener on a daemon thread;
    the UI update is marshalled back to the main thread via a
    _MainThreadDispatcher NSObject helper
"""

import logging
import threading
from collections.abc import Callable
from typing import Optional

import objc
from AppKit import (
    NSApp,
    NSBezelStyleRounded,
    NSButton,
    NSColor,
    NSFont,
    NSMakeRect,
    NSObject,
    NSPanel,
    NSScrollView,
    NSSegmentedControl,
    NSTableColumn,
    NSTableView,
    NSTextField,
    NSTextAlignmentCenter,
    NSWindowStyleMaskTitled,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskNonactivatingPanel,
    NSWorkspace,
)
from Foundation import NSURL

from config.defaults import (
    APP_NAME,
    APP_VERSION,
    VAD_SENSITIVITY_LEVELS,
)

logger = logging.getLogger(__name__)

_WIN_W: float = 440.0
_WIN_H: float = 420.0
_BACKING_STORE_BUFFERED = 2


# ── Main-thread dispatcher ────────────────────────────────────────────────────

class _MainThreadDispatcher(NSObject):
    """Utility that calls a Python callable on the main thread.

    Used to marshal pynput's key-capture callback back to the main thread
    for safe AppKit updates.
    """

    def call_(self, fn):
        fn()


_dispatcher = None  # initialised lazily on first use (must be on main thread)


def _dispatch_main(fn: Callable) -> None:
    """Schedule fn() to run on the main NSRunLoop thread."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = _MainThreadDispatcher.alloc().init()
    _dispatcher.performSelectorOnMainThread_withObject_waitUntilDone_(
        "call:", fn, False
    )


# ── Vocabulary table data source ──────────────────────────────────────────────

class _VocabDataSource(NSObject):
    """NSTableViewDataSource and delegate for the custom vocabulary list."""

    def init(self):
        self = objc.super(_VocabDataSource, self).init()
        if self is None:
            return None
        self._words: list[str] = []
        self._on_change: Optional[Callable[[list[str]], None]] = None
        return self

    # NSTableViewDataSource protocol
    def numberOfRowsInTableView_(self, table_view):
        return len(self._words)

    def tableView_objectValueForTableColumn_row_(self, table_view, column, row):
        if 0 <= row < len(self._words):
            return self._words[row]
        return ""

    def tableView_setObjectValue_forTableColumn_row_(
        self, table_view, obj, column, row
    ):
        value = str(obj).strip() if obj else ""
        if 0 <= row < len(self._words) and value:
            self._words[row] = value
            if self._on_change:
                self._on_change(list(self._words))


# ── Settings window ───────────────────────────────────────────────────────────

class SettingsWindow:
    """Native NSPanel providing user-accessible settings.

    Args:
        settings: Loaded Settings instance (read/write).
        vad: Live VoiceActivityDetector — threshold updated immediately on
             sensitivity change. Pass None if not yet available; set later
             via set_vad().
        on_hotkey_changed: Called with the new hotkey name string when the
             user captures a new hotkey. Caller is responsible for restarting
             HotkeyListener.
    """

    def __init__(
        self,
        settings,
        vad=None,
        on_hotkey_changed: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._settings = settings
        self._vad = vad
        self._on_hotkey_changed = on_hotkey_changed
        self._capture_listener = None  # pynput listener active during key capture

        self._panel = self._build_panel()
        self._populate()

    def set_vad(self, vad) -> None:
        """Inject the live VAD instance (can be called after construction)."""
        self._vad = vad

    def show(self) -> None:
        """Bring the settings panel to front. Must be called on main thread."""
        self._panel.center()
        self._panel.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)

    # ── Panel construction ────────────────────────────────────────────────────

    def _build_panel(self) -> NSPanel:
        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskNonactivatingPanel
        )
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, _WIN_W, _WIN_H),
            style,
            _BACKING_STORE_BUFFERED,
            False,
        )
        panel.setTitle_(f"{APP_NAME} — Preferences")
        panel.setReleasedWhenClosed_(False)  # keep alive for reuse
        return panel

    def _populate(self) -> None:
        cv = self._panel.contentView()
        y = _WIN_H  # layout top-to-bottom; y tracks next available top edge

        # ── Section: Hotkey ──────────────────────────────────────────────────
        y -= 30
        cv.addSubview_(self._section_label("Hotkey", y=y))

        y -= 34
        # Display current hotkey name
        self._hotkey_field = self._make_field(
            self._settings["hotkey"],
            rect=NSMakeRect(20, y, 240, 24),
            editable=False,
        )
        cv.addSubview_(self._hotkey_field)

        self._hotkey_btn = self._make_button(
            "Click to change…",
            rect=NSMakeRect(270, y, 150, 24),
            action=self._start_hotkey_capture,
        )
        cv.addSubview_(self._hotkey_btn)

        # ── Section: VAD Sensitivity ─────────────────────────────────────────
        y -= 44
        cv.addSubview_(self._section_label("VAD Sensitivity", y=y))

        y -= 34
        sensitivity_options = ["Low", "Medium", "High"]
        seg = NSSegmentedControl.segmentedControlWithLabels_trackingMode_target_action_(
            sensitivity_options,
            1,  # NSSegmentSwitchTrackingSelectOne
            None,
            None,
        )
        seg.setFrame_(NSMakeRect(20, y, 200, 24))
        current = self._settings["vad_sensitivity"]
        keys = ["low", "medium", "high"]
        if current in keys:
            seg.setSelectedSegment_(keys.index(current))

        # Wire action via target/action
        seg.setTarget_(self._make_vad_handler(seg))
        seg.setAction_("vadChanged:")
        self._vad_seg = seg
        cv.addSubview_(seg)

        y -= 20
        note = self._make_field(
            "Takes effect immediately on next recording.",
            rect=NSMakeRect(20, y, _WIN_W - 40, 18),
            editable=False,
            font_size=11.0,
            color=NSColor.secondaryLabelColor(),
        )
        cv.addSubview_(note)

        # ── Section: Custom Vocabulary ────────────────────────────────────────
        y -= 44
        cv.addSubview_(self._section_label("Custom Vocabulary", y=y))

        y -= 140
        table_h = 130
        self._vocab_source = _VocabDataSource.alloc().init()
        self._vocab_source._words = list(self._settings["custom_vocabulary"])
        self._vocab_source._on_change = self._on_vocab_changed

        col = NSTableColumn.alloc().initWithIdentifier_("word")
        col.setWidth_(_WIN_W - 60)
        col.headerCell().setStringValue_("Word or phrase")

        self._vocab_table = NSTableView.alloc().initWithFrame_(
            NSMakeRect(0, 0, _WIN_W - 44, table_h)
        )
        self._vocab_table.addTableColumn_(col)
        self._vocab_table.setDataSource_(self._vocab_source)
        self._vocab_table.setDelegate_(self._vocab_source)
        self._vocab_table.setUsesAlternatingRowBackgroundColors_(True)

        scroll = NSScrollView.alloc().initWithFrame_(
            NSMakeRect(20, y, _WIN_W - 44, table_h)
        )
        scroll.setDocumentView_(self._vocab_table)
        scroll.setHasVerticalScroller_(True)
        scroll.setBorderType_(2)  # NSBezelBorder
        cv.addSubview_(scroll)

        y -= 32
        add_btn = self._make_button("+", rect=NSMakeRect(20, y, 30, 24), action=self._add_word)
        del_btn = self._make_button("−", rect=NSMakeRect(56, y, 30, 24), action=self._del_word)
        cv.addSubview_(add_btn)
        cv.addSubview_(del_btn)

        # ── Section: About ────────────────────────────────────────────────────
        y -= 44
        cv.addSubview_(self._section_label("About", y=y))

        y -= 28
        version_label = self._make_field(
            f"{APP_NAME}  v{APP_VERSION}  •  Apache 2.0  •  Runs fully on-device",
            rect=NSMakeRect(20, y, _WIN_W - 40, 20),
            editable=False,
            font_size=12.0,
        )
        cv.addSubview_(version_label)

        y -= 30
        gh_btn = self._make_button(
            "View on GitHub",
            rect=NSMakeRect(20, y, 140, 24),
            action=self._open_github,
        )
        cv.addSubview_(gh_btn)

    # ── Section helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _section_label(text: str, y: float) -> NSTextField:
        lbl = NSTextField.labelWithString_(text.upper())
        lbl.setFrame_(NSMakeRect(20, y, _WIN_W - 40, 18))
        lbl.setFont_(NSFont.boldSystemFontOfSize_(11.0))
        lbl.setTextColor_(NSColor.secondaryLabelColor())
        return lbl

    @staticmethod
    def _make_field(
        text: str,
        rect,
        editable: bool = True,
        font_size: float = 13.0,
        color: Optional[NSColor] = None,
    ) -> NSTextField:
        if editable:
            field = NSTextField.alloc().initWithFrame_(rect)
            field.setStringValue_(text)
        else:
            field = NSTextField.labelWithString_(text)
            field.setFrame_(rect)

        field.setFont_(NSFont.systemFontOfSize_(font_size))
        if color:
            field.setTextColor_(color)
        field.setEditable_(editable)
        field.setBordered_(editable)
        field.setDrawsBackground_(editable)
        return field

    @staticmethod
    def _make_button(title: str, rect, action: Callable) -> NSButton:
        btn = NSButton.alloc().initWithFrame_(rect)
        btn.setTitle_(title)
        btn.setBezelStyle_(NSBezelStyleRounded)
        _btn_register(btn, action)
        btn.setAction_("fire:")
        btn.setTarget_(_btn_shared())
        return btn

    # ── VAD handler (needs NSObject target for setTarget_/setAction_) ─────────

    def _make_vad_handler(self, seg: NSSegmentedControl):
        """Return an NSObject target for the VAD segmented control."""
        keys = ["low", "medium", "high"]
        settings = self._settings
        vad_ref = self

        class _Handler(NSObject):
            def vadChanged_(self, sender):
                idx = sender.selectedSegment()
                key = keys[idx]
                settings["vad_sensitivity"] = key
                settings.save()
                # Update live VAD threshold (no restart needed)
                if vad_ref._vad is not None:
                    vad_ref._vad.threshold = VAD_SENSITIVITY_LEVELS[key]
                logger.debug("VAD sensitivity changed to %s", key)

        handler = _Handler.alloc().init()
        self._vad_handler = handler  # keep alive
        return handler

    # ── Hotkey capture ────────────────────────────────────────────────────────

    def _start_hotkey_capture(self) -> None:
        self._hotkey_btn.setTitle_("Press a key…")
        self._hotkey_btn.setEnabled_(False)

        from hotkey.listener import HOTKEY_MAP
        from pynput import keyboard as pynput_kb

        def on_press(key):
            # Find matching key name in HOTKEY_MAP
            key_name = None
            for name, mapped in HOTKEY_MAP.items():
                if key == mapped:
                    key_name = name
                    break

            if key_name:
                captured = key_name

                def apply():
                    self._apply_hotkey(captured)

                _dispatch_main(apply)

            return False  # Stop listener after first key

        self._capture_listener = pynput_kb.Listener(on_press=on_press)
        self._capture_listener.daemon = True
        self._capture_listener.start()

    def _apply_hotkey(self, key_name: str) -> None:
        """Called on main thread after a key is captured."""
        self._settings["hotkey"] = key_name
        self._settings.save()
        self._hotkey_field.setStringValue_(key_name)
        self._hotkey_btn.setTitle_("Click to change…")
        self._hotkey_btn.setEnabled_(True)

        if self._on_hotkey_changed:
            self._on_hotkey_changed(key_name)

        logger.info("Hotkey changed to %s", key_name)

    # ── Vocabulary actions ────────────────────────────────────────────────────

    def _add_word(self) -> None:
        self._vocab_source._words.append("")
        self._vocab_table.reloadData()
        new_row = len(self._vocab_source._words) - 1
        self._vocab_table.editColumn_row_withEvent_select_(0, new_row, None, True)

    def _del_word(self) -> None:
        row = self._vocab_table.selectedRow()
        if 0 <= row < len(self._vocab_source._words):
            self._vocab_source._words.pop(row)
            self._vocab_table.reloadData()
            self._on_vocab_changed(list(self._vocab_source._words))

    def _on_vocab_changed(self, words: list[str]) -> None:
        self._settings["custom_vocabulary"] = [w for w in words if w.strip()]
        self._settings.save()

    # ── About ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _open_github() -> None:
        url = NSURL.URLWithString_("https://github.com/LucasHE952/ondevice-dictation")
        NSWorkspace.sharedWorkspace().openURL_(url)


# ── Button trampoline ─────────────────────────────────────────────────────────
# Module-level state so no classmethods live on the NSObject subclass
# (pyobjc inspects every non-underscore method as a potential ObjC selector).

_btn_registry: dict[int, Callable] = {}
_btn_next_tag: int = 1000
_btn_trampoline: Optional["_ButtonTrampoline"] = None


def _btn_register(btn: NSButton, fn: Callable) -> None:
    """Assign a tag to btn and store fn in the dispatch table."""
    global _btn_next_tag
    tag = _btn_next_tag
    _btn_next_tag += 1
    _btn_registry[tag] = fn
    btn.setTag_(tag)


def _btn_shared() -> "_ButtonTrampoline":
    global _btn_trampoline
    if _btn_trampoline is None:
        _btn_trampoline = _ButtonTrampoline.alloc().init()
    return _btn_trampoline


class _ButtonTrampoline(NSObject):
    """Receives fire: from every NSButton and dispatches to the right callable."""

    def fire_(self, sender):
        fn = _btn_registry.get(sender.tag())
        if fn:
            fn()
