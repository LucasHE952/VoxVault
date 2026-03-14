"""Floating recording indicator overlay for ondevice-dictation.

A translucent pill that appears during recording and transcription.

Design:
- Borderless NSPanel with NSWindowStyleMaskNonactivatingPanel — never steals focus
- NSVisualEffectView with dark HUD material for native glassmorphism
- Custom WaveformView (NSView subclass) draws 5 animated bars during recording
- NSProgressIndicator (spinner) shown during transcription
- All transitions handled via NSAnimationContext fade in/out (0.15s / 0.25s)
- Animation is driven externally by calling tick() every ~50ms from the
  menu bar's rumps.Timer — avoids NSTimer selector registration complexity
"""

import enum
import logging
import math
import time
from typing import Optional

import objc
from AppKit import (
    NSAnimationContext,
    NSBezierPath,
    NSColor,
    NSFont,
    NSMakeRect,
    NSPanel,
    NSProgressIndicator,
    NSScreen,
    NSTextField,
    NSTextAlignmentCenter,
    NSView,
    NSViewWidthSizable,
    NSViewHeightSizable,
    NSVisualEffectView,
    NSVisualEffectMaterialHUDWindow,
    NSVisualEffectBlendingModeBehindWindow,
    NSVisualEffectStateActive,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorIgnoresCycle,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskNonactivatingPanel,
    NSFloatingWindowLevel,
)

from config.defaults import (
    OVERLAY_CORNER_RADIUS,
    OVERLAY_DOCK_MARGIN_PT,
    OVERLAY_HEIGHT_PT,
    OVERLAY_WIDTH_PT,
)

logger = logging.getLogger(__name__)

# NSBackingStoreBuffered — double-buffered window backing store
_BACKING_STORE_BUFFERED = 2
# NSProgressIndicatorSpinningStyle
_SPINNER_STYLE = 1
# NSSmallControlSize
_SMALL_CONTROL = 1


class OverlayState(enum.Enum):
    HIDDEN = "hidden"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    DONE = "done"  # brief pause, then fades to HIDDEN


class _WaveformView(NSView):
    """Five animated vertical bars that pulse while recording.

    Animation is driven by an external timer calling tick() rather than an
    internal NSTimer so we avoid Objective-C selector registration issues in
    pyobjc. drawRect_ uses time.monotonic() to advance the animation phase.
    """

    # Bar geometry
    _BAR_COUNT = 5
    _BAR_W = 4.0
    _BAR_GAP = 7.0
    _MIN_H = 6.0
    _MAX_H = 30.0
    _SPEED = 5.0  # radians per second

    def initWithFrame_(self, frame):
        self = objc.super(_WaveformView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._active = False
        self._start_time = 0.0
        self.setWantsLayer_(True)
        return self

    def drawRect_(self, rect):
        # Clear to transparent so the visual effect view shows through.
        NSColor.clearColor().set()
        NSBezierPath.fillRect_(self.bounds())

        total_w = self._BAR_COUNT * self._BAR_W + (self._BAR_COUNT - 1) * self._BAR_GAP
        sx = (self.bounds().size.width - total_w) / 2
        cy = self.bounds().size.height / 2

        if not self._active:
            # Flat resting bars (faint)
            NSColor.whiteColor().colorWithAlphaComponent_(0.35).set()
            for i in range(self._BAR_COUNT):
                x = sx + i * (self._BAR_W + self._BAR_GAP)
                y = cy - self._MIN_H / 2
                path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    NSMakeRect(x, y, self._BAR_W, self._MIN_H), 2.0, 2.0
                )
                path.fill()
            return

        elapsed = time.monotonic() - self._start_time
        NSColor.whiteColor().colorWithAlphaComponent_(0.9).set()

        for i in range(self._BAR_COUNT):
            phase = elapsed * self._SPEED + i * 1.1
            t = (math.sin(phase) + 1.0) / 2.0
            h = self._MIN_H + t * (self._MAX_H - self._MIN_H)
            x = sx + i * (self._BAR_W + self._BAR_GAP)
            y = cy - h / 2
            path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(x, y, self._BAR_W, h), 2.0, 2.0
            )
            path.fill()

    def start(self) -> None:
        self._active = True
        self._start_time = time.monotonic()
        self.setNeedsDisplay_(True)

    def stop(self) -> None:
        self._active = False
        self.setNeedsDisplay_(True)

    def tick(self) -> None:
        """Advance animation. Called every ~50ms from menu bar timer."""
        if self._active:
            self.setNeedsDisplay_(True)


class RecordingOverlay:
    """Floating pill overlay that shows recording/transcription state.

    Must be instantiated and operated on the main thread.
    External callers drive animation by calling tick() every ~50ms.
    """

    def __init__(self) -> None:
        self._state = OverlayState.HIDDEN
        self._done_at: float = 0.0

        self._panel, self._content = self._build_panel()
        self._waveform = self._build_waveform()
        self._spinner = self._build_spinner()
        self._label = self._build_label()

    # ── Construction ─────────────────────────────────────────────────────────

    def _build_panel(self) -> tuple:
        """Create the transparent borderless NSPanel with frosted glass backing."""
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, OVERLAY_WIDTH_PT, OVERLAY_HEIGHT_PT),
            NSWindowStyleMaskNonactivatingPanel,
            _BACKING_STORE_BUFFERED,
            False,
        )
        panel.setLevel_(NSFloatingWindowLevel)
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorIgnoresCycle
        )
        panel.setAlphaValue_(0.0)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(True)

        # NSVisualEffectView — dark frosted glass fill
        vfx = NSVisualEffectView.alloc().initWithFrame_(
            panel.contentView().bounds()
        )
        vfx.setMaterial_(NSVisualEffectMaterialHUDWindow)
        vfx.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        vfx.setState_(NSVisualEffectStateActive)
        vfx.setWantsLayer_(True)
        vfx.layer().setCornerRadius_(OVERLAY_CORNER_RADIUS)
        vfx.layer().setMasksToBounds_(True)
        vfx.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        panel.contentView().addSubview_(vfx)

        return panel, vfx

    def _build_waveform(self) -> _WaveformView:
        # Left zone: x=14..70, vertically centred
        wv = _WaveformView.alloc().initWithFrame_(
            NSMakeRect(14, 8, 56, OVERLAY_HEIGHT_PT - 16)
        )
        self._content.addSubview_(wv)
        return wv

    def _build_spinner(self) -> NSProgressIndicator:
        # Same left zone as waveform, centered 20×20
        spinner = NSProgressIndicator.alloc().initWithFrame_(
            NSMakeRect(28, (OVERLAY_HEIGHT_PT - 20) / 2, 20, 20)
        )
        spinner.setStyle_(_SPINNER_STYLE)
        spinner.setControlSize_(_SMALL_CONTROL)
        spinner.setHidden_(True)
        self._content.addSubview_(spinner)
        return spinner

    def _build_label(self) -> NSTextField:
        label = NSTextField.labelWithString_("Recording…")
        label.setFrame_(NSMakeRect(76, 0, OVERLAY_WIDTH_PT - 92, OVERLAY_HEIGHT_PT))
        label.setFont_(NSFont.systemFontOfSize_(13.0))
        label.setTextColor_(NSColor.whiteColor())
        label.setAlignment_(NSTextAlignmentCenter)
        self._content.addSubview_(label)
        return label

    # ── Public API ───────────────────────────────────────────────────────────

    def set_state(self, state: OverlayState) -> None:
        """Transition to the given state. Must be called on the main thread."""
        if state == OverlayState.RECORDING:
            self._label.setStringValue_("Recording…")
            self._waveform.setHidden_(False)
            self._waveform.start()
            self._spinner.setHidden_(True)
            self._spinner.stopAnimation_(None)
            self._show()

        elif state == OverlayState.TRANSCRIBING:
            self._label.setStringValue_("Transcribing…")
            self._waveform.setHidden_(True)
            self._waveform.stop()
            self._spinner.setHidden_(False)
            self._spinner.startAnimation_(None)
            # Panel already visible; no need to call _show() again

        elif state == OverlayState.DONE:
            self._waveform.setHidden_(True)
            self._waveform.stop()
            self._spinner.setHidden_(True)
            self._spinner.stopAnimation_(None)
            self._done_at = time.monotonic()

        elif state == OverlayState.HIDDEN:
            self._waveform.stop()
            self._spinner.stopAnimation_(None)
            self._fade_out()

        self._state = state

    def tick(self) -> None:
        """Called every ~50ms by the menu bar timer.

        Advances waveform animation and handles the auto-hide after DONE.
        """
        if self._state == OverlayState.RECORDING:
            self._waveform.tick()
        elif self._state == OverlayState.DONE:
            if time.monotonic() - self._done_at >= 0.8:
                self._state = OverlayState.HIDDEN
                self._fade_out()

    # ── Private helpers ──────────────────────────────────────────────────────

    def _show(self) -> None:
        screen = NSScreen.mainScreen()
        visible = screen.visibleFrame()
        screen_w = screen.frame().size.width
        x = (screen_w - OVERLAY_WIDTH_PT) / 2
        y = visible.origin.y + OVERLAY_DOCK_MARGIN_PT
        self._panel.setFrameOrigin_((x, y))
        self._panel.orderFront_(None)

        def _animate_in(ctx):
            ctx.setDuration_(0.15)
            self._panel.animator().setAlphaValue_(1.0)

        NSAnimationContext.runAnimationGroup_completionHandler_(_animate_in, None)

    def _fade_out(self) -> None:
        panel = self._panel

        def _animate_out(ctx):
            ctx.setDuration_(0.25)
            panel.animator().setAlphaValue_(0.0)

        def _on_done():
            panel.orderOut_(None)

        NSAnimationContext.runAnimationGroup_completionHandler_(_animate_out, _on_done)
