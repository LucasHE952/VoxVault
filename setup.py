"""py2app build configuration for OnDevice Dictation.

Build a release .app bundle:
    python setup.py py2app

Build a fast alias bundle for development/testing:
    python setup.py py2app --alias

After building:
    open 'dist/OnDevice Dictation.app'              # test it
    cp -r 'dist/OnDevice Dictation.app' /Applications/  # install
"""

import sys
from pathlib import Path
from setuptools import setup

# modulegraph (used by py2app) hits Python's default recursion limit when
# scanning deeply nested packages like torch. Raise it before the scan runs.
sys.setrecursionlimit(5000)

# Make src/ packages visible to py2app's module scanner
sys.path.insert(0, str(Path(__file__).parent / "src"))

APP = ["src/main.py"]

OPTIONS = {
    # ── Bundle behaviour ──────────────────────────────────────────────────────
    "argv_emulation": False,  # Carbon-based; not needed for AppKit/rumps apps
    "site_packages": True,   # Include all installed site-packages
    # Don't compress packages into python312.zip — native dylibs (e.g.
    # libportaudio.dylib for sounddevice) can't be dlopen'd from inside a zip.
    "no_zip": True,
    # ── Info.plist ────────────────────────────────────────────────────────────
    "plist": {
        "CFBundleName": "OnDevice Dictation",
        "CFBundleDisplayName": "OnDevice Dictation",
        "CFBundleIdentifier": "com.ondevice-dictation.app",
        "CFBundleVersion": "0.1.0",
        "CFBundleShortVersionString": "0.1.0",
        # LSUIElement = True → menu bar only; no dock icon, no app switcher entry
        "LSUIElement": True,
        # Privacy usage strings — macOS shows these in System Settings
        "NSMicrophoneUsageDescription": (
            "OnDevice Dictation records your voice to transcribe speech into text. "
            "Audio is processed entirely on-device and never leaves your Mac."
        ),
        "NSInputMonitoringUsageDescription": (
            "OnDevice Dictation monitors keyboard input to detect your push-to-talk "
            "hotkey and the Escape key to cancel recording."
        ),
        # Accessibility permission is granted manually in System Settings;
        # there is no Info.plist key for it — the app checks at runtime.
        "NSPrincipalClass": "NSApplication",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "13.0",  # macOS Ventura+
    },
    # ── Packages to bundle ────────────────────────────────────────────────────
    # List packages that py2app's static analyser might miss (C extensions,
    # lazy imports, packages loaded via importlib, etc.)
    "packages": [
        # App source (in src/)
        "audio",
        "transcription",
        "injection",
        "hotkey",
        "ui",
        "config",
        # Menu bar / UI
        "rumps",
        # Input monitoring
        "pynput",
        "pynput.keyboard",
        "pynput.mouse",
        # Audio — _sounddevice_data MUST be a real directory (not zipped) because
        # it contains libportaudio.dylib which is loaded via dlopen at runtime.
        "sounddevice",
        "_sounddevice_data",
        "numpy",
        # VAD (pulls in torch for CPU inference)
        "silero_vad",
        "torch",
        "torchaudio",
        # MLX inference (Apple Silicon Neural Engine / GPU)
        # mlx is a namespace package — imp_find_module can't locate it, so we
        # can't list it here. It's copied manually in build_app.sh post-build.
        # PyObjC frameworks
        "objc",
        "AppKit",
        "Foundation",
        "Quartz",
        "ApplicationServices",
        # Standard library — only true packages belong here (with __init__.py)
        "encodings",
    ],
    # ctypes.util is a module (not a package) — goes in includes, not packages
    "includes": [
        "ctypes",
        "ctypes.util",
    ],
    # Exclude packages we don't use — reduces bundle size and avoids SIP copy
    # errors on signed C extensions we don't need.
    "excludes": [
        "scipy",
        "matplotlib",
        "sklearn",
        "pandas",
        "IPython",
        "jupyter",
        "notebook",
        "PIL",
        "cv2",
        "tensorflow",
        "keras",
        "flask",
        "django",
        "sqlalchemy",
        "pytest",
        "docutils",
        "sphinx",
        "boto3",
        "botocore",
    ],
    # ── App icon ──────────────────────────────────────────────────────────────
    # Generate assets/app.icns with:  bash scripts/create_icon.sh
    # Leave commented out until the icon file exists.
    # "iconfile": "assets/app.icns",
}

setup(
    app=APP,
    name="OnDevice Dictation",
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
