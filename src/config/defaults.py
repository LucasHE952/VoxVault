"""Default configuration values for ondevice-dictation.

All user-facing defaults live here. Changing a default here propagates
everywhere — no other file should hardcode these values.
"""

import os
from pathlib import Path

# ── App identity ────────────────────────────────────────────────────────────
APP_NAME: str = "ondevice-dictation"
APP_VERSION: str = "0.1.0"

# ── Paths ───────────────────────────────────────────────────────────────────
CONFIG_DIR: Path = Path.home() / ".config" / APP_NAME
CONFIG_FILE: Path = CONFIG_DIR / "settings.json"
MODEL_CACHE_DIR: Path = Path.home() / ".cache" / APP_NAME / "models"
LOG_FILE: Path = CONFIG_DIR / "app.log"

# ── Model ───────────────────────────────────────────────────────────────────
# HuggingFace repo ID for Voxtral Realtime weights.
# mistralai/Voxtral-Mini-4B-Realtime-2602 is the only Voxtral model purpose-built
# for streaming/real-time transcription (causal audio encoder, sub-200ms latency).
# The mlx-community 4-bit quantisation (~2–2.5GB) runs on 8GB+ Apple Silicon Macs.
MODEL_REPO_ID: str = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
MODEL_LOCAL_DIR: Path = MODEL_CACHE_DIR / "voxtral-realtime"

# ── Audio ───────────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16_000          # Hz — Voxtral expects 16 kHz input
CHANNELS: int = 1                  # Mono
CHUNK_DURATION_MS: int = 100       # Audio chunk size fed to VAD (100 ms)
CHUNK_SIZE: int = SAMPLE_RATE * CHUNK_DURATION_MS // 1000  # samples per chunk

# ── VAD ─────────────────────────────────────────────────────────────────────
VAD_SENSITIVITY_LEVELS: dict[str, float] = {
    "low": 0.3,       # catches more speech, less filtering
    "medium": 0.5,    # balanced default
    "high": 0.7,      # aggressive silence filtering
}
VAD_DEFAULT_SENSITIVITY: str = "medium"

# ── Hotkey ──────────────────────────────────────────────────────────────────
DEFAULT_HOTKEY: str = "right_option"

# ── Transcription ───────────────────────────────────────────────────────────
DEFAULT_LANGUAGE: str = "en"
SUPPORTED_LANGUAGES: list[str] = [
    "en", "fr", "de", "es", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ko", "ar"
]

# ── Text injection ──────────────────────────────────────────────────────────
# Delay between simulated keystrokes (seconds). Too fast causes dropped chars
# in some apps; 0 works in most but 0.001 adds safety margin.
KEYSTROKE_DELAY: float = 0.001

# ── Overlay UI ───────────────────────────────────────────────────────────────
OVERLAY_WIDTH_PT: float = 80.0
OVERLAY_HEIGHT_PT: float = 26.0
OVERLAY_CORNER_RADIUS: float = 13.0
OVERLAY_DOCK_MARGIN_PT: float = 80.0  # pt above the dock / visible frame bottom

# ── Default settings dict (serialised to JSON) ──────────────────────────────
DEFAULT_SETTINGS: dict = {
    "hotkey": DEFAULT_HOTKEY,
    "language": DEFAULT_LANGUAGE,
    "vad_sensitivity": VAD_DEFAULT_SENSITIVITY,
    "custom_vocabulary": [],
    "model_path": str(MODEL_LOCAL_DIR),
}
