# ondevice-ai-dictation

> Accurate, private, local dictation for macOS — powered by Voxtral Realtime on Apple Silicon.

Dictate into any app by holding a hotkey. All transcription runs entirely on your machine using [Mistral's Voxtral Realtime](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) model — no audio or text ever leaves your device. More accurate than macOS built-in dictation, more private than cloud tools like Wispr Flow, and completely free.

---

## Requirements

> **Apple Silicon required.** Intel Macs are not supported.

| Requirement | Minimum | Recommended |
|---|---|---|
| Chip | Apple M1 | Apple M2 or later |
| Memory | 8GB unified | 16GB unified |
| macOS | 13 (Ventura) | 14 (Sonoma) or later |
| Disk space | 4GB free | — |
| Python | 3.11 | 3.12 |

---

## Demo

*GIF coming soon — add one showing recording → text appearing in an app.*

---

## Install

```bash
git clone https://github.com/your-username/ondevice-dictation.git
cd ondevice-dictation
bash setup.sh
```

`setup.sh` will:
- Check your hardware and macOS version
- Create a Python virtual environment
- Install all dependencies
- Download the Voxtral model weights (~4GB, one time only)
- Print instructions for granting required permissions

---

## macOS Permissions

The app requires two permissions:

### 1. Microphone
macOS will prompt you automatically the first time the app tries to record. Click **Allow**.

### 2. Accessibility (text injection)
macOS **will not** prompt for this automatically — you must grant it manually:

1. Open **System Settings → Privacy & Security → Accessibility**
2. Click the **+** button
3. Add your **Terminal** app (or `python3` binary)
4. Enable the toggle next to it

Without this permission, transcribed text cannot be typed into other apps.

---

## Usage

### Phase 1 — Smoke test (available now)

After setup, test that the model is working:

```bash
source .venv/bin/activate
python src/main.py --phase1
```

Speak for 5 seconds. Your transcription will be printed to the terminal.

### Full dictation mode (Phase 2 — coming soon)

Hold the **Right Option** key to dictate. Release to finish. Transcribed text is typed into whatever app is in focus.

---

## How it works

```
[Microphone] → [VAD] → [Voxtral Realtime on MLX] → [Text injection via Quartz]
```

1. **Audio capture** — `sounddevice` streams microphone audio at 16kHz
2. **Voice Activity Detection** — Silero VAD filters silence in real-time
3. **Transcription** — Voxtral Realtime runs locally via MLX on your GPU/Neural Engine
4. **Text injection** — Quartz CGEvent simulates keystrokes system-wide

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed technical breakdown.

---

## Configuration

Settings are stored at `~/.config/ondevice-dictation/settings.json`:

```json
{
  "hotkey": "right_option",
  "language": "en",
  "vad_sensitivity": "medium",
  "custom_vocabulary": [],
  "model_path": "~/.cache/ondevice-dictation/models/voxtral-realtime"
}
```

Supported languages: `en fr de es it pt nl pl ru zh ja ko ar`

---

## Contributing

Contributions are welcome. Please:

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) before making changes
2. Follow the build phase order — don't implement Phase 3 features in a Phase 2 PR
3. Keep functions small and single-purpose
4. Add type hints and Google-style docstrings to all public functions
5. Write tests for new audio, VAD, or injection logic

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
