# Live Captions v1.0

A standalone desktop application that displays real-time speech-to-text captions in a transparent overlay window.

**Test Coverage**: 154 unit tests

## Features

- 🎙️ **Microphone capture** - Direct audio input
- 🔊 **System audio capture** - WASAPI loopback for perfect quality
- 🖥️ **Transparent overlay** - Always on top, movie subtitle style
- 🔤 **High DPI support** - Crisp text on 4K displays
- 🎨 **Customizable** - Resize text with mouse wheel
- ⚡ **Real-time streaming** - WebSocket connection to ASR service
- 🔧 **Debug mode** - Verbose logging and audio saving
- **🖱️ System Tray App** - Easy access with right-click backend selection
- 📼 **Recording** - Save audio with automatic upload to Audio Notes

## Requirements

- Python 3.11+
- Windows 10/11 (for DPI awareness and WASAPI loopback)
- Running ASR service (Vosk, Parakeet, or Whisper via Docker)

## Installation

```bash
# From the AI-Tools root directory
cd apps/live-captions

# Install dependencies
pip install -r requirements.txt
```

## Usage

### System Tray App (Recommended)

The easiest way to use Live Captions is via the system tray app:

```bash
# Run the tray app
python live_captions_tray.py

# Or with auto-start
python live_captions_tray.py --auto-start
```

**Tray Controls:**
| Action | Description |
|--------|-------------|
| **Double-click** tray icon | Start/Stop with Whisper (default) |
| **Right-click** tray icon | Menu to select backend & audio source |

**Right-click Menu:**
- 🎙️ **Whisper** (GPU, Multilingual)
- 🎙️ **Parakeet** (GPU, English)
- 🎙️ **Vosk** (CPU, Lightweight)
- 🔊 **System Audio** / 🎤 **Microphone** toggle

### Build Windows Executable

To create a standalone `.exe` file:

```bash
# Run the build script
scripts\build_tray.bat
```

This creates `dist/Live Captions.exe` - a single executable you can:
- Add to Windows startup
- Pin to taskbar
- Put on your desktop

### Command Line Usage

```bash
# Make sure ASR service is running
docker compose up -d whisper-asr     # GPU-based (recommended)
# OR
docker compose up -d parakeet-asr    # GPU-based (English)
# OR
docker compose up -d vosk-asr        # CPU-based

# Run with microphone (default)
python live_captions.py

# Run with system audio (captures what you hear)
python live_captions.py --system-audio

# List available devices
python live_captions.py --list-devices

# Run with debug logging
python live_captions.py --debug
```

### Controls

| Action | Control |
|--------|---------|
| Move window | Drag with mouse |
| Resize text | Mouse wheel scroll |
| Close | Right-click or Escape |

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--system-audio` | off | Capture system audio instead of microphone |
| `--device N` | default | Microphone device index |
| `--loopback-device N` | default | System audio loopback device index |
| `--list-devices` | - | List available audio devices |
| `--backend` | vosk | ASR backend (vosk/parakeet) |
| `--debug` | off | Enable verbose debug logging |
| `--debug-save-audio` | off | Save captured audio on exit |

Backend is configured in `shared/config/backends.py`:
```python
BACKEND = "vosk"      # CPU-based, port 8001
# BACKEND = "parakeet"  # GPU-based, port 8002
```

## Project Structure

```
live-captions/
├── live_captions.py          # Main caption window entry point
├── live_captions_tray.py     # System tray app entry point
├── test_live_captions.py     # Unit tests for main app
├── test_live_captions_tray.py # Unit tests for tray app
├── requirements.txt
├── README.md
├── CHANGELOG.md
├── icon.ico                  # Application icon
├── Live Captions Tray.spec   # PyInstaller spec file
├── scripts/                  # Windows batch scripts
│   ├── run.bat               # Run caption window
│   ├── run_tray.bat          # Run tray app (development)
│   └── build_tray.bat        # Build Windows executable
└── src/                      # Source modules
    ├── __init__.py
    ├── audio/                # Audio capture module (81 tests)
    │   ├── __init__.py
    │   ├── capture.py        # MicrophoneCapture, SystemAudioCapture
    │   ├── devices.py        # Device listing
    │   ├── recorder.py       # AudioRecorder for saving
    │   ├── utils.py          # Resampling, stereo-to-mono
    │   └── test_*.py         # Unit tests
    ├── ui/                   # UI module
    │   ├── __init__.py
    │   └── window.py         # CaptionWindow overlay
    └── asr/                  # ASR client module
        ├── __init__.py
        ├── client.py         # ASRClient WebSocket
        └── test_client.py    # Unit tests
```

## Testing

```bash
# Run all live-captions tests (154 tests)
python -m pytest . -v

# Run specific test files
python -m pytest src/audio -v  # Audio module tests
python -m pytest src/asr -v    # ASR client tests
```

## Architecture

```
┌─────────────────┐     WebSocket      ┌─────────────────┐
│  Live Captions  │ ───────────────────▶│   ASR Service   │
│   (Desktop)     │     Audio Stream   │   (Docker)      │
│                 │ ◀─────────────────  │                 │
│  Mic/System     │   {id, text} JSON  │  Vosk/Parakeet  │
└─────────────────┘                    └─────────────────┘
```

## Protocol

Server sends JSON messages with segment ID and text:
```json
{"id": "s0", "text": "hello world"}
```

The TranscriptManager uses ID-based logic:
- **New ID** → Append new segment
- **Existing ID** → Replace text for that segment

## Debug Output

With `--debug` flag, you'll see:
```
[DEBUG] shared.client.transcript: [APPEND] s0 = 'hello'
[DEBUG] shared.client.transcript: [REPLACE] s0 = 'hello world'
```

## Troubleshooting

### "No module named 'pyaudio'"
```bash
pip install pyaudio
```

### "No module named 'websockets'"
```bash
pip install websockets
```

### "Connection refused"
Make sure the ASR service is running:
```bash
docker ps | grep vosk
# OR
docker ps | grep parakeet
```

### Wrong Python on Windows
Use the project's virtual environment Python:
```bash
.venv\Scripts\python.exe live_captions.py
```

### No microphone detected
Check your Windows sound settings and ensure a microphone is set as default input device.
