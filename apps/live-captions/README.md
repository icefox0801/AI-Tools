# Live Captions v10.1

A standalone desktop application that displays real-time speech-to-text captions in a transparent overlay window.

## Features

- 🎙️ **Microphone capture** - Direct audio input
- 🔊 **System audio capture** - WASAPI loopback for perfect quality
- 🖥️ **Transparent overlay** - Always on top, movie subtitle style
- 🔤 **High DPI support** - Crisp text on 4K displays
- 🎨 **Customizable** - Resize text with mouse wheel
- ⚡ **Real-time streaming** - WebSocket connection to ASR service
- 🔧 **Debug mode** - Verbose logging and audio saving

## Requirements

- Python 3.10+
- Windows 10/11 (for DPI awareness and WASAPI loopback)
- Running ASR service (Vosk or Parakeet via Docker)

## Installation

```bash
# From the AI-Tools root directory
cd apps/live-captions

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Make sure ASR service is running
docker compose up -d vosk-asr       # CPU-based (default)
# OR
docker compose up -d parakeet-asr   # GPU-based

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
├── live_captions.py      # Main entry point
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py       # Package init (version)
    ├── audio/            # Audio capture module
    │   ├── __init__.py
    │   ├── capture.py    # MicrophoneCapture, SystemAudioCapture
    │   ├── devices.py    # Device listing
    │   └── utils.py      # Resampling, stereo-to-mono
    ├── ui/               # UI module
    │   ├── __init__.py
    │   └── window.py     # CaptionWindow overlay
    └── asr/              # ASR client module
        ├── __init__.py
        └── client.py     # ASRClient WebSocket
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
If Inkscape or other apps add Python to PATH, use explicit path:
```bash
"C:\Users\<user>\AppData\Local\Microsoft\WindowsApps\python.exe" live_captions.py
```

### No microphone detected
Check your Windows sound settings and ensure a microphone is set as default input device.
