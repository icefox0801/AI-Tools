# Live Captions v8.0

A standalone desktop application that displays real-time speech-to-text captions in a transparent overlay window.

## Features

- 🎙️ **Direct microphone capture** - No browser needed
- 🖥️ **Transparent overlay** - Always on top, movie subtitle style
- 🔤 **High DPI support** - Crisp text on 4K displays
- 🎨 **Customizable** - Resize text with mouse wheel
- ⚡ **Real-time streaming** - WebSocket connection to ASR service
- 🔧 **Debug mode** - Verbose logging with `--debug` flag

## Requirements

- Python 3.10+
- Windows 10/11 (for DPI awareness)
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

# Run the caption overlay
python live_captions.py

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
| `--debug` | off | Enable verbose debug logging |

Backend is configured in `shared/config/backends.py`:
```python
BACKEND = "vosk"      # CPU-based, port 8001
# BACKEND = "parakeet"  # GPU-based, port 8002
```

## Architecture

```
┌─────────────────┐     WebSocket      ┌─────────────────┐
│  Live Captions  │ ───────────────────▶│   ASR Service   │
│   (Desktop)     │     Audio Stream   │   (Docker)      │
│                 │ ◀─────────────────  │                 │
│   Microphone    │   {id, text} JSON  │  Vosk/Parakeet  │
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
