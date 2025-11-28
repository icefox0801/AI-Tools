# Live Captions

A standalone desktop application that displays real-time speech-to-text captions in a transparent overlay window.

## Features

- 🎙️ **Direct microphone capture** - No browser needed
- 🖥️ **Transparent overlay** - Always on top, movie subtitle style
- 🔤 **High DPI support** - Crisp text on 4K displays
- 🎨 **Customizable** - Resize text with mouse wheel
- ⚡ **Real-time** - Streams audio to Parakeet ASR service

## Requirements

- Python 3.10+
- Windows 10/11 (for DPI awareness)
- Running Parakeet ASR service (Docker)

## Installation

```bash
# From the AI-Tools root directory
cd apps/live-captions

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Make sure Parakeet ASR is running
docker compose up -d parakeet-asr

# Run the caption overlay
python live_captions.py

# Or use the batch launcher from AI-Tools root
.\apps\live-captions\run.bat
```

### Controls

| Action | Control |
|--------|---------|
| Move window | Drag with mouse |
| Resize text | Mouse wheel scroll |
| Close | Right-click or Escape |

### Command Line Options

```bash
python live_captions.py --host localhost --port 8002
```

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | localhost | ASR service hostname |
| `--port` | 8002 | ASR service port |

## Architecture

```
┌─────────────────┐     WebSocket      ┌─────────────────┐
│  Live Captions  │ ───────────────────▶│  Parakeet ASR   │
│   (Desktop)     │     Audio Stream   │   (Docker)      │
│                 │ ◀─────────────────  │                 │
│   Microphone    │     Transcripts    │   GPU (RTX)     │
└─────────────────┘                    └─────────────────┘
```

## Troubleshooting

### "No module named 'pyaudio'"
```bash
pip install pyaudio
```

### "Connection refused"
Make sure the Parakeet ASR service is running:
```bash
docker ps | grep parakeet
```

### No microphone detected
Check your Windows sound settings and ensure a microphone is set as default input device.
