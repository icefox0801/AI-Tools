# CLAUDE.md - AI Assistant Context for AI-Tools

## Project Overview

AI-Tools is a Docker-based suite of AI services for voice transcription with a focus on real-time streaming ASR (Automatic Speech Recognition).

## Architecture

```
┌─────────────────────┐     WebSocket      ┌─────────────────────┐
│   Client Apps       │ ──────────────────▶│   ASR Services      │
│                     │    Audio Stream    │                     │
│  • Live Captions    │ ◀──────────────────│  • Vosk (CPU:8001)  │
│    (Desktop)        │    {id, text}      │  • Parakeet (GPU)   │
└─────────────────────┘                    └─────────────────────┘
```

## Key Components

### ASR Services (Docker)
- **Vosk** (`services/vosk/`) - CPU-based, lightweight, port 8001
- **Parakeet** (`services/parakeet/`) - GPU-based (NVIDIA NeMo), port 8002

### Client Applications
- **Live Captions** (`apps/live-captions/`) - Desktop overlay with microphone capture

### Shared Library (`shared/`)
- `shared/client/websocket_client.py` - WebSocket client for ASR streaming
- `shared/client/transcript.py` - TranscriptManager with ID-based replace/append
- `shared/config/backends.py` - Backend configuration (BACKEND = "vosk" or "parakeet")

## WebSocket Protocol (v3.0)

Server sends JSON messages:
```json
{"id": "s0", "text": "hello world"}
```

- **id**: Segment identifier (e.g., "s0", "s1", "s2")
- **text**: Current transcription for that segment

Client logic (TranscriptManager):
- If `id` exists → **REPLACE** text for that segment
- If `id` is new → **APPEND** new segment

## Running Commands

```bash
# Start ASR services
docker compose up -d vosk-asr

# Run Live Captions with debug logging
cd apps/live-captions
python live_captions.py --debug

# IMPORTANT: Use correct Python on Windows (not Inkscape's Python)
"C:\Users\icefo\AppData\Local\Microsoft\WindowsApps\python.exe" apps/live-captions/live_captions.py --debug
```

## Configuration

Backend selection in `shared/config/backends.py`:
```python
BACKEND = "vosk"  # or "parakeet"
```

## Debug Logging

Enable with `--debug` flag:
```bash
python live_captions.py --debug
```

Shows:
- `[APPEND] s0 = 'hello'` - New segment added
- `[REPLACE] s0 = 'hello world'` - Existing segment updated

## Common Issues

1. **Wrong Python interpreter on Windows**: Inkscape's Python 3.12 may be in PATH. Use explicit path to Windows Python.

2. **Module not found**: Install dependencies in the correct Python environment:
   ```bash
   pip install websockets pyaudio
   ```

3. **ASR service not running**: Check Docker containers:
   ```bash
   docker ps | grep vosk
   ```

## File Structure

```
AI-Tools/
├── docker-compose.yaml      # Service orchestration
├── apps/
│   └── live-captions/       # Desktop caption overlay
│       ├── live_captions.py # Main application (v8.0)
│       └── requirements.txt
├── services/
│   ├── vosk/                # CPU ASR service
│   └── parakeet/            # GPU ASR service
└── shared/
    ├── client/
    │   ├── websocket_client.py
    │   └── transcript.py    # ID-based replace/append
    └── config/
        └── backends.py      # BACKEND selection
```
