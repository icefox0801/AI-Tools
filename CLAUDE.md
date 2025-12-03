# CLAUDE.md - AI Assistant Context for AI-Tools

## Project Overview

AI-Tools is a Docker-based AI toolkit for real-time speech-to-text, audio transcription, and LLM-powered summarization. It includes ASR services, a web UI for audio analysis, and a desktop live captions app.

## Architecture

```
+---------------------+     WebSocket      +---------------------+
|   Client Apps       | -----------------> |   ASR Services      |
|                     |    Audio Stream    |                     |
|  - Audio Notes      | <----------------- |  - Vosk (CPU:8001)  |
|    (Web UI :7860)   |    {id, text}      |  - Parakeet (:8002) |
|  - Live Captions    |                    |  - Whisper (:8003)  |
|    (Desktop)        |                    |                     |
+---------------------+                    |  Text Refiner:8010  |
        |                                  +---------------------+
        |
        +----------------> Ollama LLM (:11434)
              Chat &       (Qwen, Llama, etc.)
              Summarize
```

## Key Components

### Web Applications (Docker)
- **Audio Notes** (`services/audio-notes/`) - Gradio web UI, port 7860
  - Upload/record audio files for transcription
  - ASR backend selection (Whisper or Parakeet)
  - AI summarization with local LLM (Ollama)
  - Chat about transcript content with context
  - Batch transcription support

### ASR Services (Docker)
- **Vosk** (`services/vosk/`) - CPU-based, lightweight, port 8001
- **Parakeet** (`services/parakeet/`) - GPU-based (NVIDIA NeMo TDT), port 8002
  - Uses TDT (Token-and-Duration Transducer) model - better for streaming
  - 300ms sliding window overlap for word boundary handling
  - FP16 inference enabled for performance
- **Whisper** (`services/whisper/`) - GPU-based (OpenAI Whisper Large V3 Turbo), port 8003
- **Text Refiner** (`services/text-refiner/`) - Punctuation & ASR correction, port 8010

### Client Applications
- **Live Captions** (`apps/live-captions/`) - Desktop overlay with microphone capture

### Shared Library (`shared/`)
- `shared/client/websocket_client.py` - WebSocket client for ASR streaming
- `shared/client/transcript.py` - TranscriptManager with ID-based replace/append
- `shared/config/backends.py` - Backend configuration (vosk, parakeet, or whisper)
- `shared/text_refiner/` - Text refiner client module for punctuation & correction

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
# Start Audio Notes with dependencies
docker compose up -d audio-notes ollama

# Start ASR services (choose one)
docker compose up -d vosk-asr      # CPU
docker compose up -d parakeet-asr  # GPU
docker compose up -d whisper-asr   # GPU (multilingual)

# Access Audio Notes
open http://localhost:7860

# Run Live Captions with debug logging
cd apps/live-captions
python live_captions.py --debug

# IMPORTANT: Use correct Python on Windows (not Inkscape's Python)
"C:\Users\icefo\AppData\Local\Microsoft\WindowsApps\python.exe" apps/live-captions/live_captions.py --debug
```

## Configuration

Backend selection in `shared/config/backends.py`:
```python
BACKEND = "vosk"      # CPU, lightweight
BACKEND = "parakeet"  # GPU, high accuracy
BACKEND = "whisper"   # GPU, multilingual
```

Or via command line: `--backend vosk|parakeet|whisper`

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
│       ├── live_captions.py       # Main caption window
│       ├── live_captions_tray.py  # System tray launcher
│       ├── build_tray.bat         # Build Windows executable
│       └── requirements.txt
├── services/
│   ├── audio-notes/         # Gradio Web UI for transcription & summarization
│   │   ├── audio_notes.py         # FastAPI entry point
│   │   ├── ui/                    # Gradio UI components
│   │   ├── services/              # LLM, recordings, ASR services
│   │   └── api/                   # REST API endpoints
│   ├── vosk/                # CPU ASR service
│   ├── parakeet/            # GPU ASR service (NVIDIA NeMo TDT)
│   ├── whisper/             # GPU ASR service (OpenAI Whisper)
│   └── text-refiner/        # Punctuation & correction service
└── shared/
    ├── client/
    │   ├── websocket_client.py
    │   └── transcript.py    # ID-based replace/append
    ├── config/
    │   └── backends.py      # BACKEND selection
    └── text_refiner/        # Text refiner client module
```

## Parakeet Model Configuration

Parakeet model can be configured via `.env` or docker-compose.yaml:
```bash
PARAKEET_MODEL=nvidia/parakeet-tdt-1.1b  # TDT - better for streaming
PARAKEET_MODEL=nvidia/parakeet-rnnt-1.1b # RNNT - alternative model
PARAKEET_FP16=true                        # Enable FP16 for TDT (required)
```

TDT (Token-and-Duration Transducer) is preferred for streaming because it handles
chunk boundaries better than RNNT.
