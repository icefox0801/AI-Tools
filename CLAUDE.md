# CLAUDE.md - AI Assistant Context for AI-Tools

## Project Overview

AI-Tools is a Docker-based AI toolkit for real-time speech-to-text, audio transcription, and LLM-powered summarization. It includes ASR services, a web UI for audio analysis, and a desktop live captions app.

**Version**: 1.0.0  
**Python**: 3.11+  
**Test Coverage**: 321 unit tests across 21 test files

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
- **Live Captions** (`apps/live-captions/`) - Desktop overlay with system tray
  - Real-time transparent overlay window
  - WASAPI loopback for system audio capture
  - Microphone input support
  - Recording with auto-upload to Audio Notes
  - System tray app with backend selection

### Shared Library (`shared/`)
- `shared/client/websocket_client.py` - WebSocket client for ASR streaming
- `shared/client/transcript.py` - TranscriptManager with ID-based replace/append
- `shared/config/backends.py` - Backend configuration (vosk, parakeet, or whisper)
- `shared/text_refiner/` - Text refiner client module for punctuation & correction
- `shared/logging/` - Centralized logging configuration

## Testing

Run all tests:
```bash
python -m pytest apps/ services/ tests/ -v
```

Run specific test suites:
```bash
# Live Captions (154 tests)
python -m pytest apps/live-captions -v

# Audio Notes (70 tests)
python -m pytest services/audio-notes -v

# ASR Services
python -m pytest services/parakeet services/whisper services/vosk -v

# E2E tests (require Docker services)
python -m pytest tests/e2e -v -m e2e
```

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
├── pyproject.toml           # Project config, ruff, pytest settings
├── docker-compose.yaml      # Service orchestration
├── apps/
│   └── live-captions/       # Desktop caption overlay (v1.0)
│       ├── live_captions.py       # Main caption window
│       ├── live_captions_tray.py  # System tray launcher
│       ├── src/
│       │   ├── audio/             # Audio capture (WASAPI, microphone)
│       │   ├── asr/               # ASR client
│       │   └── ui/                # Caption window UI
│       ├── scripts/               # Build and run scripts
│       └── test_*.py              # Unit tests (154 tests)
├── services/
│   ├── audio-notes/         # Gradio Web UI (v1.0)
│   │   ├── audio_notes.py         # FastAPI entry point
│   │   ├── ui/                    # Gradio UI components
│   │   ├── services/              # LLM, recordings, ASR services
│   │   ├── api/                   # REST API endpoints
│   │   └── test_*.py              # Unit tests (70 tests)
│   ├── vosk/                # CPU ASR service
│   ├── parakeet/            # GPU ASR service (NVIDIA NeMo TDT)
│   ├── whisper/             # GPU ASR service (OpenAI Whisper)
│   └── text-refiner/        # Punctuation & correction service
├── shared/
│   ├── client/              # WebSocket client, transcript manager
│   ├── config/              # Backend configuration
│   ├── logging/             # Centralized logging
│   └── text_refiner/        # Text refiner client
└── tests/
    ├── e2e/                 # End-to-end tests
    └── fixtures/            # Test audio files
```

## Code Quality

Lint with ruff:
```bash
ruff check .        # Check all files
ruff check --fix .  # Auto-fix issues
```

Key lint rules enabled: E, W, F, I, B, C4, UP, SIM, RUF

## Parakeet Model Configuration

Parakeet models are configured in docker-compose.yaml:
```yaml
environment:
  - PARAKEET_STREAMING_MODEL=nvidia/parakeet-tdt-1.1b  # TDT - better for streaming
  - PARAKEET_OFFLINE_MODEL=nvidia/parakeet-rnnt-1.1b   # RNNT - better accuracy
```

TDT (Token-and-Duration Transducer) is preferred for streaming because it handles
chunk boundaries better than RNNT.
