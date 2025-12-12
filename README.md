# AI-Tools

Docker-based AI toolkit for real-time speech-to-text, audio transcription, and LLM-powered summarization.

<img width="291" height="242" alt="image" src="https://github.com/user-attachments/assets/45f07f39-fe52-47bc-abb7-d103326087e2" />

<img width="3080" height="268" alt="image" src="https://github.com/user-attachments/assets/7c3d5e5e-6827-42e1-a6c6-86f944837721" />

<img width="2848" height="982" alt="image" src="https://github.com/user-attachments/assets/47fbc1b1-70c2-45fd-885c-ca000714c1d7" />

## Features

- **Real-time Speech-to-Text**: Multiple ASR backends (Vosk CPU, Parakeet GPU, Whisper GPU)
- **Audio Notes Web App**: Upload recordings, transcribe with AI, summarize with LLM, chat about content
- **Live Captions Desktop App**: Real-time overlay for meetings, videos, and calls
- **Local LLM Integration**: Ollama-powered summarization and chat (Qwen, Llama, etc.)

## Architecture

```
+-------------------+       WebSocket       +---------------------+
|   Client Apps     |  ---------------->    |    ASR Services     |
|                   |     Audio Stream      |    (Docker)         |
| - Audio Notes     |  <----------------    |  - Vosk (CPU)       |
|   (Web UI)        |     Transcripts       |  - Parakeet (GPU)   |
| - Live Captions   |                       |  - Whisper (GPU)    |
|   (Desktop)       |                       +---------------------+
+-------------------+                                |
        |                                            v
        |                               +---------------------+
        +-----------------------------> |   Ollama LLM        |
              Chat & Summarize          |   (Local Models)    |
                                        +---------------------+
```

## Services

| Service | Port | Description |
|---------|------|--------------|
| `audio-notes` | 7860 | Web UI for transcription & summarization |
| `vosk-asr` | 8001 | Vosk streaming ASR (CPU, lightweight) |
| `parakeet-asr` | 8002 | NVIDIA NeMo Parakeet (GPU, high accuracy) |
| `whisper-asr` | 8003 | OpenAI Whisper Large V3 Turbo (GPU, multilingual) |
| `text-refiner` | 8010 | Punctuation & ASR error correction |
| `ollama` | 11434 | Local LLM runtime |
| `lobe-chat` | 3210 | Chat UI for Ollama |
| `pyspark-notebook` | 8888 | Jupyter with ML tools |

## Quick Start

```bash
# Start Audio Notes (includes ASR + LLM services)
docker compose up -d audio-notes ollama

# Or start individual ASR services
docker compose up -d vosk-asr      # CPU-based (lightweight)
docker compose up -d parakeet-asr  # GPU-based (high accuracy)
docker compose up -d whisper-asr   # GPU-based (multilingual)

# Access Audio Notes web UI
open http://localhost:7860

# Check service health
curl http://localhost:8001/health  # Vosk ASR
curl http://localhost:8002/health  # Parakeet ASR
curl http://localhost:8003/health  # Whisper ASR
```

## Audio Notes (Web App)

Full-featured web UI for audio transcription and AI-powered analysis.

**Features:**
- Upload audio files or use recordings from Live Captions
- Choose ASR backend (Whisper or Parakeet)
- AI summarization with local LLM (Ollama)
- Chat about transcript content
- Batch transcription support

Access at: http://localhost:7860

## Live Captions (Desktop App)

Real-time speech-to-text overlay for your desktop.

```bash
cd apps/live-captions
pip install -r requirements.txt
python live_captions.py --debug
```

See [apps/live-captions/README.md](apps/live-captions/README.md) for details.

## Project Structure

```
AI-Tools/
├── pyproject.toml           # Project config, linting, testing
├── docker-compose.yaml      # Service orchestration
├── apps/
│   └── live-captions/       # Desktop caption overlay (v1.0)
│       └── src/             # Audio, ASR, UI modules
├── services/
│   ├── audio-notes/         # Web UI for transcription (v1.0)
│   ├── vosk/                # CPU ASR (Vosk)
│   ├── parakeet/            # GPU ASR (NVIDIA NeMo)
│   ├── whisper/             # GPU ASR (OpenAI Whisper)
│   ├── text-refiner/        # Punctuation & correction
│   └── pyspark/             # Jupyter ML environment
├── shared/
│   ├── client/              # WebSocket client library
│   ├── config/              # Backend selection
│   ├── logging/             # Centralized logging
│   └── text_refiner/        # Text refiner client module
└── tests/                   # E2E tests and fixtures
```

## Testing

```bash
# Run all tests (unit tests are co-located with their modules)
python -m pytest apps/ services/ shared/ integration/ -v

# Run with coverage
python -m pytest --cov=apps --cov=services --cov=shared
```

## Configuration

Select ASR backend in `shared/config/backends.py`:

```python
BACKEND = "vosk"      # CPU-based, lightweight
# BACKEND = "parakeet"  # GPU-based, high accuracy
# BACKEND = "whisper"   # GPU-based, multilingual
```

Or use command line:
```bash
python live_captions.py --backend whisper --system-audio
```



