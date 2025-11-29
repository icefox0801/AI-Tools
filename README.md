# AI-Tools

Docker-based AI services for real-time voice transcription and local LLM.

## Architecture

```
┌─────────────────────┐     WebSocket      ┌─────────────────────┐
│   Client Apps       │ ──────────────────▶│   ASR Services      │
│                     │    Audio Stream    │   (Docker)          │
│  • Live Captions    │ ◀──────────────────│  • Vosk (CPU)       │
│    (Desktop)        │    Transcripts     │  • Parakeet (GPU)   │
└─────────────────────┘                    │  • Whisper (GPU)    │
                                           └─────────────────────┘
```

## Services

| Service | Port | Description |
|---------|------|--------------|
| `vosk-asr` | 8001 | Vosk streaming ASR (CPU, lightweight) |
| `parakeet-asr` | 8002 | NVIDIA NeMo Parakeet (GPU, high accuracy) |
| `whisper-asr` | 8004 | OpenAI Whisper Large V3 Turbo (GPU, multilingual) |
| `text-refiner` | 8003 | Punctuation & ASR error correction |
| `ollama` | 11434 | Local LLM runtime |
| `lobe-chat` | 3210 | Chat UI for Ollama |
| `pyspark-notebook` | 8888 | Jupyter with ML tools |

## Quick Start

```bash
# Start ASR service (choose one)
docker compose up -d vosk-asr      # CPU-based (lightweight)
docker compose up -d parakeet-asr  # GPU-based (high accuracy)
docker compose up -d whisper-asr   # GPU-based (multilingual)

# Check service health
curl http://localhost:8001/health  # Vosk ASR
curl http://localhost:8002/health  # Parakeet ASR
curl http://localhost:8004/health  # Whisper ASR
```

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
├── docker-compose.yaml      # Service orchestration
├── apps/
│   └── live-captions/       # Desktop caption overlay
├── services/
│   ├── vosk/                # CPU ASR (Vosk)
│   ├── parakeet/            # GPU ASR (NVIDIA NeMo)
│   ├── whisper/             # GPU ASR (OpenAI Whisper)
│   ├── text-refiner/        # Punctuation & correction
│   └── pyspark/             # Jupyter ML environment
└── shared/
    ├── client/              # WebSocket client library
    │   ├── websocket_client.py
    │   └── transcript.py    # ID-based transcript manager
    ├── config/
    │   └── backends.py      # Backend selection
    └── text_refiner/        # Text refiner client module
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
