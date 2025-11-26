# AI-Tools

Docker-based AI services for voice transcription, translation, and local LLM.

## Services

| Service | Port | Description |
|---------|------|-------------|
| `whisper-asr` | 8001 | Faster-Whisper ASR (large-v3, GPU) |
| `voice-api` | 8000 | Voice orchestration API |
| `ollama` | 11434 | Local LLM runtime |
| `lobe-chat` | 3210 | Chat UI for Ollama |
| `pyspark-notebook` | 8888 | Jupyter with ML tools |

## Quick Start

```bash
# Start all services
docker compose up -d

# Check service health
curl http://localhost:8001/health  # Whisper ASR
curl http://localhost:8000/health  # Voice API
```

## GUI Client

See [GUI_README.md](GUI_README.md) for the voice transcription desktop app.

```bash
# Install dependencies
pip install -r requirements_gui.txt

# Launch GUI
python voice_transcription_gui.py
```

## Files

| File | Purpose |
|------|---------|
| `docker-compose.yaml` | Service orchestration |
| `Dockerfile.whisper` | Whisper ASR container |
| `Dockerfile.api` | Voice API container |
| `Dockerfile.pyspark` | Jupyter ML container |
| `whisper_service.py` | ASR service code |
| `api_service.py` | Voice API service code |
| `voice_transcription_gui.py` | Desktop GUI client |
