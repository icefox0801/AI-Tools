# Transcription Gateway Service

Unified API for all speech transcription needs. Routes requests to the optimal model backend based on use case.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Transcription Gateway                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │  /stream        │  │  /transcribe    │  │  /models/*          │ │
│  │  (Real-time)    │  │  (Offline)      │  │  (Management)       │ │
│  └────────┬────────┘  └────────┬────────┘  └─────────┬───────────┘ │
│           │                    │                     │             │
│           ▼                    ▼                     ▼             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Model Router                              │   │
│  │   - Routes to streaming model (TDT) for /stream             │   │
│  │   - Routes to offline model (RNNT/Whisper) for /transcribe  │   │
│  │   - Manages model loading/unloading based on usage          │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌────────────┐      ┌────────────┐      ┌────────────┐
   │  Parakeet  │      │  Whisper   │      │   Vosk     │
   │  (GPU)     │      │  (GPU)     │      │   (CPU)    │
   └────────────┘      └────────────┘      └────────────┘
```

## Benefits

1. **Unified API** - Single endpoint for all transcription needs
2. **Smart Routing** - Automatically routes to the best model for each use case
3. **GPU Memory Management** - Unloads idle models to free memory
4. **Decoupled UI** - Apps only need to know one service
5. **Easy Model Switching** - Change models without updating clients

## API Endpoints

### Health & Info
- `GET /health` - Service health check
- `GET /info` - Service information and available models
- `GET /backends/status` - Check status of all backend services

### Model Management
- `GET /models` - List all available models and their status
- `POST /models/{id}/load` - Load a specific model
- `POST /models/{id}/unload` - Unload a model to free GPU memory

### Transcription
- `WS /stream` - Real-time streaming transcription
- `POST /transcribe` - Offline file transcription

## Streaming Protocol

1. Connect to WebSocket at `/stream`
2. Send config JSON: `{"model": "parakeet-streaming", "chunk_ms": 300}`
3. Stream raw PCM audio bytes (int16, 16kHz, mono)
4. Receive JSON responses: `{"partial": "..."}` or `{"text": "...", "final": true}`
5. Close connection when done

## Available Models

| Model ID | Name | Type | Device | Description |
|----------|------|------|--------|-------------|
| `parakeet-streaming` | Parakeet TDT | Streaming | GPU | NVIDIA NeMo TDT - optimized for real-time |
| `parakeet-offline` | Parakeet RNNT | Offline | GPU | NVIDIA NeMo RNNT - highest accuracy |
| `whisper` | Whisper Large V3 | Both | GPU | OpenAI Whisper - fast multilingual |
| `vosk` | Vosk (CPU) | Streaming | CPU | Lightweight CPU-based ASR |

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `PARAKEET_URL` | `http://parakeet-asr:8000` | Parakeet service URL |
| `WHISPER_URL` | `http://whisper-asr:8000` | Whisper service URL |
| `VOSK_URL` | `http://vosk-asr:8000` | Vosk service URL |
| `DEFAULT_STREAMING_MODEL` | `parakeet-streaming` | Default model for streaming |
| `DEFAULT_OFFLINE_MODEL` | `parakeet-offline` | Default model for offline |
| `IDLE_UNLOAD_SECONDS` | `300` | Unload models after N seconds idle |

## Usage Examples

### Python - Streaming
```python
import asyncio
import websockets
import json

async def stream_audio():
    async with websockets.connect("ws://localhost:8000/stream") as ws:
        # Send config
        await ws.send(json.dumps({"model": "parakeet-streaming", "chunk_ms": 300}))
        
        # Stream audio
        with open("audio.raw", "rb") as f:
            while chunk := f.read(9600):  # 300ms at 16kHz
                await ws.send(chunk)
                response = await ws.recv()
                print(json.loads(response))
```

### Python - Offline
```python
import httpx

with open("audio.wav", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/transcribe",
        files={"file": ("audio.wav", f, "audio/wav")}
    )
    result = response.json()
    print(result["text"])
```

### cURL
```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Transcribe file
curl -X POST -F "file=@audio.wav" http://localhost:8000/transcribe
```

## Docker

```bash
# Build and run
docker compose up -d --build transcription

# View logs
docker logs -f transcription
```
