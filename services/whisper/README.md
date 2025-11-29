# Whisper ASR Service

GPU-accelerated speech recognition using OpenAI Whisper Large V3 Turbo.

## Features

- **Model**: `openai/whisper-large-v3-turbo` - Fast, accurate multilingual ASR
- **GPU Acceleration**: CUDA-optimized with Flash Attention 2
- **Streaming**: WebSocket API with segment-based protocol
- **Text Refinement**: Integration with text-refiner service for punctuation

## API

### Health Check
```
GET /health
```

### Streaming Transcription
```
WebSocket /stream
```

Protocol:
1. Connect to WebSocket
2. Send raw PCM audio (int16, 16kHz, mono)
3. Receive JSON: `{"id": "s1", "text": "transcribed text"}`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `openai/whisper-large-v3-turbo` | HuggingFace model ID |
| `TEXT_REFINER_URL` | `http://text-refiner:8000` | Text refiner service URL |
| `ENABLE_TEXT_REFINER` | `true` | Enable punctuation/correction |

## Resource Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **RAM**: 16GB recommended
- **Disk**: ~3GB for model weights
