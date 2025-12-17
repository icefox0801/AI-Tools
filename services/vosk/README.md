# Vosk ASR Service

CPU-optimized speech recognition using Vosk lightweight models.

## Features

- **Model**: `vosk-model-small-en-us-0.15` - Lightweight English model
- **CPU Optimized**: No GPU required, runs on any hardware
- **Streaming**: WebSocket API with real-time transcription
- **Low Latency**: Fast processing with small model footprint
- **Text Refinement**: Integration with text-refiner service for punctuation

## API

### Health Check
```
GET /health
```

Returns service status.

### Streaming Transcription
```
WebSocket /stream
```

Protocol:
1. Connect to WebSocket
2. Send config: `{"chunk_ms": 200}`
3. Send raw PCM audio chunks (int16, 16kHz, mono)
4. Receive JSON: `{"id": "s1", "text": "transcribed text"}`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VOSK_MODEL` | `vosk-model-small-en-us-0.15` | Vosk model directory name |
| `TEXT_REFINER_URL` | `http://text-refiner:8000` | Text refiner service URL |
| `ENABLE_TEXT_REFINER` | `true` | Enable punctuation/correction |

## Resource Requirements

- **CPU**: 2+ cores recommended
- **RAM**: 2GB minimum, 4GB recommended
- **Disk**: ~50MB for small model

## Performance

- **WER**: ~12-15% (higher than GPU models, but very fast)
- **Latency**: ~100-200ms for streaming chunks
- **Memory**: ~1GB RAM
- **Throughput**: Real-time+ on modern CPUs

## Model Details

| Property | Value |
|----------|-------|
| Size | ~40MB compressed |
| Language | English (US) |
| Sample Rate | 16kHz |
| Architecture | Kaldi-based lightweight model |

## References

- [Vosk Documentation](https://alphacephei.com/vosk/)
- [Model Downloads](https://alphacephei.com/vosk/models)
- [GitHub Repository](https://github.com/alphacep/vosk-api)
