# Whisper ASR Service

GPU-accelerated speech recognition using OpenAI Whisper Large V3 Turbo.

## Features

- **Model**: `openai/whisper-large-v3-turbo` - Fast, accurate multilingual ASR
- **GPU Acceleration**: CUDA-optimized with Flash Attention 2
- **Streaming**: WebSocket API with segment-based protocol
- **Multilingual**: Supports 99+ languages
- **Text Refinement**: Integration with text-refiner service for punctuation

## API

### Health Check
```
GET /health
```

Returns model status and GPU info.

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
| `DEVICE` | `cuda` | Device: `cuda` or `cpu` (auto-detected) |

## Resource Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **RAM**: 16GB recommended
- **Disk**: ~3GB for model weights

## Performance

- **WER**: ~8-10% (multilingual average)
- **Latency**: ~500-800ms for streaming segments
- **VRAM**: ~6-8GB
- **Languages**: 99+ supported

## Model Details

| Property | Value |
|----------|-------|
| Parameters | 809M |
| Architecture | Transformer encoder-decoder |
| Training Data | 680,000 hours multilingual |
| Sample Rate | 16kHz |
| Context Length | 30 seconds |

## References

- [Model Card](https://huggingface.co/openai/whisper-large-v3-turbo)
- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [OpenAI Whisper](https://github.com/openai/whisper)
