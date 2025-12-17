# Parakeet ASR Service

GPU-accelerated speech recognition using NVIDIA Parakeet RNNT models.

## Features

- **Dual Models**: TDT-1.1B for streaming, RNNT-1.1B for offline transcription
- **GPU Acceleration**: CUDA-optimized with FP16 inference
- **Streaming**: WebSocket API with chunk-based processing
- **Offline**: File upload endpoint with word-level timestamps
- **Text Refinement**: Integration with text-refiner service for punctuation

## API

### Health Check
```
GET /health
```

Returns model status, GPU info, and loaded models.

### Streaming Transcription
```
WebSocket /stream
```

Protocol:
1. Connect to WebSocket
2. Send config: `{"chunk_ms": 200, "sample_rate": 16000}`
3. Send raw PCM audio chunks (int16, 16kHz, mono)
4. Send empty chunk `b""` to signal end
5. Receive JSON: `{"text": "...", "partial": true/false, "final": true/false}`

### File Transcription
```
POST /transcribe
Content-Type: multipart/form-data

file: audio file (WAV, MP3, etc.)
```

Response:
```json
{
  "text": "full transcription",
  "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
  "model": "nvidia/parakeet-rnnt-1.1b"
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PARAKEET_STREAMING_MODEL` | `nvidia/parakeet-tdt-1.1b` | Streaming model (TDT preferred) |
| `PARAKEET_OFFLINE_MODEL` | `nvidia/parakeet-rnnt-1.1b` | Offline model with timestamps |
| `TEXT_REFINER_URL` | `http://text-refiner:8000` | Text refiner service URL |
| `ENABLE_TEXT_REFINER` | `true` | Enable punctuation/correction |
| `DEVICE` | `cuda` | Device: `cuda` or `cpu` (auto-detected) |

## Resource Requirements

- **GPU**: NVIDIA GPU with 6GB+ VRAM
- **RAM**: 12GB recommended
- **Disk**: ~5GB for both model weights

## Performance

- **WER**: ~5-7% (LibriSpeech test-clean)
- **Latency**: ~200-300ms for streaming chunks
- **VRAM**: ~4-5GB (both models loaded)
- **Throughput**: Real-time+ for streaming, 10x real-time for offline

## Model Details

| Model | Parameters | Use Case |
|-------|------------|----------|
| Parakeet TDT-1.1B | 1.1B | Streaming (better chunk boundaries) |
| Parakeet RNNT-1.1B | 1.1B | Offline (word timestamps) |

## References

- [Model Card (TDT)](https://huggingface.co/nvidia/parakeet-tdt-1.1b)
- [Model Card (RNNT)](https://huggingface.co/nvidia/parakeet-rnnt-1.1b)
- [NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html)
