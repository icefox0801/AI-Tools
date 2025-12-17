# FastConformer ASR Service

GPU-accelerated streaming speech recognition using NVIDIA FastConformer Hybrid model.

## Features

- **Model**: `nvidia/stt_en_fastconformer_hybrid_large_streaming_multi` - 114M params, streaming-optimized
- **Architecture**: Cache-aware FastConformer (Conformer + Transducer hybrid)
- **GPU Acceleration**: CUDA-optimized with FP16 inference
- **Streaming**: WebSocket API with low-latency incremental decoding
- **Native Punctuation**: Built-in punctuation, no external refiner needed
- **Configurable Latency**: 0ms, 80ms, 480ms, 1040ms modes

## API

### Health Check
```
GET /health
```

Returns model status and GPU memory usage.

### Model Info
```
GET /info
```

Returns service and model configuration details.

### Unload Model
```
POST /unload
```

Unload model from GPU to free memory.

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
| `FASTCONFORMER_MODEL` | `nvidia/stt_en_fastconformer_hybrid_large_streaming_multi` | NeMo model ID |
| `DECODER_TYPE` | `rnnt` | Decoder: `rnnt` (better accuracy) or `ctc` (faster) |
| `ATT_CONTEXT_SIZE` | `[70,6]` | Latency mode: `[70,0]`=0ms, `[70,1]`=80ms, `[70,6]`=480ms, `[70,33]`=1040ms |
| `DEVICE` | `cuda` | Device: `cuda` or `cpu` (auto-detected) |

## Resource Requirements

- **GPU**: NVIDIA GPU with 3GB+ VRAM (RTX 30/40/50 series recommended)
- **RAM**: 8GB recommended
- **Disk**: ~3GB for model weights

## Performance

- **WER**: 5.4% (LibriSpeech test-clean, RNNT decoder)
- **Latency**: 480ms worst-case, 240ms average (with `[70,6]` context)
- **VRAM**: ~2-3GB with FP16
- **Throughput**: Real-time streaming with cache-aware incremental decoding

## Model Details

| Property | Value |
|----------|-------|
| Parameters | 114M |
| Training Data | NeMo ASRSET 3.0 (several thousand hours, English) |
| Architecture | FastConformer with cache-aware streaming |
| Decoder | Hybrid RNNT/CTC (multitask training) |
| Sample Rate | 16kHz |
| Input Format | PCM16, mono |

## References

- [Model Card](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi)
- [NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#cache-aware-streaming-conformer)
- [FastConformer Paper](https://arxiv.org/abs/2305.05084)
- [Cache-aware Streaming Paper](https://arxiv.org/abs/2312.17279)
