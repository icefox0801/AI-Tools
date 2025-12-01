# Text Refiner Service

GPU-accelerated text refinement for ASR output:
- **Punctuation restoration** (punctuators ONNX - fast, ~10ms)
- **ASR error correction** (T5-based - ~50-100ms)

## Architecture

```
Audio → ASR Service → Text Refiner Service → Client
        (raw text)    (punctuation + correction)
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/process` | POST | Full pipeline: punctuation + correction |
| `/punctuate` | POST | Punctuation only (fast path) |
| `/correct` | POST | Error correction only |
| `/batch` | POST | Batch processing multiple texts |
| `/stream` | WebSocket | Streaming with context buffering |
| `/health` | GET | Health check |
| `/info` | GET | Service info |

## Usage

### REST API

```bash
# Full processing
curl -X POST http://localhost:8010/process \
  -H "Content-Type: application/json" \
  -d '{"text": "i went to the store yesterday and brought some apples"}'

# Response:
# {
#   "text": "I went to the store yesterday and bought some apples.",
#   "original": "i went to the store yesterday and brought some apples",
#   "punctuated": true,
#   "corrected": true,
#   "latency_ms": 85.3
# }

# Punctuation only (faster)
curl -X POST http://localhost:8010/punctuate \
  -H "Content-Type: application/json" \
  -d '{"text": "hello world how are you"}'
```

### WebSocket Streaming

```python
import websockets
import json

async with websockets.connect("ws://localhost:8010/stream") as ws:
    # Send config
    await ws.send(json.dumps({"punctuate": True, "correct": True}))
    config_response = await ws.recv()
    
    # Stream text
    await ws.send(json.dumps({
        "text": "some asr output",
        "segment_id": "seg_001",
        "final": True
    }))
    result = await ws.recv()
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PUNCTUATION_MODEL` | `pcs_en` | Punctuators ONNX model |
| `CORRECTION_MODEL` | `oliverguhr/spelling-correction-english-base` | T5 correction model |
| `ENABLE_CORRECTION` | `true` | Enable/disable correction |
| `CORRECTION_MIN_WORDS` | `4` | Min words before applying correction |

## Models

### Punctuation
- **pcs_en** (~100MB) - English punctuation, capitalization, segmentation

### Correction Options
- `oliverguhr/spelling-correction-english-base` (~900MB) - Fast spelling correction
- `Bhuvana/t5-base-spellchecker` - Alternative spelling
- `vennify/t5-base-grammar-correction` - Grammar-focused
- `grammarly/coedit-large` - State-of-art (larger, slower)

## Performance

| Operation | Latency | GPU Memory |
|-----------|---------|------------|
| Punctuation only | ~10ms | ~100MB |
| Correction only | ~50-100ms | ~2GB |
| Full pipeline | ~60-110ms | ~2.1GB |

## Docker

```bash
# Build
docker compose build text-refiner

# Run
docker compose up -d text-refiner

# Logs
docker logs -f text-refiner
```
