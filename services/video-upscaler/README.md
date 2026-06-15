# Video Upscaler Service

GPU-accelerated video super-resolution (clarity enhancement) using **Real-ESRGAN**.
Designed for **offline batch processing** on the RTX 5070 Ti (Blackwell, sm_120).

## Why Real-ESRGAN?

| Engine | Deployable on Blackwell / torch 2.9 | Quality | Notes |
|--------|-------------------------------------|---------|-------|
| **Real-ESRGAN** (this service) | ✅ pure PyTorch | High | Per-frame; robust, fast to deploy |
| RealBasicVSR / BasicVSR++ | ❌ needs mmcv-full/mmediting (won't build on torch 2.9/sm_120) | Higher (temporal) | Impractical to containerize here |

Real-ESRGAN runs on PyTorch 2.9.1 + CUDA 12.8, the same stack proven for the
FastConformer service on this GPU.

## Models

| Model | Scale | Best for |
|-------|-------|----------|
| `realesr-general-x4v3` *(default)* | 4x | Real video; adjustable **denoise** strength |
| `RealESRGAN_x4plus` | 4x | Maximum detail (heavier/slower) |
| `RealESRGAN_x4plus_anime_6B` | 4x | Anime / animation (lighter) |
| `RealESRGAN_x2plus` | 2x | Native 2x upscale |

`outscale` (final factor, 1.0–4.0) is independent of the model's native scale.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service + GPU status |
| GET | `/info` | Configuration |
| GET | `/models` | Available models |
| POST | `/upscale` | Upload video → `{job_id}` (returns immediately) |
| GET | `/jobs` | List jobs |
| GET | `/jobs/{id}` | Job status + progress (`progress` 0–1) |
| GET | `/jobs/{id}/download` | Download upscaled MP4 |
| DELETE | `/jobs/{id}` | Cancel a queued/running job |

### Example

```bash
# Submit
curl -F file=@input.mp4 -F model=realesr-general-x4v3 -F outscale=4 \
     http://localhost:8005/upscale
# -> {"job_id":"ab12cd34ef56","status":"queued"}

# Poll
curl http://localhost:8005/jobs/ab12cd34ef56

# Download when status == "done"
curl -OJ http://localhost:8005/jobs/ab12cd34ef56/download
```

## How it works

1. FFmpeg extracts frames to PNG.
2. Real-ESRGAN upscales each frame on the GPU (tiled to fit 16 GB VRAM).
3. FFmpeg reassembles H.264 (CRF 16) and muxes the **original audio** back in.

A single background worker processes one video at a time so VRAM stays bounded.
`tile` (default 512) caps VRAM per frame — lower it if you hit out-of-memory on
4K sources, set `0` to disable tiling for small clips (faster).

## Run

The service is wired into `docker-compose.yaml` on **port 8005**:

```bash
docker compose up -d video-upscaler
docker compose exec video-upscaler bash download_models.sh   # one-time weight download
```

## Tests

```bash
python -m pytest services/video-upscaler -v
```
