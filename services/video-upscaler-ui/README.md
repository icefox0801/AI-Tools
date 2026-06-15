# Video Upscaler Web UI

Gradio web interface for the [`video-upscaler`](../video-upscaler/README.md) API.

Upload a video, choose a model and scale, watch live progress, then preview and
download the enhanced result.

## Features

- Drag-and-drop video upload
- Model selector (fetched live from the backend)
- Output scale (1x–4x) and denoise strength controls
- Tile-size control for VRAM management
- Live progress bar with frame counter
- In-browser preview + download of the upscaled video

## Run

Wired into `docker-compose.yaml` on **port 7861**:

```bash
docker compose up -d video-upscaler video-upscaler-ui
```

Then open http://localhost:7861

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `UPSCALER_URL` | `http://video-upscaler:8000` | Backend API URL |
| `GRADIO_SERVER_PORT` | `7861` | UI port |
| `POLL_INTERVAL_SEC` | `2.0` | Job polling interval |
| `UPLOAD_TIMEOUT_SEC` | `600` | Upload/download timeout for large videos |
