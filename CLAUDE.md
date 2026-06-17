# CLAUDE.md - AI Assistant Context

This file provides context for AI assistants working on the AI-Tools codebase.

## Project Summary

AI-Tools is a comprehensive Docker-based AI toolkit featuring:
- **Live Captions**: Desktop app for real-time speech-to-text transcription
- **Audio Notes**: Web UI for audio transcription, summarization, and chat
- **Video Upscaler**: GPU-accelerated video super-resolution using Real-ESRGAN
- **Image Super Resolution**: GPU-accelerated image upscaling with ESRGAN and Stable Diffusion
- **ASR Services**: Vosk (CPU), Parakeet (GPU), Whisper (GPU), FastConformer (GPU)
- **Text Refiner**: Punctuation and correction service
- **Ollama**: Local LLM for summarization and chat

## Architecture

```
┌─────────────────┐     WebSocket      ┌──────────────────────┐
│  Client Apps    │ ◄───────────────► │   ASR Services       │
│                 │   Audio Stream     │                      │
│ • Audio Notes   │   {id, text}       │ • Vosk (:8001)       │
│   (:7860)       │                    │ • Parakeet (:8002)   │
│ • Live Captions │                    │ • Whisper (:8003)    │
└────────┬────────┘                    │ • FastConformer(:8004│
         │ Chat/Summarize              │ • Refiner (:8010)    │
         ▼                             └──────────────────────┘
   ┌─────────────┐
   │ Ollama LLM  │
   │  (:11434)   │
   └─────────────┘

┌──────────────────────────────────────────────────────────────┐
│  Video Upscaler Pipeline (Independent)                       │
├──────────────────────────────────────────────────────────────┤
│ • Video Upload → FastAPI Backend (:8005)                    │
│   - Frame extraction (FFmpeg)                                │
│   - Real-ESRGAN upscaling (4x per frame)                     │
│   - Optional temporal mixing (FFmpeg tmix)                   │
│   - Optional BasicVSR++ (temporal-aware, 4x fixed)           │
│   - Frame reassembly + audio mux (FFmpeg)                    │
│ • Gradio UI (:7861) → Poll job status → Download            │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  Image Super Resolution Pipeline (Independent)                │
├──────────────────────────────────────────────────────────────┤
│ • Image Upload → FastAPI Backend (:8006)                      │
│   - Real-ESRGAN upscaling (up to 4x)                          │
│   - Optional Stable Diffusion x2/x4 creative upscale           │
│ • Gradio UI (:7862) → 3-column layout (settings/upload/result)│
└──────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
AI-Tools/
├── apps/live-captions/          # Desktop app (Python + PyQt)
│   ├── live_captions.py         # Main caption window
│   ├── live_captions_tray.py    # System tray launcher
│   ├── src/audio/               # Audio capture (WASAPI)
│   ├── src/asr/                 # ASR client
│   ├── src/ui/                  # UI components
│   └── tests/                   # Unit tests
├── services/
│   ├── audio-notes/             # Web UI (Gradio + FastAPI)
│   │   ├── audio_notes.py       # Entry point
│   │   ├── ui/                  # Gradio components
│   │   ├── services/            # Business logic
│   │   └── api/                 # REST endpoints
│   ├── video-upscaler/          # FastAPI backend for video super-resolution
│   │   ├── upscaler_service.py  # FastAPI application
│   │   ├── pipeline.py          # Real-ESRGAN, BasicVSR++, FFmpeg pipelines
│   │   ├── jobs.py              # Job queue manager
│   │   ├── download_models.sh   # Pre-download RealESRGAN weights
│   │   └── tests/               # Unit tests
│   ├── video-upscaler-ui/       # Gradio frontend for video upscaler
│   │   └── app.py               # UI with model selection, tile controls
│   ├── image-superres/          # FastAPI backend for image super-resolution
│   │   ├── image_service.py     # FastAPI application
│   │   ├── image_model.py       # ESRGAN + Stable Diffusion models
│   │   ├── image_generative.py  # Stable Diffusion pipeline
│   │   └── download_models.sh   # Pre-download model weights
│   ├── image-superres-ui/       # Gradio frontend for image upscaling
│   │   └── app.py               # 3-column UI with presets and GAI controls
│   ├── parakeet/                # NeMo Parakeet ASR (streaming + offline)
│   ├── whisper/                 # OpenAI Whisper ASR
│   ├── vosk/                    # Vosk ASR (CPU-based)
│   ├── fastconformer/           # NVIDIA FastConformer ASR (GPU)
│   └── text-refiner/            # Punctuation & correction service
├── shared/
│   ├── client/                  # WebSocket client library
│   ├── config/                  # Backend configuration
│   └── text_refiner/            # Text refiner client
└── integration/e2e/             # End-to-end tests
```

## Testing

### Test Organization

**Unit tests are co-located with their code:**
```
apps/live-captions/tests/        # Live Captions unit tests
services/audio-notes/tests/      # Audio Notes unit tests
services/parakeet/test_*.py      # Parakeet unit tests
shared/client/tests/             # Shared library tests
integration/e2e/                 # E2E tests ONLY
```

**Never put unit tests in `integration/`!**

### Running Tests

```bash
# All tests
python -m pytest apps/ services/ shared/ integration/ -v

# Unit tests only (no Docker)
python -m pytest apps/ services/ shared/ -v

# Specific component
python -m pytest apps/live-captions -v
python -m pytest services/audio-notes -v

# E2E tests (requires Docker)
python -m pytest integration/e2e -v -m e2e

# With coverage
python -m pytest --cov=apps --cov=services --cov=shared
```

## WebSocket Protocol

ASR services send JSON messages:
```json
{"id": "s0", "text": "hello world"}
```

Client logic (TranscriptManager):
- If `id` exists → REPLACE text
- If `id` is new → APPEND segment

## Common Commands

```bash
# Start services
docker compose up -d audio-notes ollama
docker compose up -d whisper-asr parakeet-asr vosk-asr fastconformer-asr
docker compose up -d video-upscaler video-upscaler-ui
docker compose up -d image-superres image-superres-ui

# Run Live Captions
cd apps/live-captions
python live_captions.py --backend whisper --system-audio --debug

# Check code quality
ruff check apps/ services/ shared/
black apps/ services/ shared/

# Build Live Captions executable
cd apps/live-captions
python -m PyInstaller live_captions.spec
```

## Git Commit Format

Use Conventional Commits:
```
<type>(<scope>): <description>

Types: feat, fix, refactor, chore, docs, test
Scope: live-captions, audio-notes, video-upscaler, image-superres, parakeet, whisper, vosk, fastconformer, text-refiner
```

Examples:
```
feat(live-captions): Add language selector
fix(audio-notes): Fix upload timeout
chore(parakeet): Bump model version
```

## Version Bump Checklist

1. Update version in source files:
   - `apps/live-captions/live_captions_tray.py`: `APP_VERSION`
   - `apps/live-captions/live_captions.py`: description

2. Update CHANGELOG.md (user-facing changes only)

3. Run tests: `python -m pytest apps/live-captions -v`

4. Commit: `chore(live-captions): Bump version to vX.Y`

## Changelog Guidelines

Only include **user-visible changes**:

✅ Include:
- New UI features
- Behavior changes
- User-facing bug fixes

❌ Exclude:
- Internal refactoring
- Performance tweaks
- Test changes
- Background reliability fixes

## Parakeet Model Configuration

```yaml
# docker-compose.yaml
environment:
  - PARAKEET_STREAMING_MODEL=nvidia/parakeet-tdt-1.1b
  - PARAKEET_OFFLINE_MODEL=nvidia/parakeet-rnnt-1.1b
```

TDT is preferred for streaming (better chunk boundary handling).

## Known Issues

1. **Wrong Python on Windows**: Inkscape's Python may be in PATH. Use explicit `.venv` path.

2. **Module not found**: Install in correct environment:
   ```bash
   .venv/Scripts/pip install websockets pyaudio
   ```

3. **ASR not responding**: Check Docker:
   ```bash
   docker ps | grep asr
   docker logs parakeet-asr
   ```
