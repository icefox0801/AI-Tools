# CLAUDE.md - AI Assistant Context

This file provides context for AI assistants working on the AI-Tools codebase.

## Project Summary

AI-Tools is a Docker-based toolkit for speech-to-text and audio analysis:
- **Live Captions**: Desktop app for real-time transcription
- **Audio Notes**: Web UI for transcription, summarization, and chat
- **ASR Services**: Vosk (CPU), Parakeet (GPU), Whisper (GPU)
- **Text Refiner**: Punctuation and correction service
- **Ollama**: Local LLM for summarization and chat

## Architecture

```
┌─────────────────┐     WebSocket      ┌─────────────────┐
│  Client Apps    │ ◄───────────────► │   ASR Services  │
│                 │   Audio Stream     │                 │
│ • Audio Notes   │   {id, text}       │ • Vosk (:8001)  │
│   (:7860)       │                    │ • Parakeet(:8002│
│ • Live Captions │                    │ • Whisper(:8003)│
└────────┬────────┘                    │ • Refiner(:8010)│
         │                             └─────────────────┘
         │ Chat/Summarize
         ▼
   ┌─────────────┐
   │ Ollama LLM  │
   │  (:11434)   │
   └─────────────┘
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
│   ├── parakeet/                # NeMo Parakeet ASR
│   ├── whisper/                 # OpenAI Whisper ASR
│   ├── vosk/                    # Vosk ASR
│   └── text-refiner/            # Punctuation service
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
docker compose up -d whisper-asr parakeet-asr vosk-asr

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
Scope: live-captions, audio-notes, parakeet, whisper, vosk
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
