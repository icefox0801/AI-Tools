# Changelog

## [1.3] - 2025-12-17
- Added configurable VAD filter and threshold via environment variables
- Added configurable beam size for transcription quality/speed tradeoff
- Added configurable chunk timing (duration and minimum audio length)
- Added dynamic WebSocket configuration protocol for runtime parameter changes
- Fixed lambda variable binding in async threading
- Fixed type hints compliance with PEP 484

## [1.2] - 2025-12-08
- Added dual model support (Turbo for streaming, Large-v3 for offline)
- Added separate pipelines for streaming and offline transcription
- Added auto-loading models on first request
- Faster startup with pre-downloaded models

## [1.1] - 2025-12-05

### Added
- Language parameter support for multilingual transcription

## [1.0] - 2025-12-04

### Added
- GPU-accelerated ASR using OpenAI Whisper Large V3 Turbo
- Silero VAD for speech detection (skips silence)
- WebSocket streaming and HTTP file transcription
- Native punctuation and capitalization
- GPU memory management with model unload/reload
