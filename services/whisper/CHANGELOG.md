# Changelog

All notable changes to the Whisper ASR Service will be documented in this file.

## [1.3] - 2025-12-17

### Added
- Configurable VAD filter and threshold via environment variables
- Configurable beam size for transcription quality/speed tradeoff
- Configurable chunk timing (duration and minimum audio length)
- Dynamic WebSocket configuration protocol for runtime parameter changes
- Environment-based configuration (WHISPER_VAD_FILTER, WHISPER_VAD_THRESHOLD, WHISPER_BEAM_SIZE, WHISPER_LANGUAGE, WHISPER_CHUNK_DURATION_SEC, WHISPER_MIN_AUDIO_SEC)

### Changed
- WebSocket /stream endpoint accepts configuration messages before audio streaming
- All configuration parameters now sourced from environment variables

### Fixed
- Lambda variable binding in async threading (functools.partial)
- Type hints compliance with PEP 484

## [1.2] - 2025-12-08

### Added
- Dual model support: Turbo for streaming, Large-v3 for offline
- Separate pipelines for streaming and offline transcription modes
- Auto-loading models on first request

### Changed
- Models are now pre-downloaded to cache volume (no network requests)
- Use `local_files_only=True` to prevent runtime downloads
- Removed model download logic from Python code

### Performance
- Faster startup (no model validation/download)
- Guaranteed offline operation after initial setup

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
