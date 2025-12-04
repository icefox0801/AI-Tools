# Changelog

All notable changes to the Whisper ASR Service will be documented in this file.

## [1.0] - 2025-12-04

### Added
- GPU-accelerated streaming ASR using OpenAI Whisper Large V3 Turbo (809M params)
- Silero VAD integration for speech detection (skips silence, reduces GPU load)
- Flash Attention 2 / SDPA support for optimized inference
- Streaming speech recognition via WebSocket `/stream` endpoint
- File transcription via HTTP POST `/transcribe` endpoint
- Native punctuation and capitalization (no external refiner needed)
- Model unload/reload for GPU memory management (`/unload` endpoint)
- Health check endpoint `/health` with GPU and VAD status
- Service info endpoint `/info` with API version
