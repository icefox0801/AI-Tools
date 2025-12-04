# Changelog

All notable changes to the Vosk ASR Service will be documented in this file.

## [1.0] - 2025-12-04

### Added
- Faster real-time captions with async thread pool processing
- Punctuation and ASR error correction for final results via text-refiner service
- Streaming speech recognition via WebSocket `/stream` endpoint
- Health check endpoint `/health`
- Automatic model download on first run
- Lazy model loading on first connection
