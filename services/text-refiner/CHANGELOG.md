# Changelog

All notable changes to the Text Refiner Service will be documented in this file.

## [1.0] - 2025-12-04

### Added
- Punctuation restoration using punctuators ONNX model (CPU efficient)
- ASR error correction using T5-based seq2seq model (GPU accelerated)
- Full processing via `/process` endpoint (punctuation + correction)
- Health check endpoint `/health`
- Service info endpoint `/info` with API version
- Lazy model loading on first request for faster startup
- GPU memory check before loading correction model
- Batch processing support for multiple texts
