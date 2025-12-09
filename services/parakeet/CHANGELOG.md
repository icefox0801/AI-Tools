# Changelog

All notable changes to the Parakeet ASR Service will be documented in this file.

## [1.1] - 2025-12-09

### Added
- Final/partial indicator in streaming responses
- On-demand model loading for faster startup

### Changed
- Health endpoint returns healthy when service ready

## [1.0] - 2025-12-04

### Added
- GPU-accelerated ASR using NVIDIA NeMo Parakeet
- Dual models: TDT for streaming, RNNT for offline
- WebSocket streaming and HTTP file transcription
- Text refinement via text-refiner integration
- GPU memory management with model unload/reload
