# Changelog

## [1.1] - 2025-12-09
- Added final/partial indicator in streaming responses
- Added on-demand model loading for faster startup
- Changed health endpoint to return healthy when service ready

## [1.0] - 2025-12-04
- Initial release with GPU-accelerated NVIDIA NeMo Parakeet
- Dual models (TDT for streaming, RNNT for offline)
- WebSocket streaming and HTTP file transcription
- Text refinement integration
- GPU memory management with model unload/reload
