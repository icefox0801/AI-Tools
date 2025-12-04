# Changelog

All notable changes to the Parakeet ASR Service will be documented in this file.

## [1.0] - 2025-12-04

### Added
- GPU-accelerated streaming ASR using NVIDIA NeMo Parakeet models
- Dual model architecture: TDT for streaming (FP16), RNNT for offline (FP32)
- Cache-aware conformer encoder for incremental output without duplicates
- Long audio chunking with overlap for files >20s (NeMo ~20s limit)
- Streaming speech recognition via WebSocket `/stream` endpoint
- File transcription via HTTP POST `/transcribe` endpoint with word timestamps
- Text refinement integration via text-refiner service
- Model unload/reload for GPU memory management (`/unload` endpoint)
- Health check endpoint `/health` with GPU and model status
- Service info endpoint `/info` with API version
- Extracted audio utilities to `audio.py` module
- Extracted model management to `model.py` module
