# Changelog

All notable changes to the FastConformer ASR Service will be documented in this file.

## [1.0] - 2025-12-17

### Added
- GPU-accelerated streaming ASR with NVIDIA FastConformer model
- WebSocket streaming endpoint for real-time transcription
- Configurable latency modes (0ms to 1040ms)
- Native punctuation and capitalization
- Silence detection to skip non-speech audio

### Performance
- 5.4% Word Error Rate (best streaming accuracy)
- 240ms average latency, 480ms worst-case
- 2-3GB GPU memory (FP16)
