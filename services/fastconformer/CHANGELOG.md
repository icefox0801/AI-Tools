# Changelog

All notable changes to the FastConformer ASR Service will be documented in this file.

## [1.1] - 2025-12-17

### Added
- Intelligent streaming with automatic API detection
- Batched transcription fallback (accumulates ~1 second chunks)
- Live Captions tray menu integration

### Changed
- Streaming now checks for native `transcribe_step()` API
- Falls back to batched `transcribe()` when native streaming unavailable
- Improved warmup using standard `transcribe()` method
- Enhanced streaming state management with cache support

### Performance
- Better latency handling with 1-second batch accumulation
- Optimized for FastConformer-Hybrid-Transducer-CTC-BPE (114M params)

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
