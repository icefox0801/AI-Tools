# Changelog

All notable changes to Live Captions will be documented in this file.

## [1.6] - 2025-12-17

### Added
- FastConformer backend support (NVIDIA FastConformer Hybrid, 114M params)
- Intelligent streaming with batched fallback for all backends
- PySpark Docker container with JupyterLab for ASR benchmarking
- ASR benchmark notebook for streaming performance analysis

### Fixed
- Backend selection menu now properly switches models
- Transcript shifting issue after ~2 minutes (truncation to 300 words)
- Menu checked state synchronization with current backend

### Changed
- Bind mount configuration for notebooks (local/Docker sync)
- Improved backend detection and auto-restart logic

## [1.5] - 2025-12-11

### Added
- Auto-start at login feature (toggle in system tray menu)

## [1.4] - 2025-12-10

### Changed
- Simplified tray menu layout
- Audio source icons (speaker/microphone)

### Fixed
- Tray Stop button now works correctly

## [1.3] - 2025-12-08

### Added
- Version display in tray menu
- Double-click to start/stop
- No-audio detection prompt

## [1.2] - 2025-12-05

### Added
- Language support (English and Cantonese)
- Language selector in tray menu

## [1.1] - 2025-12-05

### Added
- Animated tray icon when captions are active
- Single-click tray icon to start/stop
- Recording timer in caption window

## [1.0] - 2025-12-04

### Added
- Real-time speech-to-text overlay window
- System audio capture (WASAPI loopback) and microphone input
- Multi-backend ASR support (Whisper, Parakeet, Vosk)
- System tray application with backend health indicators
- Audio recording with auto-upload to Audio Notes
- High DPI support, text resizing, draggable window
- Standalone executable via PyInstaller

