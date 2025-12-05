# Changelog

All notable changes to Live Captions will be documented in this file.

## [1.2] - 2025-12-05

### Added
- Language support with English and Cantonese (粵語)
- Language selector in system tray menu
- Language indicator in caption window
- Language parameter for CLI (`--language en|yue`)
- Unit tests for language functionality

### Changed
- Use .venv Python instead of system PATH
- Build script now stops running exe before rebuilding
- Removed unused run scripts (run.bat, run_tray.bat)

### Fixed
- Hardcoded system paths replaced with portable paths
- PyInstaller spec file uses dynamic path resolution

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

