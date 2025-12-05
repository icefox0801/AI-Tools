# Changelog

All notable changes to Live Captions will be documented in this file.

## [1.1] - 2025-12-05

### Added
- Animated tray icon with 3-frame vertical loading bars when running
- Single left-click to start/stop (right-click for menu)
- Recording status indicator (🔴 REC 00:00) in caption window
- Unit tests for CaptionWindow class

### Changed
- Simplified headless mode (recording-only) to use simple loop
- Improved subprocess error monitoring in tray app

### Removed
- Unused RecordingOverlay module
- Unused MiniOverlay module

## [1.0] - 2025-12-04

### Added
- Real-time speech-to-text overlay window with transparent background
- Microphone audio capture with device selection
- System audio capture via WASAPI loopback for perfect quality
- Multi-backend ASR support (Whisper, Parakeet, Vosk)
- WebSocket streaming to ASR services
- System tray application with right-click menu
- Backend health checking with status indicators
- Audio recording with automatic upload to Audio Notes
- Recording-only mode (transcription can be disabled)
- Double-click tray icon to start/stop
- High DPI support for crisp text on 4K displays
- Mouse wheel text resizing
- Draggable window positioning
- PyInstaller build script for standalone executable
