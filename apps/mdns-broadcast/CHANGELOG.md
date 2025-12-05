# Changelog

All notable changes to mDNS Broadcast will be documented in this file.

## [1.0] - 2025-12-05

### Added
- mDNS service broadcaster using Python zeroconf library
- Broadcasts AI-Tools services (.local domains) to local network
- Automatic local IP detection
- Windows service support via pywin32
- Comprehensive PowerShell service manager (service-manager.ps1)
  - Install/uninstall Windows service
  - Start/stop/restart service
  - Status checking with detailed info
  - Log viewing and diagnostics
  - Interactive menu mode
  - Auto-elevation for admin operations
- Service definitions for:
  - ai-tools.local (port 80)
  - audio-notes.local (port 7860)
  - lobe-chat.local (port 3210)
  - ollama.local (port 11434)
- Command-line standalone mode for testing
- Unit tests for core functionality
