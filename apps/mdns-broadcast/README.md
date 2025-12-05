# mDNS Broadcast for AI-Tools

Broadcasts `.local` domain names to your local network using mDNS (multicast DNS).

This allows other devices on your network (phones, tablets, other computers) to access AI-Tools services using friendly names like `http://audio-notes.local:7860`.

## Prerequisites

Install required packages:

```bash
pip install zeroconf
```

Or use the project's virtual environment.

## Quick Start

### Install as Scheduled Task (Recommended)

Run PowerShell as Administrator:

```powershell
.\service-manager.ps1 install
```

The broadcaster will:
- Start automatically with Windows
- Run in the background
- Broadcast mDNS services continuously

### Service Manager Commands

```powershell
.\service-manager.ps1 install    # Install and start
.\service-manager.ps1 uninstall  # Stop and remove
.\service-manager.ps1 start      # Start broadcaster
.\service-manager.ps1 stop       # Stop broadcaster
.\service-manager.ps1 status     # Show status
.\service-manager.ps1 logs       # View logs
.\service-manager.ps1 help       # Show help
```

### Run Manually (Foreground)

```bash
python mdns_broadcast.py
```

## Registered Services

| Domain | Port | Service |
|--------|------|---------|
| `ai-tools.local` | 80 | Main portal |
| `audio-notes.local` | 7860 | Audio Notes app |
| `lobe-chat.local` | 3210 | Lobe Chat |
| `ollama.local` | 11434 | Ollama LLM API |

## How It Works

1. Uses `zeroconf` library (pure Python mDNS implementation)
2. Broadcasts service names as `_http._tcp` mDNS records
3. Devices with mDNS support (iOS, macOS, Linux, Android) can discover services

## Troubleshooting

### Services not visible

1. Check Windows Firewall allows UDP port 5353 (mDNS)
2. Run `.\service-manager.ps1 status` to verify task is running
3. Restart with `.\service-manager.ps1 stop` then `.\service-manager.ps1 start`

### Android devices

Android has limited mDNS support. You may need an mDNS browser app or configure router DNS.
