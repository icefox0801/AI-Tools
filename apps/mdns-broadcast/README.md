# mDNS Broadcast for AI-Tools

Broadcasts `.local` domain names to your local network using mDNS (multicast DNS).

This allows other devices on your network (phones, tablets, other computers) to access AI-Tools services using friendly names like `http://audio-notes.local:7860`.

## Prerequisites

Install required packages:

```bash
pip install zeroconf pywin32
```

Or use the project's virtual environment (auto-installs on first run).

## Usage

### Option 1: Install as Windows Service (Recommended)

Run as Administrator:

```cmd
install_service.bat
```

The service will:
- Start automatically with Windows
- Run in the background
- Broadcast mDNS services continuously

To uninstall:

```cmd
uninstall_service.bat
```

### Option 2: Run manually (foreground)

Double-click `run.bat` or run from command line:

```cmd
run.bat
```

### Option 3: Run Python directly

```bash
python mdns_broadcast.py
```

### With custom IP

```bash
python mdns_broadcast.py --ip 192.168.50.130
```

### List services

```bash
python mdns_broadcast.py --list
```

## Service Manager (Recommended)

The `service-manager.ps1` provides a unified interface for all service operations:

```powershell
# Interactive menu
.\service-manager.ps1

# Command line
.\service-manager.ps1 install      # Install and start service
.\service-manager.ps1 uninstall    # Remove service
.\service-manager.ps1 restart      # Restart service
.\service-manager.ps1 status       # Show status
.\service-manager.ps1 logs         # View logs
.\service-manager.ps1 logs -Live   # Live log monitoring
.\service-manager.ps1 diagnostics  # Run diagnostics
.\service-manager.ps1 help         # Show help
```

## Quick Install/Uninstall

```cmd
# Install (run as Administrator)
install_service.bat

# Uninstall (run as Administrator)
uninstall_service.bat
```

## Windows Service Commands

After installing:

```cmd
net start AIToolsMDNS    # Start service
net stop AIToolsMDNS     # Stop service
sc query AIToolsMDNS     # Check status
```

## Registered Services

| Domain | Port | Service |
|--------|------|---------|
| `ai-tools.local` | 80 | Main portal (via nginx) |
| `audio-notes.local` | 7860 | Audio Notes Gradio app |
| `lobe-chat.local` | 3210 | Lobe Chat interface |
| `ollama.local` | 11434 | Ollama LLM API |

## How It Works

1. Uses `zeroconf` library (pure Python mDNS implementation)
2. Broadcasts service names as `_http._tcp` mDNS records
3. Other devices with mDNS support (iOS, macOS, Linux with avahi, Android) can discover these services

## Troubleshooting

### Run diagnostics

```powershell
.\service-manager.ps1 diagnostics
```

### Services not visible on other devices

1. Make sure Windows Firewall allows mDNS:
   - UDP port 5353 (mDNS)

2. Check service status:
   ```powershell
   .\service-manager.ps1 status
   ```

3. Verify mDNS is working:
   ```powershell
   dns-sd -B _http._tcp local
   ```

### Android devices

Android has limited mDNS support. You may need to:
- Use an mDNS browser app
- Or add entries to your router's DNS/DHCP settings

## Notes

- The script must remain running for services to be discoverable
- Press Ctrl+C to stop broadcasting
- For permanent service, consider setting up as a Windows Task Scheduler job
