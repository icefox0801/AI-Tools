"""
mDNS Broadcast Windows Service
Runs the mDNS broadcaster as a Windows service.

Install:   python mdns_service.py install
Start:     python mdns_service.py start
Stop:      python mdns_service.py stop
Remove:    python mdns_service.py remove
Debug:     python mdns_service.py debug
"""

import logging
import os
import socket
import sys
import time
from pathlib import Path

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import servicemanager
    import win32event
    import win32service
    import win32serviceutil
except ImportError:
    print("ERROR: pywin32 package not installed.")
    print("Run: pip install pywin32")
    sys.exit(1)

try:
    from zeroconf import ServiceInfo, Zeroconf
except ImportError:
    print("ERROR: zeroconf package not installed.")
    print("Run: pip install zeroconf")
    sys.exit(1)


# Service definitions
SERVICES = [
    {"name": "ai-tools", "port": 80, "description": "AI-Tools Main Portal"},
    {"name": "audio-notes", "port": 7860, "description": "Audio Notes Gradio App"},
    {"name": "lobe-chat", "port": 3210, "description": "Lobe Chat Interface"},
    {"name": "ollama", "port": 11434, "description": "Ollama LLM API"},
]


def get_local_ip() -> str:
    """Get the local IP address (192.168.x.x)"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def create_service_info(name: str, port: int, description: str, ip: str) -> ServiceInfo:
    """Create a ServiceInfo object for mDNS registration"""
    ip_bytes = socket.inet_aton(ip)
    return ServiceInfo(
        type_="_http._tcp.local.",
        name=f"{name}._http._tcp.local.",
        addresses=[ip_bytes],
        port=port,
        properties={"description": description, "path": "/"},
        server=f"{name}.local.",
    )


class MDNSBroadcastService(win32serviceutil.ServiceFramework):
    """Windows Service for mDNS Broadcasting"""

    _svc_name_ = "AIToolsMDNS"
    _svc_display_name_ = "AI-Tools mDNS Broadcaster"
    _svc_description_ = (
        "Broadcasts AI-Tools services (.local domains) to the local network via mDNS"
    )

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        self.running = True
        self.zeroconf = None
        self.service_infos = []

        # Setup logging
        self.log_path = Path(__file__).parent / "mdns_service.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.log_path),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger("MDNSService")

    def SvcStop(self):
        """Stop the service"""
        self.logger.info("Stopping mDNS service...")
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.stop_event)
        self.running = False

    def SvcDoRun(self):
        """Run the service"""
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, ""),
        )
        self.main()

    def main(self):
        """Main service logic"""
        self.logger.info("Starting mDNS Broadcast Service")

        try:
            host_ip = get_local_ip()
            self.logger.info(f"Host IP: {host_ip}")

            self.zeroconf = Zeroconf()

            # Register all services
            for svc in SERVICES:
                name = svc["name"]
                port = svc["port"]
                description = svc["description"]

                self.logger.info(f"Registering: {name}.local:{port}")

                info = create_service_info(name, port, description, host_ip)
                self.zeroconf.register_service(info)
                self.service_infos.append(info)

            self.logger.info("All services registered successfully")

            # Run until stopped
            while self.running:
                # Wait for stop event with timeout
                result = win32event.WaitForSingleObject(self.stop_event, 5000)
                if result == win32event.WAIT_OBJECT_0:
                    break

        except Exception as e:
            self.logger.error(f"Service error: {e}")
            servicemanager.LogErrorMsg(f"mDNS Service Error: {e}")

        finally:
            self._cleanup()

    def _cleanup(self):
        """Clean up mDNS registrations"""
        self.logger.info("Cleaning up mDNS registrations...")

        if self.zeroconf:
            for info in self.service_infos:
                try:
                    self.zeroconf.unregister_service(info)
                except Exception as e:
                    self.logger.error(f"Error unregistering service: {e}")

            try:
                self.zeroconf.close()
            except Exception as e:
                self.logger.error(f"Error closing zeroconf: {e}")

        self.logger.info("mDNS service stopped")


def main():
    """Entry point for command-line usage"""
    if len(sys.argv) == 1:
        # Running as service
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(MDNSBroadcastService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        # Command line
        win32serviceutil.HandleCommandLine(MDNSBroadcastService)


if __name__ == "__main__":
    main()
