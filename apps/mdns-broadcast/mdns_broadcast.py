"""
mDNS Service Broadcaster for AI-Tools
Broadcasts .local domains to LAN using zeroconf (pure Python mDNS)
"""

__version__ = "1.0"

import signal
import socket
import sys
import time
from typing import Any

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
        # Create a socket to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def create_service_info(name: str, port: int, description: str, ip: str) -> ServiceInfo:
    """Create a ServiceInfo object for mDNS registration"""
    # Convert IP string to bytes
    ip_bytes = socket.inet_aton(ip)

    return ServiceInfo(
        type_="_http._tcp.local.",
        name=f"{name}._http._tcp.local.",
        addresses=[ip_bytes],
        port=port,
        properties={"description": description, "path": "/"},
        server=f"{name}.local.",
    )


class MDNSBroadcaster:
    """Manages mDNS service registration and broadcasting"""

    def __init__(self, services: list[dict[str, Any]], host_ip: str | None = None):
        self.host_ip = host_ip or get_local_ip()
        self.zeroconf = Zeroconf()
        self.service_infos: list[ServiceInfo] = []
        self.services = services
        self.running = False

    def start(self):
        """Register all services"""
        print("=" * 50)
        print("  AI-Tools mDNS Service Broadcaster")
        print("=" * 50)
        print()
        print(f"Host IP: {self.host_ip}")
        print()
        print("Registering mDNS services...")
        print()

        for svc in self.services:
            name = svc["name"]
            port = svc["port"]
            description = svc["description"]

            print(f"  [+] {name}.local:{port} - {description}")

            info = create_service_info(name, port, description, self.host_ip)
            try:
                self.zeroconf.register_service(info, allow_name_change=True)
            except Exception as e:
                print(f"      Warning: {e}")
            self.service_infos.append(info)

        print()
        print("=" * 50)
        print("  Services are now broadcasting!")
        print("=" * 50)
        print()
        print("Other devices on your network can now access:")
        print()
        for svc in self.services:
            name = svc["name"]
            port = svc["port"]
            if port == 80:
                print(f"  http://{name}.local/")
            else:
                print(f"  http://{name}.local:{port}/")
        print()
        print("Press Ctrl+C to stop broadcasting...")
        print()

        self.running = True

    def stop(self):
        """Unregister all services"""
        if not self.running:
            return

        print()
        print("Stopping mDNS broadcasts...")

        for info in self.service_infos:
            self.zeroconf.unregister_service(info)

        self.zeroconf.close()
        self.running = False
        print("Done.")

    def run_forever(self):
        """Run until interrupted"""
        self.start()

        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Broadcast AI-Tools services via mDNS")
    parser.add_argument(
        "--ip", "-i", type=str, help="Host IP address (auto-detected if not specified)"
    )
    parser.add_argument("--list", "-l", action="store_true", help="List services and exit")
    args = parser.parse_args()

    if args.list:
        print("Available services:")
        for svc in SERVICES:
            print(f"  {svc['name']}.local:{svc['port']} - {svc['description']}")
        return

    broadcaster = MDNSBroadcaster(SERVICES, host_ip=args.ip)
    broadcaster.run_forever()


if __name__ == "__main__":
    main()
