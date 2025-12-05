"""
Unit tests for mDNS broadcast module.

Tests the mDNS service broadcaster functionality.
"""

import socket
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add apps/mdns-broadcast to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "apps" / "mdns-broadcast"))


class TestGetLocalIP:
    """Tests for get_local_ip function."""

    def test_get_local_ip_returns_string(self):
        """get_local_ip returns a string IP address."""
        from mdns_broadcast import get_local_ip

        ip = get_local_ip()
        assert isinstance(ip, str)

    def test_get_local_ip_format(self):
        """get_local_ip returns a valid IP format."""
        from mdns_broadcast import get_local_ip

        ip = get_local_ip()
        parts = ip.split(".")
        assert len(parts) == 4
        for part in parts:
            assert 0 <= int(part) <= 255

    @patch("socket.socket")
    def test_get_local_ip_fallback(self, mock_socket):
        """get_local_ip falls back to 127.0.0.1 on error."""
        from mdns_broadcast import get_local_ip

        mock_socket.side_effect = Exception("Network error")
        ip = get_local_ip()
        assert ip == "127.0.0.1"


class TestCreateServiceInfo:
    """Tests for create_service_info function."""

    def test_create_service_info_type(self):
        """create_service_info returns ServiceInfo object."""
        from mdns_broadcast import create_service_info

        try:
            from zeroconf import ServiceInfo
        except ImportError:
            pytest.skip("zeroconf not installed")

        info = create_service_info("test-service", 8080, "Test Service", "192.168.1.100")
        assert isinstance(info, ServiceInfo)

    def test_create_service_info_name(self):
        """ServiceInfo has correct name."""
        from mdns_broadcast import create_service_info

        try:
            from zeroconf import ServiceInfo
        except ImportError:
            pytest.skip("zeroconf not installed")

        info = create_service_info("my-app", 3000, "My App", "192.168.1.50")
        assert info.name == "my-app._http._tcp.local."

    def test_create_service_info_port(self):
        """ServiceInfo has correct port."""
        from mdns_broadcast import create_service_info

        try:
            from zeroconf import ServiceInfo
        except ImportError:
            pytest.skip("zeroconf not installed")

        info = create_service_info("api", 9000, "API Service", "10.0.0.1")
        assert info.port == 9000

    def test_create_service_info_server(self):
        """ServiceInfo has correct server name."""
        from mdns_broadcast import create_service_info

        try:
            from zeroconf import ServiceInfo
        except ImportError:
            pytest.skip("zeroconf not installed")

        info = create_service_info("web", 80, "Web", "192.168.1.1")
        assert info.server == "web.local."


class TestMDNSBroadcaster:
    """Tests for MDNSBroadcaster class."""

    def test_broadcaster_init(self):
        """MDNSBroadcaster initializes correctly."""
        try:
            from mdns_broadcast import MDNSBroadcaster
        except ImportError:
            pytest.skip("zeroconf not installed")

        services = [{"name": "test", "port": 80, "description": "Test"}]

        with patch("mdns_broadcast.Zeroconf"):
            broadcaster = MDNSBroadcaster(services, host_ip="192.168.1.100")
            assert broadcaster.host_ip == "192.168.1.100"
            assert broadcaster.services == services
            assert broadcaster.running is False

    def test_broadcaster_auto_detect_ip(self):
        """MDNSBroadcaster auto-detects IP if not provided."""
        try:
            from mdns_broadcast import MDNSBroadcaster
        except ImportError:
            pytest.skip("zeroconf not installed")

        services = [{"name": "test", "port": 80, "description": "Test"}]

        with patch("mdns_broadcast.Zeroconf"):
            with patch("mdns_broadcast.get_local_ip", return_value="10.0.0.5"):
                broadcaster = MDNSBroadcaster(services)
                assert broadcaster.host_ip == "10.0.0.5"


class TestServicesConfig:
    """Tests for service configuration."""

    def test_services_defined(self):
        """SERVICES list is defined and non-empty."""
        from mdns_broadcast import SERVICES

        assert isinstance(SERVICES, list)
        assert len(SERVICES) > 0

    def test_services_have_required_fields(self):
        """Each service has name, port, description."""
        from mdns_broadcast import SERVICES

        for svc in SERVICES:
            assert "name" in svc
            assert "port" in svc
            assert "description" in svc

    def test_services_ports_are_valid(self):
        """Service ports are valid port numbers."""
        from mdns_broadcast import SERVICES

        for svc in SERVICES:
            assert 1 <= svc["port"] <= 65535

    def test_ai_tools_service_exists(self):
        """ai-tools service is defined."""
        from mdns_broadcast import SERVICES

        names = [s["name"] for s in SERVICES]
        assert "ai-tools" in names

    def test_audio_notes_service_exists(self):
        """audio-notes service is defined."""
        from mdns_broadcast import SERVICES

        names = [s["name"] for s in SERVICES]
        assert "audio-notes" in names
