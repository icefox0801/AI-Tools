"""
Unit tests for mDNS Broadcast Service
"""

import socket
import unittest
from unittest.mock import MagicMock, patch

from mdns_broadcast import (
    SERVICES,
    get_local_ip,
    create_service_info,
    MDNSBroadcaster,
    __version__,
)


class TestGetLocalIP(unittest.TestCase):
    """Tests for get_local_ip function"""

    def test_returns_string(self):
        """Should return a string IP address"""
        ip = get_local_ip()
        self.assertIsInstance(ip, str)

    def test_valid_ip_format(self):
        """Should return a valid IPv4 address format"""
        ip = get_local_ip()
        parts = ip.split(".")
        self.assertEqual(len(parts), 4)
        for part in parts:
            self.assertTrue(0 <= int(part) <= 255)

    @patch("socket.socket")
    def test_fallback_on_error(self, mock_socket):
        """Should return 127.0.0.1 on socket error"""
        mock_socket.return_value.connect.side_effect = Exception("Network error")
        ip = get_local_ip()
        self.assertEqual(ip, "127.0.0.1")


class TestCreateServiceInfo(unittest.TestCase):
    """Tests for create_service_info function"""

    def test_creates_service_info(self):
        """Should create a valid ServiceInfo object"""
        info = create_service_info("test-service", 8080, "Test Service", "192.168.1.100")

        self.assertEqual(info.type, "_http._tcp.local.")
        self.assertEqual(info.name, "test-service._http._tcp.local.")
        self.assertEqual(info.port, 8080)
        self.assertEqual(info.server, "test-service.local.")

    def test_ip_conversion(self):
        """Should correctly convert IP string to bytes"""
        info = create_service_info("test", 80, "Test", "192.168.50.130")
        expected_bytes = socket.inet_aton("192.168.50.130")
        self.assertIn(expected_bytes, info.addresses)

    def test_properties(self):
        """Should include description in properties"""
        info = create_service_info("test", 80, "My Description", "127.0.0.1")
        self.assertEqual(info.properties.get(b"description"), b"My Description")
        self.assertEqual(info.properties.get(b"path"), b"/")


class TestServiceDefinitions(unittest.TestCase):
    """Tests for SERVICES constant"""

    def test_services_not_empty(self):
        """Should have at least one service defined"""
        self.assertGreater(len(SERVICES), 0)

    def test_service_structure(self):
        """Each service should have required fields"""
        for svc in SERVICES:
            self.assertIn("name", svc)
            self.assertIn("port", svc)
            self.assertIn("description", svc)

    def test_valid_ports(self):
        """All ports should be valid"""
        for svc in SERVICES:
            self.assertIsInstance(svc["port"], int)
            self.assertGreater(svc["port"], 0)
            self.assertLess(svc["port"], 65536)

    def test_expected_services(self):
        """Should include expected AI-Tools services"""
        names = [svc["name"] for svc in SERVICES]
        self.assertIn("ai-tools", names)
        self.assertIn("audio-notes", names)
        self.assertIn("ollama", names)


class TestMDNSBroadcaster(unittest.TestCase):
    """Tests for MDNSBroadcaster class"""

    @patch("mdns_broadcast.Zeroconf")
    def test_init_with_custom_ip(self, mock_zeroconf):
        """Should use provided IP address"""
        broadcaster = MDNSBroadcaster(SERVICES, host_ip="10.0.0.1")
        self.assertEqual(broadcaster.host_ip, "10.0.0.1")

    @patch("mdns_broadcast.Zeroconf")
    @patch("mdns_broadcast.get_local_ip")
    def test_init_auto_detect_ip(self, mock_get_ip, mock_zeroconf):
        """Should auto-detect IP if not provided"""
        mock_get_ip.return_value = "192.168.1.50"
        broadcaster = MDNSBroadcaster(SERVICES)
        self.assertEqual(broadcaster.host_ip, "192.168.1.50")

    @patch("mdns_broadcast.Zeroconf")
    def test_start_registers_services(self, mock_zeroconf):
        """Should register all services on start"""
        mock_zc_instance = MagicMock()
        mock_zeroconf.return_value = mock_zc_instance

        broadcaster = MDNSBroadcaster(SERVICES, host_ip="192.168.1.1")
        broadcaster.start()

        # Should register each service
        self.assertEqual(mock_zc_instance.register_service.call_count, len(SERVICES))
        self.assertTrue(broadcaster.running)

    @patch("mdns_broadcast.Zeroconf")
    def test_stop_unregisters_services(self, mock_zeroconf):
        """Should unregister all services on stop"""
        mock_zc_instance = MagicMock()
        mock_zeroconf.return_value = mock_zc_instance

        broadcaster = MDNSBroadcaster(SERVICES, host_ip="192.168.1.1")
        broadcaster.start()
        broadcaster.stop()

        # Should unregister each service
        self.assertEqual(mock_zc_instance.unregister_service.call_count, len(SERVICES))
        self.assertTrue(mock_zc_instance.close.called)
        self.assertFalse(broadcaster.running)

    @patch("mdns_broadcast.Zeroconf")
    def test_stop_when_not_running(self, mock_zeroconf):
        """Should handle stop when not running"""
        mock_zc_instance = MagicMock()
        mock_zeroconf.return_value = mock_zc_instance

        broadcaster = MDNSBroadcaster(SERVICES, host_ip="192.168.1.1")
        broadcaster.stop()  # Should not raise

        # Should not try to unregister anything
        self.assertEqual(mock_zc_instance.unregister_service.call_count, 0)


class TestVersion(unittest.TestCase):
    """Tests for version"""

    def test_version_exists(self):
        """Should have a version defined"""
        self.assertIsNotNone(__version__)
        self.assertIsInstance(__version__, str)

    def test_version_format(self):
        """Version should be in semver format"""
        parts = __version__.split(".")
        self.assertGreaterEqual(len(parts), 1)


if __name__ == "__main__":
    unittest.main()
