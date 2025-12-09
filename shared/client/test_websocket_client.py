"""
Unit tests for shared.client.websocket_client module.

Tests the ASRClient and ConnectionConfig classes.
"""

from .websocket_client import ASRClient, ConnectionConfig


class TestConnectionConfig:
    """Tests for ConnectionConfig data class."""

    def test_default_values(self):
        """Default values are set correctly."""
        config = ConnectionConfig()
        assert config.host == "localhost"
        assert config.port == 8001
        assert config.endpoint == "/stream"
        assert config.chunk_ms == 200
        assert config.timeout == 30.0

    def test_custom_values(self):
        """Custom values are set correctly."""
        config = ConnectionConfig(
            host="192.168.1.100",
            port=9000,
            endpoint="/ws",
            chunk_ms=300,
            timeout=60.0,
        )
        assert config.host == "192.168.1.100"
        assert config.port == 9000
        assert config.endpoint == "/ws"
        assert config.chunk_ms == 300
        assert config.timeout == 60.0

    def test_uri_property(self):
        """URI property constructs correct WebSocket URI."""
        config = ConnectionConfig(host="localhost", port=8001, endpoint="/stream")
        assert config.uri == "ws://localhost:8001/stream"

    def test_uri_custom_endpoint(self):
        """URI with custom endpoint."""
        config = ConnectionConfig(host="example.com", port=443, endpoint="/v2/stream")
        assert config.uri == "ws://example.com:443/v2/stream"


class TestASRClient:
    """Tests for ASRClient class."""

    def test_default_initialization(self):
        """Default initialization sets correct values."""
        client = ASRClient()
        assert client.config.host == "localhost"
        assert client.config.port == 8001
        assert client.config.endpoint == "/stream"
        assert client.config.chunk_ms == 200
        assert client.config.timeout == 30.0

    def test_custom_initialization(self):
        """Custom initialization."""
        client = ASRClient(
            host="192.168.1.100",
            port=9000,
            endpoint="/ws",
            chunk_ms=300,
            timeout=60.0,
        )
        assert client.config.host == "192.168.1.100"
        assert client.config.port == 9000

    def test_uri_property(self):
        """URI property delegates to config."""
        client = ASRClient(host="localhost", port=8001)
        assert client.uri == "ws://localhost:8001/stream"

    def test_from_backend_config(self):
        """Create client from backend config dict."""
        config = {
            "host": "parakeet-host",
            "port": 8002,
            "chunk_ms": 300,
        }
        client = ASRClient.from_backend_config(config)
        assert client.config.host == "parakeet-host"
        assert client.config.port == 8002
        assert client.config.chunk_ms == 300

    def test_from_backend_config_defaults(self):
        """Create client with missing config values uses defaults."""
        config = {"port": 8003}
        client = ASRClient.from_backend_config(config)
        assert client.config.host == "localhost"
        assert client.config.port == 8003
        assert client.config.chunk_ms == 200

    def test_from_backend_config_empty(self):
        """Create client from empty config uses all defaults."""
        client = ASRClient.from_backend_config({})
        assert client.config.host == "localhost"
        assert client.config.port == 8001
        assert client.config.chunk_ms == 200
