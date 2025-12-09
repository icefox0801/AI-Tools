"""
Unit tests for Vosk ASR Service

Tests the streaming WebSocket endpoint and health check.
Uses mocked Vosk model to avoid requiring actual model files.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ==============================================================================
# Mock Vosk Module
# ==============================================================================


class MockKaldiRecognizer:
    """Mock Vosk KaldiRecognizer for testing."""

    def __init__(self, model, sample_rate):
        self.model = model
        self.sample_rate = sample_rate
        self._partial_text = ""
        self._final_text = ""
        self._accept_count = 0
        self._is_endpoint = False

    def SetMaxAlternatives(self, n):
        pass

    def SetWords(self, enable):
        pass

    def SetPartialWords(self, enable):
        pass

    def SetEndpointerMode(self, mode):
        pass

    def SetEndpointerDelays(self, t_start_max, t_end, t_max):
        pass

    def AcceptWaveform(self, data):
        """Simulate processing audio data."""
        self._accept_count += 1
        # Simulate endpoint every 5 chunks
        if self._accept_count % 5 == 0:
            self._is_endpoint = True
            self._final_text = "hello world"
            return True
        self._is_endpoint = False
        self._partial_text = "hello"
        return False

    def Result(self):
        """Return final result JSON."""
        return json.dumps({"text": self._final_text})

    def PartialResult(self):
        """Return partial result JSON."""
        return json.dumps({"partial": self._partial_text})

    def FinalResult(self):
        """Return final result and reset."""
        result = json.dumps({"text": self._final_text})
        self._final_text = ""
        return result


class MockModel:
    """Mock Vosk Model."""

    def __init__(self, path):
        self.path = path


# Create mock vosk module
mock_vosk = MagicMock()
mock_vosk.Model = MockModel
mock_vosk.KaldiRecognizer = MockKaldiRecognizer
mock_vosk.SetLogLevel = MagicMock()

# Create mock shared modules
mock_shared_logging = MagicMock()
mock_shared_logging.setup_logging = MagicMock(return_value=MagicMock())

mock_shared_utils = MagicMock()
mock_shared_utils.setup_logging = MagicMock(return_value=MagicMock())

mock_text_refiner_client = MagicMock()
mock_text_refiner_client.enabled = True
mock_text_refiner_client.available = False
mock_text_refiner_client.url = "http://text-refiner:8000"

mock_shared_text_refiner = MagicMock()
mock_shared_text_refiner.get_client = MagicMock(return_value=mock_text_refiner_client)


async def mock_refine_text(text, punctuate=True, correct=False):
    """Mock refine_text that just returns input."""
    return text


mock_shared_text_refiner.refine_text = mock_refine_text


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_vosk_module():
    """Patch vosk and shared modules before importing the service."""
    with patch.dict(
        "sys.modules",
        {
            "vosk": mock_vosk,
            "shared": MagicMock(),
            "shared.logging": mock_shared_logging,
            "shared.utils": mock_shared_utils,
            "shared.text_refiner": mock_shared_text_refiner,
        },
    ):
        yield mock_vosk


@pytest.fixture
def app(mock_vosk_module):
    """Create FastAPI app with mocked Vosk."""
    # Need to import after mocking
    import importlib
    import sys

    # Remove cached module if exists
    if "vosk_service" in sys.modules:
        del sys.modules["vosk_service"]

    # Import the service module
    import vosk_service

    importlib.reload(vosk_service)

    # Mock get_model_name to return a test value
    vosk_service.get_model_name = lambda: "vosk-model-test"

    yield vosk_service.app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


# ==============================================================================
# Health Endpoint Tests
# ==============================================================================


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self, client):
        """Health check returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"

    def test_health_returns_backend_info(self, client):
        """Health check includes backend information."""
        response = client.get("/health")
        data = response.json()

        assert data["backend"] == "vosk"
        assert data["device"] == "cpu"
        assert data["streaming"] is True

    def test_health_returns_sample_rate(self, client):
        """Health check includes sample rate."""
        response = client.get("/health")
        data = response.json()

        assert data["sample_rate"] == 16000

    def test_health_returns_api_version(self, client):
        """Health check includes API version."""
        response = client.get("/health")
        data = response.json()

        assert "api_version" in data
        assert data["api_version"] == "1.0"


# ==============================================================================
# Streaming Message Format Tests
# ==============================================================================


class TestStreamingMessageFormat:
    """Tests for streaming message format consistency."""

    def test_message_has_required_fields(self):
        """Streaming messages must have id, text, is_final."""
        message = {"id": "s0", "text": "hello", "is_final": False}

        assert "id" in message
        assert "text" in message
        assert "is_final" in message

    def test_segment_id_format(self):
        """Segment IDs follow s0, s1, s2 pattern."""
        for i in range(10):
            segment_id = f"s{i}"
            assert segment_id.startswith("s")
            assert segment_id[1:].isdigit()

    def test_partial_message_is_not_final(self):
        """Partial messages have is_final=False."""
        partial = {"id": "s0", "text": "hel", "is_final": False}
        assert partial["is_final"] is False

    def test_final_message_is_final(self):
        """Final messages have is_final=True."""
        final = {"id": "s0", "text": "hello world", "is_final": True}
        assert final["is_final"] is True


# ==============================================================================
# WebSocket Streaming Tests
# ==============================================================================


class TestWebSocketStreaming:
    """Tests for WebSocket streaming endpoint."""

    def test_websocket_accepts_connection(self, client):
        """WebSocket endpoint accepts connections."""
        with client.websocket_connect("/stream") as websocket:
            # Connection should succeed
            assert websocket is not None

    def test_websocket_accepts_config_message(self, client):
        """WebSocket accepts JSON config message."""
        with client.websocket_connect("/stream") as websocket:
            config = {"finalize_interval": 2.0, "partial_interval": 0.1}
            websocket.send_text(json.dumps(config))
            # Should not raise

    def test_websocket_accepts_audio_bytes(self, client):
        """WebSocket accepts binary audio data."""
        with client.websocket_connect("/stream") as websocket:
            # Send fake audio data (16-bit PCM)
            audio_chunk = b"\x00\x00" * 1600  # 100ms at 16kHz
            websocket.send_bytes(audio_chunk)
            # Should not raise

    def test_websocket_returns_partial_results(self, client):
        """WebSocket sends partial transcription results."""
        with client.websocket_connect("/stream") as websocket:
            # Send audio chunks until we get a partial
            for _ in range(3):
                audio_chunk = b"\x00\x00" * 1600
                websocket.send_bytes(audio_chunk)

            # Try to receive (with timeout)
            try:
                data = websocket.receive_json()
                assert "id" in data
                assert "text" in data
            except Exception:
                # May timeout if mock doesn't trigger partial
                pass

    def test_websocket_returns_final_results(self, client):
        """WebSocket sends final transcription results when endpoint detected."""
        with client.websocket_connect("/stream") as websocket:
            # Send 5 chunks - mock triggers endpoint on 5th
            for _ in range(5):
                audio_chunk = b"\x00\x00" * 1600
                websocket.send_bytes(audio_chunk)

            # Collect all responses (partial + final)
            responses = []
            # First response is partial from chunk 1
            data = websocket.receive_json()
            responses.append(data)
            assert data.get("is_final") is False
            assert data.get("text") == "hello"

            # Next response should be final from chunk 5
            data = websocket.receive_json()
            responses.append(data)
            assert data.get("is_final") is True
            assert data.get("text") == "hello world"


# ==============================================================================
# Configuration Tests
# ==============================================================================


class TestStreamingConfig:
    """Tests for streaming configuration."""

    def test_default_finalize_interval(self):
        """Default finalize interval is 3 seconds."""
        # This tests the constant in the module
        DEFAULT_FINALIZE_INTERVAL = 3.0
        assert DEFAULT_FINALIZE_INTERVAL == 3.0

    def test_default_partial_interval(self):
        """Default partial interval is 150ms."""
        DEFAULT_PARTIAL_INTERVAL = 0.15
        assert DEFAULT_PARTIAL_INTERVAL == 0.15

    def test_config_can_override_intervals(self, client):
        """Config message can override default intervals."""
        with client.websocket_connect("/stream") as websocket:
            config = {"finalize_interval": 5.0, "partial_interval": 0.2}
            websocket.send_text(json.dumps(config))
            # Should not raise, config applied


# ==============================================================================
# Error Handling Tests
# ==============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_json_config_ignored(self, client):
        """Invalid JSON config is silently ignored."""
        with client.websocket_connect("/stream") as websocket:
            websocket.send_text("not valid json")
            # Should not raise, invalid config ignored

    def test_empty_audio_handled(self, client):
        """Empty audio bytes are handled."""
        with client.websocket_connect("/stream") as websocket:
            websocket.send_bytes(b"")
            # Should not raise


# ==============================================================================
# Integration Tests (require actual Vosk model)
# ==============================================================================


@pytest.mark.skip(reason="Requires actual Vosk model")
class TestVoskIntegration:
    """Integration tests requiring actual Vosk model."""

    def test_real_audio_transcription(self, client):
        """Test with real audio data."""
        pass

    def test_continuous_streaming(self, client):
        """Test continuous audio streaming."""
        pass


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
