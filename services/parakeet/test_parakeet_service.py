"""
Unit tests for Parakeet ASR Service

Tests the streaming WebSocket endpoint, health check, and transcription.
Uses mocked models to avoid requiring actual GPU/NeMo.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ==============================================================================
# Mock Dependencies
# ==============================================================================

# Create mock for torch
mock_torch = MagicMock()
mock_torch.cuda.is_available = MagicMock(return_value=True)
mock_torch.cuda.get_device_name = MagicMock(return_value="NVIDIA GeForce RTX Test")
mock_torch.cuda.memory_allocated = MagicMock(return_value=1024 * 1024 * 1024)  # 1GB
mock_torch.cuda.memory_reserved = MagicMock(return_value=2 * 1024 * 1024 * 1024)  # 2GB
mock_torch.cuda.get_device_properties = MagicMock(
    return_value=MagicMock(total_memory=16 * 1024 * 1024 * 1024)
)
mock_torch.cuda.synchronize = MagicMock()
mock_torch.cuda.empty_cache = MagicMock()
mock_torch.backends = MagicMock()
mock_torch.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
mock_torch.from_numpy = MagicMock(
    return_value=MagicMock(
        unsqueeze=MagicMock(
            return_value=MagicMock(to=MagicMock(return_value=MagicMock(half=MagicMock())))
        )
    )
)
mock_torch.tensor = MagicMock(return_value=MagicMock())
mock_torch.amp = MagicMock()
mock_torch.amp.autocast = MagicMock(
    return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
)
mock_torch.Tensor = MagicMock()
mock_torch.__version__ = "2.0.0"

# Create mock for soundfile
mock_soundfile = MagicMock()
mock_soundfile.read = MagicMock(return_value=(np.zeros(16000, dtype=np.float32), 16000))
mock_soundfile.write = MagicMock()

# Create mock for uvicorn
mock_uvicorn = MagicMock()

# Create mock for shared modules
mock_shared_logging = MagicMock()
mock_shared_logging.setup_logging = MagicMock(return_value=MagicMock())

mock_text_refiner = MagicMock()
mock_text_refiner.get_client = MagicMock(return_value=MagicMock(available=True, enabled=True))
mock_text_refiner.check_text_refiner = MagicMock()
mock_text_refiner.capitalize_text = MagicMock(side_effect=lambda x: x.capitalize() if x else "")
mock_text_refiner.refine_text = MagicMock(side_effect=lambda x: x)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_env(monkeypatch):
    """Set required environment variables."""
    monkeypatch.setenv("PARAKEET_STREAMING_MODEL", "nvidia/test-streaming")
    monkeypatch.setenv("PARAKEET_OFFLINE_MODEL", "nvidia/test-offline")


@pytest.fixture
def mock_modules():
    """Patch all external modules before importing the service."""
    with patch.dict(
        "sys.modules",
        {
            "torch": mock_torch,
            "torch.amp": mock_torch.amp,
            "soundfile": mock_soundfile,
            "uvicorn": mock_uvicorn,
            "shared": MagicMock(),
            "shared.logging": mock_shared_logging,
            "shared.text_refiner": mock_text_refiner,
        },
    ):
        yield


@pytest.fixture
def app(mock_env, mock_modules):
    """Create FastAPI app with mocked dependencies."""
    import sys

    # Remove cached modules if exist
    for mod_name in list(sys.modules.keys()):
        if mod_name in ("model", "audio", "parakeet_service"):
            del sys.modules[mod_name]

    # Import modules
    import model
    import parakeet_service

    # Set up mocked model state
    model._model_state.streaming_model = MagicMock()
    model._model_state.streaming_preprocessor = MagicMock()
    model._model_state.streaming_loaded = True
    model._model_state.streaming_model_name = "nvidia/test-streaming"
    model._model_state.offline_model = MagicMock()
    model._model_state.offline_preprocessor = MagicMock()
    model._model_state.offline_loaded = True
    model._model_state.offline_model_name = "nvidia/test-offline"

    yield parakeet_service.app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def service(mock_env, mock_modules):
    """Get service module for direct function testing."""
    import sys

    for mod_name in list(sys.modules.keys()):
        if mod_name in ("model", "audio", "parakeet_service"):
            del sys.modules[mod_name]

    import model
    import parakeet_service

    # Set up mocked model state
    model._model_state.streaming_model = MagicMock()
    model._model_state.streaming_preprocessor = MagicMock()
    model._model_state.streaming_loaded = True
    model._model_state.streaming_model_name = "nvidia/test-streaming"

    yield parakeet_service


# ==============================================================================
# Health Endpoint Tests
# ==============================================================================


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self, client) -> None:
        """Health check returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_returns_model_loaded(self, client) -> None:
        """Health check shows streaming model is loaded."""
        response = client.get("/health")
        data = response.json()
        assert data["streaming_loaded"] is True

    def test_health_returns_device(self, client) -> None:
        """Health check includes device info."""
        response = client.get("/health")
        data = response.json()
        assert "device" in data

    def test_health_returns_model_names(self, client) -> None:
        """Health check includes model names."""
        response = client.get("/health")
        data = response.json()
        assert "streaming_model" in data
        assert "offline_model" in data


# ==============================================================================
# Info Endpoint Tests
# ==============================================================================


class TestInfoEndpoint:
    """Tests for the /info endpoint."""

    def test_info_returns_ok(self, client) -> None:
        """Info endpoint returns OK."""
        response = client.get("/info")
        assert response.status_code == 200

    def test_info_returns_streaming_model(self, client) -> None:
        """Info endpoint includes streaming model."""
        response = client.get("/info")
        data = response.json()
        assert "streaming_model" in data

    def test_info_returns_offline_model(self, client) -> None:
        """Info endpoint includes offline model."""
        response = client.get("/info")
        data = response.json()
        assert "offline_model" in data

    def test_info_returns_device(self, client) -> None:
        """Info endpoint includes device."""
        response = client.get("/info")
        data = response.json()
        assert "device" in data

    def test_info_returns_text_refiner_status(self, client) -> None:
        """Info endpoint includes text refiner status."""
        response = client.get("/info")
        data = response.json()
        assert "text_refiner" in data


# ==============================================================================
# StreamingState Tests
# ==============================================================================


class TestStreamingState:
    """Tests for StreamingState dataclass."""

    def test_streaming_state_defaults(self, service) -> None:
        """StreamingState has correct default values."""
        state = service.StreamingState()
        assert state.cache_last_channel is None
        assert state.cache_last_time is None
        assert state.cache_last_channel_len is None
        assert state.previous_hypotheses is None
        assert state.accumulated_text == ""

    def test_streaming_state_reset(self, service) -> None:
        """Reset clears all state."""
        state = service.StreamingState()
        state.cache_last_channel = MagicMock()
        state.cache_last_time = MagicMock()
        state.cache_last_channel_len = MagicMock()
        state.previous_hypotheses = MagicMock()
        state.accumulated_text = "Hello world"

        state.reset()

        assert state.cache_last_channel is None
        assert state.cache_last_time is None
        assert state.cache_last_channel_len is None
        assert state.previous_hypotheses is None
        assert state.accumulated_text == ""


# ==============================================================================
# Extract Word Timestamps Tests
# ==============================================================================


class TestExtractWordTimestamps:
    """Tests for _extract_word_timestamps function."""

    def test_extract_empty_results(self, service) -> None:
        """Handle empty results."""
        result = service._extract_word_timestamps([], np.zeros(16000), 16000)
        assert result == []

    def test_extract_from_timestamp_dict(self, service) -> None:
        """Extract timestamps from dict format."""
        hyp = MagicMock()
        hyp.timestamp = {
            "word": [
                {"word": "hello", "start": 0.0, "end": 0.5},
                {"word": "world", "start": 0.5, "end": 1.0},
            ]
        }

        audio = np.zeros(16000)
        result = service._extract_word_timestamps([hyp], audio, 16000)

        assert len(result) == 2
        assert result[0]["word"] == "hello"
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 0.5
        assert result[1]["word"] == "world"

    def test_extract_fallback_estimation(self, service) -> None:
        """Fallback to timestamp estimation from text."""
        hyp = MagicMock()
        hyp.timestamp = None
        hyp.text = "hello world test"

        audio = np.zeros(16000)  # 1 second
        result = service._extract_word_timestamps([hyp], audio, 16000)

        assert len(result) == 3
        assert result[0]["word"] == "hello"
        assert result[1]["word"] == "world"
        assert result[2]["word"] == "test"


# ==============================================================================
# Configuration Tests
# ==============================================================================


class TestConfiguration:
    """Tests for configuration constants."""

    def test_chunk_duration_sec(self, service) -> None:
        """CHUNK_DURATION_SEC is reasonable."""
        assert service.CHUNK_DURATION_SEC >= 0.5
        assert service.CHUNK_DURATION_SEC <= 5.0

    def test_max_audio_chunk_sec(self, service) -> None:
        """MAX_AUDIO_CHUNK_SEC is within NeMo limits."""
        assert service.MAX_AUDIO_CHUNK_SEC <= 20.0
        assert service.MAX_AUDIO_CHUNK_SEC >= 10.0

    def test_overlap_sec(self, service) -> None:
        """OVERLAP_SEC is reasonable."""
        assert service.OVERLAP_SEC > 0
        assert service.OVERLAP_SEC < service.MAX_AUDIO_CHUNK_SEC / 2

    def test_min_words_for_punctuation(self, service) -> None:
        """MIN_WORDS_FOR_PUNCTUATION is set."""
        assert service.MIN_WORDS_FOR_PUNCTUATION >= 1


# ==============================================================================
# Model Unload Tests
# ==============================================================================


class TestModelUnload:
    """Tests for model unloading."""

    def test_unload_success(self, client) -> None:
        """Unload succeeds when model is loaded."""
        response = client.post("/unload")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unloaded"


# ==============================================================================
# WebSocket Streaming Tests
# ==============================================================================


class TestWebSocketStreaming:
    """Tests for WebSocket streaming endpoint."""

    def test_websocket_connect(self, client) -> None:
        """WebSocket connection is accepted."""
        with client.websocket_connect("/stream") as ws:
            assert ws is not None

    def test_websocket_accepts_audio(self, client) -> None:
        """WebSocket accepts audio bytes."""
        with client.websocket_connect("/stream") as ws:
            # Send some audio bytes
            audio_bytes = b"\x00" * 3200  # 100ms of audio
            ws.send_bytes(audio_bytes)
            # Should not raise error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
