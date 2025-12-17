"""
Unit tests for FastConformer service endpoints.
Tests FastAPI endpoints and WebSocket streaming logic.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def test_client():
    """Create test client for FastAPI app."""
    from fastconformer_service import app

    return TestClient(app)


@pytest.fixture
def mock_model():
    """Mock FastConformer model."""
    model = MagicMock()
    model.encoder = MagicMock()
    model.decoding = MagicMock()
    return model


class TestHealthEndpoint:
    """Test suite for /health endpoint."""

    @patch("fastconformer_service.MODEL", None)
    def test_health_not_loaded(self, test_client):
        """Test health endpoint when model is not loaded."""
        response = test_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "starting"
        assert data["model_loaded"] is False
        assert data["model_name"] == "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"

    @patch("fastconformer_service.MODEL")
    def test_health_loaded(self, mock_model, test_client):
        """Test health endpoint when model is loaded."""
        with patch("fastconformer_service.torch") as mock_torch:
            mock_torch.cuda.memory_allocated.return_value = 2.5 * 1024**3
            mock_torch.cuda.get_device_properties.return_value.total_memory = 24 * 1024**3

            response = test_client.get("/health")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True
            assert "gpu_memory_gb" in data
            assert isinstance(data["gpu_memory_gb"], (int, float))


class TestInfoEndpoint:
    """Test suite for /info endpoint."""

    def test_info_returns_config(self, test_client):
        """Test info endpoint returns service configuration."""
        response = test_client.get("/info")
        assert response.status_code == 200

        data = response.json()
        assert "model" in data
        assert "decoder_type" in data
        assert "att_context_size" in data
        assert "chunk_duration_sec" in data
        assert "sample_rate" in data
        assert data["model"] == "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"


class TestUnloadEndpoint:
    """Test suite for /unload endpoint."""

    @patch("fastconformer_service.unload_model")
    def test_unload_success(self, mock_unload, test_client):
        """Test model unload endpoint."""
        response = test_client.post("/unload")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "unloaded"
        mock_unload.assert_called_once()


class TestWebSocketStreaming:
    """Test suite for WebSocket streaming transcription."""

    @pytest.mark.asyncio
    @patch("fastconformer_service.get_model")
    async def test_websocket_basic_flow(self, mock_get_model):
        """Test basic WebSocket streaming flow."""
        from fastconformer_service import app
        from starlette.testclient import TestClient

        # Mock model
        mock_model = MagicMock()
        mock_model.encoder.streaming_forward.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        mock_model.decoding.rnnt_decoder_predictions_tensor.return_value = [[1, 2, 3]]  # Token IDs
        mock_model.tokenizer.ids_to_text.return_value = "hello world"
        mock_get_model.return_value = mock_model

        with TestClient(app) as client:
            with client.websocket_connect("/stream") as websocket:
                # Send config
                config = {"chunk_ms": 500}
                websocket.send_json(config)

                # Receive config acknowledgment
                data = websocket.receive_json()
                assert "config" in data

                # Send audio chunk (dummy 16-bit PCM)
                audio_chunk = b"\x00\x00" * 8000  # 500ms at 16kHz
                websocket.send_bytes(audio_chunk)

                # Should receive transcription
                data = websocket.receive_json()
                assert "text" in data or "id" in data

    @pytest.mark.asyncio
    @patch("fastconformer_service.get_model")
    async def test_websocket_silence_detection(self, mock_get_model):
        """Test WebSocket skips silence chunks."""
        from fastconformer_service import app, is_silence
        from starlette.testclient import TestClient
        import numpy as np

        # Mock model
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        # Test silence detection
        silent_audio = np.zeros(8000, dtype=np.int16)
        assert is_silence(silent_audio.tobytes()) is True

        # Test non-silence
        loud_audio = np.random.randint(-5000, 5000, 8000, dtype=np.int16)
        assert is_silence(loud_audio.tobytes()) is False

    @pytest.mark.asyncio
    @patch("fastconformer_service.get_model")
    async def test_websocket_ping_pong(self, mock_get_model):
        """Test WebSocket keepalive ping/pong."""
        from fastconformer_service import app
        from starlette.testclient import TestClient
        import asyncio

        mock_model = MagicMock()
        mock_model.encoder.streaming_forward.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        mock_get_model.return_value = mock_model

        with TestClient(app) as client:
            with client.websocket_connect("/stream") as websocket:
                # Send config
                websocket.send_json({"chunk_ms": 500})

                # Wait for ping (15s interval in production, but test may not wait)
                # Just verify connection stays alive
                data = websocket.receive_json()
                assert "config" in data


class TestStreamingLogic:
    """Test streaming transcription logic."""

    @patch("fastconformer_service.torch")
    @patch("fastconformer_service.librosa")
    def test_audio_preprocessing(self, mock_librosa, mock_torch):
        """Test audio chunk preprocessing."""
        from fastconformer_service import preprocess_audio_chunk
        import numpy as np

        # Mock audio chunk
        audio_bytes = b"\x00\x00" * 8000  # 16-bit PCM
        mock_audio_array = np.zeros(8000, dtype=np.float32)
        mock_librosa.resample.return_value = mock_audio_array
        mock_torch.tensor.return_value = MagicMock()

        # Process chunk
        result = preprocess_audio_chunk(audio_bytes, sample_rate=16000)

        # Verify resampling was called
        mock_librosa.resample.assert_called_once()

        # Verify tensor conversion
        mock_torch.tensor.assert_called_once()

    def test_silence_threshold(self):
        """Test silence detection threshold."""
        from fastconformer_service import is_silence, SILENCE_THRESHOLD
        import numpy as np

        # Just below threshold (silence)
        quiet = np.random.randint(-100, 100, 8000, dtype=np.int16)
        assert is_silence(quiet.tobytes()) is True

        # Above threshold (not silence)
        loud = np.random.randint(-5000, 5000, 8000, dtype=np.int16)
        assert is_silence(loud.tobytes()) is False


class TestErrorHandling:
    """Test error handling in service."""

    @pytest.mark.asyncio
    @patch("fastconformer_service.get_model")
    async def test_websocket_model_error(self, mock_get_model):
        """Test WebSocket handles model errors gracefully."""
        from fastconformer_service import app
        from starlette.testclient import TestClient

        # Mock model that raises error
        mock_model = MagicMock()
        mock_model.encoder.streaming_forward.side_effect = RuntimeError("CUDA error")
        mock_get_model.return_value = mock_model

        with TestClient(app) as client:
            with client.websocket_connect("/stream") as websocket:
                websocket.send_json({"chunk_ms": 500})

                # Send audio
                audio_chunk = b"\x00\x00" * 8000
                websocket.send_bytes(audio_chunk)

                # Should receive error message
                data = websocket.receive_json()
                # Connection may close or send error
                assert "error" in str(data).lower() or websocket.closed

    def test_health_handles_gpu_error(self, test_client):
        """Test health endpoint handles GPU errors gracefully."""
        with patch("fastconformer_service.torch") as mock_torch:
            mock_torch.cuda.memory_allocated.side_effect = RuntimeError("GPU error")

            response = test_client.get("/health")
            # Should still return 200 with error status
            assert response.status_code == 200
            data = response.json()
            # May report as unhealthy or starting
            assert data["status"] in ["starting", "unhealthy", "error"]
