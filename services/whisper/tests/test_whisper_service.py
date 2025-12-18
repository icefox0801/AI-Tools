"""
Unit tests for Whisper ASR Service

Tests the streaming WebSocket endpoint, health check, and transcription.
Uses mocked models to avoid requiring actual GPU/model files.
"""

import json
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
mock_torch.float16 = MagicMock()
mock_torch.float32 = MagicMock()
mock_torch.hub = MagicMock()

# Mock tensor operations
mock_tensor = MagicMock()
mock_tensor.float = MagicMock(return_value=mock_tensor)
mock_tensor.to = MagicMock(return_value=mock_tensor)
mock_torch.from_numpy = MagicMock(return_value=mock_tensor)


class MockPipeline:
    """Mock HuggingFace pipeline for testing."""

    def __call__(self, audio_array, **kwargs):
        """Simulate transcription."""
        if len(audio_array) < 4800:  # < 0.3s
            return {"text": "", "chunks": []}
        elif len(audio_array) < 16000:  # < 1s
            return {"text": "Hello.", "chunks": []}
        else:
            return {"text": "Hello world, this is a test.", "chunks": []}


# Create mock for transformers
mock_transformers = MagicMock()
mock_model = MagicMock()
mock_model.to = MagicMock(return_value=mock_model)
mock_transformers.AutoModelForSpeechSeq2Seq = MagicMock()
mock_transformers.AutoModelForSpeechSeq2Seq.from_pretrained = MagicMock(return_value=mock_model)
mock_transformers.AutoProcessor = MagicMock()
mock_transformers.AutoProcessor.from_pretrained = MagicMock(return_value=MagicMock())
mock_transformers.pipeline = MagicMock(return_value=MockPipeline())

# Create mock for soundfile
mock_soundfile = MagicMock()
mock_soundfile.read = MagicMock(return_value=(np.zeros(16000, dtype=np.float32), 16000))

# Create mock for uvicorn
mock_uvicorn = MagicMock()

# Create mock for shared modules
mock_shared_logging = MagicMock()
mock_shared_logging.setup_logging = MagicMock(return_value=MagicMock())

# Create mock for shared.core
mock_shared_core = MagicMock()
mock_shared_core.clear_gpu_cache = MagicMock(side_effect=lambda: mock_torch.cuda.empty_cache())
mock_shared_core.get_gpu_manager = MagicMock(
    return_value=MagicMock(
        ensure_model_ready=MagicMock(return_value=True),
        register_model=MagicMock(),
        unregister_model=MagicMock(),
    )
)

# Create mock for shared.utils
mock_shared_utils = MagicMock()
mock_shared_utils.setup_logging = MagicMock(return_value=MagicMock())


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_modules():
    """Patch all external modules before importing the service."""
    with patch.dict(
        "sys.modules",
        {
            "torch": mock_torch,
            "torch.hub": mock_torch.hub,
            "transformers": mock_transformers,
            "soundfile": mock_soundfile,
            "uvicorn": mock_uvicorn,
            "shared": MagicMock(),
            "shared.core": mock_shared_core,
            "shared.utils": mock_shared_utils,
            "shared.logging": mock_shared_logging,
        },
    ):
        yield


@pytest.fixture
def app(mock_modules):
    """Create FastAPI app with mocked dependencies."""
    import os
    import sys

    # Remove cached module if exists
    if "whisper_service" in sys.modules:
        del sys.modules["whisper_service"]

    # Set required env vars for testing
    os.environ["WHISPER_MODEL"] = "openai/whisper-test"
    os.environ["WHISPER_VAD_FILTER"] = "true"
    os.environ["WHISPER_VAD_THRESHOLD"] = "0.5"
    os.environ["WHISPER_BEAM_SIZE"] = "5"
    os.environ["WHISPER_LANGUAGE"] = "en"
    os.environ["WHISPER_CHUNK_DURATION_SEC"] = "1.5"
    os.environ["WHISPER_MIN_AUDIO_SEC"] = "0.3"

    # Import the service module
    import whisper_service

    # Set up mocked models
    whisper_service.whisper_pipe = MockPipeline()

    # Create mock VAD model
    mock_vad = MagicMock()
    mock_vad.reset_states = MagicMock()
    mock_vad.return_value = MagicMock(item=MagicMock(return_value=0.9))
    whisper_service.vad_model = mock_vad

    yield whisper_service.app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def service(mock_modules):
    """Get service module for direct function testing."""
    import os
    import sys

    if "whisper_service" in sys.modules:
        del sys.modules["whisper_service"]

    # Set required env var for testing
    os.environ["WHISPER_MODEL"] = "openai/whisper-test"

    import whisper_service

    whisper_service.whisper_pipe = MockPipeline()

    mock_vad = MagicMock()
    mock_vad.reset_states = MagicMock()
    mock_vad.return_value = MagicMock(item=MagicMock(return_value=0.9))
    whisper_service.vad_model = mock_vad

    yield whisper_service


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

    def test_health_returns_model_loaded(self, client):
        """Health check shows model status."""
        response = client.get("/health")
        data = response.json()
        # Model may or may not be loaded in test environment
        assert "model_loaded" in data

    def test_health_returns_vad_status(self, client):
        """Health check includes VAD status."""
        response = client.get("/health")
        data = response.json()
        assert "vad_enabled" in data
        assert "vad_loaded" in data

    def test_health_returns_device(self, client):
        """Health check includes device info."""
        response = client.get("/health")
        data = response.json()
        assert "device" in data


# ==============================================================================
# Info Endpoint Tests
# ==============================================================================


class TestInfoEndpoint:
    """Tests for the /info endpoint."""

    def test_info_returns_ok(self, client):
        """Info endpoint returns OK."""
        response = client.get("/info")
        assert response.status_code == 200

    def test_info_returns_service_name(self, client):
        """Info endpoint includes service name."""
        response = client.get("/info")
        data = response.json()
        assert data["service"] == "whisper-asr"

    def test_info_returns_version(self, client):
        """Info endpoint includes API version."""
        response = client.get("/info")
        data = response.json()
        assert data["version"] == "1.3"

    def test_info_returns_model(self, client):
        """Info endpoint includes model name."""
        response = client.get("/info")
        data = response.json()
        assert "model" in data

    def test_info_returns_vad_enabled(self, client):
        """Info endpoint includes VAD status."""
        response = client.get("/info")
        data = response.json()
        assert "vad_enabled" in data


# ==============================================================================
# VAD Tests
# ==============================================================================


class TestVAD:
    """Tests for Voice Activity Detection."""

    def test_detect_speech_returns_true_with_speech(self, service):
        """VAD returns True when speech is detected."""
        service.vad_model.return_value = MagicMock(item=MagicMock(return_value=0.9))
        audio = np.zeros(1024, dtype=np.float32)
        result = service.detect_speech(audio)
        assert result is True

    def test_detect_speech_returns_false_for_silence(self, service):
        """VAD returns False for silence."""
        service.vad_model.return_value = MagicMock(item=MagicMock(return_value=0.1))
        audio = np.zeros(1024, dtype=np.float32)
        result = service.detect_speech(audio)
        assert result is False

    def test_detect_speech_no_vad_model(self, service):
        """Returns True when VAD model not loaded."""
        service.vad_model = None
        audio = np.zeros(1024, dtype=np.float32)
        result = service.detect_speech(audio)
        assert result is True

    def test_detect_speech_short_audio(self, service):
        """Returns True for audio shorter than window size."""
        audio = np.zeros(100, dtype=np.float32)
        result = service.detect_speech(audio)
        assert result is True


# ==============================================================================
# Transcription Tests
# ==============================================================================


# Transcribe_audio function removed - transcription now handled via /transcribe endpoint


# ==============================================================================
# Configuration Tests
# ==============================================================================


class TestConfiguration:
    """Tests for configuration values."""

    def test_version(self, service):
        """Version is set correctly."""
        assert service.__version__ == "1.3"

    def test_chunk_duration(self, service):
        """Default chunk duration is 1.5s."""
        assert service.DEFAULT_CHUNK_DURATION_SEC == 1.5

    def test_min_audio_duration(self, service):
        """Default minimum audio is 0.3s."""
        assert service.DEFAULT_MIN_AUDIO_SEC == 0.3

    def test_vad_enabled_by_default(self, service):
        """VAD is enabled by default."""
        assert service.USE_VAD is True

    def test_sample_rate(self, service):
        """Sample rate is 16kHz."""
        assert service.SAMPLE_RATE == 16000


# ==============================================================================
# WebSocket Streaming Tests
# ==============================================================================


class TestWebSocketStreaming:
    """Tests for WebSocket streaming endpoint."""

    def test_websocket_connect(self, client):
        """WebSocket connection is accepted."""
        with client.websocket_connect("/stream") as ws:
            assert ws is not None

    def test_websocket_accepts_config(self, client):
        """WebSocket accepts config message."""
        with client.websocket_connect("/stream") as ws:
            config = {"chunk_ms": 200, "sample_rate": 16000}
            ws.send_text(json.dumps(config))
            # Should not raise error

    def test_websocket_end_of_stream(self, client):
        """Empty bytes signal end of stream."""
        with client.websocket_connect("/stream") as ws:
            ws.send_bytes(b"")
            data = ws.receive_json()
            assert data.get("final") is True

    def test_websocket_returns_transcription(self, client):
        """WebSocket returns transcription for audio."""
        with client.websocket_connect("/stream") as ws:
            # Send config
            config = {"chunk_ms": 200, "sample_rate": 16000}
            ws.send_text(json.dumps(config))

            # Send 1.5s of audio (24000 samples = 48000 bytes for int16)
            audio_bytes = b"\x00" * 48000
            ws.send_bytes(audio_bytes)

            # Signal end of stream
            ws.send_bytes(b"")

            # Collect responses
            responses = []
            while True:
                data = ws.receive_json()
                responses.append(data)
                if data.get("final"):
                    break

            # Should have at least the final message
            assert len(responses) >= 1
            final = [r for r in responses if r.get("final")]
            assert len(final) == 1


# ==============================================================================
# Model Unload Tests
# ==============================================================================


class TestModelUnload:
    """Tests for model unloading."""

    def test_unload_success(self, client):
        """Unload returns status (may not be loaded in test)."""
        response = client.post("/unload")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["unloaded", "not_loaded"]

    def test_unload_twice_returns_not_loaded(self, client):
        """Second unload returns not_loaded."""
        # First unload
        client.post("/unload")
        # Second unload
        response = client.post("/unload")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_loaded"

    # ==============================================================================
    # Dual Model Tests
    # ==============================================================================

    # Dual-model support removed - service now uses single whisper-turbo model

    def test_transcribe_returns_model_used(self, client):
        """Transcribe endpoint returns which model was used."""
        # Create a simple WAV file
        import io
        import struct

        # WAV header + 1 second of silence at 16kHz
        sample_rate = 16000
        num_samples = sample_rate
        audio_data = struct.pack("<" + "h" * num_samples, *([0] * num_samples))

        # WAV header
        wav_header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + len(audio_data),
            b"WAVE",
            b"fmt ",
            16,  # fmt chunk size
            1,  # PCM
            1,  # mono
            sample_rate,
            sample_rate * 2,  # byte rate
            2,  # block align
            16,  # bits per sample
            b"data",
            len(audio_data),
        )

        wav_file = io.BytesIO(wav_header + audio_data)

        response = client.post(
            "/transcribe",
            files={"file": ("test.wav", wav_file, "audio/wav")},
            data={"language": "en"},
        )

        assert response.status_code == 200
        data = response.json()
        # Should indicate offline model was used
        assert "model" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
