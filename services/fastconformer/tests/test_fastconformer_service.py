"""
Unit tests for FastConformer ASR Service

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
mock_torch.backends.cuda = MagicMock()
mock_torch.backends.cuda.matmul = MagicMock()
mock_torch.backends.cudnn = MagicMock()
mock_torch.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
mock_torch.from_numpy = MagicMock(
    return_value=MagicMock(
        unsqueeze=MagicMock(
            return_value=MagicMock(to=MagicMock(return_value=MagicMock(half=MagicMock())))
        )
    )
)
mock_torch.tensor = MagicMock(return_value=MagicMock(to=MagicMock()))
mock_torch.Tensor = MagicMock()
mock_torch.float16 = MagicMock()
mock_torch.__version__ = "2.0.0"

# Create mock for uvicorn
mock_uvicorn = MagicMock()

# Create mock for shared modules
mock_shared_utils = MagicMock()
mock_shared_utils.setup_logging = MagicMock(return_value=MagicMock())

# Create mock for shared.core
mock_shared_core = MagicMock()
mock_shared_core.clear_gpu_cache = MagicMock()


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_env(monkeypatch):
    """Set required environment variables."""
    monkeypatch.setenv(
        "FASTCONFORMER_MODEL", "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
    )
    monkeypatch.setenv("DECODER_TYPE", "rnnt")
    monkeypatch.setenv("ATT_CONTEXT_SIZE", "[70,6]")


@pytest.fixture
def mock_modules():
    """Patch all external modules before importing the service."""
    with patch.dict(
        "sys.modules",
        {
            "torch": mock_torch,
            "uvicorn": mock_uvicorn,
            "shared": MagicMock(),
            "shared.core": mock_shared_core,
            "shared.utils": mock_shared_utils,
        },
    ):
        yield


@pytest.fixture
def mock_model_module():
    """Mock the fastconformer_model module."""
    from dataclasses import dataclass

    @dataclass
    class MockModelState:
        model: object | None = None
        loaded: bool = False
        model_name: str = ""
        decoder_type: str = "rnnt"
        att_context_size: list = None

    mock_module = MagicMock()
    mock_module.DECODER_TYPE = "rnnt"
    mock_module.DEVICE = "cuda"
    mock_module.MODEL_NAME = "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
    mock_module.ATT_CONTEXT_SIZE = [70, 6]
    mock_module.get_model = MagicMock()
    mock_module.get_model_state = MagicMock(return_value=MockModelState())
    mock_module.setup_cuda = MagicMock()
    mock_module.unload_model = MagicMock()

    with patch.dict("sys.modules", {"fastconformer_model": mock_module}):
        yield mock_module


@pytest.fixture
def app(mock_env, mock_modules, mock_model_module):
    """Create FastAPI test app with mocked dependencies."""
    import sys

    # Remove cached service module
    if "fastconformer_service" in sys.modules:
        del sys.modules["fastconformer_service"]

    from fastconformer_service import app

    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


# ==============================================================================
# Test Health Endpoint
# ==============================================================================


def test_health_endpoint_model_not_loaded(client, mock_model_module):
    """Test health endpoint when model is not loaded."""
    mock_model_module.get_model_state.return_value.loaded = False

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "starting"
    assert data["model_loaded"] is False


def test_health_endpoint_model_loaded(client, mock_model_module):
    """Test health endpoint when model is loaded."""
    mock_model_module.get_model_state.return_value.loaded = True
    mock_model_module.get_model_state.return_value.model_name = (
        "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
    )
    mock_model_module.get_model_state.return_value.decoder_type = "rnnt"
    mock_model_module.get_model_state.return_value.att_context_size = [70, 6]

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_name"] == "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
    assert data["decoder_type"] == "rnnt"
    assert data["att_context_size"] == [70, 6]
    assert data["device"] == "cuda"


# ==============================================================================
# Test Info Endpoint
# ==============================================================================


def test_info_endpoint(client):
    """Test info endpoint returns configuration."""
    response = client.get("/info")

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "fastconformer-asr"
    assert "version" in data
    assert "model_name" in data
    assert "decoder_type" in data
    assert "att_context_size" in data
    assert data["streaming_only"] is True


# ==============================================================================
# Test Unload Endpoint
# ==============================================================================


def test_unload_endpoint(client, mock_model_module):
    """Test unload endpoint."""
    response = client.post("/unload")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["message"] == "Model unloaded from GPU"
    mock_model_module.unload_model.assert_called_once()


# ==============================================================================
# Test Streaming (mocked)
# ==============================================================================


def test_pcm_to_float():
    """Test PCM to float conversion."""
    # Remove cached service module
    import sys

    if "fastconformer_service" in sys.modules:
        del sys.modules["fastconformer_service"]

    # Import with mocks
    with patch.dict(
        "sys.modules",
        {
            "torch": mock_torch,
            "uvicorn": mock_uvicorn,
            "shared": MagicMock(),
            "shared.core": mock_shared_core,
            "shared.utils": mock_shared_utils,
            "fastconformer_model": MagicMock(
                DECODER_TYPE="rnnt",
                DEVICE="cuda",
                MODEL_NAME="test",
                ATT_CONTEXT_SIZE=[70, 6],
                get_model=MagicMock(),
                get_model_state=MagicMock(),
                setup_cuda=MagicMock(),
                unload_model=MagicMock(),
            ),
        },
    ):
        from fastconformer_service import pcm_to_float

        # Create test PCM audio
        audio_int16 = np.array([0, 16384, -16384, 32767], dtype=np.int16)
        audio_bytes = audio_int16.tobytes()

        # Convert
        audio_float = pcm_to_float(audio_bytes)

        assert audio_float.shape == (4,)
        assert audio_float.dtype == np.float32
        assert -1.0 <= audio_float.min() <= audio_float.max() <= 1.0
