"""
Unit tests for Parakeet model management

Tests model state, loading, unloading, and CUDA setup.
Uses mocked dependencies to avoid requiring actual GPU/NeMo.
"""

from unittest.mock import MagicMock, patch

import pytest

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
mock_torch.randn = MagicMock(return_value=MagicMock(to=MagicMock(return_value=MagicMock(half=MagicMock()))))
mock_torch.tensor = MagicMock(return_value=MagicMock(to=MagicMock()))

# Create mock for shared modules
mock_shared_logging = MagicMock()
mock_shared_logging.setup_logging = MagicMock(return_value=MagicMock())


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
    """Patch all external modules before importing."""
    with patch.dict(
        "sys.modules",
        {
            "torch": mock_torch,
            "shared": MagicMock(),
            "shared.logging": mock_shared_logging,
        },
    ):
        yield


@pytest.fixture
def model_module(mock_env, mock_modules):
    """Import model module with mocked dependencies."""
    import sys

    # Remove cached module if exists
    for mod_name in list(sys.modules.keys()):
        if "model" in mod_name and "parakeet" not in mod_name:
            continue
        if mod_name == "model":
            del sys.modules[mod_name]

    import model

    # Reset model state for each test
    model._model_state = model.ModelState()

    yield model


# ==============================================================================
# ModelState Tests
# ==============================================================================


class TestModelState:
    """Tests for ModelState dataclass."""

    def test_model_state_defaults(self, model_module) -> None:
        """ModelState has correct default values."""
        state = model_module.ModelState()
        assert state.streaming_model is None
        assert state.streaming_preprocessor is None
        assert state.streaming_loaded is False
        assert state.streaming_model_name == ""
        assert state.offline_model is None
        assert state.offline_preprocessor is None
        assert state.offline_loaded is False
        assert state.offline_model_name == ""

    def test_model_state_can_set_values(self, model_module) -> None:
        """ModelState fields can be set."""
        state = model_module.ModelState()
        mock_model = MagicMock()
        state.streaming_model = mock_model
        state.streaming_loaded = True
        state.streaming_model_name = "test-model"

        assert state.streaming_model is mock_model
        assert state.streaming_loaded is True
        assert state.streaming_model_name == "test-model"


# ==============================================================================
# get_model_state Tests
# ==============================================================================


class TestGetModelState:
    """Tests for get_model_state function."""

    def test_returns_global_state(self, model_module) -> None:
        """get_model_state returns the global ModelState instance."""
        result = model_module.get_model_state()
        assert result is model_module._model_state

    def test_returns_same_instance(self, model_module) -> None:
        """Multiple calls return the same instance."""
        result1 = model_module.get_model_state()
        result2 = model_module.get_model_state()
        assert result1 is result2


# ==============================================================================
# setup_cuda Tests
# ==============================================================================


class TestSetupCuda:
    """Tests for setup_cuda function."""

    def test_setup_cuda_enables_optimizations(self, model_module) -> None:
        """setup_cuda configures CUDA optimizations."""
        model_module.setup_cuda()

        # Verify CUDA settings were configured
        assert mock_torch.backends.cudnn.benchmark is True
        assert mock_torch.backends.cudnn.enabled is True
        mock_torch.cuda.empty_cache.assert_called()

    def test_setup_cuda_skips_on_cpu(self, model_module) -> None:
        """setup_cuda does nothing when DEVICE is cpu."""
        original_device = model_module.DEVICE
        model_module.DEVICE = "cpu"

        mock_torch.cuda.empty_cache.reset_mock()
        model_module.setup_cuda()

        # Should not call CUDA functions
        mock_torch.cuda.empty_cache.assert_not_called()

        model_module.DEVICE = original_device


# ==============================================================================
# unload_models Tests
# ==============================================================================


class TestUnloadModels:
    """Tests for unload_models function."""

    def test_unload_no_models_loaded(self, model_module) -> None:
        """unload_models returns not_loaded when no models are loaded."""
        result = model_module.unload_models()

        assert result["status"] == "not_loaded"
        assert "No models were loaded" in result["message"]

    def test_unload_streaming_model(self, model_module) -> None:
        """unload_models clears streaming model state."""
        state = model_module._model_state
        state.streaming_model = MagicMock()
        state.streaming_preprocessor = MagicMock()
        state.streaming_loaded = True
        state.streaming_model_name = "test-model"

        result = model_module.unload_models()

        assert result["status"] == "unloaded"
        assert state.streaming_model is None
        assert state.streaming_preprocessor is None
        assert state.streaming_loaded is False
        assert state.streaming_model_name == ""

    def test_unload_offline_model(self, model_module) -> None:
        """unload_models clears offline model state."""
        state = model_module._model_state
        state.offline_model = MagicMock()
        state.offline_preprocessor = MagicMock()
        state.offline_loaded = True
        state.offline_model_name = "test-offline"

        result = model_module.unload_models()

        assert result["status"] == "unloaded"
        assert state.offline_model is None
        assert state.offline_preprocessor is None
        assert state.offline_loaded is False
        assert state.offline_model_name == ""

    def test_unload_both_models(self, model_module) -> None:
        """unload_models clears both models."""
        state = model_module._model_state
        state.streaming_model = MagicMock()
        state.streaming_loaded = True
        state.streaming_model_name = "streaming"
        state.offline_model = MagicMock()
        state.offline_loaded = True
        state.offline_model_name = "offline"

        result = model_module.unload_models()

        assert result["status"] == "unloaded"
        assert state.streaming_loaded is False
        assert state.offline_loaded is False

    def test_unload_returns_gpu_memory_info(self, model_module) -> None:
        """unload_models returns GPU memory info on CUDA."""
        state = model_module._model_state
        state.streaming_model = MagicMock()
        state.streaming_loaded = True
        state.streaming_model_name = "test"

        result = model_module.unload_models()

        assert "gpu_memory_free_gb" in result
        assert "gpu_memory_total_gb" in result


# ==============================================================================
# Configuration Tests
# ==============================================================================


class TestConfiguration:
    """Tests for configuration values."""

    def test_streaming_model_from_env(self, model_module) -> None:
        """STREAMING_MODEL is set from environment."""
        assert model_module.STREAMING_MODEL == "nvidia/test-streaming"

    def test_offline_model_from_env(self, model_module) -> None:
        """OFFLINE_MODEL is set from environment."""
        assert model_module.OFFLINE_MODEL == "nvidia/test-offline"

    def test_device_is_cuda_when_available(self, model_module) -> None:
        """DEVICE is cuda when GPU available."""
        assert model_module.DEVICE == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
