"""Unit tests for Whisper model management module."""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock torch before importing model module
torch_mock = MagicMock()
torch_mock.cuda.is_available.return_value = False
torch_mock.float32 = "float32"
torch_mock.float16 = "float16"
# Mock CUDA memory functions to return numbers
torch_mock.cuda.memory_allocated.return_value = 1073741824  # 1GB in bytes
torch_mock.cuda.memory_reserved.return_value = 2147483648  # 2GB in bytes
torch_mock.cuda.get_device_properties.return_value.total_memory = 17179869184  # 16GB
torch_mock.cuda.get_device_name.return_value = "NVIDIA RTX Test"


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock all external dependencies."""
    # Create a mock GPU manager
    mock_gpu_mgr = MagicMock()
    mock_gpu_mgr.ensure_model_ready.return_value = True
    mock_gpu_mgr._get_available_memory.return_value = 10.0  # 10GB available

    with (
        patch.dict(
            "sys.modules",
            {
                "torch": torch_mock,
                "huggingface_hub": MagicMock(),
                "transformers": MagicMock(),
                "shared.logging": MagicMock(),
            },
        ),
        patch("model.get_gpu_manager", return_value=mock_gpu_mgr),
    ):
        yield


class TestConfiguration:
    """Test configuration values."""

    def test_default_model(self):
        """Test default model is turbo."""
        with patch.dict(os.environ, {}, clear=True):
            import importlib

            import model

            importlib.reload(model)
            assert model.MODEL == "openai/whisper-large-v3-turbo"

    def test_custom_model(self):
        """Test custom model from environment."""
        with patch.dict(os.environ, {"WHISPER_MODEL": "custom/model"}):
            import importlib

            import model

            importlib.reload(model)
            assert model.MODEL == "custom/model"

    def test_device_cpu_when_no_cuda(self):
        """Test device defaults to CPU when CUDA unavailable."""
        torch_mock.cuda.is_available.return_value = False
        import importlib

        import model

        importlib.reload(model)
        assert model.DEVICE == "cpu"

    def test_device_cuda_when_available(self):
        """Test device uses CUDA when available."""
        torch_mock.cuda.is_available.return_value = True
        import importlib

        import model

        importlib.reload(model)
        assert model.DEVICE == "cuda"


class TestModelState:
    """Test ModelState dataclass."""

    def test_model_state_initialization(self):
        """Test ModelState initializes with correct defaults."""
        import model

        state = model.ModelState()
        assert state.pipe is None
        assert state.loaded is False
        assert state.model_name == ""
        assert state.use_flash_attn is False

    def test_get_model_state(self):
        """Test get_model_state returns global state."""
        import model

        state1 = model.get_model_state()
        state2 = model.get_model_state()
        assert state1 is state2  # Same instance


class TestModelLoading:
    """Test model loading functions."""

    @patch("model.AutoModelForSpeechSeq2Seq")
    @patch("model.AutoProcessor")
    def test_create_pipeline_uses_local_files_only(self, mock_processor, mock_model):
        """Test that _create_pipeline uses local_files_only=True."""
        import model

        mock_model.from_pretrained.return_value = Mock()
        mock_processor.from_pretrained.return_value = Mock()

        model._create_pipeline("openai/whisper-large-v3")

        # Verify local_files_only=True is passed
        mock_model.from_pretrained.assert_called_once()
        call_kwargs = mock_model.from_pretrained.call_args[1]
        assert call_kwargs.get("local_files_only") is True

        mock_processor.from_pretrained.assert_called_once()
        call_kwargs = mock_processor.from_pretrained.call_args[1]
        assert call_kwargs.get("local_files_only") is True

    @patch("model._create_pipeline")
    def test_load_model(self, mock_create):
        """Test loading model."""
        import model

        mock_pipe = Mock()
        mock_create.return_value = mock_pipe

        model.load_model()

        state = model.get_model_state()
        assert state.loaded is True
        assert state.pipe is mock_pipe
        assert state.model_name == model.MODEL
        mock_create.assert_called_once_with(model.MODEL)

    @patch("model._create_pipeline")
    def test_load_model_already_loaded(self, mock_create):
        """Test loading model when already loaded does nothing."""
        import model

        # Reset state
        model._model_state.loaded = True
        model._model_state.pipe = Mock()

        model.load_model()

        # Should not create pipeline again
        mock_create.assert_not_called()

    @patch("model.load_model")
    def test_get_pipeline_auto_loads(self, mock_load):
        """Test get_pipeline auto-loads model."""
        import model

        model._model_state.loaded = False
        model._model_state.pipe = None

        model.get_pipeline()

        mock_load.assert_called_once()

    @patch("model.AutoModelForSpeechSeq2Seq")
    @patch("model.AutoProcessor")
    def test_load_model_raises_on_missing_cache(self, mock_processor, mock_model):
        """Test that loading fails gracefully when models not in cache."""
        import model

        # Simulate models not in cache
        mock_model.from_pretrained.side_effect = OSError("Model not found")

        with pytest.raises(OSError, match="Model not found"):
            model.load_model()

        # Verify it tried with local_files_only=True
        call_kwargs = mock_model.from_pretrained.call_args[1]
        assert call_kwargs.get("local_files_only") is True


class TestModelUnloading:
    """Test model unloading functions."""

    def test_unload_model(self):
        """Test unloading model."""
        import model

        # Set up state
        model._model_state.loaded = True
        model._model_state.pipe = Mock()
        model._model_state.model_name = "test"

        result = model.unload_model()

        assert result is True
        state = model.get_model_state()
        assert state.loaded is False
        assert state.pipe is None
        assert state.model_name == ""

    def test_unload_when_not_loaded(self):
        """Test unloading when model is not loaded."""
        import model

        # Reset state
        model._model_state.loaded = False

        result = model.unload_model()

        assert result is False


class TestCUDASetup:
    """Test CUDA setup functionality."""

    def test_setup_cuda_when_available(self):
        """Test CUDA setup when GPU is available."""
        import model

        torch_mock.cuda.is_available.return_value = True
        torch_mock.cuda.get_device_name.return_value = "NVIDIA RTX 4090"

        model.setup_cuda()

        assert torch_mock.backends.cudnn.benchmark is True
        assert torch_mock.backends.cudnn.enabled is True
        assert torch_mock.cuda.empty_cache.called

    def test_setup_cuda_when_not_available(self):
        """Test CUDA setup when GPU is not available."""
        import model

        # Reset DEVICE to cpu for this test
        with patch.object(model, "DEVICE", "cpu"):
            model.setup_cuda()

        # Should return early without configuring CUDA


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
