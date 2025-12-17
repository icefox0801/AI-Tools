"""
Unit tests for FastConformer model management.
Tests model loading, decoder configuration, and latency settings.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastconformer_model import (
    load_model,
    get_model,
    unload_model,
    _warmup_model,
    setup_cuda,
    MODEL_ID,
    DECODER_TYPE,
    ATT_CONTEXT_SIZE,
)


class TestFastConformerModel:
    """Test suite for FastConformer model operations."""

    @patch("fastconformer_model.EncDecHybridRNNTCTCBPEModel")
    @patch("fastconformer_model.torch")
    def test_load_model_success(self, mock_torch, mock_model_class):
        """Test successful model loading with correct configuration."""
        # Mock GPU availability
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        mock_torch.cuda.get_device_properties.return_value.total_memory = 24 * 1024**3

        # Mock model instance
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        # Load model
        result = load_model()

        # Verify model was loaded from correct ID
        mock_model_class.from_pretrained.assert_called_once_with(MODEL_ID)

        # Verify decoder type was set
        mock_model.change_decoding_strategy.assert_called_once()
        call_kwargs = mock_model.change_decoding_strategy.call_args.kwargs
        assert call_kwargs["decoder_type"] == DECODER_TYPE

        # Verify attention context size was set
        mock_model.encoder.set_default_att_context_size.assert_called_once_with(ATT_CONTEXT_SIZE)

        # Verify model was moved to GPU and converted to FP16
        mock_model.cuda.assert_called_once()
        mock_model.half.assert_called_once()

        # Verify eval mode
        mock_model.eval.assert_called_once()

        assert result is True

    @patch("fastconformer_model.EncDecHybridRNNTCTCBPEModel")
    @patch("fastconformer_model.torch")
    def test_load_model_no_gpu(self, mock_torch, mock_model_class):
        """Test model loading fails gracefully when no GPU is available."""
        mock_torch.cuda.is_available.return_value = False

        with pytest.raises(RuntimeError, match="CUDA not available"):
            load_model()

    @patch("fastconformer_model.EncDecHybridRNNTCTCBPEModel")
    @patch("fastconformer_model.torch")
    @patch("fastconformer_model._warmup_model")
    def test_warmup_called_after_load(self, mock_warmup, mock_torch, mock_model_class):
        """Test that model warmup is called after successful loading."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_properties.return_value.total_memory = 24 * 1024**3

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        load_model()

        # Verify warmup was called with the model
        mock_warmup.assert_called_once_with(mock_model)

    @patch("fastconformer_model.MODEL", None)
    @patch("fastconformer_model.load_model")
    def test_get_model_loads_if_not_loaded(self, mock_load):
        """Test get_model loads model if not already loaded."""
        mock_load.return_value = True

        # First call should trigger load
        with patch("fastconformer_model.MODEL", None):
            get_model()
            mock_load.assert_called_once()

    @patch("fastconformer_model.MODEL")
    def test_get_model_returns_existing(self, mock_existing_model):
        """Test get_model returns existing model without reloading."""
        with patch("fastconformer_model.load_model") as mock_load:
            model = get_model()
            assert model is mock_existing_model
            mock_load.assert_not_called()

    @patch("fastconformer_model.MODEL")
    @patch("fastconformer_model.torch")
    def test_unload_model(self, mock_torch, mock_model):
        """Test model unloading clears memory."""
        with patch("fastconformer_model.MODEL", mock_model):
            unload_model()

            # Verify CUDA cache was cleared
            mock_torch.cuda.empty_cache.assert_called_once()

    @patch("fastconformer_model.torch")
    def test_setup_cuda(self, mock_torch):
        """Test CUDA optimization settings."""
        mock_torch.cuda.is_available.return_value = True

        setup_cuda()

        # Verify TF32 was enabled for matmul and cudnn
        assert mock_torch.backends.cuda.matmul.allow_tf32 is True
        assert mock_torch.backends.cudnn.allow_tf32 is True
        # Verify cudnn benchmark was enabled
        mock_torch.backends.cudnn.benchmark = True

    @patch("fastconformer_model.torch")
    def test_warmup_model(self, mock_torch):
        """Test model warmup with dummy audio."""
        mock_model = MagicMock()
        mock_torch.randn.return_value = MagicMock()

        _warmup_model(mock_model)

        # Verify dummy audio tensor was created (1s of audio at 16kHz)
        mock_torch.randn.assert_called_once()
        call_args = mock_torch.randn.call_args[0]
        assert call_args[0] == 16000  # 1 second at 16kHz

        # Verify model was called with no_grad
        mock_torch.no_grad.assert_called_once()


class TestDecoderConfiguration:
    """Test decoder type configuration."""

    @pytest.mark.parametrize(
        "decoder_type,expected",
        [
            ("rnnt", "rnnt"),
            ("ctc", "ctc"),
        ],
    )
    @patch("fastconformer_model.EncDecHybridRNNTCTCBPEModel")
    @patch("fastconformer_model.torch")
    def test_decoder_types(self, mock_torch, mock_model_class, decoder_type, expected):
        """Test different decoder type configurations."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_properties.return_value.total_memory = 24 * 1024**3

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        with patch("fastconformer_model.DECODER_TYPE", decoder_type):
            load_model()

            call_kwargs = mock_model.change_decoding_strategy.call_args.kwargs
            assert call_kwargs["decoder_type"] == expected


class TestLatencyConfiguration:
    """Test attention context size (latency) configuration."""

    @pytest.mark.parametrize(
        "att_context,expected_latency_ms",
        [
            ([70, 0], 0),  # Zero latency mode
            ([70, 1], 80),  # 80ms latency
            ([70, 6], 480),  # 480ms latency (default)
            ([70, 33], 1040),  # 1040ms latency
        ],
    )
    @patch("fastconformer_model.EncDecHybridRNNTCTCBPEModel")
    @patch("fastconformer_model.torch")
    def test_latency_modes(self, mock_torch, mock_model_class, att_context, expected_latency_ms):
        """Test different latency mode configurations."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_properties.return_value.total_memory = 24 * 1024**3

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        with patch("fastconformer_model.ATT_CONTEXT_SIZE", att_context):
            load_model()

            mock_model.encoder.set_default_att_context_size.assert_called_once_with(att_context)
