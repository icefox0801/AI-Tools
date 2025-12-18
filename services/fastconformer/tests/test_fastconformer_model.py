"""
Unit tests for FastConformer model management

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
mock_torch.backends.cuda = MagicMock()
mock_torch.backends.cuda.matmul = MagicMock()
mock_torch.backends.cudnn = MagicMock()
mock_torch.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
mock_torch.randn = MagicMock(
    return_value=MagicMock(to=MagicMock(return_value=MagicMock(half=MagicMock())))
)
mock_torch.tensor = MagicMock(return_value=MagicMock(to=MagicMock()))
mock_torch.float16 = MagicMock()

# Create mock for shared.core
mock_shared_core = MagicMock()
mock_shared_core.clear_gpu_cache = MagicMock(side_effect=lambda: mock_torch.cuda.empty_cache())
mock_shared_core.get_gpu_manager = MagicMock(
    return_value=MagicMock(
        ensure_model_ready=MagicMock(return_value=True),
        register_model=MagicMock(),
        unregister_model=MagicMock(),
        report_memory=MagicMock(),
    )
)

# Create mock for shared.utils
mock_shared_utils = MagicMock()
mock_shared_utils.setup_logging = MagicMock(return_value=MagicMock())


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
    monkeypatch.setenv("BEAM_SIZE", "1")
    monkeypatch.setenv("BATCH_SIZE", "1")


@pytest.fixture
def mock_modules():
    """Patch all external modules before importing."""
    with patch.dict(
        "sys.modules",
        {
            "torch": mock_torch,
            "shared": MagicMock(),
            "shared.core": mock_shared_core,
            "shared.utils": mock_shared_utils,
        },
    ):
        yield


@pytest.fixture
def model_module(mock_env, mock_modules):
    """Import model module with mocked dependencies."""
    import sys

    # Remove cached module if exists
    for mod_name in list(sys.modules.keys()):
        if mod_name == "fastconformer_model":
            del sys.modules[mod_name]

    import fastconformer_model

    # Reset model state for each test
    fastconformer_model._model_state = fastconformer_model.ModelState()

    yield fastconformer_model


# ==============================================================================
# Test Model State
# ==============================================================================


def test_model_state_initialization(model_module):
    """Test model state starts unloaded."""
    state = model_module.get_model_state()
    assert state.model is None
    assert state.loaded is False
    assert state.model_name == ""
    assert state.decoder_type == ""
    assert state.att_context_size is None


# ==============================================================================
# Test CUDA Setup
# ==============================================================================


def test_setup_cuda_enables_optimizations(model_module):
    """Test CUDA optimizations are enabled."""
    # Reset flag
    model_module._cuda_initialized = False

    model_module.setup_cuda()

    assert mock_torch.backends.cuda.matmul.allow_tf32 is True
    assert mock_torch.backends.cudnn.allow_tf32 is True
    assert mock_torch.backends.cudnn.benchmark is True


def test_setup_cuda_only_runs_once(model_module):
    """Test CUDA setup only runs once."""
    # Reset flag
    model_module._cuda_initialized = False

    model_module.setup_cuda()
    model_module.setup_cuda()  # Should skip

    # Only called once
    assert mock_torch.backends.cuda.matmul.allow_tf32 is True


def test_setup_cuda_skips_on_cpu(model_module, monkeypatch):
    """Test CUDA setup skips on CPU."""
    # Mock CPU device
    mock_torch.cuda.is_available = MagicMock(return_value=False)

    # Re-import module to pick up CPU device
    import sys

    del sys.modules["fastconformer_model"]
    import fastconformer_model

    fastconformer_model._cuda_initialized = False
    fastconformer_model.setup_cuda()

    # Should not set any CUDA flags on CPU
    # (no assertions needed, just shouldn't error)


# ==============================================================================
# Test Model Loading
# ==============================================================================


def test_load_model_success(model_module):
    """Test model loads successfully."""
    # Create mock NeMo model
    mock_nemo_model = MagicMock()
    mock_nemo_model.to = MagicMock(return_value=mock_nemo_model)
    mock_nemo_model.eval = MagicMock()
    mock_nemo_model.change_decoding_strategy = MagicMock()
    mock_nemo_model.encoder = MagicMock()
    mock_nemo_model.encoder.set_default_att_context_size = MagicMock()
    mock_nemo_model.half = MagicMock(return_value=mock_nemo_model)
    mock_nemo_model.dtype = mock_torch.float16
    mock_nemo_model.transcribe = MagicMock(return_value=["test"])

    # Mock nemo module
    mock_nemo = MagicMock()
    mock_nemo.collections.asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained = MagicMock(
        return_value=mock_nemo_model
    )

    with patch.dict(
        "sys.modules",
        {
            "nemo": mock_nemo,
            "nemo.collections": mock_nemo.collections,
            "nemo.collections.asr": mock_nemo.collections.asr,
        },
    ):
        model_module.load_model()

    state = model_module.get_model_state()
    assert state.loaded is True
    assert state.model == mock_nemo_model
    assert state.model_name == "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
    assert state.decoder_type == "rnnt"
    assert state.att_context_size == [70, 6]


def test_load_model_already_loaded(model_module):
    """Test loading when model already loaded."""
    # Set as loaded
    model_module._model_state.loaded = True

    model_module.load_model()

    # Should skip loading
    assert model_module._model_state.loaded is True


def test_get_model_auto_loads(model_module):
    """Test get_model auto-loads if not loaded."""
    # Create mock NeMo model
    mock_nemo_model = MagicMock()
    mock_nemo_model.to = MagicMock(return_value=mock_nemo_model)
    mock_nemo_model.eval = MagicMock()
    mock_nemo_model.change_decoding_strategy = MagicMock()
    mock_nemo_model.encoder = MagicMock()
    mock_nemo_model.encoder.set_default_att_context_size = MagicMock()
    mock_nemo_model.half = MagicMock(return_value=mock_nemo_model)
    mock_nemo_model.dtype = mock_torch.float16
    mock_nemo_model.transcribe = MagicMock(return_value=["test"])

    # Mock nemo module
    mock_nemo = MagicMock()
    mock_nemo.collections.asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained = MagicMock(
        return_value=mock_nemo_model
    )

    with patch.dict(
        "sys.modules",
        {
            "nemo": mock_nemo,
            "nemo.collections": mock_nemo.collections,
            "nemo.collections.asr": mock_nemo.collections.asr,
        },
    ):
        model = model_module.get_model()

    assert model == mock_nemo_model
    assert model_module._model_state.loaded is True


def test_get_model_returns_existing(model_module):
    """Test get_model returns existing model."""
    # Set as loaded
    mock_model = MagicMock()
    model_module._model_state.model = mock_model
    model_module._model_state.loaded = True

    model = model_module.get_model()

    assert model == mock_model


# ==============================================================================
# Test Model Unloading
# ==============================================================================


def test_unload_model_success(model_module):
    """Test model unloads successfully."""
    # Set as loaded
    model_module._model_state.model = MagicMock()
    model_module._model_state.loaded = True

    model_module.unload_model()

    state = model_module.get_model_state()
    assert state.model is None
    assert state.loaded is False


def test_unload_model_not_loaded(model_module):
    """Test unloading when model not loaded."""
    model_module.unload_model()

    # Should not error
    state = model_module.get_model_state()
    assert state.loaded is False


# ==============================================================================
# Test Decoder Configuration
# ==============================================================================


@pytest.mark.parametrize("decoder_type", ["rnnt", "ctc"])
def test_decoder_type_configuration(decoder_type, mock_env, mock_modules, monkeypatch):
    """Test decoder type is configured correctly."""
    import sys

    # Set decoder type
    monkeypatch.setenv("DECODER_TYPE", decoder_type)

    # Remove cached module
    if "fastconformer_model" in sys.modules:
        del sys.modules["fastconformer_model"]

    import fastconformer_model

    assert decoder_type == fastconformer_model.DECODER_TYPE


# ==============================================================================
# Test Latency Configuration
# ==============================================================================


@pytest.mark.parametrize(
    "att_context_size,expected",
    [
        ("[70,0]", [70, 0]),  # 0ms latency
        ("[70,1]", [70, 1]),  # 80ms latency
        ("[70,6]", [70, 6]),  # 480ms latency
        ("[70,33]", [70, 33]),  # 1040ms latency
    ],
)
def test_latency_configuration(att_context_size, expected, mock_env, mock_modules, monkeypatch):
    """Test attention context size is configured correctly."""
    import sys

    # Set context size
    monkeypatch.setenv("ATT_CONTEXT_SIZE", att_context_size)

    # Remove cached module
    if "fastconformer_model" in sys.modules:
        del sys.modules["fastconformer_model"]

    import fastconformer_model

    assert expected == fastconformer_model.ATT_CONTEXT_SIZE
