"""Model management for Parakeet ASR service."""

import logging
import os
from dataclasses import dataclass

import torch

from shared.core import clear_gpu_cache, get_gpu_manager
from shared.utils import setup_logging

logger = setup_logging(__name__)

# Suppress NeMo warnings about training/validation/test data
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

# Service name for GPU manager
SERVICE_NAME = "parakeet-asr"

# ==============================================================================
# Configuration
# ==============================================================================

STREAMING_MODEL = os.environ["PARAKEET_STREAMING_MODEL"]
OFFLINE_MODEL = os.environ["PARAKEET_OFFLINE_MODEL"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# CUDA Optimization
# ==============================================================================


def setup_cuda():
    """Configure CUDA optimizations."""
    if DEVICE != "cuda":
        return

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    clear_gpu_cache()

    logger.info("CUDA optimizations enabled")


# ==============================================================================
# Global Model State
# ==============================================================================


@dataclass
class ModelState:
    """Container for ASR model and related components."""

    # Streaming model (TDT optimized, uses FP16)
    streaming_model: object | None = None
    streaming_preprocessor: object | None = None
    streaming_loaded: bool = False
    streaming_model_name: str = ""
    # Offline model (RNNT for accuracy, uses FP32)
    offline_model: object | None = None
    offline_preprocessor: object | None = None
    offline_loaded: bool = False
    offline_model_name: str = ""


_model_state = ModelState()


def get_model_state() -> ModelState:
    """Get the global model state."""
    return _model_state


def get_model(mode: str = "streaming"):
    """Get the loaded ASR model for the specified mode. Auto-loads if not loaded.

    Uses GPU manager to coordinate memory across services.
    """
    if mode == "offline":
        if not _model_state.offline_loaded:
            logger.info("Offline model not loaded, auto-loading...")
            load_model(mode="offline")
        return _model_state.offline_model
    else:
        if not _model_state.streaming_loaded:
            logger.info("Streaming model not loaded, auto-loading...")
            load_model(mode="streaming")
        return _model_state.streaming_model


def get_preprocessor(mode: str = "streaming"):
    """Get the audio preprocessor for the specified mode. Auto-loads if not loaded."""
    if mode == "offline":
        if not _model_state.offline_loaded:
            load_model(mode="offline")
        return _model_state.offline_preprocessor
    else:
        if not _model_state.streaming_loaded:
            load_model(mode="streaming")
        return _model_state.streaming_preprocessor


# ==============================================================================
# Model Loading
# ==============================================================================


def load_model(mode: str = "streaming"):
    """Load NeMo Parakeet model for the specified mode.

    Args:
        mode: "streaming" for TDT model (FP16) or "offline" for RNNT model (FP32)
    """
    if mode == "offline":
        if _model_state.offline_loaded:
            return _model_state.offline_model
        model_name = OFFLINE_MODEL
        use_fp16 = False  # RNNT uses FP32 for best accuracy
        # Offline model (8GB) + NeMo dataloader (3GB) + inference buffers (3GB) = 14GB
        required_memory_gb = 14.0
    else:
        if _model_state.streaming_loaded:
            return _model_state.streaming_model
        model_name = STREAMING_MODEL
        use_fp16 = True  # TDT can use FP16
        required_memory_gb = 6.5  # Streaming model + overhead

    # Request GPU memory from manager (will unload other services if needed)
    gpu_mgr = get_gpu_manager()
    if not gpu_mgr.request_memory(SERVICE_NAME, mode, required_memory_gb):
        raise RuntimeError(
            f"Insufficient GPU memory for {mode} model. " f"Required: {required_memory_gb:.1f}GB"
        )

    # Clear GPU cache before loading to maximize available memory
    if DEVICE == "cuda":
        clear_gpu_cache()

    logger.info(f"Loading {mode} model: {model_name}")
    logger.info(f"Device: {DEVICE}, FP16: {use_fp16}")

    if DEVICE == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        import nemo.collections.asr as nemo_asr

        # Load model from pre-downloaded cache only (no network requests)
        model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name, map_location=DEVICE)
        model = model.to(DEVICE)
        model.eval()

        # Apply FP16 for streaming model only
        if DEVICE == "cuda" and use_fp16:
            try:
                model = model.half()
                logger.info("Model converted to FP16")
            except Exception as e:
                logger.warning(f"FP16 conversion failed: {e}")

        # Store references based on mode
        if mode == "offline":
            _model_state.offline_model = model
            _model_state.offline_preprocessor = model.preprocessor
            _model_state.offline_loaded = True
            _model_state.offline_model_name = model_name
        else:
            _model_state.streaming_model = model
            _model_state.streaming_preprocessor = model.preprocessor
            _model_state.streaming_loaded = True
            _model_state.streaming_model_name = model_name

        # Log streaming config
        if hasattr(model.encoder, "streaming_cfg"):
            logger.info(f"Encoder streaming config: {model.encoder.streaming_cfg}")

        # GPU warmup
        _warmup_model(model, mode)

        if DEVICE == "cuda":
            mem_gb = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory allocated: {mem_gb:.2f} GB")

            # Register with GPU manager
            gpu_mgr = get_gpu_manager()
            gpu_mgr.register_model(
                service_name=SERVICE_NAME,
                model_name=mode,
                memory_gb=mem_gb,
                unload_callback=lambda: (
                    _unload_streaming() if mode == "streaming" else _unload_offline()
                ),
            )

        logger.info(f"{mode.capitalize()} model loaded successfully")
        return model

    except Exception as e:
        logger.error(f"Failed to load {mode} model: {e}")
        raise


def _warmup_model(model, mode: str = "streaming"):
    """Warm up model with a test inference."""
    if DEVICE != "cuda":
        return

    use_fp16 = mode == "streaming"  # Only streaming uses FP16

    logger.info(f"Warming up {mode} model...")
    try:
        dummy_audio = torch.randn(1, 16000).to(DEVICE)
        if use_fp16:
            dummy_audio = dummy_audio.half()

        preprocessor = (
            _model_state.offline_preprocessor
            if mode == "offline"
            else _model_state.streaming_preprocessor
        )

        with torch.no_grad():
            processed, processed_len = preprocessor(
                input_signal=dummy_audio, length=torch.tensor([16000]).to(DEVICE)
            )
            _ = model.conformer_stream_step(
                processed_signal=processed,
                processed_signal_length=processed_len,
                cache_last_channel=None,
                cache_last_time=None,
                cache_last_channel_len=None,
                keep_all_outputs=True,
                previous_hypotheses=None,
                return_transcription=True,
            )

        torch.cuda.synchronize()
        logger.info(f"{mode.capitalize()} model warmup complete")
    except Exception as e:
        logger.warning(f"GPU warmup skipped: {e}")


def _unload_streaming() -> None:
    """Internal helper to unload streaming model only."""
    import gc

    if _model_state.streaming_loaded:
        if _model_state.streaming_model is not None:
            del _model_state.streaming_model
        if _model_state.streaming_preprocessor is not None:
            del _model_state.streaming_preprocessor
        _model_state.streaming_model = None
        _model_state.streaming_preprocessor = None
        _model_state.streaming_loaded = False

        if DEVICE == "cuda":
            gc.collect()
            clear_gpu_cache()
            torch.cuda.synchronize()

        # Unregister from GPU manager
        gpu_mgr = get_gpu_manager()
        gpu_mgr.unregister_model(SERVICE_NAME, "streaming")

        logger.info(f"Streaming model unloaded ({_model_state.streaming_model_name})")
        _model_state.streaming_model_name = ""


def _unload_offline() -> None:
    """Internal helper to unload offline model only."""
    import gc

    if _model_state.offline_loaded:
        if _model_state.offline_model is not None:
            del _model_state.offline_model
        if _model_state.offline_preprocessor is not None:
            del _model_state.offline_preprocessor
        _model_state.offline_model = None
        _model_state.offline_preprocessor = None
        _model_state.offline_loaded = False

        if DEVICE == "cuda":
            gc.collect()
            clear_gpu_cache()
            torch.cuda.synchronize()

        # Unregister from GPU manager
        gpu_mgr = get_gpu_manager()
        gpu_mgr.unregister_model(SERVICE_NAME, "offline")

        logger.info(f"Offline model unloaded ({_model_state.offline_model_name})")
        _model_state.offline_model_name = ""


def unload_models() -> dict:
    """Unload all models from GPU to free memory.

    Returns:
        Dict with status and unloaded models info
    """
    models_unloaded = []

    # Unload streaming model
    if _model_state.streaming_loaded:
        model_name = _model_state.streaming_model_name
        _unload_streaming()
        models_unloaded.append(f"streaming ({model_name})")

    # Unload offline model
    if _model_state.offline_loaded:
        model_name = _model_state.offline_model_name
        _unload_offline()
        models_unloaded.append(f"offline ({model_name})")

    if not models_unloaded:
        return {"status": "not_loaded", "message": "No models were loaded"}

    if DEVICE == "cuda":
        free_mem = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory
        logger.info(
            f"Models unloaded: {models_unloaded}. "
            f"GPU memory freed. Available: {free_mem/1e9:.2f}GB / {total_mem/1e9:.2f}GB"
        )
        return {
            "status": "unloaded",
            "message": f"Models unloaded: {', '.join(models_unloaded)}",
            "gpu_memory_free_gb": round(free_mem / 1e9, 2),
            "gpu_memory_total_gb": round(total_mem / 1e9, 2),
        }
    else:
        logger.info(f"Models unloaded from CPU: {models_unloaded}")
        return {
            "status": "unloaded",
            "message": f"Models unloaded: {', '.join(models_unloaded)}",
        }
