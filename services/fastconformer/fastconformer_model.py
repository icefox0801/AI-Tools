"""Model management for FastConformer ASR service."""

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
SERVICE_NAME = "fastconformer-asr"

# ==============================================================================
# Configuration
# ==============================================================================

MODEL_NAME = os.getenv(
    "FASTCONFORMER_MODEL", "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
)
DECODER_TYPE = os.getenv("DECODER_TYPE", "rnnt")  # rnnt or ctc
ATT_CONTEXT_SIZE = eval(
    os.getenv("ATT_CONTEXT_SIZE", "[70,6]")
)  # [70,0]=0ms, [70,1]=80ms, [70,6]=480ms, [70,33]=1040ms
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Flag to track CUDA initialization
_cuda_initialized = False

# ==============================================================================
# CUDA Optimization
# ==============================================================================


def setup_cuda():
    """Set up CUDA optimizations for FastConformer."""
    global _cuda_initialized

    if _cuda_initialized or DEVICE != "cuda":
        return

    logger.info("Initializing CUDA optimizations for FastConformer...")

    try:
        # Enable TF32 for faster Ampere/Blackwell GPU computation
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Use cudnn benchmark for optimal performance
        torch.backends.cudnn.benchmark = True

        logger.info("CUDA optimizations enabled (TF32, cudnn benchmark)")
        _cuda_initialized = True

    except Exception as e:
        logger.warning(f"Failed to enable CUDA optimizations: {e}")


# ==============================================================================
# Global Model State
# ==============================================================================


@dataclass
class ModelState:
    """Container for FastConformer model and components."""

    model: object | None = None
    loaded: bool = False
    model_name: str = ""
    decoder_type: str = ""
    att_context_size: list = None


_model_state = ModelState()


def get_model_state() -> ModelState:
    """Get the global model state."""
    return _model_state


# ==============================================================================
# Model Loading
# ==============================================================================


def load_model():
    """Load the FastConformer model.

    Uses GPU manager to coordinate memory across services.
    """
    global _model_state

    if _model_state.loaded:
        logger.info("Model already loaded")
        return

    logger.info(f"Loading FastConformer model: {MODEL_NAME}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Decoder: {DECODER_TYPE}")
    logger.info(f"Context size: {ATT_CONTEXT_SIZE}")

    if DEVICE == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        import nemo.collections.asr as nemo_asr

        # Load model from pre-downloaded cache only (no network requests)
        model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
            model_name=MODEL_NAME, map_location=DEVICE
        )
        model = model.to(DEVICE)
        model.eval()

        # Set decoder type (rnnt or ctc)
        model.change_decoding_strategy(decoder_type=DECODER_TYPE)
        logger.info(f"Decoder set to: {DECODER_TYPE}")

        # Set attention context size (latency)
        model.encoder.set_default_att_context_size(ATT_CONTEXT_SIZE)
        logger.info(f"Context size set to: {ATT_CONTEXT_SIZE}")

        # Enable FP16 for faster inference on GPU
        if DEVICE == "cuda":
            try:
                model = model.half()
                logger.info("Model converted to FP16")
            except Exception as e:
                logger.warning(f"FP16 conversion failed: {e}")

        # Warm up model
        _warmup_model(model)

        # Update state
        _model_state.model = model
        _model_state.loaded = True
        _model_state.model_name = MODEL_NAME
        _model_state.decoder_type = DECODER_TYPE
        _model_state.att_context_size = ATT_CONTEXT_SIZE

        # Log GPU memory
        if DEVICE == "cuda":
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(
                f"GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved"
            )

            # Report to GPU manager
            gpu_mgr = get_gpu_manager()
            gpu_mgr.report_memory(SERVICE_NAME, mem_allocated)

        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise


def _warmup_model(model):
    """Warm up model with a test inference."""
    if DEVICE != "cuda":
        return

    logger.info("Warming up model...")
    try:
        dummy_audio = torch.randn(1, 16000).to(DEVICE)
        if model.dtype == torch.float16:
            dummy_audio = dummy_audio.half()

        with torch.no_grad():
            _ = model.transcribe([dummy_audio], batch_size=1)

        logger.info("Model warmup complete")
        clear_gpu_cache()

    except Exception as e:
        logger.warning(f"Model warmup failed: {e}")


def get_model():
    """Get the loaded model. Auto-loads if not loaded."""
    if not _model_state.loaded:
        logger.info("Model not loaded, auto-loading...")
        load_model()
    return _model_state.model


def unload_model():
    """Unload model from GPU to free memory."""
    global _model_state

    if not _model_state.loaded:
        logger.info("No model loaded")
        return

    logger.info("Unloading model from GPU...")

    try:
        # Clear references
        _model_state.model = None
        _model_state.loaded = False

        # Clear GPU cache
        if DEVICE == "cuda":
            clear_gpu_cache()

            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU Memory after unload: {mem_allocated:.2f}GB")

            # Report to GPU manager
            gpu_mgr = get_gpu_manager()
            gpu_mgr.report_memory(SERVICE_NAME, 0.0)

        logger.info("Model unloaded successfully")

    except Exception as e:
        logger.error(f"Failed to unload model: {e}", exc_info=True)
        raise
