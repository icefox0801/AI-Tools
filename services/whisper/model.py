"""Model management for Whisper ASR service.

Uses whisper-large-v3-turbo for fast, accurate transcription (~3GB VRAM).
Model is pre-downloaded to the cache volume.
"""

import gc
import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from shared.core import clear_gpu_cache, get_gpu_manager
from shared.utils import setup_logging

logger = setup_logging(__name__)

# Service name for GPU manager
SERVICE_NAME = "whisper-asr"

# ==============================================================================
# Configuration
# ==============================================================================

MODEL = os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3-turbo")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
SAMPLE_RATE = 16000

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

    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info("CUDA optimizations enabled")


# ==============================================================================
# Global Model State
# ==============================================================================


@dataclass
class ModelState:
    """Container for Whisper model and pipeline."""

    # Whisper turbo model
    pipe: object | None = None
    loaded: bool = False
    model_name: str = ""

    # Flash attention availability
    use_flash_attn: bool = False


_model_state = ModelState()


def get_model_state() -> ModelState:
    """Get the global model state."""
    return _model_state


def get_pipeline():
    """Get the Whisper pipeline. Auto-loads if not loaded.

    Returns:
        Transformers pipeline for automatic-speech-recognition
    """
    if not _model_state.loaded:
        logger.info("Model not loaded, auto-loading...")
        load_model()
    return _model_state.pipe


# ==============================================================================
# Model Loading
# ==============================================================================


def _check_flash_attention() -> bool:
    """Check if Flash Attention 2 is available."""
    if DEVICE != "cuda":
        return False

    try:
        import flash_attn  # noqa: F401

        logger.info("Flash Attention 2 available")
        return True
    except ImportError:
        logger.info("Flash Attention 2 not installed, using SDPA")
        return False


def _create_pipeline(model_name: str) -> object:
    """Create a Whisper pipeline for the given model from pre-cached files.

    Args:
        model_name: HuggingFace model name (must be pre-downloaded in cache)

    Returns:
        Transformers pipeline for automatic-speech-recognition
    """
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Device: {DEVICE}, dtype: {TORCH_DTYPE}")

    # Check Flash Attention
    use_flash_attn = _check_flash_attention()
    _model_state.use_flash_attn = use_flash_attn

    # Model loading kwargs
    model_kwargs = {
        "torch_dtype": TORCH_DTYPE,
        "low_cpu_mem_usage": True,
        "use_safetensors": True,
        "local_files_only": True,  # Only use pre-cached models
    }

    if DEVICE == "cuda":
        if use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            model_kwargs["attn_implementation"] = "sdpa"

    # Load model from cache
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, **model_kwargs)
    model.to(DEVICE)

    processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=TORCH_DTYPE,
        device=DEVICE,
    )

    # Warmup
    if DEVICE == "cuda":
        logger.info("Warming up GPU...")
        import numpy as np

        dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 second
        _ = pipe(dummy, generate_kwargs={"max_new_tokens": 10})
        torch.cuda.synchronize()

        mem_gb = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"GPU Memory: {mem_gb:.2f}GB allocated")

    logger.info(f"Model {model_name} loaded successfully")
    return pipe


def load_model(mode: str = "streaming"):
    """Load Whisper model.

    Args:
        mode: Kept for backward compatibility, ignored (always loads turbo)
    """
    if _model_state.loaded:
        logger.info("Model already loaded")
        return

    model_name = MODEL

    # Ensure GPU is ready - unloads ALL other models (same-service + other-services)
    gpu_mgr = get_gpu_manager()
    gpu_mgr.ensure_model_ready(SERVICE_NAME, "whisper-turbo")

    # Load the model
    pipe = _create_pipeline(model_name)

    # Store model
    _model_state.pipe = pipe
    _model_state.loaded = True
    _model_state.model_name = model_name

    # Register with GPU manager
    if DEVICE == "cuda":
        mem_gb = torch.cuda.memory_allocated() / 1024**3
        gpu_mgr.register_model(
            service_name=SERVICE_NAME,
            model_name="whisper-turbo",
            memory_gb=mem_gb,
            unload_callback=_unload_model,
        )


# ==============================================================================
# Model Unloading
# ==============================================================================


def _unload_model() -> None:
    """Internal helper to unload model."""
    if _model_state.loaded:
        if _model_state.pipe is not None:
            del _model_state.pipe
        _model_state.pipe = None
        _model_state.loaded = False

        if DEVICE == "cuda":
            gc.collect()
            clear_gpu_cache()
            torch.cuda.synchronize()

        # Unregister from GPU manager
        gpu_mgr = get_gpu_manager()
        gpu_mgr.unregister_model(SERVICE_NAME, "whisper-turbo")

        logger.info(f"Model unloaded ({_model_state.model_name})")
        _model_state.model_name = ""


def unload_model():
    """Unload model from GPU to free memory."""
    if _model_state.loaded:
        model_name = _model_state.model_name
        _unload_model()

        if DEVICE == "cuda":
            free_mem = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory
            logger.info(
                f"Model unloaded: {model_name}. "
                f"Available: {free_mem/1e9:.2f}GB / {total_mem/1e9:.2f}GB"
            )
        return True

    return False
