"""Model management for Whisper ASR service.

Supports dual models:
- Streaming: whisper-large-v3-turbo (fast, ~3GB VRAM)
- Offline: whisper-large-v3 (accurate, ~10GB VRAM)

Models are pre-downloaded to the cache volume.
"""

import gc
import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from shared.logging import setup_logging

logger = setup_logging(__name__)

# ==============================================================================
# Configuration
# ==============================================================================

STREAMING_MODEL = os.environ.get("WHISPER_STREAMING_MODEL", "openai/whisper-large-v3-turbo")
OFFLINE_MODEL = os.environ.get("WHISPER_OFFLINE_MODEL", "openai/whisper-large-v3")
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
    torch.cuda.empty_cache()

    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info("CUDA optimizations enabled")


# ==============================================================================
# Global Model State
# ==============================================================================


@dataclass
class ModelState:
    """Container for Whisper models and pipelines."""

    # Streaming model (turbo - fast)
    streaming_pipe: object | None = None
    streaming_loaded: bool = False
    streaming_model_name: str = ""

    # Offline model (large-v3 - accurate)
    offline_pipe: object | None = None
    offline_loaded: bool = False
    offline_model_name: str = ""

    # Flash attention availability
    use_flash_attn: bool = False


_model_state = ModelState()


def get_model_state() -> ModelState:
    """Get the global model state."""
    return _model_state


def get_pipeline(mode: str = "streaming"):
    """Get the Whisper pipeline for the specified mode. Auto-loads if not loaded.

    Args:
        mode: 'streaming' for turbo model, 'offline' for large-v3 model

    Returns:
        Transformers pipeline for automatic-speech-recognition
    """
    if mode == "offline":
        if not _model_state.offline_loaded:
            logger.info("Offline model not loaded, auto-loading...")
            load_model(mode="offline")
        return _model_state.offline_pipe
    else:
        if not _model_state.streaming_loaded:
            logger.info("Streaming model not loaded, auto-loading...")
            load_model(mode="streaming")
        return _model_state.streaming_pipe


# ==============================================================================
# Model Loading
# ==============================================================================


def _check_flash_attention() -> bool:
    """Check if Flash Attention 2 is available."""
    if DEVICE != "cuda":
        return False

    try:
        import flash_attn

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
    """Load Whisper model for the specified mode.

    Args:
        mode: 'streaming' for turbo model, 'offline' for large-v3
    """
    if mode == "offline":
        if _model_state.offline_loaded:
            logger.info("Offline model already loaded")
            return

        model_name = OFFLINE_MODEL
        _model_state.offline_pipe = _create_pipeline(model_name)
        _model_state.offline_loaded = True
        _model_state.offline_model_name = model_name

    else:  # streaming
        if _model_state.streaming_loaded:
            logger.info("Streaming model already loaded")
            return

        model_name = STREAMING_MODEL
        _model_state.streaming_pipe = _create_pipeline(model_name)
        _model_state.streaming_loaded = True
        _model_state.streaming_model_name = model_name


def unload_model(mode: str = "all"):
    """Unload model(s) from GPU to free memory.

    Args:
        mode: 'streaming', 'offline', or 'all'
    """
    freed_memory = False

    if mode in ("streaming", "all") and _model_state.streaming_loaded:
        logger.info("Unloading streaming model...")
        _model_state.streaming_pipe = None
        _model_state.streaming_loaded = False
        _model_state.streaming_model_name = ""
        freed_memory = True

    if mode in ("offline", "all") and _model_state.offline_loaded:
        logger.info("Unloading offline model...")
        _model_state.offline_pipe = None
        _model_state.offline_loaded = False
        _model_state.offline_model_name = ""
        freed_memory = True

    if freed_memory:
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            free_mem = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory
            logger.info(
                f"GPU memory freed. Available: {free_mem/1e9:.2f}GB / {total_mem/1e9:.2f}GB"
            )

    return freed_memory
