"""
Backend Definitions

Single source of truth for ASR backend configurations.
Change BACKEND constant to switch between services.
"""

import os
from typing import Dict, Any


# ============== Backend Definitions ==============
# Local development settings (localhost)
BACKENDS_LOCAL: Dict[str, Dict[str, Any]] = {
    "vosk": {
        "name": "Vosk small-en",
        "device": "CPU",
        "host": "localhost",
        "port": 8001,
        "chunk_ms": 200,
        "mode": "streaming",
        "description": "Lightweight CPU-based streaming ASR",
    },
    "parakeet": {
        "name": "Parakeet TDT 1.1B",
        "device": "GPU",
        "host": "localhost",
        "port": 8002,
        "chunk_ms": 300,
        "mode": "300ms chunks",
        "description": "High-accuracy GPU-based ASR with word timestamps",
    },
}

# Docker internal settings (service names)
BACKENDS_DOCKER: Dict[str, Dict[str, Any]] = {
    "vosk": {
        "name": "Vosk small-en",
        "device": "CPU",
        "host": "vosk-asr",  # Docker service name
        "port": 8000,        # Docker internal port
        "chunk_ms": 200,
        "mode": "streaming",
        "description": "Lightweight CPU-based streaming ASR",
    },
    "parakeet": {
        "name": "Parakeet TDT 1.1B",
        "device": "GPU",
        "host": "parakeet-asr",  # Docker service name
        "port": 8000,            # Docker internal port
        "chunk_ms": 300,
        "mode": "300ms chunks",
        "description": "High-accuracy GPU-based ASR with word timestamps",
    },
}

# Auto-detect environment: use Docker config if running in container
IS_DOCKER = os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER", "").lower() == "true"
BACKENDS = BACKENDS_DOCKER if IS_DOCKER else BACKENDS_LOCAL

# ========== SWITCH BACKEND HERE ==========
BACKEND = os.getenv("ASR_BACKEND", "vosk")
# =========================================


def get_backend_config(backend: str = None) -> Dict[str, Any]:
    """
    Get configuration for a backend.
    
    Args:
        backend: Backend name ('vosk' or 'parakeet'). Uses default if None.
    
    Returns:
        Backend configuration dictionary.
    
    Raises:
        KeyError: If backend name is not found.
    """
    key = backend or BACKEND
    if key not in BACKENDS:
        raise KeyError(f"Unknown backend: {key}. Available: {list(BACKENDS.keys())}")
    return BACKENDS[key]


def get_display_info(backend: str = None) -> str:
    """
    Get human-readable display string for a backend.
    
    Args:
        backend: Backend name. Uses default if None.
    
    Returns:
        Formatted string like "🤖 Vosk small-en (CPU) | 🔄 streaming"
    """
    cfg = get_backend_config(backend)
    return f"🤖 {cfg['name']} ({cfg['device']}) | 🔄 {cfg['mode']}"


def list_backends() -> Dict[str, str]:
    """
    List all available backends with descriptions.
    
    Returns:
        Dict mapping backend name to description.
    """
    return {name: cfg["description"] for name, cfg in BACKENDS.items()}
