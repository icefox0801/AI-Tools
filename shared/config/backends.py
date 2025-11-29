"""
Backend Definitions

Single source of truth for ASR backend configurations.
Change BACKEND constant to switch between services.
"""

import os
import urllib.request
import json
from typing import Dict, Any, Optional


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
        "name": "Parakeet",  # Will be updated from service
        "device": "GPU",
        "host": "localhost",
        "port": 8002,
        "chunk_ms": 300,
        "mode": "300ms chunks",
        "description": "High-accuracy GPU-based ASR with word timestamps",
    },
    "whisper": {
        "name": "Whisper",  # Will be updated from service
        "device": "GPU",
        "host": "localhost",
        "port": 8004,
        "chunk_ms": 500,
        "mode": "500ms chunks",
        "description": "OpenAI Whisper Large V3 Turbo - fast, multilingual ASR",
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
        "name": "Parakeet",  # Will be updated from service
        "device": "GPU",
        "host": "parakeet-asr",  # Docker service name
        "port": 8000,            # Docker internal port
        "chunk_ms": 300,
        "mode": "300ms chunks",
        "description": "High-accuracy GPU-based ASR with word timestamps",
    },
    "whisper": {
        "name": "Whisper",  # Will be updated from service
        "device": "GPU",
        "host": "whisper-asr",  # Docker service name
        "port": 8000,           # Docker internal port
        "chunk_ms": 500,
        "mode": "500ms chunks",
        "description": "OpenAI Whisper Large V3 Turbo - fast, multilingual ASR",
    },
}

# Auto-detect environment: use Docker config if running in container
IS_DOCKER = os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER", "").lower() == "true"
BACKENDS = BACKENDS_DOCKER if IS_DOCKER else BACKENDS_LOCAL

# ========== SWITCH BACKEND HERE ==========
# Options: "vosk" (CPU, lightweight), "parakeet" (GPU, high accuracy), or "whisper" (GPU, multilingual)
BACKEND = os.getenv("ASR_BACKEND", "parakeet")
# =========================================


def get_backend_config(backend: str = None) -> Dict[str, Any]:
    """
    Get configuration for a backend.
    
    Args:
        backend: Backend name ('vosk', 'parakeet', or 'whisper'). Uses default if None.
    
    Returns:
        Backend configuration dictionary.
    
    Raises:
        KeyError: If backend name is not found.
    """
    key = backend or BACKEND
    if key not in BACKENDS:
        raise KeyError(f"Unknown backend: {key}. Available: {list(BACKENDS.keys())}")
    return BACKENDS[key]


def fetch_service_model_name(backend: str = None, timeout: float = 2.0) -> Optional[str]:
    """
    Fetch the actual model name from the ASR service's /info endpoint.
    
    Args:
        backend: Backend name ('vosk', 'parakeet', or 'whisper'). Uses default if None.
        timeout: Request timeout in seconds.
    
    Returns:
        Model name string (e.g., 'nvidia/parakeet-rnnt-1.1b') or None if unavailable.
    """
    cfg = get_backend_config(backend)
    url = f"http://{cfg['host']}:{cfg['port']}/info"
    
    try:
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data.get('model_name') or data.get('model')
    except Exception:
        return None


def format_model_name(model_id: Optional[str], backend: str = None) -> str:
    """
    Format a model ID into a display-friendly name.
    
    Args:
        model_id: Full model ID (e.g., 'nvidia/parakeet-rnnt-1.1b', 'openai/whisper-large-v3-turbo')
        backend: Backend type to help format appropriately
    
    Returns:
        Formatted name (e.g., 'Parakeet RNNT 1.1B', 'Whisper Large V3 Turbo')
    """
    if not model_id:
        if backend == "whisper":
            return "Whisper"
        return "Parakeet"
    
    # Extract model name from path (e.g., 'nvidia/parakeet-rnnt-1.1b' -> 'parakeet-rnnt-1.1b')
    name = model_id.split('/')[-1]
    
    # Format: parakeet-rnnt-1.1b -> Parakeet RNNT 1.1B
    # Format: whisper-large-v3-turbo -> Whisper Large V3 Turbo
    parts = name.replace('-', ' ').split()
    formatted = []
    for part in parts:
        lower = part.lower()
        if lower in ('tdt', 'rnnt', 'ctc'):
            formatted.append(part.upper())
        elif lower.startswith('v') and len(lower) > 1 and lower[1].isdigit():
            # Version like v3 -> V3
            formatted.append(part.upper())
        elif part[0].isdigit():
            formatted.append(part.upper())  # Version numbers like 1.1b -> 1.1B
        else:
            formatted.append(part.capitalize())
    
    return ' '.join(formatted)


def get_display_info(backend: str = None) -> str:
    """
    Get human-readable display string for a backend.
    Fetches actual model name from the service if available.
    
    Args:
        backend: Backend name. Uses default if None.
    
    Returns:
        Formatted string like "🤖 Parakeet RNNT 1.1B (GPU) | 🔄 300ms chunks"
    """
    cfg = get_backend_config(backend)
    name = cfg['name']
    
    # For GPU backends (Parakeet, Whisper), try to fetch actual model name from service
    key = backend or BACKEND
    if key in ("parakeet", "whisper"):
        model_id = fetch_service_model_name(backend)
        if model_id:
            name = format_model_name(model_id, backend=key)
    
    return f"🤖 {name} ({cfg['device']}) | 🔄 {cfg['mode']}"


def list_backends() -> Dict[str, str]:
    """
    List all available backends with descriptions.
    
    Returns:
        Dict mapping backend name to description.
    """
    return {name: cfg["description"] for name, cfg in BACKENDS.items()}
