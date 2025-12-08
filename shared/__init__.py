"""
Shared Modules for AI-Tools

Provides common functionality for ASR applications:
- core: GPU management, logging, models
- config: Backend definitions and configuration
- client: TranscriptManager with simple ID-based replace/append
- utils: Text refinement and other utilities

Usage:
    from shared.client import TranscriptManager
    from shared.core import setup_logging, get_gpu_manager

    logger = setup_logging(__name__)
    gpu_mgr = get_gpu_manager()
"""

# Client - simple TranscriptManager
from .client import TranscriptManager
from .config import BACKEND, BACKENDS, get_backend_config, get_display_info

# Core modules (backward compatibility)
from .core import (
    clear_gpu_cache,
    get_free_memory_gb,
    get_gpu_manager,
    get_gpu_memory_info,
)
from .utils import get_logger, setup_logging

__all__ = [
    "BACKEND",
    "BACKENDS",
    "TranscriptManager",
    "clear_gpu_cache",
    "get_backend_config",
    "get_display_info",
    "get_free_memory_gb",
    "get_gpu_manager",
    "get_gpu_memory_info",
    "get_logger",
    "setup_logging",
]

__version__ = "3.2.0"
