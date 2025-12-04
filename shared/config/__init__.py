"""
ASR Configuration Module

Centralized backend definitions and configuration helpers.
"""

from .backends import (
    BACKEND,
    BACKENDS,
    fetch_service_model_name,
    format_model_name,
    get_backend_config,
    get_display_info,
)

__all__ = [
    "BACKEND",
    "BACKENDS",
    "fetch_service_model_name",
    "format_model_name",
    "get_backend_config",
    "get_display_info",
]
