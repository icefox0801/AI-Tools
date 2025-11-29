"""
ASR Configuration Module

Centralized backend definitions and configuration helpers.
"""

from .backends import (
    BACKENDS, 
    BACKEND, 
    get_backend_config, 
    get_display_info,
    fetch_service_model_name,
    format_model_name,
)

__all__ = [
    'BACKENDS', 
    'BACKEND', 
    'get_backend_config', 
    'get_display_info',
    'fetch_service_model_name',
    'format_model_name',
]
