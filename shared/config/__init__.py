"""
ASR Configuration Module

Centralized backend definitions and configuration helpers.
"""

from .backends import BACKENDS, BACKEND, get_backend_config, get_display_info

__all__ = ['BACKENDS', 'BACKEND', 'get_backend_config', 'get_display_info']
