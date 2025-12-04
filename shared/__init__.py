"""
Shared Modules for AI-Tools

Provides common functionality for ASR applications:
- config: Backend definitions and configuration
- client: TranscriptManager with simple ID-based replace/append
- text_refiner: Text refinement client for punctuation/correction
- logging: Consistent logging setup across all services

Usage:
    from shared.client import TranscriptManager

    manager = TranscriptManager()
    manager.on_change = lambda: update_ui(manager.get_text())

    # On receiving message from server:
    manager.update(msg['id'], msg['text'])

Logging:
    from shared.logging import setup_logging
    logger = setup_logging(__name__)
"""

# Config
# Client - simple TranscriptManager
from .client import TranscriptManager
from .config import BACKEND, BACKENDS, get_backend_config, get_display_info
from .logging import get_logger, setup_logging

__all__ = [
    "BACKEND",
    # Config
    "BACKENDS",
    # Client
    "TranscriptManager",
    "get_backend_config",
    "get_display_info",
    # Logging
    "get_logger",
    "setup_logging",
]

__version__ = "3.1.0"
