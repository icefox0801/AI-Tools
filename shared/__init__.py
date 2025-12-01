"""
Shared Modules for AI-Tools

Provides common functionality for ASR applications:
- config: Backend definitions and configuration
- client: TranscriptManager with simple ID-based replace/append
- text_refiner: Text refinement client for punctuation/correction

Usage:
    from shared.client import TranscriptManager
    
    manager = TranscriptManager()
    manager.on_change = lambda: update_ui(manager.get_text())
    
    # On receiving message from server:
    manager.update(msg['id'], msg['text'])
"""

# Config
from .config import BACKENDS, BACKEND, get_backend_config, get_display_info

# Client - simple TranscriptManager
from .client import TranscriptManager

__all__ = [
    # Config
    'BACKENDS',
    'BACKEND', 
    'get_backend_config',
    'get_display_info',
    # Client
    'TranscriptManager',
]

__version__ = "3.1.0"
