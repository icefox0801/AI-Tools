"""
ASR Client Module

Provides TranscriptManager for simple ID-based replace/append logic.

Protocol:
  Server sends: {"id": "s1", "text": "hello"}
  Client logic: ID exists → replace, ID new → append

Usage:
    from shared.client import TranscriptManager
    
    manager = TranscriptManager()
    manager.on_change = lambda: update_ui(manager.get_text())
    
    # On receiving message from server:
    manager.update(msg['id'], msg['text'])
"""

from .transcript import TranscriptManager

# Legacy imports for backward compatibility
from .websocket_client import ASRClient
from .result import ASRResult

__all__ = [
    # Simple API
    'TranscriptManager',
    # Legacy
    'ASRClient',
    'ASRResult',
]
