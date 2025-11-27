"""
Session State Module

Manages transcription session state and history.
"""

import uuid
import numpy as np
from dataclasses import dataclass, field


@dataclass
class SessionState:
    """
    Manages state for a transcription session.
    
    Server-side accumulation model:
    - Server accumulates raw text and applies punctuation
    - Server returns FULL transcript each time
    - Client just displays what server sends
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pending_audio: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    full_text: str = ""  # Full transcript from server (already punctuated)
    last_display: str = ""
    sample_rate: int = 16000
    
    def reset(self) -> None:
        """Reset session to initial state with new ID."""
        self.session_id = str(uuid.uuid4())
        self.pending_audio = np.array([], dtype=np.float32)
        self.full_text = ""
        self.last_display = ""
    
    def append_audio(self, audio: np.ndarray) -> None:
        """Append audio to pending buffer."""
        self.pending_audio = np.concatenate([self.pending_audio, audio])
    
    def set_full_transcript(self, text: str) -> None:
        """Set the full transcript (from server, already punctuated)."""
        if text:
            self.full_text = text.strip()
    
    def clear_pending_audio(self) -> None:
        """Clear pending audio after successful transcription."""
        self.pending_audio = np.array([], dtype=np.float32)
    
    def get_pending_audio(self) -> np.ndarray:
        """Get pending audio for transcription."""
        return self.pending_audio
    
    @property
    def pending_duration(self) -> float:
        """Duration of pending audio in seconds."""
        return len(self.pending_audio) / self.sample_rate
    
    @property
    def full_transcript(self) -> str:
        """Get complete transcript."""
        return self.full_text
    
    def build_display(self, show_pending_indicator: bool = True) -> str:
        """
        Build display string with transcript and status.
        
        Args:
            show_pending_indicator: Show "…" when audio is pending transcription
        
        Returns:
            Formatted display string
        """
        if self.full_text:
            text = self.full_text
            if show_pending_indicator and self.pending_duration > 0.5:
                text += " …"  # Unicode ellipsis - more audio pending
            self.last_display = text
        elif self.pending_duration > 0.5:
            self.last_display = "🎤 Listening…"
        
        return self.last_display
    
    def __repr__(self) -> str:
        return (
            f"SessionState(id={self.session_id[:8]}..., "
            f"text_len={len(self.full_text)}, "
            f"pending={self.pending_duration:.1f}s)"
        )
