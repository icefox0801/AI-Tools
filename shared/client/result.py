"""
ASR Result Data Class

Represents the result of an ASR transcription request.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ASRResult:
    """
    Result from ASR transcription.
    
    Attributes:
        text: Transcribed text
        is_final: Whether this is a final result (not partial)
        success: Whether the transcription succeeded
        error: Error message if failed
        confidence: Confidence score (0-1)
        words: Word-level timestamps if available
        chunk_ids: IDs of audio chunks that produced this result
        metadata: Additional backend-specific metadata
    """
    text: str = ""
    is_final: bool = False
    success: bool = True
    error: Optional[str] = None
    confidence: float = 1.0
    words: List[Dict[str, Any]] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_partial(cls, text: str, chunk_ids: List[str] = None) -> "ASRResult":
        """Create a partial result."""
        return cls(
            text=text,
            is_final=False,
            chunk_ids=chunk_ids or []
        )
    
    @classmethod
    def from_final(cls, text: str, chunk_ids: List[str] = None, 
                   confidence: float = 1.0) -> "ASRResult":
        """Create a final result."""
        return cls(
            text=text,
            is_final=True,
            confidence=confidence,
            chunk_ids=chunk_ids or []
        )
    
    @classmethod
    def from_error(cls, error: str) -> "ASRResult":
        """Create an error result."""
        return cls(
            text="",
            is_final=True,
            success=False,
            error=error
        )
    
    def __bool__(self) -> bool:
        """Result is truthy if successful with text."""
        return self.success and bool(self.text)
    
    def __str__(self) -> str:
        if not self.success:
            return f"ASRResult(error={self.error})"
        status = "final" if self.is_final else "partial"
        return f"ASRResult({status}: {self.text[:50]}...)" if len(self.text) > 50 else f"ASRResult({status}: {self.text})"
