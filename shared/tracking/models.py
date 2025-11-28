"""
Tracking Data Models

Data classes for audio chunks and transcript segments.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class SegmentStatus(Enum):
    """Status of a transcription segment."""
    PARTIAL = "partial"          # Interim result, may change
    FINAL = "final"              # Confirmed transcription
    REPLACED = "replaced"        # Replaced by correction
    REGENERATING = "regenerating"  # Being re-transcribed


@dataclass
class AudioChunk:
    """
    Represents an audio chunk sent to ASR.
    
    Attributes:
        id: Unique chunk ID (e.g., "c0001")
        timestamp: Unix timestamp when chunk was created
        duration_ms: Audio duration in milliseconds
        audio_hash: Short hash of audio data for deduplication
        status: Current processing status
        sequence: Sequence number in session
    """
    id: str
    timestamp: float
    duration_ms: int
    audio_hash: str = ""
    status: str = "pending"  # pending, transcribed, regenerating
    sequence: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "audio_hash": self.audio_hash,
            "status": self.status,
            "sequence": self.sequence,
        }
    
    def __repr__(self) -> str:
        return f"AudioChunk({self.id}, {self.duration_ms}ms, {self.status})"


@dataclass
class TranscriptSegment:
    """
    A transcription result linked to source chunk(s).
    
    Attributes:
        id: Segment ID (e.g., "s0001")
        text: Transcribed text
        chunk_ids: Source chunk IDs that produced this text
        status: Current segment status
        timestamp: Unix timestamp when segment was created
        replaced_by: If replaced, the new segment ID
        replaces: If this replaces another, the old segment ID
        confidence: Confidence score (0-1)
        word_timestamps: Word-level timing if available
    """
    id: str
    text: str
    chunk_ids: List[str]
    status: SegmentStatus
    timestamp: float
    replaced_by: Optional[str] = None
    replaces: Optional[str] = None
    confidence: float = 1.0
    word_timestamps: List[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "chunk_ids": self.chunk_ids,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "replaced_by": self.replaced_by,
            "replaces": self.replaces,
            "confidence": self.confidence,
        }
    
    @property
    def is_active(self) -> bool:
        """Check if segment is active (not replaced)."""
        return self.status != SegmentStatus.REPLACED
    
    @property
    def is_final(self) -> bool:
        """Check if segment is finalized."""
        return self.status == SegmentStatus.FINAL
    
    def __repr__(self) -> str:
        text_preview = self.text[:30] + "..." if len(self.text) > 30 else self.text
        return f"TranscriptSegment({self.id}, {self.status.value}: '{text_preview}')"
