"""
Shared Models for Transcription Services

Defines common data structures for consistent API responses across all ASR backends.
These models are used by:
- Transcription Gateway service
- Parakeet, Whisper, Vosk backends
- Client applications

Protocol:
- Streaming: Send {"id": "s0", "text": "Hello"} for real-time updates
- Offline: Return TranscriptionResponse with segments
"""

from dataclasses import dataclass, field

# Try to import Pydantic for API models, fallback to dataclasses
try:
    from pydantic import BaseModel

    class StreamingMessage(BaseModel):
        """Message format for streaming transcription.

        Protocol:
        - Client displays text by ID
        - If ID exists, replace text; if new ID, append as new segment
        - Optionally mark as final (no more updates for this ID)
        """

        id: str  # Segment ID (e.g., "s0", "s1")
        text: str  # Transcribed text
        is_final: bool = False  # Whether this segment is finalized

    class WordTiming(BaseModel):
        """Word-level timing information."""

        word: str
        start: float
        end: float
        confidence: float | None = None

    class TranscriptionSegment(BaseModel):
        """A segment of transcribed text with timing."""

        id: str
        text: str
        start: float = 0.0
        end: float = 0.0
        is_final: bool = True
        words: list[WordTiming] | None = None

    class TranscriptionResponse(BaseModel):
        """Response from offline transcription."""

        text: str
        segments: list[TranscriptionSegment] = []
        duration: float = 0.0
        model: str = ""
        language: str | None = None

    class HealthResponse(BaseModel):
        """Health check response from ASR backends."""

        status: str = "healthy"
        backend: str  # parakeet, whisper, vosk
        device: str = "cpu"  # cpu, cuda, etc.
        model: str | None = None
        sample_rate: int = 16000
        streaming: bool = True
        api_version: str = "3.2"

except ImportError:
    # Fallback to dataclasses if Pydantic not available
    @dataclass
    class StreamingMessage:
        """Message format for streaming transcription."""

        id: str
        text: str
        is_final: bool = False

    @dataclass
    class WordTiming:
        """Word-level timing information."""

        word: str
        start: float
        end: float
        confidence: float | None = None

    @dataclass
    class TranscriptionSegment:
        """A segment of transcribed text with timing."""

        id: str
        text: str
        start: float = 0.0
        end: float = 0.0
        is_final: bool = True
        words: list["WordTiming"] | None = None

    @dataclass
    class TranscriptionResponse:
        """Response from offline transcription."""

        text: str
        segments: list[TranscriptionSegment] = field(default_factory=list)
        duration: float = 0.0
        model: str = ""
        language: str | None = None

    @dataclass
    class HealthResponse:
        """Health check response from ASR backends."""

        status: str = "healthy"
        backend: str = ""
        device: str = "cpu"
        model: str | None = None
        sample_rate: int = 16000
        streaming: bool = True
        api_version: str = "3.2"


# Utility functions for creating messages
def streaming_message(segment_id: str, text: str, is_final: bool = False) -> dict:
    """Create a streaming message dict for WebSocket send."""
    return {"id": segment_id, "text": text, "is_final": is_final}


def next_segment_id(current_id: str) -> str:
    """Generate the next segment ID.

    Args:
        current_id: Current segment ID (e.g., "s5")

    Returns:
        Next segment ID (e.g., "s6")
    """
    if current_id.startswith("s"):
        try:
            num = int(current_id[1:])
            return f"s{num + 1}"
        except ValueError:
            pass
    return "s0"


__all__ = [
    "HealthResponse",
    "StreamingMessage",
    "TranscriptionResponse",
    "TranscriptionSegment",
    "WordTiming",
    "next_segment_id",
    "streaming_message",
]
