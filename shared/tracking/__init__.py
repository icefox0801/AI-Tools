"""
Chunk Tracking Module

Provides input/output paired tracking for audio transcription:
- Each audio chunk gets a unique ID
- Transcription results are linked to their source chunk(s)
- Supports replacement/regeneration with full traceability
- Renders to HTML with span IDs for UI integration
"""

from .models import AudioChunk, TranscriptSegment, SegmentStatus
from .tracker import ChunkTracker
from .styles import CHUNK_TRACKER_CSS, CHUNK_TRACKER_JS

__all__ = [
    'AudioChunk',
    'TranscriptSegment',
    'SegmentStatus',
    'ChunkTracker',
    'CHUNK_TRACKER_CSS',
    'CHUNK_TRACKER_JS',
]
