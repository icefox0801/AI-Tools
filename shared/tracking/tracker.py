"""
Chunk Tracker

Main tracking class that manages audio chunks and transcript segments.
"""

import time
import uuid
import hashlib
import json
from typing import List, Dict, Optional, Callable

from .models import AudioChunk, TranscriptSegment, SegmentStatus


class ChunkTracker:
    """
    Tracks audio chunks and their transcription results.
    
    Provides:
    - Unique ID generation for chunks and segments
    - Mapping between chunks and transcription segments
    - Replacement/regeneration tracking
    - HTML rendering with span IDs
    - JSON export for debugging/logging
    
    Usage:
        tracker = ChunkTracker()
        
        # When sending audio
        chunk_id = tracker.create_chunk(duration_ms=200)
        
        # When receiving partial results
        tracker.add_partial("hello wor")
        
        # When receiving final results
        tracker.finalize("hello world")
        
        # Get HTML output
        html = tracker.to_html()
        
        # Replace a segment (e.g., for punctuation)
        tracker.replace_segment("s0001", "Hello, world!")
    
    Threading: NOT thread-safe. Use locks if accessing from multiple threads.
    """
    
    def __init__(self, session_id: str = None):
        """
        Initialize chunk tracker.
        
        Args:
            session_id: Optional session identifier (auto-generated if not provided)
        """
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.chunks: Dict[str, AudioChunk] = {}
        self.segments: Dict[str, TranscriptSegment] = {}
        self._chunk_counter = 0
        self._segment_counter = 0
        self._current_partial: Optional[TranscriptSegment] = None
        self._pending_chunk_ids: List[str] = []
        
        # Callbacks for UI updates
        self._on_segment_added: Optional[Callable[[TranscriptSegment], None]] = None
        self._on_segment_replaced: Optional[Callable[[str, str], None]] = None
    
    # ==================== Callbacks ====================
    
    def on_segment_added(self, callback: Callable[[TranscriptSegment], None]):
        """Register callback for when a segment is added."""
        self._on_segment_added = callback
    
    def on_segment_replaced(self, callback: Callable[[str, str], None]):
        """Register callback for when a segment is replaced. Args: (old_id, new_id)"""
        self._on_segment_replaced = callback
    
    # ==================== ID Generation ====================
    
    def _next_chunk_id(self) -> str:
        """Generate next chunk ID."""
        self._chunk_counter += 1
        return f"c{self._chunk_counter:04d}"
    
    def _next_segment_id(self) -> str:
        """Generate next segment ID."""
        self._segment_counter += 1
        return f"s{self._segment_counter:04d}"
    
    @staticmethod
    def compute_audio_hash(audio_bytes: bytes) -> str:
        """Compute short hash of audio data."""
        return hashlib.md5(audio_bytes).hexdigest()[:8]
    
    # ==================== Chunk Management ====================
    
    def create_chunk(self, duration_ms: int = 0, audio_bytes: bytes = None) -> str:
        """
        Register a new audio chunk.
        
        Args:
            duration_ms: Duration of audio in milliseconds
            audio_bytes: Optional audio data (for hash computation)
        
        Returns:
            Chunk ID (e.g., "c0001")
        """
        chunk_id = self._next_chunk_id()
        audio_hash = self.compute_audio_hash(audio_bytes) if audio_bytes else ""
        
        self.chunks[chunk_id] = AudioChunk(
            id=chunk_id,
            timestamp=time.time(),
            duration_ms=duration_ms,
            audio_hash=audio_hash,
            status="pending",
            sequence=self._chunk_counter
        )
        
        self._pending_chunk_ids.append(chunk_id)
        return chunk_id
    
    def get_chunk(self, chunk_id: str) -> Optional[AudioChunk]:
        """Get chunk by ID."""
        return self.chunks.get(chunk_id)
    
    def get_pending_chunks(self) -> List[AudioChunk]:
        """Get all pending (not yet transcribed) chunks."""
        return [self.chunks[cid] for cid in self._pending_chunk_ids if cid in self.chunks]
    
    # ==================== Segment Management ====================
    
    def add_partial(self, text: str, chunk_ids: List[str] = None) -> str:
        """
        Add or update partial (interim) transcription.
        
        Partial results update in place until finalized.
        
        Args:
            text: Partial transcription text
            chunk_ids: Source chunk IDs (uses pending if not provided)
        
        Returns:
            Segment ID
        """
        chunk_ids = chunk_ids or self._pending_chunk_ids.copy()
        
        if self._current_partial:
            # Update existing partial
            self._current_partial.text = text
            self._current_partial.chunk_ids = chunk_ids
            return self._current_partial.id
        
        # Create new partial
        segment_id = self._next_segment_id()
        self._current_partial = TranscriptSegment(
            id=segment_id,
            text=text,
            chunk_ids=chunk_ids,
            status=SegmentStatus.PARTIAL,
            timestamp=time.time()
        )
        self.segments[segment_id] = self._current_partial
        return segment_id
    
    def finalize(self, text: str, chunk_ids: List[str] = None) -> str:
        """
        Finalize transcription (convert partial to final or add new final).
        
        Args:
            text: Final transcription text
            chunk_ids: Source chunk IDs (uses pending if not provided)
        
        Returns:
            Segment ID
        """
        chunk_ids = chunk_ids or self._pending_chunk_ids.copy()
        
        if self._current_partial:
            # Finalize existing partial
            segment = self._current_partial
            segment.text = text
            segment.chunk_ids = chunk_ids
            segment.status = SegmentStatus.FINAL
            self._current_partial = None
        else:
            # Create new final segment
            segment_id = self._next_segment_id()
            segment = TranscriptSegment(
                id=segment_id,
                text=text,
                chunk_ids=chunk_ids,
                status=SegmentStatus.FINAL,
                timestamp=time.time()
            )
            self.segments[segment_id] = segment
        
        # Mark chunks as transcribed
        for cid in chunk_ids:
            if cid in self.chunks:
                self.chunks[cid].status = "transcribed"
        
        # Clear pending chunks that were finalized
        self._pending_chunk_ids = [
            cid for cid in self._pending_chunk_ids 
            if cid not in chunk_ids
        ]
        
        # Fire callback
        if self._on_segment_added:
            self._on_segment_added(segment)
        
        return segment.id
    
    def get_segment(self, segment_id: str) -> Optional[TranscriptSegment]:
        """Get segment by ID."""
        return self.segments.get(segment_id)
    
    # ==================== Replacement/Regeneration ====================
    
    def replace_segment(
        self,
        segment_id: str,
        new_text: str,
        reason: str = "correction"
    ) -> Optional[str]:
        """
        Replace a segment with corrected text.
        
        Args:
            segment_id: ID of segment to replace
            new_text: Corrected text
            reason: Reason for replacement (e.g., "punctuation", "grammar")
        
        Returns:
            New segment ID, or None if original not found
        """
        if segment_id not in self.segments:
            return None
        
        old_segment = self.segments[segment_id]
        old_segment.status = SegmentStatus.REPLACED
        
        # Create replacement segment
        new_segment_id = self._next_segment_id()
        new_segment = TranscriptSegment(
            id=new_segment_id,
            text=new_text,
            chunk_ids=old_segment.chunk_ids,
            status=SegmentStatus.FINAL,
            timestamp=time.time(),
            replaces=segment_id,
            confidence=old_segment.confidence
        )
        self.segments[new_segment_id] = new_segment
        old_segment.replaced_by = new_segment_id
        
        # Fire callback
        if self._on_segment_replaced:
            self._on_segment_replaced(segment_id, new_segment_id)
        
        return new_segment_id
    
    def replace_by_chunk(
        self,
        chunk_id: str,
        new_text: str,
        reason: str = "correction"
    ) -> Optional[str]:
        """
        Replace segment(s) associated with a chunk.
        
        Args:
            chunk_id: Chunk ID whose segment should be replaced
            new_text: Corrected text
            reason: Reason for replacement
        
        Returns:
            New segment ID, or None if no segment found for chunk
        """
        # Find active segment containing this chunk
        for segment in self.segments.values():
            if (chunk_id in segment.chunk_ids and segment.is_active):
                return self.replace_segment(segment.id, new_text, reason)
        return None
    
    def mark_regenerating(self, chunk_ids: List[str]):
        """Mark chunks as being regenerated."""
        for cid in chunk_ids:
            if cid in self.chunks:
                self.chunks[cid].status = "regenerating"
        
        # Mark affected segments
        for segment in self.segments.values():
            if any(cid in chunk_ids for cid in segment.chunk_ids):
                if segment.is_active:
                    segment.status = SegmentStatus.REGENERATING
    
    # ==================== Query Methods ====================
    
    def get_active_segments(self) -> List[TranscriptSegment]:
        """Get all active (non-replaced) segments in order."""
        active = [s for s in self.segments.values() if s.is_active]
        return sorted(active, key=lambda s: s.timestamp)
    
    def get_final_segments(self) -> List[TranscriptSegment]:
        """Get only finalized segments (no partials)."""
        return [s for s in self.get_active_segments() if s.is_final]
    
    def get_text(self, include_partial: bool = True) -> str:
        """
        Get full transcript text.
        
        Args:
            include_partial: Include partial/interim text
        
        Returns:
            Full transcript string
        """
        if include_partial:
            segments = self.get_active_segments()
        else:
            segments = self.get_final_segments()
        
        return " ".join(s.text for s in segments if s.text)
    
    def get_partial_text(self) -> Optional[str]:
        """Get current partial text, if any."""
        if self._current_partial:
            return self._current_partial.text
        return None
    
    def get_chunks_for_segment(self, segment_id: str) -> List[AudioChunk]:
        """Get all chunks that produced a segment."""
        segment = self.segments.get(segment_id)
        if not segment:
            return []
        return [self.chunks[cid] for cid in segment.chunk_ids if cid in self.chunks]
    
    def get_segments_for_chunk(self, chunk_id: str) -> List[TranscriptSegment]:
        """Get all segments produced from a chunk (including replaced)."""
        return [s for s in self.segments.values() if chunk_id in s.chunk_ids]
    
    def get_chunk_segment_map(self) -> Dict[str, List[str]]:
        """Get mapping of chunk IDs to segment IDs."""
        mapping: Dict[str, List[str]] = {}
        for segment in self.segments.values():
            for chunk_id in segment.chunk_ids:
                if chunk_id not in mapping:
                    mapping[chunk_id] = []
                mapping[chunk_id].append(segment.id)
        return mapping
    
    def get_replacement_chain(self, segment_id: str) -> List[str]:
        """
        Get the chain of replacements for a segment.
        
        Returns list from original to most recent.
        """
        chain = [segment_id]
        
        # Walk backwards to find original
        current = segment_id
        while current in self.segments:
            segment = self.segments[current]
            if segment.replaces:
                chain.insert(0, segment.replaces)
                current = segment.replaces
            else:
                break
        
        # Walk forwards to find latest
        current = segment_id
        while current in self.segments:
            segment = self.segments[current]
            if segment.replaced_by:
                chain.append(segment.replaced_by)
                current = segment.replaced_by
            else:
                break
        
        return chain
    
    # ==================== Output Methods ====================
    
    def to_html(
        self, 
        include_partial: bool = True,
        wrapper_class: str = "transcript"
    ) -> str:
        """
        Render transcript as HTML with tracking spans.
        
        Each segment is wrapped in a span with data attributes:
        - data-segment: Segment ID
        - data-chunks: Comma-separated chunk IDs
        - data-status: Segment status
        - class: Status class (partial, final, regenerating)
        
        Args:
            include_partial: Include partial results (shown with "...")
            wrapper_class: CSS class for outer container
        
        Returns:
            HTML string
        """
        segments = self.get_active_segments()
        html_parts = []
        
        for segment in segments:
            if not segment.text:
                continue
            
            if not include_partial and segment.status == SegmentStatus.PARTIAL:
                continue
            
            chunk_ids_str = ",".join(segment.chunk_ids)
            status_class = segment.status.value
            
            # Escape HTML in text
            text = (segment.text
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"))
            
            # Add visual indicator for partials
            if segment.status == SegmentStatus.PARTIAL:
                text = f"{text}<span class='partial-indicator'>...</span>"
            
            html = (
                f'<span data-segment="{segment.id}" '
                f'data-chunks="{chunk_ids_str}" '
                f'data-status="{segment.status.value}" '
                f'class="segment {status_class}">'
                f'{text}</span>'
            )
            html_parts.append(html)
        
        content = " ".join(html_parts)
        return f'<div class="{wrapper_class}">{content}</div>'
    
    def to_json(self) -> str:
        """Export tracker state as JSON for debugging."""
        return json.dumps({
            "session_id": self.session_id,
            "chunks": {cid: c.to_dict() for cid, c in self.chunks.items()},
            "segments": {sid: s.to_dict() for sid, s in self.segments.items()},
            "pending_chunks": self._pending_chunk_ids,
            "text": self.get_text(),
        }, indent=2)
    
    def to_dict(self) -> dict:
        """Export tracker state as dictionary."""
        return {
            "session_id": self.session_id,
            "chunks": {cid: c.to_dict() for cid, c in self.chunks.items()},
            "segments": {sid: s.to_dict() for sid, s in self.segments.items()},
            "pending_chunks": self._pending_chunk_ids,
            "text": self.get_text(),
            "stats": self.stats,
        }
    
    # ==================== Lifecycle ====================
    
    def clear(self):
        """Clear all tracking data."""
        self.chunks.clear()
        self.segments.clear()
        self._chunk_counter = 0
        self._segment_counter = 0
        self._current_partial = None
        self._pending_chunk_ids.clear()
    
    @property
    def stats(self) -> dict:
        """Get tracking statistics."""
        return {
            "total_chunks": len(self.chunks),
            "pending_chunks": len(self._pending_chunk_ids),
            "total_segments": len(self.segments),
            "active_segments": len(self.get_active_segments()),
            "final_segments": len(self.get_final_segments()),
            "has_partial": self._current_partial is not None,
        }
    
    def __repr__(self) -> str:
        stats = self.stats
        return (
            f"ChunkTracker(session={self.session_id}, "
            f"chunks={stats['total_chunks']}, "
            f"segments={stats['active_segments']})"
        )
