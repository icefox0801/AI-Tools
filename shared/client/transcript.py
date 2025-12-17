"""
Transcript Manager

Simple transcript management with ID-based replace/append logic.
Decoupled from services and UI - handles all transcription state.

Protocol:
  Server sends: {"id": "s1", "text": "hello"}           # single segment
  Server sends: {"id": "s0-s5", "text": "final text"}   # replace range

Client logic:
  - Single ID (e.g., "s1"): If exists → replace, if new → append
  - Range ID (e.g., "s0-s5"): Remove s0,s1,s2,s3,s4,s5 and replace with one segment

The server controls segmentation - client just displays.
"""

import logging
from collections import OrderedDict
from collections.abc import Callable

# Set up logger for this module
logger = logging.getLogger(__name__)


def parse_segment_id(segment_id: str) -> tuple[str | None, str | None]:
    """
    Parse segment ID which can be single or range.

    Returns:
        (start_id, end_id) - if single, both are the same

    Examples:
        "s5" -> ("s5", "s5")
        "s0-s5" -> ("s0", "s5")
    """
    if "-" in segment_id and segment_id.count("-") == 1:
        # Check if it's a range like "s0-s5"
        parts = segment_id.split("-")
        if len(parts) == 2:
            start, end = parts
            # Verify both look like segment IDs (start with 's' followed by number)
            if (
                start.startswith("s")
                and start[1:].isdigit()
                and end.startswith("s")
                and end[1:].isdigit()
            ):
                return (start, end)

    # Single ID
    return (segment_id, segment_id)


def get_segment_number(segment_id: str) -> int | None:
    """Extract number from segment ID like 's5' -> 5"""
    if segment_id.startswith("s") and segment_id[1:].isdigit():
        return int(segment_id[1:])
    return None


class TranscriptManager:
    """
    Manages transcript segments with ID-based replace/append.

    Simple API:
        manager = TranscriptManager()
        manager.update("s0", "hello")       # Appends new segment
        manager.update("s0", "hello world") # Replaces existing (same ID)
        manager.update("s1", "how are")     # Appends new segment (new ID)
        manager.update("s1", "how are you") # Replaces existing (same ID)

        print(manager.get_text())  # "hello world how are you"

    Callbacks:
        manager.on_change = lambda: update_ui(manager.get_text())

    The manager doesn't care about partial/final - that's server logic.
    It just maintains an ordered dict of {id: text} and renders them.
    """

    def __init__(self, max_segments: int = 100):
        """
        Initialize transcript manager.

        Args:
            max_segments: Maximum segments to keep (oldest removed first)
        """
        # OrderedDict preserves insertion order
        self._segments: OrderedDict[str, str] = OrderedDict()
        self._max_segments = max_segments

        # Callback when transcript changes
        self.on_change: Callable[[], None] | None = None

    def update(self, segment_id: str, text: str) -> bool:
        """
        Update or add a segment. Supports single ID or range.

        Args:
            segment_id: Segment ID from server
                - Single: "s0", "s1" etc
                - Range: "s0-s5" (replaces s0,s1,s2,s3,s4,s5 with one segment)
            text: Transcribed text

        Returns:
            True if segment(s) replaced, False if new segment appended
        """
        start_id, end_id = parse_segment_id(segment_id)
        start_num = get_segment_number(start_id)
        end_num = get_segment_number(end_id)

        is_replace = False

        # Handle range replacement (e.g., "s0-s5")
        if start_id != end_id and start_num is not None and end_num is not None:
            # Remove all segments in range
            ids_to_remove = []
            for sid in self._segments:
                num = get_segment_number(sid)
                if num is not None and start_num <= num <= end_num:
                    ids_to_remove.append(sid)

            if ids_to_remove:
                is_replace = True
                logger.debug(f"[RANGE] {segment_id}: removing {ids_to_remove}")
                for sid in ids_to_remove:
                    del self._segments[sid]

            # Add consolidated segment with the range ID
            self._segments[segment_id] = text
            logger.debug(f"[RANGE] {segment_id} = '{text[:50]}...' (consolidated)")
        else:
            # Single segment update
            is_replace = start_id in self._segments
            action = "REPLACE" if is_replace else "APPEND"
            logger.debug(f"[{action}] {start_id} = '{text[:50] if len(text) > 50 else text}'")
            self._segments[start_id] = text

        # Enforce max segments (remove oldest)
        while len(self._segments) > self._max_segments:
            self._segments.popitem(last=False)

        # Fire callback
        if self.on_change:
            try:
                self.on_change()
            except Exception:
                pass  # Don't let callback errors break the manager

        return is_replace

    def get_text(self, max_words: int | None = 300) -> str:
        """
        Get full transcript text.

        Args:
            max_words: Maximum words to return (keeps last N words). None for all.

        Returns:
            Concatenated transcript from all segments (space-separated)
        """
        full_text = " ".join(text for text in self._segments.values() if text)

        # Truncate to max words if specified
        if max_words is not None:
            words = full_text.split()
            if len(words) > max_words:
                full_text = " ".join(words[-max_words:])

        return full_text

    def get_segments(self) -> list[tuple]:
        """
        Get all segments as (id, text) tuples in order.

        Returns:
            List of (segment_id, text) tuples
        """
        return list(self._segments.items())

    def get_segment(self, segment_id: str) -> str | None:
        """Get text for a specific segment ID."""
        return self._segments.get(segment_id)

    def get_last_segment(self) -> tuple | None:
        """
        Get the most recent segment.

        Returns:
            (segment_id, text) tuple or None if empty
        """
        if self._segments:
            segment_id = next(reversed(self._segments))
            return (segment_id, self._segments[segment_id])
        return None

    def clear(self) -> None:
        """Clear all segments."""
        self._segments.clear()
        if self.on_change:
            try:
                self.on_change()
            except Exception:
                pass

    def remove(self, segment_id: str) -> bool:
        """
        Remove a specific segment.

        Returns:
            True if removed, False if not found
        """
        if segment_id in self._segments:
            del self._segments[segment_id]
            if self.on_change:
                try:
                    self.on_change()
                except Exception:
                    pass
            return True
        return False

    @property
    def segment_count(self) -> int:
        """Number of segments."""
        return len(self._segments)

    @property
    def is_empty(self) -> bool:
        """Check if transcript is empty."""
        return len(self._segments) == 0

    def to_html(self) -> str:
        """
        Render transcript as HTML with span-wrapped segments.

        Each segment gets a span with data-id attribute for potential
        styling or JavaScript manipulation.

        Returns:
            HTML string
        """
        parts = []
        for segment_id, text in self._segments.items():
            if text:
                parts.append(f'<span class="segment" data-id="{segment_id}">{text}</span>')
        return " ".join(parts)

    def __repr__(self) -> str:
        return f"TranscriptManager({self.segment_count} segments)"

    def __len__(self) -> int:
        return self.segment_count
