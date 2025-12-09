"""
Unit tests for shared.client.transcript module.
"""

import pytest

from shared.client.transcript import (
    TranscriptManager,
    get_segment_number,
    parse_segment_id,
)


class TestParseSegmentId:
    """Tests for parse_segment_id function."""

    def test_parse_single_id(self):
        """Test parsing single segment ID."""
        assert parse_segment_id("s5") == ("s5", "s5")
        assert parse_segment_id("s0") == ("s0", "s0")
        assert parse_segment_id("s123") == ("s123", "s123")

    def test_parse_range_id(self):
        """Test parsing range segment ID."""
        assert parse_segment_id("s0-s5") == ("s0", "s5")
        assert parse_segment_id("s10-s20") == ("s10", "s20")

    def test_parse_non_standard_id(self):
        """Test parsing non-standard segment IDs treated as single."""
        # IDs that don't match 's[digit]' pattern
        assert parse_segment_id("segment1") == ("segment1", "segment1")
        assert parse_segment_id("custom-id") == ("custom-id", "custom-id")


class TestGetSegmentNumber:
    """Tests for get_segment_number function."""

    def test_get_number_from_standard_id(self):
        """Test extracting number from standard ID."""
        assert get_segment_number("s0") == 0
        assert get_segment_number("s5") == 5
        assert get_segment_number("s123") == 123

    def test_get_number_from_invalid_id(self):
        """Test invalid IDs return None."""
        assert get_segment_number("segment1") is None
        assert get_segment_number("abc") is None
        assert get_segment_number("") is None


class TestTranscriptManager:
    """Tests for TranscriptManager class."""

    def test_init_empty(self):
        """Test manager initializes empty."""
        manager = TranscriptManager()
        assert manager.is_empty
        assert manager.segment_count == 0
        assert manager.get_text() == ""

    def test_update_append_new_segment(self):
        """Test appending new segment."""
        manager = TranscriptManager()
        result = manager.update("s0", "hello")

        assert result is False  # False = appended (not replaced)
        assert manager.segment_count == 1
        assert manager.get_text() == "hello"

    def test_update_replace_existing_segment(self):
        """Test replacing existing segment."""
        manager = TranscriptManager()
        manager.update("s0", "hello")
        result = manager.update("s0", "hello world")

        assert result is True  # True = replaced
        assert manager.segment_count == 1
        assert manager.get_text() == "hello world"

    def test_update_multiple_segments(self):
        """Test multiple segments concatenate with spaces."""
        manager = TranscriptManager()
        manager.update("s0", "hello")
        manager.update("s1", "world")

        assert manager.segment_count == 2
        assert manager.get_text() == "hello world"

    def test_update_range_consolidates_segments(self):
        """Test range ID consolidates multiple segments."""
        manager = TranscriptManager()
        manager.update("s0", "hello")
        manager.update("s1", "there")
        manager.update("s2", "world")

        # Consolidate s0-s2 into one segment
        manager.update("s0-s2", "hello world")

        # Should have only one segment now
        assert manager.segment_count == 1
        assert manager.get_text() == "hello world"

    def test_get_segment(self):
        """Test getting specific segment."""
        manager = TranscriptManager()
        manager.update("s0", "hello")
        manager.update("s1", "world")

        assert manager.get_segment("s0") == "hello"
        assert manager.get_segment("s1") == "world"
        assert manager.get_segment("s999") is None

    def test_get_segments(self):
        """Test getting all segments as list."""
        manager = TranscriptManager()
        manager.update("s0", "hello")
        manager.update("s1", "world")

        segments = manager.get_segments()
        assert segments == [("s0", "hello"), ("s1", "world")]

    def test_get_last_segment(self):
        """Test getting last segment."""
        manager = TranscriptManager()
        assert manager.get_last_segment() is None

        manager.update("s0", "hello")
        manager.update("s1", "world")

        assert manager.get_last_segment() == ("s1", "world")

    def test_clear(self):
        """Test clearing all segments."""
        manager = TranscriptManager()
        manager.update("s0", "hello")
        manager.update("s1", "world")

        manager.clear()

        assert manager.is_empty
        assert manager.get_text() == ""

    def test_remove_segment(self):
        """Test removing specific segment."""
        manager = TranscriptManager()
        manager.update("s0", "hello")
        manager.update("s1", "world")

        result = manager.remove("s0")
        assert result is True
        assert manager.segment_count == 1
        assert manager.get_text() == "world"

        # Remove non-existent segment
        result = manager.remove("s999")
        assert result is False

    def test_max_segments_limit(self):
        """Test max segments limit removes oldest."""
        manager = TranscriptManager(max_segments=3)

        manager.update("s0", "one")
        manager.update("s1", "two")
        manager.update("s2", "three")
        manager.update("s3", "four")  # Should remove s0

        assert manager.segment_count == 3
        assert manager.get_segment("s0") is None
        assert manager.get_text() == "two three four"

    def test_on_change_callback(self):
        """Test on_change callback is called."""
        manager = TranscriptManager()
        callback_count = [0]

        def on_change():
            callback_count[0] += 1

        manager.on_change = on_change
        manager.update("s0", "hello")  # +1
        manager.update("s1", "world")  # +1
        manager.clear()  # +1

        assert callback_count[0] == 3

    def test_on_change_callback_error_doesnt_break(self):
        """Test callback errors don't break manager."""
        manager = TranscriptManager()

        def bad_callback():
            raise RuntimeError("Callback error")

        manager.on_change = bad_callback
        # Should not raise
        manager.update("s0", "hello")
        assert manager.get_text() == "hello"

    def test_to_html(self):
        """Test HTML rendering with spans."""
        manager = TranscriptManager()
        manager.update("s0", "hello")
        manager.update("s1", "world")

        html = manager.to_html()
        assert '<span class="segment" data-id="s0">hello</span>' in html
        assert '<span class="segment" data-id="s1">world</span>' in html

    def test_repr(self):
        """Test string representation."""
        manager = TranscriptManager()
        manager.update("s0", "hello")
        manager.update("s1", "world")

        assert repr(manager) == "TranscriptManager(2 segments)"

    def test_len(self):
        """Test len() returns segment count."""
        manager = TranscriptManager()
        assert len(manager) == 0

        manager.update("s0", "hello")
        assert len(manager) == 1

        manager.update("s1", "world")
        assert len(manager) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
