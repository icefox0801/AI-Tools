"""
Unit tests for shared.client.result module.

Tests the ASRResult data class and its factory methods.
"""

import pytest

from shared.client.result import ASRResult


class TestASRResult:
    """Tests for ASRResult data class."""

    def test_default_values(self):
        """Default values are set correctly."""
        result = ASRResult()
        assert result.text == ""
        assert result.is_final is False
        assert result.success is True
        assert result.error is None
        assert result.confidence == 1.0
        assert result.words == []
        assert result.chunk_ids == []
        assert result.metadata == {}

    def test_with_text(self):
        """Result with text is created correctly."""
        result = ASRResult(text="hello world", is_final=True)
        assert result.text == "hello world"
        assert result.is_final is True

    def test_from_partial(self):
        """Partial result factory method works correctly."""
        result = ASRResult.from_partial("hello")
        assert result.text == "hello"
        assert result.is_final is False
        assert result.success is True

    def test_from_partial_with_chunk_ids(self):
        """Partial result with chunk IDs."""
        result = ASRResult.from_partial("hello", chunk_ids=["c1", "c2"])
        assert result.chunk_ids == ["c1", "c2"]

    def test_from_final(self):
        """Final result factory method works correctly."""
        result = ASRResult.from_final("hello world")
        assert result.text == "hello world"
        assert result.is_final is True
        assert result.success is True

    def test_from_final_with_confidence(self):
        """Final result with custom confidence."""
        result = ASRResult.from_final("hello", confidence=0.95)
        assert result.confidence == 0.95

    def test_from_error(self):
        """Error result factory method works correctly."""
        result = ASRResult.from_error("Connection failed")
        assert result.text == ""
        assert result.is_final is True
        assert result.success is False
        assert result.error == "Connection failed"

    def test_bool_success_with_text(self):
        """Result is truthy when successful with text."""
        result = ASRResult(text="hello", success=True)
        assert bool(result) is True

    def test_bool_success_empty_text(self):
        """Result is falsy when successful but empty text."""
        result = ASRResult(text="", success=True)
        assert bool(result) is False

    def test_bool_error(self):
        """Result is falsy on error."""
        result = ASRResult.from_error("Error")
        assert bool(result) is False

    def test_str_final(self):
        """String representation for final result."""
        result = ASRResult.from_final("hello")
        assert "final" in str(result)
        assert "hello" in str(result)

    def test_str_partial(self):
        """String representation for partial result."""
        result = ASRResult.from_partial("hello")
        assert "partial" in str(result)
        assert "hello" in str(result)

    def test_str_error(self):
        """String representation for error result."""
        result = ASRResult.from_error("Connection failed")
        assert "error" in str(result)
        assert "Connection failed" in str(result)

    def test_str_long_text_truncated(self):
        """Long text is truncated in string representation."""
        long_text = "a" * 100
        result = ASRResult(text=long_text)
        assert "..." in str(result)
        assert len(str(result)) < len(long_text) + 50

    def test_words_list(self):
        """Words list with timestamps."""
        words = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]
        result = ASRResult(text="hello world", words=words)
        assert len(result.words) == 2
        assert result.words[0]["word"] == "hello"

    def test_metadata(self):
        """Metadata dictionary."""
        result = ASRResult(
            text="hello",
            metadata={"model": "parakeet", "language": "en"},
        )
        assert result.metadata["model"] == "parakeet"
        assert result.metadata["language"] == "en"
