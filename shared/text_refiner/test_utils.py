"""
Unit tests for shared.text_refiner.utils module.

Tests text processing utility functions.
"""

from .utils import capitalize_text


class TestCapitalizeText:
    """Tests for capitalize_text function."""

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert capitalize_text("") == ""

    def test_single_char_lower(self):
        """Single lowercase char is capitalized."""
        assert capitalize_text("a") == "A"

    def test_single_char_upper(self):
        """Single uppercase char stays uppercase."""
        assert capitalize_text("A") == "A"

    def test_word(self):
        """Word is capitalized."""
        assert capitalize_text("hello") == "Hello"

    def test_already_capitalized(self):
        """Already capitalized word stays the same."""
        assert capitalize_text("Hello") == "Hello"

    def test_sentence(self):
        """Sentence first letter is capitalized."""
        assert capitalize_text("hello world") == "Hello world"

    def test_all_uppercase(self):
        """All uppercase string stays uppercase."""
        assert capitalize_text("HELLO") == "HELLO"

    def test_with_numbers_start(self):
        """String starting with number."""
        assert capitalize_text("123 hello") == "123 hello"

    def test_with_punctuation_start(self):
        """String starting with punctuation."""
        assert capitalize_text("...hello") == "...hello"

    def test_whitespace_start(self):
        """String starting with whitespace."""
        assert capitalize_text(" hello") == " hello"

    def test_unicode(self):
        """Unicode text is handled."""
        assert capitalize_text("über") == "Über"

    def test_none_returns_none(self):
        """None input returns None (falsy check returns early)."""
        # The function checks `if not text` which returns None for None input
        result = capitalize_text(None)
        assert result is None
