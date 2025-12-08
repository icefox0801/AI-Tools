"""
Unit tests for text_refiner module.
"""

import pytest

from .utils import capitalize_text


class TestCapitalizeText:
    """Tests for capitalize_text function."""

    def test_capitalize_normal_text(self):
        """Test capitalizing normal lowercase text."""
        assert capitalize_text("hello world") == "Hello world"

    def test_capitalize_already_capitalized(self):
        """Test text that's already capitalized."""
        assert capitalize_text("Hello world") == "Hello world"

    def test_capitalize_all_uppercase(self):
        """Test all uppercase text (only first char stays uppercase)."""
        assert capitalize_text("HELLO") == "HELLO"

    def test_capitalize_single_char(self):
        """Test single character."""
        assert capitalize_text("a") == "A"
        assert capitalize_text("A") == "A"

    def test_capitalize_empty_string(self):
        """Test empty string returns empty."""
        assert capitalize_text("") == ""

    def test_capitalize_whitespace(self):
        """Test string starting with whitespace."""
        assert capitalize_text(" hello") == " hello"

    def test_capitalize_numbers(self):
        """Test string starting with numbers."""
        assert capitalize_text("123 abc") == "123 abc"


class TestGetClient:
    """Tests for get_client singleton pattern."""

    def test_get_client_returns_instance(self):
        """Test get_client returns a TextRefinerClient instance."""
        from . import get_client
        from .client import TextRefinerClient

        client = get_client()
        assert isinstance(client, TextRefinerClient)

    def test_get_client_singleton(self):
        """Test get_client returns same instance on repeated calls."""
        from . import get_client

        client1 = get_client()
        client2 = get_client()
        assert client1 is client2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
