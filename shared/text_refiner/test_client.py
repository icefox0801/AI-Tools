"""
Unit tests for shared.text_refiner.client module.

Tests the TextRefinerClient class with mocked HTTP responses.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestTextRefinerClient:
    """Tests for TextRefinerClient class."""

    @pytest.fixture
    def client(self):
        """Create a TextRefinerClient instance with mocked config."""
        with patch("shared.text_refiner.client.ENABLE_TEXT_REFINER", True):
            with patch("shared.text_refiner.client.TEXT_REFINER_URL", "http://localhost:8000"):
                with patch("shared.text_refiner.client.TEXT_REFINER_TIMEOUT", 5.0):
                    from .client import TextRefinerClient

                    return TextRefinerClient()

    @pytest.fixture
    def disabled_client(self):
        """Create a disabled TextRefinerClient."""
        with patch("shared.text_refiner.client.ENABLE_TEXT_REFINER", False):
            from .client import TextRefinerClient

            return TextRefinerClient()

    def test_initialization(self, client):
        """Client initializes with correct values."""
        assert client.enabled is True
        assert client.available is False
        assert client.url == "http://localhost:8000"

    def test_initialization_disabled(self, disabled_client):
        """Disabled client has enabled=False."""
        assert disabled_client.enabled is False

    @pytest.mark.asyncio
    async def test_check_availability_success(self, client):
        """Check availability returns True when service is healthy."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)

        with patch.object(client, "_get_http", return_value=mock_http):
            result = await client.check_availability()

        assert result is True
        assert client.available is True
        mock_http.get.assert_called_once_with("http://localhost:8000/health")

    @pytest.mark.asyncio
    async def test_check_availability_failure(self, client):
        """Check availability returns False when service is unavailable."""
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=Exception("Connection refused"))

        with patch.object(client, "_get_http", return_value=mock_http):
            result = await client.check_availability()

        assert result is False
        assert client.available is False

    @pytest.mark.asyncio
    async def test_check_availability_disabled(self, disabled_client):
        """Check availability returns False when disabled."""
        result = await disabled_client.check_availability()
        assert result is False

    @pytest.mark.asyncio
    async def test_refine_empty_text(self, client):
        """Refine returns empty text unchanged."""
        result = await client.refine("")
        assert result == ""

    @pytest.mark.asyncio
    async def test_refine_whitespace_only(self, client):
        """Refine returns whitespace-only text unchanged."""
        result = await client.refine("   ")
        assert result == "   "

    @pytest.mark.asyncio
    async def test_refine_success(self, client):
        """Refine returns processed text on success."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Hello world."}

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.get = AsyncMock(return_value=MagicMock(status_code=200))  # For availability check

        client.available = True

        with patch.object(client, "_get_http", return_value=mock_http):
            result = await client.refine("hello world")

        assert result == "Hello world."
        mock_http.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_refine_fallback_on_error(self, client):
        """Refine returns capitalized text when service fails."""
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=Exception("Error"))
        mock_http.get = AsyncMock(return_value=MagicMock(status_code=200))

        client.available = True

        with patch.object(client, "_get_http", return_value=mock_http):
            result = await client.refine("hello world")

        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_refine_fallback_when_unavailable(self, client):
        """Refine returns capitalized text when service unavailable."""
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=Exception("Connection refused"))

        with patch.object(client, "_get_http", return_value=mock_http):
            result = await client.refine("hello world")

        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_close(self, client):
        """Close properly closes HTTP client."""
        mock_http = AsyncMock()
        mock_http.aclose = AsyncMock()
        client._http = mock_http

        await client.close()

        mock_http.aclose.assert_called_once()
        assert client._http is None

    @pytest.mark.asyncio
    async def test_close_no_client(self, client):
        """Close handles case when no HTTP client exists."""
        client._http = None
        await client.close()  # Should not raise
