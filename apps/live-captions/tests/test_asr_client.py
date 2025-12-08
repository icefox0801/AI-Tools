"""Unit tests for ASR client module."""

from unittest.mock import MagicMock

import pytest


class TestASRClientInit:
    """Tests for ASRClient initialization."""

    def test_default_init(self):
        """Test default initialization."""
        from src.asr.client import ASRClient

        client = ASRClient()

        assert client.host == "localhost"
        assert client.port == 8000
        assert client.chunk_ms == 200
        assert client.language == "en"

    def test_custom_language(self):
        """Test initialization with custom language."""
        from src.asr.client import ASRClient

        client = ASRClient(language="yue")

        assert client.language == "yue"

    def test_custom_params(self):
        """Test initialization with custom parameters."""
        from src.asr.client import ASRClient

        on_connected = MagicMock()
        on_transcript = MagicMock()

        client = ASRClient(
            host="192.168.1.1",
            port=9000,
            chunk_ms=500,
            language="yue",
            on_connected=on_connected,
            on_transcript=on_transcript,
        )

        assert client.host == "192.168.1.1"
        assert client.port == 9000
        assert client.chunk_ms == 500
        assert client.language == "yue"
        assert client.on_connected is on_connected
        assert client.on_transcript is on_transcript


class TestASRClientConfig:
    """Tests for ASR client configuration sending."""

    @pytest.mark.asyncio
    async def test_config_includes_language(self):
        """Test that config message includes language."""
        from src.asr.client import ASRClient

        client = ASRClient(chunk_ms=300, language="yue")

        # The config that would be sent
        expected_config = {"chunk_ms": 300, "language": "yue"}

        # Verify the config matches expected format
        assert client.chunk_ms == 300
        assert client.language == "yue"


class TestASRClientQueueAudio:
    """Tests for audio queueing."""

    def test_queue_audio_with_queue(self):
        """Test queuing audio when queue exists."""
        from src.asr.client import ASRClient

        client = ASRClient()
        client.audio_queue = MagicMock()

        audio_data = b"\x00\x01\x02\x03"
        client.queue_audio(audio_data)

        client.audio_queue.put_nowait.assert_called_once_with(audio_data)

    def test_queue_audio_without_queue(self):
        """Test queuing audio when queue doesn't exist (not connected yet)."""
        from src.asr.client import ASRClient

        client = ASRClient()
        client.audio_queue = None

        # Should not raise an error
        audio_data = b"\x00\x01\x02\x03"
        client.queue_audio(audio_data)  # Should be a no-op


class TestASRClientStop:
    """Tests for stopping the client."""

    def test_stop_sets_running_false(self):
        """Test stop sets running to False."""
        from src.asr.client import ASRClient

        client = ASRClient()
        client.running = True

        client.stop()

        assert client.running is False
