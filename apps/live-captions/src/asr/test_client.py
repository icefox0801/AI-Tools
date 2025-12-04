"""
Unit tests for ASR client module.

Tests the ASRClient class including:
- Initialization and configuration
- URI generation
- Message processing (ID-based, Vosk legacy, plain text)
- Audio queue management
- Connection lifecycle callbacks
"""

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from .client import ASRClient


class TestASRClientInit:
    """Tests for ASRClient initialization."""

    def test_default_values(self):
        """Test default initialization values."""
        client = ASRClient()

        assert client.host == "localhost"
        assert client.port == 8000
        assert client.chunk_ms == 200
        assert client.on_connected is None
        assert client.on_transcript is None
        assert client.running is False
        assert client.audio_queue is None

    def test_custom_values(self):
        """Test initialization with custom values."""
        on_connected = Mock()
        on_transcript = Mock()

        client = ASRClient(
            host="transcription",
            port=9000,
            chunk_ms=300,
            on_connected=on_connected,
            on_transcript=on_transcript,
        )

        assert client.host == "transcription"
        assert client.port == 9000
        assert client.chunk_ms == 300
        assert client.on_connected is on_connected
        assert client.on_transcript is on_transcript


class TestASRClientURI:
    """Tests for URI property."""

    def test_default_uri(self):
        """Test default WebSocket URI."""
        client = ASRClient()
        assert client.uri == "ws://localhost:8000/stream"

    def test_custom_uri(self):
        """Test custom host/port URI."""
        client = ASRClient(host="transcription", port=9000)
        assert client.uri == "ws://transcription:9000/stream"

    def test_uri_with_ip(self):
        """Test URI with IP address."""
        client = ASRClient(host="192.168.1.100", port=8080)
        assert client.uri == "ws://192.168.1.100:8080/stream"


class TestASRClientProcessMessage:
    """Tests for message processing."""

    def test_id_based_protocol(self):
        """Test processing ID-based protocol messages (Parakeet/Whisper)."""
        transcripts = []
        client = ASRClient(on_transcript=lambda sid, text: transcripts.append((sid, text)))

        # Test with segment ID
        client._process_message('{"id": "s1", "text": "Hello world"}')
        assert transcripts == [("s1", "Hello world")]

        # Test with another segment
        client._process_message('{"id": "s2", "text": "How are you"}')
        assert transcripts == [("s1", "Hello world"), ("s2", "How are you")]

    def test_id_based_empty_text_ignored(self):
        """Test that empty text in ID-based messages is ignored."""
        transcripts = []
        client = ASRClient(on_transcript=lambda sid, text: transcripts.append((sid, text)))

        client._process_message('{"id": "s1", "text": ""}')
        assert transcripts == []

        client._process_message('{"id": "s2", "text": "   "}')
        assert transcripts == []

    def test_vosk_partial_protocol(self):
        """Test processing Vosk partial result messages."""
        transcripts = []
        client = ASRClient(on_transcript=lambda sid, text: transcripts.append((sid, text)))

        client._process_message('{"partial": "Hello"}')
        assert transcripts == [("_partial", "Hello")]

    def test_vosk_final_protocol(self):
        """Test processing Vosk final result messages."""
        transcripts = []
        client = ASRClient(on_transcript=lambda sid, text: transcripts.append((sid, text)))

        client._process_message('{"text": "Hello world"}')
        assert transcripts == [("_final", "Hello world")]

    def test_vosk_empty_ignored(self):
        """Test that empty Vosk messages are ignored."""
        transcripts = []
        client = ASRClient(on_transcript=lambda sid, text: transcripts.append((sid, text)))

        client._process_message('{"partial": ""}')
        client._process_message('{"text": "   "}')
        assert transcripts == []

    def test_plain_text_fallback(self):
        """Test processing plain text messages (non-JSON)."""
        transcripts = []
        client = ASRClient(on_transcript=lambda sid, text: transcripts.append((sid, text)))

        client._process_message("Hello world")
        assert transcripts == [("_text", "Hello world")]

    def test_plain_text_empty_ignored(self):
        """Test that empty plain text is ignored."""
        transcripts = []
        client = ASRClient(on_transcript=lambda sid, text: transcripts.append((sid, text)))

        client._process_message("")
        client._process_message("   ")
        assert transcripts == []

    def test_no_callback_no_error(self):
        """Test that processing works without callback."""
        client = ASRClient()  # No on_transcript callback

        # Should not raise any errors
        client._process_message('{"id": "s1", "text": "Hello"}')
        client._process_message('{"partial": "Hello"}')
        client._process_message("Hello")

    def test_text_is_stripped(self):
        """Test that text is properly stripped of whitespace."""
        transcripts = []
        client = ASRClient(on_transcript=lambda sid, text: transcripts.append((sid, text)))

        client._process_message('{"id": "s1", "text": "  Hello world  "}')
        assert transcripts == [("s1", "Hello world")]


class TestASRClientQueueAudio:
    """Tests for audio queue management."""

    def test_queue_audio_no_queue(self):
        """Test queuing audio when queue not initialized."""
        client = ASRClient()
        # Should not raise - just silently ignore
        client.queue_audio(b"\x00" * 100)

    @pytest.mark.asyncio
    async def test_queue_audio_success(self):
        """Test successfully queuing audio data."""
        client = ASRClient()
        client.audio_queue = asyncio.Queue(maxsize=10)

        audio_data = b"\x00\x01\x02\x03"
        client.queue_audio(audio_data)

        assert client.audio_queue.qsize() == 1
        queued = await client.audio_queue.get()
        assert queued == audio_data

    @pytest.mark.asyncio
    async def test_queue_audio_full_drops(self):
        """Test that full queue drops audio with warning."""
        client = ASRClient()
        client.audio_queue = asyncio.Queue(maxsize=2)

        # Fill the queue
        client.queue_audio(b"chunk1")
        client.queue_audio(b"chunk2")

        # This should be dropped (queue full)
        with patch("src.asr.client.logger") as mock_logger:
            client.queue_audio(b"chunk3")
            mock_logger.warning.assert_called_once()

        assert client.audio_queue.qsize() == 2


class TestASRClientStop:
    """Tests for stopping the client."""

    def test_stop_sets_running_false(self):
        """Test that stop() sets running to False."""
        client = ASRClient()
        client.running = True

        client.stop()

        assert client.running is False


class TestASRClientRun:
    """Tests for the run() method and connection lifecycle."""

    @pytest.mark.asyncio
    async def test_run_initializes_queue(self):
        """Test that run() initializes the audio queue."""
        client = ASRClient()

        # Mock websockets to fail immediately
        with patch("websockets.connect", side_effect=ConnectionRefusedError):
            # Run briefly then stop
            async def stop_soon():
                await asyncio.sleep(0.1)
                client.stop()

            await asyncio.gather(client.run(), stop_soon(), return_exceptions=True)

        assert client.audio_queue is not None
        assert client.audio_queue.maxsize == 100

    @pytest.mark.asyncio
    async def test_connection_callback_on_success(self):
        """Test on_connected callback is called on successful connection."""
        connected_states = []
        client = ASRClient(on_connected=lambda x: connected_states.append(x))

        # Create mock WebSocket
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.TimeoutError)

        mock_connect = AsyncMock()
        mock_connect.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_connect.__aexit__ = AsyncMock(return_value=False)

        with patch("websockets.connect", return_value=mock_connect):

            async def stop_soon():
                await asyncio.sleep(0.1)
                client.stop()

            await asyncio.gather(client.run(), stop_soon(), return_exceptions=True)

        assert True in connected_states

    @pytest.mark.asyncio
    async def test_connection_callback_on_failure(self):
        """Test on_connected(False) is called on connection failure."""
        connected_states = []
        client = ASRClient(on_connected=lambda x: connected_states.append(x))

        with patch("websockets.connect", side_effect=ConnectionRefusedError):

            async def stop_soon():
                await asyncio.sleep(0.1)
                client.stop()

            await asyncio.gather(client.run(), stop_soon(), return_exceptions=True)

        assert False in connected_states

    @pytest.mark.asyncio
    async def test_sends_config_on_connect(self):
        """Test that config is sent on successful connection."""
        client = ASRClient(chunk_ms=300)

        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.TimeoutError)

        mock_connect = AsyncMock()
        mock_connect.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_connect.__aexit__ = AsyncMock(return_value=False)

        with patch("websockets.connect", return_value=mock_connect):

            async def stop_soon():
                await asyncio.sleep(0.1)
                client.stop()

            await asyncio.gather(client.run(), stop_soon(), return_exceptions=True)

        # Check config was sent
        calls = mock_ws.send.call_args_list
        config_sent = any(
            json.loads(call[0][0]) == {"chunk_ms": 300}
            for call in calls
            if isinstance(call[0][0], str)
        )
        assert config_sent


class TestASRClientSendAudio:
    """Tests for _send_audio method."""

    @pytest.mark.asyncio
    async def test_send_audio_from_queue(self):
        """Test sending audio from queue to WebSocket."""
        client = ASRClient()
        client.running = True
        client.audio_queue = asyncio.Queue()

        mock_ws = AsyncMock()

        # Queue some audio
        await client.audio_queue.put(b"audio_chunk_1")
        await client.audio_queue.put(b"audio_chunk_2")

        # Stop after processing
        async def stop_after_queue_empty():
            while not client.audio_queue.empty():
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.1)
            client.stop()

        await asyncio.gather(
            client._send_audio(mock_ws), stop_after_queue_empty(), return_exceptions=True
        )

        # Verify audio was sent
        assert mock_ws.send.call_count >= 2


class TestASRClientReceiveTranscripts:
    """Tests for _receive_transcripts method."""

    @pytest.mark.asyncio
    async def test_receive_and_process(self):
        """Test receiving and processing transcripts."""
        transcripts = []
        client = ASRClient(on_transcript=lambda sid, text: transcripts.append((sid, text)))
        client.running = True

        messages = [
            '{"id": "s1", "text": "Hello"}',
            '{"id": "s2", "text": "World"}',
        ]
        message_iter = iter(messages)

        async def mock_recv():
            try:
                return next(message_iter)
            except StopIteration:
                raise TimeoutError from None

        mock_ws = AsyncMock()
        mock_ws.recv = mock_recv

        async def stop_soon():
            await asyncio.sleep(0.2)
            client.stop()

        await asyncio.gather(
            client._receive_transcripts(mock_ws), stop_soon(), return_exceptions=True
        )

        assert ("s1", "Hello") in transcripts
        assert ("s2", "World") in transcripts
