"""
E2E Tests for ASR Services

Tests the full transcription flow: Audio -> ASR Backend -> Response
"""

import asyncio
import json
from typing import Any

import pytest
import websockets

# Mark all tests in this module as e2e with 30s timeout
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(30),
]

# Timeout for receiving WebSocket responses
RECV_TIMEOUT = 5.0


class TestVoskASR:
    """End-to-end tests for Vosk ASR service."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, vosk_service: dict[str, Any]) -> None:
        """Test Vosk health endpoint."""
        import httpx

        url = (
            f"http://{vosk_service['host']}:{vosk_service['port']}{vosk_service['health_endpoint']}"
        )
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_websocket_streaming(
        self, vosk_service: dict[str, Any], numbers_audio: bytes
    ) -> None:
        """Test Vosk WebSocket streaming transcription."""
        uri = f"ws://{vosk_service['host']}:{vosk_service['port']}{vosk_service['ws_endpoint']}"

        responses = []

        async with websockets.connect(uri) as ws:
            # Send config
            config = {"chunk_ms": 200}
            await ws.send(json.dumps(config))

            # Skip WAV header and send audio in chunks
            audio_data = numbers_audio[44:]
            chunk_size = 6400  # 200ms at 16kHz mono 16-bit

            # Receiver task to collect responses while sending
            async def receiver():
                try:
                    async for msg in ws:
                        if isinstance(msg, str):
                            data = json.loads(msg)
                            responses.append(data)
                except websockets.exceptions.ConnectionClosed:
                    pass

            recv_task = asyncio.create_task(receiver())

            # Send audio chunks at ~real-time rate
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                await ws.send(chunk)
                await asyncio.sleep(0.1)  # ~100ms between 200ms chunks

            # Wait for final processing
            await asyncio.sleep(2.0)

            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

        # Verify we got some transcription responses
        assert len(responses) > 0, "No responses received from Vosk"

        # Get all text from responses
        all_text = " ".join(r.get("text", "") for r in responses).lower()
        # Numbers audio should contain number words
        assert any(
            word in all_text for word in ["one", "two", "three", "four", "five"]
        ), f"Expected number words in transcription, got: {all_text}"


class TestParakeetASR:
    """End-to-end tests for Parakeet ASR service."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, parakeet_service: dict[str, Any]) -> None:
        """Test Parakeet health endpoint."""
        import httpx

        url = f"http://{parakeet_service['host']}:{parakeet_service['port']}{parakeet_service['health_endpoint']}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_websocket_streaming(
        self, parakeet_service: dict[str, Any], hello_audio: bytes
    ) -> None:
        """Test Parakeet WebSocket streaming transcription."""
        uri = f"ws://{parakeet_service['host']}:{parakeet_service['port']}{parakeet_service['ws_endpoint']}"

        async with websockets.connect(uri) as ws:
            config = {"chunk_ms": 200, "sample_rate": 16000}
            await ws.send(json.dumps(config))

            audio_data = hello_audio[44:]
            chunk_size = 3200

            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                await ws.send(chunk)
                await asyncio.sleep(0.05)

            await ws.send(b"")

            responses = []
            try:
                async for msg in ws:
                    if isinstance(msg, str):
                        data = json.loads(msg)
                        responses.append(data)
                        if data.get("final"):
                            break
            except websockets.exceptions.ConnectionClosed:
                pass

            assert len(responses) > 0
            final_responses = [r for r in responses if r.get("final")]
            assert len(final_responses) > 0

            full_text = " ".join(r.get("text", "") for r in final_responses).lower()
            assert "hello" in full_text, f"Expected 'hello' in transcription, got: {full_text}"

    @pytest.mark.asyncio
    async def test_transcribe_endpoint(
        self, parakeet_service: dict[str, Any], hello_audio: bytes
    ) -> None:
        """Test Parakeet /transcribe file upload endpoint."""
        import httpx

        url = f"http://{parakeet_service['host']}:{parakeet_service['port']}/transcribe"

        async with httpx.AsyncClient(timeout=60.0) as client:
            files = {"file": ("test.wav", hello_audio, "audio/wav")}
            response = await client.post(url, files=files)

            assert response.status_code == 200
            data = response.json()

            assert "text" in data, f"Expected 'text' in response, got: {data}"
            assert "words" in data, "Expected word timestamps in response"
            assert "model" in data, "Expected model name in response"

            text = data["text"].lower()
            assert "hello" in text, f"Expected 'hello' in transcription, got: {text}"

    @pytest.mark.asyncio
    async def test_transcribe_returns_full_text(
        self, parakeet_service: dict[str, Any], numbers_audio: bytes
    ) -> None:
        """Test that Parakeet /transcribe returns full transcription without truncation."""
        import httpx

        url = f"http://{parakeet_service['host']}:{parakeet_service['port']}/transcribe"

        async with httpx.AsyncClient(timeout=60.0) as client:
            files = {"file": ("numbers.wav", numbers_audio, "audio/wav")}
            response = await client.post(url, files=files)

            assert response.status_code == 200
            data = response.json()

            # Check that words match the text
            words = data.get("words", [])
            text = data.get("text", "")

            if words:
                # Word count should approximately match
                word_count = len(words)
                text_word_count = len(text.split())
                # Allow some variance due to punctuation
                assert (
                    abs(word_count - text_word_count) < 5
                ), f"Word count mismatch: {word_count} words vs {text_word_count} in text"

    @pytest.mark.asyncio
    async def test_info_endpoint(self, parakeet_service: dict[str, Any]) -> None:
        """Test Parakeet info endpoint."""
        import httpx

        url = f"http://{parakeet_service['host']}:{parakeet_service['port']}/info"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            assert response.status_code == 200
            data = response.json()
            assert "service" in data
            assert "models" in data


class TestWhisperASR:
    """End-to-end tests for Whisper ASR service."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, whisper_service: dict[str, Any]) -> None:
        """Test Whisper health endpoint."""
        import httpx

        url = f"http://{whisper_service['host']}:{whisper_service['port']}{whisper_service['health_endpoint']}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_websocket_streaming(
        self, whisper_service: dict[str, Any], hello_audio: bytes
    ) -> None:
        """Test Whisper WebSocket streaming transcription."""
        uri = f"ws://{whisper_service['host']}:{whisper_service['port']}{whisper_service['ws_endpoint']}"

        async with websockets.connect(uri) as ws:
            config = {"chunk_ms": 200, "sample_rate": 16000}
            await ws.send(json.dumps(config))

            audio_data = hello_audio[44:]
            chunk_size = 3200

            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                await ws.send(chunk)
                await asyncio.sleep(0.05)

            await ws.send(b"")

            responses = []
            try:
                async for msg in ws:
                    if isinstance(msg, str):
                        data = json.loads(msg)
                        responses.append(data)
                        if data.get("final"):
                            break
            except websockets.exceptions.ConnectionClosed:
                pass

            assert len(responses) > 0
            final_responses = [r for r in responses if r.get("final")]
            assert len(final_responses) > 0

            # Get text from all non-final responses (final has empty text)
            text_responses = [r for r in responses if not r.get("final") and r.get("text")]
            full_text = " ".join(r.get("text", "") for r in text_responses).lower()
            assert "hello" in full_text, f"Expected 'hello' in transcription, got: {full_text}"

    @pytest.mark.asyncio
    async def test_transcribe_endpoint(
        self, whisper_service: dict[str, Any], hello_audio: bytes
    ) -> None:
        """Test Whisper /transcribe file upload endpoint."""
        import httpx

        url = f"http://{whisper_service['host']}:{whisper_service['port']}/transcribe"

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Upload audio file
            files = {"file": ("test.wav", hello_audio, "audio/wav")}
            response = await client.post(url, files=files)

            assert response.status_code == 200
            data = response.json()

            # Should have text field
            assert "text" in data, f"Expected 'text' in response, got: {data}"
            assert "error" not in data or data["error"] is None

            # Should have duration
            assert "duration" in data

            # Text should contain "hello"
            text = data["text"].lower()
            assert "hello" in text, f"Expected 'hello' in transcription, got: {text}"


class TestTextRefiner:
    """End-to-end tests for Text Refiner service."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, text_refiner_service: dict[str, Any]) -> None:
        """Test Text Refiner health endpoint."""
        import httpx

        url = f"http://{text_refiner_service['host']}:{text_refiner_service['port']}{text_refiner_service['health_endpoint']}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_info_endpoint(self, text_refiner_service: dict[str, Any]) -> None:
        """Test Text Refiner info endpoint."""
        import httpx

        url = f"http://{text_refiner_service['host']}:{text_refiner_service['port']}/info"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            assert response.status_code == 200
            data = response.json()
            assert data["service"] == "text-refiner"
            # Check for model info (different structure in old vs new versions)
            assert "models" in data or "punctuation_model" in data

    @pytest.mark.asyncio
    async def test_process_punctuation(self, text_refiner_service: dict[str, Any]) -> None:
        """Test text processing with punctuation."""
        import httpx

        url = f"http://{text_refiner_service['host']}:{text_refiner_service['port']}/process"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                json={"text": "hello world how are you", "punctuate": True, "correct": False},
            )
            assert response.status_code == 200
            data = response.json()
            assert "text" in data
            assert "original" in data
            assert data["original"] == "hello world how are you"
            assert data["punctuated"] is True
            # Check that punctuation was added (capital letter or punctuation)
            assert data["text"][0].isupper() or any(c in data["text"] for c in ".!?,")

    @pytest.mark.asyncio
    async def test_punctuate_endpoint(self, text_refiner_service: dict[str, Any]) -> None:
        """Test punctuation-only endpoint."""
        import httpx

        url = f"http://{text_refiner_service['host']}:{text_refiner_service['port']}/punctuate"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json={"text": "this is a test"})
            assert response.status_code == 200
            data = response.json()
            assert "text" in data
            assert "latency_ms" in data

    @pytest.mark.asyncio
    async def test_batch_processing(self, text_refiner_service: dict[str, Any]) -> None:
        """Test batch processing endpoint."""
        import httpx

        url = f"http://{text_refiner_service['host']}:{text_refiner_service['port']}/batch"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                json={
                    "texts": ["hello world", "how are you", "this is a test"],
                    "punctuate": True,
                    "correct": False,
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert "texts" in data
            assert len(data["texts"]) == 3
            assert "latency_ms" in data

    @pytest.mark.asyncio
    async def test_empty_text(self, text_refiner_service: dict[str, Any]) -> None:
        """Test handling of empty text."""
        import httpx

        url = f"http://{text_refiner_service['host']}:{text_refiner_service['port']}/process"
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, json={"text": "", "punctuate": True, "correct": False}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["text"] == ""
