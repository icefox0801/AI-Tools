"""
E2E Tests for FastConformer ASR Service

Tests the full streaming transcription flow with NVIDIA FastConformer model.
"""

import asyncio
import json
from typing import Any

import pytest
import websockets
from conftest import assert_gpu_available, assert_model_loaded, get_gpu_memory_gb, get_gpu_status

# Mark all tests in this module as e2e with 30s timeout
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(30),
]

# Timeout for receiving WebSocket responses
RECV_TIMEOUT = 5.0


class TestFastConformerASR:
    """End-to-end tests for FastConformer ASR service."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, fastconformer_service: dict[str, Any]) -> None:
        """Test FastConformer health endpoint."""
        import httpx

        url = f"http://{fastconformer_service['host']}:{fastconformer_service['port']}{fastconformer_service['health_endpoint']}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ["healthy", "starting"]
            assert "model_loaded" in data
            assert data["model_name"] == "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"

    @pytest.mark.asyncio
    async def test_info_endpoint(self, fastconformer_service: dict[str, Any]) -> None:
        """Test FastConformer info endpoint returns configuration."""
        import httpx

        url = f"http://{fastconformer_service['host']}:{fastconformer_service['port']}/info"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            assert response.status_code == 200
            data = response.json()

            # Verify configuration fields
            assert data["model_name"] == "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
            assert "decoder_type" in data
            assert data["decoder_type"] in ["rnnt", "ctc"]
            assert "att_context_size" in data
            assert data["streaming_only"] is True

    @pytest.mark.asyncio
    async def test_gpu_available(self, fastconformer_service: dict[str, Any]) -> None:
        """Test that GPU is available for FastConformer service."""
        assert_gpu_available(fastconformer_service)
        status = get_gpu_status(fastconformer_service)
        assert "NVIDIA" in status.get("cuda_device", ""), f"Expected NVIDIA GPU: {status}"

    @pytest.mark.asyncio
    async def test_websocket_streaming(
        self, fastconformer_service: dict[str, Any], hello_audio: bytes
    ) -> None:
        """Test FastConformer WebSocket streaming transcription."""
        uri = f"ws://{fastconformer_service['host']}:{fastconformer_service['port']}{fastconformer_service['ws_endpoint']}"

        # Extended timeout for model loading on first connection
        async with websockets.connect(uri, open_timeout=60) as ws:
            # Send config
            config = {"chunk_ms": 500}  # FastConformer uses 500ms chunks
            await ws.send(json.dumps(config))

            # Wait for config acknowledgment
            config_msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
            config_data = json.loads(config_msg)
            assert "config" in config_data, f"Expected config acknowledgment, got: {config_data}"

            # Skip WAV header and send audio in chunks
            audio_data = hello_audio[44:]
            chunk_size = 16000  # 500ms at 16kHz mono 16-bit PCM

            responses = []

            # Receiver task to collect responses while sending
            async def receiver():
                try:
                    async for msg in ws:
                        if isinstance(msg, str):
                            data = json.loads(msg)
                            if "text" in data or "partial" in data or "final" in data:
                                responses.append(data)
                except websockets.exceptions.ConnectionClosed:
                    pass

            recv_task = asyncio.create_task(receiver())

            # Send audio chunks at ~real-time rate
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                await ws.send(chunk)
                await asyncio.sleep(0.3)  # ~300ms between 500ms chunks

            # Send empty chunk to signal end
            await ws.send(b"")

            # Wait for final processing
            await asyncio.sleep(2.0)

            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

        # Verify we got some transcription responses
        assert len(responses) > 0, "No responses received from FastConformer"

        # Get all text from responses
        all_text = " ".join(r.get("text", "") for r in responses).lower()
        assert len(all_text) > 0, "Expected non-empty transcription"
        assert "hello" in all_text, f"Expected 'hello' in transcription, got: {all_text}"

    @pytest.mark.asyncio
    async def test_websocket_numbers_transcription(
        self, fastconformer_service: dict[str, Any], numbers_audio: bytes
    ) -> None:
        """Test FastConformer transcribes number words correctly."""
        uri = f"ws://{fastconformer_service['host']}:{fastconformer_service['port']}{fastconformer_service['ws_endpoint']}"

        async with websockets.connect(uri, open_timeout=60) as ws:
            # Send config
            config = {"chunk_ms": 500}
            await ws.send(json.dumps(config))

            # Wait for config acknowledgment
            await asyncio.wait_for(ws.recv(), timeout=10.0)

            # Skip WAV header and send audio in chunks
            audio_data = numbers_audio[44:]
            chunk_size = 16000  # 500ms chunks

            responses = []

            # Receiver task
            async def receiver():
                try:
                    async for msg in ws:
                        if isinstance(msg, str):
                            data = json.loads(msg)
                            if "text" in data or "partial" in data:
                                responses.append(data)
                except websockets.exceptions.ConnectionClosed:
                    pass

            recv_task = asyncio.create_task(receiver())

            # Send audio chunks
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                await ws.send(chunk)
                await asyncio.sleep(0.3)

            # Signal end
            await ws.send(b"")
            await asyncio.sleep(2.0)

            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

        # Verify transcription contains number words
        all_text = " ".join(r.get("text", "") for r in responses).lower()
        assert len(responses) > 0, "No responses received"
        assert any(
            word in all_text for word in ["one", "two", "three", "four", "five"]
        ), f"Expected number words in transcription, got: {all_text}"

    @pytest.mark.asyncio
    async def test_model_loaded_after_stream(
        self, fastconformer_service: dict[str, Any], hello_audio: bytes
    ) -> None:
        """Test that model is loaded after streaming transcription."""
        # First, do a streaming transcription to ensure model is loaded
        uri = f"ws://{fastconformer_service['host']}:{fastconformer_service['port']}{fastconformer_service['ws_endpoint']}"

        async with websockets.connect(uri, open_timeout=60) as ws:
            config = {"chunk_ms": 500}
            await ws.send(json.dumps(config))
            await asyncio.wait_for(ws.recv(), timeout=10.0)

            # Send a small chunk
            audio_data = hello_audio[44 : 44 + 16000]
            await ws.send(audio_data)
            await asyncio.sleep(1.0)

        # Wait a moment for model to settle
        await asyncio.sleep(1.0)

        # Now verify model is loaded
        assert_model_loaded(fastconformer_service, "streaming")

        # Verify GPU memory is being used (FastConformer is only 0.23GB)
        mem_gb = get_gpu_memory_gb(fastconformer_service)
        assert mem_gb > 0.2, f"Expected >0.2GB GPU memory for FastConformer model, got: {mem_gb}GB"

    @pytest.mark.asyncio
    async def test_websocket_silence_handling(self, fastconformer_service: dict[str, Any]) -> None:
        """Test that FastConformer handles silence correctly (no transcription)."""
        import struct

        import numpy as np

        uri = f"ws://{fastconformer_service['host']}:{fastconformer_service['port']}{fastconformer_service['ws_endpoint']}"

        async with websockets.connect(uri, open_timeout=60) as ws:
            # Send config
            config = {"chunk_ms": 500}
            await ws.send(json.dumps(config))
            await asyncio.wait_for(ws.recv(), timeout=10.0)

            # Generate silent audio (all zeros)
            silent_samples = np.zeros(16000, dtype=np.int16)  # 500ms silence
            silent_audio = struct.pack(f"<{len(silent_samples)}h", *silent_samples)

            responses = []

            # Receiver task
            async def receiver():
                try:
                    async for msg in ws:
                        if isinstance(msg, str):
                            data = json.loads(msg)
                            if "text" in data:
                                responses.append(data)
                except websockets.exceptions.ConnectionClosed:
                    pass

            recv_task = asyncio.create_task(receiver())

            # Send silent chunks
            for _ in range(3):
                await ws.send(silent_audio)
                await asyncio.sleep(0.3)

            await asyncio.sleep(1.0)

            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

        # Should get very few or no responses for silence
        # (service may still send empty strings or skip silent chunks)
        all_text = " ".join(r.get("text", "") for r in responses).strip()
        assert len(all_text) < 10, f"Expected minimal transcription for silence, got: {all_text}"

    @pytest.mark.asyncio
    async def test_unload_endpoint(self, fastconformer_service: dict[str, Any]) -> None:
        """Test FastConformer model unload endpoint."""
        import httpx

        url = f"http://{fastconformer_service['host']}:{fastconformer_service['port']}/unload"
        async with httpx.AsyncClient() as client:
            response = await client.post(url)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"

        # Wait a moment for GPU to clear
        await asyncio.sleep(1.0)

        # Check health again - model should be unloaded
        health_url = (
            f"http://{fastconformer_service['host']}:{fastconformer_service['port']}/health"
        )
        async with httpx.AsyncClient() as client:
            response = await client.get(health_url)
            assert response.status_code == 200
            data = response.json()
            # Model should be unloaded or starting
            assert data["model_loaded"] in [False, None]

    @pytest.mark.asyncio
    async def test_websocket_concurrent_connections(
        self, fastconformer_service: dict[str, Any], hello_audio: bytes
    ) -> None:
        """Test FastConformer handles concurrent WebSocket connections."""
        uri = f"ws://{fastconformer_service['host']}:{fastconformer_service['port']}{fastconformer_service['ws_endpoint']}"

        async def stream_audio(ws_uri: str, audio: bytes):
            async with websockets.connect(ws_uri, open_timeout=60) as ws:
                config = {"chunk_ms": 500}
                await ws.send(json.dumps(config))
                await asyncio.wait_for(ws.recv(), timeout=10.0)

                # Send one chunk
                audio_data = audio[44 : 44 + 16000]
                await ws.send(audio_data)
                await asyncio.sleep(1.0)

                # Collect responses
                responses = []
                try:
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        if isinstance(msg, str):
                            data = json.loads(msg)
                            if "text" in data:
                                responses.append(data)
                except TimeoutError:
                    pass

                return responses

        # Run 2 concurrent streams
        results = await asyncio.gather(
            stream_audio(uri, hello_audio),
            stream_audio(uri, hello_audio),
            return_exceptions=True,
        )

        # At least one should succeed (service may limit concurrent connections)
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) > 0, f"No successful concurrent connections: {results}"

    @pytest.mark.asyncio
    async def test_websocket_keepalive(self, fastconformer_service: dict[str, Any]) -> None:
        """Test FastConformer WebSocket keepalive ping/pong mechanism."""
        uri = f"ws://{fastconformer_service['host']}:{fastconformer_service['port']}{fastconformer_service['ws_endpoint']}"

        async with websockets.connect(uri, open_timeout=60, ping_interval=None) as ws:
            # Send config
            config = {"chunk_ms": 500}
            await ws.send(json.dumps(config))
            await asyncio.wait_for(ws.recv(), timeout=10.0)

            # Wait for keepalive ping (service sends every 15s)
            # Just verify connection stays alive for a few seconds without data
            await asyncio.sleep(5.0)

            # Try to send a small chunk to verify still functional (no closed attr in newer websockets)
            try:
                await ws.send(b"\x00\x00" * 1000)
                await asyncio.sleep(0.5)
            except Exception as e:
                pytest.fail(f"WebSocket closed unexpectedly: {e}")
