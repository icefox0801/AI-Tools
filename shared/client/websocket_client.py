"""
WebSocket ASR Client

Handles WebSocket connections to ASR services with the unified streaming protocol.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from .result import ASRResult


@dataclass
class ConnectionConfig:
    """WebSocket connection configuration."""

    host: str = "localhost"
    port: int = 8001
    endpoint: str = "/stream"
    chunk_ms: int = 200
    timeout: float = 30.0

    @property
    def uri(self) -> str:
        """Get WebSocket URI."""
        return f"ws://{self.host}:{self.port}{self.endpoint}"


class ASRClient:
    """
    WebSocket client for ASR services.

    Implements the unified streaming protocol:
    1. Connect to WebSocket endpoint
    2. Send config JSON: {"chunk_ms": X}
    3. Stream raw audio bytes
    4. Receive JSON responses: {"partial": "..."} or {"text": "...", "final": true}

    Usage:
        client = ASRClient(host="localhost", port=8001)

        async with client.connect() as ws:
            await client.send_config(ws, chunk_ms=200)
            await client.send_audio(ws, audio_bytes)
            result = await client.receive(ws)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8001,
        endpoint: str = "/stream",
        chunk_ms: int = 200,
        timeout: float = 30.0,
    ):
        """
        Initialize ASR client.

        Args:
            host: ASR service hostname
            port: ASR service port
            endpoint: WebSocket endpoint path
            chunk_ms: Audio chunk duration in milliseconds
            timeout: Connection timeout in seconds
        """
        self.config = ConnectionConfig(
            host=host, port=port, endpoint=endpoint, chunk_ms=chunk_ms, timeout=timeout
        )
        self._ws = None

    @classmethod
    def from_backend_config(cls, backend_config: dict[str, Any]) -> "ASRClient":
        """
        Create client from backend configuration dictionary.

        Args:
            backend_config: Config dict with host, port, chunk_ms keys

        Returns:
            Configured ASRClient instance
        """
        return cls(
            host=backend_config.get("host", "localhost"),
            port=backend_config.get("port", 8001),
            chunk_ms=backend_config.get("chunk_ms", 200),
        )

    @property
    def uri(self) -> str:
        """Get WebSocket URI."""
        return self.config.uri

    async def connect(self):
        """
        Create WebSocket connection context manager.

        Usage:
            async with client.connect() as ws:
                await client.send_audio(ws, data)
        """
        import websockets

        return websockets.connect(self.config.uri, close_timeout=self.config.timeout)

    async def send_config(self, ws, chunk_ms: int | None = None) -> None:
        """
        Send configuration to ASR service.

        Args:
            ws: WebSocket connection
            chunk_ms: Override chunk duration (uses default if None)
        """
        config = {"chunk_ms": chunk_ms or self.config.chunk_ms}
        await ws.send(json.dumps(config))

    async def send_audio(self, ws, audio_bytes: bytes) -> None:
        """
        Send audio data to ASR service.

        Args:
            ws: WebSocket connection
            audio_bytes: Raw PCM audio (16kHz, mono, int16)
        """
        await ws.send(audio_bytes)

    async def receive(self, ws, timeout: float | None = None) -> ASRResult | None:
        """
        Receive transcription result from ASR service.

        Args:
            ws: WebSocket connection
            timeout: Receive timeout (uses default if None)

        Returns:
            ASRResult or None on timeout
        """
        try:
            message = await asyncio.wait_for(ws.recv(), timeout=timeout or self.config.timeout)
            return self._parse_response(message)
        except TimeoutError:
            return None
        except Exception as e:
            return ASRResult.from_error(str(e))

    async def stream(
        self, ws, audio_chunks: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[ASRResult, None]:
        """
        Stream audio and yield transcription results.

        Args:
            ws: WebSocket connection
            audio_chunks: Async generator of audio bytes

        Yields:
            ASRResult for each transcription response
        """
        # Start receive task
        receive_task = asyncio.create_task(self._receive_loop(ws))

        try:
            async for chunk in audio_chunks:
                await self.send_audio(ws, chunk)

                # Check for results without blocking
                try:
                    result = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    parsed = self._parse_response(result)
                    if parsed:
                        yield parsed
                except TimeoutError:
                    continue
        finally:
            receive_task.cancel()

    async def _receive_loop(self, ws) -> AsyncGenerator[ASRResult, None]:
        """Internal receive loop."""
        while True:
            try:
                message = await ws.recv()
                result = self._parse_response(message)
                if result:
                    yield result
            except Exception:
                break

    def _parse_response(self, message: str) -> ASRResult | None:
        """
        Parse ASR service response.

        Unified protocol:
        - {"partial": "..."}: Interim result
        - {"text": "...", "final": true}: Final result
        """
        try:
            data = json.loads(message)

            if "partial" in data:
                return ASRResult.from_partial(data["partial"])
            elif "text" in data:
                return ASRResult.from_final(
                    text=data["text"], confidence=data.get("confidence", 1.0)
                )
            else:
                return None

        except json.JSONDecodeError:
            # Plain text fallback
            if message.strip():
                return ASRResult.from_final(message.strip())
            return None
