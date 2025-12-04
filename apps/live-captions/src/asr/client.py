"""ASR WebSocket client for streaming audio to transcription service."""

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


class ASRClient:
    """WebSocket client for ASR (Automatic Speech Recognition) service."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        chunk_ms: int = 200,
        on_connected: Callable[[bool], None] | None = None,
        on_transcript: Callable[[str, str], None] | None = None,
    ):
        """
        Initialize ASR client.

        Args:
            host: ASR service host
            port: ASR service port
            chunk_ms: Chunk duration in milliseconds for config
            on_connected: Callback when connection status changes (bool: connected)
            on_transcript: Callback for transcripts (segment_id, text)
        """
        self.host = host
        self.port = port
        self.chunk_ms = chunk_ms
        self.on_connected = on_connected
        self.on_transcript = on_transcript

        self.running = False
        self.audio_queue: asyncio.Queue | None = None

    @property
    def uri(self) -> str:
        """WebSocket URI for the ASR service."""
        return f"ws://{self.host}:{self.port}/stream"

    def queue_audio(self, audio_data: bytes):
        """
        Queue audio data for sending to ASR service.

        Args:
            audio_data: Raw 16-bit PCM audio bytes (mono, 16kHz)
        """
        if self.audio_queue:
            try:
                self.audio_queue.put_nowait(audio_data)
            except asyncio.QueueFull:
                logger.warning("Audio queue full, dropping audio chunk")

    async def run(self):
        """Run the ASR client connection loop."""
        import websockets

        self.running = True
        self.audio_queue = asyncio.Queue(maxsize=100)

        while self.running:
            try:
                logger.info(f"Connecting to ASR: {self.uri}")
                async with websockets.connect(self.uri) as ws:
                    if self.on_connected:
                        self.on_connected(True)

                    # Send config
                    config = {"chunk_ms": self.chunk_ms}
                    await ws.send(json.dumps(config))

                    # Run send and receive concurrently
                    send_task = asyncio.create_task(self._send_audio(ws))
                    recv_task = asyncio.create_task(self._receive_transcripts(ws))

                    _done, pending = await asyncio.wait(
                        [send_task, recv_task], return_when=asyncio.FIRST_COMPLETED
                    )

                    for task in pending:
                        task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await task

            except (ConnectionRefusedError, OSError) as e:
                if self.on_connected:
                    self.on_connected(False)
                logger.warning(f"Connection failed: {e}")
                await asyncio.sleep(2)
            except Exception as e:
                if self.on_connected:
                    self.on_connected(False)
                logger.error(f"ASR error: {e}")
                await asyncio.sleep(2)

    async def _send_audio(self, ws):
        """Send audio to ASR service."""
        while self.running:
            try:
                audio_data = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)
                await ws.send(audio_data)
            except TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Send error: {e}")
                break

    async def _receive_transcripts(self, ws):
        """Receive transcripts from ASR service."""
        while self.running:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                self._process_message(message)
            except TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Receive error: {e}")
                break

    def _process_message(self, message: str):
        """Process a message from the ASR service."""
        if not self.on_transcript:
            return

        try:
            data = json.loads(message)

            # ID-based protocol (Parakeet/Whisper)
            if "id" in data:
                segment_id = data["id"]
                text = data.get("text", "").strip()
                if text:
                    self.on_transcript(segment_id, text)

            # Legacy Vosk protocol
            elif "partial" in data:
                partial = data["partial"].strip()
                if partial:
                    self.on_transcript("_partial", partial)
            elif "text" in data:
                text = data["text"].strip()
                if text:
                    self.on_transcript("_final", text)

        except json.JSONDecodeError:
            # Plain text message
            if message.strip():
                self.on_transcript("_text", message.strip())

    def stop(self):
        """Stop the ASR client."""
        self.running = False
