"""
Transcription Client Module

WebSocket client for communicating with ASR services.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import websockets

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Configuration for transcription client."""
    host: str = os.getenv("PARAKEET_HOST", "parakeet-asr")
    port: int = 8000
    endpoint: str = "/stream"
    connect_timeout: float = 10.0
    recv_timeout: float = 10.0
    close_timeout: float = 5.0
    
    @property
    def uri(self) -> str:
        return f"ws://{self.host}:{self.port}{self.endpoint}"


@dataclass
class TranscriptionResult:
    """Result from a transcription request."""
    text: str
    duration: float
    success: bool
    error: Optional[str] = None


class TranscriptionClient:
    """
    WebSocket client for ASR service.
    
    Handles connection management and transcription requests.
    """
    
    def __init__(self, config: ClientConfig = None):
        self.config = config or ClientConfig()
    
    async def transcribe(self, audio_data: bytes, session_id: str) -> TranscriptionResult:
        """
        Send audio to ASR service for transcription.
        
        Args:
            audio_data: Raw PCM audio (16-bit, 16kHz, mono)
            session_id: Session identifier
        
        Returns:
            TranscriptionResult with text and metadata
        """
        try:
            async with websockets.connect(
                self.config.uri,
                close_timeout=self.config.close_timeout
            ) as ws:
                # Set session
                await ws.send(json.dumps({"session_id": session_id}))
                
                # Send audio
                await ws.send(audio_data)
                
                # Request transcription
                await ws.send(b"")
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(
                        ws.recv(),
                        timeout=self.config.recv_timeout
                    )
                    result = json.loads(response)
                    
                    if result.get("type") == "transcription":
                        text = result.get("text", "").strip()
                        duration = result.get("duration", 0)
                        logger.info(f"Transcribed ({duration:.1f}s): {text[:60]}...")
                        return TranscriptionResult(
                            text=text,
                            duration=duration,
                            success=True
                        )
                    else:
                        return TranscriptionResult(
                            text="",
                            duration=0,
                            success=False,
                            error=f"Unexpected response type: {result.get('type')}"
                        )
                        
                except asyncio.TimeoutError:
                    logger.warning("Transcription timeout")
                    return TranscriptionResult(
                        text="",
                        duration=0,
                        success=False,
                        error="Timeout waiting for transcription"
                    )
                    
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return TranscriptionResult(
                text="",
                duration=0,
                success=False,
                error=str(e)
            )
    
    async def clear_session(self, session_id: str) -> bool:
        """
        Clear server-side audio buffer for a session.
        
        Args:
            session_id: Session to clear
        
        Returns:
            True if successful
        """
        try:
            async with websockets.connect(
                self.config.uri,
                close_timeout=self.config.close_timeout
            ) as ws:
                await ws.send(json.dumps({"session_id": session_id}))
                await ws.send(json.dumps({"action": "clear"}))
                await asyncio.wait_for(ws.recv(), timeout=2.0)
                logger.info(f"Cleared session: {session_id[:8]}...")
                return True
        except Exception as e:
            logger.error(f"Clear session error: {e}")
            return False
    
    def transcribe_sync(self, audio_data: bytes, session_id: str) -> TranscriptionResult:
        """Synchronous wrapper for transcribe."""
        return asyncio.run(self.transcribe(audio_data, session_id))
    
    def clear_session_sync(self, session_id: str) -> bool:
        """Synchronous wrapper for clear_session."""
        return asyncio.run(self.clear_session(session_id))
