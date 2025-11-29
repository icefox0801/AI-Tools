"""
Text Refiner Client - Shared module for ASR services

Provides punctuation and ASR error correction via the text-refiner service.
Used by both Parakeet and Vosk ASR services.

Usage:
    from shared.text_refiner_client import TextRefinerClient
    
    # Initialize (call once at startup)
    client = TextRefinerClient()
    await client.check_availability()
    
    # Use in processing
    refined = await client.refine_text("hello world")
"""

import os
import logging
from typing import Optional
import httpx

logger = logging.getLogger(__name__)

# Configuration from environment
TEXT_REFINER_URL = os.getenv("TEXT_REFINER_URL", "http://text-refiner:8000")
ENABLE_TEXT_REFINER = os.getenv("ENABLE_TEXT_REFINER", "true").lower() == "true"
TEXT_REFINER_TIMEOUT = float(os.getenv("TEXT_REFINER_TIMEOUT", "2.0"))


class TextRefinerClient:
    """
    Async client for text-refiner service.
    
    Handles:
    - Connection pooling with persistent HTTP client
    - Availability checking
    - Graceful fallback when service unavailable
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        enabled: Optional[bool] = None,
        timeout: Optional[float] = None
    ):
        self.url = url or TEXT_REFINER_URL
        self.enabled = enabled if enabled is not None else ENABLE_TEXT_REFINER
        self.timeout = timeout or TEXT_REFINER_TIMEOUT
        self.available = False
        self._client: Optional[httpx.AsyncClient] = None
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client with connection pooling."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def check_availability(self) -> bool:
        """
        Check if text-refiner service is available.
        Call this at startup to verify connectivity.
        """
        if not self.enabled:
            logger.info("Text refiner disabled by configuration")
            self.available = False
            return False
        
        try:
            client = await self.get_client()
            response = await client.get(f"{self.url}/health")
            if response.status_code == 200:
                self.available = True
                logger.info(f"Text refiner connected: {self.url}")
                return True
        except Exception as e:
            logger.warning(f"Text refiner not available: {e}")
        
        self.available = False
        return False
    
    async def refine_text(
        self,
        text: str,
        punctuate: bool = True,
        correct: bool = True
    ) -> str:
        """
        Send text to refiner service for punctuation and correction.
        
        Args:
            text: Raw text from ASR
            punctuate: Enable punctuation restoration
            correct: Enable ASR error correction
            
        Returns:
            Refined text, or fallback capitalized text if service unavailable
        """
        if not text or not text.strip():
            return text
        
        if not self.enabled or not self.available:
            return self._capitalize(text)
        
        try:
            client = await self.get_client()
            response = await client.post(
                f"{self.url}/process",
                json={"text": text, "punctuate": punctuate, "correct": correct}
            )
            if response.status_code == 200:
                result = response.json()
                return result.get("text", text)
            else:
                logger.warning(f"Text refiner returned {response.status_code}")
        except httpx.TimeoutException:
            logger.debug("Text refiner timeout, using fallback")
        except Exception as e:
            logger.debug(f"Text refiner error: {e}")
        
        return self._capitalize(text)
    
    async def punctuate_only(self, text: str) -> str:
        """Punctuation only (faster, no correction)."""
        return await self.refine_text(text, punctuate=True, correct=False)
    
    async def correct_only(self, text: str) -> str:
        """Correction only (no punctuation)."""
        return await self.refine_text(text, punctuate=False, correct=True)
    
    @staticmethod
    def _capitalize(text: str) -> str:
        """Fallback: Capitalize first letter of text."""
        if not text:
            return text
        return text[0].upper() + text[1:] if len(text) > 1 else text.upper()


# Singleton instance for simple usage
_default_client: Optional[TextRefinerClient] = None


def get_client() -> TextRefinerClient:
    """
    Get the default text refiner client singleton (sync version).
    Creates client but doesn't check availability until first use.
    """
    global _default_client
    if _default_client is None:
        _default_client = TextRefinerClient()
    return _default_client


async def get_text_refiner() -> TextRefinerClient:
    """Get the default text refiner client singleton and check availability."""
    global _default_client
    if _default_client is None:
        _default_client = TextRefinerClient()
        await _default_client.check_availability()
    return _default_client


async def check_text_refiner() -> bool:
    """Check if text-refiner service is available (convenience function)."""
    client = get_client()
    return await client.check_availability()


async def refine_text(text: str, punctuate: bool = True, correct: bool = True) -> str:
    """
    Convenience function to refine text using default client.
    
    Usage:
        from shared.text_refiner_client import refine_text
        refined = await refine_text("hello world")
    """
    client = await get_text_refiner()
    return await client.refine_text(text, punctuate, correct)


def capitalize_text(text: str) -> str:
    """Capitalize first letter of text (sync fallback)."""
    return TextRefinerClient._capitalize(text)
