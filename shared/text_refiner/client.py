"""Async HTTP client for text-refiner service."""

import logging
from typing import Optional
import httpx

from .config import TEXT_REFINER_URL, ENABLE_TEXT_REFINER, TEXT_REFINER_TIMEOUT
from .utils import capitalize_text

logger = logging.getLogger(__name__)


class TextRefinerClient:
    """Async HTTP client for text-refiner service with connection pooling."""
    
    def __init__(self):
        self.url = TEXT_REFINER_URL
        self.enabled = ENABLE_TEXT_REFINER
        self.timeout = TEXT_REFINER_TIMEOUT
        self.available = False
        self._http: Optional[httpx.AsyncClient] = None
    
    async def _get_http(self) -> httpx.AsyncClient:
        """Lazy-init HTTP client with connection pooling."""
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.timeout)
        return self._http
    
    async def check_availability(self) -> bool:
        """Check if text-refiner service is reachable."""
        if not self.enabled:
            logger.info("Text refiner disabled by configuration")
            return False
        
        try:
            http = await self._get_http()
            response = await http.get(f"{self.url}/health")
            if response.status_code == 200:
                self.available = True
                logger.info(f"Text refiner connected: {self.url}")
                return True
        except Exception as e:
            logger.warning(f"Text refiner not available: {e}")
        
        self.available = False
        return False
    
    async def refine(self, text: str, punctuate: bool = True, correct: bool = True) -> str:
        """
        Send text to refiner service for punctuation and correction.
        Returns fallback capitalized text if service unavailable.
        """
        if not text or not text.strip():
            return text
        
        # Check availability on first call
        if self.enabled and not self.available:
            await self.check_availability()
        
        if not self.available:
            return capitalize_text(text)
        
        try:
            http = await self._get_http()
            response = await http.post(
                f"{self.url}/process",
                json={"text": text, "punctuate": punctuate, "correct": correct}
            )
            if response.status_code == 200:
                return response.json().get("text", text)
            logger.warning(f"Text refiner returned {response.status_code}")
        except httpx.TimeoutException:
            logger.debug("Text refiner timeout")
        except Exception as e:
            logger.debug(f"Text refiner error: {e}")
        
        return capitalize_text(text)
    
    async def close(self):
        """Close HTTP client."""
        if self._http:
            await self._http.aclose()
            self._http = None
