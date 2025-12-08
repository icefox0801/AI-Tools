"""
Text Refiner - Shared module for ASR services

Provides punctuation and ASR error correction via the text-refiner service.

Usage:
    from shared.text_refiner import get_client, refine_text, capitalize_text

    # Get singleton client (for health endpoint info)
    client = get_client()
    print(client.enabled, client.url)

    # Refine text (auto-connects on first call)
    refined = await refine_text("hello world")
"""

from .client import TextRefinerClient
from .config import ENABLE_TEXT_REFINER, TEXT_REFINER_TIMEOUT, TEXT_REFINER_URL
from .utils import capitalize_text

# Singleton instance
_client: TextRefinerClient | None = None


def get_client() -> TextRefinerClient:
    """Get singleton TextRefinerClient instance."""
    global _client
    if _client is None:
        _client = TextRefinerClient()
    return _client


async def check_text_refiner() -> bool:
    """Check if text-refiner service is available."""
    return await get_client().check_availability()


async def refine_text(text: str, punctuate: bool = True, correct: bool = True) -> str:
    """Refine text with punctuation and/or spelling correction."""
    return await get_client().refine(text, punctuate, correct)


__all__ = [
    "ENABLE_TEXT_REFINER",
    "TEXT_REFINER_TIMEOUT",
    "TEXT_REFINER_URL",
    "TextRefinerClient",
    "capitalize_text",
    "check_text_refiner",
    "get_client",
    "refine_text",
]
