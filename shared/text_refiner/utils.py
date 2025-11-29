"""Utility functions for text processing."""


def capitalize_text(text: str) -> str:
    """Capitalize first letter (sync fallback when service unavailable)."""
    if not text:
        return text
    return text[0].upper() + text[1:] if len(text) > 1 else text.upper()
