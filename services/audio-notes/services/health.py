"""Health check functions for backend services."""

import logging

import requests
from config import OLLAMA_MODEL, OLLAMA_URL, PARAKEET_URL, WHISPER_URL, logger

logger = logging.getLogger(__name__)


def check_whisper_health() -> tuple[bool, str]:
    """Check if Whisper service is available."""
    try:
        resp = requests.get(f"{WHISPER_URL}/health", timeout=2)
        if resp.status_code == 200:
            return True, "Whisper ASR ready"
        return False, f"Whisper returned status {resp.status_code}"
    except Exception as e:
        return False, f"Whisper not available: {e}"


def check_parakeet_health() -> tuple[bool, str]:
    """Check if Parakeet service is available."""
    try:
        resp = requests.get(f"{PARAKEET_URL}/health", timeout=2)
        if resp.status_code == 200:
            return True, "Parakeet ASR ready"
        return False, f"Parakeet returned status {resp.status_code}"
    except Exception as e:
        return False, f"Parakeet not available: {e}"


def check_ollama_health() -> tuple[bool, str]:
    """Check if Ollama service is available."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            if any(OLLAMA_MODEL in m for m in models):
                return True, f"Ollama ready ({OLLAMA_MODEL})"
            return False, f"Model {OLLAMA_MODEL} not found. Available: {models}"
        return False, f"Ollama returned status {resp.status_code}"
    except Exception as e:
        return False, f"Ollama not available: {e}"


def get_ollama_models() -> list[str]:
    """Get list of available Ollama models."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            if OLLAMA_MODEL in models:
                models.remove(OLLAMA_MODEL)
                models.insert(0, OLLAMA_MODEL)
            return models if models else [OLLAMA_MODEL]
        return [OLLAMA_MODEL]
    except Exception:
        return [OLLAMA_MODEL]
