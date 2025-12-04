"""Backend services for Audio Notes."""

from .asr import transcribe_audio, unload_asr_model
from .health import (
    check_ollama_health,
    check_parakeet_health,
    check_whisper_health,
    get_ollama_models,
)
from .llm import (
    chat_with_context,
    chat_with_context_streaming,
    generate_chat_title,
    summarize_streaming,
)
from .recordings import get_audio_duration, list_recordings

__all__ = [
    "chat_with_context",
    "chat_with_context_streaming",
    "check_ollama_health",
    "check_parakeet_health",
    "check_whisper_health",
    "generate_chat_title",
    "get_audio_duration",
    "get_ollama_models",
    "list_recordings",
    "summarize_streaming",
    "transcribe_audio",
    "unload_asr_model",
]
