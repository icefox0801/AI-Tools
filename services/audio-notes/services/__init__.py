"""Backend services for Audio Notes."""

from .asr import transcribe_audio, unload_asr_model
from .health import (
    check_ollama_health,
    check_parakeet_health,
    check_whisper_health,
    get_ollama_models,
)
from .langchain_chat import chat_streaming as langchain_chat_streaming
from .langchain_chat import generate_title as langchain_generate_title
from .llm import (
    generate_chat_title,
    summarize_streaming,
)
from .recordings import get_audio_duration, list_recordings

__all__ = [
    "check_ollama_health",
    "check_parakeet_health",
    "check_whisper_health",
    "generate_chat_title",
    "get_audio_duration",
    "get_ollama_models",
    "langchain_chat_streaming",
    "langchain_generate_title",
    "list_recordings",
    "summarize_streaming",
    "transcribe_audio",
    "unload_asr_model",
]
