"""Backend services for Audio Notes."""

from .health import check_whisper_health, check_parakeet_health, check_ollama_health, get_ollama_models
from .recordings import list_recordings, get_audio_duration
from .asr import transcribe_audio, unload_asr_model
from .llm import summarize_streaming, chat_with_context, chat_with_context_streaming, generate_chat_title

__all__ = [
    'check_whisper_health', 'check_parakeet_health', 'check_ollama_health', 'get_ollama_models',
    'list_recordings', 'get_audio_duration',
    'transcribe_audio', 'unload_asr_model',
    'summarize_streaming', 'chat_with_context', 'chat_with_context_streaming', 'generate_chat_title',
]
