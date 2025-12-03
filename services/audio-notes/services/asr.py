"""ASR (Automatic Speech Recognition) services."""

import logging
from pathlib import Path

import requests

from config import WHISPER_URL, PARAKEET_URL, logger
from services.recordings import get_audio_duration

logger = logging.getLogger(__name__)


def transcribe_audio(audio_path: str, backend: str = "parakeet") -> tuple[str, float]:
    """Transcribe audio file using selected ASR backend."""
    if backend == "whisper":
        base_url = WHISPER_URL
    else:
        base_url = PARAKEET_URL
    
    logger.info(f"Transcribing with {backend}: {audio_path}")
    
    with open(audio_path, 'rb') as f:
        audio_data = f.read()
    
    duration = get_audio_duration(audio_path)
    
    ext = Path(audio_path).suffix.lower()
    content_types = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.m4a': 'audio/mp4',
        '.ogg': 'audio/ogg',
        '.flac': 'audio/flac'
    }
    content_type = content_types.get(ext, 'audio/wav')
    
    files = {'file': (Path(audio_path).name, audio_data, content_type)}
    
    try:
        resp = requests.post(
            f"{base_url}/transcribe",
            files=files,
            timeout=300
        )
        
        if resp.status_code == 200:
            data = resp.json()
            text = data.get("text", "")
            duration = data.get("duration", duration)
            logger.info(f"Transcription complete ({backend}): {len(text)} chars, {duration:.1f}s")
            return text, duration
        else:
            logger.error(f"Transcription failed: {resp.status_code} - {resp.text}")
            return f"Error: {resp.status_code}", duration
            
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return f"Error: {e}", duration


def unload_asr_model(backend: str = "Parakeet") -> tuple[bool, str]:
    """Unload ASR model from GPU to free memory."""
    url = PARAKEET_URL if backend == "Parakeet" else WHISPER_URL
    try:
        resp = requests.post(f"{url}/unload", timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            logger.info(f"ASR model unloaded: {data}")
            return True, data.get("message", "Model unloaded")
        else:
            logger.warning(f"Failed to unload ASR model: {resp.status_code}")
            return False, f"Error: {resp.status_code}"
    except Exception as e:
        logger.warning(f"Could not unload ASR model: {e}")
        return False, str(e)
