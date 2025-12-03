"""Recording file management."""

import os
import wave
import logging
from pathlib import Path
from typing import List
from datetime import datetime

from config import RECORDINGS_DIR

logger = logging.getLogger(__name__)


def get_audio_duration(audio_path: str) -> float:
    """Get audio file duration in seconds."""
    try:
        with wave.open(audio_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / rate
    except Exception:
        try:
            size = os.path.getsize(audio_path)
            return size / 32000
        except Exception:
            return 0.0


def list_recordings() -> List[dict]:
    """List all recordings in the shared recordings directory."""
    recordings = []
    
    if not RECORDINGS_DIR.exists():
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        return recordings
    
    extensions = ['.wav', '.mp3', '.m4a', '.ogg', '.flac']
    for ext in extensions:
        for audio_file in RECORDINGS_DIR.glob(f"*{ext}"):
            try:
                stat = audio_file.stat()
                size_mb = stat.st_size / (1024 * 1024)
                mod_time = datetime.fromtimestamp(stat.st_mtime)
                duration = get_audio_duration(str(audio_file))
                
                transcript_path = audio_file.with_suffix('.txt')
                has_transcript = transcript_path.exists()
                
                recordings.append({
                    'path': str(audio_file),
                    'name': audio_file.name,
                    'size_mb': size_mb,
                    'duration': duration,
                    'duration_str': f"{int(duration // 60):02d}:{int(duration % 60):02d}",
                    'date': mod_time.strftime("%Y-%m-%d %H:%M"),
                    'timestamp': stat.st_mtime,
                    'has_transcript': has_transcript
                })
            except Exception as e:
                logger.warning(f"Error reading {audio_file}: {e}")
    
    recordings.sort(key=lambda x: x['timestamp'], reverse=True)
    return recordings
