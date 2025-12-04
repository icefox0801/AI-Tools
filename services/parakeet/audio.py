"""Audio utilities for Parakeet ASR service."""

import io

import numpy as np
import soundfile as sf


def pcm_to_float(pcm_bytes: bytes) -> np.ndarray:
    """Convert PCM int16 bytes to float32 array normalized to [-1, 1]."""
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def float_to_pcm(audio: np.ndarray) -> bytes:
    """Convert float32 array to PCM int16 bytes."""
    return (audio * 32768).astype(np.int16).tobytes()


def load_audio_file(audio_data: bytes, target_sr: int = 16000) -> np.ndarray:
    """Load and preprocess audio file to target sample rate."""
    try:
        audio_io = io.BytesIO(audio_data)
        audio_array, sr = sf.read(audio_io)

        # Resample if needed
        if sr != target_sr:
            import librosa

            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)

        # Convert stereo to mono
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        return audio_array
    except Exception:
        # Assume raw PCM if file parsing fails
        return pcm_to_float(audio_data)
