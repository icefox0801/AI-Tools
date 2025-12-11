"""Audio utility functions for resampling and format conversion."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Audio settings
TARGET_SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 100  # 100ms chunks for faster real-time updates (was 200ms)


def resample_audio(audio_data: bytes, from_rate: int, to_rate: int) -> bytes:
    """
    Resample audio using polyphase filtering for better quality.

    Args:
        audio_data: Raw 16-bit PCM audio bytes
        from_rate: Source sample rate in Hz
        to_rate: Target sample rate in Hz

    Returns:
        Resampled audio as bytes
    """
    if from_rate == to_rate:
        return audio_data

    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

    try:
        from math import gcd

        from scipy import signal

        g = gcd(from_rate, to_rate)
        up = to_rate // g
        down = from_rate // g
        resampled = signal.resample_poly(audio_np, up, down)
    except (ImportError, TypeError, Exception) as e:
        # Fallback: linear interpolation
        # TypeError can occur with scipy/torch compatibility issues
        # Catch all exceptions to handle scipy internal errors
        if not isinstance(e, (ImportError, TypeError)):
            logger.warning(f"scipy resampling failed with {type(e).__name__}: {e}, using fallback")
        else:
            logger.debug(
                "scipy not available or incompatible, using linear interpolation for resampling"
            )
        ratio = to_rate / from_rate
        new_length = int(len(audio_np) * ratio)
        indices = np.linspace(0, len(audio_np) - 1, new_length)
        resampled = np.interp(indices, np.arange(len(audio_np)), audio_np)

    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


def stereo_to_mono(audio_data: bytes) -> bytes:
    """
    Convert stereo audio to mono by averaging channels.

    Args:
        audio_data: Stereo 16-bit PCM audio bytes (interleaved L/R)

    Returns:
        Mono audio as bytes
    """
    stereo = np.frombuffer(audio_data, dtype=np.int16)
    left = stereo[0::2]
    right = stereo[1::2]
    mono = ((left.astype(np.int32) + right.astype(np.int32)) // 2).astype(np.int16)
    return mono.tobytes()


def calculate_chunk_size(sample_rate: int, duration_ms: int = CHUNK_DURATION_MS) -> int:
    """Calculate chunk size in samples for given duration."""
    return int(sample_rate * duration_ms / 1000)
