"""
Audio Processing Module

Handles audio format conversion, resampling, and preprocessing.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    target_sample_rate: int = 16000
    bit_depth: int = 16
    
    @property
    def max_int16(self) -> int:
        return 32767
    
    @property
    def max_int32(self) -> int:
        return 2147483647


class AudioProcessor:
    """
    Audio preprocessing pipeline.
    
    Handles format conversion, mono mixing, and resampling.
    """
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
    
    def to_float32(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to float32 normalized to [-1, 1].
        
        Args:
            audio: Audio in any numpy dtype
        
        Returns:
            Float32 audio normalized to [-1, 1]
        """
        if audio.dtype == np.int16:
            return audio.astype(np.float32) / self.config.max_int16
        elif audio.dtype == np.int32:
            return audio.astype(np.float32) / self.config.max_int32
        elif audio.dtype == np.float64:
            return audio.astype(np.float32)
        else:
            return audio.astype(np.float32)
    
    def to_mono(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert multi-channel audio to mono.
        
        Args:
            audio: Audio samples, possibly multi-channel
        
        Returns:
            Mono audio samples
        """
        if len(audio.shape) > 1:
            return audio.mean(axis=1).astype(np.float32)
        return audio
    
    def resample(self, audio: np.ndarray, source_rate: int) -> np.ndarray:
        """
        Resample audio to target sample rate using linear interpolation.
        
        Args:
            audio: Audio samples
            source_rate: Source sample rate in Hz
        
        Returns:
            Resampled audio at target sample rate
        """
        target_rate = self.config.target_sample_rate
        
        if source_rate == target_rate:
            return audio
        
        ratio = target_rate / source_rate
        new_length = int(len(audio) * ratio)
        
        if new_length <= 0:
            return np.array([], dtype=np.float32)
        
        indices = np.linspace(0, len(audio) - 1, new_length).astype(int)
        return audio[indices]
    
    def to_int16_bytes(self, audio: np.ndarray) -> bytes:
        """
        Convert float32 audio to int16 PCM bytes.
        
        Args:
            audio: Float32 audio [-1, 1]
        
        Returns:
            Raw PCM bytes (16-bit little-endian)
        """
        audio_clipped = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_clipped * self.config.max_int16).astype(np.int16)
        return audio_int16.tobytes()
    
    def preprocess(self, sample_rate: int, audio_data: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline.
        
        Args:
            sample_rate: Source sample rate
            audio_data: Raw audio from input device
        
        Returns:
            Preprocessed float32 mono audio at target sample rate
        """
        audio = self.to_float32(audio_data)
        audio = self.to_mono(audio)
        audio = self.resample(audio, sample_rate)
        return audio
    
    def get_duration(self, audio: np.ndarray) -> float:
        """Get duration of audio in seconds."""
        return len(audio) / self.config.target_sample_rate
