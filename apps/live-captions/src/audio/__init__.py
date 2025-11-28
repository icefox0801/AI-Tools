"""Audio capture module for Live Captions."""

from .capture import MicrophoneCapture, SystemAudioCapture, AudioCapture
from .utils import (
    resample_audio, 
    stereo_to_mono, 
    calculate_chunk_size,
    TARGET_SAMPLE_RATE, 
    CHUNK_DURATION_MS
)
from .devices import (
    list_devices, 
    get_default_microphone_info, 
    get_default_loopback_info
)

__all__ = [
    # Capture classes
    'AudioCapture',
    'MicrophoneCapture',
    'SystemAudioCapture', 
    # Utility functions
    'resample_audio',
    'stereo_to_mono',
    'calculate_chunk_size',
    # Device functions
    'list_devices',
    'get_default_microphone_info',
    'get_default_loopback_info',
    # Constants
    'TARGET_SAMPLE_RATE',
    'CHUNK_DURATION_MS',
]
