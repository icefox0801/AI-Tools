"""Audio capture module for Live Captions."""

from .capture import AudioCapture, MicrophoneCapture, SystemAudioCapture
from .devices import get_default_loopback_info, get_default_microphone_info, list_devices
from .recorder import AudioRecorder, get_recorder, set_recorder
from .utils import (
    CHUNK_DURATION_MS,
    TARGET_SAMPLE_RATE,
    calculate_chunk_size,
    resample_audio,
    stereo_to_mono,
)

__all__ = [
    "CHUNK_DURATION_MS",
    # Constants
    "TARGET_SAMPLE_RATE",
    # Capture classes
    "AudioCapture",
    # Recorder
    "AudioRecorder",
    "MicrophoneCapture",
    "SystemAudioCapture",
    "calculate_chunk_size",
    "get_default_loopback_info",
    "get_default_microphone_info",
    "get_recorder",
    # Device functions
    "list_devices",
    # Utility functions
    "resample_audio",
    "set_recorder",
    "stereo_to_mono",
]
