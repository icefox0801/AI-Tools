"""Audio capture classes for microphone and system audio."""

import logging
from typing import Callable, Optional
from abc import ABC, abstractmethod

from .utils import (
    resample_audio, stereo_to_mono, 
    TARGET_SAMPLE_RATE, CHUNK_DURATION_MS, calculate_chunk_size
)

logger = logging.getLogger(__name__)


class AudioCapture(ABC):
    """Abstract base class for audio capture."""
    
    def __init__(self, callback: Callable[[bytes], None]):
        """
        Initialize audio capture.
        
        Args:
            callback: Function to call with captured audio data (16-bit PCM, mono, 16kHz)
        """
        self.callback = callback
        self.running = False
    
    @abstractmethod
    def start(self) -> bool:
        """Start capturing audio. Returns True on success."""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop capturing audio."""
        pass
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return human-readable source name."""
        pass


class MicrophoneCapture(AudioCapture):
    """Capture audio from microphone using PyAudio."""
    
    def __init__(self, callback: Callable[[bytes], None], device_index: Optional[int] = None):
        """
        Initialize microphone capture.
        
        Args:
            callback: Function to call with captured audio data
            device_index: Specific input device index, or None for default
        """
        super().__init__(callback)
        self.device_index = device_index
        self.pyaudio_instance = None
        self.stream = None
        self.capture_rate = TARGET_SAMPLE_RATE
        self.capture_channels = 1
        self._device_name = "Microphone"
    
    @property
    def source_name(self) -> str:
        return f"🎤 {self._device_name}"
    
    def start(self) -> bool:
        try:
            import pyaudio
            
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Get device info
            if self.device_index is not None:
                device_info = self.pyaudio_instance.get_device_info_by_index(self.device_index)
            else:
                device_info = self.pyaudio_instance.get_default_input_device_info()
            
            self._device_name = device_info['name']
            native_rate = int(device_info['defaultSampleRate'])
            max_channels = int(device_info['maxInputChannels'])
            
            # Use stereo for stereo mix devices
            is_stereo_mix = '立体声混音' in self._device_name or 'stereo mix' in self._device_name.lower()
            self.capture_channels = 2 if (is_stereo_mix and max_channels >= 2) else 1
            self.capture_rate = native_rate
            
            chunk_size = calculate_chunk_size(native_rate)
            
            logger.info(f"Microphone: {self._device_name}")
            logger.info(f"Rate: {native_rate}Hz → {TARGET_SAMPLE_RATE}Hz, Channels: {self.capture_channels}")
            
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=self.capture_channels,
                rate=native_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.stream.start_stream()
            self.running = True
            logger.info("Microphone capture started")
            return True
            
        except Exception as e:
            logger.error(f"Microphone start failed: {e}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        import pyaudio
        
        if not self.running:
            return (None, pyaudio.paComplete)
        
        try:
            audio_data = in_data
            
            # Stereo to mono
            if self.capture_channels == 2:
                audio_data = stereo_to_mono(audio_data)
            
            # Resample
            audio_data = resample_audio(audio_data, self.capture_rate, TARGET_SAMPLE_RATE)
            
            self.callback(audio_data)
            
        except Exception as e:
            logger.error(f"Mic callback error: {e}")
        
        return (None, pyaudio.paContinue)
    
    def stop(self):
        self.running = False
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except Exception:
                pass
            self.pyaudio_instance = None
        
        logger.info("Microphone capture stopped")


class SystemAudioCapture(AudioCapture):
    """Capture system audio via WASAPI loopback using PyAudioWPatch."""
    
    def __init__(self, callback: Callable[[bytes], None], device_index: Optional[int] = None):
        """
        Initialize system audio capture.
        
        Args:
            callback: Function to call with captured audio data
            device_index: Specific loopback device index, or None for default
        """
        super().__init__(callback)
        self.device_index = device_index
        self.pyaudio_instance = None
        self.stream = None
        self.capture_rate = 48000
        self._device_name = "System Audio"
    
    @property
    def source_name(self) -> str:
        return f"🔊 {self._device_name}"
    
    def start(self) -> bool:
        try:
            import pyaudiowpatch as pyaudio
            
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Find loopback device
            loopback_device = None
            
            if self.device_index is not None:
                # Use specified device
                loopback_device = self.pyaudio_instance.get_device_info_by_index(self.device_index)
                if not loopback_device.get('isLoopbackDevice'):
                    logger.warning(f"Device {self.device_index} is not a loopback device")
            else:
                # Find default speaker's loopback
                try:
                    wasapi_info = self.pyaudio_instance.get_host_api_info_by_type(pyaudio.paWASAPI)
                    default_output = self.pyaudio_instance.get_device_info_by_index(
                        wasapi_info['defaultOutputDevice']
                    )
                    default_name = default_output['name']
                    
                    # Find corresponding loopback device
                    for i in range(self.pyaudio_instance.get_device_count()):
                        dev = self.pyaudio_instance.get_device_info_by_index(i)
                        if dev.get('isLoopbackDevice') and default_name in dev['name']:
                            loopback_device = dev
                            self.device_index = i
                            break
                except Exception as e:
                    logger.error(f"Failed to find default loopback: {e}")
            
            if not loopback_device:
                raise RuntimeError("No loopback device found. Install PyAudioWPatch: pip install pyaudiowpatch")
            
            self._device_name = loopback_device['name'].replace(' [Loopback]', '')
            self.capture_rate = int(loopback_device['defaultSampleRate'])
            channels = loopback_device['maxInputChannels']
            
            chunk_size = calculate_chunk_size(self.capture_rate)
            
            logger.info(f"System Audio: {self._device_name}")
            logger.info(f"Rate: {self.capture_rate}Hz → {TARGET_SAMPLE_RATE}Hz, Channels: {channels}")
            
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=self.capture_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.stream.start_stream()
            self.running = True
            logger.info("System audio capture started (WASAPI loopback)")
            return True
            
        except ImportError:
            logger.error("pyaudiowpatch not installed. Run: pip install pyaudiowpatch")
            return False
        except Exception as e:
            logger.error(f"System audio start failed: {e}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        import pyaudiowpatch as pyaudio
        
        if not self.running:
            return (None, pyaudio.paComplete)
        
        try:
            # Always stereo from loopback, convert to mono
            audio_data = stereo_to_mono(in_data)
            
            # Resample from capture rate to target rate
            audio_data = resample_audio(audio_data, self.capture_rate, TARGET_SAMPLE_RATE)
            
            self.callback(audio_data)
            
        except Exception as e:
            logger.error(f"Loopback callback error: {e}")
        
        return (None, pyaudio.paContinue)
    
    def stop(self):
        self.running = False
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except Exception:
                pass
            self.pyaudio_instance = None
        
        logger.info("System audio capture stopped")