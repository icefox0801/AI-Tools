"""Audio recorder for saving streamed audio to WAV file.

This module provides a simple recorder that accumulates audio chunks
and saves them to a WAV file for later full transcription.

The audio is saved as 16-bit PCM WAV at 16kHz mono, which is:
- ~1.92 MB per minute (uncompressed)
- Compatible with all ASR services
"""

import os
import io
import wave
import time
import threading
import logging
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Audio format constants (must match capture.py)
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
CHANNELS = 1


class AudioRecorder:
    """Records streamed audio to a WAV file.
    
    Usage:
        recorder = AudioRecorder()
        recorder.start()
        
        # In audio callback:
        recorder.add_chunk(audio_bytes)
        
        # When done:
        path = recorder.stop()  # Returns path to saved WAV
        
        # Or to clear without saving:
        recorder.clear()
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        on_duration_change: Optional[Callable[[float], None]] = None
    ):
        """Initialize recorder.
        
        Args:
            output_dir: Directory for saving recordings. Defaults to project's recordings/ folder.
            on_duration_change: Callback when duration changes (for UI updates)
        """
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Default to project's .recordings/ folder (shared with Docker audio-notes service)
            # Fall back to RECORDINGS_DIR env var or temp dir
            project_recordings = Path(__file__).parent.parent.parent.parent.parent / '.recordings'
            if project_recordings.exists():
                self.output_dir = project_recordings
            else:
                self.output_dir = Path(os.environ.get('RECORDINGS_DIR', 
                    os.path.join(os.environ.get('TEMP', '/tmp'), 'live-captions-recordings')))
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.on_duration_change = on_duration_change
        
        # Recording state
        self._chunks: list[bytes] = []
        self._recording = False
        self._start_time: Optional[datetime] = None
        self._current_file: Optional[Path] = None
        self._lock = threading.Lock()
        
        # Track total bytes for duration calculation
        self._total_bytes = 0
    
    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording
    
    @property
    def duration(self) -> float:
        """Get current recording duration in seconds."""
        if self._total_bytes == 0:
            return 0.0
        # Calculate from bytes: bytes / (sample_rate * sample_width * channels)
        return self._total_bytes / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
    
    @property
    def duration_str(self) -> str:
        """Get duration as formatted string (MM:SS)."""
        total_seconds = int(self.duration)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    @property
    def file_size_mb(self) -> float:
        """Get approximate file size in MB."""
        return self._total_bytes / (1024 * 1024)
    
    @property
    def current_file(self) -> Optional[Path]:
        """Get path to current recording file (if any)."""
        return self._current_file
    
    def start(self) -> bool:
        """Start recording.
        
        Returns:
            True if started successfully
        """
        with self._lock:
            if self._recording:
                logger.warning("Already recording")
                return False
            
            self._chunks = []
            self._total_bytes = 0
            self._start_time = datetime.now()
            self._recording = True
            
            # Generate filename with timestamp
            timestamp = self._start_time.strftime("%Y%m%d_%H%M%S")
            self._current_file = self.output_dir / f"recording_{timestamp}.wav"
            
            logger.info(f"Started recording: {self._current_file}")
            return True
    
    def add_chunk(self, audio_bytes: bytes):
        """Add an audio chunk to the recording.
        
        Args:
            audio_bytes: Raw PCM audio data (16-bit, 16kHz, mono)
        """
        if not self._recording:
            return
        
        with self._lock:
            self._chunks.append(audio_bytes)
            self._total_bytes += len(audio_bytes)
        
        # Notify duration change (for UI updates)
        if self.on_duration_change:
            try:
                self.on_duration_change(self.duration)
            except Exception:
                pass
    
    def stop(self) -> Optional[Path]:
        """Stop recording and save to WAV file.
        
        Returns:
            Path to saved WAV file, or None if no audio recorded
        """
        with self._lock:
            if not self._recording:
                return None
            
            self._recording = False
            
            if not self._chunks:
                logger.warning("No audio recorded")
                return None
            
            # Combine all chunks
            audio_data = b''.join(self._chunks)
            
            # Save as WAV
            try:
                with wave.open(str(self._current_file), 'wb') as wav:
                    wav.setnchannels(CHANNELS)
                    wav.setsampwidth(SAMPLE_WIDTH)
                    wav.setframerate(SAMPLE_RATE)
                    wav.writeframes(audio_data)
                
                logger.info(
                    f"Saved recording: {self._current_file} "
                    f"({self.duration_str}, {self.file_size_mb:.1f} MB)"
                )
                return self._current_file
                
            except Exception as e:
                logger.error(f"Failed to save recording: {e}")
                return None
    
    def clear(self):
        """Clear current recording without saving."""
        with self._lock:
            was_recording = self._recording
            self._recording = False
            self._chunks = []
            self._total_bytes = 0
            self._start_time = None
            
            # Delete file if exists
            if self._current_file and self._current_file.exists():
                try:
                    self._current_file.unlink()
                    logger.info(f"Deleted recording: {self._current_file}")
                except Exception as e:
                    logger.error(f"Failed to delete recording: {e}")
            
            self._current_file = None
            
            if was_recording:
                logger.info("Recording cleared")
    
    def get_audio_bytes(self) -> Optional[bytes]:
        """Get all recorded audio as raw bytes (without saving to file).
        
        Returns:
            Raw PCM audio data, or None if no recording
        """
        with self._lock:
            if not self._chunks:
                return None
            return b''.join(self._chunks)
    
    def save(self, continue_recording: bool = True) -> Optional[Path]:
        """Save current recording to a WAV file.
        
        Unlike stop(), this saves a snapshot of the current recording
        and can optionally continue recording.
        
        Args:
            continue_recording: If True, continue recording after save
            
        Returns:
            Path to saved WAV file, or None if no audio recorded
        """
        with self._lock:
            if not self._chunks:
                logger.warning("No audio to save")
                return None
            
            # Combine all chunks
            audio_data = b''.join(self._chunks)
            
            # Generate output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"recording_{timestamp}.wav"
            
            # Save as WAV
            try:
                with wave.open(str(output_path), 'wb') as wav:
                    wav.setnchannels(CHANNELS)
                    wav.setsampwidth(SAMPLE_WIDTH)
                    wav.setframerate(SAMPLE_RATE)
                    wav.writeframes(audio_data)
                
                duration = len(audio_data) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
                file_size_mb = len(audio_data) / (1024 * 1024)
                
                logger.info(
                    f"Saved recording: {output_path} "
                    f"({int(duration // 60):02d}:{int(duration % 60):02d}, {file_size_mb:.1f} MB)"
                )
                
                if not continue_recording:
                    self._recording = False
                    self._chunks = []
                    self._total_bytes = 0
                
                return output_path
                
            except Exception as e:
                logger.error(f"Failed to save recording: {e}")
                return None


# Global recorder instance for use by tray app
_global_recorder: Optional[AudioRecorder] = None


def get_recorder() -> AudioRecorder:
    """Get the global recorder instance."""
    global _global_recorder
    if _global_recorder is None:
        _global_recorder = AudioRecorder()
    return _global_recorder


def set_recorder(recorder: AudioRecorder):
    """Set the global recorder instance."""
    global _global_recorder
    _global_recorder = recorder
