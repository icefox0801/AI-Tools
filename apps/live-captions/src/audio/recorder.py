"""Audio recorder that uploads to Audio Notes API.

This module captures audio and uploads it to the Audio Notes service.
Audio Notes handles all file storage - Live Captions just sends the audio.

Features:
- Upload to Audio Notes API every 60 seconds
- Initial upload after 3 seconds of audio
- No local file handling - that's Audio Notes' job
"""

import contextlib
import io
import logging
import os
import threading
import wave
from collections.abc import Callable
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

# Audio format constants (must match capture.py)
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
CHANNELS = 1

# Auto-upload timing
INITIAL_UPLOAD_DELAY = 3  # First upload after 3 seconds of audio
AUTO_UPLOAD_INTERVAL = 60  # Then upload every 60 seconds (1 minute)

# Audio Notes API URL (default for local development)
AUDIO_NOTES_API = os.getenv("AUDIO_NOTES_URL", "http://localhost:7860")


class AudioRecorder:
    """Records audio and uploads to Audio Notes API.

    Usage:
        recorder = AudioRecorder()
        recorder.start()

        # In audio callback:
        recorder.add_chunk(audio_bytes)

        # When done:
        recorder.stop()
    """

    def __init__(
        self, api_url: str | None = None, on_duration_change: Callable[[float], None] | None = None
    ):
        """Initialize recorder.

        Args:
            api_url: Audio Notes API URL
            on_duration_change: Callback when duration changes (for UI updates)
        """
        self.api_url = api_url or AUDIO_NOTES_API
        self.on_duration_change = on_duration_change

        # Recording state
        self._chunks: list[bytes] = []
        self._recording = False
        self._start_time: datetime | None = None
        self._current_filename: str | None = None
        self._lock = threading.Lock()

        # Track total bytes for duration calculation
        self._total_bytes = 0

        # Auto-upload state
        self._upload_timer: threading.Timer | None = None
        self._last_upload_bytes = 0
        self._first_upload_done = False

        # Status file update timer (for IPC with tray app)
        self._status_timer: threading.Timer | None = None

        logger.info(f"Audio Notes API: {self.api_url}")

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    @property
    def duration(self) -> float:
        """Get current recording duration in seconds."""
        if self._total_bytes == 0:
            return 0.0
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
            self._first_upload_done = False
            self._last_upload_bytes = 0

            # Generate filename with timestamp
            timestamp = self._start_time.strftime("%Y%m%d_%H%M%S")
            self._current_filename = f"recording_{timestamp}.wav"

            logger.info(f"Started recording: {self._current_filename}")

            # Write initial status file for IPC
            self._update_status_file()

            # Start auto-upload timer
            self._start_upload_timer()

            # Start status file update timer
            self._start_status_timer()

            return True

    def _create_wav_bytes(self, audio_data: bytes) -> bytes:
        """Create a WAV file in memory from raw PCM data."""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav:
            wav.setnchannels(CHANNELS)
            wav.setsampwidth(SAMPLE_WIDTH)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(audio_data)
        wav_buffer.seek(0)
        return wav_buffer.read()

    def _upload_to_api(self, audio_data: bytes) -> bool:
        """Upload audio to Audio Notes API.

        Returns:
            True if upload succeeded
        """
        try:
            wav_bytes = self._create_wav_bytes(audio_data)

            files = {"audio": (self._current_filename, wav_bytes, "audio/wav")}
            data = {"filename": self._current_filename, "append": "false"}

            resp = requests.post(
                f"{self.api_url}/api/upload-audio", files=files, data=data, timeout=30
            )

            if resp.status_code == 200:
                result = resp.json()
                logger.info(
                    f"Uploaded: {result.get('filename')} "
                    f"({result.get('duration', 0):.1f}s, {result.get('size_mb', 0):.2f} MB)"
                )
                return True
            else:
                logger.warning(f"Upload failed: {resp.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            logger.debug("Audio Notes API not available")
            return False
        except Exception as e:
            logger.warning(f"Upload error: {e}")
            return False

    def _start_upload_timer(self):
        """Start the auto-upload timer."""
        if self._upload_timer:
            self._upload_timer.cancel()

        # Use shorter delay for first upload, then regular interval
        delay = INITIAL_UPLOAD_DELAY if not self._first_upload_done else AUTO_UPLOAD_INTERVAL

        self._upload_timer = threading.Timer(delay, self._auto_upload)
        self._upload_timer.daemon = True
        self._upload_timer.start()

    def _auto_upload(self):
        """Auto-upload callback."""
        if not self._recording:
            return

        with self._lock:
            # Check minimum duration for first upload
            min_bytes = INITIAL_UPLOAD_DELAY * SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS

            # Skip if not enough audio yet
            if not self._first_upload_done and self._total_bytes < min_bytes:
                if self._recording:
                    self._upload_timer = threading.Timer(1, self._auto_upload)
                    self._upload_timer.daemon = True
                    self._upload_timer.start()
                return

            # Only upload if we have new data
            if self._total_bytes > self._last_upload_bytes and self._chunks:
                audio_data = b"".join(self._chunks)

                if self._upload_to_api(audio_data):
                    self._last_upload_bytes = self._total_bytes
                    upload_type = "Initial" if not self._first_upload_done else "Auto"
                    logger.info(
                        f"{upload_type} upload: {self.duration_str}, {self.file_size_mb:.1f} MB"
                    )

                self._first_upload_done = True

        # Schedule next upload
        if self._recording:
            self._start_upload_timer()

    def _start_status_timer(self):
        """Start the status file update timer (for tray app IPC)."""
        if self._status_timer:
            self._status_timer.cancel()

        self._status_timer = threading.Timer(1.0, self._update_status_periodic)
        self._status_timer.daemon = True
        self._status_timer.start()

    def _update_status_periodic(self):
        """Periodically update status file for tray app."""
        if not self._recording:
            return

        self._update_status_file()

        # Schedule next update
        if self._recording:
            self._start_status_timer()

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

        # Notify duration change
        if self.on_duration_change:
            with contextlib.suppress(Exception):
                self.on_duration_change(self.duration)

    def stop(self) -> str | None:
        """Stop recording and upload final audio.

        Returns:
            Filename if uploaded successfully, None otherwise
        """
        if self._upload_timer:
            self._upload_timer.cancel()
            self._upload_timer = None

        if self._status_timer:
            self._status_timer.cancel()
            self._status_timer = None

        with self._lock:
            if not self._recording:
                return None

            self._recording = False
            self._update_status_file()  # Update status file

            if not self._chunks:
                logger.warning("No audio recorded")
                return None

            # Final upload
            audio_data = b"".join(self._chunks)

            if self._upload_to_api(audio_data):
                logger.info(f"Final upload: {self._current_filename} ({self.duration_str})")
                return self._current_filename
            else:
                logger.warning("Final upload failed - audio not saved")
                return None

    def clear(self):
        """Clear recording without uploading."""
        if self._upload_timer:
            self._upload_timer.cancel()
            self._upload_timer = None

        if self._status_timer:
            self._status_timer.cancel()
            self._status_timer = None

        with self._lock:
            self._recording = False
            self._chunks = []
            self._total_bytes = 0
            self._current_filename = None
            self._update_status_file()  # Update status file

    def _update_status_file(self):
        """Write recording status to a file for IPC with tray app."""
        try:
            status_file = get_status_file_path()
            if self._recording:
                # Write current status
                import json

                status = {
                    "recording": True,
                    "start_time": self._start_time.isoformat() if self._start_time else None,
                    "duration": self.duration,
                    "duration_str": self.duration_str,
                }
                with open(status_file, "w") as f:
                    json.dump(status, f)
            else:
                # Clear status file
                if status_file.exists():
                    status_file.unlink()
        except Exception as e:
            logger.debug(f"Failed to update status file: {e}")


def get_status_file_path():
    """Get the path for the recording status file."""
    from pathlib import Path

    return Path(os.environ.get("TEMP", "/tmp")) / "live_captions_recording.json"


def read_recording_status() -> tuple[bool, str, float]:
    """Read recording status from the status file (for tray app).

    Returns:
        Tuple of (is_recording, duration_str, duration_seconds)
    """
    try:
        import json
        from datetime import datetime

        status_file = get_status_file_path()
        if not status_file.exists():
            return False, "00:00", 0.0

        # Check if file is stale (older than 5 seconds means process died)
        import time

        if time.time() - status_file.stat().st_mtime > 5:
            return False, "00:00", 0.0

        with open(status_file, "r") as f:
            status = json.load(f)

        if not status.get("recording"):
            return False, "00:00", 0.0

        # Calculate current duration from start_time
        start_time_str = status.get("start_time")
        if start_time_str:
            start_time = datetime.fromisoformat(start_time_str)
            duration = (datetime.now() - start_time).total_seconds()
            mins = int(duration // 60)
            secs = int(duration % 60)
            duration_str = f"{mins:02d}:{secs:02d}"
            return True, duration_str, duration

        return True, status.get("duration_str", "00:00"), status.get("duration", 0.0)
    except Exception:
        return False, "00:00", 0.0


# Global recorder instance for tray app access
_global_recorder: AudioRecorder | None = None


def get_recorder() -> AudioRecorder | None:
    """Get the global recorder instance."""
    return _global_recorder


def set_recorder(recorder: AudioRecorder | None):
    """Set the global recorder instance."""
    global _global_recorder
    _global_recorder = recorder
