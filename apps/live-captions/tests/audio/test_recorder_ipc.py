"""
Unit tests for recorder IPC (Inter-Process Communication) functions.

Tests the file-based status sharing mechanism between the Live Captions
subprocess and the system tray application.
"""

import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add apps/live-captions to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "apps" / "live-captions"))

from src.audio.recorder import (
    AudioRecorder,
    get_recorder,
    get_status_file_path,
    read_recording_status,
    set_recorder,
)


class TestGetStatusFilePath:
    """Tests for get_status_file_path function."""

    def test_returns_path_object(self):
        """Returns a Path object."""
        path = get_status_file_path()
        assert isinstance(path, Path)

    def test_filename_is_correct(self):
        """Filename is live_captions_recording.json."""
        path = get_status_file_path()
        assert path.name == "live_captions_recording.json"

    def test_uses_temp_directory(self):
        """Path is in temp directory."""
        path = get_status_file_path()
        temp_dir = os.environ.get("TEMP", "/tmp")
        assert str(path.parent) == temp_dir

    def test_respects_temp_env_var(self):
        """Respects custom TEMP environment variable."""
        custom_temp = tempfile.mkdtemp()
        with patch.dict(os.environ, {"TEMP": custom_temp}):
            path = get_status_file_path()
            assert str(path.parent) == custom_temp
        os.rmdir(custom_temp)


class TestReadRecordingStatus:
    """Tests for read_recording_status function."""

    def setup_method(self):
        """Setup test fixtures."""
        self.status_file = get_status_file_path()
        # Clean up any existing status file
        if self.status_file.exists():
            self.status_file.unlink()

    def teardown_method(self):
        """Clean up after tests."""
        if self.status_file.exists():
            self.status_file.unlink()

    def test_no_file_returns_not_recording(self):
        """Returns not recording when status file doesn't exist."""
        is_recording, duration_str, duration, _seconds_since_audio = read_recording_status()
        assert is_recording is False
        assert duration_str == "00:00"
        assert duration == 0.0

    def test_reads_recording_status(self):
        """Reads recording status from file."""
        start_time = datetime.now()
        status = {
            "recording": True,
            "start_time": start_time.isoformat(),
            "duration": 65.5,
            "duration_str": "01:05",
        }
        with open(self.status_file, "w") as f:
            json.dump(status, f)

        is_recording, _duration_str, duration, _seconds_since_audio = read_recording_status()
        assert is_recording is True
        # Duration is calculated from start_time, so it should be close to 0
        assert duration >= 0

    def test_stale_file_returns_not_recording(self):
        """Returns not recording when file is stale (>5 seconds old)."""
        status = {
            "recording": True,
            "start_time": datetime.now().isoformat(),
            "duration": 10.0,
            "duration_str": "00:10",
        }
        with open(self.status_file, "w") as f:
            json.dump(status, f)

        # Make file appear old by modifying its mtime
        old_time = time.time() - 10  # 10 seconds ago
        os.utime(self.status_file, (old_time, old_time))

        is_recording, duration_str, duration, _seconds_since_audio = read_recording_status()
        assert is_recording is False
        assert duration_str == "00:00"
        assert duration == 0.0

    def test_recording_false_in_file(self):
        """Returns not recording when file says recording=false."""
        status = {"recording": False}
        with open(self.status_file, "w") as f:
            json.dump(status, f)

        is_recording, _duration_str, _duration, _seconds_since_audio = read_recording_status()
        assert is_recording is False

    def test_calculates_duration_from_start_time(self):
        """Calculates duration from start_time field."""
        # Start time 90 seconds ago
        start_time = datetime.now()
        status = {
            "recording": True,
            "start_time": start_time.isoformat(),
        }
        with open(self.status_file, "w") as f:
            json.dump(status, f)

        is_recording, _duration_str, duration, _seconds_since_audio = read_recording_status()
        assert is_recording is True
        # Duration should be very small since we just wrote the file
        assert duration < 1.0

    def test_handles_malformed_json(self):
        """Handles malformed JSON gracefully."""
        with open(self.status_file, "w") as f:
            f.write("not valid json {")

        is_recording, duration_str, duration, _seconds_since_audio = read_recording_status()
        assert is_recording is False
        assert duration_str == "00:00"
        assert duration == 0.0

    def test_handles_missing_fields(self):
        """Handles missing fields in JSON."""
        status = {"recording": True}  # Missing start_time
        with open(self.status_file, "w") as f:
            json.dump(status, f)

        is_recording, duration_str, _duration, _seconds_since_audio = read_recording_status()
        assert is_recording is True
        # Should fall back to stored values
        assert duration_str == "00:00"


class TestGlobalRecorder:
    """Tests for global recorder get/set functions."""

    def teardown_method(self):
        """Reset global recorder after each test."""
        set_recorder(None)

    def test_initial_recorder_is_none(self):
        """Initial global recorder is None."""
        set_recorder(None)  # Reset first
        assert get_recorder() is None

    def test_set_and_get_recorder(self):
        """Can set and get a recorder instance."""
        recorder = MagicMock(spec=AudioRecorder)
        set_recorder(recorder)
        assert get_recorder() is recorder

    def test_set_recorder_to_none(self):
        """Can set recorder back to None."""
        recorder = MagicMock(spec=AudioRecorder)
        set_recorder(recorder)
        set_recorder(None)
        assert get_recorder() is None


class TestAudioRecorderStatusFile:
    """Tests for AudioRecorder status file integration."""

    def setup_method(self):
        """Setup test fixtures."""
        self.status_file = get_status_file_path()
        if self.status_file.exists():
            self.status_file.unlink()

    def teardown_method(self):
        """Clean up after tests."""
        if self.status_file.exists():
            self.status_file.unlink()

    @patch("src.audio.recorder.requests.post")
    def test_start_creates_status_file(self, mock_post):
        """Starting recording creates status file."""
        recorder = AudioRecorder()
        recorder.start()

        # Wait a bit for the timer to write
        time.sleep(0.1)

        assert self.status_file.exists()

        with open(self.status_file) as f:
            status = json.load(f)

        assert status["recording"] is True
        assert "start_time" in status

        # Cleanup
        recorder.clear()

    @patch("src.audio.recorder.requests.post")
    def test_stop_clears_status_file(self, mock_post):
        """Stopping recording clears status file."""
        recorder = AudioRecorder()
        recorder.start()
        time.sleep(0.1)

        # Add some audio data so stop doesn't return None immediately
        recorder.add_chunk(b"\x00" * 1600)  # 0.1s of audio

        recorder.stop()

        # Status file should be deleted when recording=False
        assert not self.status_file.exists()

    @patch("src.audio.recorder.requests.post")
    def test_clear_removes_status_file(self, mock_post):
        """Clearing recording removes status file."""
        recorder = AudioRecorder()
        recorder.start()
        time.sleep(0.1)

        recorder.clear()

        assert not self.status_file.exists()


class TestAudioRecorderProperties:
    """Tests for AudioRecorder properties."""

    def test_duration_str_format(self):
        """Duration string is formatted as MM:SS."""
        recorder = AudioRecorder()
        recorder._total_bytes = 0
        assert recorder.duration_str == "00:00"

        # 65 seconds of audio (16kHz, 16-bit, mono)
        recorder._total_bytes = 65 * 16000 * 2
        assert recorder.duration_str == "01:05"

    def test_file_size_mb(self):
        """File size is calculated correctly."""
        recorder = AudioRecorder()
        recorder._total_bytes = 1024 * 1024  # 1 MB
        assert recorder.file_size_mb == 1.0

    def test_is_recording_property(self):
        """is_recording property reflects state."""
        recorder = AudioRecorder()
        assert recorder.is_recording is False

        recorder._recording = True
        assert recorder.is_recording is True

    def test_duration_property(self):
        """Duration property calculates from bytes."""
        recorder = AudioRecorder()
        recorder._total_bytes = 0
        assert recorder.duration == 0.0

        # 1 second of audio (16kHz, 16-bit, mono = 32000 bytes)
        recorder._total_bytes = 32000
        assert recorder.duration == 1.0
