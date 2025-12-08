"""
Unit tests for audio recorder module.

Tests the AudioRecorder class and IPC helper functions including:
- Stop request file IPC mechanism
- Recording status file reading
- Upload timing constants
- Recorder state management
"""

import json
import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.audio.recorder import (
    AUTO_UPLOAD_INTERVAL,
    INITIAL_UPLOAD_DELAY,
    SECOND_UPLOAD_DELAY,
    AudioRecorder,
    check_stop_requested,
    clear_stop_request,
    get_recorder,
    get_status_file_path,
    get_stop_request_file_path,
    read_recording_status,
    request_stop,
    set_recorder,
)


class TestUploadTimingConstants:
    """Tests for upload timing constants."""

    def test_initial_upload_delay(self):
        """Test initial upload delay is 3 seconds."""
        assert INITIAL_UPLOAD_DELAY == 3

    def test_second_upload_delay(self):
        """Test second upload delay is 27 seconds (at 30s total)."""
        assert SECOND_UPLOAD_DELAY == 27

    def test_auto_upload_interval(self):
        """Test auto upload interval is 30 seconds."""
        assert AUTO_UPLOAD_INTERVAL == 30

    def test_upload_schedule(self):
        """Test the complete upload schedule adds up correctly."""
        # First upload at 3s
        first_upload = INITIAL_UPLOAD_DELAY
        assert first_upload == 3

        # Second upload at 30s (3 + 27)
        second_upload = first_upload + SECOND_UPLOAD_DELAY
        assert second_upload == 30

        # Third upload at 60s (30 + 30)
        third_upload = second_upload + AUTO_UPLOAD_INTERVAL
        assert third_upload == 60


class TestStopRequestIPC:
    """Tests for stop request file IPC mechanism."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"TEMP": tmpdir}):
                yield tmpdir

    def test_get_stop_request_file_path(self, temp_dir):
        """Test stop request file path generation."""
        path = get_stop_request_file_path()
        assert path.name == "live_captions_stop_request"
        assert str(path.parent) == temp_dir

    def test_request_stop_creates_file(self, temp_dir):
        """Test request_stop creates the stop request file."""
        stop_file = get_stop_request_file_path()
        assert not stop_file.exists()

        request_stop()

        assert stop_file.exists()
        # Cleanup
        stop_file.unlink()

    def test_check_stop_requested_returns_true_when_file_exists(self, temp_dir):
        """Test check_stop_requested returns True and removes file."""
        stop_file = get_stop_request_file_path()
        stop_file.touch()
        assert stop_file.exists()

        result = check_stop_requested()

        assert result is True
        assert not stop_file.exists()  # File should be removed

    def test_check_stop_requested_returns_false_when_no_file(self, temp_dir):
        """Test check_stop_requested returns False when file doesn't exist."""
        stop_file = get_stop_request_file_path()
        assert not stop_file.exists()

        result = check_stop_requested()

        assert result is False

    def test_clear_stop_request_removes_file(self, temp_dir):
        """Test clear_stop_request removes existing file."""
        stop_file = get_stop_request_file_path()
        stop_file.touch()
        assert stop_file.exists()

        clear_stop_request()

        assert not stop_file.exists()

    def test_clear_stop_request_no_error_when_no_file(self, temp_dir):
        """Test clear_stop_request doesn't error when file doesn't exist."""
        stop_file = get_stop_request_file_path()
        assert not stop_file.exists()

        # Should not raise
        clear_stop_request()

    def test_full_stop_request_cycle(self, temp_dir):
        """Test complete stop request cycle: clear, request, check."""
        # Clear any existing
        clear_stop_request()
        assert not check_stop_requested()

        # Request stop
        request_stop()

        # Check should return True and clear
        assert check_stop_requested() is True
        assert check_stop_requested() is False  # Second check should be False


class TestRecordingStatusIPC:
    """Tests for recording status file IPC."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"TEMP": tmpdir}):
                yield tmpdir

    def test_get_status_file_path(self, temp_dir):
        """Test status file path generation."""
        path = get_status_file_path()
        assert path.name == "live_captions_recording.json"
        assert str(path.parent) == temp_dir

    def test_read_recording_status_no_file(self, temp_dir):
        """Test reading status when file doesn't exist."""
        is_recording, duration_str, duration, since_audio = read_recording_status()

        assert is_recording is False
        assert duration_str == "00:00"
        assert duration == 0.0
        assert since_audio == 0.0

    def test_read_recording_status_valid_file(self, temp_dir):
        """Test reading status from valid file."""
        status_file = get_status_file_path()
        start_time = datetime.now()
        status = {
            "recording": True,
            "start_time": start_time.isoformat(),
            "last_audio_time": start_time.isoformat(),
            "duration": 45.5,
            "duration_str": "00:45",
        }
        with open(status_file, "w") as f:
            json.dump(status, f)

        is_recording, duration_str, duration, since_audio = read_recording_status()

        assert is_recording is True
        # Duration is calculated from start_time, so just check it's reasonable
        assert duration >= 0

    def test_read_recording_status_not_recording(self, temp_dir):
        """Test reading status when recording is False."""
        status_file = get_status_file_path()
        status = {"recording": False}
        with open(status_file, "w") as f:
            json.dump(status, f)

        is_recording, duration_str, duration, since_audio = read_recording_status()

        assert is_recording is False
        assert duration_str == "00:00"


class TestGlobalRecorder:
    """Tests for global recorder getter/setter."""

    def test_set_and_get_recorder(self):
        """Test setting and getting global recorder."""
        mock_recorder = MagicMock(spec=AudioRecorder)

        set_recorder(mock_recorder)
        result = get_recorder()

        assert result is mock_recorder

        # Cleanup
        set_recorder(None)

    def test_get_recorder_returns_none_initially(self):
        """Test get_recorder returns None when not set."""
        set_recorder(None)
        assert get_recorder() is None


class TestAudioRecorderInit:
    """Tests for AudioRecorder initialization."""

    def test_default_init(self):
        """Test default initialization."""
        recorder = AudioRecorder()

        assert recorder.api_url == "http://localhost:7860"
        assert recorder.on_duration_change is None
        assert recorder._recording is False
        assert recorder._chunks == []
        assert recorder._total_bytes == 0

    def test_custom_api_url(self):
        """Test initialization with custom API URL."""
        recorder = AudioRecorder(api_url="http://custom:8080")

        assert recorder.api_url == "http://custom:8080"

    def test_custom_callback(self):
        """Test initialization with duration callback."""
        callback = MagicMock()
        recorder = AudioRecorder(on_duration_change=callback)

        assert recorder.on_duration_change is callback


class TestAudioRecorderProperties:
    """Tests for AudioRecorder properties."""

    def test_duration_zero_initially(self):
        """Test duration is 0 when no audio."""
        recorder = AudioRecorder()
        assert recorder.duration == 0.0

    def test_duration_str_format(self):
        """Test duration string format (MM:SS)."""
        recorder = AudioRecorder()
        recorder._total_bytes = 16000 * 2 * 65  # 65 seconds of audio
        assert recorder.duration_str == "01:05"

    def test_file_size_mb(self):
        """Test file size calculation."""
        recorder = AudioRecorder()
        recorder._total_bytes = 1024 * 1024  # 1 MB
        assert recorder.file_size_mb == 1.0

    def test_is_recording_property(self):
        """Test is_recording property."""
        recorder = AudioRecorder()
        assert recorder.is_recording is False

        recorder._recording = True
        assert recorder.is_recording is True


class TestAudioRecorderStartStop:
    """Tests for AudioRecorder start/stop functionality."""

    def test_start_recording(self):
        """Test starting a recording."""
        recorder = AudioRecorder()

        with (
            patch.object(recorder, "_start_upload_timer"),
            patch.object(recorder, "_start_status_timer"),
            patch.object(recorder, "_update_status_file"),
        ):
            result = recorder.start()

        assert result is True
        assert recorder._recording is True
        assert recorder._current_filename is not None
        assert recorder._current_filename.startswith("recording_")
        assert recorder._current_filename.endswith(".wav")

    def test_start_when_already_recording(self):
        """Test starting when already recording."""
        recorder = AudioRecorder()
        recorder._recording = True

        result = recorder.start()

        assert result is False

    def test_stop_when_not_recording(self):
        """Test stopping when not recording."""
        recorder = AudioRecorder()

        result = recorder.stop()

        assert result is None

    def test_stop_with_no_audio(self):
        """Test stopping when no audio was recorded."""
        recorder = AudioRecorder()
        recorder._recording = True
        recorder._chunks = []

        with patch.object(recorder, "_update_status_file"):
            result = recorder.stop()

        assert result is None


class TestAudioRecorderAddChunk:
    """Tests for adding audio chunks."""

    def test_add_chunk_when_recording(self):
        """Test adding chunk while recording."""
        recorder = AudioRecorder()
        recorder._recording = True

        audio_data = b"\x00" * 3200  # 100ms of audio

        recorder.add_chunk(audio_data)

        assert len(recorder._chunks) == 1
        assert recorder._total_bytes == 3200
        assert recorder._last_audio_time is not None

    def test_add_chunk_when_not_recording(self):
        """Test adding chunk when not recording (should be ignored)."""
        recorder = AudioRecorder()
        recorder._recording = False

        audio_data = b"\x00" * 3200

        recorder.add_chunk(audio_data)

        assert len(recorder._chunks) == 0
        assert recorder._total_bytes == 0

    def test_add_chunk_calls_duration_callback(self):
        """Test that adding chunk calls duration callback."""
        callback = MagicMock()
        recorder = AudioRecorder(on_duration_change=callback)
        recorder._recording = True

        audio_data = b"\x00" * 3200

        recorder.add_chunk(audio_data)

        callback.assert_called_once()
