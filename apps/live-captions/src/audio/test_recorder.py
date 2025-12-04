"""Unit tests for audio recorder module."""

import io
import wave
from unittest.mock import MagicMock, patch

import pytest

from .recorder import (
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    CHANNELS,
    INITIAL_UPLOAD_DELAY,
    AUTO_UPLOAD_INTERVAL,
    AudioRecorder,
    get_recorder,
    set_recorder,
)


class TestAudioRecorderInit:
    """Tests for AudioRecorder initialization."""

    def test_default_api_url(self):
        """Test default API URL."""
        recorder = AudioRecorder()
        assert "localhost" in recorder.api_url or "7860" in recorder.api_url

    def test_custom_api_url(self):
        """Test custom API URL."""
        recorder = AudioRecorder(api_url="http://custom:8080")
        assert recorder.api_url == "http://custom:8080"

    def test_initial_state(self):
        """Test initial recorder state."""
        recorder = AudioRecorder()

        assert not recorder.is_recording
        assert recorder.duration == 0.0
        assert recorder.duration_str == "00:00"
        assert recorder.file_size_mb == 0.0

    def test_duration_callback(self):
        """Test duration change callback is stored."""
        callback = MagicMock()
        recorder = AudioRecorder(on_duration_change=callback)

        assert recorder.on_duration_change is callback


class TestAudioRecorderStart:
    """Tests for starting recording."""

    def test_start_returns_true(self):
        """Test start returns True on success."""
        recorder = AudioRecorder()

        result = recorder.start()

        assert result is True
        assert recorder.is_recording
        recorder.clear()

    def test_start_generates_filename(self):
        """Test start generates timestamped filename."""
        recorder = AudioRecorder()

        recorder.start()

        assert recorder._current_filename is not None
        assert recorder._current_filename.startswith("recording_")
        assert recorder._current_filename.endswith(".wav")
        recorder.clear()

    def test_start_twice_returns_false(self):
        """Test starting twice returns False."""
        recorder = AudioRecorder()
        recorder.start()

        result = recorder.start()

        assert result is False
        recorder.clear()

    def test_start_resets_state(self):
        """Test start resets recording state."""
        recorder = AudioRecorder()
        recorder._total_bytes = 1000
        recorder._chunks = [b"test"]

        recorder.start()

        assert recorder._total_bytes == 0
        assert recorder._chunks == []
        recorder.clear()


class TestAudioRecorderAddChunk:
    """Tests for adding audio chunks."""

    def test_add_chunk_when_recording(self):
        """Test adding chunk while recording."""
        recorder = AudioRecorder()
        recorder.start()

        audio_data = b"\x00" * 1600  # 100ms at 16kHz
        recorder.add_chunk(audio_data)

        assert recorder._total_bytes == 1600
        assert len(recorder._chunks) == 1
        recorder.clear()

    def test_add_chunk_when_not_recording(self):
        """Test adding chunk when not recording is ignored."""
        recorder = AudioRecorder()

        audio_data = b"\x00" * 1600
        recorder.add_chunk(audio_data)

        assert recorder._total_bytes == 0
        assert len(recorder._chunks) == 0

    def test_add_multiple_chunks(self):
        """Test adding multiple chunks."""
        recorder = AudioRecorder()
        recorder.start()

        chunk = b"\x00" * 1600
        recorder.add_chunk(chunk)
        recorder.add_chunk(chunk)
        recorder.add_chunk(chunk)

        assert recorder._total_bytes == 4800
        assert len(recorder._chunks) == 3
        recorder.clear()

    def test_add_chunk_calls_duration_callback(self):
        """Test duration callback is called when adding chunk."""
        callback = MagicMock()
        recorder = AudioRecorder(on_duration_change=callback)
        recorder.start()

        recorder.add_chunk(b"\x00" * 1600)

        callback.assert_called()
        recorder.clear()


class TestAudioRecorderDuration:
    """Tests for duration calculation."""

    def test_duration_calculation(self):
        """Test duration is calculated correctly."""
        recorder = AudioRecorder()
        recorder.start()

        # Add 1 second of audio (16000 samples * 2 bytes * 1 channel)
        one_second_bytes = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS
        recorder.add_chunk(b"\x00" * one_second_bytes)

        assert recorder.duration == pytest.approx(1.0)
        recorder.clear()

    def test_duration_str_format(self):
        """Test duration string formatting."""
        recorder = AudioRecorder()
        recorder.start()

        # Add 65 seconds of audio
        seconds = 65
        bytes_needed = seconds * SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS
        recorder.add_chunk(b"\x00" * bytes_needed)

        assert recorder.duration_str == "01:05"
        recorder.clear()


class TestAudioRecorderStop:
    """Tests for stopping recording."""

    @patch.object(AudioRecorder, "_upload_to_api", return_value=True)
    def test_stop_uploads_audio(self, mock_upload):
        """Test stop uploads the recorded audio."""
        recorder = AudioRecorder()
        recorder.start()
        recorder.add_chunk(b"\x00" * 16000)

        filename = recorder.stop()

        assert filename is not None
        mock_upload.assert_called()

    @patch.object(AudioRecorder, "_upload_to_api", return_value=False)
    def test_stop_returns_none_on_upload_failure(self, mock_upload):
        """Test stop returns None when upload fails."""
        recorder = AudioRecorder()
        recorder.start()
        recorder.add_chunk(b"\x00" * 16000)

        filename = recorder.stop()

        assert filename is None

    def test_stop_when_not_recording(self):
        """Test stop when not recording returns None."""
        recorder = AudioRecorder()

        result = recorder.stop()

        assert result is None

    def test_stop_with_no_chunks(self):
        """Test stop with no audio chunks returns None."""
        recorder = AudioRecorder()
        recorder.start()

        result = recorder.stop()

        assert result is None


class TestAudioRecorderClear:
    """Tests for clearing recording."""

    def test_clear_resets_state(self):
        """Test clear resets all state."""
        recorder = AudioRecorder()
        recorder.start()
        recorder.add_chunk(b"\x00" * 1600)

        recorder.clear()

        assert not recorder.is_recording
        assert recorder._total_bytes == 0
        assert recorder._chunks == []
        assert recorder._current_filename is None


class TestCreateWavBytes:
    """Tests for WAV file creation."""

    def test_creates_valid_wav(self):
        """Test that _create_wav_bytes creates valid WAV data."""
        recorder = AudioRecorder()

        audio_data = b"\x00\x00" * 1600  # 100ms of silence
        wav_bytes = recorder._create_wav_bytes(audio_data)

        # Verify it's valid WAV
        wav_io = io.BytesIO(wav_bytes)
        with wave.open(wav_io, "rb") as wav:
            assert wav.getnchannels() == CHANNELS
            assert wav.getsampwidth() == SAMPLE_WIDTH
            assert wav.getframerate() == SAMPLE_RATE

    def test_wav_contains_audio(self):
        """Test WAV file contains the audio data."""
        recorder = AudioRecorder()

        audio_data = b"\x01\x02" * 100
        wav_bytes = recorder._create_wav_bytes(audio_data)

        wav_io = io.BytesIO(wav_bytes)
        with wave.open(wav_io, "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            assert audio_data in wav_bytes


class TestGlobalRecorder:
    """Tests for global recorder functions."""

    def test_get_set_recorder(self):
        """Test getting and setting global recorder."""
        original = get_recorder()

        recorder = AudioRecorder()
        set_recorder(recorder)

        assert get_recorder() is recorder

        # Restore
        set_recorder(original)

    def test_set_recorder_none(self):
        """Test setting global recorder to None."""
        set_recorder(AudioRecorder())
        set_recorder(None)

        assert get_recorder() is None


class TestConstants:
    """Tests for module constants."""

    def test_sample_rate(self):
        """Test sample rate constant."""
        assert SAMPLE_RATE == 16000

    def test_sample_width(self):
        """Test sample width constant."""
        assert SAMPLE_WIDTH == 2

    def test_channels(self):
        """Test channels constant."""
        assert CHANNELS == 1

    def test_initial_upload_delay(self):
        """Test initial upload delay."""
        assert INITIAL_UPLOAD_DELAY == 3

    def test_auto_upload_interval(self):
        """Test auto upload interval."""
        assert AUTO_UPLOAD_INTERVAL == 60
