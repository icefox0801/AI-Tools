"""
Unit tests for Audio Notes recordings service module.

Tests recording listing and audio duration functions.
"""

import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def temp_recordings_dir():
    """Create a temporary recordings directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config(temp_recordings_dir):
    """Mock config module with temp directory."""
    mock_cfg = MagicMock()
    mock_cfg.RECORDINGS_DIR = temp_recordings_dir
    with patch.dict("sys.modules", {"config": mock_cfg}):
        yield mock_cfg


def create_test_wav(path: Path, duration_sec: float = 1.0, sample_rate: int = 16000):
    """Create a test WAV file with specified duration."""
    num_frames = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)


# ==============================================================================
# get_audio_duration Tests
# ==============================================================================


class TestGetAudioDuration:
    """Tests for get_audio_duration function."""

    def test_wav_duration_correct(self, temp_recordings_dir, mock_config):
        """Test correct duration for WAV file."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.recordings" in mod:
                del sys.modules[mod]

        wav_path = temp_recordings_dir / "test.wav"
        create_test_wav(wav_path, duration_sec=5.0)

        from services.recordings import get_audio_duration

        duration = get_audio_duration(str(wav_path))
        assert abs(duration - 5.0) < 0.1

    def test_wav_duration_short_file(self, temp_recordings_dir, mock_config):
        """Test duration for short WAV file."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.recordings" in mod:
                del sys.modules[mod]

        wav_path = temp_recordings_dir / "short.wav"
        create_test_wav(wav_path, duration_sec=0.5)

        from services.recordings import get_audio_duration

        duration = get_audio_duration(str(wav_path))
        assert abs(duration - 0.5) < 0.1

    def test_invalid_file_fallback(self, temp_recordings_dir, mock_config):
        """Test fallback for non-WAV file."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.recordings" in mod:
                del sys.modules[mod]

        # Create a file that's not a valid WAV
        invalid_path = temp_recordings_dir / "invalid.wav"
        invalid_path.write_bytes(b"not a wav file content here")

        from services.recordings import get_audio_duration

        duration = get_audio_duration(str(invalid_path))
        # Should fall back to size-based estimation
        assert duration >= 0

    def test_nonexistent_file(self, temp_recordings_dir, mock_config):
        """Test handling of non-existent file."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.recordings" in mod:
                del sys.modules[mod]

        from services.recordings import get_audio_duration

        duration = get_audio_duration("/nonexistent/path.wav")
        assert duration == 0.0


# ==============================================================================
# list_recordings Tests
# ==============================================================================


class TestListRecordings:
    """Tests for list_recordings function."""

    def test_empty_directory(self, temp_recordings_dir, mock_config):
        """Test listing empty recordings directory."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.recordings" in mod:
                del sys.modules[mod]

        from services.recordings import list_recordings

        recordings = list_recordings()
        assert recordings == []

    def test_list_wav_files(self, temp_recordings_dir, mock_config):
        """Test listing WAV files."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.recordings" in mod:
                del sys.modules[mod]

        # Create test recordings
        create_test_wav(temp_recordings_dir / "recording1.wav", duration_sec=10.0)
        create_test_wav(temp_recordings_dir / "recording2.wav", duration_sec=20.0)

        from services.recordings import list_recordings

        recordings = list_recordings()

        assert len(recordings) == 2
        names = [r["name"] for r in recordings]
        assert "recording1.wav" in names
        assert "recording2.wav" in names

    def test_recordings_have_required_fields(self, temp_recordings_dir, mock_config):
        """Test that recordings have all required fields."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.recordings" in mod:
                del sys.modules[mod]

        create_test_wav(temp_recordings_dir / "test.wav", duration_sec=5.0)

        from services.recordings import list_recordings

        recordings = list_recordings()

        assert len(recordings) == 1
        rec = recordings[0]

        # Check required fields
        assert "path" in rec
        assert "name" in rec
        assert "size_mb" in rec
        assert "duration" in rec
        assert "duration_str" in rec
        assert "date" in rec
        assert "timestamp" in rec
        assert "has_transcript" in rec

    def test_duration_str_format(self, temp_recordings_dir, mock_config):
        """Test duration string format MM:SS."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.recordings" in mod:
                del sys.modules[mod]

        create_test_wav(temp_recordings_dir / "test.wav", duration_sec=65.0)  # 1:05

        from services.recordings import list_recordings

        recordings = list_recordings()

        assert len(recordings) == 1
        duration_str = recordings[0]["duration_str"]
        assert duration_str == "01:05"

    def test_has_transcript_false(self, temp_recordings_dir, mock_config):
        """Test has_transcript is False when no transcript."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.recordings" in mod:
                del sys.modules[mod]

        create_test_wav(temp_recordings_dir / "test.wav")

        from services.recordings import list_recordings

        recordings = list_recordings()

        assert recordings[0]["has_transcript"] is False

    def test_has_transcript_true(self, temp_recordings_dir, mock_config):
        """Test has_transcript is True when transcript exists."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.recordings" in mod:
                del sys.modules[mod]

        create_test_wav(temp_recordings_dir / "test.wav")
        (temp_recordings_dir / "test.txt").write_text("Transcript content")

        from services.recordings import list_recordings

        recordings = list_recordings()

        assert recordings[0]["has_transcript"] is True

    def test_sorted_by_timestamp_descending(self, temp_recordings_dir, mock_config):
        """Test recordings are sorted newest first."""
        import sys
        import time

        for mod in list(sys.modules.keys()):
            if "services.recordings" in mod:
                del sys.modules[mod]

        # Create files with different timestamps
        old_file = temp_recordings_dir / "old.wav"
        create_test_wav(old_file)
        time.sleep(0.1)

        new_file = temp_recordings_dir / "new.wav"
        create_test_wav(new_file)

        from services.recordings import list_recordings

        recordings = list_recordings()

        assert len(recordings) == 2
        assert recordings[0]["name"] == "new.wav"
        assert recordings[1]["name"] == "old.wav"

    def test_multiple_extensions(self, temp_recordings_dir, mock_config):
        """Test listing files with different extensions."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.recordings" in mod:
                del sys.modules[mod]

        # Create files with different extensions
        create_test_wav(temp_recordings_dir / "audio.wav")
        (temp_recordings_dir / "audio.mp3").write_bytes(b"fake mp3")
        (temp_recordings_dir / "audio.m4a").write_bytes(b"fake m4a")
        (temp_recordings_dir / "audio.txt").write_text("not audio")  # Should be ignored

        from services.recordings import list_recordings

        recordings = list_recordings()

        # Should find wav, mp3, m4a but not txt
        assert len(recordings) == 3
        names = [r["name"] for r in recordings]
        assert "audio.wav" in names
        assert "audio.mp3" in names
        assert "audio.m4a" in names
        assert "audio.txt" not in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
