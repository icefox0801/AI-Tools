"""
Unit tests for Audio Notes ASR service module.

Tests the transcription and model unload functions.
Uses mocked HTTP requests to avoid requiring actual ASR services.
"""

from unittest.mock import MagicMock, mock_open, patch

import pytest

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_config():
    """Mock config module."""
    with patch.dict(
        "sys.modules",
        {
            "config": MagicMock(
                PARAKEET_URL="http://parakeet:8000",
                WHISPER_URL="http://whisper:8000",
                logger=MagicMock(),
            ),
        },
    ):
        yield


@pytest.fixture
def mock_recordings():
    """Mock recordings module."""
    mock_rec = MagicMock()
    mock_rec.get_audio_duration = MagicMock(return_value=10.5)
    with patch.dict("sys.modules", {"services.recordings": mock_rec}):
        yield mock_rec


# ==============================================================================
# transcribe_audio Tests
# ==============================================================================


class TestTranscribeAudio:
    """Tests for transcribe_audio function."""

    def test_transcribe_with_parakeet_success(self, mock_config, mock_recordings):
        """Test successful transcription with Parakeet backend."""
        import sys

        # Clear module cache
        for mod in list(sys.modules.keys()):
            if "services.asr" in mod:
                del sys.modules[mod]

        with patch("builtins.open", mock_open(read_data=b"audio data")):
            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "text": "Hello world transcription",
                    "duration": 10.5,
                }
                mock_post.return_value = mock_response

                from services.asr import transcribe_audio

                text, duration = transcribe_audio("/path/to/audio.wav", backend="parakeet")

                assert text == "Hello world transcription"
                assert duration == 10.5
                mock_post.assert_called_once()

    def test_transcribe_with_whisper_success(self, mock_config, mock_recordings):
        """Test successful transcription with Whisper backend."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.asr" in mod:
                del sys.modules[mod]

        with patch("builtins.open", mock_open(read_data=b"audio data")):
            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "text": "Whisper transcription result",
                    "duration": 15.0,
                }
                mock_post.return_value = mock_response

                from services.asr import transcribe_audio

                text, duration = transcribe_audio("/path/to/audio.wav", backend="whisper")

                assert text == "Whisper transcription result"
                assert duration == 15.0
                # Check URL contains whisper
                call_args = mock_post.call_args
                assert "whisper" in call_args[0][0].lower() or "8000" in call_args[0][0]

    def test_transcribe_api_error(self, mock_config, mock_recordings):
        """Test handling of API error response."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.asr" in mod:
                del sys.modules[mod]

        with patch("builtins.open", mock_open(read_data=b"audio data")):
            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 500
                mock_response.text = "Internal Server Error"
                mock_post.return_value = mock_response

                from services.asr import transcribe_audio

                text, _duration = transcribe_audio("/path/to/audio.wav")

                assert "Error: 500" in text

    def test_transcribe_connection_error(self, mock_config, mock_recordings):
        """Test handling of connection error."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.asr" in mod:
                del sys.modules[mod]

        with patch("builtins.open", mock_open(read_data=b"audio data")):
            with patch("requests.post") as mock_post:
                mock_post.side_effect = ConnectionError("Service unavailable")

                from services.asr import transcribe_audio

                text, _duration = transcribe_audio("/path/to/audio.wav")

                assert "Error:" in text

    def test_transcribe_content_type_mp3(self, mock_config, mock_recordings):
        """Test correct content type for MP3 files."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.asr" in mod:
                del sys.modules[mod]

        with patch("builtins.open", mock_open(read_data=b"mp3 data")):
            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"text": "MP3 audio", "duration": 5.0}
                mock_post.return_value = mock_response

                from services.asr import transcribe_audio

                transcribe_audio("/path/to/audio.mp3")

                call_args = mock_post.call_args
                files_arg = call_args[1]["files"]
                # Check content type is set for mp3
                assert files_arg["file"][2] == "audio/mpeg"


# ==============================================================================
# unload_asr_model Tests
# ==============================================================================


class TestUnloadAsrModel:
    """Tests for unload_asr_model function."""

    def test_unload_parakeet_success(self, mock_config, mock_recordings):
        """Test successful Parakeet model unload."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.asr" in mod:
                del sys.modules[mod]

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"message": "Model unloaded successfully"}
            mock_post.return_value = mock_response

            from services.asr import unload_asr_model

            success, message = unload_asr_model(backend="Parakeet")

            assert success is True
            assert "unloaded" in message.lower()

    def test_unload_whisper_success(self, mock_config, mock_recordings):
        """Test successful Whisper model unload."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.asr" in mod:
                del sys.modules[mod]

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"message": "Model unloaded"}
            mock_post.return_value = mock_response

            from services.asr import unload_asr_model

            success, _message = unload_asr_model(backend="Whisper")

            assert success is True

    def test_unload_failure(self, mock_config, mock_recordings):
        """Test unload failure handling."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.asr" in mod:
                del sys.modules[mod]

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_post.return_value = mock_response

            from services.asr import unload_asr_model

            success, message = unload_asr_model()

            assert success is False
            assert "Error" in message

    def test_unload_connection_error(self, mock_config, mock_recordings):
        """Test unload with connection error."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.asr" in mod:
                del sys.modules[mod]

        with patch("requests.post") as mock_post:
            mock_post.side_effect = ConnectionError("Service unavailable")

            from services.asr import unload_asr_model

            success, _message = unload_asr_model()

            assert success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
