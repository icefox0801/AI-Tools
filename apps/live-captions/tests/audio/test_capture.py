"""
Unit tests for audio capture module.

Tests the audio capture classes including:
- AudioCapture abstract base class
- MicrophoneCapture for microphone input
- SystemAudioCapture for WASAPI loopback
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestAudioCaptureBase:
    """Tests for AudioCapture base class."""

    def test_is_abstract(self):
        """Test that AudioCapture cannot be instantiated directly."""
        from src.audio.capture import AudioCapture

        with pytest.raises(TypeError):
            AudioCapture(callback=lambda x: None)

    def test_subclass_must_implement_start(self):
        """Test that subclasses must implement start method."""
        from src.audio.capture import AudioCapture

        class IncompleteCapture(AudioCapture):
            @property
            def source_name(self) -> str:
                return "Test"

            def stop(self):
                pass

        with pytest.raises(TypeError):
            IncompleteCapture(callback=lambda x: None)

    def test_subclass_must_implement_stop(self):
        """Test that subclasses must implement stop method."""
        from src.audio.capture import AudioCapture

        class IncompleteCapture(AudioCapture):
            @property
            def source_name(self) -> str:
                return "Test"

            def start(self) -> bool:
                return True

        with pytest.raises(TypeError):
            IncompleteCapture(callback=lambda x: None)

    def test_subclass_must_implement_source_name(self):
        """Test that subclasses must implement source_name property."""
        from src.audio.capture import AudioCapture

        class IncompleteCapture(AudioCapture):
            def start(self) -> bool:
                return True

            def stop(self):
                pass

        with pytest.raises(TypeError):
            IncompleteCapture(callback=lambda x: None)


class TestMicrophoneCaptureInit:
    """Tests for MicrophoneCapture initialization."""

    def test_default_init(self):
        """Test default initialization."""
        from src.audio.capture import MicrophoneCapture

        callback = MagicMock()
        capture = MicrophoneCapture(callback)

        assert capture.callback is callback
        assert capture.device_index is None
        assert capture.running is False
        assert capture.pyaudio_instance is None
        assert capture.stream is None
        assert capture.capture_channels == 1

    def test_custom_device_index(self):
        """Test initialization with custom device index."""
        from src.audio.capture import MicrophoneCapture

        capture = MicrophoneCapture(callback=lambda x: None, device_index=2)

        assert capture.device_index == 2

    def test_source_name_default(self):
        """Test default source name."""
        from src.audio.capture import MicrophoneCapture

        capture = MicrophoneCapture(callback=lambda x: None)

        assert "Microphone" in capture.source_name
        assert "🎤" in capture.source_name


class TestMicrophoneCaptureStart:
    """Tests for MicrophoneCapture start method."""

    def test_start_success(self):
        """Test successful start."""
        mock_pyaudio = MagicMock()
        mock_instance = MagicMock()
        mock_stream = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_instance
        mock_pyaudio.paInt16 = 8
        mock_instance.get_default_input_device_info.return_value = {
            "name": "Test Mic",
            "defaultSampleRate": 48000,
            "maxInputChannels": 1,
        }
        mock_instance.open.return_value = mock_stream

        with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
            from src.audio.capture import MicrophoneCapture

            capture = MicrophoneCapture(callback=lambda x: None)
            result = capture.start()

            assert result is True
            assert capture.running is True
            mock_stream.start_stream.assert_called_once()

    def test_start_with_specific_device(self):
        """Test start with specific device index."""
        mock_pyaudio = MagicMock()
        mock_instance = MagicMock()
        mock_stream = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_instance
        mock_pyaudio.paInt16 = 8
        mock_instance.get_device_info_by_index.return_value = {
            "name": "Specific Mic",
            "defaultSampleRate": 44100,
            "maxInputChannels": 2,
        }
        mock_instance.open.return_value = mock_stream

        with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
            from src.audio.capture import MicrophoneCapture

            capture = MicrophoneCapture(callback=lambda x: None, device_index=3)
            result = capture.start()

            assert result is True
            mock_instance.get_device_info_by_index.assert_called_with(3)

    def test_start_stereo_mix_uses_stereo(self):
        """Test that stereo mix devices use stereo capture."""
        mock_pyaudio = MagicMock()
        mock_instance = MagicMock()
        mock_stream = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_instance
        mock_pyaudio.paInt16 = 8
        mock_instance.get_default_input_device_info.return_value = {
            "name": "Stereo Mix",
            "defaultSampleRate": 48000,
            "maxInputChannels": 2,
        }
        mock_instance.open.return_value = mock_stream

        with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
            from src.audio.capture import MicrophoneCapture

            capture = MicrophoneCapture(callback=lambda x: None)
            capture.start()

            assert capture.capture_channels == 2

    def test_start_failure(self):
        """Test start failure handling."""
        mock_pyaudio = MagicMock()
        mock_pyaudio.PyAudio.side_effect = Exception("No audio device")

        with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
            from src.audio.capture import MicrophoneCapture

            capture = MicrophoneCapture(callback=lambda x: None)
            result = capture.start()

            assert result is False
            assert capture.running is False


class TestMicrophoneCaptureCallback:
    """Tests for MicrophoneCapture audio callback."""

    def test_callback_processes_mono_audio(self):
        """Test callback processes mono audio correctly."""
        mock_pyaudio = MagicMock()
        mock_pyaudio.paContinue = 0
        mock_pyaudio.paComplete = 1
        received_data = []

        def capture_callback(data):
            received_data.append(data)

        with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
            from src.audio.capture import MicrophoneCapture

            capture = MicrophoneCapture(callback=capture_callback)
            capture.running = True
            capture.capture_rate = 16000
            capture.capture_channels = 1

            # Simulate audio data
            audio_data = np.zeros(1600, dtype=np.int16).tobytes()
            result = capture._audio_callback(audio_data, 1600, {}, 0)

            assert result[1] == mock_pyaudio.paContinue
            assert len(received_data) > 0

    def test_callback_stops_when_not_running(self):
        """Test callback returns complete when not running."""
        mock_pyaudio = MagicMock()
        mock_pyaudio.paContinue = 0
        mock_pyaudio.paComplete = 1

        with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
            from src.audio.capture import MicrophoneCapture

            capture = MicrophoneCapture(callback=lambda x: None)
            capture.running = False

            result = capture._audio_callback(b"\x00" * 100, 50, {}, 0)

            assert result[1] == mock_pyaudio.paComplete


class TestMicrophoneCaptureStop:
    """Tests for MicrophoneCapture stop method."""

    def test_stop_cleans_up(self):
        """Test stop cleans up resources."""
        from src.audio.capture import MicrophoneCapture

        mock_stream = MagicMock()
        mock_pyaudio_instance = MagicMock()

        capture = MicrophoneCapture(callback=lambda x: None)
        capture.running = True
        capture.stream = mock_stream
        capture.pyaudio_instance = mock_pyaudio_instance

        capture.stop()

        assert capture.running is False
        assert capture.stream is None
        assert capture.pyaudio_instance is None
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()
        mock_pyaudio_instance.terminate.assert_called_once()

    def test_stop_handles_errors(self):
        """Test stop handles cleanup errors gracefully."""
        from src.audio.capture import MicrophoneCapture

        mock_stream = MagicMock()
        mock_stream.stop_stream.side_effect = Exception("Error")

        capture = MicrophoneCapture(callback=lambda x: None)
        capture.running = True
        capture.stream = mock_stream

        # Should not raise
        capture.stop()

        assert capture.running is False


class TestSystemAudioCaptureInit:
    """Tests for SystemAudioCapture initialization."""

    def test_default_init(self):
        """Test default initialization."""
        from src.audio.capture import SystemAudioCapture

        callback = MagicMock()
        capture = SystemAudioCapture(callback)

        assert capture.callback is callback
        assert capture.device_index is None
        assert capture.running is False
        assert capture.capture_rate == 48000

    def test_source_name_default(self):
        """Test default source name."""
        from src.audio.capture import SystemAudioCapture

        capture = SystemAudioCapture(callback=lambda x: None)

        assert "System Audio" in capture.source_name
        assert "🔊" in capture.source_name


class TestSystemAudioCaptureStart:
    """Tests for SystemAudioCapture start method."""

    def test_start_with_specified_device(self):
        """Test start with specific loopback device."""
        mock_pyaudio = MagicMock()
        mock_instance = MagicMock()
        mock_stream = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_instance
        mock_pyaudio.paInt16 = 8
        mock_pyaudio.paWASAPI = 1
        mock_instance.get_device_info_by_index.return_value = {
            "name": "Speakers [Loopback]",
            "isLoopbackDevice": True,
            "defaultSampleRate": 48000,
            "maxInputChannels": 2,
        }
        mock_instance.open.return_value = mock_stream

        with patch.dict("sys.modules", {"pyaudiowpatch": mock_pyaudio}):
            from src.audio.capture import SystemAudioCapture

            capture = SystemAudioCapture(callback=lambda x: None, device_index=5)
            result = capture.start()

            assert result is True
            assert capture.running is True
            mock_stream.start_stream.assert_called_once()

    def test_start_finds_default_loopback(self):
        """Test start finds default loopback device."""
        mock_pyaudio = MagicMock()
        mock_instance = MagicMock()
        mock_stream = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_instance
        mock_pyaudio.paInt16 = 8
        mock_pyaudio.paWASAPI = 1
        mock_instance.get_host_api_info_by_type.return_value = {"defaultOutputDevice": 0}
        mock_instance.get_device_count.return_value = 2
        mock_instance.get_device_info_by_index.side_effect = [
            {"name": "Speakers"},  # default output
            {"name": "Other Device", "isLoopbackDevice": False},
            {
                "name": "Speakers [Loopback]",
                "isLoopbackDevice": True,
                "defaultSampleRate": 48000,
                "maxInputChannels": 2,
            },
        ]
        mock_instance.open.return_value = mock_stream

        with patch.dict("sys.modules", {"pyaudiowpatch": mock_pyaudio}):
            from src.audio.capture import SystemAudioCapture

            capture = SystemAudioCapture(callback=lambda x: None)
            result = capture.start()

            assert result is True

    def test_start_no_loopback_device(self):
        """Test start fails when no loopback device found."""
        mock_pyaudio = MagicMock()
        mock_instance = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_instance
        mock_pyaudio.paWASAPI = 1
        mock_instance.get_host_api_info_by_type.return_value = {"defaultOutputDevice": 0}
        mock_instance.get_device_count.return_value = 1
        mock_instance.get_device_info_by_index.side_effect = [
            {"name": "Speakers"},  # default output
            {"name": "Not Loopback", "isLoopbackDevice": False},
        ]

        with patch.dict("sys.modules", {"pyaudiowpatch": mock_pyaudio}):
            from src.audio.capture import SystemAudioCapture

            capture = SystemAudioCapture(callback=lambda x: None)
            result = capture.start()

            assert result is False

    def test_start_import_error(self):
        """Test start handles missing pyaudiowpatch."""
        # Create a mock that raises ImportError on import
        import sys

        original_modules = sys.modules.copy()

        with patch.dict("sys.modules", {"pyaudiowpatch": None}):

            def mock_import(name, *args, **kwargs):
                if name == "pyaudiowpatch":
                    raise ImportError("No module named 'pyaudiowpatch'")
                return original_modules.get(name)

            from src.audio.capture import SystemAudioCapture

            capture = SystemAudioCapture(callback=lambda x: None)

            with patch("builtins.__import__", side_effect=mock_import):
                result = capture.start()

            assert result is False


class TestSystemAudioCaptureCallback:
    """Tests for SystemAudioCapture audio callback."""

    def test_callback_converts_stereo_to_mono(self):
        """Test callback converts stereo to mono."""
        mock_pyaudio = MagicMock()
        mock_pyaudio.paContinue = 0
        mock_pyaudio.paComplete = 1
        received_data = []

        def capture_callback(data):
            received_data.append(data)

        with patch.dict("sys.modules", {"pyaudiowpatch": mock_pyaudio}):
            from src.audio.capture import SystemAudioCapture

            capture = SystemAudioCapture(callback=capture_callback)
            capture.running = True
            capture.capture_rate = 48000

            # Simulate stereo audio data (interleaved L/R samples)
            stereo_data = np.zeros(3200, dtype=np.int16).tobytes()  # 1600 stereo samples
            result = capture._audio_callback(stereo_data, 1600, {}, 0)

            assert result[1] == mock_pyaudio.paContinue
            assert len(received_data) > 0

    def test_callback_stops_when_not_running(self):
        """Test callback returns complete when not running."""
        mock_pyaudio = MagicMock()
        mock_pyaudio.paContinue = 0
        mock_pyaudio.paComplete = 1

        with patch.dict("sys.modules", {"pyaudiowpatch": mock_pyaudio}):
            from src.audio.capture import SystemAudioCapture

            capture = SystemAudioCapture(callback=lambda x: None)
            capture.running = False

            result = capture._audio_callback(b"\x00" * 100, 50, {}, 0)

            assert result[1] == mock_pyaudio.paComplete


class TestSystemAudioCaptureStop:
    """Tests for SystemAudioCapture stop method."""

    def test_stop_cleans_up(self):
        """Test stop cleans up resources."""
        from src.audio.capture import SystemAudioCapture

        mock_stream = MagicMock()
        mock_pyaudio_instance = MagicMock()

        capture = SystemAudioCapture(callback=lambda x: None)
        capture.running = True
        capture.stream = mock_stream
        capture.pyaudio_instance = mock_pyaudio_instance

        capture.stop()

        assert capture.running is False
        assert capture.stream is None
        assert capture.pyaudio_instance is None
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()
        mock_pyaudio_instance.terminate.assert_called_once()
