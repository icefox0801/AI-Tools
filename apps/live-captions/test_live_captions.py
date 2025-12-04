"""
Unit tests for live_captions.py main application module.

Tests the LiveCaptions class and helper functions including:
- Initialization and configuration
- Audio callback handling
- ASR connection handling
- Transcript updates
- Application lifecycle
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


# Mock tkinter and UI before importing the module
@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock heavy dependencies before imports."""
    mock_tk = MagicMock()
    mock_font = MagicMock()
    mock_ctypes = MagicMock()
    
    # Mock root window
    mock_root = MagicMock()
    mock_root.winfo_screenwidth.return_value = 1920
    mock_root.winfo_screenheight.return_value = 1080
    mock_tk.Tk.return_value = mock_root
    mock_font.Font.return_value = MagicMock()
    
    with patch.dict(sys.modules, {
        "tkinter": mock_tk,
        "tkinter.font": mock_font,
    }):
        yield


class TestLiveCaptionsInit:
    """Tests for LiveCaptions initialization."""

    def test_default_init(self, mock_dependencies):
        """Test default initialization."""
        with patch("live_captions.CaptionWindow") as mock_window, \
             patch("live_captions.get_backend_config") as mock_config, \
             patch("live_captions.get_display_info") as mock_display, \
             patch("live_captions.AudioRecorder") as mock_recorder, \
             patch("live_captions.set_recorder"):
            
            mock_config.return_value = {"port": 8000, "chunk_ms": 200}
            mock_display.return_value = "Test Model"
            mock_window.return_value = MagicMock()
            mock_recorder.return_value = MagicMock()
            
            from live_captions import LiveCaptions
            
            app = LiveCaptions()
            
            assert app.running is True
            assert app.enable_transcription is True
            assert app.use_system_audio is False
            assert app.device_index is None
            assert app.audio_capture is None

    def test_custom_backend(self, mock_dependencies):
        """Test initialization with custom backend."""
        with patch("live_captions.CaptionWindow") as mock_window, \
             patch("live_captions.get_backend_config") as mock_config, \
             patch("live_captions.get_display_info") as mock_display, \
             patch("live_captions.AudioRecorder") as mock_recorder, \
             patch("live_captions.set_recorder"):
            
            mock_config.return_value = {"port": 9000, "chunk_ms": 100}
            mock_display.return_value = "Parakeet"
            mock_window.return_value = MagicMock()
            mock_recorder.return_value = MagicMock()
            
            from live_captions import LiveCaptions
            
            app = LiveCaptions(backend="parakeet", port=9000)
            
            assert app.backend == "parakeet"
            assert app.port == 9000

    def test_system_audio_mode(self, mock_dependencies):
        """Test initialization with system audio mode."""
        with patch("live_captions.CaptionWindow") as mock_window, \
             patch("live_captions.get_backend_config") as mock_config, \
             patch("live_captions.get_display_info") as mock_display, \
             patch("live_captions.AudioRecorder") as mock_recorder, \
             patch("live_captions.set_recorder"):
            
            mock_config.return_value = {"port": 8000, "chunk_ms": 200}
            mock_display.return_value = "Test"
            mock_window.return_value = MagicMock()
            mock_recorder.return_value = MagicMock()
            
            from live_captions import LiveCaptions
            
            app = LiveCaptions(use_system_audio=True, device_index=5)
            
            assert app.use_system_audio is True
            assert app.device_index == 5

    def test_recording_disabled(self, mock_dependencies):
        """Test initialization with recording disabled."""
        with patch("live_captions.CaptionWindow") as mock_window, \
             patch("live_captions.get_backend_config") as mock_config, \
             patch("live_captions.get_display_info") as mock_display:
            
            mock_config.return_value = {"port": 8000, "chunk_ms": 200}
            mock_display.return_value = "Test"
            mock_window.return_value = MagicMock()
            
            from live_captions import LiveCaptions
            
            app = LiveCaptions(enable_recording=False)
            
            assert app.recorder is None

    def test_transcription_disabled(self, mock_dependencies):
        """Test initialization with transcription disabled."""
        with patch("live_captions.CaptionWindow") as mock_window, \
             patch("live_captions.get_backend_config") as mock_config, \
             patch("live_captions.get_display_info") as mock_display, \
             patch("live_captions.AudioRecorder") as mock_recorder, \
             patch("live_captions.set_recorder"):
            
            mock_config.return_value = {"port": 8000, "chunk_ms": 200}
            mock_display.return_value = "Test"
            mock_window.return_value = MagicMock()
            mock_recorder.return_value = MagicMock()
            
            from live_captions import LiveCaptions
            
            app = LiveCaptions(enable_transcription=False)
            
            assert app.enable_transcription is False


class TestLiveCaptionsAudioCallback:
    """Tests for audio callback handling."""

    def test_on_audio_data_records(self, mock_dependencies):
        """Test that audio data is recorded."""
        with patch("live_captions.CaptionWindow") as mock_window, \
             patch("live_captions.get_backend_config") as mock_config, \
             patch("live_captions.get_display_info") as mock_display, \
             patch("live_captions.AudioRecorder") as mock_recorder_class, \
             patch("live_captions.set_recorder"):
            
            mock_config.return_value = {"port": 8000, "chunk_ms": 200}
            mock_display.return_value = "Test"
            mock_window.return_value = MagicMock()
            
            mock_recorder = MagicMock()
            mock_recorder.is_recording = True
            mock_recorder_class.return_value = mock_recorder
            
            from live_captions import LiveCaptions
            
            app = LiveCaptions()
            app._on_audio_data(b"\x00\x00\x01\x00")
            
            mock_recorder.add_chunk.assert_called_once_with(b"\x00\x00\x01\x00")

    def test_on_audio_data_queues_to_asr(self, mock_dependencies):
        """Test that audio data is queued to ASR client."""
        with patch("live_captions.CaptionWindow") as mock_window, \
             patch("live_captions.get_backend_config") as mock_config, \
             patch("live_captions.get_display_info") as mock_display, \
             patch("live_captions.AudioRecorder") as mock_recorder_class, \
             patch("live_captions.set_recorder"):
            
            mock_config.return_value = {"port": 8000, "chunk_ms": 200}
            mock_display.return_value = "Test"
            mock_window.return_value = MagicMock()
            
            mock_recorder = MagicMock()
            mock_recorder.is_recording = False
            mock_recorder_class.return_value = mock_recorder
            
            from live_captions import LiveCaptions
            
            app = LiveCaptions()
            app.asr_client = MagicMock()
            app._on_audio_data(b"\x00\x00\x01\x00")
            
            app.asr_client.queue_audio.assert_called_once_with(b"\x00\x00\x01\x00")

    def test_on_audio_data_debug_save(self, mock_dependencies):
        """Test that audio data is saved for debug."""
        with patch("live_captions.CaptionWindow") as mock_window, \
             patch("live_captions.get_backend_config") as mock_config, \
             patch("live_captions.get_display_info") as mock_display, \
             patch("live_captions.AudioRecorder") as mock_recorder_class, \
             patch("live_captions.set_recorder"):
            
            mock_config.return_value = {"port": 8000, "chunk_ms": 200}
            mock_display.return_value = "Test"
            mock_window.return_value = MagicMock()
            
            mock_recorder = MagicMock()
            mock_recorder.is_recording = False
            mock_recorder_class.return_value = mock_recorder
            
            from live_captions import LiveCaptions
            
            app = LiveCaptions(debug_save_audio=True)
            app._on_audio_data(b"\x00\x00\x01\x00")
            
            assert b"\x00\x00\x01\x00" in app.debug_audio_chunks


class TestLiveCaptionsASRCallback:
    """Tests for ASR callback handling."""

    def test_on_asr_connected_true(self, mock_dependencies):
        """Test ASR connected callback."""
        with patch("live_captions.CaptionWindow") as mock_window_class, \
             patch("live_captions.get_backend_config") as mock_config, \
             patch("live_captions.get_display_info") as mock_display, \
             patch("live_captions.AudioRecorder") as mock_recorder_class, \
             patch("live_captions.set_recorder"):
            
            mock_config.return_value = {"port": 8000, "chunk_ms": 200}
            mock_display.return_value = "Test"
            
            mock_window = MagicMock()
            mock_window_class.return_value = mock_window
            
            mock_recorder = MagicMock()
            mock_recorder.is_recording = False
            mock_recorder_class.return_value = mock_recorder
            
            from live_captions import LiveCaptions
            
            app = LiveCaptions()
            app._on_asr_connected(True)
            
            mock_window.set_connection_status.assert_called_with(True)
            mock_window.set_message.assert_called()

    def test_on_asr_connected_false(self, mock_dependencies):
        """Test ASR disconnected callback."""
        with patch("live_captions.CaptionWindow") as mock_window_class, \
             patch("live_captions.get_backend_config") as mock_config, \
             patch("live_captions.get_display_info") as mock_display, \
             patch("live_captions.AudioRecorder") as mock_recorder_class, \
             patch("live_captions.set_recorder"):
            
            mock_config.return_value = {"port": 8000, "chunk_ms": 200}
            mock_display.return_value = "Test"
            
            mock_window = MagicMock()
            mock_window_class.return_value = mock_window
            
            mock_recorder = MagicMock()
            mock_recorder.is_recording = False
            mock_recorder_class.return_value = mock_recorder
            
            from live_captions import LiveCaptions
            
            app = LiveCaptions()
            app._on_asr_connected(False)
            
            mock_window.set_connection_status.assert_called_with(False)

    def test_on_asr_transcript(self, mock_dependencies):
        """Test ASR transcript callback."""
        with patch("live_captions.CaptionWindow") as mock_window_class, \
             patch("live_captions.get_backend_config") as mock_config, \
             patch("live_captions.get_display_info") as mock_display, \
             patch("live_captions.AudioRecorder") as mock_recorder_class, \
             patch("live_captions.set_recorder"):
            
            mock_config.return_value = {"port": 8000, "chunk_ms": 200}
            mock_display.return_value = "Test"
            
            mock_window = MagicMock()
            mock_window_class.return_value = mock_window
            
            mock_recorder = MagicMock()
            mock_recorder.is_recording = False
            mock_recorder_class.return_value = mock_recorder
            
            from live_captions import LiveCaptions
            
            app = LiveCaptions()
            app._on_asr_transcript("seg1", "Hello world")
            
            # Transcript manager should be updated
            assert app.transcript.get_text() == "Hello world"


class TestLiveCaptionsAudioCapture:
    """Tests for audio capture functionality."""

    def test_start_microphone_capture(self, mock_dependencies):
        """Test starting microphone capture."""
        with patch("live_captions.CaptionWindow") as mock_window_class, \
             patch("live_captions.get_backend_config") as mock_config, \
             patch("live_captions.get_display_info") as mock_display, \
             patch("live_captions.AudioRecorder") as mock_recorder_class, \
             patch("live_captions.set_recorder"), \
             patch("live_captions.MicrophoneCapture") as mock_mic_capture:
            
            mock_config.return_value = {"port": 8000, "chunk_ms": 200}
            mock_display.return_value = "Test"
            
            mock_window = MagicMock()
            mock_window_class.return_value = mock_window
            
            mock_recorder = MagicMock()
            mock_recorder_class.return_value = mock_recorder
            
            mock_capture = MagicMock()
            mock_capture.start.return_value = True
            mock_capture.source_name = "🎤 Microphone"
            mock_mic_capture.return_value = mock_capture
            
            from live_captions import LiveCaptions
            
            app = LiveCaptions(use_system_audio=False)
            app._start_audio_capture()
            
            mock_mic_capture.assert_called_once()
            mock_capture.start.assert_called_once()
            mock_window.set_audio_status.assert_called_with("🎤 Microphone")

    def test_start_system_audio_capture(self, mock_dependencies):
        """Test starting system audio capture."""
        with patch("live_captions.CaptionWindow") as mock_window_class, \
             patch("live_captions.get_backend_config") as mock_config, \
             patch("live_captions.get_display_info") as mock_display, \
             patch("live_captions.AudioRecorder") as mock_recorder_class, \
             patch("live_captions.set_recorder"), \
             patch("live_captions.SystemAudioCapture") as mock_system_capture:
            
            mock_config.return_value = {"port": 8000, "chunk_ms": 200}
            mock_display.return_value = "Test"
            
            mock_window = MagicMock()
            mock_window_class.return_value = mock_window
            
            mock_recorder = MagicMock()
            mock_recorder_class.return_value = mock_recorder
            
            mock_capture = MagicMock()
            mock_capture.start.return_value = True
            mock_capture.source_name = "🔊 System Audio"
            mock_system_capture.return_value = mock_capture
            
            from live_captions import LiveCaptions
            
            app = LiveCaptions(use_system_audio=True)
            app._start_audio_capture()
            
            mock_system_capture.assert_called_once()
            mock_capture.start.assert_called_once()

    def test_stop_audio_capture(self, mock_dependencies):
        """Test stopping audio capture."""
        with patch("live_captions.CaptionWindow") as mock_window_class, \
             patch("live_captions.get_backend_config") as mock_config, \
             patch("live_captions.get_display_info") as mock_display, \
             patch("live_captions.AudioRecorder") as mock_recorder_class, \
             patch("live_captions.set_recorder"):
            
            mock_config.return_value = {"port": 8000, "chunk_ms": 200}
            mock_display.return_value = "Test"
            
            mock_window = MagicMock()
            mock_window_class.return_value = mock_window
            
            mock_recorder = MagicMock()
            mock_recorder_class.return_value = mock_recorder
            
            from live_captions import LiveCaptions
            
            app = LiveCaptions()
            app.audio_capture = MagicMock()
            app._stop_audio_capture()
            
            app.audio_capture is None  # Should be cleared


class TestLiveCaptionsClose:
    """Tests for application close."""

    def test_close_stops_asr(self, mock_dependencies):
        """Test that close stops ASR client."""
        with patch("live_captions.CaptionWindow") as mock_window_class, \
             patch("live_captions.get_backend_config") as mock_config, \
             patch("live_captions.get_display_info") as mock_display, \
             patch("live_captions.AudioRecorder") as mock_recorder_class, \
             patch("live_captions.set_recorder"):
            
            mock_config.return_value = {"port": 8000, "chunk_ms": 200}
            mock_display.return_value = "Test"
            
            mock_window = MagicMock()
            mock_window_class.return_value = mock_window
            
            mock_recorder = MagicMock()
            mock_recorder.is_recording = False
            mock_recorder_class.return_value = mock_recorder
            
            from live_captions import LiveCaptions
            
            app = LiveCaptions()
            app.asr_client = MagicMock()
            app.close()
            
            app.asr_client.stop.assert_called_once()
            mock_window.close.assert_called_once()

    def test_close_saves_recording(self, mock_dependencies):
        """Test that close saves recording."""
        with patch("live_captions.CaptionWindow") as mock_window_class, \
             patch("live_captions.get_backend_config") as mock_config, \
             patch("live_captions.get_display_info") as mock_display, \
             patch("live_captions.AudioRecorder") as mock_recorder_class, \
             patch("live_captions.set_recorder"):
            
            mock_config.return_value = {"port": 8000, "chunk_ms": 200}
            mock_display.return_value = "Test"
            
            mock_window = MagicMock()
            mock_window_class.return_value = mock_window
            
            mock_recorder = MagicMock()
            mock_recorder.is_recording = True
            mock_recorder.stop.return_value = "/path/to/recording.wav"
            mock_recorder_class.return_value = mock_recorder
            
            from live_captions import LiveCaptions
            
            app = LiveCaptions()
            app.close()
            
            mock_recorder.stop.assert_called_once()


class TestMainFunction:
    """Tests for main() entry point."""

    def test_list_devices(self, mock_dependencies):
        """Test --list-devices argument."""
        with patch("live_captions.list_devices") as mock_list, \
             patch("sys.argv", ["live_captions.py", "--list-devices"]):
            
            from live_captions import main
            
            main()
            
            mock_list.assert_called_once()

    def test_debug_logging(self, mock_dependencies):
        """Test --debug argument enables debug logging."""
        with patch("live_captions.CaptionWindow") as mock_window, \
             patch("live_captions.get_backend_config") as mock_config, \
             patch("live_captions.get_display_info") as mock_display, \
             patch("live_captions.AudioRecorder") as mock_recorder, \
             patch("live_captions.set_recorder"), \
             patch("live_captions.LiveCaptions") as mock_app_class, \
             patch("sys.argv", ["live_captions.py", "--debug"]), \
             patch("logging.getLogger") as mock_logger:
            
            mock_config.return_value = {"port": 8000, "chunk_ms": 200, "name": "Test", "device": "CPU"}
            mock_display.return_value = "Test"
            mock_window.return_value = MagicMock()
            mock_recorder.return_value = MagicMock()
            
            mock_app = MagicMock()
            mock_app_class.return_value = mock_app
            
            from live_captions import main
            
            main()
            
            # App should be created and run
            mock_app.run.assert_called_once()
