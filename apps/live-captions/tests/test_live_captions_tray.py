"""
Unit tests for live_captions_tray.py system tray application module.

Tests the LiveCaptionsTray class and helper functions including:
- Backend health checking
- Python executable finding
- Tray icon creation
- Process management
- Audio source toggling
- Recording management
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


# Mock heavy dependencies before importing
@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock heavy dependencies before imports."""
    mock_pystray = MagicMock()
    mock_pil = MagicMock()
    mock_pil_draw = MagicMock()

    # Mock Image class
    mock_image = MagicMock()
    mock_image.new.return_value = MagicMock()
    mock_image.open.return_value = MagicMock()
    mock_image.Resampling = MagicMock()
    mock_image.Resampling.LANCZOS = 1

    mock_pil.Image = mock_image
    mock_pil.ImageDraw = mock_pil_draw

    with patch.dict(
        sys.modules,
        {
            "pystray": mock_pystray,
            "PIL": mock_pil,
            "PIL.Image": mock_image,
            "PIL.ImageDraw": mock_pil_draw,
        },
    ):
        yield {
            "pystray": mock_pystray,
            "Image": mock_image,
            "ImageDraw": mock_pil_draw,
        }


class TestCheckBackendHealth:
    """Tests for check_backend_health function."""

    def test_healthy_backend(self, mock_dependencies):
        """Test checking a healthy backend."""
        with (
            patch("socket.socket") as mock_socket_class,
            patch("urllib.request.urlopen") as mock_urlopen,
        ):

            mock_socket = MagicMock()
            mock_socket.connect_ex.return_value = 0  # Port is open
            mock_socket_class.return_value = mock_socket

            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            from live_captions_tray import check_backend_health

            is_healthy, status = check_backend_health("whisper")

            assert is_healthy is True
            assert "Ready" in status

    def test_unhealthy_backend_port_closed(self, mock_dependencies):
        """Test checking an unhealthy backend (port closed)."""
        with patch("socket.socket") as mock_socket_class:

            mock_socket = MagicMock()
            mock_socket.connect_ex.return_value = 1  # Port is closed
            mock_socket_class.return_value = mock_socket

            from live_captions_tray import check_backend_health

            is_healthy, status = check_backend_health("whisper")

            assert is_healthy is False
            assert "not running" in status.lower() or "closed" in status.lower()

    def test_unknown_backend(self, mock_dependencies):
        """Test checking an unknown backend."""
        from live_captions_tray import check_backend_health

        is_healthy, status = check_backend_health("unknown_backend")

        assert is_healthy is False
        assert "Unknown" in status


class TestCheckAllBackends:
    """Tests for check_all_backends function."""

    def test_returns_dict(self, mock_dependencies):
        """Test that check_all_backends returns a dict."""
        with patch("live_captions_tray.check_backend_health") as mock_check:
            mock_check.return_value = (True, "Ready")

            from live_captions_tray import check_all_backends

            results = check_all_backends()

            assert isinstance(results, dict)
            assert len(results) > 0


class TestLiveCaptionsTrayInit:
    """Tests for LiveCaptionsTray initialization."""

    def test_default_init(self, mock_dependencies):
        """Test default initialization."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("threading.Thread") as mock_thread,
        ):

            mock_check.return_value = {"whisper": (True, "Ready")}
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()

            assert app.current_process is None
            assert app.current_backend is None
            assert app.use_system_audio is True  # Default
            assert app.enable_recording is True
            assert app.enable_transcription is True

    def test_checks_backends_on_init(self, mock_dependencies):
        """Test that backends are checked on init."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("threading.Thread"),
        ):

            mock_check.return_value = {"whisper": (True, "Ready")}

            from live_captions_tray import LiveCaptionsTray

            LiveCaptionsTray()

            mock_check.assert_called_once()


class TestLiveCaptionsTrayBackendStatus:
    """Tests for backend status checking."""

    def test_is_backend_available(self, mock_dependencies):
        """Test checking if backend is available."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("threading.Thread"),
        ):

            mock_check.return_value = {
                "whisper": (True, "Ready"),
                "vosk": (False, "Not running"),
            }

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()

            assert app.is_backend_available("whisper") is True
            assert app.is_backend_available("vosk") is False

    def test_get_backend_status_text(self, mock_dependencies):
        """Test getting backend status text."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("threading.Thread"),
        ):

            mock_check.return_value = {
                "whisper": (True, "Ready"),
                "vosk": (False, "Port 8001 closed"),
            }

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()

            assert app.get_backend_status_text("whisper") == "Ready"
            assert "closed" in app.get_backend_status_text("vosk").lower()


class TestLiveCaptionsTrayRunning:
    """Tests for process running state."""

    def test_is_running_no_process(self, mock_dependencies):
        """Test is_running when no process."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("threading.Thread"),
        ):

            mock_check.return_value = {}

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()

            assert app.is_running() is False

    def test_is_running_with_process(self, mock_dependencies):
        """Test is_running with active process."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("threading.Thread"),
        ):

            mock_check.return_value = {}

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()

            mock_process = MagicMock()
            mock_process.poll.return_value = None  # Still running
            app.current_process = mock_process

            assert app.is_running() is True

    def test_is_running_with_exited_process(self, mock_dependencies):
        """Test is_running with exited process."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("threading.Thread"),
        ):

            mock_check.return_value = {}

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()

            mock_process = MagicMock()
            mock_process.poll.return_value = 0  # Exited
            app.current_process = mock_process

            assert app.is_running() is False


class TestLiveCaptionsTrayStartStop:
    """Tests for starting and stopping captions."""

    def test_start_captions_unavailable_backend(self, mock_dependencies):
        """Test starting with unavailable backend."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("live_captions_tray.check_backend_health") as mock_health,
            patch("threading.Thread"),
        ):

            mock_check.return_value = {"whisper": (False, "Not running")}
            mock_health.return_value = (False, "Not running")

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()
            app.start_captions("whisper")

            # Should not start a process
            assert app.current_process is None

    def test_start_captions_success(self, mock_dependencies):
        """Test successful start."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("live_captions_tray.check_backend_health") as mock_health,
            patch("subprocess.Popen") as mock_popen,
            patch("threading.Thread"),
            patch("threading.Timer"),
        ):

            mock_check.return_value = {"whisper": (True, "Ready")}
            mock_health.return_value = (True, "Ready")

            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()
            app.start_captions("whisper")

            assert app.current_process is mock_process
            assert app.current_backend == "whisper"

    def test_stop_captions(self, mock_dependencies):
        """Test stopping captions."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("live_captions_tray.request_stop") as mock_request_stop,
            patch("threading.Thread"),
        ):

            mock_check.return_value = {}

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()

            mock_process = MagicMock()
            app.current_process = mock_process
            app.current_backend = "whisper"

            app.stop_captions()

            mock_request_stop.assert_called_once()
            mock_process.wait.assert_called_once_with(timeout=10)
            assert app.current_process is None
            assert app.current_backend is None


class TestLiveCaptionsTrayAudioSource:
    """Tests for audio source toggling."""

    def test_toggle_audio_source(self, mock_dependencies):
        """Test toggling audio source."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("threading.Thread"),
        ):

            mock_check.return_value = {}

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()
            initial = app.use_system_audio

            app.toggle_audio_source()

            assert app.use_system_audio != initial

    def test_toggle_audio_restarts_if_running(self, mock_dependencies):
        """Test that toggling audio restarts process if running."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("live_captions_tray.check_backend_health") as mock_health,
            patch("live_captions_tray.request_stop") as mock_request_stop,
            patch("subprocess.Popen") as mock_popen,
            patch("threading.Thread"),
            patch("threading.Timer"),
        ):

            mock_check.return_value = {"whisper": (True, "Ready")}
            mock_health.return_value = (True, "Ready")

            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_process.pid = 12345
            mock_popen.return_value = mock_process

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()
            app.current_process = mock_process
            app.current_backend = "whisper"

            app.toggle_audio_source()

            # Should restart the process
            mock_request_stop.assert_called()


class TestLiveCaptionsTrayTranscription:
    """Tests for transcription toggling."""

    def test_toggle_transcription(self, mock_dependencies):
        """Test toggling transcription."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("threading.Thread"),
        ):

            mock_check.return_value = {}

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()
            initial = app.enable_transcription

            app.toggle_transcription()

            assert app.enable_transcription != initial


class TestLiveCaptionsTrayRecording:
    """Tests for recording management."""

    def test_toggle_recording(self, mock_dependencies):
        """Test toggling recording."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("threading.Thread"),
        ):

            mock_check.return_value = {}

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()
            initial = app.enable_recording

            app.toggle_recording()

            assert app.enable_recording != initial

    def test_get_recording_info_no_recorder(self, mock_dependencies):
        """Test getting recording info when no recorder."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("live_captions_tray.read_recording_status") as mock_read_status,
            patch("threading.Thread"),
        ):

            mock_check.return_value = {}
            mock_read_status.return_value = (False, "00:00", 0.0, 0.0)

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()
            is_recording, duration_str, duration, _seconds_since_audio = app.get_recording_info()

            assert is_recording is False
            assert duration_str == "00:00"
            assert duration == 0.0

    def test_get_recording_info_with_recorder(self, mock_dependencies):
        """Test getting recording info with active recorder."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("live_captions_tray.read_recording_status") as mock_read_status,
            patch("threading.Thread"),
        ):

            mock_check.return_value = {}
            mock_read_status.return_value = (True, "01:30", 90.0, 5.0)

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()
            is_recording, duration_str, duration, _seconds_since_audio = app.get_recording_info()

            assert is_recording is True
            assert duration_str == "01:30"
            assert duration == 90.0

    def test_clear_recording(self, mock_dependencies):
        """Test clearing recording."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("threading.Thread"),
        ):

            mock_check.return_value = {}

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()
            mock_icon = MagicMock()
            app.icon = mock_icon

            app.clear_recording()

            # Should show notification that clearing from tray is not supported
            mock_icon.notify.assert_called_once()


class TestLiveCaptionsTrayIcon:
    """Tests for tray icon creation."""

    def test_create_icon_image_not_running(self, mock_dependencies):
        """Test creating icon when not running."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("threading.Thread"),
            patch("live_captions_tray.ICON_PATH") as mock_path,
        ):

            mock_check.return_value = {}
            mock_path.exists.return_value = False

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()
            img = app.create_icon_image(running=False)

            assert img is not None

    def test_create_icon_image_running(self, mock_dependencies):
        """Test creating icon when running."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("threading.Thread"),
            patch("live_captions_tray.ICON_PATH") as mock_path,
        ):

            mock_check.return_value = {}
            mock_path.exists.return_value = False

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()
            img = app.create_icon_image(running=True)

            assert img is not None


class TestLiveCaptionsTrayQuit:
    """Tests for quit handling."""

    def test_quit_stops_process(self, mock_dependencies):
        """Test quit stops running process."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("live_captions_tray.request_stop") as mock_request_stop,
            patch("threading.Thread"),
        ):

            mock_check.return_value = {}

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()

            mock_process = MagicMock()
            app.current_process = mock_process

            mock_icon = MagicMock()
            app.quit(mock_icon, None)

            mock_request_stop.assert_called()
            mock_icon.stop.assert_called()

    def test_quit_saves_recording(self, mock_dependencies):
        """Test quit calls stop_captions."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("live_captions_tray.request_stop") as mock_request_stop,
            patch("threading.Thread"),
        ):

            mock_check.return_value = {}

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()

            mock_process = MagicMock()
            app.current_process = mock_process

            mock_icon = MagicMock()
            app.quit(mock_icon, None)

            # Quit should call stop_captions which calls request_stop
            mock_request_stop.assert_called_once()


class TestLanguageSupport:
    """Tests for language support functionality."""

    def test_default_language(self, mock_dependencies):
        """Test default language is English."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("threading.Thread"),
        ):
            mock_check.return_value = {}

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()
            assert app.current_language == "en"

    def test_set_language(self, mock_dependencies):
        """Test setting language."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("threading.Thread"),
        ):
            mock_check.return_value = {}

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()
            app.set_language("yue")
            assert app.current_language == "yue"

    def test_set_same_language_no_op(self, mock_dependencies):
        """Test setting same language does nothing."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("threading.Thread"),
        ):
            mock_check.return_value = {}

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()
            app.current_language = "en"

            # Mock start_captions to track if called
            app.start_captions = MagicMock()
            app.set_language("en")  # Same language

            # Should not restart
            app.start_captions.assert_not_called()

    def test_is_language_available_whisper(self, mock_dependencies):
        """Test language availability check for Whisper."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("threading.Thread"),
        ):
            mock_check.return_value = {}

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()
            # Whisper supports both languages
            assert app.is_language_available("en") is True
            assert app.is_language_available("yue") is True

    def test_is_language_available_parakeet(self, mock_dependencies):
        """Test language availability check for Parakeet (English only)."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("live_captions_tray.check_backend_health") as mock_health,
            patch("subprocess.Popen") as mock_popen,
            patch("threading.Thread"),
            patch("threading.Timer"),
        ):
            mock_check.return_value = {"parakeet": (True, "Ready")}
            mock_health.return_value = (True, "Ready")
            mock_popen.return_value = MagicMock(pid=123, poll=MagicMock(return_value=None))

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()
            # Start with parakeet to set current_backend
            app.start_captions("parakeet")

            # Now current_backend is parakeet, so check should reflect that
            # Parakeet only supports English
            assert app.is_language_available("en") is True
            assert app.is_language_available("yue") is False

    def test_language_fallback_for_unsupported_backend(self, mock_dependencies):
        """Test language falls back to English for unsupported backends."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("live_captions_tray.check_backend_health") as mock_health,
            patch("subprocess.Popen") as mock_popen,
            patch("threading.Thread"),
            patch("threading.Timer"),
        ):
            mock_check.return_value = {"parakeet": (True, "Ready")}
            mock_health.return_value = (True, "Ready")

            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()
            app.current_language = "yue"  # Set to Cantonese
            app.start_captions("parakeet")  # Parakeet doesn't support Cantonese

            # Should still start, but with English
            assert app.current_process is mock_process
            # Check command includes --language en (fallback)
            call_args = mock_popen.call_args
            cmd = call_args[0][0]
            lang_idx = cmd.index("--language")
            assert cmd[lang_idx + 1] == "en"


class TestLanguageConstants:
    """Tests for language-related constants."""

    def test_languages_dict(self, mock_dependencies):
        """Test LANGUAGES constant is defined correctly."""
        from live_captions_tray import LANGUAGES

        assert "en" in LANGUAGES
        assert "yue" in LANGUAGES
        assert "English" in LANGUAGES["en"]
        assert "Cantonese" in LANGUAGES["yue"] or "粵語" in LANGUAGES["yue"]

    def test_backend_languages(self, mock_dependencies):
        """Test BACKEND_LANGUAGES constant is defined correctly."""
        from live_captions_tray import BACKEND_LANGUAGES

        assert "whisper" in BACKEND_LANGUAGES
        assert "parakeet" in BACKEND_LANGUAGES
        assert "vosk" in BACKEND_LANGUAGES

        # Whisper should support multiple languages
        assert "en" in BACKEND_LANGUAGES["whisper"]
        assert "yue" in BACKEND_LANGUAGES["whisper"]

        # Parakeet and Vosk are English-only
        assert BACKEND_LANGUAGES["parakeet"] == ["en"]
        assert BACKEND_LANGUAGES["vosk"] == ["en"]


class TestMainFunction:
    """Tests for main() entry point."""

    def test_main_creates_app(self, mock_dependencies):
        """Test main creates and runs app."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("live_captions_tray.LiveCaptionsTray") as mock_app_class,
            patch("threading.Thread"),
            patch("sys.argv", ["live_captions_tray.py"]),
        ):

            mock_check.return_value = {}
            mock_app = MagicMock()
            mock_app_class.return_value = mock_app

            from live_captions_tray import main

            main()

            mock_app.run.assert_called_once()

    def test_main_auto_start(self, mock_dependencies):
        """Test main with --auto-start."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("live_captions_tray.LiveCaptionsTray") as mock_app_class,
            patch("threading.Thread"),
            patch("threading.Timer") as mock_timer,
            patch("sys.argv", ["live_captions_tray.py", "--auto-start"]),
        ):

            mock_check.return_value = {}
            mock_app = MagicMock()
            mock_app_class.return_value = mock_app

            from live_captions_tray import main

            main()

            # Timer should be created for auto-start
            mock_timer.assert_called()
