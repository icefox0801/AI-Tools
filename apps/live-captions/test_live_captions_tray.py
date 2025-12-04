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


class TestFindPython:
    """Tests for find_python function."""

    def test_finds_venv_python(self, mock_dependencies):
        """Test finding Python in project venv."""
        with patch("pathlib.Path.exists") as mock_exists:
            # First call is for venv python, return True
            mock_exists.return_value = True

            from live_captions_tray import find_python

            result = find_python()

            # Should find the venv python
            assert "python" in result.lower()

    def test_fallback_to_path(self, mock_dependencies):
        """Test fallback to PATH when no specific Python found."""
        with patch("pathlib.Path.exists") as mock_exists:
            # All paths don't exist
            mock_exists.return_value = False

            from live_captions_tray import find_python

            result = find_python()

            assert result == "python"


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
            patch("threading.Thread"),
        ):

            mock_check.return_value = {}

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()

            mock_process = MagicMock()
            app.current_process = mock_process
            app.current_backend = "whisper"

            app.stop_captions()

            mock_process.terminate.assert_called_once()
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
            mock_process.terminate.assert_called()


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
            patch("live_captions_tray.get_recorder") as mock_get_recorder,
            patch("threading.Thread"),
        ):

            mock_check.return_value = {}
            mock_get_recorder.return_value = None

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()
            is_recording, duration_str, duration = app.get_recording_info()

            assert is_recording is False
            assert duration_str == "00:00"
            assert duration == 0.0

    def test_get_recording_info_with_recorder(self, mock_dependencies):
        """Test getting recording info with active recorder."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("live_captions_tray.get_recorder") as mock_get_recorder,
            patch("threading.Thread"),
        ):

            mock_check.return_value = {}

            mock_recorder = MagicMock()
            mock_recorder.is_recording = True
            mock_recorder.duration_str = "01:30"
            mock_recorder.duration = 90.0
            mock_get_recorder.return_value = mock_recorder

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()
            is_recording, duration_str, duration = app.get_recording_info()

            assert is_recording is True
            assert duration_str == "01:30"
            assert duration == 90.0

    def test_clear_recording(self, mock_dependencies):
        """Test clearing recording."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("live_captions_tray.get_recorder") as mock_get_recorder,
            patch("threading.Thread"),
        ):

            mock_check.return_value = {}

            mock_recorder = MagicMock()
            mock_get_recorder.return_value = mock_recorder

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()
            app.clear_recording()

            mock_recorder.clear.assert_called_once()


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


class TestLiveCaptionsTrayDoubleClick:
    """Tests for double-click handling."""

    def test_double_click_starts_when_stopped(self, mock_dependencies):
        """Test double-click starts captions when stopped."""
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
            app.on_double_click(None, None)

            assert app.current_process is not None

    def test_double_click_stops_when_running(self, mock_dependencies):
        """Test double-click stops captions when running."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("threading.Thread"),
        ):

            mock_check.return_value = {}

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()

            mock_process = MagicMock()
            mock_process.poll.return_value = None
            app.current_process = mock_process
            app.current_backend = "whisper"

            app.on_double_click(None, None)

            mock_process.terminate.assert_called()


class TestLiveCaptionsTrayQuit:
    """Tests for quit handling."""

    def test_quit_stops_process(self, mock_dependencies):
        """Test quit stops running process."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("live_captions_tray.get_recorder") as mock_get_recorder,
            patch("threading.Thread"),
        ):

            mock_check.return_value = {}
            mock_get_recorder.return_value = None

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()

            mock_process = MagicMock()
            app.current_process = mock_process

            mock_icon = MagicMock()
            app.quit(mock_icon, None)

            mock_process.terminate.assert_called()
            mock_icon.stop.assert_called()

    def test_quit_saves_recording(self, mock_dependencies):
        """Test quit saves recording if active."""
        with (
            patch("live_captions_tray.check_all_backends") as mock_check,
            patch("live_captions_tray.get_recorder") as mock_get_recorder,
            patch("threading.Thread"),
        ):

            mock_check.return_value = {}

            mock_recorder = MagicMock()
            mock_recorder.is_recording = True
            mock_recorder.duration = 5.0
            mock_get_recorder.return_value = mock_recorder

            from live_captions_tray import LiveCaptionsTray

            app = LiveCaptionsTray()

            mock_icon = MagicMock()
            app.quit(mock_icon, None)

            mock_recorder.stop.assert_called_once()


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
