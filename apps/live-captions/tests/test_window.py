"""Unit tests for CaptionWindow class."""

import sys
from unittest.mock import MagicMock, patch


class TestCaptionWindowUnit:
    """Unit tests for CaptionWindow that don't require tkinter."""

    def test_text_wrapping_empty_text(self):
        """Test that empty text returns empty list."""
        # Mock the tkinter module
        with patch.dict(sys.modules, {"tkinter": MagicMock(), "tkinter.font": MagicMock()}):
            from src.ui.window import CaptionWindow

            # Create mock window
            with patch.object(CaptionWindow, "_create_window"):
                window = CaptionWindow.__new__(CaptionWindow)
                window.width = 800
                window.caption_font = MagicMock()
                window.caption_font.measure = lambda x: len(x) * 10  # Simple approximation

                assert window._wrap_text("") == []
                assert window._wrap_text("   ") == []

    def test_text_wrapping_short_text(self):
        """Test that short text stays on one line."""
        with patch.dict(sys.modules, {"tkinter": MagicMock(), "tkinter.font": MagicMock()}):
            from src.ui.window import CaptionWindow

            with patch.object(CaptionWindow, "_create_window"):
                window = CaptionWindow.__new__(CaptionWindow)
                window.width = 800
                window.caption_font = MagicMock()
                window.caption_font.measure = lambda x: len(x) * 10

                result = window._wrap_text("Hello world")
                assert len(result) == 1
                assert result[0] == "Hello world"

    def test_text_wrapping_long_text(self):
        """Test that long text wraps to multiple lines."""
        with patch.dict(sys.modules, {"tkinter": MagicMock(), "tkinter.font": MagicMock()}):
            from src.ui.window import CaptionWindow

            with patch.object(CaptionWindow, "_create_window"):
                window = CaptionWindow.__new__(CaptionWindow)
                window.width = 200  # Narrow window
                window.caption_font = MagicMock()
                window.caption_font.measure = lambda x: len(x) * 10

                text = "This is a long sentence that should wrap to multiple lines"
                result = window._wrap_text(text)
                assert len(result) > 1

    def test_constants(self):
        """Test that constants are properly defined."""
        with patch.dict(sys.modules, {"tkinter": MagicMock(), "tkinter.font": MagicMock()}):
            from src.ui.window import CaptionWindow

            assert CaptionWindow.DEFAULT_FONT_SIZE == 36
            assert CaptionWindow.MIN_FONT_SIZE == 16
            assert CaptionWindow.MAX_FONT_SIZE == 72
            assert CaptionWindow.DEFAULT_ALPHA == 0.85
            assert CaptionWindow.BG_COLOR == "#1a1a1a"
            assert CaptionWindow.TEXT_COLOR == "#ffffff"


class TestCaptionWindowRecordingStatus:
    """Tests for recording status functionality."""

    def test_set_recording_status_recording(self):
        """Test recording status shows correct format when recording."""
        with patch.dict(sys.modules, {"tkinter": MagicMock(), "tkinter.font": MagicMock()}):
            from src.ui.window import CaptionWindow

            with patch.object(CaptionWindow, "_create_window"):
                window = CaptionWindow.__new__(CaptionWindow)
                window.root = MagicMock()
                window.recording_label = MagicMock()
                window.STATUS_COLOR = "#888888"

                # Capture what's passed to after
                captured_callback = None

                def capture_after(ms, func):
                    nonlocal captured_callback
                    captured_callback = func

                window.root.after = capture_after

                # Call the method
                window.set_recording_status(True, "05:30")

                # Execute the captured callback
                assert captured_callback is not None
                captured_callback()

                # Verify the recording_label was configured correctly
                window.recording_label.configure.assert_called_once()
                call_args = window.recording_label.configure.call_args
                assert "🔴 REC 05:30" in str(call_args)

    def test_set_recording_status_not_recording(self):
        """Test recording status clears when not recording."""
        with patch.dict(sys.modules, {"tkinter": MagicMock(), "tkinter.font": MagicMock()}):
            from src.ui.window import CaptionWindow

            with patch.object(CaptionWindow, "_create_window"):
                window = CaptionWindow.__new__(CaptionWindow)
                window.root = MagicMock()
                window.recording_label = MagicMock()
                window.STATUS_COLOR = "#888888"

                captured_callback = None

                def capture_after(ms, func):
                    nonlocal captured_callback
                    captured_callback = func

                window.root.after = capture_after

                # Call the method
                window.set_recording_status(False)

                # Execute the captured callback
                assert captured_callback is not None
                captured_callback()

                # Verify the recording_label was configured with empty text
                window.recording_label.configure.assert_called_once()
                call_args = window.recording_label.configure.call_args
                assert call_args[1]["text"] == ""


class TestCaptionWindowConnectionStatus:
    """Tests for connection status functionality."""

    def test_set_connection_status_connected(self):
        """Test connection status shows connected."""
        with patch.dict(sys.modules, {"tkinter": MagicMock(), "tkinter.font": MagicMock()}):
            from src.ui.window import CaptionWindow

            with patch.object(CaptionWindow, "_create_window"):
                window = CaptionWindow.__new__(CaptionWindow)
                window.root = MagicMock()
                window.status_label = MagicMock()
                window.CONNECTED_COLOR = "#4ade80"
                window.DISCONNECTED_COLOR = "#f87171"

                captured_callback = None

                def capture_after(ms, func):
                    nonlocal captured_callback
                    captured_callback = func

                window.root.after = capture_after

                window.set_connection_status(True)

                assert captured_callback is not None
                captured_callback()

                window.status_label.configure.assert_called_once()
                call_args = window.status_label.configure.call_args
                assert "Connected" in str(call_args)
                assert "#4ade80" in str(call_args)

    def test_set_connection_status_disconnected(self):
        """Test connection status shows disconnected."""
        with patch.dict(sys.modules, {"tkinter": MagicMock(), "tkinter.font": MagicMock()}):
            from src.ui.window import CaptionWindow

            with patch.object(CaptionWindow, "_create_window"):
                window = CaptionWindow.__new__(CaptionWindow)
                window.root = MagicMock()
                window.status_label = MagicMock()
                window.CONNECTED_COLOR = "#4ade80"
                window.DISCONNECTED_COLOR = "#f87171"

                captured_callback = None

                def capture_after(ms, func):
                    nonlocal captured_callback
                    captured_callback = func

                window.root.after = capture_after

                window.set_connection_status(False)

                assert captured_callback is not None
                captured_callback()

                window.status_label.configure.assert_called_once()
                call_args = window.status_label.configure.call_args
                assert "Disconnected" in str(call_args)
                assert "#f87171" in str(call_args)


class TestCaptionWindowFontResizing:
    """Tests for font resizing functionality."""

    def test_mousewheel_increase_font(self):
        """Test that scrolling up increases font size."""
        with patch.dict(sys.modules, {"tkinter": MagicMock(), "tkinter.font": MagicMock()}):
            from src.ui.window import CaptionWindow

            with patch.object(CaptionWindow, "_create_window"):
                window = CaptionWindow.__new__(CaptionWindow)
                window.base_font_size = 36
                window.caption_font = MagicMock()
                window.MAX_FONT_SIZE = 72
                window.MIN_FONT_SIZE = 16

                # Simulate scroll up (positive delta)
                mock_event = MagicMock()
                mock_event.delta = 120

                window._on_mousewheel(mock_event)

                assert window.base_font_size == 38
                window.caption_font.configure.assert_called_once_with(size=38)

    def test_mousewheel_decrease_font(self):
        """Test that scrolling down decreases font size."""
        with patch.dict(sys.modules, {"tkinter": MagicMock(), "tkinter.font": MagicMock()}):
            from src.ui.window import CaptionWindow

            with patch.object(CaptionWindow, "_create_window"):
                window = CaptionWindow.__new__(CaptionWindow)
                window.base_font_size = 36
                window.caption_font = MagicMock()
                window.MAX_FONT_SIZE = 72
                window.MIN_FONT_SIZE = 16

                # Simulate scroll down (negative delta)
                mock_event = MagicMock()
                mock_event.delta = -120

                window._on_mousewheel(mock_event)

                assert window.base_font_size == 34
                window.caption_font.configure.assert_called_once_with(size=34)

    def test_font_size_max_limit(self):
        """Test that font size cannot exceed maximum."""
        with patch.dict(sys.modules, {"tkinter": MagicMock(), "tkinter.font": MagicMock()}):
            from src.ui.window import CaptionWindow

            with patch.object(CaptionWindow, "_create_window"):
                window = CaptionWindow.__new__(CaptionWindow)
                window.base_font_size = 72  # At max
                window.caption_font = MagicMock()
                window.MAX_FONT_SIZE = 72
                window.MIN_FONT_SIZE = 16

                mock_event = MagicMock()
                mock_event.delta = 120  # Try to increase

                window._on_mousewheel(mock_event)

                assert window.base_font_size == 72  # Should stay at max

    def test_font_size_min_limit(self):
        """Test that font size cannot go below minimum."""
        with patch.dict(sys.modules, {"tkinter": MagicMock(), "tkinter.font": MagicMock()}):
            from src.ui.window import CaptionWindow

            with patch.object(CaptionWindow, "_create_window"):
                window = CaptionWindow.__new__(CaptionWindow)
                window.base_font_size = 16  # At min
                window.caption_font = MagicMock()
                window.MAX_FONT_SIZE = 72
                window.MIN_FONT_SIZE = 16

                mock_event = MagicMock()
                mock_event.delta = -120  # Try to decrease

                window._on_mousewheel(mock_event)

                assert window.base_font_size == 16  # Should stay at min
