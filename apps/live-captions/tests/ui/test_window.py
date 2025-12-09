"""Unit tests for CaptionWindow class."""

import sys
from unittest.mock import MagicMock, patch


def create_mock_window():
    """Helper to create a properly mocked CaptionWindow instance."""
    with patch.dict(sys.modules, {"tkinter": MagicMock(), "tkinter.font": MagicMock()}):
        from src.ui.window import CaptionWindow

        with patch.object(CaptionWindow, "_create_window"):
            window = CaptionWindow.__new__(CaptionWindow)
            window.width = 800
            window.caption_font = MagicMock()
            window.caption_font.measure = lambda x: len(x) * 10  # Simple approximation
            window.root = MagicMock()
            window.line1 = MagicMock()
            window.line2 = MagicMock()
            window.status_label = MagicMock()
            window.recording_label = MagicMock()
            window.audio_status_label = MagicMock()
            window.hint_label = MagicMock()
            window.model_label = MagicMock()

            # Initialize animation state
            window._word_animation_enabled = True
            window._word_timer = None
            window._current_word_index = 0
            window._words_to_animate = []
            window._animation_speed_ms = 200
            window._target_speed_words_per_5s = 25
            window._speed_start_time = None
            window._speed_word_count = 0
            window._turbo_mode = False
            window._full_text = ""
            window._previous_text = ""
            window._current_line1 = ""
            window._current_line2 = ""

            # Constants
            window.TEXT_COLOR = "#ffffff"
            window.TEXT_COLOR_DIM = "#999999"
            window.STATUS_COLOR = "#888888"
            window.CONNECTED_COLOR = "#4ade80"
            window.DISCONNECTED_COLOR = "#f87171"
            window.MAX_FONT_SIZE = 72
            window.MIN_FONT_SIZE = 16

            return window


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
        window = create_mock_window()
        window.base_font_size = 36

        # Simulate scroll up (positive delta)
        mock_event = MagicMock()
        mock_event.delta = 120

        window._on_mousewheel(mock_event)

        assert window.base_font_size == 38
        window.caption_font.configure.assert_called_once_with(size=38)

    def test_mousewheel_decrease_font(self):
        """Test that scrolling down decreases font size."""
        window = create_mock_window()
        window.base_font_size = 36

        # Simulate scroll down (negative delta)
        mock_event = MagicMock()
        mock_event.delta = -120

        window._on_mousewheel(mock_event)

        assert window.base_font_size == 34
        window.caption_font.configure.assert_called_once_with(size=34)

    def test_font_size_max_limit(self):
        """Test that font size cannot exceed maximum."""
        window = create_mock_window()
        window.base_font_size = 72  # At max

        mock_event = MagicMock()
        mock_event.delta = 120  # Try to increase

        window._on_mousewheel(mock_event)

        assert window.base_font_size == 72  # Should stay at max

    def test_font_size_min_limit(self):
        """Test that font size cannot go below minimum."""
        window = create_mock_window()
        window.base_font_size = 16  # At min

        mock_event = MagicMock()
        mock_event.delta = -120  # Try to decrease

        window._on_mousewheel(mock_event)

        assert window.base_font_size == 16  # Should stay at min


class TestWordAnimation:
    """Tests for word-by-word animation functionality."""

    def test_word_animation_enabled_by_default(self):
        """Test that word animation is enabled by default."""
        window = create_mock_window()
        assert window._word_animation_enabled is True

    def test_toggle_word_animation(self):
        """Test toggling word animation on/off."""
        window = create_mock_window()
        window._current_line2 = "test"

        # Mock _update_display to avoid side effects
        window._update_display = MagicMock()

        # Mock after to capture callback
        captured_callbacks = []

        def mock_after(ms, func):
            captured_callbacks.append((ms, func))

        window.root.after = mock_after

        # Create mock event
        mock_event = MagicMock()

        # Toggle off
        window._toggle_word_animation(mock_event)
        assert window._word_animation_enabled is False

        # Toggle on
        window._toggle_word_animation(mock_event)
        assert window._word_animation_enabled is True

    def test_update_text_detects_new_words(self):
        """Test that update_text detects newly added words."""
        window = create_mock_window()
        window._full_text = "Hello world"
        window._previous_text = "Hello world"

        # Mock animation methods
        animate_called = []

        def mock_animate():
            animate_called.append(True)

        window._animate_words = mock_animate

        # Add new words
        window.update_text("Hello world this is new")

        # Should detect 3 new words: "this", "is", "new"
        assert window._words_to_animate == ["this", "is", "new"]
        assert len(animate_called) == 1

    def test_update_text_empty_text_ignored(self):
        """Test that empty text is ignored."""
        window = create_mock_window()
        window._full_text = "existing text"

        window.update_text("")
        assert window._full_text == "existing text"

        window.update_text("   ")
        assert window._full_text == "existing text"

    def test_update_text_no_animation_when_disabled(self):
        """Test that animation is skipped when disabled."""
        window = create_mock_window()
        window._word_animation_enabled = False
        window._full_text = ""

        window.update_text("Hello world")

        # Should update directly without animation
        assert window._full_text == "Hello world"
        assert window._words_to_animate == []

    def test_turbo_mode_activation(self):
        """Test turbo mode activates when new words arrive during animation."""
        window = create_mock_window()
        window._full_text = "Hello"
        window._word_timer = "active_timer"  # Simulate active timer
        window._words_to_animate = ["world"]  # Already animating

        window.update_text("Hello world is fast")

        # Should enable turbo mode
        assert window._turbo_mode is True
        # Should extend animation queue
        assert "is" in window._words_to_animate
        assert "fast" in window._words_to_animate

    def test_calculate_dynamic_speed_normal(self):
        """Test dynamic speed calculation in normal mode."""
        window = create_mock_window()
        window._speed_start_time = None
        window._speed_word_count = 0
        window._turbo_mode = False
        window._animation_speed_ms = 200

        speed = window._calculate_dynamic_speed()

        # Should return base speed
        assert speed == 200
        # Should initialize tracking
        assert window._speed_start_time is not None

    def test_calculate_dynamic_speed_turbo(self):
        """Test dynamic speed in turbo mode is faster."""
        window = create_mock_window()
        window._turbo_mode = True
        window._animation_speed_ms = 200

        speed = window._calculate_dynamic_speed()

        # Turbo mode should be 3x faster (200/3 �?66, min 50)
        assert speed <= 67  # 200 // 3 = 66
        assert speed >= 50  # Minimum is 50

    def test_reset_speed_tracking(self):
        """Test speed tracking reset."""
        window = create_mock_window()
        window._speed_start_time = 12345
        window._speed_word_count = 100

        window._reset_speed_tracking()

        assert window._speed_word_count == 0
        assert window._speed_start_time is not None
        assert window._speed_start_time != 12345


class TestUpdateDisplayFromSegments:
    """Tests for _update_display_from_segments method."""

    def test_single_line_display(self):
        """Test display with single short line."""
        window = create_mock_window()
        window._full_text = "Hello world"
        window._words_to_animate = []

        window._update_display_from_segments()

        # Should be on line2 only (bottom)
        assert window._current_line1 == ""
        assert window._current_line2 == "Hello world"

    def test_two_lines_display(self):
        """Test display wraps to two lines."""
        window = create_mock_window()
        # Make text long enough to wrap (width 800, 10px per char)
        window._full_text = (
            "This is a longer piece of text that should definitely "
            "wrap to multiple lines when displayed"
        )
        window._words_to_animate = []

        window._update_display_from_segments()

        # Both lines should have content
        assert window._current_line1 != ""
        assert window._current_line2 != ""

    def test_animation_partial_display(self):
        """Test display during animation shows partial text."""
        window = create_mock_window()
        window._previous_text = "Hello"
        window._full_text = "Hello world is nice"
        window._words_to_animate = ["world", "is", "nice"]
        window._current_word_index = 1  # Only "world" shown

        window._update_display_from_segments()

        # Should show "Hello world" not full text
        assert "world" in window._current_line2
        assert "nice" not in window._current_line2

    def test_empty_text_display(self):
        """Test display with empty text."""
        window = create_mock_window()
        window._full_text = ""
        window._words_to_animate = []

        window._update_display_from_segments()

        assert window._current_line1 == ""
        assert window._current_line2 == ""


class TestSetMessage:
    """Tests for set_message method."""

    def test_set_message_basic(self):
        """Test setting a message clears first line."""
        window = create_mock_window()
        window._word_timer = None

        window.set_message("Test message")

        assert window._current_line1 == ""
        assert window._current_line2 == "Test message"
        assert window._full_text == "Test message"

    def test_set_message_cancels_animation(self):
        """Test set_message cancels active animation."""
        window = create_mock_window()
        window._word_timer = "active_timer"
        window._words_to_animate = ["some", "words"]

        cancelled_timers = []

        def mock_cancel(timer):
            cancelled_timers.append(timer)

        window.root.after_cancel = mock_cancel

        window.set_message("New message")

        assert "active_timer" in cancelled_timers
        assert window._word_timer is None
        assert window._words_to_animate == []


class TestAudioStatus:
    """Tests for audio status functionality."""

    def test_set_audio_status(self):
        """Test setting audio source status."""
        window = create_mock_window()

        captured_callback = None

        def capture_after(ms, func):
            nonlocal captured_callback
            captured_callback = func

        window.root.after = capture_after

        window.set_audio_status("🎤 Microphone")

        assert captured_callback is not None
        captured_callback()

        window.audio_status_label.configure.assert_called_once()
        call_args = window.audio_status_label.configure.call_args
        assert "🎤 Microphone" in str(call_args)
