"""Unit tests for audio-notes UI styles module."""

# Import styles directly - it doesn't depend on gradio
from ui.styles import CUSTOM_CSS, CUSTOM_JS


class TestCustomCSS:
    """Tests for CUSTOM_CSS constant."""

    def test_css_not_empty(self):
        """Test that CSS is defined and not empty."""
        assert CUSTOM_CSS is not None
        assert len(CUSTOM_CSS) > 0

    def test_css_hides_progress_bars(self):
        """Test that CSS includes rules for hiding progress bars."""
        assert "progress" in CUSTOM_CSS.lower()
        assert "display: none" in CUSTOM_CSS

    def test_css_hides_loaders(self):
        """Test that CSS includes rules for hiding loaders."""
        assert "loader" in CUSTOM_CSS.lower()

    def test_css_preserves_checkbox_inputs(self):
        """Test that CSS doesn't hide checkbox/radio inputs."""
        # CSS should have exceptions for radio/checkbox
        assert "radio" in CUSTOM_CSS.lower() or "checkbox" in CUSTOM_CSS.lower()


class TestCustomJS:
    """Tests for CUSTOM_JS constant."""

    def test_js_not_empty(self):
        """Test that JavaScript is defined and not empty."""
        assert CUSTOM_JS is not None
        assert len(CUSTOM_JS) > 0

    def test_js_handles_ctrl_enter(self):
        """Test that JavaScript includes Ctrl+Enter handling."""
        assert "ctrlKey" in CUSTOM_JS
        assert "Enter" in CUSTOM_JS

    def test_js_targets_chat_input(self):
        """Test that JavaScript targets chat input."""
        assert "chat-input" in CUSTOM_JS

    def test_js_clicks_send_button(self):
        """Test that JavaScript clicks send button."""
        assert "chat-send-btn" in CUSTOM_JS
        assert "click" in CUSTOM_JS.lower()

    def test_js_is_function(self):
        """Test that JavaScript is a valid function definition."""
        assert "function()" in CUSTOM_JS
        assert "return" in CUSTOM_JS
