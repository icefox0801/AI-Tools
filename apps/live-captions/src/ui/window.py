"""Caption overlay window for Live Captions."""

import contextlib
import ctypes
import logging
import tkinter as tk
from collections.abc import Callable
from tkinter import font as tkfont

logger = logging.getLogger(__name__)

# Enable high DPI awareness on Windows
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    with contextlib.suppress(Exception):
        ctypes.windll.user32.SetProcessDPIAware()


class CaptionWindow:
    """Transparent overlay window for displaying captions."""

    # UI Constants
    DEFAULT_FONT_SIZE = 36
    MIN_FONT_SIZE = 16
    MAX_FONT_SIZE = 72
    DEFAULT_ALPHA = 0.85
    BG_COLOR = "#1a1a1a"
    TEXT_COLOR = "#ffffff"
    TEXT_COLOR_DIM = "#999999"
    STATUS_COLOR = "#888888"
    HINT_COLOR = "#555555"
    CONNECTED_COLOR = "#4ade80"
    DISCONNECTED_COLOR = "#f87171"

    def __init__(
        self,
        model_display: str = "",
        language: str = "en",
        on_close: Callable[[], None] | None = None,
    ):
        """
        Initialize the caption window.

        Args:
            model_display: Text to show for model info
            language: Transcription language code
            on_close: Callback when window is closed
        """
        self.on_close = on_close
        self.model_display = model_display
        self.language = language

        # Font size
        self.base_font_size = self.DEFAULT_FONT_SIZE

        # Drag state
        self.drag_start_x = 0
        self.drag_start_y = 0

        # Create UI
        self._create_window()

    def _create_window(self):
        """Create the overlay window."""
        self.root = tk.Tk()
        self.root.title("Live Captions")

        # Window settings
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", self.DEFAULT_ALPHA)
        self.root.overrideredirect(True)
        self.root.configure(bg=self.BG_COLOR)

        # Size and position
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.width = int(screen_width * 0.8)
        self.height = 260
        x = (screen_width - self.width) // 2
        y = screen_height - self.height - 80
        self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")

        # Bindings
        self.root.bind("<Button-1>", self._start_drag)
        self.root.bind("<B1-Motion>", self._on_drag)
        self.root.bind("<Button-3>", lambda e: self._handle_close())
        self.root.bind("<Escape>", lambda e: self._handle_close())
        self.root.bind("<MouseWheel>", self._on_mousewheel)
        self.root.bind("<KeyPress-w>", self._toggle_word_animation)
        self.root.focus_set()  # Allow keyboard input

        # Fonts
        self.caption_font = tkfont.Font(family="Segoe UI", size=self.base_font_size, weight="bold")

        # Container
        self.container = tk.Frame(self.root, bg=self.BG_COLOR)
        self.container.pack(expand=True, fill="both", padx=15, pady=(15, 30))

        # Caption lines
        self.line1 = tk.Label(
            self.container,
            text="",
            font=self.caption_font,
            fg=self.TEXT_COLOR_DIM,
            bg=self.BG_COLOR,
            anchor="w",
            justify="left",
            wraplength=self.width - 40,
        )
        self.line1.pack(expand=True, fill="both")

        self.line2 = tk.Label(
            self.container,
            text="🎙️ Starting...",
            font=self.caption_font,
            fg=self.TEXT_COLOR,
            bg=self.BG_COLOR,
            anchor="w",
            justify="left",
            wraplength=self.width - 40,
        )
        self.line2.pack(expand=True, fill="both")

        # Track current display state
        self._current_line1 = ""
        self._current_line2 = ""
        self._full_text = ""  # Complete transcript (grows bottom-to-top)
        self._previous_text = ""  # Previous text for detecting new words

        # Word-by-word animation state
        self._word_animation_enabled = True
        self._word_timer = None
        self._current_word_index = 0
        self._words_to_animate = []
        self._animation_speed_ms = 200  # Base speed (ms per word)
        self._target_speed_words_per_5s = 25  # Target: 5 words per second
        self._speed_start_time = None
        self._speed_word_count = 0
        self._turbo_mode = False  # 3x speed when catching up

        # Status labels
        self._create_status_labels()

    def _create_status_labels(self):
        """Create status indicator labels."""
        status_font = tkfont.Font(size=9)

        # Connection status (top right)
        self.status_label = tk.Label(
            self.root,
            text="● Initializing...",
            font=status_font,
            fg=self.STATUS_COLOR,
            bg=self.BG_COLOR,
        )
        self.status_label.place(relx=1.0, y=5, anchor="ne", x=-10)

        # Recording status (bottom right)
        self.recording_label = tk.Label(
            self.root,
            text="",
            font=status_font,
            fg=self.STATUS_COLOR,
            bg=self.BG_COLOR,
        )
        self.recording_label.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-5)

        # Audio status (top left)
        self.audio_status_label = tk.Label(
            self.root, text="🎙️ Off", font=status_font, fg=self.STATUS_COLOR, bg=self.BG_COLOR
        )
        self.audio_status_label.place(x=10, y=5, anchor="nw")

        # Hint (bottom center)
        self.hint_label = tk.Label(
            self.root,
            text="Drag to move • Scroll to resize • Right-click or Esc to close",
            font=status_font,
            fg=self.HINT_COLOR,
            bg=self.BG_COLOR,
        )
        self.hint_label.place(relx=0.5, rely=1.0, anchor="s", y=-5)

        # Language display names
        lang_names = {"en": "EN", "yue": "粵語"}
        lang_display = lang_names.get(self.language, self.language.upper())

        # Model info (top center) - includes language
        model_text = (
            f"{self.model_display} • {lang_display}" if self.model_display else lang_display
        )
        self.model_label = tk.Label(
            self.root,
            text=model_text,
            font=status_font,
            fg=self.STATUS_COLOR,
            bg=self.BG_COLOR,
        )
        self.model_label.place(relx=0.5, y=5, anchor="n")

    def _start_drag(self, event):
        """Start window drag."""
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def _on_drag(self, event):
        """Handle window drag motion."""
        x = self.root.winfo_x() + event.x - self.drag_start_x
        y = self.root.winfo_y() + event.y - self.drag_start_y
        self.root.geometry(f"+{x}+{y}")

    def _on_mousewheel(self, event):
        """Handle mouse wheel for font resize."""
        if event.delta > 0:
            self.base_font_size = min(self.base_font_size + 2, self.MAX_FONT_SIZE)
        else:
            self.base_font_size = max(self.base_font_size - 2, self.MIN_FONT_SIZE)
        self.caption_font.configure(size=self.base_font_size)

        # Update wraplength for consistent layout
        wrap_width = self.width - 40
        self.line1.configure(wraplength=wrap_width)
        self.line2.configure(wraplength=wrap_width)

    def _toggle_word_animation(self, event):
        """Toggle word-by-word animation mode."""
        self._word_animation_enabled = not self._word_animation_enabled
        status = "ON" if self._word_animation_enabled else "OFF"
        # Show brief status message
        old_line2 = self._current_line2
        self._current_line2 = f"Word Animation: {status}"
        self._update_display(False)
        # Restore after 1 second
        self.root.after(
            1000, lambda: (setattr(self, "_current_line2", old_line2), self._update_display(False))
        )

    def _handle_close(self):
        """Handle window close."""
        # Cancel any active timers
        if self._word_timer:
            self.root.after_cancel(self._word_timer)
        if self.on_close:
            self.on_close()

    def _get_text_width(self, text: str) -> int:
        """Get pixel width of text."""
        return self.caption_font.measure(text)

    def _wrap_text(self, text: str) -> list[str]:
        """Wrap text to fit window width."""
        if not text:
            return []

        max_width = self.width - 40
        words = text.split()
        if not words:
            return []

        lines = []
        current_line = ""

        for word in words:
            test_line = (current_line + " " + word).strip() if current_line else word
            if self._get_text_width(test_line) <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines

    def set_word_animation(self, enabled: bool, speed_ms: int = 200):
        """Enable/disable word-by-word animation."""
        self._word_animation_enabled = enabled
        self._animation_speed_ms = speed_ms
        if not enabled and self._word_timer:
            self.root.after_cancel(self._word_timer)
            self._word_timer = None

    def _animate_words(self):
        """Animate words one by one with dynamic speed adaptation."""
        if not self._words_to_animate or self._current_word_index >= len(self._words_to_animate):
            # Animation complete - update previous text to include animated words
            self._previous_text = self._full_text
            self._words_to_animate = []
            self._word_timer = None
            self._turbo_mode = False
            self._reset_speed_tracking()
            # Final display update
            self._update_display_from_segments()
            return

        # Dynamic speed calculation
        current_speed = self._calculate_dynamic_speed()

        # Move to next word first, then update display
        self._current_word_index += 1

        # Build display text up to current word
        self._update_display_from_segments()

        # Track speed metrics
        self._speed_word_count += 1

        # Schedule next word
        if self._current_word_index < len(self._words_to_animate):
            self._word_timer = self.root.after(current_speed, self._animate_words)
        else:
            # Final word shown - complete animation
            self._previous_text = self._full_text
            self._words_to_animate = []
            self._word_timer = None
            self._turbo_mode = False
            self._reset_speed_tracking()

    def _calculate_dynamic_speed(self) -> int:
        """Calculate dynamic animation speed based on target rate."""
        import time

        # Initialize speed tracking
        if self._speed_start_time is None:
            self._speed_start_time = time.time()
            self._speed_word_count = 0

        # Check if we need turbo mode (catching up)
        if self._turbo_mode:
            return max(self._animation_speed_ms // 3, 50)  # 3x speed, minimum 50ms

        # Calculate current rate
        elapsed = time.time() - self._speed_start_time
        if elapsed >= 5.0 and self._speed_word_count > 0:
            current_rate = self._speed_word_count / elapsed
            target_rate = self._target_speed_words_per_5s / 5.0  # words per second

            # Adjust speed to meet target rate
            if current_rate < target_rate * 0.8:  # Too slow
                self._animation_speed_ms = max(self._animation_speed_ms - 20, 80)
            elif current_rate > target_rate * 1.2:  # Too fast
                self._animation_speed_ms = min(self._animation_speed_ms + 20, 500)

            # Reset tracking for next interval
            self._reset_speed_tracking()

        return self._animation_speed_ms

    def _reset_speed_tracking(self):
        """Reset speed tracking metrics."""
        import time

        self._speed_start_time = time.time()
        self._speed_word_count = 0

    def _update_display_from_segments(self):
        """Update display showing last two lines from full text."""
        full_display_text = self._full_text

        # For word animation: only show up to current animated word
        if self._word_animation_enabled and self._words_to_animate:
            # Show previous text + animated portion of new words
            prev_words = self._previous_text.split() if self._previous_text else []
            animated_new_words = self._words_to_animate[:self._current_word_index]
            all_words = prev_words + animated_new_words
            full_display_text = " ".join(all_words)

        # Get last two lines (bottom-to-top growth)
        if full_display_text:
            lines = self._wrap_text(full_display_text)

            if len(lines) >= 2:
                self._current_line1 = lines[-2]
                self._current_line2 = lines[-1]
            elif len(lines) == 1:
                self._current_line1 = ""
                self._current_line2 = lines[0]
            else:
                self._current_line1 = ""
                self._current_line2 = ""
        else:
            self._current_line1 = ""
            self._current_line2 = ""

        # Update display
        self.line1.configure(text=self._current_line1, fg=self.TEXT_COLOR_DIM)
        self.line2.configure(text=self._current_line2, fg=self.TEXT_COLOR)

    def update_text(self, text: str):
        """
        Update displayed caption text with bottom-to-top growth and word animation.

        Args:
            text: Full text to display (grows continuously, no truncation)
        """
        if not text or not text.strip():
            return

        text = text.strip()
        
        # Detect new words added to the text
        if self._word_animation_enabled and text != self._full_text:
            old_words = self._full_text.split() if self._full_text else []
            new_words = text.split()
            
            # Find newly added words
            if len(new_words) > len(old_words):
                # Words were added - animate them
                added_words = new_words[len(old_words):]
                
                # If already animating, enable turbo mode to catch up
                if self._word_timer and self._words_to_animate:
                    self._turbo_mode = True
                    # Add new words to animation queue
                    self._words_to_animate.extend(added_words)
                else:
                    # Start new animation
                    self._previous_text = self._full_text
                    self._words_to_animate = added_words
                    self._current_word_index = 0
                    self._reset_speed_tracking()
                    self._animate_words()
                
                # Update full text but don't display yet (animation will handle it)
                self._full_text = text
                return

        # No animation or text replaced - update directly
        self._full_text = text
        self._previous_text = text
        self._words_to_animate = []
        self._update_display_from_segments()

    def _set_lines(self, line1: str, line2: str):
        """Set caption line text."""
        self.line1.configure(text=line1)
        self.line2.configure(text=line2)
        self.line1.pack_configure(padx=(20, 0))
        self.line2.pack_configure(padx=(20, 0))

    def _update_display(self):
        """Update the display with current line text."""
        # Don't interfere with active word animation
        if self._word_timer:
            return

        # Update from segments
        self._update_display_from_segments()

        # Update padding
        self.line1.pack_configure(padx=(20, 0))
        self.line2.pack_configure(padx=(20, 0))

    def set_message(self, message: str):
        """Set a message on the second line (clears first line)."""
        # Cancel any animation
        if self._word_timer:
            self.root.after_cancel(self._word_timer)
            self._word_timer = None
        self._words_to_animate = []
        
        self._current_line1 = ""
        self._current_line2 = message
        self._full_text = message
        self._previous_text = message
        self.line1.configure(text="", fg=self.TEXT_COLOR_DIM)
        self.line2.configure(text=message, fg=self.TEXT_COLOR)

    def set_connection_status(self, connected: bool):
        """Update connection status indicator."""
        if connected:
            self.root.after(
                0, lambda: self.status_label.configure(text="● Connected", fg=self.CONNECTED_COLOR)
            )
        else:
            self.root.after(
                0,
                lambda: self.status_label.configure(
                    text="● Disconnected", fg=self.DISCONNECTED_COLOR
                ),
            )

    def set_audio_status(self, source_name: str):
        """Update audio source status."""
        self.root.after(
            0, lambda: self.audio_status_label.configure(text=source_name, fg=self.CONNECTED_COLOR)
        )

    def set_recording_status(self, is_recording: bool, duration_str: str = "00:00"):
        """Update recording status indicator.

        Args:
            is_recording: Whether recording is active
            duration_str: Duration string like "05:30"
        """
        if is_recording:
            text = f"🔴 REC {duration_str}"
            color = "#f87171"  # Red
        else:
            text = ""
            color = self.STATUS_COLOR

        self.root.after(0, lambda: self.recording_label.configure(text=text, fg=color))

    def close(self):
        """Close the window."""
        self.root.quit()
        self.root.destroy()

    def mainloop(self):
        """Start the Tkinter main loop."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self._handle_close()

    def after(self, ms: int, func: Callable):
        """Schedule a function to run after delay."""
        self.root.after(ms, func)
