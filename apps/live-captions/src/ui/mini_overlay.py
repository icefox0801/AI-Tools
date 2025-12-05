"""Mini overlay window for recording indicator."""

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


class MiniOverlay:
    """
    Tiny overlay window for minimal status display.

    Shows a small pill-shaped indicator with recording time or status.
    """

    # UI Constants
    BG_COLOR = "#1a1a1a"
    TEXT_COLOR = "#ffffff"
    RECORDING_COLOR = "#ef4444"  # Red
    IDLE_COLOR = "#666666"  # Gray

    # Size
    PILL_WIDTH = 100
    PILL_HEIGHT = 32

    def __init__(
        self,
        on_close: Callable[[], None] | None = None,
        on_expand: Callable[[], None] | None = None,
    ):
        """
        Initialize the mini overlay.

        Args:
            on_close: Callback when window is closed
            on_expand: Callback when user double-clicks to expand
        """
        self.on_close = on_close
        self.on_expand = on_expand
        self.is_recording = False
        self.recording_start_time = None
        self.update_job = None

        # Drag state
        self.drag_start_x = 0
        self.drag_start_y = 0

        # Create UI
        self._create_window()

    def _create_window(self):
        """Create the mini overlay window."""
        self.root = tk.Tk()
        self.root.title("Live Captions - Mini")

        # Window settings - small pill shape
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.9)
        self.root.overrideredirect(True)
        self.root.configure(bg=self.BG_COLOR)

        # Position - bottom right corner
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = screen_width - self.PILL_WIDTH - 20
        y = screen_height - self.PILL_HEIGHT - 60
        self.root.geometry(f"{self.PILL_WIDTH}x{self.PILL_HEIGHT}+{x}+{y}")

        # Make it rounded (using a canvas)
        self.canvas = tk.Canvas(
            self.root,
            width=self.PILL_WIDTH,
            height=self.PILL_HEIGHT,
            bg=self.BG_COLOR,
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)

        # Draw rounded rectangle
        self._draw_pill()

        # Bindings
        self.canvas.bind("<Button-1>", self._start_drag)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<Button-3>", lambda e: self._handle_close())
        self.canvas.bind("<Double-Button-1>", lambda e: self._handle_expand())
        self.canvas.bind("<Escape>", lambda e: self._handle_close())

        # Font
        self.status_font = tkfont.Font(family="Segoe UI", size=11, weight="bold")

        # Recording indicator
        self.indicator_id = self.canvas.create_oval(
            8,
            10,
            20,
            22,
            fill=self.IDLE_COLOR,
            outline="",
        )

        # Status text
        self.text_id = self.canvas.create_text(
            55,
            16,
            text="Ready",
            font=self.status_font,
            fill=self.TEXT_COLOR,
            anchor="center",
        )

    def _draw_pill(self):
        """Draw pill-shaped background."""
        radius = self.PILL_HEIGHT // 2
        w = self.PILL_WIDTH
        h = self.PILL_HEIGHT

        # Create rounded rectangle using polygon
        points = [
            radius,
            0,
            w - radius,
            0,
            w,
            0,
            w,
            radius,
            w,
            h - radius,
            w,
            h,
            w - radius,
            h,
            radius,
            h,
            0,
            h,
            0,
            h - radius,
            0,
            radius,
            0,
            0,
        ]

        # Draw with arcs for smooth corners
        self.canvas.create_arc(
            0, 0, radius * 2, h, start=90, extent=180, fill="#2a2a2a", outline=""
        )
        self.canvas.create_arc(
            w - radius * 2, 0, w, h, start=270, extent=180, fill="#2a2a2a", outline=""
        )
        self.canvas.create_rectangle(radius, 0, w - radius, h, fill="#2a2a2a", outline="")

    def _start_drag(self, event):
        """Start window drag."""
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def _on_drag(self, event):
        """Handle window drag motion."""
        x = self.root.winfo_x() + event.x - self.drag_start_x
        y = self.root.winfo_y() + event.y - self.drag_start_y
        self.root.geometry(f"+{x}+{y}")

    def _handle_close(self):
        """Handle window close."""
        if self.update_job:
            self.root.after_cancel(self.update_job)
        if self.on_close:
            self.on_close()

    def _handle_expand(self):
        """Handle double-click to expand."""
        if self.on_expand:
            self.on_expand()

    def _update_timer(self):
        """Update recording timer display."""
        if not self.is_recording or not self.recording_start_time:
            return

        import time

        elapsed = int(time.time() - self.recording_start_time)
        mins = elapsed // 60
        secs = elapsed % 60

        # Blink the indicator
        current_color = self.canvas.itemcget(self.indicator_id, "fill")
        new_color = self.RECORDING_COLOR if current_color == "#8b0000" else "#8b0000"
        self.canvas.itemconfig(self.indicator_id, fill=new_color)

        self.canvas.itemconfig(self.text_id, text=f"🔴 {mins:02d}:{secs:02d}")

        # Schedule next update
        self.update_job = self.root.after(500, self._update_timer)

    # Public API

    def set_recording(self, is_recording: bool):
        """Set recording state."""
        import time

        self.is_recording = is_recording

        if is_recording:
            self.recording_start_time = time.time()
            self.canvas.itemconfig(self.indicator_id, fill=self.RECORDING_COLOR)
            self.canvas.itemconfig(self.text_id, text="🔴 00:00")
            self._update_timer()
        else:
            if self.update_job:
                self.root.after_cancel(self.update_job)
                self.update_job = None
            self.recording_start_time = None
            self.canvas.itemconfig(self.indicator_id, fill=self.IDLE_COLOR)
            self.canvas.itemconfig(self.text_id, text="Ready")

    def set_status(self, text: str):
        """Set status text."""
        self.root.after(0, lambda: self.canvas.itemconfig(self.text_id, text=text))

    def set_idle(self):
        """Set to idle state (both transcription and recording disabled)."""
        self.canvas.itemconfig(self.indicator_id, fill=self.IDLE_COLOR)
        self.canvas.itemconfig(self.text_id, text="Idle")

    def close(self):
        """Close the window."""
        if self.update_job:
            self.root.after_cancel(self.update_job)
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
