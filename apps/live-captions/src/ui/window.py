"""Caption overlay window for Live Captions."""

import tkinter as tk
from tkinter import font as tkfont
import ctypes
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Enable high DPI awareness on Windows
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


class CaptionWindow:
    """Transparent overlay window for displaying captions."""
    
    # UI Constants
    DEFAULT_FONT_SIZE = 36
    MIN_FONT_SIZE = 16
    MAX_FONT_SIZE = 72
    DEFAULT_ALPHA = 0.85
    BG_COLOR = '#1a1a1a'
    TEXT_COLOR = '#ffffff'
    TEXT_COLOR_DIM = '#999999'
    STATUS_COLOR = '#888888'
    HINT_COLOR = '#555555'
    CONNECTED_COLOR = '#4ade80'
    DISCONNECTED_COLOR = '#f87171'
    
    def __init__(
        self, 
        model_display: str = "",
        on_close: Optional[Callable[[], None]] = None
    ):
        """
        Initialize the caption window.
        
        Args:
            model_display: Text to show for model info
            on_close: Callback when window is closed
        """
        self.on_close = on_close
        self.model_display = model_display
        
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
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', self.DEFAULT_ALPHA)
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
        self.root.bind('<Button-1>', self._start_drag)
        self.root.bind('<B1-Motion>', self._on_drag)
        self.root.bind('<Button-3>', lambda e: self._handle_close())
        self.root.bind('<Escape>', lambda e: self._handle_close())
        self.root.bind('<MouseWheel>', self._on_mousewheel)
        
        # Fonts
        self.caption_font = tkfont.Font(
            family="Segoe UI", 
            size=self.base_font_size, 
            weight="bold"
        )
        
        # Container
        self.container = tk.Frame(self.root, bg=self.BG_COLOR)
        self.container.pack(expand=True, fill='both', padx=15, pady=(15, 30))
        
        # Caption lines
        self.line1 = tk.Label(
            self.container, text="", font=self.caption_font,
            fg=self.TEXT_COLOR_DIM, bg=self.BG_COLOR, anchor='w'
        )
        self.line1.pack(expand=True, fill='both')
        
        self.line2 = tk.Label(
            self.container, text="🎙️ Starting...", font=self.caption_font,
            fg=self.TEXT_COLOR, bg=self.BG_COLOR, anchor='w'
        )
        self.line2.pack(expand=True, fill='both')
        
        # Status labels
        self._create_status_labels()
    
    def _create_status_labels(self):
        """Create status indicator labels."""
        status_font = tkfont.Font(size=9)
        
        # Connection status (top right)
        self.status_label = tk.Label(
            self.root, text="● Initializing...", font=status_font,
            fg=self.STATUS_COLOR, bg=self.BG_COLOR
        )
        self.status_label.place(relx=1.0, y=5, anchor='ne', x=-10)
        
        # Audio status (top left)
        self.audio_status_label = tk.Label(
            self.root, text="🎙️ Off", font=status_font,
            fg=self.STATUS_COLOR, bg=self.BG_COLOR
        )
        self.audio_status_label.place(x=10, y=5, anchor='nw')
        
        # Hint (bottom center)
        self.hint_label = tk.Label(
            self.root, text="Drag to move • Scroll to resize • Right-click or Esc to close",
            font=status_font, fg=self.HINT_COLOR, bg=self.BG_COLOR
        )
        self.hint_label.place(relx=0.5, rely=1.0, anchor='s', y=-5)
        
        # Model info (top center)
        self.model_label = tk.Label(
            self.root, text=self.model_display,
            font=status_font, fg=self.STATUS_COLOR, bg=self.BG_COLOR
        )
        self.model_label.place(relx=0.5, y=5, anchor='n')
    
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
    
    def _handle_close(self):
        """Handle window close."""
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
    
    # Public API
    
    def update_text(self, text: str):
        """
        Update displayed caption text.
        
        Args:
            text: Full text to display (will be wrapped and show last 2 lines)
        """
        if not text or not text.strip():
            return
        
        lines = self._wrap_text(text.strip())
        if not lines:
            return
        
        if len(lines) >= 2:
            line1_text = lines[-2]
            line2_text = lines[-1]
        else:
            line1_text = ""
            line2_text = lines[-1]
        
        self.root.after(0, lambda: self._set_lines(line1_text, line2_text))
    
    def _set_lines(self, line1: str, line2: str):
        """Set caption line text."""
        self.line1.configure(text=line1)
        self.line2.configure(text=line2)
        self.line1.pack_configure(padx=(20, 0))
        self.line2.pack_configure(padx=(20, 0))
    
    def set_message(self, message: str):
        """Set a message on the second line (clears first line)."""
        self.root.after(0, lambda: self._set_lines("", message))
    
    def set_connection_status(self, connected: bool):
        """Update connection status indicator."""
        if connected:
            self.root.after(0, lambda: self.status_label.configure(
                text="● Connected", fg=self.CONNECTED_COLOR
            ))
        else:
            self.root.after(0, lambda: self.status_label.configure(
                text="● Disconnected", fg=self.DISCONNECTED_COLOR
            ))
    
    def set_audio_status(self, source_name: str):
        """Update audio source status."""
        self.root.after(0, lambda: self.audio_status_label.configure(
            text=source_name, fg=self.CONNECTED_COLOR
        ))
    
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
