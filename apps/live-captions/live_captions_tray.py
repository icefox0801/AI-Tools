#!/usr/bin/env python3
"""
Live Captions Tray - Windows System Tray Application

A Windows system tray application that provides easy access to Live Captions
with different ASR backends.

Features:
- System tray icon with right-click menu
- Double-click to launch with default backend (Whisper)
- Right-click menu to select backends or audio source
- Shows running status in tray tooltip
- High DPI support for crisp icons

Usage:
  python live_captions_tray.py           # Run as tray app
  python live_captions_tray.py --hidden  # Start minimized to tray
"""

import sys
import os
import subprocess
import threading
import logging
import ctypes
from pathlib import Path

# ==============================================================================
# Windows App Identity (must be set before any GUI/tray imports)
# ==============================================================================

# Set Windows App User Model ID so Windows treats this as a unique app
# This allows proper taskbar grouping and notification settings
APP_ID = "LiveCaptions.TrayApp.1.0"

try:
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
except Exception:
    pass

# ==============================================================================
# Windows DPI Awareness (must be set before any GUI imports)
# ==============================================================================

try:
    # Per-Monitor DPI Aware v2 (best quality)
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        # Fall back to System DPI Aware
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# Try to import pystray
try:
    import pystray
    from PIL import Image, ImageDraw
except ImportError:
    print("Required packages not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pystray", "Pillow"])
    import pystray
    from PIL import Image, ImageDraw

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.config import BACKENDS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Dependency Checking
# ==============================================================================

def check_backend_health(backend: str, timeout: float = 2.0) -> tuple[bool, str]:
    """Check if a backend service is healthy.
    
    Args:
        backend: Backend name ('whisper', 'parakeet', 'vosk')
        timeout: Connection timeout in seconds
        
    Returns:
        Tuple of (is_healthy, status_message)
    """
    import socket
    import urllib.request
    import urllib.error
    
    config = BACKENDS.get(backend)
    if not config:
        return False, "Unknown backend"
    
    host = config["host"]
    port = config["port"]
    
    # First check if port is open
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result != 0:
            return False, f"Service not running (port {port} closed)"
    except Exception as e:
        return False, f"Connection failed: {e}"
    
    # Try to get service info via HTTP
    try:
        url = f"http://{host}:{port}/"
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                return True, "Ready"
    except urllib.error.HTTPError as e:
        # HTTP error but server is responding
        return True, "Ready"
    except Exception:
        # Port is open but HTTP failed - still consider it ready
        return True, "Ready (no HTTP)"
    
    return True, "Ready"


def check_all_backends() -> dict[str, tuple[bool, str]]:
    """Check health of all backends.
    
    Returns:
        Dict mapping backend name to (is_healthy, status_message)
    """
    results = {}
    for backend in BACKENDS:
        results[backend] = check_backend_health(backend)
    return results

# ==============================================================================
# Configuration
# ==============================================================================

APP_NAME = "Live Captions"
DEFAULT_BACKEND = "whisper"

# Detect if running as frozen executable (PyInstaller)
IS_FROZEN = getattr(sys, 'frozen', False)

# Always use the source location for live_captions.py
# When frozen: .exe is in dist/, source is in parent (apps/live-captions/)
# When script: source is in same directory
if IS_FROZEN:
    # .exe is in dist/, go up one level to find source
    SCRIPT_DIR = Path(sys.executable).parent.parent
else:
    SCRIPT_DIR = Path(__file__).parent

MAIN_SCRIPT = SCRIPT_DIR / "live_captions.py"

# Find Python executable
def find_python() -> str:
    """Find a working Python executable."""
    # Common Python locations on Windows
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WindowsApps" / "python.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python312" / "python.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python311" / "python.exe",
        Path("C:/Python312/python.exe"),
        Path("C:/Python311/python.exe"),
        Path("C:/Python310/python.exe"),
    ]
    
    for path in candidates:
        if path.exists():
            return str(path)
    
    # Fall back to PATH
    return "python"

PYTHON_EXE = find_python()

ICON_PATH = SCRIPT_DIR / "icon.ico"

# Backend display names
BACKEND_LABELS = {
    "whisper": "🎙️ Whisper (GPU, Multilingual)",
    "parakeet": "🎙️ Parakeet (GPU, English)",
    "vosk": "🎙️ Vosk (CPU, Lightweight)",
}


# ==============================================================================
# Tray Application
# ==============================================================================

class LiveCaptionsTray:
    """System tray application for Live Captions."""
    
    def __init__(self):
        self.current_process = None
        self.current_backend = None
        self.use_system_audio = True  # Default to system audio
        self.icon = None
        self.backend_status: dict[str, tuple[bool, str]] = {}  # Cache backend health
        self._running = True  # For background thread
        self._last_running_state = False  # Track state changes
        self._check_backends()  # Initial check
        
        # Start background status monitor
        self._status_thread = threading.Thread(target=self._monitor_status, daemon=True)
        self._status_thread.start()
    
    def _check_backends(self):
        """Check all backend services health."""
        self.backend_status = check_all_backends()
        for backend, (healthy, msg) in self.backend_status.items():
            status = "✓" if healthy else "✗"
            logger.info(f"Backend {backend}: {status} {msg}")
    
    def refresh_backend_status(self):
        """Refresh backend status (called from menu)."""
        logger.info("Refreshing backend status...")
        self._check_backends()
        self.update_icon()
    
    def _monitor_status(self):
        """Background thread to monitor process status and sync icon."""
        import time
        while self._running:
            try:
                # Check if running state changed
                current_running = self.is_running()
                if current_running != self._last_running_state:
                    logger.info(f"Status changed: {'Running' if current_running else 'Stopped'}")
                    self._last_running_state = current_running
                    self.update_icon()
            except Exception as e:
                logger.debug(f"Status monitor error: {e}")
            
            # Check every 1 second
            time.sleep(1)
    
    def is_backend_available(self, backend: str) -> bool:
        """Check if a specific backend is available."""
        healthy, _ = self.backend_status.get(backend, (False, "Unknown"))
        return healthy
    
    def get_backend_status_text(self, backend: str) -> str:
        """Get status text for a backend."""
        healthy, msg = self.backend_status.get(backend, (False, "Unknown"))
        return msg
        
    def create_icon_image(self, running: bool = False) -> Image.Image:
        """Create high-resolution tray icon image.
        
        Args:
            running: If True, show green indicator; otherwise gray
            
        Returns:
            256x256 RGBA image for crisp display on high-DPI screens
        """
        # Use 256x256 for high DPI displays (Windows scales down as needed)
        size = 256
        
        # Try to load custom icon
        if ICON_PATH.exists():
            try:
                img = Image.open(ICON_PATH)
                # Get the largest size from ICO or resize
                if hasattr(img, 'n_frames') and img.n_frames > 1:
                    # ICO with multiple sizes - get largest
                    img.seek(img.n_frames - 1)
                img = img.resize((size, size), Image.Resampling.LANCZOS)
                
                # Add running indicator overlay if needed
                if running:
                    img = self._add_running_indicator(img, size)
                return img
            except Exception as e:
                logger.warning(f"Failed to load icon: {e}")
        
        # Create default high-res icon
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Scale factor for coordinates
        s = size / 64  # Original design was 64x64
        
        # Background circle with anti-aliasing (draw larger, looks smoother)
        bg_color = (74, 222, 128) if running else (100, 100, 100)  # Green or gray
        padding = int(4 * s)
        draw.ellipse([padding, padding, size - padding, size - padding], fill=bg_color)
        
        # Microphone symbol (scaled up)
        center = size // 2
        mic_color = (255, 255, 255)
        
        # Mic body (rounded rectangle)
        body_width = int(16 * s)
        body_top = int(14 * s)
        body_bottom = int(38 * s)
        body_radius = int(8 * s)
        draw.rounded_rectangle(
            [center - body_width//2, body_top, center + body_width//2, body_bottom],
            radius=body_radius, fill=mic_color
        )
        
        # Mic base arc
        arc_size = int(16 * s)
        arc_top = int(24 * s)
        arc_bottom = int(48 * s)
        arc_width = max(3, int(3 * s))
        draw.arc(
            [center - arc_size, arc_top, center + arc_size, arc_bottom],
            start=0, end=180, fill=mic_color, width=arc_width
        )
        
        # Mic stand
        stand_top = int(48 * s)
        stand_bottom = int(54 * s)
        stand_width = max(3, int(3 * s))
        draw.line([center, stand_top, center, stand_bottom], fill=mic_color, width=stand_width)
        
        # Stand base
        base_width = int(10 * s)
        base_y = int(54 * s)
        draw.line([center - base_width, base_y, center + base_width, base_y], fill=mic_color, width=stand_width)
        
        return img
    
    def _add_running_indicator(self, img: Image.Image, size: int) -> Image.Image:
        """Add a green dot indicator to show running state."""
        img = img.copy()
        draw = ImageDraw.Draw(img)
        
        # Green dot in bottom-right corner
        dot_size = size // 4
        dot_x = size - dot_size - 8
        dot_y = size - dot_size - 8
        
        # White border
        draw.ellipse(
            [dot_x - 4, dot_y - 4, dot_x + dot_size + 4, dot_y + dot_size + 4],
            fill=(255, 255, 255)
        )
        # Green fill
        draw.ellipse(
            [dot_x, dot_y, dot_x + dot_size, dot_y + dot_size],
            fill=(74, 222, 128)
        )
        
        return img
    
    def update_icon(self):
        """Update tray icon based on running state."""
        if self.icon:
            running = self.current_process is not None and self.current_process.poll() is None
            self.icon.icon = self.create_icon_image(running)
            
            # Update tooltip
            if running:
                backend_name = BACKENDS.get(self.current_backend, {}).get("name", self.current_backend)
                audio = "System Audio" if self.use_system_audio else "Microphone"
                self.icon.title = f"{APP_NAME} - {backend_name} ({audio})"
            else:
                self.icon.title = f"{APP_NAME} - Stopped"
    
    def is_running(self) -> bool:
        """Check if Live Captions is currently running."""
        return self.current_process is not None and self.current_process.poll() is None
    
    def start_captions(self, backend: str = DEFAULT_BACKEND):
        """Start Live Captions with specified backend.
        
        Args:
            backend: ASR backend name ('whisper', 'parakeet', 'vosk')
        """
        # Always re-check backend availability before starting (don't use cache)
        is_available, status = check_backend_health(backend)
        
        if not is_available:
            logger.error(f"Cannot start: {backend} is not available ({status})")
            # Update cached status
            self.backend_status[backend] = (is_available, status)
            # Show notification if possible
            if self.icon:
                try:
                    self.icon.notify(
                        f"{backend.title()} is not available",
                        f"Status: {status}\n\nPlease start the Docker service first."
                    )
                except Exception:
                    pass  # Notification not supported
            return
        
        # Update cached status (it's available)
        self.backend_status[backend] = (is_available, status)
        
        # Stop existing process if running
        self.stop_captions()
        
        self.current_backend = backend
        
        # Build command - use PYTHON_EXE for both frozen and script mode
        cmd = [
            PYTHON_EXE,
            str(MAIN_SCRIPT),
            "--backend", backend
        ]
        
        if self.use_system_audio:
            cmd.append("--system-audio")
        
        logger.info(f"Starting Live Captions: {' '.join(cmd)}")
        
        try:
            # Start process (hide console window on Windows)
            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            
            self.current_process = subprocess.Popen(
                cmd,
                startupinfo=startupinfo,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info(f"Started with PID: {self.current_process.pid}")
            
            # Schedule icon update
            threading.Timer(0.5, self.update_icon).start()
            
        except Exception as e:
            logger.error(f"Failed to start: {e}")
            self.current_process = None
    
    def stop_captions(self):
        """Stop running Live Captions."""
        if self.current_process:
            try:
                self.current_process.terminate()
                self.current_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.current_process.kill()
            except Exception as e:
                logger.error(f"Error stopping process: {e}")
            
            self.current_process = None
            self.current_backend = None
            logger.info("Stopped Live Captions")
            
            # Update icon
            self.update_icon()
    
    def toggle_audio_source(self):
        """Toggle between system audio and microphone."""
        self.use_system_audio = not self.use_system_audio
        source = "System Audio" if self.use_system_audio else "Microphone"
        logger.info(f"Audio source: {source}")
        
        # Restart if running
        if self.is_running():
            self.start_captions(self.current_backend)
    
    def on_double_click(self, icon, item):
        """Handle double-click on tray icon."""
        if self.is_running():
            self.stop_captions()
        else:
            self.start_captions(DEFAULT_BACKEND)
    
    def create_menu(self):
        """Create right-click context menu."""
        
        def make_start_handler(backend):
            """Create handler for starting specific backend."""
            def handler(icon, item):
                self.start_captions(backend)
            return handler
        
        def is_backend_checked(backend):
            """Check if backend is currently running."""
            def check(item):
                return self.is_running() and self.current_backend == backend
            return check
        
        def is_backend_enabled(backend):
            """Check if backend can be started (is available)."""
            def check(item):
                return self.is_backend_available(backend)
            return check
        
        def get_backend_label(backend):
            """Get backend label with status indicator."""
            def label(item):
                base_label = BACKEND_LABELS[backend]
                if self.is_backend_available(backend):
                    return f"✓ {base_label}"
                else:
                    status = self.get_backend_status_text(backend)
                    return f"✗ {base_label} - {status}"
            return label
        
        def is_system_audio(item):
            return self.use_system_audio
        
        def is_microphone(item):
            return not self.use_system_audio
        
        # Count available backends
        def get_available_count():
            return sum(1 for b in BACKENDS if self.is_backend_available(b))
        
        # Build menu
        menu = pystray.Menu(
            # Status section
            pystray.MenuItem(
                lambda text: f"● Running: {BACKENDS.get(self.current_backend, {}).get('name', 'None')}" 
                             if self.is_running() else "○ Stopped",
                None,
                enabled=False
            ),
            pystray.MenuItem(
                lambda text: f"Services: {get_available_count()}/{len(BACKENDS)} available",
                None,
                enabled=False
            ),
            pystray.Menu.SEPARATOR,
            
            # Backend selection submenu
            pystray.MenuItem(
                "Start with...",
                pystray.Menu(
                    pystray.MenuItem(
                        get_backend_label("whisper"),
                        make_start_handler("whisper"),
                        checked=is_backend_checked("whisper"),
                        enabled=is_backend_enabled("whisper"),
                        radio=True
                    ),
                    pystray.MenuItem(
                        get_backend_label("parakeet"),
                        make_start_handler("parakeet"),
                        checked=is_backend_checked("parakeet"),
                        enabled=is_backend_enabled("parakeet"),
                        radio=True
                    ),
                    pystray.MenuItem(
                        get_backend_label("vosk"),
                        make_start_handler("vosk"),
                        checked=is_backend_checked("vosk"),
                        enabled=is_backend_enabled("vosk"),
                        radio=True
                    ),
                    pystray.Menu.SEPARATOR,
                    pystray.MenuItem(
                        "🔄 Refresh Status",
                        lambda icon, item: self.refresh_backend_status()
                    ),
                )
            ),
            
            # Audio source submenu
            pystray.MenuItem(
                "Audio Source",
                pystray.Menu(
                    pystray.MenuItem(
                        "🔊 System Audio (Speakers)",
                        lambda icon, item: setattr(self, 'use_system_audio', True) or 
                                          (self.start_captions(self.current_backend) if self.is_running() else None),
                        checked=is_system_audio,
                        radio=True
                    ),
                    pystray.MenuItem(
                        "🎤 Microphone",
                        lambda icon, item: setattr(self, 'use_system_audio', False) or 
                                          (self.start_captions(self.current_backend) if self.is_running() else None),
                        checked=is_microphone,
                        radio=True
                    ),
                )
            ),
            
            pystray.Menu.SEPARATOR,
            
            # Quick actions
            pystray.MenuItem(
                "Stop",
                lambda icon, item: self.stop_captions(),
                visible=lambda item: self.is_running()
            ),
            pystray.MenuItem(
                lambda text: f"Quick Start ({DEFAULT_BACKEND.title()})" 
                             if self.is_backend_available(DEFAULT_BACKEND)
                             else f"Quick Start ({DEFAULT_BACKEND.title()}) - Unavailable",
                lambda icon, item: self.start_captions(DEFAULT_BACKEND),
                default=True,  # Double-click action
                visible=lambda item: not self.is_running(),
                enabled=lambda item: self.is_backend_available(DEFAULT_BACKEND)
            ),
            
            pystray.Menu.SEPARATOR,
            
            # Exit
            pystray.MenuItem("Exit", self.quit)
        )
        
        return menu
    
    def quit(self, icon, item):
        """Exit the application."""
        logger.info("Exiting tray application")
        self._running = False  # Stop background thread
        self.stop_captions()
        icon.stop()
    
    def run(self):
        """Run the tray application."""
        logger.info(f"Starting {APP_NAME} Tray")
        logger.info(f"Default backend: {DEFAULT_BACKEND}")
        logger.info("Double-click icon to start/stop")
        logger.info("Right-click for options")
        
        # Create tray icon with a consistent name for Windows to identify
        # Using APP_NAME ensures Windows can track this in taskbar settings
        self.icon = pystray.Icon(
            name=APP_NAME,  # Must be consistent for Windows taskbar settings
            icon=self.create_icon_image(running=False),
            title=f"{APP_NAME} - Stopped",
            menu=self.create_menu()
        )
        
        # Run (blocks until quit)
        self.icon.run()


# ==============================================================================
# Entry Point
# ==============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description=f"{APP_NAME} Tray Application")
    parser.add_argument('--auto-start', action='store_true',
                        help='Automatically start captions on launch')
    parser.add_argument('--backend', choices=['whisper', 'parakeet', 'vosk'],
                        default=DEFAULT_BACKEND,
                        help=f'Default backend (default: {DEFAULT_BACKEND})')
    args = parser.parse_args()
    
    # Create and run tray app
    app = LiveCaptionsTray()
    
    if args.auto_start:
        # Start captions after tray is ready
        threading.Timer(1.0, lambda: app.start_captions(args.backend)).start()
    
    app.run()


if __name__ == "__main__":
    main()
