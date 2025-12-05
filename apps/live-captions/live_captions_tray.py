#!/usr/bin/env python3
"""
Live Captions Tray - Windows System Tray Application

A Windows system tray application that provides easy access to Live Captions
with different ASR backends.

Features:
- System tray icon with right-click menu
- Double-click to start/stop with default backend (Whisper)
- Right-click menu to select backends or audio source
- Shows running status in tray tooltip
- High DPI support for crisp icons

Usage:
  python live_captions_tray.py           # Run as tray app
  python live_captions_tray.py --hidden  # Start minimized to tray
"""

import contextlib
import ctypes
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

# ==============================================================================
# Windows App Identity (must be set before any GUI/tray imports)
# ==============================================================================

# Set Windows App User Model ID so Windows treats this as a unique app
# This allows proper taskbar grouping and notification settings
APP_ID = "LiveCaptions.TrayApp.1.0"

with contextlib.suppress(Exception):
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)

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
from src.audio.recorder import read_recording_status

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
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
    import urllib.error
    import urllib.request

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
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                return True, "Ready"
    except urllib.error.HTTPError:
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
IS_FROZEN = getattr(sys, "frozen", False)

# Always use the source location for live_captions.py
# When frozen: .exe is in dist/Live Captions/, source is in apps/live-captions/
# When script: source is in same directory
if IS_FROZEN:
    # .exe is in dist/Live Captions/, go up two levels to find source (apps/live-captions/)
    SCRIPT_DIR = Path(sys.executable).parent.parent.parent
else:
    SCRIPT_DIR = Path(__file__).parent

MAIN_SCRIPT = SCRIPT_DIR / "live_captions.py"

# Just use python from PATH - the user's environment is already configured
PYTHON_EXE = "python"

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

# Audio Notes API URL (for uploading recordings)
AUDIO_NOTES_URL = os.getenv("AUDIO_NOTES_URL", "http://localhost:7860")


class LiveCaptionsTray:
    """System tray application for Live Captions."""

    def __init__(self):
        self.current_process = None
        self.current_backend = None
        self.use_system_audio = True  # Default to system audio
        self.enable_recording = True  # Default to recording enabled
        self.enable_transcription = True  # Default to live transcription enabled
        self.icon = None
        self.backend_status: dict[str, tuple[bool, str]] = {}  # Cache backend health
        self._running = True  # For background thread
        self._last_running_state = False  # Track state changes
        self._animation_frame = 0  # For pulsing animation
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
        """Background thread to monitor process status, recording, and sync icon."""
        import time

        last_recording_duration = ""

        while self._running:
            try:
                # Check if running state changed
                current_running = self.is_running()
                if current_running != self._last_running_state:
                    logger.info(f"Status changed: {'Running' if current_running else 'Stopped'}")
                    self._last_running_state = current_running

                # Check recording status
                is_recording, duration_str, _ = self.get_recording_info()
                duration_changed = is_recording and duration_str != last_recording_duration
                recording_stopped = not is_recording and last_recording_duration

                if duration_changed:
                    last_recording_duration = duration_str
                elif recording_stopped:
                    last_recording_duration = ""

                # Animate icon when running - advance frame every 500ms
                # Also update when recording duration changes (for tooltip)
                if current_running:
                    new_frame = (self._animation_frame + 1) % 3
                    self._animation_frame = new_frame
                    self.update_icon()  # Always update when running (for animation + tooltip)
                elif self._animation_frame != 0:
                    # Reset animation when stopped
                    self._animation_frame = 0
                    self.update_icon()

            except Exception as e:
                logger.debug(f"Status monitor error: {e}")

            # Sleep 500ms between updates
            time.sleep(0.5)

    def is_backend_available(self, backend: str) -> bool:
        """Check if a specific backend is available."""
        healthy, _ = self.backend_status.get(backend, (False, "Unknown"))
        return healthy

    def get_backend_status_text(self, backend: str) -> str:
        """Get status text for a backend."""
        _healthy, msg = self.backend_status.get(backend, (False, "Unknown"))
        return msg

    def create_icon_image(self, running: bool = False, recording: bool = False) -> Image.Image:
        """Create high-resolution tray icon image with vertical loading animation.

        Args:
            running: If True, show vertical loading animation on mic body
            recording: If True, show recording state (same animation)

        Returns:
            256x256 RGBA image for crisp display on high-DPI screens
        """
        # Use 256x256 for high DPI displays (Windows scales down as needed)
        size = 256

        # Dark background like Docker icon
        bg_color = (36, 41, 46)

        # Create icon
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Scale factor for coordinates
        s = size / 64  # Original design was 64x64
        padding = int(4 * s)

        # Background circle
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
            [center - body_width // 2, body_top, center + body_width // 2, body_bottom],
            radius=body_radius,
            fill=mic_color,
        )

        # Vertical loading animation on mic body when running/recording
        if running or recording:
            # Animation has 3 frames that loop: show 1 bar, 2 bars, 3 bars
            frame = self._animation_frame % 3  # 0, 1, 2

            # Bar dimensions (inside the mic body) - wider bars for visibility
            bar_width = int(12 * s)
            bar_height = int(5 * s)
            bar_x = center - bar_width // 2

            # Three bars positioned vertically in the mic body
            bar_positions = [
                int(17 * s),  # Top bar
                int(24 * s),  # Middle bar
                int(31 * s),  # Bottom bar
            ]

            # Choose color based on recording state
            bar_color = (
                (255, 60, 60) if recording else (40, 120, 70)
            )  # Red if recording, dark green if running

            # Draw bars based on current frame (incremental: 1, 2, 3 bars)
            for i in range(frame + 1):  # frame 0 = 1 bar, frame 1 = 2 bars, frame 2 = 3 bars
                bar_y = bar_positions[i]
                draw.rounded_rectangle(
                    [bar_x, bar_y, bar_x + bar_width, bar_y + bar_height],
                    radius=int(2 * s),
                    fill=bar_color,
                )

        # Mic base arc
        arc_size = int(16 * s)
        arc_top = int(24 * s)
        arc_bottom = int(48 * s)
        arc_width = max(3, int(3 * s))
        draw.arc(
            [center - arc_size, arc_top, center + arc_size, arc_bottom],
            start=0,
            end=180,
            fill=mic_color,
            width=arc_width,
        )

        # Mic stand
        stand_top = int(48 * s)
        stand_bottom = int(54 * s)
        stand_width = max(3, int(3 * s))
        draw.line([center, stand_top, center, stand_bottom], fill=mic_color, width=stand_width)

        # Stand base
        base_width = int(10 * s)
        base_y = int(54 * s)
        draw.line(
            [center - base_width, base_y, center + base_width, base_y],
            fill=mic_color,
            width=stand_width,
        )

        return img

    def update_icon(self):
        """Update tray icon based on running and recording state."""
        if self.icon:
            running = self.current_process is not None and self.current_process.poll() is None
            is_recording, duration_str, _ = self.get_recording_info()

            # Create animated icon (pulses green when running, red when recording)
            img = self.create_icon_image(running, is_recording)
            self.icon.icon = img

            # Update tooltip with recording info
            if running:
                backend_name = BACKENDS.get(self.current_backend, {}).get(
                    "name", self.current_backend
                )
                audio = "System Audio" if self.use_system_audio else "Microphone"

                if is_recording:
                    self.icon.title = (
                        f"{APP_NAME} - {backend_name} ({audio})\n🔴 Recording: {duration_str}"
                    )
                else:
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
                        f"Status: {status}\n\nPlease start the Docker service first.",
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
        cmd = [PYTHON_EXE, str(MAIN_SCRIPT), "--backend", backend]

        if self.use_system_audio:
            cmd.append("--system-audio")

        if not self.enable_recording:
            cmd.append("--no-recording")

        if not self.enable_transcription:
            cmd.append("--no-transcription")

        # Use Audio Notes API for saving recordings
        cmd.extend(["--audio-notes-url", AUDIO_NOTES_URL])

        logger.info(f"Starting Live Captions: {' '.join(cmd)}")
        logger.info(f"Working directory: {SCRIPT_DIR}")

        try:
            # Start process (hide console window on Windows)
            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

            self.current_process = subprocess.Popen(
                cmd,
                cwd=str(SCRIPT_DIR),  # Set working directory to script location
                startupinfo=startupinfo,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            logger.info(f"Started with PID: {self.current_process.pid}")

            # Start thread to monitor stderr for errors
            def log_stderr():
                if self.current_process and self.current_process.stderr:
                    for line in self.current_process.stderr:
                        try:
                            logger.error(f"Subprocess: {line.decode().strip()}")
                        except Exception:
                            pass

            stderr_thread = threading.Thread(target=log_stderr, daemon=True)
            stderr_thread.start()

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

    def get_recording_info(self) -> tuple[bool, str, float]:
        """Get current recording info from status file (IPC with subprocess).

        Returns:
            Tuple of (is_recording, duration_str, duration_seconds)
        """
        return read_recording_status()

    def clear_recording(self):
        """Clear the current recording.

        Note: When running as subprocess, clearing from tray is not supported.
        The recording can only be cleared from within the Live Captions window.
        """
        # When running via subprocess, we can't directly access the recorder
        # This would require implementing a more complex IPC mechanism
        logger.info("Clear recording from tray not supported when running as subprocess")
        if self.icon:
            self.icon.notify("Clear Recording", "Use the Live Captions window to clear recording.")

    def toggle_audio_source(self):
        """Toggle between system audio and microphone."""
        self.use_system_audio = not self.use_system_audio
        source = "System Audio" if self.use_system_audio else "Microphone"
        logger.info(f"Audio source: {source}")

        # Restart if running
        if self.is_running():
            self.start_captions(self.current_backend)

    def toggle_transcription(self):
        """Toggle live transcription on/off."""
        self.enable_transcription = not self.enable_transcription
        logger.info(f"Live transcription: {'enabled' if self.enable_transcription else 'disabled'}")

        # Restart if running to apply change
        if self.is_running():
            self.start_captions(self.current_backend)

    def can_start(self) -> bool:
        """Check if we can start (at least one of recording or transcription enabled)."""
        return self.enable_recording or self.enable_transcription

    def on_click(self, icon, item):
        """Handle single left-click on tray icon to start/stop."""
        if self.is_running():
            self.stop_captions()
        else:
            # Check if at least one mode is enabled
            if not self.can_start():
                logger.warning("Cannot start: both recording and transcription are disabled")
                if self.icon:
                    try:
                        self.icon.notify(
                            "Cannot Start",
                            "Enable Recording or Live Transcription first.",
                        )
                    except Exception:
                        pass
                return
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
                lambda text: (
                    f"● Running: {BACKENDS.get(self.current_backend, {}).get('name', 'None')}"
                    if self.is_running()
                    else "○ Stopped"
                ),
                None,
                enabled=False,
            ),
            pystray.MenuItem(
                lambda text: f"Services: {get_available_count()}/{len(BACKENDS)} available",
                None,
                enabled=False,
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
                        radio=True,
                    ),
                    pystray.MenuItem(
                        get_backend_label("parakeet"),
                        make_start_handler("parakeet"),
                        checked=is_backend_checked("parakeet"),
                        enabled=is_backend_enabled("parakeet"),
                        radio=True,
                    ),
                    pystray.MenuItem(
                        get_backend_label("vosk"),
                        make_start_handler("vosk"),
                        checked=is_backend_checked("vosk"),
                        enabled=is_backend_enabled("vosk"),
                        radio=True,
                    ),
                    pystray.Menu.SEPARATOR,
                    pystray.MenuItem(
                        "🔄 Refresh Status", lambda icon, item: self.refresh_backend_status()
                    ),
                ),
            ),
            # Audio source submenu
            pystray.MenuItem(
                "Audio Source",
                pystray.Menu(
                    pystray.MenuItem(
                        "🔊 System Audio (Speakers)",
                        lambda icon, item: setattr(self, "use_system_audio", True)
                        or (
                            self.start_captions(self.current_backend) if self.is_running() else None
                        ),
                        checked=is_system_audio,
                        radio=True,
                    ),
                    pystray.MenuItem(
                        "🎤 Microphone",
                        lambda icon, item: setattr(self, "use_system_audio", False)
                        or (
                            self.start_captions(self.current_backend) if self.is_running() else None
                        ),
                        checked=is_microphone,
                        radio=True,
                    ),
                ),
            ),
            # Live transcription toggle
            pystray.MenuItem(
                "📝 Live Transcription",
                lambda icon, item: self.toggle_transcription(),
                checked=lambda item: self.enable_transcription,
            ),
            pystray.Menu.SEPARATOR,
            # Recording section
            pystray.MenuItem(
                "🎙️ Enable Recording",
                lambda icon, item: self.toggle_recording(),
                checked=lambda item: self.enable_recording,
            ),
            pystray.MenuItem(
                lambda text: (
                    f"📼 Recording: {self.get_recording_info()[1]}"
                    if self.get_recording_info()[0]
                    else "📼 No recording"
                ),
                None,
                enabled=False,
                visible=lambda item: self.enable_recording,
            ),
            pystray.MenuItem(
                "🗑️ Clear Recording",
                lambda icon, item: self.clear_recording(),
                enabled=lambda item: self.get_recording_info()[2] > 0,
                visible=lambda item: self.enable_recording,
            ),
            pystray.Menu.SEPARATOR,
            # Quick actions - both have default=True so left-click works in either state
            pystray.MenuItem(
                "Stop",
                self.on_click,
                default=True,  # Left-click action when running
                visible=lambda item: self.is_running(),
            ),
            pystray.MenuItem(
                lambda text: (
                    f"Quick Start ({DEFAULT_BACKEND.title()})"
                    if self.is_backend_available(DEFAULT_BACKEND) and self.can_start()
                    else (
                        f"Quick Start ({DEFAULT_BACKEND.title()}) - Unavailable"
                        if self.is_backend_available(DEFAULT_BACKEND)
                        else f"Quick Start ({DEFAULT_BACKEND.title()}) - Service Unavailable"
                    )
                ),
                self.on_click,
                default=True,  # Left-click action when stopped
                visible=lambda item: not self.is_running(),
                enabled=lambda item: self.is_backend_available(DEFAULT_BACKEND)
                and self.can_start(),
            ),
            pystray.Menu.SEPARATOR,
            # Exit
            pystray.MenuItem("Exit", self.quit),
        )

        return menu

    def toggle_recording(self):
        """Toggle recording on/off."""
        self.enable_recording = not self.enable_recording
        logger.info(f"Recording: {'enabled' if self.enable_recording else 'disabled'}")

        # Restart if running to apply change
        if self.is_running():
            self.start_captions(self.current_backend)

    def quit(self, icon, item):
        """Exit the application."""
        logger.info("Exiting tray application")
        self._running = False  # Stop background thread

        # When running as subprocess, the live_captions.py process handles its own cleanup
        # including saving any recording when it exits
        self.stop_captions()
        icon.stop()

    def run(self):
        """Run the tray application."""
        logger.info(f"Starting {APP_NAME} Tray")
        logger.info(f"Default backend: {DEFAULT_BACKEND}")
        logger.info("Left-click icon to start/stop")
        logger.info("Right-click for options")

        # Create tray icon with a consistent name for Windows to identify
        # Using APP_NAME ensures Windows can track this in taskbar settings
        self.icon = pystray.Icon(
            name=APP_NAME,  # Must be consistent for Windows taskbar settings
            icon=self.create_icon_image(running=False),
            title=f"{APP_NAME} - Stopped",
            menu=self.create_menu(),
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
    parser.add_argument(
        "--auto-start", action="store_true", help="Automatically start captions on launch"
    )
    parser.add_argument(
        "--backend",
        choices=["whisper", "parakeet", "vosk"],
        default=DEFAULT_BACKEND,
        help=f"Default backend (default: {DEFAULT_BACKEND})",
    )
    args = parser.parse_args()

    # Create and run tray app
    app = LiveCaptionsTray()

    if args.auto_start:
        # Start captions after tray is ready
        threading.Timer(1.0, lambda: app.start_captions(args.backend)).start()

    app.run()


if __name__ == "__main__":
    main()
