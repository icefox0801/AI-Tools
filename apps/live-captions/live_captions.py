#!/usr/bin/env python3
"""
Live Captions - Standalone Speech-to-Text Overlay v1.1

A frameless, transparent overlay window that captures audio from the microphone
or system audio (via WASAPI loopback), sends it to the ASR service for
transcription, and displays real-time captions.

Features:
- Microphone or system audio capture (WASAPI loopback)
- Works with both Vosk and Parakeet backends
- Clean modular architecture

Usage:
  python live_captions.py                    # Default microphone
  python live_captions.py --system-audio     # Capture system audio
  python live_captions.py --list-devices     # Show available devices
"""

import atexit
import os
import signal
import sys

# Add project root to path for shared module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import asyncio
import logging
import threading
import time
import wave
from datetime import datetime

from shared.client import TranscriptManager

# Import shared modules
from shared.config import BACKEND, get_backend_config, get_display_info
from src.asr import ASRClient

# Import local modules
from src.audio import (
    TARGET_SAMPLE_RATE,
    AudioRecorder,
    MicrophoneCapture,
    SystemAudioCapture,
    check_stop_requested,
    list_devices,
    set_recorder,
)
from src.ui import CaptionWindow

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class LiveCaptions:
    """Main application class coordinating UI, audio capture, and ASR."""

    def __init__(
        self,
        host: str = "localhost",
        port: int | None = None,
        backend: str | None = None,
        language: str = "en",
        use_system_audio: bool = False,
        device_index: int | None = None,
        debug_save_audio: bool = False,
        enable_recording: bool = True,
        enable_transcription: bool = True,
        audio_notes_url: str | None = None,
    ):
        """
        Initialize Live Captions application.

        Args:
            host: ASR service host
            port: ASR service port (uses backend default if None)
            backend: ASR backend ('vosk' or 'parakeet')
            language: Transcription language code (e.g., 'en', 'yue')
            use_system_audio: Use system audio instead of microphone
            device_index: Specific audio device index
            debug_save_audio: Save captured audio on exit
            enable_recording: Enable audio recording for later transcription
            enable_transcription: Enable live transcription (can be disabled if using external ASR)
            audio_notes_url: Audio Notes API URL for uploading recordings
        """
        # Backend config
        self.backend = backend or BACKEND
        self.config = get_backend_config(self.backend)
        self.host = host
        self.port = port or self.config["port"]
        self.language = language

        # State
        self.running = True
        self.enable_transcription = enable_transcription

        # Audio settings
        self.use_system_audio = use_system_audio
        self.device_index = device_index
        self.audio_capture = None

        # Recording
        self.enable_recording = enable_recording
        self.recorder: AudioRecorder | None = None
        if enable_recording:
            self.recorder = AudioRecorder(api_url=audio_notes_url)
            set_recorder(self.recorder)  # Set as global for tray app access

        # Debug
        self.debug_save_audio = debug_save_audio
        self.debug_audio_chunks = []

        # Transcript manager
        self.transcript = TranscriptManager()
        self.transcript.on_change = self._on_transcript_change

        # Async components
        self.loop = None
        self.asr_client = None

        # Determine mode based on settings:
        # - Both disabled: No overlay, exit immediately (handled by tray)
        # - Recording only (no transcription): Headless mode (tray shows status)
        # - Transcription enabled: Full caption window
        self.headless_mode = not enable_transcription
        self.idle_mode = not enable_recording and not enable_transcription

        if self.idle_mode:
            # Both disabled - nothing to do, shouldn't be started
            logger.warning("Both recording and transcription disabled - nothing to do")
            self.window = None
        elif self.headless_mode:
            # Recording only - headless mode, tray shows status
            self.window = None
        else:
            # Full transcription mode
            self.window = CaptionWindow(
                model_display=get_display_info(self.backend),
                language=self.language,
                on_close=self.close,
            )

    def _on_transcript_change(self):
        """Handle transcript updates."""
        if not self.window:
            return  # No overlay to update
        text = self.transcript.get_text()
        self.window.update_text(text)

    def _on_audio_data(self, audio_bytes: bytes):
        """Handle audio data from capture."""
        if self.debug_save_audio:
            self.debug_audio_chunks.append(audio_bytes)

        # Record audio for later transcription
        if self.recorder and self.recorder.is_recording:
            self.recorder.add_chunk(audio_bytes)

        # Only send to ASR if live transcription is enabled
        if self.enable_transcription and self.asr_client:
            self.asr_client.queue_audio(audio_bytes)

    def _on_asr_connected(self, connected: bool):
        """Handle ASR connection status change."""
        if not self.window:
            return  # No caption window

        self.window.set_connection_status(connected)
        if connected:
            self.window.set_message("🎙️ Speak now...")
        else:
            self.window.set_message("⏳ Waiting for ASR service...")

    def _on_asr_transcript(self, segment_id: str, text: str):
        """Handle transcript from ASR service."""
        self.transcript.update(segment_id, text)

    def _start_audio_capture(self):
        """Start audio capture."""
        if self.use_system_audio:
            self.audio_capture = SystemAudioCapture(
                callback=self._on_audio_data, device_index=self.device_index
            )
        else:
            self.audio_capture = MicrophoneCapture(
                callback=self._on_audio_data, device_index=self.device_index
            )

        if self.audio_capture.start():
            if self.window:
                self.window.set_audio_status(self.audio_capture.source_name)

            # Start recording
            if self.recorder:
                self.recorder.start()
                logger.info("Recording started")

                # Start periodic update of recording duration for caption window
                if self.window:
                    self._schedule_window_recording_update()
        else:
            if self.window:
                self.window.set_message("❌ Audio capture failed")
            else:
                logger.error("Audio capture failed")

    def _stop_audio_capture(self):
        """Stop audio capture."""
        if self.audio_capture:
            self.audio_capture.stop()
            self.audio_capture = None

        if self.debug_save_audio and self.debug_audio_chunks:
            self._save_debug_audio()

    def _schedule_window_recording_update(self):
        """Schedule periodic update of recording duration on caption window."""
        if not self.running or not self.window:
            return

        if self.recorder and self.recorder.is_recording:
            duration = self.recorder.duration  # Use property, not method
            mins = int(duration // 60)
            secs = int(duration % 60)
            duration_str = f"{mins:02d}:{secs:02d}"
            self.window.set_recording_status(True, duration_str)
        else:
            self.window.set_recording_status(False)

        # Schedule next update in 1 second
        if self.window and self.running:
            self.window.after(1000, self._schedule_window_recording_update)

    def _save_debug_audio(self):
        """Save captured audio to WAV file for debugging."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"debug_audio_{timestamp}.wav"

            audio_data = b"".join(self.debug_audio_chunks)

            with wave.open(filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(TARGET_SAMPLE_RATE)
                wf.writeframes(audio_data)

            duration = len(audio_data) / (TARGET_SAMPLE_RATE * 2)
            logger.info(f"Saved debug audio: {filename} ({duration:.1f}s)")
            print(f"\n💾 Debug audio saved: {filename} ({duration:.1f}s)")

        except Exception as e:
            logger.error(f"Failed to save debug audio: {e}")

    def close(self):
        """Close the application."""
        # Prevent double-close - only skip if already stopped AND no active recorder
        if not self.running and not (self.recorder and self.recorder.is_recording):
            return

        logger.info("Closing Live Captions...")
        self.running = False

        if self.asr_client:
            try:
                self.asr_client.stop()
            except Exception as e:
                logger.warning(f"Error stopping ASR client: {e}")

        self._stop_audio_capture()

        # Save recording on close
        if self.recorder and self.recorder.is_recording:
            logger.info("Saving recording before shutdown...")
            try:
                saved_path = self.recorder.stop()
                if saved_path:
                    logger.info(f"Recording saved: {saved_path}")
            except Exception as e:
                logger.error(f"Error saving recording: {e}")

        if self.window:
            try:
                self.window.close()
            except Exception as e:
                logger.warning(f"Error closing window: {e}")

    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    def _check_stop_request(self):
        """Periodically check for stop request from tray app (for window mode)."""
        if not self.window:
            return  # Only for window mode

        # Check if stop was requested via IPC file
        if check_stop_requested():
            logger.info("Stop request received from tray app, closing window...")
            self.close()
            return

        # Check again in 100ms
        if self.running:
            self.window.after(100, self._check_stop_request)

    def run(self):
        """Run the application."""
        # Check if idle mode (both disabled) - nothing to do
        if self.idle_mode:
            logger.warning("Nothing to do - both recording and transcription disabled")
            return

        # Headless (recording-only) mode - start audio capture but no window
        if self.headless_mode:
            logger.info("Running in headless mode (recording only, no window)")

            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
            if sys.platform == "win32":
                signal.signal(signal.SIGBREAK, self._signal_handler)

            # Register atexit handler to save recording
            atexit.register(self.close)

            self._start_audio_capture()
            # Keep running until stopped (via signal, keyboard, or stop request file)
            try:
                while self.running:
                    # Check for stop request from tray app (Windows IPC)
                    if check_stop_requested():
                        logger.info("Stop request received from tray app")
                        break
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
            finally:
                self.close()
            return

        # Full transcription mode
        # Start ASR thread
        asr_thread = threading.Thread(target=self._run_async, daemon=True)
        asr_thread.start()

        # Start checking for stop request from tray app
        self._check_stop_request()

        # Run UI main loop
        self.window.mainloop()

    def _run_async(self):
        """Run async event loop in thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Create ASR client
        self.asr_client = ASRClient(
            host=self.host,
            port=self.port,
            chunk_ms=self.config["chunk_ms"],
            language=self.language,
            on_connected=self._on_asr_connected,
            on_transcript=self._on_asr_transcript,
        )

        # Start audio capture after short delay
        if self.window:
            self.window.after(500, self._start_audio_capture)
        else:
            # Headless mode - start immediately (handled in _run_headless)
            pass

        # Run ASR client
        self.loop.run_until_complete(self.asr_client.run())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Live Captions v1.4")
    parser.add_argument("--host", default="localhost", help="ASR service host")
    parser.add_argument("--port", type=int, help="ASR service port")
    parser.add_argument(
        "--backend",
        choices=["vosk", "parakeet", "whisper"],
        default=BACKEND,
        help=f"ASR backend (default: {BACKEND})",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Transcription language code (e.g., 'en', 'yue'). Default: en",
    )
    parser.add_argument(
        "--system-audio", action="store_true", help="Capture system audio instead of microphone"
    )
    parser.add_argument("--device", type=int, help="Microphone device index")
    parser.add_argument(
        "--loopback-device", type=int, help="Loopback device index for system audio"
    )
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices")
    parser.add_argument(
        "--no-recording",
        action="store_true",
        help="Disable audio recording for later transcription",
    )
    parser.add_argument(
        "--no-transcription",
        action="store_true",
        help="Disable live transcription (recording only mode)",
    )
    parser.add_argument(
        "--audio-notes-url", type=str, help="Audio Notes API URL (default: http://localhost:7860)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--debug-save-audio", action="store_true", help="Save captured audio on exit"
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    cfg = get_backend_config(args.backend)
    port = args.port or cfg["port"]

    # Determine device index
    device_index = args.loopback_device if args.system_audio else args.device

    # Language display names
    LANG_NAMES = {"en": "English", "yue": "Cantonese (粵語)"}

    # Print startup banner (ASCII only for Windows console compatibility)
    print("+======================================+")
    print("|        Live Captions v1.2           |")
    print("+======================================+")
    print(f"Model: {cfg['name']} ({cfg['device']})")
    print(f"Language: {LANG_NAMES.get(args.language, args.language)}")
    print(f"ASR: ws://{args.host}:{port}/stream")

    if args.system_audio:
        print("Audio: [Speaker] System Audio (WASAPI loopback)")
    else:
        print("Audio: [Mic] Microphone")

    if not args.no_recording:
        print("Recording: ENABLED (uploads to Audio Notes)")
        if args.audio_notes_url:
            print(f"Audio Notes URL: {args.audio_notes_url}")
        else:
            print("Audio Notes URL: http://localhost:7860 (default)")

    if args.no_transcription:
        print("Mode: RECORDING ONLY (live transcription disabled)")

    if args.debug:
        print("Debug: ENABLED")
    print()
    print("Tip: Use --system-audio for best quality")
    print("     Use --list-devices to see options")
    print()

    # Run application
    app = LiveCaptions(
        host=args.host,
        port=port,
        backend=args.backend,
        language=args.language,
        use_system_audio=args.system_audio,
        device_index=device_index,
        debug_save_audio=args.debug_save_audio,
        enable_recording=not args.no_recording,
        enable_transcription=not args.no_transcription,
        audio_notes_url=args.audio_notes_url,
    )
    app.run()


if __name__ == "__main__":
    main()
