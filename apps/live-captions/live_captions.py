#!/usr/bin/env python3
"""
Live Captions - Standalone Speech-to-Text Overlay v10.1

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

import sys
import os

# Add project root to path for shared module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio
import threading
import argparse
import logging
import wave
from datetime import datetime
from typing import Optional

# Import shared modules
from shared.config import BACKEND, get_backend_config, get_display_info
from shared.client import TranscriptManager

# Import local modules
from src.audio import (
    MicrophoneCapture, SystemAudioCapture,
    list_devices, TARGET_SAMPLE_RATE
)
from src.ui import CaptionWindow
from src.asr import ASRClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class LiveCaptions:
    """Main application class coordinating UI, audio capture, and ASR."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: Optional[int] = None,
        backend: Optional[str] = None,
        use_system_audio: bool = False,
        device_index: Optional[int] = None,
        debug_save_audio: bool = False
    ):
        """
        Initialize Live Captions application.
        
        Args:
            host: ASR service host
            port: ASR service port (uses backend default if None)
            backend: ASR backend ('vosk' or 'parakeet')
            use_system_audio: Use system audio instead of microphone
            device_index: Specific audio device index
            debug_save_audio: Save captured audio on exit
        """
        # Backend config
        self.backend = backend or BACKEND
        self.config = get_backend_config(self.backend)
        self.host = host
        self.port = port or self.config["port"]
        
        # State
        self.running = True
        
        # Audio settings
        self.use_system_audio = use_system_audio
        self.device_index = device_index
        self.audio_capture = None
        
        # Debug
        self.debug_save_audio = debug_save_audio
        self.debug_audio_chunks = []
        
        # Transcript manager
        self.transcript = TranscriptManager()
        self.transcript.on_change = self._on_transcript_change
        
        # Async components
        self.loop = None
        self.asr_client = None
        
        # Create UI
        self.window = CaptionWindow(
            model_display=get_display_info(self.backend),
            on_close=self.close
        )
    
    def _on_transcript_change(self):
        """Handle transcript updates."""
        text = self.transcript.get_text()
        self.window.update_text(text)
    
    def _on_audio_data(self, audio_bytes: bytes):
        """Handle audio data from capture."""
        if self.debug_save_audio:
            self.debug_audio_chunks.append(audio_bytes)
        
        if self.asr_client:
            self.asr_client.queue_audio(audio_bytes)
    
    def _on_asr_connected(self, connected: bool):
        """Handle ASR connection status change."""
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
                callback=self._on_audio_data,
                device_index=self.device_index
            )
        else:
            self.audio_capture = MicrophoneCapture(
                callback=self._on_audio_data,
                device_index=self.device_index
            )
        
        if self.audio_capture.start():
            self.window.set_audio_status(self.audio_capture.source_name)
        else:
            self.window.set_message("❌ Audio capture failed")
    
    def _stop_audio_capture(self):
        """Stop audio capture."""
        if self.audio_capture:
            self.audio_capture.stop()
            self.audio_capture = None
        
        if self.debug_save_audio and self.debug_audio_chunks:
            self._save_debug_audio()
    
    def _save_debug_audio(self):
        """Save captured audio to WAV file for debugging."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"debug_audio_{timestamp}.wav"
            
            audio_data = b''.join(self.debug_audio_chunks)
            
            with wave.open(filename, 'wb') as wf:
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
        self.running = False
        
        if self.asr_client:
            self.asr_client.stop()
        
        self._stop_audio_capture()
        self.window.close()
    
    def run(self):
        """Run the application."""
        # Start ASR thread
        asr_thread = threading.Thread(target=self._run_async, daemon=True)
        asr_thread.start()
        
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
            on_connected=self._on_asr_connected,
            on_transcript=self._on_asr_transcript
        )
        
        # Start audio capture after short delay
        self.window.after(500, self._start_audio_capture)
        
        # Run ASR client
        self.loop.run_until_complete(self.asr_client.run())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Live Captions v10.1")
    parser.add_argument('--host', default='localhost', help='ASR service host')
    parser.add_argument('--port', type=int, help='ASR service port')
    parser.add_argument('--backend', choices=['vosk', 'parakeet'], default=BACKEND,
                        help=f'ASR backend (default: {BACKEND})')
    parser.add_argument('--system-audio', action='store_true',
                        help='Capture system audio instead of microphone')
    parser.add_argument('--device', type=int, 
                        help='Microphone device index')
    parser.add_argument('--loopback-device', type=int,
                        help='Loopback device index for system audio')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio devices')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--debug-save-audio', action='store_true',
                        help='Save captured audio on exit')
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
    
    # Print startup banner (ASCII only for Windows console compatibility)
    print("+======================================+")
    print("|        Live Captions v10.1          |")
    print("+======================================+")
    print(f"Model: {cfg['name']} ({cfg['device']})")
    print(f"ASR: ws://{args.host}:{port}/stream")
    
    if args.system_audio:
        print("Audio: [Speaker] System Audio (WASAPI loopback)")
    else:
        print("Audio: [Mic] Microphone")
    
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
        use_system_audio=args.system_audio,
        device_index=device_index,
        debug_save_audio=args.debug_save_audio
    )
    app.run()


if __name__ == "__main__":
    main()
