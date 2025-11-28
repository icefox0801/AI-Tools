#!/usr/bin/env python3
"""
Live Captions - Standalone Speech-to-Text Overlay v8.0

A frameless, transparent overlay window that captures audio from the microphone,
sends it to the ASR service for transcription, and displays real-time captions.

Features:
- Simplified client: ID-based replace/append logic
- Works with both Vosk and Parakeet backends
- Clean separation: UI only handles display, client handles transcription

Configuration is in shared/config/backends.py

Usage: python live_captions.py [--backend vosk|parakeet] [--debug]
"""

import sys
import os

# Add project root to path for shared module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tkinter as tk
from tkinter import font as tkfont
import asyncio
import json
import threading
import argparse
import ctypes
import logging
import numpy as np

# Import shared modules
from shared.config import BACKEND, get_backend_config, get_display_info
from shared.client import TranscriptManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Enable high DPI awareness on Windows for crisp text
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# Audio settings
TARGET_SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_MS = 200

class LiveCaptions:
    """Standalone caption overlay with microphone capture and ASR."""
    
    def __init__(self, host="localhost", port=None, backend=None):
        self.backend = backend or BACKEND
        self.config = get_backend_config(self.backend)
        self.host = host
        self.port = port or self.config["port"]
        self.running = True
        self.recording = False
        self.pyaudio_instance = None
        self.stream = None
        self.loop = None
        self.capture_rate = TARGET_SAMPLE_RATE
        
        # Simple transcript manager (ID-based replace/append)
        self.transcript = TranscriptManager()
        self.transcript.on_change = self._on_transcript_change
        
        # Display state
        self.display_text = ""
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Live Captions")
        
        # Window settings
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.85)
        self.root.overrideredirect(True)
        
        self.bg_color = '#1a1a1a'
        self.root.configure(bg=self.bg_color)
        
        # Window size and position
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.width = int(screen_width * 0.8)
        self.height = 260
        x = (screen_width - self.width) // 2
        y = screen_height - self.height - 80
        self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")
        
        # Bindings
        self.root.bind('<Button-1>', self.start_drag)
        self.root.bind('<B1-Motion>', self.on_drag)
        self.root.bind('<Button-3>', lambda e: self.close())
        self.root.bind('<Escape>', lambda e: self.close())
        self.root.bind('<MouseWheel>', self.on_mousewheel)
        
        # Fonts
        self.base_font_size = 36
        self.caption_font = tkfont.Font(family="Segoe UI", size=self.base_font_size, weight="bold")
        
        # UI elements
        self.container = tk.Frame(self.root, bg=self.bg_color)
        self.container.pack(expand=True, fill='both', padx=15, pady=(15, 30))
        
        self.line1 = tk.Label(
            self.container, text="", font=self.caption_font,
            fg='#999999', bg=self.bg_color, anchor='w'
        )
        self.line1.pack(expand=True, fill='both')
        
        self.line2 = tk.Label(
            self.container, text="🎙️ Starting microphone...", font=self.caption_font,
            fg='#ffffff', bg=self.bg_color, anchor='w'
        )
        self.line2.pack(expand=True, fill='both')
        
        self.status = tk.Label(
            self.root, text="● Initializing...", font=tkfont.Font(size=9),
            fg='#888888', bg=self.bg_color
        )
        self.status.place(relx=1.0, y=5, anchor='ne', x=-10)
        
        self.mic_status = tk.Label(
            self.root, text="🎙️ Off", font=tkfont.Font(size=9),
            fg='#888888', bg=self.bg_color
        )
        self.mic_status.place(x=10, y=5, anchor='nw')
        
        self.hint = tk.Label(
            self.root, text="Drag to move • Scroll to resize • Right-click or Esc to close",
            font=tkfont.Font(size=9), fg='#555555', bg=self.bg_color
        )
        self.hint.place(relx=0.5, rely=1.0, anchor='s', y=-5)
        
        self.model_label = tk.Label(
            self.root, text=get_display_info(self.backend),
            font=tkfont.Font(size=9), fg='#888888', bg=self.bg_color
        )
        self.model_label.place(relx=0.5, y=5, anchor='n')
        
        # Drag state
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        # Audio queue
        self.audio_queue = None
        
    def _on_transcript_change(self):
        """Callback when transcript changes - update display."""
        text = self.transcript.get_text()
        self.update_display(text)
        
    def start_drag(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        
    def on_drag(self, event):
        x = self.root.winfo_x() + event.x - self.drag_start_x
        y = self.root.winfo_y() + event.y - self.drag_start_y
        self.root.geometry(f"+{x}+{y}")
        
    def on_mousewheel(self, event):
        if event.delta > 0:
            self.base_font_size = min(self.base_font_size + 2, 72)
        else:
            self.base_font_size = max(self.base_font_size - 2, 16)
        self.caption_font.configure(size=self.base_font_size)
        
    def get_text_width(self, text):
        return self.caption_font.measure(text)
    
    def get_max_line_width(self):
        return self.width - 40
    
    def wrap_text_to_lines(self, text):
        if not text:
            return []
        
        max_width = self.get_max_line_width()
        words = text.split()
        
        if not words:
            return []
        
        lines = []
        current_line = ""
        
        for word in words:
            test_line = (current_line + " " + word).strip() if current_line else word
            
            if self.get_text_width(test_line) <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def update_display(self, text):
        """Update the displayed text."""
        if not text:
            return
        
        text = text.strip()
        if not text:
            return
        
        self.display_text = text
        lines = self.wrap_text_to_lines(text)
        
        if not lines:
            return
        
        if len(lines) >= 2:
            line1_text = lines[-2]
            line2_text = lines[-1]
        else:
            line1_text = ""
            line2_text = lines[-1] if lines else ""
        
        self.root.after(0, lambda l1=line1_text, l2=line2_text: self._set_lines(l1, l2))
        
    def _set_lines(self, line1, line2):
        self.line1.configure(text=line1)
        self.line2.configure(text=line2)
        self.line1.pack_configure(padx=(20, 0))
        self.line2.pack_configure(padx=(20, 0))
        
    def set_status(self, connected):
        if connected:
            self.root.after(0, lambda: self.status.configure(text="● Connected", fg='#4ade80'))
        else:
            self.root.after(0, lambda: self.status.configure(text="● Disconnected", fg='#f87171'))
    
    def set_mic_status(self, active):
        if active:
            self.root.after(0, lambda: self.mic_status.configure(text="🎙️ Listening", fg='#4ade80'))
        else:
            self.root.after(0, lambda: self.mic_status.configure(text="🎙️ Off", fg='#f87171'))
            
    def close(self):
        self.running = False
        self.stop_microphone()
        self.root.quit()
        self.root.destroy()
    
    def resample_audio(self, audio_data: bytes, from_rate: int, to_rate: int) -> bytes:
        if from_rate == to_rate:
            return audio_data
        
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        ratio = to_rate / from_rate
        new_length = int(len(audio_np) * ratio)
        indices = np.linspace(0, len(audio_np) - 1, new_length)
        resampled = np.interp(indices, np.arange(len(audio_np)), audio_np)
        return resampled.astype(np.int16).tobytes()
    
    def start_microphone(self):
        try:
            import pyaudio
            
            self.pyaudio_instance = pyaudio.PyAudio()
            default_input = self.pyaudio_instance.get_default_input_device_info()
            device_name = default_input['name']
            native_rate = int(default_input['defaultSampleRate'])
            
            self.capture_rate = native_rate
            chunk_size = int(native_rate * CHUNK_DURATION_MS / 1000)
            
            logger.info(f"Using microphone: {device_name}")
            logger.info(f"Native sample rate: {native_rate}Hz, Target: {TARGET_SAMPLE_RATE}Hz")
            logger.debug(f"Chunk size: {chunk_size} samples ({CHUNK_DURATION_MS}ms)")
            
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=native_rate,
                input=True,
                frames_per_buffer=chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.stream.start_stream()
            self.recording = True
            self.set_mic_status(True)
            logger.info("Microphone started")
            
        except ImportError:
            logger.error("pyaudio not installed. Run: pip install pyaudio")
            self.root.after(0, lambda: self.line2.configure(
                text="❌ Install pyaudio: pip install pyaudio"
            ))
        except Exception as e:
            logger.error(f"Starting microphone failed: {e}")
            self.root.after(0, lambda: self.line2.configure(
                text=f"❌ Microphone error: {e}"
            ))
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self.running and self.loop and self.audio_queue:
            try:
                resampled = self.resample_audio(in_data, self.capture_rate, TARGET_SAMPLE_RATE)
                asyncio.run_coroutine_threadsafe(
                    self.audio_queue.put(resampled),
                    self.loop
                )
            except Exception:
                pass
        
        import pyaudio
        return (None, pyaudio.paContinue)
    
    def stop_microphone(self):
        self.recording = False
        self.set_mic_status(False)
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
            
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except Exception:
                pass
            self.pyaudio_instance = None
        
    def run(self):
        asr_thread = threading.Thread(target=self._run_async, daemon=True)
        asr_thread.start()
        
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.close()
    
    def _run_async(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.audio_queue = asyncio.Queue()
        
        self.root.after(500, self.start_microphone)
        self.loop.run_until_complete(self._asr_streaming())
        
    async def _asr_streaming(self):
        """Connect to ASR service and stream audio."""
        import websockets
        
        uri = f"ws://{self.host}:{self.port}/stream"
        
        while self.running:
            try:
                logger.info(f"Connecting to ASR service: {uri}")
                async with websockets.connect(uri) as ws:
                    self.set_status(True)
                    self.root.after(0, lambda: self.line2.configure(text="🎙️ Speak now..."))
                    
                    # Send config
                    config = {"chunk_ms": self.config["chunk_ms"]}
                    await ws.send(json.dumps(config))
                    logger.debug(f"Sent config: {config}")
                    
                    # Run send and receive concurrently
                    send_task = asyncio.create_task(self._send_audio(ws))
                    recv_task = asyncio.create_task(self._receive_transcripts(ws))
                    
                    done, pending = await asyncio.wait(
                        [send_task, recv_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                            
            except (ConnectionRefusedError, OSError) as e:
                self.set_status(False)
                logger.warning(f"Connection failed: {e}")
                self.root.after(0, lambda: self.line2.configure(
                    text="⏳ Waiting for ASR service..."
                ))
                await asyncio.sleep(2)
            except Exception as e:
                self.set_status(False)
                logger.error(f"ASR streaming error: {e}")
                await asyncio.sleep(2)
    
    async def _send_audio(self, ws):
        """Send audio chunks to ASR service."""
        while self.running:
            try:
                audio_data = await asyncio.wait_for(
                    self.audio_queue.get(),
                    timeout=1.0
                )
                await ws.send(audio_data)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Send error: {e}")
                break
    
    async def _receive_transcripts(self, ws):
        """
        Receive transcription results from ASR service.
        
        Protocol:
        - {"id": "s0", "text": "hello"}
        - {"id": "s0", "text": "hello world"}  # replaces
        - {"id": "s1", "text": "new segment"}  # appends
        
        Client logic: if ID exists → replace, if not → append
        """
        while self.running:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                
                try:
                    data = json.loads(message)
                    
                    # ID-based protocol: just pass id and text to manager
                    if 'id' in data:
                        segment_id = data['id']
                        text = data.get('text', '').strip()
                        if text:
                            self.transcript.update(segment_id, text)
                    
                    # Legacy Vosk protocol support
                    elif 'partial' in data:
                        partial = data['partial'].strip()
                        if partial:
                            self.transcript.update('_partial', partial)
                    elif 'text' in data:
                        text = data['text'].strip()
                        if text:
                            self.transcript.update('_final', text)
                            
                except json.JSONDecodeError:
                    if message.strip():
                        self.transcript.update('_text', message.strip())
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Receive error: {e}")
                break


def main():
    parser = argparse.ArgumentParser(description="Live Captions - Speech-to-Text Overlay")
    parser.add_argument('--host', default='localhost', help='ASR service host')
    parser.add_argument('--port', type=int, help='ASR service port (auto from backend)')
    parser.add_argument('--backend', choices=['vosk', 'parakeet'], default=BACKEND,
                        help=f'ASR backend (default: {BACKEND})')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger('shared.client.transcript').setLevel(logging.DEBUG)
        logging.getLogger(__name__).setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    cfg = get_backend_config(args.backend)
    port = args.port or cfg["port"]
    
    print("╔══════════════════════════════════════╗")
    print("║        Live Captions v8.0            ║")
    print("║   (Simplified ID-based Protocol)     ║")
    print("╚══════════════════════════════════════╝")
    print(f"Model: {cfg['name']} ({cfg['device']})")
    print(f"Mode: {cfg['mode']}")
    print(f"ASR Service: ws://{args.host}:{port}/stream")
    if args.debug:
        print(f"Debug: ENABLED")
    print()
    print("Controls:")
    print("  • Drag to move window")
    print("  • Mouse wheel to resize text")
    print("  • Right-click or Escape to close")
    
    app = LiveCaptions(host=args.host, port=port, backend=args.backend)
    app.run()


if __name__ == "__main__":
    main()