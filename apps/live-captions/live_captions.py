#!/usr/bin/env python3
"""
Live Captions - Standalone Speech-to-Text Overlay

A frameless, transparent overlay window that captures audio from the microphone,
sends it to the ASR service for transcription, and displays real-time captions.

Usage: python live_captions.py [--host HOST] [--port PORT]
"""

import tkinter as tk
from tkinter import font as tkfont
import asyncio
import websockets
import json
import threading
import argparse
import ctypes
import numpy as np

# Enable high DPI awareness on Windows for crisp text
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()  # Fallback for older Windows
    except Exception:
        pass

# Audio settings
TARGET_SAMPLE_RATE = 16000  # ASR expects 16kHz
CHANNELS = 1
CHUNK_DURATION_MS = 250  # Chunk duration in milliseconds


class LiveCaptions:
    """Standalone caption overlay with microphone capture and ASR."""
    
    def __init__(self, host="localhost", port=8002):
        self.host = host
        self.port = port
        self.running = True
        self.full_text = ""
        self.recording = False
        self.pyaudio_instance = None
        self.stream = None
        self.loop = None
        self.capture_rate = TARGET_SAMPLE_RATE  # Will be updated on mic start
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Live Captions")
        
        # Window settings for transparency and frameless
        self.root.attributes('-topmost', True)  # Always on top
        self.root.attributes('-alpha', 0.85)    # Slight transparency
        self.root.overrideredirect(True)        # Remove window frame
        
        # Semi-transparent black background
        self.bg_color = '#1a1a1a'
        self.root.configure(bg=self.bg_color)
        
        # Window size and position (bottom center of screen, 80% width)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.width = int(screen_width * 0.8)
        self.height = 260
        x = (screen_width - self.width) // 2
        y = screen_height - self.height - 80
        self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")
        
        # Make window draggable
        self.root.bind('<Button-1>', self.start_drag)
        self.root.bind('<B1-Motion>', self.on_drag)
        
        # Right-click to close
        self.root.bind('<Button-3>', lambda e: self.close())
        
        # Escape to close
        self.root.bind('<Escape>', lambda e: self.close())
        
        # Mouse wheel to resize text
        self.root.bind('<MouseWheel>', self.on_mousewheel)
        
        # Create fonts - same size for both lines
        self.base_font_size = 36
        self.caption_font = tkfont.Font(family="Segoe UI", size=self.base_font_size, weight="bold")
        
        # Line storage for scrolling
        self.lines = []  # List of complete lines
        self.current_line_words = []  # Words in the current (incomplete) line
        
        # Create main container
        self.container = tk.Frame(self.root, bg=self.bg_color)
        self.container.pack(expand=True, fill='both', padx=15, pady=(15, 30))
        
        # Line 1 (older line, slightly dimmed) - starts at 15%
        self.line1 = tk.Label(
            self.container,
            text="",
            font=self.caption_font,
            fg='#999999',
            bg=self.bg_color,
            anchor='w'
        )
        self.line1.pack(expand=True, fill='both')
        
        # Line 2 (current line, bright) - dynamic positioning
        self.line2 = tk.Label(
            self.container,
            text="🎙️ Starting microphone...",
            font=self.caption_font,
            fg='#ffffff',
            bg=self.bg_color,
            anchor='w'
        )
        self.line2.pack(expand=True, fill='both')
        
        # Status indicator (small, top-right)
        self.status = tk.Label(
            self.root,
            text="● Initializing...",
            font=tkfont.Font(size=9),
            fg='#888888',
            bg=self.bg_color
        )
        self.status.place(relx=1.0, y=5, anchor='ne', x=-10)
        
        # Microphone indicator (top-left)
        self.mic_status = tk.Label(
            self.root,
            text="🎙️ Off",
            font=tkfont.Font(size=9),
            fg='#888888',
            bg=self.bg_color
        )
        self.mic_status.place(x=10, y=5, anchor='nw')
        
        # Hint label (bottom)
        self.hint = tk.Label(
            self.root,
            text="Drag to move • Scroll to resize • Right-click or Esc to close",
            font=tkfont.Font(size=9),
            fg='#555555',
            bg=self.bg_color
        )
        self.hint.place(relx=0.5, rely=1.0, anchor='s', y=-5)
        
        # Drag state
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        # Audio queue for sending to ASR
        self.audio_queue = None
        
    def start_drag(self, event):
        """Start dragging the window."""
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        
    def on_drag(self, event):
        """Handle window dragging."""
        x = self.root.winfo_x() + event.x - self.drag_start_x
        y = self.root.winfo_y() + event.y - self.drag_start_y
        self.root.geometry(f"+{x}+{y}")
        
    def on_mousewheel(self, event):
        """Resize text with mouse wheel."""
        if event.delta > 0:
            self.base_font_size = min(self.base_font_size + 2, 72)
        else:
            self.base_font_size = max(self.base_font_size - 2, 16)
        
        self.caption_font.configure(size=self.base_font_size)
        
    def get_text_width(self, text):
        """Calculate pixel width of text with current font."""
        return self.caption_font.measure(text)
    
    def get_max_line_width(self):
        """Get maximum line width (window width minus padding)."""
        return self.width - 40  # 15px padding on each side + some margin
    
    def update_text(self, text):
        """Update the displayed text with line-by-line scrolling."""
        if not text:
            return
        
        words = text.strip().split()
        if not words:
            return
        
        # Check if we have new words
        old_word_count = len(self.full_text.split()) if self.full_text else 0
        new_word_count = len(words)
        
        if new_word_count <= old_word_count:
            return
        
        self.full_text = text
        
        # Get the new words that were added
        new_words = words[old_word_count:]
        max_width = self.get_max_line_width()
        
        # Add new words, checking if they fit on current line
        for word in new_words:
            # Try adding word to current line
            test_line = ' '.join(self.current_line_words + [word])
            
            if self.get_text_width(test_line) <= max_width:
                # Word fits, add it to current line
                self.current_line_words.append(word)
            else:
                # Word doesn't fit, complete current line and start new one
                if self.current_line_words:
                    completed_line = ' '.join(self.current_line_words)
                    self.lines.append(completed_line)
                    # Keep only last line for display (line 1)
                    if len(self.lines) > 1:
                        self.lines = self.lines[-1:]
                
                # Start new line with this word
                self.current_line_words = [word]
            
            # Update display
            line1_text = self.lines[-1] if self.lines else ""
            line2_text = ' '.join(self.current_line_words)
            
            self.root.after(0, lambda l1=line1_text, l2=line2_text: self._set_lines(l1, l2))
        
    def _set_lines(self, line1, line2):
        """Set line text."""
        self.line1.configure(text=line1)
        self.line2.configure(text=line2)
        
        # Small left padding for readability
        self.line1.pack_configure(padx=(20, 0))
        self.line2.pack_configure(padx=(20, 0))
        
    def set_status(self, connected):
        """Update connection status."""
        if connected:
            self.root.after(0, lambda: self.status.configure(text="● Connected", fg='#4ade80'))
        else:
            self.root.after(0, lambda: self.status.configure(text="● Disconnected", fg='#f87171'))
    
    def set_mic_status(self, active):
        """Update microphone status."""
        if active:
            self.root.after(0, lambda: self.mic_status.configure(text="🎙️ Listening", fg='#4ade80'))
        else:
            self.root.after(0, lambda: self.mic_status.configure(text="🎙️ Off", fg='#f87171'))
            
    def close(self):
        """Close the overlay."""
        self.running = False
        self.stop_microphone()
        self.root.quit()
        self.root.destroy()
    
    def resample_audio(self, audio_data: bytes, from_rate: int, to_rate: int) -> bytes:
        """Resample audio from one sample rate to another."""
        if from_rate == to_rate:
            return audio_data
        
        # Convert bytes to numpy array (16-bit signed int)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        
        # Calculate resampling ratio
        ratio = to_rate / from_rate
        new_length = int(len(audio_np) * ratio)
        
        # Simple linear interpolation resampling
        indices = np.linspace(0, len(audio_np) - 1, new_length)
        resampled = np.interp(indices, np.arange(len(audio_np)), audio_np)
        
        # Convert back to 16-bit int bytes
        return resampled.astype(np.int16).tobytes()
    
    def start_microphone(self):
        """Start capturing audio from the microphone."""
        try:
            import pyaudio
            
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Find default input device and its native sample rate
            default_input = self.pyaudio_instance.get_default_input_device_info()
            device_name = default_input['name']
            native_rate = int(default_input['defaultSampleRate'])
            
            # Use native sample rate for capture, resample later if needed
            self.capture_rate = native_rate
            chunk_size = int(native_rate * CHUNK_DURATION_MS / 1000)
            
            print(f"Using microphone: {device_name}")
            print(f"Native sample rate: {native_rate}Hz, Target: {TARGET_SAMPLE_RATE}Hz")
            print(f"Chunk size: {chunk_size} samples ({CHUNK_DURATION_MS}ms)")
            
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
            print("Microphone started")
            
        except ImportError:
            print("ERROR: pyaudio not installed. Run: pip install pyaudio")
            self.root.after(0, lambda: self.line2.configure(
                text="❌ Install pyaudio: pip install pyaudio"
            ))
        except Exception as e:
            print(f"ERROR starting microphone: {e}")
            self.root.after(0, lambda: self.line2.configure(
                text=f"❌ Microphone error: {e}"
            ))
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback - receives audio chunks."""
        if self.running and self.loop and self.audio_queue:
            try:
                # Resample to 16kHz if needed
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
        """Stop capturing audio."""
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
        """Start the overlay."""
        # Start ASR connection and microphone in background thread
        asr_thread = threading.Thread(target=self._run_async, daemon=True)
        asr_thread.start()
        
        # Run tkinter main loop
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.close()
    
    def _run_async(self):
        """Run the async event loop in background thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.audio_queue = asyncio.Queue()
        
        # Start microphone after loop is ready
        self.root.after(500, self.start_microphone)
        
        # Run ASR connection
        self.loop.run_until_complete(self._asr_streaming())
        
    async def _asr_streaming(self):
        """Connect to ASR service and stream audio."""
        uri = f"ws://{self.host}:{self.port}/stream"
        
        while self.running:
            try:
                print(f"Connecting to ASR service: {uri}")
                async with websockets.connect(uri) as ws:
                    self.set_status(True)
                    self.root.after(0, lambda: self.line2.configure(text="🎙️ Speak now..."))
                    
                    # Create tasks for sending and receiving
                    send_task = asyncio.create_task(self._send_audio(ws))
                    recv_task = asyncio.create_task(self._receive_transcripts(ws))
                    
                    # Wait for either to finish
                    done, pending = await asyncio.wait(
                        [send_task, recv_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Cancel pending tasks
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                            
            except (ConnectionRefusedError, OSError) as e:
                self.set_status(False)
                print(f"Connection failed: {e}")
                self.root.after(0, lambda: self.line2.configure(
                    text="⏳ Waiting for ASR service..."
                ))
                await asyncio.sleep(2)
            except Exception as e:
                self.set_status(False)
                print(f"Error: {e}")
                await asyncio.sleep(2)
    
    async def _send_audio(self, ws):
        """Send audio chunks to ASR service."""
        chunk_count = 0
        TRIGGER_EVERY = 4  # Trigger transcription every 4 chunks (~1 second)
        
        while self.running:
            try:
                audio_data = await asyncio.wait_for(
                    self.audio_queue.get(),
                    timeout=1.0
                )
                await ws.send(audio_data)
                chunk_count += 1
                
                # Send empty bytes to trigger transcription periodically
                if chunk_count >= TRIGGER_EVERY:
                    await ws.send(b"")
                    chunk_count = 0
                
            except asyncio.TimeoutError:
                # No audio, but send trigger if we have buffered data
                if chunk_count > 0:
                    await ws.send(b"")
                    chunk_count = 0
                continue
            except websockets.ConnectionClosed:
                break
            except Exception as e:
                print(f"Send error: {e}")
                break
    
    async def _receive_transcripts(self, ws):
        """Receive transcription results from ASR service."""
        while self.running:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                
                try:
                    data = json.loads(message)
                    if 'text' in data and data['text']:
                        self.update_text(data['text'])
                except json.JSONDecodeError:
                    if message.strip():
                        self.update_text(message)
                        
            except asyncio.TimeoutError:
                continue
            except websockets.ConnectionClosed:
                break
            except Exception as e:
                print(f"Receive error: {e}")
                break


def main():
    parser = argparse.ArgumentParser(description="Live Captions - Speech-to-Text Overlay")
    parser.add_argument('--host', default='localhost', help='ASR service host (default: localhost)')
    parser.add_argument('--port', type=int, default=8002, help='ASR service port (default: 8002)')
    args = parser.parse_args()
    
    print("╔══════════════════════════════════════╗")
    print("║        Live Captions v1.0            ║")
    print("╚══════════════════════════════════════╝")
    print(f"ASR Service: ws://{args.host}:{args.port}/stream")
    print()
    print("Controls:")
    print("  • Drag to move window")
    print("  • Mouse wheel to resize text")
    print("  • Right-click or Escape to close")
    print()
    
    app = LiveCaptions(host=args.host, port=args.port)
    app.run()


if __name__ == "__main__":
    main()
