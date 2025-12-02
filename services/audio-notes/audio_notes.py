#!/usr/bin/env python3
"""
Audio Notes - Transcription & AI Summarization App

A Gradio web UI that:
1. Lists available recordings from the shared recordings directory
2. Receives audio files (from Live Captions or direct upload)
3. Transcribes using Whisper ASR service
4. Summarizes using Ollama LLM
5. Provides interactive chat for Q&A about the content

Usage:
  python audio_notes.py                    # Start web UI
  python audio_notes.py --port 7860        # Custom port
  python audio_notes.py --audio file.wav   # Open with audio file and auto-transcribe
"""

import os
import sys
import io
import json
import time
import asyncio
import logging
import tempfile
import webbrowser
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from urllib.parse import urlparse, parse_qs

import gradio as gr
import requests
import wave
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configuration
WHISPER_URL = os.getenv("WHISPER_URL", "http://localhost:8003")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")  # Default to qwen3 (commonly available)

# Shared recordings directory (same as Live Captions uses)
# In Docker: /app/recordings, Local: project_root/recordings
_default_recordings = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) / "recordings"
RECORDINGS_DIR = Path(os.getenv("RECORDINGS_DIR", str(_default_recordings)))

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def check_whisper_health() -> tuple[bool, str]:
    """Check if Whisper service is available."""
    try:
        resp = requests.get(f"{WHISPER_URL}/health", timeout=2)
        if resp.status_code == 200:
            return True, "Whisper ASR ready"
        return False, f"Whisper returned status {resp.status_code}"
    except Exception as e:
        return False, f"Whisper not available: {e}"


def check_ollama_health() -> tuple[bool, str]:
    """Check if Ollama service is available."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            if any(OLLAMA_MODEL in m for m in models):
                return True, f"Ollama ready ({OLLAMA_MODEL})"
            return False, f"Model {OLLAMA_MODEL} not found. Available: {models}"
        return False, f"Ollama returned status {resp.status_code}"
    except Exception as e:
        return False, f"Ollama not available: {e}"


def get_audio_duration(audio_path: str) -> float:
    """Get audio file duration in seconds."""
    try:
        with wave.open(audio_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / rate
    except Exception:
        # Estimate from file size for non-WAV files
        try:
            size = os.path.getsize(audio_path)
            return size / 32000  # Rough estimate for 16kHz 16-bit mono
        except Exception:
            return 0.0


def list_recordings() -> List[dict]:
    """List all recordings in the shared recordings directory.
    
    Returns:
        List of dicts with recording info: {path, name, size_mb, duration, date}
    """
    recordings = []
    
    if not RECORDINGS_DIR.exists():
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        return recordings
    
    # Find all audio files
    extensions = ['.wav', '.mp3', '.m4a', '.ogg', '.flac']
    for ext in extensions:
        for audio_file in RECORDINGS_DIR.glob(f"*{ext}"):
            try:
                stat = audio_file.stat()
                size_mb = stat.st_size / (1024 * 1024)
                mod_time = datetime.fromtimestamp(stat.st_mtime)
                duration = get_audio_duration(str(audio_file))
                
                recordings.append({
                    'path': str(audio_file),
                    'name': audio_file.name,
                    'size_mb': size_mb,
                    'duration': duration,
                    'duration_str': f"{int(duration // 60):02d}:{int(duration % 60):02d}",
                    'date': mod_time.strftime("%Y-%m-%d %H:%M"),
                    'timestamp': stat.st_mtime
                })
            except Exception as e:
                logger.warning(f"Error reading {audio_file}: {e}")
    
    # Sort by newest first
    recordings.sort(key=lambda x: x['timestamp'], reverse=True)
    return recordings


def format_recordings_table(recordings: List[dict]) -> str:
    """Format recordings list as markdown table."""
    if not recordings:
        return "📁 *No recordings found*\n\nRecord audio using Live Captions or upload a file."
    
    lines = ["| File | Duration | Size | Date |", "| --- | --- | --- | --- |"]
    for r in recordings:
        lines.append(f"| {r['name']} | {r['duration_str']} | {r['size_mb']:.1f} MB | {r['date']} |")
    
    return "\n".join(lines)


def transcribe_audio(audio_path: str) -> tuple[str, float]:
    """Transcribe audio file using Whisper service.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Tuple of (transcript_text, duration_seconds)
    """
    logger.info(f"Transcribing: {audio_path}")
    
    # Read audio file
    with open(audio_path, 'rb') as f:
        audio_data = f.read()
    
    # Get duration
    duration = get_audio_duration(audio_path)
    
    # Determine content type
    ext = Path(audio_path).suffix.lower()
    content_types = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.m4a': 'audio/mp4',
        '.ogg': 'audio/ogg',
        '.flac': 'audio/flac'
    }
    content_type = content_types.get(ext, 'audio/wav')
    
    # Send to Whisper /transcribe endpoint
    files = {'file': (Path(audio_path).name, audio_data, content_type)}
    
    try:
        resp = requests.post(
            f"{WHISPER_URL}/transcribe",
            files=files,
            timeout=300  # 5 minutes for long audio
        )
        
        if resp.status_code == 200:
            data = resp.json()
            text = data.get("text", "")
            duration = data.get("duration", duration)
            logger.info(f"Transcription complete: {len(text)} chars, {duration:.1f}s")
            return text, duration
        else:
            logger.error(f"Transcription failed: {resp.status_code} - {resp.text}")
            return f"Error: {resp.status_code}", duration
            
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return f"Error: {e}", duration


def summarize_with_ollama(transcript: str, system_prompt: Optional[str] = None) -> str:
    """Summarize transcript using Ollama.
    
    Args:
        transcript: Full transcript text
        system_prompt: Optional system prompt override
        
    Returns:
        Summary text in markdown format
    """
    if not system_prompt:
        system_prompt = """You are an expert summarizer. Analyze the following transcript and provide:

1. **Summary** (2-3 paragraphs): A concise overview of the main content
2. **Key Points** (bullet list): The most important takeaways
3. **Topics Discussed** (bullet list): Main subjects covered
4. **Action Items** (if any): Any tasks or follow-ups mentioned

Format your response in clean Markdown."""

    prompt = f"""Please analyze this transcript:

---
{transcript}
---

Provide a comprehensive summary following the format specified."""

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False
            },
            timeout=300  # 5 minutes for long transcripts
        )
        
        if resp.status_code == 200:
            data = resp.json()
            return data.get("response", "No response generated")
        else:
            return f"Error from Ollama: {resp.status_code}"
            
    except Exception as e:
        return f"Error: {e}"


def chat_with_context(message: str, history: list, transcript: str) -> str:
    """Chat with Ollama using transcript as context.
    
    Args:
        message: User's question
        history: Chat history
        transcript: Full transcript for context
        
    Returns:
        AI response
    """
    # Build conversation context
    context_limit = 8000  # Limit context size
    truncated_transcript = transcript[:context_limit]
    if len(transcript) > context_limit:
        truncated_transcript += "\n...[transcript truncated]..."
    
    system_prompt = f"""You are a helpful assistant answering questions about a transcript.

Here is the transcript for context:
---
{truncated_transcript}
---

Answer the user's questions based on this transcript. If the answer is not in the transcript, say so."""

    # Build messages from history
    messages = []
    for user_msg, ai_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": ai_msg})
    messages.append({"role": "user", "content": message})
    
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "system": system_prompt,
                "stream": False
            },
            timeout=60
        )
        
        if resp.status_code == 200:
            data = resp.json()
            return data.get("message", {}).get("content", "No response")
        else:
            return f"Error: {resp.status_code}"
            
    except Exception as e:
        return f"Error: {e}"


def process_audio(audio_file, progress=gr.Progress()) -> tuple[str, str, str]:
    """Process uploaded audio file.
    
    Args:
        audio_file: Uploaded file path or Gradio audio tuple
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (transcript, summary, status)
    """
    if audio_file is None:
        return "", "", "⚠️ Please select or upload an audio file"
    
    progress(0, desc="Checking services...")
    
    # Handle different input types
    if isinstance(audio_file, tuple):
        # Gradio audio component returns (sample_rate, data)
        sample_rate, audio_data = audio_file
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            with wave.open(f.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                if isinstance(audio_data, np.ndarray):
                    if audio_data.dtype != np.int16:
                        audio_data = (audio_data * 32767).astype(np.int16)
                    wf.writeframes(audio_data.tobytes())
                else:
                    wf.writeframes(audio_data)
            audio_path = f.name
    else:
        audio_path = audio_file
    
    # Check services
    whisper_ok, whisper_msg = check_whisper_health()
    if not whisper_ok:
        return "", "", f"❌ {whisper_msg}"
    
    ollama_ok, ollama_msg = check_ollama_health()
    if not ollama_ok:
        return "", "", f"❌ {ollama_msg}"
    
    # Transcribe
    progress(0.2, desc="Transcribing audio...")
    transcript, duration = transcribe_audio(audio_path)
    
    if transcript.startswith("Error"):
        return "", "", f"❌ Transcription failed: {transcript}"
    
    # Summarize
    progress(0.7, desc="Generating summary...")
    summary = summarize_with_ollama(transcript)
    
    progress(1.0, desc="Done!")
    status = f"✅ Done! Transcribed {duration:.1f}s of audio ({len(transcript)} chars)"
    
    return transcript, summary, status


def batch_transcribe(selected_files: List[str], progress=gr.Progress()) -> str:
    """Batch transcribe multiple audio files.
    
    Args:
        selected_files: List of file paths to transcribe
        progress: Gradio progress tracker
        
    Returns:
        Status message with results
    """
    if not selected_files:
        return "⚠️ No files selected for batch transcription"
    
    # Check Whisper service
    whisper_ok, whisper_msg = check_whisper_health()
    if not whisper_ok:
        return f"❌ {whisper_msg}"
    
    results = []
    total = len(selected_files)
    
    for i, file_path in enumerate(selected_files):
        file_name = Path(file_path).name
        progress((i / total), desc=f"Transcribing {file_name} ({i+1}/{total})...")
        
        try:
            transcript, duration = transcribe_audio(file_path)
            
            if transcript.startswith("Error"):
                results.append(f"❌ {file_name}: {transcript}")
                continue
            
            # Save transcript to .txt file
            txt_path = Path(file_path).with_suffix('.txt')
            txt_path.write_text(transcript, encoding='utf-8')
            
            results.append(f"✅ {file_name} → {txt_path.name} ({duration:.1f}s, {len(transcript)} chars)")
            logger.info(f"Batch transcribed: {file_name} -> {txt_path}")
            
        except Exception as e:
            results.append(f"❌ {file_name}: {e}")
            logger.error(f"Batch transcription error for {file_name}: {e}")
    
    progress(1.0, desc="Batch transcription complete!")
    
    return "**Batch Transcription Results:**\n\n" + "\n".join(results)


def create_ui(initial_audio: Optional[str] = None, auto_transcribe: bool = False):
    """Create Gradio interface.
    
    Args:
        initial_audio: Optional path to audio file to load on startup
        auto_transcribe: If True and initial_audio is provided, auto-start transcription
    """
    
    # State for transcript (used in chat)
    transcript_state = gr.State("")
    
    with gr.Blocks(title="Audio Notes") as demo:
        gr.Markdown("""
        # 📝 Audio Notes
        
        Transform audio recordings into searchable notes with AI-powered summarization.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Recordings browser with batch selection
                with gr.Accordion("📁 Recordings", open=True):
                    with gr.Row(equal_height=True):
                        select_all_checkbox = gr.Checkbox(
                            label="Select All",
                            value=False,
                            interactive=True
                        )
                        refresh_recordings_btn = gr.Button(
                            "🔄",
                            variant="secondary",
                            size="sm",
                            min_width=36
                        )
                    
                    recordings_checkboxes = gr.CheckboxGroup(
                        label="Select recordings to transcribe",
                        choices=[],
                        value=[],
                        interactive=True
                    )
                    
                    batch_transcribe_btn = gr.Button(
                        "📝 Batch Transcribe Selected", 
                        variant="primary",
                        size="lg"
                    )
                    batch_status = gr.Markdown("")
                
                gr.Markdown("---")
                
                # Upload section
                with gr.Accordion("📤 Upload Audio", open=False):
                    file_input = gr.File(
                        label="Drop audio file here",
                        file_types=[".wav", ".mp3", ".m4a", ".ogg", ".flac"]
                    )
                    upload_status = gr.Markdown("")
                
                status_text = gr.Textbox(
                    label="Status", 
                    interactive=False,
                    elem_classes=["status-box"],
                    visible=False
                )
                
                # Service status
                with gr.Accordion("⚙️ Service Status", open=False):
                    gr.Markdown(f"""
                    **Backend Services:**
                    - Whisper ASR: `{WHISPER_URL}`
                    - Ollama LLM: `{OLLAMA_URL}`
                    - Model: `{OLLAMA_MODEL}`
                    - Recordings: `{RECORDINGS_DIR}`
                    """)
                    refresh_btn = gr.Button("🔄 Check Services")
                    service_status = gr.Markdown("")
            
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("📋 Summary"):
                        summary_output = gr.Markdown(
                            label="Summary"
                        )
                    
                    with gr.TabItem("📜 Full Transcript"):
                        transcript_output = gr.Textbox(
                            label="Transcript",
                            lines=20,
                            max_lines=50
                        )
                        
                        with gr.Row():
                            save_transcript_btn = gr.Button("💾 Save as .txt", size="sm")
                    
                    with gr.TabItem("💬 Chat"):
                        chatbot = gr.Chatbot(
                            label="Ask questions about the content",
                            height=400
                        )
                        
                        with gr.Row():
                            chat_input = gr.Textbox(
                                label="Your question",
                                placeholder="What was discussed? What are the main points?",
                                lines=2,
                                scale=4
                            )
                            chat_btn = gr.Button("Send", variant="primary", scale=1)
                        
                        with gr.Row():
                            clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
                            example_q1 = gr.Button("What are the key points?", size="sm")
                            example_q2 = gr.Button("Summarize in 3 sentences", size="sm")
        
        # Event handlers
        def refresh_recordings():
            """Refresh the recordings list with checkboxes."""
            recordings = list_recordings()
            # Format: "filename (duration, size)" -> path
            checkbox_choices = [(f"{r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)", r['path']) for r in recordings]
            return (
                gr.update(choices=checkbox_choices, value=[]),  # checkboxes
                ""  # clear batch status
            )
        
        def on_select_all_change(select_all):
            """Handle Select All checkbox change."""
            if select_all:
                recordings = list_recordings()
                all_paths = [r['path'] for r in recordings]
                return all_paths
            else:
                return []
        
        def on_batch_transcribe(selected_files):
            """Handle batch transcription."""
            if not selected_files:
                return "⚠️ No files selected. Check the boxes next to recordings to select them."
            return batch_transcribe(selected_files)
        
        def on_file_upload(file):
            """Handle file upload - save to recordings folder with recording_YYYYMMDD_HHMMSS name."""
            if file is None:
                return "", gr.update()
            
            try:
                # Get source file extension
                src_path = Path(file.name if hasattr(file, 'name') else file)
                ext = src_path.suffix.lower() or '.wav'
                
                # New filename: recording_YYYYMMDD_HHMMSS.ext (same pattern as Live Captions)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_name = f"recording_{timestamp}{ext}"
                dest_path = RECORDINGS_DIR / new_name
                
                # Copy file to recordings directory
                import shutil
                shutil.copy2(src_path, dest_path)
                
                logger.info(f"Uploaded file saved as: {dest_path}")
                
                # Refresh recordings list
                recordings = list_recordings()
                checkbox_choices = [(f"{r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)", r['path']) for r in recordings]
                
                return (
                    f"✅ Saved as **{new_name}**",
                    gr.update(choices=checkbox_choices, value=[])
                )
            except Exception as e:
                logger.error(f"Upload error: {e}")
                return f"❌ Upload failed: {e}", gr.update()
        
        def on_chat(message, history, transcript):
            if not message.strip():
                return history, ""
            if not transcript:
                return history + [[message, "⚠️ Please transcribe audio first"]], ""
            response = chat_with_context(message, history, transcript)
            return history + [[message, response]], ""
        
        def on_refresh():
            whisper_ok, whisper_msg = check_whisper_health()
            ollama_ok, ollama_msg = check_ollama_health()
            return f"""
**Status:**
- Whisper: {"✅" if whisper_ok else "❌"} {whisper_msg}
- Ollama: {"✅" if ollama_ok else "❌"} {ollama_msg}
            """
        
        def on_example_question(question, history, transcript):
            if not transcript:
                return history + [[question, "⚠️ Please transcribe audio first"]], ""
            response = chat_with_context(question, history, transcript)
            return history + [[question, response]], ""
        
        def save_transcript(transcript):
            """Save transcript to file."""
            if not transcript:
                return "⚠️ No transcript to save"
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = RECORDINGS_DIR / f"transcript_{timestamp}.txt"
            
            try:
                output_path.write_text(transcript, encoding='utf-8')
                return f"✅ Saved to: {output_path}"
            except Exception as e:
                return f"❌ Failed to save: {e}"
        
        # Wire up events
        refresh_recordings_btn.click(
            refresh_recordings,
            outputs=[recordings_checkboxes, batch_status]
        )
        
        select_all_checkbox.change(
            on_select_all_change,
            inputs=[select_all_checkbox],
            outputs=[recordings_checkboxes]
        )
        
        batch_transcribe_btn.click(
            on_batch_transcribe,
            inputs=[recordings_checkboxes],
            outputs=[batch_status]
        )
        
        file_input.change(
            on_file_upload,
            inputs=[file_input],
            outputs=[upload_status, recordings_checkboxes]
        )
        
        chat_btn.click(
            on_chat,
            inputs=[chat_input, chatbot, transcript_state],
            outputs=[chatbot, chat_input]
        )
        
        chat_input.submit(
            on_chat,
            inputs=[chat_input, chatbot, transcript_state],
            outputs=[chatbot, chat_input]
        )
        
        save_transcript_btn.click(
            save_transcript,
            inputs=[transcript_output],
            outputs=[status_text]
        )
        
        clear_btn.click(lambda: [], None, chatbot)
        refresh_btn.click(on_refresh, None, service_status)
        
        example_q1.click(
            lambda h, t: on_example_question("What are the key points?", h, t),
            inputs=[chatbot, transcript_state],
            outputs=[chatbot, chat_input]
        )
        
        example_q2.click(
            lambda h, t: on_example_question("Summarize in 3 sentences", h, t),
            inputs=[chatbot, transcript_state],
            outputs=[chatbot, chat_input]
        )
        
        # Initial load - populate recordings list
        def on_load():
            """Called when UI loads."""
            recordings = list_recordings()
            checkbox_choices = [(f"{r['name']} ({r['duration_str']}, {r['size_mb']:.1f}MB)", r['path']) for r in recordings]
            return gr.update(choices=checkbox_choices, value=[])
        
        demo.load(
            on_load,
            outputs=[recordings_checkboxes]
        )
        
        # Auto-transcribe if initial_audio is provided (not used in Docker mode)
        pass
    
    return demo


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio Notes - Transcription & AI Summarization")
    parser.add_argument('--port', type=int, default=7860, help='Port to run on')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--share', action='store_true', help='Create public link')
    parser.add_argument('--audio', type=str, help='Audio file to load on startup')
    parser.add_argument('--auto-transcribe', action='store_true', 
                        help='Automatically transcribe when --audio is provided')
    args = parser.parse_args()
    
    print("=" * 50)
    print("📝 Audio Notes")
    print("=" * 50)
    print(f"Whisper ASR: {WHISPER_URL}")
    print(f"Ollama LLM:  {OLLAMA_URL} (model: {OLLAMA_MODEL})")
    print(f"Recordings:  {RECORDINGS_DIR}")
    print(f"Web UI:      http://localhost:{args.port}")
    print("=" * 50)
    
    # Check services
    whisper_ok, whisper_msg = check_whisper_health()
    ollama_ok, ollama_msg = check_ollama_health()
    print(f"Whisper: {'✅' if whisper_ok else '❌'} {whisper_msg}")
    print(f"Ollama:  {'✅' if ollama_ok else '❌'} {ollama_msg}")
    print()
    
    # Check recordings
    recordings = list_recordings()
    print(f"Recordings: {len(recordings)} files found")
    
    if args.audio:
        print(f"Loading audio: {args.audio}")
        if args.auto_transcribe:
            print("Auto-transcribe: ENABLED")
    
    demo = create_ui(
        initial_audio=args.audio,
        auto_transcribe=args.auto_transcribe
    )
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
