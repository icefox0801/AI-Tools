"""
Audio Summary Service - Transcription & AI Summarization

A Gradio web UI that:
1. Receives audio files from Live Captions
2. Transcribes using Whisper ASR service
3. Summarizes using Ollama LLM
4. Provides interactive chat for Q&A about the content

Usage:
  python audio_summary.py                    # Start web UI
  python audio_summary.py --port 7860        # Custom port
"""

import os
import io
import json
import time
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional
from datetime import datetime

import gradio as gr
import requests
import wave
import numpy as np

# Configuration
WHISPER_URL = os.getenv("WHISPER_URL", "http://localhost:8003")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Logging
logging.basicConfig(level=logging.INFO)
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


def transcribe_audio(audio_path: str) -> tuple[str, float]:
    """Transcribe audio file using Whisper service.
    
    Args:
        audio_path: Path to WAV file
        
    Returns:
        Tuple of (transcript_text, duration_seconds)
    """
    logger.info(f"Transcribing: {audio_path}")
    
    # Read audio file
    with open(audio_path, 'rb') as f:
        audio_data = f.read()
    
    # Get duration
    with wave.open(audio_path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / rate
    
    # Send to Whisper /transcribe endpoint
    files = {'file': ('audio.wav', audio_data, 'audio/wav')}
    
    try:
        resp = requests.post(
            f"{WHISPER_URL}/transcribe",
            files=files,
            timeout=300  # 5 minutes for long audio
        )
        
        if resp.status_code == 200:
            data = resp.json()
            text = data.get("text", "")
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
            timeout=120
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
    system_prompt = f"""You are a helpful assistant answering questions about a transcript.

Here is the transcript for context:
---
{transcript[:8000]}  # Limit context size
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


def process_audio(audio_file) -> tuple[str, str, str]:
    """Process uploaded audio file.
    
    Args:
        audio_file: Uploaded file path or Gradio audio tuple
        
    Returns:
        Tuple of (transcript, summary, status)
    """
    if audio_file is None:
        return "", "", "⚠️ Please upload an audio file"
    
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
    status = "⏳ Transcribing audio..."
    transcript, duration = transcribe_audio(audio_path)
    
    if transcript.startswith("Error"):
        return "", "", f"❌ Transcription failed: {transcript}"
    
    # Summarize
    status = "⏳ Generating summary..."
    summary = summarize_with_ollama(transcript)
    
    status = f"✅ Done! Transcribed {duration:.1f}s of audio ({len(transcript)} chars)"
    
    return transcript, summary, status


def create_ui():
    """Create Gradio interface."""
    
    # State for transcript (used in chat)
    transcript_state = gr.State("")
    
    with gr.Blocks(
        title="Audio Summary",
        theme=gr.themes.Soft(),
        css="""
        .transcript-box { font-family: monospace; }
        .summary-box { font-size: 1.1em; }
        """
    ) as demo:
        gr.Markdown("""
        # 🎙️ Audio Summary
        
        Upload audio or paste a recording from Live Captions to get:
        - **Full Transcript** - Complete text transcription
        - **AI Summary** - Key points and overview
        - **Interactive Chat** - Ask questions about the content
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                audio_input = gr.Audio(
                    label="Audio File",
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                
                file_input = gr.File(
                    label="Or drop a WAV file here",
                    file_types=[".wav", ".mp3", ".m4a", ".ogg"],
                    visible=True
                )
                
                process_btn = gr.Button("📝 Transcribe & Summarize", variant="primary", size="lg")
                status_text = gr.Textbox(label="Status", interactive=False)
                
                # Service status
                with gr.Accordion("Service Status", open=False):
                    gr.Markdown(f"""
                    - **Whisper ASR**: `{WHISPER_URL}`
                    - **Ollama LLM**: `{OLLAMA_URL}` (model: `{OLLAMA_MODEL}`)
                    """)
                    refresh_btn = gr.Button("🔄 Check Services")
                    service_status = gr.Markdown("")
            
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("📄 Summary"):
                        summary_output = gr.Markdown(
                            label="Summary",
                            elem_classes=["summary-box"]
                        )
                    
                    with gr.TabItem("📜 Full Transcript"):
                        transcript_output = gr.Textbox(
                            label="Transcript",
                            lines=20,
                            max_lines=50,
                            elem_classes=["transcript-box"]
                        )
                    
                    with gr.TabItem("💬 Chat"):
                        chatbot = gr.Chatbot(
                            label="Ask questions about the content",
                            height=400
                        )
                        chat_input = gr.Textbox(
                            label="Your question",
                            placeholder="What was discussed? What are the main points?",
                            lines=2
                        )
                        chat_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Clear Chat")
        
        # Event handlers
        def on_process(audio, file):
            # Prefer file input over audio input
            audio_path = file.name if file else audio
            transcript, summary, status = process_audio(audio_path)
            return transcript, summary, status, transcript
        
        def on_chat(message, history, transcript):
            if not transcript:
                return history + [[message, "⚠️ Please transcribe audio first"]], ""
            response = chat_with_context(message, history, transcript)
            return history + [[message, response]], ""
        
        def on_refresh():
            whisper_ok, whisper_msg = check_whisper_health()
            ollama_ok, ollama_msg = check_ollama_health()
            return f"""
            - Whisper: {"✅" if whisper_ok else "❌"} {whisper_msg}
            - Ollama: {"✅" if ollama_ok else "❌"} {ollama_msg}
            """
        
        process_btn.click(
            on_process,
            inputs=[audio_input, file_input],
            outputs=[transcript_output, summary_output, status_text, transcript_state]
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
        
        clear_btn.click(lambda: [], None, chatbot)
        refresh_btn.click(on_refresh, None, service_status)
    
    return demo


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio Summary Service")
    parser.add_argument('--port', type=int, default=7860, help='Port to run on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--share', action='store_true', help='Create public link')
    args = parser.parse_args()
    
    print("=" * 50)
    print("🎙️ Audio Summary Service")
    print("=" * 50)
    print(f"Whisper ASR: {WHISPER_URL}")
    print(f"Ollama LLM:  {OLLAMA_URL} (model: {OLLAMA_MODEL})")
    print(f"Web UI:      http://localhost:{args.port}")
    print("=" * 50)
    
    # Check services
    whisper_ok, whisper_msg = check_whisper_health()
    ollama_ok, ollama_msg = check_ollama_health()
    print(f"Whisper: {'✅' if whisper_ok else '❌'} {whisper_msg}")
    print(f"Ollama:  {'✅' if ollama_ok else '❌'} {ollama_msg}")
    print()
    
    demo = create_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
