"""
Voice Transcribe - Web Application

GPU-accelerated speech-to-text using NVIDIA Parakeet.
Gradio-based web interface with streaming audio input.
"""

import gradio as gr
import logging
from pathlib import Path

from core import AudioProcessor, TranscriptionClient, SessionState

# ============== Configuration ==============

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize components
audio_processor = AudioProcessor()
client = TranscriptionClient()
session = SessionState()

# Transcription settings
TRANSCRIBE_INTERVAL = 2.0  # Transcribe after 2 seconds of audio


# ============== Handlers ==============

def process_audio(audio_chunk):
    """
    Process streaming audio - server handles accumulation and punctuation.
    
    Strategy: Send audio chunks to server, which accumulates text and
    applies punctuation to the full transcript. We just display the result.
    
    Args:
        audio_chunk: Tuple of (sample_rate, numpy_array) from Gradio
    
    Returns:
        Current transcript display string
    """
    if audio_chunk is None:
        return session.last_display
    
    sample_rate, chunk_data = audio_chunk
    
    # Preprocess audio
    audio = audio_processor.preprocess(sample_rate, chunk_data)
    
    # Skip very short chunks
    duration = audio_processor.get_duration(audio)
    if duration < 0.05:
        return session.last_display
    
    # Accumulate audio locally
    session.append_audio(audio)
    
    # Transcribe when we have enough pending audio
    if session.pending_duration >= TRANSCRIBE_INTERVAL:
        pending_audio = session.get_pending_audio()
        
        logger.info(f"Sending {session.pending_duration:.1f}s of audio...")
        
        audio_bytes = audio_processor.to_int16_bytes(pending_audio)
        result = client.transcribe_sync(audio_bytes, session.session_id)
        
        if result.success and result.text:
            # Server returns FULL punctuated transcript - just display it
            session.set_full_transcript(result.text)
            logger.info(f"Transcript: ...{result.text[-80:]}")
        
        # Clear pending audio after transcription
        session.clear_pending_audio()
    
    return session.build_display()


def clear_transcript():
    """Clear all transcript history and reset session."""
    client.clear_session_sync(session.session_id)
    session.reset()
    logger.info(f"New session: {session.session_id[:8]}...")
    return ""


# ============== UI ==============

def create_ui():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="Voice Transcribe") as app:
        
        # Header
        gr.Markdown("# 🎙️ Voice Transcribe")
        gr.Markdown("GPU-accelerated • **Parakeet RNNT 1.1B** • Real-time streaming")
        
        # Main components
        audio_input = gr.Audio(
            sources=["microphone"],
            type="numpy",
            label="Click to Record",
            streaming=True,
        )
        
        transcript_output = gr.Textbox(
            label="Full Transcript",
            placeholder="Start speaking... transcription grows as you speak.",
            lines=12,
            max_lines=30,
            interactive=False,
        )
        
        # Controls
        clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
        
        # Help text
        gr.Markdown("""
        **How it works:** Audio is split at natural speech pauses. 
        Each segment is transcribed and appended to the full transcript.
        """)
        
        # Event handlers
        audio_input.stream(
            fn=process_audio,
            inputs=[audio_input],
            outputs=[transcript_output],
            time_limit=300,
            stream_every=0.3,
        )
        
        clear_btn.click(fn=clear_transcript, outputs=[transcript_output])
    
    return app


# ============== Main ==============

demo = create_ui()

if __name__ == "__main__":
    app_dir = Path(__file__).parent
    favicon = app_dir / "static" / "favicon.ico"
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=8080,
        show_error=True,
        favicon_path=str(favicon) if favicon.exists() else None,
    )
