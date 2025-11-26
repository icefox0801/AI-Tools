"""
Voice Transcription - Gradio UI
Simple streaming transcription via WebSocket
"""

import gradio as gr
import numpy as np
import websockets
import asyncio
import json
import time


async def transcribe_audio(audio_data: bytes, language: str = "en") -> str:
    """Send audio to whisper service and get transcription."""
    uri = "ws://whisper-asr:8000/transcribe"
    
    try:
        async with websockets.connect(uri, close_timeout=5) as ws:
            # Send config
            await ws.send(json.dumps({"language": language, "task": "transcribe"}))
            
            # Wait for ready
            response = await ws.recv()
            data = json.loads(response)
            if data.get("status") != "ready":
                return ""
            
            # Send audio
            await ws.send(audio_data)
            
            # Wait a bit for processing then close to get result
            await asyncio.sleep(0.5)
            
            # Try to receive results
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=3.0)
                result = json.loads(response)
                return result.get("text", "").strip()
            except asyncio.TimeoutError:
                return ""
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""


def process_audio(audio_input, transcript_state):
    """Process uploaded or recorded audio."""
    if audio_input is None:
        return transcript_state or "Click record and speak..."
    
    sample_rate, audio_data = audio_input
    
    # Convert to float32
    if audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32767.0
    elif audio_data.dtype == np.int32:
        audio_float = audio_data.astype(np.float32) / 2147483647.0
    else:
        audio_float = audio_data.astype(np.float32)
    
    # Mono conversion
    if len(audio_float.shape) > 1:
        audio_float = audio_float.mean(axis=1)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        ratio = 16000 / sample_rate
        new_length = int(len(audio_float) * ratio)
        if new_length > 0:
            indices = np.linspace(0, len(audio_float) - 1, new_length).astype(int)
            audio_float = audio_float[indices]
    
    # Skip if too short (less than 0.3 seconds)
    if len(audio_float) < 4800:
        return transcript_state or "Listening..."
    
    # Convert to int16 bytes for transmission
    audio_int16 = (audio_float * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    
    # Transcribe
    result = asyncio.run(transcribe_audio(audio_bytes))
    
    if result:
        current = transcript_state or ""
        if current and not current.endswith(" "):
            current += " "
        return current + result
    
    return transcript_state or "Listening..."


def clear_transcript():
    """Clear the transcript."""
    return ""


# Build Gradio interface
with gr.Blocks(title="Voice Transcribe") as demo:
    
    gr.Markdown("""
    # Voice Transcribe
    ### Real-time speech-to-text powered by Faster-Whisper AI
    
    Record audio and it will be transcribed automatically.
    Audio is filtered to isolate human voice frequencies (100Hz - 3500Hz).
    """)
    
    audio_input = gr.Audio(
        sources=["microphone"],
        type="numpy",
        label="Click to Record",
        streaming=True,
    )
    
    transcript_output = gr.Textbox(
        label="Transcript",
        placeholder="Click the microphone and start speaking...",
        lines=12,
        max_lines=25,
        interactive=False,
    )
    
    clear_btn = gr.Button("Clear Transcript", variant="secondary")
    
    gr.Markdown("""
    ---
    *Powered by Faster-Whisper | GPU Accelerated | Large-v3 Model*
    """)
    
    # Stream handler
    audio_input.stream(
        fn=process_audio,
        inputs=[audio_input, transcript_output],
        outputs=[transcript_output],
        stream_every=2.0,
    )
    
    clear_btn.click(
        fn=clear_transcript,
        outputs=[transcript_output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=8080,
        show_error=True
    )
