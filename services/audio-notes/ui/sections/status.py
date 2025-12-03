"""Service status section UI component."""

import gradio as gr

from config import PARAKEET_URL, WHISPER_URL, OLLAMA_URL, OLLAMA_MODEL, RECORDINGS_DIR


def create_status_section():
    """Create the service status section.
    
    Returns:
        Dict of UI components for status
    """
    with gr.Accordion("⚙️ Service Status", open=False) as status_accordion:
        gr.Markdown(f"""
        **Backend Services:**
        - Parakeet ASR: `{PARAKEET_URL}`
        - Whisper ASR: `{WHISPER_URL}`
        - Ollama LLM: `{OLLAMA_URL}` (model: `{OLLAMA_MODEL}`)
        - Recordings: `{RECORDINGS_DIR}`
        """)
        refresh_btn = gr.Button("🔄 Check Services")
        service_status = gr.Markdown("")
    
    return {
        'status_accordion': status_accordion,
        'refresh_btn': refresh_btn,
        'service_status': service_status,
    }
