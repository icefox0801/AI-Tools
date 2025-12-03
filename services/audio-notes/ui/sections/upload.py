"""Upload section UI component."""

import gradio as gr


def create_upload_section():
    """Create the upload section for audio files.
    
    Returns:
        Dict of UI components for upload
    """
    with gr.Accordion("📤 Upload Audio", open=False) as upload_accordion:
        file_input = gr.File(
            label="Drop audio file here",
            file_types=[".wav", ".mp3", ".m4a", ".ogg", ".flac"]
        )
        upload_status = gr.Markdown("")
    
    return {
        'upload_accordion': upload_accordion,
        'file_input': file_input,
        'upload_status': upload_status,
    }
