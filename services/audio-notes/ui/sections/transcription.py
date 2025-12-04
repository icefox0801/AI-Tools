"""Transcription section UI component."""

import gradio as gr


def create_transcription_section():
    """Create the transcription section with backend selection and button.

    Returns:
        Dict of UI components for transcription
    """
    with gr.Accordion("📝 Transcription", open=True) as transcription_accordion:
        backend_radio = gr.Radio(
            choices=["Whisper", "Parakeet"],
            value="Whisper",
            label="ASR Backend",
            interactive=True,
        )

        batch_transcribe_btn = gr.Button(
            "📝 Transcribe", variant="primary", size="lg", interactive=False
        )

        batch_status = gr.Markdown("*Select recordings and click Transcribe.*")

    return {
        "transcription_accordion": transcription_accordion,
        "backend_radio": backend_radio,
        "batch_transcribe_btn": batch_transcribe_btn,
        "batch_status": batch_status,
    }
