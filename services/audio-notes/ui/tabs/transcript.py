"""Transcript tab UI component."""

import gradio as gr


def create_transcript_tab():
    """Create the transcript tab.
    
    Returns:
        Dict of UI components for transcript tab
    """
    with gr.TabItem("📜 Full Transcript", id=1) as transcript_tab:
        transcript_output = gr.Textbox(label="Transcript", lines=25, max_lines=50)
    
    return {
        'transcript_tab': transcript_tab,
        'transcript_output': transcript_output,
    }
