"""Transcript tab UI component."""

import gradio as gr


def create_transcript_tab():
    """Create the transcript tab.

    Returns:
        Dict of UI components for transcript tab
    """
    with gr.TabItem("📜 Full Transcript", id=0) as transcript_tab:
        transcript_output = gr.Markdown(
            value="*Select recordings and transcribe to see the full transcript here.*",
            label="Transcript",
        )

    return {
        "transcript_tab": transcript_tab,
        "transcript_output": transcript_output,
    }
