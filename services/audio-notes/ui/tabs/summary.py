"""Summary tab UI component."""

import gradio as gr


def create_summary_tab():
    """Create the summary tab.
    
    Returns:
        Dict of UI components for summary tab
    """
    with gr.TabItem("📋 Summary", id=0, interactive=False) as summary_tab:
        summary_output = gr.Markdown(label="Summary")
    
    return {
        'summary_tab': summary_tab,
        'summary_output': summary_output,
    }
