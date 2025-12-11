"""Summarize section UI component."""

import gradio as gr
from config import OLLAMA_MODEL

from services import get_ollama_models


def create_summarize_section():
    """Create the summarize section with LLM selection and prompt.

    Returns:
        Dict of UI components for summarization
    """
    with gr.Accordion("✨ Summarize", open=True) as summarize_accordion:
        llm_model_dropdown = gr.Dropdown(
            choices=get_ollama_models(),
            value=OLLAMA_MODEL,
            label="LLM Model",
            interactive=True,
        )

        summary_prompt = gr.Textbox(
            label="Summary Prompt (Auto-selected based on transcript length)",
            placeholder="Prompt will be automatically populated after transcription based on length...",
            value="",
            lines=6,
            max_lines=12,
            interactive=False,
            info="Edit this prompt to customize the summarization if needed",
        )

        summarize_btn = gr.Button("✨ Summarize", variant="secondary", size="lg", interactive=False)

    return {
        "summarize_accordion": summarize_accordion,
        "llm_model_dropdown": llm_model_dropdown,
        "summary_prompt": summary_prompt,
        "summarize_btn": summarize_btn,
    }
