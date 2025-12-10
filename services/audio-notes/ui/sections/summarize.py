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
            label="Summary Prompt",
            placeholder="Custom instructions for summarization",
            value="""You are a concise summarizer. Summarize the transcript directly.

Rules:
- Keep summary length proportional to transcript length
- Short transcript (<100 words) = 1-2 sentences
- Medium transcript (100-500 words) = short paragraph with key points
- Long transcript (500+ words) = brief summary + bullet points
- No introductory phrases like "This transcript discusses..."
- Start directly with the content
- Format in clean Markdown""",
            lines=6,
            max_lines=10,
        )

        summarize_btn = gr.Button("✨ Summarize", variant="secondary", size="lg", interactive=False)

    return {
        "summarize_accordion": summarize_accordion,
        "llm_model_dropdown": llm_model_dropdown,
        "summary_prompt": summary_prompt,
        "summarize_btn": summarize_btn,
    }
