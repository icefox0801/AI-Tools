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
            value="""You are an expert summarizer. Analyze the transcript and provide:

1. **Summary** (1-2 paragraphs): A concise overview of the main content
2. **Key Points** (3-5 bullets): The most important takeaways
3. **Topics Discussed** (bullet list): Main subjects covered

Format your response in clean Markdown. Keep your response under 300 words.""",
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
