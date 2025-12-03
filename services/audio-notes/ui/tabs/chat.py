"""Chat tab UI component."""

import gradio as gr

from config import OLLAMA_MODEL
from services import get_ollama_models


def create_chat_tab():
    """Create the chat tab with chatbot and input.
    
    Returns:
        Dict of UI components for chat tab
    """
    with gr.TabItem("💬 Chat", id=2, interactive=False) as chat_tab:
        chat_title = gr.Markdown(
            value="*Start a conversation to generate a title...*",
            elem_id="chat-title"
        )
        
        gr.Markdown(
            "*ℹ️ The transcript and summary are included as context for this chat.*",
            elem_id="chat-context-info"
        )
        
        chatbot = gr.Chatbot(label="Ask questions about the content", height=300)
        
        with gr.Row(equal_height=True):
            chat_input = gr.Textbox(
                label="Your question (Ctrl+Enter to send)",
                placeholder="What was discussed? What are the main points?",
                lines=4,
                max_lines=6,
                scale=3,
                elem_id="chat-input"
            )
            
            with gr.Column(scale=1, min_width=200):
                chat_model_dropdown = gr.Dropdown(
                    choices=get_ollama_models(),
                    value=OLLAMA_MODEL,
                    label="Chat Model",
                    interactive=True
                )
                chat_btn = gr.Button("Send", variant="primary", size="lg", elem_id="chat-send-btn")
    
    return {
        'chat_tab': chat_tab,
        'chat_title': chat_title,
        'chatbot': chatbot,
        'chat_input': chat_input,
        'chat_model_dropdown': chat_model_dropdown,
        'chat_btn': chat_btn,
    }
