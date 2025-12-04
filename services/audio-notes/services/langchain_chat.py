"""LangChain-based chat service with conversation memory.

Provides streaming chat with Ollama using LangChain for better
conversation management and memory handling.
"""

import logging
from collections.abc import Generator

from config import OLLAMA_MODEL, OLLAMA_URL

logger = logging.getLogger(__name__)

# Lazy imports to avoid startup overhead
_ChatOllama = None
_ChatPromptTemplate = None
_MessagesPlaceholder = None
_HumanMessage = None
_AIMessage = None
_SystemMessage = None


def _ensure_langchain_imports():
    """Lazy import LangChain components."""
    global _ChatOllama, _ChatPromptTemplate, _MessagesPlaceholder
    global _HumanMessage, _AIMessage, _SystemMessage

    if _ChatOllama is None:
        from langchain_ollama import ChatOllama

        _ChatOllama = ChatOllama

    if _ChatPromptTemplate is None:
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        _ChatPromptTemplate = ChatPromptTemplate
        _MessagesPlaceholder = MessagesPlaceholder

    if _HumanMessage is None:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        _HumanMessage = HumanMessage
        _AIMessage = AIMessage
        _SystemMessage = SystemMessage


def _build_system_prompt(transcript: str, summary: str = "") -> str:
    """Build the system prompt with transcript and summary context."""
    context_limit = 6000
    truncated_transcript = transcript[:context_limit] if transcript else ""
    if transcript and len(transcript) > context_limit:
        truncated_transcript += "\n...[transcript truncated]..."

    summary_section = ""
    if summary and not summary.startswith("⏳") and not summary.startswith("⚠️"):
        summary_section = f"""

Here is a summary of the transcript:
---
{summary[:2000]}
---"""

    return f"""You are a helpful assistant answering questions about a transcript.

Here is the transcript for context:
---
{truncated_transcript}
---{summary_section}

Answer the user's questions based on this content. If the answer is not in the transcript or summary, say so."""


def _convert_history_to_messages(history: list) -> list:
    """Convert Gradio chat history to LangChain message format."""
    _ensure_langchain_imports()
    messages = []

    for msg in history:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Handle Gradio's complex content format (can be list with text/images)
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                content = " ".join(text_parts)
            if content and isinstance(content, str):
                if role == "user":
                    messages.append(_HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(_AIMessage(content=content))
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            user_msg, assistant_msg = msg
            if user_msg and isinstance(user_msg, str):
                messages.append(_HumanMessage(content=user_msg))
            if assistant_msg and isinstance(assistant_msg, str):
                messages.append(_AIMessage(content=assistant_msg))

    return messages


def chat_streaming(
    message: str,
    history: list,
    transcript: str,
    summary: str = "",
    model: str | None = None,
) -> Generator[str, None, None]:
    """
    Chat with Ollama using LangChain with streaming output.

    Args:
        message: User's message
        history: Conversation history from Gradio
        transcript: Full transcript for context
        summary: Summary for context
        model: Optional model name override

    Yields:
        Accumulated response chunks
    """
    use_model = model if model else OLLAMA_MODEL

    # Extract host from OLLAMA_URL (remove /api suffix if present)
    base_url = OLLAMA_URL.replace("/api", "").rstrip("/")

    logger.info(
        f"LangChain chat - transcript: {len(transcript) if transcript else 0} chars, "
        f"summary: {len(summary) if summary else 0} chars, model: {use_model}"
    )

    try:
        _ensure_langchain_imports()

        # Create ChatOllama instance
        llm = _ChatOllama(
            model=use_model,
            base_url=base_url,
            temperature=0.7,
            streaming=True,
        )

        # Build system prompt with context
        system_prompt = _build_system_prompt(transcript, summary)

        # Convert history to LangChain messages
        chat_history = _convert_history_to_messages(history)

        # Build full message list
        messages = [_SystemMessage(content=system_prompt)]
        messages.extend(chat_history)
        messages.append(_HumanMessage(content=message))

        logger.info(f"Sending {len(messages)} messages to LangChain ChatOllama")

        # Stream the response
        full_response = ""
        for chunk in llm.stream(messages):
            if hasattr(chunk, "content") and chunk.content:
                full_response += chunk.content
                yield full_response

        if not full_response:
            yield "No response received from the model."

    except ImportError as e:
        logger.error(f"LangChain import error: {e}")
        yield f"Error: LangChain not properly installed. {e}"
    except Exception as e:
        logger.error(f"LangChain chat error: {e}")
        yield f"Error: {e}"


def generate_title(first_message: str, model: str | None = None) -> str:
    """Generate a short title for the chat session using LangChain."""
    _ensure_langchain_imports()

    use_model = model if model else OLLAMA_MODEL
    base_url = OLLAMA_URL.replace("/api", "").rstrip("/")

    try:
        llm = _ChatOllama(
            model=use_model,
            base_url=base_url,
            temperature=0.7,
        )

        prompt = f'Generate a very short title (3-6 words max) for a conversation that starts with this question: "{first_message}". Return only the title, no quotes or punctuation.'
        messages = [_HumanMessage(content=prompt)]

        response = llm.invoke(messages)
        title = response.content.strip() if hasattr(response, "content") else ""
        title = title.replace('"', "").replace("'", "").strip()

        if len(title) > 50:
            title = title[:47] + "..."

        return f"### 💬 {title}" if title else "### 💬 Chat Session"

    except Exception as e:
        logger.error(f"Title generation error: {e}")
        return "### 💬 Chat Session"
