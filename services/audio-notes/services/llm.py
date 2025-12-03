"""LLM (Large Language Model) services for summarization and chat."""

import json
import logging
from typing import Optional

import requests

from config import OLLAMA_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)


def summarize_streaming(transcript: str, system_prompt: Optional[str] = None, model: Optional[str] = None):
    """Summarize transcript using Ollama with streaming output."""
    use_model = model if model else OLLAMA_MODEL
    
    if not system_prompt:
        system_prompt = """You are an expert summarizer. Analyze the following transcript and provide:

1. **Summary** (1-2 paragraphs): A concise overview of the main content
2. **Key Points** (3-5 bullets): The most important takeaways
3. **Topics Discussed** (bullet list): Main subjects covered

Format your response in clean Markdown. Keep your response under 300 words."""

    prompt = f"""Please analyze this transcript:

---
{transcript}
---

Provide a concise summary following the format specified."""

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": use_model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": True
            },
            timeout=300,
            stream=True
        )
        
        if resp.status_code == 200:
            accumulated = ""
            for line in resp.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        accumulated += chunk
                        yield accumulated
                        
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
        else:
            yield f"Error from Ollama: {resp.status_code}"
            
    except Exception as e:
        yield f"Error: {e}"


def chat_with_context(message: str, history: list, transcript: str, summary: str = "", model: Optional[str] = None) -> str:
    """Chat with Ollama using transcript and summary as context."""
    use_model = model if model else OLLAMA_MODEL
    
    logger.info(f"chat_with_context called - transcript len: {len(transcript) if transcript else 0}, summary len: {len(summary) if summary else 0}")
    
    # Build conversation context
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
    
    system_prompt = f"""You are a helpful assistant answering questions about a transcript.

Here is the transcript for context:
---
{truncated_transcript}
---{summary_section}

Answer the user's questions based on this content. If the answer is not in the transcript or summary, say so."""

    # Build messages from history - ensure content is always a string
    messages = []
    for msg in history:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Handle Gradio's complex content format (can be list with text/images)
            if isinstance(content, list):
                # Extract text from list items
                text_parts = []
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                content = " ".join(text_parts)
            if content and isinstance(content, str):
                messages.append({"role": role, "content": content})
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            user_msg, assistant_msg = msg
            if user_msg:
                if isinstance(user_msg, str):
                    messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                if isinstance(assistant_msg, str):
                    messages.append({"role": "assistant", "content": assistant_msg})
    
    messages.append({"role": "user", "content": message})
    
    logger.info(f"Chat request with {len(messages)} messages to model {use_model}")
    logger.info(f"System prompt length: {len(system_prompt)}, first 200 chars: {system_prompt[:200]}")
    
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": use_model,
                "messages": messages,
                "system": system_prompt,
                "stream": False
            },
            timeout=60
        )
        
        if resp.status_code == 200:
            data = resp.json()
            return data.get("message", {}).get("content", "No response")
        else:
            logger.error(f"Ollama chat error {resp.status_code}: {resp.text[:500]}")
            return f"Error: {resp.status_code}"
            
    except Exception as e:
        logger.error(f"Chat exception: {e}")
        return f"Error: {e}"


def generate_chat_title(first_message: str, model: Optional[str] = None) -> str:
    """Generate a short title for the chat session."""
    use_model = model if model else OLLAMA_MODEL
    
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": use_model,
                "prompt": f"Generate a very short title (3-6 words max) for a conversation that starts with this question: \"{first_message}\". Return only the title, no quotes or punctuation.",
                "stream": False
            },
            timeout=15
        )
        
        if resp.status_code == 200:
            data = resp.json()
            title = data.get("response", "").strip()
            title = title.replace('"', '').replace("'", "").strip()
            if len(title) > 50:
                title = title[:47] + "..."
            return f"### 💬 {title}" if title else "### 💬 Chat Session"
        return "### 💬 Chat Session"
            
    except Exception:
        return "### 💬 Chat Session"
