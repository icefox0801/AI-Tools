"""LLM (Large Language Model) services for summarization and chat."""

import json
import logging
from typing import Optional

import requests

from config import OLLAMA_URL, OLLAMA_MODEL, PARAKEET_URL, WHISPER_URL

logger = logging.getLogger(__name__)


def prepare_gpu_for_llm(required_memory_gb: float = 8.0) -> dict:
    """
    Prepare GPU memory for LLM by unloading ASR models if needed.
    
    This is a synchronous version that can be called before summarization.
    It unloads Parakeet/Whisper models to free GPU memory for the LLM.
    
    Args:
        required_memory_gb: Approximate memory needed for LLM
        
    Returns:
        Dict with preparation status
    """
    result = {
        "actions": [],
        "memory_freed_gb": 0.0,
        "message": ""
    }
    
    # Try to unload Parakeet
    try:
        resp = requests.post(f"{PARAKEET_URL}/unload", timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "unloaded":
                freed = data.get("gpu_memory_used_gb", 0) or 6.0  # Estimate 6GB if not reported
                result["actions"].append(f"Unloaded Parakeet ASR (freed ~{freed:.1f}GB)")
                result["memory_freed_gb"] += freed
                logger.info(f"Unloaded Parakeet ASR to free GPU memory")
            else:
                result["actions"].append(f"Parakeet: {data.get('message', 'already unloaded')}")
    except Exception as e:
        logger.debug(f"Could not unload Parakeet: {e}")
        result["actions"].append(f"Parakeet unavailable: {e}")
    
    # Try to unload Whisper
    try:
        resp = requests.post(f"{WHISPER_URL}/unload", timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "unloaded":
                freed = data.get("gpu_memory_used_gb", 0) or 4.0  # Estimate 4GB if not reported
                result["actions"].append(f"Unloaded Whisper ASR (freed ~{freed:.1f}GB)")
                result["memory_freed_gb"] += freed
                logger.info(f"Unloaded Whisper ASR to free GPU memory")
            else:
                result["actions"].append(f"Whisper: {data.get('message', 'already unloaded')}")
    except Exception as e:
        logger.debug(f"Could not unload Whisper: {e}")
        result["actions"].append(f"Whisper unavailable: {e}")
    
    if result["memory_freed_gb"] > 0:
        result["message"] = f"Freed ~{result['memory_freed_gb']:.1f}GB GPU memory for LLM"
    else:
        result["message"] = "No ASR models to unload (GPU memory already available)"
    
    logger.info(result["message"])
    return result


def summarize_streaming(transcript: str, system_prompt: Optional[str] = None, model: Optional[str] = None, prepare_gpu: bool = True):
    """
    Summarize transcript using Ollama with streaming output.
    
    Args:
        transcript: The text to summarize
        system_prompt: Optional custom system prompt
        model: Optional model name override
        prepare_gpu: If True, unload ASR models first to free GPU memory
    """
    use_model = model if model else OLLAMA_MODEL
    
    # Prepare GPU memory by unloading ASR models
    if prepare_gpu:
        logger.info("Preparing GPU memory for LLM summarization...")
        prep_result = prepare_gpu_for_llm(required_memory_gb=8.0)
        if prep_result["memory_freed_gb"] > 0:
            yield f"⏳ {prep_result['message']}\\n\\n"
    
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


def chat_with_context(message: str, history: list, transcript: str, summary: str = "", model: Optional[str] = None, prepare_gpu: bool = True) -> str:
    """
    Chat with Ollama using transcript and summary as context.
    
    Args:
        message: User's message
        history: Conversation history
        transcript: Full transcript for context
        summary: Summary for context
        model: Optional model name override
        prepare_gpu: If True, unload ASR models first to free GPU memory
    """
    use_model = model if model else OLLAMA_MODEL
    
    # Prepare GPU memory by unloading ASR models (only on first chat message)
    if prepare_gpu and len(history) == 0:
        logger.info("Preparing GPU memory for LLM chat...")
        prepare_gpu_for_llm(required_memory_gb=8.0)
    
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
