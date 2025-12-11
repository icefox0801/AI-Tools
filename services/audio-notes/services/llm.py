"""LLM (Large Language Model) services for summarization and chat."""

import json
import logging

import requests
from config import OLLAMA_MODEL, OLLAMA_URL, PARAKEET_URL, WHISPER_URL

logger = logging.getLogger(__name__)


def get_summary_prompt_for_length(word_count: int) -> str:
    """Get the appropriate summary prompt based on transcript length.

    Args:
        word_count: Number of words in the transcript

    Returns:
        Appropriate system prompt for the transcript length
    """
    if word_count < 150:
        return """You are an expert summarizer. This is a SHORT transcript.

Instructions:
- Provide 1-2 concise sentences capturing the main point
- Focus on the core message or conclusion
- Start directly with content (no "This transcript...")
- Use active voice, present tense
- Preserve any technical terms, names, or numbers accurately"""

    elif word_count < 500:
        return """You are an expert summarizer. This is a MEDIUM-length transcript.

Instructions:
- Write a focused paragraph (3-5 sentences)
- Structure: Context → Key points → Conclusion
- Include the main topic, key points, and any conclusions
- Start directly with content (no "This transcript...")
- Use active voice, present tense
- Preserve technical terms, names, and numbers accurately
- Use markdown formatting for clarity"""

    else:
        return """You are an expert summarizer. This is a LONG transcript.

Instructions:
- Start with a 2-3 sentence executive summary
- Follow with bullet points of main topics (3-7 bullets maximum)
- Each bullet should contain one clear idea with supporting detail
- End with conclusion or action items if present
- Start directly with content (no "This transcript...")
- Use active voice, present tense
- Preserve technical terms, names, and numbers accurately
- Use markdown formatting (bold for emphasis, bullets for lists)
- Prioritize clarity - ensure the summary is understandable"""


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
    result = {"actions": [], "memory_freed_gb": 0.0, "message": ""}

    # Try to unload Parakeet
    try:
        resp = requests.post(f"{PARAKEET_URL}/unload", timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "unloaded":
                freed = data.get("gpu_memory_used_gb", 0) or 6.0  # Estimate 6GB if not reported
                result["actions"].append(f"Unloaded Parakeet ASR (freed ~{freed:.1f}GB)")
                result["memory_freed_gb"] += freed
                logger.info("Unloaded Parakeet ASR to free GPU memory")
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
                logger.info("Unloaded Whisper ASR to free GPU memory")
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


def summarize_streaming(
    transcript: str,
    system_prompt: str | None = None,
    model: str | None = None,
    prepare_gpu: bool = True,
):
    """
    Summarize transcript using Ollama with streaming output.

    Args:
        transcript: The text to summarize
        system_prompt: Optional custom system prompt (if None, auto-selects based on length)
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

    # Auto-select prompt based on transcript length if not provided
    if system_prompt is None or not system_prompt.strip():
        word_count = len(transcript.split())
        system_prompt = get_summary_prompt_for_length(word_count)
        logger.info(f"Auto-selected prompt for {word_count} words")

    prompt = f"""Summarize this transcript:

{transcript}"""

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": use_model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": True,
            },
            timeout=300,
            stream=True,
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


def generate_chat_title(first_message: str, model: str | None = None) -> str:
    """Generate a short title for the chat session."""
    use_model = model if model else OLLAMA_MODEL

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": use_model,
                "prompt": f'Generate a very short title (3-6 words max) for a conversation that starts with this question: "{first_message}". Return only the title, no quotes or punctuation.',
                "stream": False,
            },
            timeout=15,
        )

        if resp.status_code == 200:
            data = resp.json()
            title = data.get("response", "").strip()
            title = title.replace('"', "").replace("'", "").strip()
            if len(title) > 50:
                title = title[:47] + "..."
            return f"### 💬 {title}" if title else "### 💬 Chat Session"
        return "### 💬 Chat Session"

    except Exception:
        return "### 💬 Chat Session"
