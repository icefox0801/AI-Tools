"""Audio generation helpers for benchmarking notebooks."""

import random
from io import BytesIO
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import requests
from IPython.display import Audio, display


def create_audio_settings_ui(
    default_word_count: int = 500, default_speed: float = 1.5
) -> Tuple[object, Callable]:
    """
    Create interactive UI widgets for audio generation settings.

    Args:
        default_word_count: Default word count (default: 500)
        default_speed: Default speech speed (default: 1.5)

    Returns:
        tuple: (settings_ui, get_audio_settings)
            - settings_ui: VBox widget to display
            - get_audio_settings: Function to get current settings dict
    """
    from ipywidgets import IntSlider, FloatSlider, VBox, Label

    # Create interactive widgets
    word_count_widget = IntSlider(
        value=default_word_count,
        min=100,
        max=2000,
        step=50,
        description="Word Count:",
        style={"description_width": "120px"},
    )

    speech_speed_widget = FloatSlider(
        value=default_speed,
        min=0.5,
        max=2.5,
        step=0.1,
        description="Speech Speed:",
        style={"description_width": "120px"},
    )

    # Combine widgets
    header = Label("⚙️ Audio Generation Settings")
    settings_ui = VBox([header, word_count_widget, speech_speed_widget])

    # Getter function
    def get_audio_settings():
        """Get current audio generation settings."""
        return {"word_count": word_count_widget.value, "speech_speed": speech_speed_widget.value}

    return settings_ui, get_audio_settings


def unload_gpu_models(backend_urls: Dict[str, str]) -> None:
    """
    Unload GPU models from all ASR backends to free memory.

    Args:
        backend_urls: Dictionary mapping backend names to their URLs
    """
    print("🔄 Unloading GPU models to free memory for Ollama...")
    for backend_name, backend_url in backend_urls.items():
        try:
            response = requests.post(f"{backend_url}/unload", timeout=5)
            if response.status_code == 200:
                print(f"  ✓ {backend_name}: GPU models unloaded")
            else:
                print(f"  ⚠ {backend_name}: unload returned status {response.status_code}")
        except Exception as e:
            print(f"  ⚠ {backend_name}: {e}")


def generate_and_prepare_audio(
    word_count: int, speech_speed: float = 1.5, sample_rate: int = 16000
) -> tuple:
    """
    Complete workflow: generate text and convert to speech.

    Args:
        word_count: Target number of words to generate
        speech_speed: Speech speed multiplier (default: 1.5)
        sample_rate: Target sample rate (default: 16000)

    Returns:
        tuple: (audio_data, text) ready for benchmarking
    """
    print("🎙️ Generating test audio...")
    print(f"⚙️ Settings: {word_count} words, {speech_speed}x speed, {sample_rate}Hz")

    # Generate text
    text = generate_text_with_ollama(word_count=word_count)

    # Convert to speech
    audio_data = text_to_speech_gtts(text, sample_rate=sample_rate, speed=speech_speed)

    print(f"✅ Audio generated! Duration: {len(audio_data)/sample_rate:.2f}s")

    # Display audio player
    print(f"\n🔊 Audio Playback")
    display(Audio(data=audio_data, rate=sample_rate, autoplay=False))

    # Display transcript preview (1 line max)
    if text:
        preview_length = 60
        if len(text) > preview_length:
            print(f"📄 Transcript: {text[:preview_length]}... ({len(text)} chars)")
        else:
            print(f"📄 Transcript: {text}")

    print(f"✅ Ready for benchmarking!")

    return audio_data, text


def generate_text_with_ollama(
    word_count: int = 500, models: Optional[list] = None, base_urls: Optional[list] = None
) -> str:
    """Generate random text using Ollama API.

    Args:
        word_count: Target number of words to generate
        models: List of model names to try (default: qwen3:14b, gemma3:12b, deepseek-r1:14b)
        base_urls: List of base URLs to try (default: localhost and ollama)

    Returns:
        Generated text string, or falls back to generate_fallback_text() if Ollama is unavailable
    """
    # Default models
    if models is None:
        models = ["qwen3:14b", "gemma3:12b", "deepseek-r1:14b"]

    # Default base URLs
    if base_urls is None:
        base_urls = ["http://localhost:11434", "http://ollama:11434"]

    for base_url in base_urls:
        for model in models:
            try:
                ollama_url = f"{base_url}/api/generate"

                prompt = f"Write a natural, coherent passage about technology, artificial intelligence, or science. Make it around {word_count} words. Be conversational and varied."

                response = requests.post(
                    ollama_url, json={"model": model, "prompt": prompt, "stream": False}, timeout=60
                )

                if response.status_code == 200:
                    response_data = response.json()
                    text = response_data.get("response", "")
                    if text:
                        print(
                            f"✓ Generated text from Ollama model '{model}' ({len(text.split())} words)"
                        )
                        return text

            except Exception:
                continue

    print("⚠️ Ollama not accessible, using high-quality fallback text")
    return generate_fallback_text(word_count)


def generate_fallback_text(word_count: int = 500) -> str:
    """Generate high-quality, varied fallback text if Ollama is unavailable.

    Args:
        word_count: Target number of words

    Returns:
        Combined text from professional passages about AI/speech technology
    """
    passages = [
        "Artificial intelligence has transformed how we interact with technology. From voice assistants that understand natural language to recommendation systems that predict our preferences, machine learning algorithms have become integral to modern life.",
        "Speech recognition technology relies on complex neural networks trained on thousands of hours of audio data. These models learn to identify patterns in sound waves and map them to written text with remarkable accuracy.",
        "The development of real-time transcription systems has opened new possibilities for accessibility. People who are deaf or hard of hearing can now follow conversations, lectures, and media content through live captions.",
        "Natural language processing combines linguistics with computer science to help machines understand human communication. This field has advanced significantly with the introduction of transformer models and attention mechanisms.",
        "Voice interfaces are changing how we interact with computers. Instead of typing commands, users can simply speak naturally, making technology more accessible to everyone regardless of their technical expertise.",
        "The future of human-computer interaction will likely involve seamless multimodal communication. Systems that can understand speech, gestures, and context simultaneously will create more intuitive user experiences.",
        "Deep learning has revolutionized acoustic modeling for speech recognition. Modern systems can handle diverse accents, background noise, and speaking styles that would have been impossible for earlier technologies.",
        "Cloud-based speech services enable developers to integrate powerful recognition capabilities into their applications without building infrastructure from scratch. This democratization of AI technology accelerates innovation.",
        "Privacy concerns around voice data have led to the development of on-device speech recognition models. These systems process audio locally, ensuring sensitive information never leaves the user's device.",
        "The accuracy of automatic speech recognition continues to improve through transfer learning and multilingual training. Models can now recognize hundreds of languages and adapt to specialized vocabularies.",
    ]

    # Shuffle for variety
    random.shuffle(passages)

    # Combine passages to reach desired word count
    current_text = []
    current_words = 0

    for passage in passages:
        current_text.append(passage)
        current_words += len(passage.split())
        if current_words >= word_count:
            break

    full_text = " ".join(current_text)

    # Trim to approximate word count
    words = full_text.split()
    if len(words) > word_count:
        trimmed = " ".join(words[:word_count])
    else:
        trimmed = full_text

    return trimmed


def text_to_speech_gtts(text: str, sample_rate: int = 16000, speed: float = 1.5) -> np.ndarray:
    """Convert text to speech using gTTS and return as numpy array.

    Args:
        text: Text to convert to speech
        sample_rate: Target sample rate for output audio
        speed: Speed multiplier (1.0 = normal, >1.0 = faster, <1.0 = slower)

    Returns:
        Audio data as numpy array (float32, mono)

    Note:
        This function will auto-install gTTS if not available.
        Speed adjustment uses librosa.effects.time_stretch()
    """
    try:
        from gtts import gTTS
        import librosa
        import tempfile

        # Create TTS - gTTS is always slow=False for normal speed
        # We'll speed it up using librosa time_stretch
        tts = gTTS(text=text, lang="en", slow=False)

        # Save to temporary bytes buffer
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)

        # Load audio from bytes
        audio, sr = librosa.load(fp, sr=sample_rate, mono=True)

        # Speed up audio using time stretch if speed > 1.0
        if speed != 1.0:
            audio = librosa.effects.time_stretch(audio, rate=speed)

        print(
            f"✓ Generated audio: {len(audio)} samples ({len(audio)/sample_rate:.2f}s) at {speed}x speed"
        )
        return audio

    except ImportError:
        print("Installing gTTS...")
        import subprocess
        import sys

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-i",
                "https://pypi.tuna.tsinghua.edu.cn/simple",
                "gtts",
            ]
        )
        from gtts import gTTS

        return text_to_speech_gtts(text, sample_rate, speed)
