"""
Whisper ASR Service - GPU-accelerated streaming transcription
Using OpenAI Whisper Large V3 Turbo via HuggingFace Transformers

Features:
- Fast GPU inference with Flash Attention 2 or SDPA
- Silero VAD for speech detection (skip silence)
- Streaming WebSocket API with segment-based protocol
- Automatic language detection
- Native punctuation and capitalization (no external refiner needed)

Protocol:
1. Client connects to /stream
2. Client streams raw PCM audio (int16, 16kHz, mono)
3. Server sends: {"id": "s1", "text": "..."} for segments
"""

import asyncio
import io
import json
import os
import time

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from shared.logging import setup_logging

# Configure logging
logger = setup_logging(__name__)

# =============================================================================
# Configuration
# =============================================================================

WHISPER_MODEL = os.environ["WHISPER_MODEL"]  # Required: set in docker-compose.yaml
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
SAMPLE_RATE = 16000

# Flash Attention - only enabled if flash_attn package is installed
USE_FLASH_ATTENTION = True

# VAD (Voice Activity Detection) - skip silence for faster processing
USE_VAD = True
VAD_THRESHOLD = 0.5  # Speech probability threshold

# Chunk settings for streaming (optimized for whisper-large-v3-turbo)
CHUNK_DURATION_SEC = 1.5  # Process in 1.5s chunks
MIN_AUDIO_SEC = 0.3  # Minimum audio to process

# API version
API_VERSION = "1.0"

# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(title="Whisper ASR Service", version=API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model
whisper_pipe = None
vad_model = None


def load_vad_model():
    """Load Silero VAD model for speech detection."""
    global vad_model

    if vad_model is not None:
        return vad_model

    try:
        logger.info("Loading Silero VAD model...")
        vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        vad_model = vad_model.to(DEVICE)
        logger.info("Silero VAD model loaded")
        return vad_model
    except Exception as e:
        logger.warning(f"Failed to load VAD model: {e}. Continuing without VAD.")
        return None


def detect_speech(audio_array: np.ndarray) -> bool:
    """
    Detect if audio contains speech using Silero VAD.

    Silero VAD requires processing audio in 512-sample windows (32ms at 16kHz).
    We check multiple windows and return True if any window has speech.

    Args:
        audio_array: Audio samples (float32, normalized to [-1, 1])

    Returns:
        True if speech detected, False otherwise
    """
    global vad_model

    # Lazy load VAD model
    if vad_model is None:
        load_vad_model()

    if vad_model is None:
        return True  # If VAD failed to load, assume speech

    try:
        # Silero VAD expects 512 samples (32ms) at 16kHz
        window_size = 512
        num_windows = len(audio_array) // window_size

        if num_windows == 0:
            return True  # Audio too short, assume speech

        # Reset VAD state for fresh detection
        vad_model.reset_states()

        # Check windows and return True if any has speech
        max_speech_prob = 0.0
        for i in range(num_windows):
            start = i * window_size
            window = audio_array[start : start + window_size]

            # VAD expects float32 tensor
            audio_tensor = torch.from_numpy(window).float().to(DEVICE)
            speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()
            max_speech_prob = max(max_speech_prob, speech_prob)

            # Early exit if speech detected
            if speech_prob >= VAD_THRESHOLD:
                return True

        logger.debug(f"VAD: max_speech_prob={max_speech_prob:.2f}, threshold={VAD_THRESHOLD}")
        return False
    except Exception as e:
        logger.warning(f"VAD error: {e}")
        return True  # On error, assume speech


def load_model():
    """Load Whisper model with optimizations."""
    global whisper_pipe

    if whisper_pipe is not None:
        return whisper_pipe

    logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
    logger.info(f"Device: {DEVICE}, dtype: {TORCH_DTYPE}")

    if DEVICE == "cuda":
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    # Check if Flash Attention 2 is available
    use_flash_attn = False
    if DEVICE == "cuda":
        try:
            import flash_attn

            use_flash_attn = True
            logger.info("Flash Attention 2 available")
        except ImportError:
            logger.info("Flash Attention 2 not installed, using SDPA")

    # Load model with appropriate attention implementation
    model_kwargs = {
        "torch_dtype": TORCH_DTYPE,
        "low_cpu_mem_usage": True,
        "use_safetensors": True,
    }

    if DEVICE == "cuda":
        if use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            # Use PyTorch's native scaled dot product attention (SDPA)
            model_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(WHISPER_MODEL, **model_kwargs)
    model.to(DEVICE)

    processor = AutoProcessor.from_pretrained(WHISPER_MODEL)

    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=TORCH_DTYPE,
        device=DEVICE,
    )

    # Warmup
    if DEVICE == "cuda":
        logger.info("Warming up GPU...")
        dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 second
        _ = whisper_pipe(dummy, generate_kwargs={"max_new_tokens": 10})
        torch.cuda.synchronize()

        mem_gb = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"GPU Memory: {mem_gb:.2f}GB allocated")

    logger.info("Whisper model loaded successfully")
    return whisper_pipe


@app.on_event("startup")
async def startup():
    """Initialize service - models load lazily on first request."""
    logger.info("Service ready, models will load on first request")


@app.get("/health")
async def health_check():
    model_loaded = whisper_pipe is not None
    vad_loaded = vad_model is not None

    gpu_info = {}
    if DEVICE == "cuda" and torch.cuda.is_available():
        gpu_info = {
            "cuda_device": torch.cuda.get_device_name(0),
            "memory_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        }

    return {
        "status": "healthy",
        "model": WHISPER_MODEL,
        "model_loaded": model_loaded,
        "vad_enabled": USE_VAD,
        "vad_loaded": vad_loaded,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        **gpu_info,
    }


@app.get("/info")
async def info():
    """Return model information for backend config display."""
    return {
        "service": "whisper-asr",
        "version": API_VERSION,
        "model": WHISPER_MODEL,
        "device": DEVICE,
        "vad_enabled": USE_VAD,
        "model_loaded": whisper_pipe is not None,
        "chunk_duration_sec": CHUNK_DURATION_SEC,
    }


@app.post("/unload")
async def unload_model():
    """Unload model from GPU to free memory for other services."""
    global whisper_pipe

    if whisper_pipe is None:
        return {"status": "not_loaded", "message": "Model was not loaded"}

    try:
        # Store reference and set global to None before deletion
        pipe_ref = whisper_pipe
        whisper_pipe = None
        del pipe_ref

        # Force garbage collection and clear CUDA cache
        import gc

        gc.collect()

        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Get memory info
            free_mem = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory

            logger.info(
                f"Whisper model unloaded. GPU memory freed. Available: {free_mem/1e9:.2f}GB / {total_mem/1e9:.2f}GB"
            )

            return {
                "status": "unloaded",
                "message": "Whisper model unloaded from GPU",
                "gpu_memory_free_gb": round(free_mem / 1e9, 2),
                "gpu_memory_total_gb": round(total_mem / 1e9, 2),
            }
        else:
            logger.info("Whisper model unloaded from CPU")
            return {"status": "unloaded", "message": "Model unloaded from CPU"}

    except Exception as e:
        logger.error(f"Error unloading model: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...)):
    """
    Transcribe an uploaded audio file.

    Args:
        file: Audio file (WAV, MP3, etc.)

    Returns:
        JSON with text transcription
    """
    global whisper_pipe

    # Auto-reload model if it was unloaded
    if whisper_pipe is None:
        logger.info("Model not loaded, auto-reloading...")
        load_model()

    if whisper_pipe is None:
        return {"error": "Failed to load model", "text": ""}

    try:
        # Read audio file
        contents = await file.read()

        # Load audio using soundfile
        audio_array, sample_rate = sf.read(io.BytesIO(contents))

        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        # Resample to 16kHz if needed
        if sample_rate != SAMPLE_RATE:
            import scipy.signal

            num_samples = int(len(audio_array) * SAMPLE_RATE / sample_rate)
            audio_array = scipy.signal.resample(audio_array, num_samples)

        # Ensure float32
        audio_array = audio_array.astype(np.float32)

        # Normalize if needed
        if np.abs(audio_array).max() > 1.0:
            audio_array = audio_array / 32768.0

        logger.info(
            f"Transcribing file: {file.filename}, duration: {len(audio_array)/SAMPLE_RATE:.1f}s"
        )

        # Transcribe with timestamps for longer audio
        result = whisper_pipe(
            audio_array,
            generate_kwargs={
                "task": "transcribe",
                "language": "en",
            },
            return_timestamps=True,
        )

        text = result.get("text", "").strip()

        logger.info(f"Transcription complete: {len(text)} chars")

        return {
            "text": text,
            "duration": len(audio_array) / SAMPLE_RATE,
            "chunks": result.get("chunks", []),
        }

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return {"error": str(e), "text": ""}


def transcribe_audio(audio_array: np.ndarray) -> str:
    """Transcribe audio array to text."""
    global whisper_pipe

    # Auto-reload model if needed
    if whisper_pipe is None:
        logger.info("Model not loaded for transcribe_audio, auto-reloading...")
        load_model()

    if whisper_pipe is None:
        logger.error("Failed to load model for transcription")
        return ""

    if len(audio_array) < SAMPLE_RATE * MIN_AUDIO_SEC:
        return ""

    # Ensure float32
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)

    # Normalize if needed
    if audio_array.max() > 1.0:
        audio_array = audio_array / 32768.0

    try:
        result = whisper_pipe(
            audio_array,
            generate_kwargs={
                "task": "transcribe",
                "language": "en",
            },
            return_timestamps=False,
        )
        return result.get("text", "").strip()
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""


@app.websocket("/stream")
async def stream_transcribe(websocket: WebSocket):
    """
    Streaming transcription endpoint.

    Protocol:
    1. Connect to websocket
    2. Stream raw PCM audio bytes (int16, 16kHz, mono)
    3. Receive JSON: {"id": "s1", "text": "..."}
    """
    await websocket.accept()
    logger.info("Stream connection established")

    # Settings
    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION_SEC)
    min_samples = int(SAMPLE_RATE * MIN_AUDIO_SEC)

    # State
    audio_buffer = bytearray()
    segment_counter = 0
    last_process_time = time.time()
    consecutive_silence = 0

    try:
        while True:
            data = await websocket.receive()

            # Handle config messages
            if "text" in data:
                try:
                    msg = json.loads(data["text"])
                    logger.info(f"Config received: {msg}")
                    continue
                except json.JSONDecodeError:
                    continue

            # Handle audio data
            if "bytes" in data:
                audio_bytes = data["bytes"]

                # Empty bytes signals end of stream
                if len(audio_bytes) == 0:
                    logger.info("End of stream signal received")
                    # Process any remaining audio
                    if len(audio_buffer) >= min_samples * 2:
                        audio_array = (
                            np.frombuffer(bytes(audio_buffer), dtype=np.int16).astype(np.float32)
                            / 32768.0
                        )
                        text = await asyncio.to_thread(transcribe_audio, audio_array)
                        if text:
                            await websocket.send_json({"id": f"s{segment_counter}", "text": text})
                            segment_counter += 1
                    # Send final message
                    await websocket.send_json(
                        {"id": f"s{segment_counter}", "text": "", "final": True}
                    )
                    break

                audio_buffer.extend(audio_bytes)
                current_time = time.time()

                # Convert to samples count (int16 = 2 bytes per sample)
                buffer_samples = len(audio_buffer) // 2

                # Process when we have enough audio or time elapsed
                should_process = buffer_samples >= chunk_samples or (
                    buffer_samples >= min_samples
                    and current_time - last_process_time >= CHUNK_DURATION_SEC
                )

                if should_process and buffer_samples >= min_samples:
                    # Convert to numpy array
                    audio_array = (
                        np.frombuffer(bytes(audio_buffer), dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )

                    # VAD check - skip transcription if no speech detected
                    # Run in thread to avoid blocking event loop
                    has_speech = await asyncio.to_thread(detect_speech, audio_array)
                    if not has_speech:
                        consecutive_silence += 1
                        if consecutive_silence <= 2:  # Log first few silences
                            logger.debug(f"No speech detected, skipping ({consecutive_silence})")
                        audio_buffer.clear()
                        last_process_time = current_time
                        continue

                    consecutive_silence = 0
                    logger.info(
                        f"Processing {buffer_samples} samples ({buffer_samples/SAMPLE_RATE:.2f}s)"
                    )

                    # Transcribe in thread to avoid blocking event loop
                    text = await asyncio.to_thread(transcribe_audio, audio_array)
                    logger.info(f"Transcribed: '{text[:100] if text else '(empty)'}'")

                    if text:
                        # Send result (Whisper already outputs punctuated text)
                        logger.info(f"Sending segment s{segment_counter}: '{text[:50]}'")
                        await websocket.send_json({"id": f"s{segment_counter}", "text": text})
                        segment_counter += 1

                    # Clear buffer
                    audio_buffer.clear()
                    last_process_time = current_time

    except WebSocketDisconnect:
        logger.info("Stream disconnected")
    except Exception as e:
        logger.error(f"Stream error: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
