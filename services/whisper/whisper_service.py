"""
Whisper ASR Service - GPU-accelerated streaming transcription
Using OpenAI Whisper via HuggingFace Transformers

Features:
- Dual models: Turbo for streaming (fast), Large-v3 for offline (accurate)
- Fast GPU inference with Flash Attention 2 or SDPA
- Silero VAD for speech detection (skip silence)
- Streaming WebSocket API with segment-based protocol
- Automatic language detection
- Native punctuation and capitalization (no external refiner needed)
- Robust model download with retry logic

Protocol:
1. Client connects to /stream
2. Client streams raw PCM audio (int16, 16kHz, mono)
3. Server sends: {"id": "s1", "text": "..."} for segments
"""

import asyncio
import io
import json
import time
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from model import (
    DEVICE,
    OFFLINE_MODEL,
    SAMPLE_RATE,
    STREAMING_MODEL,
    TORCH_DTYPE,
    get_model_state,
    get_pipeline,
    load_model,
    setup_cuda,
    unload_model,
)

from shared.logging import setup_logging

# Configure logging
logger = setup_logging(__name__)

# =============================================================================
# Configuration
# =============================================================================

# VAD (Voice Activity Detection) - skip silence for faster processing
USE_VAD = True
VAD_THRESHOLD = 0.5  # Speech probability threshold

# Chunk settings for streaming (optimized for whisper-large-v3-turbo)
CHUNK_DURATION_SEC = 1.5  # Process in 1.5s chunks
MIN_AUDIO_SEC = 0.3  # Minimum audio to process

# API version
API_VERSION = "1.1"


# =============================================================================
# FastAPI App
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    setup_cuda()
    logger.info("Service ready, models will load on first request")
    logger.info(f"Streaming model: {STREAMING_MODEL}")
    logger.info(f"Offline model: {OFFLINE_MODEL}")
    yield
    # Shutdown (if needed)


app = FastAPI(title="Whisper ASR Service", version=API_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global VAD model
vad_model = None


def load_vad_model():
    """Load Silero VAD model for speech detection."""
    global vad_model

    if vad_model is not None:
        return vad_model

    if not USE_VAD:
        return None

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


@app.get("/health")
async def health_check():
    state = get_model_state()

    gpu_info = {}
    if DEVICE == "cuda" and torch.cuda.is_available():
        gpu_info = {
            "cuda_device": torch.cuda.get_device_name(0),
            "memory_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        }

    return {
        "status": "healthy",
        "streaming_model": STREAMING_MODEL,
        "streaming_loaded": state.streaming_loaded,
        "offline_model": OFFLINE_MODEL,
        "offline_loaded": state.offline_loaded,
        "vad_enabled": USE_VAD,
        "vad_loaded": vad_model is not None,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        **gpu_info,
    }


@app.get("/info")
async def info():
    """Return model information for backend config display."""
    state = get_model_state()
    return {
        "service": "whisper-asr",
        "version": API_VERSION,
        "streaming_model": STREAMING_MODEL,
        "offline_model": OFFLINE_MODEL,
        "device": DEVICE,
        "vad_enabled": USE_VAD,
        "streaming_loaded": state.streaming_loaded,
        "offline_loaded": state.offline_loaded,
        "chunk_duration_sec": CHUNK_DURATION_SEC,
    }


@app.post("/unload")
async def unload_models_endpoint(mode: str = "all"):
    """Unload model(s) from GPU to free memory.

    Args:
        mode: 'streaming', 'offline', or 'all' (default)
    """
    state = get_model_state()

    if mode not in ("streaming", "offline", "all"):
        return {
            "status": "error",
            "message": f"Invalid mode: {mode}. Use 'streaming', 'offline', or 'all'",
        }

    if not state.streaming_loaded and not state.offline_loaded:
        return {"status": "not_loaded", "message": "No models were loaded"}

    try:
        freed = unload_model(mode)

        if not freed:
            return {"status": "not_loaded", "message": f"Model '{mode}' was not loaded"}

        if DEVICE == "cuda":
            free_mem = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory

            return {
                "status": "unloaded",
                "message": f"Model(s) '{mode}' unloaded from GPU",
                "gpu_memory_free_gb": round(free_mem / 1e9, 2),
                "gpu_memory_total_gb": round(total_mem / 1e9, 2),
            }
        else:
            return {"status": "unloaded", "message": f"Model(s) '{mode}' unloaded from CPU"}

    except Exception as e:
        logger.error(f"Error unloading model: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...), language: str = "en"):
    """
    Transcribe an uploaded audio file using the OFFLINE model (large-v3).

    Uses the more accurate offline model for file transcription.

    Args:
        file: Audio file (WAV, MP3, etc.)
        language: Language code (e.g., 'en', 'yue' for Cantonese)

    Returns:
        JSON with text transcription
    """
    # Get offline pipeline (auto-loads if needed)
    whisper_pipe = get_pipeline(mode="offline")

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
            f"Transcribing file: {file.filename}, duration: {len(audio_array)/SAMPLE_RATE:.1f}s, language: {language}, model: {OFFLINE_MODEL}"
        )

        # Transcribe with timestamps for longer audio
        result = whisper_pipe(
            audio_array,
            generate_kwargs={
                "task": "transcribe",
                "language": language,
            },
            return_timestamps=True,
        )

        text = result.get("text", "").strip()

        logger.info(f"Transcription complete: {len(text)} chars")

        return {
            "text": text,
            "duration": len(audio_array) / SAMPLE_RATE,
            "chunks": result.get("chunks", []),
            "model": OFFLINE_MODEL,
        }

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return {"error": str(e), "text": ""}


def transcribe_audio_streaming(audio_array: np.ndarray, language: str = "en") -> str:
    """Transcribe audio array to text using STREAMING model (turbo).

    Used by the WebSocket streaming endpoint for low-latency transcription.

    Args:
        audio_array: Audio samples (float32, normalized to [-1, 1])
        language: Language code (e.g., 'en', 'yue' for Cantonese)

    Returns:
        Transcribed text
    """
    # Get streaming pipeline (auto-loads if needed)
    whisper_pipe = get_pipeline(mode="streaming")

    if whisper_pipe is None:
        logger.error("Failed to load streaming model for transcription")
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
                "language": language,
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
    Streaming transcription endpoint using STREAMING model (turbo).

    Uses the faster turbo model for real-time transcription.

    Protocol:
    1. Connect to websocket
    2. Send config: {"chunk_ms": 500, "language": "en"}
    3. Stream raw PCM audio bytes (int16, 16kHz, mono)
    4. Receive JSON: {"id": "s1", "text": "..."}
    """
    await websocket.accept()
    logger.info(f"Stream connection established (model: {STREAMING_MODEL})")

    # Settings
    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION_SEC)
    min_samples = int(SAMPLE_RATE * MIN_AUDIO_SEC)

    # State
    audio_buffer = bytearray()
    segment_counter = 0
    last_process_time = time.time()
    consecutive_silence = 0
    language = "en"  # Default language, can be overridden by config

    try:
        while True:
            data = await websocket.receive()

            # Handle config messages
            if "text" in data:
                try:
                    msg = json.loads(data["text"])
                    logger.info(f"Config received: {msg}")
                    # Update language from config if provided
                    if "language" in msg:
                        language = msg["language"]
                        logger.info(f"Language set to: {language}")
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
                        text = await asyncio.to_thread(
                            transcribe_audio_streaming, audio_array, language
                        )
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
                    text = await asyncio.to_thread(
                        transcribe_audio_streaming, audio_array, language
                    )
                    logger.info(f"Transcribed ({language}): '{text[:100] if text else '(empty)'}'")

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
