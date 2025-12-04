"""
NVIDIA NeMo Parakeet ASR Service v4.0

GPU-accelerated streaming ASR using NeMo's cache-aware conformer streaming.

Model: nvidia/parakeet-tdt-1.1b (TDT - optimized for streaming)

Key Features:
- NeMo's native conformer_stream_step() for proper streaming
- Cache-aware encoder maintains context across chunks
- No duplicate words - incremental output only
- Text refinement via text-refiner service

Endpoints:
- GET  /health     - Health check
- GET  /info       - Service information
- POST /transcribe - Transcribe audio file
- WS   /stream     - Real-time streaming transcription
"""

import asyncio
import io
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)

from shared.logging import setup_logging
from shared.text_refiner import (
    capitalize_text,
    check_text_refiner,
    get_client,
    refine_text,
)

# ==============================================================================
# Configuration
# ==============================================================================

# Model selection:
# - STREAMING_MODEL: Used for WebSocket /stream (TDT is optimized for streaming)
# - OFFLINE_MODEL: Used for /transcribe endpoint (RNNT has better accuracy)
STREAMING_MODEL = os.environ["PARAKEET_STREAMING_MODEL"]  # Required: set in docker-compose.yaml
OFFLINE_MODEL = os.environ["PARAKEET_OFFLINE_MODEL"]  # Required: set in docker-compose.yaml

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = True  # TDT/streaming can use FP16
# Note: RNNT requires FP32, TDT/CTC can use FP16
OFFLINE_USE_FP16 = False  # RNNT offline model requires FP32

# Streaming parameters
CHUNK_DURATION_SEC = 1.0  # Audio chunk size for processing
MIN_CHUNK_SEC = 0.3  # Minimum audio to process
SILENCE_THRESHOLD_SEC = 2.0  # Finalize segment after silence
MAX_WORDS_PER_SEGMENT = 30  # Force finalize long segments

# Offline transcription parameters
MAX_AUDIO_CHUNK_SEC = 18.0  # Max chunk for offline transcription (model limit is 20s)
OVERLAP_SEC = 1.0  # Overlap between chunks to avoid word cutting

# Text refinement
MIN_WORDS_FOR_PUNCTUATION = 6

# Logging
logger = setup_logging(__name__)

# ==============================================================================
# CUDA Optimization
# ==============================================================================


def setup_cuda():
    """Configure CUDA optimizations."""
    if DEVICE != "cuda":
        return

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()

    logger.info(f"CUDA optimizations enabled (FP16={USE_FP16})")


setup_cuda()

# ==============================================================================
# Global Model State
# ==============================================================================


@dataclass
class ModelState:
    """Container for ASR model and related components."""

    # Streaming model (TDT optimized)
    streaming_model: object | None = None
    streaming_preprocessor: object | None = None
    streaming_loaded: bool = False
    streaming_model_name: str = ""
    # Offline model (RNNT for accuracy)
    offline_model: object | None = None
    offline_preprocessor: object | None = None
    offline_loaded: bool = False
    offline_model_name: str = ""
    # Current active model for backwards compatibility
    current_mode: str = "streaming"  # "streaming" or "offline"


_model_state = ModelState()


def get_model(mode: str = "streaming"):
    """Get the loaded ASR model for the specified mode. Auto-loads if not loaded."""
    if mode == "offline":
        if not _model_state.offline_loaded:
            logger.info("Offline model not loaded, auto-loading...")
            load_model(mode="offline")
        return _model_state.offline_model
    else:
        if not _model_state.streaming_loaded:
            logger.info("Streaming model not loaded, auto-loading...")
            load_model(mode="streaming")
        return _model_state.streaming_model


def get_preprocessor(mode: str = "streaming"):
    """Get the audio preprocessor for the specified mode. Auto-loads if not loaded."""
    if mode == "offline":
        if not _model_state.offline_loaded:
            load_model(mode="offline")
        return _model_state.offline_preprocessor
    else:
        if not _model_state.streaming_loaded:
            load_model(mode="streaming")
        return _model_state.streaming_preprocessor


# ==============================================================================
# Model Loading
# ==============================================================================


def load_model(mode: str = "streaming"):
    """Load NeMo Parakeet model for the specified mode.

    Args:
        mode: "streaming" for TDT model or "offline" for RNNT model
    """
    if mode == "offline":
        if _model_state.offline_loaded:
            return _model_state.offline_model
        model_name = OFFLINE_MODEL
        use_fp16 = OFFLINE_USE_FP16
    else:
        if _model_state.streaming_loaded:
            return _model_state.streaming_model
        model_name = STREAMING_MODEL
        use_fp16 = USE_FP16

    logger.info(f"Loading {mode} model: {model_name}")
    logger.info(f"Device: {DEVICE}, FP16: {use_fp16}")

    if DEVICE == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        import nemo.collections.asr as nemo_asr

        # Load model
        model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name)
        model = model.to(DEVICE)
        model.eval()

        # Apply FP16 if enabled
        if DEVICE == "cuda" and use_fp16:
            try:
                model = model.half()
                logger.info("Model converted to FP16")
            except Exception as e:
                logger.warning(f"FP16 conversion failed: {e}")

        # Store references based on mode
        if mode == "offline":
            _model_state.offline_model = model
            _model_state.offline_preprocessor = model.preprocessor
            _model_state.offline_loaded = True
            _model_state.offline_model_name = model_name
        else:
            _model_state.streaming_model = model
            _model_state.streaming_preprocessor = model.preprocessor
            _model_state.streaming_loaded = True
            _model_state.streaming_model_name = model_name

        # Log streaming config
        if hasattr(model.encoder, "streaming_cfg"):
            logger.info(f"Encoder streaming config: {model.encoder.streaming_cfg}")

        # GPU warmup
        _warmup_model(model, mode)

        if DEVICE == "cuda":
            mem_gb = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory allocated: {mem_gb:.2f} GB")

        logger.info(f"{mode.capitalize()} model loaded successfully")
        return model

    except Exception as e:
        logger.error(f"Failed to load {mode} model: {e}")
        raise


def _warmup_model(model, mode: str = "streaming"):
    """Warm up model with a test inference."""
    if DEVICE != "cuda":
        return

    use_fp16 = OFFLINE_USE_FP16 if mode == "offline" else USE_FP16

    logger.info(f"Warming up {mode} model...")
    try:
        dummy_audio = torch.randn(1, 16000).to(DEVICE)
        if use_fp16:
            dummy_audio = dummy_audio.half()

        preprocessor = (
            _model_state.offline_preprocessor
            if mode == "offline"
            else _model_state.streaming_preprocessor
        )

        with torch.no_grad():
            processed, processed_len = preprocessor(
                input_signal=dummy_audio, length=torch.tensor([16000]).to(DEVICE)
            )
            _ = model.conformer_stream_step(
                processed_signal=processed,
                processed_signal_length=processed_len,
                cache_last_channel=None,
                cache_last_time=None,
                cache_last_channel_len=None,
                keep_all_outputs=True,
                previous_hypotheses=None,
                return_transcription=True,
            )

        torch.cuda.synchronize()
        logger.info(f"{mode.capitalize()} model warmup complete")
    except Exception as e:
        logger.warning(f"GPU warmup skipped: {e}")


# ==============================================================================
# Audio Utilities
# ==============================================================================


def pcm_to_float(pcm_bytes: bytes) -> np.ndarray:
    """Convert PCM int16 bytes to float32 array normalized to [-1, 1]."""
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def float_to_pcm(audio: np.ndarray) -> bytes:
    """Convert float32 array to PCM int16 bytes."""
    return (audio * 32768).astype(np.int16).tobytes()


def load_audio_file(audio_data: bytes, target_sr: int = 16000) -> np.ndarray:
    """Load and preprocess audio file to target sample rate."""
    try:
        audio_io = io.BytesIO(audio_data)
        audio_array, sr = sf.read(audio_io)

        # Resample if needed
        if sr != target_sr:
            import librosa

            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)

        # Convert stereo to mono
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        return audio_array
    except Exception:
        # Assume raw PCM if file parsing fails
        return pcm_to_float(audio_data)


# ==============================================================================
# Transcription Functions
# ==============================================================================


def transcribe_with_timestamps(audio_data: bytes, sample_rate: int = 16000) -> list[dict]:
    """
    Transcribe audio and return word-level timestamps.

    Uses the OFFLINE model for better accuracy on file transcription.
    For long audio (>18s), splits into overlapping chunks to work around
    NeMo's max_duration=20s limit.

    Args:
        audio_data: Raw PCM audio bytes (int16)
        sample_rate: Audio sample rate

    Returns:
        List of dicts: [{'word': str, 'start': float, 'end': float}, ...]
    """
    model = get_model(mode="offline")
    audio_array = pcm_to_float(audio_data)

    # Skip very short audio
    if len(audio_array) < 1600:  # < 0.1 second
        return []

    duration_sec = len(audio_array) / sample_rate
    logger.info(
        f"Transcribing audio: {duration_sec:.1f}s (offline model: {_model_state.offline_model_name})"
    )

    # For short audio, transcribe directly
    if duration_sec <= MAX_AUDIO_CHUNK_SEC:
        return _transcribe_chunk(model, audio_array, sample_rate, time_offset=0.0)

    # For long audio, split into overlapping chunks
    logger.info(f"Long audio detected ({duration_sec:.1f}s), splitting into chunks...")

    chunk_samples = int(MAX_AUDIO_CHUNK_SEC * sample_rate)
    overlap_samples = int(OVERLAP_SEC * sample_rate)
    step_samples = chunk_samples - overlap_samples

    all_words = []
    offset = 0
    chunk_idx = 0

    while offset < len(audio_array):
        # Extract chunk
        end = min(offset + chunk_samples, len(audio_array))
        chunk = audio_array[offset:end]

        chunk_duration = len(chunk) / sample_rate
        time_offset = offset / sample_rate

        logger.info(f"Chunk {chunk_idx}: {time_offset:.1f}s - {time_offset + chunk_duration:.1f}s")

        # Transcribe chunk
        chunk_words = _transcribe_chunk(model, chunk, sample_rate, time_offset=time_offset)

        if chunk_words:
            # For overlapping chunks, skip words that overlap with previous chunk
            if all_words and chunk_idx > 0:
                # Find words that start after the overlap region
                overlap_end_time = time_offset + OVERLAP_SEC
                chunk_words = [w for w in chunk_words if w["start"] >= overlap_end_time - 0.1]

            all_words.extend(chunk_words)
            logger.info(f"Chunk {chunk_idx}: {len(chunk_words)} words")

        offset += step_samples
        chunk_idx += 1

    logger.info(f"Total: {len(all_words)} words from {chunk_idx} chunks")
    return all_words


def _transcribe_chunk(
    model, audio_array: np.ndarray, sample_rate: int, time_offset: float = 0.0
) -> list[dict]:
    """
    Transcribe a single audio chunk (must be <= MAX_AUDIO_CHUNK_SEC).

    Args:
        model: NeMo ASR model
        audio_array: Float32 audio array
        sample_rate: Audio sample rate
        time_offset: Time offset to add to timestamps

    Returns:
        List of word dicts with adjusted timestamps
    """
    # Save to temp file (NeMo requires file path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        sf.write(tmp_path, audio_array, sample_rate)

    try:
        with torch.no_grad():
            if DEVICE == "cuda":
                torch.cuda.synchronize()
                with torch.amp.autocast("cuda", enabled=USE_FP16):
                    results = model.transcribe(
                        [tmp_path],
                        timestamps=True,
                        return_hypotheses=True,
                        verbose=False,
                    )
                torch.cuda.synchronize()
            else:
                results = model.transcribe(
                    [tmp_path], timestamps=True, return_hypotheses=True, verbose=False
                )

        words = _extract_word_timestamps(results, audio_array, sample_rate)

        # Adjust timestamps by offset
        if time_offset > 0:
            for w in words:
                w["start"] += time_offset
                w["end"] += time_offset

        return words

    except Exception as e:
        logger.error(f"Chunk transcription error: {e}")
        return []
    finally:
        os.unlink(tmp_path)


def _extract_word_timestamps(results, audio_array: np.ndarray, sample_rate: int) -> list[dict]:
    """Extract word timestamps from NeMo transcription results."""
    if not results or len(results) == 0:
        return []

    hypothesis = results[0]

    # Try to get word timestamps
    if hasattr(hypothesis, "timestamp") and hypothesis.timestamp:
        word_timestamps = hypothesis.timestamp.get("word", [])
        words = []
        for wt in word_timestamps:
            if isinstance(wt, dict):
                words.append(
                    {
                        "word": wt.get("word", wt.get("char", "")),
                        "start": wt.get("start", 0),
                        "end": wt.get("end", 0),
                    }
                )
            elif hasattr(wt, "word"):
                words.append(
                    {
                        "word": wt.word,
                        "start": getattr(wt, "start", 0),
                        "end": getattr(wt, "end", 0),
                    }
                )
        return words

    # Fallback: estimate timestamps from text
    if hasattr(hypothesis, "text") and hypothesis.text:
        duration = len(audio_array) / sample_rate
        words = hypothesis.text.strip().split()
        if words:
            word_dur = duration / len(words)
            return [
                {"word": w, "start": i * word_dur, "end": (i + 1) * word_dur}
                for i, w in enumerate(words)
            ]

    return []


# ==============================================================================
# Streaming State
# ==============================================================================


@dataclass
class StreamingState:
    """State for cache-aware streaming session."""

    # Encoder caches
    cache_last_channel: torch.Tensor | None = None
    cache_last_time: torch.Tensor | None = None
    cache_last_channel_len: torch.Tensor | None = None
    # Decoder state
    previous_hypotheses: object | None = None
    # Accumulated transcription
    accumulated_text: str = ""
    # Segment tracking
    segment_counter: int = 0

    def reset(self):
        """Reset state for new segment."""
        self.cache_last_channel = None
        self.cache_last_time = None
        self.cache_last_channel_len = None
        self.previous_hypotheses = None
        self.accumulated_text = ""


def stream_transcribe_chunk(
    audio_chunk: bytes, state: StreamingState, sample_rate: int = 16000
) -> tuple[str, StreamingState]:
    """
    Transcribe audio chunk using NeMo's cache-aware streaming.

    Uses conformer_stream_step() which maintains encoder caches across chunks,
    producing incremental output without duplicates.

    Args:
        audio_chunk: PCM audio bytes
        state: Current streaming state
        sample_rate: Audio sample rate

    Returns:
        Tuple of (accumulated_text, updated_state)
    """
    if not _model_state.streaming_loaded:
        return state.accumulated_text, state

    model = _model_state.streaming_model
    preprocessor = _model_state.streaming_preprocessor

    try:
        # Convert to float tensor
        audio_array = pcm_to_float(audio_chunk)

        # Skip very short chunks
        if len(audio_array) < 400:  # < 25ms
            return state.accumulated_text, state

        # Prepare tensor
        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).to(DEVICE)
        audio_len = torch.tensor([len(audio_array)], device=DEVICE)

        if USE_FP16 and DEVICE == "cuda":
            audio_tensor = audio_tensor.half()

        with torch.no_grad():
            # Preprocess: audio -> mel spectrogram
            processed_signal, processed_signal_len = preprocessor(
                input_signal=audio_tensor, length=audio_len
            )

            # Cache-aware streaming step
            result = model.conformer_stream_step(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_len,
                cache_last_channel=state.cache_last_channel,
                cache_last_time=state.cache_last_time,
                cache_last_channel_len=state.cache_last_channel_len,
                keep_all_outputs=True,
                previous_hypotheses=state.previous_hypotheses,
                return_transcription=True,
            )

            # Unpack: (greedy_preds, transcriptions, cache_ch, cache_t, cache_ch_len, best_hyp)
            _, transcriptions, cache_ch, cache_t, cache_ch_len, best_hyp = result

            # Update state with new caches
            state.cache_last_channel = cache_ch
            state.cache_last_time = cache_t
            state.cache_last_channel_len = cache_ch_len
            state.previous_hypotheses = best_hyp

            # Extract text
            if transcriptions and len(transcriptions) > 0:
                hyp = transcriptions[0]
                text = (
                    hyp.text if hasattr(hyp, "text") else str(hyp) if isinstance(hyp, str) else ""
                )
                if text:
                    state.accumulated_text = text

            return state.accumulated_text, state

    except Exception as e:
        logger.error(f"Stream transcribe error: {e}")
        return state.accumulated_text, state


# ==============================================================================
# FastAPI Application
# ==============================================================================

app = FastAPI(
    title="Parakeet ASR Service",
    description="GPU-accelerated streaming ASR with NeMo cache-aware streaming",
    version="4.0.0",
)

text_refiner = get_client()


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    try:
        # Load streaming model on startup (used for WebSocket /stream)
        load_model(mode="streaming")
        await check_text_refiner()
        logger.info(
            f"Service startup complete. Streaming model: {STREAMING_MODEL}, Offline model: {OFFLINE_MODEL}"
        )
        if STREAMING_MODEL != OFFLINE_MODEL:
            logger.info("Note: Offline model will be loaded on first /transcribe request")
    except Exception as e:
        logger.error(f"Startup failed: {e}")


# ==============================================================================
# HTTP Endpoints
# ==============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    any_loaded = _model_state.streaming_loaded or _model_state.offline_loaded

    info = {
        "status": "healthy" if any_loaded else "loading",
        "streaming_model": STREAMING_MODEL,
        "streaming_loaded": _model_state.streaming_loaded,
        "offline_model": OFFLINE_MODEL,
        "offline_loaded": _model_state.offline_loaded,
        "device": DEVICE,
        "text_refiner_available": text_refiner.available,
    }

    if DEVICE == "cuda":
        info.update(
            {
                "cuda_device": torch.cuda.get_device_name(0),
                "memory_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
                "streaming_fp16": USE_FP16,
                "offline_fp16": OFFLINE_USE_FP16,
            }
        )

    return info


@app.get("/info")
async def model_info():
    """Service information endpoint."""
    return {
        "streaming_model": STREAMING_MODEL,
        "offline_model": OFFLINE_MODEL,
        "streaming_loaded": _model_state.streaming_loaded,
        "offline_loaded": _model_state.offline_loaded,
        "device": DEVICE,
        "torch_version": torch.__version__,
        "streaming": {
            "api_version": "v4.0",
            "approach": "cache_aware_conformer_streaming",
            "chunk_duration_sec": CHUNK_DURATION_SEC,
            "description": "NeMo conformer_stream_step with encoder caches",
            "model": STREAMING_MODEL,
            "fp16": USE_FP16,
        },
        "offline": {
            "model": OFFLINE_MODEL,
            "fp16": OFFLINE_USE_FP16,
            "max_chunk_sec": MAX_AUDIO_CHUNK_SEC,
            "overlap_sec": OVERLAP_SEC,
        },
        "text_refiner": {
            "enabled": text_refiner.enabled,
            "available": text_refiner.available,
            "url": text_refiner.url,
        },
    }


@app.post("/unload")
async def unload_model():
    """Unload all models from GPU to free memory for other services."""
    global _model_state

    models_unloaded = []

    try:
        # Unload streaming model
        if _model_state.streaming_loaded:
            if _model_state.streaming_model is not None:
                del _model_state.streaming_model
            if _model_state.streaming_preprocessor is not None:
                del _model_state.streaming_preprocessor
            _model_state.streaming_model = None
            _model_state.streaming_preprocessor = None
            _model_state.streaming_loaded = False
            models_unloaded.append(f"streaming ({_model_state.streaming_model_name})")
            _model_state.streaming_model_name = ""

        # Unload offline model
        if _model_state.offline_loaded:
            if _model_state.offline_model is not None:
                del _model_state.offline_model
            if _model_state.offline_preprocessor is not None:
                del _model_state.offline_preprocessor
            _model_state.offline_model = None
            _model_state.offline_preprocessor = None
            _model_state.offline_loaded = False
            models_unloaded.append(f"offline ({_model_state.offline_model_name})")
            _model_state.offline_model_name = ""

        if not models_unloaded:
            return {"status": "not_loaded", "message": "No models were loaded"}

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
                f"Models unloaded: {models_unloaded}. GPU memory freed. Available: {free_mem/1e9:.2f}GB / {total_mem/1e9:.2f}GB"
            )

            return {
                "status": "unloaded",
                "message": f"Models unloaded: {', '.join(models_unloaded)}",
                "gpu_memory_free_gb": round(free_mem / 1e9, 2),
                "gpu_memory_total_gb": round(total_mem / 1e9, 2),
            }
        else:
            logger.info(f"Models unloaded from CPU: {models_unloaded}")
            return {
                "status": "unloaded",
                "message": f"Models unloaded: {', '.join(models_unloaded)}",
            }

    except Exception as e:
        logger.error(f"Error unloading models: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file."""
    try:
        audio_data = await file.read()

        # Load and preprocess audio
        audio_array = load_audio_file(audio_data)
        pcm_data = float_to_pcm(audio_array)

        # Transcribe
        words = transcribe_with_timestamps(pcm_data)
        raw_text = " ".join(w["word"] for w in words)

        # Apply text refinement
        if text_refiner.available and len(words) >= MIN_WORDS_FOR_PUNCTUATION:
            text = await refine_text(raw_text)
        else:
            text = capitalize_text(raw_text)

        return {"text": text, "words": words, "model": _model_state.offline_model_name}

    except Exception as e:
        logger.error(f"Transcribe error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ==============================================================================
# WebSocket Streaming
# ==============================================================================


@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Real-time streaming transcription endpoint.

    Uses NeMo's cache-aware conformer_stream_step() for proper incremental
    transcription without duplicate words.

    Protocol:
        1. Connect to WebSocket
        2. (Optional) Send config: {"chunk_ms": 500}
        3. Stream audio: raw PCM bytes (16kHz, mono, int16)
        4. Receive transcriptions: {"id": "s0", "text": "Hello world."}
    """
    await websocket.accept()
    logger.info("WebSocket connected (v4.0 - cache-aware streaming)")

    # Initialize state
    state = StreamingState()
    audio_buffer = bytearray()
    executor = ThreadPoolExecutor(max_workers=1)
    send_queue: asyncio.Queue = asyncio.Queue()

    # Timing parameters
    chunk_bytes = int(CHUNK_DURATION_SEC * 16000 * 2)  # 1 second of audio
    min_chunk_bytes = int(MIN_CHUNK_SEC * 16000 * 2)  # minimum chunk

    # Session state
    output_counter = 0
    last_audio_time = time.time()
    last_sent_text = ""
    running = True
    processing = False

    async def emit_transcription(text: str, is_final: bool = False):
        """Emit transcription to client."""
        nonlocal output_counter, last_sent_text

        if not text or text == last_sent_text:
            return

        word_count = len(text.split())

        if is_final:
            # Apply full refinement for final results
            if text_refiner.available and word_count >= MIN_WORDS_FOR_PUNCTUATION:
                text = await refine_text(text)
            else:
                text = capitalize_text(text)
                if word_count >= MIN_WORDS_FOR_PUNCTUATION:
                    text = text.rstrip(".") + "."

            await send_queue.put({"id": f"s{output_counter}", "text": text})
            output_counter += 1
            last_sent_text = ""
        else:
            # Interim: capitalize only
            text = capitalize_text(text)
            await send_queue.put({"id": f"s{output_counter}", "text": text})
            last_sent_text = text

    async def receive_audio():
        """Receive audio from WebSocket."""
        nonlocal running, audio_buffer, last_audio_time

        while running:
            try:
                data = await websocket.receive()

                if data.get("bytes"):
                    audio_buffer.extend(data["bytes"])
                    last_audio_time = time.time()
                # Ignore text/config messages

            except WebSocketDisconnect:
                running = False
                break
            except Exception as e:
                logger.error(f"Receive error: {e}")
                running = False
                break

    async def process_audio():
        """Process accumulated audio chunks."""
        nonlocal running, processing, audio_buffer, state

        loop = asyncio.get_event_loop()

        while running:
            await asyncio.sleep(0.1)  # 100ms check interval

            current_time = time.time()
            time_since_audio = current_time - last_audio_time

            # Check for silence timeout
            if state.accumulated_text and time_since_audio > SILENCE_THRESHOLD_SEC:
                await emit_transcription(state.accumulated_text, is_final=True)
                state.reset()
                audio_buffer = bytearray()
                continue

            # Skip if processing or not enough audio
            if processing or len(audio_buffer) < min_chunk_bytes:
                continue

            # Process when we have enough audio or after short delay
            if len(audio_buffer) >= chunk_bytes or time_since_audio > 0.5:
                processing = True

                try:
                    # Extract chunk from buffer
                    chunk_size = min(len(audio_buffer), chunk_bytes)
                    chunk = bytes(audio_buffer[:chunk_size])
                    audio_buffer = audio_buffer[chunk_size:]

                    # Transcribe
                    text, state = await loop.run_in_executor(
                        executor, stream_transcribe_chunk, chunk, state, 16000
                    )

                    if text:
                        # Check segment length limit
                        if len(text.split()) >= MAX_WORDS_PER_SEGMENT:
                            await emit_transcription(text, is_final=True)
                            state.reset()
                        else:
                            await emit_transcription(text, is_final=False)

                except Exception as e:
                    logger.error(f"Processing error: {e}")
                finally:
                    processing = False

    async def send_results():
        """Send results to WebSocket client."""
        nonlocal running

        while running:
            try:
                msg = await asyncio.wait_for(send_queue.get(), timeout=0.5)
                await websocket.send_json(msg)
            except TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Send error: {e}")
                break

    # Run concurrent tasks
    try:
        recv_task = asyncio.create_task(receive_audio())
        proc_task = asyncio.create_task(process_audio())
        send_task = asyncio.create_task(send_results())

        await recv_task  # Wait for connection to close

        # Process remaining audio
        if len(audio_buffer) >= min_chunk_bytes:
            loop = asyncio.get_event_loop()
            text, _ = await loop.run_in_executor(
                executor, stream_transcribe_chunk, bytes(audio_buffer), state, 16000
            )
            if text:
                await emit_transcription(text, is_final=True)
        elif state.accumulated_text:
            await emit_transcription(state.accumulated_text, is_final=True)

        # Cleanup
        running = False
        proc_task.cancel()
        send_task.cancel()

        # Flush remaining messages
        while not send_queue.empty():
            try:
                await websocket.send_json(send_queue.get_nowait())
            except:
                break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        executor.shutdown(wait=False)
        logger.info("WebSocket disconnected")


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
