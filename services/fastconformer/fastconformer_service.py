"""
FastConformer ASR Service - GPU-accelerated streaming with cache-aware inference

Features:
- NVIDIA FastConformer Hybrid model (114M params, optimized for streaming)
- Cache-aware encoder for low-latency streaming
- Hybrid RNNT/CTC decoder (RNNT default for better accuracy)
- Multiple latency modes: 0ms, 80ms, 480ms, 1040ms
- Native punctuation and capitalization (no external refiner needed)
- WebSocket streaming only (no offline transcription)

Protocol:
1. Client connects to /stream
2. Client streams raw PCM audio (int16, 16kHz, mono)
3. Server sends: {"id": "s1", "text": "..."} for segments
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastconformer_model import (
    DECODER_TYPE,
    DEVICE,
    MODEL_NAME,
    ATT_CONTEXT_SIZE,
    get_model,
    get_model_state,
    setup_cuda,
    unload_model,
)

from shared.utils import setup_logging

# ==============================================================================
# Version
# ==============================================================================

__version__ = "1.0"

# ==============================================================================
# Configuration
# ==============================================================================

# Streaming parameters
CHUNK_DURATION_SEC = 0.5  # Process every 500ms
MIN_CHUNK_SEC = 0.2  # Minimum chunk size
SILENCE_THRESHOLD_SEC = 2.0
MAX_WORDS_PER_SEGMENT = 30

# Logging
logger = setup_logging(__name__)

# Initialize CUDA
setup_cuda()

# ==============================================================================
# Streaming State
# ==============================================================================


@dataclass
class StreamingState:
    """State container for cache-aware streaming."""

    cache_last_channel: torch.Tensor | None = None
    cache_last_time: torch.Tensor | None = None
    cache_last_channel_len: torch.Tensor | None = None
    previous_hypotheses: object | None = None
    accumulated_text: str = ""
    word_count: int = 0


# ==============================================================================
# Transcription Functions
# ==============================================================================


def pcm_to_float(audio_bytes: bytes) -> np.ndarray:
    """Convert PCM16 bytes to float32 audio array."""
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32768.0
    return audio_float


def stream_transcribe_chunk(
    audio_chunk: bytes, state: StreamingState, sample_rate: int = 16000
) -> tuple[str, StreamingState]:
    """Transcribe audio chunk using cache-aware streaming."""
    model_state = get_model_state()
    if not model_state.loaded:
        # Auto-load model
        try:
            get_model()
            model_state = get_model_state()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return state.accumulated_text, state

    model = model_state.model

    try:
        audio_array = pcm_to_float(audio_chunk)

        if len(audio_array) < 400:  # < 25ms
            return state.accumulated_text, state

        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).to(DEVICE)
        audio_len = torch.tensor([len(audio_array)], device=DEVICE)

        # Convert to FP16 if model is FP16
        if DEVICE == "cuda" and model.dtype == torch.float16:
            audio_tensor = audio_tensor.half()

        with torch.no_grad():
            # Get preprocessor from model
            processed_signal, processed_signal_len = model.preprocessor(
                input_signal=audio_tensor, length=audio_len
            )

            # Cache-aware streaming step
            result = model.encoder.streaming_forward(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_len,
                cache_last_channel=state.cache_last_channel,
                cache_last_time=state.cache_last_time,
                cache_last_channel_len=state.cache_last_channel_len,
            )

            encoded, encoded_len, cache_ch, cache_t, cache_ch_len = result

            # Decode with RNNT or CTC
            if model_state.decoder_type == "rnnt":
                # RNNT decoding
                hypotheses = model.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=encoded,
                    encoded_lengths=encoded_len,
                    return_hypotheses=True,
                )[0]
                text = hypotheses[0].text if hypotheses else ""
            else:
                # CTC decoding
                log_probs = model.decoder(encoder_output=encoded)
                hypotheses = model.decoding.ctc_decoder_predictions_tensor(
                    decoder_output=log_probs,
                    decoder_lengths=encoded_len,
                    return_hypotheses=True,
                )[0]
                text = hypotheses[0].text if hypotheses else ""

            # Update cache
            state.cache_last_channel = cache_ch
            state.cache_last_time = cache_t
            state.cache_last_channel_len = cache_ch_len

            # Accumulate text
            if text:
                state.accumulated_text = text.strip()
                state.word_count = len(state.accumulated_text.split())

        return state.accumulated_text, state

    except Exception as e:
        logger.error(f"Streaming transcription error: {e}", exc_info=True)
        return state.accumulated_text, state


# ==============================================================================
# FastAPI Application
# ==============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup: Auto-load model in background
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        await loop.run_in_executor(executor, get_model)
        logger.info("Model pre-loaded during startup")
    except Exception as e:
        logger.warning(f"Model pre-load failed (will load on first request): {e}")

    yield

    # Shutdown: Unload model
    logger.info("Shutting down, unloading model...")
    try:
        await loop.run_in_executor(executor, unload_model)
    except Exception as e:
        logger.error(f"Model unload failed: {e}")
    finally:
        executor.shutdown(wait=False)


app = FastAPI(
    title="FastConformer ASR Service",
    description="GPU-accelerated streaming ASR with cache-aware FastConformer",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    state = get_model_state()

    health_info = {
        "status": "healthy" if state.loaded else "starting",
        "model_loaded": state.loaded,
        "model_name": state.model_name if state.loaded else MODEL_NAME,
        "decoder_type": state.decoder_type if state.loaded else DECODER_TYPE,
        "att_context_size": state.att_context_size if state.loaded else ATT_CONTEXT_SIZE,
        "device": DEVICE,
    }

    # Add GPU memory info
    if DEVICE == "cuda" and state.loaded:
        mem_allocated = torch.cuda.memory_allocated() / 1024**3
        health_info["memory_allocated_gb"] = round(mem_allocated, 2)
        health_info["cuda_device"] = torch.cuda.get_device_name(0)

    return health_info


@app.get("/info")
async def model_info():
    """Service information endpoint."""
    state = get_model_state()
    return {
        "service": "fastconformer-asr",
        "version": __version__,
        "model_name": state.model_name if state.loaded else MODEL_NAME,
        "decoder_type": state.decoder_type if state.loaded else DECODER_TYPE,
        "att_context_size": state.att_context_size if state.loaded else ATT_CONTEXT_SIZE,
        "device": DEVICE,
        "streaming_only": True,
    }


@app.post("/unload")
async def unload_endpoint():
    """Unload model from GPU to free memory."""
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        await loop.run_in_executor(executor, unload_model)
        return {"status": "success", "message": "Model unloaded from GPU"}
    except Exception as e:
        logger.error(f"Unload failed: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        executor.shutdown(wait=False)


# ==============================================================================
# WebSocket Streaming
# ==============================================================================


@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """Real-time streaming transcription endpoint."""
    await websocket.accept()
    logger.info("WebSocket connected")

    # Ensure model is loaded before processing
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, lambda: get_model())
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        await websocket.close(code=1011, reason="Model loading failed")
        return

    state = StreamingState()
    audio_buffer = bytearray()
    executor = ThreadPoolExecutor(max_workers=1)
    send_queue: asyncio.Queue = asyncio.Queue()

    chunk_bytes = int(CHUNK_DURATION_SEC * 16000 * 2)
    min_chunk_bytes = int(MIN_CHUNK_SEC * 16000 * 2)

    output_counter = 0
    last_audio_time = time.time()
    last_sent_text = ""
    running = True
    processing = False

    # Sender coroutine
    async def sender():
        while running or not send_queue.empty():
            try:
                msg = await asyncio.wait_for(send_queue.get(), timeout=0.1)
                await websocket.send_json(msg)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.debug(f"Sender error: {e}")
                break

    # Ping coroutine
    async def ping_loop():
        while running:
            try:
                await asyncio.sleep(15)
                if running:
                    await websocket.send_json({"type": "ping"})
            except Exception:
                break

    sender_task = asyncio.create_task(sender())
    ping_task = asyncio.create_task(ping_loop())

    try:
        while True:
            data = await websocket.receive()

            if "bytes" not in data:
                continue

            audio_bytes = data["bytes"]
            audio_buffer.extend(audio_bytes)
            last_audio_time = time.time()

            # Process when buffer reaches chunk size
            if len(audio_buffer) >= chunk_bytes and not processing:
                processing = True
                chunk = bytes(audio_buffer[:chunk_bytes])
                audio_buffer = audio_buffer[chunk_bytes:]

                # Transcribe in thread pool
                text, state = await loop.run_in_executor(
                    executor, stream_transcribe_chunk, chunk, state
                )

                # Send if text changed
                if text and text != last_sent_text:
                    output_counter += 1
                    msg = {"id": f"s{output_counter}", "text": text}
                    await send_queue.put(msg)
                    logger.debug(f"Segment s{output_counter}: '{text}' ({state.word_count} words)")
                    last_sent_text = text

                processing = False

            # Handle silence (send accumulated text as final)
            silence_duration = time.time() - last_audio_time
            if (
                silence_duration >= SILENCE_THRESHOLD_SEC
                and state.accumulated_text
                and not processing
            ):
                logger.debug(f"Silence detected ({silence_duration:.1f}s), finalizing segment")

                # Send final segment
                output_counter += 1
                msg = {"id": f"s{output_counter}", "text": state.accumulated_text}
                await send_queue.put(msg)
                logger.info(f"Final segment s{output_counter}: '{state.accumulated_text}'")

                # Reset state
                state = StreamingState()
                last_sent_text = ""
                last_audio_time = time.time()

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        running = False
        sender_task.cancel()
        ping_task.cancel()
        executor.shutdown(wait=False)
        logger.info("WebSocket session ended")


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=False,
    )
