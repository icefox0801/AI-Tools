"""
Parakeet ASR Service - GPU-accelerated streaming with NeMo cache-aware conformer

Features:
- Dual models: TDT for streaming, RNNT for offline (better accuracy)
- Cache-aware encoder for incremental output without duplicates
- Long audio chunking with overlap for >20s files
- Text refinement via text-refiner service

Protocol:
1. Client connects to /stream
2. Client streams raw PCM audio (int16, 16kHz, mono)
3. Server sends: {"id": "s1", "text": "..."} for segments
"""

import asyncio
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch
import uvicorn
from audio import float_to_pcm, load_audio_file, pcm_to_float
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from model import (
    DEVICE,
    OFFLINE_MODEL,
    STREAMING_MODEL,
    get_model,
    get_model_state,
    load_model,
    setup_cuda,
    unload_models,
)

from shared.text_refiner import capitalize_text, check_text_refiner, get_client, refine_text
from shared.utils import setup_logging

# ==============================================================================
# Configuration
# ==============================================================================

# Streaming parameters
CHUNK_DURATION_SEC = 1.0
MIN_CHUNK_SEC = 0.3
SILENCE_THRESHOLD_SEC = 2.0
MAX_WORDS_PER_SEGMENT = 30

# Offline transcription parameters
MAX_AUDIO_CHUNK_SEC = 18.0
OVERLAP_SEC = 1.0

# Text refinement
MIN_WORDS_FOR_PUNCTUATION = 6

# Logging
logger = setup_logging(__name__)

# Initialize CUDA
setup_cuda()

# ==============================================================================
# Transcription Functions
# ==============================================================================


def transcribe_with_timestamps(audio_data: bytes, sample_rate: int = 16000) -> list[dict]:
    """Transcribe audio and return word-level timestamps.

    Uses the OFFLINE model for better accuracy on file transcription.
    For long audio (>18s), splits into overlapping chunks.
    """
    model = get_model(mode="offline")
    state = get_model_state()
    audio_array = pcm_to_float(audio_data)

    if len(audio_array) < 1600:  # < 0.1 second
        return []

    duration_sec = len(audio_array) / sample_rate
    logger.info(
        f"Transcribing audio: {duration_sec:.1f}s (offline model: {state.offline_model_name})"
    )

    if duration_sec <= MAX_AUDIO_CHUNK_SEC:
        return _transcribe_chunk(model, audio_array, sample_rate, time_offset=0.0)

    # Long audio: split into overlapping chunks
    logger.info(f"Long audio detected ({duration_sec:.1f}s), splitting into chunks...")
    chunk_samples = int(MAX_AUDIO_CHUNK_SEC * sample_rate)
    overlap_samples = int(OVERLAP_SEC * sample_rate)
    step_samples = chunk_samples - overlap_samples

    all_words = []
    offset = 0
    chunk_idx = 0

    while offset < len(audio_array):
        end = min(offset + chunk_samples, len(audio_array))
        chunk = audio_array[offset:end]
        time_offset = offset / sample_rate

        chunk_words = _transcribe_chunk(model, chunk, sample_rate, time_offset=time_offset)

        if chunk_words:
            if all_words and chunk_idx > 0:
                overlap_end_time = time_offset + OVERLAP_SEC
                chunk_words = [w for w in chunk_words if w["start"] >= overlap_end_time - 0.1]
            all_words.extend(chunk_words)

        offset += step_samples
        chunk_idx += 1

    logger.info(f"Total: {len(all_words)} words from {chunk_idx} chunks")
    return all_words


def _transcribe_chunk(
    model, audio_array: np.ndarray, sample_rate: int, time_offset: float = 0.0
) -> list[dict]:
    """Transcribe a single audio chunk (must be <= MAX_AUDIO_CHUNK_SEC)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        sf.write(tmp_path, audio_array, sample_rate)

    try:
        with torch.no_grad():
            if DEVICE == "cuda":
                torch.cuda.synchronize()
                with torch.amp.autocast("cuda", enabled=False):  # RNNT uses FP32
                    results = model.transcribe(
                        [tmp_path], timestamps=True, return_hypotheses=True, verbose=False
                    )
                torch.cuda.synchronize()
            else:
                results = model.transcribe(
                    [tmp_path], timestamps=True, return_hypotheses=True, verbose=False
                )

        words = _extract_word_timestamps(results, audio_array, sample_rate)

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
    if not results:
        return []

    hypothesis = results[0]

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

    cache_last_channel: torch.Tensor | None = None
    cache_last_time: torch.Tensor | None = None
    cache_last_channel_len: torch.Tensor | None = None
    previous_hypotheses: object | None = None
    accumulated_text: str = ""

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
    """Transcribe audio chunk using NeMo's cache-aware streaming."""
    model_state = get_model_state()
    if not model_state.streaming_loaded:
        return state.accumulated_text, state

    model = model_state.streaming_model
    preprocessor = model_state.streaming_preprocessor

    try:
        audio_array = pcm_to_float(audio_chunk)

        if len(audio_array) < 400:  # < 25ms
            return state.accumulated_text, state

        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).to(DEVICE)
        audio_len = torch.tensor([len(audio_array)], device=DEVICE)

        if DEVICE == "cuda":
            audio_tensor = audio_tensor.half()  # Streaming uses FP16

        with torch.no_grad():
            processed_signal, processed_signal_len = preprocessor(
                input_signal=audio_tensor, length=audio_len
            )

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

            _, transcriptions, cache_ch, cache_t, cache_ch_len, best_hyp = result

            state.cache_last_channel = cache_ch
            state.cache_last_time = cache_t
            state.cache_last_channel_len = cache_ch_len
            state.previous_hypotheses = best_hyp

            if transcriptions:
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    try:
        setup_cuda()
        await check_text_refiner()
        logger.info("Service ready, models will load on first request")
        logger.info(f"Streaming: {STREAMING_MODEL}, Offline: {OFFLINE_MODEL}")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
    yield
    # Shutdown (if needed)


app = FastAPI(
    title="Parakeet ASR Service",
    description="GPU-accelerated streaming ASR with NeMo cache-aware streaming",
    version="1.0",
    lifespan=lifespan,
)

text_refiner = get_client()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    state = get_model_state()
    any_loaded = state.streaming_loaded or state.offline_loaded

    info = {
        "status": "healthy" if any_loaded else "loading",
        "streaming_model": STREAMING_MODEL,
        "streaming_loaded": state.streaming_loaded,
        "offline_model": OFFLINE_MODEL,
        "offline_loaded": state.offline_loaded,
        "device": DEVICE,
        "text_refiner_available": text_refiner.available,
    }

    if DEVICE == "cuda":
        info.update(
            {
                "cuda_device": torch.cuda.get_device_name(0),
                "memory_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            }
        )

    return info


@app.get("/info")
async def model_info():
    """Service information endpoint."""
    state = get_model_state()
    return {
        "streaming_model": STREAMING_MODEL,
        "offline_model": OFFLINE_MODEL,
        "streaming_loaded": state.streaming_loaded,
        "offline_loaded": state.offline_loaded,
        "device": DEVICE,
        "torch_version": torch.__version__,
        "text_refiner": {
            "enabled": text_refiner.enabled,
            "available": text_refiner.available,
        },
    }


@app.post("/unload")
async def unload_endpoint():
    """Unload all models from GPU to free memory."""
    try:
        return unload_models()
    except Exception as e:
        logger.error(f"Error unloading models: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file.

    Parakeet TDT model natively produces punctuated text, so no text-refiner
    post-processing is needed for file transcription.
    """
    try:
        audio_data = await file.read()
        audio_array = load_audio_file(audio_data)
        pcm_data = float_to_pcm(audio_array)

        words = transcribe_with_timestamps(pcm_data)
        # Parakeet TDT outputs punctuated words natively
        text = " ".join(w["word"] for w in words)

        state = get_model_state()
        return {"text": text, "words": words, "model": state.offline_model_name}

    except Exception as e:
        logger.error(f"Transcribe error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ==============================================================================
# WebSocket Streaming
# ==============================================================================


@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """Real-time streaming transcription endpoint."""
    await websocket.accept()
    logger.info("WebSocket connected")

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

    async def emit_transcription(text: str, is_final: bool = False):
        nonlocal output_counter, last_sent_text

        if not text or text == last_sent_text:
            return

        word_count = len(text.split())

        if is_final:
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
            text = capitalize_text(text)
            await send_queue.put({"id": f"s{output_counter}", "text": text})
            last_sent_text = text

    async def receive_audio():
        nonlocal running, audio_buffer, last_audio_time

        while running:
            try:
                data = await websocket.receive()
                if data.get("bytes"):
                    audio_buffer.extend(data["bytes"])
                    last_audio_time = time.time()
            except WebSocketDisconnect:
                running = False
                break
            except Exception as e:
                logger.error(f"Receive error: {e}")
                running = False
                break

    async def process_audio():
        nonlocal running, processing, audio_buffer, state

        loop = asyncio.get_event_loop()

        while running:
            await asyncio.sleep(0.1)

            time_since_audio = time.time() - last_audio_time

            if state.accumulated_text and time_since_audio > SILENCE_THRESHOLD_SEC:
                await emit_transcription(state.accumulated_text, is_final=True)
                state.reset()
                audio_buffer = bytearray()
                continue

            if processing or len(audio_buffer) < min_chunk_bytes:
                continue

            if len(audio_buffer) >= chunk_bytes or time_since_audio > 0.5:
                processing = True

                try:
                    chunk_size = min(len(audio_buffer), chunk_bytes)
                    chunk = bytes(audio_buffer[:chunk_size])
                    audio_buffer = audio_buffer[chunk_size:]

                    text, state = await loop.run_in_executor(
                        executor, stream_transcribe_chunk, chunk, state, 16000
                    )

                    if text:
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

    try:
        recv_task = asyncio.create_task(receive_audio())
        proc_task = asyncio.create_task(process_audio())
        send_task = asyncio.create_task(send_results())

        await recv_task

        if len(audio_buffer) >= min_chunk_bytes:
            loop = asyncio.get_event_loop()
            text, _ = await loop.run_in_executor(
                executor, stream_transcribe_chunk, bytes(audio_buffer), state, 16000
            )
            if text:
                await emit_transcription(text, is_final=True)
        elif state.accumulated_text:
            await emit_transcription(state.accumulated_text, is_final=True)

        running = False
        proc_task.cancel()
        send_task.cancel()

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
