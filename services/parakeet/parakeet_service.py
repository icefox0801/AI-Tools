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
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
import uvicorn

from shared.text_refiner import get_client, check_text_refiner, refine_text, capitalize_text

# ==============================================================================
# Configuration
# ==============================================================================

MODEL_NAME = os.getenv("PARAKEET_MODEL", "nvidia/parakeet-tdt-1.1b")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = os.getenv("USE_FP16", "true").lower() == "true"

# Streaming parameters
CHUNK_DURATION_SEC = 1.0      # Audio chunk size for processing
MIN_CHUNK_SEC = 0.3           # Minimum audio to process
SILENCE_THRESHOLD_SEC = 2.0   # Finalize segment after silence
MAX_WORDS_PER_SEGMENT = 30    # Force finalize long segments

# Text refinement
MIN_WORDS_FOR_PUNCTUATION = 6

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
    model: Optional[object] = None
    preprocessor: Optional[object] = None
    loaded: bool = False


_model_state = ModelState()


def get_model():
    """Get the loaded ASR model."""
    if not _model_state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _model_state.model


def get_preprocessor():
    """Get the audio preprocessor."""
    if not _model_state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _model_state.preprocessor


# ==============================================================================
# Model Loading
# ==============================================================================

def load_model():
    """Load NeMo Parakeet TDT model for streaming ASR."""
    if _model_state.loaded:
        return _model_state.model
    
    logger.info(f"Loading model: {MODEL_NAME}")
    logger.info(f"Device: {DEVICE}")
    
    if DEVICE == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        import nemo.collections.asr as nemo_asr
        
        # Load model
        model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(MODEL_NAME)
        model = model.to(DEVICE)
        model.eval()
        
        # Apply FP16 if enabled
        if DEVICE == "cuda" and USE_FP16:
            try:
                model = model.half()
                logger.info("Model converted to FP16")
            except Exception as e:
                logger.warning(f"FP16 conversion failed: {e}")
        
        # Store references
        _model_state.model = model
        _model_state.preprocessor = model.preprocessor
        _model_state.loaded = True
        
        # Log streaming config
        if hasattr(model.encoder, 'streaming_cfg'):
            logger.info(f"Encoder streaming config: {model.encoder.streaming_cfg}")
        
        # GPU warmup
        _warmup_model(model)
        
        if DEVICE == "cuda":
            mem_gb = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory allocated: {mem_gb:.2f} GB")
        
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def _warmup_model(model):
    """Warm up model with a test inference."""
    if DEVICE != "cuda":
        return
    
    logger.info("Warming up GPU...")
    try:
        dummy_audio = torch.randn(1, 16000).to(DEVICE)
        if USE_FP16:
            dummy_audio = dummy_audio.half()
        
        with torch.no_grad():
            processed, processed_len = _model_state.preprocessor(
                input_signal=dummy_audio,
                length=torch.tensor([16000]).to(DEVICE)
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
        logger.info("GPU warmup complete")
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
    
    Args:
        audio_data: Raw PCM audio bytes (int16)
        sample_rate: Audio sample rate
        
    Returns:
        List of dicts: [{'word': str, 'start': float, 'end': float}, ...]
    """
    model = get_model()
    audio_array = pcm_to_float(audio_data)
    
    # Skip very short audio
    if len(audio_array) < 1600:  # < 0.1 second
        return []
    
    # Save to temp file (NeMo requires file path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        sf.write(tmp_path, audio_array, sample_rate)
    
    try:
        with torch.no_grad():
            if DEVICE == "cuda":
                torch.cuda.synchronize()
                with torch.amp.autocast('cuda', enabled=USE_FP16):
                    results = model.transcribe(
                        [tmp_path],
                        timestamps=True,
                        return_hypotheses=True,
                        verbose=False
                    )
                torch.cuda.synchronize()
            else:
                results = model.transcribe(
                    [tmp_path],
                    timestamps=True,
                    return_hypotheses=True,
                    verbose=False
                )
        
        return _extract_word_timestamps(results, audio_array, sample_rate)
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return []
    finally:
        os.unlink(tmp_path)


def _extract_word_timestamps(results, audio_array: np.ndarray, sample_rate: int) -> list[dict]:
    """Extract word timestamps from NeMo transcription results."""
    if not results or len(results) == 0:
        return []
    
    hypothesis = results[0]
    
    # Try to get word timestamps
    if hasattr(hypothesis, 'timestamp') and hypothesis.timestamp:
        word_timestamps = hypothesis.timestamp.get('word', [])
        words = []
        for wt in word_timestamps:
            if isinstance(wt, dict):
                words.append({
                    'word': wt.get('word', wt.get('char', '')),
                    'start': wt.get('start', 0),
                    'end': wt.get('end', 0)
                })
            elif hasattr(wt, 'word'):
                words.append({
                    'word': wt.word,
                    'start': getattr(wt, 'start', 0),
                    'end': getattr(wt, 'end', 0)
                })
        return words
    
    # Fallback: estimate timestamps from text
    if hasattr(hypothesis, 'text') and hypothesis.text:
        duration = len(audio_array) / sample_rate
        words = hypothesis.text.strip().split()
        if words:
            word_dur = duration / len(words)
            return [
                {'word': w, 'start': i * word_dur, 'end': (i + 1) * word_dur}
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
    cache_last_channel: Optional[torch.Tensor] = None
    cache_last_time: Optional[torch.Tensor] = None
    cache_last_channel_len: Optional[torch.Tensor] = None
    # Decoder state
    previous_hypotheses: Optional[object] = None
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


def stream_transcribe_chunk(audio_chunk: bytes, state: StreamingState, sample_rate: int = 16000) -> tuple[str, StreamingState]:
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
    if not _model_state.loaded:
        return state.accumulated_text, state
    
    model = _model_state.model
    preprocessor = _model_state.preprocessor
    
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
                input_signal=audio_tensor,
                length=audio_len
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
                text = hyp.text if hasattr(hyp, 'text') else str(hyp) if isinstance(hyp, str) else ""
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
    version="4.0.0"
)

text_refiner = get_client()


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    try:
        load_model()
        await check_text_refiner()
        logger.info("Service startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}")


# ==============================================================================
# HTTP Endpoints
# ==============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    info = {
        "status": "healthy" if _model_state.loaded else "loading",
        "model": MODEL_NAME,
        "model_loaded": _model_state.loaded,
        "device": DEVICE,
        "text_refiner_available": text_refiner.available,
    }
    
    if DEVICE == "cuda":
        info.update({
            "cuda_device": torch.cuda.get_device_name(0),
            "memory_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "fp16": USE_FP16,
        })
    
    return info


@app.get("/info")
async def model_info():
    """Service information endpoint."""
    return {
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "torch_version": torch.__version__,
        "streaming": {
            "api_version": "v4.0",
            "approach": "cache_aware_conformer_streaming",
            "chunk_duration_sec": CHUNK_DURATION_SEC,
            "description": "NeMo conformer_stream_step with encoder caches"
        },
        "text_refiner": {
            "enabled": text_refiner.enabled,
            "available": text_refiner.available,
            "url": text_refiner.url,
        },
    }


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
        raw_text = ' '.join(w['word'] for w in words)
        
        # Apply text refinement
        if text_refiner.available and len(words) >= MIN_WORDS_FOR_PUNCTUATION:
            text = await refine_text(raw_text)
        else:
            text = capitalize_text(raw_text)
        
        return {"text": text, "words": words, "model": MODEL_NAME}
        
    except Exception as e:
        logger.error(f"Transcribe error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    min_chunk_bytes = int(MIN_CHUNK_SEC * 16000 * 2)   # minimum chunk
    
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
                    text = text.rstrip('.') + '.'
            
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
                
                if "bytes" in data and data["bytes"]:
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
            except asyncio.TimeoutError:
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
