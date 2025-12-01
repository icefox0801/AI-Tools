"""
NVIDIA NeMo Parakeet ASR Service v3.4
GPU-accelerated speech recognition with word timestamps

Supports all Parakeet models:
- nvidia/parakeet-tdt-1.1b  (TDT: best for streaming)
- nvidia/parakeet-rnnt-1.1b (RNNT: highest accuracy)
- nvidia/parakeet-ctc-1.1b  (CTC: balanced)

Features:
- Sliding window overlap (300ms) to prevent word fragmentation
- Text refinement via text-refiner service (punctuation + correction)
- Fallback to pause-based punctuation if text-refiner unavailable
"""

import io
import os
import json
import logging
import tempfile
import time
from typing import Optional, List

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
import uvicorn

# Import shared text refiner module
from shared.text_refiner import get_client, check_text_refiner, refine_text, capitalize_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============== CONFIGURATION ==============
MODEL_NAME = os.getenv("PARAKEET_MODEL", "nvidia/parakeet-tdt-1.1b")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = os.getenv("USE_FP16", "true").lower() == "true"

# Streaming settings
DEFAULT_CHUNK_MS = 500
MIN_CHUNK_MS = 300
TRANSCRIBE_INTERVAL_MS = 400

# Sliding window overlap (prevents word fragmentation at chunk boundaries)
OVERLAP_MS = 300
OVERLAP_BYTES = int(OVERLAP_MS * 16 * 2)  # 300ms @ 16kHz, int16

# Punctuation thresholds (fallback when text-refiner unavailable)
COMMA_PAUSE_THRESHOLD = 0.5
PERIOD_PAUSE_THRESHOLD = 1.0
MIN_WORDS_FOR_PUNCTUATION = 6

# Text refiner client
text_refiner = get_client()

# CUDA optimizations
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
    logger.info(f"CUDA optimizations enabled: FP16={USE_FP16}")

# FastAPI app
app = FastAPI(
    title="Parakeet ASR Service",
    description="GPU-accelerated streaming ASR with word timestamps",
    version="3.4.0"
)

# Global model
asr_model = None


# ============== MODEL LOADING ==============

def load_model():
    """Load NeMo Parakeet model (TDT/RNNT/CTC)"""
    global asr_model
    
    if asr_model is not None:
        return asr_model
    
    logger.info(f"Loading model: {MODEL_NAME}")
    logger.info(f"Device: {DEVICE}")
    
    if DEVICE == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        import nemo.collections.asr as nemo_asr
        
        asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(MODEL_NAME)
        asr_model = asr_model.to(DEVICE)
        asr_model.eval()
        
        if DEVICE == "cuda" and USE_FP16:
            try:
                asr_model = asr_model.half()
                logger.info("Model converted to FP16")
            except Exception as e:
                logger.warning(f"FP16 conversion failed: {e}")
        
        # GPU warmup
        if DEVICE == "cuda":
            logger.info("Warming up GPU...")
            try:
                dummy = torch.randn(1, 16000).to(DEVICE)
                if USE_FP16:
                    dummy = dummy.half()
                with torch.no_grad():
                    _ = asr_model.preprocessor(input_signal=dummy, length=torch.tensor([16000]).to(DEVICE))
                torch.cuda.synchronize()
                logger.info("GPU warmup complete")
            except Exception as e:
                logger.warning(f"GPU warmup skipped: {e}")
            
            mem = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU Memory: {mem:.2f}GB allocated")
        
        return asr_model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# ============== TRANSCRIPTION ==============

def transcribe_with_timestamps(audio_data: bytes, sample_rate: int = 16000) -> List[dict]:
    """
    Transcribe audio and return word-level timestamps.
    
    Returns:
        List of dicts: [{'word': str, 'start': float, 'end': float}, ...]
    """
    global asr_model
    
    if asr_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert PCM to float
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
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
                        results = asr_model.transcribe(
                            [tmp_path], 
                            timestamps=True,
                            return_hypotheses=True,
                            verbose=False
                        )
                    torch.cuda.synchronize()
                else:
                    results = asr_model.transcribe(
                        [tmp_path], 
                        timestamps=True,
                        return_hypotheses=True,
                        verbose=False
                    )
            
            # Extract word timestamps
            if results and len(results) > 0:
                hypothesis = results[0]
                
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
                
                # Fallback: no timestamps
                if hasattr(hypothesis, 'text') and hypothesis.text:
                    duration = len(audio_array) / sample_rate
                    words = hypothesis.text.strip().split()
                    if words:
                        word_dur = duration / len(words)
                        return [{'word': w, 'start': i * word_dur, 'end': (i + 1) * word_dur} 
                                for i, w in enumerate(words)]
            
            return []
            
        finally:
            os.unlink(tmp_path)
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return []


def apply_pause_punctuation(words: List[dict]) -> str:
    """Apply punctuation based on inter-word pauses (fallback mode)."""
    if not words:
        return ""
    
    result = []
    for i, w in enumerate(words):
        result.append(w['word'])
        
        if i < len(words) - 1:
            pause = words[i + 1]['start'] - w['end']
            if pause > PERIOD_PAUSE_THRESHOLD:
                result.append('.')
            elif pause > COMMA_PAUSE_THRESHOLD:
                result.append(',')
    
    # Build text with capitalization
    text = ""
    cap_next = True
    for part in result:
        if part in '.?!':
            text = text.rstrip() + part + ' '
            cap_next = True
        elif part == ',':
            text = text.rstrip() + part + ' '
        else:
            if cap_next and part:
                part = part[0].upper() + part[1:] if len(part) > 1 else part.upper()
                cap_next = False
            text += part + ' '
    
    return text.strip()


# ============== HTTP ENDPOINTS ==============

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
        await check_text_refiner()
    except Exception as e:
        logger.error(f"Startup failed: {e}")


@app.get("/health")
async def health_check():
    """Health check"""
    model_loaded = asr_model is not None
    
    info = {
        "status": "healthy" if model_loaded else "loading",
        "model": MODEL_NAME,
        "model_loaded": model_loaded,
        "device": DEVICE,
        "text_refiner_available": text_refiner.available,
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_device": torch.cuda.get_device_name(0),
            "memory_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "fp16": USE_FP16,
        })
    
    return info


@app.get("/info")
async def model_info():
    """Service information"""
    return {
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "torch_version": torch.__version__,
        "streaming": {
            "api_version": "v3.4",
            "overlap_ms": OVERLAP_MS,
            "default_chunk_ms": DEFAULT_CHUNK_MS,
            "transcribe_interval_ms": TRANSCRIBE_INTERVAL_MS,
        },
        "text_refiner": {
            "enabled": text_refiner.enabled,
            "available": text_refiner.available,
            "url": text_refiner.url,
        },
    }


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe an audio file"""
    try:
        audio_data = await file.read()
        
        # Try to decode audio file
        try:
            audio_io = io.BytesIO(audio_data)
            audio_array, sr = sf.read(audio_io)
            if sr != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            audio_data = (audio_array * 32768).astype(np.int16).tobytes()
        except Exception:
            pass
        
        words = transcribe_with_timestamps(audio_data)
        text = apply_pause_punctuation(words)
        
        return {"text": text, "words": words, "model": MODEL_NAME}
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== WEBSOCKET STREAMING ==============

@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Streaming transcription endpoint.
    
    Protocol:
    1. Send config: {"chunk_ms": 500}
    2. Stream audio: raw PCM bytes (16kHz, mono, int16)
    3. Receive: {"id": "s0", "text": "..."}
    
    Client: if id exists → replace, else → append
    """
    await websocket.accept()
    logger.info("WebSocket connected (API v3.4)")
    
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    executor = ThreadPoolExecutor(max_workers=1)
    
    # State
    chunk_ms = DEFAULT_CHUNK_MS
    audio_buffer = bytearray()
    overlap_buffer = bytearray()
    last_words_from_overlap = []
    transcribed_words = []
    output_counter = 0
    segment_start = time.time()
    last_speech = time.time()
    running = True
    transcribing = False
    send_queue = asyncio.Queue()
    
    # Segment settings
    MAX_WORDS = 20
    MAX_DURATION = 8.0
    SILENCE_THRESHOLD = 1.5
    PAUSE_CUT = 0.4
    MIN_WORDS_CUT = 5
    
    def find_cut_point(words):
        if len(words) < MIN_WORDS_CUT:
            return -1
        for i in range(len(words) - 2, MIN_WORDS_CUT - 2, -1):
            if i + 1 < len(words):
                pause = words[i + 1]['start'] - words[i]['end']
                if pause >= PAUSE_CUT:
                    return i + 1
        return -1
    
    async def process_result(chunk_words, is_final=False):
        nonlocal transcribed_words, output_counter, segment_start, last_speech
        
        if chunk_words:
            offset = transcribed_words[-1]['end'] if transcribed_words else 0.0
            for w in chunk_words:
                w['start'] += offset
                w['end'] += offset
                transcribed_words.append(w)
            last_speech = time.time()
        
        if not transcribed_words:
            return
        
        duration = time.time() - segment_start
        cut = find_cut_point(transcribed_words)
        
        should_finalize = (is_final or 
                          len(transcribed_words) >= MAX_WORDS or 
                          duration >= MAX_DURATION or
                          (cut > 0 and len(transcribed_words) >= MIN_WORDS_CUT + 3))
        
        if should_finalize:
            if 0 < cut < len(transcribed_words):
                to_finalize = transcribed_words[:cut]
                to_keep = transcribed_words[cut:]
            else:
                to_finalize = transcribed_words
                to_keep = []
            
            if to_finalize:
                raw = ' '.join(w['word'] for w in to_finalize)
                
                if text_refiner.available and len(to_finalize) >= MIN_WORDS_FOR_PUNCTUATION:
                    text = await refine_text(raw)
                else:
                    text = capitalize_text(raw)
                    if is_final or len(to_finalize) >= MIN_WORDS_FOR_PUNCTUATION:
                        text += '.'
                
                await send_queue.put({"id": f"s{output_counter}", "text": text})
                output_counter += 1
                segment_start = time.time()
            
            if to_keep:
                offset = to_keep[0]['start']
                for w in to_keep:
                    w['start'] -= offset
                    w['end'] -= offset
                transcribed_words = to_keep
            else:
                transcribed_words = []
        else:
            text = capitalize_text(' '.join(w['word'] for w in transcribed_words))
            await send_queue.put({"id": f"s{output_counter}", "text": text})
    
    async def receive_audio():
        nonlocal running, chunk_ms, audio_buffer
        
        while running:
            try:
                data = await websocket.receive()
                
                if "text" in data:
                    try:
                        msg = json.loads(data["text"])
                        if "chunk_ms" in msg:
                            chunk_ms = max(MIN_CHUNK_MS, msg["chunk_ms"])
                    except json.JSONDecodeError:
                        pass
                    continue
                
                if "bytes" in data and data["bytes"]:
                    audio_buffer.extend(data["bytes"])
                            
            except WebSocketDisconnect:
                running = False
                break
            except Exception as e:
                logger.error(f"Receive error: {e}")
                running = False
                break
    
    async def transcription_loop():
        nonlocal running, transcribing, audio_buffer, overlap_buffer, last_words_from_overlap
        nonlocal transcribed_words, last_speech
        
        loop = asyncio.get_event_loop()
        min_bytes = int(0.3 * 16000 * 2)
        
        while running:
            await asyncio.sleep(TRANSCRIBE_INTERVAL_MS / 1000.0)
            
            if transcribing or len(audio_buffer) < min_bytes:
                continue
            
            new_audio = bytes(audio_buffer)
            audio_buffer = bytearray()
            
            # Sliding window: combine overlap + new
            if overlap_buffer:
                combined = bytes(overlap_buffer) + new_audio
                overlap_dur = len(overlap_buffer) / (16000 * 2)
            else:
                combined = new_audio
                overlap_dur = 0.0
            
            transcribing = True
            try:
                words = await loop.run_in_executor(executor, transcribe_with_timestamps, combined)
                
                # Deduplicate overlap region
                if overlap_dur > 0 and words and last_words_from_overlap:
                    start_idx = 0
                    for i, w in enumerate(words):
                        if w['start'] >= overlap_dur - 0.1:
                            start_idx = i
                            break
                        if last_words_from_overlap and w['word'].lower() == last_words_from_overlap[-1].lower():
                            start_idx = i + 1
                    
                    if start_idx > 0:
                        words = words[start_idx:]
                        for w in words:
                            w['start'] = max(0, w['start'] - overlap_dur)
                            w['end'] = max(0, w['end'] - overlap_dur)
                
                # Save overlap for next iteration
                if len(new_audio) >= OVERLAP_BYTES:
                    overlap_buffer = bytearray(new_audio[-OVERLAP_BYTES:])
                    last_words_from_overlap = [w['word'] for w in words[-3:]] if words else []
                else:
                    overlap_buffer = bytearray(new_audio)
                    last_words_from_overlap = [w['word'] for w in words] if words else []
                
                await process_result(words)
                
                # Check silence
                if transcribed_words and time.time() - last_speech > SILENCE_THRESHOLD:
                    await process_result([], is_final=True)
                        
            except Exception as e:
                logger.error(f"Transcription error: {e}")
            finally:
                transcribing = False
    
    async def send_results():
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
    
    try:
        recv_task = asyncio.create_task(receive_audio())
        trans_task = asyncio.create_task(transcription_loop())
        send_task = asyncio.create_task(send_results())
        
        await recv_task
        
        # Final transcription
        if len(audio_buffer) >= int(0.1 * 16000 * 2):
            loop = asyncio.get_event_loop()
            words = await loop.run_in_executor(executor, transcribe_with_timestamps, bytes(audio_buffer))
            await process_result(words, is_final=True)
        
        running = False
        trans_task.cancel()
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
