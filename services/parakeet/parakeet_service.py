"""
NVIDIA NeMo Parakeet ASR Service v3
GPU-accelerated speech recognition using Parakeet TDT with word timestamps

Key improvements:
- Uses word-level timestamps from Parakeet TDT (no overlap needed)
- Pause-based punctuation using actual inter-word gaps from audio
- Rule: pause >0.8s → period, 0.3-0.8s → comma
- No text-based punctuation model - uses prosodic cues only
- Continuous audio accumulation with timestamp-based deduplication
"""

import io
import os
import json
import logging
import tempfile
import time
import hashlib
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============== GPU OPTIMIZATION SETTINGS ==============
MODEL_NAME = os.getenv("PARAKEET_MODEL", "nvidia/parakeet-tdt-1.1b")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_FP16 = os.getenv("USE_FP16", "true").lower() == "true"
USE_CUDNN_BENCHMARK = True

# Punctuation thresholds (in seconds)
# Based on research: short pause (0.3-0.8s) = comma, long pause (>0.8s) = period
# However, TDT timestamps may have gaps between words even in continuous speech
# So we use slightly higher thresholds to account for natural word spacing
COMMA_PAUSE_THRESHOLD = 0.5   # Pause > 0.5s → comma
PERIOD_PAUSE_THRESHOLD = 1.0  # Pause > 1.0s → period

# Torch compile settings (PyTorch 2.x JIT compiler for ~1.5-2x speedup)
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "false").lower() == "true"

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = USE_CUDNN_BENCHMARK
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
    logger.info(f"CUDA optimizations enabled: FP16={USE_FP16}, cuDNN benchmark={USE_CUDNN_BENCHMARK}")

app = FastAPI(
    title="Parakeet ASR Service",
    description="GPU-accelerated streaming ASR with word timestamps and pause-based punctuation",
    version="3.0.0"
)

# Global model instance
asr_model = None

# Session storage
audio_sessions: Dict[str, dict] = {}
SESSION_TIMEOUT = 60


@dataclass
class TimestampedWord:
    """Word with timing information"""
    word: str
    start: float  # Start time in seconds (relative to session start)
    end: float    # End time in seconds


def cleanup_old_sessions():
    """Remove sessions that haven't been accessed recently"""
    current_time = time.time()
    expired = [sid for sid, data in audio_sessions.items() 
               if current_time - data['last_access'] > SESSION_TIMEOUT]
    for sid in expired:
        logger.info(f"Cleaning up expired session: {sid[:8]}...")
        del audio_sessions[sid]


def get_or_create_session(session_id: str) -> dict:
    """Get existing session or create new one"""
    cleanup_old_sessions()
    
    if session_id not in audio_sessions:
        audio_sessions[session_id] = {
            'audio': bytearray(),           # Full audio buffer for session
            'last_access': time.time(),
            'words': [],                    # List of TimestampedWord objects
            'audio_offset': 0.0,            # Total audio duration processed so far
            'last_transcribed_end': 0.0,    # End time of last transcribed word
        }
        logger.info(f"Created new session: {session_id[:8]}...")
    else:
        audio_sessions[session_id]['last_access'] = time.time()
    
    return audio_sessions[session_id]


def load_model():
    """Load the Parakeet TDT model"""
    global asr_model
    
    if asr_model is not None:
        return asr_model
    
    logger.info(f"Loading Parakeet model: {MODEL_NAME}")
    logger.info(f"Device: {DEVICE}")
    
    if DEVICE == "cuda":
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    try:
        import nemo.collections.asr as nemo_asr
        
        asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(MODEL_NAME)
        asr_model = asr_model.to(DEVICE)
        asr_model.eval()
        
        if DEVICE == "cuda" and USE_FP16:
            try:
                asr_model = asr_model.half()
                logger.info("ASR model converted to FP16")
            except Exception as e:
                logger.warning(f"FP16 conversion failed: {e}")
        
        # Apply torch.compile() for faster inference (PyTorch 2.x)
        if USE_TORCH_COMPILE and DEVICE == "cuda":
            try:
                logger.info("Applying torch.compile() optimization...")
                # Use 'reduce-overhead' mode for best latency
                asr_model = torch.compile(asr_model, mode="reduce-overhead")
                logger.info("torch.compile() applied successfully")
            except Exception as e:
                logger.warning(f"torch.compile() failed (will use eager mode): {e}")
        
        # Warmup
        if DEVICE == "cuda":
            logger.info("Warming up GPU...")
            try:
                dummy_audio = torch.randn(1, 16000).to(DEVICE)
                if USE_FP16:
                    dummy_audio = dummy_audio.half()
                with torch.no_grad():
                    _ = asr_model.preprocessor(input_signal=dummy_audio, length=torch.tensor([16000]).to(DEVICE))
                torch.cuda.synchronize()
                logger.info("GPU warmup complete")
            except Exception as e:
                logger.warning(f"GPU warmup skipped: {e}")
            
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
        
        logger.info(f"ASR model loaded in {'FP16' if USE_FP16 and DEVICE == 'cuda' else 'FP32'} mode")
        return asr_model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


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
        # Convert raw PCM to float array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        if len(audio_array) < 1600:  # Less than 0.1 second
            return []
        
        # Save to temp file (NeMo requires file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, audio_array, sample_rate)
        
        try:
            # Transcribe with timestamps enabled
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
            
            # Extract word timestamps from hypothesis
            if results and len(results) > 0:
                hypothesis = results[0]
                
                if hasattr(hypothesis, 'timestamp') and hypothesis.timestamp:
                    word_timestamps = hypothesis.timestamp.get('word', [])
                    
                    words_with_times = []
                    for wt in word_timestamps:
                        if isinstance(wt, dict):
                            # Use 'start' and 'end' which are in seconds
                            # (not 'start_offset'/'end_offset' which are in frames)
                            words_with_times.append({
                                'word': wt.get('word', wt.get('char', '')),
                                'start': wt.get('start', 0),
                                'end': wt.get('end', 0)
                            })
                        elif hasattr(wt, 'word'):
                            words_with_times.append({
                                'word': wt.word,
                                'start': getattr(wt, 'start', 0),
                                'end': getattr(wt, 'end', 0)
                            })
                    
                    return words_with_times
                
                # Fallback: no timestamps, just return text
                if hasattr(hypothesis, 'text') and hypothesis.text:
                    audio_duration = len(audio_array) / sample_rate
                    words = hypothesis.text.strip().split()
                    if words:
                        # Estimate timing evenly distributed
                        word_duration = audio_duration / len(words)
                        return [{
                            'word': w,
                            'start': i * word_duration,
                            'end': (i + 1) * word_duration
                        } for i, w in enumerate(words)]
            
            return []
            
        finally:
            os.unlink(tmp_path)
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return []


def apply_pause_punctuation(words: List[dict], debug: bool = False) -> str:
    """
    Apply punctuation based on inter-word pauses.
    
    Rules:
    - pause > PERIOD_PAUSE_THRESHOLD → period (end of sentence)
    - pause > COMMA_PAUSE_THRESHOLD → comma (clause break)
    - Otherwise no punctuation
    
    Args:
        words: List of {'word': str, 'start': float, 'end': float}
        debug: If True, log pause durations
    
    Returns:
        Punctuated text string
    """
    if not words:
        return ""
    
    result_parts = []
    pause_info = []  # For debugging
    
    for i, word_info in enumerate(words):
        word = word_info['word']
        
        # Add the word
        result_parts.append(word)
        
        # Calculate pause after this word (if not last word)
        if i < len(words) - 1:
            next_word = words[i + 1]
            pause_duration = next_word['start'] - word_info['end']
            
            if debug and pause_duration > 0.1:
                pause_info.append(f"{word}→{pause_duration:.2f}s")
            
            if pause_duration > PERIOD_PAUSE_THRESHOLD:
                # Long pause → period, capitalize next word
                result_parts.append('.')
            elif pause_duration > COMMA_PAUSE_THRESHOLD:
                # Medium pause → comma
                result_parts.append(',')
    
    if debug and pause_info:
        logger.info(f"Pauses detected: {', '.join(pause_info[:10])}")  # Log first 10
    
    # Build final text
    text = ""
    capitalize_next = True
    
    for part in result_parts:
        if part in '.?!':
            text = text.rstrip() + part + ' '
            capitalize_next = True
        elif part == ',':
            text = text.rstrip() + part + ' '
        else:
            if capitalize_next and part:
                part = part[0].upper() + part[1:] if len(part) > 1 else part.upper()
                capitalize_next = False
            text += part + ' '
    
    return text.strip()


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
    except Exception as e:
        logger.error(f"Startup failed: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = asr_model is not None
    cuda_available = torch.cuda.is_available()
    
    gpu_info = {}
    if cuda_available:
        gpu_info = {
            "cuda_device": torch.cuda.get_device_name(0),
            "memory_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "memory_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
            "fp16_enabled": USE_FP16,
        }
    
    return {
        "status": "healthy" if model_loaded else "loading",
        "model": MODEL_NAME,
        "model_loaded": model_loaded,
        "device": DEVICE,
        "cuda_available": cuda_available,
        "punctuation_mode": "pause-based",
        **gpu_info
    }


@app.get("/info")
async def model_info():
    """Get model information"""
    return {
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "punctuation": {
            "mode": "pause-based",
            "comma_threshold_sec": COMMA_PAUSE_THRESHOLD,
            "period_threshold_sec": PERIOD_PAUSE_THRESHOLD,
        }
    }


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = "en"
):
    """Transcribe an audio file with pause-based punctuation"""
    try:
        audio_data = await file.read()
        
        # Try to read as audio file
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
        
        return {
            "text": text,
            "words": words,
            "language": language,
            "model": MODEL_NAME
        }
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# UNIFIED STREAMING API v3.0
# ============================================================================
# Protocol:
# 1. Client connects to /stream WebSocket
# 2. Client sends config JSON: {"chunk_ms": 500}
# 3. Client streams raw audio bytes (16kHz, mono, int16 PCM)
# 4. Server buffers audio and transcribes at chunk_ms intervals
# 5. Server sends: {"id": "s1", "text": "...", "is_final": false/true}
#
# Client logic: if ID exists → replace, if not → append
# ============================================================================

DEFAULT_CHUNK_MS = 500  # Default chunk duration for Parakeet
MIN_CHUNK_MS = 300  # Minimum chunk duration


@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Unified WebSocket streaming endpoint with segment IDs.
    
    Protocol:
    1. First message: JSON config {"chunk_ms": 500}
    2. Then: raw audio bytes (16kHz, mono, int16 PCM)
    
    Server responses:
    - {"id": "s0", "text": "...", "is_final": false} - partial (replace by ID)
    - {"id": "s1", "text": "...", "is_final": true} - finalized (new segment)
    
    Client logic: if ID exists → replace text, if not → append new segment
    """
    await websocket.accept()
    logger.info("WebSocket connection established (Unified API v3.0)")
    
    # Configuration
    chunk_ms = DEFAULT_CHUNK_MS
    configured = False
    
    # State
    pending_audio = bytearray()
    transcribed_words = []  # Words for CURRENT segment only
    last_transcribe_time = time.time()
    last_speech_time = time.time()  # Track when we last got speech
    segment_counter = 0
    current_segment_id = "s0"
    total_audio_time = 0.0  # Track total audio duration for timestamps
    
    # Silence detection settings
    SILENCE_THRESHOLD_SEC = 1.5  # Finalize after 1.5s of silence
    
    async def finalize_segment():
        """Finalize current segment and start new one"""
        nonlocal transcribed_words, segment_counter, current_segment_id, total_audio_time
        
        if transcribed_words:
            text = apply_pause_punctuation(transcribed_words, debug=False)
            # Send final text for this segment
            await websocket.send_json({
                "id": current_segment_id,
                "text": text
            })
            logger.info(f"Finalized [{current_segment_id}]: {text[:50]}...")
            
            # Start new segment
            segment_counter += 1
            current_segment_id = f"s{segment_counter}"
            transcribed_words = []
    
    async def transcribe_pending(is_final: bool = False):
        """Transcribe pending audio and send results"""
        nonlocal pending_audio, transcribed_words, last_transcribe_time, last_speech_time
        nonlocal segment_counter, current_segment_id, total_audio_time
        
        min_bytes = int(0.3 * 16000 * 2)
        
        if len(pending_audio) < min_bytes:
            if transcribed_words and is_final:
                await finalize_segment()
            return
        
        audio_bytes = bytes(pending_audio)
        audio_duration = len(pending_audio) / (16000 * 2)
        
        logger.info(f"Transcribing chunk: {audio_duration:.2f}s")
        
        chunk_words = transcribe_with_timestamps(audio_bytes)
        
        if chunk_words:
            # Adjust timestamps to be relative to segment start
            time_offset = transcribed_words[-1]['end'] if transcribed_words else 0.0
            
            for w in chunk_words:
                w['start'] += time_offset
                w['end'] += time_offset
                transcribed_words.append(w)
            
            last_speech_time = time.time()  # Got speech!
            logger.info(f"Added {len(chunk_words)} words (segment: {len(transcribed_words)})")
        
        pending_audio = bytearray()
        total_audio_time += audio_duration
        last_transcribe_time = time.time()
        
        if transcribed_words:
            text = apply_pause_punctuation(transcribed_words, debug=False)
            
            if is_final:
                await finalize_segment()
            else:
                # Send update - client will replace by ID
                await websocket.send_json({
                    "id": current_segment_id,
                    "text": text
                })
                logger.debug(f"Update [{current_segment_id}]: {text[-80:] if len(text) > 80 else text}")
    
    try:
        while True:
            data = await websocket.receive()
            
            if "text" in data:
                try:
                    msg = json.loads(data["text"])
                    
                    if "chunk_ms" in msg:
                        chunk_ms = max(MIN_CHUNK_MS, msg.get("chunk_ms", DEFAULT_CHUNK_MS))
                        configured = True
                        logger.info(f"Configured: chunk_ms={chunk_ms}")
                        continue
                    
                    if msg.get("action") == "clear":
                        pending_audio = bytearray()
                        transcribed_words = []
                        segment_counter += 1
                        current_segment_id = f"s{segment_counter}"
                        logger.info("Session cleared")
                        await websocket.send_json({
                            "id": current_segment_id,
                            "text": ""
                        })
                        continue
                        
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON received")
                continue
            
            if "bytes" in data:
                audio_data = data["bytes"]
                
                if not configured:
                    configured = True
                    logger.info(f"Auto-configured with defaults: chunk_ms={chunk_ms}")
                
                if len(audio_data) == 0:
                    await transcribe_pending(is_final=False)
                    continue
                
                pending_audio.extend(audio_data)
                
                elapsed_ms = (time.time() - last_transcribe_time) * 1000
                if elapsed_ms >= chunk_ms:
                    await transcribe_pending(is_final=False)
                    
                    # Check for silence - finalize segment after SILENCE_THRESHOLD_SEC of no speech
                    silence_duration = time.time() - last_speech_time
                    if transcribed_words and silence_duration > SILENCE_THRESHOLD_SEC:
                        await finalize_segment()
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        if transcribed_words or len(pending_audio) > 0:
            try:
                await transcribe_pending(is_final=True)
            except:
                pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass


@app.post("/clear_session/{session_id}")
async def clear_session(session_id: str):
    """Clear audio buffer for a session"""
    if session_id in audio_sessions:
        audio_sessions[session_id]['audio'].clear()
        audio_sessions[session_id]['words'] = []
        audio_sessions[session_id]['last_transcribed_end'] = 0.0
        return {"status": "cleared", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
