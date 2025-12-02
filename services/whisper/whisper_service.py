"""
Whisper ASR Service - GPU-accelerated streaming transcription
Using OpenAI Whisper Large V3 Turbo via HuggingFace Transformers

Features:
- Fast GPU inference with Flash Attention 2
- Streaming WebSocket API with segment-based protocol
- Text refinement integration (punctuation + correction)
- Automatic language detection

Protocol:
1. Client connects to /stream
2. Client streams raw PCM audio (int16, 16kHz, mono)
3. Server sends: {"id": "s1", "text": "..."} for segments
"""

import os
import io
import json
import time
import asyncio
import logging
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import shared text refiner module
from shared.text_refiner import get_client, refine_text, capitalize_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-large-v3-turbo")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
SAMPLE_RATE = 16000

# Flash Attention - only enabled if flash_attn package is installed
USE_FLASH_ATTENTION = os.getenv("USE_FLASH_ATTENTION", "true").lower() == "true"

# Chunk settings for streaming
CHUNK_DURATION_SEC = 3.0  # Process audio in 3-second chunks
MIN_AUDIO_SEC = 0.5       # Minimum audio to process

# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(title="Whisper ASR Service", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model
whisper_pipe = None
text_refiner = get_client()


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
    if DEVICE == "cuda" and USE_FLASH_ATTENTION:
        try:
            import flash_attn
            use_flash_attn = True
            logger.info("Flash Attention 2 available and enabled")
        except ImportError:
            logger.info("Flash Attention 2 not installed, using SDPA (scaled dot product attention)")
    
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
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL,
        **model_kwargs
    )
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
    """Load model on startup."""
    load_model()
    await get_client().check_availability()


@app.get("/health")
async def health_check():
    model_loaded = whisper_pipe is not None
    
    gpu_info = {}
    if DEVICE == "cuda" and torch.cuda.is_available():
        gpu_info = {
            "cuda_device": torch.cuda.get_device_name(0),
            "memory_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        }
    
    return {
        "status": "healthy" if model_loaded else "loading",
        "model": WHISPER_MODEL,
        "model_loaded": model_loaded,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "text_refiner_enabled": text_refiner.enabled,
        "text_refiner_available": text_refiner.available,
        **gpu_info
    }


@app.get("/info")
async def info():
    """Return model information for backend config display."""
    return {
        "model_name": WHISPER_MODEL,
        "model": WHISPER_MODEL,
        "device": DEVICE,
    }


@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...)):
    """
    Transcribe an uploaded audio file.
    
    Args:
        file: Audio file (WAV, MP3, etc.)
        
    Returns:
        JSON with text transcription
    """
    if whisper_pipe is None:
        return {"error": "Model not loaded", "text": ""}
    
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
        
        logger.info(f"Transcribing file: {file.filename}, duration: {len(audio_array)/SAMPLE_RATE:.1f}s")
        
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
        
        # Optionally refine text
        if text and text_refiner.available:
            text = await refine_text(text)
        
        logger.info(f"Transcription complete: {len(text)} chars")
        
        return {
            "text": text,
            "duration": len(audio_array) / SAMPLE_RATE,
            "chunks": result.get("chunks", [])
        }
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return {"error": str(e), "text": ""}


def transcribe_audio(audio_array: np.ndarray) -> str:
    """Transcribe audio array to text."""
    if whisper_pipe is None:
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
                audio_buffer.extend(data["bytes"])
                current_time = time.time()
                
                # Convert to samples count (int16 = 2 bytes per sample)
                buffer_samples = len(audio_buffer) // 2
                
                # Process when we have enough audio or time elapsed
                should_process = (
                    buffer_samples >= chunk_samples or
                    (buffer_samples >= min_samples and 
                     current_time - last_process_time >= CHUNK_DURATION_SEC)
                )
                
                if should_process and buffer_samples >= min_samples:
                    # Convert to numpy array
                    audio_array = np.frombuffer(
                        bytes(audio_buffer), dtype=np.int16
                    ).astype(np.float32) / 32768.0
                    
                    # Transcribe
                    text = transcribe_audio(audio_array)
                    
                    if text:
                        # Apply text refinement (spelling correction only)
                        # Whisper already outputs punctuated text
                        refined_text = await refine_text(text, punctuate=False, correct=True)
                        
                        # Send result
                        await websocket.send_json({
                            "id": f"s{segment_counter}",
                            "text": refined_text
                        })
                        segment_counter += 1
                    
                    # Clear buffer
                    audio_buffer.clear()
                    last_process_time = current_time
                    
    except WebSocketDisconnect:
        logger.info("Stream disconnected")
    except Exception as e:
        logger.error(f"Stream error: {e}")
    finally:
        # Process remaining audio
        if len(audio_buffer) >= min_samples * 2:
            audio_array = np.frombuffer(
                bytes(audio_buffer), dtype=np.int16
            ).astype(np.float32) / 32768.0
            
            text = transcribe_audio(audio_array)
            if text:
                try:
                    refined_text = await refine_text(text)
                    await websocket.send_json({
                        "id": f"s{segment_counter}",
                        "text": refined_text
                    })
                except Exception:
                    pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
