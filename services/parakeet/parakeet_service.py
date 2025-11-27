"""
NVIDIA NeMo Parakeet ASR Service
GPU-accelerated speech recognition using Parakeet TDT for real-time streaming

Features:
- TDT model for fastest real-time inference (skips blank predictions)
- FP16 mixed precision for 2x GPU throughput
- CUDA optimizations: cudnn benchmark, memory pooling
- Batched inference for efficiency
- English-only punctuation restoration
"""

import io
import os
import logging
import tempfile
import time
import hashlib
from typing import Optional, Dict, List
from collections import defaultdict

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
# Model configuration - TDT for faster real-time streaming
MODEL_NAME = os.getenv("PARAKEET_MODEL", "nvidia/parakeet-tdt-1.1b")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# GPU Performance settings
USE_FP16 = os.getenv("USE_FP16", "true").lower() == "true"  # Mixed precision
USE_CUDNN_BENCHMARK = True  # Optimize cuDNN kernels for fixed input sizes
USE_CUDA_GRAPHS = False  # Disabled - requires cuda-python package
INFERENCE_BATCH_SIZE = 1  # Single file inference

# Apply CUDA optimizations at module load
if DEVICE == "cuda":
    # Enable cuDNN autotuner - finds fastest algorithms
    torch.backends.cudnn.benchmark = USE_CUDNN_BENCHMARK
    torch.backends.cudnn.enabled = True
    # Use TF32 for faster matmul on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Memory management
    torch.cuda.empty_cache()
    logger.info(f"CUDA optimizations enabled: FP16={USE_FP16}, cuDNN benchmark={USE_CUDNN_BENCHMARK}")

app = FastAPI(
    title="Parakeet ASR Service",
    description="GPU-accelerated streaming ASR using NVIDIA NeMo Parakeet RNNT",
    version="2.0.0"
)

# Global model instances
asr_model = None
punctuation_model = None

# Session-based audio accumulation
audio_sessions: Dict[str, dict] = {}
SESSION_TIMEOUT = 60


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
            'audio': bytearray(),       # Audio buffer for current chunk
            'last_access': time.time(),
            'raw_text': '',             # Accumulated raw text (no punctuation)
            'full_text': '',            # Punctuated full transcription
        }
        logger.info(f"Created new session: {session_id[:8]}...")
    else:
        audio_sessions[session_id]['last_access'] = time.time()
    
    return audio_sessions[session_id]


def load_model():
    """Load the Parakeet RNNT model and punctuation model"""
    global asr_model, punctuation_model
    
    if asr_model is not None:
        return asr_model
    
    logger.info(f"Loading Parakeet model: {MODEL_NAME}")
    logger.info(f"Device: {DEVICE}")
    
    if DEVICE == "cuda":
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    
    try:
        import nemo.collections.asr as nemo_asr
        
        # Load RNNT model for streaming (or CTC as fallback)
        if "rnnt" in MODEL_NAME.lower() or "transducer" in MODEL_NAME.lower():
            try:
                asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(MODEL_NAME)
                logger.info(f"Loaded RNNT model: {MODEL_NAME}")
            except Exception as e:
                # Fallback to FastConformer Transducer from NGC if HuggingFace fails
                logger.warning(f"Failed to load {MODEL_NAME}: {e}")
                logger.info("Falling back to stt_en_fastconformer_transducer_large from NGC...")
                asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                    "stt_en_fastconformer_transducer_large"
                )
                logger.info("Loaded FastConformer Transducer (RNNT) from NGC")
        elif "tdt" in MODEL_NAME.lower():
            asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(MODEL_NAME)
            logger.info("Loaded TDT model")
        else:
            asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(MODEL_NAME)
            logger.info("Loaded CTC model - batch processing")
        
        asr_model = asr_model.to(DEVICE)
        asr_model.eval()
        
        # Apply GPU optimizations
        if DEVICE == "cuda":
            # Convert to FP16 for faster inference (2x speedup)
            if USE_FP16:
                try:
                    asr_model = asr_model.half()
                    logger.info("ASR model converted to FP16 for faster inference")
                except Exception as e:
                    logger.warning(f"FP16 conversion failed, using FP32: {e}")
            
            # Warmup GPU with dummy inference
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
            
            # Log GPU memory usage
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
        
        logger.info(f"ASR model loaded in {'FP16' if USE_FP16 and DEVICE == 'cuda' else 'FP32'} mode")
        
        # Load punctuation model (English-only)
        try:
            from punctuators.models import PunctCapSegModelONNX
            punctuation_model = PunctCapSegModelONNX.from_pretrained("pcs_en")
            logger.info("Punctuation model loaded: pcs_en (English-only)")
        except Exception as e:
            logger.warning(f"Failed to load punctuation model: {e}")
            logger.warning("Transcription will work without punctuation")
            punctuation_model = None
        
        return asr_model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # Don't crash - allow health checks to report unhealthy


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
            "cudnn_benchmark": USE_CUDNN_BENCHMARK,
        }
    
    return {
        "status": "healthy" if model_loaded else "loading",
        "model": MODEL_NAME,
        "model_loaded": model_loaded,
        "punctuation_loaded": punctuation_model is not None,
        "device": DEVICE,
        "cuda_available": cuda_available,
        **gpu_info
    }


@app.get("/info")
async def model_info():
    """Get model information"""
    info = {
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "optimizations": {
            "fp16": USE_FP16,
            "cudnn_benchmark": USE_CUDNN_BENCHMARK,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32 if DEVICE == "cuda" else False,
        }
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "cuda_arch": torch.cuda.get_device_capability(0),
            "gpu_memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        })
    
    return info


def process_audio(audio_data: bytes, sample_rate: int = 16000) -> str:
    """Process audio data and return transcription"""
    global asr_model
    
    if asr_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Save audio to temporary file (NeMo requires file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
            # Try to read as audio file first
            try:
                audio_io = io.BytesIO(audio_data)
                audio_array, sr = sf.read(audio_io)
                
                # Resample if necessary
                if sr != sample_rate:
                    import librosa
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=sample_rate)
                
                # Convert to mono if stereo
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.mean(axis=1)
                
                sf.write(tmp_path, audio_array, sample_rate)
                
            except Exception:
                # Assume raw PCM audio
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                sf.write(tmp_path, audio_array, sample_rate)
        
        # Transcribe with maximum GPU optimization
        with torch.no_grad():
            if DEVICE == "cuda":
                # Synchronize before inference for accurate timing
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                # Use autocast for mixed precision - faster on Tensor Cores
                with torch.amp.autocast('cuda', enabled=USE_FP16):
                    result = asr_model.transcribe([tmp_path])[0]
                
                torch.cuda.synchronize()
                inference_time = time.perf_counter() - start_time
                audio_duration = len(audio_array) / sample_rate
                rtf = inference_time / audio_duration if audio_duration > 0 else 0
                logger.debug(f"Inference: {inference_time:.3f}s for {audio_duration:.2f}s audio (RTF={rtf:.3f})")
            else:
                result = asr_model.transcribe([tmp_path])[0]
        
        # Handle different return types from NeMo
        # Newer versions return Hypothesis objects, older versions return strings
        if hasattr(result, 'text'):
            transcription = result.text
        elif hasattr(result, 'words'):
            transcription = ' '.join(result.words)
        else:
            transcription = str(result)
        
        # Clean up
        os.unlink(tmp_path)
        
        return transcription
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = "en"
):
    """
    Transcribe an audio file
    
    - **file**: Audio file (WAV, MP3, FLAC, etc.)
    - **language**: Language code (default: en)
    """
    try:
        audio_data = await file.read()
        transcription = process_audio(audio_data)
        
        # Apply punctuation if model is loaded
        if punctuation_model is not None and transcription:
            try:
                results = punctuation_model.infer([transcription])
                if results and results[0]:
                    transcription = ' '.join(results[0])
            except Exception as e:
                logger.warning(f"Punctuation failed: {e}")
        
        return {
            "text": transcription,
            "language": language,
            "model": MODEL_NAME
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming audio transcription
    
    Send raw PCM audio (16-bit, 16kHz, mono) and receive transcriptions.
    
    Protocol:
    1. Optionally send JSON {"session_id": "xxx"} to set session
    2. Send audio bytes (VAD-segmented chunks from client)
    3. Send empty bytes b"" to trigger transcription
    4. Receive JSON with transcription result
    
    The client handles VAD segmentation - this service just transcribes
    what it receives. Audio accumulates per session until transcribed.
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    # Session for this connection
    session_id = hashlib.md5(f"{id(websocket)}{time.time()}".encode()).hexdigest()
    session = get_or_create_session(session_id)
    
    async def transcribe_buffer():
        """Transcribe accumulated audio"""
        audio_buffer = session['audio']
        total_bytes = len(audio_buffer)
        
        if total_bytes < 16000:  # Less than 0.5 second
            return
        
        audio_sec = total_bytes / (16000 * 2)
        
        try:
            logger.info(f"Transcribing {audio_sec:.2f}s of audio")
            
            transcription = process_audio(bytes(audio_buffer))
            chunk_text = transcription.strip() if isinstance(transcription, str) else str(transcription).strip()
            
            if chunk_text:
                # Accumulate raw text
                if session['raw_text']:
                    session['raw_text'] += ' ' + chunk_text
                else:
                    session['raw_text'] = chunk_text
                
                # Apply punctuation to full accumulated text
                if punctuation_model is not None:
                    try:
                        results = punctuation_model.infer([session['raw_text']])
                        if results and results[0]:
                            session['full_text'] = ' '.join(results[0])
                        else:
                            session['full_text'] = session['raw_text']
                    except Exception as e:
                        logger.warning(f"Punctuation failed: {e}")
                        session['full_text'] = session['raw_text']
                else:
                    session['full_text'] = session['raw_text']
                
                logger.info(f"Full text: ...{session['full_text'][-100:]}")
                
                await websocket.send_json({
                    "type": "transcription",
                    "text": session['full_text'],
                    "duration": audio_sec
                })
            else:
                # No new text, but still send current full text
                await websocket.send_json({
                    "type": "transcription",
                    "text": session['full_text'],
                    "duration": audio_sec
                })
            
            # Clear audio buffer after transcription
            session['audio'].clear()
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
    
    try:
        while True:
            data = await websocket.receive()
            
            # Handle text messages (JSON commands)
            if "text" in data:
                import json
                try:
                    msg = json.loads(data["text"])
                    
                    if "session_id" in msg:
                        session_id = msg["session_id"]
                        session = get_or_create_session(session_id)
                        logger.debug(f"Using session: {session_id[:8]}...")
                        continue
                    
                    if msg.get("action") == "clear":
                        session['audio'].clear()
                        session['raw_text'] = ''
                        session['full_text'] = ''
                        logger.info(f"Cleared session: {session_id[:8]}...")
                        await websocket.send_json({"type": "cleared"})
                        continue
                        
                except json.JSONDecodeError:
                    pass
                continue
            
            # Handle binary audio data
            if "bytes" in data:
                audio_data = data["bytes"]
                
                # Empty data = transcribe now
                if len(audio_data) == 0:
                    await transcribe_buffer()
                    continue
                
                # Accumulate audio
                session['audio'].extend(audio_data)
                session['last_access'] = time.time()
                
                # Cap buffer at 30 seconds max
                max_buffer_bytes = 30 * 16000 * 2
                if len(session['audio']) > max_buffer_bytes:
                    session['audio'] = session['audio'][-max_buffer_bytes:]
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected, session {session_id[:8]}...")
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
        return {"status": "cleared", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
