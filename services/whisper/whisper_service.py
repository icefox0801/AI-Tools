"""
Faster-Whisper ASR Service
Optimized for real-time speech recognition with GPU acceleration
Uses CTranslate2 backend for efficient inference
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tempfile
import os
import logging
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Faster-Whisper model with GPU (CUDA)
# Using "large-v3" for best accuracy, float16 for GPU efficiency
logger.info("Loading Faster-Whisper model (large-v3) on GPU...")
try:
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    device_used = "cuda"
    logger.info("Whisper model loaded on GPU with float16")
except Exception as e:
    logger.warning(f"GPU loading failed: {e}, falling back to CPU")
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    device_used = "cpu"
    logger.info("Whisper model loaded on CPU with int8 quantization")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "openai/whisper-large-v3",
        "backend": "faster-whisper (CTranslate2)",
        "device": device_used,
        "sample_rate": 16000
    }

@app.websocket("/transcribe")
async def transcribe_audio(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    # Buffer for accumulating audio
    audio_buffer = []
    sample_rate = 16000
    chunk_duration = 2.0  # Process every 2 seconds for better accuracy
    samples_per_chunk = int(sample_rate * chunk_duration)
    
    try:
        # Wait for initial config message (JSON)
        first_msg = await websocket.receive_text()
        import json
        config = json.loads(first_msg)
        logger.info(f"Received config: {config}")
        
        # Send ready confirmation
        await websocket.send_json({"status": "ready", "message": "Whisper ASR ready"})
        
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Convert bytes to numpy array (expecting int16 PCM from browser)
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            # Convert int16 to float32 (-1.0 to 1.0)
            audio_chunk = audio_int16.astype(np.float32) / 32768.0
            audio_buffer.extend(audio_chunk.tolist())
            
            # Process when we have enough audio
            if len(audio_buffer) >= samples_per_chunk:
                # Convert to numpy array
                audio_array = np.array(audio_buffer[:samples_per_chunk], dtype=np.float32)
                audio_buffer = audio_buffer[samples_per_chunk:]
                
                # Transcribe with settings optimized for accuracy
                segments, info = model.transcribe(
                    audio_array,
                    beam_size=3,  # Better accuracy
                    language="en",
                    vad_filter=False,  # Disabled - let all audio through
                    without_timestamps=True,
                    condition_on_previous_text=False
                )
                
                # Collect transcription
                transcription = " ".join([segment.text for segment in segments]).strip()
                
                if transcription:
                    logger.info(f"Transcription: {transcription}")
                    await websocket.send_json({
                        "text": transcription,
                        "is_final": True,
                        "language": info.language,
                        "language_probability": info.language_probability
                    })
                    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in transcription: {e}")
        try:
            await websocket.close()
        except:
            pass

@app.post("/transcribe_file")
async def transcribe_file(audio_data: bytes):
    """Transcribe an audio file"""
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name
        
        # Transcribe
        segments, info = model.transcribe(temp_path, beam_size=5)
        transcription = " ".join([segment.text for segment in segments])
        
        # Cleanup
        os.unlink(temp_path)
        
        return {
            "text": transcription,
            "language": info.language,
            "language_probability": info.language_probability
        }
    except Exception as e:
        logger.error(f"Error in file transcription: {e}")
        return {"error": str(e)}
