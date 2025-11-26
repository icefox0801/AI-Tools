"""
Vosk ASR Service
Lightweight, fast, offline speech recognition with native streaming support
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
import logging
from vosk import Model, KaldiRecognizer, SetLogLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress Vosk internal logs
SetLogLevel(-1)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Vosk model
MODEL_PATH = "/app/model"
SAMPLE_RATE = 16000

logger.info(f"Loading Vosk model from {MODEL_PATH}...")
try:
    model = Model(MODEL_PATH)
    logger.info("Vosk model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Vosk model: {e}")
    raise


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "vosk-model-en-us-0.22",
        "backend": "vosk (Kaldi)",
        "device": "cpu",
        "sample_rate": SAMPLE_RATE,
        "streaming": True
    }


@app.websocket("/transcribe")
async def transcribe_audio(websocket: WebSocket):
    """
    WebSocket endpoint for streaming speech recognition.
    
    Protocol:
    1. Client sends JSON config: {"language": "en", "task": "transcribe"}
    2. Server responds: {"status": "ready", "message": "Vosk ASR ready"}
    3. Client streams raw PCM audio (int16, 16kHz, mono)
    4. Server sends partial results: {"partial": "hello wor"}
    5. Server sends final results: {"text": "hello world"}
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    # Create recognizer for this session
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(True)  # Include word-level timestamps
    
    try:
        # Wait for initial config message (JSON)
        first_msg = await websocket.receive_text()
        config = json.loads(first_msg)
        logger.info(f"Received config: {config}")
        
        # Send ready confirmation
        await websocket.send_json({
            "status": "ready",
            "message": "Vosk ASR ready",
            "streaming": True
        })
        
        while True:
            # Receive audio data (raw PCM int16)
            data = await websocket.receive_bytes()
            
            # Empty data signals end of audio stream
            if len(data) == 0:
                # Get final result
                final = json.loads(recognizer.FinalResult())
                text = final.get("text", "").strip()
                if text:
                    logger.info(f"Final (end signal): {text}")
                    await websocket.send_json({"text": text})
                continue
            
            # Process audio through Vosk
            if recognizer.AcceptWaveform(data):
                # Final result for this utterance
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                
                if text:
                    logger.info(f"Final: {text}")
                    await websocket.send_json({"text": text})
            else:
                # Partial result (still processing)
                partial = json.loads(recognizer.PartialResult())
                partial_text = partial.get("partial", "").strip()
                
                if partial_text:
                    await websocket.send_json({"partial": partial_text})
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        # Get any remaining audio
        final = json.loads(recognizer.FinalResult())
        text = final.get("text", "").strip()
        if text:
            logger.info(f"Final on disconnect: {text}")
    except Exception as e:
        logger.error(f"Error in transcription: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass


@app.websocket("/stream")
async def stream_transcribe(websocket: WebSocket):
    """
    Alternative streaming endpoint with simpler protocol.
    Just streams audio bytes and receives text results.
    """
    await websocket.accept()
    logger.info("Stream connection established")
    
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    await websocket.send_text(text)
            else:
                partial = json.loads(recognizer.PartialResult())
                partial_text = partial.get("partial", "")
                if partial_text:
                    await websocket.send_json({"partial": partial_text})
    
    except WebSocketDisconnect:
        logger.info("Stream disconnected")
        final = json.loads(recognizer.FinalResult())
        text = final.get("text", "").strip()
        if text:
            try:
                await websocket.send_text(text)
            except:
                pass
    except Exception as e:
        logger.error(f"Stream error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
