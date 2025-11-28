"""
Vosk ASR Service - Unified Streaming API v3.0
Lightweight, fast, offline speech recognition with native streaming support

Unified Protocol:
1. Client connects to /stream
2. (Optional) Client sends JSON config: {"chunk_ms": 200}
3. Client streams raw PCM audio (int16, 16kHz, mono)
4. Server sends: {"id": "s1", "text": "...", "is_final": false} for interim results
5. Server sends: {"id": "s1", "text": "...", "is_final": true} for finalized results

Client logic: if ID exists → replace, if not → append
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
import time
from vosk import Model, KaldiRecognizer, SetLogLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress Vosk internal logs
SetLogLevel(-1)

app = FastAPI(title="Vosk ASR Service", version="3.0")

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
        "backend": "vosk",
        "device": "cpu",
        "sample_rate": SAMPLE_RATE,
        "streaming": True,
        "native_streaming": True,
        "api_version": "3.0"
    }


@app.websocket("/stream")
async def stream_transcribe(websocket: WebSocket):
    """
    Unified streaming endpoint with segment IDs.
    
    Protocol:
    1. Connect to websocket
    2. (Optional) Send JSON config: {"chunk_ms": 200}
    3. Stream raw PCM audio bytes (int16, 16kHz, mono)
    4. Receive JSON responses:
       - {"id": "s1", "text": "...", "is_final": false} - interim (replace by ID)
       - {"id": "s2", "text": "...", "is_final": true} - finalized (new segment)
    
    Client logic: if ID exists → replace text, if not → append new segment
    """
    await websocket.accept()
    logger.info("Stream connection established")
    
    # Config - longer interval for smoother output
    finalize_interval = 2.0  # Force finalize every 2 seconds for smoother output
    min_partial_length = 2   # Minimum words before sending partial
    
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetMaxAlternatives(0)
    recognizer.SetWords(True)  # Enable word-level info for better accuracy
    
    # State
    last_finalize_time = time.time()
    last_partial = ""
    segment_counter = 0
    current_segment_id = "s0"  # ID for current partial
    
    try:
        while True:
            data = await websocket.receive()
            
            # Handle text messages (JSON config)
            if "text" in data:
                try:
                    msg = json.loads(data["text"])
                    if "chunk_ms" in msg:
                        logger.info(f"Config: chunk_ms={msg['chunk_ms']}")
                    if "finalize_interval" in msg:
                        finalize_interval = float(msg["finalize_interval"])
                        logger.info(f"Config: finalize_interval={finalize_interval}")
                    continue
                except json.JSONDecodeError:
                    continue
            
            # Handle binary audio data
            if "bytes" in data:
                audio_data = data["bytes"]
                current_time = time.time()
                
                # Process audio
                is_endpoint = recognizer.AcceptWaveform(audio_data)
                
                if is_endpoint:
                    # Natural finalization (pause detected)
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        # Send final with current segment ID
                        await websocket.send_json({
                            "id": current_segment_id,
                            "text": text
                        })
                        # Move to next segment
                        segment_counter += 1
                        current_segment_id = f"s{segment_counter}"
                        last_partial = ""
                    last_finalize_time = current_time
                else:
                    # Check if we should force finalization
                    elapsed = current_time - last_finalize_time
                    if finalize_interval > 0 and elapsed >= finalize_interval:
                        # Force finalization
                        final = json.loads(recognizer.FinalResult())
                        text = final.get("text", "").strip()
                        if text:
                            await websocket.send_json({
                                "id": current_segment_id,
                                "text": text
                            })
                            # Move to next segment
                            segment_counter += 1
                            current_segment_id = f"s{segment_counter}"
                            last_partial = ""
                        last_finalize_time = current_time
                    else:
                        # Send partial with current segment ID
                        partial = json.loads(recognizer.PartialResult())
                        partial_text = partial.get("partial", "").strip()
                        # Only send if changed and has minimum length
                        if partial_text and partial_text != last_partial:
                            word_count = len(partial_text.split())
                            if word_count >= min_partial_length or elapsed > 0.5:
                                await websocket.send_json({
                                    "id": current_segment_id,
                                    "text": partial_text
                                })
                                last_partial = partial_text
    
    except WebSocketDisconnect:
        logger.info("Stream disconnected")
        # Send any remaining audio
        final = json.loads(recognizer.FinalResult())
        text = final.get("text", "").strip()
        if text:
            try:
                await websocket.send_json({
                    "id": current_segment_id,
                    "text": text
                })
            except:
                pass
    except Exception as e:
        logger.error(f"Stream error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
