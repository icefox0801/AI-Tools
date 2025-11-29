"""
Vosk ASR Service - Unified Streaming API v3.2
Lightweight, fast, offline speech recognition with native streaming support

Features:
- Low-latency streaming with immediate partial results
- Post-processing via Text Refiner service (punctuation + correction)
- Segment-based ID protocol for clean UI updates

Unified Protocol:
1. Client connects to /stream
2. (Optional) Client sends JSON config: {"chunk_ms": 200}
3. Client streams raw PCM audio (int16, 16kHz, mono)
4. Server sends: {"id": "s1", "text": "..."} for all results

Client logic: if ID exists → replace, if not → append
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
import time
import os
import asyncio
from vosk import Model, KaldiRecognizer, SetLogLevel

# Import shared text refiner module
from shared.text_refiner import get_client, refine_text, capitalize_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress Vosk internal logs
SetLogLevel(-1)

app = FastAPI(title="Vosk ASR Service", version="3.2")

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

# Get text refiner client for config info
text_refiner = get_client()


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
        "text_refiner_enabled": text_refiner.enabled,
        "text_refiner_url": text_refiner.url if text_refiner.enabled else None,
        "api_version": "3.2"
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
       - {"id": "s1", "text": "..."} - update segment (replace by ID)
    
    Client logic: if ID exists → replace text, if not → append new segment
    """
    await websocket.accept()
    logger.info("Stream connection established (v3.2 - text refiner)")
    
    # Config - optimized for quality punctuation
    finalize_interval = 3.0  # Force finalize every 3s (longer chunks for better punctuation)
    partial_interval = 0.15  # Send partial updates every 150ms
    min_partial_words = 1    # Send even single words
    min_words_for_punctuation = 6  # Buffer at least 6 words before running punctuation
    
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetMaxAlternatives(0)
    recognizer.SetWords(True)
    
    # State
    last_finalize_time = time.time()
    last_partial_time = time.time()
    last_partial = ""
    segment_counter = 0
    current_segment_id = "s0"
    text_buffer = []  # Buffer for accumulating text before punctuation
    
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
                        text_buffer.append(text)
                        buffered_text = ' '.join(text_buffer)
                        word_count = len(buffered_text.split())
                        
                        # Only run punctuation if we have enough words
                        if word_count >= min_words_for_punctuation:
                            # Apply text refinement (punctuation + correction)
                            punctuated = await refine_text(buffered_text)
                            await websocket.send_json({
                                "id": current_segment_id,
                                "text": punctuated
                            })
                            # Move to next segment, clear buffer
                            segment_counter += 1
                            current_segment_id = f"s{segment_counter}"
                            text_buffer = []
                            last_partial = ""
                        else:
                            # Not enough words yet, just show capitalized buffer
                            await websocket.send_json({
                                "id": current_segment_id,
                                "text": capitalize_text(buffered_text)
                            })
                    last_finalize_time = current_time
                    last_partial_time = current_time
                else:
                    elapsed_since_finalize = current_time - last_finalize_time
                    elapsed_since_partial = current_time - last_partial_time
                    
                    # Check if we should force finalization (longer pause = flush buffer)
                    if finalize_interval > 0 and elapsed_since_finalize >= finalize_interval:
                        final = json.loads(recognizer.FinalResult())
                        text = final.get("text", "").strip()
                        if text:
                            text_buffer.append(text)
                        
                        # Flush buffer with punctuation (regardless of word count)
                        if text_buffer:
                            buffered_text = ' '.join(text_buffer)
                            punctuated = await refine_text(buffered_text)
                            await websocket.send_json({
                                "id": current_segment_id,
                                "text": punctuated
                            })
                            segment_counter += 1
                            current_segment_id = f"s{segment_counter}"
                            text_buffer = []
                            last_partial = ""
                        last_finalize_time = current_time
                        last_partial_time = current_time
                    
                    # Send partial updates frequently for low latency
                    elif elapsed_since_partial >= partial_interval:
                        partial = json.loads(recognizer.PartialResult())
                        partial_text = partial.get("partial", "").strip()
                        
                        if partial_text and partial_text != last_partial:
                            word_count = len(partial_text.split())
                            if word_count >= min_partial_words:
                                # Show buffer + current partial (just capitalize, no punctuation)
                                buffer_prefix = ' '.join(text_buffer) + ' ' if text_buffer else ''
                                display_text = capitalize_text(buffer_prefix + partial_text)
                                
                                await websocket.send_json({
                                    "id": current_segment_id,
                                    "text": display_text
                                })
                                last_partial = partial_text
                                last_partial_time = current_time
    
    except WebSocketDisconnect:
        logger.info("Stream disconnected")
        # Flush any remaining audio and buffer
        final = json.loads(recognizer.FinalResult())
        text = final.get("text", "").strip()
        if text:
            text_buffer.append(text)
        if text_buffer:
            try:
                buffered_text = ' '.join(text_buffer)
                # Use sync fallback on disconnect (can't await)
                punctuated = capitalize_text(buffered_text) + '.'
                await websocket.send_json({
                    "id": current_segment_id,
                    "text": punctuated
                })
            except:
                pass
    except Exception as e:
        logger.error(f"Stream error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
