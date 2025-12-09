"""
Vosk ASR Service - Lightweight CPU-based Streaming Speech Recognition

A simple, fast, offline-capable ASR service optimized for real-time streaming.

Endpoints:
- GET  /health  - Health check
- WS   /stream  - Streaming transcription

Protocol:
1. Client connects to /stream
2. Client streams raw PCM audio (int16, 16kHz, mono)
3. Server sends: {"id": "s0", "text": "...", "is_final": false/true}
"""

import json
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from vosk import KaldiRecognizer, Model, SetLogLevel

from shared.text_refiner import get_client, refine_text
from shared.utils import setup_logging

# Configure logging
logger = setup_logging(__name__)

# Suppress Vosk internal logs
SetLogLevel(-1)

# ==============================================================================
# Configuration
# ==============================================================================

MODEL_PATH = "/app/model"  # Standard Docker path
SAMPLE_RATE = 16000
__version__ = "1.0"


# ==============================================================================
# Model Loading (Lazy)
# ==============================================================================

_model: Model | None = None


def get_model() -> Model:
    """Lazy load the Vosk model on first use."""
    global _model
    if _model is None:
        logger.info(f"Loading Vosk model from {MODEL_PATH}...")
        try:
            _model = Model(MODEL_PATH)
            logger.info(f"Vosk model '{get_model_name()}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Vosk model: {e}")
            raise
    return _model


def get_model_name() -> str:
    """Get the model name from .model_name file or return default."""
    model_name_file = os.path.join(MODEL_PATH, ".model_name")
    try:
        with open(model_name_file) as f:
            return f.read().strip()
    except FileNotFoundError:
        return "vosk-model-en-us"  # Default model name


# ==============================================================================
# FastAPI Application
# ==============================================================================

app = FastAPI(
    title="Vosk ASR Service",
    description="Lightweight CPU-based streaming speech recognition",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    text_refiner = get_client()
    return {
        "status": "healthy",
        "backend": "vosk",
        "model": get_model_name(),
        "device": "cpu",
        "sample_rate": SAMPLE_RATE,
        "streaming": True,
        "version": __version__,
        "api_version": "1.0",
        "text_refiner": {
            "enabled": text_refiner.enabled,
            "available": text_refiner.available,
            "url": text_refiner.url,
        },
    }


@app.websocket("/stream")
async def stream_transcribe(websocket: WebSocket):
    """
    Streaming transcription endpoint.

    Protocol:
    1. Connect to WebSocket
    2. Stream raw PCM audio bytes (int16, 16kHz, mono)
    3. Receive JSON: {"id": "s0", "text": "hello world", "is_final": true/false}
    """
    await websocket.accept()
    logger.info("Stream connection established")

    # Create recognizer (lazy loads model on first connection)
    # Endpointer timing is configured in model conf/model.conf
    recognizer = KaldiRecognizer(get_model(), SAMPLE_RATE)
    recognizer.SetMaxAlternatives(0)
    recognizer.SetWords(True)
    recognizer.SetPartialWords(True)  # Word timings in partials for accuracy

    # Segment tracking
    segment_counter = 0
    current_segment_id = "s0"
    last_partial_text = ""

    try:
        while True:
            data = await websocket.receive()

            # Only handle binary audio data
            if "bytes" not in data:
                continue

            audio_data = data["bytes"]

            # Process audio through Vosk
            if recognizer.AcceptWaveform(audio_data):
                # Natural endpoint (pause detected)
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    # Add punctuation and correction to final results
                    text = await refine_text(text, punctuate=True, correct=True)
                    await websocket.send_json(
                        {
                            "id": current_segment_id,
                            "text": text,
                            "is_final": True,
                        }
                    )
                    # Move to next segment
                    segment_counter += 1
                    current_segment_id = f"s{segment_counter}"
                    last_partial_text = ""
            else:
                # Partial result
                partial = json.loads(recognizer.PartialResult())
                text = partial.get("partial", "").strip()
                if text and text != last_partial_text:
                    await websocket.send_json(
                        {
                            "id": current_segment_id,
                            "text": text,
                            "is_final": False,
                        }
                    )
                    last_partial_text = text

    except WebSocketDisconnect:
        logger.info("Stream disconnected")
        # Flush remaining audio
        try:
            final = json.loads(recognizer.FinalResult())
            text = final.get("text", "").strip()
            if text:
                # Add punctuation and correction to final results
                text = await refine_text(text, punctuate=True, correct=True)
                await websocket.send_json(
                    {
                        "id": current_segment_id,
                        "text": text,
                        "is_final": True,
                    }
                )
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Stream error: {e}")


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
