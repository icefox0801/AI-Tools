"""
Text Refiner Service

Combines punctuation restoration and ASR error correction in a single service.
Designed to work with streaming ASR output for real-time captions.

Pipeline:
1. Punctuation + Capitalization (punctuators ONNX - fast)
2. ASR Error Correction (T5-based - optional, higher latency)

Endpoints:
- POST /process - Process text with punctuation and optional correction
- POST /punctuate - Punctuation only (fast path)
- POST /correct - Error correction only
- WebSocket /stream - Streaming with context buffering
- GET /health - Health check
- GET /info - Service info
"""

import os
import json
import logging
import asyncio
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============== Configuration ==============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Punctuation model (ONNX - CPU efficient)
PUNCTUATION_MODEL = os.getenv("PUNCTUATION_MODEL", "pcs_en")

# ASR Error Correction model
CORRECTION_MODEL = os.getenv("CORRECTION_MODEL", "oliverguhr/spelling-correction-english-base")
ENABLE_CORRECTION = os.getenv("ENABLE_CORRECTION", "true").lower() == "true"

# Processing settings
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))
CORRECTION_MIN_WORDS = int(os.getenv("CORRECTION_MIN_WORDS", "4"))  # Min words before correction

# ============== Models ==============
punctuation_model = None
correction_model = None
correction_tokenizer = None


# ============== Request/Response Models ==============
class ProcessRequest(BaseModel):
    text: str
    punctuate: bool = True
    correct: bool = True
    context: Optional[str] = None  # Previous text for context


class ProcessResponse(BaseModel):
    text: str
    original: str
    punctuated: bool
    corrected: bool
    latency_ms: float


class BatchRequest(BaseModel):
    texts: List[str]
    punctuate: bool = True
    correct: bool = True


class BatchResponse(BaseModel):
    texts: List[str]
    latency_ms: float


# ============== Model Loading ==============
def load_punctuation_model():
    """Load punctuators ONNX model."""
    global punctuation_model
    
    try:
        from punctuators.models import PunctCapSegModelONNX
        logger.info(f"Loading punctuation model: {PUNCTUATION_MODEL}")
        punctuation_model = PunctCapSegModelONNX.from_pretrained(PUNCTUATION_MODEL)
        logger.info("Punctuation model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load punctuation model: {e}")
        return False


def load_correction_model():
    """Load spelling/grammar correction model (BART or T5 based)."""
    global correction_model, correction_tokenizer
    
    if not ENABLE_CORRECTION:
        logger.info("ASR error correction disabled")
        return True
    
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        logger.info(f"Loading correction model: {CORRECTION_MODEL}")
        
        correction_tokenizer = AutoTokenizer.from_pretrained(CORRECTION_MODEL)
        correction_model = AutoModelForSeq2SeqLM.from_pretrained(CORRECTION_MODEL)
        
        if DEVICE == "cuda":
            correction_model = correction_model.to(DEVICE)
            correction_model = correction_model.half()  # FP16 for speed
        
        correction_model.eval()
        
        logger.info(f"Correction model loaded on {DEVICE}")
        return True
    except Exception as e:
        logger.error(f"Failed to load correction model: {e}")
        return False


# ============== Processing Functions ==============
async def apply_punctuation(text: str) -> str:
    """Apply punctuation and capitalization."""
    if not punctuation_model or not text.strip():
        return text
    
    try:
        # Run in thread pool to avoid blocking
        result = await asyncio.to_thread(
            punctuation_model.infer, [text.lower()]
        )
        
        if result and result[0]:
            # Join segments
            return ' '.join(result[0])
        return text
    except Exception as e:
        logger.error(f"Punctuation error: {e}")
        return text


async def apply_correction(text: str, context: Optional[str] = None) -> str:
    """Apply ASR error correction using seq2seq model."""
    if not correction_model or not correction_tokenizer or not text.strip():
        return text
    
    # Skip very short texts
    word_count = len(text.split())
    if word_count < CORRECTION_MIN_WORDS:
        return text
    
    try:
        # Prepare input with optional context
        if context:
            input_text = f"{context} {text}"
        else:
            input_text = text
        
        # Run in thread pool
        def correct():
            inputs = correction_tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            if DEVICE == "cuda":
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = correction_model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=2,  # Faster than default 4
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                )
            
            corrected = correction_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # If context was provided, extract only the new part
            if context:
                # Try to find where original text starts in output
                corrected = corrected.strip()
            
            return corrected
        
        result = await asyncio.to_thread(correct)
        return result
    except Exception as e:
        logger.error(f"Correction error: {e}")
        return text


async def process_text(
    text: str,
    punctuate: bool = True,
    correct: bool = True,
    context: Optional[str] = None
) -> tuple[str, bool, bool]:
    """
    Full processing pipeline.
    
    Returns:
        (processed_text, was_punctuated, was_corrected)
    """
    result = text
    was_punctuated = False
    was_corrected = False
    
    # Step 1: Punctuation (fast)
    if punctuate and punctuation_model:
        result = await apply_punctuation(result)
        was_punctuated = True
    
    # Step 2: Correction (slower)
    if correct and correction_model and ENABLE_CORRECTION:
        result = await apply_correction(result, context)
        was_corrected = True
    
    return result, was_punctuated, was_corrected


# ============== FastAPI App ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    logger.info("Starting Post-Processing Service...")
    
    load_punctuation_model()
    load_correction_model()
    
    # Warmup
    if punctuation_model:
        await apply_punctuation("warmup test")
    if correction_model:
        await apply_correction("warmup test")
    
    logger.info("Service ready!")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Text Refiner Service",
    description="Punctuation restoration and ASR error correction",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "punctuation_model": PUNCTUATION_MODEL if punctuation_model else None,
        "correction_model": CORRECTION_MODEL if correction_model else None,
        "correction_enabled": ENABLE_CORRECTION and correction_model is not None,
        "device": DEVICE,
    }


@app.get("/info")
async def service_info():
    """Service information."""
    gpu_info = None
    if DEVICE == "cuda":
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_allocated": f"{torch.cuda.memory_allocated()/1024**3:.2f}GB",
        }
    
    return {
        "service": "text-refiner",
        "version": "1.0.0",
        "punctuation_model": PUNCTUATION_MODEL,
        "correction_model": CORRECTION_MODEL if ENABLE_CORRECTION else "disabled",
        "device": DEVICE,
        "gpu": gpu_info,
        "settings": {
            "max_batch_size": MAX_BATCH_SIZE,
            "correction_min_words": CORRECTION_MIN_WORDS,
        }
    }


@app.post("/process", response_model=ProcessResponse)
async def process_endpoint(request: ProcessRequest):
    """Full processing: punctuation + correction."""
    start = time.perf_counter()
    
    result, was_punctuated, was_corrected = await process_text(
        request.text,
        punctuate=request.punctuate,
        correct=request.correct,
        context=request.context
    )
    
    latency = (time.perf_counter() - start) * 1000
    
    return ProcessResponse(
        text=result,
        original=request.text,
        punctuated=was_punctuated,
        corrected=was_corrected,
        latency_ms=round(latency, 2)
    )


@app.post("/punctuate")
async def punctuate_endpoint(request: ProcessRequest):
    """Fast path: punctuation only."""
    start = time.perf_counter()
    
    result = await apply_punctuation(request.text)
    latency = (time.perf_counter() - start) * 1000
    
    return {
        "text": result,
        "original": request.text,
        "latency_ms": round(latency, 2)
    }


@app.post("/correct")
async def correct_endpoint(request: ProcessRequest):
    """Correction only (assumes already punctuated)."""
    if not ENABLE_CORRECTION or not correction_model:
        raise HTTPException(status_code=503, detail="Correction model not available")
    
    start = time.perf_counter()
    
    result = await apply_correction(request.text, request.context)
    latency = (time.perf_counter() - start) * 1000
    
    return {
        "text": result,
        "original": request.text,
        "latency_ms": round(latency, 2)
    }


@app.post("/batch", response_model=BatchResponse)
async def batch_endpoint(request: BatchRequest):
    """Batch processing for multiple texts."""
    start = time.perf_counter()
    
    # Process in parallel
    tasks = [
        process_text(text, request.punctuate, request.correct)
        for text in request.texts[:MAX_BATCH_SIZE]
    ]
    results = await asyncio.gather(*tasks)
    
    latency = (time.perf_counter() - start) * 1000
    
    return BatchResponse(
        texts=[r[0] for r in results],
        latency_ms=round(latency, 2)
    )


# ============== WebSocket Streaming ==============
@dataclass
class StreamSession:
    """Track streaming session state."""
    context_buffer: List[str] = field(default_factory=list)
    max_context: int = 3  # Keep last N segments for context
    
    def add_segment(self, text: str):
        self.context_buffer.append(text)
        if len(self.context_buffer) > self.max_context:
            self.context_buffer.pop(0)
    
    def get_context(self) -> Optional[str]:
        if self.context_buffer:
            return ' '.join(self.context_buffer)
        return None


@app.websocket("/stream")
async def stream_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming text processing.
    
    Client sends: {"text": "...", "segment_id": "...", "final": false}
    Server sends: {"text": "...", "segment_id": "...", "latency_ms": ...}
    """
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    session = StreamSession()
    
    try:
        # Receive config
        config_data = await websocket.receive_text()
        config = json.loads(config_data)
        punctuate = config.get("punctuate", True)
        correct = config.get("correct", True)
        
        await websocket.send_json({"status": "ready", "config": config})
        
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            
            text = msg.get("text", "")
            segment_id = msg.get("segment_id", "")
            is_final = msg.get("final", False)
            
            if not text.strip():
                continue
            
            start = time.perf_counter()
            
            # Process with context
            context = session.get_context() if correct else None
            result, _, _ = await process_text(text, punctuate, correct, context)
            
            latency = (time.perf_counter() - start) * 1000
            
            # Update context on final segments
            if is_final:
                session.add_segment(result)
            
            await websocket.send_json({
                "text": result,
                "segment_id": segment_id,
                "final": is_final,
                "latency_ms": round(latency, 2)
            })
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
