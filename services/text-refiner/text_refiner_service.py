"""
Text Refiner Service - Punctuation Restoration and ASR Error Correction

A GPU-accelerated service for post-processing ASR transcriptions.
Combines fast punctuation with optional spelling/grammar correction.

Endpoints:
- GET  /health    - Health check
- GET  /info      - Service information
- POST /process   - Full processing (punctuation + correction)
- POST /punctuate - Punctuation only (fast path)
- POST /correct   - Correction only

Protocol:
1. Client sends: {"text": "...", "punctuate": true, "correct": true}
2. Server returns: {"text": "...", "original": "...", "latency_ms": ...}
"""

import asyncio
import logging
import os
import time

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================

__version__ = "1.0"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_MEMORY_THRESHOLD_GB = 2.0  # Minimum GPU memory for correction model

# Punctuation model (ONNX - CPU efficient)
PUNCTUATION_MODEL = "pcs_en"

# ASR Error Correction model
CORRECTION_MODEL = "oliverguhr/spelling-correction-english-base"
ENABLE_CORRECTION = os.environ.get("ENABLE_CORRECTION", "true").lower() == "true"

# Processing settings
MAX_BATCH_SIZE = 8
CORRECTION_MIN_WORDS = 4


# ==============================================================================
# Model Loading (Lazy)
# ==============================================================================

_punctuation_model = None
_correction_model = None
_correction_tokenizer = None


def check_gpu_memory() -> tuple[bool, float]:
    """Check if GPU has enough free memory for models."""
    if DEVICE != "cuda":
        return True, 0.0

    try:
        free_memory = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        free_gb = (free_memory - allocated) / (1024**3)
        return free_gb >= GPU_MEMORY_THRESHOLD_GB, free_gb
    except Exception as e:
        logger.warning(f"GPU memory check failed: {e}")
        return False, 0.0


def get_punctuation_model():
    """Lazy load punctuation model on first use."""
    global _punctuation_model

    if _punctuation_model is None:
        try:
            from punctuators.models import PunctCapSegModelONNX

            logger.info(f"Loading punctuation model: {PUNCTUATION_MODEL}")
            # PunctCapSegModelONNX doesn't support local_files_only (uses cache by default)
            _punctuation_model = PunctCapSegModelONNX.from_pretrained(PUNCTUATION_MODEL)
            logger.info("Punctuation model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load punctuation model: {e}")
            raise

    return _punctuation_model


def get_correction_model():
    """Lazy load correction model on first use."""
    global _correction_model, _correction_tokenizer

    if not ENABLE_CORRECTION:
        return None, None

    if _correction_model is None:
        # Check GPU memory first
        has_memory, free_gb = check_gpu_memory()
        if DEVICE == "cuda" and not has_memory:
            logger.warning(
                f"Insufficient GPU memory ({free_gb:.1f}GB free, "
                f"need {GPU_MEMORY_THRESHOLD_GB}GB). Using CPU."
            )
            device = "cpu"
        else:
            device = DEVICE

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            logger.info(f"Loading correction model: {CORRECTION_MODEL}")

            # Load from pre-downloaded cache only (no network requests)
            _correction_tokenizer = AutoTokenizer.from_pretrained(
                CORRECTION_MODEL, local_files_only=True
            )
            _correction_model = AutoModelForSeq2SeqLM.from_pretrained(
                CORRECTION_MODEL, local_files_only=True
            )

            if device == "cuda":
                _correction_model = _correction_model.to(device)
                _correction_model = _correction_model.half()  # FP16 for speed

            _correction_model.eval()
            logger.info(f"Correction model loaded on {device}")
        except Exception as e:
            logger.error(f"Failed to load correction model: {e}")
            raise

    return _correction_model, _correction_tokenizer


def get_model_info() -> dict:
    """Get information about loaded models."""
    return {
        "punctuation": {
            "name": PUNCTUATION_MODEL,
            "loaded": _punctuation_model is not None,
        },
        "correction": {
            "name": CORRECTION_MODEL if ENABLE_CORRECTION else "disabled",
            "loaded": _correction_model is not None,
            "enabled": ENABLE_CORRECTION,
        },
    }


# ==============================================================================
# Request/Response Models
# ==============================================================================


class ProcessRequest(BaseModel):
    text: str
    punctuate: bool = True
    correct: bool = True
    context: str | None = None


class ProcessResponse(BaseModel):
    text: str
    original: str
    punctuated: bool
    corrected: bool
    latency_ms: float


class BatchRequest(BaseModel):
    texts: list[str]
    punctuate: bool = True
    correct: bool = True


class BatchResponse(BaseModel):
    texts: list[str]
    latency_ms: float


# ==============================================================================
# Processing Functions
# ==============================================================================


async def apply_punctuation(text: str) -> str:
    """Apply punctuation and capitalization."""
    if not text.strip():
        return text

    try:
        model = get_punctuation_model()
        result = await asyncio.to_thread(model.infer, [text.lower()])

        if result and result[0]:
            return " ".join(result[0])
        return text
    except Exception as e:
        logger.error(f"Punctuation error: {e}")
        return text


def _split_into_chunks(text: str, max_tokens: int = 400) -> list[str]:
    """
    Split text into chunks that fit within token limits.

    Uses sentence boundaries (. ! ?) as natural split points.
    Falls back to word-based splitting if sentences are too long.
    """
    import re

    # Split on sentence boundaries, keeping the punctuation
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        # If a single sentence is too long, split by words
        if sentence_words > max_tokens:
            # Flush current chunk first
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0

            # Split long sentence by words
            words = sentence.split()
            for i in range(0, len(words), max_tokens):
                chunk_words = words[i : i + max_tokens]
                chunks.append(" ".join(chunk_words))
        elif current_word_count + sentence_words > max_tokens:
            # Current chunk is full, start a new one
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_words
        else:
            # Add to current chunk
            current_chunk.append(sentence)
            current_word_count += sentence_words

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks if chunks else [text]


async def apply_correction(text: str, context: str | None = None) -> str:
    """Apply ASR error correction using seq2seq model."""
    if not text.strip():
        return text

    # Skip very short texts
    word_count = len(text.split())
    if word_count < CORRECTION_MIN_WORDS:
        return text

    try:
        model, tokenizer = get_correction_model()
        if model is None or tokenizer is None:
            return text

        # Split long texts into manageable chunks (400 words ≈ ~500 tokens with headroom)
        chunks = _split_into_chunks(text, max_tokens=400)

        def correct_chunk(chunk_text: str, use_context: bool = False) -> str:
            """Correct a single chunk."""
            input_text = f"{context} {chunk_text}" if (use_context and context) else chunk_text
            device = next(model.parameters()).device
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=2,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                )

            return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        def correct_all():
            """Process all chunks sequentially on the model's device."""
            corrected_chunks = []
            for i, chunk in enumerate(chunks):
                # Only use context for the first chunk
                corrected = correct_chunk(chunk, use_context=(i == 0))
                corrected_chunks.append(corrected)
            return " ".join(corrected_chunks)

        if len(chunks) > 1:
            logger.info(f"Correction: splitting {word_count} words into {len(chunks)} chunks")

        return await asyncio.to_thread(correct_all)
    except Exception as e:
        logger.error(f"Correction error: {e}")
        return text


async def process_text(
    text: str,
    punctuate: bool = True,
    correct: bool = True,
    context: str | None = None,
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
    if punctuate:
        try:
            result = await apply_punctuation(result)
            was_punctuated = True
        except Exception:
            pass

    # Step 2: Correction (slower)
    if correct and ENABLE_CORRECTION:
        try:
            result = await apply_correction(result, context)
            was_corrected = True
        except Exception:
            pass

    return result, was_punctuated, was_corrected


# ==============================================================================
# FastAPI Application
# ==============================================================================

app = FastAPI(
    title="Text Refiner Service",
    description="Punctuation restoration and ASR error correction",
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
    models = get_model_info()
    return {
        "status": "healthy",
        "punctuation_model": PUNCTUATION_MODEL if models["punctuation"]["loaded"] else None,
        "correction_model": CORRECTION_MODEL if models["correction"]["loaded"] else None,
        "correction_enabled": ENABLE_CORRECTION,
        "device": DEVICE,
        "version": __version__,
    }


@app.get("/info")
async def info():
    """Service information endpoint."""
    gpu_info = None
    if DEVICE == "cuda":
        try:
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "memory_total_gb": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}",
                "memory_allocated_gb": f"{torch.cuda.memory_allocated() / 1024**3:.2f}",
            }
        except Exception:
            pass

    return {
        "service": "text-refiner",
        "api_version": __version__,
        "models": get_model_info(),
        "device": DEVICE,
        "gpu": gpu_info,
        "settings": {
            "max_batch_size": MAX_BATCH_SIZE,
            "correction_min_words": CORRECTION_MIN_WORDS,
            "gpu_memory_threshold_gb": GPU_MEMORY_THRESHOLD_GB,
        },
    }


@app.post("/process", response_model=ProcessResponse)
async def process_endpoint(request: ProcessRequest):
    """Full processing: punctuation + correction."""
    start = time.perf_counter()

    result, was_punctuated, was_corrected = await process_text(
        request.text,
        punctuate=request.punctuate,
        correct=request.correct,
        context=request.context,
    )

    latency = (time.perf_counter() - start) * 1000

    return ProcessResponse(
        text=result,
        original=request.text,
        punctuated=was_punctuated,
        corrected=was_corrected,
        latency_ms=round(latency, 2),
    )


@app.post("/punctuate")
async def punctuate_endpoint(request: ProcessRequest):
    """Fast path: punctuation only."""
    start = time.perf_counter()

    result = await apply_punctuation(request.text)
    latency = (time.perf_counter() - start) * 1000

    return {"text": result, "original": request.text, "latency_ms": round(latency, 2)}


@app.post("/correct")
async def correct_endpoint(request: ProcessRequest):
    """Correction only (assumes already punctuated)."""
    if not ENABLE_CORRECTION:
        raise HTTPException(status_code=503, detail="Correction model not enabled")

    start = time.perf_counter()

    result = await apply_correction(request.text, request.context)
    latency = (time.perf_counter() - start) * 1000

    return {"text": result, "original": request.text, "latency_ms": round(latency, 2)}


@app.post("/batch", response_model=BatchResponse)
async def batch_endpoint(request: BatchRequest):
    """Batch processing for multiple texts."""
    start = time.perf_counter()

    tasks = [
        process_text(text, request.punctuate, request.correct)
        for text in request.texts[:MAX_BATCH_SIZE]
    ]
    results = await asyncio.gather(*tasks)

    latency = (time.perf_counter() - start) * 1000

    return BatchResponse(texts=[r[0] for r in results], latency_ms=round(latency, 2))


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
