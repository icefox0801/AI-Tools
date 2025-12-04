"""
Transcription Gateway Service

Unified API for all transcription needs - routes to appropriate model backends.
Manages GPU memory, model loading/unloading, and provides consistent API surface.

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                    Transcription Gateway                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │  /stream        │  │  /transcribe    │  │  /models/*          │ │
│  │  (Real-time)    │  │  (Offline)      │  │  (Management)       │ │
│  └────────┬────────┘  └────────┬────────┘  └─────────┬───────────┘ │
│           │                    │                     │             │
│           ▼                    ▼                     ▼             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Model Router                              │   │
│  │   - Routes to streaming model (TDT) for /stream             │   │
│  │   - Routes to offline model (RNNT/Whisper) for /transcribe  │   │
│  │   - Manages model loading/unloading based on usage          │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌────────────┐      ┌────────────┐      ┌────────────┐
   │  Parakeet  │      │  Whisper   │      │   Vosk     │
   │  (GPU)     │      │  (GPU)     │      │   (CPU)    │
   └────────────┘      └────────────┘      └────────────┘

API Endpoints:
- GET  /health            - Service health check
- GET  /info              - Service information and available models
- WS   /stream            - Real-time streaming transcription
- POST /transcribe        - Offline file transcription
- GET  /models            - List available models and their status
- POST /models/{id}/load  - Load a specific model
- POST /models/{id}/unload - Unload a model to free GPU memory
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ==============================================================================
# Configuration
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Model backend URLs (Docker service names)
PARAKEET_URL = os.getenv("PARAKEET_URL", "http://parakeet-asr:8000")
WHISPER_URL = os.getenv("WHISPER_URL", "http://whisper-asr:8000")
VOSK_URL = os.getenv("VOSK_URL", "http://vosk-asr:8000")

# Default model preferences
DEFAULT_STREAMING_MODEL = os.getenv("DEFAULT_STREAMING_MODEL", "parakeet-streaming")
DEFAULT_OFFLINE_MODEL = os.getenv("DEFAULT_OFFLINE_MODEL", "parakeet-offline")

# GPU memory management
GPU_MEMORY_THRESHOLD_MB = int(os.getenv("GPU_MEMORY_THRESHOLD_MB", "8000"))
IDLE_UNLOAD_SECONDS = int(os.getenv("IDLE_UNLOAD_SECONDS", "300"))  # 5 minutes

# ==============================================================================
# Data Models
# ==============================================================================

class ModelType(str, Enum):
    STREAMING = "streaming"
    OFFLINE = "offline"
    BOTH = "both"


class ModelStatus(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


class ModelInfo(BaseModel):
    """Information about a model backend."""
    id: str
    name: str
    backend: str  # parakeet, whisper, vosk
    type: ModelType
    status: ModelStatus = ModelStatus.UNLOADED
    device: str = "GPU"
    description: str = ""
    last_used: Optional[datetime] = None
    memory_mb: int = 0


class TranscriptionRequest(BaseModel):
    """Request for offline transcription."""
    model: Optional[str] = None  # Use default if not specified
    language: Optional[str] = None
    return_timestamps: bool = True


class TranscriptionSegment(BaseModel):
    """A segment of transcribed text with timing."""
    id: str
    text: str
    start: float
    end: float
    words: Optional[List[Dict[str, Any]]] = None


class TranscriptionResponse(BaseModel):
    """Response from transcription."""
    text: str
    segments: List[TranscriptionSegment] = []
    duration: float = 0.0
    model: str = ""
    language: Optional[str] = None


class StreamConfig(BaseModel):
    """Configuration for streaming session."""
    model: Optional[str] = None
    chunk_ms: int = 300
    language: Optional[str] = None


# ==============================================================================
# Model Registry
# ==============================================================================

@dataclass
class ModelBackend:
    """Backend model configuration and state."""
    id: str
    name: str
    backend: str
    url: str
    type: ModelType
    device: str = "GPU"
    description: str = ""
    status: ModelStatus = ModelStatus.UNLOADED
    last_used: Optional[float] = None
    memory_mb: int = 0
    ws_endpoint: str = "/stream"
    transcribe_endpoint: str = "/transcribe"
    
    def to_info(self) -> ModelInfo:
        """Convert to API response model."""
        return ModelInfo(
            id=self.id,
            name=self.name,
            backend=self.backend,
            type=self.type,
            status=self.status,
            device=self.device,
            description=self.description,
            last_used=datetime.fromtimestamp(self.last_used) if self.last_used else None,
            memory_mb=self.memory_mb
        )


class ModelRegistry:
    """Registry of available model backends."""
    
    def __init__(self):
        self.models: Dict[str, ModelBackend] = {}
        self._init_models()
    
    def _init_models(self):
        """Initialize available models."""
        # Parakeet streaming (TDT)
        self.models["parakeet-streaming"] = ModelBackend(
            id="parakeet-streaming",
            name="Parakeet TDT (Streaming)",
            backend="parakeet",
            url=PARAKEET_URL,
            type=ModelType.STREAMING,
            device="GPU",
            description="NVIDIA NeMo TDT - optimized for real-time streaming",
            memory_mb=4000,
            ws_endpoint="/stream",
        )
        
        # Parakeet offline (RNNT)
        self.models["parakeet-offline"] = ModelBackend(
            id="parakeet-offline",
            name="Parakeet RNNT (Offline)",
            backend="parakeet",
            url=PARAKEET_URL,
            type=ModelType.OFFLINE,
            device="GPU",
            description="NVIDIA NeMo RNNT - highest accuracy for offline transcription",
            memory_mb=8000,
            transcribe_endpoint="/transcribe",
        )
        
        # Whisper (both streaming and offline)
        self.models["whisper"] = ModelBackend(
            id="whisper",
            name="Whisper Large V3 Turbo",
            backend="whisper",
            url=WHISPER_URL,
            type=ModelType.BOTH,
            device="GPU",
            description="OpenAI Whisper - fast multilingual ASR",
            memory_mb=3000,
            ws_endpoint="/stream",
            transcribe_endpoint="/transcribe",
        )
        
        # Vosk (CPU, streaming only)
        self.models["vosk"] = ModelBackend(
            id="vosk",
            name="Vosk (CPU)",
            backend="vosk",
            url=VOSK_URL,
            type=ModelType.STREAMING,
            device="CPU",
            description="Lightweight CPU-based streaming ASR",
            memory_mb=500,
            ws_endpoint="/stream",
        )
    
    def get(self, model_id: str) -> Optional[ModelBackend]:
        """Get a model by ID."""
        return self.models.get(model_id)
    
    def get_streaming_model(self, preferred: Optional[str] = None) -> ModelBackend:
        """Get the best streaming model."""
        if preferred and preferred in self.models:
            model = self.models[preferred]
            if model.type in (ModelType.STREAMING, ModelType.BOTH):
                return model
        
        # Use default
        return self.models.get(DEFAULT_STREAMING_MODEL, self.models["parakeet-streaming"])
    
    def get_offline_model(self, preferred: Optional[str] = None) -> ModelBackend:
        """Get the best offline model."""
        if preferred and preferred in self.models:
            model = self.models[preferred]
            if model.type in (ModelType.OFFLINE, ModelType.BOTH):
                return model
        
        # Use default
        return self.models.get(DEFAULT_OFFLINE_MODEL, self.models["parakeet-offline"])
    
    def list_all(self) -> List[ModelInfo]:
        """List all available models."""
        return [m.to_info() for m in self.models.values()]
    
    def update_status(self, model_id: str, status: ModelStatus):
        """Update model status."""
        if model_id in self.models:
            self.models[model_id].status = status
            if status == ModelStatus.LOADED:
                self.models[model_id].last_used = time.time()


# Global registry
model_registry = ModelRegistry()


# ==============================================================================
# GPU Memory Manager
# ==============================================================================

class GPUMemoryManager:
    """Manages GPU memory by tracking model usage and unloading idle models."""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self._check_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the background memory management task."""
        self._check_task = asyncio.create_task(self._memory_check_loop())
        logger.info("GPU memory manager started")
    
    async def stop(self):
        """Stop the memory management task."""
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("GPU memory manager stopped")
    
    async def _memory_check_loop(self):
        """Periodically check for idle models to unload."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._check_idle_models()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory check error: {e}")
    
    async def _check_idle_models(self):
        """Check and unload idle models."""
        now = time.time()
        
        for model in self.registry.models.values():
            if model.status != ModelStatus.LOADED:
                continue
            
            if model.last_used and (now - model.last_used) > IDLE_UNLOAD_SECONDS:
                logger.info(f"Unloading idle model: {model.id}")
                await self.unload_model(model.id)
    
    async def load_model(self, model_id: str) -> bool:
        """Request a model backend to load its model."""
        model = self.registry.get(model_id)
        if not model:
            return False
        
        self.registry.update_status(model_id, ModelStatus.LOADING)
        
        try:
            # For now, just check health - models load on first request
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(f"{model.url}/health")
                if resp.status_code == 200:
                    self.registry.update_status(model_id, ModelStatus.LOADED)
                    return True
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            self.registry.update_status(model_id, ModelStatus.ERROR)
        
        return False
    
    async def unload_model(self, model_id: str) -> bool:
        """Request a model backend to unload its model."""
        model = self.registry.get(model_id)
        if not model:
            return False
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{model.url}/unload")
                if resp.status_code == 200:
                    self.registry.update_status(model_id, ModelStatus.UNLOADED)
                    return True
        except Exception as e:
            logger.error(f"Failed to unload model {model_id}: {e}")
        
        return False
    
    def mark_used(self, model_id: str):
        """Mark a model as recently used."""
        if model_id in self.registry.models:
            self.registry.models[model_id].last_used = time.time()
            self.registry.models[model_id].status = ModelStatus.LOADED


# Global memory manager
memory_manager: Optional[GPUMemoryManager] = None


# ==============================================================================
# FastAPI Application
# ==============================================================================

app = FastAPI(
    title="Transcription Gateway",
    description="Unified API for speech transcription - routes to optimal model backends",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Initialize service on startup."""
    global memory_manager
    memory_manager = GPUMemoryManager(model_registry)
    await memory_manager.start()
    logger.info("Transcription Gateway started")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    if memory_manager:
        await memory_manager.stop()
    logger.info("Transcription Gateway stopped")


# ==============================================================================
# Health & Info Endpoints
# ==============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "transcription-gateway"}


@app.get("/info")
async def info():
    """Service information."""
    return {
        "service": "Transcription Gateway",
        "version": "1.0.0",
        "description": "Unified API for speech transcription",
        "default_streaming_model": DEFAULT_STREAMING_MODEL,
        "default_offline_model": DEFAULT_OFFLINE_MODEL,
        "models": model_registry.list_all()
    }


# ==============================================================================
# Model Management Endpoints
# ==============================================================================

@app.get("/models")
async def list_models():
    """List all available models and their status."""
    # Update status from backends
    await _refresh_model_status()
    return {"models": model_registry.list_all()}


@app.post("/models/{model_id}/load")
async def load_model(model_id: str):
    """Load a specific model."""
    model = model_registry.get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    
    success = await memory_manager.load_model(model_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to load model")
    
    return {"status": "loaded", "model": model.to_info()}


@app.post("/models/{model_id}/unload")
async def unload_model(model_id: str):
    """Unload a model to free GPU memory."""
    model = model_registry.get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    
    success = await memory_manager.unload_model(model_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to unload model")
    
    return {"status": "unloaded", "model": model.to_info()}


async def _refresh_model_status():
    """Refresh model status from backends."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        for model in model_registry.models.values():
            try:
                resp = await client.get(f"{model.url}/health")
                if resp.status_code == 200:
                    model.status = ModelStatus.LOADED
                else:
                    model.status = ModelStatus.ERROR
            except:
                model.status = ModelStatus.UNLOADED


# ==============================================================================
# Streaming Transcription (Real-time)
# ==============================================================================

@app.websocket("/stream")
async def stream_transcribe(websocket: WebSocket):
    """
    Real-time streaming transcription via WebSocket.
    
    Protocol:
    1. Connect to WebSocket
    2. Send config JSON: {"model": "parakeet-streaming", "chunk_ms": 300}
    3. Stream raw PCM audio bytes (int16, 16kHz, mono)
    4. Receive JSON: {"partial": "..."} or {"text": "...", "final": true}
    5. Close connection when done
    """
    await websocket.accept()
    logger.info("Streaming client connected")
    
    # Get config from first message
    try:
        config_data = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        config = json.loads(config_data)
    except asyncio.TimeoutError:
        config = {}
    except json.JSONDecodeError:
        config = {}
    
    # Select streaming model
    preferred_model = config.get("model")
    model = model_registry.get_streaming_model(preferred_model)
    logger.info(f"Using streaming model: {model.id}")
    
    # Connect to backend and proxy
    import websockets
    backend_ws = None
    
    try:
        backend_uri = f"ws://{model.url.replace('http://', '')}{model.ws_endpoint}"
        backend_ws = await websockets.connect(backend_uri)
        
        # Forward config to backend
        await backend_ws.send(json.dumps(config))
        memory_manager.mark_used(model.id)
        
        # Bidirectional proxy
        async def forward_to_backend():
            try:
                while True:
                    data = await websocket.receive()
                    if "bytes" in data:
                        await backend_ws.send(data["bytes"])
                    elif "text" in data:
                        await backend_ws.send(data["text"])
            except WebSocketDisconnect:
                pass
        
        async def forward_to_client():
            try:
                async for message in backend_ws:
                    if isinstance(message, str):
                        await websocket.send_text(message)
                    else:
                        await websocket.send_bytes(message)
            except:
                pass
        
        # Run both directions concurrently
        await asyncio.gather(
            forward_to_backend(),
            forward_to_client(),
            return_exceptions=True
        )
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        if backend_ws:
            await backend_ws.close()
        try:
            await websocket.close()
        except:
            pass
        logger.info("Streaming client disconnected")


# ==============================================================================
# Offline Transcription (File-based)
# ==============================================================================

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_file(
    file: UploadFile = File(...),
    model: Optional[str] = None,
    language: Optional[str] = None,
    return_timestamps: bool = True
):
    """
    Transcribe an audio file.
    
    Args:
        file: Audio file (WAV, MP3, etc.)
        model: Model to use (default: parakeet-offline or whisper)
        language: Language code (auto-detect if not specified)
        return_timestamps: Whether to return word/segment timestamps
    
    Returns:
        TranscriptionResponse with text and optional timestamps
    """
    # Select offline model
    backend = model_registry.get_offline_model(model)
    logger.info(f"Using offline model: {backend.id}")
    
    # Read file content
    content = await file.read()
    
    # Forward to backend
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            files = {"file": (file.filename, content, file.content_type)}
            params = {}
            if language:
                params["language"] = language
            if return_timestamps:
                params["return_timestamps"] = "true"
            
            resp = await client.post(
                f"{backend.url}{backend.transcribe_endpoint}",
                files=files,
                params=params
            )
            
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Backend error: {resp.text}"
                )
            
            memory_manager.mark_used(backend.id)
            result = resp.json()
            
            # Normalize response
            return TranscriptionResponse(
                text=result.get("text", ""),
                segments=[
                    TranscriptionSegment(**s) for s in result.get("segments", [])
                ],
                duration=result.get("duration", 0.0),
                model=backend.id,
                language=result.get("language")
            )
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Transcription timeout")
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# Backend Status Check
# ==============================================================================

@app.get("/backends/status")
async def backend_status():
    """Check the status of all backend services."""
    status = {}
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in [("parakeet", PARAKEET_URL), ("whisper", WHISPER_URL), ("vosk", VOSK_URL)]:
            try:
                resp = await client.get(f"{url}/health")
                if resp.status_code == 200:
                    data = resp.json()
                    status[name] = {
                        "status": "healthy",
                        "url": url,
                        "info": data
                    }
                else:
                    status[name] = {"status": "unhealthy", "url": url}
            except Exception as e:
                status[name] = {"status": "unavailable", "url": url, "error": str(e)}
    
    return status


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
