"""
Video Upscaler Service - Real-ESRGAN GPU video super-resolution (FastAPI).

Increases video clarity / resolution. Designed for offline batch
processing: submit a video, poll for progress, download the result.

Endpoints:
  GET  /health                 - service + GPU status
  GET  /info                   - configuration
  GET  /models                 - available upscaling models
  POST /upscale                - upload a video, returns {job_id}
  GET  /jobs                    - list jobs
  GET  /jobs/{job_id}          - job status + progress
  GET  /jobs/{job_id}/preview  - latest upscaled frame (live preview JPEG)
  GET  /jobs/{job_id}/download - download the upscaled video
  DELETE /jobs/{job_id}        - cancel a queued/running job
"""

import os
import shutil
import uuid

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from jobs import JobManager
from pipeline import upscale_video
from log_setup import setup_logging
from upscaler_model import DEFAULT_MODEL, MODELS, list_models

logger = setup_logging(__name__)

__version__ = "1.0"

DEVICE = os.environ.get("DEVICE", "cuda")
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
INPUT_DIR = os.path.join(DATA_DIR, "inputs")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
DEFAULT_TILE = int(os.environ.get("UPSCALER_TILE", "512"))

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv"}


def _process(job, progress_cb, cancel_cb) -> dict:
    """Adapter between JobManager and the upscaling pipeline."""
    return upscale_video(
        input_path=job.input_path,
        output_path=job.output_path,
        model_name=job.model,
        outscale=job.outscale,
        denoise=job.denoise,
        tile=job.tile,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
        preview_path=job.output_path + ".preview.jpg",
    )


manager = JobManager(processor=_process)

app = FastAPI(title="Video Upscaler Service", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    gpu_info = {}
    cuda_available = False
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_info = {
                "cuda_device": torch.cuda.get_device_name(0),
                "memory_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            }
    except Exception as exc:  # pragma: no cover
        logger.warning("Torch/GPU probe failed: %s", exc)

    return {
        "status": "healthy",
        "device": DEVICE,
        "cuda_available": cuda_available,
        "models": list(MODELS),
        **gpu_info,
    }


@app.get("/info")
async def info():
    return {
        "service": "video-upscaler",
        "version": __version__,
        "device": DEVICE,
        "default_model": DEFAULT_MODEL,
        "default_tile": DEFAULT_TILE,
        "data_dir": DATA_DIR,
        "allowed_extensions": sorted(ALLOWED_EXT),
    }


@app.get("/models")
async def models():
    return {"models": list_models(), "default": DEFAULT_MODEL}


@app.post("/upscale")
async def upscale(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    outscale: float = Form(4.0),
    denoise: float = Form(1.0),
    tile: int = Form(DEFAULT_TILE),
):
    """Accept a video and queue it for upscaling. Returns a job id immediately."""
    if model not in MODELS:
        raise HTTPException(400, f"Unknown model '{model}'. Available: {list(MODELS)}")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXT)}")

    if not 1.0 <= outscale <= 4.0:
        raise HTTPException(400, "outscale must be between 1.0 and 4.0")
    if not 0.0 <= denoise <= 1.0:
        raise HTTPException(400, "denoise must be between 0.0 and 1.0")

    token = uuid.uuid4().hex[:12]
    safe_stem = os.path.splitext(os.path.basename(file.filename or "video"))[0]
    input_path = os.path.join(INPUT_DIR, f"{token}{ext}")
    output_path = os.path.join(OUTPUT_DIR, f"{safe_stem}_x{outscale:g}_{token}.mp4")

    try:
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    job = manager.submit(
        filename=file.filename or f"video{ext}",
        input_path=input_path,
        output_path=output_path,
        model=model,
        outscale=outscale,
        denoise=denoise,
        tile=tile if tile > 0 else None,
    )
    logger.info("Queued job %s (%s, model=%s, x%s)", job.id, file.filename, model, outscale)
    return {"job_id": job.id, "status": job.status}


@app.get("/jobs")
async def jobs():
    return {"jobs": manager.list()}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


@app.get("/jobs/{job_id}/preview")
async def preview(job_id: str):
    """Return a small JPEG of the latest upscaled frame (live preview)."""
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    ppath = job.output_path + ".preview.jpg"
    if not os.path.exists(ppath):
        raise HTTPException(404, "No preview available yet")
    return FileResponse(ppath, media_type="image/jpeg")


@app.get("/jobs/{job_id}/download")
async def download(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "done":
        raise HTTPException(409, f"Job not finished (status={job.status})")
    if not os.path.exists(job.output_path):
        raise HTTPException(410, "Output file no longer available")
    return FileResponse(
        job.output_path,
        media_type="video/mp4",
        filename=os.path.basename(job.output_path),
    )


@app.delete("/jobs/{job_id}")
async def cancel(job_id: str):
    if not manager.get(job_id):
        raise HTTPException(404, "Job not found")
    cancelled = manager.cancel(job_id)
    return {"job_id": job_id, "cancelled": cancelled}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
