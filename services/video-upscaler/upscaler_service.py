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

import base64
import json
import os
import shutil
import time
import uuid

import cv2
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from jobs import Job, JobManager
from log_setup import setup_logging
from pipeline import enhance_video_ffmpeg, upscale_video
from upscaler_model import DEFAULT_MODEL, MODELS, list_models

_FFMPEG_MODELS = {"ffmpeg-enhance"}

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


def _save_job_meta(job_dir: str, job: "Job") -> None:
    """Atomically write job metadata to <job_dir>/job.json."""
    try:
        path = os.path.join(job_dir, "job.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(job.to_dict(), f)
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning("Could not save job metadata: %s", exc)


def _on_job_finish(job: "Job") -> None:
    """Persist final job state to disk after the worker finishes."""
    job_dir = os.path.dirname(job.output_path)
    if os.path.isdir(job_dir):
        _save_job_meta(job_dir, job)


def _process(job, progress_cb, cancel_cb) -> dict:
    """Adapter between JobManager and the upscaling pipeline."""
    job_dir = os.path.dirname(job.output_path)
    preview_video_dir = os.path.join(job_dir, "preview_frames")
    os.makedirs(preview_video_dir, exist_ok=True)
    preview_video_path = os.path.join(job_dir, "preview_video.mp4")

    kwargs = {
        "progress_cb": progress_cb,
        "cancel_cb": cancel_cb,
        "preview_path": os.path.join(job_dir, "preview.jpg"),
        "preview_video_dir": preview_video_dir,
        "preview_video_path": preview_video_path,
    }
    if job.model == "ffmpeg-enhance":
        return enhance_video_ffmpeg(
            input_path=job.input_path,
            output_path=job.output_path,
            **kwargs,
        )
    return upscale_video(
        input_path=job.input_path,
        output_path=job.output_path,
        model_name=job.model,
        outscale=job.outscale,
        denoise=job.denoise,
        tile=job.tile,
        temporal_mode=job.temporal_mode,
        **kwargs,
    )


manager = JobManager(processor=_process, on_finish=_on_job_finish)

app = FastAPI(title="Video Upscaler Service", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _restore_disk_jobs() -> None:
    """On startup, reload completed jobs from disk so history survives restarts."""
    if not os.path.isdir(OUTPUT_DIR):
        return
    count = 0
    for token in os.listdir(OUTPUT_DIR):
        meta_path = os.path.join(OUTPUT_DIR, token, "job.json")
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path) as f:
                data = json.load(f)
            # Mark jobs that were still running when the service died as errors.
            if data.get("status") in ("queued", "processing"):
                data["status"] = "error"
                data["error"] = "Interrupted (service restarted)"
                data.setdefault("finished_at", time.time())
            job = Job(**{k: v for k, v in data.items() if k in Job.__dataclass_fields__})
            with manager._lock:
                if job.id not in manager._jobs:
                    manager._jobs[job.id] = job
                    count += 1
        except Exception as exc:
            logger.warning("Could not restore job from %s: %s", meta_path, exc)
    if count:
        logger.info("Restored %d historical job(s) from disk", count)


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
    ffmpeg_entry = [
        {
            "name": "ffmpeg-enhance",
            "netscale": 1,
            "supports_denoise": False,
            "description": "FFmpeg only (hqdn3d + unsharp) — no GPU, real-time speed",
        }
    ]
    return {"models": list_models() + ffmpeg_entry, "default": DEFAULT_MODEL}


@app.post("/upscale")
async def upscale(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    outscale: float = Form(4.0),
    denoise: float = Form(1.0),
    tile: int = Form(DEFAULT_TILE),
    temporal_mode: str = Form("standard"),
):
    """Accept a video and queue it for upscaling. Returns a job id immediately."""
    all_valid = set(MODELS) | _FFMPEG_MODELS
    if model not in all_valid:
        raise HTTPException(400, f"Unknown model '{model}'. Available: {sorted(all_valid)}")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXT)}")

    if not 1.0 <= outscale <= 4.0:
        raise HTTPException(400, "outscale must be between 1.0 and 4.0")
    if not 0.0 <= denoise <= 1.0:
        raise HTTPException(400, "denoise must be between 0.0 and 1.0")
    if temporal_mode not in ("standard", "tmix", "basicvsr"):
        raise HTTPException(400, f"Unknown temporal_mode '{temporal_mode}'")

    token = uuid.uuid4().hex[:12]
    safe_stem = os.path.splitext(os.path.basename(file.filename or "video"))[0]
    job_dir = os.path.join(OUTPUT_DIR, token)
    os.makedirs(job_dir, exist_ok=True)
    input_path = os.path.join(INPUT_DIR, f"{token}{ext}")
    output_path = os.path.join(job_dir, f"{safe_stem}_x{outscale:g}.mp4")

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
        temporal_mode=temporal_mode,
    )
    # Persist initial state so the job survives a service restart.
    _save_job_meta(job_dir, job)
    logger.info(
        "Queued job %s (%s, model=%s, x%s, mode=%s)",
        job.id,
        file.filename,
        model,
        outscale,
        temporal_mode,
    )
    return {"job_id": job.id, "status": job.status}


@app.get("/jobs/discover")
async def discover_jobs():
    """Scan the output directory and return all jobs found on disk (history)."""
    if not os.path.isdir(OUTPUT_DIR):
        return {"jobs": []}
    tokens = sorted(
        os.listdir(OUTPUT_DIR),
        key=lambda t: os.path.getmtime(os.path.join(OUTPUT_DIR, t)),
        reverse=True,
    )
    result = []
    for token in tokens:
        meta_path = os.path.join(OUTPUT_DIR, token, "job.json")
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path) as f:
                data = json.load(f)
            # Merge with live in-memory state if available.
            live = manager.get(data.get("id", ""))
            if live:
                data = live.to_dict()
            result.append(data)
        except Exception as exc:
            logger.warning("Error reading %s: %s", meta_path, exc)
    return {"jobs": result}


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
    ppath = os.path.join(os.path.dirname(job.output_path), "preview.jpg")
    if not os.path.exists(ppath):
        raise HTTPException(404, "No preview available yet")
    return FileResponse(ppath, media_type="image/jpeg")


@app.get("/jobs/{job_id}/preview-video")
async def preview_video(job_id: str):
    """Return a rolling 3-5 second preview video of the most recent frames."""
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    pvpath = os.path.join(os.path.dirname(job.output_path), "preview_video.mp4")
    if not os.path.exists(pvpath):
        raise HTTPException(404, "No preview video available yet")
    return FileResponse(pvpath, media_type="video/mp4")


@app.get("/jobs/{job_id}/comparison-frames")
async def comparison_frames(job_id: str):
    """List available frame indices for before/after comparison."""
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    preview_dir = os.path.join(os.path.dirname(job.output_path), "preview_frames")
    if not os.path.exists(preview_dir):
        return {"frames": []}
    orig_files = sorted(
        [f for f in os.listdir(preview_dir) if f.startswith("orig_")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )
    frame_indices = [int(f.split("_")[1].split(".")[0]) for f in orig_files]
    return {"frames": frame_indices}


@app.get("/jobs/{job_id}/comparison/{frame_idx}")
async def comparison(job_id: str, frame_idx: int):
    """Return original and upscaled images for a specific frame."""
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    preview_dir = os.path.join(os.path.dirname(job.output_path), "preview_frames")
    orig_path = os.path.join(preview_dir, f"orig_{frame_idx:06d}.jpg")
    upscaled_path = os.path.join(preview_dir, f"upscaled_{frame_idx:06d}.jpg")
    if not os.path.exists(orig_path) or not os.path.exists(upscaled_path):
        raise HTTPException(404, "Comparison not available for this frame")

    def _encode(path: str) -> str:
        """Read a JPEG, downscale to max 640px wide, return base64 data-URI."""
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            if w > 640:
                scale = 640.0 / w
                img = cv2.resize(img, (640, round(h * scale)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                return base64.b64encode(buf.tobytes()).decode()
        # fallback: return raw file bytes as-is
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    return {
        "frame_idx": frame_idx,
        "original": f"data:image/jpeg;base64,{_encode(orig_path)}",
        "upscaled": f"data:image/jpeg;base64,{_encode(upscaled_path)}",
    }


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
