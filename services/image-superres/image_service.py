"""Image Super Resolution API service."""

import os
import uuid

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from image_generative import upscale_image_generative
from image_model import (
    DEFAULT_MODEL,
    MODELS,
    get_model_meta,
    get_upsampler,
    is_generative_model,
    list_models,
)
from log_setup import setup_logging

logger = setup_logging(__name__)

__version__ = "1.0"

DEVICE = os.environ.get("DEVICE", "cuda")
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
DEFAULT_TILE = int(os.environ.get("UPSCALER_TILE", "512"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

app = FastAPI(title="Image SuperRes Service", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    cuda_available = False
    gpu_info = {}
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_info = {
                "cuda_device": torch.cuda.get_device_name(0),
                "memory_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            }
    except Exception as exc:
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
        "service": "image-superres",
        "version": __version__,
        "device": DEVICE,
        "default_model": DEFAULT_MODEL,
        "default_tile": DEFAULT_TILE,
        "data_dir": DATA_DIR,
        "supir_local_only": os.environ.get("SUPIR_LOCAL_ONLY", "true"),
        "supir_base_model_id": os.environ.get(
            "SUPIR_BASE_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0"
        ),
        "allowed_extensions": sorted(ALLOWED_EXT),
    }


@app.get("/models")
async def models():
    return {"models": list_models(), "default": DEFAULT_MODEL}


@app.post("/upscale")
async def upscale_image(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    outscale: float = Form(4.0),
    denoise: float = Form(1.0),
    tile: int = Form(DEFAULT_TILE),
    prompt: str = Form("high quality, detailed"),
    negative_prompt: str = Form("text artifacts, blurry text, watermark, logo distortion"),
    steps: int = Form(12),
    guidance_scale: float = Form(4.0),
    seed: int = Form(-1),
):
    if model not in MODELS:
        raise HTTPException(400, f"Unknown model '{model}'. Available: {sorted(MODELS)}")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXT)}")

    if not 1.0 <= outscale <= 4.0:
        raise HTTPException(400, "outscale must be between 1.0 and 4.0")
    if not 0.0 <= denoise <= 1.0:
        raise HTTPException(400, "denoise must be between 0.0 and 1.0")

    payload = await file.read()
    await file.close()

    arr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(400, "Could not decode image")

    meta = get_model_meta(model)
    model_netscale = float(meta.get("netscale", 4))
    if outscale <= 1.0:
        outscale = model_netscale

    if is_generative_model(model):
        # Map denoise 0.0-1.0 to strength 0.1-0.9 (keep some structure)
        strength = 0.1 + denoise * 0.8
        out_img = upscale_image_generative(
            bgr_image=arr,
            model_name=model,
            outscale=outscale,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed if seed >= 0 else None,
            denoise=strength,
        )
    else:
        upsampler, _ = get_upsampler(model, denoise=denoise, tile=tile if tile > 0 else None)
        out_img, _ = upsampler.enhance(arr, outscale=outscale)

    token = uuid.uuid4().hex[:12]
    output_path = os.path.join(OUTPUT_DIR, f"{token}.png")
    ok = cv2.imwrite(output_path, out_img)
    if not ok:
        raise HTTPException(500, "Failed to write output image")

    return FileResponse(output_path, media_type="image/png", filename="upscaled.png")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
