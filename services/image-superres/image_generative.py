"""Generative image upscaling helpers (diffusers-based)."""

import os

import cv2
import numpy as np
from PIL import Image

from image_model import get_model_meta
from log_setup import setup_logging

logger = setup_logging(__name__)

DEVICE = os.environ.get("DEVICE", "cuda")
GAI_MAX_INPUT_SIDE = int(os.environ.get("GAI_MAX_INPUT_SIDE", "1024"))
GAI_CPU_OFFLOAD = os.environ.get("GAI_CPU_OFFLOAD", "true").lower() == "true"

_pipe_cache: dict[str, object] = {}


def _load_pipe(model_name: str):
    meta = get_model_meta(model_name)
    model_id = meta["hf_model_id"]

    if model_name in _pipe_cache:
        return _pipe_cache[model_name]

    import torch

    torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    if model_name == "stable-diffusion-x4-upscaler":
        from diffusers import StableDiffusionUpscalePipeline

        pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    elif model_name == "sd-x2-latent-upscaler":
        from diffusers import StableDiffusionLatentUpscalePipeline

        pipe = StableDiffusionLatentUpscalePipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        )
    else:
        raise ValueError(f"Unsupported generative model: {model_name}")

    if DEVICE == "cuda":
        if GAI_CPU_OFFLOAD:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    _pipe_cache[model_name] = pipe
    logger.info("Loaded generative upscaler pipeline: %s", model_name)
    return pipe


def upscale_image_generative(
    bgr_image: np.ndarray,
    model_name: str,
    outscale: float,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    seed: int | None,
) -> np.ndarray:
    import torch

    pipe = _load_pipe(model_name)

    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    src_h, src_w = rgb.shape[:2]
    work_rgb = rgb
    if max(src_w, src_h) > GAI_MAX_INPUT_SIDE:
        scale = float(GAI_MAX_INPUT_SIDE) / float(max(src_w, src_h))
        work_w = max(64, int(round(src_w * scale)))
        work_h = max(64, int(round(src_h * scale)))
        work_rgb = cv2.resize(rgb, (work_w, work_h), interpolation=cv2.INTER_AREA)
        logger.warning(
            "GAI input resized from %dx%d to %dx%d to reduce VRAM usage",
            src_w,
            src_h,
            work_w,
            work_h,
        )

    input_image = Image.fromarray(work_rgb)

    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device="cuda" if DEVICE == "cuda" else "cpu").manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            image=input_image,
            negative_prompt=negative_prompt,
            num_inference_steps=max(5, min(30, int(steps))),
            guidance_scale=max(1.0, min(9.0, float(guidance_scale))),
            generator=generator,
        ).images[0]

    if DEVICE == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    out_rgb = np.array(result.convert("RGB"))

    target_w = max(1, round(src_w * outscale))
    target_h = max(1, round(src_h * outscale))
    if out_rgb.shape[1] != target_w or out_rgb.shape[0] != target_h:
        out_rgb = cv2.resize(out_rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
