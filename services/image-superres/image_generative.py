"""Generative image upscaling helpers (diffusers-based)."""

import glob
import os

import cv2
import numpy as np
from PIL import Image

from image_model import get_model_meta
from log_setup import setup_logging

logger = setup_logging(__name__)

DEVICE = os.environ.get("DEVICE", "cuda")
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/app/weights")
GAI_MAX_INPUT_SIDE = int(os.environ.get("GAI_MAX_INPUT_SIDE", "1024"))
GAI_CPU_OFFLOAD = os.environ.get("GAI_CPU_OFFLOAD", "true").lower() == "true"
SUPIR_LOCAL_ONLY = os.environ.get("SUPIR_LOCAL_ONLY", "true").lower() == "true"
SUPIR_BASE_MODEL_ID = os.environ.get(
    "SUPIR_BASE_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0"
)

_pipe_cache: dict[str, object] = {}


def _find_supir_ckpt() -> str:
    explicit = os.environ.get("SUPIR_CKPT", "").strip()
    if explicit and os.path.exists(explicit):
        return explicit

    candidates = [
        os.path.join(WEIGHTS_DIR, "SUPIR-v0Q.ckpt"),
        os.path.join(WEIGHTS_DIR, "SUPIR-v0Q.safetensors"),
        os.path.join(
            WEIGHTS_DIR,
            "huggingface",
            "hub",
            "models--camenduru--SUPIR",
            "snapshots",
            "97b24ec4d42bbdf6b3c5a8d701f78ac67aacba04",
            "SUPIR-v0Q.ckpt",
        ),
    ]

    patterns = [
        os.path.join(
            WEIGHTS_DIR,
            "huggingface",
            "hub",
            "models--camenduru--SUPIR",
            "snapshots",
            "*",
            "SUPIR-v0Q.ckpt",
        ),
        os.path.join(
            WEIGHTS_DIR,
            "huggingface",
            "hub",
            "models--Kijai--SUPIR_pruned",
            "snapshots",
            "*",
            "SUPIR-v0Q*.safetensors",
        ),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "SUPIR checkpoint not found. Set SUPIR_CKPT or place SUPIR-v0Q under /app/weights."
    )


def _load_supir_model():
    """Load Kijai SUPIR-pruned adapter onto SDXL base (component-by-component to
    avoid SIGSEGV on PyTorch 2.9 + CUDA 12.9)."""
    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline, UNet2DConditionModel, AutoencoderKL
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    model_id = SUPIR_BASE_MODEL_ID
    dtype = torch.float16

    logger.info("Loading SDXL base components...")
    te = CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=dtype,
        variant="fp16",
        local_files_only=SUPIR_LOCAL_ONLY,
    )
    te2 = CLIPTextModelWithProjection.from_pretrained(
        model_id,
        subfolder="text_encoder_2",
        torch_dtype=dtype,
        variant="fp16",
        local_files_only=SUPIR_LOCAL_ONLY,
    )
    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=dtype,
        variant="fp16",
        local_files_only=SUPIR_LOCAL_ONLY,
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=dtype,
        variant="fp16",
        local_files_only=SUPIR_LOCAL_ONLY,
    )
    tok = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer",
        local_files_only=SUPIR_LOCAL_ONLY,
    )
    tok2 = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer_2",
        local_files_only=SUPIR_LOCAL_ONLY,
    )

    # Find Kijai SUPIR adapter checkpoint
    import glob

    candidates = sorted(
        glob.glob(
            "/app/weights/huggingface/hub/models--Kijai--SUPIR_pruned/snapshots/*/SUPIR-v0F*.safetensors"
        )
    )
    if not candidates:
        raise FileNotFoundError("Kijai SUPIR_pruned checkpoint not found")
    adapter_path = candidates[0]
    logger.info("Loading SUPIR adapter from %s", adapter_path)

    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
        adapter_path,
        torch_dtype=dtype,
        text_encoder=te,
        text_encoder_2=te2,
        tokenizer=tok,
        tokenizer_2=tok2,
        vae=vae,
        unet=unet,
        local_files_only=True,
    )

    if DEVICE == "cuda":
        if GAI_CPU_OFFLOAD:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass

    logger.info("SUPIR model loaded on %s", DEVICE)
    return pipe


def _load_supir_pipe(torch_dtype):
    """Compatibility wrapper — returns the SUPIR model in _pipe_cache."""
    return _load_supir_model()


def _load_pipe(model_name: str):
    meta = get_model_meta(model_name)
    model_id = meta.get("hf_model_id")

    if model_name in _pipe_cache:
        return _pipe_cache[model_name]

    import torch

    torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    if model_name == "SUPIR-v0Q":
        pipe = _load_supir_pipe(torch_dtype)
    elif model_name == "sdxl-img2img":
        from diffusers import StableDiffusionXLImg2ImgPipeline

        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            variant="fp16",
            local_files_only=SUPIR_LOCAL_ONLY,
        )
        if DEVICE == "cuda":
            if GAI_CPU_OFFLOAD:
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass
    else:
        raise ValueError(f"Unsupported generative model: {model_name} (hf_model_id={model_id})")

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
    denoise: float = 0.4,
) -> np.ndarray:
    import torch

    pipe = _load_pipe(model_name)

    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    src_h, src_w = rgb.shape[:2]

    if model_name == "SUPIR-v0Q":
        # SUPIR uses same img2img pipeline as sdxl-img2img (adapter weights
        # are loaded onto the base SDXL UNet by _load_supir_model).
        pass  # fall through to generic path below

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
    # Pad to multiple of 64 for SD latent upscaler compatibility
    h, w = work_rgb.shape[:2]
    pad_h = (64 - h % 64) % 64
    pad_w = (64 - w % 64) % 64
    if pad_h or pad_w:
        work_rgb = cv2.copyMakeBorder(work_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    input_image = Image.fromarray(work_rgb)

    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device="cuda" if DEVICE == "cuda" else "cpu").manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            image=input_image,
            negative_prompt=negative_prompt,
            strength=denoise,
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

    # Crop off padding added for SD compatibility
    if pad_h or pad_w:
        out_h, out_w = out_rgb.shape[:2]
        crop_h = int(out_h * (h / (h + pad_h))) if pad_h else out_h
        crop_w = int(out_w * (w / (w + pad_w))) if pad_w else out_w
        out_rgb = out_rgb[:crop_h, :crop_w]

    target_w = max(1, round(src_w * outscale))
    target_h = max(1, round(src_h * outscale))
    if out_rgb.shape[1] != target_w or out_rgb.shape[0] != target_h:
        out_rgb = cv2.resize(out_rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
