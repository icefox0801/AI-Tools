"""
Real-ESRGAN model registry and loader for the Video Upscaler service.

Each model entry describes the network architecture, its native scale, and the
weight file(s) needed. Models are loaded lazily and cached so the GPU only holds
one model at a time (videos are processed sequentially).

Supported models:
  - realesr-general-x4v3      General-purpose v3 (fast, supports denoise strength)
  - RealESRGAN_x4plus         General 4x, highest detail (heavier)
  - RealESRGAN_x4plus_anime_6B Anime/animation 4x (lighter)
  - RealESRGAN_x2plus         General 2x
"""

import os

from log_setup import setup_logging

logger = setup_logging(__name__)

WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/app/weights")
DEVICE = os.environ.get("DEVICE", "cuda")
DEFAULT_TILE = int(os.environ.get("UPSCALER_TILE", "512"))
USE_HALF = os.environ.get("UPSCALER_HALF", "true").lower() == "true"

# GitHub release base for Real-ESRGAN weights
_RELEASE = "https://github.com/xinntao/Real-ESRGAN/releases/download"

# Model registry: name -> metadata
MODELS = {
    "realesr-general-x4v3": {
        "arch": "srvgg",
        "netscale": 4,
        "files": ["realesr-general-x4v3.pth", "realesr-general-wdn-x4v3.pth"],
        "urls": [
            f"{_RELEASE}/v0.2.5.0/realesr-general-x4v3.pth",
            f"{_RELEASE}/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        ],
        "supports_denoise": True,
        "description": "General v3 - fast, adjustable denoise. Best default for real video.",
    },
    "RealESRGAN_x4plus": {
        "arch": "rrdb",
        "netscale": 4,
        "num_block": 23,
        "files": ["RealESRGAN_x4plus.pth"],
        "urls": [f"{_RELEASE}/v0.1.0/RealESRGAN_x4plus.pth"],
        "supports_denoise": False,
        "description": "General 4x - highest detail, slower/heavier.",
    },
    "RealESRGAN_x4plus_anime_6B": {
        "arch": "rrdb",
        "netscale": 4,
        "num_block": 6,
        "files": ["RealESRGAN_x4plus_anime_6B.pth"],
        "urls": [f"{_RELEASE}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"],
        "supports_denoise": False,
        "description": "Anime/animation 4x - lighter, optimized for drawn content.",
    },
    "RealESRGAN_x2plus": {
        "arch": "rrdb",
        "netscale": 2,
        "num_block": 23,
        "files": ["RealESRGAN_x2plus.pth"],
        "urls": [f"{_RELEASE}/v0.2.1/RealESRGAN_x2plus.pth"],
        "supports_denoise": False,
        "description": "General 2x native model.",
    },
}

DEFAULT_MODEL = "RealESRGAN_x4plus"

# Simple in-process cache: only one upsampler held at a time
_current = {"key": None, "upsampler": None}


def list_models() -> list[dict]:
    """Return a JSON-serializable list of available models."""
    return [
        {
            "name": name,
            "netscale": meta["netscale"],
            "supports_denoise": meta["supports_denoise"],
            "description": meta["description"],
        }
        for name, meta in MODELS.items()
    ]


def _build_arch(meta: dict):
    """Instantiate the network architecture for a model."""
    if meta["arch"] == "srvgg":
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact

        return SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=meta["netscale"],
            act_type="prelu",
        )

    from basicsr.archs.rrdbnet_arch import RRDBNet

    return RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=meta["num_block"],
        num_grow_ch=32,
        scale=meta["netscale"],
    )


def get_upsampler(model_name: str, denoise: float = 1.0, tile: int | None = None):
    """
    Build (or reuse) a RealESRGANer for the given model.

    Args:
        model_name: key in MODELS.
        denoise: 0.0-1.0 denoise strength (only used by models that support it).
                 1.0 = full denoise, 0.0 = keep noise/detail.
        tile: tile size for limited VRAM (0 disables tiling). Defaults to env UPSCALER_TILE.

    Returns:
        (RealESRGANer, metadata dict)
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODELS)}")

    meta = MODELS[model_name]
    tile = DEFAULT_TILE if tile is None else tile

    # dni (Deep Network Interpolation) weight for adjustable denoise on general-v3
    dni_weight = None
    model_path = os.path.join(WEIGHTS_DIR, meta["files"][0])

    if meta["supports_denoise"] and denoise < 1.0:
        wdn_path = os.path.join(WEIGHTS_DIR, meta["files"][1])
        model_path = [model_path, wdn_path]
        dni_weight = [denoise, 1.0 - denoise]

    cache_key = (model_name, round(denoise, 3), tile)
    if _current["key"] == cache_key and _current["upsampler"] is not None:
        return _current["upsampler"], meta

    logger.info(
        f"Loading upsampler: model={model_name}, denoise={denoise}, tile={tile}, half={USE_HALF}"
    )

    from realesrgan import RealESRGANer

    upsampler = RealESRGANer(
        scale=meta["netscale"],
        model_path=model_path,
        dni_weight=dni_weight,
        model=_build_arch(meta),
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=USE_HALF and DEVICE == "cuda",
        device=DEVICE,
    )

    _current["key"] = cache_key
    _current["upsampler"] = upsampler
    return upsampler, meta


def unload():
    """Release the cached upsampler and free GPU memory."""
    _current["key"] = None
    _current["upsampler"] = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - best effort cleanup
        pass
