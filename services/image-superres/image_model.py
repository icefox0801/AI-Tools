"""Real-ESRGAN model registry and loader for image super-resolution."""

import os

from log_setup import setup_logging

logger = setup_logging(__name__)

WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/app/weights")
DEVICE = os.environ.get("DEVICE", "cuda")
DEFAULT_TILE = int(os.environ.get("UPSCALER_TILE", "512"))
USE_HALF = os.environ.get("UPSCALER_HALF", "true").lower() == "true"

_RELEASE = "https://github.com/xinntao/Real-ESRGAN/releases/download"

MODELS = {
    "RealESRGAN_x4plus": {
        "arch": "rrdb",
        "netscale": 4,
        "num_block": 23,
        "files": ["RealESRGAN_x4plus.pth"],
        "urls": [f"{_RELEASE}/v0.1.0/RealESRGAN_x4plus.pth"],
        "supports_denoise": False,
        "class": "restoration",
        "description": "Best detail reconstruction model.",
    },
    "realesr-general-x4v3": {
        "arch": "srvgg",
        "netscale": 4,
        "num_block": 32,
        "files": ["realesr-general-x4v3.pth", "realesr-general-wdn-x4v3.pth"],
        "urls": [
            f"{_RELEASE}/v0.2.5.0/realesr-general-x4v3.pth",
            f"{_RELEASE}/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        ],
        "supports_denoise": True,
        "class": "restoration",
        "description": "Balanced model with adjustable denoise strength.",
    },
    "stable-diffusion-x4-upscaler": {
        "arch": "diffusion",
        "netscale": 4,
        "supports_denoise": False,
        "class": "generative",
        "hf_model_id": "stabilityai/stable-diffusion-x4-upscaler",
        "description": "GAI creative x4 upscaler (can hallucinate details/text).",
    },
    "sd-x2-latent-upscaler": {
        "arch": "diffusion",
        "netscale": 2,
        "supports_denoise": False,
        "class": "generative",
        "hf_model_id": "stabilityai/sd-x2-latent-upscaler",
        "description": "GAI creative x2 latent upscaler (lower hallucination than x4).",
    },
}

DEFAULT_MODEL = "RealESRGAN_x4plus"

_current = {"key": None, "upsampler": None}


def _build_arch(meta: dict):
    if meta["arch"] == "srvgg":
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact

        return SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=meta["num_block"],
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


def list_models() -> list[dict]:
    return [
        {
            "name": name,
            "netscale": meta["netscale"],
            "supports_denoise": meta["supports_denoise"],
            "class": meta.get("class", "restoration"),
            "description": meta["description"],
        }
        for name, meta in MODELS.items()
    ]


def get_model_meta(model_name: str) -> dict:
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODELS)}")
    return MODELS[model_name]


def is_generative_model(model_name: str) -> bool:
    return get_model_meta(model_name).get("class") == "generative"


def get_upsampler(model_name: str, denoise: float = 1.0, tile: int | None = None):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODELS)}")

    meta = MODELS[model_name]
    if meta.get("arch") == "diffusion":
        raise ValueError(
            f"Model '{model_name}' is generative; use the generative pipeline path instead."
        )
    tile = DEFAULT_TILE if tile is None else tile

    dni_weight = None
    model_path = os.path.join(WEIGHTS_DIR, meta["files"][0])
    if meta["supports_denoise"] and denoise < 1.0:
        wdn_path = os.path.join(WEIGHTS_DIR, meta["files"][1])
        model_path = [model_path, wdn_path]
        dni_weight = [denoise, 1.0 - denoise]

    cache_key = (model_name, round(denoise, 3), tile)
    if _current["key"] == cache_key and _current["upsampler"] is not None:
        return _current["upsampler"], meta

    from realesrgan import RealESRGANer

    logger.info(
        "Loading image upsampler: model=%s, denoise=%s, tile=%s, half=%s",
        model_name,
        denoise,
        tile,
        USE_HALF,
    )

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
