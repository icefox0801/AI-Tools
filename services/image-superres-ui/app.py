#!/usr/bin/env python3
"""Gradio UI for image super-resolution service."""

import io

import gradio as gr
import httpx
from PIL import Image

from config import IMAGE_SR_URL, SERVER_NAME, SERVER_PORT, logger

__version__ = "1.0"


GAI_MODELS = {"stable-diffusion-x4-upscaler", "sd-x2-latent-upscaler"}
PRESET_CHOICES = [
    "💎 Best Fidelity",
    "🧹 Balanced Denoise",
    "🎨 GAI Creative x2",
    "🧠 GAI Creative x4",
]


def _preset_for_model(model_name: str) -> str:
    if model_name == "sd-x2-latent-upscaler":
        return "🎨 GAI Creative x2"
    if model_name == "stable-diffusion-x4-upscaler":
        return "🧠 GAI Creative x4"
    if model_name == "realesr-general-x4v3":
        return "🧹 Balanced Denoise"
    return "💎 Best Fidelity"


def _model_for_preset(preset_name: str) -> str:
    if preset_name == "🎨 GAI Creative x2":
        return "sd-x2-latent-upscaler"
    if preset_name == "🧠 GAI Creative x4":
        return "stable-diffusion-x4-upscaler"
    if preset_name == "🧹 Balanced Denoise":
        return "realesr-general-x4v3"
    return "RealESRGAN_x4plus"


def fetch_models() -> tuple[list[str], str]:
    try:
        r = httpx.get(f"{IMAGE_SR_URL}/models", timeout=10)
        r.raise_for_status()
        data = r.json()
        models = [m["name"] for m in data.get("models", [])]
        default = data.get("default") or (models[0] if models else "RealESRGAN_x4plus")
        return models, default
    except Exception as exc:
        logger.warning("Could not fetch models: %s", exc)
        fallback = [
            "RealESRGAN_x4plus",
            "realesr-general-x4v3",
            "stable-diffusion-x4-upscaler",
            "sd-x2-latent-upscaler",
        ]
        return fallback, fallback[0]




def choose_model(model_name: str):
    visible = model_name in GAI_MODELS
    note = (
        "GAI model selected: text fidelity may be reduced."
        if visible
        else "Fidelity model selected: best for text/UI details."
    )
    outscale_value = 2.0 if model_name == "sd-x2-latent-upscaler" else 4.0
    return (
        gr.update(value=model_name),
        gr.update(visible=visible),
        gr.update(value=outscale_value),
        gr.update(interactive=not visible),
        gr.update(interactive=not visible),
        note,
    )


def apply_preset(preset_name: str):
    return choose_model(_model_for_preset(preset_name))


def upscale_image(
    image: Image.Image,
    model: str,
    outscale: float,
    denoise: float,
    tile: int,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    seed: int,
):
    if image is None:
        return None, "Please upload an image first."

    buff = io.BytesIO()
    image.save(buff, format="PNG")
    buff.seek(0)

    files = {"file": ("input.png", buff.getvalue(), "image/png")}
    generative = model in GAI_MODELS

    data = {
        "model": model,
        "outscale": str(outscale),
        "denoise": str(denoise),
        "tile": str(tile),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": str(steps),
        "guidance_scale": str(guidance_scale),
        "seed": str(seed),
    }

    try:
        with httpx.Client(timeout=600) as client:
            resp = client.post(f"{IMAGE_SR_URL}/upscale", files=files, data=data)
            if resp.status_code != 200:
                return None, f"Backend error: {resp.status_code} {resp.text}"
            out = Image.open(io.BytesIO(resp.content)).convert("RGB")
        if generative:
            return out, "Done (GAI creative mode: text fidelity may be reduced)"
        return out, "Done"
    except Exception as exc:
        return None, f"Request failed: {exc}"


MODELS, DEFAULT_MODEL = fetch_models()


def _set_run_btn_state(image: Image.Image | None):
    has_image = image is not None
    return gr.update(interactive=has_image, value="🚀 Upscale")


def _disable_controls():
    """Lock all settings while processing."""
    return (
        gr.update(interactive=False),  # preset
        gr.update(interactive=False),  # model
        gr.update(interactive=False),  # outscale
        gr.update(interactive=False),  # denoise
        gr.update(interactive=False),  # tile
        gr.update(interactive=False, value="⏳ Processing…"),  # run_btn
    )


def _enable_controls(image: Image.Image | None):
    """Restore settings after processing."""
    has_image = image is not None
    return (
        gr.update(interactive=True),  # preset
        gr.update(interactive=True),  # model
        gr.update(interactive=True),  # outscale
        gr.update(interactive=True),  # denoise
        gr.update(interactive=True),  # tile
        gr.update(interactive=has_image, value="🚀 Upscale"),  # run_btn
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Image Super Resolution",
        theme=gr.themes.Soft(),
        css="""
        #left_panel {
            position: relative;
            z-index: 20;
        }
        #left_panel .gradio-slider,
        #left_panel .gradio-button,
        #left_panel .gradio-dropdown,
        #left_panel .gradio-radio {
            position: relative;
            z-index: 21;
        }
        #center_panel {
            position: relative;
            z-index: 20;
        }
        #right_panel {
            position: relative;
            z-index: 10;
            overflow: hidden;
        }
        #output_image_panel,
        #output_image_panel .wrap,
        #output_image_panel .empty,
        #output_image_panel img {
            background: #ffffff !important;
        }
    """,
    ) as demo:
        # ── header ────────────────────────────────────────────────────────────
        gr.Markdown(
            "# 🖼️ Image Super Resolution\n"
            "Enhance image clarity with **Real-ESRGAN** or **Stable Diffusion** on your GPU.\n\n"
            "_Configure settings on the left, upload in the center, view results on the right._"
        )

        # ── three-column layout ──────────────────────────────────────────────
        with gr.Row():
            # ── LEFT COLUMN: settings ───────────────────────────────────────
            with gr.Column(scale=1, elem_id="left_panel"):
                gr.Markdown("### ⚙️ Settings")

                preset = gr.Radio(
                    choices=PRESET_CHOICES,
                    value=_preset_for_model(DEFAULT_MODEL),
                    label="Preset",
                    info="Fidelity for text/UI detail, GAI for creative enhancement.",
                )
                model = gr.Dropdown(
                    MODELS, value=DEFAULT_MODEL, label="Model"
                )

                with gr.Accordion("Advanced", open=False):
                    outscale = gr.Slider(
                        minimum=1.0,
                        maximum=4.0,
                        value=4.0,
                        step=0.1,
                        label="Output scale",
                        info="4 = native model scale, maximum AI detail",
                    )
                    denoise = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0,
                        step=0.05,
                        label="Detail strength",
                        info="1.0 = sharpest  ·  0.0 = softest / denoised",
                    )
                    tile = gr.Slider(
                        minimum=0,
                        maximum=1024,
                        value=0,
                        step=64,
                        label="Tile size (VRAM control)",
                        info="0 = no tiling (fastest). Raise only on OOM.",
                    )

                with gr.Accordion(
                    "🧠 GAI Creative Controls", open=False, visible=False
                ) as gai_controls:
                    prompt = gr.Textbox(
                        label="Prompt",
                        value="high quality, detailed",
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value="text artifacts, blurry text, watermark, logo distortion",
                    )
                    with gr.Row():
                        steps = gr.Slider(
                            minimum=5, maximum=50, value=12, step=1, label="Steps"
                        )
                        guidance_scale = gr.Slider(
                            minimum=1.0,
                            maximum=12.0,
                            value=4.0,
                            step=0.5,
                            label="Guidance Scale",
                        )
                    seed = gr.Number(
                        value=-1, precision=0, label="Seed (-1 = random)"
                    )

                run_btn = gr.Button(
                    "🚀 Upscale",
                    variant="primary",
                    size="lg",
                    interactive=False,
                )

            # ── CENTER COLUMN: upload ───────────────────────────────────────
            with gr.Column(scale=1, elem_id="center_panel"):
                input_image = gr.Image(type="pil", label="📂 Upload image")

            # ── RIGHT COLUMN: result ────────────────────────────────────────
            with gr.Column(scale=1, elem_id="right_panel"):
                output_image = gr.Image(
                    type="pil",
                    label="✅ Upscaled Image",
                    elem_id="output_image_panel",
                )
                status = gr.Markdown("_Upload an image and click Upscale._")

        gr.Markdown(
            "_GPU-bound — one job at a time.  "
            "GAI models may hallucinate details; use ESRGAN for text-accurate results._"
        )

        # ── events ──────────────────────────────────────────────────────────
        preset.change(
            apply_preset,
            inputs=[preset],
            outputs=[model, gai_controls, outscale, denoise, tile, status],
        )

        model.change(
            choose_model,
            inputs=[model],
            outputs=[model, gai_controls, outscale, denoise, tile, status],
        )

        input_image.change(
            _set_run_btn_state,
            inputs=[input_image],
            outputs=[run_btn],
        )

        run_evt = run_btn.click(
            _disable_controls,
            inputs=None,
            outputs=[preset, model, outscale, denoise, tile, run_btn],
            queue=False,
        )

        run_evt.then(
            upscale_image,
            inputs=[
                input_image,
                model,
                outscale,
                denoise,
                tile,
                prompt,
                negative_prompt,
                steps,
                guidance_scale,
                seed,
            ],
            outputs=[output_image, status],
        ).then(
            _enable_controls,
            inputs=[input_image],
            outputs=[preset, model, outscale, denoise, tile, run_btn],
            queue=False,
        )

    return demo


if __name__ == "__main__":
    logger.info(
        "Starting Image Super Resolution UI v%s (backend: %s)",
        __version__,
        IMAGE_SR_URL,
    )
    build_ui().queue().launch(server_name=SERVER_NAME, server_port=SERVER_PORT)
