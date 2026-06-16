#!/usr/bin/env python3
"""
Video Upscaler - Gradio Web UI

Upload a video, pick an upscaling model / scale, and let the GPU enhance its
clarity. Progress is shown live; the result can be previewed and
downloaded when finished.

The heavy lifting happens in the `video-upscaler` API service; this UI is a thin
client that uploads the file, polls the job, and fetches the output.

Usage:
  python app.py
"""

import os
import tempfile
import time

import gradio as gr
import httpx

from config import (
    POLL_INTERVAL_SEC,
    SERVER_NAME,
    SERVER_PORT,
    UPLOAD_TIMEOUT_SEC,
    UPSCALER_URL,
    logger,
)

__version__ = "1.0"


def fetch_models() -> list[str]:
    """Get available model names from the backend (with a sensible fallback)."""
    try:
        resp = httpx.get(f"{UPSCALER_URL}/models", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data["models"]]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch models from backend: %s", exc)
        return [
            "realesr-general-x4v3",
            "RealESRGAN_x4plus",
            "RealESRGAN_x4plus_anime_6B",
            "RealESRGAN_x2plus",
        ]


def _progress_bar(pct: float) -> str:
    """Inline HTML progress bar that updates smoothly without flicker."""
    p = max(0.0, min(1.0, pct)) * 100
    return (
        f'<div style="background:#e5e7eb;border-radius:6px;height:14px;width:100%;overflow:hidden">'
        f'<div style="background:#6366f1;height:100%;width:{p:.1f}%;transition:width .3s"></div></div>'
    )


def upscale(video_path, model, outscale, denoise, tile):
    """
    Submit a video to the backend and stream progress until completion.

    Yields per-component updates for (status, output_video, download, live_preview).
    Unchanged components are sent gr.skip() so they never re-render (no flicker).
    """
    keep = gr.skip()  # sentinel: leave a component untouched

    if not video_path:
        yield "⚠️ Please upload a video first.", keep, keep, keep
        return

    try:
        with open(video_path, "rb") as f:
            files = {"file": (os.path.basename(video_path), f, "video/mp4")}
            form = {
                "model": model,
                "outscale": str(outscale),
                "denoise": str(denoise),
                "tile": str(int(tile)),
            }
            resp = httpx.post(
                f"{UPSCALER_URL}/upscale",
                files=files,
                data=form,
                timeout=UPLOAD_TIMEOUT_SEC,
            )
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
    except Exception as exc:  # noqa: BLE001
        logger.error("Upload failed: %s", exc)
        yield f"❌ Upload failed: {exc}", keep, keep, keep
        return

    logger.info("Submitted job %s", job_id)
    yield f"⏳ Queued (job `{job_id}`). Processing will start shortly…", keep, keep, keep

    # Poll for progress. If the browser tab is closed/refreshed, Gradio closes
    # this generator (GeneratorExit), so `finished` stays False and the finally
    # block cancels the backend job instead of leaving the GPU busy.
    finished = False
    try:
        while True:
            time.sleep(POLL_INTERVAL_SEC)
            try:
                jr = httpx.get(f"{UPSCALER_URL}/jobs/{job_id}", timeout=15)
                jr.raise_for_status()
                job = jr.json()
            except Exception as exc:  # noqa: BLE001
                finished = True
                yield f"❌ Lost connection to backend: {exc}", keep, keep, keep
                return

            status = job["status"]
            pct = job.get("progress", 0.0)
            done_f = job.get("done_frames", 0)
            total_f = job.get("total_frames", 0)

            if status == "queued":
                yield "⏳ Waiting in queue…", keep, keep, keep
            elif status == "processing":
                frames = f"{done_f}/{total_f} frames" if total_f else "preparing frames…"
                msg = (
                    f"### 🚀 Processing\n{frames} &nbsp; **{pct*100:.1f}%**\n\n{_progress_bar(pct)}"
                )
                preview = _download_preview(job_id, done_f)
                # Only touch the preview when a new frame is available.
                yield msg, keep, keep, (preview if preview else keep)
            elif status == "done":
                out_path = _download_result(job_id)
                if out_path is None:
                    finished = True
                    yield "❌ Finished but could not download the result.", keep, keep, keep
                    return
                res = job.get("result") or {}
                summary = (
                    f"### ✅ Done\n"
                    f"{res.get('source_resolution', '?')} → "
                    f"**{res.get('output_resolution', '?')}** "
                    f"({res.get('frames', total_f)} frames, model `{job['model']}`, "
                    f"x{job['outscale']:g})"
                )
                finished = True
                yield summary, out_path, out_path, keep
                return
            elif status == "cancelled":
                finished = True
                yield "🛑 Job cancelled.", keep, keep, keep
                return
            else:  # error
                finished = True
                yield f"❌ Failed: {job.get('error', 'unknown error')}", keep, keep, keep
                return
    finally:
        if not finished:
            # Client disconnected (page refresh/close) before completion.
            try:
                httpx.delete(f"{UPSCALER_URL}/jobs/{job_id}", timeout=10)
                logger.info("Cancelled job %s (client disconnected)", job_id)
            except Exception:  # noqa: BLE001
                pass


def _download_preview(job_id: str, frame_no: int) -> str | None:
    """Fetch the latest live-preview frame to a temp file for Gradio to show."""
    try:
        r = httpx.get(f"{UPSCALER_URL}/jobs/{job_id}/preview", timeout=15)
        if r.status_code != 200:
            return None
        out_dir = os.path.join(tempfile.gettempdir(), f"upscale_preview_{job_id}")
        os.makedirs(out_dir, exist_ok=True)
        # New filename per update so Gradio refreshes the image.
        out_path = os.path.join(out_dir, f"p_{frame_no}.jpg")
        with open(out_path, "wb") as f:
            f.write(r.content)
        return out_path
    except Exception as exc:  # noqa: BLE001
        logger.debug("Preview fetch failed: %s", exc)
        return None


def _download_result(job_id: str) -> str | None:
    """Stream the finished video to a temp file for Gradio to serve."""
    try:
        out_dir = tempfile.mkdtemp(prefix="upscaled_")
        out_path = os.path.join(out_dir, f"{job_id}.mp4")
        with httpx.stream(
            "GET", f"{UPSCALER_URL}/jobs/{job_id}/download", timeout=UPLOAD_TIMEOUT_SEC
        ) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        return out_path
    except Exception as exc:  # noqa: BLE001
        logger.error("Download failed: %s", exc)
        return None


def build_ui() -> gr.Blocks:
    models = fetch_models()
    default_model = "RealESRGAN_x4plus" if "RealESRGAN_x4plus" in models else models[0]

    with gr.Blocks(title="Video Upscaler") as demo:
        gr.Markdown(
            "# 🎬 Video Upscaler\n"
            "Enhance video clarity and resolution with **Real-ESRGAN** on your GPU."
        )

        with gr.Row(equal_height=False):
            # ---- Left: input + controls ----
            with gr.Column(scale=4):
                input_video = gr.Video(label="Input video", sources=["upload"], height=320)
                with gr.Row():
                    model = gr.Dropdown(
                        choices=models,
                        value=default_model,
                        label="Model",
                        scale=2,
                        info="x4plus = max AI detail (default); general-x4v3 for noisy video; anime_6B for animation",
                    )
                    outscale = gr.Slider(
                        1.0,
                        4.0,
                        value=4.0,
                        step=0.5,
                        label="Output scale",
                        scale=1,
                        info="Final size multiplier (4 = max AI detail, native model scale)",
                    )
                with gr.Accordion("Advanced settings", open=False):
                    denoise = gr.Slider(
                        0.0,
                        1.0,
                        value=1.0,
                        step=0.05,
                        label="Detail strength (general-x4v3 only)",
                        info="1.0 = sharpest, maximum AI detail · 0.0 = softer / denoised",
                    )
                    tile = gr.Slider(
                        0,
                        1024,
                        value=0,
                        step=64,
                        label="Tile size (VRAM control)",
                        info="0 = no tiling (fastest, recommended for 16GB+ GPUs). "
                        "Raise only if you hit out-of-memory on 4K input.",
                    )
                run_btn = gr.Button("🚀 Upscale video", variant="primary", size="lg")

            # ---- Right: status + live preview + result ----
            with gr.Column(scale=5):
                status = gr.Markdown("Upload a video and click **Upscale video** to begin.")
                live_preview = gr.Image(
                    label="Live preview — latest upscaled frame",
                    interactive=False,
                    height=320,
                )
                output_video = gr.Video(label="Upscaled result", interactive=False, height=320)
                download = gr.File(label="Download result", interactive=False)

        gr.Markdown(
            "ℹ️ GPU-bound and runs one video at a time. The live preview updates as frames "
            "finish; the full playable result appears when processing completes."
        )

        run_btn.click(
            upscale,
            inputs=[input_video, model, outscale, denoise, tile],
            outputs=[status, output_video, download, live_preview],
            show_progress="hidden",
        )

    return demo


if __name__ == "__main__":
    logger.info("Starting Video Upscaler UI v%s (backend: %s)", __version__, UPSCALER_URL)
    build_ui().queue().launch(
        server_name=SERVER_NAME, server_port=SERVER_PORT, theme=gr.themes.Soft()
    )
