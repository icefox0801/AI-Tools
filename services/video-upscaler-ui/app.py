#!/usr/bin/env python3
"""
Video Upscaler - Gradio Web UI

Progressive layout:
  1. Initially only the upload widget is shown.
  2. After a video is uploaded, settings + results panels appear and all
     video/image components are resized to match the source aspect ratio.

Three processing modes:
  🎬 Standard               – Real-ESRGAN frame-by-frame (fast)
  ✨ Standard + Smooth      – Real-ESRGAN + ffmpeg tmix temporal blend (reduces flicker)
  🎯 Temporal (BasicVSR++)  – BasicVSR++ video-aware SR (preserves motion blur)
"""

import base64
import io
import json
import os
import subprocess
import tempfile
import time

import gradio as gr
import httpx
from PIL import Image

from config import (
    POLL_INTERVAL_SEC,
    SERVER_NAME,
    SERVER_PORT,
    UPLOAD_TIMEOUT_SEC,
    UPSCALER_URL,
    logger,
)

__version__ = "1.1"

# ── constants ─────────────────────────────────────────────────────────────────
_VID_H_MAX = 900  # comparison video/preview — tall enough for 9:16 portrait
_VID_H_MIN = 180
_IMG_H_MAX = 900  # comparison image — tall enough for 9:16 portrait
_IMG_H_MIN = 140
_INPUT_VID_H_MAX = 560  # upload widget — left-column, keep compact
_INPUT_VID_TARGET_W = 400  # assumed left-column width for height calculation
_VID_TARGET_W = 420  # each video inside the 2-column results row
_IMG_TARGET_W = 420  # each image inside the 2-column comparison row


# ── helpers ───────────────────────────────────────────────────────────────────


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


def _extract_video_path(video) -> str | None:
    """Extract a local file path from whatever gr.Video passes.

    Gradio 4 passes a str; Gradio 5/6 passes a VideoData dataclass
    (video.video.path) or a FileData (video.path) or a dict.
    """
    if video is None:
        return None
    if isinstance(video, str):
        return video or None
    # Gradio 6 VideoData: .video is a FileData with .path
    if hasattr(video, "video"):
        inner = video.video
        return (getattr(inner, "path", None) or str(inner)) or None
    # Gradio 5/6 FileData: .path directly
    if hasattr(video, "path"):
        return video.path or None
    # Dict fallback
    if isinstance(video, dict):
        inner = video.get("video")
        if isinstance(inner, dict):
            return inner.get("path") or None
        return video.get("path") or None
    return None


def _probe_video_dims(path: str) -> tuple[int, int]:
    """Return (width, height) of *path* using ffprobe.  Falls back to (1920,1080)."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        data = json.loads(result.stdout)
        s = data["streams"][0]
        return int(s["width"]), int(s["height"])
    except Exception:  # noqa: BLE001
        return 1920, 1080


def _compute_heights(w: int, h: int) -> dict:
    """Proportional display heights for upload widget, result videos, and comparison images."""
    if h == 0:
        return {"input": 360, "video": 240, "image": 240}
    aspect = w / h
    input_h = int(_INPUT_VID_TARGET_W / aspect)
    input_h = max(_VID_H_MIN, min(_INPUT_VID_H_MAX, input_h))  # compact cap for full-width widget
    video_h = int(_VID_TARGET_W / aspect)
    video_h = max(_VID_H_MIN, min(_VID_H_MAX, video_h))
    image_h = int(_IMG_TARGET_W / aspect)
    image_h = max(_IMG_H_MIN, min(_IMG_H_MAX, image_h))
    return {"input": input_h, "video": video_h, "image": image_h}


def _progress_bar(pct: float) -> str:
    p = max(0.0, min(1.0, pct)) * 100
    return (
        '<div style="background:#e5e7eb;border-radius:6px;height:14px;width:100%;overflow:hidden">'
        f'<div style="background:#6366f1;height:100%;width:{p:.1f}%;transition:width .3s"></div></div>'
    )


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ── main processing generator ─────────────────────────────────────────────────


def upscale(video_path, model, outscale, denoise, tile, temporal_mode):
    """
    Submit a video to the backend and stream progress until completion.

    Yields updates for: (status, output_video, original_frame, upscaled_frame,
                         frame_info, preview_video, frame_slider, frame_number,
                         active_job_id).
    """
    keep = gr.skip()
    hide_selector = gr.update(visible=False)

    path = _extract_video_path(video_path)
    if not path:
        yield (
            "⚠️ Please upload a video first.",
            keep,
            keep,
            keep,
            keep,
            keep,
            hide_selector,
            "",
        )
        return

    mode_map = {
        "🎬 Standard": "standard",
        "✨ Standard + Smooth": "tmix",
        "🎯 BasicVSR++": "basicvsr",
    }
    api_mode = mode_map.get(temporal_mode, "standard")

    try:
        with open(path, "rb") as f:
            files = {"file": (os.path.basename(path), f, "video/mp4")}
            form = {
                "model": model,
                "outscale": str(outscale),
                "denoise": str(denoise),
                "tile": str(int(tile)),
                "temporal_mode": api_mode,
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
        yield (
            f"❌ Upload failed: {exc}",
            keep,
            keep,
            keep,
            keep,
            keep,
            hide_selector,
            "",
        )
        return

    logger.info("Submitted job %s (mode=%s)", job_id, api_mode)
    yield (
        f"⏳ Queued (job `{job_id}`). Processing will start shortly…",
        keep,
        keep,
        keep,
        keep,
        keep,
        hide_selector,
        job_id,
    )

    finished = False
    proc_start: float | None = None
    proc_start_frame = 0
    poll_count = 0
    try:
        while True:
            time.sleep(POLL_INTERVAL_SEC)
            poll_count += 1
            try:
                jr = httpx.get(f"{UPSCALER_URL}/jobs/{job_id}", timeout=15)
                jr.raise_for_status()
                job = jr.json()
            except Exception as exc:  # noqa: BLE001
                finished = True
                yield (
                    f"❌ Lost connection to backend: {exc}",
                    keep,
                    keep,
                    keep,
                    keep,
                    keep,
                    hide_selector,
                    job_id,
                )
                return

            status = job["status"]
            pct = job.get("progress", 0.0)
            done_f = job.get("done_frames", 0)
            total_f = job.get("total_frames", 0)

            if status == "queued":
                try:
                    jobs_resp = httpx.get(f"{UPSCALER_URL}/jobs", timeout=10)
                    if jobs_resp.status_code == 200:
                        all_jobs = jobs_resp.json().get("jobs", [])
                        processing = [j for j in all_jobs if j["status"] == "processing"]
                        if processing:
                            other = processing[0]
                            other_pct = other.get("progress", 0) * 100
                            other_done = other.get("done_frames", 0)
                            other_total = other.get("total_frames", 0)
                            frames_txt = (
                                f"{other_done}/{other_total} frames"
                                if other_total
                                else "preparing…"
                            )
                            yield (
                                f"### ⏳ Waiting in queue\n"
                                f"Another job is processing — {frames_txt} ({other_pct:.1f}% done).\n\n"
                                f"💡 Refresh the page to cancel your queued job.",
                                keep,
                                keep,
                                keep,
                                keep,
                                keep,
                                hide_selector,
                                job_id,
                            )
                            continue
                except Exception:  # noqa: BLE001
                    pass
                yield (
                    "⏳ Waiting in queue…",
                    keep,
                    keep,
                    keep,
                    keep,
                    keep,
                    hide_selector,
                    job_id,
                )

            elif status == "processing":
                now = time.monotonic()
                if proc_start is None and done_f > 0:
                    proc_start = now
                    proc_start_frame = done_f
                eta_txt = ""
                if proc_start is not None and total_f:
                    elapsed = now - proc_start
                    advanced = done_f - proc_start_frame
                    if advanced > 0 and elapsed > 0:
                        rate = advanced / elapsed
                        remaining = (total_f - done_f) / rate
                        eta_txt = f" &nbsp; ⏱ ~{_fmt_duration(remaining)} left ({rate:.1f} fps)"

                frames = f"{done_f}/{total_f} frames" if total_f else "preparing frames…"
                msg = (
                    f"### 🚀 Processing\n"
                    f"{frames} &nbsp; **{pct*100:.1f}%**{eta_txt}\n\n"
                    f"{_progress_bar(pct)}"
                )

                comparison = _fetch_latest_comparison(job_id) if poll_count % 5 == 1 else None
                orig_img = comparison["original"] if comparison else keep
                upscaled_img = comparison["upscaled"] if comparison else keep
                slider_upd = keep
                if comparison and comparison.get("frames"):
                    frames_list = comparison["frames"]
                    slider_upd = gr.update(
                        minimum=frames_list[0],
                        maximum=max(frames_list[-1], frames_list[0] + 1),
                        value=comparison["frame_idx"],
                        step=1,
                        visible=True,
                    )

                preview_vid = keep
                if poll_count % 15 == 0:
                    preview_vid = _download_preview_video(job_id) or keep

                yield (
                    msg,
                    keep,
                    orig_img,
                    upscaled_img,
                    keep,
                    preview_vid,
                    slider_upd,
                    job_id,
                )

            elif status == "done":
                out_path = _download_result(job_id)
                if out_path is None:
                    finished = True
                    yield (
                        "❌ Finished but could not download the result.",
                        keep,
                        keep,
                        keep,
                        "No frames available",
                        keep,
                        job_id,
                    )
                    return
                res = job.get("result") or {}
                mode_label = job.get("temporal_mode", "standard")
                summary = (
                    f"### ✅ Done\n"
                    f"{res.get('source_resolution', '?')} → "
                    f"**{res.get('output_resolution', '?')}** "
                    f"({res.get('frames', total_f)} frames · model `{job['model']}` · "
                    f"x{job['outscale']:g} · mode `{mode_label}`)"
                )
                finished = True
                yield summary, out_path, keep, keep, keep, keep, keep, job_id
                return

            elif status == "cancelled":
                finished = True
                yield (
                    "🛑 Job cancelled.",
                    keep,
                    keep,
                    keep,
                    keep,
                    keep,
                    hide_selector,
                    job_id,
                )
                return

            else:
                finished = True
                yield (
                    f"❌ Failed: {job.get('error', 'unknown error')}",
                    keep,
                    keep,
                    keep,
                    keep,
                    keep,
                    hide_selector,
                    job_id,
                )
                return

    finally:
        if not finished:
            try:
                httpx.delete(f"{UPSCALER_URL}/jobs/{job_id}", timeout=10)
                logger.info("Cancelled job %s (client disconnected)", job_id)
            except Exception:  # noqa: BLE001
                pass


# ── result / preview helpers ──────────────────────────────────────────────────


def _download_result(job_id: str) -> str | None:
    try:
        r = httpx.get(f"{UPSCALER_URL}/jobs/{job_id}/download", timeout=600)
        if r.status_code != 200:
            return None
        out_dir = os.path.join(tempfile.gettempdir(), f"upscale_result_{job_id}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "result.mp4")
        with open(out_path, "wb") as f:
            f.write(r.content)
        return out_path
    except Exception as exc:  # noqa: BLE001
        logger.error("Download failed: %s", exc)
        return None


def _download_preview_video(job_id: str) -> str | None:
    try:
        r = httpx.get(f"{UPSCALER_URL}/jobs/{job_id}/preview-video", timeout=30)
        if r.status_code != 200:
            return None
        out_dir = os.path.join(tempfile.gettempdir(), f"upscale_preview_{job_id}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "preview_video.mp4")
        with open(out_path, "wb") as f:
            f.write(r.content)
        return out_path
    except Exception as exc:  # noqa: BLE001
        logger.debug("Preview video download failed: %s", exc)
        return None


def _b64_to_pil(data_uri: str) -> Image.Image | None:
    try:
        b64 = data_uri.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64)))
    except Exception:
        return None


def _fetch_comparison(job_id: str, frame_idx: int) -> dict | None:
    """Fetch a specific before/after comparison frame."""
    try:
        r = httpx.get(f"{UPSCALER_URL}/jobs/{job_id}/comparison/{frame_idx}", timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        return {
            "frame_idx": data["frame_idx"],
            "original": _b64_to_pil(data["original"]),
            "upscaled": _b64_to_pil(data["upscaled"]),
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("Comparison fetch failed: %s", exc)
        return None


def _fetch_latest_comparison(job_id: str) -> dict | None:
    try:
        r = httpx.get(f"{UPSCALER_URL}/jobs/{job_id}/comparison-frames", timeout=10)
        if r.status_code != 200:
            return None
        frames = r.json().get("frames", [])
        if not frames:
            return None
        latest_frame = frames[-1]
        data = _fetch_comparison(job_id, latest_frame)
        if not data:
            return None
        data["frames"] = frames
        return data
    except Exception as exc:  # noqa: BLE001
        logger.debug("Comparison fetch failed: %s", exc)
        return None


# ── job queue helpers ─────────────────────────────────────────────────────────


def _fetch_jobs() -> list[dict]:
    """Fetch all jobs (newest first) from the backend."""
    r = httpx.get(f"{UPSCALER_URL}/jobs", timeout=10)
    if r.status_code != 200:
        raise RuntimeError("Failed to fetch job queue")
    return r.json().get("jobs", [])


def _format_job_line(job: dict) -> str:
    """Readable single-line job summary for queue display and selector labels."""
    status = job.get("status", "unknown")
    pct = job.get("progress", 0.0) * 100
    done = job.get("done_frames", 0)
    total = job.get("total_frames", 0)
    file_name = os.path.basename(job.get("filename", "video"))
    if total:
        return f"{job['id'][:8]} | {status:<10} | {done:>5}/{total:<5} | {pct:>5.1f}% | {file_name}"
    return f"{job['id'][:8]} | {status:<10} | {'-':>11} | {pct:>5.1f}% | {file_name}"


def _queue_view(closed_ids: list[str] | None) -> tuple[str, list[tuple[str, str]], str | None]:
    """Build queue markdown + dropdown choices for active jobs only."""
    closed = set(closed_ids or [])
    jobs = [
        j
        for j in _fetch_jobs()
        if j.get("id") not in closed and j.get("status") in ("queued", "processing")
    ]
    if not jobs:
        return "✅ No running jobs", [], None

    lines = ["### Running Jobs", ""]
    for j in jobs:
        status = j.get("status", "unknown")
        icon = {
            "queued": "⏳",
            "processing": "🔄",
            "done": "✅",
            "error": "❌",
            "cancelled": "🛑",
        }.get(status, "•")
        lines.append(f"- {icon} {_format_job_line(j)}")

    choices = [(_format_job_line(j), j["id"]) for j in jobs]
    return "\n".join(lines), choices, None


def refresh_job_queue(closed_ids: list[str] | None):
    """Refresh queue display and selector from backend."""
    try:
        text, choices, value = _queue_view(closed_ids)
        return text, gr.update(choices=choices, value=value)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Queue refresh failed: %s", exc)
        return "⚠️ Failed to fetch job queue", gr.update(choices=[], value=None)


def cancel_selected_job(selected_id: str | None, closed_ids: list[str] | None):
    """Cancel selected queued/processing job, then refresh queue list."""
    closed = list(closed_ids or [])
    if not selected_id:
        text, choices, value = _queue_view(closed)
        return "⚠️ Select a job first\n\n" + text, gr.update(choices=choices, value=value), closed

    note = ""
    try:
        r = httpx.delete(f"{UPSCALER_URL}/jobs/{selected_id}", timeout=10)
        if r.status_code == 200 and r.json().get("cancelled"):
            note = f"✅ Cancelled {selected_id[:8]}\n\n"
        else:
            note = f"⚠️ Could not cancel {selected_id[:8]} (already finished?)\n\n"
    except Exception as exc:  # noqa: BLE001
        logger.debug("Cancel selected failed: %s", exc)
        note = "⚠️ Cancel failed\n\n"

    text, choices, value = _queue_view(closed)
    return note + text, gr.update(choices=choices, value=value), closed


def close_selected_job(selected_id: str | None, closed_ids: list[str] | None):
    """Hide selected job from queue list in this UI session (does not delete backend data)."""
    closed = list(closed_ids or [])
    if selected_id and selected_id not in closed:
        closed.append(selected_id)
    text, choices, value = _queue_view(closed)
    return text, gr.update(choices=choices, value=value), closed


# ── UI builder ────────────────────────────────────────────────────────────────


def build_ui() -> gr.Blocks:
    esrgan_models = fetch_models()
    default_model = (
        "RealESRGAN_x4plus" if "RealESRGAN_x4plus" in esrgan_models else esrgan_models[0]
    )
    _dv, _di = 320, 240  # default heights before video is loaded

    with gr.Blocks(
        title="Video Upscaler",
        theme=gr.themes.Soft(),
        css="""
        #left_panel {
            position: relative;
            z-index: 20;
        }
        #left_panel .gradio-slider,
        #left_panel .gradio-button,
        #left_panel .gradio-dropdown,
        #left_panel .gradio-radio,
        #left_panel .gradio-video {
            position: relative;
            z-index: 21;
        }
        #right_panel {
            position: relative;
            z-index: 10;
            overflow: hidden;
        }
        #mode_radio .wrap {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            gap: 4px !important;
        }
        #mode_radio .wrap > label {
            flex: 0 1 auto !important;
            white-space: nowrap !important;
            padding: 6px 10px !important;
        }
    """,
    ) as demo:

        # ── header ────────────────────────────────────────────────────────────
        gr.Markdown(
            "# 🎬 Video Upscaler\n"
            "Enhance video clarity with **Real-ESRGAN** or **BasicVSR++** on your GPU.\n\n"
            "_Upload a video to get started — settings appear automatically._"
        )

        # ── two-column layout (always visible) ───────────────────────────────
        with gr.Row():

            # ── LEFT COLUMN: upload + settings ──────────────────────────────
            with gr.Column(scale=1, min_width=280, elem_id="left_panel"):
                input_video = gr.Video(
                    label="📂 Upload video",
                    sources=["upload"],
                    height=_dv,
                    elem_id="input_video",
                )
                # Injected after upload to constrain widget to the video's aspect ratio
                css_injector = gr.HTML("", visible=False, elem_id="css_injector")

                gr.Markdown("### ⚙️ Settings")

                temporal_mode = gr.Radio(
                    choices=["🎬 Standard", "✨ Standard + Smooth", "🎯 BasicVSR++"],
                    value="🎬 Standard",
                    label="Processing mode",
                    elem_id="mode_radio",
                    info=(
                        "Standard: fastest, frame-by-frame ESRGAN  ·  "
                        "Standard + Smooth: ESRGAN then ffmpeg tmix temporal blend (reduces flicker)  ·  "
                        "Temporal: BasicVSR++ — preserves motion blur, 4× only"
                    ),
                )

                with gr.Group() as esrgan_controls:
                    preset = gr.Radio(
                        choices=["⚡ Fast", "⚖️ Balanced", "💎 Best"],
                        value="💎 Best",
                        label="Preset",
                        info="Fast: 2× ESRGAN-v3  ·  Balanced: 3×  ·  Best: 4× x4plus",
                    )
                    model = gr.Dropdown(
                        choices=esrgan_models,
                        value=default_model,
                        label="Model",
                        info="x4plus = max detail · general-x4v3 = noise-aware · anime_6B = animation",
                    )
                    outscale = gr.Slider(
                        1.0,
                        4.0,
                        value=4.0,
                        step=0.5,
                        label="Output scale",
                        info="4 = native model scale, maximum AI detail",
                    )
                    with gr.Accordion("Advanced", open=False):
                        denoise = gr.Slider(
                            0.0,
                            1.0,
                            value=1.0,
                            step=0.05,
                            label="Detail strength (general-x4v3 only)",
                            info="1.0 = sharpest  ·  0.0 = softest / denoised",
                        )
                        tile = gr.Slider(
                            0,
                            1024,
                            value=0,
                            step=64,
                            label="Tile size (VRAM control)",
                            info="0 = no tiling (fastest, suits 16 GB GPU).  Raise only on OOM.",
                        )

                with gr.Group(visible=False) as basicvsr_info:
                    gr.Markdown(
                        "> **BasicVSR++** processes frames in temporal windows — motion blur and "
                        "inter-frame consistency are preserved.  Scale is fixed at **4×**.  "
                        "First run downloads ~70 MB of weights automatically."
                    )
                    _bvsr_outscale = gr.Number(value=4.0, visible=False)
                    _bvsr_denoise = gr.Number(value=1.0, visible=False)
                    _bvsr_tile = gr.Number(value=0, visible=False)

                run_btn = gr.Button("🚀 Start upscaling", variant="primary", size="lg")

            # ── RIGHT COLUMN: results ──────────────────────────────────────────
            with gr.Column(scale=2, elem_id="right_panel"):
                status = gr.Markdown("_Waiting for job…_")
                active_job_id = gr.State("")

                with gr.Accordion("📋 Job Queue", open=False):
                    closed_jobs = gr.State([])
                    queue_display = gr.Markdown("No jobs yet")
                    queue_selector = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Select job",
                        info="Select a queued or processing job to cancel or hide",
                    )
                    with gr.Row():
                        refresh_queue_btn = gr.Button("🔄 Refresh", size="sm")
                        cancel_selected_btn = gr.Button(
                            "🛑 Cancel Selected", size="sm", variant="stop"
                        )
                        close_selected_btn = gr.Button("✖ Close Selected", size="sm")

                gr.Markdown("### 🔍 Before / After Comparison")
                with gr.Row():
                    with gr.Column(scale=1):
                        original_frame = gr.Image(
                            label="Original",
                            interactive=False,
                            height=_di,
                        )
                    with gr.Column(scale=1):
                        upscaled_frame = gr.Image(
                            label="AI Upscaled",
                            interactive=False,
                            height=_di,
                        )
                frame_info = gr.Markdown("No frames yet", visible=False)
                with gr.Row():
                    frame_slider = gr.Slider(
                        0,
                        1,
                        value=0,
                        step=1,
                        label="Comparison frame",
                        interactive=True,
                        visible=False,
                    )

                with gr.Row():
                    preview_video = gr.Video(
                        label="📹 Live preview (last ~3 s, updates every 30 s)",
                        interactive=False,
                        height=_dv,
                        autoplay=True,
                    )
                    output_video = gr.Video(
                        label="✅ Final result",
                        interactive=False,
                        height=_dv,
                    )

        gr.Markdown(
            "_GPU-bound — one job at a time.  "
            "Comparison frames update every ~10 s; preview clip every ~30 s._"
        )

        # ── event: video upload → set aspect-ratio heights ─────────────────────
        def on_video_upload(video):
            fpath = _extract_video_path(video)
            if not fpath:
                return (
                    gr.update(height=_dv),  # input_video
                    gr.update(height=_di),  # original_frame
                    gr.update(height=_di),  # upscaled_frame
                    gr.update(height=_dv),  # preview_video
                    gr.update(height=_dv),  # output_video
                    gr.update(value="", visible=False),  # css_injector
                )
            w, h = _probe_video_dims(fpath)
            heights = _compute_heights(w, h)
            ih = heights["image"]
            vh = heights["video"]
            input_h = heights["input"]
            # Constrain widget width so portrait video isn't pillarboxed inside the column.
            input_max_w = int(input_h * w / h) if h > 0 else 400
            css = (
                "<style>"
                f"#input_video{{max-width:{input_max_w}px!important;"
                "margin-left:auto!important;margin-right:auto!important;}}"
                "</style>"
            )
            return (
                gr.update(height=input_h),  # input_video
                gr.update(height=ih),  # original_frame
                gr.update(height=ih),  # upscaled_frame
                gr.update(height=vh),  # preview_video
                gr.update(height=vh),  # output_video
                gr.update(value=css, visible=True),  # css_injector
            )

        input_video.change(
            on_video_upload,
            inputs=[input_video],
            outputs=[
                input_video,
                original_frame,
                upscaled_frame,
                preview_video,
                output_video,
                css_injector,
            ],
        )

        # ── event: mode change → toggle ESRGAN vs BasicVSR++ controls ─────────
        def on_mode_change(mode):
            is_bvsr = mode == "🎯 BasicVSR++"
            return (
                gr.update(visible=not is_bvsr),  # esrgan_controls
                gr.update(visible=is_bvsr),  # basicvsr_info
            )

        temporal_mode.change(
            on_mode_change,
            inputs=[temporal_mode],
            outputs=[esrgan_controls, basicvsr_info],
        )

        # ── event: preset → model / scale / denoise ───────────────────────────
        def apply_preset(preset_name):
            if preset_name == "⚡ Fast":
                return "realesr-general-x4v3", 2.0, 0.5
            if preset_name == "⚖️ Balanced":
                return "realesr-general-x4v3", 3.0, 0.8
            return "RealESRGAN_x4plus", 4.0, 1.0

        preset.change(apply_preset, inputs=[preset], outputs=[model, outscale, denoise])

        def on_select_frame(frame_idx, job_id):
            keep = gr.skip()
            if frame_idx is None or not job_id:
                return keep, keep, keep, keep
            try:
                idx = int(frame_idx)
            except Exception:  # noqa: BLE001
                return keep, keep, keep, keep
            comp = _fetch_comparison(job_id, idx)
            if not comp:
                return keep, keep, keep, keep
            return (
                comp["original"],
                comp["upscaled"],
                gr.update(value=comp["frame_idx"], visible=True),
                gr.skip(),  # Don't update active_job_id when manually scrubbing
            )

        # ── lock/unlock controls while a job is running ─────────────────────
        def _disable_controls():
            return (
                gr.update(interactive=False),  # input_video
                gr.update(interactive=False),  # temporal_mode
                gr.update(interactive=False),  # preset
                gr.update(interactive=False),  # model
                gr.update(interactive=False),  # outscale
                gr.update(interactive=False),  # denoise
                gr.update(interactive=False),  # tile
                gr.update(interactive=False, value="⏳ Processing…"),  # run_btn
                gr.update(interactive=True),  # refresh_queue_btn
                gr.update(interactive=True),  # cancel_selected_btn
                gr.update(interactive=True),  # close_selected_btn
            )

        def _enable_controls():
            return (
                gr.update(interactive=True),  # input_video
                gr.update(interactive=True),  # temporal_mode
                gr.update(interactive=True),  # preset
                gr.update(interactive=True),  # model
                gr.update(interactive=True),  # outscale
                gr.update(interactive=True),  # denoise
                gr.update(interactive=True),  # tile
                gr.update(interactive=True, value="🚀 Start upscaling"),  # run_btn
                gr.update(interactive=True),  # refresh_queue_btn
                gr.update(interactive=True),  # cancel_selected_btn
                gr.update(interactive=True),  # close_selected_btn
            )

        # ── run ───────────────────────────────────────────────────────────────
        run_evt = run_btn.click(
            _disable_controls,
            inputs=None,
            outputs=[
                input_video,
                temporal_mode,
                preset,
                model,
                outscale,
                denoise,
                tile,
                run_btn,
                refresh_queue_btn,
                cancel_selected_btn,
                close_selected_btn,
            ],
            queue=False,
        )

        run_evt.then(
            upscale,
            inputs=[input_video, model, outscale, denoise, tile, temporal_mode],
            outputs=[
                status,
                output_video,
                original_frame,
                upscaled_frame,
                frame_info,
                preview_video,
                frame_slider,
                active_job_id,
            ],
            show_progress="hidden",
        ).then(
            _enable_controls,
            inputs=None,
            outputs=[
                input_video,
                temporal_mode,
                preset,
                model,
                outscale,
                denoise,
                tile,
                run_btn,
                refresh_queue_btn,
                cancel_selected_btn,
                close_selected_btn,
            ],
            queue=False,
        )

        def load_selected_job(job_id):
            """Load and display comparison frames for a selected job from the queue."""
            keep = gr.skip()
            if not job_id:
                return keep, keep, gr.update(visible=False), keep

            # Fetch latest comparison for the selected job
            comp = _fetch_latest_comparison(job_id)
            if not comp:
                return keep, keep, gr.update(visible=False), keep

            # Update comparison images and make slider visible
            frames_list = comp.get("frames", [])
            slider_upd = gr.update(visible=False)
            if frames_list:
                slider_upd = gr.update(
                    minimum=frames_list[0],
                    maximum=frames_list[-1],
                    value=comp["frame_idx"],
                    step=1,
                    visible=True,
                )

            return (
                comp["original"],
                comp["upscaled"],
                slider_upd,
                job_id,  # Update active_job_id so frame slider can fetch frames
            )

        frame_slider.change(
            on_select_frame,
            inputs=[frame_slider, active_job_id],
            outputs=[original_frame, upscaled_frame, frame_slider, frame_info],
            queue=False,
        )

        queue_selector.change(
            load_selected_job,
            inputs=[queue_selector],
            outputs=[original_frame, upscaled_frame, frame_slider, active_job_id],
            queue=False,
        )

        demo.load(
            refresh_job_queue,
            inputs=[closed_jobs],
            outputs=[queue_display, queue_selector],
            queue=False,
        )
        refresh_queue_btn.click(
            refresh_job_queue,
            inputs=[closed_jobs],
            outputs=[queue_display, queue_selector],
            queue=False,
        )
        cancel_selected_btn.click(
            cancel_selected_job,
            inputs=[queue_selector, closed_jobs],
            outputs=[queue_display, queue_selector, closed_jobs],
            queue=False,
        )
        close_selected_btn.click(
            close_selected_job,
            inputs=[queue_selector, closed_jobs],
            outputs=[queue_display, queue_selector, closed_jobs],
            queue=False,
        )

    return demo


if __name__ == "__main__":
    logger.info("Starting Video Upscaler UI v%s (backend: %s)", __version__, UPSCALER_URL)
    build_ui().queue().launch(server_name=SERVER_NAME, server_port=SERVER_PORT)
