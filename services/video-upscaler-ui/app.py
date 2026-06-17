#!/usr/bin/env python3
"""
Video Upscaler - Gradio Web UI

Progressive layout:
  1. Initially only the upload widget is shown.
  2. After a video is uploaded, settings + results panels appear and all
     video/image components are resized to match the source aspect ratio.

Three processing modes:
  🎬 Standard               - Real-ESRGAN frame-by-frame (fast)
  ✨ Standard + Smooth      - Real-ESRGAN + ffmpeg tmix temporal blend (reduces flicker)
  🎯 Temporal (BasicVSR++)  - BasicVSR++ video-aware SR (preserves motion blur)
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
from config import (
    POLL_INTERVAL_SEC,
    SERVER_NAME,
    SERVER_PORT,
    UPLOAD_TIMEOUT_SEC,
    UPSCALER_URL,
    logger,
)
from PIL import Image

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
_DEFAULT_VID_H = 320  # compact placeholder height (before source dims are known)
_DEFAULT_IMG_H = 240  # compact placeholder height for comparison images


# ── helpers ───────────────────────────────────────────────────────────────────


def fetch_models() -> list[str]:
    """Get available model names from the backend (with a sensible fallback)."""
    try:
        resp = httpx.get(f"{UPSCALER_URL}/models", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data["models"]]
    except Exception as exc:
        logger.warning("Could not fetch models from backend: %s", exc)
        return [
            "RealESRGAN_x4plus",
            "ffmpeg-enhance",
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
    except Exception:
        return 1920, 1080


def _compute_heights(w: int, h: int) -> dict:
    """Proportional display heights for upload widget, result videos, and comparison images."""
    if h == 0:
        return {"input": 360, "video": 240, "image": 240, "image_w": 320}
    aspect = w / h
    input_h = int(_INPUT_VID_TARGET_W / aspect)
    input_h = max(_VID_H_MIN, min(_INPUT_VID_H_MAX, input_h))  # compact cap for full-width widget
    video_h = int(_VID_TARGET_W / aspect)
    video_h = max(_VID_H_MIN, min(_VID_H_MAX, video_h))
    image_h = int(_IMG_TARGET_W / aspect)
    image_h = max(_IMG_H_MIN, min(_IMG_H_MAX, image_h))
    image_w = int(image_h * aspect)  # width at correct aspect ratio for the clamped height
    return {"input": input_h, "video": video_h, "image": image_h, "image_w": image_w}


def _progress_bar(pct: float) -> str:
    p = max(0.0, min(1.0, pct)) * 100
    return (
        '<div style="background:#e5e7eb;border-radius:6px;height:14px;width:100%;overflow:hidden">'
        f'<div style="background:#6366f1;height:100%;width:{p:.1f}%;transition:width .3s"></div></div>'
    )


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _heights_from_res(resolution: str | None) -> dict | None:
    """Parse 'WxH' resolution string and return height dict, or None on failure."""
    if not resolution:
        return None
    try:
        w, h = map(int, resolution.lower().split("x"))
        return _compute_heights(w, h)
    except Exception:
        return None


def _heights_from_image(img) -> dict | None:
    """Derive height dict from a PIL Image's dimensions, or None on failure."""
    try:
        w, h = img.size
        return _compute_heights(w, h)
    except Exception:
        return None


# gr.skip() returns the same singleton each call — capture it once for identity checks.
_SKIP = gr.skip()


def _wrap_h(val, height: int | None):
    """Return gr.update(value=val, height=height) or val unchanged."""
    if height is None or val is None or isinstance(val, dict) or val is _SKIP:
        return val
    return gr.update(value=val, height=height)


# ── shared job-polling generator ─────────────────────────────────────────────


def _stream_job(job_id: str, cancel_on_disconnect: bool = True):
    """
    Poll job_id until terminal state, yielding UI updates.

    Yields 8-tuples: (status, output_video, original_frame, upscaled_frame,
                      frame_info, preview_video, frame_slider, active_job_id)

    cancel_on_disconnect=True  → cancel the backend job if the client
                                  disconnects (used when *we* submitted it).
    cancel_on_disconnect=False → just stop watching (used when reattaching
                                  to a job we didn't submit).
    """
    keep = gr.skip()
    hide_selector = gr.update(visible=False)

    finished = False
    proc_start: float | None = None
    proc_start_frame = 0
    poll_count = 0
    img_h: int | None = None  # set once on first comparison frame
    vid_h: int | None = None
    try:
        while True:
            time.sleep(POLL_INTERVAL_SEC)
            poll_count += 1
            try:
                jr = httpx.get(f"{UPSCALER_URL}/jobs/{job_id}", timeout=15)
                jr.raise_for_status()
                job = jr.json()
            except Exception as exc:
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
                except Exception:
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
                # Detect dimensions once from first comparison frame.
                first_heights = False
                if comparison and img_h is None:
                    heights = _heights_from_image(comparison.get("original"))
                    if heights:
                        img_h = heights["image"]
                        vid_h = heights["video"]
                        first_heights = True
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

                # On first height detection set video container heights to match
                # images (height-only gr.update is safe; value+height crashes in
                # Gradio 6.x via Video.postprocess).
                if first_heights and vid_h is not None:
                    output_vid_upd = gr.update(height=vid_h)
                    preview_vid_upd = gr.update(height=vid_h)
                else:
                    output_vid_upd = keep
                    preview_vid_upd = preview_vid

                yield (
                    msg,
                    output_vid_upd,
                    _wrap_h(orig_img, img_h),
                    _wrap_h(upscaled_img, img_h),
                    keep,
                    preview_vid_upd,
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
                done_f = job.get("done_frames", 0)
                total_f = job.get("total_frames", 0)
                pct = job.get("progress", 0.0) * 100
                progress_txt = f"{done_f}/{total_f} frames ({pct:.1f}%)" if total_f else "partial"
                comp = _fetch_latest_comparison(job_id)
                orig_img = comp["original"] if comp else keep
                upscaled_img = comp["upscaled"] if comp else keep
                slider_upd = hide_selector
                if comp and comp.get("frames"):
                    frames_list = comp["frames"]
                    slider_upd = gr.update(
                        minimum=frames_list[0],
                        maximum=max(frames_list[-1], frames_list[0] + 1),
                        value=comp["frame_idx"],
                        step=1,
                        visible=True,
                    )
                preview_vid = _download_preview_video(job_id) or keep
                yield (
                    f"🛑 Cancelled — {progress_txt} completed.",
                    keep,
                    orig_img,
                    upscaled_img,
                    keep,
                    preview_vid,
                    slider_upd,
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
        if not finished and cancel_on_disconnect:
            try:
                httpx.delete(f"{UPSCALER_URL}/jobs/{job_id}", timeout=10)
                logger.info("Cancelled job %s (client disconnected)", job_id)
            except Exception:
                pass


# ── main processing generator ─────────────────────────────────────────────────


def upscale(video_path, model, outscale, denoise, tile, temporal_mode):
    """
    Submit a video to the backend and stream progress until completion.

    Yields updates for: (status, output_video, original_frame, upscaled_frame,
                         frame_info, preview_video, frame_slider, active_job_id).
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

    api_mode = temporal_mode  # State already holds the API string (standard/tmix/basicvsr)

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
    except Exception as exc:
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
        None,  # clear output video
        gr.update(value=None, height=_DEFAULT_IMG_H),  # clear + reset original frame
        gr.update(value=None, height=_DEFAULT_IMG_H),  # clear + reset upscaled frame
        keep,
        None,  # clear preview video
        hide_selector,
        job_id,
    )
    yield from _stream_job(job_id, cancel_on_disconnect=True)


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
    except Exception as exc:
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
    except Exception as exc:
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
    except Exception as exc:
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
    except Exception as exc:
        logger.debug("Comparison fetch failed: %s", exc)
        return None


# ── job queue helpers ─────────────────────────────────────────────────────────


def _fetch_jobs() -> list[dict]:
    """Fetch all jobs (newest first) from the backend."""
    r = httpx.get(f"{UPSCALER_URL}/jobs", timeout=10)
    if r.status_code != 200:
        raise RuntimeError("Failed to fetch job queue")
    return r.json().get("jobs", [])


def _fetch_disk_jobs() -> list[dict]:
    """Fetch all jobs discovered from disk (full history)."""
    try:
        r = httpx.get(f"{UPSCALER_URL}/jobs/discover", timeout=10)
        r.raise_for_status()
        return r.json().get("jobs", [])
    except Exception as exc:
        logger.debug("Disk jobs fetch failed: %s", exc)
        return []


def _format_prev_job_label(j: dict) -> str:
    """Human-readable label for a historical job dropdown entry."""
    status = j.get("status", "?")
    icon = {"done": "✅", "error": "❌", "cancelled": "🛑", "processing": "🔄", "queued": "⏳"}.get(
        status, "•"
    )
    filename = os.path.basename(j.get("filename", "video"))
    ts = j.get("finished_at") or j.get("created_at")
    date_str = time.strftime("%m/%d %H:%M", time.localtime(ts)) if ts else "?"
    model = j.get("model", "")
    short_model = model.split("_")[0] if model else ""
    outscale = j.get("outscale", "")
    return f"{icon} {j['id'][:8]} | {filename} | x{outscale:g} {short_model} | {date_str}"


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


def refresh_prev_jobs():
    """Return updated choices for the previous-jobs dropdown."""
    jobs = _fetch_disk_jobs()
    choices = [(_format_prev_job_label(j), j["id"]) for j in jobs]
    return gr.update(choices=choices, value=None)


def load_previous_job(job_id: str | None):
    """Load a historical job's results; reattach as live stream if still running."""
    keep = gr.skip()
    hide_slider = gr.update(visible=False)

    if not job_id:
        yield "⚠️ Select a job to load.", keep, keep, keep, keep, keep, hide_slider, ""
        return

    try:
        r = httpx.get(f"{UPSCALER_URL}/jobs/{job_id}", timeout=10)
        if r.status_code == 404:
            yield f"❌ Job `{job_id[:8]}` not found.", keep, keep, keep, keep, keep, hide_slider, ""
            return
        r.raise_for_status()
        job = r.json()
    except Exception as exc:
        yield f"❌ Failed to fetch job: {exc}", keep, keep, keep, keep, keep, hide_slider, ""
        return

    status = job.get("status", "unknown")

    # Still running — clear stale UI and reattach to the live stream.
    if status in ("queued", "processing"):
        yield (
            f"⏳ Reattaching to job `{job_id[:8]}`…",
            None,
            gr.update(value=None, height=_DEFAULT_IMG_H),
            gr.update(value=None, height=_DEFAULT_IMG_H),
            keep,
            None,
            hide_slider,
            job_id,
        )
        # cancel_on_disconnect=False: closing the page must not kill the job.
        yield from _stream_job(job_id, cancel_on_disconnect=False)
        return

    # Terminal states — build status banner then load whatever artifacts exist.
    if status == "done":
        res = job.get("result") or {}
        banner = (
            f"### ✅ Done (loaded from history)\n"
            f"{res.get('source_resolution', '?')} → "
            f"**{res.get('output_resolution', '?')}** "
            f"({res.get('frames', '?')} frames · model `{job['model']}` · "
            f"x{job['outscale']:g} · mode `{job.get('temporal_mode', 'standard')}`)"
        )
    elif status == "cancelled":
        done_f = job.get("done_frames", 0)
        total_f = job.get("total_frames", 0)
        pct = job.get("progress", 0.0) * 100
        progress_txt = f"{done_f}/{total_f} frames ({pct:.1f}%)" if total_f else "partial"
        banner = f"### 🛑 Cancelled — {progress_txt} completed\n_Showing frames captured before cancellation._"
    elif status == "error":
        banner = f"### ❌ Failed: {job.get('error', 'unknown error')}\n_Showing frames captured before the error._"
    else:
        yield f"⚠️ Unknown status: {status}", keep, keep, keep, keep, keep, hide_slider, job_id
        return

    out_path = _download_result(job_id) if status == "done" else None
    preview_vid = _download_preview_video(job_id)
    comp = _fetch_latest_comparison(job_id)
    orig_img = comp["original"] if comp else keep
    upscaled_img = comp["upscaled"] if comp else keep

    # Compute image dimensions from comparison frame for proper aspect-ratio sizing.
    img_h = None
    if comp:
        _h = _heights_from_image(comp.get("original"))
        if _h:
            img_h = _h["image"]

    # For empty video slots reset to compact height (height-only gr.update is safe;
    # gr.update(value=path, height=x) crashes via Gradio 6.x Video.postprocess).
    out_video_upd = (
        out_path if out_path is not None else gr.update(value=None, height=_DEFAULT_VID_H)
    )
    prev_video_upd = (
        preview_vid if preview_vid is not None else gr.update(value=None, height=_DEFAULT_VID_H)
    )

    slider_upd = hide_slider
    if comp and comp.get("frames"):
        frames_list = comp["frames"]
        slider_upd = gr.update(
            minimum=frames_list[0],
            maximum=max(frames_list[-1], frames_list[0] + 1),
            value=comp["frame_idx"],
            step=1,
            visible=True,
        )
    yield (
        banner,
        out_video_upd,
        _wrap_h(orig_img, img_h),
        _wrap_h(upscaled_img, img_h),
        keep,
        prev_video_upd,
        slider_upd,
        job_id,
    )


def refresh_job_queue(closed_ids: list[str] | None):
    """Refresh queue display and selector from backend."""
    try:
        text, choices, value = _queue_view(closed_ids)
        return text, gr.update(choices=choices, value=value)
    except Exception as exc:
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
    except Exception as exc:
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

                with gr.Accordion("📂 Load Previous Job", open=False):
                    prev_job_selector = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Select a previous job",
                        interactive=True,
                    )
                    with gr.Row():
                        refresh_prev_btn = gr.Button("🔄 Refresh list", size="sm")
                        load_prev_btn = gr.Button(
                            "📂 Load selected", size="sm", variant="secondary"
                        )

                gr.Markdown("### ⚙️ Settings")

                temporal_mode = gr.State(value="standard")

                preset = gr.Radio(
                    choices=[
                        "🔍 Clarity",
                        "💎 Best",
                        "🎯 BasicVSR++",
                        "🎬 FFmpeg Fast",
                    ],
                    value="💎 Best",
                    label="Preset",
                    info="Clarity: ESRGAN sharpen 1x  ·  Best: ESRGAN 4x  ·  BasicVSR++: temporal 4x  ·  FFmpeg Fast: no GPU",
                )

                noise_cb = gr.Checkbox(
                    label="Noise reduction",
                    value=False,
                    visible=True,
                )

                with gr.Group() as esrgan_controls:
                    model = gr.Dropdown(
                        choices=esrgan_models,
                        value=default_model,
                        label="Model",
                        info="x4plus = traditional ESRGAN",
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
                            label="Detail strength",
                            info="1.0 = sharpest  ·  0.0 = softest / denoised  (ESRGAN only)",
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
                        "inter-frame consistency are preserved.  Scale is fixed at **4x**.  "
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
            iw = heights["image_w"]  # noqa: F841 (kept for reference)
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

        # ── event: preset → model / scale / denoise / temporal_mode / group visibility ───
        def apply_preset(preset_name):
            # returns: model, outscale, denoise, temporal_mode, esrgan_controls, basicvsr_info, noise_cb
            show = gr.update(visible=True)
            hide = gr.update(visible=False)
            if preset_name == "🔍 Clarity":
                return (
                    "RealESRGAN_x4plus",
                    gr.update(value=1.0, visible=True),
                    1.0,
                    "standard",
                    show,
                    hide,
                    gr.update(value=False, visible=True),
                )
            if preset_name == "🎯 BasicVSR++":
                return (
                    "RealESRGAN_x4plus",
                    gr.update(value=4.0, visible=False),
                    1.0,
                    "basicvsr",
                    hide,
                    show,
                    gr.update(value=False, visible=False),
                )
            if preset_name == "🎬 FFmpeg Fast":
                return (
                    "ffmpeg-enhance",
                    gr.update(value=1.0, visible=False),
                    1.0,
                    "standard",
                    show,
                    hide,
                    gr.update(value=False, visible=False),
                )
            return (
                "RealESRGAN_x4plus",
                gr.update(value=4.0, visible=True),
                1.0,
                "standard",
                show,
                hide,
                gr.update(value=False, visible=True),
            )  # 💎 Best

        preset.change(
            apply_preset,
            inputs=[preset],
            outputs=[
                model,
                outscale,
                denoise,
                temporal_mode,
                esrgan_controls,
                basicvsr_info,
                noise_cb,
            ],
        )

        def on_noise_toggle(checked):
            return "tmix" if checked else "standard"

        noise_cb.change(
            on_noise_toggle,
            inputs=[noise_cb],
            outputs=[temporal_mode],
        )

        def on_select_frame(frame_idx, job_id):
            keep = gr.skip()
            if frame_idx is None or not job_id:
                return keep, keep, keep, keep
            try:
                idx = int(frame_idx)
            except Exception:
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
                gr.update(interactive=False),  # preset
                gr.update(interactive=False),  # model
                gr.update(interactive=False),  # outscale
                gr.update(interactive=False),  # denoise
                gr.update(interactive=False),  # tile
                gr.update(interactive=False),  # noise_cb
                gr.update(interactive=False, value="⏳ Processing…"),  # run_btn
                gr.update(interactive=True),  # refresh_queue_btn
                gr.update(interactive=True),  # cancel_selected_btn
                gr.update(interactive=True),  # close_selected_btn
            )

        def _enable_controls():
            return (
                gr.update(interactive=True),  # input_video
                gr.update(interactive=True),  # preset
                gr.update(interactive=True),  # model
                gr.update(interactive=True),  # outscale
                gr.update(interactive=True),  # denoise
                gr.update(interactive=True),  # tile
                gr.update(interactive=True),  # noise_cb
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
                preset,
                model,
                outscale,
                denoise,
                tile,
                noise_cb,
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
                preset,
                model,
                outscale,
                denoise,
                tile,
                noise_cb,
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
        demo.load(
            refresh_prev_jobs,
            inputs=None,
            outputs=[prev_job_selector],
            queue=False,
        )
        refresh_queue_btn.click(
            refresh_job_queue,
            inputs=[closed_jobs],
            outputs=[queue_display, queue_selector],
            queue=False,
        )
        refresh_prev_btn.click(
            refresh_prev_jobs,
            inputs=None,
            outputs=[prev_job_selector],
            queue=False,
        )
        load_prev_btn.click(
            load_previous_job,
            inputs=[prev_job_selector],
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
