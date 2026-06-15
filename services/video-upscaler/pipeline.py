"""
Video upscaling pipeline using FFmpeg + Real-ESRGAN.

Steps (offline, quality-first):
  1. Probe input for fps / audio.
  2. Extract frames to PNG with FFmpeg.
  3. Upscale each frame with Real-ESRGAN (GPU).
  4. Reassemble frames into H.264 video, muxing the original audio back in.

Progress is reported through a callback so the job manager can expose it.
"""

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable

import cv2

from log_setup import setup_logging
from upscaler_model import get_upsampler

logger = setup_logging(__name__)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a subprocess, raising with captured stderr on failure."""
    logger.debug("RUN: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({' '.join(cmd[:2])}): {proc.stderr.strip()}")
    return proc


def _write_preview(img, path: str, max_w: int = 640) -> None:
    """Atomically write a small JPEG preview of the latest upscaled frame."""
    try:
        h, w = img.shape[:2]
        if w > max_w:
            scale = max_w / w
            img = cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
        tmp = f"{path}.tmp"
        cv2.imwrite(tmp, img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 - preview is best-effort, never fail the job
        logger.debug("Preview write failed: %s", exc)


def probe_video(path: str) -> dict:
    """Return basic video info: fps, has_audio, width, height, nb_frames (best effort)."""
    proc = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            path,
        ]
    )
    data = json.loads(proc.stdout)
    info = {"fps": 30.0, "has_audio": False, "width": 0, "height": 0, "nb_frames": 0}

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            # r_frame_rate is like "30000/1001"
            rate = stream.get("r_frame_rate", "30/1")
            try:
                num, den = rate.split("/")
                info["fps"] = float(num) / float(den) if float(den) else 30.0
            except (ValueError, ZeroDivisionError):
                info["fps"] = 30.0
            info["width"] = int(stream.get("width", 0))
            info["height"] = int(stream.get("height", 0))
            try:
                info["nb_frames"] = int(stream.get("nb_frames", 0))
            except (TypeError, ValueError):
                info["nb_frames"] = 0
        elif stream.get("codec_type") == "audio":
            info["has_audio"] = True

    return info


def upscale_video(
    input_path: str,
    output_path: str,
    model_name: str,
    outscale: float = 4.0,
    denoise: float = 1.0,
    tile: int | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
    preview_path: str | None = None,
) -> dict:
    """
    Upscale a video file.

    Args:
        input_path: source video.
        output_path: destination .mp4.
        model_name: Real-ESRGAN model key.
        outscale: final scale factor relative to source (e.g. 2.0, 4.0).
        denoise: 0.0-1.0 denoise strength (general-v3 only).
        tile: tile size for VRAM control.
        progress_cb: called with (done_frames, total_frames).
        cancel_cb: returns True if the job should abort.
        preview_path: if set, a small JPEG of the latest upscaled frame is
            written here periodically so clients can show a live preview.

    Returns:
        dict with output info.
    """
    info = probe_video(input_path)
    fps = info["fps"]

    upsampler, meta = get_upsampler(model_name, denoise=denoise, tile=tile)

    work = tempfile.mkdtemp(prefix="upscale_")
    frames_in = os.path.join(work, "in")
    frames_out = os.path.join(work, "out")
    os.makedirs(frames_in, exist_ok=True)
    os.makedirs(frames_out, exist_ok=True)

    try:
        # 1. Extract frames
        logger.info("Extracting frames from %s", input_path)
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                input_path,
                "-qscale:v",
                "1",
                "-qmin",
                "1",
                os.path.join(frames_in, "frame_%08d.png"),
            ]
        )

        frame_files = sorted(f for f in os.listdir(frames_in) if f.endswith(".png"))
        total = len(frame_files)
        if total == 0:
            raise RuntimeError("No frames were extracted from the input video.")

        logger.info("Upscaling %d frames (model=%s, outscale=%s)", total, model_name, outscale)

        # 2. Upscale each frame
        for idx, fname in enumerate(frame_files, start=1):
            if cancel_cb and cancel_cb():
                raise RuntimeError("Job cancelled")

            img = cv2.imread(os.path.join(frames_in, fname), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Failed to read frame {fname}")

            output, _ = upsampler.enhance(img, outscale=outscale)
            cv2.imwrite(os.path.join(frames_out, fname), output)

            # Update the live preview every few frames (best-effort, cheap).
            if preview_path and (idx == 1 or idx % 5 == 0):
                _write_preview(output, preview_path)

            if progress_cb:
                progress_cb(idx, total)

        # 3. Reassemble video (+ mux original audio if present)
        logger.info("Reassembling video -> %s", output_path)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            f"{fps}",
            "-i",
            os.path.join(frames_out, "frame_%08d.png"),
        ]
        if info["has_audio"]:
            cmd += ["-i", input_path]
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
        ]
        if info["has_audio"]:
            cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += [output_path]
        _run(cmd)

        out_info = probe_video(output_path)
        return {
            "output_path": output_path,
            "frames": total,
            "model": model_name,
            "outscale": outscale,
            "source_resolution": f"{info['width']}x{info['height']}",
            "output_resolution": f"{out_info['width']}x{out_info['height']}",
            "fps": round(fps, 3),
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)
