"""
Video upscaling pipeline using FFmpeg + Real-ESRGAN.

Streaming design (low latency, CPU/GPU overlap):
  1. Probe input for fps / audio / size.
  2. Decode frames straight from FFmpeg's stdout (raw BGR, no PNGs on disk).
  3. Upscale each frame with Real-ESRGAN (GPU) as it arrives.
  4. Pipe each upscaled frame into an FFmpeg encoder (H.264 + original audio).

Decode (CPU), upscale (GPU) and encode (CPU) all run concurrently, so the
GPU starts working within seconds instead of waiting for a full disk
extraction pass. Progress is reported through a callback.
"""

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable

import cv2
import numpy as np

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


def _first_frame_size(path: str) -> tuple[int, int]:
    """Decode a single frame to learn the true (display-oriented) width/height."""
    tmp = tempfile.mkdtemp(prefix="probe_")
    fp = os.path.join(tmp, "f.png")
    try:
        _run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path, "-frames:v", "1", fp]
        )
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Could not decode the first frame.")
        h, w = img.shape[:2]
        return w, h
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _read_exact(stream, n: int) -> bytes | None:
    """Read exactly n bytes from a stream; return None at end-of-stream."""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None  # EOF (a trailing partial frame is discarded)
        buf.extend(chunk)
    return bytes(buf)


def _start_encoder(
    width: int, height: int, fps: float, input_path: str, has_audio: bool, output_path: str
) -> subprocess.Popen:
    """Launch an FFmpeg encoder that reads raw BGR frames from stdin."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-framerate",
        f"{fps}",
        "-i",
        "pipe:0",
    ]
    if has_audio:
        cmd += ["-i", input_path]
    # crf 14 + slow preset: visually-lossless, keeps the AI-added high-frequency
    # detail (larger files, which is acceptable here).
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p"]
    if has_audio:
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += [output_path]
    logger.debug("ENCODER: %s", " ".join(cmd))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def _write_preview(img, path: str, max_w: int = 960) -> None:
    """Atomically write a small JPEG preview of the latest upscaled frame."""
    try:
        h, w = img.shape[:2]
        if w > max_w:
            scale = max_w / w
            img = cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
        # Encode explicitly (cv2.imwrite infers the format from the file
        # extension, which would break on the ".tmp" temp filename below).
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            f.write(buf.tobytes())
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
            "-show_format",
            path,
        ]
    )
    data = json.loads(proc.stdout)
    info = {
        "fps": 30.0,
        "has_audio": False,
        "width": 0,
        "height": 0,
        "nb_frames": 0,
        "duration": 0.0,
    }

    try:
        info["duration"] = float(data.get("format", {}).get("duration", 0.0))
    except (TypeError, ValueError):
        info["duration"] = 0.0

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

    # Best-effort total frame count for progress reporting.
    total = info.get("nb_frames") or 0
    if total <= 0 and info.get("duration", 0) > 0:
        total = int(round(info["duration"] * fps))

    upsampler, meta = get_upsampler(model_name, denoise=denoise, tile=tile)

    # True (display-oriented) frame size, so raw decoding lines up byte-for-byte.
    width, height = _first_frame_size(input_path)
    frame_bytes = width * height * 3

    logger.info(
        "Streaming upscale: %dx%d @ %.3ffps, ~%d frames (model=%s, outscale=%s)",
        width,
        height,
        fps,
        total,
        model_name,
        outscale,
    )

    # Decoder: raw BGR frames straight from FFmpeg's stdout (no PNGs on disk).
    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            input_path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=frame_bytes,
    )
    encoder: subprocess.Popen | None = None
    idx = 0

    try:
        while True:
            if cancel_cb and cancel_cb():
                raise RuntimeError("Job cancelled")

            buf = _read_exact(decoder.stdout, frame_bytes)
            if buf is None:
                break  # end of stream

            frame = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3).copy()
            output, _ = upsampler.enhance(frame, outscale=outscale)
            idx += 1

            if encoder is None:
                oh, ow = output.shape[:2]
                encoder = _start_encoder(ow, oh, fps, input_path, info["has_audio"], output_path)

            encoder.stdin.write(np.ascontiguousarray(output).tobytes())

            # Live preview (best-effort, cheap).
            if preview_path and (idx == 1 or idx % 5 == 0):
                _write_preview(output, preview_path)

            if progress_cb:
                progress_cb(idx, total or idx)

        if idx == 0:
            raise RuntimeError("No frames were decoded from the input video.")

        # Flush and finalize the encoder.
        if encoder is not None:
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("FFmpeg encoder failed while writing the output video.")

        if decoder.wait() not in (0, None):
            logger.warning("Decoder exited with code %s", decoder.returncode)

        out_info = probe_video(output_path)
        return {
            "output_path": output_path,
            "frames": idx,
            "model": model_name,
            "outscale": outscale,
            "source_resolution": f"{info['width']}x{info['height']}",
            "output_resolution": f"{out_info['width']}x{out_info['height']}",
            "fps": round(fps, 3),
        }
    finally:
        # Make sure no FFmpeg process is left running on cancel/error.
        for proc in (decoder, encoder):
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
