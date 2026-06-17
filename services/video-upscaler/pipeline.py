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
import re
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


def _read_exact(stream, n: int, cancel_cb: Callable[[], bool] | None = None) -> bytes | None:
    """
    Read exactly n bytes from a stream; return None at end-of-stream or on cancellation.

    Reads in small chunks to allow frequent cancellation checks.
    """
    buf = bytearray()
    chunk_size = min(65536, n)  # Read 64KB at a time to allow cancellation checks

    while len(buf) < n:
        # Check for cancellation before each chunk
        if cancel_cb and cancel_cb():
            return None  # Signal cancellation as EOF

        # Read next chunk (up to remaining bytes needed)
        to_read = min(chunk_size, n - len(buf))
        chunk = stream.read(to_read)
        if not chunk:
            return None  # EOF (a trailing partial frame is discarded)
        buf.extend(chunk)
    return bytes(buf)


def _resize_for_preview(img: np.ndarray, max_w: int = 640) -> np.ndarray:
    """Downscale an image to at most ``max_w`` wide for cheap preview transfer."""
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / float(w)
    return cv2.resize(img, (max_w, round(h * scale)), interpolation=cv2.INTER_AREA)


def _clarity_enhance(upsampler, frame: np.ndarray, outscale: float) -> np.ndarray:
    """Run the model at 4x, sharpen in the high-res domain, then Lanczos-downsample.

    When outscale <= 1 the caller wants maximum AI-assisted clarity at the
    original resolution.  Sharpening *before* the final downsample retains far
    more fine detail than letting ESRGAN's own INTER_AREA resize do the job.
    """
    src_h, src_w = frame.shape[:2]
    # Always run at 4x so the AI operates at maximum detail.
    hires, _ = upsampler.enhance(frame, outscale=4.0)

    # --- unsharp masking in 4x space ----------------------------------------
    # sigma=2.5 at 4x ≈ sigma=0.6 at 1x → sharpens features at sub-pixel scale
    # without amplifying noise or creating halos.
    blurred = cv2.GaussianBlur(hires, (0, 0), sigmaX=2.5, sigmaY=2.5)
    hires = cv2.addWeighted(hires, 1.35, blurred, -0.35, 0)
    hires = np.clip(hires, 0, 255).astype(np.uint8)

    # --- high-quality downsample to target resolution -----------------------
    target_w = max(1, round(src_w * outscale))
    target_h = max(1, round(src_h * outscale))
    if hires.shape[1] != target_w or hires.shape[0] != target_h:
        hires = cv2.resize(hires, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    return hires


def _extract_comparison_frames(
    input_path: str,
    output_path: str,
    preview_dir: str,
    duration: float,
    count: int = 5,
) -> None:
    """Extract before/after JPEG pairs at evenly-spaced timestamps.

    Written as ``orig_{idx:06d}.jpg`` / ``upscaled_{idx:06d}.jpg`` so the
    existing comparison-frame API endpoints can serve them without changes.
    """
    if duration <= 0 or not os.path.exists(output_path):
        return
    fracs = [0.1, 0.3, 0.5, 0.7, 0.9][:count]
    for i, frac in enumerate(fracs, start=1):
        t = duration * frac
        idx_str = f"{i * 5:06d}"  # multiples of 5 → matches idx % 5 == 0 convention
        for src, prefix in [(input_path, "orig"), (output_path, "upscaled")]:
            out_jpg = os.path.join(preview_dir, f"{prefix}_{idx_str}.jpg")
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-ss",
                        str(t),
                        "-i",
                        src,
                        "-vframes",
                        "1",
                        "-vf",
                        "scale=640:-2",  # 640 px wide, even height
                        "-q:v",
                        "3",
                        "-y",
                        out_jpg,
                    ],
                    check=True,
                    timeout=15,
                )
            except Exception:
                pass


def enhance_video_ffmpeg(
    input_path: str,
    output_path: str,
    progress_cb=None,
    cancel_cb=None,
    preview_path: str | None = None,
    preview_video_dir: str | None = None,
    preview_video_path: str | None = None,
) -> dict:
    """Enhance clarity using FFmpeg filters only — no AI, very fast.

    Pipeline: ``hqdn3d`` (temporal + spatial denoise) → ``unsharp`` (luma USM).
    Progress is reported via FFmpeg's ``-progress pipe:1`` output.
    After processing, before/after comparison frames are extracted from both
    the source and output files.
    """
    info = probe_video(input_path)
    total: int = info.get("nb_frames", 0) or 0
    width: int = info.get("width", 0)
    height: int = info.get("height", 0)
    duration: float = info.get("duration", 0.0)

    if progress_cb:
        progress_cb(0, total)

    # hqdn3d: 4 params = luma_spatial:chroma_spatial:luma_tmp:chroma_tmp
    vf = "hqdn3d=4:4:4:4,unsharp=5:5:1.0:5:5:0.0,format=yuv420p"
    audio_opts = ["-c:a", "copy"] if info.get("has_audio") else ["-an"]
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        *audio_opts,
        "-progress",
        "pipe:1",
        "-y",
        output_path,
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    frame_count = 0
    try:
        for line in proc.stdout:
            if cancel_cb and cancel_cb():
                proc.kill()
                raise RuntimeError("Job cancelled")
            line = line.strip()
            if line.startswith("frame="):
                try:
                    frame_count = int(line.split("=", 1)[1])
                except ValueError:
                    pass
                if progress_cb:
                    progress_cb(frame_count, total or frame_count)
        proc.wait()
        if proc.returncode not in (0, None):
            raise RuntimeError(f"FFmpeg exited with code {proc.returncode}")
    except RuntimeError:
        if proc.poll() is None:
            proc.kill()
        raise

    # Extract before/after comparison frames from source + output.
    if preview_video_dir and os.path.isdir(preview_video_dir):
        _extract_comparison_frames(input_path, output_path, preview_video_dir, duration)

    if progress_cb and total:
        progress_cb(total, total)

    return {
        "output_path": output_path,
        "frames": frame_count or total,
        "model": "ffmpeg-enhance",
        "outscale": 1.0,
        "width": width,
        "height": height,
    }


def _start_encoder(
    width: int,
    height: int,
    fps: float,
    input_path: str,
    has_audio: bool,
    output_path: str,
    hqdn3d_tmp: int = 4,
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
    # crf 14 + medium preset: visually-lossless, keeps the AI-added high-frequency
    # detail, 2-3x faster encoding than slow with minimal quality difference.
    # hqdn3d temporal-only denoising (no spatial blur): removes frame-to-frame
    # inconsistency (flicker/shimmer) that frame-independent ESRGAN introduces while
    # preserving the spatial resolution gains.  Standard+Smooth uses strength 8
    # (vs Standard's 4) instead of tmix, which caused ghosting on moving subjects.
    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "14",
        "-vf",
        f"hqdn3d=0:0:{hqdn3d_tmp}:{hqdn3d_tmp},format=yuv420p",
    ]
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
    except Exception as exc:
        logger.debug("Preview write failed: %s", exc)


def _generate_preview_video(
    preview_frames_dir: str,
    output_path: str,
    fps: float,
    max_frames: int = 90,
) -> None:
    """Generate a short preview video from the last N frames (best-effort)."""
    try:
        frames = sorted(
            [
                f
                for f in os.listdir(preview_frames_dir)
                if f.startswith("frame_") and f.endswith(".jpg")
            ],
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
        if not frames:
            return
        # Use the last N frames for a ~3-5 second clip.
        frames = frames[-max_frames:]
        if len(frames) < 10:  # Need at least 10 frames for a meaningful clip.
            return

        # Read first frame to get dimensions.
        first = cv2.imread(os.path.join(preview_frames_dir, frames[0]))
        if first is None:
            return
        _h, _w = first.shape[:2]

        # Encode frames into a short MP4 (crf 23 for small size, preset fast).
        vf_args: list[str] = []

        tmp = f"{output_path}.tmp.mp4"
        proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "image2pipe",
                "-framerate",
                str(fps),
                "-i",
                "pipe:0",
                *vf_args,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                tmp,
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        for frame_name in frames:
            frame_path = os.path.join(preview_frames_dir, frame_name)
            with open(frame_path, "rb") as f:
                proc.stdin.write(f.read())
        proc.stdin.close()
        if proc.wait() == 0:
            os.replace(tmp, output_path)
        else:
            logger.debug("Preview video encoding failed")
    except Exception as exc:
        logger.debug("Preview video generation failed: %s", exc)


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
    preview_video_dir: str | None = None,
    preview_video_path: str | None = None,
    temporal_mode: str = "standard",
) -> dict:
    """
    Upscale a video file.

    temporal_mode:
      "standard" - Real-ESRGAN frame-by-frame (default)
      "tmix"     - Real-ESRGAN then ffmpeg tmix temporal-smoothing pass
      "basicvsr" - BasicVSR++ temporal-aware SR (4x fixed scale)
    """
    if temporal_mode == "basicvsr":
        return _upscale_video_basicvsr(
            input_path=input_path,
            output_path=output_path,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
            preview_path=preview_path,
            preview_video_dir=preview_video_dir,
            preview_video_path=preview_video_path,
        )

    result = _upscale_video_esrgan(
        input_path=input_path,
        output_path=output_path,
        model_name=model_name,
        outscale=outscale,
        denoise=denoise,
        tile=tile,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
        preview_path=preview_path,
        preview_video_dir=preview_video_dir,
        preview_video_path=preview_video_path,
        # "tmix" (Standard+Smooth) uses stronger temporal denoising instead of
        # ghosting tmix filter: hqdn3d strength 8 vs Standard's 4.
        hqdn3d_tmp=8 if temporal_mode == "tmix" else 4,
    )

    return result


# ── Real-ESRGAN pipeline (original, renamed) ──────────────────────────────────


def _upscale_video_esrgan(
    input_path: str,
    output_path: str,
    model_name: str,
    outscale: float = 4.0,
    denoise: float = 1.0,
    tile: int | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
    preview_path: str | None = None,
    preview_video_dir: str | None = None,
    preview_video_path: str | None = None,
    hqdn3d_tmp: int = 4,
) -> dict:
    """Real-ESRGAN frame-by-frame upscaling pipeline."""
    info = probe_video(input_path)
    fps = info["fps"]

    # Best-effort total frame count for progress reporting.
    total = info.get("nb_frames") or 0
    if total <= 0 and info.get("duration", 0) > 0:
        total = round(info["duration"] * fps)

    # True (display-oriented) frame size, so raw decoding lines up byte-for-byte.
    width, height = _first_frame_size(input_path)
    frame_bytes = width * height * 3

    # Auto-tile large frames so that enhance() returns in reasonable time and
    # cancellation checks between frames stay responsive.  Without tiling a single
    # 1080p frame can block the thread for 30-120 s, making Cancel unresponsive.
    AUTO_TILE_PIXELS = 512 * 512  # anything larger than ~512x512 gets auto-tiled
    AUTO_TILE_SIZE = 512
    if not tile and (width * height) > AUTO_TILE_PIXELS:
        tile = AUTO_TILE_SIZE
        logger.debug(
            "Auto-tiling: frame %dx%d > %d px, using tile=%d",
            width,
            height,
            AUTO_TILE_PIXELS,
            AUTO_TILE_SIZE,
        )

    upsampler, __meta = get_upsampler(model_name, denoise=denoise, tile=tile)

    logger.info(
        "Streaming upscale: %dx%d @ %.3ffps, ~%d frames (model=%s, outscale=%s)",
        width,
        height,
        fps,
        total,
        model_name,
        outscale,
    )

    # Report total frame count immediately so UI can show "0/N frames" instead of "preparing..."
    if progress_cb:
        progress_cb(0, total)

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
    _preview_frame_window: list[str] = []  # rolling window tracked without os.listdir
    _preview_frame_keep = max(30, int(fps * 3))

    try:
        while True:
            # Check for cancellation and kill subprocesses to unblock reads
            if cancel_cb and cancel_cb():
                for proc in (decoder, encoder):
                    if proc is not None and proc.poll() is None:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                raise RuntimeError("Job cancelled")

            buf = _read_exact(decoder.stdout, frame_bytes, cancel_cb)
            if buf is None:
                break  # end of stream or cancelled

            frame = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3).copy()
            if outscale <= 1.0:
                output = _clarity_enhance(upsampler, frame, outscale)
            else:
                output, _ = upsampler.enhance(frame, outscale=outscale)
            idx += 1

            if encoder is None:
                oh, ow = output.shape[:2]
                encoder = _start_encoder(
                    ow, oh, fps, input_path, info["has_audio"], output_path, hqdn3d_tmp=hqdn3d_tmp
                )

            encoder.stdin.write(np.ascontiguousarray(output).tobytes())

            # Live preview JPEG + comparison frames (best-effort, cheap).
            if preview_path and (idx == 1 or idx % 5 == 0):
                _write_preview(output, preview_path)
                # Save original and upscaled side-by-side for comparison.
                if preview_video_dir:
                    # Downscale to a small preview size so the base64 payload sent
                    # to the UI stays tiny (full-res frames would choke the connection).
                    orig_small = _resize_for_preview(frame, max_w=640)
                    upscaled_small = _resize_for_preview(output, max_w=640)
                    # Save original frame.
                    orig_path = os.path.join(preview_video_dir, f"orig_{idx:06d}.jpg")
                    ok, buf = cv2.imencode(".jpg", orig_small, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        with open(orig_path, "wb") as f:
                            f.write(buf.tobytes())
                    # Save upscaled frame.
                    upscaled_path = os.path.join(preview_video_dir, f"upscaled_{idx:06d}.jpg")
                    ok, buf = cv2.imencode(".jpg", upscaled_small, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        with open(upscaled_path, "wb") as f:
                            f.write(buf.tobytes())
                    # Keep only the last 15 comparison pairs.
                    all_orig = sorted(
                        [f for f in os.listdir(preview_video_dir) if f.startswith("orig_")],
                        key=lambda x: int(x.split("_")[1].split(".")[0]),
                    )
                    if len(all_orig) > 15:
                        for old_frame in all_orig[:-15]:
                            frame_num = old_frame.split("_")[1].split(".")[0]
                            os.remove(os.path.join(preview_video_dir, f"orig_{frame_num}.jpg"))
                            upscaled_file = os.path.join(
                                preview_video_dir, f"upscaled_{frame_num}.jpg"
                            )
                            if os.path.exists(upscaled_file):
                                os.remove(upscaled_file)

            # Rolling preview video: downscale then save, regenerate clip every 30 frames.
            if preview_video_dir and preview_video_path:
                frame_path = os.path.join(preview_video_dir, f"frame_{idx:06d}.jpg")
                # Downscale to at most 960px wide — preview clip doesn't need full 4K resolution.
                preview_frame = _resize_for_preview(output, max_w=960)
                ok, buf = cv2.imencode(".jpg", preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with open(frame_path, "wb") as f:
                        f.write(buf.tobytes())
                    # Rolling window: track saved paths in a list, no os.listdir needed.
                    _preview_frame_window.append(frame_path)
                    while len(_preview_frame_window) > _preview_frame_keep:
                        old = _preview_frame_window.pop(0)
                        try:
                            os.remove(old)
                        except OSError:
                            pass
                # Regenerate preview video every 30 frames at original fps for smooth playback.
                if idx % 30 == 0:
                    _generate_preview_video(
                        preview_video_dir,
                        preview_video_path,
                        fps,
                    )

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
                    proc.wait()  # Reap the zombie to prevent defunct processes
                except Exception:
                    pass


# ── BasicVSR++ temporal pipeline ─────────────────────────────────────────────

_BASICVSR_CHUNK = 4  # frames per temporal chunk — small keeps VRAM under control
# Max input pixels for BasicVSR++ before auto-downscale.
# At 720p (921 k px) a chunk of 4 fits safely in 16 GB VRAM with fp16 autocast.
_BASICVSR_MAX_PIXELS = 1280 * 720
_BASICVSR_MODEL_DIR = os.environ.get("WEIGHTS_DIR", "/app/weights")
_BASICVSR_CKPT = os.path.join(
    _BASICVSR_MODEL_DIR,
    "basicvsr_plusplus_c64n7_8x1_600k_reds4.pth",
)
_BASICVSR_SPYNET = os.path.join(
    _BASICVSR_MODEL_DIR,
    "spynet_20210409-c6c1bd09.pth",
)

_basicvsr_model = None  # singleton, loaded on first use


def _get_basicvsr_model():
    global _basicvsr_model
    if _basicvsr_model is not None:
        return _basicvsr_model
    import torch
    from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus

    logger.info("Loading BasicVSR++ weights from %s", _BASICVSR_CKPT)
    # Pass spynet_path=None so basicsr doesn't try to load it (it always expects
    # a {'params': ...} wrapper which not all checkpoint files provide).
    model = BasicVSRPlusPlus(
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=True,
        spynet_path=None,
    )
    # Load SPyNet separately — handles both raw OrderedDict and {'params': sd}
    # and remaps the '.conv.' key infix used by newer MMEditing checkpoints.
    logger.info("Loading SPyNet weights from %s", _BASICVSR_SPYNET)
    spynet_ckpt = torch.load(_BASICVSR_SPYNET, map_location="cpu", weights_only=False)
    if isinstance(spynet_ckpt, dict) and "params" in spynet_ckpt:
        spynet_ckpt = spynet_ckpt["params"]
    # The mmediting checkpoint uses ConvModule wrappers (.conv. infix) and stores
    # ConvModule blocks at sequential indices 0,1,2,3,4 (ReLUs not counted).
    # basicsr's SpyNet uses nn.Sequential with interleaved ReLUs, so the 5 conv
    # layers sit at positions 0,2,4,6,8.  Fix in two steps:
    # Step 1: strip the ConvModule .conv. infix
    spynet_ckpt = {k.replace(".conv.", "."): v for k, v in spynet_ckpt.items()}
    # Step 2: remap ConvModule index N -> basicsr sequential index 2*N
    remapped: dict = {}
    for k, v in spynet_ckpt.items():
        m = re.match(r"^(basic_module\.\d+\.basic_module\.)(\d+)\.(weight|bias)$", k)
        if m:
            prefix, idx, suffix = m.groups()
            remapped[f"{prefix}{int(idx) * 2}.{suffix}"] = v
        else:
            remapped[k] = v
    model.spynet.load_state_dict(remapped, strict=False)

    state = torch.load(_BASICVSR_CKPT, map_location="cpu", weights_only=False)
    state = state.get("params_ema") or state.get("params") or state
    model.load_state_dict(state, strict=False)
    _basicvsr_model = model.cuda().eval()
    logger.info("BasicVSR++ ready")
    return _basicvsr_model


def _upscale_video_basicvsr(
    input_path: str,
    output_path: str,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
    preview_path: str | None = None,
    preview_video_dir: str | None = None,
    preview_video_path: str | None = None,
) -> dict:
    """BasicVSR++ temporal-aware 4x upscaling pipeline."""
    import torch

    info = probe_video(input_path)
    fps = info["fps"]
    total = info.get("nb_frames") or 0
    if total <= 0 and info.get("duration", 0) > 0:
        total = round(info["duration"] * fps)

    width, height = _first_frame_size(input_path)

    # ── auto-downscale oversized inputs ────────────────────────────────────
    # BasicVSR++ intermediate feature maps grow with input area.  Above ~720p
    # a chunk of 4 frames exceeds 16 GB VRAM in fp32; fp16 autocast helps but
    # the safe ceiling is still ~720p.  Downscale here so the 4x output is
    # still larger than the original (e.g. 1080p input → 720p → 2880p out).
    vf_scale: list[str] = []
    if width * height > _BASICVSR_MAX_PIXELS:
        scale = (_BASICVSR_MAX_PIXELS / (width * height)) ** 0.5
        safe_w = int(width * scale) & ~1  # ensure even
        safe_h = int(height * scale) & ~1
        logger.warning(
            "Input %dx%d exceeds BasicVSR++ VRAM budget — downscaling to %dx%d before SR",
            width,
            height,
            safe_w,
            safe_h,
        )
        vf_scale = ["-vf", f"scale={safe_w}:{safe_h}"]
        width, height = safe_w, safe_h

    frame_bytes = width * height * 3

    logger.info(
        "BasicVSR++ upscale: %dx%d @ %.3ffps, ~%d frames",
        width,
        height,
        fps,
        total,
    )

    if progress_cb:
        progress_cb(0, total)

    model = _get_basicvsr_model()
    out_h, out_w = height * 4, width * 4

    # ── stream frames into chunks — never buffer the full video ────────────
    # Only _BASICVSR_CHUNK frames live in RAM at a time.
    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            input_path,
            *vf_scale,
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
    done = 0
    frames_read = 0
    chunk: list[np.ndarray] = []

    def _run_chunk(frames: list[np.ndarray]) -> None:
        """Infer one temporal chunk and pipe upscaled frames to the encoder."""
        nonlocal encoder, done
        tensors = []
        for f in frames:
            t = torch.from_numpy(f[:, :, ::-1].copy()).float() / 255.0  # RGB float
            tensors.append(t.permute(2, 0, 1))  # CHW
        seq = torch.stack(tensors).unsqueeze(0).cuda()  # (1,T,C,H,W)
        with torch.no_grad(), torch.autocast("cuda"):
            out_seq = model(seq).float().squeeze(0).cpu()  # (T,C,H*4,W*4) in fp32
        torch.cuda.empty_cache()
        for i, out_t in enumerate(out_seq):
            out_np = (out_t.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
            out_bgr = out_np[:, :, ::-1]
            if encoder is None:
                encoder = _start_encoder(
                    out_w, out_h, fps, input_path, info["has_audio"], output_path
                )
            encoder.stdin.write(np.ascontiguousarray(out_bgr).tobytes())
            done += 1
            if preview_path and (done == 1 or done % 5 == 0):
                _write_preview(out_bgr, preview_path)
                if preview_video_dir:
                    # Keep comparison payloads small for UI transport.
                    orig_small = _resize_for_preview(frames[i], max_w=640)
                    upscaled_small = _resize_for_preview(out_bgr, max_w=640)

                    orig_path = os.path.join(preview_video_dir, f"orig_{done:06d}.jpg")
                    ok, buf = cv2.imencode(".jpg", orig_small, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        with open(orig_path, "wb") as f:
                            f.write(buf.tobytes())

                    upscaled_path = os.path.join(preview_video_dir, f"upscaled_{done:06d}.jpg")
                    ok, buf = cv2.imencode(".jpg", upscaled_small, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        with open(upscaled_path, "wb") as f:
                            f.write(buf.tobytes())

                    # Keep only the last 15 comparison pairs.
                    all_orig = sorted(
                        [f for f in os.listdir(preview_video_dir) if f.startswith("orig_")],
                        key=lambda x: int(x.split("_")[1].split(".")[0]),
                    )
                    if len(all_orig) > 15:
                        for old_frame in all_orig[:-15]:
                            frame_num = old_frame.split("_")[1].split(".")[0]
                            os.remove(os.path.join(preview_video_dir, f"orig_{frame_num}.jpg"))
                            upscaled_file = os.path.join(
                                preview_video_dir, f"upscaled_{frame_num}.jpg"
                            )
                            if os.path.exists(upscaled_file):
                                os.remove(upscaled_file)

            # Rolling preview video: save every upscaled frame, regenerate clip every 15.
            if preview_video_dir:
                frame_path = os.path.join(preview_video_dir, f"frame_{done:06d}.jpg")
                ok, buf = cv2.imencode(".jpg", out_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ok:
                    with open(frame_path, "wb") as fh:
                        fh.write(buf.tobytes())
                    # Keep only the last 3 seconds worth of frames.
                    keep = max(30, int(fps * 3))
                    all_frames = sorted(
                        [
                            ff
                            for ff in os.listdir(preview_video_dir)
                            if ff.startswith("frame_") and ff.endswith(".jpg")
                        ],
                        key=lambda x: int(x.split("_")[1].split(".")[0]),
                    )
                    if len(all_frames) > keep:
                        for old in all_frames[:-keep]:
                            os.remove(os.path.join(preview_video_dir, old))
                if preview_video_path and done % 15 == 0:
                    _generate_preview_video(
                        preview_video_dir,
                        preview_video_path,
                        fps,
                    )

            if progress_cb:
                progress_cb(done, total or frames_read)

    try:
        while True:
            if cancel_cb and cancel_cb():
                raise RuntimeError("Job cancelled")
            buf = _read_exact(decoder.stdout, frame_bytes, cancel_cb)
            if buf is None:
                break
            chunk.append(np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3).copy())
            frames_read += 1
            if len(chunk) >= _BASICVSR_CHUNK:
                _run_chunk(chunk)
                chunk = []

        # flush any remaining frames (last partial chunk)
        if chunk:
            _run_chunk(chunk)

        if not done:
            raise RuntimeError("No frames decoded from input video.")

        if encoder is not None:
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("FFmpeg encoder failed.")

        out_info = probe_video(output_path)
        return {
            "output_path": output_path,
            "frames": done,
            "model": "basicvsr_plusplus",
            "outscale": 4.0,
            "source_resolution": f"{info['width']}x{info['height']}",
            "output_resolution": f"{out_info['width']}x{out_info['height']}",
            "fps": round(fps, 3),
        }
    finally:
        if decoder.poll() is None:
            try:
                decoder.kill()
                decoder.wait()
            except Exception:
                pass
        if encoder is not None and encoder.poll() is None:
            try:
                encoder.kill()
                encoder.wait()
            except Exception:
                pass
