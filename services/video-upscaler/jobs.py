"""
In-memory job manager for video upscaling.

Videos can take a long time to process, so the API accepts a job and returns
immediately with a job_id. A single background worker thread processes jobs
sequentially (the GPU can only run one upscale at a time), which keeps VRAM
usage predictable.

This module is intentionally free of torch/cv2 imports at module load so it can
be unit-tested without the heavy ML stack. The actual processing function is
injected, defaulting to pipeline.upscale_video.
"""

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from queue import Queue


@dataclass
class Job:
    id: str
    filename: str
    input_path: str
    output_path: str
    model: str
    outscale: float
    denoise: float
    tile: int | None
    status: str = "queued"  # queued | processing | done | error | cancelled
    progress: float = 0.0  # 0.0 - 1.0
    done_frames: int = 0
    total_frames: int = 0
    error: str | None = None
    result: dict | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    _cancel: bool = field(default=False, repr=False)

    def to_dict(self) -> dict:
        data = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        return data


class JobManager:
    """Thread-safe job queue with a single worker."""

    def __init__(self, processor: Callable | None = None, retention: int = 50):
        # processor(job, progress_cb, cancel_cb) -> result dict
        self._processor = processor
        self._jobs: dict[str, Job] = {}
        self._queue: Queue[str] = Queue()
        self._lock = threading.Lock()
        self._retention = retention
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # -- public API -------------------------------------------------------
    def submit(self, **kwargs) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], **kwargs)
        with self._lock:
            self._jobs[job.id] = job
            self._evict_locked()
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [j.to_dict() for j in jobs]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            if job.status == "queued":
                job.status = "cancelled"
                job.finished_at = time.time()
                return True
            if job.status == "processing":
                job._cancel = True
                return True
            return False

    # -- worker -----------------------------------------------------------
    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None or job.status == "cancelled":
                self._queue.task_done()
                continue

            job.status = "processing"
            job.started_at = time.time()

            def progress_cb(done: int, total: int, _job=job) -> None:
                _job.done_frames = done
                _job.total_frames = total
                _job.progress = (done / total) if total else 0.0

            def cancel_cb(_job=job) -> bool:
                return _job._cancel

            try:
                if self._processor is None:
                    raise RuntimeError("No processor configured")
                result = self._processor(job, progress_cb, cancel_cb)
                job.result = result
                job.progress = 1.0
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 - surface any failure to the client
                if job._cancel:
                    job.status = "cancelled"
                else:
                    job.status = "error"
                    job.error = str(exc)
            finally:
                job.finished_at = time.time()
                self._queue.task_done()

    def _evict_locked(self) -> None:
        """Drop oldest finished jobs beyond the retention limit."""
        if len(self._jobs) <= self._retention:
            return
        finished = sorted(
            (j for j in self._jobs.values() if j.finished_at),
            key=lambda j: j.finished_at,
        )
        excess = len(self._jobs) - self._retention
        for job in finished[:excess]:
            self._jobs.pop(job.id, None)
