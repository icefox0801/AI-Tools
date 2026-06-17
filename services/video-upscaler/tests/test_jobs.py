"""Unit tests for the JobManager (no torch/cv2/GPU required)."""

import sys
import time
from pathlib import Path

# Make the service modules importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jobs import JobManager


def _submit(manager, **overrides):
    params = {
        "filename": "clip.mp4",
        "input_path": "/tmp/in.mp4",
        "output_path": "/tmp/out.mp4",
        "model": "realesr-general-x4v3",
        "outscale": 4.0,
        "denoise": 1.0,
        "tile": 512,
    }
    params.update(overrides)
    return manager.submit(**params)


def _wait_for(manager, job_id, statuses, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get(job_id)
        if job and job.status in statuses:
            return job
        time.sleep(0.02)
    return manager.get(job_id)


def test_successful_job_reports_progress_and_result():
    def processor(job, progress_cb, cancel_cb):
        progress_cb(5, 10)
        progress_cb(10, 10)
        return {"output_path": job.output_path, "frames": 10}

    manager = JobManager(processor=processor)
    job = _submit(manager)

    done = _wait_for(manager, job.id, {"done", "error"})
    assert done.status == "done"
    assert done.progress == 1.0
    assert done.result == {"output_path": "/tmp/out.mp4", "frames": 10}
    assert done.total_frames == 10


def test_failed_job_captures_error():
    def processor(job, progress_cb, cancel_cb):
        raise ValueError("boom")

    manager = JobManager(processor=processor)
    job = _submit(manager)

    failed = _wait_for(manager, job.id, {"error", "done"})
    assert failed.status == "error"
    assert "boom" in failed.error


def test_cancel_running_job():
    started = []

    def processor(job, progress_cb, cancel_cb):
        started.append(True)
        for i in range(1000):
            if cancel_cb():
                raise RuntimeError("Job cancelled")
            progress_cb(i, 1000)
            time.sleep(0.005)
        return {"frames": 1000}

    manager = JobManager(processor=processor)
    job = _submit(manager)

    # Wait until it actually starts, then cancel
    _wait_for(manager, job.id, {"processing"})
    assert manager.cancel(job.id) is True

    cancelled = _wait_for(manager, job.id, {"cancelled", "error", "done"})
    assert cancelled.status == "cancelled"


def test_to_dict_excludes_private_fields():
    manager = JobManager(processor=lambda j, p, c: {})
    job = _submit(manager)
    data = job.to_dict()
    assert "_cancel" not in data
    assert data["id"] == job.id
    assert data["model"] == "realesr-general-x4v3"


def test_list_returns_newest_first():
    manager = JobManager(processor=lambda j, p, c: {})
    first = _submit(manager, filename="a.mp4")
    time.sleep(0.01)
    second = _submit(manager, filename="b.mp4")

    listed = manager.list()
    ids = [j["id"] for j in listed]
    assert ids.index(second.id) < ids.index(first.id)
