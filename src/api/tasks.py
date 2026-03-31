"""Background task runner for video processing jobs.

Uses a thread pool to run analysis pipelines without blocking the API.
Job state is stored in-memory (sufficient for single-process MVP).
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from src.api.schemas import JobStatus, PipelineStep, StepProgress

logger = logging.getLogger(__name__)

# Single worker — one video at a time to avoid GPU/memory contention
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")


@dataclass
class Job:
    """Represents a video processing job."""
    job_id: str
    video_path: str
    filename: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    current_step: str = ""
    steps: list[StepProgress] = field(default_factory=list)
    results: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    def set_step(self, step: PipelineStep, status: JobStatus, message: str = "") -> None:
        """Update or add a step's progress."""
        for s in self.steps:
            if s.step == step:
                s.status = status
                s.message = message
                return
        self.steps.append(StepProgress(step=step, status=status, message=message))

    def update_progress(self, step: PipelineStep, message: str = "") -> None:
        """Mark a step as processing and update overall progress."""
        self.current_step = step.value
        self.set_step(step, JobStatus.PROCESSING, message)
        # Compute progress from completed steps
        all_steps = list(PipelineStep)
        done = sum(
            1 for s in self.steps if s.status == JobStatus.COMPLETE
        )
        self.progress = done / len(all_steps)

    def complete_step(self, step: PipelineStep, message: str = "") -> None:
        """Mark a step as complete."""
        self.set_step(step, JobStatus.COMPLETE, message)
        all_steps = list(PipelineStep)
        done = sum(1 for s in self.steps if s.status == JobStatus.COMPLETE)
        self.progress = done / len(all_steps)


# In-memory job store (thread-safe via GIL for simple dict ops)
_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def create_job(video_path: str, filename: str) -> Job:
    """Create a new processing job and return it."""
    job_id = uuid.uuid4().hex[:12]
    job = Job(job_id=job_id, video_path=video_path, filename=filename)
    with _lock:
        _jobs[job_id] = job
    return job


def get_job(job_id: str) -> Job | None:
    """Retrieve a job by ID."""
    with _lock:
        return _jobs.get(job_id)


def submit_job(job: Job, pipeline_fn: Callable[[Job], None]) -> None:
    """Submit a job to the thread pool for background processing."""
    def _run() -> None:
        try:
            job.status = JobStatus.PROCESSING
            pipeline_fn(job)
            job.status = JobStatus.COMPLETE
            job.progress = 1.0
        except Exception as exc:
            logger.exception("Pipeline failed for job %s", job.job_id)
            job.status = JobStatus.FAILED
            job.error = str(exc)
        finally:
            job.completed_at = datetime.now(timezone.utc)

    _executor.submit(_run)


def list_jobs() -> list[Job]:
    """Return all jobs (newest first)."""
    with _lock:
        return sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)


def clear_completed_jobs() -> int:
    """Remove completed/failed jobs from memory. Returns count removed."""
    with _lock:
        to_remove = [
            jid for jid, j in _jobs.items()
            if j.status in (JobStatus.COMPLETE, JobStatus.FAILED)
        ]
        for jid in to_remove:
            del _jobs[jid]
        return len(to_remove)
