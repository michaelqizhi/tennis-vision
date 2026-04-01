"""Background task runner for video processing jobs.

Uses a thread pool to run analysis pipelines without blocking the API.
Job state is stored in-memory (sufficient for single-process MVP).
"""

from __future__ import annotations

import copy
import logging
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
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
    _mutation_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_step(self, step: PipelineStep, status: JobStatus, message: str = "") -> None:
        """Update or add a step's progress."""
        with self._mutation_lock:
            for s in self.steps:
                if s.step == step:
                    s.status = status
                    s.message = message
                    return
            self.steps.append(StepProgress(step=step, status=status, message=message))

    def update_progress(self, step: PipelineStep, message: str = "") -> None:
        """Mark a step as processing and update overall progress."""
        with self._mutation_lock:
            self.current_step = step.value
            self._set_step_unlocked(step, JobStatus.PROCESSING, message)
            all_steps = list(PipelineStep)
            done = sum(
                1 for s in self.steps if s.status == JobStatus.COMPLETE
            )
            self.progress = done / len(all_steps)

    def complete_step(self, step: PipelineStep, message: str = "") -> None:
        """Mark a step as complete."""
        with self._mutation_lock:
            self._set_step_unlocked(step, JobStatus.COMPLETE, message)
            all_steps = list(PipelineStep)
            done = sum(1 for s in self.steps if s.status == JobStatus.COMPLETE)
            self.progress = done / len(all_steps)

    def _set_step_unlocked(self, step: PipelineStep, status: JobStatus, message: str) -> None:
        """Update or add a step (caller must hold _mutation_lock)."""
        for s in self.steps:
            if s.step == step:
                s.status = status
                s.message = message
                return
        self.steps.append(StepProgress(step=step, status=status, message=message))

    def snapshot(self) -> dict[str, Any]:
        """Return a thread-safe snapshot of job state for API responses.

        Returns a deep copy of results to prevent callers from mutating
        shared state through the returned reference.
        """
        with self._mutation_lock:
            return {
                "job_id": self.job_id,
                "status": self.status,
                "progress": self.progress,
                "current_step": self.current_step,
                "steps": list(self.steps),
                "results": copy.deepcopy(self.results),
                "error": self.error,
                "created_at": self.created_at,
                "completed_at": self.completed_at,
            }


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
            with job._mutation_lock:
                job.status = JobStatus.PROCESSING
            pipeline_fn(job)
            with job._mutation_lock:
                job.status = JobStatus.COMPLETE
                job.progress = 1.0
        except Exception as exc:
            logger.exception("Pipeline failed for job %s", job.job_id)
            with job._mutation_lock:
                job.status = JobStatus.FAILED
                job.error = str(exc)
        finally:
            with job._mutation_lock:
                job.completed_at = datetime.now(timezone.utc)

    _executor.submit(_run)


def list_jobs() -> list[Job]:
    """Return all jobs (newest first)."""
    with _lock:
        return sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)


def clear_completed_jobs() -> int:
    """Remove completed/failed jobs from memory and clean up output directories.

    Returns count removed.
    """
    from src.config import load_config

    with _lock:
        to_remove = [
            jid for jid, j in _jobs.items()
            if j.status in (JobStatus.COMPLETE, JobStatus.FAILED)
        ]
        for jid in to_remove:
            del _jobs[jid]

    # Clean up output directories on disk
    if to_remove:
        try:
            cfg = load_config()
            for jid in to_remove:
                job_dir = cfg.output_dir / jid
                if job_dir.is_dir():
                    shutil.rmtree(job_dir, ignore_errors=True)
                    logger.info("Removed output directory: %s", job_dir)
        except Exception:
            logger.warning("Failed to clean up some output directories")

    return len(to_remove)
