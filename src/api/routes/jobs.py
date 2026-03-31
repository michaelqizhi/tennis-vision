"""Jobs route — list all processing jobs."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import StatusResponse
from src.api.tasks import list_jobs

router = APIRouter()


@router.get(
    "/jobs",
    response_model=list[StatusResponse],
    summary="List all jobs",
    description="Returns all processing jobs, newest first.",
)
async def get_jobs() -> list[StatusResponse]:
    """Return all jobs with their current status."""
    jobs = list_jobs()
    return [
        StatusResponse(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            current_step=job.current_step,
            steps=job.steps,
            error=job.error,
        )
        for job in jobs
    ]
