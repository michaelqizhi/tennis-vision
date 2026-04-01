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
    responses = []
    for job in jobs:
        snap = job.snapshot()
        responses.append(StatusResponse(
            job_id=snap["job_id"],
            status=snap["status"],
            progress=snap["progress"],
            current_step=snap["current_step"],
            steps=snap["steps"],
            error=snap["error"],
        ))
    return responses
