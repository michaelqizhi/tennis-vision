"""Status route — check processing progress for a job."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import StatusResponse, ErrorResponse
from src.api.tasks import get_job

router = APIRouter()


@router.get(
    "/status/{job_id}",
    response_model=StatusResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get processing status for a job",
    description="Returns the current status and step-by-step progress of a video analysis job.",
)
async def get_status(job_id: str) -> StatusResponse:
    """Return the current status and progress of a processing job."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return StatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        current_step=job.current_step,
        steps=job.steps,
        error=job.error,
    )
