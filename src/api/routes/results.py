"""Results route — retrieve completed analysis results."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    AnalysisResults,
    JobStatus,
    ResultsResponse,
    ErrorResponse,
)
from src.api.tasks import get_job

router = APIRouter()


@router.get(
    "/results/{job_id}",
    response_model=ResultsResponse,
    responses={404: {"model": ErrorResponse}, 202: {"model": ResultsResponse}},
    summary="Get analysis results for a completed job",
    description="Returns the full analysis results including ball positions, rally stats, serve stats, and visualization paths.",
)
async def get_results(job_id: str) -> ResultsResponse:
    """Return analysis results for a completed job."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job.status == JobStatus.FAILED:
        return ResultsResponse(
            job_id=job.job_id,
            status=job.status,
            error=job.error,
        )

    if job.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
        return ResultsResponse(
            job_id=job.job_id,
            status=job.status,
            error="Job is still processing. Check /status/{job_id} for progress.",
        )

    # Job is complete — return results
    results = None
    if job.results:
        results = AnalysisResults(**job.results)

    return ResultsResponse(
        job_id=job.job_id,
        status=job.status,
        results=results,
    )
