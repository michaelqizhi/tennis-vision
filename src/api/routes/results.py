"""Results route — retrieve completed analysis results."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

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
async def get_results(job_id: str) -> ResultsResponse | JSONResponse:
    """Return analysis results for a completed job."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    snap = job.snapshot()

    if snap["status"] == JobStatus.FAILED:
        return ResultsResponse(
            job_id=snap["job_id"],
            status=snap["status"],
            error=snap["error"],
        )

    if snap["status"] in (JobStatus.QUEUED, JobStatus.PROCESSING):
        body = ResultsResponse(
            job_id=snap["job_id"],
            status=snap["status"],
            error="Job is still processing. Check /status/{job_id} for progress.",
        )
        return JSONResponse(content=body.model_dump(), status_code=202)

    # Job is complete — return results
    results = None
    if snap["results"]:
        results = AnalysisResults(**snap["results"])

    return ResultsResponse(
        job_id=snap["job_id"],
        status=snap["status"],
        results=results,
    )
