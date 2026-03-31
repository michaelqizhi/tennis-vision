"""Upload route — accept video file and queue processing."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from src.api.pipeline import run_pipeline
from src.api.schemas import UploadResponse, ErrorResponse
from src.api.tasks import create_job, submit_job
from src.config import load_config

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_FILE_SIZE_MB = 500


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    summary="Upload a tennis video for analysis",
    description="Accepts a video file, saves it, and queues it for the full analysis pipeline.",
)
async def upload_video(file: UploadFile = File(..., description="Tennis match video file")) -> UploadResponse:
    """Upload a video and start background analysis."""
    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Save uploaded file to a temp location
    config = load_config()
    uploads_dir = config.project_root / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Read into a temp file then move (handles large files)
        with tempfile.NamedTemporaryFile(
            dir=str(uploads_dir), suffix=ext, delete=False
        ) as tmp:
            size = 0
            while chunk := await file.read(1024 * 1024):  # 1 MB chunks
                size += len(chunk)
                if size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    os.unlink(tmp.name)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB} MB.",
                    )
                tmp.write(chunk)
            tmp_path = tmp.name
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to save uploaded file")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.") from exc

    # Create job and submit to pipeline
    job = create_job(video_path=tmp_path, filename=file.filename)
    submit_job(job, run_pipeline)

    logger.info("Job %s created for file %s (%d bytes)", job.job_id, file.filename, size)
    return UploadResponse(job_id=job.job_id)
