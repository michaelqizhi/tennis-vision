"""FastAPI application for Tennis Vision.

Provides REST API endpoints for video upload, processing status, and results retrieval.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import upload, status, results, jobs
from src.config import Config, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler — startup/shutdown."""
    logger.info("Tennis Vision API starting up")
    # Ensure output directory exists for static file serving
    cfg = load_config()
    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # Mount output directory to serve generated heatmap images
    app.mount("/output", StaticFiles(directory=str(output_dir)), name="output")

    # Clean up orphaned temp files from previous runs
    _cleanup_orphaned_uploads(cfg)

    yield
    logger.info("Tennis Vision API shutting down")


def _cleanup_orphaned_uploads(cfg: Config) -> None:
    """Remove orphaned temp files in uploads/ older than 1 hour."""
    import time

    uploads_dir = cfg.project_root / "uploads"
    if not uploads_dir.is_dir():
        return

    max_age_seconds = 3600  # 1 hour
    now = time.time()
    removed = 0

    for f in uploads_dir.iterdir():
        if f.is_file():
            try:
                age = now - f.stat().st_mtime
                if age > max_age_seconds:
                    f.unlink()
                    removed += 1
            except OSError:
                pass

    if removed:
        logger.info("Cleaned up %d orphaned upload file(s)", removed)


app = FastAPI(
    title="Tennis Vision API",
    description=(
        "Automated tennis video analysis — ball tracking, court detection, "
        "rally stats, serve analysis, and heatmap visualizations."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — restrict origins for security (configurable via TENNIS_VISION_CORS_ORIGINS env var)
import os as _os

_cors_origins = [
    origin.strip()
    for origin in _os.environ.get(
        "TENNIS_VISION_CORS_ORIGINS", "http://localhost:3000"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(upload.router, tags=["Upload"])
app.include_router(status.router, tags=["Status"])
app.include_router(results.router, tags=["Results"])
app.include_router(jobs.router, tags=["Jobs"])


@app.get("/health", summary="Health check")
async def health_check() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok", "service": "tennis-vision"}


@app.get("/", summary="Root", include_in_schema=False)
async def root():
    """Redirect root to API docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all exception handler to return clean JSON errors."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )
