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
from src.config import load_config

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
    yield
    logger.info("Tennis Vision API shutting down")


app = FastAPI(
    title="Tennis Vision API",
    description=(
        "Automated tennis video analysis — ball tracking, court detection, "
        "rally stats, serve analysis, and heatmap visualizations."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow all origins for MVP (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all exception handler to return clean JSON errors."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )
