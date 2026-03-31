"""Pydantic schemas for the Tennis Vision API request/response models."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Processing job status."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class PipelineStep(str, Enum):
    """Individual pipeline processing steps."""
    COURT_DETECTION = "court_detection"
    BALL_TRACKING = "ball_tracking"
    RALLY_DETECTION = "rally_detection"
    RALLY_STATS = "rally_stats"
    SERVE_ANALYSIS = "serve_analysis"
    VISUALIZATION = "visualization"


class UploadResponse(BaseModel):
    """Response returned after successful video upload."""
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    message: str = "Video uploaded successfully. Processing will begin shortly."


class StepProgress(BaseModel):
    """Progress for a single pipeline step."""
    step: PipelineStep
    status: JobStatus
    message: str = ""


class StatusResponse(BaseModel):
    """Response for job status queries."""
    job_id: str
    status: JobStatus
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Overall progress 0.0–1.0")
    current_step: str = ""
    steps: list[StepProgress] = []
    error: str | None = None


class BallPositionOut(BaseModel):
    """A single ball detection result."""
    frame_number: int
    x: float | None = None
    y: float | None = None
    confidence: float = 0.0
    interpolated: bool = False
    court_x: float | None = None
    court_y: float | None = None


class RallyOut(BaseModel):
    """A detected rally."""
    rally_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    shot_count: int = 0


class RallyStatsOut(BaseModel):
    """Aggregate rally statistics."""
    rally_count: int = 0
    total_shots: int = 0
    avg_rally_length: float = 0.0
    avg_shots_per_rally: float = 0.0
    longest_rally_duration: float = 0.0
    longest_rally_shots: int = 0
    shortest_rally_duration: float = 0.0
    shortest_rally_shots: int = 0
    median_rally_duration: float = 0.0
    median_shots_per_rally: float = 0.0
    shots_per_rally: list[int] = []
    shot_count_distribution: dict[int, int] = {}


class ServeStatsOut(BaseModel):
    """Serve analysis statistics."""
    total_serves: int = 0
    first_serves: int = 0
    second_serves: int = 0
    first_serve_in: int = 0
    second_serve_in: int = 0
    first_serve_faults: int = 0
    double_faults: int = 0
    aces: int = 0
    first_serve_pct: float = 0.0
    second_serve_pct: float = 0.0


class VisualizationPaths(BaseModel):
    """Paths to generated visualization assets."""
    shot_heatmap: str | None = None
    serve_heatmap_all: str | None = None
    serve_heatmap_first: str | None = None
    serve_heatmap_second: str | None = None


class AnalysisResults(BaseModel):
    """Complete analysis results for a processed video."""
    job_id: str
    video_filename: str
    fps: float = 0.0
    total_frames: int = 0
    ball_positions: list[BallPositionOut] = []
    rallies: list[RallyOut] = []
    rally_stats: RallyStatsOut = Field(default_factory=RallyStatsOut)
    serve_stats: ServeStatsOut = Field(default_factory=ServeStatsOut)
    visualizations: VisualizationPaths = Field(default_factory=VisualizationPaths)


class ResultsResponse(BaseModel):
    """Response for job results queries."""
    job_id: str
    status: JobStatus
    results: AnalysisResults | None = None
    error: str | None = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
