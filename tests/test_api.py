"""Tests for Sprint 6 — FastAPI backend, routes, schemas, tasks, and pipeline."""

import io
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import (
    AnalysisResults,
    BallPositionOut,
    JobStatus,
    PipelineStep,
    RallyOut,
    RallyStatsOut,
    ResultsResponse,
    ServeStatsOut,
    StatusResponse,
    StepProgress,
    UploadResponse,
    VisualizationPaths,
)
from src.api.tasks import Job, create_job, get_job, submit_job, clear_completed_jobs, _jobs, _lock


client = TestClient(app)


# ─── Health Check ────────────────────────────────────────────────


class TestHealthCheck:
    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "tennis-vision"


# ─── Schemas ─────────────────────────────────────────────────────


class TestSchemas:
    def test_upload_response(self):
        r = UploadResponse(job_id="abc123")
        assert r.job_id == "abc123"
        assert r.status == JobStatus.QUEUED
        assert "successfully" in r.message

    def test_status_response(self):
        r = StatusResponse(job_id="x", status=JobStatus.PROCESSING, progress=0.5, current_step="ball_tracking")
        assert r.progress == 0.5
        assert r.error is None

    def test_analysis_results_defaults(self):
        r = AnalysisResults(job_id="j1", video_filename="test.mp4")
        assert r.rally_stats.rally_count == 0
        assert r.serve_stats.total_serves == 0
        assert r.visualizations.shot_heatmap is None
        assert r.ball_positions == []

    def test_ball_position_out(self):
        bp = BallPositionOut(frame_number=10, x=100.0, y=200.0, confidence=0.95)
        assert bp.court_x is None
        assert bp.court_y is None

    def test_results_response_complete(self):
        results = AnalysisResults(job_id="j1", video_filename="v.mp4", fps=30.0, total_frames=900)
        r = ResultsResponse(job_id="j1", status=JobStatus.COMPLETE, results=results)
        assert r.results is not None
        assert r.results.total_frames == 900

    def test_results_response_failed(self):
        r = ResultsResponse(job_id="j1", status=JobStatus.FAILED, error="boom")
        assert r.results is None
        assert r.error == "boom"


# ─── Task Manager ───────────────────────────────────────────────


class TestTaskManager:
    def setup_method(self):
        """Clear job store between tests."""
        with _lock:
            _jobs.clear()

    def test_create_and_get_job(self):
        job = create_job("/tmp/test.mp4", "test.mp4")
        assert job.status == JobStatus.QUEUED
        assert job.progress == 0.0
        fetched = get_job(job.job_id)
        assert fetched is not None
        assert fetched.job_id == job.job_id

    def test_get_nonexistent_job(self):
        assert get_job("nonexistent") is None

    def test_job_step_progress(self):
        job = create_job("/tmp/test.mp4", "test.mp4")
        job.update_progress(PipelineStep.BALL_TRACKING, "Running...")
        assert job.current_step == "ball_tracking"
        assert job.progress == 0.0  # none complete yet

        job.complete_step(PipelineStep.BALL_TRACKING, "Done")
        assert job.progress > 0.0
        assert any(s.status == JobStatus.COMPLETE for s in job.steps)

    def test_submit_job_success(self):
        job = create_job("/tmp/test.mp4", "test.mp4")
        done = threading.Event()

        def fake_pipeline(j: Job):
            j.results = {"test": True}
            done.set()

        submit_job(job, fake_pipeline)
        done.wait(timeout=5)
        time.sleep(0.1)  # let the wrapper finish
        assert job.status == JobStatus.COMPLETE
        assert job.progress == 1.0
        assert job.results == {"test": True}

    def test_submit_job_failure(self):
        job = create_job("/tmp/test.mp4", "test.mp4")
        done = threading.Event()

        def failing_pipeline(j: Job):
            done.set()
            raise ValueError("Pipeline exploded")

        submit_job(job, failing_pipeline)
        done.wait(timeout=5)
        time.sleep(0.1)
        assert job.status == JobStatus.FAILED
        assert "exploded" in job.error

    def test_clear_completed_jobs(self):
        j1 = create_job("/tmp/a.mp4", "a.mp4")
        j1.status = JobStatus.COMPLETE
        j2 = create_job("/tmp/b.mp4", "b.mp4")
        j2.status = JobStatus.PROCESSING
        count = clear_completed_jobs()
        assert count == 1
        assert get_job(j1.job_id) is None
        assert get_job(j2.job_id) is not None


# ─── Upload Route ────────────────────────────────────────────────


class TestUploadRoute:
    def setup_method(self):
        with _lock:
            _jobs.clear()

    @patch("src.api.routes.upload.submit_job")
    def test_upload_valid_video(self, mock_submit):
        """Upload a valid .mp4 file should return 200 with job_id."""
        mock_submit.return_value = None
        # Minimal valid MP4 header (ftyp box at offset 4)
        video_content = b"\x00\x00\x00\x1c" + b"ftyp" + b"isom" + b"\x00" * 1008
        resp = client.post(
            "/upload",
            files={"file": ("test_video.mp4", io.BytesIO(video_content), "video/mp4")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        mock_submit.assert_called_once()

    def test_upload_invalid_extension(self):
        """Upload a .txt file should return 400."""
        resp = client.post(
            "/upload",
            files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

    def test_upload_no_file(self):
        """Missing file should return 422."""
        resp = client.post("/upload")
        assert resp.status_code == 422


# ─── Status Route ────────────────────────────────────────────────


class TestStatusRoute:
    def setup_method(self):
        with _lock:
            _jobs.clear()

    def test_status_not_found(self):
        resp = client.get("/status/nonexistent")
        assert resp.status_code == 404

    def test_status_queued(self):
        job = create_job("/tmp/test.mp4", "test.mp4")
        resp = client.get(f"/status/{job.job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["progress"] == 0.0

    def test_status_processing(self):
        job = create_job("/tmp/test.mp4", "test.mp4")
        job.status = JobStatus.PROCESSING
        job.update_progress(PipelineStep.COURT_DETECTION, "Detecting...")
        resp = client.get(f"/status/{job.job_id}")
        data = resp.json()
        assert data["status"] == "processing"
        assert data["current_step"] == "court_detection"
        assert len(data["steps"]) >= 1


# ─── Results Route ───────────────────────────────────────────────


class TestResultsRoute:
    def setup_method(self):
        with _lock:
            _jobs.clear()

    def test_results_not_found(self):
        resp = client.get("/results/nonexistent")
        assert resp.status_code == 404

    def test_results_still_processing(self):
        job = create_job("/tmp/test.mp4", "test.mp4")
        job.status = JobStatus.PROCESSING
        resp = client.get(f"/results/{job.job_id}")
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "processing"
        assert "still processing" in data["error"]

    def test_results_failed(self):
        job = create_job("/tmp/test.mp4", "test.mp4")
        job.status = JobStatus.FAILED
        job.error = "GPU out of memory"
        resp = client.get(f"/results/{job.job_id}")
        data = resp.json()
        assert data["status"] == "failed"
        assert data["error"] == "GPU out of memory"
        assert data["results"] is None

    def test_results_complete(self):
        job = create_job("/tmp/test.mp4", "test.mp4")
        job.status = JobStatus.COMPLETE
        results = AnalysisResults(
            job_id=job.job_id,
            video_filename="test.mp4",
            fps=30.0,
            total_frames=900,
            ball_positions=[BallPositionOut(frame_number=0, x=100.0, y=200.0, confidence=0.9)],
            rallies=[RallyOut(rally_id=1, start_frame=0, end_frame=100, start_time=0.0, end_time=3.33, duration=3.33, shot_count=5)],
        )
        job.results = results.model_dump()
        resp = client.get(f"/results/{job.job_id}")
        data = resp.json()
        assert data["status"] == "complete"
        assert data["results"]["fps"] == 30.0
        assert len(data["results"]["ball_positions"]) == 1
        assert len(data["results"]["rallies"]) == 1
        assert data["results"]["rallies"][0]["shot_count"] == 5


# ─── OpenAPI / Swagger ───────────────────────────────────────────


class TestOpenAPI:
    def test_openapi_schema_available(self):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        assert "/upload" in schema["paths"]
        assert "/status/{job_id}" in schema["paths"]
        assert "/results/{job_id}" in schema["paths"]
        assert "/health" in schema["paths"]

    def test_swagger_ui_available(self):
        resp = client.get("/docs")
        assert resp.status_code == 200
        assert "swagger" in resp.text.lower() or "openapi" in resp.text.lower()


# ─── Pipeline (unit-level, mocked models) ────────────────────────


class TestPipelineUnit:
    """Test pipeline orchestration with mocked ML models."""

    @patch("src.api.pipeline.render_serve_placement")
    @patch("src.api.pipeline.render_shot_heatmap")
    @patch("src.api.pipeline.compute_serve_stats")
    @patch("src.api.pipeline.detect_double_faults")
    @patch("src.api.pipeline.reclassify_serves")
    @patch("src.api.pipeline.detect_serves")
    @patch("src.api.pipeline.compute_rally_stats")
    @patch("src.api.pipeline.count_all_rally_shots")
    @patch("src.api.pipeline.detect_rallies")
    @patch("src.api.pipeline.CourtDetector")
    @patch("src.api.pipeline.BallTracker")
    @patch("src.api.pipeline.VideoReader")
    def test_pipeline_runs_all_steps(
        self,
        MockReader, MockTracker, MockCourt,
        mock_rallies, mock_shots, mock_rally_stats,
        mock_detect_serves, mock_reclassify, mock_double_faults,
        mock_serve_stats, mock_heatmap, mock_serve_heatmap,
    ):
        from src.api.pipeline import run_pipeline
        from src.features.ball_tracking import BallDetection
        from src.features.auto_clip import Rally
        from src.features.rally_stats import RallyShots, RallyStats
        from src.features.serve_analysis import ServeStats

        # Mock video reader
        reader = MagicMock()
        reader.fps = 30.0
        reader.frame_count = 90
        reader.width = 640
        reader.height = 360
        reader.read_sampled_frames.return_value = [np.zeros((360, 640, 3), dtype=np.uint8)] * 20
        reader.iter_frames.return_value = iter([np.zeros((360, 640, 3), dtype=np.uint8)] * 90)
        reader.__enter__ = MagicMock(return_value=reader)
        reader.__exit__ = MagicMock(return_value=False)
        MockReader.return_value = reader

        # Mock ball tracker
        tracker_inst = MagicMock()
        detections = [
            BallDetection(frame_number=i, x=float(100 + i), y=float(200 + i), confidence=0.9)
            for i in range(90)
        ]
        tracker_inst.detect_streaming.return_value = detections
        MockTracker.return_value = tracker_inst

        # Mock court detector
        court_inst = MagicMock()
        court_inst.get_stable_homography.return_value = np.eye(3)
        MockCourt.return_value = court_inst

        # Mock rally detection
        rally = Rally(rally_id=1, start_frame=0, end_frame=89, fps=30.0, num_detections=90)
        mock_rallies.return_value = [rally]

        # Mock shot counting
        shots = RallyShots(rally_id=1, shot_count=8, net_crossings=4, start_frame=0, end_frame=89, duration=3.0)
        mock_shots.return_value = [shots]

        # Mock rally stats
        mock_rally_stats.return_value = RallyStats(
            rally_count=1, total_shots=8, shots_per_rally=[8],
            rally_durations=[3.0], avg_rally_length=3.0, avg_shots_per_rally=8.0,
            longest_rally_duration=3.0, longest_rally_shots=8,
            shortest_rally_duration=3.0, shortest_rally_shots=8,
            median_rally_duration=3.0, median_shots_per_rally=8.0,
            shot_count_distribution={8: 1},
        )

        # Mock serve analysis
        mock_detect_serves.return_value = []
        mock_reclassify.return_value = []
        mock_double_faults.return_value = []
        mock_serve_stats.return_value = ServeStats(
            total_serves=0, first_serves=0, second_serves=0,
            first_serve_in=0, second_serve_in=0, first_serve_faults=0,
            double_faults=0, aces=0, first_serve_pct=0.0, second_serve_pct=0.0,
            first_serve_positions=[], second_serve_positions=[],
            all_serve_positions=[], serve_details=[],
        )

        # Run pipeline
        job = create_job("/tmp/test.mp4", "test.mp4")
        run_pipeline(job)

        # Verify all steps ran
        assert job.results is not None
        assert job.results["video_filename"] == "test.mp4"
        assert job.results["fps"] == 30.0
        assert job.results["total_frames"] == 90
        assert len(job.results["ball_positions"]) == 90
        assert len(job.results["rallies"]) == 1
        assert job.results["rally_stats"]["rally_count"] == 1
        assert job.results["rally_stats"]["total_shots"] == 8

        # Verify all pipeline steps were called
        MockTracker.assert_called_once()
        MockCourt.assert_called_once()
        mock_rallies.assert_called_once()
        mock_shots.assert_called_once()
        mock_rally_stats.assert_called_once()


# ─── Job step tracking ──────────────────────────────────────────


class TestJobStepTracking:
    def test_step_progress_through_pipeline(self):
        job = create_job("/tmp/t.mp4", "t.mp4")

        # Simulate pipeline progress
        all_steps = list(PipelineStep)
        for i, step in enumerate(all_steps):
            job.update_progress(step, f"Running {step.value}...")
            assert job.current_step == step.value

            job.complete_step(step, f"{step.value} done")
            expected_progress = (i + 1) / len(all_steps)
            assert abs(job.progress - expected_progress) < 0.01

        assert job.progress == 1.0
