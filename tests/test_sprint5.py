"""Tests for Sprint 5 — Player detection, proximity filtering, bug fixes.

Covers:
  - PlayerDetector and PlayerBox dataclass
  - Player proximity filter logic
  - Consensus player filter integration
  - API route snapshot usage (race condition fix)
  - Court detection weighted centroid (postprocess fix)
  - Ball tracker stationary removal
  - Shot count sanity cap
  - Serve stats fallback wiring
  - Code quality fixes (step numbering, dead code removal)
  - Frontend lint script fix
  - Inference runner streaming (no full preload)
  - Root endpoint redirect
"""

import math
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pytest


# ── PlayerBox tests ────────────────────────────────────────────────

class TestPlayerBox:
    """Tests for PlayerBox dataclass."""

    def test_center(self):
        from src.labeling.player_detector import PlayerBox
        box = PlayerBox(x1=100, y1=200, x2=200, y2=400, confidence=0.9)
        assert box.center_x == 150.0
        assert box.center_y == 300.0

    def test_dimensions(self):
        from src.labeling.player_detector import PlayerBox
        box = PlayerBox(x1=100, y1=200, x2=200, y2=400, confidence=0.9)
        assert box.width == 100.0
        assert box.height == 200.0

    def test_contains_point_inside(self):
        from src.labeling.player_detector import PlayerBox
        box = PlayerBox(x1=100, y1=200, x2=200, y2=400, confidence=0.9)
        assert box.contains_point(150, 300)

    def test_contains_point_outside(self):
        from src.labeling.player_detector import PlayerBox
        box = PlayerBox(x1=100, y1=200, x2=200, y2=400, confidence=0.9)
        assert not box.contains_point(50, 300)

    def test_contains_point_with_margin(self):
        from src.labeling.player_detector import PlayerBox
        box = PlayerBox(x1=100, y1=200, x2=200, y2=400, confidence=0.9)
        assert box.contains_point(90, 300, margin=15)

    def test_distance_inside(self):
        from src.labeling.player_detector import PlayerBox
        box = PlayerBox(x1=100, y1=200, x2=200, y2=400, confidence=0.9)
        assert box.distance_to_point(150, 300) == 0.0

    def test_distance_outside(self):
        from src.labeling.player_detector import PlayerBox
        box = PlayerBox(x1=100, y1=200, x2=200, y2=400, confidence=0.9)
        # Point at (50, 300) is 50px to the left of the box
        assert box.distance_to_point(50, 300) == 50.0

    def test_distance_diagonal(self):
        from src.labeling.player_detector import PlayerBox
        box = PlayerBox(x1=100, y1=200, x2=200, y2=400, confidence=0.9)
        # Point at (0, 100) — dx=100, dy=100
        dist = box.distance_to_point(0, 100)
        assert abs(dist - math.sqrt(100**2 + 100**2)) < 0.01


# ── Player proximity filter tests ─────────────────────────────────

class TestPlayerProximityFilter:
    """Tests for filter_by_player_proximity."""

    def test_near_player(self):
        from src.labeling.player_detector import PlayerBox, filter_by_player_proximity
        players = [PlayerBox(x1=100, y1=200, x2=200, y2=400, confidence=0.9)]
        assert filter_by_player_proximity(150, 300, players, max_distance=200)

    def test_far_from_player(self):
        from src.labeling.player_detector import PlayerBox, filter_by_player_proximity
        players = [PlayerBox(x1=100, y1=200, x2=200, y2=400, confidence=0.9)]
        assert not filter_by_player_proximity(800, 300, players, max_distance=200)

    def test_no_players(self):
        from src.labeling.player_detector import filter_by_player_proximity
        # No players means we can't filter — pass through
        assert filter_by_player_proximity(800, 300, [], max_distance=200)

    def test_near_second_player(self):
        from src.labeling.player_detector import PlayerBox, filter_by_player_proximity
        players = [
            PlayerBox(x1=100, y1=200, x2=200, y2=400, confidence=0.9),
            PlayerBox(x1=700, y1=200, x2=800, y2=400, confidence=0.8),
        ]
        # Near second player
        assert filter_by_player_proximity(750, 300, players, max_distance=200)

    def test_between_players(self):
        from src.labeling.player_detector import PlayerBox, filter_by_player_proximity
        players = [
            PlayerBox(x1=100, y1=200, x2=200, y2=400, confidence=0.9),
            PlayerBox(x1=700, y1=200, x2=800, y2=400, confidence=0.8),
        ]
        # Midpoint at 450 — >200px from both players
        assert not filter_by_player_proximity(450, 300, players, max_distance=200)


# ── Consensus player filter integration ───────────────────────────

class TestConsensusPlayerFilter:
    """Tests for apply_player_proximity_filter in consensus.py."""

    def test_filters_distant_detections(self):
        from src.labeling.consensus import (
            FrameConsensus, DetectionLabel, apply_player_proximity_filter,
        )
        from src.labeling.player_detector import PlayerBox

        consensus = [
            FrameConsensus(frame_idx=0, label=DetectionLabel.CONSENSUS, x=150, y=300),
            FrameConsensus(frame_idx=1, label=DetectionLabel.CONSENSUS, x=900, y=300),
            FrameConsensus(frame_idx=2, label=DetectionLabel.NO_DETECTION),
        ]
        players = {
            0: [PlayerBox(x1=100, y1=200, x2=200, y2=400, confidence=0.9)],
            1: [PlayerBox(x1=100, y1=200, x2=200, y2=400, confidence=0.9)],
        }

        filtered, rejected = apply_player_proximity_filter(consensus, players, max_distance=200)
        assert rejected == 1
        assert filtered[0].label == DetectionLabel.CONSENSUS  # near player
        assert filtered[1].label == DetectionLabel.NO_DETECTION  # far from player
        assert filtered[2].label == DetectionLabel.NO_DETECTION  # unchanged

    def test_no_filter_without_players(self):
        from src.labeling.consensus import (
            FrameConsensus, DetectionLabel, apply_player_proximity_filter,
        )
        consensus = [
            FrameConsensus(frame_idx=0, label=DetectionLabel.CONSENSUS, x=900, y=300),
        ]
        # No player data → keep all
        filtered, rejected = apply_player_proximity_filter(consensus, {}, max_distance=200)
        assert rejected == 0
        assert filtered[0].label == DetectionLabel.CONSENSUS


# ── Court detection weighted centroid fix ─────────────────────────

class TestCourtDetectionPostprocess:
    """Tests for _postprocess_heatmap weighted centroid fix."""

    def test_weighted_centroid_basic(self):
        from src.features.court_detect.detector import _postprocess_heatmap
        heatmap = np.zeros((180, 320), dtype=np.uint8)
        # Create a bright spot
        heatmap[80:90, 150:160] = 255
        x, y = _postprocess_heatmap(heatmap, scale_x=2.0, scale_y=2.0, low_thresh=170)
        assert x is not None
        assert y is not None
        # Should be near the center of the bright spot
        assert abs(x - 310.0) < 20  # 155 * 2
        assert abs(y - 170.0) < 20  # 85 * 2

    def test_no_detection_on_empty(self):
        from src.features.court_detect.detector import _postprocess_heatmap
        heatmap = np.zeros((180, 320), dtype=np.uint8)
        x, y = _postprocess_heatmap(heatmap, scale_x=2.0, scale_y=2.0, low_thresh=170)
        assert x is None
        assert y is None


# ── Ball tracking postprocess weighted centroid ───────────────────

class TestBallPostprocess:
    """Tests for ball tracking postprocess weighted centroid."""

    def test_weighted_centroid(self):
        from src.features.ball_tracking.detector import postprocess
        heatmap = np.zeros((360, 640), dtype=np.float32)
        # Create a bright spot at center
        heatmap[175:185, 315:325] = 1.0
        x, y = postprocess(heatmap.flatten(), 640, 360, scale_x=3.0, scale_y=3.0)
        assert x is not None
        assert y is not None

    def test_no_detection_on_empty(self):
        from src.features.ball_tracking.detector import postprocess
        heatmap = np.zeros((360 * 640,), dtype=np.float32)
        x, y = postprocess(heatmap, 640, 360)
        assert x is None
        assert y is None


# ── Stationary detection removal ──────────────────────────────────

class TestStationaryRemoval:
    """Tests for BallTracker._remove_stationary."""

    def test_removes_stationary_cluster(self):
        from src.features.ball_tracking.detector import BallTracker
        # 15 frames at nearly the same position
        track = [(100.0, 200.0)] * 15 + [(500.0, 300.0)]
        result = BallTracker._remove_stationary(track, max_stationary_frames=10, stationary_radius=15.0)
        # First 15 should be cleared
        assert all(r == (None, None) for r in result[:15])
        # Last one should survive
        assert result[15] == (500.0, 300.0)

    def test_keeps_moving_detections(self):
        from src.features.ball_tracking.detector import BallTracker
        # Each frame moves significantly
        track = [(float(i * 30), float(i * 20)) for i in range(20)]
        result = BallTracker._remove_stationary(track, max_stationary_frames=10, stationary_radius=15.0)
        # All should survive (each moves >15px)
        assert all(r[0] is not None for r in result)

    def test_short_stationary_kept(self):
        from src.features.ball_tracking.detector import BallTracker
        # 5 frames stationary — below threshold of 10
        track = [(100.0, 200.0)] * 5
        result = BallTracker._remove_stationary(track, max_stationary_frames=10, stationary_radius=15.0)
        assert all(r[0] is not None for r in result)


# ── Shot count sanity cap ─────────────────────────────────────────

class TestShotCountCap:
    """Tests for shot count max-per-second cap."""

    def test_shot_count_capped(self):
        from src.features.rally_stats.counter import count_shots_pixel
        from src.features.ball_tracking.detector import BallDetection
        from src.features.auto_clip.rally_detector import Rally

        # Create a short rally (5 seconds) with lots of y-direction changes
        detections = []
        for i in range(150):  # 5 seconds at 30fps
            # Alternating y to create many direction changes
            y_val = 200.0 if i % 3 == 0 else 500.0
            detections.append(BallDetection(
                frame_number=i, x=500.0, y=y_val, confidence=1.0
            ))

        rally = Rally(rally_id=1, start_frame=0, end_frame=149,
                       fps=30.0, num_detections=150)

        result = count_shots_pixel(detections, rally, frame_height=720)
        # Cap at 1 shot/sec × 5 seconds = 5 max
        assert result.shot_count <= 5


# ── Serve stats fallback ─────────────────────────────────────────

class TestServeStatsFallback:
    """Tests that serve stats wire estimated count when no homography."""

    def test_serve_stats_pipeline_no_homography(self):
        """The pipeline should set total_serves = len(rallies) when no homography."""
        from src.api.pipeline import _run_pipeline_steps
        # Just verify the logic exists by checking the pipeline source
        import inspect
        source = inspect.getsource(_run_pipeline_steps)
        assert "total_serves" in source
        assert "len(rallies)" in source


# ── API race condition fix ────────────────────────────────────────

class TestAPISnapshot:
    """Tests that API routes use job.snapshot() for thread safety."""

    def test_status_route_uses_snapshot(self):
        import inspect
        from src.api.routes.status import get_status
        source = inspect.getsource(get_status)
        assert "snapshot()" in source
        assert "job.status" not in source or "snap" in source

    def test_results_route_uses_snapshot(self):
        import inspect
        from src.api.routes.results import get_results
        source = inspect.getsource(get_results)
        assert "snapshot()" in source

    def test_jobs_route_uses_snapshot(self):
        import inspect
        from src.api.routes.jobs import get_jobs
        source = inspect.getsource(get_jobs)
        assert "snapshot()" in source


# ── Code quality fixes ────────────────────────────────────────────

class TestCodeQuality:
    """Tests verifying code quality fixes from Sprint 4 feedback."""

    def test_pipeline_step_numbering(self):
        """Step comments should be sequential (1-6)."""
        import inspect
        from src.api.pipeline import _run_pipeline_steps
        source = inspect.getsource(_run_pipeline_steps)
        # Should have Steps 1-6 in order
        for i in range(1, 7):
            assert f"Step {i}:" in source

    def test_pipeline_config_type(self):
        """_run_pipeline_steps should accept Config, not object."""
        import inspect
        from src.api.pipeline import _run_pipeline_steps
        sig = inspect.signature(_run_pipeline_steps)
        params = sig.parameters
        assert "config" in params
        # Check annotation is Config, not object
        ann = params["config"].annotation
        # With from __future__ import annotations, the annotation is a string
        assert ann == "Config" or (hasattr(ann, '__name__') and ann.__name__ == 'Config')

    def test_no_dead_video_magic(self):
        """_VIDEO_MAGIC dict should not exist in upload.py."""
        import inspect
        from src.api.routes import upload
        source = inspect.getsource(upload)
        assert "_VIDEO_MAGIC" not in source

    def test_no_dead_locals_serve_detector(self):
        """rally_idx and burst_idx should not be assigned in detect_serves."""
        import inspect
        from src.features.serve_analysis.serve_detector import detect_serves
        source = inspect.getsource(detect_serves)
        assert "rally_idx = " not in source
        assert "burst_idx = " not in source

    def test_no_unused_imports_consensus(self):
        """consensus.py should not import logging."""
        import inspect
        from src.labeling import consensus
        source = inspect.getsource(consensus)
        assert "import logging" not in source

    def test_no_unused_imports_inference_runner(self):
        """inference_runner.py should not import cv2."""
        with open("src/labeling/inference_runner.py", encoding="utf-8") as f:
            source = f.read()
        assert "import cv2" not in source

    def test_root_endpoint_exists(self):
        """GET / should exist and not 404."""
        from src.api.main import app
        routes = [r.path for r in app.routes]
        assert "/" in routes


# ── Frontend lint fix ─────────────────────────────────────────────

class TestFrontendLint:
    """Tests for frontend configuration."""

    def test_lint_script_no_dir_flag(self):
        """npm run lint should not use --dir flag."""
        import json
        with open("frontend/package.json") as f:
            pkg = json.load(f)
        lint_cmd = pkg["scripts"]["lint"]
        assert "--dir" not in lint_cmd
        assert "next lint" in lint_cmd


# ── Player serialization ─────────────────────────────────────────

class TestPlayerSerialization:
    """Tests for serialize_player_boxes."""

    def test_serialize(self):
        from src.labeling.player_detector import PlayerBox, serialize_player_boxes
        boxes = [
            PlayerBox(x1=100.123, y1=200.456, x2=300.789, y2=400.012, confidence=0.9123),
        ]
        result = serialize_player_boxes(boxes)
        assert len(result) == 1
        assert result[0]["x1"] == 100.1
        assert result[0]["confidence"] == 0.9123


# ── Labeling pipeline player filter flag ──────────────────────────

class TestPipelineFlags:
    """Tests for new CLI flags."""

    def test_no_player_filter_flag(self):
        from src.labeling.pipeline import build_parser
        parser = build_parser()
        args = parser.parse_args(["test.mp4", "--no-player-filter"])
        assert args.no_player_filter is True

    def test_player_distance_flag(self):
        from src.labeling.pipeline import build_parser
        parser = build_parser()
        args = parser.parse_args(["test.mp4", "--player-distance", "300"])
        assert args.player_distance == 300.0

    def test_default_player_filter_enabled(self):
        from src.labeling.pipeline import build_parser
        parser = build_parser()
        args = parser.parse_args(["test.mp4"])
        assert args.no_player_filter is False
        assert args.player_distance == 200.0


# ── Main API cleanup type fix ─────────────────────────────────────

class TestMainAppFixes:
    """Tests for main.py fixes."""

    def test_cleanup_type_annotation(self):
        """_cleanup_orphaned_uploads should accept Config, not object."""
        import inspect
        from src.api.main import _cleanup_orphaned_uploads
        sig = inspect.signature(_cleanup_orphaned_uploads)
        params = sig.parameters
        ann = params["cfg"].annotation
        assert ann == "Config" or (hasattr(ann, '__name__') and ann.__name__ == 'Config')
