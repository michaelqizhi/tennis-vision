"""Tests for Sprint 3 deliverables: visualize, rally boundaries, pipeline CLI, and bug fixes.

Tests use real logic — no mocks. They use synthetic frames/data to exercise
the overlay renderer, rally boundary detector, and consensus clustering fix.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.labeling.consensus import (
    Detection,
    DetectionLabel,
    FrameConsensus,
    _euclidean_distance,
    _find_largest_cluster,
    compute_consensus,
)
from src.labeling.rally_boundaries import (
    RallyBoundary,
    RallyBoundaryConfig,
    detect_rally_boundaries,
    serialize_rally_boundaries,
)
from src.labeling.visualize import generate_review_video, _draw_overlay
from src.labeling.pipeline import build_parser, run_labeling_pipeline


# ── Helpers ─────────────────────────────────────────────────────────


def _make_consensus(
    label: DetectionLabel,
    frame_idx: int,
    x: float | None = None,
    y: float | None = None,
    confidence: float | None = None,
    models: list[str] | None = None,
) -> FrameConsensus:
    """Create a FrameConsensus for testing."""
    return FrameConsensus(
        frame_idx=frame_idx,
        label=label,
        x=x, y=y,
        confidence=confidence,
        agreeing_models=models or [],
    )


def _make_test_video(tmp_dir: str, num_frames: int = 30, fps: float = 30.0) -> str:
    """Create a small synthetic test video file."""
    path = os.path.join(tmp_dir, "test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (320, 240))
    for i in range(num_frames):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Add some variation so video isn't all black
        cv2.putText(frame, str(i), (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    return path


# ══════════════════════════════════════════════════════════════════
# Consensus clustering fix (Critical Bug #1)
# ══════════════════════════════════════════════════════════════════


class TestClusteringAllPairs:
    """Verify that _find_largest_cluster enforces all-pairs distance."""

    def test_linear_spread_rejects_distant_pair(self):
        """Detections at (0,0), (14,0), (28,0) with threshold=15.

        (0,0)↔(14,0)=14 ✓, (14,0)↔(28,0)=14 ✓, but (0,0)↔(28,0)=28 ✗.
        The old seed-based algorithm would form a cluster of 3 from seed (14,0).
        The fixed algorithm should return at most 2.
        """
        dets = [
            Detection("A", 0, 0, 0.9),
            Detection("B", 14, 0, 0.9),
            Detection("C", 28, 0, 0.9),
        ]
        cluster = _find_largest_cluster(dets, threshold_px=15.0)
        # Should be exactly 2 — any pair of adjacent detections
        assert len(cluster) == 2
        # Verify all pairs in the cluster are within threshold
        for i, d1 in enumerate(cluster):
            for j, d2 in enumerate(cluster):
                if i < j:
                    dist = _euclidean_distance(d1.x, d1.y, d2.x, d2.y)
                    assert dist <= 15.0, f"Pair {d1.model_name}-{d2.model_name} distance {dist} > 15"

    def test_all_within_threshold_keeps_all(self):
        """All detections are within threshold — cluster should include all."""
        dets = [
            Detection("A", 0, 0, 0.9),
            Detection("B", 5, 5, 0.9),
            Detection("C", 3, 3, 0.9),
        ]
        cluster = _find_largest_cluster(dets, threshold_px=15.0)
        assert len(cluster) == 3

    def test_consensus_with_linear_spread(self):
        """Full pipeline should classify linear spread as uncertain, not consensus."""
        raw = {
            "0": {
                "A": {"x": 0, "y": 0, "conf": 0.9},
                "B": {"x": 14, "y": 0, "conf": 0.9},
                "C": {"x": 28, "y": 0, "conf": 0.9},
            }
        }
        results = compute_consensus(raw, threshold_px=15.0)
        assert len(results) == 1
        # Should still be consensus (2 of 3 agree), but NOT include the distant one
        fc = results[0]
        assert fc.label == DetectionLabel.CONSENSUS
        assert len(fc.agreeing_models) == 2

    def test_two_detections_within_threshold(self):
        """Two detections within threshold should form consensus."""
        dets = [
            Detection("A", 100, 200, 0.9),
            Detection("B", 110, 205, 0.8),
        ]
        cluster = _find_largest_cluster(dets, 15.0)
        assert len(cluster) == 2

    def test_four_models_two_clusters(self):
        """4 detections forming two separate pairs — largest cluster wins."""
        dets = [
            Detection("A", 0, 0, 0.9),
            Detection("B", 5, 0, 0.9),
            Detection("C", 100, 0, 0.9),
            Detection("D", 105, 0, 0.9),
        ]
        cluster = _find_largest_cluster(dets, 15.0)
        # Both clusters have 2 members — either is valid
        assert len(cluster) == 2


# ══════════════════════════════════════════════════════════════════
# Rally boundary detection
# ══════════════════════════════════════════════════════════════════


class TestRallyBoundaries:
    """Tests for rally boundary detection from detection density."""

    def test_empty_consensus(self):
        """Empty consensus → empty boundaries."""
        assert detect_rally_boundaries([], fps=30.0) == []

    def test_invalid_fps(self):
        """fps ≤ 0 raises ValueError."""
        fc = [_make_consensus(DetectionLabel.CONSENSUS, 0, 100, 200, 0.9)]
        with pytest.raises(ValueError, match="fps must be positive"):
            detect_rally_boundaries(fc, fps=0)

    def test_all_detections_single_rally(self):
        """All frames have detections → single rally segment."""
        consensus = [
            _make_consensus(DetectionLabel.CONSENSUS, i, 100, 200, 0.9, ["A", "B"])
            for i in range(150)  # 5 seconds at 30fps
        ]
        boundaries = detect_rally_boundaries(consensus, fps=30.0)
        rallies = [b for b in boundaries if b.is_rally]
        assert len(rallies) >= 1
        # All should be rally
        for r in rallies:
            assert r.detection_rate >= 0.3

    def test_no_detections_no_rally(self):
        """All frames are no_detection → no rally segments."""
        consensus = [
            _make_consensus(DetectionLabel.NO_DETECTION, i)
            for i in range(150)
        ]
        boundaries = detect_rally_boundaries(consensus, fps=30.0)
        rallies = [b for b in boundaries if b.is_rally]
        assert len(rallies) == 0

    def test_rally_then_gap_then_rally(self):
        """Dense → sparse → dense should produce 2 rally segments."""
        consensus = []
        # Rally 1: frames 0-149 (5 seconds) — all detected
        for i in range(150):
            consensus.append(
                _make_consensus(DetectionLabel.CONSENSUS, i, 100, 200, 0.9, ["A", "B"])
            )
        # Gap: frames 150-299 (5 seconds) — no detections
        for i in range(150, 300):
            consensus.append(
                _make_consensus(DetectionLabel.NO_DETECTION, i)
            )
        # Rally 2: frames 300-449 (5 seconds) — all detected
        for i in range(300, 450):
            consensus.append(
                _make_consensus(DetectionLabel.CONSENSUS, i, 100, 200, 0.9, ["A", "B"])
            )

        boundaries = detect_rally_boundaries(consensus, fps=30.0)
        rallies = [b for b in boundaries if b.is_rally]
        assert len(rallies) == 2

    def test_configurable_thresholds(self):
        """Custom config should affect rally detection."""
        consensus = [
            _make_consensus(DetectionLabel.CONSENSUS, i, 100, 200, 0.9, ["A"])
            for i in range(120)  # 4 seconds at 30fps
        ]
        # With high min_rally_duration, this short rally should be filtered
        config = RallyBoundaryConfig(min_rally_duration_sec=10.0)
        boundaries = detect_rally_boundaries(consensus, fps=30.0, config=config)
        rallies = [b for b in boundaries if b.is_rally]
        assert len(rallies) == 0

    def test_serialize_rally_boundaries(self):
        """Serialization produces valid JSON-compatible dicts."""
        boundaries = [
            RallyBoundary(
                start_frame=0, end_frame=89,
                start_sec=0.0, end_sec=2.967,
                detection_rate=0.85, is_rally=True,
            ),
        ]
        serialized = serialize_rally_boundaries(boundaries)
        assert len(serialized) == 1
        entry = serialized[0]
        assert entry["start_frame"] == 0
        assert entry["end_frame"] == 89
        assert entry["is_rally"] is True
        assert "duration_sec" in entry
        # Should be JSON-serializable
        json.dumps(serialized)

    def test_short_video_single_segment(self):
        """Video shorter than window size → treated as single segment."""
        consensus = [
            _make_consensus(DetectionLabel.CONSENSUS, i, 100, 200, 0.9, ["A"])
            for i in range(10)
        ]
        config = RallyBoundaryConfig(window_size_sec=2.0)
        boundaries = detect_rally_boundaries(consensus, fps=30.0, config=config)
        assert len(boundaries) == 1


# ══════════════════════════════════════════════════════════════════
# Review visualization overlay
# ══════════════════════════════════════════════════════════════════


class TestVisualizationOverlay:
    """Tests for the review visualization overlay generator."""

    def test_draw_overlay_consensus(self):
        """Consensus detection draws green circle."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        fc = _make_consensus(DetectionLabel.CONSENSUS, 0, 160, 120, 0.95, ["A", "B"])
        _draw_overlay(frame, 0, 100, fc)
        # Frame should not be all black anymore (overlay was drawn)
        assert frame.sum() > 0

    def test_draw_overlay_uncertain(self):
        """Uncertain detection draws yellow circle."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        fc = _make_consensus(DetectionLabel.UNCERTAIN, 0, 160, 120, 0.5, ["A"])
        _draw_overlay(frame, 0, 100, fc)
        assert frame.sum() > 0

    def test_draw_overlay_no_detection(self):
        """No detection draws red border."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        fc = _make_consensus(DetectionLabel.NO_DETECTION, 0)
        _draw_overlay(frame, 0, 100, fc)
        # Red border should be visible — check edges
        assert frame.sum() > 0
        # Red channel should have activity on the border
        assert frame[0, :, 2].sum() > 0  # top edge, red channel

    def test_draw_overlay_none_consensus(self):
        """None consensus (frame not in consensus map) draws red border."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        _draw_overlay(frame, 0, 100, None)
        assert frame.sum() > 0

    def test_generate_review_video(self):
        """Full review video generation with synthetic video."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = _make_test_video(tmp_dir, num_frames=10, fps=30.0)

            consensus = []
            for i in range(10):
                if i < 4:
                    consensus.append(_make_consensus(
                        DetectionLabel.CONSENSUS, i, 160, 120, 0.9, ["A", "B"]
                    ))
                elif i < 7:
                    consensus.append(_make_consensus(
                        DetectionLabel.UNCERTAIN, i, 100, 80, 0.5, ["A"]
                    ))
                else:
                    consensus.append(_make_consensus(
                        DetectionLabel.NO_DETECTION, i
                    ))

            output_path = os.path.join(tmp_dir, "review.mp4")
            result = generate_review_video(video_path, consensus, output_path)
            assert result == output_path
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0


# ══════════════════════════════════════════════════════════════════
# Pipeline CLI
# ══════════════════════════════════════════════════════════════════


class TestPipelineCLI:
    """Tests for the CLI argument parser."""

    def test_parser_defaults(self):
        """Parser provides sensible defaults."""
        parser = build_parser()
        args = parser.parse_args(["test.mp4"])
        assert args.video_path == "test.mp4"
        assert args.output == "output/labeling"
        assert args.device == "cpu"
        assert args.max_frames == 0
        assert args.threshold == 15.0
        assert args.skip_review_video is False
        assert args.verbose is False

    def test_parser_custom_args(self):
        """Parser handles custom arguments."""
        parser = build_parser()
        args = parser.parse_args([
            "video.mp4",
            "--output", "/tmp/out",
            "--device", "cuda",
            "--max-frames", "100",
            "--threshold", "20.0",
            "--skip-review-video",
            "-v",
        ])
        assert args.video_path == "video.mp4"
        assert args.output == "/tmp/out"
        assert args.device == "cuda"
        assert args.max_frames == 100
        assert args.threshold == 20.0
        assert args.skip_review_video is True
        assert args.verbose is True

    def test_parser_rally_config_args(self):
        """Parser handles rally detection config arguments."""
        parser = build_parser()
        args = parser.parse_args([
            "v.mp4",
            "--window-size", "3.0",
            "--rally-threshold", "0.5",
            "--min-rally", "5.0",
            "--min-gap", "6.0",
        ])
        assert args.window_size == 3.0
        assert args.rally_threshold == 0.5
        assert args.min_rally == 5.0
        assert args.min_gap == 6.0
