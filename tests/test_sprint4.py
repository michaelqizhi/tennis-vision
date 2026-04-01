"""Tests for Sprint 4: Bug fixes + TrackNet V4 + Benchmark Report.

Covers:
  - Pixel-coordinate heatmap fallback (#1)
  - Shot counting smoothing / min_displacement fix (#2)
  - Rally boundary config validation (#8)
  - Rally re-merge after filtering (#9)
  - Upload magic byte validation (#10)
  - snapshot() deep copy (#6)
  - TrackNet V4 model architecture
  - Benchmark report generation
  - Updated inference runner with V4
  - Output directory cleanup (#5)
"""

import copy
import io
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── Pixel Heatmap Fallback Tests ──────────────────────────────────────


class TestPixelHeatmapFallback:
    """Test render_pixel_heatmap function."""

    def test_basic_render(self, tmp_path):
        """Renders a pixel heatmap PNG with valid positions."""
        from src.features.visualization.renderer import render_pixel_heatmap

        positions = [(100.0, 200.0), (300.0, 400.0), (500.0, 600.0)]
        output = str(tmp_path / "heatmap.png")
        result = render_pixel_heatmap(positions, output, frame_width=1920, frame_height=1080)
        assert Path(result).exists()
        assert os.path.getsize(result) > 0

    def test_render_with_many_points_kde(self, tmp_path):
        """Renders KDE heatmap when enough points are provided."""
        from src.features.visualization.renderer import render_pixel_heatmap

        rng = np.random.default_rng(42)
        positions = [(float(rng.uniform(0, 1920)), float(rng.uniform(0, 1080))) for _ in range(50)]
        output = str(tmp_path / "heatmap_kde.png")
        result = render_pixel_heatmap(positions, output, frame_width=1920, frame_height=1080)
        assert Path(result).exists()

    def test_render_empty_positions(self, tmp_path):
        """Renders without error even with zero positions."""
        from src.features.visualization.renderer import render_pixel_heatmap

        output = str(tmp_path / "empty.png")
        result = render_pixel_heatmap([], output, frame_width=1920, frame_height=1080)
        assert Path(result).exists()

    def test_render_with_background_frame(self, tmp_path):
        """Renders with a background frame without error."""
        from src.features.visualization.renderer import render_pixel_heatmap

        bg = np.zeros((1080, 1920, 3), dtype=np.uint8)
        positions = [(100.0, 200.0), (300.0, 400.0), (500.0, 600.0)]
        output = str(tmp_path / "bg_heatmap.png")
        result = render_pixel_heatmap(positions, output, frame_width=1920, frame_height=1080, background_frame=bg)
        assert Path(result).exists()


# ── Shot Counting Fix Tests ───────────────────────────────────────────


class TestShotCountingFix:
    """Test that shot counting produces plausible results."""

    def test_smooth_trajectory(self):
        """Smoothing reduces noise in trajectory."""
        from src.features.rally_stats.counter import _smooth_trajectory

        noisy = [100.0, 102.0, 98.0, 101.0, 99.0, 100.0]
        smoothed = _smooth_trajectory(noisy, window=3)
        assert len(smoothed) == len(noisy)
        # Smoothed values should have less variance
        assert np.std(smoothed) <= np.std(noisy)

    def test_smooth_trajectory_short(self):
        """Smoothing returns input unchanged for very short sequences."""
        from src.features.rally_stats.counter import _smooth_trajectory

        assert _smooth_trajectory([1.0], window=5) == [1.0]
        assert _smooth_trajectory([1.0, 2.0], window=5) == [1.0, 2.0]

    def test_pixel_shot_count_reasonable(self):
        """Pixel-space shot count should be reasonable for a 60s rally."""
        from src.features.rally_stats.counter import count_shots_pixel
        from src.features.ball_tracking.detector import BallDetection
        from src.features.auto_clip.rally_detector import Rally

        # Simulate 1800 frames (60s at 30fps) with sinusoidal y-motion
        # 3-second period → 20 full oscillations → ~40 net crossings → ~41 shots
        # This is correct for a fast rally. The key test is that it's NOT 125.
        fps = 30.0
        detections = []
        for i in range(1800):
            y = 360 + 200 * np.sin(2 * np.pi * i / (fps * 3))
            detections.append(BallDetection(
                frame_number=i, x=640.0, y=float(y),
                confidence=0.9,
            ))

        rally = Rally(rally_id=1, start_frame=0, end_frame=1799, fps=fps, num_detections=1800)
        result = count_shots_pixel(detections, rally, frame_height=720)

        # With 3s period: 20 cycles × 2 crossings = ~40 shots (+ 1)
        # Much less than the original 125 (implausible) but more than trivial
        assert result.shot_count < 60, f"Shot count {result.shot_count} is implausibly high"
        assert result.shot_count >= 10, f"Shot count {result.shot_count} is too low"
        # Crucially, much less than the old broken value of 125
        assert result.shot_count < 125, "Shot count should be much less than pre-fix 125"

    def test_pixel_shot_count_noisy_signal(self):
        """Noisy detections should not produce excessive shot count."""
        from src.features.rally_stats.counter import count_shots_pixel
        from src.features.ball_tracking.detector import BallDetection
        from src.features.auto_clip.rally_detector import Rally

        fps = 30.0
        rng = np.random.default_rng(42)
        detections = []
        for i in range(900):  # 30 seconds
            # Base trajectory with noise
            y = 360 + 200 * np.sin(2 * np.pi * i / (fps * 4)) + rng.normal(0, 15)
            detections.append(BallDetection(
                frame_number=i, x=640.0, y=float(y),
                confidence=0.9,
            ))

        rally = Rally(rally_id=1, start_frame=0, end_frame=899, fps=fps, num_detections=900)
        result = count_shots_pixel(detections, rally, frame_height=720)

        # With 4-second period over 30 seconds, ~15 direction changes max
        assert result.shot_count < 25, f"Shot count {result.shot_count} too high for noisy signal"


# ── Rally Boundary Config Validation Tests ────────────────────────────


class TestRallyBoundaryConfigValidation:
    """Test config validation in rally boundary detection."""

    def test_negative_window_size(self):
        """Negative window_size_sec raises ValueError."""
        from src.labeling.rally_boundaries import RallyBoundaryConfig, detect_rally_boundaries
        from src.labeling.consensus import FrameConsensus, DetectionLabel

        consensus = [FrameConsensus(frame_idx=i, label=DetectionLabel.CONSENSUS, x=1.0, y=1.0, confidence=0.9, all_detections=[])
                     for i in range(10)]
        config = RallyBoundaryConfig(window_size_sec=-1.0)
        with pytest.raises(ValueError, match="window_size_sec"):
            detect_rally_boundaries(consensus, 30.0, config=config)

    def test_threshold_out_of_range(self):
        """rally_threshold > 1.0 raises ValueError."""
        from src.labeling.rally_boundaries import RallyBoundaryConfig, detect_rally_boundaries
        from src.labeling.consensus import FrameConsensus, DetectionLabel

        consensus = [FrameConsensus(frame_idx=i, label=DetectionLabel.CONSENSUS, x=1.0, y=1.0, confidence=0.9, all_detections=[])
                     for i in range(10)]
        config = RallyBoundaryConfig(rally_threshold=1.5)
        with pytest.raises(ValueError, match="rally_threshold"):
            detect_rally_boundaries(consensus, 30.0, config=config)

    def test_negative_min_rally_duration(self):
        """Negative min_rally_duration_sec raises ValueError."""
        from src.labeling.rally_boundaries import RallyBoundaryConfig, detect_rally_boundaries
        from src.labeling.consensus import FrameConsensus, DetectionLabel

        consensus = [FrameConsensus(frame_idx=i, label=DetectionLabel.CONSENSUS, x=1.0, y=1.0, confidence=0.9, all_detections=[])
                     for i in range(10)]
        config = RallyBoundaryConfig(min_rally_duration_sec=-1.0)
        with pytest.raises(ValueError, match="min_rally_duration_sec"):
            detect_rally_boundaries(consensus, 30.0, config=config)


# ── Rally Re-merge Tests ──────────────────────────────────────────────


class TestRallyRemerge:
    """Test that adjacent rallies are merged after filtering."""

    def test_adjacent_rallies_merged(self):
        """After filtering a short rally between two long rallies, the remaining are merged."""
        from src.labeling.rally_boundaries import _merge_adjacent_rallies, RallyBoundary

        segments = [
            RallyBoundary(start_frame=0, end_frame=100, start_sec=0.0, end_sec=3.3, detection_rate=0.8, is_rally=True),
            RallyBoundary(start_frame=101, end_frame=200, start_sec=3.37, end_sec=6.67, detection_rate=0.9, is_rally=True),
        ]
        merged = _merge_adjacent_rallies(segments)
        assert len(merged) == 1
        assert merged[0].start_frame == 0
        assert merged[0].end_frame == 200

    def test_non_adjacent_not_merged(self):
        """Rally followed by gap followed by rally — not merged."""
        from src.labeling.rally_boundaries import _merge_adjacent_rallies, RallyBoundary

        segments = [
            RallyBoundary(start_frame=0, end_frame=100, start_sec=0.0, end_sec=3.3, detection_rate=0.8, is_rally=True),
            RallyBoundary(start_frame=101, end_frame=200, start_sec=3.37, end_sec=6.67, detection_rate=0.05, is_rally=False),
            RallyBoundary(start_frame=201, end_frame=300, start_sec=6.7, end_sec=10.0, detection_rate=0.7, is_rally=True),
        ]
        merged = _merge_adjacent_rallies(segments)
        assert len(merged) == 3


# ── Upload Magic Byte Validation Tests ────────────────────────────────


class TestUploadMagicBytes:
    """Test magic byte validation in upload route."""

    def test_looks_like_mp4(self):
        """MP4 header (ftyp at offset 4) is recognized."""
        from src.api.routes.upload import _looks_like_video

        header = b"\x00\x00\x00\x1c" + b"ftyp" + b"isom" + b"\x00" * 4
        assert _looks_like_video(header)

    def test_looks_like_avi(self):
        """AVI header (RIFF...AVI ) is recognized."""
        from src.api.routes.upload import _looks_like_video

        header = b"RIFF" + b"\x00\x00\x00\x00" + b"AVI " + b"\x00"
        assert _looks_like_video(header)

    def test_looks_like_mkv(self):
        """MKV/WebM header (EBML) is recognized."""
        from src.api.routes.upload import _looks_like_video

        header = b"\x1a\x45\xdf\xa3" + b"\x00" * 8
        assert _looks_like_video(header)

    def test_random_bytes_rejected(self):
        """Random bytes are not recognized as video."""
        from src.api.routes.upload import _looks_like_video

        assert not _looks_like_video(b"\x00" * 12)
        assert not _looks_like_video(b"hello world!")

    def test_too_short_rejected(self):
        """Header too short is rejected."""
        from src.api.routes.upload import _looks_like_video

        assert not _looks_like_video(b"\x00\x00")


# ── snapshot() Deep Copy Test ─────────────────────────────────────────


class TestSnapshotDeepCopy:
    """Test that snapshot() returns a deep copy."""

    def test_snapshot_results_independent(self):
        """Mutating the snapshot's results dict should not affect the job."""
        from src.api.tasks import Job
        from src.api.schemas import JobStatus

        job = Job(job_id="test", video_path="/tmp/test.mp4", filename="test.mp4")
        job.results = {"ball_positions": [{"x": 1, "y": 2}]}

        snap = job.snapshot()
        snap["results"]["ball_positions"].append({"x": 3, "y": 4})

        assert len(job.results["ball_positions"]) == 1  # original unchanged


# ── Output Directory Cleanup Test ─────────────────────────────────────


class TestOutputCleanup:
    """Test clear_completed_jobs cleans up output dirs."""

    def test_cleanup_removes_output_dirs(self, tmp_path):
        """clear_completed_jobs removes output directories."""
        from src.api.tasks import Job, _jobs, _lock, clear_completed_jobs
        from src.api.schemas import JobStatus

        # Clean up any leftover jobs from other tests
        with _lock:
            _jobs.clear()

        # Create a completed job with an output dir
        job = Job(job_id="cleanup_test", video_path="/tmp/test.mp4", filename="test.mp4")
        job.status = JobStatus.COMPLETE

        with _lock:
            _jobs["cleanup_test"] = job

        # Create fake output dir
        with patch("src.config.load_config") as mock_cfg:
            cfg = MagicMock()
            cfg.output_dir = tmp_path
            mock_cfg.return_value = cfg

            output_dir = tmp_path / "cleanup_test"
            output_dir.mkdir()
            (output_dir / "dummy.png").write_text("test")

            count = clear_completed_jobs()

        assert count == 1
        assert not output_dir.exists()


# ── TrackNet V4 Architecture Tests ────────────────────────────────────


class TestTrackNetV4Architecture:
    """Test TrackNet V4 model architecture."""

    def test_model_instantiates(self):
        """TrackNetV4Model can be instantiated."""
        from src.labeling.tracknetv4_detector import TrackNetV4Model

        model = TrackNetV4Model()
        assert model is not None

    def test_model_forward_pass(self):
        """Model forward pass produces correct output shape."""
        import torch
        from src.labeling.tracknetv4_detector import TrackNetV4Model

        model = TrackNetV4Model()
        model.eval()

        inp = torch.randn(1, 9, 288, 512)
        with torch.no_grad():
            out = model(inp)

        assert out.shape == (1, 3, 288, 512)
        # Output should be sigmoid (0-1 range)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_model_parameter_count(self):
        """Model has expected parameter count (approx)."""
        from src.labeling.tracknetv4_detector import TrackNetV4Model

        model = TrackNetV4Model()
        params = sum(p.numel() for p in model.parameters())
        # TrackNet V4 should have roughly 10-20M parameters
        assert 1_000_000 < params < 50_000_000

    def test_detector_name(self):
        """Detector has correct name."""
        from src.labeling.tracknetv4_detector import TrackNetV4Detector

        det = TrackNetV4Detector()
        assert det.name == "tracknet_v4"

    def test_detector_detect_without_load_raises(self):
        """Calling detect before load raises RuntimeError."""
        from src.labeling.tracknetv4_detector import TrackNetV4Detector

        det = TrackNetV4Detector()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="not loaded"):
            det.detect(frame)

    def test_detector_buffer_needs_3_frames(self):
        """Detector returns None for first two frames."""
        import torch
        from src.labeling.tracknetv4_detector import TrackNetV4Detector, TrackNetV4Model

        det = TrackNetV4Detector()
        det._model = TrackNetV4Model()
        det._model.eval()
        det._device = "cpu"

        frame = np.zeros((288, 512, 3), dtype=np.uint8)
        assert det.detect(frame) is None  # frame 1
        assert det.detect(frame) is None  # frame 2
        # Frame 3 should produce a result (or None if no ball detected)
        result = det.detect(frame)
        # With blank frames, likely None
        assert result is None or (isinstance(result, tuple) and len(result) == 3)

    def test_peak_component_centroid(self):
        """Peak-component centroid extracts position from heatmap."""
        from src.labeling.tracknetv4_detector import TrackNetV4Detector

        # Create a heatmap with a bright spot at (100, 50) in 512x288
        heatmap = np.zeros((288, 512), dtype=np.float32)
        heatmap[50, 100] = 0.9
        heatmap[51, 100] = 0.8
        heatmap[50, 101] = 0.8

        x, y, conf = TrackNetV4Detector._peak_component_centroid(heatmap, scale_x=2.0, scale_y=2.0)
        assert x is not None
        assert y is not None
        assert conf >= 0.5
        # Position should be near (200, 100) after 2x scaling
        assert abs(x - 200.0) < 10
        assert abs(y - 100.0) < 10

    def test_peak_component_rejects_large_blob(self):
        """Large diffuse blobs (player-sized) are rejected."""
        from src.labeling.tracknetv4_detector import TrackNetV4Detector, _MAX_BLOB_AREA

        # Create a heatmap with a large diffuse blob (40 pixels)
        heatmap = np.zeros((288, 512), dtype=np.float32)
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                heatmap[140 + dy, 256 + dx] = 0.6
        assert (heatmap > 0.5).sum() > _MAX_BLOB_AREA  # confirm it exceeds limit

        x, y, conf = TrackNetV4Detector._peak_component_centroid(heatmap, scale_x=2.0, scale_y=2.0)
        assert x is None  # rejected as player-sized blob


# ── Benchmark Report Tests ────────────────────────────────────────────


class TestBenchmarkReport:
    """Test benchmark report generation."""

    def _make_test_data(self):
        """Create minimal test data for benchmark report."""
        from src.labeling.consensus import FrameConsensus, DetectionLabel

        detections = {
            "0": {"model_a": {"x": 100, "y": 200, "conf": 0.9}, "model_b": {"x": 102, "y": 201, "conf": 0.8}},
            "1": {"model_a": {"x": 110, "y": 210, "conf": 0.85}, "model_b": None},
            "2": {"model_a": None, "model_b": {"x": 120, "y": 220, "conf": 0.7}},
            "3": {"model_a": None, "model_b": None},
        }
        consensus = [
            FrameConsensus(frame_idx=0, label=DetectionLabel.CONSENSUS, x=101.0, y=200.5, confidence=0.85, all_detections=[]),
            FrameConsensus(frame_idx=1, label=DetectionLabel.UNCERTAIN, x=110.0, y=210.0, confidence=0.85, all_detections=[]),
            FrameConsensus(frame_idx=2, label=DetectionLabel.UNCERTAIN, x=120.0, y=220.0, confidence=0.7, all_detections=[]),
            FrameConsensus(frame_idx=3, label=DetectionLabel.NO_DETECTION, x=None, y=None, confidence=0.0, all_detections=[]),
        ]
        timing = {"model_a": 1.0, "model_b": 2.0}
        return detections, consensus, timing

    def test_report_generated(self, tmp_path):
        """Report is generated and saved."""
        from src.labeling.benchmark import generate_benchmark_report

        detections, consensus, timing = self._make_test_data()
        output = str(tmp_path / "report.md")
        report = generate_benchmark_report(detections, consensus, timing, output)

        assert Path(output).exists()
        assert "Model Benchmark Report" in report
        assert "model_a" in report
        assert "model_b" in report

    def test_report_contains_tables(self, tmp_path):
        """Report contains comparison and consensus tables."""
        from src.labeling.benchmark import generate_benchmark_report

        detections, consensus, timing = self._make_test_data()
        output = str(tmp_path / "report.md")
        report = generate_benchmark_report(detections, consensus, timing, output)

        assert "Per-Model Comparison" in report
        assert "Consensus Summary" in report
        assert "VRAM Estimates" in report
        assert "Recommendations" in report

    def test_report_with_video_info(self, tmp_path):
        """Report includes video info when provided."""
        from src.labeling.benchmark import generate_benchmark_report

        detections, consensus, timing = self._make_test_data()
        output = str(tmp_path / "report.md")
        video_info = {"path": "test.mp4", "frames": 100, "fps": 30.0}
        report = generate_benchmark_report(detections, consensus, timing, output, video_info=video_info)

        assert "Video Info" in report
        assert "test.mp4" in report

    def test_report_empty_detections(self, tmp_path):
        """Report handles empty detections gracefully."""
        from src.labeling.benchmark import generate_benchmark_report

        output = str(tmp_path / "report.md")
        report = generate_benchmark_report({}, [], {}, output)
        assert "No frames" in report


# ── Updated Inference Runner Tests ────────────────────────────────────


class TestInferenceRunnerV4:
    """Test that inference runner includes TrackNet V4."""

    def test_default_detectors_includes_v4(self):
        """Default detectors list includes TrackNet V4."""
        from src.labeling.inference_runner import get_default_detectors

        detectors = get_default_detectors()
        names = [d.name for d in detectors]
        assert "tracknet_v4" in names
        assert "tracknet_v2" in names
        assert len(detectors) == 4  # V2, V4, Florence, YOLO-World

    def test_return_timing_flag(self):
        """run_inference with return_timing returns tuple."""
        from src.labeling.inference_runner import run_inference
        from src.labeling.models import BaseDetector

        class DummyDetector(BaseDetector):
            @property
            def name(self): return "dummy"
            def load(self, device="cpu"): pass
            def detect(self, frame): return (10.0, 20.0, 0.5)
            def unload(self): pass

        # Create a tiny test video
        import cv2
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(tmp.name, fourcc, 30.0, (64, 64))
            for _ in range(5):
                writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
            writer.release()

            result = run_inference(tmp.name, [DummyDetector()], return_timing=True)
            assert isinstance(result, tuple)
            detections, timing = result
            assert "dummy" in timing
            assert timing["dummy"] > 0

            # Without return_timing, returns just dict
            result2 = run_inference(tmp.name, [DummyDetector()], return_timing=False)
            assert isinstance(result2, dict)
        finally:
            os.unlink(tmp.name)


# ── Pipeline Integration Tests ────────────────────────────────────────


class TestPipelineStepLabels:
    """Test that labeling pipeline step labels are updated."""

    def test_pipeline_cli_parser(self):
        """CLI parser has all expected arguments."""
        from src.labeling.pipeline import build_parser

        parser = build_parser()
        args = parser.parse_args(["test.mp4", "--output", "out/", "--skip-review-video"])
        assert args.video_path == "test.mp4"
        assert args.output == "out/"
        assert args.skip_review_video is True
