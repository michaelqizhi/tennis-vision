"""Tests for Sprint 5 — serve analysis, fault detection, double faults, and stats.

Also includes regression tests for Sprint 4 feedback fixes.
"""

import numpy as np
import pytest

from src.features.ball_tracking.detector import BallDetection
from src.features.auto_clip.rally_detector import (
    Rally,
    RallyDetectorConfig,
    detect_rallies,
)
from src.features.serve_analysis.serve_detector import (
    ServeEvent,
    ServeDetectorConfig,
    detect_serves,
    _find_fault_bursts,
    _determine_server_end,
    _transform_detections,
)
from src.features.serve_analysis.fault_detector import (
    is_in_service_box,
    is_in_target_box,
    classify_serve_fault,
    reclassify_serves,
    _renumber_serves,
)
from src.features.serve_analysis.double_fault import (
    DoubleFault,
    detect_double_faults,
)
from src.features.serve_analysis.stats import (
    ServeStats,
    compute_serve_stats,
    format_serve_summary,
)


# ---------- Helpers ----------

def make_detection(frame: int, x: float | None, y: float | None, conf: float = 1.0) -> BallDetection:
    """Create a single ball detection."""
    return BallDetection(frame_number=frame, x=x, y=y, confidence=conf)


def make_rally(rally_id: int, start: int, end: int, fps: float = 30.0, num_dets: int = 20) -> Rally:
    """Create a rally for testing."""
    return Rally(
        rally_id=rally_id,
        start_frame=start,
        end_frame=end,
        fps=fps,
        num_detections=num_dets,
    )


def make_serve(
    serve_id: int,
    rally_id: int | None = None,
    server_end: str = "near",
    serve_number: int = 1,
    start_frame: int = 0,
    end_frame: int = 10,
    landing: tuple[float, float] | None = (1.0, -3.0),
    is_fault: bool = False,
    is_ace: bool = False,
) -> ServeEvent:
    """Create a serve event for testing."""
    return ServeEvent(
        serve_id=serve_id,
        rally_id=rally_id,
        server_end=server_end,
        serve_number=serve_number,
        start_frame=start_frame,
        end_frame=end_frame,
        landing_position=landing,
        is_fault=is_fault,
        is_ace=is_ace,
    )


def identity_homography() -> np.ndarray:
    """Create an identity homography (pixel coords = court coords)."""
    return np.eye(3, dtype=np.float64)


# ---------- Service Box Tests ----------

class TestIsInServiceBox:
    """Tests for is_in_service_box."""

    def test_center_of_service_box(self):
        assert is_in_service_box(1.0, -3.0) is True

    def test_on_service_line(self):
        assert is_in_service_box(0.0, 6.4) is True
        assert is_in_service_box(0.0, -6.4) is True

    def test_at_net(self):
        assert is_in_service_box(0.0, 0.0) is True

    def test_outside_service_box_width(self):
        assert is_in_service_box(5.0, -3.0) is False

    def test_outside_service_box_depth(self):
        assert is_in_service_box(1.0, -7.0) is False

    def test_at_baseline(self):
        assert is_in_service_box(0.0, 11.885) is False

    def test_negative_coords(self):
        assert is_in_service_box(-2.0, -3.0) is True


class TestIsInTargetBox:
    """Tests for is_in_target_box."""

    def test_near_server_far_target(self):
        # Server at near end → target is far-side service boxes (y < 0)
        assert is_in_target_box(1.0, -3.0, "near") is True
        assert is_in_target_box(1.0, 3.0, "near") is False

    def test_far_server_near_target(self):
        # Server at far end → target is near-side service boxes (y > 0)
        assert is_in_target_box(1.0, 3.0, "far") is True
        assert is_in_target_box(1.0, -3.0, "far") is False

    def test_at_net(self):
        # y=0 is boundary — "near" target includes y=0
        assert is_in_target_box(0.0, 0.0, "near") is True
        assert is_in_target_box(0.0, 0.0, "far") is True

    def test_outside_width(self):
        assert is_in_target_box(5.0, -3.0, "near") is False

    def test_invalid_server_end(self):
        with pytest.raises(ValueError, match="server_end must be"):
            is_in_target_box(0.0, 0.0, "left")


class TestClassifyServeFault:
    """Tests for classify_serve_fault."""

    def test_no_landing_is_fault(self):
        serve = make_serve(1, landing=None)
        assert classify_serve_fault(serve) is True

    def test_in_target_box_not_fault(self):
        serve = make_serve(1, server_end="near", landing=(1.0, -3.0))
        assert classify_serve_fault(serve) is False

    def test_outside_target_box_is_fault(self):
        serve = make_serve(1, server_end="near", landing=(1.0, 3.0))
        assert classify_serve_fault(serve) is True

    def test_far_server_in(self):
        serve = make_serve(1, server_end="far", landing=(1.0, 3.0))
        assert classify_serve_fault(serve) is False


# ---------- Serve Numbering Tests ----------

class TestRenumberServes:
    """Tests for _renumber_serves."""

    def test_all_first_serves_in(self):
        serves = [make_serve(i, is_fault=False) for i in range(3)]
        _renumber_serves(serves)
        assert all(s.serve_number == 1 for s in serves)

    def test_fault_followed_by_second(self):
        serves = [
            make_serve(1, is_fault=True, serve_number=1),
            make_serve(2, is_fault=False, serve_number=1),
        ]
        _renumber_serves(serves)
        assert serves[0].serve_number == 1
        assert serves[1].serve_number == 2

    def test_double_fault_resets(self):
        serves = [
            make_serve(1, is_fault=True, serve_number=1),
            make_serve(2, is_fault=True, serve_number=1),
            make_serve(3, is_fault=False, serve_number=1),
        ]
        _renumber_serves(serves)
        assert serves[0].serve_number == 1
        assert serves[1].serve_number == 2
        assert serves[2].serve_number == 1  # reset after double fault


# ---------- Double Fault Tests ----------

class TestDetectDoubleFaults:
    """Tests for detect_double_faults."""

    def test_no_faults(self):
        serves = [make_serve(i, is_fault=False) for i in range(3)]
        dfs = detect_double_faults(serves)
        assert dfs == []

    def test_single_fault_no_double(self):
        serves = [
            make_serve(1, is_fault=True, serve_number=1),
            make_serve(2, is_fault=False, serve_number=2),
        ]
        dfs = detect_double_faults(serves)
        assert dfs == []

    def test_double_fault_detected(self):
        serves = [
            make_serve(1, is_fault=True, serve_number=1, start_frame=0, end_frame=10),
            make_serve(2, is_fault=True, serve_number=2, start_frame=20, end_frame=30),
        ]
        dfs = detect_double_faults(serves)
        assert len(dfs) == 1
        assert dfs[0].fault_id == 1
        assert dfs[0].first_serve is serves[0]
        assert dfs[0].second_serve is serves[1]
        assert dfs[0].frame_range == (0, 30)

    def test_multiple_double_faults(self):
        serves = [
            make_serve(1, is_fault=True, serve_number=1),
            make_serve(2, is_fault=True, serve_number=2),
            make_serve(3, is_fault=True, serve_number=1),
            make_serve(4, is_fault=True, serve_number=2),
        ]
        dfs = detect_double_faults(serves)
        assert len(dfs) == 2

    def test_empty_serves(self):
        assert detect_double_faults([]) == []


class TestDoubleFaultProperties:
    """Test DoubleFault dataclass properties."""

    def test_landing_positions(self):
        s1 = make_serve(1, landing=(1.0, -3.0), is_fault=True, serve_number=1)
        s2 = make_serve(2, landing=(2.0, -4.0), is_fault=True, serve_number=2)
        df = DoubleFault(fault_id=1, first_serve=s1, second_serve=s2,
                         server_end="near", frame_range=(0, 30))
        assert df.first_landing == (1.0, -3.0)
        assert df.second_landing == (2.0, -4.0)


# ---------- Serve Stats Tests ----------

class TestComputeServeStats:
    """Tests for compute_serve_stats."""

    def test_empty_serves(self):
        stats = compute_serve_stats([], [])
        assert stats.total_serves == 0
        assert stats.first_serve_pct == 0.0

    def test_all_first_serves_in(self):
        serves = [
            make_serve(1, serve_number=1, is_fault=False, landing=(1.0, -3.0)),
            make_serve(2, serve_number=1, is_fault=False, landing=(-1.0, -2.0)),
        ]
        stats = compute_serve_stats(serves, [])
        assert stats.total_serves == 2
        assert stats.first_serves == 2
        assert stats.second_serves == 0
        assert stats.first_serve_in == 2
        assert stats.first_serve_pct == 100.0
        assert stats.double_faults == 0
        assert len(stats.first_serve_positions) == 2

    def test_mixed_serves(self):
        serves = [
            make_serve(1, serve_number=1, is_fault=True, landing=(6.0, -3.0)),
            make_serve(2, serve_number=2, is_fault=False, landing=(1.0, -3.0)),
            make_serve(3, serve_number=1, is_fault=False, is_ace=True, landing=(2.0, -3.0)),
        ]
        stats = compute_serve_stats(serves, [])
        assert stats.total_serves == 3
        assert stats.first_serves == 2
        assert stats.second_serves == 1
        assert stats.first_serve_in == 1
        assert stats.first_serve_faults == 1
        assert stats.aces == 1
        assert stats.first_serve_pct == 50.0
        assert stats.second_serve_pct == 100.0

    def test_to_dict(self):
        serves = [make_serve(1, serve_number=1, is_fault=False, landing=(1.5, -3.2))]
        stats = compute_serve_stats(serves, [])
        d = stats.to_dict()
        assert d["total_serves"] == 1
        assert "serve_details" in d
        assert d["all_serve_positions"] == [[1.5, -3.2]]


class TestFormatServeSummary:
    """Tests for format_serve_summary."""

    def test_no_serves(self):
        stats = compute_serve_stats([], [])
        assert format_serve_summary(stats) == "No serves detected."

    def test_format_output(self):
        serves = [make_serve(1, serve_number=1, is_fault=False, landing=(1.0, -3.0))]
        stats = compute_serve_stats(serves, [])
        output = format_serve_summary(stats)
        assert "Serve Statistics" in output
        assert "Total serves" in output
        assert "1st serve %" in output


# ---------- Transform Detections ----------

class TestTransformDetections:
    """Tests for _transform_detections."""

    def test_identity_transform(self):
        dets = [make_detection(0, 1.0, 2.0), make_detection(1, 3.0, 4.0)]
        H = identity_homography()
        result = _transform_detections(dets, H)
        assert len(result) == 2
        assert abs(result[0][0] - 1.0) < 0.01
        assert abs(result[0][1] - 2.0) < 0.01

    def test_empty_detections(self):
        H = identity_homography()
        assert _transform_detections([], H) == []

    def test_skips_non_detected(self):
        dets = [
            make_detection(0, 1.0, 2.0),
            make_detection(1, None, None),
            make_detection(2, 3.0, 4.0),
        ]
        H = identity_homography()
        result = _transform_detections(dets, H)
        assert len(result) == 2


# ---------- Server End Detection ----------

class TestDetermineServerEnd:
    """Tests for _determine_server_end."""

    def test_ball_moving_negative_y(self):
        """Ball moving from positive to negative y → server at near end."""
        dets = [
            make_detection(0, 0.0, 10.0),
            make_detection(1, 0.0, 5.0),
            make_detection(2, 0.0, 0.0),
        ]
        H = identity_homography()
        config = ServeDetectorConfig(min_y_displacement_meters=2.0)
        assert _determine_server_end(dets, H, config) == "near"

    def test_ball_moving_positive_y(self):
        """Ball moving from negative to positive y → server at far end."""
        dets = [
            make_detection(0, 0.0, -10.0),
            make_detection(1, 0.0, -5.0),
            make_detection(2, 0.0, 0.0),
        ]
        H = identity_homography()
        config = ServeDetectorConfig(min_y_displacement_meters=2.0)
        assert _determine_server_end(dets, H, config) == "far"

    def test_insufficient_displacement(self):
        """Small displacement → guess from starting position."""
        dets = [
            make_detection(0, 0.0, 8.0),
            make_detection(1, 0.0, 7.9),
        ]
        H = identity_homography()
        config = ServeDetectorConfig(min_y_displacement_meters=2.0)
        assert _determine_server_end(dets, H, config) == "near"

    def test_single_detection(self):
        dets = [make_detection(0, 0.0, 5.0)]
        H = identity_homography()
        config = ServeDetectorConfig()
        assert _determine_server_end(dets, H, config) == "near"


# ---------- Fault Burst Detection ----------

class TestFindFaultBursts:
    """Tests for _find_fault_bursts."""

    def test_no_gaps_no_bursts(self):
        """All detections inside rallies → no fault bursts."""
        dets = [make_detection(i, 1.0, 2.0) for i in range(100)]
        rallies = [make_rally(1, 0, 99, fps=30.0)]
        config = ServeDetectorConfig(min_serve_detections=3)
        bursts = _find_fault_bursts(dets, rallies, 30.0, config)
        assert bursts == []

    def test_burst_between_rallies(self):
        """Short detection burst between two rallies → fault burst."""
        dets = []
        # Rally 1: frames 0-100
        for i in range(0, 100):
            dets.append(make_detection(i, 1.0, 2.0))
        # Gap with fault burst: frames 120-125
        for i in range(120, 126):
            dets.append(make_detection(i, 1.0, 2.0))
        # Rally 2: frames 200-300
        for i in range(200, 300):
            dets.append(make_detection(i, 1.0, 2.0))

        rallies = [
            make_rally(1, 0, 99, fps=30.0),
            make_rally(2, 200, 299, fps=30.0),
        ]
        config = ServeDetectorConfig(
            min_serve_detections=3,
            max_serve_duration_seconds=3.0,
            min_gap_before_seconds=1.0,
        )
        bursts = _find_fault_bursts(dets, rallies, 30.0, config)
        assert len(bursts) == 1
        assert bursts[0][0] == 120
        assert bursts[0][1] == 125

    def test_burst_too_few_detections(self):
        """Burst with fewer than min_serve_detections is ignored."""
        dets = [
            make_detection(0, 1.0, 2.0),  # in rally
            make_detection(50, 1.0, 2.0),  # gap — only 1 detection
            make_detection(100, 1.0, 2.0),  # in rally
        ]
        rallies = [make_rally(1, 0, 0, fps=30.0), make_rally(2, 100, 100, fps=30.0)]
        config = ServeDetectorConfig(min_serve_detections=3)
        bursts = _find_fault_bursts(dets, rallies, 30.0, config)
        assert bursts == []


# ---------- Full Serve Detection Pipeline ----------

class TestDetectServes:
    """Tests for detect_serves end-to-end."""

    def test_empty_inputs(self):
        H = identity_homography()
        assert detect_serves([], [], H, 30.0) == []

    def test_invalid_homography(self):
        with pytest.raises(ValueError, match="3x3"):
            detect_serves(
                [make_detection(0, 1.0, 2.0)],
                [make_rally(1, 0, 100)],
                np.eye(2),
                30.0,
            )

    def test_serves_detected_for_rallies(self):
        """Each rally should produce a serve event."""
        # Create detections that simulate ball moving from near baseline toward net
        dets = []
        for i in range(0, 100):
            y = 10.0 - (i * 0.2)  # ball moves from y=10 to y=-10
            dets.append(make_detection(i, 0.0, y))
        # Second rally
        for i in range(200, 300):
            y = 10.0 - ((i - 200) * 0.2)
            dets.append(make_detection(i, 0.0, y))

        rallies = [
            make_rally(1, 0, 99, fps=30.0, num_dets=100),
            make_rally(2, 200, 299, fps=30.0, num_dets=100),
        ]
        H = identity_homography()
        serves = detect_serves(dets, rallies, H, 30.0)
        assert len(serves) == 2
        assert all(s.rally_id is not None for s in serves)


# ---------- Reclassify Tests ----------

class TestReclassifyServes:
    """Tests for reclassify_serves."""

    def test_reclassify_updates_fault_status(self):
        serves = [
            # Landing in far-side service box — should be in for near server
            make_serve(1, server_end="near", landing=(1.0, -3.0), is_fault=True),
            # Landing outside — should be fault
            make_serve(2, server_end="near", landing=(1.0, 3.0), is_fault=False),
        ]
        result = reclassify_serves(serves)
        assert result[0].is_fault is False
        assert result[1].is_fault is True

    def test_reclassify_updates_serve_numbers(self):
        serves = [
            make_serve(1, server_end="near", landing=(1.0, 8.0), is_fault=False),
            make_serve(2, server_end="near", landing=(1.0, -3.0), is_fault=False),
        ]
        result = reclassify_serves(serves)
        # First is fault (lands on wrong side), second is in
        assert result[0].is_fault is True
        assert result[0].serve_number == 1
        assert result[1].is_fault is False
        assert result[1].serve_number == 2


# ---------- Sprint 4 Regression Tests ----------

class TestSprint4Fixes:
    """Regression tests for Sprint 4 feedback fixes."""

    def test_detect_rallies_sorts_frames(self):
        """detected_frames should be sorted before splitting (feedback #1)."""
        # Create detections with intentionally shuffled frame numbers
        dets = [
            make_detection(50, 1.0, 200.0),
            make_detection(10, 1.0, 200.0),
            make_detection(30, 1.0, 200.0),
            make_detection(20, 1.0, 200.0),
            make_detection(40, 1.0, 200.0),
            make_detection(11, 1.0, 200.0),
            make_detection(31, 1.0, 200.0),
            make_detection(21, 1.0, 200.0),
            make_detection(41, 1.0, 200.0),
            make_detection(51, 1.0, 200.0),
        ]
        config = RallyDetectorConfig(min_gap_seconds=0.5, min_rally_seconds=0.1,
                                     min_detections=3, min_detection_density=0.01)
        rallies = detect_rallies(dets, fps=30.0, config=config)
        # Should detect one rally from frame 10-51 (all within gap threshold)
        assert len(rallies) == 1
        assert rallies[0].start_frame == 10
        assert rallies[0].end_frame == 51

    def test_compute_rally_stats_validates_length(self):
        """rallies and rally_shots must have same length (feedback #2)."""
        from src.features.rally_stats.stats import compute_rally_stats
        from src.features.rally_stats.counter import RallyShots

        rallies = [make_rally(1, 0, 100)]
        shots = [
            RallyShots(rally_id=1, shot_count=5, net_crossings=3,
                       start_frame=0, end_frame=100, duration=3.3),
            RallyShots(rally_id=2, shot_count=3, net_crossings=2,
                       start_frame=200, end_frame=300, duration=3.3),
        ]
        with pytest.raises(ValueError, match="same length"):
            compute_rally_stats(rallies, shots)

    def test_clip_rally_validates_frame_range(self):
        """clip_rally should validate start_frame and end_frame (feedback #3)."""
        from src.features.auto_clip.clipper import clip_rally

        with pytest.raises(ValueError, match="start_frame must be >= 0"):
            clip_rally("dummy.mp4", "out.mp4", -1, 10)

        with pytest.raises(ValueError, match="end_frame .* must be >= start_frame"):
            clip_rally("dummy.mp4", "out.mp4", 10, 5)

    def test_count_shots_court_validates_homography(self):
        """count_shots_court should reject non-3x3 homography (feedback #5)."""
        from src.features.rally_stats.counter import count_shots_court

        rally = make_rally(1, 0, 100)
        dets = [make_detection(i, 1.0, 2.0) for i in range(100)]

        with pytest.raises(ValueError, match="3x3"):
            count_shots_court(dets, rally, np.eye(2))

    def test_get_video_info_resource_safety(self):
        """get_video_info should use try/finally for cap.release() (feedback code quality #1).

        We verify the function signature accepts a path and raises on invalid input.
        """
        from scripts.run_auto_clip import get_video_info
        with pytest.raises(FileNotFoundError):
            get_video_info("nonexistent_video.mp4")


# ---------- ServeEvent Properties ----------

class TestServeEventProperties:
    """Test ServeEvent dataclass."""

    def test_is_in_property(self):
        serve = make_serve(1, is_fault=False)
        assert serve.is_in is True

        fault = make_serve(2, is_fault=True)
        assert fault.is_in is False


# ---------- Module Import Tests ----------

class TestModuleImports:
    """Verify all serve_analysis module imports work."""

    def test_import_serve_analysis(self):
        from src.features.serve_analysis import (
            ServeEvent,
            ServeDetectorConfig,
            detect_serves,
            is_in_service_box,
            is_in_target_box,
            classify_serve_fault,
            reclassify_serves,
            DoubleFault,
            detect_double_faults,
            ServeStats,
            compute_serve_stats,
            format_serve_summary,
        )
        assert ServeEvent is not None
        assert ServeDetectorConfig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
