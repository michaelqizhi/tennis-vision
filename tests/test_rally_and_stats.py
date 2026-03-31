"""Tests for rally detection, shot counting, and stats."""

import numpy as np
import pytest

from src.features.ball_tracking.detector import BallDetection
from src.features.auto_clip.rally_detector import (
    Rally,
    RallyDetectorConfig,
    detect_rallies,
    get_rally_clips,
    _split_at_gaps,
)
from src.features.rally_stats.counter import (
    RallyShots,
    count_shots_pixel,
    count_all_rally_shots,
    _count_direction_changes,
    _count_crossings,
)
from src.features.rally_stats.stats import (
    compute_rally_stats,
    format_stats_summary,
)


# --- Helpers ---

def make_detections(
    pattern: list[bool],
    fps: float = 30.0,
    y_base: float = 200.0,
) -> list[BallDetection]:
    """Create ball detections from a boolean pattern.

    True = ball detected, False = no detection.
    """
    dets = []
    for i, detected in enumerate(pattern):
        if detected:
            dets.append(BallDetection(
                frame_number=i, x=320.0, y=y_base + (i % 20) * 5.0, confidence=1.0,
            ))
        else:
            dets.append(BallDetection(
                frame_number=i, x=None, y=None, confidence=0.0,
            ))
    return dets


def make_rally(rally_id: int, start: int, end: int, fps: float = 30.0, num_dets: int = 20) -> Rally:
    return Rally(rally_id=rally_id, start_frame=start, end_frame=end, fps=fps, num_detections=num_dets)


# --- Rally Detection Tests ---

class TestSplitAtGaps:
    def test_single_segment(self):
        frames = [0, 1, 2, 3, 4, 5]
        result = _split_at_gaps(frames, min_gap_frames=10)
        assert len(result) == 1
        assert result[0] == (0, 5, 6)

    def test_two_segments(self):
        frames = [0, 1, 2, 100, 101, 102]
        result = _split_at_gaps(frames, min_gap_frames=10)
        assert len(result) == 2
        assert result[0] == (0, 2, 3)
        assert result[1] == (100, 102, 3)

    def test_empty(self):
        assert _split_at_gaps([], 10) == []

    def test_gap_just_below_threshold(self):
        frames = [0, 1, 2, 11, 12, 13]
        result = _split_at_gaps(frames, min_gap_frames=10)
        assert len(result) == 1  # gap of 9, below threshold of 10

    def test_gap_at_threshold(self):
        frames = [0, 1, 2, 12, 13, 14]
        result = _split_at_gaps(frames, min_gap_frames=10)
        assert len(result) == 2  # gap of 10, at threshold


class TestDetectRallies:
    def test_no_detections(self):
        assert detect_rallies([], 30.0) == []

    def test_too_few_detections(self):
        dets = make_detections([True, True, False, False])
        config = RallyDetectorConfig(min_detections=5)
        assert detect_rallies(dets, 30.0, config) == []

    def test_single_rally(self):
        # 90 frames = 3 seconds of activity
        pattern = [True] * 90
        dets = make_detections(pattern)
        config = RallyDetectorConfig(min_rally_seconds=1.0, min_detections=5)
        rallies = detect_rallies(dets, 30.0, config)
        assert len(rallies) == 1
        assert rallies[0].rally_id == 1

    def test_two_rallies_with_gap(self):
        # Rally 1: frames 0-89, gap: frames 90-179, Rally 2: frames 180-269
        pattern = [True] * 90 + [False] * 90 + [True] * 90
        dets = make_detections(pattern)
        config = RallyDetectorConfig(
            min_gap_seconds=2.0, min_rally_seconds=1.0, min_detections=5,
        )
        rallies = detect_rallies(dets, 30.0, config)
        assert len(rallies) == 2
        assert rallies[0].rally_id == 1
        assert rallies[1].rally_id == 2

    def test_short_segment_filtered(self):
        # Short burst: 10 frames = 0.33s at 30fps (below 1s min)
        pattern = [True] * 10 + [False] * 90 + [True] * 90
        dets = make_detections(pattern)
        config = RallyDetectorConfig(
            min_gap_seconds=2.0, min_rally_seconds=1.0, min_detections=5,
        )
        rallies = detect_rallies(dets, 30.0, config)
        assert len(rallies) == 1  # only the 90-frame rally survives

    def test_zero_fps(self):
        dets = make_detections([True] * 30)
        assert detect_rallies(dets, 0.0) == []


class TestRallyProperties:
    def test_duration(self):
        rally = make_rally(1, start=0, end=150, fps=30.0)
        assert abs(rally.duration - 5.0) < 0.01

    def test_start_end_time(self):
        rally = make_rally(1, start=30, end=90, fps=30.0)
        assert abs(rally.start_time - 1.0) < 0.01
        assert abs(rally.end_time - 3.0) < 0.01

    def test_detection_density(self):
        rally = Rally(rally_id=1, start_frame=0, end_frame=99, fps=30.0, num_detections=50)
        assert abs(rally.detection_density - 0.5) < 0.01


class TestGetRallyClips:
    def test_padding(self):
        rallies = [make_rally(1, start=30, end=90, fps=30.0)]
        config = RallyDetectorConfig(pad_start_seconds=1.0, pad_end_seconds=1.0)
        clips = get_rally_clips(rallies, total_frames=200, fps=30.0, config=config)
        assert len(clips) == 1
        assert clips[0][0] == 0  # 30 - 30 = 0
        assert clips[0][1] == 120  # 90 + 30

    def test_clamp_to_bounds(self):
        rallies = [make_rally(1, start=5, end=195, fps=30.0)]
        config = RallyDetectorConfig(pad_start_seconds=1.0, pad_end_seconds=1.0)
        clips = get_rally_clips(rallies, total_frames=200, fps=30.0, config=config)
        assert clips[0][0] == 0
        assert clips[0][1] == 199


# --- Shot Counting Tests ---

class TestDirectionChanges:
    def test_monotonic_increasing(self):
        assert _count_direction_changes([1, 2, 3, 4, 5]) == 0

    def test_single_reversal(self):
        assert _count_direction_changes([0, 5, 10, 5, 0], min_displacement=2.0) == 1

    def test_multiple_reversals(self):
        # Up, down, up pattern
        values = [0, 5, 10, 5, 0, 5, 10]
        assert _count_direction_changes(values, min_displacement=2.0) == 2

    def test_noise_filtered(self):
        # Small oscillation below threshold
        values = [0, 0.1, -0.1, 0.2, -0.2]
        assert _count_direction_changes(values, min_displacement=1.0) == 0

    def test_too_few_values(self):
        assert _count_direction_changes([1, 2]) == 0
        assert _count_direction_changes([1]) == 0
        assert _count_direction_changes([]) == 0


class TestCountCrossings:
    def test_single_crossing(self):
        assert _count_crossings([-1, 1], crossing_value=0.0) == 1

    def test_no_crossing(self):
        assert _count_crossings([1, 2, 3], crossing_value=0.0) == 0

    def test_multiple_crossings(self):
        assert _count_crossings([-1, 1, -1, 1], crossing_value=0.0) == 3

    def test_crossing_at_custom_value(self):
        assert _count_crossings([0, 200, 100, 300], crossing_value=150) == 3

    def test_empty(self):
        assert _count_crossings([], 0.0) == 0
        assert _count_crossings([5], 0.0) == 0


class TestCountShotsPixel:
    def test_basic_rally(self):
        # Simulate ball bouncing up and down (direction changes = shots)
        dets = []
        for i in range(60):
            y = 200 + 100 * np.sin(i * 0.2)  # oscillating y
            dets.append(BallDetection(frame_number=i, x=320.0, y=float(y), confidence=1.0))

        rally = make_rally(1, start=0, end=59, fps=30.0)
        result = count_shots_pixel(dets, rally, frame_height=480)
        assert result.shot_count > 0
        assert result.rally_id == 1

    def test_no_detections_in_rally(self):
        dets = [BallDetection(frame_number=i, x=None, y=None, confidence=0.0) for i in range(30)]
        rally = make_rally(1, start=0, end=29, fps=30.0)
        result = count_shots_pixel(dets, rally, frame_height=480)
        assert result.shot_count == 0


# --- Stats Tests ---

class TestComputeRallyStats:
    def test_basic_stats(self):
        rallies = [
            make_rally(1, 0, 150, fps=30.0),   # 5s
            make_rally(2, 200, 290, fps=30.0),  # 3s
            make_rally(3, 400, 700, fps=30.0),  # 10s
        ]
        shots = [
            RallyShots(1, shot_count=4, net_crossings=3, start_frame=0, end_frame=150, duration=5.0),
            RallyShots(2, shot_count=2, net_crossings=1, start_frame=200, end_frame=290, duration=3.0),
            RallyShots(3, shot_count=8, net_crossings=7, start_frame=400, end_frame=700, duration=10.0),
        ]
        stats = compute_rally_stats(rallies, shots)

        assert stats.rally_count == 3
        assert stats.total_shots == 14
        assert stats.longest_rally_shots == 8
        assert stats.shortest_rally_shots == 2
        assert len(stats.shots_per_rally) == 3
        assert len(stats.shot_count_distribution) > 0

    def test_empty_stats(self):
        stats = compute_rally_stats([], [])
        assert stats.rally_count == 0
        assert stats.total_shots == 0

    def test_to_dict(self):
        rallies = [make_rally(1, 0, 150, fps=30.0)]
        shots = [RallyShots(1, shot_count=4, net_crossings=3, start_frame=0, end_frame=150, duration=5.0)]
        stats = compute_rally_stats(rallies, shots)
        d = stats.to_dict()
        assert isinstance(d, dict)
        assert "rally_count" in d
        assert "shots_per_rally" in d


class TestFormatStatsSummary:
    def test_no_rallies(self):
        stats = compute_rally_stats([], [])
        summary = format_stats_summary(stats)
        assert "No rallies" in summary

    def test_with_rallies(self):
        rallies = [make_rally(1, 0, 150, fps=30.0)]
        shots = [RallyShots(1, shot_count=4, net_crossings=3, start_frame=0, end_frame=150, duration=5.0)]
        stats = compute_rally_stats(rallies, shots)
        summary = format_stats_summary(stats)
        assert "Rally Statistics" in summary
        assert "1" in summary


# --- Visualization Feedback Fix Tests ---

class TestFilterCourtPositionsValidation:
    def test_invalid_region_raises(self):
        from src.features.visualization.heatmap import filter_court_positions
        with pytest.raises(ValueError, match="Invalid region"):
            filter_court_positions([(0, 0)], region="typo")

    def test_valid_regions(self):
        from src.features.visualization.heatmap import filter_court_positions
        positions = [(0.0, 0.0)]
        for region in ["full", "service_boxes", "near_service", "far_service", "near_half", "far_half"]:
            filter_court_positions(positions, region=region)  # should not raise


class TestEmbedOverlayPositionValidation:
    def test_invalid_position_raises(self):
        from src.features.visualization.renderer import embed_overlay_in_frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        overlay = np.zeros((100, 80, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Invalid position"):
            embed_overlay_in_frame(frame, overlay, position="invalid")

    def test_valid_positions(self):
        from src.features.visualization.renderer import embed_overlay_in_frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        overlay = np.zeros((100, 80, 3), dtype=np.uint8)
        for pos in ["top-left", "top-right", "bottom-left", "bottom-right"]:
            result = embed_overlay_in_frame(frame, overlay, position=pos)
            assert result.shape == frame.shape


# --- Court Template Tests ---

class TestCourtTemplate:
    def test_itf_dimensions(self):
        from src.features.court_detect.court_template import ITF_DIMS
        assert abs(ITF_DIMS.court_length - 23.77) < 0.01
        assert abs(ITF_DIMS.singles_width - 8.23) < 0.01
        assert abs(ITF_DIMS.doubles_width - 10.97) < 0.01

    def test_reference_keypoints_count(self):
        from src.features.court_detect.court_template import REFERENCE_KPS_METERS
        assert len(REFERENCE_KPS_METERS) == 14


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
