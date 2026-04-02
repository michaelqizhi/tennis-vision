"""Tests for the three-layer scoring harness.

Covers metrics computation, ground truth parsing, and report generation
with known inputs/outputs. Tests edge cases: no detections, perfect
detections, all false positives, occluded frames, etc.
"""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import pytest

from src.features.scoring.ground_truth import (
    GroundTruthData,
    GroundTruthFrame,
    _extract_rallies,
    parse_cvat_xml,
)
from src.features.scoring.metrics import (
    EventMetrics,
    FrameMetrics,
    Prediction,
    ScoringConfig,
    ScoringResult,
    TrajectoryMetrics,
    _temporal_iou,
    compute_event_metrics,
    compute_frame_metrics,
    compute_trajectory_metrics,
    run_scoring,
)
from src.features.scoring.report import format_json_report, format_text_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gt(frames_dict: dict[int, tuple[float, float, bool]], gap_tolerance: int = 5) -> GroundTruthData:
    """Build GroundTruthData from a simple dict: {frame: (x, y, visible)}."""
    frames = {
        f: GroundTruthFrame(frame_number=f, x=x, y=y, visible=vis)
        for f, (x, y, vis) in frames_dict.items()
    }
    rallies = _extract_rallies(sorted(frames.keys()), gap_tolerance)
    return GroundTruthData(frames=frames, rallies=rallies)


def _preds(entries: list[tuple[int, float | None, float | None]]) -> list[Prediction]:
    """Build predictions from [(frame, x, y), ...]."""
    return [
        Prediction(frame_number=f, x=x, y=y)
        for f, x, y in entries
    ]


# ============================================================================
# Layer 1: Frame-level localization
# ============================================================================

class TestFrameMetrics:
    """Tests for compute_frame_metrics (Layer 1)."""

    def test_perfect_detections(self):
        """All predictions exactly match ground truth."""
        gt = _gt({0: (100, 200, True), 1: (110, 210, True), 2: (120, 220, True)})
        preds = _preds([(0, 100, 200), (1, 110, 210), (2, 120, 220)])
        m = compute_frame_metrics(preds, gt)

        assert m.true_positives == 3
        assert m.false_positives == 0
        assert m.false_negatives == 0
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.mean_error == 0.0

    def test_no_detections(self):
        """No predictions at all — all FN."""
        gt = _gt({0: (100, 200, True), 1: (110, 210, True)})
        preds: list[Prediction] = []
        m = compute_frame_metrics(preds, gt)

        assert m.true_positives == 0
        assert m.false_positives == 0
        assert m.false_negatives == 2
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_no_detections_with_none_coords(self):
        """Predictions exist but with None coordinates — treated as no detection."""
        gt = _gt({0: (100, 200, True), 1: (110, 210, True)})
        preds = _preds([(0, None, None), (1, None, None)])
        m = compute_frame_metrics(preds, gt)

        assert m.true_positives == 0
        assert m.false_negatives == 2

    def test_all_false_positives(self):
        """Predictions during dead time within annotated range → FP."""
        # GT exists at frame 0 and 10 (defines annotated range 0-10)
        # Predictions at frames 3 and 5 are in dead time → FP
        gt = _gt({0: (100, 200, True), 10: (200, 300, True)})
        preds = _preds([
            (0, 100, 200),   # TP
            (3, 500, 500),   # FP (dead time, no GT)
            (5, 600, 600),   # FP (dead time, no GT)
            (10, 200, 300),  # TP
        ])
        m = compute_frame_metrics(preds, gt)

        assert m.true_positives == 2
        assert m.false_positives == 2
        assert m.false_negatives == 0

    def test_predictions_outside_gt_range_ignored(self):
        """Predictions outside annotated frame range are not scored."""
        gt = _gt({10: (100, 200, True), 20: (110, 210, True)})
        preds = _preds([
            (0, 500, 500),    # outside GT range → ignored
            (10, 100, 200),   # TP
            (20, 110, 210),   # TP
            (100, 500, 500),  # outside GT range → ignored
        ])
        m = compute_frame_metrics(preds, gt)

        assert m.true_positives == 2
        assert m.false_positives == 0
        assert m.false_negatives == 0

    def test_empty_gt_no_scoring(self):
        """Empty GT → nothing to score."""
        gt = _gt({})
        preds = _preds([(0, 100, 200), (1, 110, 210)])
        m = compute_frame_metrics(preds, gt)

        assert m.true_positives == 0
        assert m.false_positives == 0
        assert m.false_negatives == 0

    def test_distance_threshold_boundary(self):
        """Prediction exactly at threshold distance = TP."""
        tau = 15.0
        # Distance = 15.0 exactly (3-4-5 triangle scaled: 9, 12 → dist=15)
        gt = _gt({0: (100, 200, True)})
        preds = _preds([(0, 109, 212)])
        cfg = ScoringConfig(distance_threshold=tau)
        m = compute_frame_metrics(preds, gt, cfg)

        assert m.true_positives == 1
        assert m.false_positives == 0

    def test_distance_threshold_exceeded(self):
        """Prediction just beyond threshold = FP + FN."""
        gt = _gt({0: (100, 200, True)})
        preds = _preds([(0, 200, 200)])  # distance = 100
        cfg = ScoringConfig(distance_threshold=15.0)
        m = compute_frame_metrics(preds, gt, cfg)

        assert m.true_positives == 0
        assert m.false_positives == 1
        assert m.false_negatives == 1

    def test_localization_error_stats(self):
        """Check mean, P50, P90 error computation."""
        gt = _gt({
            0: (100, 200, True),
            1: (110, 210, True),
            2: (120, 220, True),
            3: (130, 230, True),
        })
        # Errors: 0, 5, 10, 2
        preds = _preds([
            (0, 100, 200),      # err = 0
            (1, 113, 214),      # err = 5
            (2, 126, 228),      # err = 10
            (3, 131, 231),      # err ≈ √5 ≈ 2.236
        ])
        cfg = ScoringConfig(distance_threshold=15.0)
        m = compute_frame_metrics(preds, gt, cfg)

        assert m.true_positives == 4
        assert m.mean_error == pytest.approx(sum(m.errors) / 4, abs=0.01)
        # P50: median of sorted [0, ~2.24, 5, 10]
        assert m.p50_error > 0
        assert m.p90_error >= m.p50_error

    def test_occluded_frames_tracked_separately(self):
        """Occluded frames count as TP/FN but tracked in separate counters."""
        gt = _gt({
            0: (100, 200, True),
            1: (110, 210, False),  # occluded
            2: (120, 220, False),  # occluded
        })
        preds = _preds([
            (0, 100, 200),  # TP visible
            (1, 110, 210),  # TP occluded
            # frame 2 missing → FN occluded
        ])
        m = compute_frame_metrics(preds, gt)

        assert m.true_positives == 2
        assert m.false_negatives == 1
        assert m.occluded_tp == 1
        assert m.occluded_fn == 1

    def test_mixed_scenario(self):
        """Mix of TP, FP, FN across frames."""
        gt = _gt({0: (100, 200, True), 2: (120, 220, True), 4: (140, 240, True)})
        preds = _preds([
            (0, 101, 201),  # TP (close)
            (1, 500, 500),  # FP (no GT for frame 1)
            (2, 999, 999),  # FP+FN (too far from GT)
            # frame 4 missing → FN
        ])
        cfg = ScoringConfig(distance_threshold=15.0)
        m = compute_frame_metrics(preds, gt, cfg)

        assert m.true_positives == 1
        assert m.false_positives == 2  # frame 1 + frame 2
        assert m.false_negatives == 2  # frame 2 + frame 4

    def test_empty_gt_empty_preds(self):
        """Both empty — no metrics, no errors."""
        gt = _gt({})
        preds: list[Prediction] = []
        m = compute_frame_metrics(preds, gt)

        assert m.true_positives == 0
        assert m.false_positives == 0
        assert m.false_negatives == 0
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_configurable_threshold(self):
        """Custom threshold changes TP/FP classification."""
        gt = _gt({0: (100, 200, True)})
        preds = _preds([(0, 120, 200)])  # distance = 20

        # With τ=15: FP+FN
        m1 = compute_frame_metrics(preds, gt, ScoringConfig(distance_threshold=15.0))
        assert m1.true_positives == 0

        # With τ=25: TP
        m2 = compute_frame_metrics(preds, gt, ScoringConfig(distance_threshold=25.0))
        assert m2.true_positives == 1


# ============================================================================
# Layer 2: Trajectory continuity
# ============================================================================

class TestTrajectoryMetrics:
    """Tests for compute_trajectory_metrics (Layer 2)."""

    def test_no_breaks(self):
        """Continuous detections within a rally — zero breaks."""
        gt = _gt({f: (100 + f, 200 + f, True) for f in range(30)})
        preds = _preds([(f, 100 + f, 200 + f) for f in range(30)])
        m = compute_trajectory_metrics(preds, gt, ScoringConfig(fps=30.0))

        assert m.num_breaks == 0
        assert m.max_break_duration == 0

    def test_single_break(self):
        """One gap in detections within GT region."""
        gt_dict = {f: (100 + f, 200, True) for f in range(20)}
        gt = _gt(gt_dict)
        # Missing frames 8-12 (5 frames)
        preds = _preds(
            [(f, 100 + f, 200) for f in range(8)] +
            [(f, 100 + f, 200) for f in range(13, 20)]
        )
        m = compute_trajectory_metrics(preds, gt, ScoringConfig(fps=30.0))

        assert m.num_breaks == 1
        assert m.max_break_duration == 5

    def test_multiple_breaks(self):
        """Multiple gaps within GT region."""
        gt_dict = {f: (100 + f, 200, True) for f in range(30)}
        gt = _gt(gt_dict)
        # Missing frames 5-7 (3 frames) and 15-19 (5 frames)
        detected = [f for f in range(30) if not (5 <= f <= 7) or not (15 <= f <= 19)]
        detected = [f for f in range(30) if f < 5 or (8 <= f <= 14) or f >= 20]
        preds = _preds([(f, 100 + f, 200) for f in detected])
        m = compute_trajectory_metrics(preds, gt, ScoringConfig(fps=30.0))

        assert m.num_breaks == 2
        assert m.max_break_duration == 5  # frames 15-19

    def test_breaks_per_minute(self):
        """Breaks per minute calculated correctly."""
        # 60 frames at 30fps = 2 seconds = 1/30 minutes
        gt_dict = {f: (100, 200, True) for f in range(60)}
        gt = _gt(gt_dict)
        # Two breaks
        detected = [f for f in range(60) if f < 10 or (15 <= f < 30) or f >= 35]
        preds = _preds([(f, 100, 200) for f in detected])
        m = compute_trajectory_metrics(preds, gt, ScoringConfig(fps=30.0))

        assert m.num_breaks == 2
        # 60 frames at 30fps = 2 seconds = 1/30 minutes → 2 / (1/30) = 60 breaks/min
        assert m.breaks_per_minute == pytest.approx(60.0, rel=0.01)

    def test_velocity_outliers(self):
        """Detect velocity outliers."""
        gt_dict = {f: (100, 200, True) for f in range(5)}
        gt = _gt(gt_dict)
        # Frame 0→1: move 10px (ok), 1→2: move 100px (outlier at max_vel=50)
        preds = _preds([
            (0, 100, 200),
            (1, 110, 200),  # vel = 10 px/frame
            (2, 210, 200),  # vel = 100 px/frame (outlier!)
            (3, 215, 200),  # vel = 5 px/frame
        ])
        cfg = ScoringConfig(max_velocity=50.0)
        m = compute_trajectory_metrics(preds, gt, cfg)

        assert m.velocity_outliers == 1

    def test_acceleration_outliers(self):
        """Detect acceleration outliers."""
        gt_dict = {f: (100, 200, True) for f in range(5)}
        gt = _gt(gt_dict)
        # Velocities: 5, 5, 45 → accelerations: 0, 40
        preds = _preds([
            (0, 100, 200),
            (1, 105, 200),  # vel = 5
            (2, 110, 200),  # vel = 5, accel = 0
            (3, 155, 200),  # vel = 45, accel = 40 (outlier!)
        ])
        cfg = ScoringConfig(max_velocity=50.0, max_acceleration=30.0)
        m = compute_trajectory_metrics(preds, gt, cfg)

        assert m.acceleration_outliers == 1

    def test_physical_consistency_perfect(self):
        """No outliers → consistency = 1.0."""
        gt_dict = {f: (100 + f, 200, True) for f in range(10)}
        gt = _gt(gt_dict)
        preds = _preds([(f, 100 + f, 200) for f in range(10)])
        cfg = ScoringConfig(max_velocity=50.0, max_acceleration=30.0)
        m = compute_trajectory_metrics(preds, gt, cfg)

        assert m.physical_consistency == 1.0

    def test_empty_gt(self):
        """No ground truth → empty trajectory metrics."""
        gt = _gt({})
        preds = _preds([(0, 100, 200)])
        m = compute_trajectory_metrics(preds, gt)

        assert m.num_breaks == 0
        assert m.physical_consistency == 1.0

    def test_no_predictions(self):
        """No predictions → everything is a break."""
        gt_dict = {f: (100 + f, 200, True) for f in range(30)}
        gt = _gt(gt_dict)
        preds: list[Prediction] = []
        m = compute_trajectory_metrics(preds, gt, ScoringConfig(fps=30.0))

        # Entire rally is one break
        assert m.num_breaks == 1
        assert m.max_break_duration == 30


# ============================================================================
# Layer 3: Event-level (rally segmentation)
# ============================================================================

class TestEventMetrics:
    """Tests for compute_event_metrics (Layer 3)."""

    def test_perfect_rally_match(self):
        """Prediction rally exactly matches GT rally."""
        gt = _gt({f: (100, 200, True) for f in range(100)})
        preds = _preds([(f, 100, 200) for f in range(100)])
        m = compute_event_metrics(preds, gt)

        assert m.gt_rally_count == 1
        assert m.pred_rally_count == 1
        assert m.mean_iou == pytest.approx(1.0)
        assert m.rally_recall == 1.0
        assert m.rally_false_positive_rate == 0.0

    def test_no_predictions_for_rallies(self):
        """No predictions → recall = 0."""
        gt = _gt({f: (100, 200, True) for f in range(100)})
        preds: list[Prediction] = []
        m = compute_event_metrics(preds, gt)

        assert m.rally_recall == 0.0

    def test_extra_predicted_rallies(self):
        """Predictions in non-GT regions → false positives."""
        gt = _gt({f: (100, 200, True) for f in range(50, 150)})
        preds = _preds(
            [(f, 100, 200) for f in range(50, 150)] +  # matches GT
            [(f, 100, 200) for f in range(200, 250)]    # spurious rally
        )
        cfg = ScoringConfig(rally_gap_tolerance=5, rally_iou_threshold=0.5)
        m = compute_event_metrics(preds, gt, cfg)

        assert m.pred_rally_count == 2
        assert m.rally_recall == 1.0
        assert m.rally_false_positive_rate == pytest.approx(0.5)

    def test_partial_overlap(self):
        """Predicted rally partially overlaps GT rally."""
        gt = _gt({f: (100, 200, True) for f in range(0, 100)})
        # Predictions cover frames 50-149 (50% overlap)
        preds = _preds([(f, 100, 200) for f in range(50, 150)])
        cfg = ScoringConfig(rally_iou_threshold=0.5)
        m = compute_event_metrics(preds, gt, cfg)

        # IoU = 50/150 ≈ 0.333 < 0.5 threshold
        assert m.mean_iou == pytest.approx(50 / 150, abs=0.01)
        assert m.rally_recall == 0.0  # Below threshold

    def test_multiple_gt_rallies(self):
        """Multiple GT rallies with matching predictions."""
        gt = _gt({
            **{f: (100, 200, True) for f in range(0, 50)},
            **{f: (100, 200, True) for f in range(100, 150)},
        }, gap_tolerance=5)
        preds = _preds(
            [(f, 100, 200) for f in range(0, 50)] +
            [(f, 100, 200) for f in range(100, 150)]
        )
        cfg = ScoringConfig(rally_gap_tolerance=5)
        m = compute_event_metrics(preds, gt, cfg)

        assert m.gt_rally_count == 2
        assert m.rally_recall == 1.0

    def test_temporal_iou_no_overlap(self):
        """Non-overlapping ranges → IoU = 0."""
        assert _temporal_iou((0, 10), (20, 30)) == 0.0

    def test_temporal_iou_full_overlap(self):
        """Identical ranges → IoU = 1."""
        assert _temporal_iou((5, 15), (5, 15)) == 1.0

    def test_temporal_iou_partial(self):
        """Partial overlap → correct IoU."""
        # (0,10) ∩ (5,15): intersection = 5-10 = 6 frames, union = 0-15 = 16 frames
        iou = _temporal_iou((0, 10), (5, 15))
        assert iou == pytest.approx(6 / 16, abs=0.001)


# ============================================================================
# Ground truth parser
# ============================================================================

class TestGroundTruthParser:
    """Tests for CVAT XML parsing."""

    def test_parse_track_format(self, tmp_path):
        """Parse CVAT track format with points."""
        xml_content = textwrap.dedent("""\
            <?xml version="1.0" encoding="utf-8"?>
            <annotations>
              <track id="0" label="ball">
                <points frame="0" points="100.5,200.3" outside="0" occluded="0"/>
                <points frame="1" points="110.0,210.0" outside="0" occluded="1"/>
                <points frame="2" points="120.0,220.0" outside="0" occluded="0"/>
                <points frame="10" points="180.0,280.0" outside="0" occluded="0"/>
              </track>
            </annotations>
        """)
        xml_file = tmp_path / "annotations.xml"
        xml_file.write_text(xml_content, encoding="utf-8")
        data = parse_cvat_xml(xml_file, gap_tolerance=5)

        assert len(data.frames) == 4
        assert data.frames[0].x == pytest.approx(100.5)
        assert data.frames[0].y == pytest.approx(200.3)
        assert data.frames[0].visible is True
        assert data.frames[1].visible is False  # occluded
        # Rallies: frames 0-2 (continuous), frame 10 (gap > 5 → separate)
        assert len(data.rallies) == 2
        assert data.rallies[0] == (0, 2)
        assert data.rallies[1] == (10, 10)

    def test_parse_box_format(self, tmp_path):
        """Parse CVAT track format with bounding boxes."""
        xml_content = textwrap.dedent("""\
            <?xml version="1.0" encoding="utf-8"?>
            <annotations>
              <track id="0" label="tennis_ball">
                <box frame="5" xtl="95" ytl="195" xbr="105" ybr="205" outside="0" occluded="0"/>
              </track>
            </annotations>
        """)
        xml_file = tmp_path / "annotations.xml"
        xml_file.write_text(xml_content, encoding="utf-8")
        data = parse_cvat_xml(xml_file)

        assert 5 in data.frames
        assert data.frames[5].x == pytest.approx(100.0)  # center
        assert data.frames[5].y == pytest.approx(200.0)

    def test_outside_frames_excluded(self, tmp_path):
        """Frames marked as outside=1 are excluded."""
        xml_content = textwrap.dedent("""\
            <?xml version="1.0" encoding="utf-8"?>
            <annotations>
              <track id="0" label="ball">
                <points frame="0" points="100,200" outside="0" occluded="0"/>
                <points frame="1" points="110,210" outside="1" occluded="0"/>
                <points frame="2" points="120,220" outside="0" occluded="0"/>
              </track>
            </annotations>
        """)
        xml_file = tmp_path / "annotations.xml"
        xml_file.write_text(xml_content, encoding="utf-8")
        data = parse_cvat_xml(xml_file)

        assert len(data.frames) == 2
        assert 1 not in data.frames

    def test_visibility_attribute_override(self, tmp_path):
        """Visibility attribute overrides occluded flag."""
        xml_content = textwrap.dedent("""\
            <?xml version="1.0" encoding="utf-8"?>
            <annotations>
              <track id="0" label="ball">
                <points frame="0" points="100,200" outside="0" occluded="0">
                  <attribute name="visibility">occluded</attribute>
                </points>
              </track>
            </annotations>
        """)
        xml_file = tmp_path / "annotations.xml"
        xml_file.write_text(xml_content, encoding="utf-8")
        data = parse_cvat_xml(xml_file)

        assert data.frames[0].visible is False

    def test_per_image_format(self, tmp_path):
        """Parse CVAT per-image (not track) format."""
        xml_content = textwrap.dedent("""\
            <?xml version="1.0" encoding="utf-8"?>
            <annotations>
              <image id="0" name="frame_0.png" width="1920" height="1080">
                <points label="ball" points="100,200" occluded="0"/>
              </image>
              <image id="1" name="frame_1.png" width="1920" height="1080">
                <points label="ball" points="110,210" occluded="0"/>
              </image>
            </annotations>
        """)
        xml_file = tmp_path / "annotations.xml"
        xml_file.write_text(xml_content, encoding="utf-8")
        data = parse_cvat_xml(xml_file)

        assert len(data.frames) == 2
        assert data.frames[0].x == pytest.approx(100.0)
        assert data.frames[1].x == pytest.approx(110.0)

    def test_empty_xml(self, tmp_path):
        """Empty annotations → empty data."""
        xml_content = '<?xml version="1.0"?><annotations></annotations>'
        xml_file = tmp_path / "annotations.xml"
        xml_file.write_text(xml_content, encoding="utf-8")
        data = parse_cvat_xml(xml_file)

        assert len(data.frames) == 0
        assert len(data.rallies) == 0

    def test_non_ball_labels_ignored(self, tmp_path):
        """Non-ball labels are ignored."""
        xml_content = textwrap.dedent("""\
            <?xml version="1.0" encoding="utf-8"?>
            <annotations>
              <track id="0" label="player">
                <points frame="0" points="500,300" outside="0" occluded="0"/>
              </track>
              <track id="1" label="ball">
                <points frame="0" points="100,200" outside="0" occluded="0"/>
              </track>
            </annotations>
        """)
        xml_file = tmp_path / "annotations.xml"
        xml_file.write_text(xml_content, encoding="utf-8")
        data = parse_cvat_xml(xml_file)

        assert len(data.frames) == 1
        assert data.frames[0].x == pytest.approx(100.0)


# ============================================================================
# Rally extraction
# ============================================================================

class TestRallyExtraction:
    """Tests for _extract_rallies."""

    def test_single_contiguous_block(self):
        rallies = _extract_rallies([0, 1, 2, 3, 4], gap_tolerance=5)
        assert rallies == [(0, 4)]

    def test_two_rallies_with_gap(self):
        rallies = _extract_rallies([0, 1, 2, 20, 21, 22], gap_tolerance=5)
        assert rallies == [(0, 2), (20, 22)]

    def test_gap_within_tolerance(self):
        """Gap of 3 frames with tolerance=5 → single rally."""
        rallies = _extract_rallies([0, 1, 2, 5, 6, 7], gap_tolerance=5)
        assert rallies == [(0, 7)]

    def test_gap_at_tolerance_boundary(self):
        """Gap exactly at tolerance → single rally."""
        rallies = _extract_rallies([0, 5], gap_tolerance=5)
        assert rallies == [(0, 5)]

    def test_gap_exceeds_tolerance(self):
        """Gap beyond tolerance → two rallies."""
        rallies = _extract_rallies([0, 6], gap_tolerance=5)
        assert rallies == [(0, 0), (6, 6)]

    def test_empty_input(self):
        assert _extract_rallies([], gap_tolerance=5) == []

    def test_single_frame(self):
        assert _extract_rallies([42], gap_tolerance=5) == [(42, 42)]


# ============================================================================
# Report formatting
# ============================================================================

class TestReporting:
    """Tests for report generation."""

    def _make_result(self) -> ScoringResult:
        gt = _gt({f: (100 + f, 200, True) for f in range(30)})
        preds = _preds([(f, 100 + f, 200) for f in range(30)])
        cfg = ScoringConfig(fps=30.0)
        return run_scoring(preds, gt, cfg)

    def test_json_report_is_valid_json(self):
        result = self._make_result()
        report = format_json_report(result)
        parsed = json.loads(report)

        assert "layer1_frame" in parsed
        assert "layer2_trajectory" in parsed
        assert "layer3_event" in parsed
        assert "config" in parsed

    def test_json_report_values(self):
        result = self._make_result()
        parsed = json.loads(format_json_report(result))

        assert parsed["layer1_frame"]["f1"] == 1.0
        assert parsed["layer1_frame"]["precision"] == 1.0
        assert parsed["layer2_trajectory"]["num_breaks"] == 0
        assert parsed["layer3_event"]["rally_recall"] == 1.0

    def test_text_summary_contains_key_sections(self):
        result = self._make_result()
        text = format_text_summary(result)

        assert "LAYER 1" in text
        assert "LAYER 2" in text
        assert "LAYER 3" in text
        assert "Precision" in text
        assert "Breaks" in text
        assert "Rally recall" in text


# ============================================================================
# Integration: run_scoring end-to-end
# ============================================================================

class TestRunScoring:
    """Integration tests for the full scoring pipeline."""

    def test_perfect_tracking(self):
        """Perfect predictions → all metrics ideal."""
        gt = _gt({f: (100 + f * 2, 200 + f, True) for f in range(90)})
        preds = _preds([(f, 100 + f * 2, 200 + f) for f in range(90)])
        result = run_scoring(preds, gt, ScoringConfig(fps=30.0))

        assert result.frame_metrics.f1 == 1.0
        assert result.trajectory_metrics.num_breaks == 0
        assert result.event_metrics.rally_recall == 1.0

    def test_complete_failure(self):
        """No detections → worst-case metrics."""
        gt = _gt({f: (100, 200, True) for f in range(90)})
        preds: list[Prediction] = []
        result = run_scoring(preds, gt, ScoringConfig(fps=30.0))

        assert result.frame_metrics.f1 == 0.0
        assert result.frame_metrics.recall == 0.0
        assert result.trajectory_metrics.num_breaks >= 1
        assert result.event_metrics.rally_recall == 0.0

    def test_half_detected(self):
        """Only first half detected."""
        gt = _gt({f: (100 + f, 200, True) for f in range(60)})
        preds = _preds([(f, 100 + f, 200) for f in range(30)])
        result = run_scoring(preds, gt, ScoringConfig(fps=30.0))

        assert 0 < result.frame_metrics.recall < 1.0
        assert result.frame_metrics.precision == 1.0
        assert result.trajectory_metrics.num_breaks >= 1

    def test_scoring_result_has_config(self):
        """ScoringResult preserves the config used."""
        cfg = ScoringConfig(distance_threshold=20.0, fps=60.0)
        gt = _gt({0: (100, 200, True)})
        preds = _preds([(0, 100, 200)])
        result = run_scoring(preds, gt, cfg)

        assert result.config.distance_threshold == 20.0
        assert result.config.fps == 60.0
