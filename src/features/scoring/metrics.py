"""Three-layer metric computation for tennis ball tracking evaluation.

Layer 1: Frame-level localization (precision, recall, F1, localization error)
Layer 2: Trajectory continuity (break rate, physical consistency)
Layer 3: Event-level (rally segmentation IoU, recall, FP rate)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.features.scoring.ground_truth import GroundTruthData, GroundTruthFrame


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ScoringConfig:
    """Configuration for the scoring harness."""
    # Layer 1
    distance_threshold: float = 15.0  # τ pixels for TP matching
    # Layer 2
    max_velocity: float = 50.0  # px/frame — flag if exceeded
    max_acceleration: float = 30.0  # px/frame² — flag if exceeded
    # Layer 3
    rally_gap_tolerance: int = 5  # frames — bridge gaps in predicted rallies
    rally_iou_threshold: float = 0.5  # min IoU for rally match
    # General
    fps: float = 30.0


# ---------------------------------------------------------------------------
# Prediction representation (lightweight — avoid importing torch/cv2)
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    """Minimal prediction record for scoring (decoupled from BallDetection)."""
    frame_number: int
    x: float | None
    y: float | None
    confidence: float = 1.0
    interpolated: bool = False

    @property
    def detected(self) -> bool:
        return self.x is not None and self.y is not None


# ---------------------------------------------------------------------------
# Layer 1: Frame-level localization
# ---------------------------------------------------------------------------

@dataclass
class FrameMetrics:
    """Frame-level localization metrics."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    # Occluded-specific counts (subset of above)
    occluded_tp: int = 0
    occluded_fn: int = 0
    # Localization errors for TP frames
    errors: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def mean_error(self) -> float:
        return sum(self.errors) / len(self.errors) if self.errors else 0.0

    @property
    def p50_error(self) -> float:
        return _percentile(self.errors, 50)

    @property
    def p90_error(self) -> float:
        return _percentile(self.errors, 90)


def _percentile(values: list[float], pct: float) -> float:
    """Compute percentile without numpy dependency."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def compute_frame_metrics(
    predictions: list[Prediction],
    ground_truth: GroundTruthData,
    config: ScoringConfig | None = None,
) -> FrameMetrics:
    """Compute Layer 1 frame-level localization metrics.

    Args:
        predictions: Model predictions (one per frame).
        ground_truth: Parsed ground truth data.
        config: Scoring configuration.

    Returns:
        FrameMetrics with precision, recall, F1, and error stats.
    """
    cfg = config or ScoringConfig()
    tau = cfg.distance_threshold
    metrics = FrameMetrics()

    pred_map: dict[int, Prediction] = {p.frame_number: p for p in predictions}

    # Restrict evaluation to the annotated frame range.
    # Predictions outside this range are not scored (no GT to judge them).
    # Within the range, predictions during dead time (no GT) count as FP.
    if not ground_truth.frames:
        return metrics
    min_f = min(ground_truth.frames.keys())
    max_f = max(ground_truth.frames.keys())
    all_frames = set(ground_truth.frames.keys()) | {
        f for f in pred_map if min_f <= f <= max_f
    }

    for f in all_frames:
        gt: GroundTruthFrame | None = ground_truth.frames.get(f)
        pred: Prediction | None = pred_map.get(f)
        pred_exists = pred is not None and pred.detected
        gt_exists = gt is not None

        if gt_exists and pred_exists:
            dist = math.hypot(pred.x - gt.x, pred.y - gt.y)
            if dist <= tau:
                metrics.true_positives += 1
                metrics.errors.append(dist)
                if not gt.visible:
                    metrics.occluded_tp += 1
            else:
                metrics.false_positives += 1
                metrics.false_negatives += 1
                if not gt.visible:
                    metrics.occluded_fn += 1
        elif pred_exists and not gt_exists:
            metrics.false_positives += 1
        elif gt_exists and not pred_exists:
            metrics.false_negatives += 1
            if not gt.visible:
                metrics.occluded_fn += 1

    return metrics


# ---------------------------------------------------------------------------
# Layer 2: Trajectory continuity
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryMetrics:
    """Trajectory-level continuity metrics."""
    num_breaks: int = 0
    max_break_duration: int = 0  # frames
    break_durations: list[int] = field(default_factory=list)
    breaks_per_minute: float = 0.0
    # Physical consistency
    total_consecutive_pairs: int = 0
    velocity_outliers: int = 0
    acceleration_outliers: int = 0

    @property
    def physical_consistency(self) -> float:
        """Fraction of consecutive-prediction frames WITHOUT outliers."""
        if self.total_consecutive_pairs == 0:
            return 1.0
        outlier_frames = self.velocity_outliers + self.acceleration_outliers
        return 1.0 - min(outlier_frames / self.total_consecutive_pairs, 1.0)


def compute_trajectory_metrics(
    predictions: list[Prediction],
    ground_truth: GroundTruthData,
    config: ScoringConfig | None = None,
) -> TrajectoryMetrics:
    """Compute Layer 2 trajectory continuity metrics.

    Breaks are measured only within annotated (GT) regions.

    Args:
        predictions: Model predictions.
        ground_truth: Parsed ground truth data.
        config: Scoring configuration.

    Returns:
        TrajectoryMetrics with break rate and physical consistency.
    """
    cfg = config or ScoringConfig()
    metrics = TrajectoryMetrics()

    pred_map: dict[int, Prediction] = {p.frame_number: p for p in predictions}

    # --- Break analysis (within GT annotated regions) ---
    gt_frames_sorted = sorted(ground_truth.frames.keys())
    if not gt_frames_sorted:
        return metrics

    annotated_duration_frames = 0
    for start, end in ground_truth.rallies:
        annotated_duration_frames += end - start + 1

    # Walk through each GT rally and find breaks (consecutive missing preds)
    for rally_start, rally_end in ground_truth.rallies:
        in_break = False
        break_len = 0
        for f in range(rally_start, rally_end + 1):
            if f not in ground_truth.frames:
                continue
            pred = pred_map.get(f)
            has_pred = pred is not None and pred.detected

            if not has_pred:
                if not in_break:
                    in_break = True
                    break_len = 1
                else:
                    break_len += 1
            else:
                if in_break:
                    metrics.num_breaks += 1
                    metrics.break_durations.append(break_len)
                    in_break = False
                    break_len = 0

        # Handle break that extends to end of rally
        if in_break and break_len > 0:
            metrics.num_breaks += 1
            metrics.break_durations.append(break_len)

    metrics.max_break_duration = max(metrics.break_durations) if metrics.break_durations else 0
    annotated_minutes = annotated_duration_frames / cfg.fps / 60.0
    metrics.breaks_per_minute = (
        metrics.num_breaks / annotated_minutes if annotated_minutes > 0 else 0.0
    )

    # --- Physical consistency (velocity/acceleration) ---
    sorted_preds = sorted(
        [p for p in predictions if p.detected],
        key=lambda p: p.frame_number,
    )

    velocities: list[tuple[int, float]] = []  # (frame, velocity)
    for i in range(1, len(sorted_preds)):
        p0 = sorted_preds[i - 1]
        p1 = sorted_preds[i]
        dt = p1.frame_number - p0.frame_number
        if dt <= 0 or dt > 3:
            # Skip non-consecutive or large gaps
            continue
        dist = math.hypot(p1.x - p0.x, p1.y - p0.y)
        vel = dist / dt  # px/frame
        velocities.append((p1.frame_number, vel))
        metrics.total_consecutive_pairs += 1
        if vel > cfg.max_velocity:
            metrics.velocity_outliers += 1

    # Acceleration: consecutive velocity differences
    for i in range(1, len(velocities)):
        f0, v0 = velocities[i - 1]
        f1, v1 = velocities[i]
        dt = f1 - f0
        if dt <= 0 or dt > 3:
            continue
        accel = abs(v1 - v0) / dt
        if accel > cfg.max_acceleration:
            metrics.acceleration_outliers += 1

    return metrics


# ---------------------------------------------------------------------------
# Layer 3: Event-level (rally segmentation)
# ---------------------------------------------------------------------------

@dataclass
class RallyMatch:
    """Match between a GT rally and a predicted rally."""
    gt_rally: tuple[int, int]
    pred_rally: tuple[int, int]
    iou: float


@dataclass
class EventMetrics:
    """Event-level rally segmentation metrics."""
    gt_rally_count: int = 0
    pred_rally_count: int = 0
    rally_matches: list[RallyMatch] = field(default_factory=list)
    mean_iou: float = 0.0
    rally_recall: float = 0.0
    rally_false_positive_rate: float = 0.0


def _temporal_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Compute temporal IoU between two frame ranges (inclusive)."""
    inter_start = max(a[0], b[0])
    inter_end = min(a[1], b[1])
    if inter_start > inter_end:
        return 0.0
    intersection = inter_end - inter_start + 1
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - intersection
    return intersection / union if union > 0 else 0.0


def _extract_prediction_rallies(
    predictions: list[Prediction],
    gap_tolerance: int,
) -> list[tuple[int, int]]:
    """Extract rally segments from predictions.

    Consecutive detected frames with gaps <= gap_tolerance are merged.

    Args:
        predictions: Model predictions.
        gap_tolerance: Max frames without detection to bridge.

    Returns:
        List of (start_frame, end_frame) inclusive.
    """
    detected_frames = sorted(p.frame_number for p in predictions if p.detected)
    if not detected_frames:
        return []

    rallies: list[tuple[int, int]] = []
    start = detected_frames[0]
    prev = detected_frames[0]

    for f in detected_frames[1:]:
        if f - prev > gap_tolerance:
            rallies.append((start, prev))
            start = f
        prev = f

    rallies.append((start, prev))
    return rallies


def compute_event_metrics(
    predictions: list[Prediction],
    ground_truth: GroundTruthData,
    config: ScoringConfig | None = None,
) -> EventMetrics:
    """Compute Layer 3 event-level rally segmentation metrics.

    Args:
        predictions: Model predictions.
        ground_truth: Parsed ground truth data.
        config: Scoring configuration.

    Returns:
        EventMetrics with rally IoU, recall, and FP rate.
    """
    cfg = config or ScoringConfig()
    metrics = EventMetrics()

    gt_rallies = ground_truth.rallies
    pred_rallies = _extract_prediction_rallies(predictions, cfg.rally_gap_tolerance)

    metrics.gt_rally_count = len(gt_rallies)
    metrics.pred_rally_count = len(pred_rallies)

    if not gt_rallies or not pred_rallies:
        metrics.rally_recall = 0.0
        metrics.rally_false_positive_rate = 1.0 if pred_rallies else 0.0
        return metrics

    # For each GT rally find the best-matching predicted rally
    matched_pred_indices: set[int] = set()
    ious: list[float] = []
    recalled = 0

    for gt_rally in gt_rallies:
        best_iou = 0.0
        best_idx = -1
        for j, pred_rally in enumerate(pred_rallies):
            iou = _temporal_iou(gt_rally, pred_rally)
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_iou > 0:
            metrics.rally_matches.append(RallyMatch(
                gt_rally=gt_rally,
                pred_rally=pred_rallies[best_idx],
                iou=best_iou,
            ))
            ious.append(best_iou)
            if best_iou >= cfg.rally_iou_threshold:
                recalled += 1
                matched_pred_indices.add(best_idx)

    metrics.mean_iou = sum(ious) / len(ious) if ious else 0.0
    metrics.rally_recall = recalled / len(gt_rallies) if gt_rallies else 0.0

    # FP rate: predicted rallies that don't match any GT rally above threshold
    unmatched = 0
    for j in range(len(pred_rallies)):
        if j not in matched_pred_indices:
            # Check if this pred rally has any decent IoU with any GT rally
            has_match = any(
                _temporal_iou(gt_r, pred_rallies[j]) >= cfg.rally_iou_threshold
                for gt_r in gt_rallies
            )
            if not has_match:
                unmatched += 1

    metrics.rally_false_positive_rate = (
        unmatched / len(pred_rallies) if pred_rallies else 0.0
    )

    return metrics


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

@dataclass
class ScoringResult:
    """Complete three-layer scoring result."""
    frame_metrics: FrameMetrics
    trajectory_metrics: TrajectoryMetrics
    event_metrics: EventMetrics
    config: ScoringConfig


def run_scoring(
    predictions: list[Prediction],
    ground_truth: GroundTruthData,
    config: ScoringConfig | None = None,
) -> ScoringResult:
    """Run the full three-layer scoring harness.

    Args:
        predictions: Model predictions.
        ground_truth: Parsed ground truth data.
        config: Scoring configuration.

    Returns:
        ScoringResult with all three metric layers.
    """
    cfg = config or ScoringConfig()
    return ScoringResult(
        frame_metrics=compute_frame_metrics(predictions, ground_truth, cfg),
        trajectory_metrics=compute_trajectory_metrics(predictions, ground_truth, cfg),
        event_metrics=compute_event_metrics(predictions, ground_truth, cfg),
        config=cfg,
    )
