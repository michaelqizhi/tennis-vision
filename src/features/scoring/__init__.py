"""Three-layer scoring harness for tennis ball tracking evaluation.

Measures tracking quality across three layers:
  Layer 1 — Frame-level localization (precision, recall, F1, error)
  Layer 2 — Trajectory continuity (break rate, physical consistency)
  Layer 3 — Event-level (rally segmentation IoU, recall, FP rate)
"""

from src.features.scoring.ground_truth import (
    GroundTruthFrame,
    GroundTruthData,
    parse_cvat_xml,
)
from src.features.scoring.metrics import (
    ScoringConfig,
    FrameMetrics,
    TrajectoryMetrics,
    EventMetrics,
    ScoringResult,
    compute_frame_metrics,
    compute_trajectory_metrics,
    compute_event_metrics,
    run_scoring,
)
from src.features.scoring.report import format_json_report, format_text_summary

__all__ = [
    "GroundTruthFrame",
    "GroundTruthData",
    "parse_cvat_xml",
    "ScoringConfig",
    "FrameMetrics",
    "TrajectoryMetrics",
    "EventMetrics",
    "ScoringResult",
    "compute_frame_metrics",
    "compute_trajectory_metrics",
    "compute_event_metrics",
    "run_scoring",
    "format_json_report",
    "format_text_summary",
]
