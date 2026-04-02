"""Reporting utilities for the scoring harness.

Formats ScoringResult as JSON and human-readable text.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from src.features.scoring.metrics import ScoringResult


def _scoring_result_to_dict(result: ScoringResult) -> dict[str, Any]:
    """Convert ScoringResult to a JSON-serialisable dictionary."""
    fm = result.frame_metrics
    tm = result.trajectory_metrics
    em = result.event_metrics

    return {
        "config": {
            "distance_threshold": result.config.distance_threshold,
            "max_velocity": result.config.max_velocity,
            "max_acceleration": result.config.max_acceleration,
            "rally_gap_tolerance": result.config.rally_gap_tolerance,
            "rally_iou_threshold": result.config.rally_iou_threshold,
            "fps": result.config.fps,
        },
        "layer1_frame": {
            "true_positives": fm.true_positives,
            "false_positives": fm.false_positives,
            "false_negatives": fm.false_negatives,
            "precision": round(fm.precision, 4),
            "recall": round(fm.recall, 4),
            "f1": round(fm.f1, 4),
            "mean_error_px": round(fm.mean_error, 2),
            "p50_error_px": round(fm.p50_error, 2),
            "p90_error_px": round(fm.p90_error, 2),
            "occluded_tp": fm.occluded_tp,
            "occluded_fn": fm.occluded_fn,
        },
        "layer2_trajectory": {
            "num_breaks": tm.num_breaks,
            "max_break_duration_frames": tm.max_break_duration,
            "breaks_per_minute": round(tm.breaks_per_minute, 2),
            "total_consecutive_pairs": tm.total_consecutive_pairs,
            "velocity_outliers": tm.velocity_outliers,
            "acceleration_outliers": tm.acceleration_outliers,
            "physical_consistency": round(tm.physical_consistency, 4),
        },
        "layer3_event": {
            "gt_rally_count": em.gt_rally_count,
            "pred_rally_count": em.pred_rally_count,
            "mean_iou": round(em.mean_iou, 4),
            "rally_recall": round(em.rally_recall, 4),
            "rally_false_positive_rate": round(em.rally_false_positive_rate, 4),
            "rally_matches": [
                {
                    "gt": list(m.gt_rally),
                    "pred": list(m.pred_rally),
                    "iou": round(m.iou, 4),
                }
                for m in em.rally_matches
            ],
        },
    }


def format_json_report(result: ScoringResult, indent: int = 2) -> str:
    """Format ScoringResult as a JSON string.

    Args:
        result: The scoring result.
        indent: JSON indentation level.

    Returns:
        JSON string.
    """
    return json.dumps(_scoring_result_to_dict(result), indent=indent)


def format_text_summary(result: ScoringResult) -> str:
    """Format ScoringResult as a human-readable text summary.

    Args:
        result: The scoring result.

    Returns:
        Multi-line summary string.
    """
    fm = result.frame_metrics
    tm = result.trajectory_metrics
    em = result.event_metrics

    lines = [
        "=" * 60,
        "  TENNIS BALL TRACKING — SCORING REPORT",
        "=" * 60,
        "",
        f"  Config: τ={result.config.distance_threshold}px, "
        f"fps={result.config.fps}, "
        f"max_vel={result.config.max_velocity} px/f",
        "",
        "  LAYER 1: Frame-level Localization",
        "  " + "-" * 40,
        f"    Precision:     {fm.precision:.4f}",
        f"    Recall:        {fm.recall:.4f}",
        f"    F1:            {fm.f1:.4f}",
        f"    TP / FP / FN:  {fm.true_positives} / {fm.false_positives} / {fm.false_negatives}",
        f"    Mean error:    {fm.mean_error:.2f} px",
        f"    P50 error:     {fm.p50_error:.2f} px",
        f"    P90 error:     {fm.p90_error:.2f} px",
        f"    Occluded TP:   {fm.occluded_tp}",
        f"    Occluded FN:   {fm.occluded_fn}",
        "",
        "  LAYER 2: Trajectory Continuity",
        "  " + "-" * 40,
        f"    Breaks:        {tm.num_breaks}",
        f"    Breaks/min:    {tm.breaks_per_minute:.2f}",
        f"    Max break:     {tm.max_break_duration} frames",
        f"    Phys consist.: {tm.physical_consistency:.4f}",
        f"    Vel outliers:  {tm.velocity_outliers}",
        f"    Accel outliers:{tm.acceleration_outliers}",
        "",
        "  LAYER 3: Event-level (Rally Segmentation)",
        "  " + "-" * 40,
        f"    GT rallies:    {em.gt_rally_count}",
        f"    Pred rallies:  {em.pred_rally_count}",
        f"    Mean IoU:      {em.mean_iou:.4f}",
        f"    Rally recall:  {em.rally_recall:.4f}",
        f"    Rally FP rate: {em.rally_false_positive_rate:.4f}",
        "",
        "=" * 60,
    ]

    return "\n".join(lines)
