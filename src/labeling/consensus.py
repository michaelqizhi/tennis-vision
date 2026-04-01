"""Consensus engine — merges multi-model detections into unified annotations.

For each frame, compares detections from all models:
  - **≥2 models within ``threshold_px``** → ``consensus`` with averaged position
  - **1 model only** → ``uncertain``
  - **0 models** → ``no_detection``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DetectionLabel(str, Enum):
    """Classification of a frame's consensus status."""
    CONSENSUS = "consensus"
    UNCERTAIN = "uncertain"
    NO_DETECTION = "no_detection"


@dataclass
class Detection:
    """A single model's detection for one frame."""
    model_name: str
    x: float
    y: float
    confidence: float


@dataclass
class FrameConsensus:
    """Consensus result for a single frame."""
    frame_idx: int
    label: DetectionLabel
    x: Optional[float] = None
    y: Optional[float] = None
    confidence: Optional[float] = None
    agreeing_models: list[str] = field(default_factory=list)
    all_detections: list[Detection] = field(default_factory=list)


def _euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two 2D points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _find_largest_cluster(
    detections: list[Detection], threshold_px: float
) -> list[Detection]:
    """Find the largest group of detections where all pairs are within threshold.

    Uses a greedy seed approach followed by all-pairs verification: for each
    detection, gather candidates within ``threshold_px`` of the seed, then
    prune any candidate whose distance to another candidate exceeds the
    threshold.  This guarantees every pair in the returned cluster is within
    ``threshold_px`` of each other.

    Returns:
        List of detections in the winning cluster (may be empty if
        ``detections`` is empty).
    """
    if not detections:
        return []

    best_cluster: list[Detection] = []

    for seed in detections:
        candidates = [
            d for d in detections
            if _euclidean_distance(seed.x, seed.y, d.x, d.y) <= threshold_px
        ]
        # Verify all pairs within threshold (prune until valid)
        cluster = _verify_all_pairs(candidates, threshold_px)
        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    return best_cluster


def _verify_all_pairs(
    candidates: list[Detection], threshold_px: float
) -> list[Detection]:
    """Prune candidates until all pairs are within threshold of each other.

    Iteratively removes the candidate with the most violations until
    the remaining set forms a valid clique.
    """
    remaining = list(candidates)
    while len(remaining) > 1:
        # Find any pair that violates the threshold
        worst_idx = -1
        worst_violations = 0
        for i, d1 in enumerate(remaining):
            violations = sum(
                1 for j, d2 in enumerate(remaining)
                if i != j and _euclidean_distance(d1.x, d1.y, d2.x, d2.y) > threshold_px
            )
            if violations > worst_violations:
                worst_violations = violations
                worst_idx = i
        if worst_violations == 0:
            break  # all pairs valid
        remaining.pop(worst_idx)
    return remaining


def compute_consensus(
    raw_detections: dict[str, dict[str, Optional[dict]]],
    threshold_px: float = 15.0,
) -> list[FrameConsensus]:
    """Compute per-frame consensus from multi-model detection results.

    Args:
        raw_detections: Output of ``inference_runner.run_inference()``.
            Mapping of ``frame_idx_str → {model_name → {x, y, conf} | None}``.
        threshold_px: Maximum pixel distance for two detections to be
            considered "agreeing". Default 15px.

    Returns:
        List of ``FrameConsensus`` objects, one per frame, sorted by frame index.
    """
    if threshold_px < 0:
        raise ValueError(f"threshold_px must be non-negative, got {threshold_px}")

    results: list[FrameConsensus] = []
    frame_indices = sorted(int(k) for k in raw_detections.keys())

    for frame_idx in frame_indices:
        frame_key = str(frame_idx)
        model_results = raw_detections[frame_key]

        # Collect valid detections
        detections: list[Detection] = []
        for model_name, det_dict in model_results.items():
            if det_dict is not None:
                detections.append(Detection(
                    model_name=model_name,
                    x=det_dict["x"],
                    y=det_dict["y"],
                    confidence=det_dict["conf"],
                ))

        if len(detections) == 0:
            results.append(FrameConsensus(
                frame_idx=frame_idx,
                label=DetectionLabel.NO_DETECTION,
            ))
        elif len(detections) == 1:
            d = detections[0]
            results.append(FrameConsensus(
                frame_idx=frame_idx,
                label=DetectionLabel.UNCERTAIN,
                x=d.x,
                y=d.y,
                confidence=d.confidence,
                agreeing_models=[d.model_name],
                all_detections=detections,
            ))
        else:
            # ≥2 detections — check for cluster agreement
            cluster = _find_largest_cluster(detections, threshold_px)

            if len(cluster) >= 2:
                # Consensus: average position of agreeing models
                avg_x = sum(d.x for d in cluster) / len(cluster)
                avg_y = sum(d.y for d in cluster) / len(cluster)
                avg_conf = sum(d.confidence for d in cluster) / len(cluster)
                results.append(FrameConsensus(
                    frame_idx=frame_idx,
                    label=DetectionLabel.CONSENSUS,
                    x=round(avg_x, 2),
                    y=round(avg_y, 2),
                    confidence=round(avg_conf, 4),
                    agreeing_models=[d.model_name for d in cluster],
                    all_detections=detections,
                ))
            else:
                # Multiple detections but none agree — treat as uncertain
                # Use the detection with highest confidence
                best = max(detections, key=lambda d: d.confidence)
                results.append(FrameConsensus(
                    frame_idx=frame_idx,
                    label=DetectionLabel.UNCERTAIN,
                    x=best.x,
                    y=best.y,
                    confidence=best.confidence,
                    agreeing_models=[best.model_name],
                    all_detections=detections,
                ))

    return results


def compute_consensus_stats(consensus: list[FrameConsensus]) -> dict[str, object]:
    """Compute summary statistics from consensus results.

    Returns:
        Dict with keys: total_frames, consensus_count, uncertain_count,
        no_detection_count, consensus_pct, uncertain_pct, no_detection_pct.
    """
    total = len(consensus)
    if total == 0:
        return {
            "total_frames": 0,
            "consensus_count": 0,
            "uncertain_count": 0,
            "no_detection_count": 0,
            "consensus_pct": 0.0,
            "uncertain_pct": 0.0,
            "no_detection_pct": 0.0,
        }

    consensus_count = sum(1 for c in consensus if c.label == DetectionLabel.CONSENSUS)
    uncertain_count = sum(1 for c in consensus if c.label == DetectionLabel.UNCERTAIN)
    no_det_count = sum(1 for c in consensus if c.label == DetectionLabel.NO_DETECTION)

    return {
        "total_frames": total,
        "consensus_count": consensus_count,
        "uncertain_count": uncertain_count,
        "no_detection_count": no_det_count,
        "consensus_pct": round(100.0 * consensus_count / total, 1),
        "uncertain_pct": round(100.0 * uncertain_count / total, 1),
        "no_detection_pct": round(100.0 * no_det_count / total, 1),
    }


def serialize_consensus(consensus: list[FrameConsensus]) -> dict[str, dict]:
    """Serialize consensus results to a JSON-compatible dict.

    Returns:
        Mapping of ``frame_idx_str → {label, x, y, confidence, agreeing_models,
        all_detections}``.
    """
    result: dict[str, dict] = {}
    for fc in consensus:
        entry: dict = {
            "label": fc.label.value,
            "x": fc.x,
            "y": fc.y,
            "confidence": fc.confidence,
            "agreeing_models": fc.agreeing_models,
        }
        if fc.all_detections:
            entry["all_detections"] = [
                {"model": d.model_name, "x": d.x, "y": d.y, "conf": d.confidence}
                for d in fc.all_detections
            ]
        result[str(fc.frame_idx)] = entry
    return result


def apply_player_proximity_filter(
    consensus: list[FrameConsensus],
    player_detections: dict[int, list],
    max_distance: float = 200.0,
) -> tuple[list[FrameConsensus], int]:
    """Filter consensus results by player proximity.

    Rejects ball detections that are too far from any detected player.
    This reduces false positives on net posts, fences, and static features.

    Args:
        consensus: Original consensus results.
        player_detections: Mapping of frame_idx → list of PlayerBox objects.
        max_distance: Maximum pixel distance from nearest player bounding box.

    Returns:
        Tuple of (filtered_consensus, rejected_count).
    """
    from src.labeling.player_detector import filter_by_player_proximity

    filtered: list[FrameConsensus] = []
    rejected = 0

    for fc in consensus:
        if fc.label == DetectionLabel.NO_DETECTION or fc.x is None or fc.y is None:
            filtered.append(fc)
            continue

        players = player_detections.get(fc.frame_idx, [])
        if not players:
            # No player data for this frame — keep detection
            filtered.append(fc)
            continue

        if filter_by_player_proximity(fc.x, fc.y, players, max_distance):
            filtered.append(fc)
        else:
            # Replace with no_detection
            filtered.append(FrameConsensus(
                frame_idx=fc.frame_idx,
                label=DetectionLabel.NO_DETECTION,
            ))
            rejected += 1

    return filtered, rejected
