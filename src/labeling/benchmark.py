"""Benchmark report generator — compare model detection performance.

Produces a Markdown table comparing all detectors on the same video,
including detection rate, consensus agreement, confidence, inference
speed, and estimated VRAM usage.

Usage::

    from src.labeling.benchmark import generate_benchmark_report
    report = generate_benchmark_report(detections, consensus, timing, output_path)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.labeling.consensus import DetectionLabel, FrameConsensus

logger = logging.getLogger(__name__)


def generate_benchmark_report(
    detections: dict[str, dict[str, Optional[dict]]],
    consensus: list[FrameConsensus],
    timing: dict[str, float],
    output_path: str,
    *,
    video_info: dict[str, str | int | float] | None = None,
    player_filter_stats: dict | None = None,
) -> str:
    """Generate a Markdown benchmark report comparing all models.

    Args:
        detections: Raw detections from inference runner
            (frame_idx → model_name → detection_or_null).
        consensus: Consensus results from the consensus engine.
        timing: Per-model inference timing in seconds.
        output_path: Path for the output Markdown file.
        video_info: Optional dict with 'path', 'frames', 'fps',
            'width', 'height' keys for the report header.
        player_filter_stats: Optional dict with player filter statistics
            ('enabled', 'rejected', 'max_distance', 'frames_with_players').

    Returns:
        The Markdown report content as a string.
    """
    total_frames = len(detections)
    if total_frames == 0:
        return "No frames to benchmark."

    # Discover model names from the first frame
    first_frame = detections[next(iter(detections))]
    model_names = sorted(first_frame.keys())

    if not model_names:
        return "No models found in detections."

    # ── Per-model metrics ─────────────────────────────────────────
    model_stats: dict[str, dict] = {}

    for model in model_names:
        det_count = 0
        confidences: list[float] = []
        agrees_with_consensus = 0

        for frame_idx_str, frame_dets in detections.items():
            det = frame_dets.get(model)
            if det is not None:
                det_count += 1
                confidences.append(det.get("conf", 0.0))

        # Check agreement with consensus
        consensus_by_frame = {fc.frame_idx: fc for fc in consensus}
        for frame_idx_str, frame_dets in detections.items():
            frame_idx = int(frame_idx_str)
            det = frame_dets.get(model)
            fc = consensus_by_frame.get(frame_idx)
            if fc is None:
                continue

            if fc.label == DetectionLabel.CONSENSUS:
                if det is not None and fc.x is not None and fc.y is not None:
                    dx = det["x"] - fc.x
                    dy = det["y"] - fc.y
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist < 20.0:
                        agrees_with_consensus += 1

        elapsed = timing.get(model, 0.0)
        fps = total_frames / elapsed if elapsed > 0 else 0.0
        det_rate = 100.0 * det_count / total_frames
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        consensus_count = sum(1 for fc in consensus if fc.label == DetectionLabel.CONSENSUS)
        agreement_rate = (
            100.0 * agrees_with_consensus / consensus_count
            if consensus_count > 0
            else 0.0
        )

        model_stats[model] = {
            "detections": det_count,
            "detection_rate": det_rate,
            "avg_confidence": avg_conf,
            "consensus_agreement": agreement_rate,
            "agrees_count": agrees_with_consensus,
            "consensus_total": consensus_count,
            "fps": fps,
            "elapsed": elapsed,
        }

    # ── Consensus summary ─────────────────────────────────────────
    consensus_count = sum(1 for fc in consensus if fc.label == DetectionLabel.CONSENSUS)
    uncertain_count = sum(1 for fc in consensus if fc.label == DetectionLabel.UNCERTAIN)
    no_det_count = sum(1 for fc in consensus if fc.label == DetectionLabel.NO_DETECTION)

    # ── Build Markdown ────────────────────────────────────────────
    lines: list[str] = []
    lines.append("# Model Benchmark Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    if video_info:
        lines.append("## Video Info")
        lines.append("")
        lines.append(f"| Property | Value |")
        lines.append(f"|----------|-------|")
        for k, v in video_info.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    lines.append("## Per-Model Comparison")
    lines.append("")
    lines.append(
        "| Model | Detection Rate | Avg Confidence | "
        "Consensus Agreement | Inference Speed | Time |"
    )
    lines.append(
        "|-------|---------------|----------------|"
        "--------------------|-----------------|------|"
    )

    for model in model_names:
        s = model_stats[model]
        lines.append(
            f"| {model} | {s['detection_rate']:.1f}% ({s['detections']}/{total_frames}) | "
            f"{s['avg_confidence']:.3f} | "
            f"{s['consensus_agreement']:.1f}% ({s['agrees_count']}/{s['consensus_total']}) | "
            f"{s['fps']:.1f} fps | "
            f"{s['elapsed']:.1f}s |"
        )

    lines.append("")
    lines.append("## Consensus Summary")
    lines.append("")
    lines.append(f"| Metric | Count | Percentage |")
    lines.append(f"|--------|-------|------------|")
    lines.append(
        f"| Consensus (≥2 agree) | {consensus_count} | "
        f"{100.0 * consensus_count / total_frames:.1f}% |"
    )
    lines.append(
        f"| Uncertain (1 only) | {uncertain_count} | "
        f"{100.0 * uncertain_count / total_frames:.1f}% |"
    )
    lines.append(
        f"| No detection | {no_det_count} | "
        f"{100.0 * no_det_count / total_frames:.1f}% |"
    )
    lines.append(f"| **Total frames** | {total_frames} | 100% |")

    lines.append("")
    lines.append("## VRAM Estimates")
    lines.append("")
    lines.append("| Model | Estimated VRAM |")
    lines.append("|-------|---------------|")

    vram_estimates = {
        "tracknet_v2": "~1.5 GB",
        "tracknet_v4": "~2.0 GB",
        "florence_2": "~3.0 GB",
        "yolo_world": "~1.5 GB",
    }
    for model in model_names:
        est = vram_estimates.get(model, "Unknown")
        lines.append(f"| {model} | {est} |")

    lines.append("")
    lines.append("## Recommendations")
    lines.append("")

    # Player filter stats
    if player_filter_stats and player_filter_stats.get("enabled"):
        lines.append("## Player Proximity Filter")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Enabled | Yes |")
        lines.append(f"| Max distance | {player_filter_stats.get('max_distance', 200)}px |")
        lines.append(f"| Frames with players | {player_filter_stats.get('frames_with_players', 0)} |")
        lines.append(f"| Detections rejected | {player_filter_stats.get('rejected', 0)} |")
        if total_frames > 0:
            fp_reduction = 100.0 * player_filter_stats.get('rejected', 0) / total_frames
            lines.append(f"| FP reduction | {fp_reduction:.1f}% of frames |")
        lines.append("")

    # Find best model by detection rate
    best_det = max(model_names, key=lambda m: model_stats[m]["detection_rate"])
    best_agree = max(model_names, key=lambda m: model_stats[m]["consensus_agreement"])
    fastest = max(model_names, key=lambda m: model_stats[m]["fps"])

    lines.append(f"- **Highest detection rate:** {best_det} ({model_stats[best_det]['detection_rate']:.1f}%)")
    lines.append(f"- **Best consensus agreement:** {best_agree} ({model_stats[best_agree]['consensus_agreement']:.1f}%)")
    lines.append(f"- **Fastest inference:** {fastest} ({model_stats[fastest]['fps']:.1f} fps)")
    lines.append("")

    report = "\n".join(lines)

    # Write to file
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    logger.info("Benchmark report saved to %s", output_path)

    return report
