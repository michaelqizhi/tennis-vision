#!/usr/bin/env python
"""CLI entry point for the three-layer scoring harness.

Usage:
    python scripts/run_scoring.py \
        --predictions predictions.json \
        --ground-truth annotations.xml \
        --fps 30 \
        --output report.json

Prediction format (JSON array):
    [{"frame_number": 0, "x": 100.5, "y": 200.3, "confidence": 0.9, "interpolated": false}, ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path for `src.*` imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.features.scoring.ground_truth import parse_cvat_xml
from src.features.scoring.metrics import Prediction, ScoringConfig, run_scoring
from src.features.scoring.report import format_json_report, format_text_summary


def _load_predictions_from_json(path: Path) -> list[Prediction]:
    """Load predictions from a JSON file (array of detection dicts)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")

    predictions: list[Prediction] = []
    for entry in data:
        predictions.append(Prediction(
            frame_number=int(entry["frame_number"]),
            x=float(entry["x"]) if entry.get("x") is not None else None,
            y=float(entry["y"]) if entry.get("y") is not None else None,
            confidence=float(entry.get("confidence", 1.0)),
            interpolated=bool(entry.get("interpolated", False)),
        ))
    return predictions


def _load_predictions_from_dir(dir_path: Path) -> list[Prediction]:
    """Load predictions from a directory of per-frame JSON files.

    Expected file naming: <frame_number>.json or frame_<frame_number>.json
    Each file: {"x": ..., "y": ..., "confidence": ..., "interpolated": ...}
    """
    predictions: list[Prediction] = []
    json_files = sorted(dir_path.glob("*.json"))

    for jf in json_files:
        stem = jf.stem
        # Try to extract frame number from filename
        if stem.startswith("frame_"):
            frame_num = int(stem.replace("frame_", ""))
        else:
            try:
                frame_num = int(stem)
            except ValueError:
                continue

        with open(jf, "r", encoding="utf-8") as f:
            entry = json.load(f)

        predictions.append(Prediction(
            frame_number=frame_num,
            x=float(entry["x"]) if entry.get("x") is not None else None,
            y=float(entry["y"]) if entry.get("y") is not None else None,
            confidence=float(entry.get("confidence", 1.0)),
            interpolated=bool(entry.get("interpolated", False)),
        ))

    return predictions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Three-layer scoring harness for tennis ball tracking evaluation.",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions JSON file or directory of per-frame JSONs.",
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Path to CVAT XML ground truth annotations.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video frame rate (default: 30).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write JSON report (default: stdout).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=15.0,
        help="Distance threshold τ in pixels (default: 15).",
    )
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=50.0,
        help="Max velocity (px/frame) for physical consistency (default: 50).",
    )
    parser.add_argument(
        "--max-acceleration",
        type=float,
        default=30.0,
        help="Max acceleration (px/frame²) for physical consistency (default: 30).",
    )
    parser.add_argument(
        "--rally-gap",
        type=int,
        default=5,
        help="Gap tolerance (frames) for rally segmentation (default: 5).",
    )
    parser.add_argument(
        "--rally-iou",
        type=float,
        default=0.5,
        help="Min IoU for rally match (default: 0.5).",
    )
    parser.add_argument(
        "--max-frame",
        type=int,
        default=None,
        help="Discard ground truth annotations beyond this frame number.",
    )

    args = parser.parse_args(argv)

    # Load predictions
    pred_path = Path(args.predictions)
    if pred_path.is_dir():
        predictions = _load_predictions_from_dir(pred_path)
    elif pred_path.is_file() and pred_path.suffix == ".json":
        predictions = _load_predictions_from_json(pred_path)
    else:
        parser.error(f"Predictions path must be a .json file or directory: {pred_path}")

    # Load ground truth
    gt_path = Path(args.ground_truth)
    if not gt_path.is_file():
        parser.error(f"Ground truth file not found: {gt_path}")

    ground_truth = parse_cvat_xml(gt_path, max_frame=args.max_frame)

    # Configure
    config = ScoringConfig(
        distance_threshold=args.threshold,
        max_velocity=args.max_velocity,
        max_acceleration=args.max_acceleration,
        rally_gap_tolerance=args.rally_gap,
        rally_iou_threshold=args.rally_iou,
        fps=args.fps,
    )

    # Run scoring
    result = run_scoring(predictions, ground_truth, config)

    # Output
    json_report = format_json_report(result)
    text_summary = format_text_summary(result)

    print(text_summary)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_report)
        print(f"\nJSON report written to: {output_path}")
    else:
        print("\n--- JSON Report ---")
        print(json_report)


if __name__ == "__main__":
    main()
