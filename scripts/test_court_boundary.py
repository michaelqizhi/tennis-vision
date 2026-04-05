#!/usr/bin/env python
"""Test court boundary filtering on the real match video.

Runs the full pipeline:
1. Smart frame selection (garbage filter + court detection)
2. Court boundary filtering on existing V2 predictions
3. Re-score filtered predictions vs ground truth
4. Compare before/after metrics

Usage:
    python scripts/test_court_boundary.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import get_config
from src.features.court_detect.detector import CourtDetector
from src.features.court_detect.frame_selector import (
    select_best_court_frames,
    find_valid_frame_indices,
    compute_court_color_score,
    compute_sharpness,
)
from src.features.court_detect.boundary_filter import CourtBoundaryFilter
from src.features.court_detect.homography import transform_point_to_court
from src.features.scoring.ground_truth import parse_cvat_xml
from src.features.scoring.metrics import Prediction, ScoringConfig, run_scoring
from src.features.scoring.report import format_text_summary
from src.features.ball_tracking.detector import BallDetection


# --- Config ---
VIDEO_PATH = str(project_root / "tests" / "fixtures" / "real_match_10min.mp4")
PREDICTIONS_PATH = str(project_root / "output" / "v2_baseline_predictions.json")
GT_PATH = str(project_root / "tests" / "fixtures" / "annotations" / "real_match_10min.xml")
MAX_FRAME = 6000


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quieter sub-loggers
    logging.getLogger("src.features.court_detect.boundary_filter").setLevel(logging.WARNING)


def load_predictions(path: str) -> list[Prediction]:
    with open(path, "r") as f:
        data = json.load(f)
    return [
        Prediction(
            frame_number=int(e["frame_number"]),
            x=float(e["x"]) if e.get("x") is not None else None,
            y=float(e["y"]) if e.get("y") is not None else None,
            confidence=float(e.get("confidence", 1.0)),
            interpolated=bool(e.get("interpolated", False)),
        )
        for e in data
    ]


def predictions_to_detections(preds: list[Prediction]) -> list[BallDetection]:
    return [
        BallDetection(
            frame_number=p.frame_number,
            x=p.x,
            y=p.y,
            confidence=p.confidence,
            interpolated=p.interpolated,
        )
        for p in preds
    ]


def detections_to_predictions(dets: list[BallDetection]) -> list[Prediction]:
    return [
        Prediction(
            frame_number=d.frame_number,
            x=d.x,
            y=d.y,
            confidence=d.confidence,
            interpolated=d.interpolated,
        )
        for d in dets
    ]


def main():
    setup_logging()
    logger = logging.getLogger("court_boundary_test")
    config = get_config()

    # ===================================================================
    # PHASE 1: Smart Frame Selection
    # ===================================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: Smart Frame Selection")
    logger.info("=" * 60)

    t0 = time.time()

    # Phase A: Quick scan
    logger.info("Phase A: Scanning for valid frames...")
    valid_indices = find_valid_frame_indices(
        VIDEO_PATH,
        sample_interval_sec=config.court_detection.frame_sample_interval,
        min_court_color=config.court_detection.min_court_color_score,
        min_sharpness=config.court_detection.min_sharpness,
    )
    logger.info(f"Phase A complete: {len(valid_indices)} valid frames found")

    # Phase B: Run court detector on candidates
    logger.info("Phase B: Running court detector on candidates...")
    detector = CourtDetector(config)
    best_results = select_best_court_frames(
        VIDEO_PATH,
        detector,
        sample_interval_sec=config.court_detection.frame_sample_interval,
        min_court_color=config.court_detection.min_court_color_score,
        min_sharpness=config.court_detection.min_sharpness,
    )

    t_select = time.time() - t0

    if not best_results:
        logger.error("FAILED: No court detected in video!")
        sys.exit(1)

    best = best_results[0]
    logger.info(f"Best frame: #{best.frame_number}")
    logger.info(f"  Keypoints: {best.num_detected}/14")
    logger.info(f"  Confidence: {best.confidence:.1%}")
    logger.info(f"  Has homography: {best.homography_img_to_court is not None}")
    logger.info(f"Frame selection took: {t_select:.1f}s")

    if best.homography_img_to_court is None:
        logger.error("FAILED: No valid homography computed!")
        sys.exit(1)

    H = best.homography_img_to_court

    # Show what the homography maps key court points to
    logger.info("")
    logger.info("Homography sanity check — court corners in meters:")
    test_points = [
        ("Net center", 960, 400),
        ("Near baseline center", 960, 900),
        ("Top-left corner", 200, 200),
        ("Bottom-right corner", 1700, 950),
    ]
    for label, px, py in test_points:
        try:
            xm, ym = transform_point_to_court(px, py, H)
            logger.info(f"  {label} ({px}, {py}) → ({xm:.1f}m, {ym:.1f}m)")
        except Exception as e:
            logger.warning(f"  {label} ({px}, {py}) → ERROR: {e}")

    # ===================================================================
    # PHASE 2: Apply Boundary Filter to V2 Predictions
    # ===================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 2: Court Boundary Filtering")
    logger.info("=" * 60)

    # Load V2 predictions
    if not Path(PREDICTIONS_PATH).exists():
        logger.error(f"V2 predictions not found at {PREDICTIONS_PATH}")
        logger.info("Run V2 baseline first to generate predictions.")
        sys.exit(1)

    preds_original = load_predictions(PREDICTIONS_PATH)
    logger.info(f"Loaded {len(preds_original)} V2 predictions")

    # Convert to BallDetections for filtering
    detections = predictions_to_detections(preds_original)
    detected_before = sum(1 for d in detections if d.detected)
    logger.info(f"Detections before filter: {detected_before}")

    # Apply boundary filter
    boundary_filter = CourtBoundaryFilter(
        homography=H,
        x_margin=config.court_detection.boundary_x_margin,
        y_margin=config.court_detection.boundary_y_margin,
    )
    logger.info(
        f"Court bounds: |x| ≤ {boundary_filter.x_bound:.1f}m, "
        f"|y| ≤ {boundary_filter.y_bound:.1f}m"
    )

    t0 = time.time()
    filtered = boundary_filter.filter_detections(detections)
    t_filter = time.time() - t0

    detected_after = sum(1 for d in filtered if d.detected)
    removed = detected_before - detected_after
    logger.info(f"Detections after filter: {detected_after}")
    logger.info(f"Removed: {removed} ({removed * 100 / detected_before:.1f}% of detections)")
    logger.info(f"Filtering took: {t_filter * 1000:.1f}ms")

    # ===================================================================
    # PHASE 3: Re-score and Compare
    # ===================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 3: Scoring Comparison")
    logger.info("=" * 60)

    gt = parse_cvat_xml(Path(GT_PATH), max_frame=MAX_FRAME)
    scoring_cfg = ScoringConfig(fps=30.0)

    # Score original
    result_before = run_scoring(preds_original, gt, scoring_cfg)

    # Score filtered
    preds_filtered = detections_to_predictions(filtered)
    result_after = run_scoring(preds_filtered, gt, scoring_cfg)

    # Print comparison
    fb = result_before.frame_metrics
    fa = result_after.frame_metrics
    tb = result_before.trajectory_metrics
    ta = result_after.trajectory_metrics
    rb = result_before.rally_metrics
    ra = result_after.rally_metrics

    logger.info("")
    logger.info(f"{'Metric':<25} {'Before':>10} {'After':>10} {'Delta':>10}")
    logger.info("-" * 55)
    logger.info(f"{'Precision':<25} {fb.precision:>10.3f} {fa.precision:>10.3f} {fa.precision - fb.precision:>+10.3f}")
    logger.info(f"{'Recall':<25} {fb.recall:>10.3f} {fa.recall:>10.3f} {fa.recall - fb.recall:>+10.3f}")
    logger.info(f"{'F1':<25} {fb.f1:>10.3f} {fa.f1:>10.3f} {fa.f1 - fb.f1:>+10.3f}")
    logger.info(f"{'Mean error (px)':<25} {fb.mean_error:>10.2f} {fa.mean_error:>10.2f} {fa.mean_error - fb.mean_error:>+10.2f}")
    logger.info(f"{'Breaks/min':<25} {tb.breaks_per_minute:>10.1f} {ta.breaks_per_minute:>10.1f} {ta.breaks_per_minute - tb.breaks_per_minute:>+10.1f}")
    logger.info(f"{'Rally recall':<25} {rb.recall:>10.3f} {ra.recall:>10.3f} {ra.recall - rb.recall:>+10.3f}")
    logger.info(f"{'Rally precision':<25} {rb.precision:>10.3f} {ra.precision:>10.3f} {ra.precision - rb.precision:>+10.3f}")

    # Save filtered predictions
    output_path = project_root / "output" / "v2_court_filtered_predictions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            [
                {
                    "frame_number": p.frame_number,
                    "x": p.x,
                    "y": p.y,
                    "confidence": p.confidence,
                    "interpolated": p.interpolated,
                }
                for p in preds_filtered
            ],
            f,
        )
    logger.info(f"\nFiltered predictions saved to {output_path}")

    # Save report
    report_path = project_root / "output" / "v2_court_filtered_report.json"
    from src.features.scoring.report import format_json_report
    with open(report_path, "w") as f:
        f.write(format_json_report(result_after))
    logger.info(f"Scoring report saved to {report_path}")


if __name__ == "__main__":
    main()
