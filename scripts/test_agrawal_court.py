#!/usr/bin/env python3
"""Agrawal et al. (2024) inspired court detection pipeline - test runner.

Thin wrapper that imports the pipeline from src.features.court_detect.agrawal
and handles frame loading, iteration, summary printing, and report saving.

Usage:
    python scripts/test_agrawal_court.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.court_detect.agrawal import (
    HomographyKalmanFilter,
    run_agrawal_pipeline,
)
from src.features.court_detect.shadow_removal import ShadowRemover

# ---------------------------------------------------------------------------
# Script-level configuration
# ---------------------------------------------------------------------------
VIDEO_PATH = PROJECT_ROOT / "tests" / "fixtures" / "real_match_10min.mp4"
OUTPUT_DIR = PROJECT_ROOT / "output"
FRAME_INDICES = [16200]
IMAGE_DIR = PROJECT_ROOT / "tests" / "court_detection_frames"  # Set to None to use VIDEO_PATH
ENABLE_SHADOW_REMOVAL = True


def main():
    """Entry point: process frames and generate report."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frames: dict[int | str, np.ndarray] = {}

    if IMAGE_DIR is not None and IMAGE_DIR.is_dir():
        # Load from image directory
        print(f"Loading images from: {IMAGE_DIR}")
        image_files = sorted(
            p for p in IMAGE_DIR.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        )
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                frames[img_path.stem] = img
                print(f"  Loaded {img_path.name}: {img.shape[1]}\u00d7{img.shape[0]}")
            else:
                print(f"  WARNING: Could not read {img_path.name}")
        video_stem = IMAGE_DIR.name
    else:
        # Load from video
        print(f"Opening video: {VIDEO_PATH}")
        cap = cv2.VideoCapture(str(VIDEO_PATH))
        if not cap.isOpened():
            print(f"ERROR: Cannot open video {VIDEO_PATH}")
            sys.exit(1)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video: {total_frames} frames, {fps:.1f} fps")

        for idx in FRAME_INDICES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames[idx] = frame
                print(f"  Extracted frame {idx}: {frame.shape[1]}\u00d7{frame.shape[0]}")
            else:
                print(f"  WARNING: Could not read frame {idx}")
        cap.release()
        video_stem = VIDEO_PATH.stem

    if not frames:
        print("ERROR: No frames extracted")
        sys.exit(1)

    # Initialize YOLO net detector
    net_detector = None
    net_weights = PROJECT_ROOT / "weights" / "net_detector.pt"
    if net_weights.exists():
        from ultralytics import YOLO
        net_detector = YOLO(str(net_weights))
        print(f"  YOLO net detector loaded: {net_weights}")
    else:
        print(f"  YOLO net detector not found at {net_weights}, using heuristic fallback")

    # Initialize Kalman smoother for temporal consistency
    # Disabled for IMAGE_DIR mode (independent frames from different videos)
    kalman = None if IMAGE_DIR is not None else HomographyKalmanFilter()

    # Initialize shadow remover (optional)
    shadow_remover = ShadowRemover() if ENABLE_SHADOW_REMOVAL else None

    # Derive video stem for unique output filenames
    # (video_stem already set above)

    # Process each frame
    all_results = []
    for frame_key in frames:
        print(f"\n{'='*70}")
        print(f"Processing frame {frame_key}")
        print(f"{'='*70}")

        frame_idx = frame_key if isinstance(frame_key, int) else 0

        result = run_agrawal_pipeline(
            frames[frame_key], frame_idx, kalman, shadow_remover,
            net_detector=net_detector,
            output_dir=OUTPUT_DIR,
            video_stem=video_stem)

        # Save diagnostic composite
        out_path = OUTPUT_DIR / f"agrawal_{video_stem}_{frame_key}.jpg"
        cv2.imwrite(str(out_path), result["composite"])
        print(f"  Saved: {out_path}")

        all_results.append(result)

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Frame':>8} {'Near KPs':>9} {'Total KPs':>10} "
          f"{'H valid':>8} {'Raw err':>9} {'Smth err':>9} {'Far err*':>9} "
          f"{'Spread':>10} {'Time':>6}")
    print("-" * 95)
    for r in all_results:
        m = r["metrics"]
        print(f"{r['frame_idx']:>8} "
              f"{m['n_near_kps']:>9} "
              f"{m['n_total_kps']:>10} "
              f"{'YES' if m['homography_valid'] else 'NO':>8} "
              f"{r['raw_near_err']:>9.1f} "
              f"{r['smoothed_near_err']:>9.1f} "
              f"{m['far_reprojection_error_px']:>9.1f} "
              f"{m['keypoint_spread']:>10.0f} "
              f"{r['total_time']:>6.2f}")
    print("  * Far err is self-referential (projected from near-half H)")
    print("  * Raw err = before Kalman, Smth err = after Kalman smoothing")

    # Save JSON report (without numpy arrays)
    report = []
    for r in all_results:
        entry = {k: v for k, v in r.items() if k != "composite"}
        # Convert metrics inf to string for JSON
        for mk in entry.get("metrics", {}):
            val = entry["metrics"][mk]
            if isinstance(val, float) and (np.isinf(val) or np.isnan(val)):
                entry["metrics"][mk] = str(val)
        report.append(entry)

    report_path = OUTPUT_DIR / f"agrawal_{video_stem}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
