#!/usr/bin/env python
"""Threshold sweep for TrackNet V4 ball detection.

Runs V4 inference once (saving raw heatmap peaks for every frame), then
sweeps detection thresholds offline and reports F1 / precision / recall
at each threshold.

Usage:
    python scripts/threshold_sweep.py --input <video.mp4> --max-frames 6000
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.config import get_config
from src.features.ball_tracking.tracknetv4_tracker import postprocess_v4
from src.features.scoring.ground_truth import parse_cvat_xml
from src.features.scoring.metrics import (
    FrameMetrics,
    Prediction,
    ScoringConfig,
    compute_frame_metrics,
)
from src.models.tracknetv4 import TrackNetV4Model, load_tracknet_v4
from src.video.reader import VideoReader

GT_PATH = "tests/fixtures/annotations/real_match_10min.xml"

THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]


@dataclass
class RawDetection:
    """Stores raw heatmap data for offline threshold sweeping."""
    frame_number: int
    heatmap: np.ndarray  # (H, W) sigmoid heatmap
    scale_x: float
    scale_y: float


# ---------------------------------------------------------------------------
# Inference phase
# ---------------------------------------------------------------------------

def run_inference(
    video_path: str,
    max_frames: int,
    device: str,
) -> list[RawDetection]:
    """Run V4 inference and collect raw detections with peak values."""
    cfg = get_config()
    inp_w = cfg.model.tracknet_v4_input_width
    inp_h = cfg.model.tracknet_v4_input_height

    model = load_tracknet_v4(
        weights_path=cfg.model.tracknet_v4_weights,
        device=device,
    )
    model.eval()

    raw: list[RawDetection] = []
    frame_buffer: list[np.ndarray] = []

    with VideoReader(video_path, max_frames=max_frames) as reader:
        orig_h, orig_w = None, None
        scale_x, scale_y = 1.0, 1.0

        for num, frame in enumerate(tqdm(reader.iter_frames(), total=max_frames, desc="Inference")):
            if orig_h is None:
                orig_h, orig_w = frame.shape[:2]
                scale_x = orig_w / inp_w
                scale_y = orig_h / inp_h

            resized = cv2.resize(frame, (inp_w, inp_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frame_buffer.append(rgb)

            # Need at least 3 frames to form a window
            if len(frame_buffer) < 3:
                continue

            # Build input: [oldest, middle, newest] = last 3 frames
            imgs = np.concatenate(frame_buffer[-3:], axis=2)  # (H, W, 9)
            imgs = imgs.astype(np.float32) / 255.0
            imgs = np.rollaxis(imgs, 2, 0)  # (9, H, W)
            inp = np.expand_dims(imgs, axis=0)  # (1, 9, H, W)

            with torch.no_grad():
                out = model(torch.from_numpy(inp).float().to(device))

            # Channel 1 = center frame prediction
            heatmap = out[0, 1].cpu().numpy()
            peak_value = float(heatmap.max())

            # Store raw heatmap for offline threshold sweep.
            center_frame = num - 1
            raw.append(RawDetection(
                frame_number=center_frame,
                heatmap=heatmap.copy(),
                scale_x=scale_x,
                scale_y=scale_y,
            ))

            # Keep only last 3 frames in buffer
            if len(frame_buffer) > 3:
                frame_buffer = frame_buffer[-3:]

    return raw


# ---------------------------------------------------------------------------
# Sweep phase
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    threshold: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    detection_rate: float
    mean_error: float


def sweep_thresholds(
    raw_detections: list[RawDetection],
    max_frames: int,
) -> list[SweepResult]:
    """Sweep thresholds and compute metrics against ground truth."""
    gt = parse_cvat_xml(GT_PATH, max_frame=max_frames - 1)
    results: list[SweepResult] = []

    for thresh in THRESHOLDS:
        # Run postprocessing at this threshold for every stored heatmap
        pred_by_frame: dict[int, tuple[float, float, float]] = {}
        for det in raw_detections:
            x, y, conf = postprocess_v4(
                det.heatmap, det.scale_x, det.scale_y, threshold=thresh,
            )
            if x is not None:
                pred_by_frame[det.frame_number] = (x, y, conf)

        predictions: list[Prediction] = []
        for f in range(max_frames):
            if f in pred_by_frame:
                x, y, conf = pred_by_frame[f]
                predictions.append(Prediction(
                    frame_number=f, x=x, y=y, confidence=conf,
                ))
            else:
                predictions.append(Prediction(frame_number=f, x=None, y=None))

        scoring_cfg = ScoringConfig(distance_threshold=15.0)
        metrics: FrameMetrics = compute_frame_metrics(predictions, gt, scoring_cfg)

        tp = metrics.true_positives
        fp = metrics.false_positives
        fn = metrics.false_negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        det_rate = len(pred_by_frame) / max_frames
        mean_err = float(np.mean(metrics.errors)) if metrics.errors else 0.0

        results.append(SweepResult(
            threshold=thresh,
            tp=tp, fp=fp, fn=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            detection_rate=det_rate,
            mean_error=mean_err,
        ))

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(results: list[SweepResult]) -> None:
    """Print a formatted results table."""
    best = max(results, key=lambda r: r.f1)

    header = (
        f"{'Thresh':>7s}  {'TP':>5s}  {'FP':>5s}  {'FN':>5s}  "
        f"{'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}  "
        f"{'Det%':>6s}  {'MeanErr':>7s}"
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print("TrackNet V4 — Threshold Sweep Results")
    print(sep)
    print(header)
    print(sep)

    for r in results:
        marker = " ***" if r.threshold == best.threshold else ""
        print(
            f"{r.threshold:7.2f}  {r.tp:5d}  {r.fp:5d}  {r.fn:5d}  "
            f"{r.precision:6.3f}  {r.recall:6.3f}  {r.f1:6.3f}  "
            f"{r.detection_rate:6.1%}  {r.mean_error:7.2f}{marker}"
        )

    print(sep)
    print(
        f"\n★ Best F1 = {best.f1:.4f} at threshold = {best.threshold:.2f}"
        f"  (P={best.precision:.3f}  R={best.recall:.3f}  "
        f"MeanErr={best.mean_error:.2f}px)"
    )
    print(
        f"\nRecommendation: set v4_detection_threshold to {best.threshold} "
        f"in config.yaml\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep V4 detection thresholds and report F1/precision/recall."
    )
    parser.add_argument("--input", required=True, help="Path to input video (.mp4)")
    parser.add_argument(
        "--max-frames", type=int, default=6000,
        help="Number of frames to process (default: 6000)",
    )
    args = parser.parse_args()

    cfg = get_config()
    device = cfg.device
    print(f"Device: {device}")
    print(f"Video:  {args.input}")
    print(f"Frames: {args.max_frames}")

    t0 = time.time()
    raw = run_inference(args.input, args.max_frames, device)
    t_inf = time.time() - t0
    print(f"\nInference complete: {len(raw)} windows in {t_inf:.1f}s")

    results = sweep_thresholds(raw, args.max_frames)
    print_results(results)


if __name__ == "__main__":
    main()
