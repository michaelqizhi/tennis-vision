"""Evaluate the trained tennis net detector.

Runs the trained YOLOv8n net detector on:
  1. The held-out test split
  2. Sample frames from the two existing test videos

Prints metrics (mAP, precision, recall) and saves annotated images.

Usage:
    python scripts/net_dataset/eval_net_detector.py
"""

import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS = PROJECT_ROOT / "weights" / "net_detector.pt"
DATASET_YAML = PROJECT_ROOT / "data" / "net_detection" / "dataset" / "dataset.yaml"
TEST_VIDEOS = [
    PROJECT_ROOT / "tests" / "fixtures" / "courtside_match_2.mp4",
    PROJECT_ROOT / "tests" / "fixtures" / "real_match_10min.mp4",
]
OUTPUT_DIR = PROJECT_ROOT / "output" / "net_detection_eval"
NUM_FRAMES = 5


def extract_frames(video_path: Path, n: int) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append((int(idx), frame))
    cap.release()
    return frames


def draw_detections(frame: np.ndarray, boxes, scores) -> np.ndarray:
    vis = frame.copy()
    for i in range(len(scores)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        score = float(scores[i])
        color = (0, 255, 0) if score >= 0.5 else (0, 165, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text = f"net {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(vis, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return vis


def main():
    if not WEIGHTS.exists():
        print(f"ERROR: Trained weights not found: {WEIGHTS}")
        print("Run train_net_detector.py first.")
        return

    model = YOLO(str(WEIGHTS))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Part 1: Evaluate on test split ──
    print("=" * 60)
    print("Part 1: Test Split Evaluation")
    print("=" * 60)

    if DATASET_YAML.exists():
        metrics = model.val(data=str(DATASET_YAML), split="test", verbose=False)
        print(f"  mAP50:      {metrics.box.map50:.3f}")
        print(f"  mAP50-95:   {metrics.box.map:.3f}")
        print(f"  Precision:  {metrics.box.mp:.3f}")
        print(f"  Recall:     {metrics.box.mr:.3f}")
    else:
        print(f"  SKIP: Dataset YAML not found ({DATASET_YAML})")

    # ── Part 2: Inference on test videos ──
    print(f"\n{'='*60}")
    print("Part 2: Test Video Inference")
    print("=" * 60)

    for video_path in TEST_VIDEOS:
        vname = video_path.stem
        print(f"\n  ── {vname} ──")
        if not video_path.exists():
            print(f"    SKIP: not found")
            continue

        frames = extract_frames(video_path, NUM_FRAMES)
        for frame_idx, frame in frames:
            t0 = time.perf_counter()
            results = model.predict(frame, conf=0.25, verbose=False)
            dt_ms = (time.perf_counter() - t0) * 1000

            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0, 4))
            scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.empty(0)
            best = float(scores.max()) if len(scores) > 0 else 0.0

            vis = draw_detections(frame, boxes, scores)
            fname = f"{vname}_frame{frame_idx}.jpg"
            cv2.imwrite(str(OUTPUT_DIR / fname), vis)

            status = "✓" if best >= 0.5 else "~" if best > 0 else "✗"
            print(f"    [{status}] Frame {frame_idx:5d} | {len(scores)} det | best={best:.3f} | {dt_ms:.1f}ms")

    print(f"\nOutput images: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
