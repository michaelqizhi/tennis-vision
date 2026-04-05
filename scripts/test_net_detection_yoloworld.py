"""Test YOLO-World zero-shot tennis net detection.

Compares YOLOv8s-worldv2 and YOLOv8l-worldv2 on sampled frames from
test videos using set_classes(["tennis net"]).  Shows top-1 detection
per frame with inference timing for comparison against Grounding DINO.

Usage:
    python scripts/test_net_detection_yoloworld.py
"""

import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────────────────────
NUM_FRAMES = 5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "yolov8s-worldv2": PROJECT_ROOT / "yolov8s-worldv2.pt",
    "yolov8l-worldv2": PROJECT_ROOT / "yolov8l-worldv2.pt",
}
VIDEOS = [
    PROJECT_ROOT / "tests" / "fixtures" / "courtside_match_2.mp4",
    PROJECT_ROOT / "tests" / "fixtures" / "real_match_10min.mp4",
]
OUTPUT_DIR = PROJECT_ROOT / "output" / "net_detection_test" / "yoloworld"
CLASS_NAMES = ["tennis net"]
CONF_THRESHOLD = 0.05  # low threshold to see what the model finds


def extract_frames(video_path: Path, n: int = NUM_FRAMES) -> list[tuple[int, np.ndarray]]:
    """Extract *n* evenly-spaced frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        print(f"  ERROR: No frames in {video_path}")
        cap.release()
        return []

    indices = np.linspace(0, total - 1, n, dtype=int)
    frames: list[tuple[int, np.ndarray]] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append((int(idx), frame))
    cap.release()
    return frames


def draw_top1(frame: np.ndarray, boxes, scores, label: str) -> tuple[np.ndarray, float]:
    """Draw only the top-1 detection on a frame copy. Returns (vis, best_score)."""
    vis = frame.copy()
    if len(scores) == 0:
        return vis, 0.0

    best_idx = int(np.argmax(scores))
    score = float(scores[best_idx])
    box = boxes[best_idx].astype(int)
    x1, y1, x2, y2 = box

    color = (0, 255, 0) if score >= 0.3 else (0, 165, 255)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {score:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
    cv2.putText(vis, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return vis, score


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*75}")
    print("YOLO-World Zero-Shot Tennis Net Detection Test")
    print(f"{'='*75}")
    print(f"Classes: {CLASS_NAMES} | Conf threshold: {CONF_THRESHOLD}")
    print(f"Output: {OUTPUT_DIR}\n")

    # Pre-extract frames so we reuse them across models
    all_frames: dict[str, list[tuple[int, np.ndarray]]] = {}
    for video_path in VIDEOS:
        vname = video_path.stem
        if not video_path.exists():
            print(f"SKIP: {video_path} not found")
            continue
        frames = extract_frames(video_path)
        all_frames[vname] = frames
        print(f"Extracted {len(frames)} frames from {vname}")

    summary_rows = []

    for model_name, model_path in MODELS.items():
        print(f"\n{'─'*75}")
        print(f"Model: {model_name} ({model_path.stat().st_size / 1e6:.1f} MB)")
        print(f"{'─'*75}")

        model = YOLO(str(model_path))
        model.set_classes(CLASS_NAMES)

        # Warmup
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        model.predict(dummy, conf=CONF_THRESHOLD, verbose=False)

        for vname, frames in all_frames.items():
            print(f"\n  ── Video: {vname} ──")

            for frame_idx, frame in frames:
                t0 = time.perf_counter()
                results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
                dt_ms = (time.perf_counter() - t0) * 1000

                r = results[0]
                boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0, 4))
                scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.empty(0)
                n_det = len(scores)

                vis, best_score = draw_top1(frame, boxes, scores, "tennis net")

                fname = f"{model_name}_{vname}_frame{frame_idx}.jpg"
                cv2.imwrite(str(OUTPUT_DIR / fname), vis)

                status = "✓" if best_score >= 0.3 else "~" if n_det > 0 else "✗"
                print(f"    [{status}] Frame {frame_idx:5d} | {n_det:2d} det | best={best_score:.3f} | {dt_ms:6.1f}ms | {fname}")

                summary_rows.append({
                    "model": model_name,
                    "video": vname,
                    "frame": frame_idx,
                    "n_det": n_det,
                    "best_score": best_score,
                    "time_ms": dt_ms,
                })

        del model

    # Summary
    print(f"\n{'='*75}")
    print("SUMMARY")
    print(f"{'='*75}")
    print(f"{'Model':<22} {'Video':<25} {'Frame':>6} {'#Det':>5} {'Best':>6} {'Time':>8}")
    print("-" * 78)
    for row in summary_rows:
        print(
            f"{row['model']:<22} {row['video']:<25} {row['frame']:>6} "
            f"{row['n_det']:>5} {row['best_score']:>6.3f} {row['time_ms']:>7.1f}ms"
        )

    # Per-model aggregates
    print(f"\n{'─'*75}")
    print("Per-model averages:")
    for model_name in MODELS:
        rows = [r for r in summary_rows if r["model"] == model_name]
        if not rows:
            continue
        avg_score = np.mean([r["best_score"] for r in rows])
        avg_time = np.mean([r["time_ms"] for r in rows])
        good = sum(1 for r in rows if r["best_score"] >= 0.3)
        total = len(rows)
        print(f"  {model_name}: avg_conf={avg_score:.3f}, avg_time={avg_time:.1f}ms, good={good}/{total}")

    print(f"\nOutput images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
