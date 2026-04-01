"""Diagnostic script: test YOLO-World detection with different prompts,
confidence thresholds, and input resolutions on a single frame from the
courtside tennis video.

Usage:
    python scripts/diagnose_yoloworld.py
"""

import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "tests/fixtures/tennis_test.mp4"
MODEL_NAME = "yolov8l-worldv2.pt"

# Sample multiple frames spread across the video to get a representative picture
FRAME_INDICES = [50, 150, 250, 350, 450]

PROMPTS = [
    ["tennis ball"],
    ["ball"],
    ["small ball"],
    ["yellow ball"],
    ["sports ball"],
    ["small yellow tennis ball"],
    ["tennis ball", "ball"],
]

CONF_THRESHOLDS = [0.1, 0.05, 0.01, 0.005, 0.001]

IMGSZ_OPTIONS = [640, 1280]


def extract_frames(video_path: str, indices: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in indices:
        if idx >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def run_diagnostics():
    print("=" * 80)
    print("YOLO-World Tennis Ball Detection Diagnostics")
    print("=" * 80)

    frames = extract_frames(VIDEO_PATH, FRAME_INDICES)
    print(f"\nLoaded {len(frames)} frames from {VIDEO_PATH}")
    print(f"Frame shape: {frames[0].shape}")
    print(f"Model: {MODEL_NAME}\n")

    model = YOLO(MODEL_NAME)

    # ── Phase 1: Prompt sweep at default conf=0.1, imgsz=640 ──
    print("=" * 80)
    print("PHASE 1: Prompt sweep  (conf=0.1, imgsz=640)")
    print("=" * 80)
    for prompt in PROMPTS:
        model.set_classes(prompt)
        detections = 0
        total_conf = 0.0
        for frame in frames:
            results = model.predict(frame, device="cpu", verbose=False,
                                    conf=0.1, imgsz=640)
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                best_conf = float(boxes.conf.max())
                detections += 1
                total_conf += best_conf
        avg_conf = total_conf / detections if detections > 0 else 0
        print(f"  Prompt {str(prompt):45s} → {detections}/{len(frames)} frames, "
              f"avg_conf={avg_conf:.4f}")

    # ── Phase 2: Confidence threshold sweep (best prompt candidates) ──
    print("\n" + "=" * 80)
    print("PHASE 2: Confidence threshold sweep  (imgsz=640)")
    print("=" * 80)
    for prompt in PROMPTS:
        model.set_classes(prompt)
        print(f"\n  Prompt: {prompt}")
        for conf in CONF_THRESHOLDS:
            detections = 0
            confs_seen = []
            for frame in frames:
                results = model.predict(frame, device="cpu", verbose=False,
                                        conf=conf, imgsz=640)
                if results and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    best_conf = float(boxes.conf.max())
                    detections += 1
                    confs_seen.append(best_conf)
            avg_c = np.mean(confs_seen) if confs_seen else 0
            min_c = min(confs_seen) if confs_seen else 0
            max_c = max(confs_seen) if confs_seen else 0
            print(f"    conf≥{conf:<6.3f} → {detections}/{len(frames)} "
                  f"(avg={avg_c:.4f}, min={min_c:.4f}, max={max_c:.4f})")

    # ── Phase 3: Input resolution sweep ──
    print("\n" + "=" * 80)
    print("PHASE 3: Input resolution sweep  (conf=0.01)")
    print("=" * 80)
    for prompt in [["tennis ball"], ["ball"], ["small yellow tennis ball"]]:
        model.set_classes(prompt)
        print(f"\n  Prompt: {prompt}")
        for imgsz in IMGSZ_OPTIONS:
            detections = 0
            confs_seen = []
            bbox_sizes = []
            for frame in frames:
                results = model.predict(frame, device="cpu", verbose=False,
                                        conf=0.01, imgsz=imgsz)
                if results and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    best_idx = int(boxes.conf.argmax())
                    best_conf = float(boxes.conf[best_idx])
                    xyxy = boxes.xyxy[best_idx].cpu().numpy()
                    w = xyxy[2] - xyxy[0]
                    h = xyxy[3] - xyxy[1]
                    detections += 1
                    confs_seen.append(best_conf)
                    bbox_sizes.append((w, h))
            avg_c = np.mean(confs_seen) if confs_seen else 0
            avg_w = np.mean([s[0] for s in bbox_sizes]) if bbox_sizes else 0
            avg_h = np.mean([s[1] for s in bbox_sizes]) if bbox_sizes else 0
            print(f"    imgsz={imgsz:5d} → {detections}/{len(frames)} "
                  f"(avg_conf={avg_c:.4f}, avg_bbox={avg_w:.1f}x{avg_h:.1f})")

    # ── Phase 4: Detailed detection dump on frame 0 ──
    print("\n" + "=" * 80)
    print("PHASE 4: Detailed dump — frame 0, conf=0.001, imgsz=1280")
    print("=" * 80)
    frame0 = frames[0]
    for prompt in PROMPTS:
        model.set_classes(prompt)
        results = model.predict(frame0, device="cpu", verbose=False,
                                conf=0.001, imgsz=1280)
        n = len(results[0].boxes) if results else 0
        print(f"\n  Prompt: {prompt}  — {n} detection(s)")
        if results and n > 0:
            boxes = results[0].boxes
            for i in range(min(n, 10)):
                conf = float(boxes.conf[i])
                xyxy = boxes.xyxy[i].cpu().numpy()
                w = xyxy[2] - xyxy[0]
                h = xyxy[3] - xyxy[1]
                cx = (xyxy[0] + xyxy[2]) / 2
                cy = (xyxy[1] + xyxy[3]) / 2
                print(f"    [{i}] conf={conf:.4f}  bbox=({xyxy[0]:.0f},{xyxy[1]:.0f})-"
                      f"({xyxy[2]:.0f},{xyxy[3]:.0f})  size={w:.1f}x{h:.1f}  "
                      f"center=({cx:.1f},{cy:.1f})")

    # ── Phase 5: Full video scan — best combo ──
    print("\n" + "=" * 80)
    print("PHASE 5: Full video scan (500 frames) — candidate configs")
    print("=" * 80)

    cap = cv2.VideoCapture(VIDEO_PATH)
    all_frames = []
    for i in range(500):
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    print(f"  Loaded {len(all_frames)} frames\n")

    configs = [
        (["tennis ball"], 0.1, 640, "CURRENT"),
        (["tennis ball"], 0.01, 640, "lower-conf"),
        (["tennis ball"], 0.01, 1280, "lower-conf+hires"),
        (["ball"], 0.01, 1280, "ball+lowconf+hires"),
        (["small yellow tennis ball"], 0.01, 1280, "verbose-prompt"),
        (["tennis ball", "ball"], 0.01, 1280, "multi-class"),
    ]

    for classes, conf, imgsz, label in configs:
        model.set_classes(classes)
        det_count = 0
        for frame in all_frames:
            results = model.predict(frame, device="cpu", verbose=False,
                                    conf=conf, imgsz=imgsz)
            if results and len(results[0].boxes) > 0:
                det_count += 1
        pct = 100.0 * det_count / len(all_frames)
        print(f"  [{label:25s}] classes={str(classes):35s} conf={conf:.3f} "
              f"imgsz={imgsz} → {det_count}/{len(all_frames)} ({pct:.1f}%)")


if __name__ == "__main__":
    run_diagnostics()
