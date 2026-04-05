"""Test Grounding DINO zero-shot tennis net detection.

Extracts sample frames from test videos and runs Grounding DINO with
a "tennis net." prompt to evaluate whether zero-shot detection is viable,
potentially avoiding the need to finetune YOLOv5.

Usage:
    python scripts/test_net_detection.py
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ── Config ──────────────────────────────────────────────────────────────
MODEL_ID = "IDEA-Research/grounding-dino-base"
PROMPTS = ["tennis net."]  # trailing period required by GDINO tokenizer
BOX_THRESHOLD = 0.1
TEXT_THRESHOLD = 0.1
NUM_FRAMES = 5  # frames to sample per video

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOS = [
    PROJECT_ROOT / "tests" / "fixtures" / "courtside_match_2.mp4",
    PROJECT_ROOT / "tests" / "fixtures" / "real_match_10min.mp4",
]
OUTPUT_DIR = PROJECT_ROOT / "output" / "net_detection_test"


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


def load_model(device: str):
    """Load Grounding DINO model and processor."""
    print(f"Loading {MODEL_ID} on {device} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_ID, torch_dtype=torch_dtype
    ).to(device)
    model.eval()
    print("Model loaded.")
    return model, processor, torch_dtype


def detect_net(
    model,
    processor,
    frame: np.ndarray,
    prompt: str,
    device: str,
    dtype,
) -> list[dict]:
    """Run Grounding DINO on a single frame. Returns list of detections."""
    rgb = frame[:, :, ::-1]
    pil_img = Image.fromarray(rgb)

    inputs = processor(images=pil_img, text=prompt, return_tensors="pt")
    inputs = {
        k: v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        if device == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[(pil_img.height, pil_img.width)],
    )

    detections = []
    if results and len(results[0]["boxes"]) > 0:
        for i in range(len(results[0]["boxes"])):
            box = results[0]["boxes"][i].cpu().numpy()  # xyxy
            score = float(results[0]["scores"][i])
            label = results[0]["text_labels"][i] if "text_labels" in results[0] else prompt.rstrip(".")
            detections.append({
                "box": box,  # [x1, y1, x2, y2]
                "score": score,
                "label": label,
            })
    return detections


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes and labels on a frame copy."""
    vis = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"].astype(int)
        score = det["score"]
        label = det.get("label", "net")

        color = (0, 255, 0) if score >= 0.3 else (0, 165, 255)  # green if good, orange if weak
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(vis, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return vis


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor, dtype = load_model(device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Grounding DINO Zero-Shot Tennis Net Detection Test")
    print(f"{'='*70}")
    print(f"Device: {device} | Prompt: {PROMPTS} | Thresholds: box={BOX_THRESHOLD}, text={TEXT_THRESHOLD}")
    print(f"Output: {OUTPUT_DIR}\n")

    summary_rows = []

    for video_path in VIDEOS:
        vname = video_path.stem
        print(f"\n── Video: {vname} ──")
        if not video_path.exists():
            print(f"  SKIP: {video_path} not found")
            continue

        frames = extract_frames(video_path)
        print(f"  Extracted {len(frames)} frames")

        for frame_idx, frame in frames:
            for prompt in PROMPTS:
                detections = detect_net(model, processor, frame, prompt, device, dtype)
                n_det = len(detections)
                best_score = max((d["score"] for d in detections), default=0.0)

                # Keep only the top-1 detection for cleaner visualization
                top1 = [max(detections, key=lambda d: d["score"])] if detections else []

                tag = prompt.rstrip(".").replace(" ", "_")
                fname = f"{vname}_frame{frame_idx}_{tag}.jpg"
                vis = draw_detections(frame, top1)
                out_path = OUTPUT_DIR / fname
                cv2.imwrite(str(out_path), vis)

                status = "✓" if best_score >= 0.3 else "~" if n_det > 0 else "✗"
                print(f"  [{status}] Frame {frame_idx:5d} | {n_det} det (showing top-1) | best={best_score:.3f} | {fname}")

                summary_rows.append({
                    "video": vname,
                    "frame": frame_idx,
                    "prompt": prompt,
                    "n_detections": n_det,
                    "best_score": best_score,
                })

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Video':<25} {'Frame':>6} {'Prompt':<15} {'#Det':>5} {'Best':>6}")
    print("-" * 62)
    for row in summary_rows:
        print(
            f"{row['video']:<25} {row['frame']:>6} {row['prompt']:<15} "
            f"{row['n_detections']:>5} {row['best_score']:>6.3f}"
        )

    good = sum(1 for r in summary_rows if r["best_score"] >= 0.3)
    weak = sum(1 for r in summary_rows if 0 < r["best_score"] < 0.3)
    miss = sum(1 for r in summary_rows if r["best_score"] == 0)
    total = len(summary_rows)
    print(f"\nResults: {good}/{total} good (≥0.3), {weak}/{total} weak (<0.3), {miss}/{total} missed")
    print(f"Output images saved to: {OUTPUT_DIR}")

    if good >= total * 0.6:
        print("\n→ Grounding DINO looks VIABLE for tennis net detection!")
    elif good + weak >= total * 0.5:
        print("\n→ Grounding DINO shows PARTIAL success — may need prompt tuning or post-processing.")
    else:
        print("\n→ Grounding DINO struggles with tennis nets — consider YOLOv5 finetuning.")


if __name__ == "__main__":
    main()
