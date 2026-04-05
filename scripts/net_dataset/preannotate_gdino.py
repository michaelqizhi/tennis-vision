"""Pre-annotate tennis net dataset using Grounding DINO.

Runs GDINO on all extracted frames, saves top-1 detection as:
  - YOLO-format .txt labels in data/net_detection/labels/
  - Annotated preview images in output/net_detection_preannotated/

The YOLO labels can be imported into CVAT for review/correction.

Usage:
    python scripts/net_dataset/preannotate_gdino.py
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRAME_DIR = PROJECT_ROOT / "data" / "net_detection" / "frames"
LABEL_DIR = PROJECT_ROOT / "data" / "net_detection" / "labels_gdino"
PREVIEW_DIR = PROJECT_ROOT / "output" / "net_detection_preannotated"

MODEL_ID = "IDEA-Research/grounding-dino-base"
PROMPT = "tennis net."
BOX_THRESHOLD = 0.15
TEXT_THRESHOLD = 0.15
CLASS_ID = 0  # tennis_net


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {MODEL_ID} on {device}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_ID, torch_dtype=torch_dtype
    ).to(device)
    model.eval()
    print("Model loaded.\n")

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    frames = sorted(FRAME_DIR.glob("*.jpg"))
    print(f"Processing {len(frames)} frames...")
    print(f"Labels → {LABEL_DIR}")
    print(f"Previews → {PREVIEW_DIR}\n")

    stats = {"detected": 0, "missed": 0, "total": 0}

    for i, img_path in enumerate(frames):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        rgb = frame[:, :, ::-1]
        pil_img = Image.fromarray(rgb)

        inputs = processor(images=pil_img, text=PROMPT, return_tensors="pt")
        inputs = {
            k: v.to(device, dtype=torch_dtype) if v.is_floating_point() else v.to(device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            if device == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch_dtype):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"],
            threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
            target_sizes=[(h, w)],
        )

        stats["total"] += 1
        label_path = LABEL_DIR / f"{img_path.stem}.txt"

        if results and len(results[0]["boxes"]) > 0:
            boxes = results[0]["boxes"].cpu().numpy()
            scores = results[0]["scores"].cpu().numpy()

            # Take top-1 only
            best_idx = int(scores.argmax())
            box = boxes[best_idx]  # xyxy
            score = float(scores[best_idx])

            # Convert to YOLO format: class cx cy w h (normalized)
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            label_path.write_text(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            stats["detected"] += 1

            # Draw preview
            vis = frame.copy()
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            color = (0, 255, 0) if score >= 0.3 else (0, 165, 255)
            cv2.rectangle(vis, (ix1, iy1), (ix2, iy2), color, 2)
            text = f"net {score:.2f}"
            cv2.putText(vis, text, (ix1, iy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imwrite(str(PREVIEW_DIR / img_path.name), vis)

            status = "✓" if score >= 0.3 else "~"
        else:
            # Empty label file = no detection (negative sample)
            label_path.write_text("")
            stats["missed"] += 1
            # Save clean frame as preview
            cv2.imwrite(str(PREVIEW_DIR / img_path.name), frame)
            status = "✗"

        if (i + 1) % 20 == 0 or (i + 1) == len(frames):
            print(f"  [{i+1}/{len(frames)}] {status} {img_path.name}")

    print(f"\n{'='*60}")
    print(f"DONE: {stats['detected']}/{stats['total']} frames detected, {stats['missed']} missed")
    print(f"Labels: {LABEL_DIR}")
    print(f"Previews: {PREVIEW_DIR}")


if __name__ == "__main__":
    main()
