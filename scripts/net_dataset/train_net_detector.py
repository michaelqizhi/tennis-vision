"""Train YOLOv8n for tennis net detection.

Expects annotated data in YOLO format (from CVAT export):
  data/net_detection/frames/  — images
  data/net_detection/labels/  — YOLO .txt label files

Automatically splits into train/val/test (80/10/10), creates the
dataset YAML, and trains YOLOv8n with transfer learning.

Usage:
    python scripts/net_dataset/train_net_detector.py
    python scripts/net_dataset/train_net_detector.py --epochs 150 --batch 16
"""

import argparse
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRAME_DIR = PROJECT_ROOT / "data" / "net_detection" / "frames"
LABEL_DIR = PROJECT_ROOT / "data" / "net_detection" / "labels"
DATASET_DIR = PROJECT_ROOT / "data" / "net_detection" / "dataset"
DATASET_YAML = DATASET_DIR / "dataset.yaml"
WEIGHTS_SRC = PROJECT_ROOT / "yolov8n.pt"
WEIGHTS_OUT = PROJECT_ROOT / "weights" / "net_detector.pt"


def create_split(seed: int = 42):
    """Split annotated images into train/val/test (80/10/10)."""
    # Find images that have corresponding label files
    labeled_images = []
    unlabeled_images = []

    for img_path in sorted(FRAME_DIR.glob("*.jpg")):
        label_path = LABEL_DIR / f"{img_path.stem}.txt"
        if label_path.exists():
            labeled_images.append(img_path.stem)
        else:
            unlabeled_images.append(img_path.stem)

    if not labeled_images:
        print("ERROR: No labeled images found.")
        print(f"  Images dir: {FRAME_DIR}")
        print(f"  Labels dir: {LABEL_DIR}")
        print("  Export from CVAT in 'YOLO 1.1' format and place .txt files in the labels dir.")
        return False

    print(f"Found {len(labeled_images)} labeled + {len(unlabeled_images)} unlabeled images")

    random.seed(seed)
    random.shuffle(labeled_images)

    n = len(labeled_images)
    n_val = max(1, int(n * 0.1))
    n_test = max(1, int(n * 0.1))
    n_train = n - n_val - n_test

    splits = {
        "train": labeled_images[:n_train],
        "val": labeled_images[n_train : n_train + n_val],
        "test": labeled_images[n_train + n_val :],
    }

    # Copy files into split directories
    for split_name, stems in splits.items():
        img_dir = DATASET_DIR / split_name / "images"
        lbl_dir = DATASET_DIR / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for stem in stems:
            shutil.copy2(FRAME_DIR / f"{stem}.jpg", img_dir / f"{stem}.jpg")
            shutil.copy2(LABEL_DIR / f"{stem}.txt", lbl_dir / f"{stem}.txt")

    print(f"Split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Add some unlabeled images as negative samples to training set
    n_negatives = min(len(unlabeled_images), int(n_train * 0.1))
    if n_negatives > 0:
        neg_samples = random.sample(unlabeled_images, n_negatives)
        neg_img_dir = DATASET_DIR / "train" / "images"
        neg_lbl_dir = DATASET_DIR / "train" / "labels"
        for stem in neg_samples:
            shutil.copy2(FRAME_DIR / f"{stem}.jpg", neg_img_dir / f"{stem}.jpg")
            # Empty label file = negative sample (no objects)
            (neg_lbl_dir / f"{stem}.txt").touch()
        print(f"Added {n_negatives} negative samples to training set")

    # Write dataset.yaml
    yaml_content = f"""# Tennis Net Detection Dataset
path: {DATASET_DIR.as_posix()}
train: train/images
val: val/images
test: test/images

names:
  0: tennis_net
"""
    DATASET_YAML.write_text(yaml_content)
    print(f"Dataset YAML: {DATASET_YAML}")
    return True


def train(epochs: int, batch: int, imgsz: int):
    """Train YOLOv8n on the prepared dataset."""
    from ultralytics import YOLO

    if not WEIGHTS_SRC.exists():
        print(f"ERROR: Base weights not found: {WEIGHTS_SRC}")
        return

    print(f"\nTraining YOLOv8n for {epochs} epochs (batch={batch}, imgsz={imgsz})")
    print(f"Base weights: {WEIGHTS_SRC}")
    print(f"Dataset: {DATASET_YAML}\n")

    model = YOLO(str(WEIGHTS_SRC))
    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(PROJECT_ROOT / "output" / "net_training"),
        name="yolov8n_net",
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
        verbose=True,
    )

    # Copy best weights
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, WEIGHTS_OUT)
        print(f"\nBest weights saved to: {WEIGHTS_OUT}")
    else:
        print(f"\nWARNING: best.pt not found at {best_pt}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n for tennis net detection")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--split-only", action="store_true", help="Only create train/val/test split, don't train")
    args = parser.parse_args()

    print("=" * 60)
    print("YOLOv8n Tennis Net Detector — Training")
    print("=" * 60)

    # Clean previous split
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    if not create_split():
        return

    if args.split_only:
        print("\n--split-only: Skipping training.")
        return

    train(args.epochs, args.batch, args.imgsz)


if __name__ == "__main__":
    main()
