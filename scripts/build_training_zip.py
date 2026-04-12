"""Build the Kaggle training zip for near-half court detector.

Creates output/near_half_train.zip with a flat structure that Kaggle expects:
  train_near_half_court.py
  court_detector.pt
  src/__init__.py
  src/models/__init__.py
  src/models/common.py
  src/models/court_net.py
  src/features/__init__.py
  src/features/court_detect/__init__.py
  src/features/court_detect/dataset.py
  src/features/court_detect/loss.py
  near_half_train/labels.json
  near_half_train/crops/*.jpg
"""

import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SRC_FILES = {
    # arcname -> source path (relative to ROOT)
    "train_near_half_court.py": "scripts/train_near_half_court.py",
    "court_detector.pt": "weights/court_detector.pt",
    "src/models/common.py": "src/models/common.py",
    "src/models/court_net.py": "src/models/court_net.py",
    "src/features/court_detect/dataset.py": "src/features/court_detect/dataset.py",
    "src/features/court_detect/loss.py": "src/features/court_detect/loss.py",
}

# These __init__.py files must be empty stubs (the real ones import modules
# not included in the zip).
STUB_INITS = [
    "src/__init__.py",
    "src/models/__init__.py",
    "src/features/__init__.py",
    "src/features/court_detect/__init__.py",
]

LABELS_JSON = ROOT / "data" / "near_half_train" / "labels.json"
CROPS_DIR = ROOT / "data" / "near_half_train" / "crops"
OUTPUT_ZIP = ROOT / "output" / "near_half_train.zip"


def main():
    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)

    # Load labels to find referenced images
    with open(LABELS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", data if isinstance(data, list) else [])
    referenced = {os.path.basename(fr["image_path"]) for fr in frames}
    print(f"Labels reference {len(referenced)} unique images")

    missing = 0
    added_images = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Source files
        for arcname, relpath in SRC_FILES.items():
            src = ROOT / relpath
            if not src.exists():
                print(f"  WARNING: {src} not found, skipping")
                continue
            zf.write(src, arcname)
            print(f"  + {arcname}")

        # 2. Stub __init__.py files
        for arcname in STUB_INITS:
            zf.writestr(arcname, "")
            print(f"  + {arcname} (stub)")

        # 3. Labels
        zf.write(LABELS_JSON, "near_half_train/labels.json")
        print(f"  + near_half_train/labels.json ({len(frames)} frames)")

        # 4. Crop images (only those referenced in labels)
        for img_name in sorted(referenced):
            img_path = CROPS_DIR / img_name
            if img_path.exists():
                zf.write(img_path, f"near_half_train/crops/{img_name}")
                added_images += 1
            else:
                missing += 1

        print(f"\nAdded {added_images} images, {missing} referenced but missing")

    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"\n✅ Created {OUTPUT_ZIP} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
