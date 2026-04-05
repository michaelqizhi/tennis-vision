"""Create a YOLO-format ZIP for CVAT annotation import.

Packages the GDINO pre-annotations (labels_gdino/*.txt) with a
classes.txt into a ZIP that CVAT can import via "Upload annotations"
→ "YOLO 1.1" format.

Usage:
    python scripts/net_dataset/create_cvat_zip.py
"""

import shutil
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LABEL_DIR = PROJECT_ROOT / "data" / "net_detection" / "labels_gdino"
OUTPUT_ZIP = PROJECT_ROOT / "data" / "net_detection" / "cvat_preannotations"


def main():
    labels = sorted(LABEL_DIR.glob("*.txt"))
    if not labels:
        print(f"No label files found in {LABEL_DIR}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Create obj_train_data dir with label files
        obj_dir = tmp / "obj_train_data"
        obj_dir.mkdir()
        for lbl in labels:
            shutil.copy2(lbl, obj_dir / lbl.name)

        # Create classes.txt (CVAT expects this in YOLO format)
        (tmp / "classes.txt").write_text("tennis_net\n")

        # Create obj.names (alias used by some YOLO importers)
        (tmp / "obj.names").write_text("tennis_net\n")

        # Create obj.data (required by CVAT's YOLO 1.1 importer)
        obj_data = (
            f"classes = 1\n"
            f"names = obj.names\n"
            f"train = train.txt\n"
            f"backup = backup/\n"
        )
        (tmp / "obj.data").write_text(obj_data)

        # Create train.txt listing all label files
        train_lines = [f"obj_train_data/{lbl.name.replace('.txt', '.jpg')}\n" for lbl in labels]
        (tmp / "train.txt").write_text("".join(train_lines))

        # Zip it
        zip_path = shutil.make_archive(str(OUTPUT_ZIP), "zip", tmpdir)
        print(f"Created: {zip_path}")
        print(f"Contains: {len(labels)} label files + classes.txt")
        print(f"\nImport in CVAT: Task → ⋮ → Upload annotations → YOLO 1.1 → select this ZIP")


if __name__ == "__main__":
    main()
