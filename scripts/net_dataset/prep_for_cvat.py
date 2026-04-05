"""Prepare extracted frames for CVAT annotation.

Organizes frames and prints instructions for importing into CVAT
and annotating tennis nets with bounding boxes.

Usage:
    python scripts/net_dataset/prep_for_cvat.py
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRAME_DIR = PROJECT_ROOT / "data" / "net_detection" / "frames"


def main():
    frames = sorted(FRAME_DIR.glob("*.jpg"))
    if not frames:
        print(f"No frames found in {FRAME_DIR}")
        print("Run extract_frames.py first.")
        return

    print(f"Found {len(frames)} frames in {FRAME_DIR}")
    print()

    # Print CVAT setup instructions
    print("=" * 70)
    print("CVAT ANNOTATION SETUP")
    print("=" * 70)
    print("""
1. START CVAT (if not running):
   docker compose up -d   # from your CVAT installation directory
   Open http://localhost:8080 in your browser

2. CREATE A NEW PROJECT:
   - Name: "Tennis Net Detection"
   - Labels: Add one label → "tennis_net"
   - Click "Submit & Open"

3. CREATE A TASK:
   - Click "+ Create new task"
   - Name: "net_detection_batch1"
   - Project: "Tennis Net Detection"
   - Upload files: Select ALL images from:
""")
    print(f"     {FRAME_DIR}")
    print("""
   - Click "Submit & Open"

4. ANNOTATE:
   - Open the task, click "Job #1"
   - Use the rectangle/bounding box tool
   - Draw a box around the tennis net in each frame
   - Label: "tennis_net"

5. EXPORT:
   - Go to the task page
   - Click "..." → "Export task dataset"
   - Format: "YOLO 1.1"
   - Download and extract to:
""")
    print(f"     {PROJECT_ROOT / 'data' / 'net_detection' / 'labels'}")
    print()

    # Annotation guidelines
    print("=" * 70)
    print("ANNOTATION GUIDELINES")
    print("=" * 70)
    print("""
WHAT TO BOX:
  ✓ The entire visible tennis net including the net band (white strip on top)
  ✓ Include the net posts if they are part of the net structure
  ✓ Box should be tight — just the net, not the court behind it

EDGE CASES:
  ✓ Partially visible net (e.g., cropped by frame edge) — box the visible part
  ✓ Net partially occluded by player — still box the full estimated net extent
  ✗ Skip frames where the net is completely invisible or <10% visible
  ✗ Do NOT box fences, barriers, or other nets in the background

BOX PLACEMENT:
  - Top edge: align with the top of the net band
  - Bottom edge: align with the bottom of the net where it meets the court
  - Left/right edges: align with the net posts or where the net exits the frame
  - Keep it tight — minimal padding

QUALITY CHECKS:
  - Every frame should have exactly 0 or 1 "tennis_net" box
  - If no net is visible, leave the frame unannotated (it becomes a negative sample)
""")

    print("=" * 70)
    print(f"Ready: {len(frames)} frames waiting for annotation")
    print("=" * 70)


if __name__ == "__main__":
    main()
