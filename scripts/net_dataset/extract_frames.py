"""Extract diverse frames from downloaded tennis videos for annotation.

Samples 15-20 evenly-spaced frames per video (skipping first/last 5%),
filters out blurry frames, and saves to data/net_detection/frames/.
Generates a manifest CSV for tracking.

Usage:
    python scripts/net_dataset/extract_frames.py
    python scripts/net_dataset/extract_frames.py --per-video 20
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VIDEO_DIR = PROJECT_ROOT / "data" / "net_detection" / "videos"
FRAME_DIR = PROJECT_ROOT / "data" / "net_detection" / "frames"
MANIFEST = PROJECT_ROOT / "data" / "net_detection" / "manifest.csv"

BLUR_THRESHOLD = 50.0  # Laplacian variance below this = blurry
SKIP_PERCENT = 0.05  # skip first/last 5% of video


def laplacian_variance(frame: np.ndarray) -> float:
    """Compute Laplacian variance as a sharpness measure."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_from_video(
    video_path: Path, n_frames: int, blur_thresh: float
) -> list[dict]:
    """Extract up to n_frames sharp frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path.name}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if total <= 0:
        print(f"  ERROR: No frames in {video_path.name}")
        cap.release()
        return []

    start = int(total * SKIP_PERCENT)
    end = int(total * (1 - SKIP_PERCENT))
    # Sample more candidates than needed to account for blur filtering
    n_candidates = int(n_frames * 1.5)
    indices = np.linspace(start, end, n_candidates, dtype=int)

    vid_id = video_path.stem
    results = []
    kept = 0

    for idx in indices:
        if kept >= n_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue

        sharpness = laplacian_variance(frame)
        if sharpness < blur_thresh:
            continue

        timestamp = idx / fps
        fname = f"{vid_id}_f{idx:06d}.jpg"
        out_path = FRAME_DIR / fname
        cv2.imwrite(str(out_path), frame)

        results.append({
            "frame_path": fname,
            "source_video": video_path.name,
            "frame_index": int(idx),
            "timestamp_sec": round(timestamp, 2),
            "sharpness": round(sharpness, 1),
        })
        kept += 1

    cap.release()
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract frames for net detection dataset")
    parser.add_argument("--per-video", type=int, default=15, help="Frames to extract per video")
    parser.add_argument("--blur-thresh", type=float, default=BLUR_THRESHOLD, help="Blur threshold")
    args = parser.parse_args()

    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        print(f"No videos found in {VIDEO_DIR}")
        print("Run download_videos.py first.")
        return

    print(f"Extracting ~{args.per_video} frames each from {len(videos)} videos")
    print(f"Blur threshold: {args.blur_thresh} | Output: {FRAME_DIR}\n")

    all_rows = []
    for video_path in videos:
        print(f"  {video_path.name}...", end=" ")
        rows = extract_from_video(video_path, args.per_video, args.blur_thresh)
        print(f"{len(rows)} frames")
        all_rows.extend(rows)

    # Write manifest
    with open(MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_path", "source_video", "frame_index", "timestamp_sec", "sharpness"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nTotal: {len(all_rows)} frames extracted")
    print(f"Manifest: {MANIFEST}")
    print(f"Frames:   {FRAME_DIR}")


if __name__ == "__main__":
    main()
