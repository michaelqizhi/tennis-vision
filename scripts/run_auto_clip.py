"""CLI — rally detection, auto-clipping, and shot statistics.

Usage:
    python scripts/run_auto_clip.py --input video.mp4 --output output_dir/
    python scripts/run_auto_clip.py --input video.mp4 --csv ball_positions.csv --output output_dir/
    python scripts/run_auto_clip.py --input video.mp4 --csv ball_positions.csv --output output_dir/ --no-clip
    python scripts/run_auto_clip.py --csv ball_positions.csv --output output_dir/ --stats-only

Options:
    --input         Source video file (required for clipping)
    --csv           Pre-computed ball positions CSV (frame,x,y,confidence)
    --output        Output directory for clips and stats
    --homography    Path to homography matrix (.npy) for court-coordinate shot counting
    --no-clip       Skip video clipping, only produce stats
    --stats-only    Only compute stats from CSV (no video needed)
    --min-gap       Minimum gap between rallies in seconds (default: 2.0)
    --min-rally     Minimum rally duration in seconds (default: 1.0)
    --min-dets      Minimum ball detections per rally (default: 5)
    --pad-start     Padding before rally start in seconds (default: 0.5)
    --pad-end       Padding after rally end in seconds (default: 1.0)
    --fps           Override FPS (auto-detected from video if not given)
"""

import argparse
import csv
import json
from pathlib import Path

from src.features.ball_tracking.detector import BallDetection
from src.features.auto_clip.rally_detector import (
    Rally,
    RallyDetectorConfig,
    detect_rallies,
    get_rally_clips,
)
from src.features.auto_clip.clipper import clip_rallies
from src.features.rally_stats.counter import count_all_rally_shots
from src.features.rally_stats.stats import compute_rally_stats, format_stats_summary


def load_detections_from_csv(csv_path: str) -> list[BallDetection]:
    """Load ball detections from a CSV file.

    Expected columns: frame_number (or frame), x, y, confidence
    """
    detections: list[BallDetection] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_key = "frame_number" if "frame_number" in row else "frame"
            frame_num = int(row[frame_key])
            x_val = row.get("x", "")
            y_val = row.get("y", "")
            conf = float(row.get("confidence", 0.0))

            x = float(x_val) if x_val and x_val.lower() not in ("", "none", "nan") else None
            y = float(y_val) if y_val and y_val.lower() not in ("", "none", "nan") else None

            detections.append(BallDetection(
                frame_number=frame_num,
                x=x,
                y=y,
                confidence=conf,
            ))
    return detections


def get_video_info(video_path: str) -> tuple[float, int, int, int]:
    """Get FPS, frame count, width, height from a video file."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return fps, frame_count, width, height


def build_stats_json(
    rallies: list[Rally],
    stats: "RallyStats",
    rally_clips: list[str] | None = None,
) -> dict:
    """Build the output JSON structure."""
    rally_details = []
    for i, rally in enumerate(rallies):
        detail = {
            "rally_id": rally.rally_id,
            "start_frame": rally.start_frame,
            "end_frame": rally.end_frame,
            "start_time": round(rally.start_time, 2),
            "end_time": round(rally.end_time, 2),
            "duration": round(rally.duration, 2),
            "num_detections": rally.num_detections,
            "detection_density": round(rally.detection_density, 3),
        }
        if rally_clips and i < len(rally_clips):
            detail["clip_file"] = rally_clips[i]
        if i < len(stats.shots_per_rally):
            detail["shot_count"] = stats.shots_per_rally[i]
        rally_details.append(detail)

    return {
        "summary": stats.to_dict(),
        "rallies": rally_details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rally detection, auto-clipping, and shot statistics.",
    )
    parser.add_argument("--input", type=str, help="Source video file")
    parser.add_argument("--csv", type=str, help="Ball positions CSV")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--homography", type=str, help="Homography matrix .npy file")
    parser.add_argument("--no-clip", action="store_true", help="Skip video clipping")
    parser.add_argument("--stats-only", action="store_true", help="Only compute stats")
    parser.add_argument("--min-gap", type=float, default=2.0, help="Min gap between rallies (seconds)")
    parser.add_argument("--min-rally", type=float, default=1.0, help="Min rally duration (seconds)")
    parser.add_argument("--min-dets", type=int, default=5, help="Min ball detections per rally")
    parser.add_argument("--pad-start", type=float, default=0.5, help="Padding before rally (seconds)")
    parser.add_argument("--pad-end", type=float, default=1.0, help="Padding after rally (seconds)")
    parser.add_argument("--fps", type=float, help="Override video FPS")
    args = parser.parse_args()

    if not args.csv and not args.input:
        parser.error("Either --input (video) or --csv (ball positions) is required")

    if not args.stats_only and not args.no_clip and not args.input:
        parser.error("--input is required for video clipping (use --no-clip or --stats-only to skip)")

    # Get video info
    fps = args.fps or 30.0
    frame_count = 0
    frame_height = 720
    if args.input:
        vid_fps, frame_count, _, frame_height = get_video_info(args.input)
        if not args.fps:
            fps = vid_fps
        print(f"Video: {args.input} ({fps:.1f} fps, {frame_count} frames)")

    # Load or compute ball detections
    if args.csv:
        print(f"Loading ball positions from {args.csv}")
        detections = load_detections_from_csv(args.csv)
        if not frame_count:
            frame_count = max((d.frame_number for d in detections), default=0) + 1
    else:
        print("Running ball tracking on video...")
        from src.video.reader import VideoReader
        from src.features.ball_tracking.detector import BallTracker

        reader = VideoReader(args.input)
        frames = reader.read_all()
        reader.release()
        frame_count = len(frames)

        tracker = BallTracker()
        detections = tracker.detect(frames)
        print(f"Detected ball in {sum(1 for d in detections if d.detected)}/{len(detections)} frames")

    # Configure rally detection
    config = RallyDetectorConfig(
        min_gap_seconds=args.min_gap,
        min_rally_seconds=args.min_rally,
        min_detections=args.min_dets,
        pad_start_seconds=args.pad_start,
        pad_end_seconds=args.pad_end,
    )

    # Detect rallies
    rallies = detect_rallies(detections, fps, config)
    print(f"\nDetected {len(rallies)} rallies:")
    for rally in rallies:
        print(f"  Rally {rally.rally_id}: frames {rally.start_frame}-{rally.end_frame} "
              f"({rally.duration:.1f}s, {rally.num_detections} detections)")

    # Load homography if available
    import numpy as np
    homography = None
    if args.homography:
        homography = np.load(args.homography)
        print(f"Loaded homography from {args.homography}")

    # Count shots per rally
    rally_shots = count_all_rally_shots(
        detections, rallies,
        homography=homography,
        frame_height=frame_height,
    )

    # Compute aggregate stats
    from src.features.rally_stats.stats import RallyStats
    stats = compute_rally_stats(rallies, rally_shots)
    print(f"\n{format_stats_summary(stats)}")

    # Clip rallies from video
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_paths: list[str] | None = None

    if not args.stats_only and not args.no_clip and args.input:
        clip_ranges = get_rally_clips(rallies, frame_count, fps, config)
        print(f"\nClipping {len(clip_ranges)} rally segments...")
        clip_paths = clip_rallies(
            args.input,
            str(output_dir / "clips"),
            clip_ranges,
        )
        for path in clip_paths:
            print(f"  Saved: {path}")

    # Save stats JSON
    stats_json = build_stats_json(rallies, stats, clip_paths)
    stats_path = str(output_dir / "rally_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats_json, f, indent=2)
    print(f"\nStats saved to {stats_path}")

    # Save detections CSV if we computed them (not loaded from file)
    if not args.csv and detections:
        csv_path = str(output_dir / "ball_positions.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_number", "x", "y", "confidence"])
            for d in detections:
                writer.writerow([d.frame_number, d.x, d.y, d.confidence])
        print(f"Ball positions saved to {csv_path}")


if __name__ == "__main__":
    main()
