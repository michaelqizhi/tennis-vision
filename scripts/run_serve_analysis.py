"""CLI — serve analysis, fault detection, and serve placement heatmap.

Usage:
    python scripts/run_serve_analysis.py --csv ball_positions.csv --homography H.npy --output output_dir/
    python scripts/run_serve_analysis.py --input video.mp4 --homography H.npy --output output_dir/

Options:
    --input         Source video file (used for FPS if no --csv)
    --csv           Pre-computed ball positions CSV (frame,x,y,confidence)
    --homography    Path to homography matrix (.npy) — REQUIRED
    --output        Output directory for stats and heatmaps
    --fps           Override FPS (auto-detected from video if not given)
    --min-gap       Minimum gap between rallies in seconds (default: 2.0)
    --min-rally     Minimum rally duration in seconds (default: 1.0)
    --min-dets      Minimum ball detections per rally (default: 5)
    --no-heatmap    Skip heatmap image generation
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.features.ball_tracking.detector import BallDetection
from src.features.auto_clip.rally_detector import (
    RallyDetectorConfig,
    detect_rallies,
)
from src.features.serve_analysis.serve_detector import (
    ServeDetectorConfig,
    detect_serves,
)
from src.features.serve_analysis.fault_detector import reclassify_serves
from src.features.serve_analysis.double_fault import detect_double_faults
from src.features.serve_analysis.stats import (
    compute_serve_stats,
    format_serve_summary,
)


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

            try:
                conf = float(row.get("confidence", "0.0") or "0.0")
            except (ValueError, TypeError):
                conf = 0.0

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve analysis — detect serves, faults, double faults, and generate placement heatmaps.",
    )
    parser.add_argument("--input", type=str, help="Source video file")
    parser.add_argument("--csv", type=str, help="Ball positions CSV")
    parser.add_argument("--homography", type=str, required=True,
                        help="Homography matrix .npy file (required)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--fps", type=float, help="Override video FPS")
    parser.add_argument("--min-gap", type=float, default=2.0,
                        help="Min gap between rallies (seconds)")
    parser.add_argument("--min-rally", type=float, default=1.0,
                        help="Min rally duration (seconds)")
    parser.add_argument("--min-dets", type=int, default=5,
                        help="Min ball detections per rally")
    parser.add_argument("--no-heatmap", action="store_true",
                        help="Skip heatmap generation")
    args = parser.parse_args()

    if not args.csv and not args.input:
        parser.error("Either --input (video) or --csv (ball positions) is required")

    # Load homography
    homography = np.load(args.homography)
    print(f"Loaded homography from {args.homography}")

    # Get video info / FPS
    fps = args.fps or 30.0
    if args.input:
        vid_fps, frame_count, _, _ = get_video_info(args.input)
        if not args.fps:
            fps = vid_fps
        print(f"Video: {args.input} ({fps:.1f} fps, {frame_count} frames)")

    # Load or compute ball detections
    if args.csv:
        print(f"Loading ball positions from {args.csv}")
        detections = load_detections_from_csv(args.csv)
    else:
        print("Running ball tracking on video...")
        from src.video.reader import VideoReader
        from src.features.ball_tracking.detector import BallTracker

        reader = VideoReader(args.input)
        frames = reader.read_all()
        reader.release()

        tracker = BallTracker()
        detections = tracker.detect(frames)
        print(f"Detected ball in {sum(1 for d in detections if d.detected)}/{len(detections)} frames")

    # Detect rallies
    rally_config = RallyDetectorConfig(
        min_gap_seconds=args.min_gap,
        min_rally_seconds=args.min_rally,
        min_detections=args.min_dets,
    )
    rallies = detect_rallies(detections, fps, rally_config)
    print(f"Detected {len(rallies)} rallies")

    # Detect serves
    serve_config = ServeDetectorConfig()
    serves = detect_serves(detections, rallies, homography, fps, serve_config)
    print(f"Detected {len(serves)} serve events")

    # Reclassify faults using court geometry
    serves = reclassify_serves(serves)

    # Detect double faults
    dfs = detect_double_faults(serves)

    # Compute stats
    stats = compute_serve_stats(serves, dfs)
    print(f"\n{format_serve_summary(stats)}")

    # Save output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save stats JSON
    stats_json = stats.to_dict()
    stats_path = str(output_dir / "serve_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats_json, f, indent=2)
    print(f"\nServe stats saved to {stats_path}")

    # Generate heatmaps
    if not args.no_heatmap and stats.all_serve_positions:
        from src.features.visualization.renderer import (
            render_serve_placement,
            render_shot_heatmap,
        )

        # All serves heatmap
        all_path = str(output_dir / "serve_placement_all.png")
        render_serve_placement(
            stats.all_serve_positions,
            all_path,
            title="All Serve Placement",
        )
        print(f"All serves heatmap saved to {all_path}")

        # 1st serve heatmap
        if stats.first_serve_positions:
            first_path = str(output_dir / "serve_placement_1st.png")
            render_serve_placement(
                stats.first_serve_positions,
                first_path,
                title="1st Serve Placement",
            )
            print(f"1st serve heatmap saved to {first_path}")

        # 2nd serve heatmap
        if stats.second_serve_positions:
            second_path = str(output_dir / "serve_placement_2nd.png")
            render_serve_placement(
                stats.second_serve_positions,
                second_path,
                title="2nd Serve Placement",
            )
            print(f"2nd serve heatmap saved to {second_path}")


if __name__ == "__main__":
    main()
