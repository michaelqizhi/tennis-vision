"""Run ball tracking on a tennis video.

Usage:
    python scripts/run_ball_tracking.py --input video.mp4 --output out.mp4
    python scripts/run_ball_tracking.py --input video.mp4 --output out.mp4 --csv positions.csv
    python scripts/run_ball_tracking.py --input video.mp4 --output out.mp4 --no-interpolate
"""

import argparse
import csv
import time
from pathlib import Path

from src.config import get_config
from src.video.reader import VideoReader
from src.video.writer import VideoWriter
from src.features.ball_tracking.detector import BallTracker
from src.features.ball_tracking.visualizer import annotate_video_frames


def save_detections_csv(detections, csv_path: str) -> None:
    """Save ball detections to a CSV file.

    Args:
        detections: List of BallDetection objects.
        csv_path: Output CSV file path.
    """
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_number", "x", "y", "confidence"])
        for det in detections:
            writer.writerow([
                det.frame_number,
                f"{det.x:.1f}" if det.x is not None else "",
                f"{det.y:.1f}" if det.y is not None else "",
                f"{det.confidence:.2f}",
            ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TrackNetV2 ball tracking on a tennis video.",
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to input video file.",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to output annotated video file.",
    )
    parser.add_argument(
        "--csv", "-c", default=None,
        help="Path to output CSV with ball positions. "
             "Defaults to <output_dir>/<input_name>_positions.csv",
    )
    parser.add_argument(
        "--weights", "-w", default=None,
        help="Path to TrackNet weights file. Defaults to weights/tracknet.pt",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Maximum number of frames to process (0 = all).",
    )
    parser.add_argument(
        "--no-interpolate", action="store_true",
        help="Disable ball position interpolation.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device to use for inference (cuda/cpu). Auto-detected if not set.",
    )
    args = parser.parse_args()

    config = get_config()

    # Override config with CLI args
    if args.weights:
        config.model.tracknet_weights = args.weights
    if args.device:
        config.device = args.device
    if args.max_frames:
        config.video.max_frames = args.max_frames

    # Determine CSV output path
    csv_path = args.csv
    if csv_path is None:
        input_stem = Path(args.input).stem
        output_dir = Path(args.output).parent
        csv_path = str(output_dir / f"{input_stem}_positions.csv")

    print(f"Device: {config.device}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"CSV:    {csv_path}")
    print()

    # Read video
    print("Reading video...")
    with VideoReader(args.input, max_frames=config.video.max_frames) as reader:
        fps = reader.fps
        width = reader.width
        height = reader.height
        frames = reader.read_all()
    print(f"Read {len(frames)} frames ({width}x{height} @ {fps:.1f} fps)")

    # Run ball tracking
    print("\nRunning ball tracking...")
    start_time = time.time()
    tracker = BallTracker(config)
    detections = tracker.detect(frames, interpolate=not args.no_interpolate)
    elapsed = time.time() - start_time

    detected_count = sum(1 for d in detections if d.detected)
    total = len(detections)
    pct = (detected_count / total * 100) if total > 0 else 0.0
    print(f"\nDetected ball in {detected_count}/{total} frames ({pct:.1f}%)")
    if total > 0:
        print(f"Processing time: {elapsed:.1f}s "
              f"({len(frames)/elapsed:.1f} fps)")

    # Save CSV
    print(f"\nSaving positions to {csv_path}...")
    save_detections_csv(detections, csv_path)

    # Annotate and write output video
    print("Annotating video frames...")
    bt_cfg = config.ball_tracking
    annotated_frames = annotate_video_frames(
        frames, detections,
        trace_length=bt_cfg.trace_length,
        color=bt_cfg.ball_color,
        radius=bt_cfg.ball_radius,
    )

    print(f"Writing output video to {args.output}...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with VideoWriter(args.output, fps, width, height) as writer:
        writer.write_frames(annotated_frames)

    print("\nDone!")


if __name__ == "__main__":
    main()
