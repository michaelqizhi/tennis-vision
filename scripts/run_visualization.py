"""Generate court heatmap visualizations from ball tracking data.

Usage:
    python scripts/run_visualization.py --csv positions.csv --homography homography.npy --output heatmap.png
    python scripts/run_visualization.py --csv positions.csv --homography homography.npy --output heatmap.png --serve-only
    python scripts/run_visualization.py --csv positions.csv --homography homography.npy --output heatmap.png --video input.mp4 --video-output overlay.mp4
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from src.features.court_detect.homography import transform_points_to_court
from src.features.visualization.renderer import (
    render_shot_heatmap,
    render_serve_placement,
    render_court_overlay,
    embed_overlay_in_frame,
)


def load_ball_positions_csv(csv_path: str) -> list[tuple[float, float]]:
    """Load ball pixel positions from a CSV file.

    Expected columns: frame_number, x, y, confidence

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of (x, y) pixel positions (only detected frames).
    """
    positions = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_str = row.get("x", "")
            y_str = row.get("y", "")
            if x_str and y_str:
                try:
                    positions.append((float(x_str), float(y_str)))
                except ValueError:
                    continue
    return positions


def load_court_positions_csv(csv_path: str) -> list[tuple[float, float]]:
    """Load pre-transformed court positions from a CSV file.

    Expected columns: frame_number, pixel_x, pixel_y, court_x_m, court_y_m, confidence

    Args:
        csv_path: Path to the court-positions CSV file.

    Returns:
        List of (x_meters, y_meters) court positions.
    """
    positions = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cx = row.get("court_x_m", "")
            cy = row.get("court_y_m", "")
            if cx and cy:
                try:
                    positions.append((float(cx), float(cy)))
                except ValueError:
                    continue
    return positions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate court heatmap visualizations from ball tracking data.",
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to ball positions CSV (from run_ball_tracking.py) or "
             "court positions CSV (from run_court_detect.py --ball-csv).",
    )
    parser.add_argument(
        "--homography", default=None,
        help="Path to homography .npy file (from run_court_detect.py). "
             "Required if CSV has pixel coordinates. "
             "Not needed if CSV already has court_x_m/court_y_m columns.",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to output PNG heatmap image.",
    )
    parser.add_argument(
        "--title", default=None,
        help="Title for the heatmap. Defaults to auto-generated.",
    )
    parser.add_argument(
        "--serve-only", action="store_true",
        help="Filter to only show positions in service boxes.",
    )
    parser.add_argument(
        "--region", default="full",
        choices=["full", "service_boxes", "near_service", "far_service",
                 "near_half", "far_half"],
        help="Court region filter (default: full).",
    )
    parser.add_argument(
        "--scatter", action="store_true",
        help="Use scatter plot instead of KDE heatmap.",
    )
    parser.add_argument(
        "--video", default=None,
        help="Path to source video (for overlay output).",
    )
    parser.add_argument(
        "--video-output", default=None,
        help="Path to output video with court overlay embedded.",
    )
    args = parser.parse_args()

    # Determine if CSV has court coordinates or pixel coordinates
    court_positions = _load_positions(args.csv, args.homography)

    if not court_positions:
        print("ERROR: No valid positions found. Check CSV and homography files.")
        sys.exit(1)

    print(f"Loaded {len(court_positions)} ball positions in court coordinates.")

    # Determine region and title
    region = "service_boxes" if args.serve_only else args.region
    if args.title:
        title = args.title
    elif args.serve_only:
        title = "Serve Placement Heatmap"
    else:
        title = "Shot Placement Heatmap"

    use_kde = not args.scatter

    # Render heatmap
    if args.serve_only:
        output = render_serve_placement(
            court_positions, args.output, title=title, use_kde=use_kde,
        )
    else:
        output = render_shot_heatmap(
            court_positions, args.output, title=title,
            region=region, use_kde=use_kde,
        )
    print(f"Heatmap saved to {output}")

    # Video overlay if requested
    if args.video and args.video_output:
        _create_video_overlay(
            args.video, args.video_output, court_positions, use_kde,
        )


def _load_positions(
    csv_path: str,
    homography_path: str | None,
) -> list[tuple[float, float]]:
    """Load positions, transforming with homography if needed."""
    # Check if CSV has court coordinates already
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []

    if "court_x_m" in fields and "court_y_m" in fields:
        print("CSV contains court coordinates — using directly.")
        return load_court_positions_csv(csv_path)

    # Need homography to transform pixel positions
    if homography_path is None:
        print("ERROR: CSV has pixel coordinates but no --homography provided.")
        print("Run court detection first or provide a pre-transformed CSV.")
        sys.exit(1)

    homography_file = Path(homography_path)
    if not homography_file.exists():
        print(f"ERROR: Homography file not found: {homography_path}")
        sys.exit(1)

    H = np.load(str(homography_file))
    print(f"Loaded homography from {homography_path}")

    pixel_positions = load_ball_positions_csv(csv_path)
    if not pixel_positions:
        return []

    court_positions = transform_points_to_court(pixel_positions, H)
    return court_positions


def _create_video_overlay(
    video_path: str,
    output_path: str,
    positions: list[tuple[float, float]],
    use_kde: bool,
) -> None:
    """Create a video with a court heatmap overlay in the corner."""
    import cv2
    from src.video.reader import VideoReader
    from src.video.writer import VideoWriter

    print(f"\nCreating video with overlay...")

    # Pre-render the overlay
    overlay = render_court_overlay(positions, use_kde=use_kde)
    print(f"Overlay size: {overlay.shape[1]}x{overlay.shape[0]}")

    with VideoReader(video_path) as reader:
        fps = reader.fps
        width = reader.width
        height = reader.height
        frames = reader.read_all()

    if not frames:
        print("ERROR: No frames read from video.")
        return

    print(f"Processing {len(frames)} frames...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with VideoWriter(output_path, fps, width, height) as writer:
        for frame in frames:
            annotated = embed_overlay_in_frame(frame, overlay)
            writer.write_frame(annotated)

    print(f"Video with overlay saved to {output_path}")


if __name__ == "__main__":
    main()
