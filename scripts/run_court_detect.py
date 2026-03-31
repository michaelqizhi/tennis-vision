"""Run court detection on a tennis video.

Usage:
    python scripts/run_court_detect.py --input video.mp4 --output out.mp4
    python scripts/run_court_detect.py --input video.mp4 --output out.mp4 --sample-every 30
    python scripts/run_court_detect.py --input video.mp4 --output out.mp4 --no-refine --no-homography
"""

import argparse
import cv2
import json
import time
from pathlib import Path

from src.config import get_config
from src.video.reader import VideoReader
from src.video.writer import VideoWriter
from src.features.court_detect.detector import CourtDetector
from src.features.court_detect.visualizer import annotate_court_video


def save_court_data(
    results: list,
    output_path: str,
    homography_path: str | None = None,
) -> None:
    """Save court detection results to JSON.

    Args:
        results: List of CourtDetectionResult objects.
        output_path: Path to save keypoints JSON.
        homography_path: Optional path to save homography matrix as .npy.
    """
    import numpy as np

    data = {
        "frames": [],
        "summary": {
            "total_frames": len(results),
            "detected_frames": sum(1 for r in results if r.detected),
        },
    }

    best_result = None
    best_confidence = 0.0

    for r in results:
        frame_data = {
            "frame_number": r.frame_number,
            "num_detected": r.num_detected,
            "confidence": round(r.confidence, 3),
            "keypoints": [
                {"x": round(x, 1) if x is not None else None,
                 "y": round(y, 1) if y is not None else None}
                for x, y in r.keypoints
            ],
        }
        data["frames"].append(frame_data)

        if r.confidence > best_confidence:
            best_confidence = r.confidence
            best_result = r

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    # Save best homography matrix
    if homography_path and best_result and best_result.homography_img_to_court is not None:
        np.save(homography_path, best_result.homography_img_to_court)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run court keypoint detection on a tennis video.",
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
        "--json", "-j", default=None,
        help="Path to output JSON with keypoints. "
             "Defaults to <output_dir>/<input_name>_court.json",
    )
    parser.add_argument(
        "--homography", default=None,
        help="Path to save homography matrix (.npy). "
             "Defaults to <output_dir>/<input_name>_homography.npy",
    )
    parser.add_argument(
        "--ball-csv", default=None,
        help="Path to ball positions CSV (from Sprint 1). "
             "If provided, transforms ball positions to court coordinates.",
    )
    parser.add_argument(
        "--weights", "-w", default=None,
        help="Path to court detector weights. Defaults to weights/court_detector.pt",
    )
    parser.add_argument(
        "--sample-every", type=int, default=1,
        help="Process every Nth frame (default: 1 = all frames).",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Maximum number of frames to process (0 = all).",
    )
    parser.add_argument(
        "--no-refine", action="store_true",
        help="Disable keypoint refinement via line intersection.",
    )
    parser.add_argument(
        "--no-homography", action="store_true",
        help="Disable homography-based keypoint correction.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device (cuda/cpu). Auto-detected if not set.",
    )
    args = parser.parse_args()

    config = get_config()

    if args.weights:
        config.model.court_weights = args.weights
    if args.device:
        config.device = args.device
    if args.max_frames:
        config.video.max_frames = args.max_frames

    # Determine output paths
    input_stem = Path(args.input).stem
    output_dir = Path(args.output).parent

    json_path = args.json or str(output_dir / f"{input_stem}_court.json")
    homography_path = args.homography or str(
        output_dir / f"{input_stem}_homography.npy"
    )

    print(f"Device: {config.device}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"JSON:   {json_path}")
    print(f"Homography: {homography_path}")
    print(f"Sample every: {args.sample_every} frames")
    print(f"Refine KPs: {not args.no_refine}")
    print(f"Homography correction: {not args.no_homography}")
    print()

    # Read video
    print("Reading video...")
    with VideoReader(args.input, max_frames=config.video.max_frames) as reader:
        fps = reader.fps
        width = reader.width
        height = reader.height
        frames = reader.read_all()

    if not frames:
        print("ERROR: No frames read from video.")
        sys.exit(1)

    print(f"Read {len(frames)} frames ({width}x{height} @ {fps:.1f} fps)")

    # Run court detection
    print("\nRunning court detection...")
    start_time = time.time()
    detector = CourtDetector(
        config,
        use_refine_kps=not args.no_refine,
        use_homography=not args.no_homography,
    )
    results = detector.detect_video(frames, sample_every=args.sample_every)
    elapsed = time.time() - start_time

    detected_count = sum(1 for r in results if r.detected)
    print(f"\nDetected court in {detected_count}/{len(results)} sampled frames")
    print(f"Processing time: {elapsed:.1f}s")

    # Save court data
    print(f"\nSaving court data to {json_path}...")
    save_court_data(results, json_path, homography_path)

    # Transform ball positions if CSV provided
    if args.ball_csv:
        _transform_ball_positions(args.ball_csv, results, output_dir, input_stem)

    # Annotate and write output video
    print("\nAnnotating video frames...")
    annotated = annotate_court_video(frames, results, draw_labels=True)

    print(f"Writing output video to {args.output}...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with VideoWriter(args.output, fps, width, height) as writer:
        writer.write_frames(annotated)

    print("\nDone!")


def _transform_ball_positions(
    csv_path: str,
    results: list,
    output_dir: Path,
    input_stem: str,
) -> None:
    """Transform ball pixel positions to court coordinates using homography."""
    import csv
    import numpy as np

    # Find best homography
    best_result = max(
        (r for r in results if r.detected),
        key=lambda r: r.confidence,
        default=None,
    )
    if best_result is None or best_result.homography_img_to_court is None:
        print("WARNING: No valid homography found, cannot transform ball positions.")
        return

    H = best_result.homography_img_to_court
    print(f"\nTransforming ball positions using homography (confidence: {best_result.confidence:.0%})...")

    # Read ball positions
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Transform and write
    out_path = str(output_dir / f"{input_stem}_court_positions.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_number", "pixel_x", "pixel_y",
                          "court_x_m", "court_y_m", "confidence"])
        for row in rows:
            frame_num = row["frame_number"]
            px = row.get("x", "")
            py = row.get("y", "")
            conf = row.get("confidence", "0")

            if px and py:
                try:
                    px_f, py_f = float(px), float(py)
                    pt = np.array([[[px_f, py_f]]], dtype=np.float32)
                    court_pt = cv2.perspectiveTransform(pt, H)
                    cx, cy = court_pt[0][0]
                    writer.writerow([
                        frame_num, px, py,
                        f"{cx:.3f}", f"{cy:.3f}", conf,
                    ])
                except (ValueError, cv2.error):
                    writer.writerow([frame_num, px, py, "", "", conf])
            else:
                writer.writerow([frame_num, "", "", "", "", conf])

    print(f"Court coordinates saved to {out_path}")


if __name__ == "__main__":
    main()
