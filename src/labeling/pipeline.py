"""End-to-end CLI orchestrator for the labeling pipeline.

Runs the full multi-model consensus labeling pipeline on a video file
and produces all output artifacts in a single directory.

Usage::

    python -m src.labeling.pipeline <video_path> --output <dir> [options]

Outputs:
    - ``detections.json``       — raw per-model detections
    - ``consensus.json``        — consensus results with labels
    - ``annotations_cvat.xml``  — CVAT XML 1.1 format
    - ``annotations_coco.json`` — COCO JSON format
    - ``review.mp4``            — overlay visualization video
    - ``rallies.json``          — rally boundary time ranges

Pipeline stats are printed at the end.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from src.labeling.consensus import (
    compute_consensus,
    compute_consensus_stats,
    serialize_consensus,
    apply_player_proximity_filter,
)
from src.labeling.export import export_cvat_xml, export_coco_json
from src.labeling.inference_runner import get_default_detectors, run_inference
from src.labeling.benchmark import generate_benchmark_report
from src.labeling.player_detector import PlayerDetector, serialize_player_boxes
from src.labeling.rally_boundaries import (
    RallyBoundaryConfig,
    detect_rally_boundaries,
    serialize_rally_boundaries,
)
from src.labeling.visualize import generate_review_video
from src.video.reader import VideoReader

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.labeling.pipeline",
        description="Multi-model consensus labeling pipeline for tennis ball detection.",
    )
    parser.add_argument(
        "video_path",
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output", "-o",
        default="output/labeling",
        help="Output directory for all artifacts (default: output/labeling).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device (default: cpu). Use 'cuda' for GPU.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process (0 = all, default: 0).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=15.0,
        help="Consensus agreement threshold in pixels (default: 15.0).",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=2.0,
        help="Rally detection sliding window size in seconds (default: 2.0).",
    )
    parser.add_argument(
        "--rally-threshold",
        type=float,
        default=0.30,
        help="Minimum detection rate for a rally window (default: 0.30).",
    )
    parser.add_argument(
        "--min-rally",
        type=float,
        default=3.0,
        help="Minimum rally duration in seconds (default: 3.0).",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=4.0,
        help="Minimum gap duration in seconds (default: 4.0).",
    )
    parser.add_argument(
        "--skip-review-video",
        action="store_true",
        help="Skip generating the review overlay video (faster).",
    )
    parser.add_argument(
        "--no-player-filter",
        action="store_true",
        help="Disable player-proximity filtering of ball detections.",
    )
    parser.add_argument(
        "--player-distance",
        type=float,
        default=200.0,
        help="Max pixel distance from player for ball detection (default: 200).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser


def run_labeling_pipeline(
    video_path: str,
    output_dir: str,
    *,
    device: str = "cpu",
    max_frames: int = 0,
    threshold_px: float = 15.0,
    rally_config: RallyBoundaryConfig | None = None,
    skip_review_video: bool = False,
    use_player_filter: bool = True,
    player_distance: float = 200.0,
) -> dict:
    """Run the full labeling pipeline programmatically.

    Args:
        video_path: Path to the source video file.
        output_dir: Directory for output artifacts.
        device: PyTorch device string.
        max_frames: Max frames to process (0 = all).
        threshold_px: Consensus threshold in pixels.
        rally_config: Rally boundary detection configuration.
        skip_review_video: If True, skip generating review.mp4.
        use_player_filter: If True, apply player-proximity filtering.
        player_distance: Max pixel distance from player for ball detection.

    Returns:
        Dict with pipeline statistics and output paths.
    """
    t0 = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Read video metadata
    with VideoReader(video_path, max_frames=max_frames) as reader:
        fps = reader.fps
        width = reader.width
        height = reader.height
        frame_count = reader.frame_count
    video_filename = Path(video_path).name

    logger.info(
        "Pipeline start: %s — %d frames, %.1f fps, %dx%d",
        video_path, frame_count, fps, width, height,
    )

    # ── Step 1: Inference ──────────────────────────────────────────
    logger.info("Step 1/9: Running multi-model inference...")
    detectors = get_default_detectors()
    detections_path = str(out / "detections.json")
    raw_detections, inference_timing = run_inference(
        video_path,
        detectors,
        device=device,
        output_path=detections_path,
        max_frames=max_frames,
        return_timing=True,
    )
    logger.info("Step 1 complete: %d frames processed", len(raw_detections))

    # ── Step 2: Consensus ──────────────────────────────────────────
    logger.info("Step 2/9: Computing consensus (threshold=%.1fpx)...", threshold_px)
    consensus = compute_consensus(raw_detections, threshold_px=threshold_px)
    stats = compute_consensus_stats(consensus)

    consensus_path = str(out / "consensus.json")
    consensus_data = serialize_consensus(consensus)
    with open(consensus_path, "w") as f:
        json.dump(consensus_data, f, indent=2)
    logger.info(
        "Step 2 complete: %d consensus, %d uncertain, %d no_detection",
        stats["consensus_count"], stats["uncertain_count"], stats["no_detection_count"],
    )

    # ── Step 3: Player detection & proximity filtering ─────────────
    player_data: dict[int, list] = {}
    player_rejected = 0
    if use_player_filter:
        logger.info("Step 3/9: Running player detection (YOLOv8-nano)...")
        player_det = PlayerDetector()
        try:
            player_det.load(device)
            with VideoReader(video_path, max_frames=max_frames) as reader:
                for frame_idx, frame in enumerate(reader.iter_frames()):
                    boxes = player_det.detect(frame)
                    if boxes:
                        player_data[frame_idx] = boxes
        finally:
            player_det.unload()

        # Save player detections
        player_path = str(out / "players.json")
        player_serialized = {
            str(k): serialize_player_boxes(v) for k, v in player_data.items()
        }
        with open(player_path, "w") as f:
            json.dump(player_serialized, f, indent=2)

        # Apply proximity filter
        consensus, player_rejected = apply_player_proximity_filter(
            consensus, player_data, max_distance=player_distance,
        )
        stats_after = compute_consensus_stats(consensus)
        logger.info(
            "Step 3 complete: %d detections rejected by player filter, "
            "players found in %d/%d frames",
            player_rejected, len(player_data), len(raw_detections),
        )

        # Update consensus file with filtered data
        consensus_data = serialize_consensus(consensus)
        with open(consensus_path, "w") as f:
            json.dump(consensus_data, f, indent=2)
        stats = stats_after
    else:
        logger.info("Step 3/9: Player detection disabled (--no-player-filter)")

    # ── Step 4: CVAT XML export ────────────────────────────────────
    logger.info("Step 4/9: Exporting CVAT XML...")
    cvat_path = str(out / "annotations_cvat.xml")
    export_cvat_xml(
        consensus, cvat_path,
        video_filename=video_filename,
        width=width, height=height, fps=fps,
    )

    # ── Step 5: COCO JSON export ───────────────────────────────────
    logger.info("Step 5/9: Exporting COCO JSON...")
    coco_path = str(out / "annotations_coco.json")
    export_coco_json(
        consensus, coco_path,
        video_filename=video_filename,
        width=width, height=height, fps=fps,
    )

    # ── Step 6: Review video ───────────────────────────────────────
    review_path = str(out / "review.mp4")
    if skip_review_video:
        logger.info("Step 6/9: Skipping review video (--skip-review-video)")
        review_path = None
    else:
        logger.info("Step 6/9: Generating review overlay video...")
        generate_review_video(
            video_path, consensus, review_path,
            fps=fps, max_frames=max_frames,
            player_detections=player_data if use_player_filter else None,
        )

    # ── Step 7: Rally boundaries ───────────────────────────────────
    logger.info("Step 7/9: Detecting rally boundaries...")
    rally_config = rally_config or RallyBoundaryConfig()
    boundaries = detect_rally_boundaries(consensus, fps, config=rally_config)
    rallies = [b for b in boundaries if b.is_rally]

    rallies_path = str(out / "rallies.json")
    rallies_data = serialize_rally_boundaries(boundaries)
    with open(rallies_path, "w") as f:
        json.dump(rallies_data, f, indent=2)

    # ── Step 8: Benchmark report ──────────────────────────────────
    logger.info("Step 8/9: Generating benchmark report...")
    benchmark_path = str(out / "benchmark_report.md")
    video_info_dict = {
        "path": video_path,
        "frames": frame_count,
        "fps": fps,
        "resolution": f"{width}x{height}",
    }
    generate_benchmark_report(
        raw_detections, consensus, inference_timing,
        benchmark_path, video_info=video_info_dict,
        player_filter_stats={
            "enabled": use_player_filter,
            "rejected": player_rejected,
            "max_distance": player_distance,
            "frames_with_players": len(player_data),
        },
    )

    elapsed = time.perf_counter() - t0

    # ── Summary ────────────────────────────────────────────────────
    summary = {
        "video": video_path,
        "total_frames": stats["total_frames"],
        "consensus_count": stats["consensus_count"],
        "consensus_pct": stats["consensus_pct"],
        "uncertain_count": stats["uncertain_count"],
        "uncertain_pct": stats["uncertain_pct"],
        "no_detection_count": stats["no_detection_count"],
        "no_detection_pct": stats["no_detection_pct"],
        "player_filter_rejected": player_rejected,
        "rally_count": len(rallies),
        "total_segments": len(boundaries),
        "elapsed_sec": round(elapsed, 1),
        "outputs": {
            "detections": detections_path,
            "consensus": consensus_path,
            "cvat_xml": cvat_path,
            "coco_json": coco_path,
            "review_video": review_path,
            "rallies": rallies_path,
            "benchmark": benchmark_path,
        },
    }

    return summary


def _print_summary(summary: dict) -> None:
    """Print pipeline summary to stdout."""
    print("\n" + "=" * 60)
    print("LABELING PIPELINE — SUMMARY")
    print("=" * 60)
    print(f"  Video:            {summary['video']}")
    print(f"  Total frames:     {summary['total_frames']}")
    print(f"  Consensus:        {summary['consensus_count']} ({summary['consensus_pct']:.1f}%)")
    print(f"  Uncertain:        {summary['uncertain_count']} ({summary['uncertain_pct']:.1f}%)")
    print(f"  No detection:     {summary['no_detection_count']} ({summary['no_detection_pct']:.1f}%)")
    if summary.get('player_filter_rejected', 0) > 0:
        print(f"  Player-filtered:  {summary['player_filter_rejected']} detections rejected")
    print(f"  Rallies detected: {summary['rally_count']}")
    print(f"  Elapsed:          {summary['elapsed_sec']:.1f}s")
    print("-" * 60)
    print("  Outputs:")
    for name, path in summary["outputs"].items():
        if path:
            print(f"    {name:20s} → {path}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    # Validate input
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        return 1

    rally_config = RallyBoundaryConfig(
        window_size_sec=args.window_size,
        rally_threshold=args.rally_threshold,
        min_rally_duration_sec=args.min_rally,
        min_gap_duration_sec=args.min_gap,
    )

    try:
        summary = run_labeling_pipeline(
            str(video_path),
            args.output,
            device=args.device,
            max_frames=args.max_frames,
            threshold_px=args.threshold,
            rally_config=rally_config,
            skip_review_video=args.skip_review_video,
            use_player_filter=not args.no_player_filter,
            player_distance=args.player_distance,
        )
        _print_summary(summary)
        return 0
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        return 1
    except (ValueError, RuntimeError) as exc:
        logger.error("Pipeline error: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 130
    except Exception:
        logger.exception("Unexpected pipeline failure")
        return 1


if __name__ == "__main__":
    sys.exit(main())
