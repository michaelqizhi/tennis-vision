"""Full analysis pipeline: court detection → ball tracking → rally detection → serve analysis → visualizations.

Orchestrates all feature modules into a single end-to-end workflow.
Frames are streamed (not loaded into memory) to avoid excessive RAM usage.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.api.schemas import (
    AnalysisResults,
    BallPositionOut,
    PipelineStep,
    RallyOut,
    RallyStatsOut,
    ServeStatsOut,
    VisualizationPaths,
)
from src.api.tasks import Job
from src.config import load_config
from src.features.ball_tracking import BallDetection, BallTracker
from src.features.court_detect import (
    CourtDetector,
    transform_points_to_court,
)
from src.features.auto_clip import detect_rallies, Rally
from src.features.rally_stats import (
    count_all_rally_shots,
    compute_rally_stats,
    RallyShots,
)
from src.features.serve_analysis import (
    detect_serves,
    reclassify_serves,
    detect_double_faults,
    compute_serve_stats,
)
from src.features.visualization import render_shot_heatmap, render_serve_placement
from src.video.reader import VideoReader

logger = logging.getLogger(__name__)


def _ensure_output_dir(job: Job) -> Path:
    """Create and return the output directory for a job."""
    cfg = load_config()
    output_dir = cfg.output_dir / job.job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_pipeline(job: Job) -> None:
    """Run the full analysis pipeline for a video processing job.

    Frames are streamed through the pipeline to keep memory usage low.
    Court detection samples ~20 frames; ball tracking uses a 3-frame
    sliding window. Peak memory is O(1) relative to video length.

    Updates job.results with an AnalysisResults dict on success.
    Raises on failure (caught by task runner).
    """
    config = load_config()
    output_dir = _ensure_output_dir(job)

    # ── Read video metadata (no frames loaded) ─────────────────────
    logger.info("[%s] Opening video: %s", job.job_id, job.video_path)
    with VideoReader(job.video_path) as reader:
        fps = reader.fps
        frame_width = reader.width
        frame_height = reader.height
        total_frames = reader.frame_count

    if total_frames <= 0:
        raise ValueError("Video contains no readable frames")

    logger.info("[%s] Video: %d frames (est.), %.1f fps, %dx%d",
                job.job_id, total_frames, fps, frame_width, frame_height)

    # ── Step 1: Court detection (samples ~20 frames) ───────────────
    job.update_progress(PipelineStep.COURT_DETECTION, "Detecting court keypoints...")
    court_detector = CourtDetector(config)

    with VideoReader(job.video_path) as reader:
        sampled_frames = reader.read_sampled_frames(max_samples=20)

    homography = court_detector.get_stable_homography(sampled_frames) if sampled_frames else None
    del sampled_frames  # free memory

    if homography is None:
        logger.warning("[%s] Court detection failed — spatial features will be limited", job.job_id)
        job.complete_step(PipelineStep.COURT_DETECTION, "Court not detected — using pixel coordinates only")
    else:
        job.complete_step(PipelineStep.COURT_DETECTION, "Court detected, homography computed")
        logger.info("[%s] Court homography computed", job.job_id)

    # ── Step 2: Ball tracking (streaming, 3-frame window) ──────────
    job.update_progress(PipelineStep.BALL_TRACKING, "Running TrackNet inference...")
    tracker = BallTracker(config)

    def _ball_tracking_progress(current: int, total: int) -> None:
        """Report granular progress during ball tracking."""
        step_progress = current / total if total > 0 else 0.0
        # Court detection is done (1/6), ball tracking fills the next 1/6
        all_steps = len(list(PipelineStep))
        job.progress = (1 + step_progress) / all_steps
        if current % max(1, total // 10) == 0:
            job.set_step(
                PipelineStep.BALL_TRACKING,
                job.status,
                f"Processing frame {current}/{total}...",
            )

    with VideoReader(job.video_path) as reader:
        detections: list[BallDetection] = tracker.detect_streaming(
            reader.iter_frames(),
            total_frames=total_frames,
            interpolate=True,
            progress_callback=_ball_tracking_progress,
        )

    if not detections:
        raise ValueError("Video contains no readable frames")

    # Use actual detection count (may differ from container metadata)
    total_frames = len(detections)

    job.complete_step(PipelineStep.BALL_TRACKING,
                      f"Detected ball in {sum(1 for d in detections if d.detected)}/{len(detections)} frames")
    logger.info("[%s] Ball tracking complete: %d detections",
                job.job_id, sum(1 for d in detections if d.detected))

    # ── Step 4: Rally detection ─────────────────────────────────────
    job.update_progress(PipelineStep.RALLY_DETECTION, "Detecting rally boundaries...")
    rallies: list[Rally] = detect_rallies(detections, fps)
    job.complete_step(PipelineStep.RALLY_DETECTION, f"Found {len(rallies)} rallies")
    logger.info("[%s] Rally detection: %d rallies", job.job_id, len(rallies))

    # ── Step 5: Rally stats ─────────────────────────────────────────
    job.update_progress(PipelineStep.RALLY_STATS, "Counting shots per rally...")
    rally_shots: list[RallyShots] = count_all_rally_shots(
        detections, rallies, homography=homography, frame_height=frame_height
    )
    rally_stats_obj = compute_rally_stats(rallies, rally_shots)
    job.complete_step(PipelineStep.RALLY_STATS, f"{rally_stats_obj.total_shots} total shots across {rally_stats_obj.rally_count} rallies")
    logger.info("[%s] Rally stats: %d shots in %d rallies",
                job.job_id, rally_stats_obj.total_shots, rally_stats_obj.rally_count)

    # ── Step 6: Serve analysis ──────────────────────────────────────
    job.update_progress(PipelineStep.SERVE_ANALYSIS, "Analyzing serves...")
    serves = []
    double_faults = []
    serve_stats_dict: dict = {}
    if homography is not None and len(rallies) > 0:
        serves = detect_serves(detections, rallies, homography, fps)
        serves = reclassify_serves(serves)
        double_faults = detect_double_faults(serves)
        serve_stats_obj = compute_serve_stats(serves, double_faults)
        serve_stats_dict = serve_stats_obj.to_dict()
        job.complete_step(PipelineStep.SERVE_ANALYSIS,
                          f"{serve_stats_obj.total_serves} serves, {serve_stats_obj.double_faults} double faults")
    else:
        job.complete_step(PipelineStep.SERVE_ANALYSIS,
                          "Skipped — no homography or no rallies detected")

    # ── Step 7: Visualization ───────────────────────────────────────
    job.update_progress(PipelineStep.VISUALIZATION, "Generating heatmaps...")
    viz_paths = VisualizationPaths()

    # Collect court-coordinate ball positions for heatmaps
    court_positions: list[tuple[float, float]] = []
    if homography is not None:
        pixel_positions = [
            (d.x, d.y) for d in detections if d.detected and d.x is not None and d.y is not None
        ]
        if pixel_positions:
            court_positions = transform_points_to_court(pixel_positions, homography)

    if len(court_positions) >= 3:
        try:
            heatmap_path = str(output_dir / "shot_heatmap.png")
            render_shot_heatmap(court_positions, heatmap_path, title="Shot Placement Heatmap")
            viz_paths.shot_heatmap = f"/output/{job.job_id}/shot_heatmap.png"
        except Exception:
            logger.exception("[%s] Failed to render shot heatmap", job.job_id)

    # Serve placement heatmaps
    if serve_stats_dict:
        all_positions = serve_stats_dict.get("all_serve_positions", [])
        first_positions = serve_stats_dict.get("first_serve_positions", [])
        second_positions = serve_stats_dict.get("second_serve_positions", [])

        for label, positions, attr in [
            ("All Serves", all_positions, "serve_heatmap_all"),
            ("1st Serve", first_positions, "serve_heatmap_first"),
            ("2nd Serve", second_positions, "serve_heatmap_second"),
        ]:
            if len(positions) >= 3:
                try:
                    filename = f"serve_heatmap_{attr.split('_')[-1]}.png"
                    path = str(output_dir / filename)
                    render_serve_placement(
                        [tuple(p) for p in positions], path, title=f"{label} Placement"
                    )
                    setattr(viz_paths, attr, f"/output/{job.job_id}/{filename}")
                except Exception:
                    logger.exception("[%s] Failed to render %s heatmap", job.job_id, label)

    job.complete_step(PipelineStep.VISUALIZATION, "Heatmaps generated")

    # ── Build results ───────────────────────────────────────────────
    ball_positions_out: list[BallPositionOut] = []
    for det in detections:
        bp = BallPositionOut(
            frame_number=det.frame_number,
            x=det.x,
            y=det.y,
            confidence=det.confidence,
            interpolated=det.interpolated,
        )
        ball_positions_out.append(bp)

    # Add court coordinates if homography available
    if homography is not None:
        for bp in ball_positions_out:
            if bp.x is not None and bp.y is not None:
                try:
                    court_pts = transform_points_to_court([(bp.x, bp.y)], homography)
                    if court_pts:
                        bp.court_x, bp.court_y = court_pts[0]
                except Exception:
                    pass

    rallies_out = [
        RallyOut(
            rally_id=r.rally_id,
            start_frame=r.start_frame,
            end_frame=r.end_frame,
            start_time=r.start_time,
            end_time=r.end_time,
            duration=r.duration,
            shot_count=rs.shot_count,
        )
        for r, rs in zip(rallies, rally_shots)
    ]

    rally_stats_out = RallyStatsOut(
        rally_count=rally_stats_obj.rally_count,
        total_shots=rally_stats_obj.total_shots,
        avg_rally_length=rally_stats_obj.avg_rally_length,
        avg_shots_per_rally=rally_stats_obj.avg_shots_per_rally,
        longest_rally_duration=rally_stats_obj.longest_rally_duration,
        longest_rally_shots=rally_stats_obj.longest_rally_shots,
        shortest_rally_duration=rally_stats_obj.shortest_rally_duration,
        shortest_rally_shots=rally_stats_obj.shortest_rally_shots,
        median_rally_duration=rally_stats_obj.median_rally_duration,
        median_shots_per_rally=rally_stats_obj.median_shots_per_rally,
        shots_per_rally=rally_stats_obj.shots_per_rally,
        shot_count_distribution=rally_stats_obj.shot_count_distribution,
    )

    serve_stats_out = ServeStatsOut(**{
        k: v for k, v in serve_stats_dict.items()
        if k in ServeStatsOut.model_fields
    }) if serve_stats_dict else ServeStatsOut()

    results = AnalysisResults(
        job_id=job.job_id,
        video_filename=job.filename,
        fps=fps,
        total_frames=total_frames,
        ball_positions=ball_positions_out,
        rallies=rallies_out,
        rally_stats=rally_stats_out,
        serve_stats=serve_stats_out,
        visualizations=viz_paths,
    )

    job.results = results.model_dump()

    # ── Cleanup: remove uploaded temp file ─────────────────────────
    try:
        video_path = Path(job.video_path)
        if video_path.exists():
            video_path.unlink()
            logger.info("[%s] Cleaned up temp file: %s", job.job_id, job.video_path)
    except Exception:
        logger.warning("[%s] Failed to clean up temp file: %s", job.job_id, job.video_path)

    logger.info("[%s] Pipeline complete", job.job_id)
