#!/usr/bin/env python3
"""Profile the Agrawal tennis court detection pipeline with per-stage timing.

Directly instruments run_agrawal_pipeline() internals to get fine-grained
per-stage timing without relying on stdout parsing.

Usage:
    python scripts/profile_pipeline.py                  # 20-frame sample
    python scripts/profile_pipeline.py --n 50            # 50-frame sample
    python scripts/profile_pipeline.py --full            # all test frames
    python scripts/profile_pipeline.py --no-composite    # skip diagnostic panel
    python scripts/profile_pipeline.py --no-shadow       # skip shadow removal

Outputs a ranked table of bottlenecks and per-frame breakdown.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.court_detect.agrawal import (
    TOPHAT_KERNEL_SIZE,
    agrawal_clahe_line_filter,
    agrawal_local_contrast_line_filter,
    agrawal_saturation_line_filter,
    classify_lines_courtside,
    compute_distance_transform,
    compute_metrics,
    compute_near_half_keypoints,
    compute_near_half_homography,
    crop_near_half,
    debug_per_line_chamfer,
    detect_lines_near_half,
    draw_pipeline_stages,
    estimate_net_y,
    extend_to_full_court,
    fallback_service_line_detection,
    filter_short_merged,
    identify_near_half_lines,
    merge_collinear_segments,
    REFERENCE_KPS_METERS,
    refine_homography_edge_align,
    reprojection_error,
    score_court_detection,
    validate_near_half_keypoints,
)
from src.features.court_detect.shadow_removal import ShadowRemover


# ── Timing helper ──────────────────────────────────────────────────
class StageTimer:
    """Accumulates per-stage timings for a single frame."""

    def __init__(self):
        self.stages: list[tuple[str, float]] = []
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        yield
        self.stages.append((name, time.perf_counter() - t))

    @property
    def total(self) -> float:
        return time.perf_counter() - self._t0

    def as_dict(self) -> dict[str, float]:
        d = {name: dur for name, dur in self.stages}
        d["TOTAL"] = self.total
        return d


# ── Instrumented pipeline ──────────────────────────────────────────
def run_profiled_pipeline(
    frame: np.ndarray,
    frame_idx: int,
    net_detector=None,
    shadow_remover: ShadowRemover | None = None,
    skip_composite: bool = False,
) -> dict[str, float]:
    """Run the Agrawal pipeline with per-stage timing.

    Returns dict mapping stage name -> seconds.
    """
    T = StageTimer()

    # Stage 1: Net detection & crop
    with T.stage("1_net_detect"):
        net_y, net_source, crop_margin = estimate_net_y(
            frame, net_detector=net_detector)
        near_half, y_offset = crop_near_half(frame, net_y, margin=crop_margin)

    # Stage 1.5: Shadow removal
    with T.stage("1.5_shadow_removal"):
        if shadow_remover is not None:
            should_apply, _, _ = ShadowRemover.should_remove_shadows(near_half)
            if should_apply:
                near_half = shadow_remover.remove_shadows(near_half)

    # Stage 2+3: Saturation filter (always computed for court_color)
    with T.stage("2_saturation_filter"):
        line_mask, court_color_mask, hsv_full, eff_tophat = (
            agrawal_saturation_line_filter(near_half))
        chromatic_px = hsv_full[court_color_mask > 0]
        if len(chromatic_px) > 0:
            court_color = np.array([
                int(np.median(chromatic_px[:, 0])),
                int(np.median(chromatic_px[:, 1])),
                int(np.median(chromatic_px[:, 2])),
            ], dtype=np.uint8)
        else:
            court_color = np.array([0, 0, 0], dtype=np.uint8)

    # Stage 4: Initial Hough (on saturation mask — used for baseline only)
    with T.stage("4_initial_hough"):
        raw_lines, baseline_extent, spatial_mask, bl_angle = (
            detect_lines_near_half(line_mask))
        horizontal, vertical = classify_lines_courtside(
            raw_lines, baseline_angle=bl_angle)
        horizontal = filter_short_merged(merge_collinear_segments(horizontal))
        vertical = filter_short_merged(merge_collinear_segments(vertical))

    nh_h, nh_w = near_half.shape[:2]

    # Stage 4b-candidate helper (same as _run_candidate in agrawal.py)
    def _run_candidate(cand_line_mask):
        c_raw, c_bl_ext, c_sm, c_bl_angle = detect_lines_near_half(cand_line_mask)
        if c_bl_ext is not None:
            cand_line_mask = cv2.bitwise_and(cand_line_mask, c_sm)
            c_raw, _, _, c_bl_angle2 = detect_lines_near_half(cand_line_mask)
            if c_bl_angle2 is not None:
                c_bl_angle = c_bl_angle2
        c_h, c_v = classify_lines_courtside(c_raw, baseline_angle=c_bl_angle)
        c_h = filter_short_merged(merge_collinear_segments(c_h))
        c_v_merged = merge_collinear_segments(c_v)
        c_v = filter_short_merged(c_v_merged)
        c_identified = identify_near_half_lines(
            c_h, c_v, nh_h, nh_w, pre_filter_verticals=c_v_merged)
        c_near_kps = compute_near_half_keypoints(c_identified, y_offset, c_v, nh_h)
        c_near_kps = validate_near_half_keypoints(c_near_kps)
        c_H, c_near_kps = compute_near_half_homography(c_near_kps)
        c_score = (score_court_detection(near_half, c_H, y_offset, cand_line_mask)
                   if c_H is not None else 0)
        c_err = reprojection_error(c_near_kps, c_H) if c_H is not None else float('inf')
        return c_H, c_near_kps, c_identified, c_h, c_v, c_raw, cand_line_mask, c_score, c_err, c_v_merged

    # Stage 4b: Candidate 1 — LocalContrast filter
    with T.stage("4b_localcontrast_filter"):
        lc_line_mask, lc_court_mask, _, _, _ = (
            agrawal_local_contrast_line_filter(near_half))

    with T.stage("4b_localcontrast_hough_score"):
        c1 = _run_candidate(lc_line_mask)

    # Stage 4b: Candidate 2 — CLAHE filter
    with T.stage("4b_clahe_filter"):
        cl_line_mask, cl_court_mask, _, _ = agrawal_clahe_line_filter(near_half)

    with T.stage("4b_clahe_hough_score"):
        c2 = _run_candidate(cl_line_mask)

    # Pick best candidate
    candidates = [
        ("local_contrast", c1, lc_court_mask, court_color),
        ("clahe", c2, cl_court_mask, court_color),
    ]
    valid = [(n, c, cm, cc) for n, c, cm, cc in candidates if c[7] > 0]
    if valid:
        best_name, best_cand, best_cm, best_cc = max(valid, key=lambda x: x[1][7])
        _, best_near_kps, identified, horizontal, vertical, raw_lines, line_mask, _, _, pre_v = best_cand
        court_color_mask = best_cm
        winning_mode = best_name
    else:
        best_cand = max(candidates, key=lambda x: len(x[1][1]))[1]
        horizontal, vertical = best_cand[3], best_cand[4]
        pre_v = best_cand[9]
        line_mask = best_cand[6]
        identified = best_cand[2]
        winning_mode = "local_contrast"

    # Stage 5b: Fallback service line
    with T.stage("5b_fallback_service"):
        if (identified.get("near_service") is None
                and identified.get("near_baseline") is not None):
            fb = fallback_service_line_detection(line_mask, identified, nh_h, nh_w)
            if fb is not None:
                identified["near_service"] = fb

    # Stage 6: Keypoints
    with T.stage("6_keypoints"):
        near_kps = compute_near_half_keypoints(identified, y_offset, vertical, nh_h)
        near_kps = validate_near_half_keypoints(near_kps)

    # Stage 7: Homography + edge-align + extend
    with T.stage("7_homography"):
        H_near, near_kps = compute_near_half_homography(near_kps)
        H_full = None
        all_kps = list(near_kps)

    with T.stage("7_edge_align"):
        if H_near is not None:
            H_near = refine_homography_edge_align(
                H_near, line_mask, near_kps, y_offset,
                lambda_reg=0.3, max_iter=150)
            try:
                H_inv = np.linalg.inv(H_near)
                meter_pts = np.array(
                    [REFERENCE_KPS_METERS[idx] for idx, _ in near_kps],
                    dtype=np.float64).reshape(-1, 1, 2)
                updated_px = cv2.perspectiveTransform(meter_pts, H_inv).reshape(-1, 2)
                near_kps = [
                    (idx, (float(updated_px[i, 0]), float(updated_px[i, 1])))
                    for i, (idx, _) in enumerate(near_kps)
                ]
            except (np.linalg.LinAlgError, cv2.error):
                pass

    with T.stage("7_extend_full"):
        if H_near is not None:
            H_full, all_kps = extend_to_full_court(H_near, near_kps)

    # Stage 8: Metrics
    with T.stage("8_metrics"):
        metrics = compute_metrics(near_kps, all_kps, H_full)

    # Composite panel (diagnostic only — can be skipped)
    with T.stage("9_composite_panel"):
        if not skip_composite:
            draw_pipeline_stages(
                original_frame=frame,
                net_y=net_y,
                near_half=near_half,
                court_color_hsv=court_color,
                court_color_mask=court_color_mask,
                line_mask=line_mask,
                horizontal=horizontal,
                vertical=vertical,
                identified_lines=identified,
                near_kps=near_kps,
                all_kps=all_kps,
                H_full=H_full,
                y_offset=y_offset,
                court_mode=winning_mode,
                pre_filter_verticals=pre_v,
            )

    return T.as_dict()


# ── Frame sampling ─────────────────────────────────────────────────
def collect_test_frames(
    frame_dir: Path, n: int | None = None, seed: int = 42,
) -> list[tuple[str, Path]]:
    """Collect test frames, optionally sampling n from diverse videos."""
    videos = sorted(
        d for d in frame_dir.iterdir()
        if d.is_dir() and d.name.startswith("vid")
    )
    all_frames: list[tuple[str, Path]] = []
    for vdir in videos:
        imgs = sorted(vdir.glob("*.jpg"))
        for p in imgs:
            all_frames.append((f"{vdir.name}/{p.stem}", p))

    if n is not None and n < len(all_frames):
        # Stratified sample: pick proportionally from each video
        rng = random.Random(seed)
        by_video: dict[str, list[tuple[str, Path]]] = defaultdict(list)
        for name, path in all_frames:
            vid = name.split("/")[0]
            by_video[vid].append((name, path))

        sampled: list[tuple[str, Path]] = []
        per_vid = max(1, n // len(by_video))
        for vid in sorted(by_video):
            frames = by_video[vid]
            k = min(per_vid, len(frames))
            sampled.extend(rng.sample(frames, k))

        # Fill remainder
        remaining = [f for f in all_frames if f not in sampled]
        rng.shuffle(remaining)
        while len(sampled) < n and remaining:
            sampled.append(remaining.pop())

        return sorted(sampled[:n])

    return all_frames


# ── Reporting ──────────────────────────────────────────────────────
def print_report(
    results: list[tuple[str, dict[str, float]]],
) -> None:
    """Print ranked bottleneck table and per-frame breakdown."""
    n = len(results)
    if n == 0:
        print("No results to report.")
        return

    # Aggregate across frames
    stage_totals: dict[str, float] = defaultdict(float)
    stage_counts: dict[str, int] = defaultdict(int)
    for _, timings in results:
        for stage, dur in timings.items():
            stage_totals[stage] += dur
            stage_counts[stage] += 1

    grand_total = stage_totals.pop("TOTAL", 0)
    avg_total = grand_total / n

    # ── Ranked bottleneck table ──
    print("\n" + "=" * 90)
    print(f"PROFILING REPORT — {n} frames, avg {avg_total:.3f}s/frame "
          f"(total {grand_total:.1f}s)")
    print("=" * 90)
    print(f"{'Stage':<35} {'Total':>8} {'Avg':>9} {'Max':>9} {'%Total':>8}")
    print("-" * 90)

    # Compute max per stage
    stage_max: dict[str, float] = defaultdict(float)
    for _, timings in results:
        for stage, dur in timings.items():
            if stage != "TOTAL":
                stage_max[stage] = max(stage_max[stage], dur)

    ranked = sorted(stage_totals.keys(), key=lambda s: stage_totals[s], reverse=True)
    for s in ranked:
        total_s = stage_totals[s]
        count = stage_counts[s]
        avg = total_s / count if count > 0 else 0
        mx = stage_max.get(s, 0)
        pct = (total_s / grand_total * 100) if grand_total > 0 else 0
        bar = "█" * int(pct / 2.5)
        print(f"{s:<35} {total_s:>7.2f}s {avg:>8.4f}s {mx:>8.4f}s {pct:>6.1f}%  {bar}")

    accounted = sum(stage_totals.values())
    unaccounted = grand_total - accounted
    if abs(unaccounted) > 0.01:
        pct = unaccounted / grand_total * 100
        print(f"{'(unaccounted overhead)':<35} {unaccounted:>7.2f}s "
              f"{unaccounted/n:>8.4f}s {'':>8}  {pct:>6.1f}%")

    print("-" * 90)
    print(f"{'TOTAL':<35} {grand_total:>7.2f}s {avg_total:>8.4f}s")

    # ── Top 10 slowest frames ──
    print("\n" + "=" * 90)
    print("Top 10 Slowest Frames")
    print("=" * 90)
    sorted_results = sorted(results, key=lambda r: r[1].get("TOTAL", 0), reverse=True)
    for name, timings in sorted_results[:10]:
        total = timings.get("TOTAL", 0)
        # Find biggest stage
        stages_only = {k: v for k, v in timings.items() if k != "TOTAL"}
        if stages_only:
            biggest = max(stages_only, key=stages_only.get)
            biggest_val = stages_only[biggest]
        else:
            biggest, biggest_val = "?", 0
        print(f"  {name:<50} {total:.3f}s  ({biggest}: {biggest_val:.3f}s)")

    # ── Quick optimization estimate ──
    print("\n" + "=" * 90)
    print("Optimization Opportunities (estimated savings)")
    print("=" * 90)

    # Composite panel savings
    composite_total = stage_totals.get("9_composite_panel", 0)
    if composite_total > 0:
        print(f"  Skip composite panel in eval:  save ~{composite_total/n:.3f}s/frame "
              f"({composite_total/grand_total*100:.1f}% of total)")

    # Redundant filter (3 filters when 1 would suffice on easy frames)
    filter_total = (stage_totals.get("2_saturation_filter", 0)
                    + stage_totals.get("4b_localcontrast_filter", 0)
                    + stage_totals.get("4b_clahe_filter", 0))
    print(f"  3 line filters total:          {filter_total/n:.3f}s/frame "
          f"({filter_total/grand_total*100:.1f}%) — could skip CLAHE if LC wins")

    # Hough passes total
    hough_total = (stage_totals.get("4_initial_hough", 0)
                   + stage_totals.get("4b_localcontrast_hough_score", 0)
                   + stage_totals.get("4b_clahe_hough_score", 0))
    print(f"  Hough+scoring passes total:    {hough_total/n:.3f}s/frame "
          f"({hough_total/grand_total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Profile Agrawal pipeline")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of frames to sample (default: 20)")
    parser.add_argument("--full", action="store_true",
                        help="Run on all test frames")
    parser.add_argument("--no-composite", action="store_true",
                        help="Skip composite panel generation")
    parser.add_argument("--no-shadow", action="store_true",
                        help="Skip shadow removal")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-frame pipeline stdout")
    args = parser.parse_args()

    frame_dir = PROJECT_ROOT / "tests" / "court_detection_frames"
    if not frame_dir.is_dir():
        print(f"ERROR: Frame directory not found: {frame_dir}")
        sys.exit(1)

    n_sample = None if args.full else args.n
    frames = collect_test_frames(frame_dir, n=n_sample)
    print(f"Profiling {len(frames)} frames "
          f"(composite={'OFF' if args.no_composite else 'ON'}, "
          f"shadow={'OFF' if args.no_shadow else 'ON'})")

    # Load shadow remover once (amortize model load)
    shadow_remover = None
    if not args.no_shadow:
        try:
            shadow_remover = ShadowRemover()
        except Exception as e:
            print(f"  Shadow remover unavailable: {e}")

    results: list[tuple[str, dict[str, float]]] = []
    import io
    from contextlib import redirect_stdout as _redirect

    for i, (name, path) in enumerate(frames):
        img = cv2.imread(str(path))
        if img is None:
            print(f"  SKIP {name} (unreadable)")
            continue

        # Suppress pipeline's own print statements
        if args.quiet:
            buf = io.StringIO()
            with _redirect(buf):
                timings = run_profiled_pipeline(
                    img, i,
                    shadow_remover=shadow_remover,
                    skip_composite=args.no_composite,
                )
        else:
            timings = run_profiled_pipeline(
                img, i,
                shadow_remover=shadow_remover,
                skip_composite=args.no_composite,
            )

        results.append((name, timings))
        total = timings.get("TOTAL", 0)
        print(f"  [{i+1:3d}/{len(frames)}] {name:<50} {total:.3f}s")

    print_report(results)


if __name__ == "__main__":
    main()
