# Checkpoint — Court Detection Pipeline (Agrawal-inspired)

**Last updated:** 2026-04-07

## Current Status

**27/32 valid homographies** on test set (`tests/court_detection_frames/`, 32 frames from 10 amateur courtside videos).

### 5 Remaining Failures
- `vid01_recreational_1set_t2_half` — noisy mask, insufficient keypoints
- `vid04_courtside_singles_t3_twothird` — noisy blue court, Hough overwhelmed
- `vid05_recreational_doubles_t3_twothird` — no lines detected
- `vid09_match_3D3t_t1_third` — insufficient keypoints
- `vid09_match_3D3t_t3_twothird` — no lines detected

## Pipeline Architecture (`scripts/test_agrawal_court.py`, ~2325 lines)

8-stage classical CV pipeline for courtside amateur tennis footage:

1. **Net detection & crop** — YOLO net detector → crop below net (near-half court)
2. **Line mask generation** — 3 competing filters: saturation (pre-pass), LocalContrast, CLAHE
3. **Court color detection** — HSV-based court surface color identification
4. **Hough line detection** — Two-pass HoughLinesP with baseline-anchored spatial masking, multi-candidate scoring across filter variants
5. **Line identification** — Classify detected lines as baseline, service line, sidelines (4 doubles + singles)
   - **5b. Fallback service line** — Band-masked Hough with aggressive params when service line missing
6. **Keypoint computation** — Line intersections → court keypoints
7. **Homography** — RANSAC + edge-align refinement + extend to full court
   - Safety revert if fallback service line breaks homography
8. **Metrics** — Reprojection error, condition number, keypoint spread

### Key Parameters
- HOUGH_THRESHOLD=30, HOUGH_MIN_LENGTH=30, HOUGH_MAX_GAP=40
- MIN_MERGED_LENGTH=200, MIN_BASELINE_FRAC=0.70
- Service line y-band: [0.50, 0.78] of baseline y
- Fallback Hough: threshold=15, minLength=20, maxGap=60, angle±8°

## What Was Done (chronological)

### Weeks 0-1 (Apr 2-4)
- Built initial Agrawal pipeline (color-agnostic saturation filter)
- Added ShadowFormer shadow removal with auto-gating
- Integrated YOLO net detector for reliable net-y cropping
- Two-pass Hough with baseline-anchored spatial masking
- Service line geometric constraints (y-band, width≤baseline, angle)
- Edge-alignment homography refinement optimizer
- Oriented line kernels (6 directions, 0°-150°)

### Week 1 continued (Apr 5-6)
- LocalContrast + CLAHE as competing line filter candidates
- Multi-candidate scoring algorithm (best score wins)
- Tight net crop (0.3× net height margin)
- Multi-color court surface support
- Nighttime court detection (adaptive CLAHE S-gate)
- Merge tolerance tuning (angle_tol=8°, dist_tol=15px)
- Baseline fragment merging for occluded baselines
- Sideline chimera merge fixes
- Homography chamfer-based outlier detection (experimental)

### Week 1 final (Apr 7)
- Removed barrel distortion fallback (never saved any frame)
- Removed CC filtering (regressed 27→24)
- Evaluated density gating (FAIL/PASS ranges overlap, unusable)
- **Added fallback service line detection** — band-masked Hough using baseline geometry to predict service line y-position
- Stage 7 safety revert when fallback breaks homography
- Restored pipeline from git blob after accidental refactor by another agent

## Architecture Decisions

- **No CNN keypoint fallback** — Removed; the 27/32 rate is achieved purely through classical CV improvements
- **No barrel distortion correction** — Never rescued any frame; removed
- **Band-masked Hough for service line** — Uses baseline y-position (empirically validated: y_ratio 0.65-0.73, band [0.50, 0.78]) and angle (±8°) to constrain search
- **Multi-candidate filter competition** — Run LocalContrast and CLAHE, score each, pick best. Saturation filter as pre-pass only (too conservative to win)
- **IMAGE_DIR mode** — Load test frames from directory instead of video, Kalman disabled for independent frames

## Known Issues

- Fallback service line not yet tested for rescuing the 5 failures
- 3 brainstormed but unimplemented approaches for remaining failures:
  1. Adaptive Hough threshold (scale with mask density) — ~5 lines
  2. Line support verification (mask coverage scoring) — ~20 lines
  3. Directional morphological opening — ~15 lines

## Next Steps

1. Test fallback service line on the 5 failures — vid04_t3 was the primary target
2. Implement adaptive Hough threshold for noisy masks
3. Consider line support verification for post-Hough filtering
4. Commit and stabilize before moving to CNN near-half keypoint detector (Phase 2)
