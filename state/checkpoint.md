# Checkpoint — Sprint 12 (Fix Round)

## Completed
All 10 issues from Sprint 11 feedback fixed:

### Bug Fixes
1. **pipeline.py max_samples override removed (Issue #1)** — Removed `max_samples=10` from `court_detector.get_stable_homography(frames)` call in pipeline.py. Now uses the method default of 20, matching the Sprint 9 intent.
2. **Court detector passes min_radius from config (Issue #2)** — Added `min_radius=cfg.court_detection.min_radius` to `_postprocess_heatmap()` call in detector.py. Config value of 8 is now used instead of the function default of 10.
3. **Court detection improved (Issue #3)** — Fixes #1 and #2 were root causes. With correct max_samples=20 and min_radius=8, court detection has a much better chance of finding keypoints.
4. **Frontend CourtHeatmap coordinate transform fixed (Issue #4)** — Added transform `cx = pos.court_x! + COURT_WIDTH / 2`, `cy = pos.court_y! + COURT_LENGTH / 2` to convert center-origin backend coords to SVG top-left origin. Updated bounds check and density calculations to use transformed coords.
5. **Rally detection improved (Issue #5)** — Lowered `min_gap_seconds` from 2.0 to 1.5. Added position-reset detection: large ball position jumps (>400px within 0.5s) are treated as rally boundaries, splitting rallies even without detection gaps.
6. **`_interpolate_subtrack` handles all-None input (Issue #6)** — Added early return for all-NaN case: returns original coords instead of NaN values.
7. **`read_all()` dead code removed (Issue #7)** — Removed unused `read_all()` method from `VideoReader`.
8. **`/jobs` route added (Issue #8)** — Created `src/api/routes/jobs.py` with `GET /jobs` endpoint. Returns all jobs newest-first using `list_jobs()`. Registered in main.py.
9. **`BallPositionOut` includes `interpolated` field (Issue #9)** — Added `interpolated: bool = False` to `BallPositionOut` schema. Updated pipeline.py results builder to set `interpolated=det.interpolated`. Updated frontend `BallPosition` TypeScript interface.
10. **Stale output directories cleaned (Issue #10)** — Removed all 22 empty directories from `output/`.

## Architecture Decisions
- **Position-reset rally splitting** — Supplements gap-based detection. When consecutive raw detections show a position jump >400px within 0.5s, the frame is treated as a rally boundary. This catches point transitions where ball detection continues (e.g., player walking with ball).
- **Coordinate transform in frontend** — Backend uses center-origin court coordinates (x ∈ [-5.485, 5.485], y ∈ [-11.885, 11.885]). Frontend SVG uses top-left origin (0,0 to 10.97, 23.77). Transform applied at render time.

## Known Issues
- **Court detection may still fail on some camera angles** — Threshold/parameter changes improve but don't guarantee success on all real footage.
- **Frames still materialized in memory** — `list(reader.iter_frames())` loads all frames into a list.
- **CORS wildcard** — Still present for dev flexibility.
- **Thread safety of Job mutations** — CPython GIL prevents crashes but it's a data race.

## File Changes
- Modified: `src/api/pipeline.py` — removed max_samples override, added interpolated field to results
- Modified: `src/features/court_detect/detector.py` — pass min_radius from config
- Modified: `src/features/ball_tracking/detector.py` — handle all-None interpolation
- Modified: `src/features/auto_clip/rally_detector.py` — lowered gap threshold, added position-reset detection
- Modified: `src/video/reader.py` — removed dead `read_all()` method
- Modified: `src/api/schemas.py` — added `interpolated` field to BallPositionOut
- Modified: `src/api/main.py` — registered /jobs route
- Created: `src/api/routes/jobs.py` — GET /jobs endpoint
- Modified: `frontend/src/components/CourtHeatmap.tsx` — fixed coordinate transform
- Modified: `frontend/src/lib/api.ts` — added `interpolated` to BallPosition interface
- Cleaned: `output/` — removed 22 stale empty directories

## Test Results
- 119 tests pass (4.81s)
- All module imports clean
- API starts successfully with 5 routes (health, upload, status, results, jobs)
- NaN interpolation fix verified
- /jobs route verified
