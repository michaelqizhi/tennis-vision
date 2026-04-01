# Checkpoint — Week 0, Sprint 5: YOLOv8-nano Player Detection & Bug Fixes

## Completed

### Bug Fixes from Sprint 4 Feedback

1. **Race condition fix** (Critical #4) — All API route handlers (`status.py`, `results.py`, `jobs.py`) now use `job.snapshot()` for thread-safe reads instead of accessing job fields directly. Prevents inconsistent state (e.g., `status=processing` but `progress=1.0`).

2. **Court detection weighted centroid** (Critical #1) — Replaced HoughCircles with `cv2.moments` weighted centroid as the primary keypoint extraction method in `_postprocess_heatmap()`. HoughCircles is now a fallback only. This fixes the confidence=-1.0 issue where HoughCircles couldn't find circles in courtside footage heatmaps that are often elliptical, not circular. Same fix applied to ball tracking `postprocess()`.

3. **Stationary detection removal** (Critical #2) — Added `_remove_stationary()` method to `BallTracker`. Detections that cluster within 15px for >10 consecutive frames are removed as false positives (player positions, net posts). This directly addresses the heatmap showing two dense clusters at player positions instead of ball trajectories.

4. **Shot count sanity cap** (Critical #3) — Both `count_shots_pixel()` and `count_shots_court()` now cap shot count at 1 shot/second × rally duration. A 60-second rally maxes out at 60 shots instead of 71+.

5. **Serve stats fallback wiring** (Non-critical #7) — When no homography, the pipeline now sets `total_serves = len(rallies)` and `first_serves = len(rallies)` in the `ServeStatsOut` response data, not just in the step status message.

6. **Frontend lint fix** (Non-critical #5) — Changed `npm run lint` from `next lint --dir src` to `next lint`. The `--dir` flag is not valid in Next.js 16.

7. **Root endpoint** (Non-critical #8) — Added `GET /` that redirects to `/docs`. No longer returns 404.

8. **Pipeline type annotations** — Fixed `_run_pipeline_steps` config parameter from `object` to `Config`. Fixed `_cleanup_orphaned_uploads` parameter from `object` to `Config`.

9. **Dead code removal** — Removed unused `_VIDEO_MAGIC` dict from `upload.py`. Removed dead `rally_idx`/`burst_idx` locals from `serve_detector.py`.

10. **Dead import cleanup** — Removed unused `logging`/`logger` from `consensus.py`, `rally_boundaries.py`, `models.py`. Removed unused `cv2` from `inference_runner.py`. Removed unused `Optional` from `export.py`. Removed unused `os` from `tracknetv4_detector.py`.

11. **Step numbering fix** — Pipeline step comments now count sequentially (Steps 1-6) instead of skipping Step 3.

12. **Streaming frames in inference runner** — Replaced RAM-hogging `list(reader.iter_frames())` with disk-cached streaming using pickle temp file. A 10-min 1080p video no longer requires ~55GB RAM.

### Sprint 5 Deliverables

13. **YOLOv8-nano player detector** (`src/labeling/player_detector.py`) — Full implementation:
    - `PlayerBox` dataclass with center, dimensions, containment, and distance-to-point methods
    - `PlayerDetector` class wrapping ultralytics YOLOv8n for person detection
    - Auto-downloads weights on first use; configurable confidence threshold and max players
    - `filter_by_player_proximity()` function: rejects ball detections >N px from nearest player
    - `serialize_player_boxes()` for JSON export

14. **Player-proximity consensus filter** (`src/labeling/consensus.py`) — New `apply_player_proximity_filter()` function:
    - Takes consensus results + per-frame player detections
    - Rejects ball detections far from all detected players (configurable max_distance, default 200px)
    - Returns (filtered_consensus, rejected_count) tuple
    - Preserves no_detection frames unchanged

15. **Pipeline integration** (`src/labeling/pipeline.py`) — Updated to 9-step pipeline (was 7):
    - Step 3: Player detection (YOLOv8-nano) + proximity filtering
    - Saves `players.json` with per-frame player bounding boxes
    - `--no-player-filter` CLI flag to disable filtering
    - `--player-distance` CLI flag (default 200px)
    - Benchmark report includes player filter stats (enabled, rejected count, FP reduction %)

16. **Visualization with player boxes** (`src/labeling/visualize.py`) — Updated review video:
    - Blue rectangles for player bounding boxes
    - Confidence label on each player box
    - `player_detections` parameter in `generate_review_video()`

## Architecture Decisions

- **Weighted centroid over HoughCircles** — The court keypoint heatmaps from courtside footage produce elliptical peaks, not circular. `cv2.moments` handles arbitrary blob shapes. HoughCircles is kept as a fallback for edge cases where moments fail (m00=0).
- **Stationary cluster removal at 10 frames** — A tennis ball in play never stays in the same 15px radius for >10 frames at 30fps. This catches stationary FPs while preserving legitimate slow ball detections (lobs, drops).
- **Player filter as consensus post-processing** — Rather than filtering individual model detections, the filter is applied to consensus results. This is simpler and catches FPs that multiple models agree on (which would be consensus but still wrong).
- **Disk-cached frame streaming** — The inference runner now pickles frames to a temp file instead of holding them all in RAM. Each detector reads from the cache sequentially. Trades disk I/O for ~55GB RAM savings on 10-min videos.

## Known Issues

- **TrackNet V4 weights not yet tested on real video** — Pre-existing from Sprint 4.
- **Florence-2 and YOLO-World not smoke-tested** — Pre-existing from Sprint 3.
- **Player detector requires ultralytics package** — Listed in `requirements-labeling.txt` but not `requirements.txt`.
- **Court detection still depends on model quality** — The weighted centroid fix improves keypoint extraction but can't fix fundamentally poor model predictions on very oblique camera angles.

## File Changes

### Created
- `src/labeling/player_detector.py` — YOLOv8-nano player detector + proximity filter
- `tests/test_sprint5.py` — 40 tests for Sprint 5

### Modified
- `src/api/routes/status.py` — Use `job.snapshot()` for thread safety
- `src/api/routes/results.py` — Use `job.snapshot()` for thread safety
- `src/api/routes/jobs.py` — Use `job.snapshot()` for thread safety
- `src/api/pipeline.py` — Config type fix, step numbering, serve stats fallback, import fix
- `src/api/main.py` — Config type fix, root endpoint, import fix
- `src/api/routes/upload.py` — Removed dead `_VIDEO_MAGIC` dict
- `src/features/court_detect/detector.py` — Weighted centroid keypoint extraction
- `src/features/ball_tracking/detector.py` — Weighted centroid postprocess, stationary removal
- `src/features/rally_stats/counter.py` — Shot count max-per-second cap
- `src/features/serve_analysis/serve_detector.py` — Removed dead locals
- `src/labeling/consensus.py` — Added `apply_player_proximity_filter()`, removed dead imports
- `src/labeling/pipeline.py` — 9-step pipeline with player detection, new CLI flags
- `src/labeling/visualize.py` — Player bounding box overlay
- `src/labeling/benchmark.py` — Player filter stats in report
- `src/labeling/inference_runner.py` — Disk-cached streaming, removed dead cv2 import
- `src/labeling/models.py` — Removed dead logging import
- `src/labeling/rally_boundaries.py` — Removed dead logging import
- `src/labeling/export.py` — Removed dead Optional import
- `src/labeling/tracknetv4_detector.py` — Removed dead os import
- `src/labeling/__init__.py` — Updated docstring
- `frontend/package.json` — Fixed lint script

## Test Results

- 283 tests pass (7.58s) — 243 existing + 40 new
- All imports clean
- All 12 bug fixes verified by tests
- Zero regressions

## Next Sprint (Sprint 6 / Week 1)

- Run pipeline on real test video with weighted centroid + stationary filter
- Measure detection rate improvement vs Sprint 4 baseline
- Test player detector on real footage
- Multi-model benchmark with all 4 detectors
- Ground truth annotation on test clip

## Environment Notes

- No new pip dependencies required for core Sprint 5 features
- Player detector requires `ultralytics` (already in `requirements-labeling.txt`)
- YOLOv8-nano weights auto-download on first use (~6MB)
- CLI: `python -m src.labeling.pipeline <video_path> --output <dir>` now produces 9 output files including `players.json`
