# Checkpoint — Week 0, Sprint 1 (Fix Pass): Model Loader & Inference Harness

## Completed

Built the multi-model inference harness for the data labeling pipeline (Week 0, Sprint 1), then applied critical bug fixes from evaluator feedback.

### New Files (Sprint 1)
1. **`src/labeling/__init__.py`** — Package init for the labeling pipeline module.
2. **`src/labeling/models.py`** — `BaseDetector` abstract base class. All detectors implement `detect(frame) → Optional[(x, y, conf)]`, plus `load(device)` / `unload()` for sequential VRAM management.
3. **`src/labeling/tracknet_detector.py`** — TrackNet V2 wrapper. Uses `cv2.moments` weighted centroid instead of `HoughCircles` for postprocessing. Handles elliptical heatmap blobs from oblique courtside cameras. Maintains an internal 3-frame sliding window.
4. **`src/labeling/florence_detector.py`** — Florence-2 zero-shot wrapper. Uses `microsoft/Florence-2-large` with `<OD>` task, filters for "ball" labels.
5. **`src/labeling/yoloworld_detector.py`** — YOLO-World wrapper. Uses `yolov8l-worldv2` with `set_classes(["tennis ball"])`.
6. **`src/labeling/inference_runner.py`** — Sequential inference runner. Loads one model at a time, runs all frames, unloads, then loads the next.
7. **`requirements-labeling.txt`** — Additional deps: `transformers>=4.40.0`, `ultralytics>=8.1.0`, `supervision>=0.19.0`, `Pillow>=10.0.0`.
8. **`tests/test_labeling.py`** — 29 tests (expanded from 19 in fix pass).

### Fixes Applied (from feedback.md)

#### Critical Fixes
1. **TrackNet argmax→softmax** (`src/labeling/tracknet_detector.py:131`) — Replaced `out.argmax(dim=1)` with `out.softmax(dim=1)[0, 1]`. `argmax` produced integer class indices (0, 1, 2…) that corrupted the weighted centroid after `*255` and uint8 truncation. `softmax` produces proper [0,1] probability heatmap for the ball class. Added test `test_softmax_produces_heatmap` to verify.
2. **VRAM leak: unload() in finally** (`src/labeling/inference_runner.py`) — Wrapped the detect loop + timing in `try/finally` so `detector.unload()` runs even on unexpected exceptions. Prevents VRAM leak that could OOM the next detector. Added test `test_detector_unloaded_on_detect_error`.

#### Code Quality Fixes
3. **Florence-2 model.eval()** (`src/labeling/florence_detector.py`) — Added `self._model.eval()` after loading to disable BatchNorm/Dropout training behavior during inference.
4. **JSON write guarded** (`src/labeling/inference_runner.py`) — Wrapped `json.dump` in try/except so results are returned even if file I/O fails (disk full, permissions). Added test `test_json_write_failure_returns_results`.
5. **Tests for Florence & YOLO-World** (`tests/test_labeling.py`) — Added 10 new tests: `TestFlorenceDetector` (name, detect-without-load, unload-without-load), `TestYOLOWorldDetector` (name, detect-without-load, unload-without-load, custom model size), TrackNet softmax test, runner unload-on-error test, JSON write failure test.

### Maintenance
- Cleaned 8 stale empty directories from `output/`.

## Architecture Decisions
- **`cv2.moments` weighted centroid** — Replaces `HoughCircles` in the labeling pipeline's TrackNet wrapper. `cv2.moments` computes the centroid of any thresholded region regardless of shape. The existing `src/features/ball_tracking/detector.py` still uses HoughCircles (unchanged — separate code path).
- **`softmax` for heatmap extraction** — `out.softmax(dim=1)[0, 1]` extracts the ball-class probability as a proper [0,1] heatmap, which the weighted centroid can threshold and process correctly.
- **Sequential model loading** — Models are loaded one at a time to fit in 8GB VRAM. `unload()` is always called via `finally` to prevent VRAM leaks.
- **Per-frame interface** — Each detector exposes `detect(frame) → Optional[(x, y, conf)]`. TrackNet internally buffers 3 frames. This uniform interface lets the runner treat all detectors identically.
- **Graceful failure** — If a detector fails to load, the runner fills all frames with `null` for that model. If JSON output write fails, results are still returned in memory.

## Known Issues
- **Florence-2 and YOLO-World not yet smoke-tested with real weights** — Requires `pip install -r requirements-labeling.txt` and model downloads.
- **Frames loaded into memory** — For very long videos, a disk-based cache would be better.
- **Court detection still fails on some camera angles** — Pre-existing.
- **CORS wildcard** — Still present for dev flexibility (pre-existing).
- **Thread safety of Job mutations** — Pre-existing.
- **Results route returns 200 for in-progress jobs** — Should be 202 (pre-existing, noted in feedback).
- **Pipeline temp file cleanup only on success** — Pre-existing.

## File Changes
- Modified: `src/labeling/tracknet_detector.py` — argmax→softmax fix
- Modified: `src/labeling/inference_runner.py` — unload in finally, JSON write guard
- Modified: `src/labeling/florence_detector.py` — added model.eval()
- Modified: `tests/test_labeling.py` — 10 new tests (29 total)

## Test Results
- 148 tests pass (5.34s) — 119 existing + 29 labeling tests
- All labeling module imports clean
- All critical and code quality fixes verified by new tests

## What to Do Next (Sprint 2)
- Consensus engine: compare multi-model detections per frame (≥2 within 15px → consensus, 1 → uncertain, 0 → no detection)
- CVAT XML and COCO JSON export
- Unit tests with real consensus logic (not mocked)
