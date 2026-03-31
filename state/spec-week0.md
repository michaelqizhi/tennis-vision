# Sprint Spec — Week 0: Data Labeling Pipeline

> Auto-generated from ROADMAP.md Week 0.
> This spec drives the Planner → Generator → Evaluator harness loop.

## Goal

Build a multi-model consensus labeling pipeline that takes any courtside tennis video and outputs near-complete ball position annotations with minimal human review.

## Sprint Breakdown

### Sprint 1: Model Loader & Inference Harness

**Deliverable:** A unified inference runner that loads multiple ball detection models and runs them on video frames.

**Files to create/modify:**
- `src/labeling/__init__.py`
- `src/labeling/models.py` — model loader abstraction (BaseDetector interface)
- `src/labeling/tracknet_detector.py` — TrackNet V2 wrapper (uses existing `src/features/ball_tracking/detector.py` but replaces HoughCircles with weighted centroid via `cv2.moments`)
- `src/labeling/florence_detector.py` — Florence-2 zero-shot wrapper (prompt: "tennis ball", uses `microsoft/Florence-2-large` from HuggingFace)
- `src/labeling/yoloworld_detector.py` — YOLO-World wrapper (prompt: "tennis ball")
- `src/labeling/inference_runner.py` — runs all detectors on a video, outputs per-frame results as JSON
- `requirements-labeling.txt` — additional deps: `transformers`, `ultralytics`, `supervision`

**Acceptance criteria:**
- Each detector implements `detect(frame: np.ndarray) -> Optional[Tuple[float, float, float]]` (x, y, confidence)
- Runner processes a video file and outputs `detections.json` with shape: `{frame_idx: {model_name: {x, y, conf} | null}}`
- **Critical fix:** TrackNet wrapper uses `cv2.moments` weighted centroid instead of HoughCircles
- Runs without crashing on a 30-second test clip (can use any video for smoke test)

**VRAM note:** Models must run sequentially (not simultaneously) to fit in 8GB. Load one, run all frames, unload, load next.

**Dependencies:** None (first sprint)

---

### Sprint 2: Consensus Engine & Annotation Export

**Deliverable:** Takes multi-model detections and produces consensus annotations in CVAT-compatible format.

**Files to create/modify:**
- `src/labeling/consensus.py` — consensus logic
- `src/labeling/export.py` — CVAT XML and COCO JSON exporters
- `tests/test_labeling_consensus.py` — unit tests with real logic (NOT mocked)

**Acceptance criteria:**
- For each frame, compare all model detections:
  - **≥2 models within 15px** → label as `consensus` with averaged position
  - **1 model only** → label as `uncertain`
  - **0 models** → label as `no_detection`
- Adjustable agreement threshold (default 15px, configurable)
- Export to CVAT XML 1.1 (video annotation format) — must load in CVAT without errors
- Export to COCO JSON format (for training compatibility)
- Unit tests cover: all-agree, partial-agree, no-agree, edge cases (exactly on threshold)

**Dependencies:** Sprint 1

---

### Sprint 3: Review Visualization & Rally Boundary Detection

**Deliverable:** Visual overlay video for human review + automatic rally boundary detection from detection density.

**Files to create/modify:**
- `src/labeling/visualize.py` — overlay generator
- `src/labeling/rally_boundaries.py` — detection density → rally time ranges
- `src/labeling/pipeline.py` — end-to-end CLI orchestrator

**Acceptance criteria:**
- Overlay video output:
  - Green circle: consensus detection
  - Yellow circle: uncertain (single model)
  - Red border: no detection frame
  - Model name labels showing which models detected (small text)
  - Frame counter overlay
- Rally boundary detection:
  - Sliding window over detection density (configurable window size, default 2 sec)
  - Rally = window with ≥30% frames having any detection; dead time = <10% detection density
  - Output: list of `{start_sec, end_sec, detection_rate}` time ranges
  - Configurable min rally duration (default 3 sec) and min gap duration (default 4 sec)
- CLI: `python -m src.labeling.pipeline <video_path> --output <dir>` produces:
  - `detections.json` (raw per-model detections)
  - `consensus.json` (consensus results with labels)
  - `annotations_cvat.xml` (CVAT format)
  - `annotations_coco.json` (COCO format)
  - `review.mp4` (overlay visualization video)
  - `rallies.json` (rally boundary time ranges)
- Pipeline stats printed at end: total frames, consensus %, uncertain %, no-detection %, rally count

**Dependencies:** Sprint 1, Sprint 2

---

### Sprint 4: TrackNet V4 Integration & Model Comparison Report

**Deliverable:** Add TrackNet V4 as a fourth detector, run full comparison, output benchmark report.

**Files to create/modify:**
- `src/labeling/tracknetv4_detector.py` — TrackNet V4 wrapper using [AnInsomniacy/tracknet-series-pytorch](https://github.com/AnInsomniacy/tracknet-series-pytorch) pretrained weights
- `src/labeling/benchmark.py` — comparison report generator
- Update `src/labeling/inference_runner.py` to include V4

**Acceptance criteria:**
- TrackNet V4 detector downloads/loads pretrained weights automatically
- Benchmark report (markdown table) comparing all models:
  - Detection rate (% of frames with detection)
  - Agreement rate with consensus
  - Average confidence score
  - Inference speed (fps)
  - VRAM usage
- Report saved as `benchmark_report.md` in output directory

**Dependencies:** Sprint 1, Sprint 3

---

### Sprint 5: YOLOv8-nano Player Detection & Spatial Filtering

**Deliverable:** Add player detection as a spatial prior for filtering ball detections.

**Files to create/modify:**
- `src/labeling/player_detector.py` — YOLOv8-nano person detector wrapper
- Update `src/labeling/consensus.py` — add player-proximity filter
- Update `src/labeling/pipeline.py` — integrate player detection

**Acceptance criteria:**
- YOLOv8-nano runs on each frame, outputs player bounding boxes
- Player-proximity filter: reject ball detections that are >200px from nearest player bounding box (configurable threshold)
- Visualization overlay includes player bounding boxes (blue rectangles)
- Filter is optional (can be disabled via `--no-player-filter` flag)
- Report updated to show FP reduction from player filtering

**Dependencies:** Sprint 3, Sprint 4

---

## Model Selection

| Model | Source | Purpose | VRAM Est. |
|-------|--------|---------|-----------|
| TrackNet V2 | Existing code (`src/features/ball_tracking/detector.py`) | Baseline ball detection | ~1.5 GB |
| TrackNet V4 | [AnInsomniacy/tracknet-series-pytorch](https://github.com/AnInsomniacy/tracknet-series-pytorch) | Improved ball detection with motion attention | ~2 GB |
| Florence-2 Large | [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) | Zero-shot ball detection (no domain gap) | ~3 GB |
| YOLO-World | [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World) | Fast zero-shot ball detection | ~1.5 GB |
| YOLOv8-nano | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | Player detection (spatial filtering) | ~0.5 GB |

All models run sequentially to stay within 8GB VRAM.

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Florence-2 too slow (<5fps) or OOM on 8GB | Fall back to Florence-2-base (0.2B) or skip Florence entirely |
| YOLO-World can't detect 3-5px tennis balls | Expected — document the failure threshold (minimum ball size in pixels) |
| TrackNet V4 weights incompatible with wrapper | Use soumvincent/TracknetV3-tennis as alternative |
| Consensus rate <30% (models too divergent) | Likely means courtside video is genuinely hard. This IS the signal — validates that fine-tuning is needed. Pipeline still useful for pre-labeling. |
| No test video available on dev machine | Download any amateur courtside tennis match from YouTube for smoke testing |

## Conventions

- Python 3.10+, type hints everywhere
- All detectors inherit from `BaseDetector` abstract class
- No mocked tests — test with real (small) video clips or synthetic frames
- Intermediate results always saved to disk (resume-friendly)
- CLI interface via `argparse`, not hardcoded paths
- Log with `logging` module, not print statements
