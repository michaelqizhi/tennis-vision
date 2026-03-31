# Tennis Vision — Technical Roadmap

> SwingVision competitor for amateur recreational tennis players.
> Single source of truth for project direction. Last updated: 2026-03-31.

## Current State

- **Repo:** `github.com/michaelqizhi/tennis-vision` (private)
- **Code:** 5 files pushed (pipeline.py, schemas.py, detector.py, reader.py, test_api.py); ~12-15 modules exist on Windows but not yet pushed (court_detect, auto_clip, rally_stats, serve_analysis, visualization, config, models/tracknet, tasks, etc.)
- **Ball tracking:** TrackNet v2, trained on broadcast TV footage
- **Real-world test:** 10-min match video → 0 serves detected, 4 rallies found, longest rally "1+ min". Wildly inaccurate.
- **Tests:** 119 tests, all mocked. Zero real video tested.
- **Output:** JSON stats + heatmap PNGs only. No video trimming/clipping.
- **Hardware:** Windows PC (RTX 2060 Super, 8GB VRAM, CUDA), MacBook Air for orchestration
- **Dev setup:** Copilot CLI harness (Planner/Generator/Evaluator agents), PowerShell

## Root Cause Analysis

The core problem is **domain gap**. TrackNet v2 was trained on broadcast footage (fixed overhead cameras, professional courts, consistent lighting). Courtside phone video is a completely different domain:

- Oblique viewing angle (side-on, not overhead)
- Camera shake and drift
- Variable lighting, shadows crossing court
- Cluttered backgrounds (fences, trees, spectators)
- Ball size varies wildly with distance (large near court, tiny far court)

TrackNet's F1 drops from ~95% (broadcast) to potentially **below 50%** on amateur courtside without adaptation. Since the entire pipeline depends on ball tracking, everything downstream fails.

## Architecture Decision: Detect-Then-Project

**Do NOT warp frames to top-down before tracking.** This was evaluated and rejected:

- Warping stretches far-court pixels (15% of frame → 50%), destroying resolution
- Ball shape distorts into ellipses, breaking TrackNet's learned features
- Motion blur direction rotates incorrectly
- Homography jitter between frames breaks temporal consistency

**Correct approach:** Detect ball on **raw frames** (maximum image quality), then use court homography to **project coordinates** to court space for heatmaps/stats. This is the standard in every published tennis CV pipeline.

---

## Phase 1: MVP (3 Weeks)

**Goal:** Upload a 10-min match video → get auto-trimmed highlights with rally stats.

### Week 1: Foundation + Metrics

**Objective:** Pipeline runs end-to-end on real video with measurable baseline.

| Task | Details |
|------|---------|
| Push all missing code | Get all ~12-15 modules from Windows → GitHub |
| Add `requirements.txt` / `pyproject.toml` | Nobody can install or run the project without this |
| Add `config.yaml` | Wire up the config system that every module imports |
| **Fix HoughCircles → weighted centroid** | Replace `postprocess()` HoughCircles with `cv2.moments` / `scipy.ndimage.center_of_mass`. Current code silently drops frames where HoughCircles finds 0 or 2+ circles — on oblique courtside video the ball heatmap is often elliptical, not circular. This is likely the single biggest detection rate improvement. ~10 lines of code. |
| Multi-model benchmark | **Don't blindly pick one model.** Set up a comparison harness for: TrackNet V2 (current), TrackNet V4 ([AnInsomniacy/tracknet-series-pytorch](https://github.com/AnInsomniacy/tracknet-series-pytorch)), [Florence-2](https://huggingface.co/microsoft/Florence-2-large) (zero-shot, prompt "tennis ball"), [YOLO-World](https://github.com/AILab-CVC/YOLO-World) (zero-shot). Run all four on the same test clip, compare detection rate / FP rate / spatial error. Let data decide which model to use. |
| Add YOLOv8-nano player detection | Off-the-shelf person detector (~100+ fps). Use player bounding boxes as a prior: reject ball detections far from both players. Zero training required. Free signal. |
| Video stabilization | Test with and without OpenCV VidStab. May help (shaky video) or hurt (interpolation artifacts on small targets). Keep whichever scores better on the metrics harness. |
| Ground truth annotation | Annotate **3 minutes** (~5400 frames) for ball position + **10-15 rally boundaries** on 1 test clip. Use [CVAT](https://github.com/cvat-ai/cvat) video interpolation mode (annotate every 5th frame, let CVAT interpolate between). ~3-4 hours. For scaling beyond MVP, use **multi-model ensemble pre-labeling**: run all 4 models, auto-accept frames where ≥2 models agree, only manually review disagreements (~10% of frames). |
| Scoring harness | Build automated metrics: per-frame detection rate, false positive rate, spatial error (px), rally boundary IoU. |
| Baseline measurement | Run all models on test video, record metrics per model. Pick the best performer for Week 2. |

**Go/No-Go:**
- ✅ Pipeline runs end-to-end without crashing
- ✅ Best model detects ball in **≥10% of frames** (across V2/V4/Florence-2/YOLO-World benchmark)
- ✅ Scoring harness outputs metrics automatically
- ✅ HoughCircles replaced with weighted centroid
- ❌ <10% detection rate on ALL models → test video may be unusable (wrong angle/resolution). Record a new one from a slightly elevated position (e.g., on a bench or tripod at ~5ft height).

### Week 2: Court Detection + Ball Tracking Improvements

**Objective:** Significantly improve ball detection accuracy via filtering and multi-scale detection.

| Task | Details |
|------|---------|
| Court detection | Integrate [TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector) (14 keypoints, 96.1% on broadcast). Test on courtside footage — expect 4-6 keypoints visible. Also review [arxiv:2404.06977](https://arxiv.org/abs/2404.06977) for amateur-specific court detection. |
| Homography computation | Compute perspective transform from detected court keypoints. Use for coordinate projection (detect-then-project). |
| Court-boundary filtering | Reject any ball detection whose pixel coordinates project outside the court polygon. Major false positive reduction. |
| Far-court crop detection | Run TrackNet a **second time** on a 2x-zoomed crop of the far court region (where ball is tiny). Merge detections from full-frame + crop passes. This directly addresses the "ball too small at far end" problem. |
| Measure improvement | Re-run scoring harness. Compare detection rate, FP rate, spatial error against Week 1 baseline. |

**Go/No-Go:**
- ✅ Court detection finds **≥4 keypoints on ≥80%** of sampled frames
- ✅ Court-boundary filtering reduces false positives by **≥20%**
- ✅ Overall detection rate **≥25% of frames**
- ❌ <4 keypoints → courtside angle too extreme. Fallback: manual 4-corner annotation for this video (annotate once, derive homography).
- ❌ <25% detection after filtering → domain gap too large for pretrained weights. Trigger Phase 2 fine-tuning early.

### Week 3: Rally Detection + Video Trimming (User-Facing MVP)

**Objective:** Working demo — upload match → get trimmed highlights + stats.

| Task | Details |
|------|---------|
| Rally boundary detection | **Primary signal:** ball activity pattern — continuous detections = rally, sustained gaps = dead time. Tune gap threshold, minimum rally length, minimum detection density. |
| Audio confirmation (secondary) | Extract audio track. Use [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) or simple FFT peak detection for ball-impact sounds (2-4kHz). Use as confidence boost for vision-detected boundaries, NOT as primary signal. Outdoor courts are too noisy for audio-only. |
| ffmpeg auto-clipping | Cut video at rally timestamps with 2-3 sec buffer. Output: one clip per rally + one full highlights video with dead time removed. |
| Rally stats JSON | Count, duration, estimated shot count per rally. Output alongside video. |
| End-to-end demo | Upload 10-min raw match → pipeline → trimmed highlights + `results.json`. |

**Go/No-Go:**
- ✅ Rally boundaries match manual annotation with **≥70% IoU**
- ✅ **≥50% of real rallies** detected with **≤30% false positives**
- ✅ Trimmed output looks reasonable to a human viewer
- ❌ <50% rally detection → ball tracking still too sparse. Pivot to audio-primary rally detection (YAMNet) with ball activity as secondary.
- ❌ Output is garbage → validates that pretrained models aren't enough. Budget 4-6 weeks for fine-tuning (Phase 2).

---

## Phase 2: Post-MVP Improvements

Prioritized by user value per effort. Only pursue after Phase 1 MVP demo works.

### 2A: Fine-Tuning (4-6 weeks) — Only If Needed

Trigger: Phase 1 metrics don't meet go/no-go criteria.

- Collect 20-30 diverse amateur courtside match videos from YouTube
- Run current model as **pre-labeler** → export detections as CVAT annotations
- Manually correct labels (~2-4 hours per minute of video — this is the bottleneck)
- Fine-tune TrackNet V4 on courtside dataset
- Alternative: [soumvincent/TracknetV3-tennis](https://github.com/soumvincent/TracknetV3-tennis) (pre-fine-tuned, >94.8% accuracy) — try this first before labeling from scratch

### 2B: Bounce Detection + Shot Placement Heatmaps (1-2 weeks)

- Detect ball direction reversals in y-component relative to court plane → bounce points
- Bounce positions are where the ball *lands*, not where it flies — required for accurate heatmaps
- Separate serve landing positions (in service boxes) from rally shot placements

### 2C: Shot Classification (3-4 weeks)

- [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) → extract skeleton keypoints per frame
- Classify: forehand, backhand, serve, volley from pose sequences
- Train LSTM or small transformer on ~200-300 labeled shots
- Semi-automated: once rally boundaries work, label shot types within detected rallies

### 2D: Serve Analysis (2 weeks)

- Detect serve motion: ball toss (ball going UP in service box area) as point-start anchor
- Classify 1st serve vs 2nd serve from sequence
- Detect double faults (two consecutive faults)
- Serve speed estimation (requires calibrated court distance + frame timestamps)

### 2E: On-Device Deployment (6-8 weeks)

- PyTorch → ONNX → CoreML (iOS) via [coremltools](https://github.com/apple/coremltools)
- PyTorch → ONNX → [ONNX Runtime](https://github.com/microsoft/onnxruntime) (Android)
- Quantize to float16 / int8 for Neural Engine / NNAPI acceleration
- TrackNet V4 is lightweight enough for ~30fps on A15+ chips
- Start with **post-game processing** (upload → wait → results). Real-time is a separate project.

### 2F: Real-Time Processing (8+ weeks)

- Requires on-device deployment first
- Process frames during recording via camera pipeline
- Immediate feedback on Apple Watch (like SwingVision)
- This is SwingVision's key differentiator — long-term goal, not MVP

---

## Open-Source Dependencies

| Component | Resource | URL |
|-----------|----------|-----|
| Zero-shot detection | Florence-2 | <https://huggingface.co/microsoft/Florence-2-large> |
| Zero-shot detection | YOLO-World | <https://github.com/AILab-CVC/YOLO-World> |
| Point tracking | CoTracker (Meta) | <https://github.com/facebookresearch/co-tracker> |
| Player detection | YOLOv8-nano (ultralytics) | <https://github.com/ultralytics/ultralytics> |
| Ball tracking (V4) | tracknet-series-pytorch | <https://github.com/AnInsomniacy/tracknet-series-pytorch> |
| Ball tracking (V3 fine-tuned) | TracknetV3-tennis | <https://github.com/soumvincent/TracknetV3-tennis> |
| Court detection | TennisCourtDetector | <https://github.com/yastrebksv/TennisCourtDetector> |
| Court detection (amateur) | arxiv:2404.06977 | <https://arxiv.org/abs/2404.06977> |
| Reference pipeline | tennis_analysis | <https://github.com/abdullahtarek/tennis_analysis> |
| Rally detection ref | tennis-rally-detector | <https://github.com/lwc-alex/tennis-rally-detector> |
| Audio classification | YAMNet | <https://github.com/tensorflow/models/tree/master/research/audioset/yamnet> |
| Pose estimation | MediaPipe Pose | <https://developers.google.com/mediapipe/solutions/vision/pose_landmarker> |
| Annotation tool | CVAT | <https://github.com/cvat-ai/cvat> |
| Video stabilization | VidStab (OpenCV) | <https://github.com/AdamSpannbauer/python_video_stab> |
| iOS export | coremltools | <https://github.com/apple/coremltools> |

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **TrackNet V4 still fails on courtside video** | Medium | Critical — blocks everything | Try soumvincent V3 fork first. If both fail, trigger Phase 2A fine-tuning early. Far-court crop + court filtering may bridge the gap. |
| **Court detection unreliable at oblique angles** | Medium | High — blocks filtering + heatmaps | Fallback: manual 4-corner annotation per video (one-time, ~30 sec). Not scalable but unblocks MVP demo. |
| **Labeling bottleneck (if fine-tuning needed)** | High (if triggered) | High — 4-6 weeks of tedious work | Use current model as pre-labeler. Pre-labels make correction 3-4x faster than labeling from scratch. Start with 5 diverse videos, not 30. |
| **VRAM constraints (8GB)** | Low | Medium | TrackNet V4 ~2GB. Sequence court detection and ball tracking (don't run simultaneously). Monitor with `nvidia-smi`. |
| **Audio unreliable outdoors** | Medium | Low — audio is secondary signal | Audio is supplementary only. If it doesn't help, skip it. Rally detection works on vision alone. |
| **Scope creep** | High | Medium | This roadmap is the scope. No shot classification, no heatmaps, no on-device in MVP. Resist. |
| **Missing test video** | Low | Medium — blocks Week 1 | Record one at next court session. Or download courtside amateur match from YouTube for initial testing. |

---

## Competitive Context

SwingVision's moat is **data** — years of labeled footage from professional partnerships. We cannot match this. Our strategy:

1. **Leverage open-source models** (TrackNet V4, TennisCourtDetector) instead of training from scratch
2. **Focus on the auto-clip use case first** — this is what amateur players want most and requires less precision than line calling
3. **Build a data flywheel**: every video processed becomes potential training data for fine-tuning later
4. **Target "good enough" accuracy** — users want highlights and rough stats, not Hawk-Eye precision

---

## Dev Workflow

- **Harness:** Copilot CLI (Planner/Generator/Evaluator) for code generation sprints
- **Primary dev:** Windows machine (GPU)
- **Orchestration/monitoring:** MacBook Air via OpenClaw
- **Metrics-driven:** Every change measured against baseline. No vibes-based evaluation.
- **Weekly checkpoints:** Go/no-go gates with concrete thresholds. Pivot early if numbers are bad.
