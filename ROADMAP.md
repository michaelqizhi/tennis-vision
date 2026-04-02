# Tennis Vision — Technical Roadmap

> SwingVision competitor for amateur recreational tennis players.
> Single source of truth for project direction. Last updated: 2026-04-01.

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

## Primary Model Decision: TrackNet V5

**TrackNet V5 is the default primary model.** This is not a multi-model shootout — V5 is the clear successor in the TrackNet lineage:

- **V2 → V4 → V5** is a clean progression. V4 introduced motion attention maps (frame differencing) to handle occlusion/low-visibility, but used absolute difference which **loses direction information** (ball moving left vs right looks identical).
- **V5 fixes this** with direction-decoupled motion channels + lightweight Transformer spatiotemporal refinement.
- **Performance:** F1 = 0.9859 on TrackNet V2 benchmark dataset, FLOPs only +3.7% vs V4, 114 FPS on T4 GPU (real-time capable).
- **No realistic scenario where V2 or V4 beats V5** on the same data.

Other models (Florence-2, YOLO-World) serve as **pseudo-labeling contributors** in the consensus engine, not as primary model candidates.

## Recording Specification (Input Quality Gate)

> **The cheapest way to improve model accuracy is to constrain the input.**

SwingVision requires specific recording conditions even for imported videos. We adopt similar constraints as hard requirements — videos that don't meet these are rejected at pipeline entry with guidance to re-record.

### Hard Requirements

| Constraint | Value | Rationale |
|-----------|-------|-----------|
| Resolution | ≥ 1080p | Ball is 3-5px at far court on 1080p; lower resolution makes it undetectable |
| Frame rate | ≥ 60 fps | Ball moves 10-30px/frame at 60fps; at 30fps motion blur doubles and inter-frame gap is too large for temporal models |
| Camera position | Behind baseline, centered | Side-on views cause extreme foreshortening; behind-baseline maximizes court visibility |
| Camera height | ≥ 5 ft (tripod/bench/fence mount) | Ground-level causes extreme oblique angles; elevated position shows more court surface |
| Court coverage | Both baselines visible in frame | Required for court detection and homography computation |
| Stability | Tripod or fixed mount strongly preferred | Camera shake corrupts motion attention features in V4/V5 and degrades TrackNet temporal input |

### Auto Quality Check (pipeline entry)

The pipeline automatically validates:
- Resolution and frame rate from video metadata
- Court coverage via court keypoint detection (≥4 keypoints visible)
- Stability score via frame-to-frame motion estimation

Videos failing hard requirements → rejected with specific guidance ("mount higher", "move behind baseline", etc.).
Videos failing soft requirements (stability) → warning + optional stabilization pass.

---

### Week 0: Data Labeling Pipeline (Pre-MVP Foundation)

**Objective:** Build an automated multi-model labeling pipeline that produces near-complete ball position annotations for any courtside tennis video with minimal human review.

| Task | Details |
|------|---------|
| Multi-model inference runner | Run TrackNet V5 (primary), Florence-2, and YOLO-World on every frame of an input video. Output per-frame ball coordinates (or "no detection") from each model. |
| Consensus engine | For each frame, compare detections across models. Apply **three-layer validation** before accepting: (1) **Pixel agreement:** ≥2 models agree within 15px. (2) **Motion consistency:** predicted velocity/acceleration within physically plausible range given frame rate (reject teleporting detections). (3) **Appearance consistency:** local patch around candidate matches ball color/texture profile (reject white shoes, net tape reflections, line markings). Auto-accept frames passing all three → ground truth. Fail any layer → "uncertain" for human review. |
| CVAT-compatible export | Output annotations in CVAT XML or COCO JSON format. "Uncertain" frames flagged for human review. |
| Review interface | Generate a video overlay showing: green dots (consensus), yellow dots (uncertain), red frames (no detection). Human reviews by scrubbing through the overlay video and correcting errors. |
| Rally boundary auto-labeling | As a bonus signal: detect rally boundaries from ball activity density (clusters of detections = rally, gaps = dead time). Export as time ranges. |
| CLI interface | `python -m src.labeling.pipeline <video_path> --output <annotations_dir>` |

**Go/No-Go:**
- ✅ Pipeline runs end-to-end on a test video without crashing
- ✅ Consensus rate ≥60% of frames (≥2 models agree + pass validation)
- ✅ Output loads correctly in CVAT for human review
- ❌ Consensus rate <30% → models are too divergent on courtside video. Fall back to single-model (V5) pre-labeling with full manual review.

**Long-term value:** Every new match video you record feeds into this pipeline. Over time, accumulated labeled data enables fine-tuning. This is the data flywheel.

---

## Phase 1: MVP (3 Weeks)

**Goal:** Upload a 10-min match video → get auto-trimmed highlights with rally stats.

### Evaluation Framework (applies to all weeks)

Metrics are organized in **three layers** — improvements must be measured at all three levels, because "better frame-level F1" can sometimes hurt trajectory continuity or rally segmentation.

**Layer 1: Frame-level localization** (is the ball found correctly?)
- Precision / Recall / F1 at adaptive distance threshold τ (scaled by ball size in frame, not fixed 15px)
- Mean localization error (px) and P50/P90 error percentiles

**Layer 2: Trajectory continuity** (is the track stable enough for downstream?)
- Trajectory break rate: breaks per minute, maximum break duration
- Physical consistency: % of frames with velocity/acceleration outliers

**Layer 3: Event-level** (does the MVP output make sense?)
- Rally segmentation IoU
- Rally recall (% of real rallies detected) and false positive rate
- Human evaluation: "shareable vs not shareable" on trimmed highlights

**Golden test set:** 3-5 short video clips (2-3 min each), diverse courts/lighting/backgrounds, fully human-annotated for ball position + rally boundaries. All experiments report metrics on this set. Integrated into CI for regression testing.

### Week 1: Foundation + V5 Baseline

**Objective:** Pipeline runs end-to-end on real video with measurable V5 baseline.

| Task | Details |
|------|---------|
| Push all missing code | Get all ~12-15 modules from Windows → GitHub |
| Add `requirements.txt` / `pyproject.toml` | Nobody can install or run the project without this |
| Add `config.yaml` | Wire up the config system that every module imports |
| **Fix HoughCircles → weighted centroid** | Replace `postprocess()` HoughCircles with `cv2.moments` / `scipy.ndimage.center_of_mass`. Current code silently drops frames where HoughCircles finds 0 or 2+ circles — on oblique courtside video the ball heatmap is often elliptical, not circular. This is likely the single biggest detection rate improvement. ~10 lines of code. |
| **Integrate TrackNet V5** | V5 is the primary model. Set up inference pipeline, verify it runs on RTX 2060. Also run V2 (current) as baseline comparison to quantify improvement. |
| Auxiliary model runs | Run Florence-2 and YOLO-World on the same test clips — **not as primary model candidates**, but to (a) characterize their detection patterns for pseudo-labeling consensus, and (b) identify V5's specific failure modes by comparing where they disagree. |
| Add YOLOv8-nano player detection | Off-the-shelf person detector (~100+ fps). Use player bounding boxes as a prior: reject ball detections far from both players. Zero training required. Free signal. |
| Video stabilization | Test with and without OpenCV VidStab. May help (shaky video) or hurt (interpolation artifacts on small targets, corruption of V5 motion features). Keep whichever scores better on the metrics harness. |
| Ground truth annotation | Annotate **3 minutes** (~5400 frames at 30fps, ~10800 at 60fps) for ball position + visibility attribute (`visible` / `occluded`) + **10-15 rally boundaries** on 1 test clip. Use [CVAT](https://github.com/cvat-ai/cvat) video interpolation mode (annotate every 5th frame, let CVAT interpolate between). ~3-4 hours. Label only in-play balls (during rallies/serves), not dead balls between points. |
| Scoring harness | Build automated three-layer metrics (frame-level, trajectory, event-level). |
| V5 baseline measurement | Run V5 on test video, record all three metric layers. Compare against V2 to quantify the upgrade. |

**Go/No-Go:**
- ✅ Pipeline runs end-to-end without crashing
- ✅ V5 detects ball in **≥30% of frames** on compliant video (recording spec met)
- ✅ Three-layer scoring harness outputs metrics automatically
- ✅ HoughCircles replaced with weighted centroid
- ❌ <30% detection rate → check recording compliance first. If compliant, domain gap is too severe for pretrained V5 — trigger Phase 2A fine-tuning early.
- ❌ <10% detection rate on non-compliant video → expected. Re-record with proper setup before proceeding.

#### Week 1 Results (2026-04-02)

**V5 status:** Unavailable. No public weights or permissive license exist. The V5 SDK repo (`codelancera-offical/TrackNetV5-SDK`) has architecture code but empty `weights/` directory. Falling back to V2/V3/V4.

**Test set:** 6000 frames (3.3 min at 30fps) of real courtside amateur match video, 3883 annotated frames across 14 rallies. Human-annotated in CVAT with visibility attributes.

**Baseline comparison (all pretrained, no fine-tuning):**

| Metric | V2 (tennis) | V3 (badminton) | V4 (badminton) |
|--------|-------------|----------------|----------------|
| Raw detection rate | 62.9% | **55.5%** | 53.5% |
| After interpolation | **73.8%** | 61.4% | 64.5% |
| Precision | 0.42 | **0.43** | 0.28 |
| Recall | **0.47** | 0.39 | 0.27 |
| F1 | 0.44 | **0.41** | 0.28 |
| Mean error (px) | **3.02** | 4.67 | 7.46 |
| Breaks/min | 83 | **64** | 126 |
| Physical consistency | 0.79 | **0.82** | 0.74 |
| Rally recall | **28.6%** | 14.3% | 7.1% |
| Rally FP rate | **96.6%** | 98.5% | 99.4% |
| Inference speed | 14 fps | **78 fps** | 22 fps |

**Key findings:**

1. **V3 has the best trajectory continuity** (64 breaks/min vs 83 for V2) and highest precision (0.43) despite being trained on badminton. Its 8-frame window + background concat provides strong temporal context. Its 78 fps inference speed (non-overlapping stride-8 windows) is also the fastest.

2. **V2 has the best raw recall** (47%) because it was trained on tennis data. But its trajectory is more fragmented than V3.

3. **V4 had catastrophic postprocessing bugs** that were found and fixed: (a) selected smallest blob instead of largest, (b) rejected blobs >30px area when real blobs were 31-53px, (c) used output channel 2 instead of channel 1 (center frame with bidirectional context). After fixes: detection rate jumped from 20.9% → 53.5%, but still worst overall due to badminton-only weights with no skip connections.

4. **V4 also had BGR→RGB and frame order bugs** found and fixed in earlier investigation round.

5. **All rally detection is poor** (best: 28.6% recall, 96.6% FP rate). This is expected — raw detections without trajectory smoothing produce hyper-fragmented predicted rallies. Week 2's trajectory layer is critical.

**Available training data for fine-tuning:**

| Dataset | Frames | Sport | Access |
|---------|--------|-------|--------|
| TrackNet Original (Huang et al.) | ~21K | Tennis | Free (Google Drive) |
| CoachAI Badminton | ~78K | Badminton | Free (SharePoint) |
| TrackNet V3 raw_data | Varies | Tennis | In GitHub repo |
| TrackNet V4 multiball | Unknown | Tennis+Badminton | Request form |
| Our annotations | 3,883 | Tennis (courtside) | Local |

**Go/No-Go assessment:**
- ✅ Pipeline runs end-to-end without crashing
- ✅ Best model detects ball in ≥30% of frames (V2: 62.9%, V3: 55.5%)
- ✅ Three-layer scoring harness outputs metrics automatically
- ✅ HoughCircles replaced with weighted centroid
- ⚠️ Rally detection broken — needs Week 2 trajectory layer
- ⚠️ V5 unavailable — V3 is the most promising architecture for fine-tuning (best trajectory continuity, fastest inference, U-Net with skip connections)

**Recommendation:** Fine-tune V3 on tennis data (download free 21K-frame TrackNet dataset + our 3,883 frames). V3's architecture (U-Net + skip connections + background concat) is superior to V2, and fine-tuning on tennis should close the precision gap. Proceed with Week 2 trajectory layer in parallel.

### Week 2: Court Detection + Trajectory Layer

**Objective:** Significantly improve tracking quality via court geometry and trajectory-level processing.

| Task | Details |
|------|---------|
| Court detection | Integrate [TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector) (14 keypoints, 96.1% on broadcast). Test on courtside footage — expect 4-6 keypoints visible. Also review [arxiv:2404.06977](https://arxiv.org/abs/2404.06977) for amateur-specific court detection. |
| Homography computation | Compute perspective transform from detected court keypoints. **Use keyframe-based updates + temporal smoothing** — do NOT estimate homography independently per frame (jitter from frame-to-frame keypoint noise gets amplified into coordinate space). |
| Court-boundary filtering | Reject any ball detection whose pixel coordinates project outside the court polygon (with margin for out-of-bounds shots and ball toss). Major false positive reduction. |
| **Trajectory layer** | Add post-detection trajectory processing: (1) **Temporal smoothing** — Kalman filter or Savitzky-Golay on detected positions. (2) **Gap interpolation** — for gaps ≤ N frames, interpolate using motion model (constant velocity / constant acceleration). (3) **Physical constraint filtering** — reject detections that violate max ball speed (~80 m/s) or imply impossible acceleration. This converts sparse, noisy per-frame detections into continuous, physically plausible trajectories. |
| Far-court crop detection | Run V5 a **second time** on a 2x-zoomed crop of the far court region (where ball is tiny). Merge detections from full-frame + crop passes. This directly addresses the "ball too small at far end" problem. |
| Measure improvement | Re-run three-layer scoring harness. Compare all metrics against Week 1 baseline. |

**Go/No-Go:**
- ✅ Court detection finds **≥4 keypoints on ≥80%** of sampled frames
- ✅ Court-boundary filtering reduces false positives by **≥20%**
- ✅ Trajectory layer reduces break rate by **≥50%** vs raw detections
- ✅ Overall detection rate **≥40% of frames** (after trajectory interpolation)
- ❌ <4 keypoints → courtside angle too extreme. Fallback: manual 4-corner annotation for this video (annotate once, derive homography).
- ❌ <40% detection after filtering + trajectory → domain gap too large for pretrained weights. Trigger Phase 2A fine-tuning early.

### Week 3: Rally Detection + Video Trimming (User-Facing MVP)

**Objective:** Working demo — upload match → get trimmed highlights + stats.

| Task | Details |
|------|---------|
| Rally boundary detection | **Primary signal:** ball activity pattern from trajectory layer — continuous trajectory segments = rally, sustained gaps = dead time. Tune gap threshold, minimum rally length, minimum detection density. The trajectory layer from Week 2 should make this significantly more reliable than raw per-frame detections. |
| Audio confirmation (optional) | Extract audio track. Use [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) or simple FFT peak detection for ball-impact sounds (2-4kHz). Use as confidence boost for vision-detected boundaries, NOT as primary signal. Outdoor courts are too noisy for audio-only. **Only pursue if vision-only rally detection doesn't meet go/no-go.** |
| ffmpeg auto-clipping | Cut video at rally timestamps with 2-3 sec buffer. Output: one clip per rally + one full highlights video with dead time removed. |
| Rally stats JSON | Count, duration, estimated shot count per rally. Output alongside video. |
| End-to-end demo | Upload 10-min raw match → pipeline → trimmed highlights + `results.json`. |

**Go/No-Go:**
- ✅ Rally boundaries match manual annotation with **≥70% IoU**
- ✅ **≥50% of real rallies** detected with **≤30% false positives**
- ✅ Trimmed output looks reasonable to a human viewer ("shareable" quality)
- ❌ <50% rally detection → check if trajectory layer gaps are the bottleneck. If so, consider CoTracker for trajectory completion (see Ablation Experiments E5). If trajectory is fine but rally logic is wrong, tune thresholds.
- ❌ Output is garbage → validates that pretrained models aren't enough. Budget 4-6 weeks for fine-tuning (Phase 2).

---

## Ablation Experiments

Run on the golden test set. Report all three metric layers. Purpose: **make data-driven decisions about which components are worth the complexity.**

| ID | Experiment | What it tests |
|----|-----------|---------------|
| E1 | V2 (current) vs V5 (no other changes) | Is V5 a meaningful upgrade on our domain? |
| E2 | V5 + trajectory layer vs V5 raw | Does smoothing/interpolation improve rally detection? |
| E3 | E2 + court-boundary filtering vs E2 | Does geometric filtering help or hurt (risk: filtering out valid lobs/toss)? |
| E4 | E3 + far-court multi-scale ROI vs E3 | Is "ball too small at far end" the main bottleneck? |
| E5 | E4 + CoTracker trajectory completion vs E4 + simple interpolation | Is a learned tracker worth the complexity over linear/motion-model interpolation? Only run if trajectory breaks remain a problem after E4. |
| E6 | Vision-only rally segmentation vs vision + YAMNet audio | Does audio actually improve rally boundary detection? Test across indoor/outdoor/windy conditions. |

---

## Phase 2: Post-MVP Improvements

Prioritized by user value per effort. Only pursue after Phase 1 MVP demo works.

### 2A: Fine-Tuning (4-6 weeks) — Only If Needed

Trigger: Phase 1 metrics don't meet go/no-go criteria.

- Collect 20-30 diverse amateur courtside match videos from YouTube
- Run V5 as **pre-labeler** → export detections as CVAT annotations
- Manually correct labels (~2-4 hours per minute of video — this is the bottleneck)
- Fine-tune TrackNet V5 on courtside dataset
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

- PyTorch → CoreML directly via [coremltools](https://github.com/apple/coremltools) (preferred for iOS; ONNX intermediate step not required)
- PyTorch → ONNX → [ONNX Runtime](https://github.com/microsoft/onnxruntime) (Android)
- Quantize to float16 / int8 for Neural Engine / NNAPI acceleration
- TrackNet V5 is lightweight enough for ~30fps on A15+ chips (similar parameter count to V4)
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
| Ball tracking (primary) | TrackNet V5 | TBD — check for official repo release |
| Ball tracking (V4 fallback) | tracknet-series-pytorch | <https://github.com/AnInsomniacy/tracknet-series-pytorch> |
| Ball tracking (V3 fine-tuned) | TracknetV3-tennis | <https://github.com/soumvincent/TracknetV3-tennis> |
| Zero-shot detection (pseudo-labeling) | Florence-2 | <https://huggingface.co/microsoft/Florence-2-large> |
| Zero-shot detection (pseudo-labeling) | YOLO-World | <https://github.com/AILab-CVC/YOLO-World> |
| Trajectory completion (optional) | CoTracker (Meta) | <https://github.com/facebookresearch/co-tracker> |
| Player detection | YOLOv8-nano (ultralytics) | <https://github.com/ultralytics/ultralytics> |
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
| **TrackNet V5 repo not publicly available yet** | Medium | High — blocks primary model | Fallback: use V4 from tracknet-series-pytorch. V4 is still a significant upgrade over V2. Monitor for V5 release. |
| **V5 still fails on courtside video** | Medium | Critical — blocks everything | Recording spec compliance first. Then try soumvincent V3 fork. If both fail, trigger Phase 2A fine-tuning early. Far-court crop + court filtering + trajectory layer may bridge the gap. |
| **Court detection unreliable at oblique angles** | Medium | High — blocks filtering + heatmaps | Fallback: manual 4-corner annotation per video (one-time, ~30 sec). Not scalable but unblocks MVP demo. |
| **Labeling bottleneck (if fine-tuning needed)** | High (if triggered) | High — 4-6 weeks of tedious work | Use V5 as pre-labeler. Pre-labels make correction 3-4x faster than labeling from scratch. Start with 5 diverse videos, not 30. |
| **VRAM constraints (8GB)** | Low | Medium | TrackNet V5 ~2GB (similar to V4). Sequence court detection and ball tracking (don't run simultaneously). Monitor with `nvidia-smi`. |
| **Non-compliant recording kills detection** | High | Critical | Input quality gate rejects bad video upfront with actionable guidance. Users learn correct setup on first attempt. |
| **Audio unreliable outdoors** | Medium | Low — audio is optional secondary signal | Audio is supplementary only. If it doesn't help, skip it. Rally detection works on vision + trajectory layer alone. |
| **Scope creep** | High | Medium | This roadmap is the scope. No shot classification, no heatmaps, no on-device in MVP. Resist. |

---

## Competitive Context

SwingVision's moat is **data** — years of labeled footage from professional partnerships. We cannot match this. Our strategy:

1. **Leverage open-source models** (TrackNet V5, TennisCourtDetector) instead of training from scratch
2. **Focus on the auto-clip use case first** — this is what amateur players want most and requires less precision than line calling
3. **Build a data flywheel**: every video processed becomes potential training data for fine-tuning later
4. **Target "good enough" accuracy** — users want highlights and rough stats, not Hawk-Eye precision
5. **Constrain input quality** — like SwingVision, enforce recording requirements to keep the problem tractable

---

## Annotation Guidelines

### What to label
- **In-play balls only:** label ball position during rallies and serves (ball in flight, bouncing, or rolling during active play)
- **Dead balls:** do NOT label balls sitting still between points. TrackNet training datasets (V2/V3/V4) are rally-centric — they only contain frames from serve-to-score segments
- **One ball per frame:** label only the match ball, ignore stray balls on sidelines

### Visibility attribute
Add a `visibility` attribute to the `ball` label in CVAT:
- `visible` — ball identifiable in frame, position marked (default)
- `occluded` — ball obscured by player/net but position estimated from neighboring frames

### Why this matters
- TrackNet training uses binary heatmap supervision (present vs absent). Occluded balls labeled with estimated positions preserve positive supervision and teach the model to track through occlusion.
- Downstream rally detection uses detection density as primary signal. Dead ball detections would blur rally/non-rally boundaries.
- This is consistent with the original TrackNet dataset methodology (clips from serve-to-score, visibility classes 0-3).

---

## Dev Workflow

- **Harness:** Copilot CLI (Planner/Generator/Evaluator) for code generation sprints
- **Primary dev:** Windows machine (GPU)
- **Orchestration/monitoring:** MacBook Air via OpenClaw
- **Metrics-driven:** Every change measured against golden test set baseline. No vibes-based evaluation.
- **Weekly checkpoints:** Go/no-go gates with concrete thresholds. Pivot early if numbers are bad.
- **CI regression:** Golden test set runs automatically on every model/pipeline change. Metric regression beyond threshold blocks merge.
