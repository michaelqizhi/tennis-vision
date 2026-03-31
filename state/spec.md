# Tennis Vision — Sprint Plan

*Last updated: 2026-03-31*

## Model Selection

### 1. Ball Tracking: TrackNetV2 (PyTorch)

- **Repository:** https://github.com/yastrebksv/TrackNet
- **Pretrained weights:** Google Drive link in repo README ([direct](https://drive.google.com/file/d/1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl/view))
- **Why this model:**
  - Domain-specific architecture designed for small fast-moving sports balls
  - PyTorch (matches our stack), community-maintained, well-documented
  - ~88% accuracy on tennis datasets out of the box
  - Minimal dependencies — just torch + opencv + numpy
- **Input:** 3 consecutive RGB frames stacked → tensor of shape `(B, 9, 512, 288)`
- **Output:** 3 heatmaps `(B, 3, 512, 288)` — argmax gives ball (x, y) per frame
- **VRAM:** ~2–3 GB inference at 512×288 (well within 8 GB target)
- **Fallback chain (in priority order):**
  1. wolfyeva/TrackNetV2 — improved V2 with separate pretrained weights
  2. soumvincent/TracknetV3-tennis — TrackNetV3 fine-tuned specifically on tennis data
  3. asigatchov/TrackNetV4-PyTorch — PyTorch port of TrackNetV4 (ICASSP 2025), adds motion attention maps for better occlusion handling
  4. alenzenx/TrackNetV3 — official V3 with attention mechanisms (provides both V2/V3 weights)

### 2. Court Detection + Homography: TennisCourtDetector

- **Repository:** https://github.com/yastrebksv/TennisCourtDetector
- **Pretrained weights:** Google Drive link in repo README (trained on 8,841 images across hard/clay/grass)
- **Why this model:**
  - Same author ecosystem as our TrackNet choice (consistent code style, compatible pipeline)
  - Outputs 14 court keypoints → compute homography matrix → pixel-to-court-coordinate transform
  - Includes CV-based postprocessing for refinement
  - Integrated in yastrebksv/TennisProject which demonstrates full pipeline
- **Input:** Single RGB frame resized to `640×360`
- **Output:** 15 heatmaps (14 keypoints + 1 center); postprocessed to 14 (x, y) court keypoints
- **Homography:** `cv2.findHomography()` from detected keypoints to known court template coordinates (ITF standard: 23.77m × 10.97m)
- **VRAM:** ~1–2 GB inference (lightweight, single-frame)
- **Fallback chain:**
  1. TEXflip/tennis-court-detection — uses LETR/HAWP line detection + model-based homography fitting
  2. Semi-automatic mode — user clicks 4 court corners, compute homography from those
- **Amateur footage concern:** This model was primarily trained on broadcast footage. For courtside amateur angles, keypoint detection may degrade. See Risk Assessment for mitigation. Reference: [arXiv:2404.06977](https://arxiv.org/abs/2404.06977) — "Accurate Tennis Court Line Detection on Amateur Recorded Matches" (April 2024) proposes shadow-removal preprocessing that may help.

### 3. No Additional Models Needed for MVP

- **Rally boundary detection:** Heuristic — gap in ball detections + ball trajectory analysis (no ML model needed)
- **Serve detection:** Heuristic — ball position in service box region + trajectory from baseline (uses court homography)
- **Double fault detection:** Two consecutive serve faults (logical rule on top of serve detection)
- **Shot counting:** Count ball direction changes or net crossings using tracked trajectory + homography

---

## Sprint Breakdown

### Sprint 1: Ball Tracking End-to-End Proof

**Deliverable:** Feed a tennis video → get ball positions per frame → draw detected positions on output video

**Files to create:**
- `src/__init__.py` — package init
- `src/config.py` — project config (paths, model params, device selection)
- `src/video/__init__.py` — package init
- `src/video/reader.py` — video file → frame iterator (OpenCV)
- `src/video/writer.py` — frames + overlays → output video
- `src/models/__init__.py` — package init
- `src/models/tracknet.py` — TrackNetV2 model definition + weight loading
- `src/features/__init__.py` — package init
- `src/features/ball_tracking/__init__.py` — package init
- `src/features/ball_tracking/detector.py` — run TrackNetV2 inference on frame triplets, extract (x, y) positions
- `src/features/ball_tracking/visualizer.py` — draw ball positions on frames
- `scripts/download_weights.py` — download TrackNet + court detector pretrained weights
- `scripts/run_ball_tracking.py` — CLI entry point: video in → annotated video out
- `requirements.txt` — torch, opencv-python, numpy, fastapi, uvicorn, etc.

**Dependencies:** None (first sprint)

**Acceptance criteria:**
- [ ] `python scripts/run_ball_tracking.py --input sample.mp4 --output out.mp4` produces a video with green circles on detected ball positions
- [ ] Ball positions also saved to CSV: `frame_number, x, y, confidence`
- [ ] Works on a freely available tennis rally clip (user provides or script downloads one)
- [ ] Runs on GPU (CUDA) and falls back to CPU gracefully
- [ ] No import errors — all `__init__.py` files in place, clean module structure

---

### Sprint 2: Court Detection + Homography

**Deliverable:** Detect court keypoints in video frames, compute homography, transform ball pixel positions to real-world court coordinates

**Files to create:**
- `src/models/court_net.py` — Court keypoint model definition + weight loading
- `src/features/court_detect/__init__.py` — package init
- `src/features/court_detect/detector.py` — run court keypoint inference, postprocess to 14 (x, y) points
- `src/features/court_detect/homography.py` — compute homography matrix from keypoints to ITF court template; transform arbitrary pixel points to court coords
- `src/features/court_detect/court_template.py` — ITF standard court dimensions and keypoint reference coordinates
- `src/features/court_detect/visualizer.py` — draw detected keypoints + court overlay on frame
- `scripts/run_court_detect.py` — CLI: video in → court overlay video + homography matrix saved

**Dependencies:** Sprint 1 (shares video reader, config, model loading patterns)

**Acceptance criteria:**
- [ ] Detected keypoints visually align with court lines on test video
- [ ] Homography correctly maps a pixel in the service box to ~correct real-world coordinates (within 0.5m error)
- [ ] Ball positions from Sprint 1 can be transformed to court coordinates
- [ ] Handles frames where court is partially occluded (returns confidence / skips gracefully)

---

### Sprint 3: Visualization — Court Diagram + Heatmaps

**Deliverable:** 2D court diagram with ball landing positions plotted as a heatmap; serve placement view

**Files to create:**
- `src/features/visualization/__init__.py` — package init
- `src/features/visualization/court_diagram.py` — draw a 2D top-down court diagram (matplotlib or PIL)
- `src/features/visualization/heatmap.py` — plot ball landing positions as a heatmap on court diagram
- `src/features/visualization/renderer.py` — composite court diagram + heatmap, save as image/video overlay
- `scripts/run_visualization.py` — CLI: ball positions CSV + homography → heatmap image

**Dependencies:** Sprint 1 (ball positions), Sprint 2 (court coordinates)

**Acceptance criteria:**
- [ ] Produces a top-down court diagram image with ball positions plotted
- [ ] Heatmap shows density of ball landings (kernel density or binned)
- [ ] Serve placement can be filtered to show only service-box landings
- [ ] Output saved as PNG and/or embedded in output video

---

### Sprint 4: Rally Detection + Auto-Clip

**Deliverable:** Automatically detect rally boundaries (start/end of points), clip video into individual rallies

**Files to create:**
- `src/features/auto_clip/__init__.py` — package init
- `src/features/auto_clip/rally_detector.py` — detect rally boundaries from ball tracking data (gaps in detection, ball speed changes, trajectory resets)
- `src/features/auto_clip/clipper.py` — extract rally segments from source video using ffmpeg or OpenCV
- `src/features/rally_stats/__init__.py` — package init
- `src/features/rally_stats/counter.py` — count shots per rally (net crossings or direction changes)
- `src/features/rally_stats/stats.py` — aggregate stats: rally lengths, shot count distribution
- `scripts/run_auto_clip.py` — CLI: video + ball data → individual rally clips + stats JSON

**Dependencies:** Sprint 1 (ball tracking), Sprint 2 (court homography for net-crossing detection)

**Acceptance criteria:**
- [ ] Rally start/end times detected within ±1 second of actual boundaries
- [ ] Individual rally clips saved as separate video files
- [ ] Shot count per rally is within ±1 of manual count on test video
- [ ] Stats JSON includes: rally_count, shots_per_rally[], avg_rally_length, longest_rally

---

### Sprint 5: Serve Analysis + Double Fault Detection

**Deliverable:** Identify serves, classify 1st/2nd serve, detect double faults, serve placement heatmap

**Files to create:**
- `src/features/serve_analysis/__init__.py` — package init
- `src/features/serve_analysis/serve_detector.py` — detect serve events (ball trajectory from baseline toward service box at rally start)
- `src/features/serve_analysis/fault_detector.py` — detect serve faults (ball landing outside service box or into net)
- `src/features/serve_analysis/double_fault.py` — detect consecutive faults = double fault
- `src/features/serve_analysis/stats.py` — serve stats: 1st serve %, double fault count, placement distribution
- `scripts/run_serve_analysis.py` — CLI: ball data + homography → serve stats JSON + placement heatmap

**Dependencies:** Sprint 2 (homography), Sprint 3 (heatmap visualization), Sprint 4 (rally boundaries to find serve moments)

**Acceptance criteria:**
- [ ] Serves correctly identified at start of each rally
- [ ] 1st vs 2nd serve distinguished (2nd serve follows a fault)
- [ ] Double faults detected (two consecutive faults)
- [ ] Serve placement heatmap shows landing positions in service boxes
- [ ] Stats JSON: total_serves, first_serve_pct, double_faults, ace_candidates

---

### Sprint 6: FastAPI Backend + Integration Pipeline

**Deliverable:** REST API that accepts video upload, runs full pipeline, returns results

**Files to create:**
- `src/api/__init__.py` — package init
- `src/api/main.py` — FastAPI app setup, CORS, error handling
- `src/api/routes/__init__.py` — package init
- `src/api/routes/upload.py` — POST /upload — accept video, queue processing
- `src/api/routes/results.py` — GET /results/{job_id} — return analysis results
- `src/api/routes/status.py` — GET /status/{job_id} — processing progress
- `src/api/pipeline.py` — orchestrate: video → ball tracking → court detect → rally detect → serve analysis → stats + visualizations
- `src/api/schemas.py` — Pydantic models for request/response
- `src/api/tasks.py` — background task runner (simple threading or asyncio for MVP)

**Dependencies:** All previous sprints

**Acceptance criteria:**
- [ ] `POST /upload` with video file returns job_id
- [ ] `GET /status/{job_id}` returns progress (queued → processing → complete)
- [ ] `GET /results/{job_id}` returns JSON with: ball_positions, rally_stats, serve_stats, heatmap_url
- [ ] Full pipeline runs end-to-end on uploaded video
- [ ] API documented with OpenAPI/Swagger (automatic with FastAPI)

---

### Sprint 7: React Frontend

**Deliverable:** Basic web UI — upload video, view results with court heatmap overlay

**Files to create:**
- `frontend/` — Next.js project (via `npx create-next-app`)
- `frontend/src/app/page.tsx` — main page: upload form + results display
- `frontend/src/components/VideoUpload.tsx` — drag-and-drop video upload
- `frontend/src/components/CourtHeatmap.tsx` — SVG/Canvas 2D court with heatmap overlay
- `frontend/src/components/RallyStats.tsx` — rally count, shot distribution chart
- `frontend/src/components/ServeStats.tsx` — serve stats display + serve placement map
- `frontend/src/components/ProcessingStatus.tsx` — progress indicator during analysis
- `frontend/src/lib/api.ts` — API client (fetch wrapper for upload/status/results)

**Dependencies:** Sprint 6 (API must be running)

**Acceptance criteria:**
- [ ] User can upload a video via browser
- [ ] Processing progress shown in real-time
- [ ] Results page shows: court heatmap, rally stats table, serve stats
- [ ] Court heatmap is interactive (hover to see position details)
- [ ] Responsive layout (works on desktop; mobile is stretch goal)

---

## Risk Assessment

### High Risk

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| TrackNet pretrained weights don't generalize to amateur courtside footage (trained on broadcast angles) | Ball tracking fails completely | Medium-High | Test early in Sprint 1 with courtside clips. Fallback chain: wolfyeva/TrackNetV2 → soumvincent/TracknetV3-tennis (tennis-finetuned) → asigatchov/TrackNetV4-PyTorch (motion attention for occlusion). Ultimate fallback: augment with frame-difference motion detection as a secondary signal. |
| Court keypoint detection fails on non-broadcast camera angles | Homography is wrong → all spatial features broken | Medium-High | Test with courtside footage in Sprint 2. Fallback: TEXflip/tennis-court-detection (different architecture). Reference arXiv:2404.06977 for shadow-removal preprocessing on amateur footage. Ultimate fallback: semi-automatic mode where user clicks 4 court corners. |
| Single camera occlusion — ball hidden behind player | Gaps in tracking lead to incorrect shot counts | Medium | Interpolate ball positions through short occlusion gaps (≤5 frames); use trajectory prediction (polynomial/spline fit). TrackNetV4's motion attention may help here. |

### Medium Risk

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Rally boundary detection heuristics are too noisy | Auto-clip produces wrong segments | Medium | Tune thresholds on multiple test videos; add a manual override/correction UI |
| Serve vs. groundstroke classification errors | Wrong serve stats | Medium | Use strong positional heuristic (must start from baseline, land in service box) + validate with rally-start timing |
| Video processing too slow for acceptable UX | Users wait too long | Low-Medium | Process at lower resolution for detection; use batch GPU inference; show progress bar; consider processing at 15fps instead of full frame rate |
| Google Drive weight downloads break (links expire or quota limits) | Can't set up project | Low-Medium | Cache weights in project `weights/` directory (gitignored); document mirror locations; consider hosting on HuggingFace Hub |

### Low Risk

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| VRAM overflow on 8 GB GPU | Can't run models | Low | Both models well under 4 GB at inference; only risk is if both run simultaneously — sequence them |
| Frontend complexity grows | Delays Sprint 7 | Low | Keep MVP frontend minimal; use a component library (shadcn/ui) |

---

## Dependency Graph

```
Sprint 1 (Ball Tracking)
    ├── Sprint 2 (Court Detection + Homography)
    │       ├── Sprint 3 (Visualization + Heatmaps)
    │       ├── Sprint 4 (Rally Detection + Auto-Clip)
    │       │       └── Sprint 5 (Serve Analysis)
    │       └── Sprint 5 (Serve Analysis)
    └── Sprint 4 (Rally Detection — partial, needs ball data)

Sprint 6 (API) ← depends on Sprints 1–5
Sprint 7 (Frontend) ← depends on Sprint 6
```

## Key Architecture Decisions

1. **yastrebksv ecosystem as primary choice** — TrackNet + TennisCourtDetector from the same author gives us a consistent PyTorch codebase, compatible preprocessing, and proven integration (see TennisProject repo). Multiple fallback models identified for each component.

2. **Heuristic-first for rally/serve detection** — No ML model needed for rally segmentation or serve detection. Ball trajectory + court homography provide sufficient signal. This avoids additional model dependencies and training data requirements.

3. **CLI scripts before API** — Each sprint produces a standalone CLI tool. This makes testing and debugging fast without needing the full web stack. API wraps these in Sprint 6.

4. **512×288 inference resolution** — Balances accuracy and speed. Can bump to 640×360 if accuracy is insufficient, still within 8 GB VRAM budget.

5. **Explicit `__init__.py` in every package** — Prevents import errors and makes the module structure explicit. Each sprint must create these for any new directories.
