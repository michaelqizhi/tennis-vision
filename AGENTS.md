# Tennis Vision — Project Specification

## What Is This
A tennis video analysis app (SwingVision competitor). Users upload match videos recorded from a single camera, and the app provides automated analysis.

## MVP Features
1. **Ball tracking + shot placement heatmap** — detect the ball frame-by-frame, map landing positions to court coordinates
2. **Auto-clip** — detect rally boundaries, remove dead time between points
3. **Rally length / shot count** — count shots per rally, show distribution
4. **Serve placement heatmap** — filtered view of first-serve and second-serve landing positions in service boxes
5. **Double fault detection** — detect consecutive serve faults

## Technical Constraints
- **Single camera input** — one phone/camera placed courtside, no extra hardware
- **Use pretrained domain-specific models** — do NOT train from scratch
  - For ball tracking: prefer TrackNet (v2/v3) or equivalent tennis-specific models with public weights
  - Do NOT use generic YOLO for small ball detection — tennis balls are too small and fast
  - For court detection: use court line detection / homography estimation models
- **All models must have publicly available weights and reproducible inference code**
- Court homography (pixel → real-world court coordinates) is the foundation for all spatial features — invest heavily here

## Tech Stack
- **Backend:** Python + FastAPI
- **ML/CV:** PyTorch, OpenCV
- **Frontend:** React + Next.js (can be basic for MVP)
- **Visualization:** court diagram overlays (simple 2D is fine for MVP, no Three.js needed yet)
- **Video processing:** ffmpeg for preprocessing, OpenCV for frame extraction

## Code Conventions
- Python: type hints, docstrings on public functions
- Keep ML inference separate from API routes (clean separation of concerns)
- Config via environment variables or a single `config.yaml`
- Each feature in its own module under `src/features/`

## Sprint Workflow
- Each sprint must update `state/checkpoint.md` with:
  - What was implemented
  - Key architecture decisions and why
  - Known issues
  - What to do next
- Generator MUST read `state/checkpoint.md` and `state/feedback.md` (if exists) before starting work
- Evaluator writes findings to `state/feedback.md`

## File Structure
```
tennis-vision/
├── AGENTS.md              # This file
├── src/
│   ├── api/               # FastAPI routes
│   ├── features/
│   │   ├── ball_tracking/  # TrackNet integration
│   │   ├── court_detect/   # Court line detection + homography
│   │   ├── auto_clip/      # Rally boundary detection
│   │   ├── rally_stats/    # Shot counting, rally length
│   │   ├── serve_analysis/ # Serve placement, double fault
│   │   └── visualization/  # Heatmaps, court overlays
│   ├── models/             # Model loading, inference wrappers
│   ├── video/              # Video I/O, frame extraction
│   └── config.py
├── frontend/               # React/Next.js app
├── tests/                  # Test videos + expected outputs
├── state/                  # Sprint state files
│   ├── spec.md
│   ├── checkpoint.md
│   └── feedback.md
├── logs/                   # Per-sprint logs
└── scripts/                # Utility scripts
```
