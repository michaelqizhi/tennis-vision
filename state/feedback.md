# Feedback — Sprint 5 Evaluation

**Date:** 2026-04-01
**Tests:** 283 passed in 7.68s ✓ (243 existing + 40 new)
**Frontend build:** Next.js 16.2.1 (Turbopack), compiled in 1992ms, zero TS errors ✓
**Backend startup:** FastAPI starts cleanly, all routes respond correctly ✓

---

## What Works

- **All 283 tests pass** (243 existing + 40 Sprint 5): zero failures, zero regressions (7.68s)
- **Backend starts cleanly**: `python -m uvicorn src.api.main:app` starts, all endpoints respond correctly
- **Root endpoint fixed**: `GET /` now redirects 307 → `/docs` (was 404)
- **Upload validation**: rejects bad extensions (400), rejects non-video magic bytes (400), enforces 500MB limit (413). Cleanup on startup removed 14 orphaned files
- **Error responses are clean**: bad job IDs → 404 with `{"detail":"Job 'bad-id' not found."}`, global exception handler catches unhandled errors
- **Frontend builds**: Next.js 16.2.1 + Turbopack, compiled in 1992ms, zero TypeScript errors, 3 static pages
- **Frontend components well-structured**: VideoUpload (drag & drop + file input), ProcessingStatus, CourtHeatmap (SVG tennis court with density coloring), RallyStats, ServeStats
- **Frontend–backend integration correct**: Next.js rewrites `/api/*` → FastAPI, typed polling via `pollUntilComplete()`, proper error handling
- **Code architecture clean**: ML inference in `src/features/`, thin API routes, orchestration in `src/api/pipeline.py`, labeling pipeline separate
- **Thread safety infrastructure**: `Job.snapshot()` with `_mutation_lock` + `copy.deepcopy()` exists, single-worker thread pool prevents GPU contention

### Sprint 5 Bug Fixes — Verified ✓
1. **Race condition fix (Critical #4)** — Routes now use `job.snapshot()`. ✓
2. **Weighted centroid (Critical #1)** — `cv2.moments` replaces HoughCircles as primary keypoint extraction. ✓
3. **Stationary detection removal (Critical #2)** — `_remove_stationary()` added, 15px/10-frame threshold. ✓
4. **Shot count cap (Critical #3)** — Capped at 1 shot/sec × rally duration. ✓
5. **Serve stats fallback (Non-critical #7)** — Pipeline sets `total_serves = len(rallies)` in response data. ✓
6. **Frontend lint fix (Non-critical #5)** — Changed to `next lint` (removed `--dir`). ✓
7. **Root endpoint (Non-critical #8)** — `GET /` → 307 `/docs`. ✓
8. **Pipeline type annotations** — `Config` type instead of `object`. ✓
9. **Dead code removal** — `_VIDEO_MAGIC`, dead locals, dead imports all cleaned. ✓
10. **Step numbering** — Pipeline comments now count Steps 1-6 sequentially. ✓
11. **Streaming frames in inference runner** — Disk-cached with pickle instead of RAM. ✓
12. **Player detector + proximity filter** — New Sprint 5 feature, integrated into labeling pipeline. ✓

---

## Issues Found

### Critical

### 1. `job.progress` written without lock from pipeline callback
- **File:** `src/api/pipeline.py:123`
- **Problem:** `_ball_tracking_progress()` writes `job.progress = (1 + step_progress) / all_steps` directly, without acquiring `job._mutation_lock`. The pipeline runs in a background thread. Meanwhile, `job.snapshot()` reads `self.progress` under the lock. This creates a data race — `snapshot()` may read a partially-written float or observe `progress` inconsistent with the step status.
- **Impact:** API consumers can see momentarily inconsistent progress values during ball tracking. Low probability of causing actual crashes, but violates the thread-safety model the codebase otherwise follows.
- **Fix:** Either use `job.update_progress()` (which acquires the lock) or wrap the assignment in `with job._mutation_lock:`.

### 2. Inference runner temp file not closed before unlink on Windows
- **File:** `src/labeling/inference_runner.py:90-99`
- **Problem:** `tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")` opens a file handle. If frame caching raises mid-loop (line 94-98), execution jumps to the `finally` block (line 142-149) which calls `os.unlink(frames_cache_path)`. On Windows, unlinking an open file handle fails silently (caught by `except OSError: pass`), leaving a leaked temp file on disk. Even on the happy path, `tmp.close()` at line 99 only runs after the full frame iteration completes.
- **Fix:** Use `try/finally` to ensure `tmp.close()` is called even if frame iteration raises. Or use a context manager wrapping the write phase.

### 3. No API versioning — routes are at `/upload`, `/status/`, not `/api/v1/`
- **File:** `src/api/main.py:100-103`
- **Problem:** Routes are mounted at root (`/upload`, `/status/{job_id}`, etc.) with no API prefix. The AGENTS.md spec structure and frontend proxy both assume `/api/` prefix handling, but the actual backend serves at root. This works currently because the Next.js rewrite strips `/api/` before forwarding, but:
  - API docs show unprefixed paths, confusing for API consumers
  - No versioning path for breaking changes
  - Direct backend access (without Next.js proxy) has no API namespace
- **Impact:** Works but fragile. Not a runtime bug, but an architectural issue that will cause pain when adding API versions or deploying behind a reverse proxy.

### 4. Result payload can be very large — no pagination or streaming for ball positions
- **File:** `src/api/pipeline.py:258-268`, `src/api/schemas.py:111-121`
- **Problem:** `AnalysisResults.ball_positions` contains one entry per frame. A 10-min video at 30fps = 18,000 entries → ~2.1 MB JSON response. A 60-min match = ~12.6 MB. This is returned as a single JSON blob with no pagination.
- **Impact:** Slow frontend rendering, high memory usage, potential timeout on mobile clients.
- **Fix:** Add pagination to the results endpoint, or return ball positions separately from aggregate stats.

---

## Non-Critical Issues

### 5. Frontend polling cannot be cancelled
- **File:** `frontend/src/lib/api.ts:153-177`
- `pollUntilComplete()` uses recursive `setTimeout` with no cancellation mechanism. If the user resets or the component unmounts, poll timers keep running and may update stale React state (potential "Can't perform state update on unmounted component" warning).
- **Fix:** Return an abort function or use `AbortController`.

### 6. Next.js proxy hardcoded to port 8000
- **File:** `frontend/next.config.ts:11`
- `destination: "http://127.0.0.1:8000/:path*"` is hardcoded. No startup script or docs coordinate backend port. The backend defaults to port 8000 (uvicorn default), but if started on another port the frontend silently fails to reach it.
- **Fix:** Use an env variable like `BACKEND_URL`.

### 7. `CourtHeatmap` has O(n²) density computation
- **File:** `frontend/src/components/CourtHeatmap.tsx:126-146, 173-200`
- Each ball position computes density by iterating all positions. Even with sampling, `computeMaxDensity` iterates 200 samples × all positions. For 18,000 positions this is 3.6 million distance computations per render.
- **Fix:** Grid-based binning for density, or use canvas/WebGL instead of SVG for large datasets.

### 8. `load_config()` called repeatedly without caching in pipeline
- **File:** `src/api/pipeline.py:65`, `src/api/main.py:33`, `src/api/routes/upload.py:62`
- `load_config()` creates a new `Config` instance each time. `get_config()` (singleton) exists but isn't used. Minor perf issue — Config construction imports torch, which is slow.
- **Fix:** Use `get_config()` consistently, or cache `load_config()`.

---

## Code Quality

- **Thread safety model is solid** — `Job._mutation_lock`, `snapshot()`, `_set_step_unlocked()` are well-designed. Only the `pipeline.py:123` bypass violates it.
- **Upload security is good** — temp filenames are securely generated (no path traversal), extensions validated, magic bytes checked, explicit `file.close()`.
- **No circular imports detected** — import graph is clean and layered.
- **Type hints present on most public functions** — good coverage.
- **Config `resolve_path()` allows `..` traversal** — `src/config.py:85-90` doesn't block `../../../etc/passwd` style paths. Low risk since it's only used for model weight paths from config, not user input.

---

## Suggestions for Sprint 6

1. **Fix the `job.progress` lock bypass** — One-line fix in `pipeline.py:123`. Use `with job._mutation_lock:` or `job.update_progress()`.
2. **Fix inference runner temp file leak on Windows** — Wrap `tmp` write phase in try/finally to ensure `tmp.close()`.
3. **Test on real video end-to-end** — Sprint 5 checkpoint says "TrackNet V4 weights not yet tested on real video." The weighted centroid + stationary filter fixes need validation on actual footage.
4. **Add result pagination** — Don't return 18,000+ ball positions in a single JSON response. Either paginate or separate ball_positions into its own endpoint.
5. **Add frontend polling cancellation** — Use AbortController or cleanup function.
6. **Coordinate backend port** — Add a startup script or `.env` that ensures frontend proxy and backend use the same port.

---

## Test Results

```
$ python -m pytest tests\ --tb=short -q
283 passed, 1 warning in 7.68s

$ python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8099
INFO: Application startup complete
Cleaned up 14 orphaned upload file(s)

$ # API endpoint tests (via httpx)
GET  /                     → 307 redirect to /docs
GET  /health               → 200 {"status":"ok","service":"tennis-vision"}
GET  /jobs                 → 200 []
GET  /status/bad-id        → 404 {"detail":"Job 'bad-id' not found."}
GET  /results/bad-id       → 404 {"detail":"Job 'bad-id' not found."}
POST /upload (text file)   → 400 "Unsupported file type '.txt'"

$ # Frontend
next build: ✓ compiled in 1992ms, zero TS errors, 3 static pages

$ # Result payload size test
18,000-frame result (10-min video) → 2.1 MB JSON
```
