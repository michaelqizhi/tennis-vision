"""Inference runner — runs all detectors sequentially on a video.

Each detector is loaded, run on every frame, then unloaded before
loading the next one.  This keeps peak VRAM usage within 8 GB.

Output format (``detections.json``)::

    {
      "0": {
        "tracknet_v2": {"x": 123.4, "y": 456.7, "conf": 0.92},
        "florence_2": null,
        "yolo_world": {"x": 120.1, "y": 458.0, "conf": 0.65}
      },
      "1": { ... },
      ...
    }
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from src.labeling.models import BaseDetector
from src.video.reader import VideoReader

logger = logging.getLogger(__name__)


def _serialize_detection(
    det: Optional[tuple[float, float, float]],
) -> Optional[dict]:
    """Convert a detection tuple to a JSON-serializable dict."""
    if det is None:
        return None
    return {"x": round(det[0], 2), "y": round(det[1], 2), "conf": round(det[2], 4)}


def run_inference(
    video_path: str,
    detectors: list[BaseDetector],
    device: str = "cpu",
    output_path: Optional[str] = None,
    max_frames: int = 0,
    return_timing: bool = False,
) -> dict[str, dict[str, Optional[dict]]] | tuple[dict[str, dict[str, Optional[dict]]], dict[str, float]]:
    """Run all detectors on a video, sequentially to conserve VRAM.

    Args:
        video_path: Path to the input video file.
        detectors: List of detector instances to run.
        device: PyTorch device string.
        output_path: If provided, write ``detections.json`` to this path.
        max_frames: Maximum number of frames to process (0 = all).
        return_timing: If True, return a tuple of (detections, timing_dict).

    Returns:
        Nested dict mapping ``frame_idx → model_name → detection_or_null``.
        If ``return_timing`` is True, returns ``(detections, timing_dict)``
        where timing_dict maps model_name → elapsed seconds.
    """
    with VideoReader(video_path, max_frames=max_frames) as reader:
        logger.info(
            "Video: %s — %d frames, %.1f fps, %dx%d",
            video_path,
            reader.frame_count,
            reader.fps,
            reader.width,
            reader.height,
        )
        total = reader.frame_count if max_frames == 0 else min(max_frames, reader.frame_count)

        # Initialize results: frame_idx → {model_name → detection}
        results: dict[str, dict[str, Optional[dict]]] = {}
        timing: dict[str, float] = {}

        # Stream frames to disk cache to avoid holding all in RAM.
        # For a 10-min 1080p video at 30fps, full preload would be ~55GB.
        import tempfile
        import pickle

        frames_cache_path = None
        try:
            # First pass: cache frames to a temp file
            logger.info("Caching frames to disk …")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
            frames_cache_path = tmp.name
            frame_offsets: list[int] = []
            actual_count = 0
            for frame in reader.iter_frames():
                frame_offsets.append(tmp.tell())
                pickle.dump(frame, tmp)
                results[str(actual_count)] = {}
                actual_count += 1
            tmp.close()
            total = actual_count
            logger.info("Cached %d frames", total)

            for detector in detectors:
                model_name = detector.name
                logger.info("=== Running %s ===", model_name)

                try:
                    detector.load(device)
                except Exception:
                    logger.exception("Failed to load %s — skipping", model_name)
                    for i in range(total):
                        results[str(i)][model_name] = None
                    continue

                try:
                    t0 = time.perf_counter()

                    with open(frames_cache_path, "rb") as cache_f:
                        for i in tqdm(range(total), desc=model_name):
                            frame = pickle.load(cache_f)
                            try:
                                det = detector.detect(frame)
                            except Exception:
                                logger.exception("Error in %s on frame %d", model_name, i)
                                det = None
                            results[str(i)][model_name] = _serialize_detection(det)

                    elapsed = time.perf_counter() - t0
                    fps_rate = total / elapsed if elapsed > 0 else 0.0
                    timing[model_name] = elapsed
                    logger.info(
                        "%s: %.1fs (%.1f fps), %d/%d detections",
                        model_name,
                        elapsed,
                        fps_rate,
                        sum(1 for i in range(total) if results[str(i)][model_name] is not None),
                        total,
                    )
                finally:
                    detector.unload()

        finally:
            # Clean up temp cache
            if frames_cache_path:
                try:
                    import os
                    os.unlink(frames_cache_path)
                except OSError:
                    pass

    # Summary
    logger.info("--- Inference Summary ---")
    for name, elapsed in timing.items():
        det_count = sum(
            1 for i in range(total) if results[str(i)].get(name) is not None
        )
        logger.info(
            "  %s: %d/%d detections (%.1f%%), %.1f fps",
            name,
            det_count,
            total,
            100.0 * det_count / total if total else 0,
            total / elapsed if elapsed > 0 else 0,
        )

    # Write output
    if output_path:
        try:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(results, f, indent=2)
            logger.info("Saved detections to %s", out)
        except Exception:
            logger.exception(
                "Failed to write detections to %s — returning in-memory results",
                output_path,
            )

    if return_timing:
        return results, timing
    return results


def get_default_detectors() -> list[BaseDetector]:
    """Return the standard set of detectors for the labeling pipeline.

    Returns:
        List of detector instances (not yet loaded).
    """
    from src.labeling.tracknet_detector import TrackNetDetector
    from src.labeling.florence_detector import FlorenceDetector
    from src.labeling.yoloworld_detector import YOLOWorldDetector
    from src.labeling.yoloworld_sahi_detector import YOLOWorldSAHIDetector
    from src.labeling.groundingdino_detector import GroundingDINODetector

    return [
        TrackNetDetector(),
        FlorenceDetector(),
        YOLOWorldDetector(),
        YOLOWorldSAHIDetector(),
        GroundingDINODetector(),
    ]
