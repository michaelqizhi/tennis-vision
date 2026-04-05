"""Smart frame selector for court detection.

Finds the best frames for court detection by filtering out garbage frames
(pre-match, intermission, replays) and selecting frames with clear court views.
"""

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.features.court_detect.detector import CourtDetector, CourtDetectionResult

logger = logging.getLogger(__name__)


def compute_court_color_score(frame_bgr: np.ndarray) -> float:
    """Score frame by presence of court-colored pixels (HSV).

    Checks for typical tennis court colors:
    - Blue hard court (hue ~100-130°)
    - Green court/grass (hue ~40-80°)
    - Red/orange clay (hue ~0-20°)

    Focuses on the middle 60% of the frame to avoid scoreboard/crowd pixels.

    Args:
        frame_bgr: Input frame in BGR format.

    Returns:
        Score in range [0, 1], where higher = more court-colored pixels.
    """
    # Crop to middle 60% (avoid scoreboard, crowd)
    h, w = frame_bgr.shape[:2]
    y1, y2 = int(h * 0.2), int(h * 0.8)
    x1, x2 = int(w * 0.2), int(w * 0.8)
    roi = frame_bgr[y1:y2, x1:x2]

    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define court color ranges in HSV
    # Blue hard court: hue ~100-130 (OpenCV hue is 0-179)
    blue_lower = np.array([90, 50, 50])
    blue_upper = np.array([125, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # Green court/grass: hue ~40-80
    green_lower = np.array([25, 40, 40])
    green_upper = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # Red/orange clay: hue ~0-20 and ~160-179 (wraps around)
    clay_lower1 = np.array([0, 50, 50])
    clay_upper1 = np.array([20, 255, 255])
    clay_mask1 = cv2.inRange(hsv, clay_lower1, clay_upper1)
    clay_lower2 = np.array([160, 50, 50])
    clay_upper2 = np.array([179, 255, 255])
    clay_mask2 = cv2.inRange(hsv, clay_lower2, clay_upper2)

    # Combine all masks
    court_mask = cv2.bitwise_or(blue_mask, green_mask)
    court_mask = cv2.bitwise_or(court_mask, clay_mask1)
    court_mask = cv2.bitwise_or(court_mask, clay_mask2)

    # Compute fraction of pixels that are court-colored
    total_pixels = roi.shape[0] * roi.shape[1]
    court_pixels = np.count_nonzero(court_mask)
    score = court_pixels / total_pixels if total_pixels > 0 else 0.0

    return score


def compute_sharpness(frame_bgr: np.ndarray) -> float:
    """Compute image sharpness using Laplacian variance.

    Low variance indicates blur (motion blur, defocus) or garbage frames.

    Args:
        frame_bgr: Input frame in BGR format.

    Returns:
        Laplacian variance (higher = sharper).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return float(variance)


def find_valid_frame_indices(
    video_path: str,
    sample_interval_sec: float = 5.0,
    min_court_color: float = 0.15,
    min_sharpness: float = 100.0,
) -> list[int]:
    """Scan video and return frame indices that pass garbage filter.

    Phase A of the 2-phase approach: fast scan with CPU-only filtering.
    Returns only indices (not frames) to avoid storing all frames in memory.

    Args:
        video_path: Path to video file.
        sample_interval_sec: Sample every N seconds.
        min_court_color: Minimum court color score (0-1).
        min_sharpness: Minimum Laplacian variance.

    Returns:
        List of frame indices that passed the filters.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        logger.warning("Invalid FPS, using default 30")
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval_frames = max(1, int(fps * sample_interval_sec))

    logger.info(
        f"Scanning video: {total_frames} frames @ {fps:.1f} fps, "
        f"sampling every {sample_interval_sec}s ({sample_interval_frames} frames)"
    )

    valid_indices: list[int] = []
    frame_indices = range(0, total_frames, sample_interval_frames)

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        color_score = compute_court_color_score(frame)
        sharpness = compute_sharpness(frame)

        if color_score >= min_court_color and sharpness >= min_sharpness:
            valid_indices.append(frame_idx)
            logger.debug(
                f"Frame {frame_idx}: color={color_score:.3f}, "
                f"sharpness={sharpness:.1f} ✓"
            )
        else:
            logger.debug(
                f"Frame {frame_idx}: color={color_score:.3f}, "
                f"sharpness={sharpness:.1f} ✗ (rejected)"
            )

    cap.release()
    logger.info(f"Found {len(valid_indices)} valid frames out of {len(frame_indices)} sampled")
    return valid_indices


def select_best_court_frames(
    video_path: str,
    court_detector: "CourtDetector",
    max_candidates: int = 100,
    top_n: int = 3,
    sample_interval_sec: float = 5.0,
    min_court_color: float = 0.15,
    min_sharpness: float = 100.0,
) -> list["CourtDetectionResult"]:
    """Full pipeline: scan → filter → detect → return top N results.

    Phase A: Fast scan + filter (CPU-only)
    Phase B: Run court detector on surviving candidates (GPU)

    Args:
        video_path: Path to video file.
        court_detector: Initialized CourtDetector instance.
        max_candidates: Maximum frames to keep from Phase A.
        top_n: Number of best results to return.
        sample_interval_sec: Sampling interval in seconds.
        min_court_color: Minimum court color score threshold.
        min_sharpness: Minimum sharpness threshold.

    Returns:
        List of top N CourtDetectionResult objects, sorted by quality
        (most keypoints + lowest reprojection error).
    """
    # Phase A: Fast scan and filter (returns indices only — no frames in memory)
    logger.info("Phase A: Scanning for valid frames...")
    valid_indices = find_valid_frame_indices(
        video_path,
        sample_interval_sec=sample_interval_sec,
        min_court_color=min_court_color,
        min_sharpness=min_sharpness,
    )

    if not valid_indices:
        logger.warning("No valid frames found!")
        return []

    # Limit to max_candidates
    if len(valid_indices) > max_candidates:
        logger.info(f"Limiting to {max_candidates} candidates (from {len(valid_indices)})")
        step = len(valid_indices) / max_candidates
        valid_indices = [valid_indices[int(i * step)] for i in range(max_candidates)]

    # Phase B: Run court detector on candidates (read frames on-demand)
    logger.info(f"Phase B: Running court detector on {len(valid_indices)} frames...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video for Phase B: {video_path}")
        return []

    results = []
    for frame_idx in valid_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        result = court_detector.detect_frame(frame, frame_number=frame_idx)
        if result.detected:
            results.append(result)
            logger.debug(
                f"Frame {frame_idx}: {result.num_detected}/14 keypoints, "
                f"confidence={result.confidence:.3f}"
            )
        else:
            logger.debug(f"Frame {frame_idx}: detection failed")
    cap.release()

    if not results:
        logger.warning("No successful court detections!")
        return []

    # Sort by quality: most keypoints first, then by confidence (inverse reprojection error)
    results.sort(key=lambda r: (r.num_detected, r.confidence), reverse=True)

    # Return top N
    top_results = results[:top_n]
    logger.info(
        f"Selected top {len(top_results)} frames: "
        f"{[r.frame_number for r in top_results]}"
    )

    return top_results
