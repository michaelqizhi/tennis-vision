"""Court detection visualization — draw keypoints and court overlay on frames."""

import cv2
import numpy as np

from src.features.court_detect.detector import CourtDetectionResult


# Colors for different keypoint groups (BGR)
_BASELINE_COLOR = (0, 0, 255)       # Red — baselines
_SINGLES_COLOR = (0, 255, 0)        # Green — singles sidelines
_SERVICE_COLOR = (255, 165, 0)      # Orange — service line corners
_CENTER_COLOR = (255, 0, 255)       # Magenta — center service line

_KP_COLORS = [
    _BASELINE_COLOR, _BASELINE_COLOR, _BASELINE_COLOR, _BASELINE_COLOR,
    _SINGLES_COLOR, _SINGLES_COLOR, _SINGLES_COLOR, _SINGLES_COLOR,
    _SERVICE_COLOR, _SERVICE_COLOR, _SERVICE_COLOR, _SERVICE_COLOR,
    _CENTER_COLOR, _CENTER_COLOR,
]

_KP_LABELS = [
    "BL-TL", "BL-TR", "BL-BL", "BL-BR",
    "SL-TL", "SL-BL", "SL-TR", "SL-BR",
    "SV-TL", "SV-TR", "SV-BL", "SV-BR",
    "CT-T", "CT-B",
]

# Court lines connecting keypoint pairs
_COURT_LINES = [
    (0, 1),   # top baseline
    (2, 3),   # bottom baseline
    (0, 2),   # left doubles sideline
    (1, 3),   # right doubles sideline
    (4, 5),   # left singles sideline
    (6, 7),   # right singles sideline
    (8, 9),   # top service line
    (10, 11), # bottom service line
    (12, 13), # center service line
]


def draw_court_keypoints(
    frame: np.ndarray,
    result: CourtDetectionResult,
    draw_labels: bool = True,
    point_radius: int = 8,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw detected court keypoints and connecting lines on a frame.

    Args:
        frame: BGR frame to draw on (will be modified in-place).
        result: CourtDetectionResult with keypoints.
        draw_labels: Whether to draw keypoint labels.
        point_radius: Radius of keypoint circles.
        line_thickness: Thickness of court lines.

    Returns:
        The annotated frame.
    """
    kps = result.keypoints

    # Draw court lines between detected pairs
    for i, j in _COURT_LINES:
        if i < len(kps) and j < len(kps):
            p1, p2 = kps[i], kps[j]
            if p1[0] is not None and p2[0] is not None:
                pt1 = (int(p1[0]), int(p1[1]))
                pt2 = (int(p2[0]), int(p2[1]))
                cv2.line(frame, pt1, pt2, (0, 255, 255), line_thickness)

    # Draw keypoints
    for idx, (x, y) in enumerate(kps):
        if x is None or y is None:
            continue
        center = (int(x), int(y))
        color = _KP_COLORS[idx] if idx < len(_KP_COLORS) else (255, 255, 255)
        cv2.circle(frame, center, point_radius, color, thickness=-1)
        cv2.circle(frame, center, point_radius, (255, 255, 255), thickness=1)

        if draw_labels and idx < len(_KP_LABELS):
            label = _KP_LABELS[idx]
            cv2.putText(
                frame, label,
                (center[0] + 10, center[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
            )

    # Draw confidence info
    info = f"Court: {result.num_detected}/14 kps ({result.confidence:.0%})"
    cv2.putText(
        frame, info, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )

    return frame


def annotate_court_video(
    frames: list[np.ndarray],
    results: list[CourtDetectionResult],
    draw_labels: bool = False,
) -> list[np.ndarray]:
    """Annotate video frames with court detection results.

    If results were sampled (fewer results than frames), the last available
    result is reused for intermediate frames.

    Args:
        frames: List of BGR video frames.
        results: List of CourtDetectionResult (may be fewer than frames).
        draw_labels: Whether to draw keypoint labels.

    Returns:
        List of annotated frames.
    """
    # Build a mapping of frame_number → result
    result_map: dict[int, CourtDetectionResult] = {
        r.frame_number: r for r in results
    }

    annotated: list[np.ndarray] = []
    last_result: CourtDetectionResult | None = None

    for i, frame in enumerate(frames):
        frame_copy = frame.copy()

        if i in result_map:
            last_result = result_map[i]

        if last_result is not None:
            draw_court_keypoints(frame_copy, last_result, draw_labels=draw_labels)

        annotated.append(frame_copy)

    return annotated
