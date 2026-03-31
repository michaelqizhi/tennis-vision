"""Ball tracking visualization — draw detected ball positions on video frames."""

import cv2
import numpy as np

from src.features.ball_tracking.detector import BallDetection


def draw_ball_detections(
    frame: np.ndarray,
    detections: list[BallDetection],
    current_frame: int,
    trace_length: int = 7,
    color: tuple[int, int, int] = (0, 255, 0),
    radius: int = 5,
) -> np.ndarray:
    """Draw ball detection(s) with a fading trail on a single frame.

    Args:
        frame: BGR frame to draw on (will be modified in-place).
        detections: Full list of BallDetection for all frames.
        current_frame: Index of the current frame to render.
        trace_length: Number of trailing frames to show.
        color: BGR color for the ball marker.
        radius: Base radius of the ball circle.

    Returns:
        The annotated frame.
    """
    for i in range(trace_length):
        idx = current_frame - i
        if idx < 0 or idx >= len(detections):
            break
        det = detections[idx]
        if not det.detected:
            break
        x = int(det.x)
        y = int(det.y)
        # Fade trail: reduce radius and opacity for older positions
        trail_radius = max(1, radius - i)
        alpha = 1.0 - (i / trace_length) * 0.6
        trail_color = tuple(int(c * alpha) for c in color)
        cv2.circle(frame, (x, y), trail_radius, trail_color, thickness=-1)

    return frame


def annotate_video_frames(
    frames: list[np.ndarray],
    detections: list[BallDetection],
    trace_length: int = 7,
    color: tuple[int, int, int] = (0, 255, 0),
    radius: int = 5,
) -> list[np.ndarray]:
    """Annotate all video frames with ball detections.

    Args:
        frames: List of BGR video frames.
        detections: List of BallDetection, one per frame.
        trace_length: Number of trailing frames to show.
        color: BGR color for the ball marker.
        radius: Base radius of the ball circle.

    Returns:
        List of annotated frames.
    """
    annotated: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        frame_copy = frame.copy()
        draw_ball_detections(
            frame_copy, detections, i,
            trace_length=trace_length, color=color, radius=radius,
        )
        annotated.append(frame_copy)
    return annotated
