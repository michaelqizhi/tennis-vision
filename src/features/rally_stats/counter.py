"""Shot counting — count shots per rally using ball trajectory analysis.

Counts shots by detecting net crossings (y=0 in court coordinates) or
y-direction reversals in the ball trajectory. Each net crossing or
direction change represents a shot exchange.
"""

from dataclasses import dataclass

import numpy as np

from src.features.ball_tracking.detector import BallDetection
from src.features.auto_clip.rally_detector import Rally


@dataclass
class RallyShots:
    """Shot count data for a single rally."""
    rally_id: int
    shot_count: int
    net_crossings: int
    start_frame: int
    end_frame: int
    duration: float


def count_shots_pixel(
    detections: list[BallDetection],
    rally: Rally,
    frame_height: int,
) -> RallyShots:
    """Count shots in a rally using pixel-space y-direction changes.

    Uses the approximate screen midpoint as a proxy for the net when
    court homography is not available.

    Args:
        detections: Full list of ball detections.
        rally: Rally boundaries.
        frame_height: Video frame height (used to estimate net position).

    Returns:
        RallyShots with shot count.
    """
    rally_dets = [
        d for d in detections
        if rally.start_frame <= d.frame_number <= rally.end_frame and d.detected
    ]

    if len(rally_dets) < 2:
        return RallyShots(
            rally_id=rally.rally_id,
            shot_count=0,
            net_crossings=0,
            start_frame=rally.start_frame,
            end_frame=rally.end_frame,
            duration=rally.duration,
        )

    # Count direction reversals in y-axis (ball going up vs down)
    y_values = [d.y for d in rally_dets if d.y is not None]
    direction_changes = _count_direction_changes(y_values, min_displacement=20.0)

    # Count crossings of the approximate net line (midpoint of frame)
    net_y = frame_height / 2.0
    net_crossings = _count_crossings(y_values, net_y)

    # Use the higher of the two metrics as shot count
    # +1 because the first shot doesn't produce a direction change
    shot_count = max(direction_changes, net_crossings) + 1

    return RallyShots(
        rally_id=rally.rally_id,
        shot_count=shot_count,
        net_crossings=net_crossings,
        start_frame=rally.start_frame,
        end_frame=rally.end_frame,
        duration=rally.duration,
    )


def count_shots_court(
    detections: list[BallDetection],
    rally: Rally,
    homography: np.ndarray,
) -> RallyShots:
    """Count shots in a rally using court-coordinate net crossings.

    Uses homography to transform ball positions to court coordinates
    where y=0 is the net. Much more accurate than pixel-space counting.

    Args:
        detections: Full list of ball detections.
        rally: Rally boundaries.
        homography: 3x3 pixel-to-court homography matrix.

    Returns:
        RallyShots with shot count.

    Raises:
        ValueError: If homography is not a 3x3 matrix.
    """
    import cv2

    if not isinstance(homography, np.ndarray) or homography.shape != (3, 3):
        raise ValueError(
            f"homography must be a 3x3 numpy array, got shape {getattr(homography, 'shape', 'N/A')}"
        )

    rally_dets = [
        d for d in detections
        if rally.start_frame <= d.frame_number <= rally.end_frame and d.detected
    ]

    if len(rally_dets) < 2:
        return RallyShots(
            rally_id=rally.rally_id,
            shot_count=0,
            net_crossings=0,
            start_frame=rally.start_frame,
            end_frame=rally.end_frame,
            duration=rally.duration,
        )

    # Transform pixel positions to court coordinates
    pixel_pts = np.array(
        [[d.x, d.y] for d in rally_dets],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    court_pts = cv2.perspectiveTransform(pixel_pts, homography)
    court_y = court_pts[:, 0, 1].tolist()

    # Count net crossings (y=0 in court coordinates)
    net_crossings = _count_crossings(court_y, crossing_value=0.0)

    # Count direction reversals in court y
    direction_changes = _count_direction_changes(court_y, min_displacement=1.0)

    shot_count = max(direction_changes, net_crossings) + 1

    return RallyShots(
        rally_id=rally.rally_id,
        shot_count=shot_count,
        net_crossings=net_crossings,
        start_frame=rally.start_frame,
        end_frame=rally.end_frame,
        duration=rally.duration,
    )


def count_all_rally_shots(
    detections: list[BallDetection],
    rallies: list[Rally],
    homography: np.ndarray | None = None,
    frame_height: int = 720,
) -> list[RallyShots]:
    """Count shots for all detected rallies.

    Uses court-coordinate counting if homography is available,
    falling back to pixel-space counting otherwise.

    Args:
        detections: Full list of ball detections.
        rallies: List of detected rallies.
        homography: Optional pixel-to-court homography.
        frame_height: Video frame height (used for pixel-space fallback).

    Returns:
        List of RallyShots, one per rally.
    """
    results: list[RallyShots] = []
    for rally in rallies:
        if homography is not None:
            shots = count_shots_court(detections, rally, homography)
        else:
            shots = count_shots_pixel(detections, rally, frame_height)
        results.append(shots)
    return results


def _count_direction_changes(
    values: list[float],
    min_displacement: float = 1.0,
) -> int:
    """Count significant direction reversals in a 1D signal.

    Only counts reversals where the displacement since the last
    reversal exceeds min_displacement to filter out noise.

    Args:
        values: 1D signal values.
        min_displacement: Minimum displacement to count as a reversal.

    Returns:
        Number of direction changes.
    """
    if len(values) < 3:
        return 0

    changes = 0
    last_extreme = values[0]
    direction = 0  # 0=unknown, 1=increasing, -1=decreasing

    for i in range(1, len(values)):
        diff = values[i] - last_extreme

        if abs(diff) < min_displacement:
            continue

        new_direction = 1 if diff > 0 else -1

        if direction != 0 and new_direction != direction:
            changes += 1
            last_extreme = values[i - 1]
        elif new_direction == direction:
            last_extreme = values[i]

        if direction == 0:
            last_extreme = values[i]

        direction = new_direction

    return changes


def _count_crossings(
    values: list[float],
    crossing_value: float = 0.0,
) -> int:
    """Count the number of times a signal crosses a threshold.

    Args:
        values: 1D signal values.
        crossing_value: The value to detect crossings of.

    Returns:
        Number of crossings.
    """
    if len(values) < 2:
        return 0

    crossings = 0
    prev_side = values[0] - crossing_value

    for v in values[1:]:
        curr_side = v - crossing_value
        if prev_side * curr_side < 0:
            crossings += 1
        if curr_side != 0:
            prev_side = curr_side

    return crossings
