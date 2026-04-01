"""Serve event detection from ball tracking data and rally boundaries.

Identifies serve events at the start of each rally and in gaps between
rallies (faults). Uses ball trajectory direction and court coordinates
to determine server position and serve landing location.
"""

from dataclasses import dataclass

import numpy as np

from src.features.ball_tracking.detector import BallDetection
from src.features.auto_clip.rally_detector import Rally


@dataclass
class ServeEvent:
    """A detected serve event."""
    serve_id: int
    rally_id: int | None  # None for faults that don't start a rally
    server_end: str  # "near" (y>0 baseline) or "far" (y<0 baseline)
    serve_number: int  # 1 or 2 (second serve follows a fault)
    start_frame: int
    end_frame: int
    landing_position: tuple[float, float] | None  # court coords (x, y) meters
    is_fault: bool
    is_ace: bool  # serve that wins the point without return

    @property
    def is_in(self) -> bool:
        """Whether the serve landed in the service box (not a fault)."""
        return not self.is_fault


@dataclass
class ServeDetectorConfig:
    """Configuration for serve detection."""
    # Minimum ball detections to consider a burst as a serve
    min_serve_detections: int = 3
    # Maximum duration of a serve burst in seconds (toss + hit + land)
    max_serve_duration_seconds: float = 3.0
    # Minimum gap before a burst to consider it a serve (seconds)
    min_gap_before_seconds: float = 1.0
    # How many initial detections of a rally to analyze for serve direction
    serve_analysis_frames: int = 15
    # Minimum y-displacement (meters) to determine serve direction
    min_y_displacement_meters: float = 2.0
    # Baseline proximity threshold (meters from baseline)
    baseline_proximity_meters: float = 3.0
    # Maximum shots in a rally for it to be an ace candidate
    ace_max_shots: int = 1


def detect_serves(
    detections: list[BallDetection],
    rallies: list[Rally],
    homography: np.ndarray,
    fps: float,
    config: ServeDetectorConfig | None = None,
) -> list[ServeEvent]:
    """Detect serve events from ball tracking data and rally boundaries.

    Algorithm:
    1. For each rally, analyze the ball trajectory at the start to identify
       the serve direction and server end.
    2. Find short ball-activity bursts in gaps between rallies — these are
       serve faults that didn't start a rally.
    3. Classify each serve as in/fault and 1st/2nd serve.

    Args:
        detections: Ball detections from tracking.
        rallies: Detected rallies from rally_detector.
        homography: 3x3 pixel-to-court homography matrix.
        fps: Video frame rate.
        config: Detection configuration.

    Returns:
        List of ServeEvent objects in chronological order.

    Raises:
        ValueError: If homography is not a 3x3 matrix.
    """
    if not isinstance(homography, np.ndarray) or homography.shape != (3, 3):
        raise ValueError(
            f"homography must be a 3x3 numpy array, got shape "
            f"{getattr(homography, 'shape', 'N/A')}"
        )

    config = config or ServeDetectorConfig()

    if not detections or not rallies or fps <= 0:
        return []

    # Find fault bursts in gaps between rallies
    fault_bursts = _find_fault_bursts(detections, rallies, fps, config)

    # Build chronological serve events
    serves: list[ServeEvent] = []
    serve_id = 1
    pending_fault: ServeEvent | None = None

    events: list[tuple[str, int]] = []
    for i, rally in enumerate(rallies):
        events.append(("rally", i))
    for i, burst in enumerate(fault_bursts):
        events.append(("fault", i))
    events.sort(key=lambda e: (
        rallies[e[1]].start_frame if e[0] == "rally"
        else fault_bursts[e[1]][0]
    ))

    for event_type, idx in events:
        if event_type == "fault":
            burst_start, burst_end, burst_dets = fault_bursts[idx]
            landing = _find_landing_position(burst_dets, homography)
            server_end = _determine_server_end(burst_dets, homography, config)

            serve_number = 2 if pending_fault is not None else 1

            serve = ServeEvent(
                serve_id=serve_id,
                rally_id=None,
                server_end=server_end,
                serve_number=serve_number,
                start_frame=burst_start,
                end_frame=burst_end,
                landing_position=landing,
                is_fault=True,
                is_ace=False,
            )
            serves.append(serve)
            serve_id += 1
            pending_fault = serve

        else:  # rally
            rally = rallies[idx]
            rally_dets = [
                d for d in detections
                if rally.start_frame <= d.frame_number
                <= min(rally.start_frame + int(config.serve_analysis_frames * fps / 30), rally.end_frame)
                and d.detected
            ]

            server_end = _determine_server_end(rally_dets, homography, config)
            landing = _find_landing_position(rally_dets, homography)

            serve_number = 2 if pending_fault is not None else 1

            # Check if this rally is short enough to be an ace
            is_ace = _check_ace(
                detections, rally, homography, config,
            )

            serve = ServeEvent(
                serve_id=serve_id,
                rally_id=rally.rally_id,
                server_end=server_end,
                serve_number=serve_number,
                start_frame=rally.start_frame,
                end_frame=min(rally.start_frame + int(config.serve_analysis_frames * fps / 30), rally.end_frame),
                landing_position=landing,
                is_fault=False,
                is_ace=is_ace,
            )
            serves.append(serve)
            serve_id += 1
            pending_fault = None  # Rally started, reset fault tracking

    return serves


def _find_fault_bursts(
    detections: list[BallDetection],
    rallies: list[Rally],
    fps: float,
    config: ServeDetectorConfig,
) -> list[tuple[int, int, list[BallDetection]]]:
    """Find short ball-activity bursts in gaps between rallies.

    These are serve faults — short bursts of ball activity that didn't
    produce a full rally.

    Returns:
        List of (start_frame, end_frame, detections) for each burst.
    """
    max_burst_frames = int(config.max_serve_duration_seconds * fps)
    min_gap_frames = int(config.min_gap_before_seconds * fps)

    # Build list of "protected" frame ranges (rallies)
    rally_ranges = [(r.start_frame, r.end_frame) for r in rallies]

    def is_in_rally(frame: int) -> bool:
        return any(start <= frame <= end for start, end in rally_ranges)

    # Find detections NOT in any rally
    gap_detections = [d for d in detections if d.detected and not is_in_rally(d.frame_number)]

    if not gap_detections:
        return []

    # Sort by frame number
    gap_detections.sort(key=lambda d: d.frame_number)

    # Cluster gap detections into bursts
    bursts: list[tuple[int, int, list[BallDetection]]] = []
    current_burst: list[BallDetection] = [gap_detections[0]]

    for d in gap_detections[1:]:
        gap = d.frame_number - current_burst[-1].frame_number
        if gap > min_gap_frames:
            # End current burst, start new one
            if len(current_burst) >= config.min_serve_detections:
                start = current_burst[0].frame_number
                end = current_burst[-1].frame_number
                duration = end - start
                if duration <= max_burst_frames:
                    bursts.append((start, end, list(current_burst)))
            current_burst = []
        current_burst.append(d)

    # Don't forget the last burst
    if len(current_burst) >= config.min_serve_detections:
        start = current_burst[0].frame_number
        end = current_burst[-1].frame_number
        duration = end - start
        if duration <= max_burst_frames:
            bursts.append((start, end, list(current_burst)))

    return bursts


def _determine_server_end(
    serve_detections: list[BallDetection],
    homography: np.ndarray,
    config: ServeDetectorConfig,
) -> str:
    """Determine which end of the court the server is on.

    Analyzes the initial ball trajectory direction. If the ball moves
    from positive y toward negative y, the server is at the "near" end.

    Args:
        serve_detections: First few detections of the serve.
        homography: Pixel-to-court homography.
        config: Detector configuration.

    Returns:
        "near" or "far" indicating server's baseline.
    """
    if len(serve_detections) < 2:
        return "near"  # default

    court_points = _transform_detections(serve_detections, homography)
    if len(court_points) < 2:
        return "near"

    # Look at net y-displacement of the ball
    first_y = court_points[0][1]
    last_y = court_points[-1][1]
    displacement = last_y - first_y

    if abs(displacement) < config.min_y_displacement_meters:
        # Insufficient displacement — guess based on starting position
        from src.features.court_detect.court_template import ITF_DIMS
        if first_y > 0:
            return "near"
        return "far"

    # Ball moving from positive to negative y → server at near (positive y) end
    if displacement < 0:
        return "near"
    return "far"


def _find_landing_position(
    serve_detections: list[BallDetection],
    homography: np.ndarray,
) -> tuple[float, float] | None:
    """Estimate where the serve landed in court coordinates.

    Uses the last few detections of the serve trajectory to estimate
    the landing position (approximated as the point closest to y=0
    on the receiving side, or the last detected position).

    Returns:
        (x_meters, y_meters) court coordinates, or None if not enough data.
    """
    court_points = _transform_detections(serve_detections, homography)
    if not court_points:
        return None

    if len(court_points) == 1:
        return court_points[0]

    # Find the trajectory's endpoint (last detection in the sequence)
    # For a serve, this approximates the landing position
    # In practice, the ball tracking might lose the ball at the bounce,
    # so the last detected position is our best estimate
    return court_points[-1]


def _check_ace(
    detections: list[BallDetection],
    rally: Rally,
    homography: np.ndarray,
    config: ServeDetectorConfig,
) -> bool:
    """Check if a rally could be an ace (serve wins without return).

    An ace candidate is a very short rally with minimal shot exchanges.
    We check for net crossings — an ace should have at most one (the serve itself).
    """
    rally_dets = [
        d for d in detections
        if rally.start_frame <= d.frame_number <= rally.end_frame and d.detected
    ]

    court_points = _transform_detections(rally_dets, homography)
    if len(court_points) < 2:
        return False

    # Count net crossings (y=0)
    y_vals = [p[1] for p in court_points]
    crossings = 0
    for i in range(1, len(y_vals)):
        if y_vals[i - 1] * y_vals[i] < 0:
            crossings += 1

    # Ace: short rally with at most 1 net crossing (the serve itself)
    return crossings <= config.ace_max_shots and rally.duration < 4.0


def _transform_detections(
    dets: list[BallDetection],
    homography: np.ndarray,
) -> list[tuple[float, float]]:
    """Transform ball detections from pixel to court coordinates.

    Args:
        dets: Ball detections with pixel positions.
        homography: 3x3 pixel-to-court homography.

    Returns:
        List of (x_meters, y_meters) court positions.
    """
    import cv2

    valid = [(d.x, d.y) for d in dets if d.detected and d.x is not None and d.y is not None]
    if not valid:
        return []

    pixel_pts = np.array(valid, dtype=np.float32).reshape(-1, 1, 2)
    court_pts = cv2.perspectiveTransform(pixel_pts, homography)
    return [(float(p[0][0]), float(p[0][1])) for p in court_pts]
