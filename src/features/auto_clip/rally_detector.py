"""Rally boundary detection from ball tracking data.

Detects rally start/end times by analyzing gaps in ball detections,
ball trajectory resets, and movement patterns. A rally is a continuous
period of ball activity; dead time between rallies indicates point boundaries.
"""

from dataclasses import dataclass

from src.features.ball_tracking.detector import BallDetection


@dataclass
class Rally:
    """A detected rally (point) with start/end frame information."""
    rally_id: int
    start_frame: int
    end_frame: int
    fps: float
    num_detections: int

    @property
    def start_time(self) -> float:
        """Start time in seconds."""
        return self.start_frame / self.fps if self.fps > 0 else 0.0

    @property
    def end_time(self) -> float:
        """End time in seconds."""
        return self.end_frame / self.fps if self.fps > 0 else 0.0

    @property
    def duration(self) -> float:
        """Rally duration in seconds."""
        return self.end_time - self.start_time

    @property
    def detection_density(self) -> float:
        """Fraction of frames with ball detections."""
        total_frames = self.end_frame - self.start_frame + 1
        return self.num_detections / total_frames if total_frames > 0 else 0.0


@dataclass
class RallyDetectorConfig:
    """Configuration for rally boundary detection."""
    min_gap_seconds: float = 1.5
    min_rally_seconds: float = 1.0
    min_detections: int = 5
    min_detection_density: float = 0.15
    # Large position jump threshold (pixels) — indicates a rally boundary
    # even when continuous detections exist (e.g., ball reset after point)
    position_reset_threshold: float = 400.0
    # Minimum consecutive reset frames to split at
    min_reset_frames: int = 3
    # Padding added before/after detected rally boundaries (seconds)
    pad_start_seconds: float = 0.5
    pad_end_seconds: float = 1.0


def detect_rallies(
    detections: list[BallDetection],
    fps: float,
    config: RallyDetectorConfig | None = None,
) -> list[Rally]:
    """Detect rally boundaries from ball tracking detections.

    Algorithm:
    1. Find all frames with valid ball detections.
    2. Identify gaps longer than min_gap_seconds — these separate rallies.
    3. Filter out segments shorter than min_rally_seconds or with too
       few detections (likely false positives).

    Args:
        detections: Ball detections from BallTracker, one per frame.
        fps: Video frame rate (frames per second).
        config: Detection configuration.

    Returns:
        List of Rally objects, sorted by start frame.
    """
    config = config or RallyDetectorConfig()

    if not detections or fps <= 0:
        return []

    min_gap_frames = int(config.min_gap_seconds * fps)
    min_rally_frames = int(config.min_rally_seconds * fps)

    # Find frames with valid non-interpolated detections for gap analysis.
    # Using raw (non-interpolated) detections prevents interpolation from
    # filling gaps that represent real rally boundaries.
    raw_detections = [
        d for d in detections
        if d.detected and not d.interpolated
    ]
    detected_frames = sorted(d.frame_number for d in raw_detections)

    if len(detected_frames) < config.min_detections:
        return []

    # Build a lookup of frame_number → detection for position reset analysis
    det_by_frame = {d.frame_number: d for d in raw_detections}

    # Find position-reset frames: large jumps between consecutive raw detections
    # indicate ball pickup / serve restart (rally boundary even without gap)
    reset_frames: set[int] = set()
    sorted_raw = sorted(raw_detections, key=lambda d: d.frame_number)
    for i in range(1, len(sorted_raw)):
        prev = sorted_raw[i - 1]
        curr = sorted_raw[i]
        if prev.x is not None and curr.x is not None:
            dx = curr.x - prev.x  # type: ignore[operator]
            dy = curr.y - prev.y  # type: ignore[operator]
            dist = (dx * dx + dy * dy) ** 0.5
            frame_gap = curr.frame_number - prev.frame_number
            # Large jump within a short time window signals a reset
            if dist > config.position_reset_threshold and frame_gap <= fps * 0.5:
                reset_frames.add(curr.frame_number)

    # Inject virtual gaps at reset frames so _split_at_gaps treats them as boundaries
    effective_frames = [f for f in detected_frames if f not in reset_frames]
    if len(effective_frames) < config.min_detections:
        effective_frames = detected_frames  # fall back if too aggressive

    # Split into segments at gaps
    segments = _split_at_gaps(effective_frames, min_gap_frames)

    # Convert segments to rallies, filtering by minimum criteria
    rallies: list[Rally] = []
    rally_id = 1

    for seg_start, seg_end, num_dets in segments:
        duration_frames = seg_end - seg_start + 1
        if duration_frames < min_rally_frames:
            continue
        if num_dets < config.min_detections:
            continue

        density = num_dets / duration_frames
        if density < config.min_detection_density:
            continue

        rallies.append(Rally(
            rally_id=rally_id,
            start_frame=seg_start,
            end_frame=seg_end,
            fps=fps,
            num_detections=num_dets,
        ))
        rally_id += 1

    return rallies


def _split_at_gaps(
    detected_frames: list[int],
    min_gap_frames: int,
) -> list[tuple[int, int, int]]:
    """Split detected frames into contiguous segments separated by gaps.

    Args:
        detected_frames: Sorted list of frame numbers with detections.
        min_gap_frames: Minimum gap size (frames) to split segments.

    Returns:
        List of (start_frame, end_frame, num_detections) tuples.
    """
    if not detected_frames:
        return []

    segments: list[tuple[int, int, int]] = []
    seg_start = detected_frames[0]
    seg_end = detected_frames[0]
    num_dets = 1

    for i in range(1, len(detected_frames)):
        gap = detected_frames[i] - detected_frames[i - 1]
        if gap >= min_gap_frames:
            segments.append((seg_start, seg_end, num_dets))
            seg_start = detected_frames[i]
            num_dets = 0
        seg_end = detected_frames[i]
        num_dets += 1

    segments.append((seg_start, seg_end, num_dets))
    return segments


def get_rally_clips(
    rallies: list[Rally],
    total_frames: int,
    fps: float,
    config: RallyDetectorConfig | None = None,
) -> list[tuple[int, int]]:
    """Get frame ranges for rally clips with padding.

    Adds configurable padding before and after each rally to capture
    the full action (serve windup, celebration, etc.).

    Args:
        rallies: Detected rallies.
        total_frames: Total number of frames in the video.
        fps: Video frame rate.
        config: Configuration for padding.

    Returns:
        List of (start_frame, end_frame) tuples for clipping.
    """
    config = config or RallyDetectorConfig()
    pad_start = int(config.pad_start_seconds * fps)
    pad_end = int(config.pad_end_seconds * fps)

    clips: list[tuple[int, int]] = []
    for rally in rallies:
        start = max(0, rally.start_frame - pad_start)
        end = min(total_frames - 1, rally.end_frame + pad_end)
        clips.append((start, end))

    return clips
