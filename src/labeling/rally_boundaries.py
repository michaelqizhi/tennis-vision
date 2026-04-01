"""Rally boundary detection from detection density.

Uses a sliding window over per-frame detection density to segment a video
into rally and dead-time regions. This is independent of the main API's
``auto_clip`` module — it works on the labeling pipeline's multi-model
consensus data rather than single-model ball tracking.

Algorithm:
  1. For each frame, compute whether *any* model detected the ball.
  2. Slide a window across the binary detection signal.
  3. Windows with ≥ ``rally_threshold`` detection rate → rally.
  4. Windows with < ``dead_threshold`` → dead time.
  5. Merge adjacent rally windows and apply minimum duration filters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.labeling.consensus import DetectionLabel, FrameConsensus


@dataclass
class RallyBoundary:
    """A detected rally or dead-time segment."""
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    detection_rate: float
    is_rally: bool

    @property
    def duration_sec(self) -> float:
        """Duration in seconds."""
        return self.end_sec - self.start_sec


@dataclass
class RallyBoundaryConfig:
    """Configuration for rally boundary detection."""
    # Sliding window size in seconds
    window_size_sec: float = 2.0
    # Minimum detection rate to classify a window as "rally"
    rally_threshold: float = 0.30
    # Maximum detection rate for "dead time" (below this = dead)
    dead_threshold: float = 0.10
    # Minimum rally duration in seconds (shorter rallies are discarded)
    min_rally_duration_sec: float = 3.0
    # Minimum gap duration in seconds (shorter gaps are merged with adjacent rallies)
    min_gap_duration_sec: float = 4.0


def detect_rally_boundaries(
    consensus: list[FrameConsensus],
    fps: float,
    config: Optional[RallyBoundaryConfig] = None,
) -> list[RallyBoundary]:
    """Detect rally boundaries from consensus detection density.

    Args:
        consensus: Per-frame consensus results from the labeling pipeline.
        fps: Video frame rate.
        config: Detection configuration (uses defaults if not provided).

    Returns:
        List of ``RallyBoundary`` objects in chronological order.

    Raises:
        ValueError: If fps ≤ 0, consensus is empty, or config values are invalid.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if not consensus:
        return []

    config = config or RallyBoundaryConfig()
    _validate_config(config)
    window_frames = max(1, int(config.window_size_sec * fps))

    # Build binary detection signal (1 = any detection, 0 = none)
    max_frame = max(fc.frame_idx for fc in consensus)
    detection_signal = [0] * (max_frame + 1)
    for fc in consensus:
        if fc.label != DetectionLabel.NO_DETECTION:
            detection_signal[fc.frame_idx] = 1

    total_frames = len(detection_signal)
    if total_frames < window_frames:
        # Video shorter than one window — treat as single segment
        rate = sum(detection_signal) / total_frames if total_frames > 0 else 0
        is_rally = rate >= config.rally_threshold
        return [RallyBoundary(
            start_frame=0,
            end_frame=total_frames - 1,
            start_sec=0.0,
            end_sec=(total_frames - 1) / fps,
            detection_rate=round(rate, 4),
            is_rally=is_rally,
        )]

    # Sliding window: compute per-frame detection density
    # Use prefix sum for efficient window computation
    prefix = [0] * (total_frames + 1)
    for i in range(total_frames):
        prefix[i + 1] = prefix[i] + detection_signal[i]

    # Compute detection rate at each frame (centred window)
    half_window = window_frames // 2
    frame_is_rally = [False] * total_frames
    for i in range(total_frames):
        win_start = max(0, i - half_window)
        win_end = min(total_frames, i + half_window + 1)
        win_len = win_end - win_start
        win_sum = prefix[win_end] - prefix[win_start]
        rate = win_sum / win_len
        frame_is_rally[i] = rate >= config.rally_threshold

    # Segment into contiguous runs
    segments: list[RallyBoundary] = []
    seg_start = 0
    seg_rally = frame_is_rally[0]

    for i in range(1, total_frames):
        if frame_is_rally[i] != seg_rally:
            _add_segment(segments, seg_start, i - 1, seg_rally, detection_signal, fps)
            seg_start = i
            seg_rally = frame_is_rally[i]
    _add_segment(segments, seg_start, total_frames - 1, seg_rally, detection_signal, fps)

    # Merge short gaps between rallies
    segments = _merge_short_gaps(segments, config.min_gap_duration_sec, detection_signal, fps)

    # Filter out short rallies
    segments = _filter_short_rallies(segments, config.min_rally_duration_sec)

    # Re-merge: after filtering short rallies, newly adjacent rallies may need merging
    segments = _merge_adjacent_rallies(segments)

    # Recompute detection rates after merging
    for seg in segments:
        seg_len = seg.end_frame - seg.start_frame + 1
        if seg_len > 0:
            seg.detection_rate = round(
                sum(detection_signal[seg.start_frame:seg.end_frame + 1]) / seg_len, 4
            )

    return segments


def _validate_config(config: RallyBoundaryConfig) -> None:
    """Validate rally boundary config values.

    Raises:
        ValueError: If any config value is out of valid range.
    """
    if config.window_size_sec <= 0:
        raise ValueError(f"window_size_sec must be positive, got {config.window_size_sec}")
    if not 0.0 <= config.rally_threshold <= 1.0:
        raise ValueError(f"rally_threshold must be in [0, 1], got {config.rally_threshold}")
    if not 0.0 <= config.dead_threshold <= 1.0:
        raise ValueError(f"dead_threshold must be in [0, 1], got {config.dead_threshold}")
    if config.min_rally_duration_sec < 0:
        raise ValueError(f"min_rally_duration_sec must be non-negative, got {config.min_rally_duration_sec}")
    if config.min_gap_duration_sec < 0:
        raise ValueError(f"min_gap_duration_sec must be non-negative, got {config.min_gap_duration_sec}")


def _add_segment(
    segments: list[RallyBoundary],
    start_frame: int,
    end_frame: int,
    is_rally: bool,
    detection_signal: list[int],
    fps: float,
) -> None:
    """Add a segment to the list."""
    seg_len = end_frame - start_frame + 1
    if seg_len <= 0:
        return
    rate = sum(detection_signal[start_frame:end_frame + 1]) / seg_len
    segments.append(RallyBoundary(
        start_frame=start_frame,
        end_frame=end_frame,
        start_sec=round(start_frame / fps, 3),
        end_sec=round(end_frame / fps, 3),
        detection_rate=round(rate, 4),
        is_rally=is_rally,
    ))


def _merge_short_gaps(
    segments: list[RallyBoundary],
    min_gap_sec: float,
    detection_signal: list[int],
    fps: float,
) -> list[RallyBoundary]:
    """Merge non-rally gaps shorter than min_gap_sec with adjacent rallies."""
    if len(segments) <= 1:
        return segments

    merged: list[RallyBoundary] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if (
            not seg.is_rally
            and seg.duration_sec < min_gap_sec
            and prev.is_rally
        ):
            # Merge short gap into the previous rally
            prev.end_frame = seg.end_frame
            prev.end_sec = seg.end_sec
        elif (
            seg.is_rally
            and not prev.is_rally
            and prev.duration_sec < min_gap_sec
            and len(merged) >= 2
            and merged[-2].is_rally
        ):
            # Short gap between two rallies — merge all three
            rally_before = merged[-2]
            rally_before.end_frame = seg.end_frame
            rally_before.end_sec = seg.end_sec
            merged.pop()  # remove the short gap
        elif seg.is_rally and prev.is_rally:
            # Adjacent rallies — merge
            prev.end_frame = seg.end_frame
            prev.end_sec = seg.end_sec
        else:
            merged.append(seg)

    return merged


def _filter_short_rallies(
    segments: list[RallyBoundary],
    min_duration_sec: float,
) -> list[RallyBoundary]:
    """Remove rally segments shorter than min_duration_sec."""
    return [
        seg for seg in segments
        if not seg.is_rally or seg.duration_sec >= min_duration_sec
    ]


def _merge_adjacent_rallies(
    segments: list[RallyBoundary],
) -> list[RallyBoundary]:
    """Merge rally segments that became adjacent after filtering.

    After short rallies are removed, two rally segments may be separated
    only by a now-removed segment. This function merges them.
    """
    if len(segments) <= 1:
        return segments

    merged: list[RallyBoundary] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg.is_rally and prev.is_rally:
            # Adjacent rallies — merge
            prev.end_frame = seg.end_frame
            prev.end_sec = seg.end_sec
        else:
            merged.append(seg)
    return merged


def serialize_rally_boundaries(
    boundaries: list[RallyBoundary],
) -> list[dict]:
    """Serialize rally boundaries to JSON-compatible list of dicts."""
    return [
        {
            "start_frame": rb.start_frame,
            "end_frame": rb.end_frame,
            "start_sec": rb.start_sec,
            "end_sec": rb.end_sec,
            "detection_rate": rb.detection_rate,
            "is_rally": rb.is_rally,
            "duration_sec": round(rb.duration_sec, 3),
        }
        for rb in boundaries
    ]
