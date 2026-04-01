"""Review visualization overlay — generates a video with detection annotations.

Draws coloured circles and labels on each frame so a human reviewer can
quickly assess detection quality:

  - **Green circle**: consensus detection (≥2 models agree)
  - **Yellow circle**: uncertain (single model only)
  - **Red border**: no detection in this frame
  - **Model labels**: small text showing which models detected the ball
  - **Frame counter**: current frame / total frames overlay
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.labeling.consensus import DetectionLabel, FrameConsensus
from src.video.reader import VideoReader

logger = logging.getLogger(__name__)

# Colours (BGR)
_GREEN = (0, 200, 0)
_YELLOW = (0, 220, 220)
_RED = (0, 0, 220)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)

# Drawing parameters
_CIRCLE_RADIUS = 12
_CIRCLE_THICKNESS = 2
_RED_BORDER_THICKNESS = 4
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE_LABEL = 0.4
_FONT_SCALE_COUNTER = 0.6
_FONT_THICKNESS = 1

# Player box drawing
_BLUE = (220, 130, 0)  # BGR blue for player boxes
_PLAYER_BOX_THICKNESS = 2


def generate_review_video(
    video_path: str,
    consensus: list[FrameConsensus],
    output_path: str,
    *,
    fps: Optional[float] = None,
    codec: str = "mp4v",
    max_frames: int = 0,
    player_detections: Optional[dict[int, list]] = None,
) -> str:
    """Generate a review overlay video from consensus results.

    Args:
        video_path: Path to the original source video.
        consensus: List of ``FrameConsensus`` objects (one per frame).
        output_path: Path to write the output ``.mp4`` file.
        fps: Override output FPS. If ``None``, uses source video FPS.
        codec: FourCC codec string for ``cv2.VideoWriter``.
        max_frames: Maximum frames to render (0 = all).
        player_detections: Optional dict of frame_idx → list[PlayerBox] for
            drawing player bounding boxes (blue rectangles).

    Returns:
        The ``output_path`` string.
    """
    # Build frame_idx → consensus lookup for O(1) access
    consensus_map: dict[int, FrameConsensus] = {
        fc.frame_idx: fc for fc in consensus
    }

    with VideoReader(video_path, max_frames=max_frames) as reader:
        out_fps = fps or reader.fps
        width = reader.width
        height = reader.height
        total_frames = reader.frame_count if max_frames == 0 else min(max_frames, reader.frame_count)

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))

        if not writer.isOpened():
            raise IOError(f"Cannot open video writer for: {output_path}")

        try:
            frame_idx = 0
            for frame in reader.iter_frames():
                fc = consensus_map.get(frame_idx)
                # Draw player bounding boxes first (behind ball overlay)
                if player_detections:
                    players = player_detections.get(frame_idx, [])
                    _draw_player_boxes(frame, players)
                _draw_overlay(frame, frame_idx, total_frames, fc)
                writer.write(frame)
                frame_idx += 1

                if frame_idx % 500 == 0:
                    logger.info("Rendered %d / %d frames", frame_idx, total_frames)
        finally:
            writer.release()

    logger.info(
        "Review video: %d frames written to %s", frame_idx, output_path
    )
    return output_path


def _draw_overlay(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    fc: Optional[FrameConsensus],
) -> None:
    """Draw detection overlay on a single frame (in-place)."""
    h, w = frame.shape[:2]

    if fc is None or fc.label == DetectionLabel.NO_DETECTION:
        # Red border for no-detection frames
        cv2.rectangle(
            frame,
            (0, 0),
            (w - 1, h - 1),
            _RED,
            _RED_BORDER_THICKNESS,
        )
    elif fc.x is not None and fc.y is not None:
        cx, cy = int(round(fc.x)), int(round(fc.y))

        if fc.label == DetectionLabel.CONSENSUS:
            colour = _GREEN
        else:
            colour = _YELLOW

        # Draw detection circle
        cv2.circle(frame, (cx, cy), _CIRCLE_RADIUS, colour, _CIRCLE_THICKNESS)
        # Small filled centre dot
        cv2.circle(frame, (cx, cy), 2, colour, -1)

        # Model labels — draw below the circle
        if fc.agreeing_models:
            label_text = ", ".join(fc.agreeing_models)
            text_y = cy + _CIRCLE_RADIUS + 15
            # Background rectangle for readability
            (tw, th), _ = cv2.getTextSize(label_text, _FONT, _FONT_SCALE_LABEL, _FONT_THICKNESS)
            text_x = cx - tw // 2
            cv2.rectangle(frame, (text_x - 2, text_y - th - 2), (text_x + tw + 2, text_y + 2), _BLACK, -1)
            cv2.putText(frame, label_text, (text_x, text_y), _FONT, _FONT_SCALE_LABEL, _WHITE, _FONT_THICKNESS)

        # Confidence text above circle
        if fc.confidence is not None:
            conf_text = f"{fc.confidence:.2f}"
            conf_y = cy - _CIRCLE_RADIUS - 5
            (cw, ch), _ = cv2.getTextSize(conf_text, _FONT, _FONT_SCALE_LABEL, _FONT_THICKNESS)
            conf_x = cx - cw // 2
            cv2.rectangle(frame, (conf_x - 2, conf_y - ch - 2), (conf_x + cw + 2, conf_y + 2), _BLACK, -1)
            cv2.putText(frame, conf_text, (conf_x, conf_y), _FONT, _FONT_SCALE_LABEL, colour, _FONT_THICKNESS)

    # Frame counter (top-left)
    counter_text = f"Frame {frame_idx}/{total_frames}"
    cv2.putText(frame, counter_text, (10, 25), _FONT, _FONT_SCALE_COUNTER, _BLACK, 3)
    cv2.putText(frame, counter_text, (10, 25), _FONT, _FONT_SCALE_COUNTER, _WHITE, 1)

    # Detection status (top-right)
    if fc is not None:
        status_text = fc.label.value.upper()
        if fc.label == DetectionLabel.CONSENSUS:
            status_colour = _GREEN
        elif fc.label == DetectionLabel.UNCERTAIN:
            status_colour = _YELLOW
        else:
            status_colour = _RED
        (sw, sh), _ = cv2.getTextSize(status_text, _FONT, _FONT_SCALE_COUNTER, _FONT_THICKNESS)
        cv2.putText(frame, status_text, (w - sw - 10, 25), _FONT, _FONT_SCALE_COUNTER, _BLACK, 3)
        cv2.putText(frame, status_text, (w - sw - 10, 25), _FONT, _FONT_SCALE_COUNTER, status_colour, 1)


def _draw_player_boxes(
    frame: np.ndarray,
    players: list,
) -> None:
    """Draw player bounding boxes on a frame (in-place).

    Args:
        frame: BGR image.
        players: List of PlayerBox objects.
    """
    for player in players:
        x1 = int(round(player.x1))
        y1 = int(round(player.y1))
        x2 = int(round(player.x2))
        y2 = int(round(player.y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), _BLUE, _PLAYER_BOX_THICKNESS)
        # Label with confidence
        label = f"Player {player.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE_LABEL, _FONT_THICKNESS)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), _BLUE, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 2), _FONT, _FONT_SCALE_LABEL, _WHITE, _FONT_THICKNESS)
