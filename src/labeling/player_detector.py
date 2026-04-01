"""YOLOv8-nano player detector for spatial filtering in the labeling pipeline.

Detects person bounding boxes using YOLOv8-nano (ultralytics). Player positions
serve as spatial priors for ball detection filtering — ball detections too far
from any player are likely false positives.

Usage::

    detector = PlayerDetector()
    detector.load("cuda")
    boxes = detector.detect(frame)
    detector.unload()

Each detection is a ``PlayerBox`` with ``(x1, y1, x2, y2, confidence)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PlayerBox:
    """A detected player bounding box."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Check if a point is within the box (with optional margin)."""
        return (self.x1 - margin <= x <= self.x2 + margin
                and self.y1 - margin <= y <= self.y2 + margin)

    def distance_to_point(self, x: float, y: float) -> float:
        """Minimum distance from point to bounding box edge.

        Returns 0 if the point is inside the box.
        """
        dx = max(self.x1 - x, 0.0, x - self.x2)
        dy = max(self.y1 - y, 0.0, y - self.y2)
        return (dx * dx + dy * dy) ** 0.5


class PlayerDetector:
    """YOLOv8-nano person detector for the labeling pipeline.

    Args:
        model_name: Ultralytics model identifier (default: yolov8n).
        person_class_id: COCO class ID for 'person' (default: 0).
        confidence_threshold: Minimum confidence to keep a detection.
        max_players: Maximum number of players to return (sorted by confidence).
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        person_class_id: int = 0,
        confidence_threshold: float = 0.4,
        max_players: int = 4,
    ):
        self.model_name = model_name
        self.person_class_id = person_class_id
        self.confidence_threshold = confidence_threshold
        self.max_players = max_players
        self._model = None
        self._device: str = "cpu"

    def load(self, device: str = "cpu") -> None:
        """Load YOLOv8-nano model.

        Downloads weights automatically on first use via ultralytics.

        Args:
            device: PyTorch device string ('cpu' or 'cuda').
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for player detection. "
                "Install with: pip install ultralytics"
            )

        self._device = device
        self._model = YOLO(self.model_name)
        logger.info("PlayerDetector loaded: %s on %s", self.model_name, device)

    def detect(self, frame: np.ndarray) -> list[PlayerBox]:
        """Detect players in a single frame.

        Args:
            frame: BGR image as a numpy array (H, W, 3), uint8.

        Returns:
            List of PlayerBox detections, sorted by confidence (highest first).
            Returns at most ``max_players`` results.
        """
        if self._model is None:
            raise RuntimeError("PlayerDetector not loaded. Call load() first.")

        results = self._model(
            frame,
            device=self._device,
            classes=[self.person_class_id],
            conf=self.confidence_threshold,
            verbose=False,
        )

        boxes: list[PlayerBox] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                boxes.append(PlayerBox(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    confidence=conf,
                ))

        # Sort by confidence and limit
        boxes.sort(key=lambda b: b.confidence, reverse=True)
        return boxes[:self.max_players]

    def unload(self) -> None:
        """Release model from memory."""
        self._model = None
        logger.info("PlayerDetector unloaded")


def filter_by_player_proximity(
    ball_x: float,
    ball_y: float,
    players: list[PlayerBox],
    max_distance: float = 200.0,
) -> bool:
    """Check if a ball detection is within plausible distance of any player.

    A tennis ball should always be reasonably close to at least one player
    during active play. Detections far from all players are likely false
    positives on net posts, fences, or other features.

    Args:
        ball_x: Ball x position in pixels.
        ball_y: Ball y position in pixels.
        players: Detected player bounding boxes.
        max_distance: Maximum pixel distance from nearest player box edge.

    Returns:
        True if the detection passes the filter (is near a player),
        False if it should be rejected.
    """
    if not players:
        return True  # No players detected — can't filter, pass through

    for player in players:
        if player.distance_to_point(ball_x, ball_y) <= max_distance:
            return True

    return False


def serialize_player_boxes(boxes: list[PlayerBox]) -> list[dict]:
    """Convert player boxes to JSON-serializable format."""
    return [
        {
            "x1": round(b.x1, 1),
            "y1": round(b.y1, 1),
            "x2": round(b.x2, 1),
            "y2": round(b.y2, 1),
            "confidence": round(b.confidence, 4),
        }
        for b in boxes
    ]
