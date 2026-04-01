"""TrackNet V2 detector wrapper for the labeling pipeline.

Uses the existing TrackNetV2 model from ``src.models.tracknet`` but replaces
the HoughCircles postprocessing with a weighted-centroid approach via
``cv2.moments``, which is more robust on oblique courtside footage where
the ball heatmap is often elliptical rather than circular.

TrackNet requires 3 consecutive frames as input.  This wrapper maintains
an internal buffer of the two most recent frames so the runner can call
``detect(frame)`` one frame at a time.
"""

import logging
from collections import deque
from typing import Optional

import cv2
import numpy as np
import torch

from src.config import get_config
from src.labeling.models import BaseDetector
from src.models.tracknet import BallTrackerNet, load_tracknet

logger = logging.getLogger(__name__)


def _weighted_centroid(
    feature_map: np.ndarray,
    width: int,
    height: int,
    scale_x: float,
    scale_y: float,
    threshold: int = 127,
) -> tuple[float | None, float | None, float]:
    """Extract ball (x, y) from a heatmap using weighted centroid.

    Uses ``cv2.moments`` on the thresholded heatmap, which handles
    elliptical blobs better than HoughCircles.

    Args:
        feature_map: Raw model output, shape ``(height*width,)`` or ``(height, width)``.
        width: Model input width.
        height: Model input height.
        scale_x: Scale factor to map x back to original video resolution.
        scale_y: Scale factor to map y back to original video resolution.
        threshold: Binary threshold for the heatmap.

    Returns:
        Tuple of (x, y, confidence) in original video coordinates,
        where confidence is the normalized peak intensity.
        Returns (None, None, 0.0) if no ball-like region is found.
    """
    feature_map = (feature_map * 255).astype(np.uint8)
    if feature_map.ndim == 1:
        feature_map = feature_map.reshape((height, width))

    _, heatmap = cv2.threshold(feature_map, threshold, 255, cv2.THRESH_BINARY)

    # Use moments to find the centroid of the thresholded region
    moments = cv2.moments(heatmap)
    if moments["m00"] == 0:
        return None, None, 0.0

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    # Confidence: fraction of the heatmap's peak intensity vs 255
    peak = float(feature_map.max()) / 255.0

    x = float(cx * scale_x)
    y = float(cy * scale_y)
    return x, y, peak


class TrackNetDetector(BaseDetector):
    """TrackNet V2 ball detector.

    Maintains a sliding window of 3 frames internally so the
    inference runner can call ``detect(frame)`` per-frame.
    """

    def __init__(self) -> None:
        self._model: BallTrackerNet | None = None
        self._device: str = "cpu"
        self._buffer: deque[np.ndarray] = deque(maxlen=3)
        self._cfg = get_config()
        self._inp_w = self._cfg.model.tracknet_input_width
        self._inp_h = self._cfg.model.tracknet_input_height

    @property
    def name(self) -> str:
        return "tracknet_v2"

    def load(self, device: str = "cpu") -> None:
        """Load TrackNet V2 weights."""
        weights = self._cfg.resolve_path(self._cfg.model.tracknet_weights)
        logger.info("Loading TrackNet V2 from %s onto %s", weights, device)
        self._model = load_tracknet(str(weights), device)
        self._device = device
        self._buffer.clear()

    def detect(self, frame: np.ndarray) -> Optional[tuple[float, float, float]]:
        """Detect ball using a 3-frame sliding window.

        Returns ``None`` for the first two frames (insufficient context).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")

        self._buffer.append(frame)
        if len(self._buffer) < 3:
            return None

        orig_h, orig_w = frame.shape[:2]
        scale_x = orig_w / self._inp_w
        scale_y = orig_h / self._inp_h

        img = cv2.resize(self._buffer[2], (self._inp_w, self._inp_h))
        img_prev = cv2.resize(self._buffer[1], (self._inp_w, self._inp_h))
        img_preprev = cv2.resize(self._buffer[0], (self._inp_w, self._inp_h))

        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)

        with torch.no_grad():
            out = self._model(torch.from_numpy(inp).float().to(self._device))

        # TrackNet V2 outputs 256 channels — per-pixel classification into
        # heatmap intensity levels (0 = background, 255 = ball centre).
        # argmax recovers the predicted intensity per pixel, matching the
        # main pipeline in src/features/ball_tracking/detector.py.
        output = out.argmax(dim=1).detach().cpu().numpy()[0]

        x, y, conf = _weighted_centroid(
            output,
            self._inp_w,
            self._inp_h,
            scale_x,
            scale_y,
            threshold=self._cfg.ball_tracking.confidence_threshold,
        )

        if x is None:
            return None
        return (x, y, conf)

    def unload(self) -> None:
        """Release model and clear frame buffer."""
        if self._model is not None:
            del self._model
            self._model = None
        self._buffer.clear()
        torch.cuda.empty_cache()
        logger.info("TrackNet V2 unloaded")
