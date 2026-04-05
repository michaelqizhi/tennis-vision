"""Court boundary filter for ball detections.

Filters out false positive ball detections that fall outside the court area
using homography-based coordinate transformation.
"""

import logging
from dataclasses import dataclass

import numpy as np

from src.features.court_detect.homography import transform_point_to_court
from src.features.court_detect.court_template import ITF_DIMS
from src.features.ball_tracking.detector import BallDetection

logger = logging.getLogger(__name__)


@dataclass
class CourtBoundaryFilter:
    """Filters ball detections outside the court boundary.

    Uses homography to transform pixel coordinates to court coordinates (meters)
    and rejects detections outside expanded court bounds.

    Args:
        homography: 3×3 homography matrix (image pixels → court meters).
        x_margin: Extra margin beyond sideline (meters).
        y_margin: Extra margin beyond baseline (meters).
    """

    homography: np.ndarray
    x_margin: float = 3.0
    y_margin: float = 5.0

    def __post_init__(self):
        """Compute bounds based on court dimensions and margins."""
        d = ITF_DIMS
        # Doubles half-width + margin
        self.x_bound = d.doubles_width / 2.0 + self.x_margin
        # Half court length + margin
        self.y_bound = d.net_to_baseline + self.y_margin

        logger.debug(
            f"Court bounds: |x| ≤ {self.x_bound:.2f}m, "
            f"|y| ≤ {self.y_bound:.2f}m"
        )

    def is_in_bounds(self, pixel_x: float, pixel_y: float) -> bool:
        """Check if a pixel coordinate falls within expanded court bounds.

        Args:
            pixel_x: X coordinate in image pixels.
            pixel_y: Y coordinate in image pixels.

        Returns:
            True if within bounds, False otherwise.
        """
        try:
            # Transform to court coordinates
            x_m, y_m = transform_point_to_court(pixel_x, pixel_y, self.homography)

            # Check bounds
            in_bounds = abs(x_m) <= self.x_bound and abs(y_m) <= self.y_bound

            logger.debug(
                f"Pixel ({pixel_x:.1f}, {pixel_y:.1f}) → "
                f"Court ({x_m:.2f}m, {y_m:.2f}m) → "
                f"{'IN' if in_bounds else 'OUT'}"
            )

            return in_bounds
        except Exception as e:
            logger.warning(f"Failed to transform point ({pixel_x}, {pixel_y}): {e}")
            # If transformation fails, assume out of bounds
            return False

    def filter_detections(
        self, detections: list[BallDetection]
    ) -> list[BallDetection]:
        """Filter detections, marking out-of-bounds ones as not detected.

        Creates new BallDetection objects — does not modify originals.

        Args:
            detections: List of ball detections to filter.

        Returns:
            New list of BallDetection objects with out-of-bounds detections
            set to x=None, y=None, confidence=0.0.
        """
        filtered = []
        in_count = 0
        out_count = 0

        for det in detections:
            if det.detected:
                if self.is_in_bounds(det.x, det.y):
                    # Keep detection as-is
                    filtered.append(det)
                    in_count += 1
                else:
                    # Mark as not detected
                    filtered.append(
                        BallDetection(
                            frame_number=det.frame_number,
                            x=None,
                            y=None,
                            confidence=0.0,
                            interpolated=False,
                        )
                    )
                    out_count += 1
            else:
                # Already not detected, keep as-is
                filtered.append(det)

        if in_count + out_count > 0:
            logger.info(
                f"Boundary filter: {in_count} in bounds, {out_count} out of bounds "
                f"({out_count * 100.0 / (in_count + out_count):.1f}% rejected)"
            )

        return filtered
