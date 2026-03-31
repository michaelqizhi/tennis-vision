"""ITF standard tennis court template and reference keypoint coordinates.

Provides both pixel coordinates (matching the upstream TennisCourtDetector
reference image) and real-world meter coordinates based on ITF specifications.

ITF Court Dimensions (doubles):
  - Full court length: 23.77m (baseline to baseline)
  - Singles width: 8.23m
  - Doubles width: 10.97m
  - Service line to net: 6.40m
  - Center service line length: 12.80m (6.40m each side)
  - Net position: center of court length

Keypoint ordering (0-13) matches upstream model output:
  0: top-left baseline (doubles)
  1: top-right baseline (doubles)
  2: bottom-left baseline (doubles)
  3: bottom-right baseline (doubles)
  4: top-left singles sideline at baseline
  5: bottom-left singles sideline at baseline
  6: top-right singles sideline at baseline
  7: bottom-right singles sideline at baseline
  8: top-left service box corner
  9: top-right service box corner
  10: bottom-left service box corner
  11: bottom-right service box corner
  12: top center service line (at service line)
  13: bottom center service line (at service line)
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CourtDimensions:
    """ITF standard court dimensions in meters."""
    court_length: float = 23.77
    singles_width: float = 8.23
    doubles_width: float = 10.97
    service_line_distance: float = 6.40  # from net to service line
    net_to_baseline: float = 11.885  # half court length

    @property
    def doubles_alley_width(self) -> float:
        return (self.doubles_width - self.singles_width) / 2.0

    @property
    def service_box_width(self) -> float:
        return self.singles_width / 2.0

    @property
    def service_box_depth(self) -> float:
        return self.service_line_distance


ITF_DIMS = CourtDimensions()


class CourtReference:
    """Court reference model with pixel and real-world coordinates.

    Pixel coordinates match the upstream TennisCourtDetector reference image.
    Real-world coordinates use meters with origin at court center.
    """

    def __init__(self):
        # --- Pixel coordinates (from upstream TennisCourtDetector) ---
        self.baseline_top = ((286, 561), (1379, 561))
        self.baseline_bottom = ((286, 2935), (1379, 2935))
        self.net = ((286, 1748), (1379, 1748))
        self.left_court_line = ((286, 561), (286, 2935))
        self.right_court_line = ((1379, 561), (1379, 2935))
        self.left_inner_line = ((423, 561), (423, 2935))
        self.right_inner_line = ((1242, 561), (1242, 2935))
        self.middle_line = ((832, 1110), (832, 2386))
        self.top_inner_line = ((423, 1110), (1242, 1110))
        self.bottom_inner_line = ((423, 2386), (1242, 2386))

        # 14 keypoints in the order expected by the model
        self.key_points = [
            *self.baseline_top, *self.baseline_bottom,
            *self.left_inner_line, *self.right_inner_line,
            *self.top_inner_line, *self.bottom_inner_line,
            *self.middle_line,
        ]

        # 12 court configurations of 4 points for homography fitting
        self.court_conf = {
            1: [*self.baseline_top, *self.baseline_bottom],
            2: [self.left_inner_line[0], self.right_inner_line[0],
                self.left_inner_line[1], self.right_inner_line[1]],
            3: [self.left_inner_line[0], self.right_court_line[0],
                self.left_inner_line[1], self.right_court_line[1]],
            4: [self.left_court_line[0], self.right_inner_line[0],
                self.left_court_line[1], self.right_inner_line[1]],
            5: [*self.top_inner_line, *self.bottom_inner_line],
            6: [*self.top_inner_line,
                self.left_inner_line[1], self.right_inner_line[1]],
            7: [self.left_inner_line[0], self.right_inner_line[0],
                *self.bottom_inner_line],
            8: [self.right_inner_line[0], self.right_court_line[0],
                self.right_inner_line[1], self.right_court_line[1]],
            9: [self.left_court_line[0], self.left_inner_line[0],
                self.left_court_line[1], self.left_inner_line[1]],
            10: [self.top_inner_line[0], self.middle_line[0],
                 self.bottom_inner_line[0], self.middle_line[1]],
            11: [self.middle_line[0], self.top_inner_line[1],
                 self.middle_line[1], self.bottom_inner_line[1]],
            12: [*self.bottom_inner_line,
                 self.left_inner_line[1], self.right_inner_line[1]],
        }

        # Reference image dimensions
        self.court_width = 1117
        self.court_height = 2408
        self.top_bottom_border = 549
        self.right_left_border = 274
        self.court_total_width = self.court_width + self.right_left_border * 2
        self.court_total_height = self.court_height + self.top_bottom_border * 2
        self.line_width = 1

    def build_court_reference(self) -> np.ndarray:
        """Create a court reference image using line positions.

        Returns:
            Grayscale image with court lines drawn.
        """
        court = np.zeros(
            (self.court_total_height, self.court_total_width), dtype=np.uint8,
        )
        cv2.line(court, *self.baseline_top, 1, self.line_width)
        cv2.line(court, *self.baseline_bottom, 1, self.line_width)
        cv2.line(court, *self.net, 1, self.line_width)
        cv2.line(court, *self.top_inner_line, 1, self.line_width)
        cv2.line(court, *self.bottom_inner_line, 1, self.line_width)
        cv2.line(court, *self.left_court_line, 1, self.line_width)
        cv2.line(court, *self.right_court_line, 1, self.line_width)
        cv2.line(court, *self.left_inner_line, 1, self.line_width)
        cv2.line(court, *self.right_inner_line, 1, self.line_width)
        cv2.line(court, *self.middle_line, 1, self.line_width)
        court = cv2.dilate(court, np.ones((5, 5), dtype=np.uint8))
        return court


# Module-level reference instance and derived data
COURT_REF = CourtReference()

# Reference keypoints as Nx1x2 array for cv2.perspectiveTransform
REFER_KPS = np.array(
    COURT_REF.key_points, dtype=np.float32,
).reshape((-1, 1, 2))

# Pre-compute configuration indices into key_points
COURT_CONF_INDICES: dict[int, list[int]] = {}
for _i in range(len(COURT_REF.court_conf)):
    _conf = COURT_REF.court_conf[_i + 1]
    _inds = [COURT_REF.key_points.index(_conf[j]) for j in range(4)]
    COURT_CONF_INDICES[_i + 1] = _inds


def reference_keypoints_meters() -> list[tuple[float, float]]:
    """Get the 14 reference keypoints in real-world meter coordinates.

    Origin is at court center (net midpoint).
    X-axis: left-right (positive = right from camera perspective).
    Y-axis: top-bottom (positive = away from camera / far baseline).

    Returns:
        List of 14 (x_m, y_m) tuples.
    """
    d = ITF_DIMS
    hw = d.doubles_width / 2.0
    sw = d.singles_width / 2.0
    hl = d.court_length / 2.0
    sl = d.service_line_distance

    return [
        # 0: top-left baseline (doubles)
        (-hw, -hl),
        # 1: top-right baseline (doubles)
        (hw, -hl),
        # 2: bottom-left baseline (doubles)
        (-hw, hl),
        # 3: bottom-right baseline (doubles)
        (hw, hl),
        # 4: top-left singles sideline at baseline
        (-sw, -hl),
        # 5: bottom-left singles sideline at baseline
        (-sw, hl),
        # 6: top-right singles sideline at baseline
        (sw, -hl),
        # 7: bottom-right singles sideline at baseline
        (sw, hl),
        # 8: top service line left corner
        (-sw, -sl),
        # 9: top service line right corner
        (sw, -sl),
        # 10: bottom service line left corner
        (-sw, sl),
        # 11: bottom service line right corner
        (sw, sl),
        # 12: top center service line (at service line)
        (0.0, -sl),
        # 13: bottom center service line (at service line)
        (0.0, sl),
    ]


REFERENCE_KPS_METERS = reference_keypoints_meters()
