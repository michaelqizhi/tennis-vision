"""Court detection — keypoint detection, homography, and coordinate transform."""

from src.features.court_detect.detector import CourtDetectionResult, CourtDetector
from src.features.court_detect.homography import (
    compute_pixel_to_court_homography,
    transform_point_to_court,
    transform_points_to_court,
)
from src.features.court_detect.court_template import ITF_DIMS, CourtDimensions, CourtReference
from src.features.court_detect.boundary_filter import CourtBoundaryFilter
from src.features.court_detect.frame_selector import (
    compute_court_color_score,
    compute_sharpness,
    find_valid_frame_indices,
    select_best_court_frames,
)
from src.features.court_detect.homography_monitor import (
    HomographyMonitor,
    TemplateRegion,
)

__all__ = [
    "CourtDetectionResult",
    "CourtDetector",
    "compute_pixel_to_court_homography",
    "transform_point_to_court",
    "transform_points_to_court",
    "ITF_DIMS",
    "CourtDimensions",
    "CourtReference",
    "CourtBoundaryFilter",
    "compute_court_color_score",
    "compute_sharpness",
    "find_valid_frame_indices",
    "select_best_court_frames",
    "HomographyMonitor",
    "TemplateRegion",
]