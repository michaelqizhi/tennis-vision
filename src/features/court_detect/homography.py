"""Homography computation and coordinate transformation.

Computes the best-fit homography matrix from detected court keypoints to the
reference court template, and provides functions to transform pixel coordinates
to real-world court coordinates (meters).
"""

import numpy as np
import cv2
from scipy.spatial import distance

from src.features.court_detect.court_template import (
    COURT_REF,
    REFER_KPS,
    COURT_CONF_INDICES,
    REFERENCE_KPS_METERS,
)


def find_best_homography(
    detected_points: list[tuple[float | None, float | None]],
) -> np.ndarray | None:
    """Find the best homography from reference court to detected keypoints.

    Tries all 12 court configurations of 4 reference points and picks the
    one with the lowest mean reprojection error on the remaining visible points.

    Args:
        detected_points: List of 14 (x, y) tuples. Use (None, None) for
            undetected keypoints.

    Returns:
        3×3 homography matrix (reference → image), or None if no valid
        configuration was found.
    """
    best_matrix: np.ndarray | None = None
    best_dist = np.inf

    for conf_idx in range(1, 13):
        conf = COURT_REF.court_conf[conf_idx]
        inds = COURT_CONF_INDICES[conf_idx]

        # Get the 4 detected points for this configuration
        det_pts = [detected_points[i] for i in inds]
        if any(p[0] is None or p[1] is None for p in det_pts):
            continue

        src = np.float32(conf)
        dst = np.float32(det_pts)
        matrix, _ = cv2.findHomography(src, dst, method=0)
        if matrix is None:
            continue

        # Compute reprojection error on other visible points
        trans_kps = cv2.perspectiveTransform(REFER_KPS, matrix)
        dists = []
        for i in range(14):
            if i not in inds and detected_points[i][0] is not None:
                dists.append(
                    distance.euclidean(detected_points[i], trans_kps[i].flatten())
                )

        if dists:
            mean_dist = float(np.mean(dists))
        else:
            # No other points to validate — use a high penalty
            mean_dist = 1e6

        if mean_dist < best_dist:
            best_dist = mean_dist
            best_matrix = matrix

    return best_matrix


def refine_keypoints_with_homography(
    detected_points: list[tuple[float | None, float | None]],
    homography: np.ndarray,
) -> list[tuple[float, float]]:
    """Use homography to fill in / refine all 14 keypoint positions.

    Projects reference keypoints through the homography matrix to get
    consistent positions for all keypoints, even those that were not
    directly detected.

    Args:
        detected_points: Original 14 detected (x, y) points.
        homography: 3×3 homography matrix (reference → image).

    Returns:
        List of 14 refined (x, y) tuples.
    """
    projected = cv2.perspectiveTransform(REFER_KPS, homography)
    result = []
    for i in range(14):
        pt = projected[i].flatten()
        result.append((float(pt[0]), float(pt[1])))
    return result


def compute_pixel_to_court_homography(
    detected_points: list[tuple[float | None, float | None]],
) -> np.ndarray | None:
    """Compute homography from image pixels to real-world court coordinates.

    Uses detected keypoints and their known real-world positions (meters)
    to compute a direct pixel → court-meter transform.

    Args:
        detected_points: List of 14 (x, y) tuples in image pixel coords.

    Returns:
        3×3 homography matrix (image pixels → court meters), or None if
        not enough points.
    """
    # Collect matched pairs of (pixel, meter) for detected keypoints
    src_pts = []
    dst_pts = []
    for i in range(14):
        px, py = detected_points[i]
        if px is not None and py is not None:
            src_pts.append([px, py])
            dst_pts.append(list(REFERENCE_KPS_METERS[i]))

    if len(src_pts) < 4:
        return None

    src = np.float32(src_pts)
    dst = np.float32(dst_pts)
    matrix, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    return matrix


def transform_point_to_court(
    pixel_x: float,
    pixel_y: float,
    homography: np.ndarray,
) -> tuple[float, float]:
    """Transform a single pixel coordinate to court coordinates (meters).

    Args:
        pixel_x: X coordinate in image pixels.
        pixel_y: Y coordinate in image pixels.
        homography: 3×3 homography (image pixels → court meters).

    Returns:
        (x_meters, y_meters) on court, origin at court center.
    """
    pt = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, homography)
    x_m, y_m = transformed[0][0]
    return float(x_m), float(y_m)


def transform_points_to_court(
    pixel_points: list[tuple[float, float]],
    homography: np.ndarray,
) -> list[tuple[float, float]]:
    """Transform multiple pixel coordinates to court coordinates.

    Args:
        pixel_points: List of (x, y) pixel coordinates.
        homography: 3×3 homography (image pixels → court meters).

    Returns:
        List of (x_meters, y_meters) tuples.
    """
    if not pixel_points:
        return []

    pts = np.array(pixel_points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, homography)
    return [(float(p[0]), float(p[1])) for p in transformed.reshape(-1, 2)]
