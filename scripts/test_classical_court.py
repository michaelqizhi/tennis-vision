#!/usr/bin/env python3
"""Classical computer-vision court-line detection experiment.

Detects tennis court lines via Hough transform + geometric fitting,
derives court keypoints from line intersections, and computes a
homography.  Also overlays the CNN detector's keypoints 12 & 13 for
comparison.

Usage:
    python scripts/test_classical_court.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from itertools import product
from dataclasses import dataclass, field
from collections import defaultdict

import cv2
import numpy as np

# Ensure project root is on sys.path so we can import src.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.features.court_detect.detector import CourtDetector
from src.features.court_detect.court_template import (
    REFERENCE_KPS_METERS,
    CourtReference,
    COURT_REF,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VIDEO_PATH = PROJECT_ROOT / "tests" / "fixtures" / "real_match_10min.mp4"
OUTPUT_DIR = PROJECT_ROOT / "output"
FRAME_INDICES = [1500, 3000, 4500]
FRAME_W, FRAME_H = 1920, 1080

# ROI: mask out everything above this y-coordinate (ceiling/walls/lights)
ROI_Y_MIN = 400

# Court surface HSV ranges (blue indoor court)
COURT_HSV_LOW = np.array([90, 40, 50])
COURT_HSV_HIGH = np.array([125, 255, 255])
COURT_MASK_DILATE_PX = 30

# ITF court keypoints in metres (origin = court centre / net midpoint)
KP_NAMES = [
    "BL-TL", "BL-TR", "BL-BL", "BL-BR",
    "SL-TL", "SL-BL", "SL-TR", "SL-BR",
    "SV-TL", "SV-TR", "SV-BL", "SV-BR",
    "CT-T",  "CT-B",
]

# Tuning grids
WHITE_THRESHOLDS = [170, 190, 210]
HOUGH_CONFIGS = [
    {"threshold": 50,  "minLineLength": 80,  "maxLineGap": 30},
    {"threshold": 80,  "minLineLength": 100, "maxLineGap": 40},
    {"threshold": 100, "minLineLength": 200, "maxLineGap": 50},
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class LineDetectionResult:
    """Container for one frame's line-detection outputs."""
    frame_idx: int
    white_thresh: int
    hough_cfg: dict
    lines: np.ndarray | None
    horizontal: list[np.ndarray]
    vertical: list[np.ndarray]
    intersections: list[tuple[float, float]]
    white_mask: np.ndarray
    cleaned: np.ndarray
    edges: np.ndarray
    court_color_mask: np.ndarray | None = None
    roi_mask: np.ndarray | None = None
    filtered_white_mask: np.ndarray | None = None
    lsd_lines: np.ndarray | None = None
    detection_method: str = "hough"
    n_lines: int = 0
    n_horizontal: int = 0
    n_vertical: int = 0

    def __post_init__(self):
        self.n_lines = 0 if self.lines is None else len(self.lines)
        self.n_horizontal = len(self.horizontal)
        self.n_vertical = len(self.vertical)


# ---------------------------------------------------------------------------
# Court-surface and ROI detection (Fix 1 & 2)
# ---------------------------------------------------------------------------
def detect_court_surface(frame_bgr: np.ndarray) -> np.ndarray:
    """Detect court surface pixels via HSV color segmentation (blue court)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    court_mask = cv2.inRange(hsv, COURT_HSV_LOW, COURT_HSV_HIGH)
    # Fill small holes and smooth
    kernel = np.ones((7, 7), np.uint8)
    court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return court_mask


def build_roi_mask(
    frame_bgr: np.ndarray,
    court_color_mask: np.ndarray,
) -> np.ndarray:
    """Build a region-of-interest mask combining vertical crop + court color.

    Returns a binary mask where 255 = valid court region.
    """
    h, w = frame_bgr.shape[:2]
    roi = np.zeros((h, w), dtype=np.uint8)

    # Step 1: simple vertical crop — keep bottom portion only
    roi[ROI_Y_MIN:, :] = 255

    # Step 2: dilate the court color mask to include court lines (which are
    # white, not blue) that sit *on* the court surface
    dilated_court = cv2.dilate(
        court_color_mask,
        np.ones((COURT_MASK_DILATE_PX * 2 + 1, COURT_MASK_DILATE_PX * 2 + 1), np.uint8),
    )

    # Combine: pixel must be below ROI_Y_MIN *and* near court surface
    roi = cv2.bitwise_and(roi, dilated_court)

    # Safety: also include a thin strip at the very bottom (near baseline,
    # which may not have enough blue pixels around it)
    roi[max(h - 120, 0):, :] = 255

    return roi


# ---------------------------------------------------------------------------
# Core detection functions (Fix 2: court-color-aware white detection)
# ---------------------------------------------------------------------------
def detect_court_lines(
    frame_bgr: np.ndarray,
    white_thresh: int = 190,
    hough_threshold: int = 80,
    hough_min_length: int = 100,
    hough_max_gap: int = 30,
    roi_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Detect white court lines via thresholding + Hough transform.

    If *roi_mask* is provided, only pixels inside the ROI are considered.
    Returns (lines, white_mask, cleaned, edges, filtered_white_mask).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # --- White pixel detection ---
    # Method 1: grayscale threshold
    _, white_mask = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY)

    # Method 2: HSV-based white (high V, low S)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white_hsv = cv2.inRange(hsv, (0, 0, max(white_thresh, 180)), (179, 50, 255))

    combined = cv2.bitwise_or(white_mask, white_hsv)

    # --- Apply ROI mask (Fix 1 + Fix 2) ---
    filtered_white = combined.copy()
    if roi_mask is not None:
        filtered_white = cv2.bitwise_and(combined, roi_mask)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(filtered_white, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    # Canny edge detection
    edges = cv2.Canny(cleaned, 50, 150)

    # Probabilistic Hough transform
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=hough_threshold,
        minLineLength=hough_min_length,
        maxLineGap=hough_max_gap,
    )

    return lines, white_mask, cleaned, edges, filtered_white


# ---------------------------------------------------------------------------
# LSD line detection (Fix 5: alternative to Hough)
# ---------------------------------------------------------------------------
def detect_court_lines_lsd(
    frame_bgr: np.ndarray,
    white_thresh: int = 190,
    roi_mask: np.ndarray | None = None,
    min_length: float = 60.0,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Detect court lines using OpenCV's Line Segment Detector (LSD).

    Returns (lines_Nx1x4, white_mask, cleaned, edges_placeholder, filtered_white).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    _, white_mask = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white_hsv = cv2.inRange(hsv, (0, 0, max(white_thresh, 180)), (179, 50, 255))
    combined = cv2.bitwise_or(white_mask, white_hsv)

    filtered_white = combined.copy()
    if roi_mask is not None:
        filtered_white = cv2.bitwise_and(combined, roi_mask)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(filtered_white, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    # LSD operates on grayscale; mask non-ROI to black
    gray_masked = gray.copy()
    if roi_mask is not None:
        gray_masked[roi_mask == 0] = 0

    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lsd_lines, widths, precs, nfas = lsd.detect(gray_masked)

    # Filter by length and convert to HoughLinesP format (N,1,4)
    if lsd_lines is not None:
        kept = []
        for seg in lsd_lines:
            x1, y1, x2, y2 = seg[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length >= min_length:
                kept.append([[int(round(x1)), int(round(y1)),
                              int(round(x2)), int(round(y2))]])
        lines = np.array(kept, dtype=np.int32) if kept else None
    else:
        lines = None

    # No Canny needed for LSD; provide a placeholder
    edges = cleaned.copy()

    return lines, white_mask, cleaned, edges, filtered_white


# ---------------------------------------------------------------------------
# Line classification (Fix 3: vanishing-point-aware for courtside view)
# ---------------------------------------------------------------------------
def _seg_angle(seg: np.ndarray) -> float:
    x1, y1, x2, y2 = seg
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180


def estimate_vanishing_point(lines: list[np.ndarray]) -> np.ndarray | None:
    """Estimate a vanishing point from a set of line segments.

    Uses RANSAC-style: pick pairs, compute intersection, find the point
    that has the most lines passing near it.
    """
    if len(lines) < 2:
        return None

    # Compute all pairwise intersections
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pt = line_intersection(lines[i], lines[j],
                                   w=FRAME_W * 4, h=FRAME_H * 4)
            if pt is not None:
                intersections.append(np.array(pt))

    if not intersections:
        return None

    # Find the point that is close to the most lines
    best_pt = None
    best_count = 0
    for candidate in intersections:
        count = 0
        for seg in lines:
            # Distance from candidate to line through segment
            x1, y1, x2, y2 = seg.astype(float)
            length = max(np.hypot(x2 - x1, y2 - y1), 1e-6)
            dist = abs((y2 - y1) * candidate[0] - (x2 - x1) * candidate[1]
                       + x2 * y1 - y2 * x1) / length
            if dist < 40:
                count += 1
        if count > best_count:
            best_count = count
            best_pt = candidate

    return best_pt


def classify_lines_courtside(
    lines: np.ndarray | None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Classify lines into horizontal (baselines/service lines) and vertical
    (sidelines/center line) families using vanishing point estimation.

    For a courtside camera:
    - "Vertical" lines (sidelines) converge toward a vanishing point near the
      top-center of the frame. They have steep angles but are NOT truly
      vertical — they fan out from the VP.
    - "Horizontal" lines (baselines, service lines) are roughly horizontal
      (small |angle| from 0° or 180°).

    Falls back to generous angle thresholds if VP estimation fails.
    """
    horizontal: list[np.ndarray] = []
    vertical: list[np.ndarray] = []
    if lines is None:
        return horizontal, vertical

    all_segs = [seg[0] if seg.ndim == 2 else seg for seg in lines]

    # Step 1: initial coarse split — horizontal vs "steep" lines
    h_candidates: list[np.ndarray] = []
    steep_candidates: list[np.ndarray] = []

    for seg in all_segs:
        angle = _seg_angle(seg)
        # Horizontal: within 20° of horizontal (tightened from 25°)
        if angle < 20 or angle > 160:
            h_candidates.append(seg)
        # Steep: between 45° and 135° — potential sideline / center line
        elif 45 < angle < 135:
            steep_candidates.append(seg)

    # Step 2: among steep lines, try to identify the vanishing point
    vp = estimate_vanishing_point(steep_candidates)

    if vp is not None:
        for seg in steep_candidates:
            x1, y1, x2, y2 = seg.astype(float)
            length = max(np.hypot(x2 - x1, y2 - y1), 1e-6)
            dist_to_vp = abs((y2 - y1) * vp[0] - (x2 - x1) * vp[1]
                             + x2 * y1 - y2 * x1) / length
            if dist_to_vp < 60:
                vertical.append(seg)
    else:
        vertical = steep_candidates

    horizontal = h_candidates
    return horizontal, vertical


def classify_lines(
    lines: np.ndarray | None,
    h_lo: float = 30.0,
    v_lo: float = 60.0,
    v_hi: float = 120.0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Legacy classifier — delegates to courtside-aware version."""
    return classify_lines_courtside(lines)


def line_intersection(
    line1: np.ndarray,
    line2: np.ndarray,
    w: int = FRAME_W,
    h: int = FRAME_H,
) -> tuple[float, float] | None:
    """Compute intersection of two line segments (extended to full lines)."""
    x1, y1, x2, y2 = line1.astype(float)
    x3, y3, x4, y4 = line2.astype(float)
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    # Accept only within frame bounds (with small margin)
    margin = 50
    if -margin <= ix <= w + margin and -margin <= iy <= h + margin:
        return (float(ix), float(iy))
    return None


def find_intersections(
    horizontal: list[np.ndarray],
    vertical: list[np.ndarray],
) -> list[tuple[float, float]]:
    """Find all horizontal × vertical line intersections."""
    pts: list[tuple[float, float]] = []
    for h_seg, v_seg in product(horizontal, vertical):
        pt = line_intersection(h_seg, v_seg)
        if pt is not None:
            pts.append(pt)
    return pts


def deduplicate_points(
    pts: list[tuple[float, float]],
    radius: float = 20.0,
) -> list[tuple[float, float]]:
    """Merge points closer than *radius* pixels, keeping the centroid."""
    if not pts:
        return []
    arr = np.array(pts)
    used = np.zeros(len(arr), dtype=bool)
    merged: list[tuple[float, float]] = []
    for i in range(len(arr)):
        if used[i]:
            continue
        cluster = [arr[i]]
        used[i] = True
        for j in range(i + 1, len(arr)):
            if used[j]:
                continue
            if np.linalg.norm(arr[i] - arr[j]) < radius:
                cluster.append(arr[j])
                used[j] = True
        c = np.mean(cluster, axis=0)
        merged.append((float(c[0]), float(c[1])))
    return merged


# ---------------------------------------------------------------------------
# Merge nearby collinear segments
# ---------------------------------------------------------------------------
def merge_collinear_segments(
    segments: list[np.ndarray],
    angle_tol: float = 10.0,
    dist_tol: float = 30.0,
) -> list[np.ndarray]:
    """Merge segments that are nearly collinear and close together."""
    if len(segments) <= 1:
        return segments

    merged: list[np.ndarray] = []
    used = [False] * len(segments)

    for i in range(len(segments)):
        if used[i]:
            continue
        group = [segments[i]]
        used[i] = True
        ai = _seg_angle(segments[i])
        for j in range(i + 1, len(segments)):
            if used[j]:
                continue
            aj = _seg_angle(segments[j])
            adiff = min(abs(ai - aj), 180 - abs(ai - aj))
            if adiff > angle_tol:
                continue
            # Check perpendicular distance between midpoints & segment
            mx = (segments[j][0] + segments[j][2]) / 2
            my = (segments[j][1] + segments[j][3]) / 2
            x1, y1, x2, y2 = segments[i].astype(float)
            length = max(np.hypot(x2 - x1, y2 - y1), 1e-6)
            perp_dist = abs((y2 - y1) * mx - (x2 - x1) * my + x2 * y1 - y2 * x1) / length
            if perp_dist < dist_tol:
                group.append(segments[j])
                used[j] = True

        # Merge group: project all endpoints onto principal axis, take extremes
        pts = np.concatenate([np.array([[s[0], s[1]], [s[2], s[3]]]) for s in group])
        # PCA-like: use direction of longest segment in group
        best = max(group, key=lambda s: np.hypot(s[2] - s[0], s[3] - s[1]))
        dx, dy = best[2] - best[0], best[3] - best[1]
        norm = max(np.hypot(dx, dy), 1e-6)
        ux, uy = dx / norm, dy / norm
        projs = pts[:, 0] * ux + pts[:, 1] * uy
        imin, imax = np.argmin(projs), np.argmax(projs)
        merged.append(np.array([
            int(pts[imin, 0]), int(pts[imin, 1]),
            int(pts[imax, 0]), int(pts[imax, 1]),
        ]))

    return merged


# ---------------------------------------------------------------------------
# Homography helpers
# ---------------------------------------------------------------------------
def try_compute_homography(
    image_pts: list[tuple[float, float]],
    court_pts_m: list[tuple[float, float]],
) -> np.ndarray | None:
    """Compute homography from ≥4 image↔court point correspondences."""
    if len(image_pts) < 4 or len(court_pts_m) < 4:
        return None
    src = np.array(image_pts, dtype=np.float64)
    dst = np.array(court_pts_m, dtype=np.float64)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return H


def homography_sanity_check(H: np.ndarray) -> dict:
    """Project court corners through H⁻¹ → image and verify reasonableness."""
    corners_m = np.array([
        REFERENCE_KPS_METERS[0],  # BL-TL
        REFERENCE_KPS_METERS[1],  # BL-TR
        REFERENCE_KPS_METERS[2],  # BL-BL
        REFERENCE_KPS_METERS[3],  # BL-BR
    ], dtype=np.float64).reshape(-1, 1, 2)
    try:
        H_inv = np.linalg.inv(H)
        projected = cv2.perspectiveTransform(corners_m, H_inv)
        results = {}
        for name, pt in zip(
            ["BL-TL", "BL-TR", "BL-BL", "BL-BR"],
            projected.reshape(-1, 2),
        ):
            results[name] = (float(pt[0]), float(pt[1]))
        return results
    except np.linalg.LinAlgError:
        return {}


# ---------------------------------------------------------------------------
# Nearest-keypoint matching (Fix 4: structural identification)
# ---------------------------------------------------------------------------
def _line_y_at_x(seg: np.ndarray, x: float) -> float | None:
    """Evaluate a line segment's y-value at a given x (extending the line)."""
    x1, y1, x2, y2 = seg.astype(float)
    dx = x2 - x1
    if abs(dx) < 1e-6:
        return (y1 + y2) / 2.0
    t = (x - x1) / dx
    return y1 + t * (y2 - y1)


def _seg_left_right(seg: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return (left_point, right_point) for a segment."""
    if seg[0] <= seg[2]:
        return (int(seg[0]), int(seg[1])), (int(seg[2]), int(seg[3]))
    return (int(seg[2]), int(seg[3])), (int(seg[0]), int(seg[1]))


def identify_court_lines(
    horizontal: list[np.ndarray],
    vertical: list[np.ndarray],
    cnn_kps: list[tuple[float | None, float | None]],
) -> dict[str, np.ndarray | None]:
    """Identify which detected lines correspond to court features.

    Uses KP12/KP13 to anchor the service lines, then finds baselines
    by position relative to them.

    Returns dict with keys like 'far_service', 'near_service',
    'far_baseline', 'near_baseline', 'center_service'.
    """
    result: dict[str, np.ndarray | None] = {
        "far_service": None,
        "near_service": None,
        "far_baseline": None,
        "near_baseline": None,
        "center_service": None,
    }

    kp12 = cnn_kps[12] if len(cnn_kps) > 12 else (None, None)
    kp13 = cnn_kps[13] if len(cnn_kps) > 13 else (None, None)
    if kp12[0] is None or kp13[0] is None:
        return result

    center_x = (kp12[0] + kp13[0]) / 2.0
    kp12_y = kp12[1]
    kp13_y = kp13[1]

    # Only consider lines that span a significant width and have low slope
    def is_valid_court_line(seg: np.ndarray, min_length: float = 150) -> bool:
        x1, y1, x2, y2 = seg.astype(float)
        length = np.hypot(x2 - x1, y2 - y1)
        if length < min_length:
            return False
        x_span = abs(x2 - x1)
        y_span = abs(y2 - y1)
        if x_span < 50:
            return False
        # Slope check: court lines should be nearly horizontal
        if y_span / max(x_span, 1) > 0.10:
            return False
        return True

    # Score horizontal lines: compute y_at_center for valid court lines
    scored_h: list[tuple[float, float, np.ndarray]] = []
    for seg in horizontal:
        if not is_valid_court_line(seg):
            continue
        y_at_center = _line_y_at_x(seg, center_x)
        if y_at_center is None:
            continue
        length = np.hypot(seg[2] - seg[0], seg[3] - seg[1])
        scored_h.append((y_at_center, length, seg))

    if not scored_h:
        return result

    # Sort by y_at_center
    scored_h.sort(key=lambda x: x[0])

    # Debug: print valid court line candidates (only during step 5)
    for y, ln, seg in scored_h:
        left, right = _seg_left_right(seg)
        y_kp12 = _line_y_at_x(seg, kp12[0])
        y_kp13 = _line_y_at_x(seg, kp13[0])
        if y_kp12 is not None and y_kp13 is not None:
            print(f"    [candidate] y@cx={y:.0f} y@kp12={y_kp12:.0f} y@kp13={y_kp13:.0f} "
                  f"len={ln:.0f} ({left[0]},{left[1]})->({right[0]},{right[1]})")

    # Far service line: must pass near KP12, prefer wide lines
    far_sv_candidates = []
    for y_at_cx, length, seg in scored_h:
        y_at_kp12 = _line_y_at_x(seg, kp12[0])
        if y_at_kp12 is not None and abs(y_at_kp12 - kp12_y) < 30 and length > 600:
            # Score: prefer long lines (weight length heavily)
            score = abs(y_at_kp12 - kp12_y) - length / 200.0
            far_sv_candidates.append((score, seg))
    if far_sv_candidates:
        far_sv_candidates.sort()
        result["far_service"] = far_sv_candidates[0][1]

    # Near service line: must pass near KP13, prefer wide lines
    near_sv_candidates = []
    for y_at_cx, length, seg in scored_h:
        y_at_kp13 = _line_y_at_x(seg, kp13[0])
        if y_at_kp13 is not None and abs(y_at_kp13 - kp13_y) < 30 and length > 600:
            score = abs(y_at_kp13 - kp13_y) - length / 200.0
            near_sv_candidates.append((score, seg))
    if near_sv_candidates:
        near_sv_candidates.sort()
        result["near_service"] = near_sv_candidates[0][1]

    # Fallback: synthesize near service line from KP13 + far_service slope.
    # The net (at y≈660) is between the service lines and must NOT be used
    # as the near service line.  If no detected line passes within 30px of
    # KP13, we construct a synthetic line through KP13 with the same slope
    # as far_service, spanning the same x-range.
    if result["near_service"] is None and result["far_service"] is not None:
        fs = result["far_service"].astype(float)
        fs_left, fs_right = _seg_left_right(result["far_service"])
        dx = fs[2] - fs[0]
        dy = fs[3] - fs[1]
        slope = dy / max(abs(dx), 1)
        x_left = float(fs_left[0])
        x_right = float(fs_right[0])
        y_left = kp13_y + slope * (x_left - kp13[0])
        y_right = kp13_y + slope * (x_right - kp13[0])
        synth = np.array([x_left, y_left, x_right, y_right])
        result["near_service"] = synth
        print(f"    [synth] near_service from KP13 + far_service slope: "
              f"({x_left:.0f},{y_left:.0f})->({x_right:.0f},{y_right:.0f})")

    # Far baseline: above far service line, significant length
    if result["far_service"] is not None:
        fs_y = _line_y_at_x(result["far_service"], center_x)
        far_bl_candidates = [
            (y, length, seg)
            for y, length, seg in scored_h
            if y < fs_y - 10 and length > 150
        ]
        if far_bl_candidates:
            # Pick the closest to the far service line (lowest y above it)
            far_bl_candidates.sort(key=lambda x: -x[0])  # highest y first (closest to fs)
            result["far_baseline"] = far_bl_candidates[0][2]

    # Near baseline: below near service line, significant length
    if result["near_service"] is not None:
        ns_y = _line_y_at_x(result["near_service"], center_x)
        near_bl_candidates = [
            (y, length, seg)
            for y, length, seg in scored_h
            if y > ns_y + 15 and length > 200
        ]
        if near_bl_candidates:
            # Pick the longest line below the near service line
            near_bl_candidates.sort(key=lambda x: -x[1])  # longest first
            result["near_baseline"] = near_bl_candidates[0][2]
    else:
        # Fallback: find any long line below KP13
        near_bl_candidates = [
            (y, length, seg)
            for y, length, seg in scored_h
            if y > kp13_y + 20 and length > 300
        ]
        if near_bl_candidates:
            near_bl_candidates.sort(key=lambda x: -x[1])
            result["near_baseline"] = near_bl_candidates[0][2]

    # Center service line: vertical line passing near both KP12 and KP13
    for seg in vertical:
        x1, y1, x2, y2 = seg.astype(float)
        dy = y2 - y1
        if abs(dy) > 1:
            t12 = (kp12_y - y1) / dy
            t13 = (kp13_y - y1) / dy
            x_at_kp12 = x1 + t12 * (x2 - x1)
            x_at_kp13 = x1 + t13 * (x2 - x1)
            if abs(x_at_kp12 - kp12[0]) < 40 and abs(x_at_kp13 - kp13[0]) < 40:
                result["center_service"] = seg
                break

    return result


def match_intersections_to_keypoints(
    intersections: list[tuple[float, float]],
    cnn_kps: list[tuple[float | None, float | None]],
    horizontal: list[np.ndarray] | None = None,
    vertical: list[np.ndarray] | None = None,
) -> list[tuple[int, tuple[float, float]]]:
    """Label court keypoints using structural identification.

    Strategy:
    1. Always include KP12 and KP13 from CNN.
    2. Identify service lines and baselines from horizontal lines.
    3. Estimate sideline directions from service line endpoints.
    4. Compute keypoints as intersections of identified lines.

    Returns list of (keypoint_index, (px, py)).
    """
    matches: list[tuple[int, tuple[float, float]]] = []

    kp12 = cnn_kps[12] if len(cnn_kps) > 12 else (None, None)
    kp13 = cnn_kps[13] if len(cnn_kps) > 13 else (None, None)
    if kp12[0] is None or kp13[0] is None:
        return matches

    matches.append((12, (kp12[0], kp12[1])))
    matches.append((13, (kp13[0], kp13[1])))
    assigned: set[int] = {12, 13}

    if horizontal is None or vertical is None:
        return matches

    court_lines = identify_court_lines(horizontal, vertical, cnn_kps)

    # Log identified court lines
    for name, seg in court_lines.items():
        if seg is not None:
            left, right = _seg_left_right(seg)
            print(f"    [court_line] {name}: ({left[0]},{left[1]})->({right[0]},{right[1]})")

    fs = court_lines["far_service"]
    ns = court_lines["near_service"]
    fb = court_lines["far_baseline"]
    nb = court_lines["near_baseline"]

    # Estimate sidelines using vanishing-point geometry.
    # The center service line (KP13→KP12) gives the depth vanishing point.
    # All court-depth lines (sidelines, CSL) converge to this VP.
    # The far service line is fully visible, so its endpoints approximate
    # the left/right singles sideline positions at that depth.
    # We draw lines from VP through those endpoints → sideline estimates.
    left_sideline = None  # as a 4-element segment
    right_sideline = None

    # Compute depth vanishing point from KP12/KP13
    dx_vp = kp12[0] - kp13[0]  # typically small (~-3.6px)
    dy_vp = kp12[1] - kp13[1]  # typically large (~-133px)
    # Extend to find VP (where KP13→KP12 line meets far horizon)
    if abs(dy_vp) > 1:
        # VP is where this line crosses y=0 (or very far)
        t_vp = -kp12[1] / dy_vp  # t to reach y=0
        vp_x = kp12[0] + t_vp * dx_vp
        vp_y = 0.0
        # For numerical stability, use a point 2000px above KP12
        t_far = -2000.0 / dy_vp
        vp_far_x = kp12[0] + t_far * dx_vp
        vp_far_y = kp12[1] + t_far * dy_vp
    else:
        vp_far_x = kp12[0]
        vp_far_y = -2000.0

    if fs is not None and ns is not None:
        fs_left, fs_right = _seg_left_right(fs)

        # Left sideline: from VP through far_service left endpoint
        left_sideline = np.array([vp_far_x, vp_far_y, fs_left[0], fs_left[1]])
        # Right sideline: from VP through far_service right endpoint
        right_sideline = np.array([vp_far_x, vp_far_y, fs_right[0], fs_right[1]])

        # Log: show where sidelines cross the near service line for verification
        ns_arr = np.array([ns[0], ns[1], ns[2], ns[3]])
        sl_left_ns = line_intersection(left_sideline, ns_arr, w=FRAME_W * 3, h=FRAME_H * 3)
        sl_right_ns = line_intersection(right_sideline, ns_arr, w=FRAME_W * 3, h=FRAME_H * 3)
        print(f"    [VP] depth VP far point: ({vp_far_x:.0f},{vp_far_y:.0f})")
        print(f"    [estimated] left_sideline: VP->({fs_left[0]},{fs_left[1]})")
        print(f"    [estimated] right_sideline: VP->({fs_right[0]},{fs_right[1]})")
        if sl_left_ns and sl_right_ns:
            near_sv_width = abs(sl_right_ns[0] - sl_left_ns[0])
            print(f"    [check] near service width via sidelines: {near_sv_width:.0f}px "
                  f"(L=({sl_left_ns[0]:.0f},{sl_left_ns[1]:.0f}) "
                  f"R=({sl_right_ns[0]:.0f},{sl_right_ns[1]:.0f}))")

    # Helper: intersect two line segments (extended to full lines)
    def _intersect(seg1: np.ndarray, seg2: np.ndarray) -> tuple[float, float] | None:
        return line_intersection(seg1, seg2, w=FRAME_W * 2, h=FRAME_H * 2)

    # --- Service box corners: far service line × sidelines ---
    if fs is not None:
        fs_arr = np.array([fs[0], fs[1], fs[2], fs[3]])
        if left_sideline is not None:
            pt = _intersect(fs_arr, left_sideline)
            if pt and 8 not in assigned:
                matches.append((8, pt))
                assigned.add(8)
        if right_sideline is not None:
            pt = _intersect(fs_arr, right_sideline)
            if pt and 9 not in assigned:
                matches.append((9, pt))
                assigned.add(9)

    # Near service line × sidelines → KP10, KP11
    if ns is not None:
        ns_arr = np.array([ns[0], ns[1], ns[2], ns[3]])
        if left_sideline is not None:
            pt = _intersect(ns_arr, left_sideline)
            if pt and 10 not in assigned:
                matches.append((10, pt))
                assigned.add(10)
        if right_sideline is not None:
            pt = _intersect(ns_arr, right_sideline)
            if pt and 11 not in assigned:
                matches.append((11, pt))
                assigned.add(11)

    # --- Far baseline × sidelines → KP4/KP6 (singles) or KP0/KP1 (doubles) ---
    if fb is not None:
        fb_arr = np.array([fb[0], fb[1], fb[2], fb[3]])
        fb_left_pt = None
        fb_right_pt = None
        if left_sideline is not None:
            fb_left_pt = _intersect(fb_arr, left_sideline)
        if right_sideline is not None:
            fb_right_pt = _intersect(fb_arr, right_sideline)

        # Determine if this is the singles or doubles baseline
        # by comparing width with service line width
        if fb_left_pt and fb_right_pt:
            fb_width = abs(fb_right_pt[0] - fb_left_pt[0])
            if fs is not None:
                fs_left, fs_right = _seg_left_right(fs)
                sv_width = abs(fs_right[0] - fs_left[0])
                if fb_width > sv_width * 1.15:
                    # Doubles baseline (wider than singles service line)
                    if 0 not in assigned:
                        matches.append((0, fb_left_pt))
                        assigned.add(0)
                    if 1 not in assigned:
                        matches.append((1, fb_right_pt))
                        assigned.add(1)
                else:
                    if 4 not in assigned:
                        matches.append((4, fb_left_pt))
                        assigned.add(4)
                    if 6 not in assigned:
                        matches.append((6, fb_right_pt))
                        assigned.add(6)
            else:
                if 4 not in assigned:
                    matches.append((4, fb_left_pt))
                    assigned.add(4)
                if 6 not in assigned:
                    matches.append((6, fb_right_pt))
                    assigned.add(6)

    # --- Near baseline × sidelines → KP5/KP7 (singles) or KP2/KP3 (doubles) ---
    if nb is not None:
        nb_arr = np.array([nb[0], nb[1], nb[2], nb[3]])
        nb_left_pt = None
        nb_right_pt = None
        if left_sideline is not None:
            nb_left_pt = _intersect(nb_arr, left_sideline)
        if right_sideline is not None:
            nb_right_pt = _intersect(nb_arr, right_sideline)

        if nb_left_pt and nb_right_pt:
            nb_width = abs(nb_right_pt[0] - nb_left_pt[0])
            if ns is not None:
                ns_left, ns_right = _seg_left_right(ns)
                sv_width = abs(ns_right[0] - ns_left[0])
                if nb_width > sv_width * 1.15:
                    if 2 not in assigned:
                        matches.append((2, nb_left_pt))
                        assigned.add(2)
                    if 3 not in assigned:
                        matches.append((3, nb_right_pt))
                        assigned.add(3)
                else:
                    if 5 not in assigned:
                        matches.append((5, nb_left_pt))
                        assigned.add(5)
                    if 7 not in assigned:
                        matches.append((7, nb_right_pt))
                        assigned.add(7)
            else:
                if 5 not in assigned:
                    matches.append((5, nb_left_pt))
                    assigned.add(5)
                if 7 not in assigned:
                    matches.append((7, nb_right_pt))
                    assigned.add(7)

    return matches


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def draw_lines_on_frame(
    frame: np.ndarray,
    lines: np.ndarray | None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    vis = frame.copy()
    if lines is not None:
        for seg in lines:
            x1, y1, x2, y2 = seg if seg.ndim == 1 else seg[0]
            cv2.line(vis, (x1, y1), (x2, y2), color, thickness)
    return vis


def draw_classified_lines(
    frame: np.ndarray,
    horizontal: list[np.ndarray],
    vertical: list[np.ndarray],
) -> np.ndarray:
    vis = frame.copy()
    for seg in horizontal:
        cv2.line(vis, (seg[0], seg[1]), (seg[2], seg[3]), (255, 100, 0), 2)
    for seg in vertical:
        cv2.line(vis, (seg[0], seg[1]), (seg[2], seg[3]), (0, 0, 255), 2)
    return vis


def draw_intersections(
    frame: np.ndarray,
    intersections: list[tuple[float, float]],
) -> np.ndarray:
    vis = frame.copy()
    for x, y in intersections:
        ix, iy = int(round(x)), int(round(y))
        cv2.circle(vis, (ix, iy), 8, (0, 255, 0), 2)
        cv2.circle(vis, (ix, iy), 2, (0, 255, 0), -1)
    return vis


def draw_matched_keypoints(
    frame: np.ndarray,
    matches: list[tuple[int, tuple[float, float]]],
    cnn_kps: list[tuple[float | None, float | None]] | None = None,
) -> np.ndarray:
    vis = frame.copy()
    for kp_idx, (px, py) in matches:
        ix, iy = int(round(px)), int(round(py))
        cv2.circle(vis, (ix, iy), 10, (0, 255, 255), 2)
        label = f"KP{kp_idx} ({KP_NAMES[kp_idx]})"
        cv2.putText(vis, label, (ix + 12, iy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Draw CNN keypoints in magenta for comparison
    if cnn_kps:
        for i, (cx, cy) in enumerate(cnn_kps):
            if cx is not None and cy is not None:
                ix, iy = int(round(cx)), int(round(cy))
                cv2.circle(vis, (ix, iy), 12, (255, 0, 255), 2)
                cv2.putText(vis, f"CNN-KP{i}", (ix + 14, iy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    return vis


def make_composite(images: list[np.ndarray], titles: list[str], cols: int = 3) -> np.ndarray:
    """Tile images into a composite grid with titles."""
    thumb_w, thumb_h = 640, 360
    rows = (len(images) + cols - 1) // cols
    canvas = np.zeros((rows * (thumb_h + 30), cols * thumb_w, 3), dtype=np.uint8)

    for idx, (img, title) in enumerate(zip(images, titles)):
        r, c = divmod(idx, cols)
        # Ensure 3-channel
        if img.ndim == 2:
            img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img3 = img
        thumb = cv2.resize(img3, (thumb_w, thumb_h))
        y0 = r * (thumb_h + 30) + 30
        x0 = c * thumb_w
        canvas[y0:y0 + thumb_h, x0:x0 + thumb_w] = thumb
        cv2.putText(canvas, title, (x0 + 10, y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("  Classical Court-Line Detection Experiment (v2 — courtside fixes)")
    print("=" * 72)

    # ---- Load video frames ------------------------------------------------
    print(f"\n[1] Loading frames from {VIDEO_PATH}")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open video {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"    Video: {w}x{h} @ {fps:.1f} fps, {total_frames} frames")

    frames: dict[int, np.ndarray] = {}
    for idx in FRAME_INDICES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames[idx] = frame
            print(f"    ✓ Frame {idx} loaded ({idx/fps:.1f}s)")
        else:
            print(f"    ✗ Frame {idx} could not be read")
    cap.release()

    if not frames:
        sys.exit("ERROR: no frames loaded")

    # ---- CNN court detector -----------------------------------------------
    print("\n[2] Running CNN court detector for anchor keypoints...")
    try:
        config = get_config()
        cnn_detector = CourtDetector(config, use_refine_kps=False, use_homography=False)
        cnn_results: dict[int, list[tuple[float | None, float | None]]] = {}
        for idx, frame in frames.items():
            result = cnn_detector.detect_frame(frame, frame_number=idx)
            cnn_results[idx] = result.keypoints
            detected_ids = [
                i for i, (x, y) in enumerate(result.keypoints)
                if x is not None
            ]
            print(f"    Frame {idx}: CNN detected {result.num_detected} keypoints: {detected_ids}")
            for ki in detected_ids:
                kx, ky = result.keypoints[ki]
                print(f"      KP{ki} ({KP_NAMES[ki]}): ({kx:.1f}, {ky:.1f})")
    except Exception as e:
        print(f"    CNN detector failed: {e}")
        cnn_results = {idx: [(None, None)] * 14 for idx in frames}

    # ---- Pre-compute ROI masks per frame (Fix 1 + Fix 2) ------------------
    print("\n[2b] Computing court-surface and ROI masks...")
    court_masks: dict[int, np.ndarray] = {}
    roi_masks: dict[int, np.ndarray] = {}
    for idx, frame in frames.items():
        cm = detect_court_surface(frame)
        court_masks[idx] = cm
        rm = build_roi_mask(frame, cm)
        roi_masks[idx] = rm
        court_pct = 100.0 * np.count_nonzero(cm) / cm.size
        roi_pct = 100.0 * np.count_nonzero(rm) / rm.size
        print(f"    Frame {idx}: court_surface={court_pct:.1f}% ROI={roi_pct:.1f}%")

    # ---- Classical detection: sweep parameters (Hough) --------------------
    print("\n[3] Sweeping classical detection parameters (Hough + ROI)...")
    print("-" * 72)

    best_results: dict[int, LineDetectionResult | None] = {idx: None for idx in frames}

    for idx, frame in frames.items():
        print(f"\n  === Frame {idx} (Hough) ===")
        best_score = -1
        roi = roi_masks[idx]

        for wt in WHITE_THRESHOLDS:
            for hcfg in HOUGH_CONFIGS:
                lines, wmask, cleaned, edges, filt_white = detect_court_lines(
                    frame,
                    white_thresh=wt,
                    hough_threshold=hcfg["threshold"],
                    hough_min_length=hcfg["minLineLength"],
                    hough_max_gap=hcfg["maxLineGap"],
                    roi_mask=roi,
                )
                horiz, vert = classify_lines(lines)
                intersections = find_intersections(horiz, vert)
                intersections = deduplicate_points(intersections, radius=20)

                n_lines = 0 if lines is None else len(lines)
                # Scoring: reward having both H & V lines + intersections
                score = min(len(horiz), 10) + min(len(vert), 10) + len(intersections) * 3
                tag = f"wt={wt} hough(t={hcfg['threshold']},ml={hcfg['minLineLength']},mg={hcfg['maxLineGap']})"
                print(f"    {tag}: lines={n_lines} H={len(horiz)} V={len(vert)} ints={len(intersections)} score={score}")

                if score > best_score:
                    best_score = score
                    best_results[idx] = LineDetectionResult(
                        frame_idx=idx,
                        white_thresh=wt,
                        hough_cfg=hcfg,
                        lines=lines,
                        horizontal=horiz,
                        vertical=vert,
                        intersections=intersections,
                        white_mask=wmask,
                        cleaned=cleaned,
                        edges=edges,
                        court_color_mask=court_masks[idx],
                        roi_mask=roi,
                        filtered_white_mask=filt_white,
                        detection_method="hough",
                    )

        res = best_results[idx]
        if res:
            print(f"  >> Best Hough for frame {idx}: wt={res.white_thresh} "
                  f"hough={res.hough_cfg} → L={res.n_lines} H={res.n_horizontal} "
                  f"V={res.n_vertical} ints={len(res.intersections)}")

    # ---- LSD detection (Fix 5) --------------------------------------------
    print("\n[3b] Trying LSD detection (alternative to Hough)...")
    lsd_results: dict[int, LineDetectionResult | None] = {idx: None for idx in frames}

    for idx, frame in frames.items():
        print(f"\n  === Frame {idx} (LSD) ===")
        roi = roi_masks[idx]
        best_score = -1

        for wt in WHITE_THRESHOLDS:
            lines, wmask, cleaned, edges, filt_white = detect_court_lines_lsd(
                frame, white_thresh=wt, roi_mask=roi, min_length=60,
            )
            horiz, vert = classify_lines(lines)
            intersections = find_intersections(horiz, vert)
            intersections = deduplicate_points(intersections, radius=20)

            n_lines = 0 if lines is None else len(lines)
            score = min(len(horiz), 10) + min(len(vert), 10) + len(intersections) * 3
            print(f"    wt={wt}: lines={n_lines} H={len(horiz)} V={len(vert)} ints={len(intersections)} score={score}")

            if score > best_score:
                best_score = score
                lsd_results[idx] = LineDetectionResult(
                    frame_idx=idx,
                    white_thresh=wt,
                    hough_cfg={},
                    lines=lines,
                    horizontal=horiz,
                    vertical=vert,
                    intersections=intersections,
                    white_mask=wmask,
                    cleaned=cleaned,
                    edges=edges,
                    court_color_mask=court_masks[idx],
                    roi_mask=roi,
                    filtered_white_mask=filt_white,
                    detection_method="lsd",
                )

        lsd_res = lsd_results[idx]
        if lsd_res:
            print(f"  >> Best LSD for frame {idx}: wt={lsd_res.white_thresh} "
                  f"→ L={lsd_res.n_lines} H={lsd_res.n_horizontal} "
                  f"V={lsd_res.n_vertical} ints={len(lsd_res.intersections)}")

    # ---- Pick best between Hough and LSD per frame ------------------------
    print("\n[3c] Selecting best method per frame (Hough vs LSD)...")
    for idx in frames:
        hough_r = best_results[idx]
        lsd_r = lsd_results[idx]
        h_ints = len(hough_r.intersections) if hough_r else 0
        l_ints = len(lsd_r.intersections) if lsd_r else 0
        # Prefer whichever found more intersections with spatial spread
        if lsd_r and l_ints > h_ints:
            best_results[idx] = lsd_r
            print(f"  Frame {idx}: LSD wins ({l_ints} vs {h_ints} intersections)")
        else:
            print(f"  Frame {idx}: Hough wins ({h_ints} vs {l_ints} intersections)")

    # ---- Merge collinear segments & recompute intersections ---------------
    print("\n[4] Merging collinear segments and recomputing intersections...")
    for idx in frames:
        res = best_results[idx]
        if res is None:
            continue
        merged_h = merge_collinear_segments(res.horizontal, dist_tol=40.0)
        merged_v = merge_collinear_segments(res.vertical)
        new_ints = find_intersections(merged_h, merged_v)
        new_ints = deduplicate_points(new_ints, radius=25)
        print(f"  Frame {idx}: {res.n_horizontal}→{len(merged_h)} horiz, "
              f"{res.n_vertical}→{len(merged_v)} vert, "
              f"{len(res.intersections)}→{len(new_ints)} intersections")
        res.horizontal = merged_h
        res.vertical = merged_v
        res.intersections = new_ints
        res.n_horizontal = len(merged_h)
        res.n_vertical = len(merged_v)

    # ---- Match intersections to keypoints & homography --------------------
    print("\n[5] Matching intersections to court keypoints & computing homography...")
    homographies: dict[int, np.ndarray | None] = {}

    for idx in frames:
        res = best_results[idx]
        if res is None:
            print(f"  Frame {idx}: no detection result")
            homographies[idx] = None
            continue

        cnn_kps = cnn_results.get(idx, [(None, None)] * 14)
        matches = match_intersections_to_keypoints(
            res.intersections, cnn_kps,
            horizontal=res.horizontal, vertical=res.vertical,
        )
        print(f"  Frame {idx} ({res.detection_method}): {len(matches)} matched keypoints")

        # Report spatial spread of matched points
        if len(matches) > 2:
            xs = [px for _, (px, _) in matches]
            ys = [py for _, (_, py) in matches]
            print(f"    x-range: [{min(xs):.0f}, {max(xs):.0f}]  "
                  f"y-range: [{min(ys):.0f}, {max(ys):.0f}]  "
                  f"spread: {max(xs)-min(xs):.0f}x{max(ys)-min(ys):.0f}px")

        for kp_idx, (px, py) in matches:
            print(f"    KP{kp_idx} ({KP_NAMES[kp_idx]}): ({px:.1f}, {py:.1f})")

        # Attempt homography
        if len(matches) >= 4:
            img_pts = [pt for _, pt in matches]
            court_pts = [REFERENCE_KPS_METERS[ki] for ki, _ in matches]
            H = try_compute_homography(img_pts, court_pts)
            homographies[idx] = H
            if H is not None:
                print(f"    ✓ Homography computed ({len(matches)} point correspondences)")
                sanity = homography_sanity_check(H)
                print("    Sanity check (court corners → pixel):")
                all_ok = True
                for name, (px, py) in sanity.items():
                    ok = -200 <= px <= FRAME_W + 200 and -200 <= py <= FRAME_H + 200
                    if not ok:
                        all_ok = False
                    print(f"      {name}: ({px:.1f}, {py:.1f}) {'✓' if ok else '⚠ out of frame'}")
                if all_ok:
                    print("    → Homography looks REASONABLE ✓")
                else:
                    print("    → Homography is DEGENERATE ⚠")
            else:
                print("    ✗ Homography computation failed (RANSAC)")
        else:
            homographies[idx] = None
            print(f"    ✗ Not enough correspondences for homography (need ≥4, have {len(matches)})")

    # ---- Save diagnostic images -------------------------------------------
    print("\n[6] Saving diagnostic images...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for idx, frame in frames.items():
        res = best_results[idx]
        if res is None:
            continue

        cnn_kps = cnn_results.get(idx, [(None, None)] * 14)
        matches = match_intersections_to_keypoints(
            res.intersections, cnn_kps,
            horizontal=res.horizontal, vertical=res.vertical,
        )
        vis_lines = draw_lines_on_frame(frame, res.lines)
        vis_classified = draw_classified_lines(frame, res.horizontal, res.vertical)
        vis_intersections = draw_intersections(vis_classified, res.intersections)
        vis_kps = draw_matched_keypoints(frame, matches, cnn_kps)

        # Diagnostic masks
        court_vis = court_masks.get(idx, np.zeros_like(frame[:, :, 0]))
        roi_vis = roi_masks.get(idx, np.zeros_like(frame[:, :, 0]))
        filt_white_vis = res.filtered_white_mask if res.filtered_white_mask is not None else res.cleaned

        imgs = [
            (frame, "Original"),
            (court_vis, "Court color mask"),
            (roi_vis, "ROI mask"),
            (res.white_mask, "White mask (raw)"),
            (filt_white_vis, "White mask (filtered)"),
            (res.cleaned, "Cleaned mask"),
            (res.edges, "Canny/LSD edges"),
            (vis_lines, f"All {res.detection_method} lines"),
            (vis_classified, "H(blue) V(red) lines"),
            (vis_intersections, "Intersections"),
            (vis_kps, "Matched keypoints"),
        ]

        # Save composite
        composite = make_composite(
            [im for im, _ in imgs],
            [t for _, t in imgs],
            cols=4,
        )
        comp_path = OUTPUT_DIR / f"classical_court_frame_{idx}.jpg"
        cv2.imwrite(str(comp_path), composite)
        print(f"    Saved {comp_path}")

        # Save individual classified + intersections image (high-res)
        detail_path = OUTPUT_DIR / f"classical_court_detail_{idx}.jpg"
        cv2.imwrite(str(detail_path), vis_kps)
        print(f"    Saved {detail_path}")

    # ---- Final report -----------------------------------------------------
    print("\n" + "=" * 72)
    print("  FINAL REPORT")
    print("=" * 72)

    for idx in FRAME_INDICES:
        res = best_results.get(idx)
        if res is None:
            print(f"\n  Frame {idx}: NO RESULT")
            continue

        cnn_kps = cnn_results.get(idx, [(None, None)] * 14)
        matches = match_intersections_to_keypoints(
            res.intersections, cnn_kps,
            horizontal=res.horizontal, vertical=res.vertical,
        )
        H = homographies.get(idx)

        print(f"\n  Frame {idx} (t={idx/fps:.1f}s) [{res.detection_method}]:")
        print(f"    Best params: white_thresh={res.white_thresh}, hough={res.hough_cfg}")
        print(f"    Total lines detected:         {res.n_lines}")
        print(f"    Horizontal lines (merged):    {res.n_horizontal}")
        print(f"    Vertical lines (merged):      {res.n_vertical}")
        print(f"    Intersections (deduplicated):  {len(res.intersections)}")
        print(f"    Matched court keypoints:       {len(matches)}")

        # Spatial spread analysis
        matched_kp_ids = [ki for ki, _ in matches]
        has_sideline_kps = any(ki in {0, 1, 2, 3, 4, 5, 6, 7} for ki in matched_kp_ids)
        has_service_kps = any(ki in {8, 9, 10, 11} for ki in matched_kp_ids)
        print(f"    Sideline keypoints matched:   {'YES ✓' if has_sideline_kps else 'NO ✗'}")
        print(f"    Service box kps matched:      {'YES ✓' if has_service_kps else 'NO ✗'}")

        if len(matches) > 2:
            xs = [px for _, (px, _) in matches]
            print(f"    x-spread of matches:          {max(xs)-min(xs):.0f}px "
                  f"(range [{min(xs):.0f}, {max(xs):.0f}])")

        for kp_idx, (px, py) in matches:
            print(f"      KP{kp_idx:2d} ({KP_NAMES[kp_idx]:6s}): ({px:7.1f}, {py:7.1f})")
        if H is not None:
            print(f"    Homography: COMPUTED ✓")
            sanity = homography_sanity_check(H)
            for name, (px, py) in sanity.items():
                ok = -200 <= px <= FRAME_W + 200 and -200 <= py <= FRAME_H + 200
                print(f"      {name}: ({px:.1f}, {py:.1f}) {'✓' if ok else '⚠'}")
        else:
            print(f"    Homography: NOT COMPUTED ✗")

    # ---- Summary ----------------------------------------------------------
    any_homography = any(h is not None for h in homographies.values())
    total_intersections = sum(
        len(r.intersections) for r in best_results.values() if r is not None
    )
    print(f"\n  Overall summary:")
    print(f"    Frames processed:     {len(frames)}")
    print(f"    Total intersections:  {total_intersections}")
    print(f"    Homography computed:  {'YES' if any_homography else 'NO'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
