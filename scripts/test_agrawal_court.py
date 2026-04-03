#!/usr/bin/env python3
"""Agrawal et al. (2024) inspired court detection pipeline.

Implements: court-color filtering → near-half extraction → Hough lines →
near-half homography → full-court extension.

Usage:
    python scripts/test_agrawal_court.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.features.court_detect.detector import CourtDetector, CourtDetectionResult
from src.features.court_detect.court_template import REFERENCE_KPS_METERS

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------
VIDEO_PATH = PROJECT_ROOT / "tests" / "fixtures" / "real_match_10min.mp4"
OUTPUT_DIR = PROJECT_ROOT / "output"
FRAME_INDICES = [900, 1500, 2700, 3600, 5400, 7200, 9000, 10800, 13500, 16200]
FRAME_W, FRAME_H = 1920, 1080

# Court geometry (ITF, meters from net center origin)
NET_Y_METERS = 0.0
NEAR_BASELINE_Y = 11.885
SERVICE_LINE_Y = 6.4

# Near-half keypoint indices → meter coords
NEAR_HALF_KPS = {
    2:  (-5.485, 11.885),
    3:  (5.485,  11.885),
    5:  (-4.115, 11.885),
    7:  (4.115,  11.885),
    10: (-4.115, 6.4),
    11: (4.115,  6.4),
    13: (0.0,    6.4),
}

FAR_HALF_KPS = {
    0:  (-5.485, -11.885),
    1:  (5.485,  -11.885),
    4:  (-4.115, -11.885),
    6:  (4.115,  -11.885),
    8:  (-4.115, -6.4),
    9:  (4.115,  -6.4),
    12: (0.0,    -6.4),
}

KP_NAMES = [
    "BL-TL", "BL-TR", "BL-BL", "BL-BR",
    "SL-TL", "SL-BL", "SL-TR", "SL-BR",
    "SV-TL", "SV-TR", "SV-BL", "SV-BR",
    "CT-T",  "CT-B",
]

# Algorithm parameters
N_COLOR_SAMPLES = 2000
COLOR_MATCH_H_THRESH = 15
COLOR_MATCH_S_THRESH = 40
COLOR_MATCH_V_THRESH = 60
NEIGHBORHOOD_SIZE = 7
MIN_COURT_NEIGHBORS = 4
LINE_V_THRESHOLD = 150
LINE_S_THRESHOLD = 80
HOUGH_THRESHOLD = 60
HOUGH_MIN_LENGTH = 100
HOUGH_MAX_GAP = 40


# ===================================================================
# Stage 1: Net detection / near-half crop
# ===================================================================

def estimate_net_y_from_cnn(
    frame: np.ndarray,
    court_detector: CourtDetector,
    frame_number: int = 0,
) -> tuple[int, CourtDetectionResult]:
    """Estimate net y-position using CNN keypoints.

    KP12 is the far service line center T (appears near top of court area).
    The net sits ABOVE KP12 in the frame (lower y value).
    """
    result = court_detector.detect_frame(frame, frame_number=frame_number)
    kps = result.keypoints

    # Collect y-coords of KP8, KP9, KP12 (all at far service line level)
    far_service_ys = []
    for idx in [8, 9, 12]:
        if idx < len(kps) and kps[idx][0] is not None:
            far_service_ys.append(kps[idx][1])

    if far_service_ys:
        # Net is slightly above (lower y) the far service line detections
        net_y = int(min(far_service_ys) - 20)
        net_y = max(0, net_y)
    else:
        # CNN failed entirely — use heuristic
        net_y = estimate_net_y_heuristic(frame)

    return net_y, result


def estimate_net_y_heuristic(frame: np.ndarray) -> int:
    """Fallback: find net as strongest horizontal edge in middle band."""
    h = frame.shape[0]
    y_lo = int(0.35 * h)
    y_hi = int(0.55 * h)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    band = gray[y_lo:y_hi, :]

    # Horizontal Sobel → strongest horizontal edge
    sobel = cv2.Sobel(band, cv2.CV_64F, 0, 1, ksize=3)
    row_energy = np.sum(np.abs(sobel), axis=1)
    best_row = int(np.argmax(row_energy))

    return y_lo + best_row


def crop_near_half(
    frame: np.ndarray,
    net_y: int,
    margin: int = 20,
) -> tuple[np.ndarray, int]:
    """Crop frame to near-half court (below net).

    Returns (cropped, y_offset) where y_offset is the row in the original
    frame where the crop starts.
    """
    y_start = max(0, net_y - margin)
    cropped = frame[y_start:, :].copy()
    return cropped, y_start


# ===================================================================
# Stage 2: Court-color detection
# ===================================================================

def detect_court_color_kmeans(
    frame_bgr: np.ndarray,
    roi_mask: np.ndarray | None = None,
    n_samples: int = 2000,
    k: int = 3,
) -> tuple[np.ndarray, str]:
    """Detect dominant court surface color using k-means on HSV samples.

    Samples from the center of the frame to avoid the surround area.
    Returns (court_color_hsv [H,S,V], court_type).
    """
    from sklearn.cluster import KMeans

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h_img, w_img = hsv.shape[:2]

    # Sample from the CENTER of the image (middle 50% horizontal, 40-80% vertical)
    # to focus on the actual court surface rather than surround
    if roi_mask is not None:
        ys, xs = np.where(roi_mask > 0)
    else:
        y_lo = int(h_img * 0.30)
        y_hi = int(h_img * 0.80)
        x_lo = int(w_img * 0.25)
        x_hi = int(w_img * 0.75)
        ys, xs = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        ys = ys.ravel()
        xs = xs.ravel()

    if len(ys) == 0:
        return np.array([0, 0, 0], dtype=np.uint8), "unknown"

    # Random sample
    n = min(n_samples, len(ys))
    indices = np.random.default_rng(42).choice(len(ys), size=n, replace=False)
    sampled_hsv = hsv[ys[indices], xs[indices]]  # (n, 3)

    # K-means on H, S channels (ignore V for lighting robustness)
    features = sampled_hsv[:, :2].astype(np.float32)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(features)

    # Find largest cluster
    counts = np.bincount(labels, minlength=k)
    dominant = int(np.argmax(counts))
    dominant_mask = labels == dominant
    dominant_pixels = sampled_hsv[dominant_mask]

    court_h = int(np.median(dominant_pixels[:, 0]))
    court_s = int(np.median(dominant_pixels[:, 1]))
    court_v = int(np.median(dominant_pixels[:, 2]))
    court_color = np.array([court_h, court_s, court_v], dtype=np.uint8)

    # Auto-detect court type
    if 90 <= court_h <= 125:
        court_type = "blue"
    elif 35 <= court_h <= 85:
        court_type = "green"
    elif court_h <= 25 or court_h >= 170:
        court_type = "clay"
    else:
        court_type = "unknown"

    return court_color, court_type


# ===================================================================
# Stage 3: Agrawal court-line filter
# ===================================================================

def build_court_color_mask(
    frame_bgr: np.ndarray,
    court_color_hsv: np.ndarray,
    h_thresh: int = COLOR_MATCH_H_THRESH,
    s_thresh: int = COLOR_MATCH_S_THRESH,
    v_thresh: int = COLOR_MATCH_V_THRESH,
    precomputed_hsv: np.ndarray | None = None,
) -> np.ndarray:
    """Binary mask of pixels matching the court surface color (vectorized)."""
    hsv = precomputed_hsv if precomputed_hsv is not None else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    ch, cs, cv_val = int(court_color_hsv[0]), int(court_color_hsv[1]), int(court_color_hsv[2])

    s_lo = max(0, cs - s_thresh)
    s_hi = min(255, cs + s_thresh)
    v_lo = max(0, cv_val - v_thresh)
    v_hi = min(255, cv_val + v_thresh)

    h_lo = ch - h_thresh
    h_hi = ch + h_thresh

    if h_lo < 0:
        # Hue wraps around
        mask1 = cv2.inRange(hsv, np.array([0, s_lo, v_lo]),
                            np.array([h_hi, s_hi, v_hi]))
        mask2 = cv2.inRange(hsv, np.array([180 + h_lo, s_lo, v_lo]),
                            np.array([179, s_hi, v_hi]))
        court_mask = cv2.bitwise_or(mask1, mask2)
    elif h_hi > 179:
        mask1 = cv2.inRange(hsv, np.array([h_lo, s_lo, v_lo]),
                            np.array([179, s_hi, v_hi]))
        mask2 = cv2.inRange(hsv, np.array([0, s_lo, v_lo]),
                            np.array([h_hi - 180, s_hi, v_hi]))
        court_mask = cv2.bitwise_or(mask1, mask2)
    else:
        court_mask = cv2.inRange(hsv, np.array([h_lo, s_lo, v_lo]),
                                 np.array([h_hi, s_hi, v_hi]))

    return court_mask


def agrawal_court_line_filter(
    frame_bgr: np.ndarray,
    court_color_hsv: np.ndarray,
    neighborhood: int = NEIGHBORHOOD_SIZE,
    min_court_neighbors: int = MIN_COURT_NEIGHBORS,
    h_thresh: int = COLOR_MATCH_H_THRESH,
    s_thresh: int = COLOR_MATCH_S_THRESH,
    v_thresh: int = COLOR_MATCH_V_THRESH,
) -> np.ndarray:
    """Agrawal's core algorithm: find pixels that are NOT court-colored
    but are SURROUNDED by court-colored pixels (vectorized via filter2D).

    Returns binary mask (255 = court line pixel).
    """
    # Single HSV conversion, reused for both court mask and line checks
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # 1. Build court-color binary mask (reusing pre-computed HSV)
    court_mask = build_court_color_mask(frame_bgr, court_color_hsv,
                                        h_thresh, s_thresh, v_thresh,
                                        precomputed_hsv=hsv)

    # 2. Count court-color neighbors via convolution
    court_01 = (court_mask > 0).astype(np.float32)
    kernel = np.ones((neighborhood, neighborhood), dtype=np.float32)
    neighbor_count = cv2.filter2D(court_01, cv2.CV_32F, kernel)

    # 3. Court line = NOT court-color AND enough court-color neighbors
    not_court = (court_01 == 0)
    has_neighbors = (neighbor_count >= min_court_neighbors)

    # 4. Brightness/saturation check — court lines are white/bright
    bright = hsv[:, :, 2] > LINE_V_THRESHOLD
    low_sat = hsv[:, :, 1] < LINE_S_THRESHOLD

    line_mask = (not_court & has_neighbors & bright & low_sat).astype(np.uint8) * 255
    return line_mask


# ===================================================================
# Stage 4: Hough line detection + classification (reused logic)
# ===================================================================

def _seg_angle(seg: np.ndarray) -> float:
    x1, y1, x2, y2 = seg
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180


def line_intersection(
    line1: np.ndarray,
    line2: np.ndarray,
    w: int = FRAME_W,
    h: int = FRAME_H,
) -> tuple[float, float] | None:
    """Compute intersection of two lines (extended beyond segments)."""
    x1, y1, x2, y2 = line1.astype(float)
    x3, y3, x4, y4 = line2.astype(float)
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    margin = 100
    if -margin <= ix <= w + margin and -margin <= iy <= h + margin:
        return (float(ix), float(iy))
    return None


def estimate_vanishing_point(lines: list[np.ndarray]) -> np.ndarray | None:
    """Estimate vanishing point from a set of line segments via RANSAC."""
    if len(lines) < 2:
        return None

    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pt = line_intersection(lines[i], lines[j],
                                   w=FRAME_W * 4, h=FRAME_H * 4)
            if pt is not None:
                intersections.append(np.array(pt))

    if not intersections:
        return None

    best_pt = None
    best_count = 0
    for candidate in intersections:
        count = 0
        for seg in lines:
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
    """Classify lines into horizontal (baselines/service) and vertical (sidelines)."""
    horizontal: list[np.ndarray] = []
    vertical: list[np.ndarray] = []
    if lines is None:
        return horizontal, vertical

    all_segs = [seg[0] if seg.ndim == 2 else seg for seg in lines]

    h_candidates: list[np.ndarray] = []
    steep_candidates: list[np.ndarray] = []

    for seg in all_segs:
        angle = _seg_angle(seg)
        # Horizontal: within 15° of horizontal (baselines, service lines)
        if angle < 15 or angle > 165:
            h_candidates.append(seg)
        # Vertical: everything else with noticeable slope (sidelines ~18-24°/156-162°)
        elif 15 <= angle <= 165:
            steep_candidates.append(seg)

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
            mx = (segments[j][0] + segments[j][2]) / 2
            my = (segments[j][1] + segments[j][3]) / 2
            x1, y1, x2, y2 = segments[i].astype(float)
            length = max(np.hypot(x2 - x1, y2 - y1), 1e-6)
            perp_dist = abs((y2 - y1) * mx - (x2 - x1) * my
                            + x2 * y1 - y2 * x1) / length
            if perp_dist < dist_tol:
                group.append(segments[j])
                used[j] = True

        pts = np.concatenate([np.array([[s[0], s[1]], [s[2], s[3]]]) for s in group])
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


def detect_lines_near_half(
    line_mask: np.ndarray,
    hough_threshold: int = HOUGH_THRESHOLD,
    min_length: int = HOUGH_MIN_LENGTH,
    max_gap: int = HOUGH_MAX_GAP,
) -> np.ndarray | None:
    """Run HoughLinesP on the court-line-filtered mask."""
    # Light morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    lines = cv2.HoughLinesP(
        cleaned, 1, np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_length,
        maxLineGap=max_gap,
    )
    return lines


# ===================================================================
# Stage 5: Near-half line identification
# ===================================================================

def _line_y_at_x(seg: np.ndarray, x: float) -> float | None:
    """Evaluate a line segment's y-value at a given x (extending the line)."""
    x1, y1, x2, y2 = seg.astype(float)
    dx = x2 - x1
    if abs(dx) < 1e-6:
        return (y1 + y2) / 2.0
    t = (x - x1) / dx
    return y1 + t * (y2 - y1)


def _line_x_at_y(seg: np.ndarray, y: float) -> float | None:
    """Evaluate a line segment's x-value at a given y."""
    x1, y1, x2, y2 = seg.astype(float)
    dy = y2 - y1
    if abs(dy) < 1e-6:
        return (x1 + x2) / 2.0
    t = (y - y1) / dy
    return x1 + t * (x2 - x1)


def _seg_signed_slope(seg: np.ndarray) -> float:
    """Slope of a segment after normalizing so x1 < x2 (left-to-right)."""
    x1, y1, x2, y2 = seg.astype(float)
    if x2 < x1:
        x1, y1, x2, y2 = x2, y2, x1, y1
    dx = x2 - x1
    if abs(dx) < 1e-6:
        return 0.0
    return (y2 - y1) / dx


def identify_near_half_lines(
    horizontal: list[np.ndarray],
    vertical: list[np.ndarray],
    frame_height: int,
    frame_width: int,
) -> dict[str, np.ndarray | None]:
    """Identify which near-half lines correspond to court features.

    Horizontal lines sorted by y (top→bottom in crop):
      - Near baseline: widest horizontal line at the bottom
      - Near service line: horizontal line above baseline with consistent slope

    Vertical lines sorted by x at a reference y:
      - Left/right singles sidelines, center service line, doubles sidelines
    """
    result: dict[str, np.ndarray | None] = {
        "near_service": None,
        "near_baseline": None,
        "left_singles": None,
        "right_singles": None,
        "center_service": None,
        "left_doubles": None,
        "right_doubles": None,
    }

    cx = frame_width / 2.0

    # --- Horizontal lines ---
    min_xspan = frame_width * 0.08
    scored = []
    for seg in horizontal:
        x_span = abs(seg[2] - seg[0])
        if x_span < min_xspan:
            continue
        y_at_cx = _line_y_at_x(seg, cx)
        if y_at_cx is not None:
            length = np.hypot(seg[2] - seg[0], seg[3] - seg[1])
            scored.append((y_at_cx, length, seg))
    scored.sort(key=lambda x: x[0])  # ascending y (top first)

    if len(scored) >= 1:
        # Near baseline: the bottommost wide line (highest y)
        result["near_baseline"] = scored[-1][2]
        bl_slope = _seg_signed_slope(scored[-1][2])

        # Near service line: must be above baseline, with consistent slope sign,
        # and must overlap with the center region of the frame (not a far-side fragment)
        service_candidates = []
        bl_y = scored[-1][0]
        center_margin = frame_width * 0.15
        for y_at_cx, length, seg in scored[:-1]:
            if y_at_cx >= bl_y - 10:
                continue  # too close to baseline
            seg_x_min = min(seg[0], seg[2])
            seg_x_max = max(seg[0], seg[2])
            overlaps_center = not (seg_x_max < cx - center_margin or seg_x_min > cx + center_margin)
            seg_slope = _seg_signed_slope(seg)
            # Slope consistency check (relaxed)
            slope_ok = False
            if abs(bl_slope) < 0.01:
                slope_ok = abs(seg_slope) < 0.3
            else:
                slope_ok = (bl_slope * seg_slope >= 0) or abs(seg_slope) < 0.05

            if overlaps_center and slope_ok:
                service_candidates.append((y_at_cx, length, seg))
            elif not overlaps_center and slope_ok and length > frame_width * 0.2:
                # Long line that doesn't overlap center — may be offset but valid
                service_candidates.append((y_at_cx, length, seg))

        if service_candidates:
            # Prefer the longest candidate if multiple pass the slope test
            service_candidates.sort(key=lambda x: -x[1])
            result["near_service"] = service_candidates[0][2]

    # --- Vertical lines ---
    if len(vertical) >= 1:
        ref_y = frame_height * 0.8
        scored_v = []
        for seg in vertical:
            x_at_ref = _line_x_at_y(seg, ref_y)
            if x_at_ref is not None:
                length = np.hypot(seg[2] - seg[0], seg[3] - seg[1])
                scored_v.append((x_at_ref, length, seg))
        scored_v.sort(key=lambda x: x[0])  # left to right

        # Separate into left group (x < center) and right group (x > center)
        left_lines = [(x, ln, s) for x, ln, s in scored_v if x < cx]
        right_lines = [(x, ln, s) for x, ln, s in scored_v if x >= cx]

        # Center service line: must be near the center
        center_tolerance = frame_width * 0.15
        center_candidates = [(x, ln, s) for x, ln, s in scored_v
                             if abs(x - cx) < center_tolerance]
        center_seg = None
        if center_candidates:
            best = min(center_candidates, key=lambda t: abs(t[0] - cx))
            center_seg = best[2]
            result["center_service"] = center_seg

        # Exclude center_service from sideline groups to prevent misidentification
        if center_seg is not None:
            left_lines = [(x, ln, s) for x, ln, s in left_lines
                          if not np.array_equal(s, center_seg)]
            right_lines = [(x, ln, s) for x, ln, s in right_lines
                           if not np.array_equal(s, center_seg)]

        # Assign sidelines from left and right groups
        if len(left_lines) >= 2:
            result["left_doubles"] = left_lines[0][2]
            result["left_singles"] = left_lines[-1][2]
        elif len(left_lines) == 1:
            result["left_singles"] = left_lines[0][2]

        if len(right_lines) >= 2:
            result["right_singles"] = right_lines[0][2]
            result["right_doubles"] = right_lines[-1][2]
        elif len(right_lines) == 1:
            result["right_singles"] = right_lines[0][2]

        # --- Metric ratio constraint: disambiguate singles vs doubles ---
        # When only 1 vertical line per side, use court proportions to decide
        # whether they are singles or doubles sidelines.
        bl = result.get("near_baseline")
        only_left_singles = (result["left_singles"] is not None
                             and result["left_doubles"] is None)
        only_right_singles = (result["right_singles"] is not None
                              and result["right_doubles"] is None)

        if bl is not None and only_left_singles and only_right_singles:
            ls = result["left_singles"]
            rs = result["right_singles"]
            # Intersect each vertical with baseline
            pt_l = line_intersection(bl, ls, w=FRAME_W * 2, h=FRAME_H * 2)
            pt_r = line_intersection(bl, rs, w=FRAME_W * 2, h=FRAME_H * 2)
            if pt_l is not None and pt_r is not None:
                line_sep = abs(pt_r[0] - pt_l[0])
                # Baseline pixel length (endpoint-to-endpoint)
                bl_len = np.hypot(bl[2] - bl[0], bl[3] - bl[1])
                if bl_len > 0:
                    ratio = line_sep / bl_len
                    # ratio ≈ 1.0 → doubles lines (they meet baseline at endpoints)
                    # ratio ≈ 0.75 (8.23/10.97) → singles lines
                    # threshold at 0.87 to separate
                    doubles_ratio = 1.0
                    singles_ratio = 8.23 / 10.97  # ~0.75
                    threshold = 0.87
                    print(f"    [Metric] Line sep ratio = {ratio:.3f} "
                          f"(doubles≈{doubles_ratio:.2f}, "
                          f"singles≈{singles_ratio:.2f}, "
                          f"threshold={threshold})")
                    if ratio > threshold:
                        # These are doubles lines, reclassify
                        print(f"    [Metric] Reclassifying as DOUBLES sidelines")
                        result["left_doubles"] = result["left_singles"]
                        result["right_doubles"] = result["right_singles"]
                        result["left_singles"] = None
                        result["right_singles"] = None
                    # else: they stay as singles (correct)

            # Additional check: if a vertical line's intersection with baseline
            # is within ~100px of the baseline endpoint, it's doubles
            if result["left_doubles"] is None and result["left_singles"] is not None:
                pt_l = line_intersection(bl, result["left_singles"],
                                         w=FRAME_W * 2, h=FRAME_H * 2)
                bl_left_x = min(bl[0], bl[2])
                if pt_l is not None and abs(pt_l[0] - bl_left_x) < 100:
                    print(f"    [Metric] Left line near baseline endpoint "
                          f"({abs(pt_l[0] - bl_left_x):.0f}px), "
                          f"reclassifying as doubles")
                    result["left_doubles"] = result["left_singles"]
                    result["left_singles"] = None

            if result["right_doubles"] is None and result["right_singles"] is not None:
                pt_r = line_intersection(bl, result["right_singles"],
                                         w=FRAME_W * 2, h=FRAME_H * 2)
                bl_right_x = max(bl[0], bl[2])
                if pt_r is not None and abs(pt_r[0] - bl_right_x) < 100:
                    print(f"    [Metric] Right line near baseline endpoint "
                          f"({abs(pt_r[0] - bl_right_x):.0f}px), "
                          f"reclassifying as doubles")
                    result["right_doubles"] = result["right_singles"]
                    result["right_singles"] = None

    return result


# ===================================================================
# Stage 6: Compute near-half keypoints from intersections
# ===================================================================

def _synthesize_service_line_from_vp(
    identified_lines: dict[str, np.ndarray | None],
    vertical: list[np.ndarray],
    y_offset: int,
    frame_height: int,
) -> list[tuple[int, tuple[float, float]]]:
    """Synthesize service line keypoints using VP + baseline + court proportions.

    When the service line can't be detected directly, use the vanishing point
    of the sidelines and the known court proportions to estimate where it is.
    """
    kps: list[tuple[int, tuple[float, float]]] = []
    bl = identified_lines.get("near_baseline")
    ls = identified_lines.get("left_singles")
    rs = identified_lines.get("right_singles")

    # Fall back to doubles sidelines if singles not detected
    use_doubles = False
    if ls is None and identified_lines.get("left_doubles") is not None:
        ls = identified_lines["left_doubles"]
        use_doubles = True
    if rs is None and identified_lines.get("right_doubles") is not None:
        rs = identified_lines["right_doubles"]
        use_doubles = True

    if bl is None or (ls is None and rs is None):
        return kps

    # Compute VP from the sidelines
    sidelines = [s for s in [ls, rs] if s is not None]
    vp = estimate_vanishing_point(sidelines + vertical) if len(vertical) >= 2 else None
    if vp is None and len(sidelines) >= 2:
        pt = line_intersection(sidelines[0], sidelines[1], w=FRAME_W * 4, h=FRAME_H * 4)
        if pt is not None:
            vp = np.array(pt)

    if vp is None:
        return kps

    # Baseline intersection points
    bl_left = None
    bl_right = None
    if ls is not None:
        pt = line_intersection(bl, ls, w=FRAME_W * 2, h=FRAME_H * 2)
        if pt:
            bl_left = np.array(pt)
    if rs is not None:
        pt = line_intersection(bl, rs, w=FRAME_W * 2, h=FRAME_H * 2)
        if pt:
            bl_right = np.array(pt)

    # Perspective ratio for service line
    # Service line is 5.485m from baseline, baseline is 11.885m from net
    # For a camera ~3-5m behind baseline, the ratio on the sideline is:
    # t = d_cam / (d_cam + 5.485) where d_cam ≈ 3-5m
    # Simple estimate: use t = 0.42 (corresponds to d_cam ≈ 4m)
    t_sv = 0.42  # reasonable for indoor courtside camera
    if bl_left is not None:
        sv_left = vp + t_sv * (bl_left - vp)
        kps.append((10, (float(sv_left[0]), float(sv_left[1]) + y_offset)))

    if bl_right is not None:
        sv_right = vp + t_sv * (bl_right - vp)
        kps.append((11, (float(sv_right[0]), float(sv_right[1]) + y_offset)))

    # KP13 (center T): midpoint of left and right service line points
    if len(kps) == 2:
        kp10_x, kp10_y = kps[0][1]
        kp11_x, kp11_y = kps[1][1]
        kps.append((13, ((kp10_x + kp11_x) / 2, (kp10_y + kp11_y) / 2)))

    return kps


def compute_near_half_keypoints(
    identified_lines: dict[str, np.ndarray | None],
    y_offset: int,
    vertical: list[np.ndarray] | None = None,
    frame_height: int = 0,
) -> list[tuple[int, tuple[float, float]]]:
    """Compute keypoints from line intersections.

    All pixel coordinates are converted to the original frame coordinate
    system by adding y_offset.
    """
    kps: list[tuple[int, tuple[float, float]]] = []

    def _intersect(name1: str, name2: str) -> tuple[float, float] | None:
        seg1 = identified_lines.get(name1)
        seg2 = identified_lines.get(name2)
        if seg1 is None or seg2 is None:
            return None
        pt = line_intersection(seg1, seg2, w=FRAME_W * 2, h=FRAME_H * 2)
        if pt is not None:
            return (pt[0], pt[1] + y_offset)
        return None

    # near_baseline × doubles sidelines → KP2, KP3
    pt = _intersect("near_baseline", "left_doubles")
    if pt is None:
        pt = _intersect("near_baseline", "left_singles")
        if pt:
            kps.append((5, pt))
    else:
        kps.append((2, pt))

    pt = _intersect("near_baseline", "right_doubles")
    if pt is None:
        pt = _intersect("near_baseline", "right_singles")
        if pt:
            kps.append((7, pt))
    else:
        kps.append((3, pt))

    # near_baseline × singles sidelines → KP5, KP7
    assigned_ids = {k for k, _ in kps}
    if 5 not in assigned_ids:
        pt = _intersect("near_baseline", "left_singles")
        if pt:
            kps.append((5, pt))
    if 7 not in assigned_ids:
        pt = _intersect("near_baseline", "right_singles")
        if pt:
            kps.append((7, pt))

    # near_service × singles sidelines → KP10, KP11
    kp10 = _intersect("near_service", "left_singles")
    kp11 = _intersect("near_service", "right_singles")

    # Validate: KP10 and KP11 should be well-separated (>100px apart in x)
    use_detected_service = False
    if kp10 is not None and kp11 is not None:
        if abs(kp10[0] - kp11[0]) > 100:
            kps.append((10, kp10))
            kps.append((11, kp11))
            use_detected_service = True
        else:
            print(f"    [Warning] Service line KPs too close: "
                  f"dx={abs(kp10[0]-kp11[0]):.0f}px, using VP synthesis")
    elif kp10 is not None:
        kps.append((10, kp10))
        use_detected_service = True
    elif kp11 is not None:
        kps.append((11, kp11))
        use_detected_service = True

    # If service line detection failed or gave bad results, synthesize from VP
    if not use_detected_service and vertical is not None:
        synth = _synthesize_service_line_from_vp(
            identified_lines, vertical, y_offset, frame_height)
        if synth:
            print(f"    [Synth] Service line from VP: "
                  f"{len(synth)} keypoints synthesized")
            kps.extend(synth)

    # near_service × center_service → KP13
    kp13 = _intersect("near_service", "center_service")
    assigned_ids = {k for k, _ in kps}
    if kp13 is not None and 13 not in assigned_ids:
        kps.append((13, kp13))

    # Fallback: if KP10 and KP11 are both detected but KP13 is not, compute
    # KP13 as the point on the near_service line at the x-midpoint of KP10/KP11
    assigned_ids = {k for k, _ in kps}
    if 13 not in assigned_ids and 10 in assigned_ids and 11 in assigned_ids:
        kp10_pt = next(p for k, p in kps if k == 10)
        kp11_pt = next(p for k, p in kps if k == 11)
        mid_x = (kp10_pt[0] + kp11_pt[0]) / 2.0
        mid_y = (kp10_pt[1] + kp11_pt[1]) / 2.0
        # Use intersection with service line if available, otherwise midpoint
        svc = identified_lines.get("near_service")
        if svc is not None:
            svc_y = _line_y_at_x(svc, mid_x)
            if svc_y is not None:
                mid_y = svc_y + y_offset
        kps.append((13, (mid_x, mid_y)))
        print(f"    [Fallback] KP13 computed from KP10/KP11 midpoint")

    return kps


# ===================================================================
# Stage 7: Near-half homography + extend to full court
# ===================================================================

def validate_near_half_keypoints(
    kps: list[tuple[int, tuple[float, float]]],
) -> list[tuple[int, tuple[float, float]]]:
    """Validate geometric consistency of near-half keypoints.

    Checks:
    - Baseline KPs should be below service line KPs (higher y in image)
    - Left KPs should have smaller x than right KPs
    - Removes keypoints that violate these constraints
    """
    kp_dict = {k: p for k, p in kps}

    def _check_x_order(left_id: int, right_id: int) -> bool:
        """Left keypoint should have smaller x than right keypoint."""
        if left_id in kp_dict and right_id in kp_dict:
            return kp_dict[left_id][0] < kp_dict[right_id][0]
        return True

    def _check_y_order(upper_id: int, lower_id: int) -> bool:
        """Upper keypoint (service line) should have smaller y than lower (baseline)."""
        if upper_id in kp_dict and lower_id in kp_dict:
            return kp_dict[upper_id][1] < kp_dict[lower_id][1]
        return True

    bad_ids: set[int] = set()

    # Left-right ordering checks
    if not _check_x_order(2, 3):
        print("    [Validate] KP2/KP3 left-right reversed, removing both")
        bad_ids.update([2, 3])
    if not _check_x_order(5, 7):
        print("    [Validate] KP5/KP7 left-right reversed, removing both")
        bad_ids.update([5, 7])
    if not _check_x_order(10, 11):
        print("    [Validate] KP10/KP11 left-right reversed, removing both")
        bad_ids.update([10, 11])

    # Service line should be above baseline (smaller y) in image
    if not _check_y_order(10, 5):
        print("    [Validate] KP10 not above KP5, removing KP10")
        bad_ids.add(10)
    if not _check_y_order(11, 7):
        print("    [Validate] KP11 not above KP7, removing KP11")
        bad_ids.add(11)

    if bad_ids:
        kps = [(k, p) for k, p in kps if k not in bad_ids]

    return kps


def compute_near_half_homography(
    near_kps: list[tuple[int, tuple[float, float]]],
) -> np.ndarray | None:
    """Compute homography from near-half keypoint correspondences.

    Maps image pixels → court meters.
    """
    if len(near_kps) < 4:
        return None

    src_pts = []
    dst_pts = []
    for kp_idx, (px, py) in near_kps:
        src_pts.append([px, py])
        dst_pts.append(list(REFERENCE_KPS_METERS[kp_idx]))

    src = np.array(src_pts, dtype=np.float64)
    dst = np.array(dst_pts, dtype=np.float64)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return H


# Court template lines in meter-space: (name, (x1, y1), (x2, y2))
COURT_TEMPLATE_LINES = {
    "near_baseline":   ((-5.485, 11.885), (5.485, 11.885)),
    "left_singles":    ((-4.115, 11.885), (-4.115, 6.4)),
    "right_singles":   ((4.115, 11.885), (4.115, 6.4)),
    "left_doubles":    ((-5.485, 11.885), (-5.485, 6.4)),
    "right_doubles":   ((5.485, 11.885), (5.485, 6.4)),
    "near_service":    ((-4.115, 6.4), (4.115, 6.4)),
    "center_service":  ((0.0, 11.885), (0.0, 6.4)),
}


def refine_homography_with_lines(
    H_init: np.ndarray,
    near_kps: list[tuple[int, tuple[float, float]]],
    identified_lines: dict[str, np.ndarray | None],
    y_offset: int,
    line_weight: float = 0.1,
    n_line_samples: int = 10,
) -> np.ndarray:
    """Refine homography using both point and line correspondences.

    Uses scipy.optimize.least_squares to minimize combined point reprojection
    error and line-to-line distance between detected Hough lines and projected
    court template lines.
    """
    if H_init is None or len(near_kps) < 4:
        return H_init

    # Collect point correspondences
    src_pts = []
    dst_pts = []
    for kp_idx, (px, py) in near_kps:
        src_pts.append([px, py])
        dst_pts.append(list(REFERENCE_KPS_METERS[kp_idx]))
    src_pts = np.array(src_pts, dtype=np.float64)
    dst_pts = np.array(dst_pts, dtype=np.float64)

    # Collect line correspondences: detected segment ↔ template line
    line_pairs = []  # (detected_samples_px, template_line_meters)
    for name, seg in identified_lines.items():
        if seg is None or name not in COURT_TEMPLATE_LINES:
            continue
        template = COURT_TEMPLATE_LINES[name]
        # Sample points along the detected segment (in original frame coords)
        x1, y1, x2, y2 = seg.astype(float)
        y1 += y_offset
        y2 += y_offset
        ts = np.linspace(0.0, 1.0, n_line_samples)
        samples = np.column_stack([
            x1 + ts * (x2 - x1),
            y1 + ts * (y2 - y1),
        ])
        line_pairs.append((samples, template))

    if not line_pairs:
        return H_init

    # Normalize H so H[2,2] = 1
    h0 = H_init.flatten()[:8] / H_init[2, 2]

    def _h_to_mat(h8):
        H = np.zeros((3, 3))
        H.flat[:8] = h8
        H[2, 2] = 1.0
        return H

    def _project_meter_to_pixel(H, meter_pts):
        """Project meter-space points to pixel space using H^-1."""
        H_inv = np.linalg.inv(H)
        pts = meter_pts.reshape(-1, 1, 2).astype(np.float64)
        return cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)

    def _point_on_line_distance(px, py, lx1, ly1, lx2, ly2):
        """Perpendicular distance from point to line defined by two points."""
        dx = lx2 - lx1
        dy = ly2 - ly1
        length = max(np.hypot(dx, dy), 1e-9)
        return abs(dy * px - dx * py + lx2 * ly1 - ly2 * lx1) / length

    def residuals(h8):
        H = _h_to_mat(h8)
        res = []

        # Point residuals: for each KP, project meter→pixel and compare
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return np.full(len(near_kps) * 2 +
                           sum(len(s) for s, _ in line_pairs), 1e6)

        for i in range(len(src_pts)):
            m_pt = dst_pts[i].reshape(1, 1, 2)
            proj = cv2.perspectiveTransform(m_pt, H_inv).reshape(2)
            res.append(proj[0] - src_pts[i, 0])
            res.append(proj[1] - src_pts[i, 1])

        # Line residuals: for each detected line sample point, compute
        # perpendicular distance to the projected template line
        for samples, (m1, m2) in line_pairs:
            template_pts = np.array([m1, m2], dtype=np.float64)
            try:
                proj_line = _project_meter_to_pixel(H, template_pts)
            except np.linalg.LinAlgError:
                res.extend([1e6] * len(samples))
                continue
            lx1, ly1 = proj_line[0]
            lx2, ly2 = proj_line[1]
            for sx, sy in samples:
                d = _point_on_line_distance(sx, sy, lx1, ly1, lx2, ly2)
                res.append(d * line_weight)

        return np.array(res)

    try:
        result = least_squares(residuals, h0, method='lm', max_nfev=200)
        H_refined = _h_to_mat(result.x)

        # Sanity check: refined H should not be dramatically different
        err_init = reprojection_error(near_kps, H_init)
        err_refined = reprojection_error(near_kps, H_refined)
        if err_refined < err_init:
            print(f"    [PnL] Refined H: reproj {err_init:.2f} → {err_refined:.2f}px "
                  f"(cost {result.cost:.2f})")
            return H_refined
        else:
            print(f"    [PnL] Refinement did not improve ({err_init:.2f} → {err_refined:.2f}px), "
                  f"keeping original")
            return H_init
    except Exception as e:
        print(f"    [PnL] Optimization failed: {e}")
        return H_init


def extend_to_full_court(
    H_near: np.ndarray,
    near_kps: list[tuple[int, tuple[float, float]]],
) -> tuple[np.ndarray | None, list[tuple[int, tuple[float, float]]]]:
    """Extend near-half homography to full court.

    1. Invert H_near to project far-half meter coords → image pixels
    2. Combine with near-half keypoints
    3. Re-compute full-court homography
    """
    try:
        H_inv = np.linalg.inv(H_near)
    except np.linalg.LinAlgError:
        return None, list(near_kps)

    all_kps = list(near_kps)
    assigned = {k for k, _ in all_kps}

    # Project far-half keypoints
    for kp_idx, (mx, my) in FAR_HALF_KPS.items():
        if kp_idx in assigned:
            continue
        meter_pt = np.array([[[mx, my]]], dtype=np.float64)
        projected = cv2.perspectiveTransform(meter_pt, H_inv)
        px, py = projected[0][0]
        # Sanity: projected point should be within a reasonable range
        if -500 < px < FRAME_W + 500 and -500 < py < FRAME_H + 500:
            all_kps.append((kp_idx, (float(px), float(py))))

    # Re-compute full homography from all keypoints
    if len(all_kps) < 4:
        return None, all_kps

    src_pts = []
    dst_pts = []
    for kp_idx, (px, py) in all_kps:
        src_pts.append([px, py])
        dst_pts.append(list(REFERENCE_KPS_METERS[kp_idx]))

    src = np.array(src_pts, dtype=np.float64)
    dst = np.array(dst_pts, dtype=np.float64)
    H_full, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    # Sanity check: projected court corners should form a convex quadrilateral
    # with correct orientation (far baseline above near baseline in image)
    if H_full is not None:
        try:
            H_inv_full = np.linalg.inv(H_full)
            court_corners_m = [
                REFERENCE_KPS_METERS[0], REFERENCE_KPS_METERS[1],
                REFERENCE_KPS_METERS[3], REFERENCE_KPS_METERS[2],
            ]
            corners = np.array(court_corners_m, dtype=np.float64).reshape(-1, 1, 2)
            projected = cv2.perspectiveTransform(corners, H_inv_full)
            pts_2d = [(float(p[0][0]), float(p[0][1])) for p in projected]
            if not is_convex_quadrilateral(pts_2d):
                print("    [Warning] Projected court is not a convex quadrilateral")
                return None, all_kps

            # Orientation: far baseline (KP0,KP1) should be above near baseline
            # (KP2,KP3) in image (smaller y)
            far_bl_y = (pts_2d[0][1] + pts_2d[1][1]) / 2
            near_bl_y = (pts_2d[2][1] + pts_2d[3][1]) / 2
            if far_bl_y >= near_bl_y:
                print("    [Warning] Court orientation inverted "
                      "(far baseline below near baseline)")
                return None, all_kps
        except np.linalg.LinAlgError:
            return None, all_kps

    return H_full, all_kps


def reprojection_error(
    kps: list[tuple[int, tuple[float, float]]],
    H: np.ndarray,
) -> float:
    """Mean reprojection error (pixels) of keypoints through homography."""
    if not kps or H is None:
        return float("inf")

    errors = []
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return float("inf")

    for kp_idx, (px, py) in kps:
        mx, my = REFERENCE_KPS_METERS[kp_idx]
        meter_pt = np.array([[[mx, my]]], dtype=np.float64)
        projected = cv2.perspectiveTransform(meter_pt, H_inv)
        ppx, ppy = projected[0][0]
        errors.append(np.hypot(ppx - px, ppy - py))

    return float(np.mean(errors)) if errors else float("inf")


def is_convex_quadrilateral(points: list[tuple[float, float]]) -> bool:
    """Check if 4 points form a convex quadrilateral."""
    if len(points) < 4:
        return False
    pts = np.array(points[:4], dtype=np.float32)
    hull = cv2.convexHull(pts)
    return len(hull) == 4


# ===================================================================
# Stage 8: Diagnostics and visualization
# ===================================================================

def draw_pipeline_stages(
    original_frame: np.ndarray,
    net_y: int,
    near_half: np.ndarray,
    court_color_hsv: np.ndarray,
    court_color_mask: np.ndarray,
    line_mask: np.ndarray,
    horizontal: list[np.ndarray],
    vertical: list[np.ndarray],
    identified_lines: dict[str, np.ndarray | None],
    near_kps: list[tuple[int, tuple[float, float]]],
    all_kps: list[tuple[int, tuple[float, float]]],
    H_full: np.ndarray | None,
    y_offset: int,
    cnn_kps: list[tuple[float | None, float | None]] | None = None,
) -> np.ndarray:
    """Create 2×3 diagnostic composite."""
    cell_w, cell_h = 640, 360

    def resize(img):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.resize(img, (cell_w, cell_h))

    # Panel 1: Original + net line
    p1 = original_frame.copy()
    cv2.line(p1, (0, net_y), (FRAME_W, net_y), (0, 255, 255), 2)
    cv2.putText(p1, f"Net y={net_y}", (10, net_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    p1 = resize(p1)
    cv2.putText(p1, "1. Original + net", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Panel 2: Near-half crop
    p2 = resize(near_half)
    cv2.putText(p2, "2. Near-half crop", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Panel 3: Court color mask
    p3 = resize(court_color_mask)
    ch, cs, cv_val = court_color_hsv
    cv2.putText(p3, f"3. Court mask H={ch} S={cs} V={cv_val}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Panel 4: Line filter mask
    p4 = resize(line_mask)
    cv2.putText(p4, "4. Agrawal line mask", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Panel 5: Detected + classified lines on near-half
    p5 = near_half.copy()
    for seg in horizontal:
        cv2.line(p5, (seg[0], seg[1]), (seg[2], seg[3]), (255, 100, 0), 2)
    for seg in vertical:
        cv2.line(p5, (seg[0], seg[1]), (seg[2], seg[3]), (0, 0, 255), 2)
    # Draw identified lines with labels
    colors = {
        "near_baseline": (0, 255, 0),
        "near_service": (0, 200, 200),
        "left_singles": (255, 0, 255),
        "right_singles": (255, 0, 255),
        "center_service": (255, 255, 0),
        "left_doubles": (200, 200, 0),
        "right_doubles": (200, 200, 0),
    }
    for name, seg in identified_lines.items():
        if seg is not None:
            c = colors.get(name, (255, 255, 255))
            cv2.line(p5, (seg[0], seg[1]), (seg[2], seg[3]), c, 3)
            mx = (seg[0] + seg[2]) // 2
            my = (seg[1] + seg[3]) // 2
            cv2.putText(p5, name, (mx, my - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)
    p5 = resize(p5)
    cv2.putText(p5, "5. Lines classified", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Panel 6: Final keypoints on original frame
    p6 = original_frame.copy()
    # Draw Agrawal keypoints
    for kp_idx, (px, py) in all_kps:
        ix, iy = int(round(px)), int(round(py))
        near_kp_ids = {k for k, _ in near_kps}
        if kp_idx in near_kp_ids:
            color = (0, 255, 0)     # Green for detected near-half
        else:
            color = (0, 165, 255)   # Orange for projected far-half
        cv2.circle(p6, (ix, iy), 6, color, -1)
        cv2.putText(p6, f"{kp_idx}", (ix + 8, iy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw CNN keypoints for comparison
    if cnn_kps is not None:
        for i, (cx, cy) in enumerate(cnn_kps):
            if cx is not None and cy is not None:
                cv2.circle(p6, (int(cx), int(cy)), 4, (255, 0, 0), -1)

    # Draw reprojected court outline if we have H_full
    if H_full is not None:
        try:
            H_inv = np.linalg.inv(H_full)
            court_corners_m = [
                REFERENCE_KPS_METERS[0], REFERENCE_KPS_METERS[1],
                REFERENCE_KPS_METERS[3], REFERENCE_KPS_METERS[2],
            ]
            corners = np.array(court_corners_m, dtype=np.float64).reshape(-1, 1, 2)
            projected = cv2.perspectiveTransform(corners, H_inv)
            pts = projected.reshape(-1, 2).astype(np.int32)
            cv2.polylines(p6, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        except np.linalg.LinAlgError:
            pass

    p6 = resize(p6)
    cv2.putText(p6, "6. Keypoints (green=near, orange=far, blue=CNN)", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Assemble 2×3 grid
    row1 = np.hstack([p1, p2, p3])
    row2 = np.hstack([p4, p5, p6])
    composite = np.vstack([row1, row2])
    return composite


def compute_metrics(
    near_kps: list[tuple[int, tuple[float, float]]],
    all_kps: list[tuple[int, tuple[float, float]]],
    H_full: np.ndarray | None,
    cnn_kps: list[tuple[float | None, float | None]] | None,
) -> dict:
    """Compute quality metrics for the detection."""
    metrics: dict = {
        "n_near_kps": len(near_kps),
        "n_total_kps": len(all_kps),
        "near_reprojection_error_px": float("inf"),
        "full_reprojection_error_px": float("inf"),
        "far_reprojection_error_px": float("inf"),
        "homography_condition": float("inf"),
        "keypoint_spread": 0.0,
        "cnn_agreement_px": float("inf"),
        "cnn_near_agreement_px": float("inf"),
        "cnn_far_agreement_px": float("inf"),
        "homography_valid": False,
    }

    near_ids = {k for k, _ in near_kps}

    if H_full is not None:
        metrics["homography_valid"] = True

        # Reprojection errors — separate near (detected) vs far (projected)
        metrics["near_reprojection_error_px"] = reprojection_error(near_kps, H_full)
        metrics["full_reprojection_error_px"] = reprojection_error(all_kps, H_full)
        far_kps = [(k, p) for k, p in all_kps if k not in near_ids]
        if far_kps:
            metrics["far_reprojection_error_px"] = reprojection_error(far_kps, H_full)

        # Condition number
        metrics["homography_condition"] = float(np.linalg.cond(H_full))

    # Keypoint spread
    if len(all_kps) >= 4:
        pts = np.array([p for _, p in all_kps])
        x_range = pts[:, 0].max() - pts[:, 0].min()
        y_range = pts[:, 1].max() - pts[:, 1].min()
        metrics["keypoint_spread"] = float(x_range * y_range)

    # CNN agreement — separated by near (meaningful) and far (projected)
    if cnn_kps is not None and len(all_kps) > 0:
        all_dists = []
        near_dists = []
        far_dists = []
        for kp_idx, (px, py) in all_kps:
            if kp_idx < len(cnn_kps):
                cx, cy = cnn_kps[kp_idx]
                if cx is not None and cy is not None:
                    d = np.hypot(px - cx, py - cy)
                    all_dists.append(d)
                    if kp_idx in near_ids:
                        near_dists.append(d)
                    else:
                        far_dists.append(d)
        if all_dists:
            metrics["cnn_agreement_px"] = float(np.mean(all_dists))
        if near_dists:
            metrics["cnn_near_agreement_px"] = float(np.mean(near_dists))
        if far_dists:
            metrics["cnn_far_agreement_px"] = float(np.mean(far_dists))

    return metrics


# ===================================================================
# Kalman smoother for temporal consistency
# ===================================================================

class HomographyKalmanFilter:
    """Simple Kalman filter on homography parameters for temporal smoothing.

    Adapts measurement noise based on reprojection error and keypoint count
    to handle transitions between frames with different keypoint sets.
    """

    def __init__(self):
        self.state = None   # 8-element vector
        self.P = None        # 8x8 covariance
        self.Q = np.eye(8) * 0.001  # process noise (camera is static but allow drift)
        self.initialized = False
        self.prev_n_kps = 0
        self.prev_frame_idx = -1

    def update(self, H_measured: np.ndarray, reproj_error: float,
               n_kps: int = 7, frame_idx: int = -1) -> np.ndarray:
        """Incorporate a new measurement. Returns smoothed H.

        If frames are non-consecutive (gap > 5), resets state to the new
        measurement instead of blending with stale history.

        Args:
            H_measured: raw homography from this frame
            reproj_error: near-half reprojection error in pixels
            n_kps: number of near-half keypoints used (more = more reliable)
            frame_idx: video frame index for gap detection
        """
        h = H_measured.flatten()[:8] / H_measured[2, 2]  # normalize

        # Measurement noise: lower when reproj error is low AND many KPs
        kp_factor = max(1.0, 8.0 - n_kps)  # 7 KPs→1, 5 KPs→3
        noise_scale = (reproj_error + 0.5) ** 2 * kp_factor
        R = np.eye(8) * noise_scale

        # Detect large frame gaps — reset instead of blending with stale state
        frame_gap = (frame_idx - self.prev_frame_idx
                     if self.prev_frame_idx >= 0 and frame_idx >= 0 else 0)
        if frame_gap > 5:
            print(f"    [Kalman] Frame gap {frame_gap} detected, resetting state")
            self.initialized = False

        self.prev_frame_idx = frame_idx

        if not self.initialized:
            self.state = h.copy()
            self.P = R.copy()
            self.initialized = True
            self.prev_n_kps = n_kps
            return self._to_H()

        # If keypoint count changed significantly, increase process noise
        if abs(n_kps - self.prev_n_kps) >= 2:
            Q_eff = self.Q * 100.0
        else:
            Q_eff = self.Q
        self.prev_n_kps = n_kps

        # Predict (constant state model)
        P_pred = self.P + Q_eff

        # Update
        K = P_pred @ np.linalg.inv(P_pred + R)
        self.state = self.state + K @ (h - self.state)
        self.P = (np.eye(8) - K) @ P_pred

        return self._to_H()

    def predict(self) -> np.ndarray | None:
        """Use when detection fails — coast on prediction."""
        if not self.initialized:
            return None
        self.P = self.P + self.Q
        return self._to_H()

    def _to_H(self) -> np.ndarray:
        H = np.zeros((3, 3))
        H.flat[:8] = self.state
        H[2, 2] = 1.0
        return H


# ===================================================================
# Main pipeline
# ===================================================================

def run_agrawal_pipeline(
    frame: np.ndarray,
    frame_idx: int,
    court_detector: CourtDetector,
    kalman: HomographyKalmanFilter | None = None,
) -> dict:
    """Run the full Agrawal-inspired pipeline on a single frame."""
    t0 = time.time()
    results: dict = {"frame_idx": frame_idx}

    # --- Stage 1: Net detection & crop ---
    t1 = time.time()
    net_y, cnn_result = estimate_net_y_from_cnn(frame, court_detector, frame_idx)
    cnn_kps = cnn_result.keypoints
    cnn_detected = sum(1 for p in cnn_kps if p[0] is not None)
    print(f"  [Stage 1] Net y={net_y}, CNN detected {cnn_detected}/14 keypoints "
          f"({time.time()-t1:.2f}s)")

    near_half, y_offset = crop_near_half(frame, net_y)
    print(f"  [Stage 1] Near-half crop: {near_half.shape[1]}×{near_half.shape[0]}, "
          f"y_offset={y_offset}")

    # --- Stage 2: Court color detection ---
    t2 = time.time()
    court_color, court_type = detect_court_color_kmeans(near_half, n_samples=N_COLOR_SAMPLES)
    print(f"  [Stage 2] Court color: H={court_color[0]} S={court_color[1]} "
          f"V={court_color[2]}, type={court_type} ({time.time()-t2:.2f}s)")

    # --- Stage 3: Agrawal court-line filter ---
    t3 = time.time()
    court_color_mask = build_court_color_mask(near_half, court_color)
    line_mask = agrawal_court_line_filter(near_half, court_color)
    print(f"  [Stage 3] Agrawal filter: {np.count_nonzero(line_mask)} line pixels "
          f"({time.time()-t3:.3f}s)")

    # --- Stage 4: Hough line detection ---
    t4 = time.time()
    raw_lines = detect_lines_near_half(line_mask)
    n_raw = 0 if raw_lines is None else len(raw_lines)
    horizontal, vertical = classify_lines_courtside(raw_lines)
    horizontal = merge_collinear_segments(horizontal)
    vertical = merge_collinear_segments(vertical)
    print(f"  [Stage 4] Hough: {n_raw} raw → {len(horizontal)}H + {len(vertical)}V "
          f"merged ({time.time()-t4:.2f}s)")

    # --- Stage 5: Identify near-half lines ---
    t5 = time.time()
    nh_h, nh_w = near_half.shape[:2]
    identified = identify_near_half_lines(horizontal, vertical, nh_h, nh_w)
    id_count = sum(1 for v in identified.values() if v is not None)
    print(f"  [Stage 5] Identified {id_count}/7 court features:")
    for name, seg in identified.items():
        if seg is not None:
            print(f"    {name}: ({seg[0]},{seg[1]})->({seg[2]},{seg[3]})")

    # --- Stage 6: Compute near-half keypoints ---
    t6 = time.time()
    near_kps = compute_near_half_keypoints(identified, y_offset, vertical, nh_h)
    near_kps = validate_near_half_keypoints(near_kps)
    print(f"  [Stage 6] Near-half keypoints: {len(near_kps)}")
    for kp_idx, (px, py) in near_kps:
        name = KP_NAMES[kp_idx] if kp_idx < len(KP_NAMES) else f"KP{kp_idx}"
        print(f"    KP{kp_idx} ({name}): ({px:.1f}, {py:.1f})")

    # --- Stage 7: Homography + PnL refinement + extend ---
    t7 = time.time()
    H_near = compute_near_half_homography(near_kps)
    H_full = None
    H_full_raw = None
    all_kps = list(near_kps)

    if H_near is not None:
        # PnL refinement: refine H_near using detected line correspondences
        H_near = refine_homography_with_lines(
            H_near, near_kps, identified, y_offset)

        H_full, all_kps = extend_to_full_court(H_near, near_kps)
        H_full_raw = H_full  # keep raw for comparison
        near_err = reprojection_error(near_kps, H_full) if H_full is not None else float("inf")
        full_err = reprojection_error(all_kps, H_full) if H_full is not None else float("inf")
        print(f"  [Stage 7] Homography computed: near_err={near_err:.1f}px, "
              f"full_err={full_err:.1f}px ({time.time()-t7:.2f}s)")
        print(f"    Total keypoints (near+far): {len(all_kps)}")

        # Kalman smoothing
        if kalman is not None and H_full is not None:
            H_full = kalman.update(H_full, near_err, n_kps=len(near_kps),
                                   frame_idx=frame_idx)
            smoothed_err = reprojection_error(near_kps, H_full)
            print(f"  [Kalman] Smoothed H: reproj {near_err:.2f} → {smoothed_err:.2f}px")
    else:
        print(f"  [Stage 7] Homography FAILED — only {len(near_kps)} keypoints "
              f"(need ≥4)")

        # Fallback: supplement with CNN keypoints
        if cnn_detected >= 4:
            print(f"  [Fallback] Using CNN keypoints to supplement")
            assigned = {k for k, _ in near_kps}
            for i, (cx, cy) in enumerate(cnn_kps):
                if cx is not None and cy is not None and i not in assigned:
                    near_kps.append((i, (cx, cy)))
                    assigned.add(i)
            H_near = compute_near_half_homography(near_kps)
            if H_near is not None:
                H_full, all_kps = extend_to_full_court(H_near, near_kps)
                H_full_raw = H_full
                if H_full is not None:
                    print(f"  [Fallback] Homography recovered with {len(all_kps)} keypoints")
                    if kalman is not None:
                        near_err = reprojection_error(near_kps, H_full)
                        H_full = kalman.update(H_full, near_err,
                                               n_kps=len(near_kps),
                                               frame_idx=frame_idx)

        # If still no homography, try Kalman prediction
        if H_full is None and kalman is not None:
            H_pred = kalman.predict()
            if H_pred is not None:
                H_full = H_pred
                print(f"  [Kalman] Using predicted H (coasting)")

    # Compute raw reprojection error before Kalman for reporting
    raw_near_err = reprojection_error(near_kps, H_full_raw) if H_full_raw is not None else float("inf")
    smoothed_near_err = reprojection_error(near_kps, H_full) if H_full is not None else float("inf")

    # --- Stage 8: Metrics ---
    metrics = compute_metrics(near_kps, all_kps, H_full, cnn_kps)
    print(f"  [Stage 8] Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.2f}")
        else:
            print(f"    {k}: {v}")

    total_time = time.time() - t0
    print(f"  [Total] Pipeline took {total_time:.2f}s")

    # --- Save diagnostic image ---
    composite = draw_pipeline_stages(
        original_frame=frame,
        net_y=net_y,
        near_half=near_half,
        court_color_hsv=court_color,
        court_color_mask=court_color_mask,
        line_mask=line_mask,
        horizontal=horizontal,
        vertical=vertical,
        identified_lines=identified,
        near_kps=near_kps,
        all_kps=all_kps,
        H_full=H_full,
        y_offset=y_offset,
        cnn_kps=cnn_kps,
    )

    results.update({
        "net_y": net_y,
        "court_color": court_color.tolist(),
        "court_type": court_type,
        "n_raw_lines": n_raw,
        "n_horizontal": len(horizontal),
        "n_vertical": len(vertical),
        "n_identified": id_count,
        "near_kps": [(k, list(p)) for k, p in near_kps],
        "all_kps": [(k, list(p)) for k, p in all_kps],
        "metrics": metrics,
        "composite": composite,
        "cnn_detected": cnn_detected,
        "total_time": total_time,
        "raw_near_err": raw_near_err,
        "smoothed_near_err": smoothed_near_err,
    })

    return results


def main():
    """Entry point: process frames and generate report."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Open video
    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {VIDEO_PATH}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames, {fps:.1f} fps")

    # Extract target frames
    frames: dict[int, np.ndarray] = {}
    for idx in FRAME_INDICES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames[idx] = frame
            print(f"  Extracted frame {idx}: {frame.shape[1]}×{frame.shape[0]}")
        else:
            print(f"  WARNING: Could not read frame {idx}")
    cap.release()

    if not frames:
        print("ERROR: No frames extracted")
        sys.exit(1)

    # Initialize CNN detector
    print("\nInitializing CNN court detector...")
    config = get_config()
    court_detector = CourtDetector(config, use_refine_kps=True, use_homography=True)

    # Initialize Kalman smoother for temporal consistency
    kalman = HomographyKalmanFilter()

    # Process each frame
    all_results = []
    for idx in FRAME_INDICES:
        if idx not in frames:
            continue
        print(f"\n{'='*70}")
        print(f"Processing frame {idx}")
        print(f"{'='*70}")

        result = run_agrawal_pipeline(frames[idx], idx, court_detector, kalman)

        # Save diagnostic composite
        out_path = OUTPUT_DIR / f"agrawal_frame_{idx}.jpg"
        cv2.imwrite(str(out_path), result["composite"])
        print(f"  Saved: {out_path}")

        all_results.append(result)

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Frame':>8} {'CNN KPs':>8} {'Near KPs':>9} {'Total KPs':>10} "
          f"{'H valid':>8} {'Raw err':>9} {'Smth err':>9} {'Far err*':>9} "
          f"{'CNN agr':>8} {'Spread':>10} {'Time':>6}")
    print("-" * 115)
    for r in all_results:
        m = r["metrics"]
        print(f"{r['frame_idx']:>8} "
              f"{r['cnn_detected']:>8} "
              f"{m['n_near_kps']:>9} "
              f"{m['n_total_kps']:>10} "
              f"{'YES' if m['homography_valid'] else 'NO':>8} "
              f"{r['raw_near_err']:>9.1f} "
              f"{r['smoothed_near_err']:>9.1f} "
              f"{m['far_reprojection_error_px']:>9.1f} "
              f"{m['cnn_agreement_px']:>8.1f} "
              f"{m['keypoint_spread']:>10.0f} "
              f"{r['total_time']:>6.2f}")
    print("  * Far err is self-referential (projected from near-half H)")
    print("  * Raw err = before Kalman, Smth err = after Kalman smoothing")

    # Save JSON report (without numpy arrays)
    report = []
    for r in all_results:
        entry = {k: v for k, v in r.items() if k != "composite"}
        # Convert metrics inf to string for JSON
        for mk in entry.get("metrics", {}):
            val = entry["metrics"][mk]
            if isinstance(val, float) and (np.isinf(val) or np.isnan(val)):
                entry["metrics"][mk] = str(val)
        report.append(entry)

    report_path = OUTPUT_DIR / "agrawal_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
