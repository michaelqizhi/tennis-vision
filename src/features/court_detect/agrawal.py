#!/usr/bin/env python3
"""Agrawal et al. (2024) inspired court detection pipeline.

Importable module containing: court-color filtering, near-half extraction,
Hough lines, near-half homography, full-court extension, and scoring.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

from src.features.court_detect.court_template import REFERENCE_KPS_METERS
from src.features.court_detect.shadow_removal import ShadowRemover

FRAME_W, FRAME_H = 1920, 1080

# Court geometry (ITF, meters from net center origin)
NET_Y_METERS = 0.0
NEAR_BASELINE_Y = 11.885
SERVICE_LINE_Y = 6.4
MIN_BASELINE_FRAC = 0.70  # baseline must span ≥70% of frame width
MIN_BASELINE_EXTENT_FRAC = 0.65  # reject baseline extents too narrow for useful spatial masks
CLASSIFY_ANGLE_TOL = 10   # degrees from baseline angle for H/V classification

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
NEIGHBORHOOD_SIZE = 7
MIN_COURT_NEIGHBORS = 4

# Saturation-based Agrawal (color-agnostic)
# Saturation-based Agrawal (color-agnostic)
SAT_COURT_THRESHOLD = 50    # S > this = chromatic surface for neighbor counting
SAT_LINE_S_MAX = 50         # S < this = line candidate (includes anti-aliased edges)
SAT_LINE_V_MIN = 150        # V > this = bright (line candidate)
TOPHAT_KERNEL_SIZE = 23      # Morphological kernel for white top-hat (> line width ~5-15px)
TOPHAT_THRESHOLD = 30        # Minimum top-hat response to be a line candidate
HOUGH_THRESHOLD = 30
HOUGH_MIN_LENGTH = 30
HOUGH_MAX_GAP = 40
MIN_MERGED_LENGTH = 200
MERGE_DEBUG = False
MERGE_DEBUG_X_RANGE = (400.0, 800.0)
# ===================================================================
# Stage 1: Net detection / near-half crop
# ===================================================================

def estimate_net_y_from_yolo(
    frame: np.ndarray,
    net_detector,
    conf_threshold: float = 0.25,
) -> tuple[int, int] | None:
    """Estimate net position using finetuned YOLOv8 net detector.

    Returns (net_top_y, net_bottom_y) from the highest-confidence detection,
    or None if no detection meets the confidence threshold.
    """
    results = net_detector(frame, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    # Filter by confidence
    confs = boxes.conf.cpu().numpy()
    mask = confs >= conf_threshold
    if not mask.any():
        return None

    # Take highest confidence detection
    best_idx = confs[mask].argmax()
    filtered_boxes = boxes.xyxy.cpu().numpy()[mask]
    y1 = int(filtered_boxes[best_idx][1])  # top of bbox
    y2 = int(filtered_boxes[best_idx][3])  # bottom of bbox
    return y1, y2


def estimate_net_y(
    frame: np.ndarray,
    net_detector=None,
) -> tuple[int, str, int]:
    """Estimate net y-position using YOLO net detector (primary) or heuristic fallback.

    Returns (net_y, source, crop_margin) where:
    - net_y: the net bottom position (crop reference)
    - source: description of how net_y was determined
    - crop_margin: how many pixels above net_y to include in crop
    """
    # Primary: YOLO net detector
    if net_detector is not None:
        yolo_result = estimate_net_y_from_yolo(frame, net_detector)
        if yolo_result is not None:
            net_top, net_bottom = yolo_result
            net_height = net_bottom - net_top
            crop_margin = int(net_height * 0.1)
            net_y = max(0, min(net_bottom, frame.shape[0] - 100))
            return net_y, f"yolo(top={net_top},bot={net_bottom})", crop_margin

    # Fallback: horizontal Sobel heuristic
    net_y = estimate_net_y_heuristic(frame)
    net_y = max(0, min(net_y, frame.shape[0] - 100))
    return net_y, "heuristic", 20


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
# Stage 2: Agrawal court-line filter
# ===================================================================

def agrawal_saturation_line_filter(
    frame_bgr: np.ndarray,
    neighborhood: int = NEIGHBORHOOD_SIZE,
    min_court_neighbors: int = MIN_COURT_NEIGHBORS,
    sat_court_thresh: int = SAT_COURT_THRESHOLD,
    sat_line_s_max: int = SAT_LINE_S_MAX,
    sat_line_v_min: int = SAT_LINE_V_MIN,
    tophat_ksize: int = TOPHAT_KERNEL_SIZE,
    tophat_thresh: int = TOPHAT_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Color-agnostic Agrawal filter using white top-hat + saturation gate.

    White top-hat on V channel extracts only thin bright features (lines),
    naturally ignoring large bright areas (court surface, overexposure).
    Saturation gate ensures detected features are on a chromatic surface
    (court/surround), not sky or background.

    Works on any court color combination and any court brightness.

    Returns (line_mask, court_mask, hsv) where masks are uint8 binary (0/255)
    and hsv is the precomputed HSV image.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # White top-hat: extracts thin bright features smaller than kernel
    # Court surface (large, uniform) is suppressed; lines (thin, bright) survive
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (tophat_ksize, tophat_ksize))
    tophat = cv2.morphologyEx(V, cv2.MORPH_TOPHAT, kernel_morph)

    # Adaptive top-hat threshold using Otsu on court-region top-hat values.
    # This handles both bright courts (low top-hat response ~6) and dark
    # courts (high top-hat response ~100) without a fixed threshold.
    court_region = (S > sat_court_thresh)
    tophat_court = tophat[court_region]
    if len(tophat_court) > 100:
        otsu_thresh, _ = cv2.threshold(tophat_court, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Use the higher of Otsu and a minimum floor to avoid noise
        effective_tophat_thresh = max(int(otsu_thresh * 0.7), 5)
    else:
        effective_tophat_thresh = tophat_thresh

    # Line candidate: strong top-hat response AND achromatic AND bright
    line_candidate = ((tophat > effective_tophat_thresh) &
                      (S < sat_line_s_max) &
                      (V > sat_line_v_min))

    # Court surface = any chromatic pixel (for neighbor counting)
    court_mask = (S > sat_court_thresh).astype(np.uint8) * 255

    # Agrawal neighbor logic: line pixel must be surrounded by court pixels
    court_01 = (court_mask > 0).astype(np.float32)
    kernel = np.ones((neighborhood, neighborhood), dtype=np.float32)
    neighbor_count = cv2.filter2D(court_01, cv2.CV_32F, kernel)
    has_neighbors = (neighbor_count >= min_court_neighbors)

    line_mask = (line_candidate & has_neighbors).astype(np.uint8) * 255
    return line_mask, court_mask, hsv, effective_tophat_thresh


def agrawal_local_contrast_line_filter(
    frame_bgr: np.ndarray,
    neighborhood: int = NEIGHBORHOOD_SIZE,
    min_court_neighbors: int = MIN_COURT_NEIGHBORS,
    tophat_ksize: int = TOPHAT_KERNEL_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Local-contrast Agrawal filter using oriented V top-hat + S black-hat.

    Uses oriented line kernels at multiple angles (0°-150°) instead of an
    isotropic ellipse. This suppresses isotropic texture noise (e.g. clay
    grain) while preserving court lines at any orientation. A 3×3 median
    pre-blur further reduces fine texture grain.

    Returns (line_mask, court_mask, hsv, thresh_v, thresh_s).
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Oriented top-hat/black-hat: max response across 6 line kernel angles
    angles = [0, 30, 60, 90, 120, 150]
    tophat_v = np.zeros_like(V, dtype=np.float32)
    blackhat_s = np.zeros_like(S, dtype=np.float32)
    for angle in angles:
        k = np.zeros((tophat_ksize, tophat_ksize), dtype=np.uint8)
        cx, cy = tophat_ksize // 2, tophat_ksize // 2
        half = tophat_ksize // 2
        dx = int(round(half * np.cos(np.radians(angle))))
        dy = int(round(half * np.sin(np.radians(angle))))
        cv2.line(k, (cx - dx, cy - dy), (cx + dx, cy + dy), 1, 1)
        tv = cv2.morphologyEx(V, cv2.MORPH_TOPHAT, k)
        tophat_v = np.maximum(tophat_v, tv.astype(np.float32))

        # Half-kernel blackhat: split SE into two halves, take max response
        # This allows detection of lines at court/surround boundaries
        # where only one side has chromatic court surface.
        k_left = k.copy()
        k_right = k.copy()

        # Zero out each half (split perpendicular to the line direction).
        # For a line at angle θ, the perpendicular is θ+90°.
        perp_dx = int(round(half * np.cos(np.radians(angle + 90))))
        perp_dy = int(round(half * np.sin(np.radians(angle + 90))))

        for py in range(tophat_ksize):
            for px in range(tophat_ksize):
                vx, vy = px - cx, py - cy
                proj = vx * perp_dx + vy * perp_dy
                if proj > 0:
                    k_left[py, px] = 0
                elif proj < 0:
                    k_right[py, px] = 0

        if np.count_nonzero(k_left) >= 3 and np.count_nonzero(k_right) >= 3:
            bs_left = cv2.morphologyEx(S, cv2.MORPH_BLACKHAT, k_left)
            bs_right = cv2.morphologyEx(S, cv2.MORPH_BLACKHAT, k_right)
            bs = np.maximum(bs_left.astype(np.float32), bs_right.astype(np.float32))
        else:
            bs = cv2.morphologyEx(S, cv2.MORPH_BLACKHAT, k).astype(np.float32)
        blackhat_s = np.maximum(blackhat_s, bs)

    tophat_v = np.clip(tophat_v, 0, 255).astype(np.uint8)
    blackhat_s = np.clip(blackhat_s, 0, 255).astype(np.uint8)

    # Adaptive thresholds via Otsu
    court_region = (V > 30)
    tophat_court = tophat_v[court_region]
    if len(tophat_court) > 100:
        otsu_v, _ = cv2.threshold(tophat_court, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_v = max(int(otsu_v * 0.7), 5)
    else:
        thresh_v = 20

    blackhat_court = blackhat_s[court_region]
    if len(blackhat_court) > 100:
        otsu_s, _ = cv2.threshold(blackhat_court, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_s = max(int(otsu_s * 0.85), 15)
    else:
        thresh_s = 15

    line_candidate = (tophat_v > thresh_v) & (blackhat_s > thresh_s)
    court_mask = (V > 30).astype(np.uint8) * 255

    court_01 = (court_mask > 0).astype(np.float32)
    kernel = np.ones((neighborhood, neighborhood), dtype=np.float32)
    neighbor_count = cv2.filter2D(court_01, cv2.CV_32F, kernel)
    has_neighbors = (neighbor_count >= min_court_neighbors)

    line_mask = (line_candidate & has_neighbors).astype(np.uint8) * 255
    return line_mask, court_mask, hsv, thresh_v, thresh_s


def agrawal_clahe_line_filter(
    frame_bgr: np.ndarray,
    neighborhood: int = NEIGHBORHOOD_SIZE,
    min_court_neighbors: int = MIN_COURT_NEIGHBORS,
    tophat_ksize: int = TOPHAT_KERNEL_SIZE,
    clahe_clip: float = 3.0,
    clahe_grid: int = 8,
    court_s_min: int = 40,
    line_s_max: int = SAT_LINE_S_MAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """CLAHE-enhanced Agrawal filter for low-contrast / night scenes.

    Applies CLAHE to the V channel before oriented top-hat extraction,
    amplifying local brightness contrast that floodlights and dim lighting
    reduce.  Uses an adaptive saturation ceiling (0.95 × court S median)
    so that lines on saturated courts (e.g. nighttime floodlit blue) are
    not rejected for absorbing court color.  The neighbor gate checks for
    lit pixels (V > 30) rather than chromatic pixels (S > threshold),
    allowing sideline pixels at the court-surround boundary to pass.

    Returns (line_mask, court_mask, hsv, thresh_v).
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # CLAHE on V: boosts local contrast so lines stand out under dim lighting
    clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                            tileGridSize=(clahe_grid, clahe_grid))
    V_enhanced = clahe.apply(V)

    # Oriented top-hat on CLAHE-enhanced V
    angles = [0, 30, 60, 90, 120, 150]
    tophat_v = np.zeros_like(V, dtype=np.float32)
    for angle in angles:
        k = np.zeros((tophat_ksize, tophat_ksize), dtype=np.uint8)
        cx, cy = tophat_ksize // 2, tophat_ksize // 2
        half = tophat_ksize // 2
        dx = int(round(half * np.cos(np.radians(angle))))
        dy = int(round(half * np.sin(np.radians(angle))))
        cv2.line(k, (cx - dx, cy - dy), (cx + dx, cy + dy), 1, 1)
        tv = cv2.morphologyEx(V_enhanced, cv2.MORPH_TOPHAT, k)
        tophat_v = np.maximum(tophat_v, tv.astype(np.float32))
    tophat_v = np.clip(tophat_v, 0, 255).astype(np.uint8)

    # Chromatic court mask: colored court surface, excludes dark backgrounds
    court_region = (S > court_s_min)

    # Adaptive threshold via Otsu on court-region top-hat values
    tophat_court = tophat_v[court_region]
    if len(tophat_court) > 100:
        otsu_v, _ = cv2.threshold(tophat_court, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_v = max(int(otsu_v * 0.7), 5)
    else:
        thresh_v = 20

    # Adaptive S gate: on saturated courts (nighttime floodlit),
    # lines absorb court color → moderate S.  Scale threshold to court.
    court_s_median = float(np.median(S[court_region])) if np.any(court_region) else 0
    adaptive_s_max = max(int(court_s_median * 0.95), line_s_max)

    # Line candidate: strong CLAHE top-hat AND below adaptive S ceiling
    line_candidate = (tophat_v > thresh_v) & (S < adaptive_s_max)

    court_mask = court_region.astype(np.uint8) * 255

    # Neighbor logic: line must be surrounded by lit pixels (V > 30).
    # Using brightness instead of saturation so that desaturated but
    # well-lit surfaces (e.g. green surround) count as valid context,
    # allowing sideline pixels at the court-surround boundary to pass.
    lit_region = (V > 30).astype(np.float32)
    kernel = np.ones((neighborhood, neighborhood), dtype=np.float32)
    neighbor_count = cv2.filter2D(lit_region, cv2.CV_32F, kernel)
    has_neighbors = (neighbor_count >= min_court_neighbors)

    line_mask = (line_candidate & has_neighbors).astype(np.uint8) * 255
    return line_mask, court_mask, hsv, thresh_v


# ===================================================================
# Stage 4: Hough line detection + classification (reused logic)
# ===================================================================

def _seg_angle(seg: np.ndarray) -> float:
    x1, y1, x2, y2 = seg
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180


def _angle_diff(a1: float, a2: float) -> float:
    """Shortest angular distance between two line directions in [0, 180) space."""
    diff = abs(a1 - a2) % 180
    return min(diff, 180 - diff)


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
    margin = 300
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
    baseline_angle: float | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Classify lines into horizontal (baselines/service) and vertical (sidelines).

    When baseline_angle is provided or auto-detected, classifies relative to it:
    segments within CLASSIFY_ANGLE_TOL of the baseline direction → horizontal,
    else → vertical.  This adapts to camera perspective and avoids fixed-threshold
    misclassification of foreshortened sidelines.
    """
    horizontal: list[np.ndarray] = []
    vertical: list[np.ndarray] = []
    if lines is None:
        return horizontal, vertical

    all_segs = [seg[0] if seg.ndim == 2 else seg for seg in lines]

    # Fallback: auto-detect baseline angle from the best near-horizontal
    # segment.  DEPRECATED — callers should pass baseline_angle from
    # _find_baseline_extent() via detect_lines_near_half() instead.
    if baseline_angle is None:
        best_score = 0
        for seg in all_segs:
            angle = _seg_angle(seg)
            length = np.hypot(seg[2] - seg[0], seg[3] - seg[1])
            horiz_dist = min(angle, 180 - angle)  # degrees from horizontal
            if horiz_dist < 25:
                score = length * (1.0 - horiz_dist / 25.0)
                if score > best_score:
                    best_score = score
                    baseline_angle = angle

    if baseline_angle is not None:
        for seg in all_segs:
            angle = _seg_angle(seg)
            if _angle_diff(angle, baseline_angle) < CLASSIFY_ANGLE_TOL:
                horizontal.append(seg)
            else:
                vertical.append(seg)
    else:
        # No baseline detected — fall back to absolute threshold
        for seg in all_segs:
            angle = _seg_angle(seg)
            if angle < 15 or angle > 165:
                horizontal.append(seg)
            else:
                vertical.append(seg)

    return horizontal, vertical


def merge_collinear_segments(
    segments: list[np.ndarray],
    angle_tol: float = 8.0,
    dist_tol: float = 15.0,
    max_gap: float = 80.0,
) -> list[np.ndarray]:
    """Merge segments that are nearly collinear and close together.

    Uses max-endpoint distance against the longer segment's line to avoid
    chimera merges where two lines converge locally but diverge at extremes.
    Distance tolerance scales down as angle difference increases, rejecting
    merges where slight angle differences compound over long spans.
    """
    if len(segments) <= 1:
        return segments

    def _debug_center(seg: np.ndarray) -> bool:
        x_mid = (float(seg[0]) + float(seg[2])) / 2.0
        return MERGE_DEBUG_X_RANGE[0] <= x_mid <= MERGE_DEBUG_X_RANGE[1]

    def _debug_pair(seg_a: np.ndarray, seg_b: np.ndarray) -> bool:
        return MERGE_DEBUG and _debug_center(seg_a) and _debug_center(seg_b)

    n = len(segments)

    # Precompute angles and lengths
    angles = [_seg_angle(s) for s in segments]
    lengths = [float(np.hypot(s[2] - s[0], s[3] - s[1])) for s in segments]

    # --- Union-Find ---
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    # --- Pairwise compatibility (3 conditions) ---
    for i in range(n):
        for j in range(i + 1, n):
            ai, aj = angles[i], angles[j]
            adiff = min(abs(ai - aj), 180 - abs(ai - aj))
            debug_pair = _debug_pair(segments[i], segments[j])

            if adiff > angle_tol:
                if debug_pair:
                    print(
                        f"    [MergeDebug] pair i={i} j={j} "
                        f"seg_i={tuple(int(v) for v in segments[i])} "
                        f"seg_j={tuple(int(v) for v in segments[j])} "
                        f"angle_diff={adiff:.2f} > angle_tol={angle_tol:.2f} -> NO (angle)"
                    )
                continue

            # Use longer segment as reference
            if lengths[j] > lengths[i]:
                ref, cand, ref_len = segments[j], segments[i], lengths[j]
            else:
                ref, cand, ref_len = segments[i], segments[j], lengths[i]
            ref_len = max(ref_len, 1e-6)

            # Perpendicular distance of BOTH candidate endpoints to ref line
            rx1, ry1, rx2, ry2 = ref.astype(float)
            a_coeff = ry2 - ry1
            b_coeff = -(rx2 - rx1)
            c_coeff = rx2 * ry1 - ry2 * rx1
            d1 = abs(a_coeff * cand[0] + b_coeff * cand[1] + c_coeff) / ref_len
            d2 = abs(a_coeff * cand[2] + b_coeff * cand[3] + c_coeff) / ref_len
            max_dist = max(d1, d2)

            # Adaptive tolerance: tighter when angle difference is larger
            effective_tol = dist_tol * (1.0 - 0.7 * adiff / angle_tol)

            if max_dist >= effective_tol:
                if debug_pair:
                    print(
                        f"    [MergeDebug] pair i={i} j={j} "
                        f"seg_i={tuple(int(v) for v in segments[i])} "
                        f"seg_j={tuple(int(v) for v in segments[j])} "
                        f"angle_diff={adiff:.2f} d1={d1:.2f} d2={d2:.2f} "
                        f"max_dist={max_dist:.2f} effective_tol={effective_tol:.2f} -> NO (distance)"
                    )
                continue

            # Endpoint-gap along the reference direction
            ux = (rx2 - rx1) / ref_len
            uy = (ry2 - ry1) / ref_len
            tA1 = float(segments[i][0]) * ux + float(segments[i][1]) * uy
            tA2 = float(segments[i][2]) * ux + float(segments[i][3]) * uy
            tB1 = float(segments[j][0]) * ux + float(segments[j][1]) * uy
            tB2 = float(segments[j][2]) * ux + float(segments[j][3]) * uy
            minA, maxA = min(tA1, tA2), max(tA1, tA2)
            minB, maxB = min(tB1, tB2), max(tB1, tB2)
            ep_gap = max(0.0, max(minA, minB) - min(maxA, maxB))

            if ep_gap > max_gap:
                if debug_pair:
                    print(
                        f"    [MergeDebug] pair i={i} j={j} "
                        f"seg_i={tuple(int(v) for v in segments[i])} "
                        f"seg_j={tuple(int(v) for v in segments[j])} "
                        f"angle_diff={adiff:.2f} max_dist={max_dist:.2f} "
                        f"ep_gap={ep_gap:.2f} > max_gap={max_gap:.2f} -> NO (gap)"
                    )
                continue

            if debug_pair:
                print(
                    f"    [MergeDebug] pair i={i} j={j} "
                    f"seg_i={tuple(int(v) for v in segments[i])} "
                    f"seg_j={tuple(int(v) for v in segments[j])} "
                    f"angle_diff={adiff:.2f} d1={d1:.2f} d2={d2:.2f} "
                    f"max_dist={max_dist:.2f} effective_tol={effective_tol:.2f} "
                    f"ep_gap={ep_gap:.2f} -> YES"
                )
            union(i, j)

    # --- Group by component root ---
    components: dict[int, list[int]] = {}
    for idx in range(n):
        components.setdefault(find(idx), []).append(idx)

    # Process components in deterministic order (smallest member index first)
    merged: list[np.ndarray] = []
    for root in sorted(components.keys(), key=lambda r: min(components[r])):
        member_indices = components[root]
        group = [segments[k] for k in member_indices]

        # Direction from longest segment in component
        best_local = max(range(len(group)), key=lambda k: lengths[member_indices[k]])
        best = group[best_local]
        dx, dy = best[2] - best[0], best[3] - best[1]
        norm = max(np.hypot(dx, dy), 1e-6)
        ux, uy = dx / norm, dy / norm

        intervals = []
        for s, orig_idx in zip(group, member_indices):
            p1 = s[0] * ux + s[1] * uy
            p2 = s[2] * ux + s[3] * uy
            if p1 <= p2:
                intervals.append((p1, p2, (s[0], s[1]), (s[2], s[3]), orig_idx))
            else:
                intervals.append((p2, p1, (s[2], s[3]), (s[0], s[1]), orig_idx))

        intervals.sort(key=lambda x: x[0])
        debug_group = MERGE_DEBUG and any(_debug_center(s) for s in group)
        if debug_group:
            print(
                f"    [MergeDebug] group root={root} size={len(group)} "
                f"members={member_indices} best={tuple(int(v) for v in best)}"
            )
            for idx_interval, (lo, hi, pt_lo, pt_hi, _oi) in enumerate(intervals):
                print(
                    f"    [MergeDebug] interval {idx_interval}: "
                    f"lo={lo:.2f} hi={hi:.2f} "
                    f"pt_lo={tuple(int(v) for v in pt_lo)} "
                    f"pt_hi={tuple(int(v) for v in pt_hi)}"
                )

        # Gap-split: cluster intervals by max_gap
        first = intervals[0]
        clusters = [(first[0], first[1], first[2], first[3], [first[4]])]
        for lo, hi, pt_lo, pt_hi, oi in intervals[1:]:
            prev_lo, prev_hi, prev_pt_lo, prev_pt_hi, prev_members = clusters[-1]
            gap = lo - prev_hi
            if debug_group:
                print(
                    f"    [MergeDebug] interval_gap prev_hi={prev_hi:.2f} lo={lo:.2f} "
                    f"gap={gap:.2f} max_gap={max_gap:.2f} -> "
                    f"{'MERGE' if gap <= max_gap else 'SPLIT'}"
                )
            if gap <= max_gap:
                new_hi = max(prev_hi, hi)
                new_pt_hi = pt_hi if hi >= prev_hi else prev_pt_hi
                prev_members.append(oi)
                clusters[-1] = (prev_lo, new_hi, prev_pt_lo, new_pt_hi, prev_members)
            else:
                clusters.append((lo, hi, pt_lo, pt_hi, [oi]))

        for c_lo, c_hi, c_pt_lo, c_pt_hi, _members in clusters:
            merged.append(np.array([int(c_pt_lo[0]), int(c_pt_lo[1]),
                                    int(c_pt_hi[0]), int(c_pt_hi[1])]))

    return merged


def filter_short_merged(
    segments: list[np.ndarray],
    min_length: float = MIN_MERGED_LENGTH,
) -> list[np.ndarray]:
    """Discard merged segments whose total span is below *min_length* px."""
    return [s for s in segments
            if np.hypot(s[2] - s[0], s[3] - s[1]) >= min_length]


def _group_h_segments_by_line(
    segments: list[np.ndarray],
    frame_width: int,
    y_tol: float = 30.0,
    slope_tol: float = 0.02,
) -> list[dict]:
    """Group horizontal segments into co-linear clusters.

    Groups by ``y_at_cx`` (y-value at frame center-x, ±*y_tol*) and slope
    consistency (slope difference ≤ *slope_tol*).  This prevents merging
    watermark text fragments (random short slopes) with real baseline
    segments (consistent angle).

    Returns a list of group dicts, each containing:
        - ``segments``: list of np.ndarray segments in the group
        - ``x_span``: (x_left, x_right) combined extent
        - ``y_at_cx``: centroid y at frame center-x
        - ``n_fragments``: number of segments
        - ``gap_ratio``: fraction of x-span NOT covered by segments
    """
    if not segments:
        return []

    cx = frame_width / 2.0

    # Compute y_at_cx and slope for each segment
    annotated: list[tuple[float, float, np.ndarray]] = []
    for seg in segments:
        y_cx = _line_y_at_x(seg, cx)
        if y_cx is None:
            continue
        slope = _seg_signed_slope(seg)
        annotated.append((y_cx, slope, seg))

    if not annotated:
        return []

    annotated.sort(key=lambda a: a[0])  # sort by y_at_cx

    # Build groups: same y-band AND consistent slope
    raw_groups: list[list[tuple[float, float, np.ndarray]]] = []
    current: list[tuple[float, float, np.ndarray]] = [annotated[0]]
    for entry in annotated[1:]:
        group_y = np.mean([e[0] for e in current])
        group_slope = np.median([e[1] for e in current])
        if abs(entry[0] - group_y) < y_tol and abs(entry[1] - group_slope) <= slope_tol:
            current.append(entry)
        else:
            raw_groups.append(current)
            current = [entry]
    raw_groups.append(current)

    # Build metadata for each group
    results: list[dict] = []
    for g in raw_groups:
        segs = [e[2] for e in g]
        x_left = float(min(min(s[0], s[2]) for s in segs))
        x_right = float(max(max(s[0], s[2]) for s in segs))
        x_span_width = x_right - x_left

        # gap_ratio: fraction of x-span not covered by segment projections
        covered = 0.0
        for s in segs:
            covered += abs(float(s[2]) - float(s[0]))
        gap_ratio = 1.0 - min(covered / max(x_span_width, 1e-6), 1.0)

        results.append({
            "segments": segs,
            "x_span": (x_left, x_right),
            "y_at_cx": float(np.mean([e[0] for e in g])),
            "n_fragments": len(segs),
            "gap_ratio": gap_ratio,
        })

    return results


def _find_baseline_extent(
    lines: np.ndarray | None,
    frame_height: int,
    frame_width: int,
) -> tuple[int, int, float] | None:
    """Find the court's horizontal extent from baseline-region H lines.

    Groups all near-horizontal segments in the baseline region by
    y-position (via ``_group_h_segments_by_line``) and selects the group
    with the widest combined x-span.  No minimum threshold — any extent
    is useful for spatial masking.

    Returns (x_left, x_right, baseline_angle_degrees) or None if no
    baseline found.  The angle is the length-weighted average angle of
    the best group's segments (in [0, 180) degrees).
    """
    if lines is None:
        return None

    cx = frame_width / 2.0

    # Collect all H segments in the bottom portion of the crop
    h_segs: list[np.ndarray] = []
    for line in lines:
        seg = line[0] if line.ndim == 2 else line
        x1, y1, x2, y2 = seg
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
        y_cx = _line_y_at_x(seg, cx)
        length = np.hypot(x2 - x1, y2 - y1)

        is_horizontal = angle < 15 or angle > 165
        is_bottom = (y_cx is not None and y_cx > frame_height * 0.50)
        is_long = length > frame_width * 0.10

        if is_horizontal and is_bottom and is_long:
            h_segs.append(seg.copy())

    if not h_segs:
        return None

    groups = _group_h_segments_by_line(h_segs, frame_width)
    if not groups:
        return None

    # Select the group with the widest combined x-span
    best = max(groups, key=lambda g: g["x_span"][1] - g["x_span"][0])
    x_left, x_right = best["x_span"]

    # Compute baseline angle as length-weighted average of the group's segments
    total_weight = 0.0
    weighted_sin = 0.0
    weighted_cos = 0.0
    for seg in best["segments"]:
        length = float(np.hypot(seg[2] - seg[0], seg[3] - seg[1]))
        angle_rad = np.radians(_seg_angle(seg))
        weighted_sin += length * np.sin(2 * angle_rad)
        weighted_cos += length * np.cos(2 * angle_rad)
        total_weight += length
    baseline_angle = (np.degrees(np.arctan2(weighted_sin, weighted_cos)) / 2) % 180

    return int(x_left), int(x_right), float(baseline_angle)


def create_court_spatial_mask(
    baseline_extent: tuple[int, int],
    frame_height: int,
    frame_width: int,
    margin_ratio: float = 0.05,
) -> np.ndarray:
    """Create a spatial mask covering the court's horizontal extent.

    Uses the baseline x-extent (from merged horizontal segments) to define
    the court's lateral bounds. Pixels outside these bounds are zeroed,
    eliminating surround noise that creates false vertical lines.
    """
    bl_x_left, bl_x_right = baseline_extent
    bl_span = bl_x_right - bl_x_left
    margin = max(int(margin_ratio * bl_span), 15)

    spatial_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    x_lo = max(0, bl_x_left - margin)
    x_hi = min(frame_width, bl_x_right + margin)
    spatial_mask[:, x_lo:x_hi] = 255
    return spatial_mask


def detect_lines_near_half(
    line_mask: np.ndarray,
    hough_threshold: int = HOUGH_THRESHOLD,
    min_length: int = HOUGH_MIN_LENGTH,
    max_gap: int = HOUGH_MAX_GAP,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, float | None]:
    """Two-pass Hough detection with baseline-anchored spatial masking.

    Pass 1: Run Hough on the full line mask to find the baseline (the
    strongest near-horizontal line near the bottom of the frame). The
    baseline is always detectable because it's long, high-contrast, and
    unaffected by surround noise (which creates vertical, not horizontal,
    false lines).

    Pass 2: Use the baseline endpoints to create a rectangular spatial
    mask covering only the court's horizontal extent. Apply this mask to
    the line pixels, then re-run Hough. This eliminates false vertical
    lines from the surround while preserving thin court features like
    singles sidelines.

    This solves two problems simultaneously:
    - Single-color courts (e.g., blue + green surround): surround noise
      that created false V lines is spatially excluded.
    - Multi-color courts: no hue-specific logic, works on any color.

    Returns (lines, baseline_extent, spatial_mask, baseline_angle).
    baseline_extent is (x_left, x_right) or None.
    baseline_angle is the dominant angle of the baseline group in degrees, or None.
    """
    h, w = line_mask.shape[:2]

    # Light morphological cleanup — close small gaps in line pixels.
    kernel_close = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # --- Pass 1: Find the baseline extent ---
    pass1_lines = cv2.HoughLinesP(
        cleaned, 1, np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_length,
        maxLineGap=max_gap,
    )

    bl_result = _find_baseline_extent(pass1_lines, h, w)

    if bl_result is None:
        # No baseline found — fall back to unmasked detection
        return pass1_lines, None, None, None

    baseline_extent = (bl_result[0], bl_result[1])
    baseline_angle = bl_result[2]

    # Reject baseline extents too narrow to produce useful spatial masks
    extent_span = baseline_extent[1] - baseline_extent[0]
    if extent_span < w * MIN_BASELINE_EXTENT_FRAC:
        print(f"    [ExtentGuard] Rejected: span={extent_span}px "
              f"({extent_span/w:.0%} < {MIN_BASELINE_EXTENT_FRAC:.0%})")
        return pass1_lines, None, None, baseline_angle

    # --- Create spatial mask from baseline extent ---
    spatial_mask = create_court_spatial_mask(baseline_extent, h, w)

    # --- Pass 2: Masked Hough for all lines ---
    masked = cv2.bitwise_and(cleaned, spatial_mask)

    pass2_lines = cv2.HoughLinesP(
        masked, 1, np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_length,
        maxLineGap=max_gap,
    )

    return pass2_lines, baseline_extent, spatial_mask, baseline_angle


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


def _point_to_line_dist(px: float, py: float, seg: np.ndarray) -> float:
    """Perpendicular distance from point (px, py) to the line defined by segment."""
    x1, y1, x2, y2 = seg.astype(float)
    dx, dy = x2 - x1, y2 - y1
    length = max(np.hypot(dx, dy), 1e-6)
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / length


def find_center_service_line(
    pre_filter_verticals: list[np.ndarray],
    identified: dict[str, np.ndarray | None],
    post_filter_verticals: list[np.ndarray],
    frame_height: int,
    frame_width: int,
) -> np.ndarray | None:
    """Detect the center service line from merged verticals before length filtering."""
    baseline = identified.get("near_baseline")
    if baseline is None or not pre_filter_verticals:
        return None

    bl_left_x = min(float(baseline[0]), float(baseline[2]))
    bl_right_x = max(float(baseline[0]), float(baseline[2]))
    bl_span = bl_right_x - bl_left_x
    if bl_span < 1:
        return None

    # Use sideline-baseline intersections for court-relative t (not frame-relative).
    # Only use matched pairs (both singles or both doubles) to avoid asymmetric span.
    for _sl_left_name, _sl_right_name in (
        ("left_doubles", "right_doubles"),
        ("left_singles", "right_singles"),
    ):
        _sl_left = identified.get(_sl_left_name)
        _sl_right = identified.get(_sl_right_name)
        if _sl_left is not None and _sl_right is not None:
            _lpt = line_intersection(baseline, _sl_left, w=frame_width * 2, h=frame_height * 2)
            _rpt = line_intersection(baseline, _sl_right, w=frame_width * 2, h=frame_height * 2)
            if _lpt is not None and _rpt is not None:
                court_left_x = min(float(_lpt[0]), float(_rpt[0]))
                court_right_x = max(float(_lpt[0]), float(_rpt[0]))
                court_span = court_right_x - court_left_x
                if court_span > 1:
                    bl_left_x, bl_right_x, bl_span = court_left_x, court_right_x, court_span
                    break  # prefer doubles (wider span), fall back to singles

    bl_y = _line_y_at_x(baseline, frame_width / 2.0)
    if bl_y is None or bl_y <= 0:
        return None

    expected_y = bl_y * 0.35
    min_length = bl_y * 0.23
    sideline_ids = {
        id(seg)
        for name in ("left_doubles", "left_singles", "right_singles", "right_doubles")
        if (seg := identified.get(name)) is not None
    }

    candidates = []
    for seg in pre_filter_verticals:
        pt = line_intersection(baseline, seg, w=frame_width * 2, h=frame_height * 2)
        if pt is None:
            continue

        t = (pt[0] - bl_left_x) / bl_span
        if t < 0.38 or t > 0.62:
            continue

        length = np.hypot(seg[2] - seg[0], seg[3] - seg[1])
        if length < min_length:
            print(f"    [CenterService] Gate 2 reject: length={length:.0f} < {min_length:.0f}")
            continue

        if id(seg) in sideline_ids:
            print(f"    [CenterService] Gate 3 reject: segment already matched as sideline")
            continue

        if seg[1] >= seg[3]:
            bottom_x, bottom_y = float(seg[0]), float(seg[1])
        else:
            bottom_x, bottom_y = float(seg[2]), float(seg[3])

        service_y_ratio = bottom_y / bl_y
        if service_y_ratio < 0.23 or service_y_ratio > 0.42:
            print(f"    [CenterService] Gate 4 reject: y_ratio={service_y_ratio:.2f} outside [0.23, 0.42]")
            continue

        candidates.append((abs(bottom_y - expected_y), bottom_y, t, seg))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    _, bottom_y, t, seg = candidates[0]
    print(f"    [CenterService] Selected: bottom_y={bottom_y:.0f}, t={t:.3f}")
    return seg


def identify_near_half_lines(
    horizontal: list[np.ndarray],
    vertical: list[np.ndarray],
    frame_height: int,
    frame_width: int,
    pre_filter_verticals: list[np.ndarray] | None = None,
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
        "center_service": None,
        "left_singles": None,
        "right_singles": None,
        "left_doubles": None,
        "right_doubles": None,
    }

    cx = frame_width / 2.0
    bl_y = None  # will be set if baseline is detected (used by vertical filter)

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
        # Group all horizontal segments and pick the bottommost group
        # whose combined x-span meets the minimum baseline threshold.
        h_segments = [entry[2] for entry in scored]
        groups = _group_h_segments_by_line(h_segments, frame_width)

        min_bl_span = frame_width * MIN_BASELINE_FRAC
        # Sort groups by y_at_cx descending (bottommost first)
        groups.sort(key=lambda g: g["y_at_cx"], reverse=True)

        baseline_seg = None
        for g in groups:
            x_left, x_right = g["x_span"]
            span = x_right - x_left
            if span >= min_bl_span:
                # Synthesize a segment from leftmost to rightmost extent
                # using the dominant (longest) fragment's slope
                longest = max(g["segments"], key=lambda s: abs(float(s[2]) - float(s[0])))
                slope = _seg_signed_slope(longest)
                anchor_y = _line_y_at_x(longest, (x_left + x_right) / 2.0)
                mid_x = (x_left + x_right) / 2.0
                y_left = anchor_y + slope * (x_left - mid_x)
                y_right = anchor_y + slope * (x_right - mid_x)
                baseline_seg = np.array([int(x_left), int(y_left),
                                         int(x_right), int(y_right)])
                bl_y = _line_y_at_x(baseline_seg, cx)
                result["near_baseline"] = baseline_seg
                n = g["n_fragments"]
                print(f"    [Baseline] Accepted group ({n} fragment{'s' if n > 1 else ''}): "
                      f"xspan={span:.0f} ({span/frame_width:.0%})")
                break
            else:
                print(f"    [Baseline] Rejected group: xspan={span:.0f} < {min_bl_span:.0f} "
                      f"({span/frame_width:.0%} < {MIN_BASELINE_FRAC:.0%})")

        if baseline_seg is None:
            print(f"    [Baseline] No group spans ≥{MIN_BASELINE_FRAC:.0%} of frame")
            result["near_baseline"] = None

    # --- Vertical lines (baseline-intersection matching with court proportions) ---
    bl = result.get("near_baseline")
    scored_v = []
    if len(vertical) >= 1 and bl is not None:
        if bl_y is not None:
            min_vline_length = bl_y * 0.85
        else:
            min_vline_length = frame_height * 0.40

        bl_left_x = min(bl[0], bl[2])
        bl_right_x = max(bl[0], bl[2])

        bl_span = bl_right_x - bl_left_x

        # Collect verticals that intersect the baseline, with normalized position
        candidates = []  # (t, seg) where t is normalized x along baseline
        for seg in vertical:
            length = np.hypot(seg[2] - seg[0], seg[3] - seg[1])
            if length < min_vline_length:
                continue
            pt = line_intersection(bl, seg, w=FRAME_W * 2, h=FRAME_H * 2)
            if pt is None or bl_span < 1:
                continue
            t = (pt[0] - bl_left_x) / bl_span
            candidates.append((t, seg))
            ref_y = frame_height * 0.8
            x_at_ref = _line_x_at_y(seg, ref_y)
            if x_at_ref is not None:
                scored_v.append((x_at_ref, length, seg))
        scored_v.sort(key=lambda x: x[0])

        # Expected normalized positions from ITF court proportions
        expected = {
            "left_doubles": 0.0,
            "left_singles": 1.37 / 10.97,    # ~0.125
            "right_singles": 1.0 - 1.37 / 10.97,  # ~0.875
            "right_doubles": 1.0,
        }
        inner_tol = 0.10   # toward court center
        outer_tol = 0.15   # toward frame edge (off-frame corners)

        # Greedy nearest-match: doubles first (most distinct), then singles
        # Among candidates within tolerance, prefer the one reaching closest
        # to the top of the frame (net) — sidelines always come from the net.
        assigned = set()
        matched_t: dict[str, float] = {}
        role_order = ["left_doubles", "right_doubles", "left_singles", "right_singles"]
        for role in role_order:
            exp_t = expected[role]
            best_idx, best_top_y = None, float('inf')
            for i, (t, seg) in enumerate(candidates):
                if i in assigned:
                    continue
                dist = abs(t - exp_t)
                # Asymmetric tolerance: wider toward frame edges (off-frame corners)
                is_outside = (exp_t < 0.5 and t < exp_t) or (exp_t >= 0.5 and t > exp_t)
                tol = outer_tol if is_outside else inner_tol
                if dist <= tol:
                    top_y = min(seg[1], seg[3])
                    if top_y < best_top_y:
                        best_top_y = top_y
                        best_idx = i
            if best_idx is not None:
                assigned.add(best_idx)
                result[role] = candidates[best_idx][1]
                matched_t[role] = candidates[best_idx][0]
                best_t = candidates[best_idx][0]
                print(f"    [Sideline] {role}: t={best_t:.2f} "
                      f"(expected {exp_t:.2f}, dist={abs(best_t - exp_t):.2f})")

        for i, (t, seg) in enumerate(candidates):
            if i not in assigned:
                print(f"    [Sideline] Unmatched vertical at t={t:.2f} (no role within tolerance)")

        # --- Post-processing: reassign lone sidelines for off-frame courts ---
        ld, ls = result["left_doubles"], result["left_singles"]
        rd, rs = result["right_doubles"], result["right_singles"]

        if ld is not None and ls is None and rd is not None and rs is not None:
            # Both right sidelines detected, only one left → use right gap to check
            right_gap = matched_t["right_doubles"] - matched_t["right_singles"]
            expected_ls_t = matched_t["left_doubles"] + right_gap
            if 0 <= expected_ls_t <= 1:
                print(f"    [Sideline] Reassign left_doubles→left_singles: "
                      f"companion expected at t={expected_ls_t:.2f} (in-frame but missing), "
                      f"right gap={right_gap:.2f}")
                result["left_singles"] = result["left_doubles"]
                result["left_doubles"] = None

        elif rd is not None and rs is None and ld is not None and ls is not None:
            # Both left sidelines detected, only one right → use left gap to check
            left_gap = matched_t["left_singles"] - matched_t["left_doubles"]
            expected_rs_t = matched_t["right_doubles"] - left_gap
            if 0 <= expected_rs_t <= 1:
                print(f"    [Sideline] Reassign right_doubles→right_singles: "
                      f"companion expected at t={expected_rs_t:.2f} (in-frame but missing), "
                      f"left gap={left_gap:.2f}")
                result["right_singles"] = result["right_doubles"]
                result["right_doubles"] = None

        # Re-check after cross-check: one sideline per side → default to singles
        ld, ls = result["left_doubles"], result["left_singles"]
        rd, rs = result["right_doubles"], result["right_singles"]
        if ld is not None and ls is None and rd is not None and rs is None:
            print(f"    [Sideline] One line per side, defaulting both to singles")
            result["left_singles"] = result["left_doubles"]
            result["left_doubles"] = None
            result["right_singles"] = result["right_doubles"]
            result["right_doubles"] = None

    # --- Center service line detection ---
    if bl_y is not None and pre_filter_verticals is not None:
        result["center_service"] = find_center_service_line(
            pre_filter_verticals, result, vertical, frame_height, frame_width)

    # --- Service line (uses baseline + sidelines) ---
    if bl_y is not None and len(scored) >= 2:
        baseline_xspan = abs(baseline_seg[2] - baseline_seg[0])
        bl_angle = _seg_angle(baseline_seg)
        expected_y = bl_y * (1.0 - 0.65)  # perspective-adjusted: ~65% up from baseline
        if result["center_service"] is not None:
            center_seg = result["center_service"]
            expected_y = max(float(center_seg[1]), float(center_seg[3]))

        service_candidates = []
        for y_at_cx, length, seg in scored[:-1]:
            # Constraint 1: Y-position band
            candidate_y = y_at_cx
            y_ratio = (bl_y - candidate_y) / bl_y if bl_y > 0 else 0.0
            if y_ratio < 0.30 or y_ratio > 0.78:
                print(f"    [Service] Rejected: y_ratio={y_ratio:.2f} outside [0.30, 0.78]")
                continue

            # Constraint 2: Width ≤ baseline
            candidate_xspan = abs(seg[2] - seg[0])
            if candidate_xspan > baseline_xspan:
                print(f"    [Service] Rejected: xspan={candidate_xspan:.0f} > baseline={baseline_xspan:.0f}")
                continue

            # Constraint 3: Angle within 10° of baseline
            seg_angle = _seg_angle(seg)
            angle_diff = abs(seg_angle - bl_angle)
            if angle_diff > 90:
                angle_diff = 180 - angle_diff
            if angle_diff >= 10:
                print(f"    [Service] Rejected: angle_diff={angle_diff:.1f}° >= 10°")
                continue

            # Constraint 4: Sideline endpoint test (when sidelines detected)
            left_sl = result["left_singles"]
            right_sl = result["right_singles"]
            if left_sl is not None or right_sl is not None:
                tolerance = max(40, baseline_xspan * 0.04)
                seg_left_x = min(seg[0], seg[2])
                seg_right_x = max(seg[0], seg[2])
                # Identify left/right endpoints with their y-values
                if seg[0] <= seg[2]:
                    lx, ly = float(seg[0]), float(seg[1])
                    rx, ry = float(seg[2]), float(seg[3])
                else:
                    lx, ly = float(seg[2]), float(seg[3])
                    rx, ry = float(seg[0]), float(seg[1])

                sideline_ok = True
                if left_sl is not None and right_sl is not None:
                    dist_left = _point_to_line_dist(lx, ly, left_sl)
                    dist_right = _point_to_line_dist(rx, ry, right_sl)
                    if dist_left > tolerance or dist_right > tolerance:
                        print(f"    [Service] Rejected: sideline dist L={dist_left:.1f} R={dist_right:.1f} "
                              f"(tol={tolerance:.1f})")
                        sideline_ok = False
                elif left_sl is not None:
                    dist_left = _point_to_line_dist(lx, ly, left_sl)
                    if dist_left > tolerance:
                        print(f"    [Service] Rejected: left sideline dist={dist_left:.1f} "
                              f"(tol={tolerance:.1f})")
                        sideline_ok = False
                elif right_sl is not None:
                    dist_right = _point_to_line_dist(rx, ry, right_sl)
                    if dist_right > tolerance:
                        print(f"    [Service] Rejected: right sideline dist={dist_right:.1f} "
                              f"(tol={tolerance:.1f})")
                        sideline_ok = False

                if not sideline_ok:
                    continue

            service_candidates.append((y_at_cx, length, seg))

        if service_candidates:
            # Constraint 5: Pick closest to expected y-position, NOT longest
            service_candidates.sort(key=lambda c: abs(c[0] - expected_y))
            result["near_service"] = service_candidates[0][2]
            print(f"    [Service] Selected: y={service_candidates[0][0]:.0f} "
                  f"(expected={expected_y:.0f}, "
                  f"y_ratio={(bl_y - service_candidates[0][0]) / bl_y:.2f})")

    return result


# ===================================================================
# Stage 5b: Fallback service line detection (band-masked Hough)
# ===================================================================

def fallback_service_line_detection(
    line_mask: np.ndarray,
    identified: dict[str, np.ndarray | None],
    nh_h: int,
    nh_w: int,
) -> np.ndarray | None:
    """Detect service line using band-masked Hough when normal pipeline misses it.

    Uses the baseline's y-position and angle to constrain the search:
    - Y-band: service line must be between 0.50 and 0.78 of baseline y
    - Angle: within ±8° of baseline angle
    - Aggressive Hough parameters (lower threshold, shorter min length, larger gap)
    """
    baseline = identified.get("near_baseline")
    if baseline is None or identified.get("near_service") is not None:
        return None

    bl = baseline
    bl_y = (bl[1] + bl[3]) / 2.0
    bl_angle = np.degrees(np.arctan2(bl[3] - bl[1], bl[2] - bl[0]))

    # Y-band: service line is between 50% and 78% of baseline y
    y_min = int(bl_y * 0.22)  # top of band (22% from top = 78% down from baseline)
    y_max = int(bl_y * 0.50)  # bottom of band (50% from top = 50% down from baseline)

    if y_min >= y_max or y_max <= 0:
        return None

    # Create band mask
    band_mask = np.zeros_like(line_mask)
    band_mask[y_min:y_max, :] = line_mask[y_min:y_max, :]

    # X-margin: exclude 5% on each side
    x_margin = int(nh_w * 0.05)
    if x_margin > 0:
        band_mask[:, :x_margin] = 0
        band_mask[:, nh_w - x_margin:] = 0

    if cv2.countNonZero(band_mask) < 20:
        return None

    # Aggressive Hough parameters
    raw_lines = cv2.HoughLinesP(
        band_mask,
        rho=1, theta=np.pi / 180,
        threshold=15,
        minLineLength=20,
        maxLineGap=60,
    )

    if raw_lines is None:
        return None

    # Filter by angle: within ±8° of baseline
    angle_tol = 8.0
    candidates = []
    for seg in raw_lines[:, 0]:
        x1, y1, x2, y2 = seg
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle - bl_angle) <= angle_tol or abs(angle - bl_angle + 180) <= angle_tol or abs(angle - bl_angle - 180) <= angle_tol:
            candidates.append(seg)

    if not candidates:
        return None

    # Merge nearby segments (generous tolerances)
    merged = merge_collinear_segments(list(np.array(candidates)), angle_tol=8.0, dist_tol=15.0, max_gap=80.0)

    # Filter by minimum merged length
    min_length = 80
    long_enough = []
    for seg in merged:
        length = np.hypot(seg[2] - seg[0], seg[3] - seg[1])
        if length >= min_length:
            long_enough.append(seg)

    if not long_enough:
        return None

    # Pick the longest segment
    best = max(long_enough, key=lambda s: np.hypot(s[2] - s[0], s[3] - s[1]))
    return np.array(best, dtype=np.float64)


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

    # KP13: midpoint of KP10/KP11 on the service line
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
) -> tuple[np.ndarray | None, list[tuple[int, tuple[float, float]]]]:
    """Compute homography from near-half keypoint correspondences.

    Maps image pixels → court meters. Returns (H, inlier_kps).
    """
    if len(near_kps) < 4:
        return None, near_kps

    src_pts = []
    dst_pts = []
    for kp_idx, (px, py) in near_kps:
        src_pts.append([px, py])
        dst_pts.append(list(REFERENCE_KPS_METERS[kp_idx]))

    src = np.array(src_pts, dtype=np.float64)
    dst = np.array(dst_pts, dtype=np.float64)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 0.5)

    if H is None or mask is None:
        return None, near_kps

    # Filter to inliers only
    inlier_mask = mask.ravel().astype(bool)
    outlier_indices = [near_kps[i][0] for i in range(len(near_kps)) if not inlier_mask[i]]
    if outlier_indices:
        names = [KP_NAMES[k] if k < len(KP_NAMES) else f"KP{k}" for k in outlier_indices]
        print(f"    [RANSAC] Rejected outliers: {', '.join(f'KP{k} ({n})' for k, n in zip(outlier_indices, names))}")
    inlier_kps = [near_kps[i] for i in range(len(near_kps)) if inlier_mask[i]]
    print(f"    [RANSAC] Inliers: {len(inlier_kps)}/{len(near_kps)}")

    return H, inlier_kps


EDGE_ALIGN_LINES = {
    "near_baseline":     ((-5.485, 11.885), (5.485, 11.885)),
    "left_singles":      ((-4.115, 11.885), (-4.115, 6.4)),
    "right_singles":     ((4.115, 11.885), (4.115, 6.4)),
    "left_doubles":      ((-5.485, 11.885), (-5.485, 6.4)),
    "right_doubles":     ((5.485, 11.885), (5.485, 6.4)),
    "near_service":      ((-4.115, 6.4), (4.115, 6.4)),
    "center_service":    ((0.0, 6.4), (0.0, 0.0)),
    "left_doubles_ext":  ((-5.485, 6.4), (-5.485, 0.0)),
    "right_doubles_ext": ((5.485, 6.4), (5.485, 0.0)),
}


def compute_distance_transform(line_mask: np.ndarray) -> np.ndarray:
    """Compute distance transform: dt[y,x] = distance to nearest white pixel."""
    inverted = 255 - line_mask
    dt = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)
    return dt


def debug_per_line_chamfer(
    H: np.ndarray,
    line_mask: np.ndarray,
    y_offset: int,
    n_samples: int = 50,
    dt_cap: float = 30.0,
    label: str = "",
) -> dict[str, float]:
    """Compute and print per-line chamfer distance for debugging.

    Projects each EDGE_ALIGN_LINES template line through H^-1 into pixel
    space and measures mean distance to nearest white pixel in line_mask.
    """
    dt = compute_distance_transform(line_mask)
    img_h, img_w = dt.shape

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        print(f"    [{label}] H is singular, cannot compute per-line chamfer")
        return {}

    results: dict[str, float] = {}
    for name, (m1, m2) in EDGE_ALIGN_LINES.items():
        ts = np.linspace(0.0, 1.0, n_samples)
        meter_pts = np.column_stack([
            m1[0] + ts * (m2[0] - m1[0]),
            m1[1] + ts * (m2[1] - m1[1]),
        ]).reshape(-1, 1, 2).astype(np.float64)

        try:
            pixel_pts = cv2.perspectiveTransform(meter_pts, H_inv).reshape(-1, 2)
        except cv2.error:
            results[name] = 999.0
            continue

        pixel_pts[:, 1] -= y_offset
        distances = []
        n_oob = 0
        for px, py in pixel_pts:
            ix, iy = int(round(px)), int(round(py))
            if 0 <= ix < img_w and 0 <= iy < img_h:
                distances.append(min(float(dt[iy, ix]), dt_cap))
            else:
                distances.append(dt_cap)
                n_oob += 1
        mean_d = float(np.mean(distances))
        results[name] = mean_d

        # Compute projected pixel endpoints for context
        p1 = pixel_pts[0]
        p2 = pixel_pts[-1]
        oob_str = f", {n_oob}/{n_samples} OOB" if n_oob > 0 else ""
        print(f"    [{label}] {name:20s}: chamfer={mean_d:5.1f}px  "
              f"px=({p1[0]:6.1f},{p1[1]:6.1f})->({p2[0]:6.1f},{p2[1]:6.1f})"
              f"{oob_str}")

    total = float(np.mean(list(results.values()))) if results else 999.0
    print(f"    [{label}] TOTAL mean chamfer = {total:.2f}px")
    return results


def _edge_align_cost(
    h8: np.ndarray,
    dt: np.ndarray,
    template_lines: list[tuple[tuple[float,float], tuple[float,float]]],
    y_offset: int,
    n_samples: int = 50,
    dt_cap: float = 30.0,
    penalty: float = 50.0,
    kp_pixels: np.ndarray | None = None,
    kp_meters: np.ndarray | None = None,
    lambda_reg: float = 0.5,
) -> float:
    """Compute edge-alignment cost for a candidate homography.
    
    Cost = mean truncated chamfer distance of projected template lines
           to line_mask (via distance transform)
         + lambda_reg * mean keypoint reprojection error
    """
    H = np.zeros((3, 3), dtype=np.float64)
    H.flat[:8] = h8
    H[2, 2] = 1.0
    
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return 1e6
    
    if abs(np.linalg.det(H)) < 1e-10:
        return 1e6
    
    img_h, img_w = dt.shape
    all_distances = []
    
    for (mx1, my1), (mx2, my2) in template_lines:
        ts = np.linspace(0.0, 1.0, n_samples)
        meter_pts = np.column_stack([
            mx1 + ts * (mx2 - mx1),
            my1 + ts * (my2 - my1),
        ]).reshape(-1, 1, 2).astype(np.float64)
        
        try:
            pixel_pts = cv2.perspectiveTransform(meter_pts, H_inv).reshape(-1, 2)
        except cv2.error:
            all_distances.extend([penalty] * n_samples)
            continue
        
        pixel_pts[:, 1] -= y_offset

        # Bilinear interpolation on DT for smooth sub-pixel gradients
        px_arr = pixel_pts[:, 0]
        py_arr = pixel_pts[:, 1]
        in_bounds = ((px_arr >= 0) & (px_arr < img_w - 1) &
                     (py_arr >= 0) & (py_arr < img_h - 1))
        for j in range(len(px_arr)):
            if in_bounds[j]:
                x, y = float(px_arr[j]), float(py_arr[j])
                x0, y0 = int(x), int(y)
                x1, y1 = x0 + 1, y0 + 1
                fx, fy = x - x0, y - y0
                d = (dt[y0, x0] * (1 - fx) * (1 - fy) +
                     dt[y0, x1] * fx * (1 - fy) +
                     dt[y1, x0] * (1 - fx) * fy +
                     dt[y1, x1] * fx * fy)
                all_distances.append(min(float(d), dt_cap))
            else:
                all_distances.append(penalty)
    
    if not all_distances:
        return 1e6
    
    chamfer_cost = np.mean(all_distances)
    
    reg_cost = 0.0
    if kp_pixels is not None and kp_meters is not None and lambda_reg > 0:
        meter_pts = kp_meters.reshape(-1, 1, 2).astype(np.float64)
        try:
            proj = cv2.perspectiveTransform(meter_pts, H_inv).reshape(-1, 2)
            diffs = proj - kp_pixels
            reg_cost = np.mean(np.sqrt(np.sum(diffs**2, axis=1)))
        except (cv2.error, np.linalg.LinAlgError):
            reg_cost = 100.0
    
    return chamfer_cost + lambda_reg * reg_cost


def refine_homography_edge_align(
    H_init: np.ndarray,
    line_mask: np.ndarray,
    near_kps: list[tuple[int, tuple[float, float]]],
    y_offset: int,
    lambda_reg: float = 0.5,
    n_samples: int = 50,
    dt_cap: float = 30.0,
    max_iter: int = 100,
    min_init_quality: float = 25.0,
) -> np.ndarray:
    """Refine homography by maximizing alignment with edge/line mask.
    
    Uses distance transform of the local-contrast line mask as a smooth
    cost landscape. Optimizes 8-DOF homography to minimize mean chamfer
    distance of projected court template lines to mask pixels.
    """
    if H_init is None:
        return H_init
    
    dt = compute_distance_transform(line_mask)
    template_lines = list(EDGE_ALIGN_LINES.values())
    
    kp_pixels = np.array([[px, py] for _, (px, py) in near_kps], dtype=np.float64)
    kp_meters = np.array([REFERENCE_KPS_METERS[idx] for idx, _ in near_kps],
                         dtype=np.float64)
    
    h0 = H_init.flatten()[:8] / H_init[2, 2]
    
    init_cost = _edge_align_cost(
        h0, dt, template_lines, y_offset, n_samples, dt_cap, 50.0,
        kp_pixels, kp_meters, lambda_reg)

    # Debug: show per-line chamfer and per-KP reprojection BEFORE optimization
    print(f"    [EdgeAlign] --- BEFORE optimization (total_cost={init_cost:.2f}, "
          f"lambda_reg={lambda_reg}) ---")
    debug_per_line_chamfer(H_init, line_mask, y_offset, n_samples, dt_cap,
                           label="EdgeAlign-BEFORE")
    # Per-KP reprojection error
    try:
        H_inv_init = np.linalg.inv(H_init)
        m_pts = kp_meters.reshape(-1, 1, 2).astype(np.float64)
        proj_init = cv2.perspectiveTransform(m_pts, H_inv_init).reshape(-1, 2)
        for i, (kp_idx, (px, py)) in enumerate(near_kps):
            name = KP_NAMES[kp_idx] if kp_idx < len(KP_NAMES) else f"KP{kp_idx}"
            dx = proj_init[i, 0] - px
            dy = proj_init[i, 1] - py
            err = np.hypot(dx, dy)
            print(f"    [EdgeAlign-BEFORE] KP{kp_idx:2d} ({name:6s}): "
                  f"reproj_err={err:6.1f}px  "
                  f"(proj=({proj_init[i,0]:.1f},{proj_init[i,1]:.1f}) "
                  f"vs det=({px:.1f},{py:.1f}))")
    except np.linalg.LinAlgError:
        print(f"    [EdgeAlign-BEFORE] Cannot compute KP reproj (singular H)")

    if init_cost > min_init_quality:
        print(f"    [EdgeAlign] Initial cost too high ({init_cost:.1f} > "
              f"{min_init_quality}), skipping")
        return H_init
    
    result = minimize(
        _edge_align_cost,
        h0,
        args=(dt, template_lines, y_offset, n_samples, dt_cap, 50.0,
              kp_pixels, kp_meters, lambda_reg),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-6, 'disp': False},
    )
    
    H_refined = np.zeros((3, 3), dtype=np.float64)
    H_refined.flat[:8] = result.x
    H_refined[2, 2] = 1.0
    
    refined_cost = result.fun
    
    err_init = reprojection_error(near_kps, H_init)
    err_refined = reprojection_error(near_kps, H_refined)

    # Debug: show per-line chamfer AFTER optimization
    print(f"    [EdgeAlign] --- AFTER optimization (total_cost={refined_cost:.2f}, "
          f"iterations={result.nit}, converged={result.success}) ---")
    debug_per_line_chamfer(H_refined, line_mask, y_offset, n_samples, dt_cap,
                           label="EdgeAlign-AFTER")
    
    # When lambda_reg=0, accept purely on chamfer improvement (KPs are untrusted)
    accept = refined_cost < init_cost
    if lambda_reg > 0:
        accept = accept and err_refined < err_init * 1.5

    if accept:
        print(f"    [EdgeAlign] Refined: chamfer {init_cost:.2f} → {refined_cost:.2f}px, "
              f"reproj {err_init:.2f} → {err_refined:.2f}px")
        return H_refined
    else:
        print(f"    [EdgeAlign] No improvement (chamfer {init_cost:.2f} → {refined_cost:.2f}, "
              f"reproj {err_init:.2f} → {err_refined:.2f}), keeping original")
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
    court_mode: str = "single",
) -> np.ndarray:
    """Create 2×3 diagnostic composite."""
    cell_w, cell_h = 640, 360

    def resize(img):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.resize(img, (cell_w, cell_h))

    # Panel 1: Near-half crop
    p1 = resize(near_half)
    cv2.putText(p1, "1. Near-half crop", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Panel 2: Line filter mask
    p2 = resize(line_mask)
    cv2.putText(p2, "2. Agrawal line mask", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Panel 3: All raw Hough lines on near-half
    p3 = near_half.copy()
    for seg in horizontal:
        cv2.line(p3, (seg[0], seg[1]), (seg[2], seg[3]), (255, 100, 0), 2)
    for seg in vertical:
        cv2.line(p3, (seg[0], seg[1]), (seg[2], seg[3]), (0, 0, 255), 2)
    p3 = resize(p3)
    if court_mode == "multi":
        mode_label = "MULTI"
    elif court_mode == "local_contrast":
        mode_label = "LOCAL"
    else:
        mode_label = "SINGLE"
    cv2.putText(p3, f"3. Raw Hough [{mode_label}] {len(horizontal)}H+{len(vertical)}V", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # Panel 4: Identified court lines only (no raw segments)
    p4 = near_half.copy()
    colors = {
        "near_baseline": (0, 255, 0),
        "near_service": (0, 200, 200),
        "center_service": (0, 128, 255),
        "left_singles": (255, 0, 255),
        "right_singles": (255, 0, 255),
        "left_doubles": (200, 200, 0),
        "right_doubles": (200, 200, 0),
    }
    for name, seg in identified_lines.items():
        if seg is not None:
            c = colors.get(name, (255, 255, 255))
            cv2.line(p4, (int(seg[0]), int(seg[1])), (int(seg[2]), int(seg[3])), c, 3)
            mx = int((seg[0] + seg[2]) // 2)
            my = int((seg[1] + seg[3]) // 2)
            cv2.putText(p4, name, (mx, my - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)
    p4 = resize(p4)
    cv2.putText(p4, "4. Court lines", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Panel 5: Edge-alignment overlay (line mask + projected template)
    p5 = resize(line_mask)
    if H_full is not None:
        try:
            H_inv = np.linalg.inv(H_full)
            mask_h, mask_w = line_mask.shape[:2]
            sx = cell_w / mask_w
            sy = cell_h / mask_h
            for (m1, m2) in EDGE_ALIGN_LINES.values():
                ts = np.linspace(0, 1, 50)
                pts_m = np.column_stack([
                    m1[0] + ts*(m2[0]-m1[0]),
                    m1[1] + ts*(m2[1]-m1[1]),
                ]).reshape(-1,1,2).astype(np.float64)
                pts_px = cv2.perspectiveTransform(pts_m, H_inv).reshape(-1,2)
                pts_px[:, 1] -= y_offset
                for i in range(len(pts_px)-1):
                    pt1 = (int(pts_px[i,0]*sx), int(pts_px[i,1]*sy))
                    pt2 = (int(pts_px[i+1,0]*sx), int(pts_px[i+1,1]*sy))
                    cv2.line(p5, pt1, pt2, (0, 255, 0), 1)
        except Exception:
            pass
    cv2.putText(p5, "5. Edge-align overlay", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

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
    cv2.putText(p6, "6. Keypoints (green=near, orange=far)", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Assemble 2×3 grid
    row1 = np.hstack([p1, p2, p3])
    row2 = np.hstack([p4, p5, p6])
    composite = np.vstack([row1, row2])
    return composite


def compute_metrics(
    near_kps: list[tuple[int, tuple[float, float]]],
    all_kps: list[tuple[int, tuple[float, float]]],
    H_full: np.ndarray | None,
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
# Court detection scoring (Agrawal Section 3.4)
# ===================================================================

def _project_court_lines(H_near, y_offset, nh_h, nh_w, n_samples=30):
    """Project court template lines into near-half pixel coords.

    Returns list of (pixel_pts, meter_line) for lines with >=2 in-frame points.
    pixel_pts is Nx2 array in near-half coords.
    """
    try:
        H_inv = np.linalg.inv(H_near)
    except np.linalg.LinAlgError:
        return []

    court_lines_m = [
        ((-5.485, 11.885), (5.485, 11.885)),   # Near baseline
        ((-4.115, 6.4), (4.115, 6.4)),          # Near service line
        ((-5.485, 11.885), (-5.485, 0.0)),      # Left doubles sideline
        ((5.485, 11.885), (5.485, 0.0)),        # Right doubles sideline
        ((-4.115, 11.885), (-4.115, 6.4)),      # Left singles sideline
        ((4.115, 11.885), (4.115, 6.4)),        # Right singles sideline
        ((0.0, 6.4), (0.0, 0.0)),               # Center service line
    ]

    projected = []
    for (mx1, my1), (mx2, my2) in court_lines_m:
        ts = np.linspace(0.0, 1.0, n_samples)
        meter_pts = np.column_stack([
            mx1 + ts * (mx2 - mx1),
            my1 + ts * (my2 - my1),
        ]).reshape(-1, 1, 2).astype(np.float64)

        try:
            pixel_pts = cv2.perspectiveTransform(meter_pts, H_inv).reshape(-1, 2)
        except cv2.error:
            continue

        pixel_pts[:, 1] -= y_offset

        # Keep only points within or near frame bounds
        margin = 20
        in_frame = ((pixel_pts[:, 0] >= -margin) & (pixel_pts[:, 0] < nh_w + margin) &
                    (pixel_pts[:, 1] >= -margin) & (pixel_pts[:, 1] < nh_h + margin))
        if in_frame.sum() < 2:
            continue
        projected.append(pixel_pts[in_frame])

    return projected


def _line_profile_score(gray, projected_lines, n_profile_samples=24,
                        perp_halfwidth=10, expected_line_halfwidth=2,
                        search_radius=8, offset_sigma=4.0):
    """Score how well projected lines look like white court lines in raw image.

    For each projected line, samples perpendicular brightness profiles and
    searches for the brightest peak within ±search_radius of the projected
    center.  Scores the bright-center/dark-flanks pattern at the found peak,
    then applies a Gaussian penalty based on how far off it is.

    Returns float in [0, 1].
    """
    h, w = gray.shape[:2]
    all_scores = []

    for pts in projected_lines:
        if len(pts) < 3:
            continue

        indices = np.linspace(1, len(pts) - 2, min(n_profile_samples, len(pts) - 2))
        indices = np.unique(indices.astype(int))

        for idx in indices:
            p = pts[idx]
            p_prev = pts[max(0, idx - 1)]
            p_next = pts[min(len(pts) - 1, idx + 1)]
            tangent = p_next - p_prev
            tlen = np.linalg.norm(tangent)
            if tlen < 1e-6:
                continue
            tangent = tangent / tlen
            perp = np.array([-tangent[1], tangent[0]])

            offsets = np.arange(-perp_halfwidth, perp_halfwidth + 1, dtype=np.float32)
            xs = p[0] + offsets * perp[0]
            ys = p[1] + offsets * perp[1]

            if xs.min() < 0 or xs.max() >= w or ys.min() < 0 or ys.max() >= h:
                continue

            profile = cv2.remap(
                gray,
                xs.reshape(1, -1).astype(np.float32),
                ys.reshape(1, -1).astype(np.float32),
                cv2.INTER_LINEAR,
            ).ravel().astype(np.float32)

            lo, hi = profile.min(), profile.max()
            if hi - lo < 20:
                all_scores.append(0.0)
                continue

            # Find brightness peak within search window around projected center
            center = perp_halfwidth  # index of projected center
            search_lo = max(expected_line_halfwidth + 1, center - search_radius)
            search_hi = min(len(profile) - expected_line_halfwidth - 2, center + search_radius)
            if search_lo > search_hi:
                all_scores.append(0.0)
                continue

            search_slice = profile[search_lo:search_hi + 1]
            peak_in_slice = int(np.argmax(search_slice))
            peak_idx = search_lo + peak_in_slice
            offset = abs(peak_idx - center)

            # Score contrast at the found peak position
            hw = expected_line_halfwidth
            c_lo = max(0, peak_idx - hw)
            c_hi = min(len(profile), peak_idx + hw + 1)
            center_val = profile[c_lo:c_hi].mean()

            # Flanks: pixels outside peak ± (hw+1), excluding the line region
            flank_left = profile[:max(1, peak_idx - hw)].mean()
            flank_right_start = peak_idx + hw + 1
            flank_right = (profile[flank_right_start:].mean()
                           if flank_right_start < len(profile) else flank_left)
            flank_avg = 0.5 * (flank_left + flank_right)

            contrast = (center_val - flank_avg) / (hi - lo + 1e-6)
            contrast = max(0.0, min(1.0, contrast))

            # Gaussian distance penalty: score degrades with offset
            distance_penalty = np.exp(-(offset ** 2) / (2.0 * offset_sigma ** 2))

            all_scores.append(contrast * distance_penalty)

    if not all_scores:
        return 0.0
    return float(np.mean(all_scores))


def _color_gate_score(hsv, projected_lines, nh_h, nh_w,
                      search_radius=6, offset_sigma=4.0):
    """Score whether the region around projected court lines has court-like
    color on the flanks (chromatic) and line-like color at center (low-sat, bright).

    Searches perpendicular to the projected line for the lowest-saturation
    point (the actual white line), then compares its saturation to the flanks.
    Applies a Gaussian distance penalty for offset from projected center.

    Returns float in [0, 1].
    """
    S = hsv[:, :, 1]

    scores = []

    for pts in projected_lines:
        if len(pts) < 3:
            continue

        indices = np.linspace(1, len(pts) - 2, min(15, len(pts) - 2))
        indices = np.unique(indices.astype(int))

        for idx in indices:
            p = pts[idx]
            p_prev = pts[max(0, idx - 1)]
            p_next = pts[min(len(pts) - 1, idx + 1)]
            tangent = p_next - p_prev
            tlen = np.linalg.norm(tangent)
            if tlen < 1e-6:
                continue
            perp = np.array([-tangent[1], tangent[0]]) / tlen

            # Search for lowest-saturation point within ±search_radius
            best_s = 255
            best_offset = 0
            best_cx, best_cy = -1, -1
            for d in range(-search_radius, search_radius + 1):
                cx = int(round(p[0] + d * perp[0]))
                cy = int(round(p[1] + d * perp[1]))
                if 0 <= cx < nh_w and 0 <= cy < nh_h:
                    sv = S[cy, cx]
                    if sv < best_s:
                        best_s = sv
                        best_offset = abs(d)
                        best_cx, best_cy = cx, cy

            if best_cx < 0:
                continue

            # Sample flanks at ±10px from the found line center
            flank_dist = 10
            fx1 = int(round(best_cx + flank_dist * perp[0]))
            fy1 = int(round(best_cy + flank_dist * perp[1]))
            fx2 = int(round(best_cx - flank_dist * perp[0]))
            fy2 = int(round(best_cy - flank_dist * perp[1]))

            flank_vals = []
            if 0 <= fx1 < nh_w and 0 <= fy1 < nh_h:
                flank_vals.append(float(S[fy1, fx1]))
            if 0 <= fx2 < nh_w and 0 <= fy2 < nh_h:
                flank_vals.append(float(S[fy2, fx2]))

            if not flank_vals:
                continue

            s_diff = np.mean(flank_vals) - float(best_s)
            raw_score = max(0.0, min(1.0, s_diff / 50.0))

            # Gaussian distance penalty
            distance_penalty = np.exp(-(best_offset ** 2) / (2.0 * offset_sigma ** 2))
            scores.append(raw_score * distance_penalty)

    if len(scores) < 5:
        return 0.0
    return float(np.median(scores))


def score_court_detection(
    near_half: np.ndarray,
    H_near: np.ndarray,
    y_offset: int = 0,
    line_mask: np.ndarray | None = None,
) -> int:
    """Score court detection using raw-image verification (line profile + color gate).

    Projects court template lines via H and checks:
    1. Line profile (70%): do projected lines show bright-center/dark-flanks
       pattern in raw grayscale? Independent of the detection pipeline.
    2. Color gate (30%): do projected lines have low saturation (white) with
       higher saturation flanks (court color)?

    Returns score in [0, 10000].
    """
    if H_near is None:
        return 0

    nh_h, nh_w = near_half.shape[:2]

    # Project court lines
    projected = _project_court_lines(H_near, y_offset, nh_h, nh_w)
    if len(projected) < 3:
        return 0

    gray = cv2.cvtColor(near_half, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(near_half, cv2.COLOR_BGR2HSV)

    # Line profile score (bright-dark-bright cross-section)
    profile = _line_profile_score(gray, projected)

    # Color gate (low-sat center, high-sat flanks)
    color = _color_gate_score(hsv, projected, nh_h, nh_w)

    # Combined score: 70% profile + 30% color
    combined = 0.70 * profile + 0.30 * color
    return int(round(combined * 10000))


# ===================================================================
# Main pipeline
# ===================================================================

def run_agrawal_pipeline(
    frame: np.ndarray,
    frame_idx: int,
    kalman: HomographyKalmanFilter | None = None,
    shadow_remover: "ShadowRemover | None" = None,
    net_detector=None,
    output_dir: Path | None = None,
    video_stem: str = "",
) -> dict:
    """Run the full Agrawal-inspired pipeline on a single frame."""
    t0 = time.time()
    results: dict = {"frame_idx": frame_idx}

    # --- Stage 1: Net detection & crop ---
    t1 = time.time()
    net_y, net_source, crop_margin = estimate_net_y(
        frame, net_detector=net_detector)
    print(f"  [Stage 1] Net y={net_y} (source={net_source}), "
          f"margin={crop_margin} ({time.time()-t1:.2f}s)")

    near_half, y_offset = crop_near_half(frame, net_y, margin=crop_margin)
    print(f"  [Stage 1] Near-half crop: {near_half.shape[1]}×{near_half.shape[0]}, "
          f"y_offset={y_offset}")

    # --- Stage 1.5: Shadow removal (optional, with auto-gating) ---
    if shadow_remover is not None:
        t_sr = time.time()
        should_apply, discrim, shadow_ratio = ShadowRemover.should_remove_shadows(near_half)
        if should_apply:
            near_half_sr = shadow_remover.remove_shadows(near_half)
            print(f"  [Stage 1.5] Shadow removal APPLIED "
                  f"(discrim={discrim:.3f}, shadow_ratio={shadow_ratio:.2f}, "
                  f"{time.time()-t_sr:.3f}s)")
            if output_dir is not None:
                diag_path = output_dir / f"shadow_removal_{video_stem}_frame_{frame_idx}.jpg"
                diag = np.hstack([near_half, near_half_sr])
                cv2.imwrite(str(diag_path), diag)
            near_half = near_half_sr
        else:
            print(f"  [Stage 1.5] Shadow removal SKIPPED "
                  f"(discrim={discrim:.3f}, shadow_ratio={shadow_ratio:.2f}, "
                  f"{time.time()-t_sr:.3f}s)")

    # --- Stage 2+3: Saturation-based Agrawal filter (color-agnostic) ---
    t2 = time.time()
    line_mask, court_color_mask, hsv_full, eff_tophat = agrawal_saturation_line_filter(near_half)
    # Derive court_color for logging/visualization (median HSV of chromatic pixels)
    chromatic_px = hsv_full[court_color_mask > 0]
    if len(chromatic_px) > 0:
        court_h = int(np.median(chromatic_px[:, 0]))
        court_s = int(np.median(chromatic_px[:, 1]))
        court_v = int(np.median(chromatic_px[:, 2]))
    else:
        court_h, court_s, court_v = 0, 0, 0
    court_color = np.array([court_h, court_s, court_v], dtype=np.uint8)
    # Classify for logging only
    if 90 <= court_h <= 125:
        court_type = "blue"
    elif 35 <= court_h <= 85:
        court_type = "green"
    elif court_h <= 25 or court_h >= 170:
        court_type = "clay"
    else:
        court_type = "unknown"
    print(f"  [Stage 2+3] Saturation Agrawal: {np.count_nonzero(line_mask)} line pixels, "
          f"tophat_thresh={eff_tophat}, "
          f"court={court_type}(H={court_h} S={court_s} V={court_v}) ({time.time()-t2:.3f}s)")

    # --- Stage 4: Two-pass Hough with baseline-anchored spatial masking ---
    t4 = time.time()
    raw_lines, baseline_extent, spatial_mask, bl_angle = detect_lines_near_half(line_mask)
    n_raw = 0 if raw_lines is None else len(raw_lines)

    if baseline_extent is not None:
        bl_x_left, bl_x_right = baseline_extent
        masked_px = np.count_nonzero(cv2.bitwise_and(line_mask, spatial_mask))
        total_px = np.count_nonzero(line_mask)
        print(f"  [Stage 4] Baseline anchor: x=[{bl_x_left},{bl_x_right}], "
              f"spatial mask kept {masked_px}/{total_px} line pixels")
    else:
        print(f"  [Stage 4] No baseline found — using unmasked detection")
    horizontal, vertical = classify_lines_courtside(raw_lines, baseline_angle=bl_angle)
    horizontal = filter_short_merged(merge_collinear_segments(horizontal))
    vertical = filter_short_merged(merge_collinear_segments(vertical))
    print(f"  [Stage 4] Hough: {n_raw} raw → {len(horizontal)}H + {len(vertical)}V "
          f"merged ({time.time()-t4:.2f}s)")

    # --- Stage 4b: Multi-candidate scoring (Agrawal Section 3.4) ---
    t4b = time.time()
    nh_h, nh_w = near_half.shape[:2]

    # Helper: run a candidate line mask through the full pipeline to homography.
    # Each candidate computes its own baseline extent so a weak saturation
    # mask cannot clip a stronger local-contrast candidate.
    def _run_candidate(cand_line_mask, cand_label):
        """Returns (H_near, near_kps, identified, horizontal, vertical, raw_lines, line_mask, score, reproj_err)."""
        c_raw, c_bl_ext, c_sm, c_bl_angle = detect_lines_near_half(cand_line_mask)
        if c_bl_ext is not None:
            cand_line_mask = cv2.bitwise_and(cand_line_mask, c_sm)
            c_raw, _, _, c_bl_angle2 = detect_lines_near_half(cand_line_mask)
            if c_bl_angle2 is not None:
                c_bl_angle = c_bl_angle2
            bl_span = c_bl_ext[1] - c_bl_ext[0]
            print(f"  [Stage 4b] {cand_label}: baseline x=[{c_bl_ext[0]},{c_bl_ext[1]}], "
                  f"span={bl_span}px ({bl_span/nh_w*100:.0f}%)")
        c_h, c_v = classify_lines_courtside(c_raw, baseline_angle=c_bl_angle)
        c_h = filter_short_merged(merge_collinear_segments(c_h))
        c_v_merged = merge_collinear_segments(c_v)
        c_v = filter_short_merged(c_v_merged)
        c_identified = identify_near_half_lines(c_h, c_v, nh_h, nh_w, pre_filter_verticals=c_v_merged)
        c_near_kps = compute_near_half_keypoints(c_identified, y_offset, c_v, nh_h)
        c_near_kps = validate_near_half_keypoints(c_near_kps)
        c_H, c_near_kps = compute_near_half_homography(c_near_kps)
        c_score = score_court_detection(near_half, c_H, y_offset, cand_line_mask) if c_H is not None else 0
        c_err = reprojection_error(c_near_kps, c_H) if c_H is not None else float('inf')
        n = 0 if c_raw is None else len(c_raw)
        print(f"  [Stage 4b] {cand_label}: {len(c_h)}H+{len(c_v)}V, "
              f"{len(c_near_kps)} kps, score={c_score}, reproj={c_err:.1f}px")
        return c_H, c_near_kps, c_identified, c_h, c_v, c_raw, cand_line_mask, c_score, c_err

    candidates = []

    # Candidate 1: Local-contrast AND filter (primary)
    lc_line_mask, lc_court_mask, _, lc_thresh_v, lc_thresh_s = agrawal_local_contrast_line_filter(near_half)
    lc_line_px = np.count_nonzero(lc_line_mask)
    print(f"  [Stage 2+3] LocalContrast: {lc_line_px} line pixels, "
          f"thresh_v={lc_thresh_v}, thresh_s={lc_thresh_s}")
    c1 = _run_candidate(lc_line_mask, "LocalContrast")
    candidates.append(("local_contrast", c1, lc_court_mask, court_color, "local_contrast"))

    # Candidate 2: CLAHE-enhanced filter (night / low-contrast scenes)
    cl_line_mask, cl_court_mask, _, cl_thresh_v = agrawal_clahe_line_filter(near_half)
    cl_line_px = np.count_nonzero(cl_line_mask)
    print(f"  [Stage 2+3] CLAHE: {cl_line_px} line pixels, "
          f"thresh_v={cl_thresh_v}")
    c2 = _run_candidate(cl_line_mask, "CLAHE")
    candidates.append(("clahe", c2, cl_court_mask, court_color, "clahe"))

    # Pick best candidate: among those with score > 0, choose lowest reproj error
    valid_candidates = [(name, cand, cm, cc, mode) for name, cand, cm, cc, mode in candidates
                        if cand[7] > 0]  # index 7 = score
    if valid_candidates:
        # Best = highest score (template-line overlap, index 7).
        # Reproj error is degenerate (fewer KPs → lower error), so use score.
        best_name, best_cand, best_court_mask, best_court_color, best_court_mode = max(
            valid_candidates, key=lambda x: x[1][7])  # index 7 = score
        best_H, best_near_kps, best_identified, best_h, best_v, best_raw, best_lm, best_score, best_err = best_cand

        print(f"  [Stage 4b] Winner: {best_name} (score={best_score}, reproj={best_err:.1f}px, "
              f"mode={best_court_mode}) ({time.time()-t4b:.3f}s)")
        horizontal, vertical = best_h, best_v
        raw_lines = best_raw
        n_raw = 0 if raw_lines is None else len(raw_lines)
        line_mask = best_lm
        court_color_mask = best_court_mask
        court_color = best_court_color
        winning_court_mode = best_court_mode
        identified = best_identified
    else:
        # No candidate scored > 0 — use best LC candidate for display
        best_lc = max(candidates, key=lambda x: len(x[1][1]))  # index 1 = near_kps
        lc_cand = best_lc[1]
        horizontal, vertical = lc_cand[3], lc_cand[4]
        line_mask = lc_cand[6]
        identified = lc_cand[2]
        winning_court_mode = "local_contrast"

    # --- Stage 5: Identify near-half lines (already computed in 4b) ---
    t5 = time.time()
    id_count = sum(1 for v in identified.values() if v is not None)
    print(f"  [Stage 5] Identified {id_count}/7 court features:")
    for name, seg in identified.items():
        if seg is not None:
            print(f"    {name}: ({seg[0]},{seg[1]})->({seg[2]},{seg[3]})")

    # --- Stage 5b: Fallback service line detection ---
    used_sv_fallback = False
    if identified.get("near_service") is None and identified.get("near_baseline") is not None:
        fb_service = fallback_service_line_detection(line_mask, identified, nh_h, nh_w)
        if fb_service is not None:
            identified["near_service"] = fb_service
            used_sv_fallback = True
            print(f"  [Stage 5b] Fallback service line: "
                  f"({fb_service[0]:.0f},{fb_service[1]:.0f})->"
                  f"({fb_service[2]:.0f},{fb_service[3]:.0f})")
        else:
            print(f"  [Stage 5b] Fallback service line: not found")

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
    H_near, near_kps = compute_near_half_homography(near_kps)
    H_full = None
    H_full_raw = None
    all_kps = list(near_kps)

    if H_near is not None:
        print(f"  [Stage 7] --- Initial H (from keypoints only) ---")
        debug_per_line_chamfer(H_near, line_mask, y_offset, label="InitH")

        # Edge-align with keypoint regularization (RANSAC-cleaned KPs are trustworthy)
        H_near = refine_homography_edge_align(
            H_near, line_mask, near_kps, y_offset,
            lambda_reg=0.3, max_iter=150)

        # Debug: per-line chamfer AFTER edge-align
        print(f"  [Stage 7] --- After edge-align (lambda_reg=0.3) ---")
        debug_per_line_chamfer(H_near, line_mask, y_offset, label="PostEdge")

        # Update keypoints from edge-aligned H so PnL uses corrected positions
        try:
            H_inv = np.linalg.inv(H_near)
            meter_pts = np.array([REFERENCE_KPS_METERS[idx] for idx, _ in near_kps],
                                 dtype=np.float64).reshape(-1, 1, 2)
            updated_px = cv2.perspectiveTransform(meter_pts, H_inv).reshape(-1, 2)
            near_kps = [(idx, (float(updated_px[i, 0]), float(updated_px[i, 1])))
                        for i, (idx, _) in enumerate(near_kps)]
            print(f"  [Stage 7] Updated {len(near_kps)} KPs from edge-aligned H")
        except (np.linalg.LinAlgError, cv2.error):
            print(f"  [Stage 7] Could not update KPs (singular H), keeping originals")

        H_full, all_kps = extend_to_full_court(H_near, near_kps)
        H_full_raw = H_full  # keep raw for comparison
        near_err = reprojection_error(near_kps, H_full) if H_full is not None else float("inf")
        full_err = reprojection_error(all_kps, H_full) if H_full is not None else float("inf")
        print(f"  [Stage 7] Homography computed: near_err={near_err:.1f}px, "
              f"full_err={full_err:.1f}px ({time.time()-t7:.2f}s)")
        print(f"    Total keypoints (near+far): {len(all_kps)}")

        # --- Stage 7 safety: revert fallback if it broke the homography ---
        if H_full is None and used_sv_fallback:
            print(f"  [Stage 7] Fallback service line broke homography — reverting")
            identified["near_service"] = None
            used_sv_fallback = False
            # Re-run Stage 6+7 without the fallback service line
            near_kps = compute_near_half_keypoints(identified, y_offset, vertical, nh_h)
            near_kps = validate_near_half_keypoints(near_kps)
            H_near, near_kps = compute_near_half_homography(near_kps)
            if H_near is not None:
                H_near = refine_homography_edge_align(
                    H_near, line_mask, near_kps, y_offset,
                    lambda_reg=0.3, max_iter=150)
                try:
                    H_inv = np.linalg.inv(H_near)
                    meter_pts = np.array([REFERENCE_KPS_METERS[idx] for idx, _ in near_kps],
                                         dtype=np.float64).reshape(-1, 1, 2)
                    updated_px = cv2.perspectiveTransform(meter_pts, H_inv).reshape(-1, 2)
                    near_kps = [(idx, (float(updated_px[i, 0]), float(updated_px[i, 1])))
                                for i, (idx, _) in enumerate(near_kps)]
                except (np.linalg.LinAlgError, cv2.error):
                    pass
                H_full, all_kps = extend_to_full_court(H_near, near_kps)
                H_full_raw = H_full
                near_err = reprojection_error(near_kps, H_full) if H_full is not None else float("inf")
                full_err = reprojection_error(all_kps, H_full) if H_full is not None else float("inf")
                print(f"  [Stage 7] Reverted: near_err={near_err:.1f}px, "
                      f"full_err={full_err:.1f}px")

        # Kalman smoothing
        if kalman is not None and H_full is not None:
            H_full = kalman.update(H_full, near_err, n_kps=len(near_kps),
                                   frame_idx=frame_idx)
            smoothed_err = reprojection_error(near_kps, H_full)
            print(f"  [Kalman] Smoothed H: reproj {near_err:.2f} → {smoothed_err:.2f}px")
    else:
        print(f"  [Stage 7] Homography FAILED — only {len(near_kps)} keypoints "
              f"(need ≥4)")

        # If no homography, try Kalman prediction
        if H_full is None and kalman is not None:
            H_pred = kalman.predict()
            if H_pred is not None:
                H_full = H_pred
                print(f"  [Kalman] Using predicted H (coasting)")

    # Compute raw reprojection error before Kalman for reporting
    raw_near_err = reprojection_error(near_kps, H_full_raw) if H_full_raw is not None else float("inf")
    smoothed_near_err = reprojection_error(near_kps, H_full) if H_full is not None else float("inf")

    # --- Stage 8: Metrics ---
    metrics = compute_metrics(near_kps, all_kps, H_full)
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
        court_mode=winning_court_mode,
    )

    results.update({
        "net_y": net_y,
        "y_offset": y_offset,
        "court_color": court_color.tolist(),
        "court_type": court_type,
        "court_mode": winning_court_mode,
        "horizontal_segments": [seg.copy() for seg in horizontal],
        "vertical_segments": [seg.copy() for seg in vertical],
        "identified_lines": {
            name: (seg.copy() if isinstance(seg, np.ndarray) else None)
            for name, seg in identified.items()
        },
        "n_raw_lines": n_raw,
        "n_horizontal": len(horizontal),
        "n_vertical": len(vertical),
        "n_identified": id_count,
        "near_kps": [(k, list(p)) for k, p in near_kps],
        "all_kps": [(k, list(p)) for k, p in all_kps],
        "metrics": metrics,
        "near_half": near_half,
        "line_mask": line_mask,
        "H_near": H_near,
        "H_full": H_full,
        "composite": composite,
        "total_time": total_time,
        "raw_near_err": raw_near_err,
        "smoothed_near_err": smoothed_near_err,
    })

    return results

