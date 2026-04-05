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
from src.features.court_detect.shadow_removal import ShadowRemover

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

# Saturation-based Agrawal (color-agnostic)
# Saturation-based Agrawal (color-agnostic)
SAT_COURT_THRESHOLD = 50    # S > this = chromatic surface for neighbor counting
SAT_LINE_S_MAX = 50         # S < this = line candidate (includes anti-aliased edges)
SAT_LINE_V_MIN = 150        # V > this = bright (line candidate)
TOPHAT_KERNEL_SIZE = 23      # Morphological kernel for white top-hat (> line width ~5-15px)
TOPHAT_THRESHOLD = 30        # Minimum top-hat response to be a line candidate
HOUGH_THRESHOLD = 30
HOUGH_MIN_LENGTH = 80
HOUGH_MAX_GAP = 40
ENABLE_SHADOW_REMOVAL = True


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


def estimate_net_y_from_cnn(
    frame: np.ndarray,
    court_detector: CourtDetector,
    frame_number: int = 0,
    net_detector=None,
) -> tuple[int, CourtDetectionResult, str]:
    """Estimate net y-position using YOLO net detector (primary) or CNN (fallback).

    Returns (net_y, cnn_result, source, crop_margin) where:
    - net_y: the net bottom position (crop reference)
    - crop_margin: how many pixels above net_y to include in crop
    """
    # Always run CNN for keypoints (used later in pipeline for fallback)
    result = court_detector.detect_frame(frame, frame_number=frame_number)

    # Primary: YOLO net detector
    if net_detector is not None:
        yolo_result = estimate_net_y_from_yolo(frame, net_detector)
        if yolo_result is not None:
            net_top, net_bottom = yolo_result
            # Crop at net bottom — the true half-court dividing line.
            # Use 0.3× net height as small margin: just enough for the
            # service line area at the net, without including far-court pixels.
            net_height = net_bottom - net_top
            crop_margin = int(net_height * 0.3)
            net_y = max(0, min(net_bottom, frame.shape[0] - 100))
            return net_y, result, f"yolo(top={net_top},bot={net_bottom})", crop_margin

    # Fallback: CNN keypoints (net_y ≈ net top, uses default margin)
    kps = result.keypoints
    kp12_y = kps[12][1] if len(kps) > 12 and kps[12][0] is not None else None
    kp13_y = kps[13][1] if len(kps) > 13 and kps[13][0] is not None else None

    if kp12_y is not None and kp13_y is not None:
        net_y = int(kp12_y - 30)
        source = "cnn_kp12_kp13"
    elif kp12_y is not None:
        net_y = int(kp12_y - 30)
        source = "cnn_kp12"
    else:
        net_y = estimate_net_y_heuristic(frame)
        source = "heuristic"

    net_y = max(0, min(net_y, frame.shape[0] - 100))
    return net_y, result, source, 20  # default margin for CNN path


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


def detect_court_color_simple(
    frame_bgr: np.ndarray,
    n_samples: int = 1000,
) -> tuple[list[np.ndarray], str, str]:
    """Detect dominant court color(s) using simple mode (Agrawal Section 3.3).

    Samples N random points from the center ROI, quantizes into HSV bins,
    and returns the median HSV of the top chromatic bin(s) as the court color(s).

    Returns (colors, court_type, court_mode) where:
    - colors: list of 1 or 2 HSV arrays (the court surface colors)
    - court_type: "blue", "green", "clay", "unknown" (based on dominant color)
    - court_mode: "single" or "multi"
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h_img, w_img = hsv.shape[:2]

    # Sample from center ROI: y=30-80%, x=25-75%
    y_lo = int(h_img * 0.30)
    y_hi = int(h_img * 0.80)
    x_lo = int(w_img * 0.25)
    x_hi = int(w_img * 0.75)
    ys, xs = np.mgrid[y_lo:y_hi, x_lo:x_hi]
    ys = ys.ravel()
    xs = xs.ravel()

    if len(ys) == 0:
        return [np.array([0, 0, 0], dtype=np.uint8)], "unknown", "single"

    n = min(n_samples, len(ys))
    indices = np.random.default_rng(42).choice(len(ys), size=n, replace=False)
    sampled_hsv = hsv[ys[indices], xs[indices]]  # (n, 3)

    # Quantize into bins: H→18 bins (10 each), S→5 bins (51 each), V→5 bins (51 each)
    h_bins = np.clip(sampled_hsv[:, 0].astype(int) // 10, 0, 17)
    s_bins = np.clip(sampled_hsv[:, 1].astype(int) // 51, 0, 4)
    v_bins = np.clip(sampled_hsv[:, 2].astype(int) // 51, 0, 4)

    # Combine into a single bin index
    bin_ids = h_bins * 25 + s_bins * 5 + v_bins
    counts = np.bincount(bin_ids, minlength=18 * 5 * 5)

    # Find top-2 chromatic bins (median saturation > 30)
    n_bins = len(counts)
    chromatic_bins = []
    for b in range(n_bins):
        if counts[b] == 0:
            continue
        bin_mask = bin_ids == b
        bin_pixels = sampled_hsv[bin_mask]
        median_s = int(np.median(bin_pixels[:, 1]))
        if median_s > 30:
            chromatic_bins.append((b, int(counts[b]), bin_pixels))

    # Sort chromatic bins by count descending
    chromatic_bins.sort(key=lambda x: x[1], reverse=True)

    if len(chromatic_bins) == 0:
        return [np.array([0, 0, 0], dtype=np.uint8)], "unknown", "single"

    # bin1 = most common chromatic bin
    bin1_id, bin1_count, bin1_pixels = chromatic_bins[0]
    color1_h = int(np.median(bin1_pixels[:, 0]))
    color1_s = int(np.median(bin1_pixels[:, 1]))
    color1_v = int(np.median(bin1_pixels[:, 2]))
    color1 = np.array([color1_h, color1_s, color1_v], dtype=np.uint8)

    # Determine single vs multi-color
    colors = [color1]
    court_mode = "single"
    if len(chromatic_bins) >= 2:
        bin2_id, bin2_count, bin2_pixels = chromatic_bins[1]
        if bin2_count >= 0.25 * bin1_count:
            color2_h = int(np.median(bin2_pixels[:, 0]))
            color2_s = int(np.median(bin2_pixels[:, 1]))
            color2_v = int(np.median(bin2_pixels[:, 2]))
            color2 = np.array([color2_h, color2_s, color2_v], dtype=np.uint8)
            colors.append(color2)
            court_mode = "multi"

    # Auto-detect court type from dominant color (same hue ranges as kmeans version)
    court_h = color1_h
    if 90 <= court_h <= 125:
        court_type = "blue"
    elif 35 <= court_h <= 85:
        court_type = "green"
    elif court_h <= 25 or court_h >= 170:
        court_type = "clay"
    else:
        court_type = "unknown"

    return colors, court_type, court_mode


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


def agrawal_paper_line_filter(
    frame_bgr: np.ndarray,
    court_colors: list[np.ndarray],
    neighborhood: int = NEIGHBORHOOD_SIZE,
    min_court_neighbors: int = MIN_COURT_NEIGHBORS,
    h_thresh: int = COLOR_MATCH_H_THRESH,
    s_thresh: int = COLOR_MATCH_S_THRESH,
    v_thresh: int = COLOR_MATCH_V_THRESH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paper-faithful Agrawal filter (Section 3.3): NO brightness/saturation gates.

    Line pixel = NOT court-colored AND ≥4 court-colored neighbors in 7×7 window.
    No V > 150 or S < 80 gate — matches the paper exactly.

    Returns (line_mask, court_mask, hsv).
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Build court-color binary mask as union of all color masks
    court_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for color in court_colors:
        mask = build_court_color_mask(frame_bgr, color, h_thresh, s_thresh, v_thresh, precomputed_hsv=hsv)
        court_mask = cv2.bitwise_or(court_mask, mask)

    # Count court-color neighbors via convolution
    court_01 = (court_mask > 0).astype(np.float32)
    kernel = np.ones((neighborhood, neighborhood), dtype=np.float32)
    neighbor_count = cv2.filter2D(court_01, cv2.CV_32F, kernel)

    # Line pixel = NOT court-color AND enough court-color neighbors
    not_court = (court_01 == 0)
    has_neighbors = (neighbor_count >= min_court_neighbors)

    line_mask = (not_court & has_neighbors).astype(np.uint8) * 255
    return line_mask, court_mask, hsv


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


def _find_baseline_extent(
    lines: np.ndarray | None,
    frame_height: int,
    frame_width: int,
) -> tuple[int, int] | None:
    """Find the court's horizontal extent from baseline-region H lines.

    Instead of using a single Hough segment (which may be partial),
    collects ALL near-horizontal segments in the baseline region and
    takes their full x-extent. This handles cases where Hough breaks
    the baseline into 2-3 shorter segments.

    Returns (x_left, x_right) or None if no baseline found.
    """
    if lines is None:
        return None

    # Collect all H segments in the bottom portion of the crop
    h_segs = []
    for line in lines:
        seg = line[0] if line.ndim == 2 else line
        x1, y1, x2, y2 = seg
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
        avg_y = (y1 + y2) / 2.0
        length = np.hypot(x2 - x1, y2 - y1)

        is_horizontal = angle < 15 or angle > 165
        is_bottom = avg_y > frame_height * 0.45
        is_long = length > frame_width * 0.10

        if is_horizontal and is_bottom and is_long:
            h_segs.append(seg.copy())

    if not h_segs:
        return None

    # Find the longest segment to anchor the baseline y-position
    longest = max(h_segs, key=lambda s: np.hypot(s[2]-s[0], s[3]-s[1]))
    baseline_y = (longest[1] + longest[3]) / 2.0

    # Collect all H segments near the same y-level (within ±30px)
    near_baseline = []
    for seg in h_segs:
        seg_y = (seg[1] + seg[3]) / 2.0
        if abs(seg_y - baseline_y) < 30:
            near_baseline.append(seg)

    # Full x-extent across all co-linear baseline segments
    x_left = min(min(s[0], s[2]) for s in near_baseline)
    x_right = max(max(s[0], s[2]) for s in near_baseline)

    return int(x_left), int(x_right)


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
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
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

    Returns (lines, baseline_extent, spatial_mask).
    """
    h, w = line_mask.shape[:2]

    # Light morphological cleanup — close small gaps in line pixels.
    # NOTE: Only CLOSE, no OPEN. The horizontal OPEN kernel (1,3) erodes
    # thin near-vertical features (sidelines) when using the saturation-based
    # filter. The top-hat pre-filter already provides clean output.
    kernel_close = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # --- Pass 1: Find the baseline extent ---
    pass1_lines = cv2.HoughLinesP(
        cleaned, 1, np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_length,
        maxLineGap=max_gap,
    )

    baseline_extent = _find_baseline_extent(pass1_lines, h, w)

    if baseline_extent is None:
        # No baseline found — fall back to unmasked detection
        return pass1_lines, None, None

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

    return pass2_lines, baseline_extent, spatial_mask


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
        # Baseline-relative min-length: sidelines span from baseline to net,
        # so their length ≈ bl_y. Use 0.85× for margin on partial detections.
        # Fallback to crop_height * 0.40 if no baseline was detected.
        if bl_y is not None:
            min_vline_length = bl_y * 0.85
        else:
            min_vline_length = frame_height * 0.40
        scored_v = []
        for seg in vertical:
            length = np.hypot(seg[2] - seg[0], seg[3] - seg[1])
            if length < min_vline_length:
                continue  # skip short fragments
            x_at_ref = _line_x_at_y(seg, ref_y)
            if x_at_ref is not None:
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
    court_mode: str = "single",
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
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Panel 2: Near-half crop
    p2 = resize(near_half)
    cv2.putText(p2, "2. Near-half crop", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Panel 3: Court color mask
    p3 = resize(court_color_mask)
    ch, cs, cv_val = court_color_hsv
    mode_label = "MULTI" if court_mode == "multi" else "SINGLE"
    cv2.putText(p3, f"3. Court mask [{mode_label}] H={ch} S={cs} V={cv_val}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # Panel 4: Line filter mask
    p4 = resize(line_mask)
    cv2.putText(p4, "4. Agrawal line mask", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

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
# Court detection scoring (Agrawal Section 3.4)
# ===================================================================

def score_court_detection(
    near_half: np.ndarray,
    H_near: np.ndarray,
    y_offset: int = 0,
) -> int:
    """Score a court detection by projecting court template lines and counting
    overlap with bright pixels in the original image (Agrawal Section 3.4).

    H_near maps image pixels → court meters.  We need the inverse
    (meters → pixels) to project template lines onto the image.

    Returns score = number of projected court pixels that overlap with
    bright/white pixels in the original frame.
    """
    if H_near is None:
        return 0

    nh_h, nh_w = near_half.shape[:2]

    # Invert homography: pixel→meter to meter→pixel
    try:
        H_inv = np.linalg.inv(H_near)
    except np.linalg.LinAlgError:
        return 0

    # Near-half court lines in meter coords (from NEAR_HALF_KPS)
    court_lines_m = [
        # Near baseline: KP2 → KP3
        ((-5.485, 11.885), (5.485, 11.885)),
        # Near service line: KP10 → KP11
        ((-4.115, 6.4), (4.115, 6.4)),
        # Left doubles sideline: KP2 down toward net
        ((-5.485, 11.885), (-5.485, 0.0)),
        # Right doubles sideline: KP3 down toward net
        ((5.485, 11.885), (5.485, 0.0)),
        # Left singles sideline: KP5 → KP10
        ((-4.115, 11.885), (-4.115, 6.4)),
        # Right singles sideline: KP7 → KP11
        ((4.115, 11.885), (4.115, 6.4)),
        # Center service line: KP13 toward net
        ((0.0, 6.4), (0.0, 0.0)),
    ]

    # Draw projected court lines onto a blank mask
    proj_mask = np.zeros((nh_h, nh_w), dtype=np.uint8)
    n_samples_per_line = 50

    for (mx1, my1), (mx2, my2) in court_lines_m:
        ts = np.linspace(0.0, 1.0, n_samples_per_line)
        meter_pts = np.column_stack([
            mx1 + ts * (mx2 - mx1),
            my1 + ts * (my2 - my1),
        ]).reshape(-1, 1, 2).astype(np.float64)

        try:
            pixel_pts = cv2.perspectiveTransform(meter_pts, H_inv).reshape(-1, 2)
        except cv2.error:
            continue

        # Adjust for y_offset: H_near maps full-frame pixels, but near_half
        # is cropped starting at y_offset
        pixel_pts[:, 1] -= y_offset

        # Draw line segments between consecutive projected points
        for i in range(len(pixel_pts) - 1):
            pt1 = (int(round(pixel_pts[i, 0])), int(round(pixel_pts[i, 1])))
            pt2 = (int(round(pixel_pts[i + 1, 0])), int(round(pixel_pts[i + 1, 1])))
            cv2.line(proj_mask, pt1, pt2, 255, thickness=3)

    # Create bright-pixel mask from the original near_half image
    hsv = cv2.cvtColor(near_half, cv2.COLOR_BGR2HSV)
    bright_mask = ((hsv[:, :, 2] > 170) & (hsv[:, :, 1] < 60)).astype(np.uint8) * 255

    # Score = overlap between projected court lines and bright pixels
    overlap = cv2.bitwise_and(proj_mask, bright_mask)
    score = int(np.count_nonzero(overlap))

    return score


# ===================================================================
# Main pipeline
# ===================================================================

def run_agrawal_pipeline(
    frame: np.ndarray,
    frame_idx: int,
    court_detector: CourtDetector,
    kalman: HomographyKalmanFilter | None = None,
    shadow_remover: "ShadowRemover | None" = None,
    net_detector=None,
) -> dict:
    """Run the full Agrawal-inspired pipeline on a single frame."""
    t0 = time.time()
    results: dict = {"frame_idx": frame_idx}

    # --- Stage 1: Net detection & crop ---
    t1 = time.time()
    net_y, cnn_result, net_source, crop_margin = estimate_net_y_from_cnn(
        frame, court_detector, frame_idx, net_detector=net_detector)
    cnn_kps = cnn_result.keypoints
    cnn_detected = sum(1 for p in cnn_kps if p[0] is not None)
    print(f"  [Stage 1] Net y={net_y} (source={net_source}), "
          f"margin={crop_margin}, CNN detected {cnn_detected}/14 keypoints "
          f"({time.time()-t1:.2f}s)")

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
            diag_path = OUTPUT_DIR / f"shadow_removal_{VIDEO_PATH.stem}_frame_{frame_idx}.jpg"
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
    raw_lines, baseline_extent, spatial_mask = detect_lines_near_half(line_mask)
    n_raw = 0 if raw_lines is None else len(raw_lines)
    if baseline_extent is not None:
        bl_x_left, bl_x_right = baseline_extent
        masked_px = np.count_nonzero(cv2.bitwise_and(line_mask, spatial_mask))
        total_px = np.count_nonzero(line_mask)
        print(f"  [Stage 4] Baseline anchor: x=[{bl_x_left},{bl_x_right}], "
              f"spatial mask kept {masked_px}/{total_px} line pixels")
    else:
        print(f"  [Stage 4] No baseline found — using unmasked detection")
    horizontal, vertical = classify_lines_courtside(raw_lines)
    horizontal = merge_collinear_segments(horizontal)
    vertical = merge_collinear_segments(vertical)
    print(f"  [Stage 4] Hough: {n_raw} raw → {len(horizontal)}H + {len(vertical)}V "
          f"merged ({time.time()-t4:.2f}s)")

    # --- Stage 4b: Multi-candidate scoring (Agrawal Section 3.4) ---
    # Run three candidate filters through the full pipeline, score each
    # homography by projecting court lines and counting bright-pixel overlap,
    # then pick the best.
    t4b = time.time()
    nh_h, nh_w = near_half.shape[:2]

    # Helper: run a candidate line mask through the full pipeline to homography
    def _run_candidate(cand_line_mask, cand_label):
        """Returns (H_near, near_kps, identified, horizontal, vertical, raw_lines, line_mask, score)."""
        if baseline_extent is not None:
            cand_line_mask = cv2.bitwise_and(cand_line_mask, spatial_mask)
        c_raw = detect_lines_near_half(cand_line_mask)[0]
        c_h, c_v = classify_lines_courtside(c_raw)
        c_h = merge_collinear_segments(c_h)
        c_v = merge_collinear_segments(c_v)
        c_identified = identify_near_half_lines(c_h, c_v, nh_h, nh_w)
        c_near_kps = compute_near_half_keypoints(c_identified, y_offset, c_v, nh_h)
        c_near_kps = validate_near_half_keypoints(c_near_kps)
        c_H = compute_near_half_homography(c_near_kps)
        c_score = score_court_detection(near_half, c_H, y_offset) if c_H is not None else 0
        n = 0 if c_raw is None else len(c_raw)
        print(f"  [Stage 4b] {cand_label}: {len(c_h)}H+{len(c_v)}V, "
              f"{len(c_near_kps)} kps, score={c_score}")
        return c_H, c_near_kps, c_identified, c_h, c_v, c_raw, cand_line_mask, c_score

    candidates = []

    # Candidate 1: Paper-faithful filter (simple color, no brightness gate)
    paper_colors, paper_court_type, paper_court_mode = detect_court_color_simple(near_half, n_samples=1000)
    paper_line_mask, paper_court_mask, _ = agrawal_paper_line_filter(near_half, paper_colors)
    c1 = _run_candidate(paper_line_mask, f"Paper(H={paper_colors[0][0]})")
    candidates.append(("paper", c1, paper_court_mask, paper_colors[0], paper_court_mode))

    # Candidate 2: Saturation-based (already computed above in Stage 2+3)
    c2 = _run_candidate(line_mask.copy(), "Saturation")
    candidates.append(("saturation", c2, court_color_mask, court_color, "single"))

    # Candidate 3: Hue-based K-means (original agrawal_court_line_filter)
    kmeans_color, kmeans_court_type = detect_court_color_kmeans(near_half, n_samples=N_COLOR_SAMPLES)
    kmeans_line_mask = agrawal_court_line_filter(near_half, kmeans_color)
    kmeans_court_mask = build_court_color_mask(near_half, kmeans_color)
    c3 = _run_candidate(kmeans_line_mask, f"KMeans(H={kmeans_color[0]})")
    candidates.append(("kmeans", c3, kmeans_court_mask, kmeans_color, "single"))

    # Pick candidate with highest score
    best_name, best_cand, best_court_mask, best_court_color, best_court_mode = max(
        candidates, key=lambda x: x[1][7])  # index 7 = score
    best_H, best_near_kps, best_identified, best_h, best_v, best_raw, best_lm, best_score = best_cand

    if best_score > 0:
        print(f"  [Stage 4b] Winner: {best_name} (score={best_score}, mode={best_court_mode}) ({time.time()-t4b:.3f}s)")
        horizontal, vertical = best_h, best_v
        raw_lines = best_raw
        n_raw = 0 if raw_lines is None else len(raw_lines)
        line_mask = best_lm
        court_color_mask = best_court_mask
        court_color = best_court_color
        winning_court_mode = best_court_mode
        identified = best_identified
    else:
        # No candidate produced a valid homography — fall through with
        # the saturation result from Stage 4 (already in horizontal/vertical)
        print(f"  [Stage 4b] No candidate scored > 0, using saturation fallback "
              f"({time.time()-t4b:.3f}s)")
        identified = identify_near_half_lines(horizontal, vertical, nh_h, nh_w)
        winning_court_mode = "single"

    # --- Stage 5: Identify near-half lines (already computed in 4b) ---
    t5 = time.time()
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
        court_mode=winning_court_mode,
    )

    results.update({
        "net_y": net_y,
        "court_color": court_color.tolist(),
        "court_type": court_type,
        "court_mode": winning_court_mode,
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

    # Initialize YOLO net detector
    net_detector = None
    net_weights = PROJECT_ROOT / "weights" / "net_detector.pt"
    if net_weights.exists():
        from ultralytics import YOLO
        net_detector = YOLO(str(net_weights))
        print(f"  YOLO net detector loaded: {net_weights}")
    else:
        print(f"  YOLO net detector not found at {net_weights}, using CNN fallback")

    # Initialize Kalman smoother for temporal consistency
    kalman = HomographyKalmanFilter()

    # Initialize shadow remover (optional)
    shadow_remover = ShadowRemover() if ENABLE_SHADOW_REMOVAL else None

    # Derive video stem for unique output filenames
    video_stem = VIDEO_PATH.stem

    # Process each frame
    all_results = []
    for idx in FRAME_INDICES:
        if idx not in frames:
            continue
        print(f"\n{'='*70}")
        print(f"Processing frame {idx}")
        print(f"{'='*70}")

        result = run_agrawal_pipeline(
            frames[idx], idx, court_detector, kalman, shadow_remover,
            net_detector=net_detector)

        # Save diagnostic composite
        out_path = OUTPUT_DIR / f"agrawal_{video_stem}_frame_{idx}.jpg"
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

    report_path = OUTPUT_DIR / f"agrawal_{video_stem}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
