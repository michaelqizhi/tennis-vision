"""Ball tracking detector — runs TrackNet V4 inference on video frames.

TrackNet V4 uses a motion-attention-enhanced U-Net. Key differences from V2:
  - Input resolution: 512×288 (vs 640×360 for V2)
  - Output: 3-channel sigmoid heatmap (one per input frame)
  - Motion Prompt module extracts attention from frame differences
  - Better handling of fast-moving objects via learnable motion attention

This module provides ``BallTrackerV4`` which mirrors the V2 ``BallTracker``
API but uses the V4 architecture and its own postprocessing tuned for the
sigmoid output space.
"""

from __future__ import annotations

import math
from collections import deque
from itertools import groupby
from typing import Callable, Iterator

import cv2
import numpy as np
import torch
from scipy.spatial import distance
from tqdm import tqdm

from src.config import Config, get_config
from src.features.ball_tracking.detector import BallDetection
from src.models.tracknetv4 import TrackNetV4Model, load_tracknet_v4, INPUT_W, INPUT_H


# V4 post-processing constants.
# V4 outputs sigmoid [0, 1] heatmaps — these thresholds are calibrated for
# that output space (vs V2's 256-class argmax → [0, 1] normalized heatmap).
# Matches the reference implementation: threshold 0.5, largest contour, no
# area or peak filtering (the model's sigmoid activations are already sparse).
_DETECTION_THRESHOLD = 0.5


def postprocess_v4(
    heatmap: np.ndarray,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    threshold: float = _DETECTION_THRESHOLD,
) -> tuple[float | None, float | None, float]:
    """Extract ball (x, y, confidence) from a V4 sigmoid heatmap.

    V4 outputs a [0, 1] sigmoid heatmap. We binarize at the threshold and
    pick the largest contour — matching the reference implementation which
    uses no blob-area cap or per-component peak filter.

    Args:
        heatmap: Sigmoid heatmap, shape (H, W), values in [0, 1].
        scale_x: Scale factor to map x back to original video resolution.
        scale_y: Scale factor to map y back to original video resolution.
        threshold: Activation threshold — pixels below this are ignored.

    Returns:
        Tuple of (x, y, confidence) in original video coordinates,
        or (None, None, 0.0) if no ball detected.
    """
    peak = float(heatmap.max())
    if peak < threshold:
        return None, None, 0.0

    binary = (heatmap > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0.0

    # Pick the largest contour (matches reference postprocessing)
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, None, 0.0

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    x = float(cx * scale_x)
    y = float(cy * scale_y)
    return x, y, peak


class BallTrackerV4:
    """Track tennis ball positions across video frames using TrackNet V4.

    Drop-in replacement for BallTracker (V2) with the V4 motion-attention
    architecture. Returns the same ``BallDetection`` dataclass.

    Args:
        config: Project configuration. Uses default if not provided.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._model: TrackNetV4Model | None = None

    def _ensure_model(self) -> TrackNetV4Model:
        """Lazily load the V4 model."""
        if self._model is None:
            weights = self.config.resolve_path(self.config.model.tracknet_v4_weights)
            self._model = load_tracknet_v4(
                str(weights), self.config.device, auto_download=True,
            )
        return self._model

    def detect(
        self,
        frames: list[np.ndarray],
        interpolate: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[BallDetection]:
        """Run ball detection on a list of video frames.

        Args:
            frames: List of BGR frames (original resolution).
            interpolate: Whether to interpolate gaps in the track.
            progress_callback: Optional callback(current_frame, total_frames).

        Returns:
            List of BallDetection, one per frame.
        """
        if not frames:
            return []

        if len(frames) < 3:
            return [
                BallDetection(frame_number=i, x=None, y=None, confidence=0.0)
                for i in range(len(frames))
            ]

        model = self._ensure_model()
        cfg = self.config
        inp_w = cfg.model.tracknet_v4_input_width
        inp_h = cfg.model.tracknet_v4_input_height

        orig_h, orig_w = frames[0].shape[:2]
        scale_x = orig_w / inp_w
        scale_y = orig_h / inp_h

        # First two frames have no detection (need 3-frame input)
        ball_track: list[tuple[float | None, float | None]] = [(None, None)] * 2
        confidences: list[float] = [0.0, 0.0]
        dists: list[float] = [-1.0, -1.0]

        for num in tqdm(range(2, len(frames)), desc="Ball tracking (V4)"):
            img = cv2.cvtColor(cv2.resize(frames[num], (inp_w, inp_h)), cv2.COLOR_BGR2RGB)
            img_prev = cv2.cvtColor(cv2.resize(frames[num - 1], (inp_w, inp_h)), cv2.COLOR_BGR2RGB)
            img_preprev = cv2.cvtColor(cv2.resize(frames[num - 2], (inp_w, inp_h)), cv2.COLOR_BGR2RGB)

            # Stack 3 frames chronologically [oldest, middle, newest] → (H, W, 9)
            imgs = np.concatenate((img_preprev, img_prev, img), axis=2)
            imgs = imgs.astype(np.float32) / 255.0
            imgs = np.rollaxis(imgs, 2, 0)  # (9, H, W)
            inp = np.expand_dims(imgs, axis=0)

            with torch.no_grad():
                out = model(torch.from_numpy(inp).float().to(cfg.device))

            # V4 output: (1, 3, H, W) — channel 1 is the center frame, which
            # has bidirectional temporal context (matches reference inference).
            heatmap = out[0, 1].cpu().numpy()

            x_pred, y_pred, conf = postprocess_v4(
                heatmap, scale_x, scale_y,
                threshold=cfg.ball_tracking.v4_detection_threshold,
            )
            ball_track.append((x_pred, y_pred))
            confidences.append(conf)

            if ball_track[-1][0] is not None and ball_track[-2][0] is not None:
                dist = distance.euclidean(ball_track[-1], ball_track[-2])
            else:
                dist = -1.0
            dists.append(dist)

            if progress_callback is not None:
                progress_callback(num - 1, len(frames) - 2)

        # Remove outliers
        ball_track = self._remove_outliers(ball_track, dists)

        # Save raw track before interpolation
        raw_track = list(ball_track)

        if interpolate:
            ball_track = self._interpolate_track(ball_track)

        # Convert to BallDetection list
        detections: list[BallDetection] = []
        for i, (x, y) in enumerate(ball_track):
            was_raw = raw_track[i][0] is not None
            is_interpolated = (x is not None) and not was_raw
            conf = confidences[i] if was_raw else (0.5 if is_interpolated else 0.0)
            detections.append(BallDetection(
                frame_number=i, x=x, y=y, confidence=conf,
                interpolated=is_interpolated,
            ))

        return detections

    def detect_streaming(
        self,
        frame_iter: Iterator[np.ndarray],
        total_frames: int,
        interpolate: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[BallDetection]:
        """Run ball detection by streaming frames through a sliding window.

        Memory-efficient alternative to detect(): keeps at most 3 frames
        in memory at a time.

        Args:
            frame_iter: Iterator yielding BGR frames (original resolution).
            total_frames: Total number of frames (for progress reporting).
            interpolate: Whether to interpolate gaps in the track.
            progress_callback: Optional callback(current_frame, total_frames).

        Returns:
            List of BallDetection, one per frame.
        """
        model = self._ensure_model()
        cfg = self.config
        inp_w = cfg.model.tracknet_v4_input_width
        inp_h = cfg.model.tracknet_v4_input_height

        window: deque[np.ndarray] = deque(maxlen=3)
        scale_x: float | None = None
        scale_y: float | None = None
        actual_count = 0

        ball_track: list[tuple[float | None, float | None]] = []
        confidences: list[float] = []
        dists: list[float] = []

        for frame in frame_iter:
            actual_count += 1
            window.append(frame)

            if scale_x is None:
                orig_h, orig_w = frame.shape[:2]
                scale_x = orig_w / inp_w
                scale_y = orig_h / inp_h

            if len(window) < 3:
                ball_track.append((None, None))
                confidences.append(0.0)
                dists.append(-1.0)
                continue

            img = cv2.cvtColor(cv2.resize(window[2], (inp_w, inp_h)), cv2.COLOR_BGR2RGB)
            img_prev = cv2.cvtColor(cv2.resize(window[1], (inp_w, inp_h)), cv2.COLOR_BGR2RGB)
            img_preprev = cv2.cvtColor(cv2.resize(window[0], (inp_w, inp_h)), cv2.COLOR_BGR2RGB)

            # Stack chronologically [oldest, middle, newest]
            imgs = np.concatenate((img_preprev, img_prev, img), axis=2)
            imgs = imgs.astype(np.float32) / 255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)

            with torch.no_grad():
                out = model(torch.from_numpy(inp).float().to(cfg.device))

            # Channel 1 = center frame with bidirectional context
            heatmap = out[0, 1].cpu().numpy()

            x_pred, y_pred, conf = postprocess_v4(
                heatmap, scale_x, scale_y,
                threshold=cfg.ball_tracking.v4_detection_threshold,
            )
            ball_track.append((x_pred, y_pred))
            confidences.append(conf)

            if ball_track[-1][0] is not None and ball_track[-2][0] is not None:
                dist = distance.euclidean(ball_track[-1], ball_track[-2])
            else:
                dist = -1.0
            dists.append(dist)

            if progress_callback is not None:
                progress_callback(actual_count - 2, max(1, total_frames - 2))

        if actual_count == 0:
            return []
        if actual_count < 3:
            return [
                BallDetection(frame_number=i, x=None, y=None, confidence=0.0)
                for i in range(actual_count)
            ]

        ball_track = self._remove_outliers(ball_track, dists)
        raw_track = list(ball_track)
        if interpolate:
            ball_track = self._interpolate_track(ball_track)

        detections: list[BallDetection] = []
        for i, (x, y) in enumerate(ball_track):
            was_raw = raw_track[i][0] is not None
            is_interpolated = (x is not None) and not was_raw
            conf = confidences[i] if was_raw else (0.5 if is_interpolated else 0.0)
            detections.append(BallDetection(
                frame_number=i, x=x, y=y, confidence=conf,
                interpolated=is_interpolated,
            ))

        return detections

    # ── Post-processing (shared with V2 BallTracker) ──

    def _remove_outliers(
        self,
        ball_track: list[tuple[float | None, float | None]],
        dists: list[float],
    ) -> list[tuple[float | None, float | None]]:
        """Remove outlier detections based on distance between consecutive points."""
        max_dist = self.config.ball_tracking.max_outlier_dist
        outliers = list(np.where(np.array(dists) > max_dist)[0])
        for i in outliers:
            if i + 1 < len(dists):
                if dists[i + 1] > max_dist or dists[i + 1] == -1:
                    ball_track[i] = (None, None)
                elif i > 0 and dists[i - 1] == -1:
                    ball_track[i - 1] = (None, None)

        ball_track = self._remove_stationary(ball_track)
        return ball_track

    @staticmethod
    def _remove_stationary(
        ball_track: list[tuple[float | None, float | None]],
        max_stationary_frames: int = 10,
        stationary_radius: float = 15.0,
    ) -> list[tuple[float | None, float | None]]:
        """Remove detections that remain stationary for too many frames."""
        if len(ball_track) < max_stationary_frames:
            return ball_track

        result = list(ball_track)
        i = 0
        while i < len(result):
            if result[i][0] is None:
                i += 1
                continue

            cluster_start = i
            ref_x, ref_y = result[i]
            j = i + 1
            while j < len(result):
                if result[j][0] is None:
                    break
                dx = result[j][0] - ref_x
                dy = result[j][1] - ref_y
                if (dx * dx + dy * dy) > stationary_radius * stationary_radius:
                    break
                j += 1

            cluster_len = j - cluster_start
            if cluster_len > max_stationary_frames:
                for k in range(cluster_start, j):
                    result[k] = (None, None)

            i = j

        return result

    def _interpolate_track(
        self,
        ball_track: list[tuple[float | None, float | None]],
    ) -> list[tuple[float | None, float | None]]:
        """Split track into subtracks and interpolate gaps."""
        cfg = self.config.ball_tracking
        subtracks = self._split_track(
            ball_track, cfg.max_gap, cfg.max_dist_gap, cfg.min_track_length,
        )
        for start, end in subtracks:
            subtrack = ball_track[start:end]
            interpolated = self._interpolate_subtrack(subtrack)
            ball_track[start:end] = interpolated
        return ball_track

    @staticmethod
    def _split_track(
        ball_track: list[tuple[float | None, float | None]],
        max_gap: int = 4,
        max_dist_gap: float = 80.0,
        min_track: int = 5,
    ) -> list[list[int]]:
        """Split track into coherent subtracks for interpolation."""
        list_det = [0 if x[0] is not None else 1 for x in ball_track]
        groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

        cursor = 0
        min_value = 0
        result: list[list[int]] = []
        for i, (k, length) in enumerate(groups):
            if k == 1 and i > 0 and i < len(groups) - 1:
                if cursor > 0 and cursor + length < len(ball_track):
                    dist = distance.euclidean(
                        ball_track[cursor - 1], ball_track[cursor + length],
                    )
                    if length >= max_gap or dist / length > max_dist_gap:
                        if cursor - min_value > min_track:
                            result.append([min_value, cursor])
                        min_value = cursor + length - 1
            cursor += length
        if len(list_det) - min_value > min_track:
            result.append([min_value, len(list_det)])

        result = [
            r for r in result
            if any(ball_track[i][0] is not None for i in range(r[0], r[1]))
        ]
        return result

    @staticmethod
    def _interpolate_subtrack(
        coords: list[tuple[float | None, float | None]],
    ) -> list[tuple[float, float]]:
        """Interpolate missing positions within a subtrack."""
        x_arr = np.array([c[0] if c[0] is not None else np.nan for c in coords])
        y_arr = np.array([c[1] if c[1] is not None else np.nan for c in coords])

        if np.isnan(x_arr).all():
            return list(coords)

        nans_x = np.isnan(x_arr)
        if nans_x.any() and not nans_x.all():
            indices = np.arange(len(x_arr))
            x_arr[nans_x] = np.interp(
                indices[nans_x], indices[~nans_x], x_arr[~nans_x],
            )
        nans_y = np.isnan(y_arr)
        if nans_y.any() and not nans_y.all():
            indices = np.arange(len(y_arr))
            y_arr[nans_y] = np.interp(
                indices[nans_y], indices[~nans_y], y_arr[~nans_y],
            )

        return list(zip(x_arr.tolist(), y_arr.tolist()))
