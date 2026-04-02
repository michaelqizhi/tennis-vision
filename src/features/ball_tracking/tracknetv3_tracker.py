"""Ball tracking detector — runs TrackNet V3 inference on video frames.

TrackNet V3 uses a U-Net with skip connections. This checkpoint uses:
  - ``seq_len=8``, ``bg_mode=concat``
  - Input: 27 channels [background_RGB(3) + 8_frames_RGB(24)] at 512×288
  - Output: 8-channel sigmoid heatmap (one per input frame)
  - Non-overlapping 8-frame sliding windows
  - Median background image estimated from sampled video frames

This module provides ``BallTrackerV3`` which mirrors the V2 ``BallTracker``
and V4 ``BallTrackerV4`` APIs but uses the V3 architecture.
"""

from __future__ import annotations

from itertools import groupby
from typing import Callable, Iterator

import cv2
import numpy as np
import torch
from scipy.spatial import distance
from tqdm import tqdm

from src.config import Config, get_config
from src.features.ball_tracking.detector import BallDetection
from src.features.ball_tracking.tracknetv4_tracker import postprocess_v4
from src.models.tracknetv3 import TrackNetV3Model, load_tracknet_v3, INPUT_W, INPUT_H

_SEQ_LEN = 8


class BallTrackerV3:
    """Track tennis ball positions across video frames using TrackNet V3.

    Drop-in replacement for BallTracker (V2) / BallTrackerV4 (V4) with the
    V3 U-Net architecture.  Returns the same ``BallDetection`` dataclass.

    The V3 checkpoint uses an 8-frame window with a concatenated background
    median image (``bg_mode=concat``), giving 27 input channels.

    Args:
        config: Project configuration. Uses default if not provided.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._model: TrackNetV3Model | None = None

    def _ensure_model(self) -> TrackNetV3Model:
        """Lazily load the V3 model."""
        if self._model is None:
            weights = self.config.resolve_path(self.config.model.tracknet_v3_weights)
            self._model = load_tracknet_v3(
                str(weights), self.config.device, auto_download=True,
            )
        return self._model

    # ── Background estimation ──

    @staticmethod
    def _estimate_background(
        frames: list[np.ndarray],
        sample_count: int = 100,
    ) -> np.ndarray:
        """Compute a median background image from a list of RGB 512×288 frames.

        Args:
            frames: Pre-processed frames (RGB, already resized to 512×288).
            sample_count: Max number of frames to sample for the median.  If
                the video has fewer frames than this, all frames are used.

        Returns:
            Median background as a uint8 ndarray of shape (288, 512, 3).
        """
        n = len(frames)
        if n <= sample_count:
            selected = frames
        else:
            indices = np.linspace(0, n - 1, sample_count, dtype=int)
            selected = [frames[i] for i in indices]

        stacked = np.stack(selected, axis=0)  # (N, H, W, 3)
        median_bg = np.median(stacked, axis=0).astype(np.uint8)
        return median_bg

    # ── Input construction ──

    @staticmethod
    def _build_input(
        bg: np.ndarray,
        window_frames: list[np.ndarray],
    ) -> np.ndarray:
        """Build a (27, H, W) input tensor for one 8-frame window.

        Channel layout: [bg_R, bg_G, bg_B, f1_R, f1_G, f1_B, ..., f8_R, f8_G, f8_B]

        Args:
            bg: Background median image (H, W, 3), uint8 RGB.
            window_frames: List of 8 frames (H, W, 3), uint8 RGB.

        Returns:
            Float32 array of shape (27, H, W) normalised to [0, 1].
        """
        # Concatenate bg + 8 frames along channel axis → (H, W, 27)
        imgs = np.concatenate([bg] + window_frames, axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)  # (27, H, W)
        return imgs

    # ── Inference helpers ──

    def _run_windows(
        self,
        rgb_frames: list[np.ndarray],
        bg: np.ndarray,
        scale_x: float,
        scale_y: float,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[
        list[tuple[float | None, float | None]],
        list[float],
        list[float],
    ]:
        """Process all frames through non-overlapping 8-frame windows.

        Returns ball_track, confidences, and consecutive-distance lists (one
        entry per frame).
        """
        model = self._ensure_model()
        cfg = self.config
        n = len(rgb_frames)
        threshold = cfg.ball_tracking.v3_detection_threshold

        ball_track: list[tuple[float | None, float | None]] = [(None, None)] * n
        confidences: list[float] = [0.0] * n
        dists: list[float] = [-1.0] * n

        num_windows = (n + _SEQ_LEN - 1) // _SEQ_LEN
        frames_processed = 0

        for w in tqdm(range(num_windows), desc="Ball tracking (V3)"):
            start = w * _SEQ_LEN
            end = min(start + _SEQ_LEN, n)
            window = list(rgb_frames[start:end])

            # Pad with the last frame if the window is incomplete
            valid_count = len(window)
            while len(window) < _SEQ_LEN:
                window.append(window[-1])

            inp = self._build_input(bg, window)
            inp_batch = np.expand_dims(inp, axis=0)  # (1, 27, H, W)

            with torch.no_grad():
                out = model(
                    torch.from_numpy(inp_batch).float().to(cfg.device),
                )  # (1, 8, H, W)

            out_np = out[0].cpu().numpy()  # (8, H, W)

            for f in range(valid_count):
                abs_idx = start + f
                heatmap = out_np[f]
                x_pred, y_pred, conf = postprocess_v4(
                    heatmap, scale_x, scale_y, threshold=threshold,
                )
                ball_track[abs_idx] = (x_pred, y_pred)
                confidences[abs_idx] = conf

                if (
                    abs_idx > 0
                    and ball_track[abs_idx][0] is not None
                    and ball_track[abs_idx - 1][0] is not None
                ):
                    dists[abs_idx] = distance.euclidean(
                        ball_track[abs_idx], ball_track[abs_idx - 1],
                    )

            frames_processed += valid_count
            if progress_callback is not None:
                progress_callback(frames_processed, n)

        return ball_track, confidences, dists

    # ── Public API ──

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

        model = self._ensure_model()
        cfg = self.config
        inp_w = cfg.model.tracknet_v3_input_width
        inp_h = cfg.model.tracknet_v3_input_height

        orig_h, orig_w = frames[0].shape[:2]
        scale_x = orig_w / inp_w
        scale_y = orig_h / inp_h

        # Pre-process: resize + BGR→RGB for all frames
        rgb_frames = [
            cv2.cvtColor(cv2.resize(f, (inp_w, inp_h)), cv2.COLOR_BGR2RGB)
            for f in frames
        ]

        # Background estimation
        sample_count = cfg.ball_tracking.v3_bg_sample_count
        bg = self._estimate_background(rgb_frames, sample_count)

        # Run inference in 8-frame windows
        ball_track, confidences, dists = self._run_windows(
            rgb_frames, bg, scale_x, scale_y, progress_callback,
        )

        # Post-processing
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

    def detect_streaming(
        self,
        frame_iter: Iterator[np.ndarray],
        total_frames: int,
        interpolate: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[BallDetection]:
        """Run ball detection by streaming frames from an iterator.

        V3 requires background estimation and arbitrary window access, so all
        frames are collected first (stored as resized 512×288 RGB to save
        memory).  Processing then proceeds identically to :meth:`detect`.

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
        inp_w = cfg.model.tracknet_v3_input_width
        inp_h = cfg.model.tracknet_v3_input_height

        scale_x: float | None = None
        scale_y: float | None = None
        rgb_frames: list[np.ndarray] = []

        for frame in frame_iter:
            if scale_x is None:
                orig_h, orig_w = frame.shape[:2]
                scale_x = orig_w / inp_w
                scale_y = orig_h / inp_h
            rgb_frames.append(
                cv2.cvtColor(cv2.resize(frame, (inp_w, inp_h)), cv2.COLOR_BGR2RGB),
            )

        if not rgb_frames:
            return []

        # Background estimation
        sample_count = cfg.ball_tracking.v3_bg_sample_count
        bg = self._estimate_background(rgb_frames, sample_count)

        # Run inference in 8-frame windows
        ball_track, confidences, dists = self._run_windows(
            rgb_frames, bg, scale_x, scale_y, progress_callback,
        )

        # Post-processing
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

    # ── Post-processing (shared with V2/V4 trackers) ──

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
