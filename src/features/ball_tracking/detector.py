"""Ball tracking detector — runs TrackNetV2 inference on video frames."""

import math
from collections import deque
from dataclasses import dataclass
from itertools import groupby
from typing import Callable, Iterator

import cv2
import numpy as np
import torch
from scipy.spatial import distance
from tqdm import tqdm

from src.config import Config, get_config
from src.models.tracknet import BallTrackerNet, load_tracknet


@dataclass
class BallDetection:
    """A single ball detection for one frame."""
    frame_number: int
    x: float | None
    y: float | None
    confidence: float
    interpolated: bool = False

    @property
    def detected(self) -> bool:
        if self.x is None or self.y is None:
            return False
        if math.isnan(self.x) or math.isnan(self.y):
            return False
        return True


def postprocess(feature_map: np.ndarray, width: int = 640, height: int = 360,
                scale_x: float = 1.0, scale_y: float = 1.0,
                threshold: int = 127,
                hough_min_dist: int = 1,
                hough_param1: int = 50,
                hough_param2: int = 2,
                hough_min_radius: int = 2,
                hough_max_radius: int = 7) -> tuple[float | None, float | None]:
    """Extract ball (x, y) from a model output heatmap using weighted centroid.

    Uses cv2.moments weighted centroid as the primary method for robustness
    on courtside footage. Falls back to HoughCircles if moments fail.

    Args:
        feature_map: Raw model output, shape (height*width,) or (height, width).
        width: Model input width.
        height: Model input height.
        scale_x: Scale factor to map x back to original video resolution.
        scale_y: Scale factor to map y back to original video resolution.
        threshold: Binary threshold for heatmap.
        hough_min_dist: Minimum distance between detected circle centers.
        hough_param1: First method-specific parameter for HoughCircles.
        hough_param2: Second method-specific parameter for HoughCircles.
        hough_min_radius: Minimum circle radius.
        hough_max_radius: Maximum circle radius.

    Returns:
        Tuple of (x, y) in original video coordinates, or (None, None).
    """
    feature_map = (feature_map * 255).astype(np.uint8)
    if feature_map.ndim == 1:
        feature_map = feature_map.reshape((height, width))

    _, heatmap = cv2.threshold(feature_map, threshold, 255, cv2.THRESH_BINARY)

    # Primary: weighted centroid via cv2.moments
    moments = cv2.moments(heatmap)
    if moments["m00"] > 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        x = float(cx * scale_x)
        y = float(cy * scale_y)
        return x, y

    # Fallback: HoughCircles
    circles = cv2.HoughCircles(
        heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=hough_min_dist,
        param1=hough_param1, param2=hough_param2,
        minRadius=hough_min_radius, maxRadius=hough_max_radius,
    )

    if circles is not None and len(circles) == 1:
        x = float(circles[0][0][0] * scale_x)
        y = float(circles[0][0][1] * scale_y)
        return x, y
    return None, None


class BallTracker:
    """Track tennis ball positions across video frames using TrackNetV2.

    Args:
        config: Project configuration. Uses default if not provided.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._model: BallTrackerNet | None = None

    def _ensure_model(self) -> BallTrackerNet:
        """Lazily load the model."""
        if self._model is None:
            weights = self.config.resolve_path(self.config.model.tracknet_weights)
            self._model = load_tracknet(str(weights), self.config.device)
        return self._model

    def detect(self, frames: list[np.ndarray],
               interpolate: bool = True,
               progress_callback: "Callable[[int, int], None] | None" = None) -> list[BallDetection]:
        """Run ball detection on a list of video frames.

        Args:
            frames: List of BGR frames (original resolution).
            interpolate: Whether to interpolate gaps in the track.
            progress_callback: Optional callback(current_frame, total_frames) for progress reporting.

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
        inp_w = cfg.model.tracknet_input_width
        inp_h = cfg.model.tracknet_input_height

        orig_h, orig_w = frames[0].shape[:2]
        scale_x = orig_w / inp_w
        scale_y = orig_h / inp_h

        # First two frames have no detection (need 3-frame input)
        ball_track: list[tuple[float | None, float | None]] = [(None, None)] * 2
        dists: list[float] = [-1.0, -1.0]

        for num in tqdm(range(2, len(frames)), desc="Ball tracking"):
            img = cv2.resize(frames[num], (inp_w, inp_h))
            img_prev = cv2.resize(frames[num - 1], (inp_w, inp_h))
            img_preprev = cv2.resize(frames[num - 2], (inp_w, inp_h))

            # Stack 3 frames → (H, W, 9)
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            imgs = imgs.astype(np.float32) / 255.0
            # (9, H, W)
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)

            with torch.no_grad():
                out = model(torch.from_numpy(inp).float().to(cfg.device))

            output = out.argmax(dim=1).detach().cpu().numpy()
            x_pred, y_pred = postprocess(
                output[0], inp_w, inp_h, scale_x, scale_y,
                threshold=cfg.ball_tracking.confidence_threshold,
                hough_min_dist=cfg.ball_tracking.hough_min_dist,
                hough_param1=cfg.ball_tracking.hough_param1,
                hough_param2=cfg.ball_tracking.hough_param2,
                hough_min_radius=cfg.ball_tracking.hough_min_radius,
                hough_max_radius=cfg.ball_tracking.hough_max_radius,
            )
            ball_track.append((x_pred, y_pred))

            if ball_track[-1][0] is not None and ball_track[-2][0] is not None:
                dist = distance.euclidean(ball_track[-1], ball_track[-2])
            else:
                dist = -1.0
            dists.append(dist)

            if progress_callback is not None:
                progress_callback(num - 1, len(frames) - 2)

        # Remove outliers
        ball_track = self._remove_outliers(ball_track, dists)

        # Save raw (pre-interpolation) track to mark interpolated positions
        raw_track = list(ball_track)

        # Interpolate gaps
        if interpolate:
            ball_track = self._interpolate_track(ball_track)

        # Convert to BallDetection list
        detections: list[BallDetection] = []
        for i, (x, y) in enumerate(ball_track):
            was_raw = raw_track[i][0] is not None
            is_interpolated = (x is not None) and not was_raw
            conf = 1.0 if was_raw else (0.5 if is_interpolated else 0.0)
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
        in memory at a time instead of loading the entire video.

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
        inp_w = cfg.model.tracknet_input_width
        inp_h = cfg.model.tracknet_input_height

        # Sliding window of the 3 most recent frames
        window: deque[np.ndarray] = deque(maxlen=3)
        scale_x: float | None = None
        scale_y: float | None = None
        actual_count = 0

        ball_track: list[tuple[float | None, float | None]] = []
        dists: list[float] = []

        for frame in frame_iter:
            actual_count += 1
            window.append(frame)

            # Determine scale factors from first frame
            if scale_x is None:
                orig_h, orig_w = frame.shape[:2]
                scale_x = orig_w / inp_w
                scale_y = orig_h / inp_h

            if len(window) < 3:
                # Need 3 frames before detection can start
                ball_track.append((None, None))
                dists.append(-1.0)
                continue

            img = cv2.resize(window[2], (inp_w, inp_h))
            img_prev = cv2.resize(window[1], (inp_w, inp_h))
            img_preprev = cv2.resize(window[0], (inp_w, inp_h))

            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            imgs = imgs.astype(np.float32) / 255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)

            with torch.no_grad():
                out = model(torch.from_numpy(inp).float().to(cfg.device))

            output = out.argmax(dim=1).detach().cpu().numpy()
            x_pred, y_pred = postprocess(
                output[0], inp_w, inp_h, scale_x, scale_y,
                threshold=cfg.ball_tracking.confidence_threshold,
                hough_min_dist=cfg.ball_tracking.hough_min_dist,
                hough_param1=cfg.ball_tracking.hough_param1,
                hough_param2=cfg.ball_tracking.hough_param2,
                hough_min_radius=cfg.ball_tracking.hough_min_radius,
                hough_max_radius=cfg.ball_tracking.hough_max_radius,
            )
            ball_track.append((x_pred, y_pred))

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

        # Post-processing is identical to detect()
        ball_track = self._remove_outliers(ball_track, dists)
        raw_track = list(ball_track)
        if interpolate:
            ball_track = self._interpolate_track(ball_track)

        detections: list[BallDetection] = []
        for i, (x, y) in enumerate(ball_track):
            was_raw = raw_track[i][0] is not None
            is_interpolated = (x is not None) and not was_raw
            conf = 1.0 if was_raw else (0.5 if is_interpolated else 0.0)
            detections.append(BallDetection(
                frame_number=i, x=x, y=y, confidence=conf,
                interpolated=is_interpolated,
            ))

        return detections

    def _remove_outliers(
        self,
        ball_track: list[tuple[float | None, float | None]],
        dists: list[float],
    ) -> list[tuple[float | None, float | None]]:
        """Remove outlier detections based on distance between consecutive points.

        Also rejects stationary detections: if the ball stays within a small
        radius for too many consecutive frames, those are likely false positives
        on players or static objects.
        """
        max_dist = self.config.ball_tracking.max_outlier_dist
        outliers = list(np.where(np.array(dists) > max_dist)[0])
        for i in outliers:
            if i + 1 < len(dists):
                if dists[i + 1] > max_dist or dists[i + 1] == -1:
                    ball_track[i] = (None, None)
                elif i > 0 and dists[i - 1] == -1:
                    ball_track[i - 1] = (None, None)

        # Remove stationary clusters (likely player/static object FPs)
        ball_track = self._remove_stationary(ball_track)

        return ball_track

    @staticmethod
    def _remove_stationary(
        ball_track: list[tuple[float | None, float | None]],
        max_stationary_frames: int = 10,
        stationary_radius: float = 15.0,
    ) -> list[tuple[float | None, float | None]]:
        """Remove detections that remain stationary for too many frames.

        A tennis ball in play moves rapidly. Detections that cluster in a
        small area for >max_stationary_frames are almost certainly false
        positives on players, net posts, or static background features.

        Args:
            ball_track: Ball position track.
            max_stationary_frames: Max consecutive frames allowed at same spot.
            stationary_radius: Pixel radius to consider as "same position".

        Returns:
            Cleaned ball track with stationary clusters removed.
        """
        if len(ball_track) < max_stationary_frames:
            return ball_track

        result = list(ball_track)
        i = 0
        while i < len(result):
            if result[i][0] is None:
                i += 1
                continue

            # Find how many consecutive frames stay within stationary_radius
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

        # Filter out subtracks with no valid detections
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

        # If all coordinates are None/NaN, return original coords unchanged
        if np.isnan(x_arr).all():
            return list(coords)

        # Interpolate NaN values
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
