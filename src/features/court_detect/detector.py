"""Court keypoint detection — runs court detector inference on video frames."""

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from scipy.spatial import distance

from src.config import Config, get_config
from src.models.court_net import CourtDetectorNet, load_court_detector
from src.features.court_detect.court_template import COURT_REF, REFER_KPS, COURT_CONF_INDICES
from src.features.court_detect.homography import (
    find_best_homography,
    refine_keypoints_with_homography,
    compute_pixel_to_court_homography,
)


@dataclass
class CourtDetectionResult:
    """Result of court detection on a single frame."""
    frame_number: int
    keypoints: list[tuple[float | None, float | None]]
    num_detected: int
    homography_ref_to_img: np.ndarray | None = field(default=None, repr=False)
    homography_img_to_court: np.ndarray | None = field(default=None, repr=False)
    confidence: float = 0.0

    @property
    def detected(self) -> bool:
        """Whether enough keypoints were detected for a valid homography."""
        return self.num_detected >= 4 and self.homography_img_to_court is not None


def _postprocess_heatmap(
    heatmap: np.ndarray,
    scale_x: float = 2.0,
    scale_y: float = 2.0,
    low_thresh: int = 170,
    min_radius: int = 10,
    max_radius: int = 25,
) -> tuple[float | None, float | None]:
    """Extract keypoint (x, y) from a single heatmap using HoughCircles.

    Args:
        heatmap: 2D heatmap, values in [0, 255].
        scale_x: Scale factor for X (model output width → input image width).
        scale_y: Scale factor for Y (model output height → input image height).
        low_thresh: Binary threshold for heatmap.
        min_radius: Minimum circle radius for HoughCircles.
        max_radius: Maximum circle radius for HoughCircles.

    Returns:
        (x, y) in scaled image coordinates, or (None, None).
    """
    _, binary = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        binary, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=50, param2=2, minRadius=min_radius, maxRadius=max_radius,
    )
    if circles is not None:
        x = float(circles[0][0][0] * scale_x)
        y = float(circles[0][0][1] * scale_y)
        return x, y
    return None, None


def _refine_keypoint(
    image: np.ndarray,
    x_ct: int,
    y_ct: int,
    crop_size: int = 40,
) -> tuple[int, int]:
    """Refine a keypoint using line intersection in a local crop.

    Detects lines in a crop around the keypoint and finds their intersection
    for sub-pixel accuracy.

    Args:
        image: Full-resolution BGR image.
        x_ct: Row coordinate of keypoint center.
        y_ct: Column coordinate of keypoint center.
        crop_size: Half-size of the crop window.

    Returns:
        Refined (y, x) coordinates — note the upstream convention.
    """
    img_height, img_width = image.shape[:2]
    x_min = max(x_ct - crop_size, 0)
    x_max = min(img_height, x_ct + crop_size)
    y_min = max(y_ct - crop_size, 0)
    y_max = min(img_width, y_ct + crop_size)

    crop = image[x_min:x_max, y_min:y_max]
    if crop.size == 0:
        return y_ct, x_ct

    # Detect lines in the crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(
        gray, 1, np.pi / 180, 30, minLineLength=10, maxLineGap=30,
    )

    if lines is None or len(lines) < 2:
        return y_ct, x_ct

    lines = np.squeeze(lines)
    if lines.ndim == 1:
        lines = [lines]
    else:
        lines = list(lines)

    # Merge nearby parallel lines
    lines = _merge_lines(lines)

    if len(lines) == 2:
        intersection = _line_intersection(lines[0], lines[1])
        if intersection is not None:
            new_x_ct = int(intersection[1])
            new_y_ct = int(intersection[0])
            if 0 < new_x_ct < crop.shape[0] and 0 < new_y_ct < crop.shape[1]:
                return y_min + new_y_ct, x_min + new_x_ct

    return y_ct, x_ct


def _merge_lines(lines: list[np.ndarray]) -> list[np.ndarray]:
    """Merge nearby parallel lines into single representative lines."""
    lines = sorted(lines, key=lambda item: item[0])
    mask = [True] * len(lines)
    new_lines = []

    for i, line in enumerate(lines):
        if mask[i]:
            for j, s_line in enumerate(lines[i + 1:]):
                if mask[i + j + 1]:
                    x1, y1, x2, y2 = line
                    x3, y3, x4, y4 = s_line
                    dist1 = distance.euclidean((x1, y1), (x3, y3))
                    dist2 = distance.euclidean((x2, y2), (x4, y4))
                    if dist1 < 20 and dist2 < 20:
                        line = np.array(
                            [int((x1 + x3) / 2), int((y1 + y3) / 2),
                             int((x2 + x4) / 2), int((y2 + y4) / 2)],
                            dtype=np.int32,
                        )
                        mask[i + j + 1] = False
            new_lines.append(line)
    return new_lines


def _line_intersection(
    line1: np.ndarray, line2: np.ndarray,
) -> tuple[float, float] | None:
    """Find intersection of two line segments using cross product.

    Args:
        line1: (x1, y1, x2, y2) array.
        line2: (x3, y3, x4, y4) array.

    Returns:
        (x, y) intersection point, or None if lines are parallel.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return float(ix), float(iy)


class CourtDetector:
    """Detect tennis court keypoints in video frames.

    Args:
        config: Project configuration. Uses default if not provided.
        use_refine_kps: Whether to use line-intersection refinement.
        use_homography: Whether to use homography-based keypoint correction.
    """

    def __init__(
        self,
        config: Config | None = None,
        use_refine_kps: bool = True,
        use_homography: bool = True,
    ):
        self.config = config or get_config()
        self.use_refine_kps = use_refine_kps
        self.use_homography = use_homography
        self._model: CourtDetectorNet | None = None

    def _ensure_model(self) -> CourtDetectorNet:
        """Lazily load the model."""
        if self._model is None:
            weights = self.config.resolve_path(self.config.model.court_weights)
            self._model = load_court_detector(str(weights), self.config.device)
        return self._model

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
    ) -> CourtDetectionResult:
        """Detect court keypoints in a single frame.

        Args:
            frame: BGR image at original resolution.
            frame_number: Frame index for tracking.

        Returns:
            CourtDetectionResult with keypoints and homography.
        """
        model = self._ensure_model()
        cfg = self.config
        inp_w = cfg.model.court_input_width
        inp_h = cfg.model.court_input_height

        # Scale factors: model output → original image coordinates
        orig_h, orig_w = frame.shape[:2]
        scale_x = orig_w / inp_w
        scale_y = orig_h / inp_h

        # Preprocess: resize, normalize, to tensor
        img = cv2.resize(frame, (inp_w, inp_h))
        inp = img.astype(np.float32) / 255.0
        inp = np.rollaxis(inp, 2, 0)  # (3, H, W)
        inp = torch.tensor(inp).unsqueeze(0)  # (1, 3, H, W)

        # Inference
        with torch.no_grad():
            out = model(inp.float().to(cfg.device))

        # Apply sigmoid to get heatmap probabilities
        pred = torch.sigmoid(out[0]).detach().cpu().numpy()

        # Extract keypoints from heatmaps
        points: list[tuple[float | None, float | None]] = []
        for kps_num in range(14):
            heatmap = (pred[kps_num] * 255).astype(np.uint8)
            x_pred, y_pred = _postprocess_heatmap(
                heatmap, scale_x=scale_x, scale_y=scale_y,
                low_thresh=cfg.court_detection.heatmap_threshold,
                min_radius=cfg.court_detection.min_radius,
                max_radius=cfg.court_detection.max_radius,
            )

            # Refine using line intersection (skip center-line points)
            if (self.use_refine_kps
                    and kps_num not in [8, 9, 12]
                    and x_pred is not None and y_pred is not None):
                x_pred, y_pred = _refine_keypoint(
                    frame, int(y_pred), int(x_pred),
                    crop_size=cfg.court_detection.refine_crop_size,
                )

            points.append((x_pred, y_pred))

        num_detected = sum(1 for p in points if p[0] is not None)

        # Apply homography-based correction
        homography_ref = None
        raw_detected = num_detected
        if self.use_homography and num_detected >= 4:
            homography_ref = find_best_homography(points)
            if homography_ref is not None:
                points = refine_keypoints_with_homography(points, homography_ref)
                num_detected = 14  # All points now filled via homography

        # Compute pixel → court-meters homography
        homography_court = compute_pixel_to_court_homography(points)

        # Confidence reflects actual raw detections, not homography-filled points
        confidence = raw_detected / 14.0

        return CourtDetectionResult(
            frame_number=frame_number,
            keypoints=points,
            num_detected=num_detected,
            homography_ref_to_img=homography_ref,
            homography_img_to_court=homography_court,
            confidence=confidence,
        )

    def detect_video(
        self,
        frames: list[np.ndarray],
        sample_every: int = 1,
    ) -> list[CourtDetectionResult]:
        """Detect court keypoints across multiple frames.

        For efficiency, the court is relatively static so we can sample
        fewer frames and reuse the homography.

        Args:
            frames: List of BGR video frames.
            sample_every: Process every Nth frame (1 = all frames).

        Returns:
            List of CourtDetectionResult, one per sampled frame.
        """
        if not frames:
            return []

        from tqdm import tqdm

        results: list[CourtDetectionResult] = []
        last_good_result: CourtDetectionResult | None = None

        frame_indices = range(0, len(frames), sample_every)
        for idx in tqdm(frame_indices, desc="Court detection"):
            result = self.detect_frame(frames[idx], frame_number=idx)
            if result.detected:
                last_good_result = result
            elif last_good_result is not None:
                # Reuse last good result with updated frame number
                result = CourtDetectionResult(
                    frame_number=idx,
                    keypoints=last_good_result.keypoints,
                    num_detected=last_good_result.num_detected,
                    homography_ref_to_img=last_good_result.homography_ref_to_img,
                    homography_img_to_court=last_good_result.homography_img_to_court,
                    confidence=last_good_result.confidence * 0.9,  # decay
                )
            results.append(result)

        return results

    def get_stable_homography(
        self,
        frames: list[np.ndarray],
        max_samples: int = 20,
    ) -> np.ndarray | None:
        """Get a stable pixel→court homography from the best-scoring frame.

        Selects the single homography with the highest confidence
        across all sampled frames (static camera assumption).

        Args:
            frames: List of BGR video frames.
            max_samples: Maximum number of frames to sample.

        Returns:
            3×3 homography matrix, or None if detection fails.
        """
        if not frames:
            return None

        import logging
        logger = logging.getLogger(__name__)

        step = max(1, len(frames) // max_samples)
        best_homography: np.ndarray | None = None
        best_confidence = -1.0

        for idx in range(0, len(frames), step):
            result = self.detect_frame(frames[idx], frame_number=idx)
            logger.debug(
                "Court detection frame %d: %d raw keypoints, confidence=%.3f, homography=%s",
                idx, sum(1 for p in result.keypoints if p[0] is not None),
                result.confidence,
                result.homography_img_to_court is not None,
            )
            if (result.homography_img_to_court is not None
                    and result.confidence > best_confidence):
                best_confidence = result.confidence
                best_homography = result.homography_img_to_court
            if best_confidence >= 0.85:
                break

        logger.info(
            "Court detection best confidence: %.3f across %d sampled frames",
            best_confidence, min(len(frames), max_samples),
        )
        return best_homography
