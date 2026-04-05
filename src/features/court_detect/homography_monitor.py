"""Reactive homography monitor for detecting camera shifts.

Monitors homography validity using template matching and triggers re-detection
when camera movement is detected.
"""

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.features.court_detect.detector import CourtDetector, CourtDetectionResult

logger = logging.getLogger(__name__)


@dataclass
class TemplateRegion:
    """A small image patch used for monitoring camera stability.

    Args:
        template: 48×48 grayscale patch extracted around a keypoint.
        position: (x, y) center position in the original frame.
    """

    template: np.ndarray
    position: tuple[int, int]


@dataclass
class HomographyMonitor:
    """Monitors homography validity and detects camera shifts.

    Uses template matching at detected keypoint locations to detect when the
    camera has moved, triggering re-detection of the court.

    Args:
        court_detector: CourtDetector instance for re-detection.
        current_homography: Initial homography matrix.
        templates: List of template regions (initially empty, set via extract_templates).
        validation_interval: Frames between validity checks.
        ncc_threshold: NCC correlation threshold (below = keypoint shifted).
        consensus_ratio: Fraction of template failures to trigger re-detection.
        cooldown_frames: Frames to wait after camera bump before re-detecting.
        max_retry_interval: Frames between retries on detection failure.
        disable_after_seconds: Disable filtering after this long without court.
    """

    court_detector: CourtDetector
    current_homography: np.ndarray
    templates: list[TemplateRegion] = field(default_factory=list)

    # Config
    validation_interval: int = 30
    ncc_threshold: float = 0.65
    consensus_ratio: float = 0.6
    cooldown_frames: int = 5
    max_retry_interval: int = 30
    disable_after_seconds: float = 10.0

    # Internal state
    _last_check_frame: int = field(default=0, init=False)
    _shift_detected_frame: int | None = field(default=None, init=False)
    _last_retry_frame: int = field(default=0, init=False)
    _detection_failed_time: float | None = field(default=None, init=False)
    _disabled: bool = field(default=False, init=False)

    def extract_templates(
        self, frame: np.ndarray, keypoints: list[tuple[float | None, float | None]]
    ) -> None:
        """Extract 48×48 templates around detected keypoints.

        Args:
            frame: Full frame image (BGR).
            keypoints: List of 14 (x, y) tuples. Only non-None keypoints
                are used for template extraction.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        template_size = 48
        half_size = template_size // 2

        self.templates = []

        for i, (x, y) in enumerate(keypoints):
            if x is None or y is None:
                continue

            cx, cy = int(x), int(y)

            # Check bounds
            if (
                cx - half_size < 0
                or cx + half_size > w
                or cy - half_size < 0
                or cy + half_size > h
            ):
                logger.debug(f"Keypoint {i} at ({cx}, {cy}) too close to edge, skipping")
                continue

            # Extract template
            template = gray[
                cy - half_size : cy + half_size, cx - half_size : cx + half_size
            ].copy()

            if template.shape != (template_size, template_size):
                logger.warning(
                    f"Template {i} has wrong shape {template.shape}, skipping"
                )
                continue

            self.templates.append(TemplateRegion(template=template, position=(cx, cy)))

        logger.info(f"Extracted {len(self.templates)} templates from detected keypoints")

    def check_validity(self, frame: np.ndarray) -> bool:
        """Check if current homography is still valid via template matching.

        Uses normalized cross-correlation (NCC) to match templates against
        current frame. If ≥60% of templates fail to match, camera shift is detected.

        Args:
            frame: Current frame (BGR).

        Returns:
            True if homography is still valid, False if camera shift detected.
        """
        if not self.templates:
            logger.warning("No templates available for validation")
            return True

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        failures = 0
        total = 0

        for i, tmpl_region in enumerate(self.templates):
            cx, cy = tmpl_region.position
            template = tmpl_region.template
            template_size = template.shape[0]
            half_size = template_size // 2

            # Define search region (±10 pixels around expected position)
            search_margin = 10
            x1 = max(0, cx - half_size - search_margin)
            y1 = max(0, cy - half_size - search_margin)
            x2 = min(w, cx + half_size + search_margin)
            y2 = min(h, cy + half_size + search_margin)

            search_region = gray[y1:y2, x1:x2]

            if (search_region.size == 0
                    or search_region.shape[0] < template.shape[0]
                    or search_region.shape[1] < template.shape[1]):
                continue

            # Match template
            result = cv2.matchTemplate(
                search_region, template, cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            total += 1
            if max_val < self.ncc_threshold:
                failures += 1
                logger.debug(
                    f"Template {i} at ({cx}, {cy}): NCC={max_val:.3f} < "
                    f"{self.ncc_threshold:.3f} (FAIL)"
                )
            else:
                logger.debug(
                    f"Template {i} at ({cx}, {cy}): NCC={max_val:.3f} (OK)"
                )

        if total == 0:
            logger.warning("No templates could be matched")
            return True

        failure_ratio = failures / total
        consensus_failed = failure_ratio >= self.consensus_ratio

        logger.info(
            f"Template matching: {failures}/{total} failures "
            f"({failure_ratio * 100:.1f}%) → "
            f"{'SHIFT DETECTED' if consensus_failed else 'Valid'}"
        )

        return not consensus_failed

    def update(
        self, frame: np.ndarray, frame_number: int
    ) -> np.ndarray | None:
        """Called every frame. Returns new homography if camera shifted, else None.

        Logic:
        1. Every `validation_interval` frames, run check_validity()
        2. If shift detected:
           a. Wait cooldown_frames for motion blur to settle
           b. Run court_detector.detect_frame() on current frame
           c. If ≥4 keypoints: compute new homography, extract new templates, return it
           d. If fails: retry every max_retry_interval frames
           e. If fails for disable_after_seconds: log warning, return special DISABLE signal
        3. If valid: return None (keep using current homography)

        Args:
            frame: Current frame (BGR).
            frame_number: Current frame number.

        Returns:
            - None: homography is still valid, keep using current one
            - np.ndarray: new homography matrix (camera shifted and re-detected)
            - np.array([], dtype=np.float64): DISABLE signal (detection failed for too long)
        """
        # Once disabled, stay disabled
        if self._disabled:
            return None

        # Check if we should validate
        if frame_number - self._last_check_frame < self.validation_interval:
            # Not time to check yet
            # But if we're in cooldown or retry mode, handle that
            if self._shift_detected_frame is not None:
                frames_since_shift = frame_number - self._shift_detected_frame

                # Cooldown period
                if frames_since_shift < self.cooldown_frames:
                    logger.debug(
                        f"In cooldown: {frames_since_shift}/{self.cooldown_frames}"
                    )
                    return None

                # Try re-detection
                if (
                    frames_since_shift >= self.cooldown_frames
                    and frame_number - self._last_retry_frame >= self.max_retry_interval
                ):
                    return self._attempt_redetection(frame, frame_number)

                # Check if we should disable
                if self._detection_failed_time is not None:
                    elapsed = time.time() - self._detection_failed_time
                    if elapsed > self.disable_after_seconds:
                        logger.warning(
                            f"Court detection failed for {elapsed:.1f}s, "
                            f"disabling boundary filter"
                        )
                        self._disabled = True
                        return np.array([], dtype=np.float64)  # DISABLE signal

            return None

        # Time to check validity
        self._last_check_frame = frame_number
        is_valid = self.check_validity(frame)

        if is_valid:
            # Reset all failure/shift tracking
            self._detection_failed_time = None
            self._shift_detected_frame = None
            return None

        # Shift detected
        logger.warning(f"Camera shift detected at frame {frame_number}")
        self._shift_detected_frame = frame_number
        return None  # Will retry after cooldown

    def _attempt_redetection(
        self, frame: np.ndarray, frame_number: int
    ) -> np.ndarray | None:
        """Attempt to re-detect the court and update homography.

        Args:
            frame: Current frame (BGR).
            frame_number: Current frame number.

        Returns:
            New homography matrix if successful, None otherwise.
        """
        self._last_retry_frame = frame_number
        logger.info(f"Attempting court re-detection at frame {frame_number}...")

        try:
            result: CourtDetectionResult = self.court_detector.detect_frame(
                frame, frame_number=frame_number
            )

            if result.detected:
                # Success!
                logger.info(
                    f"Re-detection successful: {result.num_detected}/14 keypoints, "
                    f"confidence={result.confidence:.3f}"
                )

                # Update homography and templates
                self.current_homography = result.homography_img_to_court
                self.extract_templates(frame, result.keypoints)

                # Reset state
                self._shift_detected_frame = None
                self._detection_failed_time = None

                return result.homography_img_to_court
            else:
                # Failed
                logger.warning(
                    f"Re-detection failed: only {result.num_detected}/14 keypoints"
                )

                # Start failure timer if not already started
                if self._detection_failed_time is None:
                    self._detection_failed_time = time.time()

                return None

        except Exception as e:
            logger.error(f"Re-detection exception: {e}")
            if self._detection_failed_time is None:
                self._detection_failed_time = time.time()
            return None
