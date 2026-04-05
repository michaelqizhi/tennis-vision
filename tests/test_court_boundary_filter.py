"""Smoke tests for court boundary filtering system."""

import numpy as np
import pytest

from src.features.court_detect.frame_selector import (
    compute_court_color_score,
    compute_sharpness,
)
from src.features.court_detect.boundary_filter import CourtBoundaryFilter
from src.features.court_detect.homography_monitor import (
    HomographyMonitor,
    TemplateRegion,
)
from src.features.ball_tracking.detector import BallDetection
from src.features.court_detect.detector import CourtDetector
from src.features.court_detect.homography import compute_pixel_to_court_homography


class TestFrameSelector:
    """Test frame selector functions."""

    def test_compute_court_color_score_blue_court(self):
        """Test that a blue court frame gets a reasonable score."""
        # Create a fake blue court frame (640x360)
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        # Fill middle region with blue (HSV: hue ~100)
        frame[72:288, 128:512] = [200, 100, 50]  # BGR blue
        
        score = compute_court_color_score(frame)
        assert 0.0 <= score <= 1.0
        assert score > 0.05  # Should detect some blue

    def test_compute_court_color_score_green_court(self):
        """Test that a green court frame gets a reasonable score."""
        # Create a fake green court frame
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[72:288, 128:512] = [50, 150, 50]  # BGR green
        
        score = compute_court_color_score(frame)
        assert 0.0 <= score <= 1.0
        assert score > 0.05  # Should detect some green

    def test_compute_court_color_score_garbage_frame(self):
        """Test that a garbage frame (black) gets low score."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        score = compute_court_color_score(frame)
        assert 0.0 <= score <= 1.0
        assert score < 0.1  # Should be very low

    def test_compute_sharpness_sharp_frame(self):
        """Test that a sharp frame (checkerboard) gets high score."""
        # Create checkerboard pattern (high frequency = sharp)
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[::2, ::2] = 255
        frame[1::2, 1::2] = 255
        
        sharpness = compute_sharpness(frame)
        assert sharpness > 100.0  # Should be sharp

    def test_compute_sharpness_blurry_frame(self):
        """Test that a blurry frame (uniform) gets low score."""
        # Uniform gray = very blurry
        frame = np.full((360, 640, 3), 128, dtype=np.uint8)
        
        sharpness = compute_sharpness(frame)
        assert sharpness < 10.0  # Should be very low


class TestCourtBoundaryFilter:
    """Test court boundary filter."""

    def test_is_in_bounds_center_court(self):
        """Test that center court point is in bounds."""
        # Create identity homography (pixel = meters, for testing)
        H = np.eye(3, dtype=np.float64)
        
        filter = CourtBoundaryFilter(homography=H, x_margin=3.0, y_margin=5.0)
        
        # Point at (0, 0) in court coords should be in bounds
        assert filter.is_in_bounds(0.0, 0.0)

    def test_is_in_bounds_near_baseline(self):
        """Test that point near baseline is in bounds."""
        # Create identity homography
        H = np.eye(3, dtype=np.float64)
        
        filter = CourtBoundaryFilter(homography=H, x_margin=3.0, y_margin=5.0)
        
        # Point at (0, 11.0) should be in bounds (baseline is at 11.885m)
        assert filter.is_in_bounds(0.0, 11.0)

    def test_is_in_bounds_far_outside(self):
        """Test that point far outside court is out of bounds."""
        # Create identity homography
        H = np.eye(3, dtype=np.float64)
        
        filter = CourtBoundaryFilter(homography=H, x_margin=3.0, y_margin=5.0)
        
        # Point at (20, 30) should be way out of bounds
        # x_bound = 5.485 + 3.0 = 8.485
        # y_bound = 11.885 + 5.0 = 16.885
        assert not filter.is_in_bounds(20.0, 30.0)

    def test_filter_detections_keeps_in_bounds(self):
        """Test that in-bounds detections are kept."""
        # Create identity homography
        H = np.eye(3, dtype=np.float64)
        
        filter = CourtBoundaryFilter(homography=H, x_margin=3.0, y_margin=5.0)
        
        # Detection at center court
        detections = [
            BallDetection(frame_number=0, x=0.0, y=0.0, confidence=0.9),
        ]
        
        filtered = filter.filter_detections(detections)
        assert len(filtered) == 1
        assert filtered[0].detected
        assert filtered[0].x == 0.0
        assert filtered[0].y == 0.0

    def test_filter_detections_rejects_out_of_bounds(self):
        """Test that out-of-bounds detections are rejected."""
        # Create identity homography
        H = np.eye(3, dtype=np.float64)
        
        filter = CourtBoundaryFilter(homography=H, x_margin=3.0, y_margin=5.0)
        
        # Detection way outside court
        detections = [
            BallDetection(frame_number=0, x=100.0, y=100.0, confidence=0.9),
        ]
        
        filtered = filter.filter_detections(detections)
        assert len(filtered) == 1
        assert not filtered[0].detected
        assert filtered[0].x is None
        assert filtered[0].y is None
        assert filtered[0].confidence == 0.0

    def test_filter_detections_preserves_already_not_detected(self):
        """Test that already non-detected detections are preserved."""
        H = np.eye(3, dtype=np.float64)
        filter = CourtBoundaryFilter(homography=H)
        
        detections = [
            BallDetection(frame_number=0, x=None, y=None, confidence=0.0),
        ]
        
        filtered = filter.filter_detections(detections)
        assert len(filtered) == 1
        assert not filtered[0].detected


class TestHomographyMonitor:
    """Test homography monitor."""

    def test_extract_templates(self):
        """Test template extraction from keypoints."""
        # Create a fake detector and homography
        detector = CourtDetector()
        H = np.eye(3, dtype=np.float64)
        
        monitor = HomographyMonitor(court_detector=detector, current_homography=H)
        
        # Create fake frame
        frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        
        # Create fake keypoints (some detected, some not)
        keypoints = [
            (100.0, 100.0),  # 0
            (200.0, 100.0),  # 1
            (None, None),    # 2 - not detected
            (100.0, 200.0),  # 3
            *[(None, None)] * 10,  # Rest not detected
        ]
        
        monitor.extract_templates(frame, keypoints)
        
        # Should have 3 templates (only non-None keypoints)
        assert len(monitor.templates) == 3
        
        # Check template properties
        for tmpl in monitor.templates:
            assert isinstance(tmpl, TemplateRegion)
            assert tmpl.template.shape == (48, 48)
            assert isinstance(tmpl.position, tuple)
            assert len(tmpl.position) == 2

    def test_check_validity_no_templates(self):
        """Test that check_validity returns True when no templates."""
        detector = CourtDetector()
        H = np.eye(3, dtype=np.float64)
        
        monitor = HomographyMonitor(court_detector=detector, current_homography=H)
        
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        
        # No templates extracted yet
        assert monitor.check_validity(frame)

    def test_check_validity_same_frame(self):
        """Test that check_validity returns True for the same frame."""
        detector = CourtDetector()
        H = np.eye(3, dtype=np.float64)
        
        monitor = HomographyMonitor(
            court_detector=detector, 
            current_homography=H,
            ncc_threshold=0.65,
        )
        
        # Create frame with distinctive pattern
        frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        
        keypoints = [
            (200.0, 200.0),
            (400.0, 200.0),
            *[(None, None)] * 12,
        ]
        
        monitor.extract_templates(frame, keypoints)
        
        # Check validity on same frame should pass (high correlation)
        is_valid = monitor.check_validity(frame)
        assert is_valid

    def test_update_skips_early_frames(self):
        """Test that update skips frames before validation_interval."""
        detector = CourtDetector()
        H = np.eye(3, dtype=np.float64)
        
        monitor = HomographyMonitor(
            court_detector=detector,
            current_homography=H,
            validation_interval=30,
        )
        
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        
        # Frame 0-29 should return None (no check yet)
        for i in range(29):
            result = monitor.update(frame, frame_number=i)
            assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
