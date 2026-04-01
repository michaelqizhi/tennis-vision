"""Tests for the YOLO-World SAHI tiled inference detector.

All tests mock the underlying YOLO model so they run without GPU or
downloaded weights.
"""

from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.labeling.yoloworld_sahi_detector import (
    YOLOWorldSAHIDetector,
    _compute_tiles,
    _iou,
    _nms,
)


# ---------------------------------------------------------------------------
# _compute_tiles
# ---------------------------------------------------------------------------

class TestComputeTiles:
    def test_small_frame_yields_one_tile(self):
        """A frame smaller than the slice size produces a single tile."""
        tiles = _compute_tiles(320, 480, 640, 640, 0.25, 0.25)
        assert len(tiles) == 1
        x1, y1, x2, y2 = tiles[0]
        assert x1 == 0 and y1 == 0
        assert x2 == 480 and y2 == 320

    def test_exact_fit_frame(self):
        """A frame exactly equal to slice size produces one tile."""
        tiles = _compute_tiles(640, 640, 640, 640, 0.25, 0.25)
        assert len(tiles) == 1

    def test_tiles_cover_full_frame(self):
        """Every pixel must be covered by at least one tile."""
        frame_h, frame_w = 1080, 1920
        slice_size = 640
        overlap = 0.25
        tiles = _compute_tiles(frame_h, frame_w, slice_size, slice_size, overlap, overlap)

        coverage = np.zeros((frame_h, frame_w), dtype=np.int32)
        for x1, y1, x2, y2 in tiles:
            coverage[y1:y2, x1:x2] += 1

        assert coverage.min() >= 1, "Some pixels are not covered by any tile"

    def test_overlap_region_covered_twice(self):
        """Overlapping tiles should cover the overlap region more than once."""
        frame_h, frame_w = 1080, 1920
        tiles = _compute_tiles(frame_h, frame_w, 640, 640, 0.25, 0.25)

        coverage = np.zeros((frame_h, frame_w), dtype=np.int32)
        for x1, y1, x2, y2 in tiles:
            coverage[y1:y2, x1:x2] += 1

        assert coverage.max() >= 2, "Expected some pixels covered by multiple tiles"

    def test_tiles_stay_within_frame(self):
        """All tile coordinates must be within frame bounds."""
        frame_h, frame_w = 1080, 1920
        tiles = _compute_tiles(frame_h, frame_w, 640, 640, 0.25, 0.25)
        for x1, y1, x2, y2 in tiles:
            assert 0 <= x1 < x2 <= frame_w
            assert 0 <= y1 < y2 <= frame_h

    def test_multiple_tiles_for_large_frame(self):
        """A 1920x1080 frame with 640-px tiles must produce >1 tile."""
        tiles = _compute_tiles(1080, 1920, 640, 640, 0.25, 0.25)
        assert len(tiles) > 1

    def test_zero_overlap(self):
        """Zero overlap still covers the full frame."""
        frame_h, frame_w = 1080, 1920
        tiles = _compute_tiles(frame_h, frame_w, 640, 640, 0.0, 0.0)
        coverage = np.zeros((frame_h, frame_w), dtype=np.int32)
        for x1, y1, x2, y2 in tiles:
            coverage[y1:y2, x1:x2] += 1
        assert coverage.min() >= 1


# ---------------------------------------------------------------------------
# _iou
# ---------------------------------------------------------------------------

class TestIoU:
    def test_identical_boxes(self):
        box = np.array([0.0, 0.0, 10.0, 10.0])
        assert _iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = np.array([0.0, 0.0, 5.0, 5.0])
        b = np.array([10.0, 10.0, 20.0, 20.0])
        assert _iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = np.array([0.0, 0.0, 10.0, 10.0])
        b = np.array([5.0, 5.0, 15.0, 15.0])
        # intersection = 5x5=25, union = 100+100-25=175
        assert _iou(a, b) == pytest.approx(25.0 / 175.0, rel=1e-4)

    def test_contained_box(self):
        outer = np.array([0.0, 0.0, 10.0, 10.0])
        inner = np.array([2.0, 2.0, 8.0, 8.0])
        # intersection = 36, union = 100
        assert _iou(outer, inner) == pytest.approx(36.0 / 100.0, rel=1e-4)


# ---------------------------------------------------------------------------
# _nms
# ---------------------------------------------------------------------------

class TestNMS:
    def test_empty_input(self):
        assert _nms([], [], 0.5) == []

    def test_single_box(self):
        boxes = [np.array([0.0, 0.0, 10.0, 10.0])]
        scores = [0.9]
        kept = _nms(boxes, scores, 0.5)
        assert kept == [0]

    def test_non_overlapping_boxes_all_kept(self):
        boxes = [
            np.array([0.0, 0.0, 5.0, 5.0]),
            np.array([10.0, 10.0, 20.0, 20.0]),
        ]
        scores = [0.9, 0.8]
        kept = _nms(boxes, scores, 0.5)
        assert set(kept) == {0, 1}

    def test_overlapping_low_score_suppressed(self):
        """Two heavily overlapping boxes — only the higher-score one kept."""
        boxes = [
            np.array([0.0, 0.0, 10.0, 10.0]),  # score 0.9, kept
            np.array([1.0, 1.0, 11.0, 11.0]),  # score 0.5, suppressed
        ]
        scores = [0.9, 0.5]
        kept = _nms(boxes, scores, 0.5)
        assert kept == [0]

    def test_ordering_by_score(self):
        """Higher-score box wins when two boxes heavily overlap."""
        boxes = [
            np.array([0.0, 0.0, 10.0, 10.0]),  # score 0.5 — should be suppressed
            np.array([1.0, 1.0, 11.0, 11.0]),  # score 0.95 — should be kept
        ]
        scores = [0.5, 0.95]
        kept = _nms(boxes, scores, 0.5)
        assert 1 in kept
        assert 0 not in kept


# ---------------------------------------------------------------------------
# YOLOWorldSAHIDetector — instantiation & interface
# ---------------------------------------------------------------------------

class TestYOLOWorldSAHIDetectorInterface:
    def test_name(self):
        d = YOLOWorldSAHIDetector()
        assert d.name == "yolo_world_sahi"

    def test_repr(self):
        d = YOLOWorldSAHIDetector()
        assert "YOLOWorldSAHIDetector" in repr(d)

    def test_default_params(self):
        d = YOLOWorldSAHIDetector()
        assert d._model_size == "l"
        assert d._conf == pytest.approx(0.003)
        assert d._slice_size == 640
        assert d._overlap == pytest.approx(0.25)

    def test_custom_params(self):
        d = YOLOWorldSAHIDetector(model_size="s", conf=0.01, slice_size=320, overlap=0.2)
        assert d._model_size == "s"
        assert d._conf == pytest.approx(0.01)
        assert d._slice_size == 320
        assert d._overlap == pytest.approx(0.2)

    def test_detect_without_load_raises(self):
        d = YOLOWorldSAHIDetector()
        with pytest.raises(RuntimeError, match="not loaded"):
            d.detect(np.zeros((360, 640, 3), dtype=np.uint8))

    def test_unload_without_load(self):
        """Unloading before loading must not raise."""
        d = YOLOWorldSAHIDetector()
        d.unload()  # should not raise


# ---------------------------------------------------------------------------
# YOLOWorldSAHIDetector — detect() with mocked YOLO
# ---------------------------------------------------------------------------

def _make_mock_yolo_no_detection():
    """Return a mock YOLO model that always reports no detections."""
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.boxes = MagicMock()
    mock_result.boxes.__len__ = MagicMock(return_value=0)
    mock_model.predict.return_value = [mock_result]
    return mock_model


def _make_mock_yolo_with_detection(cx: float, cy: float, conf: float, box_size: float = 10.0):
    """Return a mock YOLO model that detects a ball at (cx, cy) with given conf."""
    import torch

    mock_model = MagicMock()

    def _predict(frame, **kwargs):
        h, w = frame.shape[:2]
        x1 = max(0.0, cx - box_size / 2)
        y1 = max(0.0, cy - box_size / 2)
        x2 = min(float(w), cx + box_size / 2)
        y2 = min(float(h), cy + box_size / 2)
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.__len__ = MagicMock(return_value=1)
        mock_result.boxes.xyxy = torch.tensor([[x1, y1, x2, y2]])
        mock_result.boxes.conf = torch.tensor([conf])
        return [mock_result]

    mock_model.predict.side_effect = _predict
    return mock_model


class TestYOLOWorldSAHIDetect:
    def _make_detector_with_mock(self, mock_model) -> YOLOWorldSAHIDetector:
        d = YOLOWorldSAHIDetector()
        d._model = mock_model
        d._device = "cpu"
        return d

    def test_returns_none_when_no_detections(self):
        d = self._make_detector_with_mock(_make_mock_yolo_no_detection())
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        result = d.detect(frame)
        assert result is None

    def test_returns_tuple_of_three_floats(self):
        """detect() must return (cx, cy, conf) as three floats."""
        d = self._make_detector_with_mock(
            _make_mock_yolo_with_detection(cx=100.0, cy=200.0, conf=0.8)
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        result = d.detect(frame)
        assert result is not None
        assert len(result) == 3
        cx, cy, conf = result
        assert isinstance(cx, float)
        assert isinstance(cy, float)
        assert isinstance(conf, float)

    def test_conf_in_valid_range(self):
        d = self._make_detector_with_mock(
            _make_mock_yolo_with_detection(cx=100.0, cy=200.0, conf=0.75)
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        _, _, conf = d.detect(frame)
        assert 0.0 <= conf <= 1.0

    def test_detection_position_approximately_correct(self):
        """Detected centre should be close to the injected ball position."""
        target_cx, target_cy = 320.0, 180.0
        d = self._make_detector_with_mock(
            _make_mock_yolo_with_detection(cx=target_cx, cy=target_cy, conf=0.9)
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        result = d.detect(frame)
        assert result is not None
        cx, cy, _ = result
        assert abs(cx - target_cx) < 10.0
        assert abs(cy - target_cy) < 10.0

    def test_predict_called_multiple_times(self):
        """Tiling should cause predict() to be called more than once on 1080p."""
        mock_model = _make_mock_yolo_no_detection()
        d = self._make_detector_with_mock(mock_model)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        d.detect(frame)
        # Tiled inference + full-frame inference = >1 call
        assert mock_model.predict.call_count > 1

    def test_nms_deduplicates_tile_border_detections(self):
        """Duplicate detections at tile boundaries must be merged to one result."""
        import torch

        # Simulate a ball near a tile boundary.  Two adjacent tiles both see
        # it at the same full-frame position.  We inject a detection at a
        # fixed full-frame location (500, 300) which falls in the overlap
        # zone between tile-0 (0,0,640,640) and tile-1 (480,0,1120,640).
        # Both tiles should produce nearly identical full-frame boxes, and
        # NMS must suppress the duplicate.
        ball_full_x, ball_full_y = 500.0, 300.0
        box_half = 5.0

        tiles = _compute_tiles(1080, 1920, 640, 640, 0.25, 0.25)

        def _predict(frame, **kwargs):
            h, w = frame.shape[:2]
            # Figure out which tile this is by matching the shape/offset.
            # We return a detection only if the ball's local coords are
            # inside this tile crop.
            mock_result = MagicMock()
            mock_result.boxes = MagicMock()
            # Find the tile whose crop dimensions match this frame.
            for tx1, ty1, tx2, ty2 in tiles:
                tw, th = tx2 - tx1, ty2 - ty1
                if tw == w and th == h:
                    local_x = ball_full_x - tx1
                    local_y = ball_full_y - ty1
                    if 0 <= local_x < w and 0 <= local_y < h:
                        mock_result.boxes.__len__ = MagicMock(return_value=1)
                        mock_result.boxes.xyxy = torch.tensor(
                            [[local_x - box_half, local_y - box_half,
                              local_x + box_half, local_y + box_half]]
                        )
                        mock_result.boxes.conf = torch.tensor([0.7])
                        return [mock_result]
            # No detection for this tile / full-frame
            mock_result.boxes.__len__ = MagicMock(return_value=0)
            return [mock_result]

        mock_model = MagicMock()
        mock_model.predict.side_effect = _predict

        d = self._make_detector_with_mock(mock_model)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = d.detect(frame)
        assert result is not None
        cx, cy, conf = result
        # After NMS the position should match the injected ball
        assert abs(cx - ball_full_x) < 1.0
        assert abs(cy - ball_full_y) < 1.0

    def test_unload_clears_model(self):
        d = YOLOWorldSAHIDetector()
        d._model = MagicMock()
        with patch("torch.cuda.empty_cache") as mock_cache:
            d.unload()
        assert d._model is None
        mock_cache.assert_called_once()


# ---------------------------------------------------------------------------
# get_default_detectors includes YOLOWorldSAHIDetector
# ---------------------------------------------------------------------------

class TestGetDefaultDetectors:
    def test_sahi_in_default_detectors(self):
        """get_default_detectors() must include a YOLOWorldSAHIDetector."""
        from src.labeling.inference_runner import get_default_detectors
        detectors = get_default_detectors()
        names = [d.name for d in detectors]
        assert "yolo_world_sahi" in names

    def test_tracknetv4_not_in_default_detectors(self):
        """TrackNetV4 should have been removed from the default list."""
        from src.labeling.inference_runner import get_default_detectors
        detectors = get_default_detectors()
        names = [d.name for d in detectors]
        assert "tracknet_v4" not in names

    def test_exactly_five_detectors(self):
        """Default list should contain exactly 5 detectors."""
        from src.labeling.inference_runner import get_default_detectors
        assert len(get_default_detectors()) == 5
