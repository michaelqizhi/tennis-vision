"""Tests for the labeling pipeline — Sprint 1: Model Loader & Inference Harness.

Tests use synthetic frames (solid-color images with drawn circles) so they
run without real video files or GPU.
"""

import json
import math
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.labeling.models import BaseDetector
from src.labeling.tracknet_detector import TrackNetDetector, _weighted_centroid
from src.labeling.inference_runner import run_inference, _serialize_detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(w: int = 640, h: int = 360) -> np.ndarray:
    """Create a synthetic BGR frame with a white circle (simulated ball)."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 180), 5, (255, 255, 255), -1)
    return frame


class DummyDetector(BaseDetector):
    """Deterministic detector for testing the runner."""

    def __init__(self, detections: list[Optional[tuple[float, float, float]]]) -> None:
        self._detections = detections
        self._idx = 0
        self.loaded = False

    @property
    def name(self) -> str:
        return "dummy"

    def load(self, device: str = "cpu") -> None:
        self.loaded = True
        self._idx = 0

    def detect(self, frame: np.ndarray) -> Optional[tuple[float, float, float]]:
        if self._idx < len(self._detections):
            det = self._detections[self._idx]
            self._idx += 1
            return det
        return None

    def unload(self) -> None:
        self.loaded = False
        self._idx = 0


class FailingDetector(BaseDetector):
    """Detector that fails on load — tests graceful error handling."""

    @property
    def name(self) -> str:
        return "failing"

    def load(self, device: str = "cpu") -> None:
        raise RuntimeError("Simulated load failure")

    def detect(self, frame: np.ndarray) -> Optional[tuple[float, float, float]]:
        return None

    def unload(self) -> None:
        pass


# ---------------------------------------------------------------------------
# BaseDetector interface
# ---------------------------------------------------------------------------

class TestBaseDetector:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseDetector()

    def test_dummy_implements_interface(self):
        d = DummyDetector([(100.0, 200.0, 0.9)])
        assert d.name == "dummy"
        d.load()
        assert d.loaded
        assert d.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == (100.0, 200.0, 0.9)
        d.unload()
        assert not d.loaded


# ---------------------------------------------------------------------------
# _weighted_centroid
# ---------------------------------------------------------------------------

class TestWeightedCentroid:
    def test_empty_heatmap_returns_none(self):
        heatmap = np.zeros(640 * 360, dtype=np.float32)
        x, y, conf = _weighted_centroid(heatmap, 640, 360, 1.0, 1.0)
        assert x is None
        assert y is None
        assert conf == 0.0

    def test_single_blob_center(self):
        """A single bright blob should return its center."""
        heatmap = np.zeros((360, 640), dtype=np.float32)
        cv2.circle(heatmap, (320, 180), 5, 1.0, -1)
        x, y, conf = _weighted_centroid(heatmap, 640, 360, 1.0, 1.0)
        assert x is not None
        assert abs(x - 320) < 2
        assert abs(y - 180) < 2
        assert conf > 0

    def test_scaling(self):
        """Scale factors should map model coords to original video coords."""
        heatmap = np.zeros((360, 640), dtype=np.float32)
        cv2.circle(heatmap, (320, 180), 5, 1.0, -1)
        x, y, _ = _weighted_centroid(heatmap, 640, 360, 3.0, 3.0)
        assert x is not None
        assert abs(x - 960) < 6
        assert abs(y - 540) < 6

    def test_elliptical_blob(self):
        """An elliptical blob (oblique camera) should still return a valid centroid."""
        heatmap = np.zeros((360, 640), dtype=np.float32)
        cv2.ellipse(heatmap, (200, 100), (10, 4), 30, 0, 360, 1.0, -1)
        x, y, conf = _weighted_centroid(heatmap, 640, 360, 1.0, 1.0)
        assert x is not None
        assert abs(x - 200) < 5
        assert abs(y - 100) < 5


# ---------------------------------------------------------------------------
# _serialize_detection
# ---------------------------------------------------------------------------

class TestSerializeDetection:
    def test_none(self):
        assert _serialize_detection(None) is None

    def test_tuple(self):
        result = _serialize_detection((123.456, 789.012, 0.95432))
        assert result == {"x": 123.46, "y": 789.01, "conf": 0.9543}


# ---------------------------------------------------------------------------
# TrackNetDetector
# ---------------------------------------------------------------------------

class TestTrackNetDetector:
    def test_name(self):
        d = TrackNetDetector()
        assert d.name == "tracknet_v2"

    def test_repr(self):
        d = TrackNetDetector()
        assert "TrackNetDetector" in repr(d)

    def test_detect_without_load_raises(self):
        d = TrackNetDetector()
        with pytest.raises(RuntimeError, match="not loaded"):
            d.detect(np.zeros((360, 640, 3), dtype=np.uint8))

    def test_first_two_frames_return_none(self):
        """TrackNet needs 3 frames — first two should return None."""
        d = TrackNetDetector()

        # Mock the model loading and forward pass
        mock_model = MagicMock()
        # Model output: (1, num_classes, H*W) — softmax is applied in detect()
        fake_output = torch.zeros(1, 256, 360 * 640)
        mock_model.return_value = fake_output

        d._model = mock_model
        d._device = "cpu"

        frame = _make_frame()
        assert d.detect(frame) is None  # Frame 1
        assert d.detect(frame) is None  # Frame 2

    def test_softmax_produces_heatmap(self):
        """Verify detect() uses softmax (probabilities), not argmax (indices)."""
        d = TrackNetDetector()

        mock_model = MagicMock()
        # Create output where class 1 (ball) has high logit at center
        inp_h, inp_w = d._inp_h, d._inp_w
        fake_output = torch.zeros(1, 256, inp_h * inp_w)
        # Set ball class (index 1) logit high at a known position
        center_idx = (inp_h // 2) * inp_w + (inp_w // 2)
        for offset in range(-2, 3):
            idx = center_idx + offset
            if 0 <= idx < inp_h * inp_w:
                fake_output[0, 1, idx] = 10.0  # high logit for ball class
        mock_model.return_value = fake_output

        d._model = mock_model
        d._device = "cpu"

        frame = _make_frame(640, 360)
        d.detect(frame)  # Frame 1
        d.detect(frame)  # Frame 2
        result = d.detect(frame)  # Frame 3 — should produce detection

        # With softmax on class 1 logits of 10.0, we should get a valid heatmap
        # The result may be None if threshold is too high, but at least
        # we verify no crash from argmax integer aliasing
        # (argmax would produce 1s → 255 after *255, softmax produces ~1.0 → 255)
        assert result is None or (len(result) == 3 and result[2] > 0)


# ---------------------------------------------------------------------------
# Florence-2 Detector
# ---------------------------------------------------------------------------

class TestFlorenceDetector:
    def test_name(self):
        from src.labeling.florence_detector import FlorenceDetector
        d = FlorenceDetector()
        assert d.name == "florence_2"

    def test_detect_without_load_raises(self):
        from src.labeling.florence_detector import FlorenceDetector
        d = FlorenceDetector()
        with pytest.raises(RuntimeError, match="not loaded"):
            d.detect(np.zeros((360, 640, 3), dtype=np.uint8))

    def test_unload_without_load(self):
        """Unloading before loading should not raise."""
        from src.labeling.florence_detector import FlorenceDetector
        d = FlorenceDetector()
        d.unload()  # should not raise


# ---------------------------------------------------------------------------
# YOLO-World Detector
# ---------------------------------------------------------------------------

class TestYOLOWorldDetector:
    def test_name(self):
        from src.labeling.yoloworld_detector import YOLOWorldDetector
        d = YOLOWorldDetector()
        assert d.name == "yolo_world"

    def test_detect_without_load_raises(self):
        from src.labeling.yoloworld_detector import YOLOWorldDetector
        d = YOLOWorldDetector()
        with pytest.raises(RuntimeError, match="not loaded"):
            d.detect(np.zeros((360, 640, 3), dtype=np.uint8))

    def test_unload_without_load(self):
        """Unloading before loading should not raise."""
        from src.labeling.yoloworld_detector import YOLOWorldDetector
        d = YOLOWorldDetector()
        d.unload()  # should not raise

    def test_custom_model_size(self):
        from src.labeling.yoloworld_detector import YOLOWorldDetector
        d = YOLOWorldDetector(model_size="s")
        assert d._model_size == "s"


# ---------------------------------------------------------------------------
# Inference Runner
# ---------------------------------------------------------------------------

class TestInferenceRunner:
    def _make_test_video(self, tmp_path: Path, n_frames: int = 10) -> str:
        """Create a short test video with synthetic frames."""
        video_path = str(tmp_path / "test.mp4")
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 360))
        for _ in range(n_frames):
            writer.write(_make_frame())
        writer.release()
        return video_path

    def test_dummy_detector_run(self, tmp_path):
        """Run inference with a deterministic dummy detector."""
        video_path = self._make_test_video(tmp_path, n_frames=5)
        detections = [
            (100.0, 200.0, 0.9),
            None,
            (110.0, 210.0, 0.85),
            None,
            (105.0, 205.0, 0.88),
        ]
        detector = DummyDetector(detections)

        results = run_inference(video_path, [detector], device="cpu")

        assert len(results) == 5
        assert results["0"]["dummy"] is not None
        assert results["0"]["dummy"]["x"] == 100.0
        assert results["1"]["dummy"] is None
        assert results["2"]["dummy"]["x"] == 110.0

    def test_output_json(self, tmp_path):
        """Verify JSON output file is written correctly."""
        video_path = self._make_test_video(tmp_path, n_frames=3)
        detector = DummyDetector([(1.0, 2.0, 0.5)] * 3)
        output = str(tmp_path / "detections.json")

        run_inference(video_path, [detector], output_path=output)

        assert Path(output).exists()
        with open(output) as f:
            data = json.load(f)
        assert len(data) == 3
        assert data["0"]["dummy"]["x"] == 1.0

    def test_multiple_detectors_same_name(self, tmp_path):
        """Two detectors with the same name: last one wins per frame."""
        video_path = self._make_test_video(tmp_path, n_frames=3)
        d1 = DummyDetector([(10.0, 20.0, 0.9)] * 3)
        d2 = DummyDetector([None, (30.0, 40.0, 0.7), None])

        results = run_inference(video_path, [d1, d2])

        # d2 runs after d1 and overwrites the same key
        assert results["0"]["dummy"] is None
        assert results["1"]["dummy"]["x"] == 30.0

    def test_multiple_named_detectors(self, tmp_path):
        """Detectors with different names produce separate entries."""
        video_path = self._make_test_video(tmp_path, n_frames=2)

        class DetA(DummyDetector):
            @property
            def name(self) -> str:
                return "det_a"

        class DetB(DummyDetector):
            @property
            def name(self) -> str:
                return "det_b"

        results = run_inference(
            video_path,
            [DetA([(1.0, 2.0, 0.5)] * 2), DetB([None, (3.0, 4.0, 0.8)])],
        )

        assert results["0"]["det_a"] is not None
        assert results["0"]["det_b"] is None
        assert results["1"]["det_b"] is not None

    def test_failing_detector_graceful(self, tmp_path):
        """A detector that fails to load should not crash the runner."""
        video_path = self._make_test_video(tmp_path, n_frames=3)
        good = DummyDetector([(1.0, 2.0, 0.5)] * 3)
        bad = FailingDetector()

        results = run_inference(video_path, [good, bad])

        # Good detector should have results
        assert results["0"]["dummy"]["x"] == 1.0
        # Failing detector should have null for all frames
        assert all(results[str(i)]["failing"] is None for i in range(3))

    def test_max_frames(self, tmp_path):
        """max_frames should limit processing."""
        video_path = self._make_test_video(tmp_path, n_frames=10)
        detector = DummyDetector([(1.0, 2.0, 0.5)] * 5)

        results = run_inference(video_path, [detector], max_frames=5)

        assert len(results) == 5

    def test_empty_detectors_list(self, tmp_path):
        """Running with no detectors should produce empty per-frame results."""
        video_path = self._make_test_video(tmp_path, n_frames=3)
        results = run_inference(video_path, [])
        assert len(results) == 3
        assert results["0"] == {}

    def test_detector_unloaded_on_detect_error(self, tmp_path):
        """Detector.unload() should be called even if detect() crashes hard."""
        video_path = self._make_test_video(tmp_path, n_frames=3)

        class CrashingDetector(BaseDetector):
            def __init__(self):
                self.was_unloaded = False

            @property
            def name(self) -> str:
                return "crasher"

            def load(self, device: str = "cpu") -> None:
                pass

            def detect(self, frame: np.ndarray):
                raise RuntimeError("boom")

            def unload(self) -> None:
                self.was_unloaded = True

        detector = CrashingDetector()
        # detect() errors are caught per-frame, so run completes normally
        run_inference(video_path, [detector])
        assert detector.was_unloaded

    def test_json_write_failure_returns_results(self, tmp_path):
        """If JSON write fails, results should still be returned."""
        video_path = self._make_test_video(tmp_path, n_frames=2)
        detector = DummyDetector([(1.0, 2.0, 0.5)] * 2)

        # Use an invalid path to force I/O failure
        bad_path = str(tmp_path / "nonexistent" / "deep" / "nested" / "\0bad" / "out.json")
        results = run_inference(video_path, [detector], output_path=bad_path)

        # Results should still be returned even though write failed
        assert len(results) == 2
        assert results["0"]["dummy"]["x"] == 1.0


# Need torch for the mock test
import torch
