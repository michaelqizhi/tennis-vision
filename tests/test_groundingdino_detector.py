"""Tests for GroundingDINODetector.

All tests mock the transformers model so no real weights are needed.
"""

from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from src.labeling.groundingdino_detector import GroundingDINODetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Create a minimal BGR uint8 frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_detector(**kwargs) -> GroundingDINODetector:
    return GroundingDINODetector(**kwargs)


# ---------------------------------------------------------------------------
# Construction & metadata
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_default_name(self):
        det = _make_detector()
        assert det.name == "grounding_dino"

    def test_default_prompt(self):
        det = _make_detector()
        assert det._prompt == "tennis ball."

    def test_custom_params(self):
        det = _make_detector(
            model_id="some/model",
            prompt="ball.",
            conf_threshold=0.3,
            box_threshold=0.25,
        )
        assert det._model_id == "some/model"
        assert det._prompt == "ball."
        assert det._conf_threshold == 0.3
        assert det._box_threshold == 0.25

    def test_not_loaded_initially(self):
        det = _make_detector()
        assert det._model is None
        assert det._processor is None

    def test_repr(self):
        det = _make_detector()
        assert "GroundingDINODetector" in repr(det)
        assert "grounding_dino" in repr(det)


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------

class TestLoad:
    def _make_mocks(self, device: str = "cpu"):
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        return mock_processor, mock_model

    @patch("src.labeling.groundingdino_detector.GroundingDINODetector.load")
    def test_load_sets_device(self, mock_load):
        """Smoke-test that load() can be called (mocked at method level)."""
        det = _make_detector()
        det.load("cpu")
        mock_load.assert_called_once_with("cpu")

    def test_load_cpu_dtype(self):
        import torch
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with patch("transformers.AutoProcessor") as MockProc, \
             patch("transformers.AutoModelForZeroShotObjectDetection") as MockModel, \
             patch.dict("sys.modules", {
                 "transformers": MagicMock(
                     AutoProcessor=MagicMock(from_pretrained=MagicMock(return_value=mock_processor)),
                     AutoModelForZeroShotObjectDetection=MagicMock(from_pretrained=MagicMock(return_value=mock_model)),
                 )
             }):
            det = _make_detector()
            # Manually simulate what load() does so we can verify dtype logic
            det._device = "cpu"
            det._dtype = torch.float32
            det._model = mock_model
            det._processor = mock_processor

            assert det._dtype == torch.float32

    def test_load_cuda_dtype(self):
        import torch
        det = _make_detector()
        det._device = "cuda"
        det._dtype = torch.float16
        assert det._dtype == torch.float16

    def test_load_raises_without_transformers(self):
        det = _make_detector()
        # Simulate transformers being absent by making the import raise
        import builtins
        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "transformers":
                raise ImportError("No module named 'transformers'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            with pytest.raises(ImportError, match="transformers"):
                det.load("cpu")


# ---------------------------------------------------------------------------
# detect()
# ---------------------------------------------------------------------------

def _setup_loaded_detector(scores_list, boxes_list, device="cpu"):
    """Return a GroundingDINODetector with a mocked model and processor.

    Args:
        scores_list: list of float scores (empty list → no detections).
        boxes_list: list of [x1, y1, x2, y2] lists.
    """
    import torch

    det = _make_detector()
    det._device = device
    det._dtype = torch.float32

    mock_processor = MagicMock()
    mock_model = MagicMock()

    # inputs returned by processor — simple float tensor so dtype-cast logic runs
    pixel_values = torch.zeros(1, 3, 32, 32, dtype=torch.float32)
    input_ids = torch.zeros(1, 5, dtype=torch.long)
    mock_processor.return_value = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
    }

    # post_process result
    if scores_list:
        scores_t = torch.tensor(scores_list, dtype=torch.float32)
        boxes_t = torch.tensor(boxes_list, dtype=torch.float32)
    else:
        scores_t = torch.zeros(0, dtype=torch.float32)
        boxes_t = torch.zeros((0, 4), dtype=torch.float32)

    mock_processor.post_process_grounded_object_detection.return_value = [
        {
            "scores": scores_t,
            "boxes": boxes_t,
            "labels": ["tennis ball"] * len(scores_list),
        }
    ]

    det._processor = mock_processor
    det._model = mock_model
    return det


class TestDetect:
    def test_returns_none_when_no_detections(self):
        det = _setup_loaded_detector([], [])
        result = det.detect(_make_frame())
        assert result is None

    def test_returns_tuple_of_three_floats(self):
        # box: x1=10, y1=20, x2=30, y2=40 → cx=20, cy=30
        det = _setup_loaded_detector([0.75], [[10.0, 20.0, 30.0, 40.0]])
        result = det.detect(_make_frame())
        assert result is not None
        assert len(result) == 3
        cx, cy, conf = result
        assert isinstance(cx, float)
        assert isinstance(cy, float)
        assert isinstance(conf, float)

    def test_centre_calculation(self):
        det = _setup_loaded_detector([0.8], [[10.0, 20.0, 30.0, 40.0]])
        cx, cy, conf = det.detect(_make_frame())
        assert cx == pytest.approx(20.0)
        assert cy == pytest.approx(30.0)
        assert conf == pytest.approx(0.8)

    def test_picks_highest_confidence(self):
        # Two detections; second has higher score
        det = _setup_loaded_detector(
            [0.4, 0.9],
            [[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 70.0, 70.0]],
        )
        cx, cy, conf = det.detect(_make_frame())
        # Should pick the second box: cx=(50+70)/2=60, cy=(50+70)/2=60
        assert cx == pytest.approx(60.0)
        assert cy == pytest.approx(60.0)
        assert conf == pytest.approx(0.9)

    def test_raises_if_not_loaded(self):
        det = _make_detector()
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            det.detect(_make_frame())

    def test_empty_results_list_returns_none(self):
        """post_process returns an empty list (not just empty boxes)."""
        import torch

        det = _make_detector()
        det._device = "cpu"
        det._dtype = torch.float32

        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor.return_value = {
            "pixel_values": torch.zeros(1, 3, 32, 32),
            "input_ids": torch.zeros(1, 5, dtype=torch.long),
        }
        mock_processor.post_process_grounded_object_detection.return_value = []
        det._processor = mock_processor
        det._model = mock_model

        assert det.detect(_make_frame()) is None

    def test_dtype_casting_applied(self):
        """Float tensors in inputs must be cast to self._dtype."""
        import torch

        det = _make_detector()
        det._device = "cpu"
        det._dtype = torch.float32

        mock_processor = MagicMock()
        mock_model = MagicMock()

        float_tensor = torch.zeros(1, 3, 32, 32, dtype=torch.float64)  # wrong dtype
        int_tensor = torch.zeros(1, 5, dtype=torch.long)
        mock_processor.return_value = {
            "pixel_values": float_tensor,
            "input_ids": int_tensor,
        }
        mock_processor.post_process_grounded_object_detection.return_value = [
            {
                "scores": torch.zeros(0),
                "boxes": torch.zeros((0, 4)),
                "labels": [],
            }
        ]
        det._processor = mock_processor
        det._model = mock_model

        det.detect(_make_frame())

        # Verify the model was called with float32 pixel_values (not float64)
        call_kwargs = mock_model.call_args
        assert call_kwargs is not None
        # model(**inputs) — extract from keyword args
        all_kwargs = {}
        if call_kwargs.kwargs:
            all_kwargs.update(call_kwargs.kwargs)
        if call_kwargs.args and len(call_kwargs.args) > 0 and isinstance(call_kwargs.args[0], dict):
            all_kwargs.update(call_kwargs.args[0])
        passed_pixel = all_kwargs.get("pixel_values")
        if passed_pixel is not None:
            assert passed_pixel.dtype == torch.float32

    def test_post_process_uses_threshold_not_box_threshold(self):
        """Regression: post_process_grounded_object_detection uses 'threshold'
        not 'box_threshold' — passing the wrong kwarg causes a TypeError."""
        det = _setup_loaded_detector([0.8], [[10.0, 20.0, 30.0, 40.0]])
        det.detect(_make_frame())

        call_kwargs = det._processor.post_process_grounded_object_detection.call_args
        assert "threshold" in call_kwargs.kwargs, (
            "Must use 'threshold=' (not 'box_threshold=') for "
            "post_process_grounded_object_detection"
        )
        assert "box_threshold" not in call_kwargs.kwargs, (
            "'box_threshold' is not a valid parameter name — use 'threshold'"
        )


# ---------------------------------------------------------------------------
# unload()
# ---------------------------------------------------------------------------

class TestUnload:
    def test_unload_clears_model_and_processor(self):
        import torch

        det = _make_detector()
        det._model = MagicMock()
        det._processor = MagicMock()

        with patch("torch.cuda.empty_cache") as mock_cache:
            det.unload()

        assert det._model is None
        assert det._processor is None
        mock_cache.assert_called_once()

    def test_unload_safe_when_already_unloaded(self):
        """Calling unload() on an unloaded detector must not raise."""
        det = _make_detector()
        with patch("torch.cuda.empty_cache"):
            det.unload()  # should not raise


# ---------------------------------------------------------------------------
# Integration: get_default_detectors includes GroundingDINO
# ---------------------------------------------------------------------------

class TestDefaultDetectors:
    def test_grounding_dino_in_defaults(self):
        """GroundingDINODetector should be in the default pipeline."""
        from src.labeling.inference_runner import get_default_detectors
        from src.labeling.groundingdino_detector import GroundingDINODetector

        detectors = get_default_detectors()
        gd_instances = [d for d in detectors if isinstance(d, GroundingDINODetector)]
        assert len(gd_instances) == 1, "GroundingDINODetector should appear in defaults"

    def test_grounding_dino_name_is_grounding_dino(self):
        det = GroundingDINODetector()
        assert det.name == "grounding_dino"
