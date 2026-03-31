"""YOLO-World zero-shot ball detector for the labeling pipeline.

Uses ``ultralytics`` to run YOLO-World with the prompt "tennis ball".
"""

import logging
from typing import Optional

import numpy as np

from src.labeling.models import BaseDetector

logger = logging.getLogger(__name__)


class YOLOWorldDetector(BaseDetector):
    """YOLO-World open-vocabulary tennis ball detector.

    Downloads the model on first use via the ``ultralytics`` package.
    """

    def __init__(self, model_size: str = "l") -> None:
        """
        Args:
            model_size: YOLO-World model variant ('s', 'm', or 'l').
        """
        self._model_size = model_size
        self._model = None
        self._device: str = "cpu"

    @property
    def name(self) -> str:
        return "yolo_world"

    def load(self, device: str = "cpu") -> None:
        """Load YOLO-World model."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "YOLO-World requires the 'ultralytics' package. "
                "Install with: pip install ultralytics"
            ) from exc

        model_name = f"yolov8{self._model_size}-worldv2.pt"
        logger.info("Loading YOLO-World (%s) onto %s", model_name, device)
        self._model = YOLO(model_name)
        self._model.set_classes(["tennis ball"])
        self._device = device

    def detect(self, frame: np.ndarray) -> Optional[tuple[float, float, float]]:
        """Detect tennis ball using YOLO-World.

        Returns the highest-confidence 'tennis ball' detection.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")

        results = self._model.predict(
            frame,
            device=self._device,
            verbose=False,
            conf=0.1,
        )

        if not results or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes
        best_idx = int(boxes.conf.argmax())
        conf = float(boxes.conf[best_idx])
        xyxy = boxes.xyxy[best_idx].cpu().numpy()
        cx = float((xyxy[0] + xyxy[2]) / 2.0)
        cy = float((xyxy[1] + xyxy[3]) / 2.0)

        return (cx, cy, conf)

    def unload(self) -> None:
        """Release YOLO-World model from memory."""
        import torch

        if self._model is not None:
            del self._model
            self._model = None
        torch.cuda.empty_cache()
        logger.info("YOLO-World unloaded")
