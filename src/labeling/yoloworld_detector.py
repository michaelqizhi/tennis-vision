"""YOLO-World zero-shot ball detector for the labeling pipeline.

Uses ``ultralytics`` to run YOLO-World with the prompt "tennis ball".

Note on tuning: YOLO-World produces very low confidence scores for small
objects like tennis balls in 1080p footage (often 0.01–0.05 at the default
640 image size).  Two settings are critical for acceptable recall:

* **imgsz=1280** – processes the frame at higher resolution so the model can
  actually resolve a ~15-pixel tennis ball.  Confidences jump from ~0.03 to
  0.2–0.6 compared to the default 640.
* **conf ≈ 0.005** – even at 1280 some frames still score below 0.01.
  A threshold of 0.005 provides a good recall/precision trade-off for the
  downstream pipeline which already applies temporal smoothing.
"""

import logging
from typing import Optional

import numpy as np

from src.labeling.models import BaseDetector

logger = logging.getLogger(__name__)

# Defaults tuned for 1080p courtside tennis footage.
_DEFAULT_CONF = 0.005
_DEFAULT_IMGSZ = 1280


class YOLOWorldDetector(BaseDetector):
    """YOLO-World open-vocabulary tennis ball detector.

    Downloads the model on first use via the ``ultralytics`` package.
    """

    def __init__(
        self,
        model_size: str = "l",
        conf: float = _DEFAULT_CONF,
        imgsz: int = _DEFAULT_IMGSZ,
    ) -> None:
        """
        Args:
            model_size: YOLO-World model variant ('s', 'm', or 'l').
            conf: Minimum confidence threshold for detections.
            imgsz: Inference image size (longer edge). Use 1280 for 1080p
                   video to avoid down-scaling small objects beyond
                   recognition.
        """
        self._model_size = model_size
        self._conf = conf
        self._imgsz = imgsz
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
        logger.info(
            "Loading YOLO-World (%s) onto %s  [conf=%.4f, imgsz=%d]",
            model_name, device, self._conf, self._imgsz,
        )
        self._model = YOLO(model_name)
        # set_classes must be called *before* the model is moved to GPU
        # to avoid a device-mismatch bug in the CLIP text encoder.
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
            conf=self._conf,
            imgsz=self._imgsz,
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
