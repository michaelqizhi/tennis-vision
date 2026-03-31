"""Florence-2 zero-shot ball detector for the labeling pipeline.

Uses ``microsoft/Florence-2-large`` from HuggingFace with the
``<OD>`` (object detection) task and filters results for "tennis ball".
"""

import logging
from typing import Optional

import numpy as np

from src.labeling.models import BaseDetector

logger = logging.getLogger(__name__)


class FlorenceDetector(BaseDetector):
    """Florence-2 zero-shot tennis ball detector.

    Downloads the model on first use from HuggingFace Hub.
    Requires ``transformers`` and ``Pillow``.
    """

    def __init__(self, model_id: str = "microsoft/Florence-2-large") -> None:
        self._model_id = model_id
        self._model = None
        self._processor = None
        self._device: str = "cpu"

    @property
    def name(self) -> str:
        return "florence_2"

    def load(self, device: str = "cpu") -> None:
        """Load Florence-2 from HuggingFace Hub."""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Florence-2 requires the 'transformers' package. "
                "Install with: pip install transformers"
            ) from exc

        import torch

        logger.info("Loading Florence-2 from %s onto %s", self._model_id, device)
        self._processor = AutoProcessor.from_pretrained(
            self._model_id, trust_remote_code=True,
        )
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(device)
        self._model.eval()
        self._device = device

    def detect(self, frame: np.ndarray) -> Optional[tuple[float, float, float]]:
        """Detect tennis ball using open-vocabulary object detection.

        Uses the ``<OD>`` task prompt. Filters detections for labels
        containing 'ball'.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded — call load() first")

        import torch
        from PIL import Image

        # Convert BGR (OpenCV) → RGB PIL image
        rgb = frame[:, :, ::-1]
        pil_img = Image.fromarray(rgb)

        task = "<OD>"
        inputs = self._processor(text=task, images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False,
        )[0]
        result = self._processor.post_process_generation(
            generated_text, task=task, image_size=(pil_img.width, pil_img.height),
        )

        # result[task] has 'bboxes' and 'labels'
        detections = result.get(task, {})
        bboxes = detections.get("bboxes", [])
        labels = detections.get("labels", [])

        best: Optional[tuple[float, float, float]] = None
        smallest_area = float("inf")

        for bbox, label in zip(bboxes, labels):
            if "ball" not in label.lower():
                continue
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            # Pick the smallest ball-like detection (tennis ball is small)
            if area < smallest_area:
                smallest_area = area
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                # Confidence proxy: inverse of area (smaller = more likely tennis ball)
                conf = max(0.0, min(1.0, 1.0 - area / (frame.shape[0] * frame.shape[1])))
                best = (cx, cy, conf)

        return best

    def unload(self) -> None:
        """Release Florence-2 model from memory."""
        import torch

        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        torch.cuda.empty_cache()
        logger.info("Florence-2 unloaded")
