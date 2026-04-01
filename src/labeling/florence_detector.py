"""Florence-2 zero-shot ball detector for the labeling pipeline.

Uses ``microsoft/Florence-2-large`` from HuggingFace with the
``<CAPTION_TO_PHRASE_GROUNDING>`` task and the text prompt "tennis ball"
to locate the ball in each frame.
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
        self._dtype = None

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
        self._dtype = torch_dtype

    def detect(self, frame: np.ndarray) -> Optional[tuple[float, float, float]]:
        """Detect tennis ball using phrase grounding.

        Uses the ``<CAPTION_TO_PHRASE_GROUNDING>`` task with the text
        prompt ``"tennis ball"`` so Florence-2 searches specifically for
        a tennis ball rather than relying on the fixed ``<OD>`` vocabulary
        (which never includes "ball").
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded — call load() first")

        import torch
        from PIL import Image

        # Convert BGR (OpenCV) → RGB PIL image
        rgb = frame[:, :, ::-1]
        pil_img = Image.fromarray(rgb)

        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        prompt = task + "tennis ball"
        inputs = self._processor(text=prompt, images=pil_img, return_tensors="pt")
        # Cast float tensors to model dtype to avoid float32/float16 mismatch
        inputs = {
            k: v.to(self._device, dtype=self._dtype)
            if v.is_floating_point()
            else v.to(self._device)
            for k, v in inputs.items()
        }

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

        detections = result.get(task, {})
        bboxes = detections.get("bboxes", [])
        labels = detections.get("labels", [])

        best: Optional[tuple[float, float, float]] = None
        smallest_area = float("inf")

        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            # Pick the smallest detection (tennis ball is small)
            if area < smallest_area:
                smallest_area = area
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                # Florence-2 does not provide per-detection confidence
                # scores.  Use a fixed value so downstream consumers
                # (e.g. consensus engine) don't mistake this for a
                # calibrated probability.
                best = (cx, cy, 0.5)

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
