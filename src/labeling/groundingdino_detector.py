"""Grounding DINO zero-shot ball detector for the labeling pipeline.

Uses ``IDEA-Research/grounding-dino-base`` from HuggingFace with a
period-separated text prompt to locate the tennis ball in each frame.
Grounding DINO achieves 54.3 zero-shot COCO AP and excels at small
object detection.
"""

import logging
from typing import Optional

import numpy as np

from src.labeling.models import BaseDetector

logger = logging.getLogger(__name__)


class GroundingDINODetector(BaseDetector):
    """Grounding DINO zero-shot tennis ball detector.

    Downloads the model on first use from HuggingFace Hub (~900 MB).
    Requires ``transformers>=4.49.0`` and ``Pillow``.

    Note:
        Grounding DINO uses period-separated text prompts, e.g.
        ``"tennis ball."`` — the trailing period is required for the
        tokenizer to correctly parse phrase boundaries.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        prompt: str = "tennis ball.",
        conf_threshold: float = 0.1,
        box_threshold: float = 0.1,
    ) -> None:
        self._model_id = model_id
        self._prompt = prompt
        self._conf_threshold = conf_threshold
        self._box_threshold = box_threshold
        self._model = None
        self._processor = None
        self._device: str = "cpu"
        self._dtype = None

    @property
    def name(self) -> str:
        return "grounding_dino"

    def load(self, device: str = "cpu") -> None:
        """Load Grounding DINO from HuggingFace Hub."""
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        except ImportError as exc:
            raise ImportError(
                "Grounding DINO requires the 'transformers' package (>=4.49.0). "
                "Install with: pip install transformers"
            ) from exc

        import torch

        logger.info(
            "Loading Grounding DINO from %s onto %s", self._model_id, device
        )
        self._processor = AutoProcessor.from_pretrained(self._model_id)
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self._model_id, torch_dtype=torch_dtype
        ).to(device)
        self._model.eval()
        self._device = device
        self._dtype = torch_dtype
        logger.info("Grounding DINO loaded (dtype=%s)", torch_dtype)

    def detect(self, frame: np.ndarray) -> Optional[tuple[float, float, float]]:
        """Detect tennis ball using Grounding DINO text-grounded detection.

        Args:
            frame: BGR image as a numpy array (H, W, 3), uint8.

        Returns:
            Tuple of (cx, cy, confidence) in pixel coordinates, or
            ``None`` if no ball is detected above the configured thresholds.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded — call load() first")

        import torch
        from PIL import Image

        # Convert BGR (OpenCV) → RGB PIL image
        rgb = frame[:, :, ::-1]
        pil_img = Image.fromarray(rgb)

        inputs = self._processor(
            images=pil_img,
            text=self._prompt,
            return_tensors="pt",
        )
        # Cast float tensors to model dtype to avoid float32/float16 mismatch
        inputs = {
            k: v.to(self._device, dtype=self._dtype)
            if v.is_floating_point()
            else v.to(self._device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            # Grounding DINO's text backbone (BERT) internally upcasts to
            # float32 for LayerNorm/softmax, producing float32 features that
            # crash against float16 encoder Linear layers.  Autocast handles
            # the per-op dtype negotiation automatically.
            if self._device == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=self._dtype):
                    outputs = self._model(**inputs)
            else:
                outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self._box_threshold,
            text_threshold=self._conf_threshold,
            target_sizes=[(pil_img.height, pil_img.width)],
        )

        if not results or len(results[0]["boxes"]) == 0:
            return None

        boxes = results[0]["boxes"]   # (N, 4) xyxy
        scores = results[0]["scores"]  # (N,)

        best_idx = int(scores.argmax())
        box = boxes[best_idx].cpu().numpy()
        conf = float(scores[best_idx])
        cx = float((box[0] + box[2]) / 2.0)
        cy = float((box[1] + box[3]) / 2.0)
        return (cx, cy, conf)

    def unload(self) -> None:
        """Release Grounding DINO model from memory."""
        import torch

        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        torch.cuda.empty_cache()
        logger.info("Grounding DINO unloaded")
