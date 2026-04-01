"""YOLO-World detector with manual SAHI-style tiled inference.

Slices each frame into overlapping tiles, runs YOLO-World on each tile
plus the full frame, offsets detections back into full-frame coordinates,
applies NMS, and returns the highest-confidence result.  This improves
recall for small objects (tennis ball at far court: 3–8 px in 1080p).

Manual tiling is used instead of sahi.AutoDetectionModel because
YOLO-World requires ``set_classes()`` before inference, which the SAHI
AutoDetectionModel wrapper does not propagate.
"""

import logging
from typing import Optional

import numpy as np

from src.labeling.models import BaseDetector

logger = logging.getLogger(__name__)

_DEFAULT_CONF = 0.003
_DEFAULT_SLICE = 640
_DEFAULT_OVERLAP = 0.25
_NMS_IOU_THRESH = 0.5
_FULL_FRAME_IMGSZ = 1280


def _compute_tiles(
    frame_h: int,
    frame_w: int,
    slice_h: int,
    slice_w: int,
    overlap_h: float,
    overlap_w: float,
) -> list[tuple[int, int, int, int]]:
    """Return (x1, y1, x2, y2) tile coordinates covering the full frame.

    Tiles are placed on a regular grid with the requested overlap; the last
    tile in each row/column is snapped to the frame edge so no pixel is
    missed.
    """
    stride_w = max(1, int(slice_w * (1.0 - overlap_w)))
    stride_h = max(1, int(slice_h * (1.0 - overlap_h)))

    tiles: list[tuple[int, int, int, int]] = []
    y = 0
    while True:
        x = 0
        y2 = min(y + slice_h, frame_h)
        y1 = y2 - slice_h  # may be negative if frame smaller than slice
        y1 = max(0, y1)
        while True:
            x2 = min(x + slice_w, frame_w)
            x1 = max(0, x2 - slice_w)
            tiles.append((x1, y1, x2, y2))
            if x2 >= frame_w:
                break
            x += stride_w
        if y2 >= frame_h:
            break
        y += stride_h

    return tiles


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xi1 = max(box_a[0], box_b[0])
    yi1 = max(box_a[1], box_b[1])
    xi2 = min(box_a[2], box_b[2])
    yi2 = min(box_a[3], box_b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(
    boxes: list[np.ndarray],
    scores: list[float],
    iou_thresh: float,
) -> list[int]:
    """Greedy NMS; returns indices of kept boxes sorted by descending score."""
    if not boxes:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    kept: list[int] = []
    suppressed = [False] * len(order)
    for i, idx in enumerate(order):
        if suppressed[i]:
            continue
        kept.append(idx)
        for j in range(i + 1, len(order)):
            if suppressed[j]:
                continue
            if _iou(boxes[idx], boxes[order[j]]) > iou_thresh:
                suppressed[j] = True
    return kept


class YOLOWorldSAHIDetector(BaseDetector):
    """YOLO-World with manual SAHI-style tiled inference.

    Runs YOLO-World on overlapping image tiles AND on the full frame,
    merges detections with NMS, and returns the highest-confidence ball.
    """

    def __init__(
        self,
        model_size: str = "l",
        conf: float = _DEFAULT_CONF,
        slice_size: int = _DEFAULT_SLICE,
        overlap: float = _DEFAULT_OVERLAP,
    ) -> None:
        """
        Args:
            model_size: YOLO-World variant ('s', 'm', or 'l').
            conf: Minimum confidence threshold for detections.
            slice_size: Tile edge length in pixels (used for both H and W).
            overlap: Fractional overlap between adjacent tiles (0–1).
        """
        self._model_size = model_size
        self._conf = conf
        self._slice_size = slice_size
        self._overlap = overlap
        self._model = None
        self._device: str = "cpu"

    @property
    def name(self) -> str:
        return "yolo_world_sahi"

    def load(self, device: str = "cpu") -> None:
        """Load YOLO-World and set classes to 'tennis ball'."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "YOLO-World SAHI requires the 'ultralytics' package. "
                "Install with: pip install ultralytics"
            ) from exc

        model_name = f"yolov8{self._model_size}-worldv2.pt"
        logger.info(
            "Loading YOLO-World SAHI (%s) onto %s  [conf=%.4f, slice=%d, overlap=%.2f]",
            model_name, device, self._conf, self._slice_size, self._overlap,
        )
        self._model = YOLO(model_name)
        self._model.set_classes(["tennis ball"])
        self._device = device

    def _predict_tile(
        self,
        tile: np.ndarray,
        offset_x: int,
        offset_y: int,
        imgsz: int | None = None,
    ) -> tuple[list[np.ndarray], list[float]]:
        """Run YOLO-World on a single tile and return full-frame boxes + scores."""
        results = self._model.predict(
            tile,
            device=self._device,
            verbose=False,
            conf=self._conf,
            imgsz=imgsz or self._slice_size,
        )
        boxes_out: list[np.ndarray] = []
        scores_out: list[float] = []
        if not results or len(results[0].boxes) == 0:
            return boxes_out, scores_out

        for xyxy, conf in zip(
            results[0].boxes.xyxy.cpu().numpy(),
            results[0].boxes.conf.cpu().numpy(),
        ):
            # Translate tile-local coordinates to full-frame coordinates.
            full_box = np.array([
                xyxy[0] + offset_x,
                xyxy[1] + offset_y,
                xyxy[2] + offset_x,
                xyxy[3] + offset_y,
            ], dtype=np.float32)
            boxes_out.append(full_box)
            scores_out.append(float(conf))

        return boxes_out, scores_out

    def detect(self, frame: np.ndarray) -> Optional[tuple[float, float, float]]:
        """Tile the frame, run YOLO-World on each tile + full frame, return best hit."""
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")

        frame_h, frame_w = frame.shape[:2]
        all_boxes: list[np.ndarray] = []
        all_scores: list[float] = []

        # --- tiled inference ---
        tiles = _compute_tiles(
            frame_h, frame_w,
            self._slice_size, self._slice_size,
            self._overlap, self._overlap,
        )
        for x1, y1, x2, y2 in tiles:
            tile = frame[y1:y2, x1:x2]
            b, s = self._predict_tile(tile, x1, y1)
            all_boxes.extend(b)
            all_scores.extend(s)

        # --- full-frame inference (catches large/close balls) ---
        full_boxes, full_scores = self._predict_tile(
            frame, 0, 0, imgsz=_FULL_FRAME_IMGSZ,
        )
        all_boxes.extend(full_boxes)
        all_scores.extend(full_scores)

        if not all_boxes:
            return None

        # --- NMS to merge overlapping detections from different tiles ---
        kept = _nms(all_boxes, all_scores, _NMS_IOU_THRESH)
        best_idx = max(kept, key=lambda i: all_scores[i])

        conf = all_scores[best_idx]
        box = all_boxes[best_idx]
        cx = float((box[0] + box[2]) / 2.0)
        cy = float((box[1] + box[3]) / 2.0)

        return (cx, cy, conf)

    def unload(self) -> None:
        """Release YOLO-World SAHI model from memory."""
        import torch

        if self._model is not None:
            del self._model
            self._model = None
        torch.cuda.empty_cache()
        logger.info("YOLO-World SAHI unloaded")
