"""TrackNet V4 detector wrapper for the labeling pipeline.

TrackNet V4 uses a motion-attention-enhanced U-Net architecture from
``AnInsomniacy/tracknet-series-pytorch``. Key differences from V2:

  - Input resolution: 288×512 (vs 360×640 for V2)
  - Output: 3-channel sigmoid heatmap (one per input frame)
  - Motion Prompt module extracts attention from frame differences
  - Pretrained weights: ``tracknet-v4_best-model.pth``

Like the V2 wrapper, this maintains a 3-frame sliding window so the
runner can call ``detect(frame)`` one frame at a time.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.labeling.models import BaseDetector

logger = logging.getLogger(__name__)

# TrackNet V4 inference resolution (must match training config)
_INPUT_W = 512
_INPUT_H = 288
_DETECTION_THRESHOLD = 0.5

# Post-processing tuning for false-positive reduction.
# V4 was trained on badminton and produces diffuse activations on tennis
# players.  A real tennis ball at 512×288 is ≤10 px; player blobs are
# typically 20-60 px.  We keep only the smallest, sharpest component.
_MAX_BLOB_AREA = 30   # reject blobs larger than a ball
_MIN_PEAK = 0.55      # require confident activation within the component

# Where to download / find weights
_WEIGHTS_FILENAME = "tracknet_v4.pth"
_WEIGHTS_URL = (
    "https://github.com/AnInsomniacy/tracknet-v4-pytorch/releases/download/v1.0.1/tracknet-v4_best-model.pth"
)


# ── Model architecture (copied from AnInsomniacy/tracknet-series-pytorch) ──


class _MotionPrompt(nn.Module):
    """Extract motion attention from consecutive frame differences."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        B, T, C, H, W = x.shape
        gray = x.mean(dim=2)
        diffs = [gray[:, i + 1] - gray[:, i] for i in range(T - 1)]
        motion_diff = torch.stack(diffs, dim=1)
        attention = torch.sigmoid(self.a * motion_diff.abs() + self.b)
        motion_features = attention.unsqueeze(2).expand(B, 2, 3, H, W)
        return motion_features, attention, None


class _MotionFusion(nn.Module):
    """Fuse visual features with motion attention."""

    def forward(self, visual: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            visual[:, 0],
            motion[:, 0] * visual[:, 1],
            motion[:, 1] * visual[:, 2],
        ], dim=1)


class TrackNetV4Model(nn.Module):
    """Motion-Enhanced U-Net for Sports Object Tracking (TrackNet V4)."""

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.motion_prompt = _MotionPrompt()
        self.motion_fusion = _MotionFusion()

        self.enc1 = self._conv_block(9, 64, 2)
        self.enc2 = self._conv_block(64, 128, 2)
        self.enc3 = self._conv_block(128, 256, 3)
        self.enc4 = self._conv_block(256, 512, 3)

        self.dec1 = self._conv_block(768, 256, 3)
        self.dec2 = self._conv_block(384, 128, 2)
        self.dec3 = self._conv_block(192, 64, 2)

        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(dropout)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.output = nn.Conv2d(64, 3, 1)

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int, n_layers: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for i in range(n_layers):
            ch_in = in_ch if i == 0 else out_ch
            layers.extend([
                nn.Conv2d(ch_in, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        _, motion_mask, _ = self.motion_prompt(x.view(B, 3, 3, _INPUT_H, _INPUT_W))

        e1 = self.enc1(x)
        e1_pool = self.pool(e1)
        e2 = self.enc2(e1_pool)
        e2_pool = self.pool(e2)
        e3 = self.enc3(e2_pool)
        e3_pool = self.pool(e3)
        bottleneck = self.enc4(e3_pool)
        bottleneck = self.dropout(bottleneck)

        d1 = self.upsample(bottleneck)
        d1 = self.dec1(torch.cat([d1, e3], dim=1))
        d2 = self.upsample(d1)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d3 = self.upsample(d2)
        d3 = self.dec3(torch.cat([d3, e1], dim=1))

        visual_output = self.output(d3)
        enhanced_output = self.motion_fusion(visual_output, motion_mask)
        return torch.sigmoid(enhanced_output)


# ── Detector wrapper ──────────────────────────────────────────────────


def _download_weights(dest: Path) -> None:
    """Download TrackNet V4 pretrained weights if not present."""
    if dest.exists():
        return

    logger.info("Downloading TrackNet V4 weights to %s ...", dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        import urllib.request
        urllib.request.urlretrieve(_WEIGHTS_URL, str(dest))
        logger.info("Download complete: %s", dest)
    except Exception:
        logger.exception("Failed to download TrackNet V4 weights from %s", _WEIGHTS_URL)
        raise RuntimeError(
            f"Could not download TrackNet V4 weights. "
            f"Please download manually from {_WEIGHTS_URL} "
            f"and place at {dest}"
        )


class TrackNetV4Detector(BaseDetector):
    """TrackNet V4 ball detector with motion attention.

    Maintains a sliding window of 3 frames internally so the
    inference runner can call ``detect(frame)`` per-frame.
    """

    def __init__(self, weights_path: str | None = None) -> None:
        self._model: TrackNetV4Model | None = None
        self._device: str = "cpu"
        self._buffer: deque[np.ndarray] = deque(maxlen=3)
        self._weights_path = weights_path

    @property
    def name(self) -> str:
        return "tracknet_v4"

    def load(self, device: str = "cpu") -> None:
        """Load TrackNet V4 weights."""
        if self._weights_path:
            weights = Path(self._weights_path)
        else:
            from src.config import get_config
            cfg = get_config()
            weights = cfg.weights_dir / _WEIGHTS_FILENAME

        _download_weights(weights)

        logger.info("Loading TrackNet V4 from %s onto %s", weights, device)

        self._model = TrackNetV4Model()
        state_dict = torch.load(str(weights), map_location=device, weights_only=False)

        # Handle full training checkpoints (keys: model_state_dict, optimizer_state_dict, etc.)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        # Handle DDP-wrapped checkpoints (keys prefixed with 'module.')
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

        self._model.load_state_dict(state_dict)
        self._model.to(device)
        self._model.eval()
        self._device = device
        self._buffer.clear()

    def detect(self, frame: np.ndarray) -> Optional[tuple[float, float, float]]:
        """Detect ball using a 3-frame sliding window.

        Returns ``None`` for the first two frames (insufficient context).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")

        self._buffer.append(frame)
        if len(self._buffer) < 3:
            return None

        orig_h, orig_w = frame.shape[:2]
        scale_x = orig_w / _INPUT_W
        scale_y = orig_h / _INPUT_H

        # Resize and convert BGR→RGB (model was trained on RGB)
        resized = [
            cv2.cvtColor(cv2.resize(self._buffer[i], (_INPUT_W, _INPUT_H)), cv2.COLOR_BGR2RGB)
            for i in range(3)
        ]

        # Concatenate 3 frames → 9 channels (RGB × 3)
        imgs = np.concatenate(resized, axis=2)  # (H, W, 9)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)  # (9, H, W)
        inp = np.expand_dims(imgs, axis=0)  # (1, 9, H, W)

        with torch.no_grad():
            out = self._model(torch.from_numpy(inp).to(self._device))

        # Output shape: (1, 3, H, W) — use channel 2 (most recent frame)
        heatmap = out[0, 2].cpu().numpy()

        # Threshold and find centroid of the sharpest component
        x, y, conf = self._peak_component_centroid(heatmap, scale_x, scale_y)
        if x is None:
            return None
        return (x, y, conf)

    @staticmethod
    def _peak_component_centroid(
        heatmap: np.ndarray,
        scale_x: float,
        scale_y: float,
        threshold: float = _DETECTION_THRESHOLD,
    ) -> tuple[float | None, float | None, float]:
        """Extract ball position using the smallest qualifying component.

        Instead of computing a centroid over ALL above-threshold pixels
        (which drifts toward large diffuse player-region activations),
        this method:
        1. Finds connected components in the thresholded heatmap.
        2. Rejects components larger than ``_MAX_BLOB_AREA`` (player blobs).
        3. Rejects components whose peak activation < ``_MIN_PEAK``.
        4. Among survivors, picks the **smallest** — a real tennis ball
           produces a tiny, bright blob; player limbs are larger.
        5. Returns the centroid of only that component.
        """
        peak = float(heatmap.max())
        if peak < threshold:
            return None, None, 0.0

        binary = (heatmap > threshold).astype(np.uint8) * 255
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        # Evaluate each component (skip label 0 = background)
        best_cx: float | None = None
        best_cy: float | None = None
        best_area = _MAX_BLOB_AREA + 1  # start larger than any acceptable
        best_peak = 0.0

        for lbl in range(1, n_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]

            if area > _MAX_BLOB_AREA:
                continue

            comp_mask = labels == lbl
            comp_peak = float(heatmap[comp_mask].max())

            if comp_peak < _MIN_PEAK:
                continue

            # Prefer the smallest qualifying component (most ball-like)
            if area < best_area or (area == best_area and comp_peak > best_peak):
                best_area = area
                best_cx = centroids[lbl][0]
                best_cy = centroids[lbl][1]
                best_peak = comp_peak

        if best_cx is None:
            return None, None, 0.0

        x = float(best_cx * scale_x)
        y = float(best_cy * scale_y)
        return x, y, best_peak

    def unload(self) -> None:
        """Release model and clear frame buffer."""
        if self._model is not None:
            del self._model
            self._model = None
        self._buffer.clear()
        torch.cuda.empty_cache()
        logger.info("TrackNet V4 unloaded")
