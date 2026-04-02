"""TrackNet V4 model definition and weight loading.

Architecture from: https://github.com/AnInsomniacy/tracknet-series-pytorch
TrackNet V4 introduces motion-attention-enhanced U-Net architecture.

Input: 3 consecutive RGB frames stacked → tensor of shape (B, 9, H, W)
Output: (B, 3, H, W) — per-frame sigmoid heatmap (one channel per input frame)

Key differences from V2:
  - Input resolution: 512×288 (vs 640×360)
  - Output: 3-channel sigmoid heatmap (one per input frame)
  - Motion Prompt module extracts attention from frame differences
  - Learnable motion attention parameters (a, b)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Default inference resolution (must match training config)
INPUT_W = 512
INPUT_H = 288

# Where to download weights
_WEIGHTS_FILENAME = "tracknet_v4.pth"
_WEIGHTS_URL = (
    "https://github.com/AnInsomniacy/tracknet-v4-pytorch/releases/"
    "download/v1.0.1/tracknet-v4_best-model.pth"
)


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
    """Motion-Enhanced U-Net for Sports Object Tracking (TrackNet V4).

    Args:
        dropout: Dropout probability for the bottleneck layer.
    """

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
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 9, H, W).

        Returns:
            Output tensor of shape (B, 3, H, W) with sigmoid activations.
        """
        B = x.size(0)
        _, motion_mask, _ = self.motion_prompt(
            x.view(B, 3, 3, x.size(2), x.size(3))
        )

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


def load_tracknet_v4(
    weights_path: str | None = None,
    device: str = "cpu",
    auto_download: bool = True,
) -> TrackNetV4Model:
    """Load TrackNet V4 model with pretrained weights.

    Args:
        weights_path: Path to the .pth weights file. If None, uses
            default path in weights/ directory.
        device: Device to load model onto ('cpu' or 'cuda').
        auto_download: If True, download weights automatically when missing.

    Returns:
        Loaded model in eval mode.

    Raises:
        FileNotFoundError: If weights file doesn't exist and auto_download is False.
        RuntimeError: If weight download fails.
    """
    if weights_path:
        path = Path(weights_path)
    else:
        from src.config import get_config
        path = get_config().weights_dir / _WEIGHTS_FILENAME

    if auto_download:
        _download_weights(path)
    elif not path.exists():
        raise FileNotFoundError(
            f"TrackNet V4 weights not found at {path}. "
            f"Download from {_WEIGHTS_URL} or set auto_download=True."
        )

    logger.info("Loading TrackNet V4 from %s onto %s", path, device)
    model = TrackNetV4Model()
    state_dict = torch.load(str(path), map_location=device, weights_only=False)

    # Handle full training checkpoints
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    # Handle DDP-wrapped checkpoints (keys prefixed with 'module.')
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
