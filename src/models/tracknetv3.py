"""TrackNet V3 model definition and weight loading.

Architecture from: https://github.com/qaz812345/TrackNetV3
TrackNet V3 uses a U-Net with skip connections for shuttlecock/ball tracking.

Input: 3 consecutive RGB frames stacked → tensor of shape (B, 9, H, W)
Output: (B, 3, H, W) — per-frame sigmoid heatmap (one channel per input frame)

Key differences from V2:
  - U-Net skip connections (V2 has none)
  - Conv → BN → ReLU order (V2 uses Conv → ReLU → BN)
  - Conv2d bias=False (V2 uses bias=True)
  - Output: 3-channel sigmoid heatmap (V2: 256-class softmax)
  - Input resolution: 512×288 (V2: 640×360)
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

# Google Drive file ID for reference checkpoint zip
_GDRIVE_FILE_ID = "1CfzE87a0f6LhBp0kniSl-89zaLCZ8cA"
_GDRIVE_URL = (
    f"https://drive.google.com/file/d/{_GDRIVE_FILE_ID}/view?usp=sharing"
)
_WEIGHTS_FILENAME = "tracknet_v3.pt"


class _Conv2DBlock(nn.Module):
    """Conv2D (3×3, no bias) → BatchNorm → ReLU."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class _Double2DConv(nn.Module):
    """Two consecutive Conv2DBlocks."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv_1 = _Conv2DBlock(in_dim, out_dim)
        self.conv_2 = _Conv2DBlock(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_2(self.conv_1(x))


class _Triple2DConv(nn.Module):
    """Three consecutive Conv2DBlocks."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv_1 = _Conv2DBlock(in_dim, out_dim)
        self.conv_2 = _Conv2DBlock(out_dim, out_dim)
        self.conv_3 = _Conv2DBlock(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_3(self.conv_2(self.conv_1(x)))


class TrackNetV3Model(nn.Module):
    """TrackNet V3 — U-Net with skip connections for ball tracking.

    Args:
        in_dim: Number of input channels (default 9 = 3 RGB frames).
        out_dim: Number of output channels (default 3 = one heatmap per frame).
    """

    def __init__(self, in_dim: int = 9, out_dim: int = 3) -> None:
        super().__init__()
        # Encoder
        self.down_block_1 = _Double2DConv(in_dim, 64)
        self.down_block_2 = _Double2DConv(64, 128)
        self.down_block_3 = _Triple2DConv(128, 256)
        # Bottleneck
        self.bottleneck = _Triple2DConv(256, 512)
        # Decoder (with skip connections: 512+256=768, 256+128=384, 128+64=192)
        self.up_block_1 = _Triple2DConv(768, 256)
        self.up_block_2 = _Double2DConv(384, 128)
        self.up_block_3 = _Double2DConv(192, 64)
        # Output
        self.predictor = nn.Conv2d(64, out_dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, in_dim, H, W).

        Returns:
            Output tensor of shape (B, out_dim, H, W) with sigmoid activations.
        """
        # Encoder
        x1 = self.down_block_1(x)
        x = nn.MaxPool2d(2, stride=2)(x1)
        x2 = self.down_block_2(x)
        x = nn.MaxPool2d(2, stride=2)(x2)
        x3 = self.down_block_3(x)
        x = nn.MaxPool2d(2, stride=2)(x3)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)
        x = self.up_block_1(x)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)
        x = self.up_block_2(x)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)
        x = self.up_block_3(x)

        # Prediction
        x = self.predictor(x)
        x = self.sigmoid(x)
        return x


def _download_weights(dest: Path) -> None:
    """Download TrackNet V3 pretrained weights if not present.

    The reference checkpoint is hosted on Google Drive as a zip file.
    Requires the ``gdown`` package for automatic download.
    """
    if dest.exists():
        return

    logger.info("Downloading TrackNet V3 weights to %s ...", dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        import gdown  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError(
            "Could not download TrackNet V3 weights automatically.\n"
            "Install gdown (`pip install gdown`) and retry, or download manually:\n"
            f"  1. Download from {_GDRIVE_URL}\n"
            "  2. Unzip TrackNetV3_ckpts.zip\n"
            f"  3. Place TrackNet_best.pt at {dest}"
        )

    zip_path = dest.parent / "TrackNetV3_ckpts.zip"
    try:
        gdown.download(id=_GDRIVE_FILE_ID, output=str(zip_path), quiet=False)

        import zipfile
        with zipfile.ZipFile(str(zip_path)) as zf:
            # Find TrackNet_best.pt in the archive
            tracknet_name = None
            for name in zf.namelist():
                if name.endswith("TrackNet_best.pt"):
                    tracknet_name = name
                    break
            if tracknet_name is None:
                raise FileNotFoundError(
                    "TrackNet_best.pt not found in downloaded archive"
                )
            with zf.open(tracknet_name) as src:
                dest.write_bytes(src.read())
        logger.info("Download complete: %s", dest)
    except Exception:
        logger.exception(
            "Failed to download TrackNet V3 weights from Google Drive"
        )
        raise RuntimeError(
            f"Could not download TrackNet V3 weights.\n"
            f"Please download manually from {_GDRIVE_URL}\n"
            f"and place TrackNet_best.pt at {dest}"
        )
    finally:
        if zip_path.exists():
            zip_path.unlink()


def load_tracknet_v3(
    weights_path: str | None = None,
    device: str = "cpu",
    auto_download: bool = True,
) -> TrackNetV3Model:
    """Load TrackNet V3 model with pretrained weights.

    Handles two checkpoint formats:
      - Plain state_dict
      - Reference format: ``{'model': state_dict, 'param_dict': {...}, ...}``

    When loading the reference checkpoint (which may use seq_len != 3 or a
    background mode), the model is automatically configured to match the
    checkpoint dimensions.

    Args:
        weights_path: Path to the .pt weights file.  If ``None``, uses
            the default path from the project config.
        device: Device to load model onto (``'cpu'`` or ``'cuda'``).
        auto_download: If ``True``, download weights automatically when missing.

    Returns:
        Loaded model in eval mode.

    Raises:
        FileNotFoundError: If weights file doesn't exist and auto_download
            is ``False``.
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
            f"TrackNet V3 weights not found at {path}. "
            f"Download from {_GDRIVE_URL} or set auto_download=True."
        )

    logger.info("Loading TrackNet V3 from %s onto %s", path, device)
    checkpoint = torch.load(str(path), map_location=device, weights_only=False)

    # Determine model dimensions and extract state_dict
    in_dim = 9
    out_dim = 3
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        # Reference checkpoints include param_dict with seq_len and bg_mode
        if "param_dict" in checkpoint:
            pd = checkpoint["param_dict"]
            seq_len = pd.get("seq_len", 3)
            bg_mode = pd.get("bg_mode", "")
            if bg_mode == "concat":
                in_dim = (seq_len + 1) * 3
            elif bg_mode == "subtract":
                in_dim = seq_len
            elif bg_mode == "subtract_concat":
                in_dim = seq_len * 4
            else:
                in_dim = seq_len * 3
            out_dim = seq_len
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Handle DDP-wrapped checkpoints (keys prefixed with 'module.')
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    model = TrackNetV3Model(in_dim=in_dim, out_dim=out_dim)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
