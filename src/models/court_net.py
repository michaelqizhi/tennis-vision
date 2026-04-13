"""Court keypoint detection model definition and weight loading.

Architecture from: https://github.com/yastrebksv/TennisCourtDetector
Same encoder-decoder as TrackNet but with:
  - Input: 1 RGB frame (3 channels) instead of 3 stacked frames (9 channels)
  - Output: 15 heatmaps (14 keypoints + 1 court center)
  - Resolution: 640×360
"""

import torch
import torch.nn as nn
from pathlib import Path

from src.models.common import ConvBlock


class CourtDetectorNet(nn.Module):
    """Court keypoint detection network.

    Takes a single RGB frame (3 channels) and produces 15 heatmaps
    for 14 court keypoints + 1 center point.

    Args:
        out_channels: Number of output heatmaps (default: 15).
    """

    def __init__(self, out_channels: int = 15):
        super().__init__()
        self.out_channels = out_channels

        # Encoder
        self.conv1 = ConvBlock(in_channels=3, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv6 = ConvBlock(in_channels=256, out_channels=256)
        self.conv7 = ConvBlock(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = ConvBlock(in_channels=256, out_channels=512)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512)
        self.conv10 = ConvBlock(in_channels=512, out_channels=512)

        # Decoder
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(in_channels=512, out_channels=256)
        self.conv12 = ConvBlock(in_channels=256, out_channels=256)
        self.conv13 = ConvBlock(in_channels=256, out_channels=256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(in_channels=256, out_channels=128)
        self.conv15 = ConvBlock(in_channels=128, out_channels=128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(in_channels=128, out_channels=64)
        self.conv17 = ConvBlock(in_channels=64, out_channels=64)
        self.conv18 = ConvBlock(in_channels=64, out_channels=self.out_channels)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Output tensor of shape (B, out_channels, H, W).
        """
        # Encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)

        # Decoder
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)

        return x

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def load_court_detector(weights_path: str, device: str = "cpu") -> CourtDetectorNet:
    """Load court detector model with pretrained weights.

    Args:
        weights_path: Path to the .pt weights file.
        device: Device to load model onto ('cpu' or 'cuda').

    Returns:
        Loaded model in eval mode.

    Raises:
        FileNotFoundError: If weights file doesn't exist.
    """
    weights = Path(weights_path)
    if not weights.exists():
        raise FileNotFoundError(
            f"Court detector weights not found at {weights_path}. "
            "Run 'python scripts/download_weights.py' first."
        )

    model = CourtDetectorNet(out_channels=15)
    # Upstream uses BallTrackerNet(out_channels=14) but saves 15-channel weights
    # (14 keypoints + 1 center). We use out_channels=15 to match.
    state_dict = torch.load(str(weights), map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


class NearHalfCourtNet(nn.Module):
    """Near-half court keypoint detector with visibility head.

    Same VGG encoder-decoder backbone as CourtDetectorNet but:
    - 7 output heatmaps (near-half keypoints only)
    - Visibility head: GAP(encoder) → FC(512→7)
    - No sigmoid in forward() — use BCEWithLogitsLoss during training,
      apply sigmoid at inference time.

    The 7 keypoints correspond to the near-half court lines visible in
    typical amateur courtside recordings.
    """

    NEAR_HALF_KP_INDICES = [5, 2, 10, 13, 11, 3, 7]
    NUM_KEYPOINTS = 7

    def __init__(self) -> None:
        super().__init__()

        # Encoder (identical to CourtDetectorNet)
        self.conv1 = ConvBlock(in_channels=3, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv6 = ConvBlock(in_channels=256, out_channels=256)
        self.conv7 = ConvBlock(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = ConvBlock(in_channels=256, out_channels=512)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512)
        self.conv10 = ConvBlock(in_channels=512, out_channels=512)

        # Decoder (identical except final conv outputs 7 channels)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(in_channels=512, out_channels=256)
        self.conv12 = ConvBlock(in_channels=256, out_channels=256)
        self.conv13 = ConvBlock(in_channels=256, out_channels=256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(in_channels=256, out_channels=128)
        self.conv15 = ConvBlock(in_channels=128, out_channels=128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(in_channels=128, out_channels=64)
        self.conv17 = ConvBlock(in_channels=64, out_channels=64)
        self.conv18 = ConvBlock(in_channels=64, out_channels=self.NUM_KEYPOINTS)

        # Visibility head (branches off encoder bottleneck after conv10)
        self.vis_gap = nn.AdaptiveAvgPool2d(1)
        self.vis_fc = nn.Linear(512, self.NUM_KEYPOINTS)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            heatmaps: (B, 7, H, W) — raw logits, apply sigmoid for probabilities.
            vis_logits: (B, 7) — per-keypoint visibility logits.
        """
        # Encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        enc = self.conv10(x)

        # Visibility head (from encoder bottleneck, detached)
        # Detach so visibility gradients don't flow back through the encoder.
        # This keeps the encoder optimized purely for heatmap production.
        vis_logits = self.vis_fc(self.vis_gap(enc.detach()).flatten(1))

        # Decoder
        x = self.ups1(enc)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        heatmaps = self.conv18(x)

        return heatmaps, vis_logits

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        # Xavier init for visibility FC layer
        nn.init.xavier_uniform_(self.vis_fc.weight)
        nn.init.zeros_(self.vis_fc.bias)


# Mapping: NearHalfCourtNet output channel → CourtDetectorNet output channel
_NEAR_HALF_TO_ORIG = [5, 2, 10, 13, 11, 3, 7]


def load_near_half_court_detector(
    weights_path: str,
    device: str = "cpu",
    pretrained_full: bool = True,
) -> NearHalfCourtNet:
    """Load NearHalfCourtNet, optionally initializing from full-court weights.

    Transfer strategy:
    - Encoder (conv1–conv10): copy all weights directly (identical architecture).
    - Decoder (conv11–conv17): copy all weights directly (identical architecture).
    - conv18 (final layer): slice 7 channels from the pretrained 15-channel output
      using _NEAR_HALF_TO_ORIG index mapping.
    - vis_gap/vis_fc: Xavier-initialized (not present in pretrained weights).

    Args:
        weights_path: Path to the full-court .pt weights file.
        device: Device to load model onto ('cpu' or 'cuda').
        pretrained_full: If True, transfer weights from full-court model.
            If False, return model with random initialization.

    Returns:
        Loaded model in eval mode.

    Raises:
        FileNotFoundError: If pretrained_full=True and weights file doesn't exist.
    """
    model = NearHalfCourtNet()

    if pretrained_full:
        weights = Path(weights_path)
        if not weights.exists():
            raise FileNotFoundError(
                f"Full-court weights not found at {weights_path}. "
                "Run 'python scripts/download_weights.py' first."
            )

        full_state = torch.load(str(weights), map_location=device, weights_only=False)
        new_state = model.state_dict()

        for name, param in full_state.items():
            if name not in new_state:
                continue
            if name.startswith("conv18"):
                # Final layer: select the 7 relevant channels from 15
                if param.dim() > 0 and param.shape[0] == 15:
                    new_state[name] = param[_NEAR_HALF_TO_ORIG]
                else:
                    # num_batches_tracked (scalar) — copy as-is
                    new_state[name] = param
            elif new_state[name].shape == param.shape:
                new_state[name] = param

        model.load_state_dict(new_state)

        # Re-init visibility head (not in pretrained weights)
        nn.init.xavier_uniform_(model.vis_fc.weight)
        nn.init.zeros_(model.vis_fc.bias)

    model = model.to(device)
    model.eval()
    return model
