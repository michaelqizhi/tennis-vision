"""TrackNetV2 model definition and weight loading.

Architecture from: https://github.com/yastrebksv/TrackNet
Input: 3 consecutive RGB frames stacked → tensor of shape (B, 9, H, W)
Output: (B, out_channels, H*W) — per-pixel classification heatmap
"""

import torch
import torch.nn as nn
from pathlib import Path

from src.models.common import ConvBlock


class BallTrackerNet(nn.Module):
    """TrackNetV2 encoder-decoder for tennis ball detection.

    Takes 3 consecutive RGB frames (9 channels) and produces a per-pixel
    heatmap indicating the ball location.

    Args:
        out_channels: Number of output classes per pixel (default: 256).
    """

    def __init__(self, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels

        # Encoder
        self.conv1 = ConvBlock(in_channels=9, out_channels=64)
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

        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def forward(self, x: torch.Tensor, testing: bool = False) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 9, H, W).
            testing: If True, apply softmax to output.

        Returns:
            Output tensor of shape (B, out_channels, H*W).
        """
        batch_size = x.size(0)

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

        out = x.reshape(batch_size, self.out_channels, -1)
        if testing:
            out = self.softmax(out)
        return out

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def load_tracknet(weights_path: str, device: str = "cpu") -> BallTrackerNet:
    """Load TrackNetV2 model with pretrained weights.

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
            f"TrackNet weights not found at {weights_path}. "
            "Run 'python scripts/download_weights.py' first."
        )

    model = BallTrackerNet()
    model.load_state_dict(torch.load(str(weights), map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model
