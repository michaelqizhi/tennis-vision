"""Shared model building blocks."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d + ReLU + BatchNorm block.

    Used by both TrackNet (ball tracking) and CourtDetectorNet (court detection).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, pad: int = 1,
                 stride: int = 1, bias: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
