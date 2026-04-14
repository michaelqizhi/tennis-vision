"""Loss functions for near-half court keypoint detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class NearHalfCourtLoss(nn.Module):
    """Heatmap loss for near-half court detection.

    Matches upstream exactly: MSE(sigmoid(pred), target) with mean reduction
    over all pixels and channels.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        pred_heatmaps: torch.Tensor,
        target_heatmaps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute heatmap loss.

        Args:
            pred_heatmaps: Predicted heatmap logits (B, 7, H, W)
            target_heatmaps: Target heatmaps (B, 7, H, W)

        Returns:
            total_loss: MSE loss between sigmoid(pred) and target
            loss_dict: Dictionary with 'heatmap_loss' and 'total_loss'
        """
        heatmap_loss = F.mse_loss(
            torch.sigmoid(pred_heatmaps), target_heatmaps, reduction='mean'
        )

        loss_dict = {
            "heatmap_loss": heatmap_loss.item(),
            "total_loss": heatmap_loss.item(),
        }

        return heatmap_loss, loss_dict
