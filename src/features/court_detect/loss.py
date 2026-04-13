"""Loss functions for near-half court keypoint detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class NearHalfCourtLoss(nn.Module):
    """Combined loss for near-half court detection with heatmap and visibility.
    
    Heatmap loss matches upstream exactly: MSE(sigmoid(pred), target) with
    mean reduction over all pixels and channels.
    
    Visibility loss is separate BCE, weighted by lambda_vis. Since the
    encoder is detached from the visibility head, lambda_vis only affects
    the vis_fc layer's gradient magnitude — it does NOT compete with
    heatmap gradients in the shared encoder/decoder.
    """
    
    def __init__(self, lambda_vis: float = 0.1):
        """Initialize the combined loss.
        
        Args:
            lambda_vis: Weight for visibility loss component (default: 0.1).
                       With the encoder detached from the vis head, this only
                       scales the vis_fc gradient, not the backbone gradient.
        """
        super().__init__()
        self.lambda_vis = lambda_vis
        self.vis_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    
    def forward(
        self,
        pred_heatmaps: torch.Tensor,
        pred_vis_logits: torch.Tensor,
        target_heatmaps: torch.Tensor,
        target_visibility: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss.
        
        Args:
            pred_heatmaps: Predicted heatmap logits (B, 7, H, W)
            pred_vis_logits: Predicted visibility logits (B, 7)
            target_heatmaps: Target heatmaps (B, 7, H, W)
            target_visibility: Target visibility flags (B, 7)
        
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary with individual loss components
        """
        # Heatmap loss: sigmoid + MSE, matching upstream exactly:
        #   loss = criterion(F.sigmoid(out), gt_hm_hp)  # upstream base_trainer.py
        heatmap_loss = F.mse_loss(
            torch.sigmoid(pred_heatmaps), target_heatmaps, reduction='mean'
        )
        
        # Visibility loss (gradients only reach vis_fc due to encoder detach)
        vis_loss = self.vis_loss_fn(pred_vis_logits, target_visibility)
        
        # Combined loss
        total_loss = heatmap_loss + self.lambda_vis * vis_loss
        
        loss_dict = {
            "heatmap_loss": heatmap_loss.item(),
            "vis_loss": vis_loss.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss, loss_dict
