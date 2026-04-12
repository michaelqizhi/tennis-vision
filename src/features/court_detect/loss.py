"""Loss functions for near-half court keypoint detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class NearHalfCourtLoss(nn.Module):
    """Combined loss for near-half court detection with heatmap and visibility components.
    
    The total loss combines:
    - Heatmap loss: per-pixel MSE with visibility-based weighting
    - Visibility loss: per-keypoint BCE for visibility prediction
    
    Invisible keypoints receive reduced weight on their zero-target heatmaps to
    suppress phantom detections while not dominating the gradient signal.
    """
    
    def __init__(self, lambda_vis: float = 0.1, invis_weight: float = 0.1):
        """Initialize the combined loss.
        
        Args:
            lambda_vis: Weight for visibility loss component (default: 0.1)
            invis_weight: Weight for invisible keypoint heatmap pixels (default: 0.1)
                         Visible keypoints always use weight 1.0
        """
        super().__init__()
        self.lambda_vis = lambda_vis
        self.invis_weight = invis_weight
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
        # Heatmap loss with visibility-based weighting
        heatmap_loss = self._compute_heatmap_loss(
            pred_heatmaps, target_heatmaps, target_visibility
        )
        
        # Visibility loss
        vis_loss = self.vis_loss_fn(pred_vis_logits, target_visibility)
        
        # Combined loss
        total_loss = heatmap_loss + self.lambda_vis * vis_loss
        
        loss_dict = {
            "heatmap_loss": heatmap_loss.item(),
            "vis_loss": vis_loss.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def _compute_heatmap_loss(
        self,
        pred_heatmaps: torch.Tensor,
        target_heatmaps: torch.Tensor,
        visibility: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted per-pixel MSE loss for heatmaps.
        
        Visible keypoints get weight 1.0, invisible get reduced weight to
        suppress phantoms without dominating gradients.
        
        Args:
            pred_heatmaps: Predicted heatmap logits (B, 7, H, W)
            target_heatmaps: Target heatmaps (B, 7, H, W)
            visibility: Visibility flags (B, 7)
        
        Returns:
            Weighted mean heatmap loss
        """
        # MSE directly on raw logits vs [0,1] Gaussian targets.
        # Do NOT apply sigmoid first: MSE(sigmoid(z), t) has vanishing gradients
        # when z is very negative (sigmoid'→0), preventing peak learning.
        # Raw-logit MSE gradient = 2*(z - t) — always non-vanishing.
        per_pixel_loss = F.mse_loss(
            pred_heatmaps, target_heatmaps, reduction='none'
        )
        
        # Create visibility mask: (B, 7, 1, 1) for broadcasting
        vis_mask = visibility.unsqueeze(-1).unsqueeze(-1)
        
        # Weight formula: visible gets 1.0, invisible gets invis_weight
        # Weight = vis_mask * 1.0 + (1 - vis_mask) * invis_weight
        #        = vis_mask + (1 - vis_mask) * invis_weight
        weights = vis_mask + (1 - vis_mask) * self.invis_weight
        
        # Apply weights and normalize by total weighted pixel count
        weighted_loss = per_pixel_loss * weights
        
        # Normalize: weights is (B,7,1,1) so sum gives per-channel weight total;
        # multiply by spatial dims to get actual weighted pixel count.
        H, W = pred_heatmaps.shape[2], pred_heatmaps.shape[3]
        total_weight = weights.sum() * H * W
        normalized_loss = weighted_loss.sum() / total_weight.clamp(min=1.0)
        
        return normalized_loss
