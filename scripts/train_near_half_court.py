"""Training script for near-half court keypoint detector.

Fine-tunes a pretrained full-court detector for the near-half court view with
7 keypoints and visibility prediction.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.court_net import NearHalfCourtNet, load_near_half_court_detector
from src.features.court_detect.dataset import create_train_val_datasets
from src.features.court_detect.loss import NearHalfCourtLoss


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train near-half court keypoint detector"
    )
    
    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/near_half_train",
        help="Directory containing training images"
    )
    parser.add_argument(
        "--labels-json",
        type=str,
        default="data/near_half_train/labels.json",
        help="Path to labels JSON file"
    )
    
    # Model
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        default="weights/court_detector.pt",
        help="Path to pretrained full-court detector weights"
    )
    
    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=3e-5,
        help="Learning rate for backbone (conv layers)"
    )
    parser.add_argument(
        "--vis-lr",
        type=float,
        default=1e-3,
        help="Learning rate for visibility head"
    )
    parser.add_argument(
        "--lambda-vis",
        type=float,
        default=0.1,
        help="Weight for visibility loss (only affects vis_fc due to encoder detach)"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Number of linear warmup steps (0 = no warmup, matching upstream)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (epochs)"
    )
    
    # System
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, auto-detect if not specified)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="weights",
        help="Directory to save trained model"
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    
    # Checkpointing
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    # Diagnostic mode
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Run 3 epochs and print lambda_vis calibration recommendation"
    )
    
    return parser.parse_args()


def get_device(device_arg: Optional[str]) -> torch.device:
    """Get torch device, auto-detecting if not specified."""
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_optimizer(model: NearHalfCourtNet, backbone_lr: float, vis_lr: float) -> Adam:
    """Create optimizer with separate parameter groups for backbone and visibility head.
    
    Uses Adam with weight_decay=0 to match upstream training configuration.
    
    Args:
        model: The model to optimize
        backbone_lr: Learning rate for conv layers (pretrained, adapt gently)
        vis_lr: Learning rate for visibility head (randomly initialized, learn fast)
    
    Returns:
        Adam optimizer with two parameter groups
    """
    # Separate parameters: backbone (all conv layers) vs visibility head
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("vis_")]
    vis_params = [model.vis_fc.weight, model.vis_fc.bias]
    
    optimizer = Adam([
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": 0},
        {"params": vis_params, "lr": vis_lr, "weight_decay": 0}
    ])
    
    return optimizer


def create_scheduler(
    optimizer: Adam,
    warmup_steps: int,
    total_steps: int
) -> LambdaLR:
    """Create learning rate scheduler with linear warmup and cosine decay.
    
    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of linear warmup steps
        total_steps: Total training steps
    
    Returns:
        LambdaLR scheduler
    """
    initial_lr = optimizer.param_groups[0]["lr"]

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay to 1e-6
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            min_lr_ratio = 1e-6 / initial_lr
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
                1 + np.cos(np.pi * progress)
            )
    
    return LambdaLR(optimizer, lr_lambda)


def extract_keypoints_from_heatmaps(
    heatmaps: torch.Tensor,
    vis_logits: torch.Tensor,
    vis_threshold: float = 0.5
) -> List[Optional[Tuple[float, float]]]:
    """Extract keypoint coordinates from heatmap predictions.
    
    Args:
        heatmaps: Predicted heatmap logits (7, H, W) at full resolution (360, 640)
        vis_logits: Predicted visibility logits (7,)
        vis_threshold: Threshold for visibility prediction
    
    Returns:
        List of 7 keypoints in full image space (640×360), each either (x, y) or None if invisible
    """
    # Apply sigmoid to visibility for thresholding
    vis_prob = torch.sigmoid(vis_logits)  # (7,)
    
    keypoints = []
    for i in range(7):
        if vis_prob[i] < vis_threshold:
            keypoints.append(None)
        else:
            # Find argmax directly on logits (argmax is invariant to sigmoid)
            heatmap = heatmaps[i]  # (H, W)
            flat_idx = heatmap.argmax()
            h, w = heatmap.shape
            y = (flat_idx // w).item()
            x = (flat_idx % w).item()
            # Coordinates are already in full image space (H, W) = (360, 640)
            keypoints.append((float(x), float(y)))
    
    return keypoints


def compute_reprojection_error(
    pred_kps: List[Optional[Tuple[float, float]]],
    gt_kps: np.ndarray,
    visibility: np.ndarray
) -> float:
    """Compute mean reprojection error for visible keypoints.
    
    Args:
        pred_kps: List of predicted keypoints (x, y) or None, in full image space (640×360)
        gt_kps: Ground truth keypoints (7, 2) in full image space (640×360)
        visibility: Visibility flags (7,)
    
    Returns:
        Mean reprojection error in pixels (inf if no visible keypoints)
    """
    errors = []
    for i in range(7):
        if visibility[i] > 0.5 and pred_kps[i] is not None:
            dx = pred_kps[i][0] - gt_kps[i, 0]
            dy = pred_kps[i][1] - gt_kps[i, 1]
            errors.append(np.hypot(dx, dy))
    
    return np.mean(errors) if errors else float("inf")


def compute_visibility_accuracy(
    pred_vis_logits: torch.Tensor,
    target_visibility: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """Compute visibility prediction accuracy.
    
    Args:
        pred_vis_logits: Predicted visibility logits (B, 7)
        target_visibility: Target visibility flags (B, 7)
        threshold: Threshold for binary prediction
    
    Returns:
        Accuracy as fraction of correct predictions
    """
    pred_vis = (torch.sigmoid(pred_vis_logits) > threshold).float()
    correct = (pred_vis == target_visibility).float().sum()
    total = target_visibility.numel()
    return (correct / total).item()


def train_epoch(
    model: NearHalfCourtNet,
    dataloader: DataLoader,
    criterion: NearHalfCourtLoss,
    optimizer: Adam,
    scheduler: LambdaLR,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epoch: Current epoch number (for progress bar)
    
    Returns:
        Dictionary with average loss components
    """
    model.train()
    
    total_losses = {"heatmap_loss": 0.0, "vis_loss": 0.0, "total_loss": 0.0}
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        images = batch["image"].to(device)
        target_heatmaps = batch["heatmaps"].to(device)
        target_visibility = batch["visibility"].to(device)
        
        # Forward pass
        pred_heatmaps, pred_vis_logits = model(images)
        
        # Heatmaps are already at full resolution — no upsampling needed
        
        # Compute loss at full resolution
        loss, loss_dict = criterion(
            pred_heatmaps,
            pred_vis_logits,
            target_heatmaps,
            target_visibility
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Accumulate losses
        for key, value in loss_dict.items():
            total_losses[key] += value
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix(loss=loss_dict["total_loss"])
    
    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses


@torch.no_grad()
def validate(
    model: NearHalfCourtNet,
    dataloader: DataLoader,
    criterion: NearHalfCourtLoss,
    device: torch.device,
    epoch: int
) -> Tuple[Dict[str, float], float, float]:
    """Validate the model.
    
    Args:
        model: The model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number (for progress bar)
    
    Returns:
        avg_losses: Dictionary with average loss components
        avg_reproj_error: Average reprojection error in pixels
        vis_accuracy: Visibility prediction accuracy
    """
    model.eval()
    
    total_losses = {"heatmap_loss": 0.0, "vis_loss": 0.0, "total_loss": 0.0}
    num_batches = 0
    
    all_reproj_errors = []
    all_vis_logits = []
    all_vis_targets = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]  ")
    for batch in pbar:
        images = batch["image"].to(device)
        target_heatmaps = batch["heatmaps"].to(device)
        target_visibility = batch["visibility"].to(device)
        gt_coords = batch["keypoints_coords"]  # (B, 7, 2) in full image space
        
        # Forward pass
        pred_heatmaps, pred_vis_logits = model(images)
        
        # Heatmaps are already at full resolution — no upsampling needed
        
        # Compute loss at full resolution
        loss, loss_dict = criterion(
            pred_heatmaps,
            pred_vis_logits,
            target_heatmaps,
            target_visibility
        )
        
        # Accumulate losses
        for key, value in loss_dict.items():
            total_losses[key] += value
        num_batches += 1
        
        # Compute reprojection errors in full image space (640×360)
        for i in range(len(images)):
            # Extract keypoints from full-res predictions
            pred_kps = extract_keypoints_from_heatmaps(
                pred_heatmaps[i].cpu(),
                pred_vis_logits[i].cpu()
            )
            
            # Use original label coordinates for GT (no argmax quantization)
            gt_kps = gt_coords[i].numpy()  # (7, 2) — [x, y] per keypoint
            
            visibility = target_visibility[i].cpu().numpy()
            
            reproj_error = compute_reprojection_error(pred_kps, gt_kps, visibility)
            if reproj_error != float("inf"):
                all_reproj_errors.append(reproj_error)
        
        # Collect visibility predictions for accuracy
        all_vis_logits.append(pred_vis_logits.cpu())
        all_vis_targets.append(target_visibility.cpu())
    
    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    # Average reprojection error
    avg_reproj_error = np.mean(all_reproj_errors) if all_reproj_errors else float("inf")
    
    # Visibility accuracy
    all_vis_logits = torch.cat(all_vis_logits, dim=0)
    all_vis_targets = torch.cat(all_vis_targets, dim=0)
    vis_accuracy = compute_visibility_accuracy(all_vis_logits, all_vis_targets)
    
    return avg_losses, avg_reproj_error, vis_accuracy


def save_checkpoint(
    model: NearHalfCourtNet,
    optimizer: Adam,
    scheduler: LambdaLR,
    epoch: int,
    best_val_error: float,
    filepath: Path
) -> None:
    """Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        best_val_error: Best validation error so far
        filepath: Path to save checkpoint
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_error": best_val_error
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    checkpoint_path: str,
    model: NearHalfCourtNet,
    optimizer: Adam,
    scheduler: LambdaLR,
    device: torch.device
) -> Tuple[int, float]:
    """Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load tensors to
    
    Returns:
        start_epoch: Epoch to resume from
        best_val_error: Best validation error from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    start_epoch = checkpoint["epoch"] + 1
    best_val_error = checkpoint["best_val_error"]
    
    print(f"Resumed from epoch {checkpoint['epoch']}, best val error: {best_val_error:.3f}")
    
    return start_epoch, best_val_error


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset, val_dataset = create_train_val_datasets(
        labels_json=args.labels_json,
        images_dir=args.data_dir,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Load model
    print("Loading model...")
    pretrained_path = Path(args.pretrained_weights)
    if not pretrained_path.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")
    
    model = load_near_half_court_detector(
        weights_path=str(pretrained_path),
        device=device,
        pretrained_full=True
    )
    model = model.to(device)
    
    # Create optimizer, scheduler, criterion
    optimizer = create_optimizer(model, args.backbone_lr, args.vis_lr)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = create_scheduler(optimizer, args.warmup_steps, total_steps)
    
    criterion = NearHalfCourtLoss(
        lambda_vis=args.lambda_vis
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_error = float("inf")
    
    if args.resume:
        start_epoch, best_val_error = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
    
    # Diagnostic mode: run only 3 epochs
    num_epochs = 3 if args.diagnostic else args.epochs
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    epochs_without_improvement = 0
    
    # For diagnostic mode
    all_heatmap_losses = []
    all_vis_losses = []
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch + 1
        )
        
        # Validate
        val_losses, val_reproj_error, val_vis_accuracy = validate(
            model, val_loader, criterion, device, epoch + 1
        )
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"  Train - Total: {train_losses['total_loss']:.4f}, "
              f"Heatmap: {train_losses['heatmap_loss']:.4f}, "
              f"Vis: {train_losses['vis_loss']:.4f}")
        print(f"  Val   - Total: {val_losses['total_loss']:.4f}, "
              f"Heatmap: {val_losses['heatmap_loss']:.4f}, "
              f"Vis: {val_losses['vis_loss']:.4f}")
        print(f"  Val   - Reproj Error: {val_reproj_error:.3f} px, "
              f"Vis Accuracy: {val_vis_accuracy:.3f}")
        
        # Collect for diagnostic mode
        if args.diagnostic:
            all_heatmap_losses.append(val_losses['heatmap_loss'])
            all_vis_losses.append(val_losses['vis_loss'])
        
        # Save best model
        if val_reproj_error < best_val_error:
            best_val_error = val_reproj_error
            epochs_without_improvement = 0
            
            # Save best checkpoint
            best_path = output_dir / "near_half_court_best.pt"
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_error, best_path
            )
            print(f"  ✓ Saved best model (reproj error: {best_val_error:.3f} px)")
        else:
            epochs_without_improvement += 1
        
        # Save latest checkpoint
        latest_path = output_dir / "near_half_court_latest.pt"
        save_checkpoint(
            model, optimizer, scheduler, epoch, best_val_error, latest_path
        )
        
        # Early stopping
        if not args.diagnostic and epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping after {args.patience} epochs without improvement")
            break
    
    # Diagnostic mode: print calibration recommendation
    if args.diagnostic:
        mean_heatmap = np.mean(all_heatmap_losses)
        mean_vis = np.mean(all_vis_losses)
        
        print("\n" + "=" * 50)
        print("=== DIAGNOSTIC RESULTS ===")
        print(f"Mean heatmap_loss: {mean_heatmap:.4f}")
        print(f"Mean vis_loss: {mean_vis:.4f}")
        
        if mean_vis > 0:
            recommended_lambda = (mean_heatmap / mean_vis) * 0.3
            print(f"Recommended lambda_vis = (L_heatmap / L_vis) * 0.3 = {recommended_lambda:.4f}")
        else:
            print("Cannot compute recommended lambda_vis (vis_loss is zero)")
        
        print("=" * 50)
    
    # Export final model
    if not args.diagnostic:
        print("\nExporting final model...")
        final_path = output_dir / "near_half_court.pt"
        
        # Load best checkpoint and export just the model state
        best_checkpoint = torch.load(output_dir / "near_half_court_best.pt", weights_only=False)
        torch.save(best_checkpoint["model_state_dict"], final_path)
        
        print(f"✓ Final model saved to: {final_path}")
        print(f"✓ Best validation reproj error: {best_val_error:.3f} px")


if __name__ == "__main__":
    main()
