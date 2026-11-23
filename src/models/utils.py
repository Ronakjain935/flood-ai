"""
Flood-AI: Model Utilities, Loss Functions, and Checkpoint Helpers.
Implements Focal Loss, Soft Dice Loss, Combined Inundation Loss,
and robust model weight loading/saving utilities.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss addresses extreme class imbalance between flooded and non-flooded pixels:
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        loss = focal_weight * bce
        return torch.mean(loss)


class SoftDiceLoss(nn.Module):
    """Soft Dice Loss directly optimizing Intersection-over-Union."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = torch.sum(probs * targets, dim=(1, 2, 3))
        cardinality = torch.sum(probs + targets, dim=(1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return torch.mean(1.0 - dice)


class CombinedFloodLoss(nn.Module):
    """Combines Focal Loss and Soft Dice Loss for robust flood boundary delineation."""

    def __init__(self, focal_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = SoftDiceLoss()
        self.w_focal = focal_weight
        self.w_dice = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.w_focal * self.focal(logits, targets) + self.w_dice * self.dice(logits, targets)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: Union[str, Path],
):
    """Serializes model state and training metadata."""
    out_p = Path(path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(state, out_p)


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[int, Dict[str, float]]]:
    """Loads model weights and returns stored epoch and metrics."""
    chk_p = Path(path)
    if not chk_p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {chk_p}")

    checkpoint = torch.load(chk_p, map_location=device or torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {"epoch": checkpoint.get("epoch", 0), "metrics": checkpoint.get("metrics", {})}
