"""
Flood-AI: Training Pipeline for High-Resolution Flood Segmentation (ViTUNet).
Trains ViTUNet on multimodal SAR + Optical + DEM tiles using Combined Focal + Dice Loss,
AdamW optimizer with Cosine Annealing, and validation IoU tracking.
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import logging

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger("Flood-AI.TrainSeg")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

from src.models.segmentation.vit_unet import ViTUNet
from src.models.utils import CombinedFloodLoss, save_checkpoint
from src.evaluation.metrics import SegmentationMetrics


class SyntheticFloodDataset(Dataset):
    """Generates synthetic multimodal flood tiles for training and pipeline verification."""

    def __init__(self, num_samples: int = 100, in_channels: int = 4, img_size: int = 128):
        self.num_samples = num_samples
        self.in_channels = in_channels
        self.img_size = img_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Generate synthetic multimodal input (e.g., VV, VH, NDWI, DEM)
        x = torch.randn(self.in_channels, self.img_size, self.img_size)
        # Synthetic ground truth mask (water is correlated with low channel 0 SAR backscatter)
        y = (x[0:1, :, :] < -0.5).float()
        return {"image": x, "mask": y}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Runs a single epoch of training."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates validation loss, IoU, and F1-Dice."""
    model.eval()
    all_ious = []
    all_dices = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)

            metrics = SegmentationMetrics.compute_all(masks, probs, threshold=0.5)
            all_ious.append(metrics["iou"])
            all_dices.append(metrics["dice"])

    return {"val_iou": float(np.mean(all_ious)), "val_dice": float(np.mean(all_dices))}


def run_segmentation_training(
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    save_dir: str = "checkpoints/segmentation",
):
    """Executes the segmentation model training workflow."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting ViTUNet training on device: {device}")

    # Datasets and Loaders
    train_ds = SyntheticFloodDataset(num_samples=40, in_channels=4, img_size=128)
    val_ds = SyntheticFloodDataset(num_samples=10, in_channels=4, img_size=128)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ViTUNet(in_channels=4, num_classes=1, embed_dim=128, depth=2, num_heads=4, img_size=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = CombinedFloodLoss(focal_weight=0.5, dice_weight=0.5)

    best_iou = 0.0
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        val_metrics = validate(model, val_loader, device)

        logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {loss:.4f} | Val IoU: {val_metrics['val_iou']:.4f} | Val Dice: {val_metrics['val_dice']:.4f}")

        if val_metrics["val_iou"] > best_iou:
            best_iou = val_metrics["val_iou"]
            save_checkpoint(model, optimizer, epoch, val_metrics, Path(save_dir) / "vit_unet_best.pt")

    logger.info(f"ViTUNet Training Complete. Best Validation IoU: {best_iou:.4f}")


if __name__ == "__main__":
    run_segmentation_training(epochs=2, batch_size=2)
