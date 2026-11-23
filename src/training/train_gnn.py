"""
Flood-AI: Training Pipeline for River Reach Topological Routing (GNN).
Trains the Spatio-Temporal River GNN on sensor graph networks to forecast
discharge and stage levels across interconnected river stations.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import logging

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger("Flood-AI.TrainGNN")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

from src.models.forecasting.gnn_river import SpatioTemporalRiverGNN


class RiverGraphDataset(Dataset):
    """Generates synthetic graph sequences for river network training."""

    def __init__(self, num_samples: int = 40, num_nodes: int = 5, seq_len: int = 24, horizon: int = 12):
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        x = torch.randn(self.seq_len, self.num_nodes, 3)
        # Target: downstream water levels over horizon
        y = torch.randn(self.num_nodes, self.horizon)
        return x, y


def run_gnn_training(
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    save_dir: str = "checkpoints/gnn",
):
    """Executes the River GNN training workflow."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting River GNN training on device: {device}")

    num_nodes = 5
    adj = torch.tensor([
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ]).to(device)

    dataset = RiverGraphDataset(num_samples=30, num_nodes=num_nodes, seq_len=24, horizon=12)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SpatioTemporalRiverGNN(num_nodes=num_nodes, in_features=3, hidden_dim=32, forecast_steps=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x, adj)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        logger.info(f"Epoch {epoch}/{epochs} | River GNN MSE Loss: {avg_loss:.4f}")

    out_p = Path(save_dir) / "river_gnn_best.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_p)
    logger.info(f"River GNN Training Finished. Saved to: {out_p}")


if __name__ == "__main__":
    run_gnn_training(epochs=2, batch_size=2)
