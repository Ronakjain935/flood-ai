"""
Flood-AI: Training Pipeline for 0-72h Hydro-Forecasting (TFT).
Trains the Temporal Fusion Transformer with multi-quantile pinball loss
to produce robust p10, p50, and p90 river stage forecast trajectories.
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import logging

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger("Flood-AI.TrainTFT")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

from src.models.forecasting.tft_model import TemporalFusionTransformer
from src.evaluation.metrics import HydrologyMetrics


class PinballLoss(nn.Module):
    """
    Multi-quantile pinball loss for probabilistic regression:
    Loss_q = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    """

    def __init__(self, quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # predictions: (B, Horizon, num_quantiles)
        # targets: (B, Horizon)
        losses = []
        for i, q in enumerate(self.quantiles):
            error = targets - predictions[:, :, i]
            loss = torch.max(q * error, (q - 1.0) * error)
            losses.append(torch.mean(loss))
        return torch.sum(torch.stack(losses))


class SyntheticHydroDataset(Dataset):
    """Generates synthetic river stage & rainfall time series for forecasting."""

    def __init__(self, num_sequences: int = 50, past_len: int = 48, horizon: int = 24):
        self.num_sequences = num_sequences
        self.past_len = past_len
        self.horizon = horizon

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        total_len = self.past_len + self.horizon
        t = np.linspace(0, 4 * np.pi, total_len)
        stage = 2.0 + 0.8 * np.sin(t) + np.random.normal(0, 0.05, total_len)

        past_x = torch.zeros(self.past_len, 4)
        past_x[:, 0] = torch.from_numpy(stage[: self.past_len]).float()  # stage
        past_x[:, 1] = torch.randn(self.past_len) * 2.0                 # rain

        future_x = torch.randn(self.horizon, 2)                         # future rain
        targets = torch.from_numpy(stage[self.past_len :]).float()      # future stage

        return {"past_x": past_x, "future_x": future_x, "targets": targets}


def run_tft_training(
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    save_dir: str = "checkpoints/forecasting",
):
    """Executes the TFT training workflow."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting TFT training on device: {device}")

    dataset = SyntheticHydroDataset(num_sequences=40, past_len=48, horizon=24)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TemporalFusionTransformer(
        num_past_features=4,
        num_future_features=2,
        forecast_horizon=24,
        hidden_dim=32,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = PinballLoss(quantiles=(0.1, 0.5, 0.9))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in loader:
            past_x = batch["past_x"].to(device)
            future_x = batch["future_x"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            out = model(past_x, future_x)
            loss = criterion(out["quantiles"], targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        logger.info(f"Epoch {epoch}/{epochs} | TFT Pinball Loss: {avg_loss:.4f}")

    out_p = Path(save_dir) / "tft_model_best.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_p)
    logger.info(f"TFT Training Finished. Checkpoint saved to: {out_p}")


if __name__ == "__main__":
    run_tft_training(epochs=2, batch_size=2)
