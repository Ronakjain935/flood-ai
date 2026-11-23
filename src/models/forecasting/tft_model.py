"""
Flood-AI: Temporal Fusion Transformer (TFT) for Multi-Horizon Hydro-Forecasting.
Predicts river stage heights and discharge 0-72 hours in advance with
quantile uncertainty estimation (p10, p50, p90) and variable selection.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU) for feature gating."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val, gate = self.fc(x).chunk(2, dim=-1)
        return val * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) with skip connections, ELU non-linearity,
    LayerNorm, and GLU gating for non-linear feature transformation.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1, context_dim: Optional[int] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.context_fc = nn.Linear(context_dim, hidden_dim) if context_dim else None
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.glu = GatedLinearUnit(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        hidden = self.fc1(x)
        if self.context_fc is not None and context is not None:
            hidden = hidden + self.context_fc(context)
        hidden = F.elu(hidden)
        hidden = self.dropout(self.fc2(hidden))
        gated = self.glu(hidden)
        return self.norm(residual + gated)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) computes attention weights over static
    and dynamic features, dynamically zeroing out uninformative telemetry.
    """

    def __init__(self, num_inputs: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.num_inputs = num_inputs
        self.single_grns = nn.ModuleList([
            GatedResidualNetwork(hidden_dim, dropout=dropout) for _ in range(num_inputs)
        ])
        self.weight_grn = GatedResidualNetwork(num_inputs * hidden_dim, dropout=dropout)
        self.weight_proj = nn.Linear(num_inputs * hidden_dim, num_inputs)

    def forward(self, flattened_features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # flattened_features: list of num_inputs tensors of shape (B, T, hidden_dim)
        stacked = torch.cat(flattened_features, dim=-1)  # (B, T, num_inputs * hidden_dim)
        weights = F.softmax(self.weight_proj(self.weight_grn(stacked)), dim=-1)  # (B, T, num_inputs)

        transformed = []
        for i, grn in enumerate(self.single_grns):
            transformed.append(grn(flattened_features[i]))

        # Transformed stack: (B, T, num_inputs, hidden_dim)
        transformed_stack = torch.stack(transformed, dim=2)
        # Weighted sum across features
        selected = torch.sum(transformed_stack * weights.unsqueeze(-1), dim=2)
        return selected, weights


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for river stage height & discharge nowcasting.

    Args:
        num_past_features: Number of dynamic past time series variables (e.g., stage, precip, rate_of_rise).
        num_future_features: Number of known future inputs (e.g., forecasted precipitation, day-of-year).
        forecast_horizon: Number of future timesteps to predict (e.g., 72 hours).
        hidden_dim: Latent representation size across GRN and attention blocks.
        quantiles: Target quantiles for probabilistic prediction (default: p10, p50, p90).
    """

    def __init__(
        self,
        num_past_features: int = 4,
        num_future_features: int = 2,
        forecast_horizon: int = 72,
        hidden_dim: int = 64,
        quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
    ):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.hidden_dim = hidden_dim
        self.quantiles = quantiles

        self.past_embedding = nn.Linear(num_past_features, hidden_dim)
        self.future_embedding = nn.Linear(num_future_features, hidden_dim)

        self.temporal_encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.temporal_decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.post_attn_grn = GatedResidualNetwork(hidden_dim)

        # Output projection for quantiles (e.g., 3 heads for p10, p50, p90)
        self.output_head = nn.Linear(hidden_dim, len(quantiles))

    def forward(
        self,
        past_inputs: torch.Tensor,
        future_inputs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            past_inputs: Tensor of shape (Batch, Past_Steps, num_past_features)
            future_inputs: Tensor of shape (Batch, Forecast_Horizon, num_future_features)
        Returns:
            Dictionary containing 'quantiles' of shape (Batch, Forecast_Horizon, num_quantiles)
            and 'attention_weights'.
        """
        B, T_past, _ = past_inputs.shape

        if future_inputs is None:
            # If no future forecast is available, pad with zeros
            future_inputs = torch.zeros(B, self.forecast_horizon, self.future_embedding.in_features, device=past_inputs.device)

        past_emb = F.elu(self.past_embedding(past_inputs))
        future_emb = F.elu(self.future_embedding(future_inputs))

        # Encode past context
        _, (hn, cn) = self.temporal_encoder(past_emb)

        # Decode future horizon
        decoded_seq, _ = self.temporal_decoder(future_emb, (hn, cn))

        # Multi-Head Attention over temporal trajectory
        attn_out, attn_weights = self.self_attn(decoded_seq, decoded_seq, decoded_seq)
        context = self.post_attn_grn(decoded_seq + attn_out)

        # Predict multi-horizon quantiles: (Batch, Forecast_Horizon, num_quantiles)
        quantiles_pred = self.output_head(context)

        return {
            "quantiles": quantiles_pred,
            "p10": quantiles_pred[:, :, 0],
            "p50": quantiles_pred[:, :, 1],
            "p90": quantiles_pred[:, :, 2],
            "attn_weights": attn_weights,
        }


if __name__ == "__main__":
    tft = TemporalFusionTransformer(num_past_features=4, num_future_features=2, forecast_horizon=72)
    dummy_past = torch.randn(4, 96, 4)      # Past 96 hours of hourly telemetry
    dummy_future = torch.randn(4, 72, 2)    # Next 72 hours of forecasted rainfall
    predictions = tft(dummy_past, dummy_future)
    print(f"TFT Forward Pass Succeeded. Quantiles Shape: {predictions['quantiles'].shape}")
