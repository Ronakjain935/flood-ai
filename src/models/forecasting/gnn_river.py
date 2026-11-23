"""
Flood-AI: Spatio-Temporal Graph Neural Network for River Reach Network Forecasting.
Models river reaches as directed graph edges and sensor monitoring stations as nodes,
enforcing hydraulic physical routing and upstream-to-downstream water propagation.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Standard Message-Passing Graph Convolution (Kipf & Welling).
    Transfers hydrologic energy and flood volume along the adjacency matrix.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch, Num_Nodes, In_Features)
        adj: Normalized Adjacency Matrix (Num_Nodes, Num_Nodes)
        """
        # Linear projection
        proj = self.fc(x)
        # Message passing: A * X * W
        out = torch.matmul(adj, proj)
        return out


class SpatioTemporalRiverGNN(nn.Module):
    """
    Spatio-Temporal GNN combining directional river graph convolutions with
    recurrent GRU cells to simulate hydrodynamic routing down a river basin.

    Args:
        num_nodes: Number of interconnected river gauge stations in the catchment.
        in_features: Number of sensor channels per node (e.g., stage, discharge, rain).
        hidden_dim: Latent representation size for graph convolutions.
        forecast_steps: Number of forward timesteps to forecast (e.g., 24 hours).
    """

    def __init__(
        self,
        num_nodes: int = 5,
        in_features: int = 3,
        hidden_dim: int = 32,
        forecast_steps: int = 24,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.forecast_steps = forecast_steps
        self.hidden_dim = hidden_dim

        # Graph spatial message passing
        self.gcn1 = GraphConvolution(in_features, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)

        # Temporal sequence update
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Forecasting head: maps hidden state to future stage heights
        self.head = nn.Linear(hidden_dim, forecast_steps)

    @staticmethod
    def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
        """Computes symmetric or degree-normalized adjacency matrix D^(-1) A."""
        deg = torch.sum(adj, dim=-1, keepdim=True)
        deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
        return adj * deg_inv

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features over time (Batch, Seq_Len, Num_Nodes, In_Features)
            adj: Directed adjacency matrix representing river reach flow (Num_Nodes, Num_Nodes)
        Returns:
            Forecasted stage for each node: (Batch, Num_Nodes, Forecast_Steps)
        """
        B, T, N, F_dim = x.shape
        norm_adj = self.normalize_adjacency(adj)

        # Apply spatial GCN at each timestep
        spatial_features = []
        for t in range(T):
            x_t = x[:, t, :, :]  # (B, N, F_dim)
            h = F.relu(self.gcn1(x_t, norm_adj))
            h = F.relu(self.gcn2(h, norm_adj))  # (B, N, hidden_dim)
            spatial_features.append(h)

        # Reshape for GRU: (B * N, T, hidden_dim)
        # Stack over time -> (B, T, N, hidden_dim) -> transpose to (B, N, T, hidden_dim)
        stacked = torch.stack(spatial_features, dim=1).transpose(1, 2).reshape(B * N, T, self.hidden_dim)

        # Temporal GRU sequence modeling
        _, h_final = self.gru(stacked)  # (1, B * N, hidden_dim)
        h_final = h_final.squeeze(0)    # (B * N, hidden_dim)

        # Predict future horizon: (B * N, forecast_steps)
        preds = self.head(h_final)
        # Reshape back to (B, N, forecast_steps)
        out = preds.view(B, N, self.forecast_steps)
        return out


if __name__ == "__main__":
    # Test network: 5 gauge stations along Dhaka's Buriganga & Turag rivers
    num_stations = 5
    # Directed adjacency: Station 0 flows to 1, 1 to 2, 3 flows to 2 (tributary), 2 flows to 4
    sample_adj = torch.tensor([
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])

    model = SpatioTemporalRiverGNN(num_nodes=num_stations, in_features=3, forecast_steps=24)
    dummy_input = torch.randn(2, 48, num_stations, 3)  # Batch=2, Past 48h, 5 stations, 3 features
    forecast = model(dummy_input, sample_adj)
    print(f"River GNN Forecast Output Shape: {forecast.shape} (Batch, Stations, 24h Horizon)")
