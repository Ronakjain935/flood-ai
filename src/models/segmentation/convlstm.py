"""
Flood-AI: Spatio-Temporal Convolutional LSTM (ConvLSTM) for Flood Propagation Dynamics.
Replaces linear matrix multiplications in standard LSTM cells with 2D convolutions,
capturing simultaneous spatial topology and temporal evolution of floodwaters.
"""

from typing import List, Tuple, Union
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    2D Convolutional LSTM cell.
    Maintains 2D spatial dimensions (H, W) across hidden states and cell memories.
    """

    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        # Computes input gate, forget gate, cell gate, and output gate in a single conv
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (Batch, in_channels, H, W)
        hx: tuple of (h_prev, c_prev), each of shape (Batch, hidden_dim, H, W)
        """
        h_prev, c_prev = hx
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)

        # Split into 4 gates: input, forget, candidate cell, output
        i_gate, f_gate, c_cand, o_gate = torch.split(gates, self.hidden_dim, dim=1)

        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        c = f * c_prev + i * torch.tanh(c_cand)
        o = torch.sigmoid(o_gate)
        h = o * torch.tanh(c)

        return h, c


class ConvLSTM(nn.Module):
    """
    Multi-layer recurrent spatio-temporal ConvLSTM network for flood sequence modeling.

    Args:
        in_channels: Number of input channels per time step (e.g., 4 for SAR + DEM + Rain).
        hidden_dim: Number of feature maps in hidden recurrent layers.
        num_layers: Number of stacked ConvLSTM layers.
        out_channels: Number of output prediction channels (e.g., 1 for water mask).
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 32,
        num_layers: int = 2,
        out_channels: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        cells = []
        for i in range(num_layers):
            cur_in = in_channels if i == 0 else hidden_dim
            cells.append(ConvLSTMCell(cur_in, hidden_dim))
        self.cells = nn.ModuleList(cells)

        self.head = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: (Batch, Seq_Len, In_Channels, H, W)
        Returns:
            Predicted final inundation mask: (Batch, Out_Channels, H, W)
        """
        B, T, C, H, W = x_seq.shape
        device = x_seq.device

        # Initialize hidden and cell states
        h_states = [torch.zeros(B, self.hidden_dim, H, W, device=device) for _ in range(self.num_layers)]
        c_states = [torch.zeros(B, self.hidden_dim, H, W, device=device) for _ in range(self.num_layers)]

        for t in range(T):
            cur_input = x_seq[:, t, :, :, :]
            for layer_idx in range(self.num_layers):
                h_new, c_new = self.cells[layer_idx](cur_input, (h_states[layer_idx], c_states[layer_idx]))
                h_states[layer_idx] = h_new
                c_states[layer_idx] = c_new
                cur_input = h_new

        final_hidden = h_states[-1]
        out_logits = self.head(final_hidden)
        return out_logits


if __name__ == "__main__":
    model = ConvLSTM(in_channels=4, hidden_dim=16, num_layers=2, out_channels=1)
    dummy_seq = torch.randn(2, 5, 4, 64, 64)  # 5 sequential hourly time steps
    logits = model(dummy_seq)
    print(f"ConvLSTM forward pass verified: {dummy_seq.shape} -> {logits.shape}")
