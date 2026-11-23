"""
Flood-AI: ViTUNet (Vision Transformer + U-Net) for High-Resolution Flood Segmentation.
Combines a Transformer encoder for global spatial context and long-range dependencies
with a convolutional U-Net decoder for precise 10-30m boundary delineation.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Dual Convolutional Block with BatchNorm and GELU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TransformerBlock(nn.Module):
    """Standard Transformer Encoder Block with Multi-Head Self-Attention (MHSA)."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-Head Attention with residual
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        # Feedforward MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class ViTUNet(nn.Module):
    """
    Hybrid Vision Transformer + U-Net architecture for multimodal flood extent segmentation.

    Args:
        in_channels: Number of input channels (e.g., 2 for SAR VV/VH, 6 for SAR+NDWI+DEM).
        num_classes: Number of target segmentation classes (default: 1 for binary flood mask).
        embed_dim: Latent embedding dimension in the transformer bottleneck.
        depth: Number of sequential transformer blocks.
        num_heads: Number of attention heads in multi-head attention.
        img_size: Expected spatial height/width of input tiles (must be divisible by 16).
        patch_size: Patch resolution for tokenization (default: 16x16).
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 1,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        img_size: int = 128,
        patch_size: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # --- CNN Encoder Stages (Extracts hierarchical spatial features for skip connections) ---
        self.enc1 = ConvBlock(in_channels, 64)        # Scale 1: H, W
        self.pool1 = nn.MaxPool2d(2)                   # H/2, W/2

        self.enc2 = ConvBlock(64, 128)                 # Scale 2: H/2, W/2
        self.pool2 = nn.MaxPool2d(2)                   # H/4, W/4

        self.enc3 = ConvBlock(128, 256)                # Scale 3: H/4, W/4
        self.pool3 = nn.MaxPool2d(2)                   # H/8, W/8

        self.enc4 = ConvBlock(256, 512)                # Scale 4: H/8, W/8
        self.pool4 = nn.MaxPool2d(2)                   # H/16, W/16

        # --- Bottleneck Transformer ---
        # Projects H/16 x W/16 feature map into sequence of tokens
        self.proj_in = nn.Conv2d(512, embed_dim, kernel_size=1)
        self.grid_size = img_size // 16
        num_patches = self.grid_size * self.grid_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads) for _ in range(depth)
        ])

        self.proj_out = nn.Conv2d(embed_dim, 512, kernel_size=1)

        # --- U-Net Decoder Stages with Skip Connections ---
        # Decoder Stage 4: Up to H/8, W/8 (concat with s4: 512)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(256 + 512, 256)

        # Decoder Stage 3: Up to H/4, W/4 (concat with s3: 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128 + 256, 128)

        # Decoder Stage 2: Up to H/2, W/2 (concat with s2: 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64 + 128, 64)

        # Decoder Stage 1: Up to H, W (concat with s1: 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32 + 64, 32)

        # Final Segmentation Logits Head
        self.final_head = nn.Conv2d(32, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Initialize positional embeddings with truncated normal distribution."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. Hierarchical Feature Extraction (Encoder)
        s1 = self.enc1(x)         # (B, 64, H, W)
        p1 = self.pool1(s1)       # (B, 64, H/2, W/2)

        s2 = self.enc2(p1)        # (B, 128, H/2, W/2)
        p2 = self.pool2(s2)       # (B, 128, H/4, W/4)

        s3 = self.enc3(p2)        # (B, 256, H/4, W/4)
        p3 = self.pool3(s3)       # (B, 256, H/8, W/8)

        s4 = self.enc4(p3)        # (B, 512, H/8, W/8)
        p4 = self.pool4(s4)       # (B, 512, H/16, W/16)

        # 2. Vision Transformer Bottleneck
        feat = self.proj_in(p4)   # (B, embed_dim, H/16, W/16)
        _, _, h_btn, w_btn = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (B, N, embed_dim)

        # Interpolate positional embeddings if resolution varies
        if tokens.shape[1] == self.pos_embed.shape[1]:
            tokens = tokens + self.pos_embed
        else:
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2).reshape(1, self.embed_dim, self.grid_size, self.grid_size),
                size=(h_btn, w_btn),
                mode="bilinear",
                align_corners=False,
            ).flatten(2).transpose(1, 2)
            tokens = tokens + pos

        for layer in self.transformer_layers:
            tokens = layer(tokens)

        # Reshape tokens back to 2D spatial map
        feat_out = tokens.transpose(1, 2).reshape(B, self.embed_dim, h_btn, w_btn)
        bottleneck = self.proj_out(feat_out)  # (B, 512, H/16, W/16)

        # 3. Convolutional Decoder with Skip Connections
        d4 = self.up4(bottleneck)             # (B, 256, H/8, W/8)
        d4 = torch.cat([d4, s4], dim=1)       # Concat with scale 4 (H/8, W/8)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)                     # (B, 128, H/4, W/4)
        d3 = torch.cat([d3, s3], dim=1)       # Concat with scale 3 (H/4, W/4)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)                     # (B, 64, H/2, W/2)
        d2 = torch.cat([d2, s2], dim=1)       # Concat with scale 2 (H/2, W/2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)                     # (B, 32, H, W)
        d1 = torch.cat([d1, s1], dim=1)       # Concat with scale 1 (H, W)
        d1 = self.dec1(d1)

        logits = self.final_head(d1)          # (B, num_classes, H, W)
        return logits


if __name__ == "__main__":
    model = ViTUNet(in_channels=4, num_classes=1, img_size=128)
    dummy_input = torch.randn(2, 4, 128, 128)
    out = model(dummy_input)
    print(f"ViTUNet instantiated. Input: {dummy_input.shape} -> Output: {out.shape}")
