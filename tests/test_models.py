"""
Unit tests for Flood-AI deep learning model architectures.
"""

import pytest
import torch
from src.models.segmentation.vit_unet import ViTUNet


def test_vit_unet_output_shape():
    """Verify that ViTUNet maintains spatial dimensions and produces correct logits shape."""
    batch_size = 2
    in_channels = 4
    img_size = 64
    num_classes = 1

    model = ViTUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=128,
        depth=2,
        num_heads=4,
        img_size=img_size,
    )
    model.eval()

    dummy_input = torch.randn(batch_size, in_channels, img_size, img_size)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (batch_size, num_classes, img_size, img_size)
    assert not torch.isnan(output).any()


def test_vit_unet_multimodal_channels():
    """Verify ViTUNet handles multimodal inputs (e.g., 2 SAR channels vs 8 optical+SAR+DEM channels)."""
    img_size = 64
    for in_ch in [2, 6, 8]:
        model = ViTUNet(in_channels=in_ch, num_classes=1, embed_dim=64, depth=1, num_heads=2, img_size=img_size)
        x = torch.randn(1, in_ch, img_size, img_size)
        out = model(x)
        assert out.shape == (1, 1, img_size, img_size)


def test_vit_unet_backward_pass():
    """Verify loss computation and gradient backpropagation through transformer and conv stages."""
    model = ViTUNet(in_channels=3, num_classes=1, embed_dim=64, depth=2, num_heads=2, img_size=64)
    model.train()

    inputs = torch.randn(2, 3, 64, 64)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    logits = model(inputs)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    loss.backward()

    # Check gradients in transformer bottleneck and conv layers
    assert model.pos_embed.grad is not None
    assert model.enc1.block[0].weight.grad is not None
    assert model.final_head.weight.grad is not None
