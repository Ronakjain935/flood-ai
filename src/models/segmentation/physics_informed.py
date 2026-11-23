"""
Flood-AI: Physics-Informed Neural Network (PINN) Loss Regularizer.
Enforces physical hydrodynamic laws during training:
1. Gravity-driven flow continuity: Water flows downhill toward lower elevation.
2. Elevation constraint: High ridges cannot be inundated while adjacent lower valleys are dry.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsInformedFloodLoss(nn.Module):
    """
    Combines standard supervised loss with physics-based hydrologic constraints:
    Total_Loss = L_supervised + lambda_dem * L_gravity_violation + lambda_smooth * L_spatial_smoothness
    """

    def __init__(
        self,
        lambda_gravity: float = 0.15,
        lambda_smoothness: float = 0.05,
    ):
        super().__init__()
        self.lambda_gravity = lambda_gravity
        self.lambda_smoothness = lambda_smoothness
        self.bce = nn.BCEWithLogitsLoss()

    def gravity_constraint_loss(
        self,
        predicted_prob: torch.Tensor,
        dem: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalizes physically impossible configurations where a higher-elevation pixel
        is assigned higher water probability than its immediate lower-elevation neighbor.
        """
        # Compute horizontal and vertical gradients of elevation and flood probability
        dem_grad_y = dem[:, :, 1:, :] - dem[:, :, :-1, :]
        dem_grad_x = dem[:, :, :, 1:] - dem[:, :, :, :-1]

        prob_grad_y = predicted_prob[:, :, 1:, :] - predicted_prob[:, :, :-1, :]
        prob_grad_x = predicted_prob[:, :, :, 1:] - predicted_prob[:, :, :, :-1]

        # If elevation increases (dem_grad > 0), flood probability should not increase (prob_grad <= 0)
        # Violation occurs when sign(dem_grad) == sign(prob_grad) and dem_grad > 0
        violation_y = F.relu(prob_grad_y * torch.sign(F.relu(dem_grad_y)))
        violation_x = F.relu(prob_grad_x * torch.sign(F.relu(dem_grad_x)))

        loss_gravity = torch.mean(violation_y) + torch.mean(violation_x)
        return loss_gravity

    @staticmethod
    def total_variation_smoothness(predicted_prob: torch.Tensor) -> torch.Tensor:
        """Enforces spatial continuity of standing flood water bodies (Total Variation Loss)."""
        diff_i = torch.abs(predicted_prob[:, :, 1:, :] - predicted_prob[:, :, :-1, :])
        diff_j = torch.abs(predicted_prob[:, :, :, 1:] - predicted_prob[:, :, :, :-1])
        return torch.mean(diff_i) + torch.mean(diff_j)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        dem: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Unnormalized network predictions (B, 1, H, W)
            targets: Binary ground truth water mask (B, 1, H, W)
            dem: Normalized Digital Elevation Model (B, 1, H, W)
        """
        loss_supervised = self.bce(logits, targets)
        probs = torch.sigmoid(logits)

        loss_grav = self.gravity_constraint_loss(probs, dem)
        loss_smooth = self.total_variation_smoothness(probs)

        total_loss = (
            loss_supervised
            + self.lambda_gravity * loss_grav
            + self.lambda_smoothness * loss_smooth
        )
        return total_loss


if __name__ == "__main__":
    criterion = PhysicsInformedFloodLoss()
    pred_logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    ground_truth = torch.randint(0, 2, (2, 1, 64, 64)).float()
    elevation = torch.linspace(0.0, 1.0, 64).view(1, 1, 64, 1).repeat(2, 1, 1, 64)

    loss = criterion(pred_logits, ground_truth, elevation)
    loss.backward()
    print(f"PINN Loss calculated successfully: {loss.item():.4f}")
