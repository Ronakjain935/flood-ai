"""
Flood-AI: Evaluation Metrics for Geospatial Segmentation and Hydrologic Forecasting.
Implements Intersection over Union (IoU), Critical Success Index (CSI),
Nash-Sutcliffe Efficiency (NSE), and Kling-Gupta Efficiency (KGE).
"""

from typing import Dict, Union
import numpy as np
import torch


class SegmentationMetrics:
    """Standard evaluation metrics for binary flood extent segmentation."""

    @staticmethod
    def compute_all(
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        threshold: float = 0.5,
        eps: float = 1e-7,
    ) -> Dict[str, float]:
        """Computes IoU, Dice/F1, Critical Success Index (CSI), Precision, and Recall."""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        y_true = (y_true > threshold).astype(bool)
        y_pred = (y_pred > threshold).astype(bool)

        tp = np.sum(y_true & y_pred)
        fp = np.sum(~y_true & y_pred)
        fn = np.sum(y_true & ~y_pred)
        tn = np.sum(~y_true & ~y_pred)

        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        # Critical Success Index (Threat Score)
        csi = (tp + eps) / (tp + fp + fn + eps)

        return {
            "iou": float(iou),
            "dice": float(dice),
            "csi": float(csi),
            "precision": float(precision),
            "recall": float(recall),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }


class HydrologyMetrics:
    """Benchmark metrics for streamflow and river stage time-series forecasting."""

    @staticmethod
    def nash_sutcliffe_efficiency(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Nash-Sutcliffe Efficiency (NSE):
        NSE = 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
        NSE = 1.0 (perfect match), NSE > 0.65 indicates acceptable hydrological skill.
        """
        denom = np.sum((y_true - np.mean(y_true)) ** 2)
        if denom == 0:
            return 0.0
        num = np.sum((y_true - y_pred) ** 2)
        return float(1.0 - (num / denom))

    @staticmethod
    def kling_gupta_efficiency(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Kling-Gupta Efficiency (KGE) decomposing skill into correlation, variability, and bias:
        KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
        where alpha = std(y_pred) / std(y_true), beta = mean(y_pred) / mean(y_true).
        """
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        std_true = np.std(y_true)
        std_pred = np.std(y_pred)

        if std_true == 0 or mean_true == 0:
            return 0.0

        r = np.corrcoef(y_true, y_pred)[0, 1] if std_pred > 0 else 0.0
        alpha = std_pred / std_true
        beta = mean_pred / mean_true

        kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
        return float(kge)

    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error (RMSE)."""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


if __name__ == "__main__":
    t_seg = np.array([[1, 1, 0], [1, 0, 0]], dtype=float)
    p_seg = np.array([[0.9, 0.8, 0.1], [0.4, 0.1, 0.0]], dtype=float)
    seg_res = SegmentationMetrics.compute_all(t_seg, p_seg)
    print("Segmentation Metrics:", seg_res)

    t_hydro = np.array([2.0, 2.5, 3.8, 4.2, 3.1, 2.2])
    p_hydro = np.array([1.9, 2.6, 3.7, 4.0, 3.0, 2.3])
    print(f"Hydrology NSE: {HydrologyMetrics.nash_sutcliffe_efficiency(t_hydro, p_hydro):.4f}")
    print(f"Hydrology KGE: {HydrologyMetrics.kling_gupta_efficiency(t_hydro, p_hydro):.4f}")
