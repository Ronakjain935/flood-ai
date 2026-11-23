"""
Flood-AI: Geospatial & Hydrograph Visualization Utilities.
Creates false-color SAR composite maps, semi-transparent flood overlays,
and multi-horizon hydrograph forecast charts with quantile uncertainty cones.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt


class FloodVisualizer:
    """Visualization utilities for geospatial inundation and hydro-forecasting."""

    @staticmethod
    def create_sar_false_color(vv_db: np.ndarray, vh_db: np.ndarray) -> np.ndarray:
        """
        Assembles standard RGB false-color SAR composite:
        Red: VV, Green: VH, Blue: VV / VH (dB difference).
        """
        # Normalize dB ranges
        r = np.clip((vv_db + 25.0) / 25.0, 0.0, 1.0)
        g = np.clip((vh_db + 30.0) / 25.0, 0.0, 1.0)
        diff = vv_db - vh_db
        b = np.clip((diff - 3.0) / 12.0, 0.0, 1.0)
        return np.stack([r, g, b], axis=-1)

    @staticmethod
    def plot_inundation_overlay(
        background: np.ndarray,
        flood_prob: np.ndarray,
        threshold: float = 0.5,
        alpha: float = 0.45,
        title: str = "Flood-AI Inundation Map",
        output_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Overlays predicted flood boundaries in vibrant cyan/blue over satellite base layer."""
        fig, ax = plt.subplots(figsize=(8, 8))
        if background.ndim == 2:
            ax.imshow(background, cmap="gray")
        else:
            ax.imshow(background)

        # Mask dry land
        water_mask = np.ma.masked_where(flood_prob < threshold, flood_prob)
        ax.imshow(water_mask, cmap="Blues", alpha=alpha, vmin=threshold, vmax=1.0)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_p, bbox_inches="tight", dpi=300)

        return fig

    @staticmethod
    def plot_forecast_hydrograph(
        past_timestamps: List[str],
        past_values: np.ndarray,
        future_timestamps: List[str],
        p10: np.ndarray,
        p50: np.ndarray,
        p90: np.ndarray,
        thresholds: Optional[Dict[str, float]] = None,
        title: str = "0-72h River Stage Hydrograph Forecast",
        output_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Plots historical water levels and forecasted median with 10th-90th quantile
        uncertainty envelope and color-coded danger thresholds.
        """
        fig, ax = plt.subplots(figsize=(12, 5))

        # Historical curve
        ax.plot(past_timestamps, past_values, label="Observed Stage", color="black", lw=2)

        # Connect last observed to first forecast
        conn_t = [past_timestamps[-1], future_timestamps[0]]
        conn_y = [past_values[-1], p50[0]]
        ax.plot(conn_t, conn_y, color="blue", lw=2, linestyle=":")

        # Forecast median & confidence ribbon
        ax.plot(future_timestamps, p50, label="Forecast Median (p50)", color="blue", lw=2)
        ax.fill_between(future_timestamps, p10, p90, color="blue", alpha=0.2, label="80% Confidence Interval (p10-p90)")

        # Critical warning thresholds
        if thresholds:
            if "minor" in thresholds:
                ax.axhline(thresholds["minor"], color="orange", linestyle="--", label=f"Minor Flood ({thresholds['minor']}m)")
            if "major" in thresholds:
                ax.axhline(thresholds["major"], color="red", linestyle="--", label=f"Major Flood ({thresholds['major']}m)")

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("River Stage Height (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

        # Sparse x-ticks
        step = max(1, len(future_timestamps) // 6)
        all_ticks = list(past_timestamps) + list(future_timestamps)
        ax.set_xticks(range(0, len(all_ticks), step))
        ax.set_xticklabels([all_ticks[i] for i in range(0, len(all_ticks), step)], rotation=30)
        plt.tight_layout()

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_p, bbox_inches="tight", dpi=300)

        return fig


if __name__ == "__main__":
    vis = FloodVisualizer()
    past_t = [f"-{i}h" for i in reversed(range(1, 13))] + ["0h"]
    past_vals = np.array([2.1, 2.2, 2.3, 2.4, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.8, 4.0, 4.2])
    fut_t = [f"+{i}h" for i in range(1, 25)]
    median = 4.2 + 0.8 * np.sin(np.linspace(0, np.pi, 24))
    p10_v = median - 0.3
    p90_v = median + 0.4

    fig = vis.plot_forecast_hydrograph(
        past_t, past_vals, fut_t, p10_v, median, p90_v,
        thresholds={"minor": 4.0, "major": 4.8}
    )
    plt.close(fig)
    print("Visualization utilities successfully verified!")
