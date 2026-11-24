"""
Flood-AI: Multimodal Spatio-Temporal Data Fusion Engine.

Harmonizes heterogeneous spatial rasters (Sentinel-1 SAR, Sentinel-2 Optical, DEM, GPM Radar)
and point sensor time-series into unified multi-channel raster cubes.
Implements Module 1 Fusion Engine:
- Channels 0-5: SAR VV, SAR VH, VV/VH ratio, Optical NDWI, DEM, Slope.
- Channel 6: Gauge Stage (observed where adapter returns it, or river flow from openmeteo_flood with origin = 'model').
- Channel 7: Rate of Rise (observed where adapter returns it, or model rate).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import logging

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger("Flood-AI.Fusion")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


@dataclass
class FusionConfig:
    """Configuration parameters for multi-sensor tensor construction."""
    target_crs: str = "EPSG:32646"  # UTM Zone 46N for Bangladesh pilot
    target_pixel_size_m: float = 10.0
    tile_height: int = 128
    tile_width: int = 128
    fill_nodata: float = 0.0


class MultimodalDataFusion:
    """
    Fuses SAR backscatter, optical spectral water indices, topographic DEM,
    and river stage height readings into a synchronized (C, H, W) tensor.
    """

    CHANNEL_ORDER = [
        "sar_vv_norm",     # Channel 0: Normalized Sentinel-1 VV
        "sar_vh_norm",     # Channel 1: Normalized Sentinel-1 VH
        "sar_ratio_norm",  # Channel 2: Normalized VV/VH difference
        "optical_ndwi",    # Channel 3: Sentinel-2 NDWI
        "elevation_dem",   # Channel 4: Normalized Topographic DEM
        "slope",           # Channel 5: Topographic Slope in radians
        "gauge_stage",     # Channel 6: Gauge Stage / Model Discharge
        "rate_of_rise",    # Channel 7: Rate of Rise
    ]

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()

    @staticmethod
    def compute_topographic_slope(dem: np.ndarray, cell_size_m: float = 10.0) -> np.ndarray:
        """
        Computes topographic slope from a Digital Elevation Model using central differences.
        Steep slopes are less prone to standing flood inundation.
        """
        dy, dx = np.gradient(dem, cell_size_m, cell_size_m)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        return slope_rad.astype(np.float32)

    def assemble_multimodal_tensor(
        self,
        sar_vv: np.ndarray,
        sar_vh: np.ndarray,
        ndwi: Optional[np.ndarray] = None,
        dem: Optional[np.ndarray] = None,
        gauge_stage_m: Optional[float] = None,
        rate_of_rise_m_hr: Optional[float] = None,
    ) -> np.ndarray:
        """
        Combines spatial arrays into a unified (C, H, W) model tensor.
        Interpolates and normalizes missing modalities gracefully.
        """
        H, W = sar_vv.shape

        # Normalize SAR
        vv_norm = np.clip((sar_vv + 30.0) / 30.0, 0.0, 1.0).astype(np.float32)
        vh_norm = np.clip((sar_vh + 30.0) / 30.0, 0.0, 1.0).astype(np.float32)
        ratio_norm = np.clip((sar_vv - sar_vh) / 15.0, 0.0, 1.0).astype(np.float32)

        # Handle optical NDWI (clip [-1, 1] -> [0, 1])
        if ndwi is not None and ndwi.shape == (H, W):
            ndwi_norm = np.clip((ndwi + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
        else:
            ndwi_norm = np.full((H, W), fill_value=0.5, dtype=np.float32)

        # Handle elevation & slope
        if dem is not None and dem.shape == (H, W):
            d_min, d_max = np.min(dem), np.max(dem)
            denom = d_max - d_min if d_max > d_min else 1.0
            dem_norm = ((dem - d_min) / denom).astype(np.float32)
            slope = self.compute_topographic_slope(dem, cell_size_m=self.config.target_pixel_size_m)
        else:
            dem_norm = np.zeros((H, W), dtype=np.float32)
            slope = np.zeros((H, W), dtype=np.float32)

        stack = [vv_norm, vh_norm, ratio_norm, ndwi_norm, dem_norm, slope]

        # Channel 6: Gauge Stage / Discharge
        if gauge_stage_m is not None:
            stage_layer = np.full((H, W), fill_value=float(gauge_stage_m) / 10.0, dtype=np.float32)
            stack.append(stage_layer)

        # Channel 7: Rate of Rise
        if rate_of_rise_m_hr is not None:
            ror_layer = np.full((H, W), fill_value=float(rate_of_rise_m_hr), dtype=np.float32)
            stack.append(ror_layer)

        tensor = np.stack(stack, axis=0).astype(np.float32)
        logger.info(f"Assembled multimodal tensor shape: {tensor.shape} (Channels: {len(stack)})")
        return tensor
