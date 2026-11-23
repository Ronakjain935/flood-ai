"""
Flood-AI: End-to-End Real-Time Inundation & Forecasting Pipeline.
Orchestrates multi-sensor data fusion, ViTUNet spatial segmentation,
TFT 0-72h hydro-forecasting, and automated early warning alert dispatch.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import logging

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger("Flood-AI.Pipeline")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

from src.data.fusion import MultimodalDataFusion
from src.data.preprocess import GeospatialPreprocessor
from src.models.segmentation.vit_unet import ViTUNet
from src.models.forecasting.tft_model import TemporalFusionTransformer
from src.inference.alert_generator import FloodAlertGenerator


@dataclass
class PipelineResult:
    """Consolidated inference outputs from the real-time pipeline."""
    timestamp: datetime
    flood_extent_hectares: float
    max_forecast_stage_m: float
    alert_level: str
    advisory_english: str
    advisory_local: str
    inundation_probability_map: np.ndarray
    forecast_quantiles: Dict[str, np.ndarray]


class RealtimeFloodPipeline:
    """
    Production inference engine coordinating satellite raster tiling,
    deep learning prediction, and multi-channel advisory generation.
    """

    def __init__(
        self,
        segmentation_model: Optional[ViTUNet] = None,
        forecast_model: Optional[TemporalFusionTransformer] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seg_model = segmentation_model or ViTUNet(in_channels=6, num_classes=1, img_size=128)
        self.seg_model = self.seg_model.to(self.device).eval()

        self.forecast_model = forecast_model or TemporalFusionTransformer(num_past_features=4, num_future_features=2, forecast_horizon=72)
        self.forecast_model = self.forecast_model.to(self.device).eval()

        self.fusion = MultimodalDataFusion()
        self.preprocessor = GeospatialPreprocessor()
        self.alert_gen = FloodAlertGenerator()

    def run_inference(
        self,
        sar_vv: np.ndarray,
        sar_vh: np.ndarray,
        dem: Optional[np.ndarray] = None,
        ndwi: Optional[np.ndarray] = None,
        past_sensor_telemetry: Optional[np.ndarray] = None,
        future_rain_forecast: Optional[np.ndarray] = None,
        pixel_size_m: float = 10.0,
    ) -> PipelineResult:
        """
        Executes end-to-end inference over current scene and gauge readings.

        Args:
            sar_vv, sar_vh: 2D SAR backscatter rasters in dB.
            dem: 2D Digital Elevation Model raster.
            ndwi: 2D optical water index raster.
            past_sensor_telemetry: (1, T_past, 4) past sensor time-series.
            future_rain_forecast: (1, 72, 2) predicted precipitation.
        """
        now = datetime.utcnow()
        logger.info(f"=== [Flood-AI] Starting Real-time Pipeline at {now.isoformat()} ===")
        H, W = sar_vv.shape

        # 1. Spatio-Temporal Data Fusion
        current_stage = float(past_sensor_telemetry[0, -1, 0]) if past_sensor_telemetry is not None else 2.5
        cube = self.fusion.assemble_multimodal_tensor(sar_vv, sar_vh, ndwi=ndwi, dem=dem, gauge_stage_m=current_stage)
        C, _, _ = cube.shape

        # 2. Sliding-Window Tiling & ViTUNet Segmentation
        patches = []
        coords = []
        for patch, coord in self.preprocessor.extract_patches(cube, patch_size=128, stride=64):
            tensor_patch = torch.from_numpy(patch).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.seg_model(tensor_patch)
                probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
            patches.append(probs)
            coords.append(coord)

        # 3. Spatial Stitching
        full_prob_map = self.preprocessor.stitch_patches(patches, coords, output_shape=(H, W))
        flooded_pixels = np.sum(full_prob_map >= 0.5)
        pixel_area_m2 = pixel_size_m * pixel_size_m
        flooded_hectares = float((flooded_pixels * pixel_area_m2) / 10000.0)

        # 4. Multi-Horizon Hydro-Forecasting (TFT)
        if past_sensor_telemetry is None:
            past_sensor_telemetry = np.random.randn(1, 96, 4).astype(np.float32) + 2.5
        if future_rain_forecast is None:
            future_rain_forecast = np.random.randn(1, 72, 2).astype(np.float32)

        t_past = torch.from_numpy(past_sensor_telemetry).to(self.device)
        t_future = torch.from_numpy(future_rain_forecast).to(self.device)
        with torch.no_grad():
            forecast_out = self.forecast_model(t_past, t_future)

        p10 = forecast_out["p10"].cpu().numpy()[0]
        p50 = forecast_out["p50"].cpu().numpy()[0]
        p90 = forecast_out["p90"].cpu().numpy()[0]
        max_stage = float(np.max(p50))

        # 5. Alert & Evacuation Synthesis
        alert = self.alert_gen.generate_advisory(
            inundation_hectares=flooded_hectares,
            max_forecast_stage_m=max_stage,
            rate_of_rise_m_hr=0.35,
            region_name="Dhaka Metropolitan Area",
        )

        logger.info(f"Pipeline Completed: Flood Area = {flooded_hectares:.1f} ha, Alert = {alert.level}")

        return PipelineResult(
            timestamp=now,
            flood_extent_hectares=flooded_hectares,
            max_forecast_stage_m=max_stage,
            alert_level=alert.level,
            advisory_english=alert.english_advisory,
            advisory_local=alert.local_advisory,
            inundation_probability_map=full_prob_map,
            forecast_quantiles={"p10": p10, "p50": p50, "p90": p90},
        )


if __name__ == "__main__":
    pipeline = RealtimeFloodPipeline()
    sample_vv = np.random.uniform(-25.0, -5.0, (256, 256)).astype(np.float32)
    sample_vh = np.random.uniform(-30.0, -10.0, (256, 256)).astype(np.float32)
    sample_dem = np.random.uniform(1.0, 15.0, (256, 256)).astype(np.float32)

    result = pipeline.run_inference(sample_vv, sample_vh, dem=sample_dem)
    print(f"Result Alert Level: {result.alert_level}")
    print(f"English Advisory:\n{result.advisory_english}")
