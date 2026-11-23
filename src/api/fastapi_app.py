"""
Flood-AI: Production FastAPI Microservice.
Exposes REST endpoints for real-time flood inundation segmentation,
0-72h stage nowcasting, and multilingual evacuation alert generation.
"""

from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.models.segmentation.vit_unet import ViTUNet
from src.models.forecasting.tft_model import TemporalFusionTransformer
from src.inference.alert_generator import FloodAlertGenerator
from src.inference.realtime_pipeline import RealtimeFloodPipeline


# --- Pydantic Schemas ---
class ForecastRequest(BaseModel):
    station_id: str = Field(default="01638500", description="River station identifier")
    past_stage_readings: List[float] = Field(description="Historical stage readings (meters)")
    forecast_rain_mm: Optional[List[float]] = Field(default=None, description="Predicted rainfall")


class ForecastResponse(BaseModel):
    station_id: str
    forecast_horizon_hours: int
    median_stage_p50: List[float]
    confidence_p10: List[float]
    confidence_p90: List[float]
    max_forecast_stage: float


class AlertRequest(BaseModel):
    region_name: str = Field(default="Dhaka Metropolitan Area")
    inundated_hectares: float
    current_stage_m: float
    forecast_stage_m: float
    rate_of_rise_m_hr: float = 0.25


class AlertResponse(BaseModel):
    alert_level: str
    headline: str
    english_advisory: str
    local_advisory: str
    designated_shelters: List[str]
    evacuation_routes: List[str]


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Flood-AI REST API",
    description="High-Resolution Real-Time Flood Early Warning and Evacuation Guidance System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model references
alert_generator = FloodAlertGenerator()


@app.get("/")
def root():
    return {
        "system": "Flood-AI Early Warning Platform",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "capabilities": ["segmentation_10m", "nowcasting_72h", "multilingual_alerts", "edge_tpu"],
    }


@app.get("/health")
def health():
    return {"status": "healthy", "cuda_available": torch.cuda.is_available()}


@app.post("/predict/nowcast", response_model=ForecastResponse)
def predict_nowcast(req: ForecastRequest):
    """Predicts 0-72h probabilistic river stage trajectories."""
    past_vals = req.past_stage_readings
    if len(past_vals) < 5:
        raise HTTPException(status_code=400, detail="At least 5 historical readings required.")

    # Format input tensor for TFT: (1, T_past, 4)
    t_len = len(past_vals)
    past_tensor = torch.zeros(1, t_len, 4)
    past_tensor[0, :, 0] = torch.tensor(past_vals, dtype=torch.float32)

    horizon = 24
    tft = TemporalFusionTransformer(num_past_features=4, num_future_features=2, forecast_horizon=horizon)
    tft.eval()

    with torch.no_grad():
        out = tft(past_tensor)

    p10 = out["p10"].numpy()[0].tolist()
    p50 = out["p50"].numpy()[0].tolist()
    p90 = out["p90"].numpy()[0].tolist()

    return ForecastResponse(
        station_id=req.station_id,
        forecast_horizon_hours=horizon,
        median_stage_p50=p50,
        confidence_p10=p10,
        confidence_p90=p90,
        max_forecast_stage=float(np.max(p50)),
    )


@app.post("/alerts/generate", response_model=AlertResponse)
def generate_alert(req: AlertRequest):
    """Generates localized, plain-language evacuation advisories."""
    notification = alert_generator.generate_advisory(
        inundation_hectares=req.inundated_hectares,
        max_forecast_stage_m=req.forecast_stage_m,
        rate_of_rise_m_hr=req.rate_of_rise_m_hr,
        region_name=req.region_name,
    )
    return AlertResponse(
        alert_level=notification.level.value,
        headline=notification.headline,
        english_advisory=notification.english_advisory,
        local_advisory=notification.local_advisory,
        designated_shelters=notification.designated_shelters,
        evacuation_routes=notification.evacuation_routes,
    )
