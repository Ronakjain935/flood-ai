"""
Unit tests for Flood-AI inference pipeline, forecasting models, and metrics.
"""

import numpy as np
import pytest
import torch
from src.data.city_feature_resolver import CityFeatureResolver
from src.data.fusion import MultimodalDataFusion
from src.data.preprocess import GeospatialPreprocessor
from src.evaluation.metrics import HydrologyMetrics, SegmentationMetrics
from src.inference.alert_generator import AlertSeverity, FloodAlertGenerator
from src.inference.edge_inference import EdgeTPUInferenceEngine
from src.models.forecasting.gnn_river import SpatioTemporalRiverGNN
from src.models.forecasting.tft_model import TemporalFusionTransformer


def test_tft_model_forward():
    """Verify TFT outputs 3 quantile trajectories (p10, p50, p90) over 72h."""
    batch_size = 2
    past_steps = 48
    horizon = 72
    model = TemporalFusionTransformer(
        num_past_features=4,
        num_future_features=2,
        forecast_horizon=horizon,
        hidden_dim=32,
    )
    model.eval()

    past_x = torch.randn(batch_size, past_steps, 4)
    future_x = torch.randn(batch_size, horizon, 2)

    with torch.no_grad():
        out = model(past_x, future_x)

    assert "quantiles" in out
    assert out["quantiles"].shape == (batch_size, horizon, 3)
    assert out["p10"].shape == (batch_size, horizon)
    assert out["p50"].shape == (batch_size, horizon)
    assert out["p90"].shape == (batch_size, horizon)


def test_river_gnn_forward():
    """Verify Spatio-Temporal GNN forecasts river stage for all stations."""
    num_nodes = 4
    forecast_steps = 12
    model = SpatioTemporalRiverGNN(
        num_nodes=num_nodes,
        in_features=3,
        hidden_dim=16,
        forecast_steps=forecast_steps,
    )
    model.eval()

    adj = torch.eye(num_nodes)
    x = torch.randn(2, 24, num_nodes, 3)

    with torch.no_grad():
        out = model(x, adj)

    assert out.shape == (2, num_nodes, forecast_steps)


def test_multimodal_data_fusion():
    """Verify assembly of multi-channel tensor from SAR, DEM, and river stage."""
    fusion = MultimodalDataFusion()
    vv = np.random.uniform(-25.0, -5.0, (64, 64))
    vh = np.random.uniform(-30.0, -10.0, (64, 64))
    dem = np.random.uniform(5.0, 20.0, (64, 64))

    tensor = fusion.assemble_multimodal_tensor(vv, vh, dem=dem, gauge_stage_m=3.2)
    assert tensor.shape == (7, 64, 64)
    assert not np.isnan(tensor).any()


def test_preprocessor_tiling_and_stitching():
    """Verify sliding-window tile extraction and seamless reconstruction."""
    pre = GeospatialPreprocessor()
    img = np.random.randn(3, 100, 100).astype(np.float32)

    patches = []
    coords = []
    for p, c in pre.extract_patches(img, patch_size=64, stride=32):
        patches.append(p[0])
        coords.append(c)

    reconstructed = pre.stitch_patches(patches, coords, output_shape=(100, 100))
    assert reconstructed.shape == (100, 100)
    assert not np.isnan(reconstructed).any()


def test_alert_severity_classification():
    """Verify flood alert severity escalation logic."""
    gen = FloodAlertGenerator(stage_threshold_warning_m=4.0, stage_threshold_emergency_m=4.8)

    # Normal conditions
    assert gen.classify_severity(10.0, 2.5, 0.05) == AlertSeverity.NORMAL

    # Fast-rising water -> Emergency
    assert gen.classify_severity(100.0, 4.2, 0.55) == AlertSeverity.EMERGENCY

    # Inundation > 500 ha -> Warning
    assert gen.classify_severity(600.0, 3.5, 0.1) == AlertSeverity.WARNING


def test_evaluation_metrics():
    """Verify Segmentation (IoU, Dice) and Hydrology (NSE, KGE) metric correctness."""
    # Perfect segmentation match
    y_t = np.ones((10, 10))
    y_p = np.ones((10, 10))
    seg_res = SegmentationMetrics.compute_all(y_t, y_p)
    assert pytest.approx(seg_res["iou"], rel=1e-3) == 1.0

    # Perfect hydrology forecast
    hydro_t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    hydro_p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    nse = HydrologyMetrics.nash_sutcliffe_efficiency(hydro_t, hydro_p)
    kge = HydrologyMetrics.kling_gupta_efficiency(hydro_t, hydro_p)
    assert pytest.approx(nse, rel=1e-3) == 1.0
    assert pytest.approx(kge, rel=1e-3) == 1.0


def test_edge_inference_engine():
    """Verify edge engine fallback and benchmark execution."""
    edge = EdgeTPUInferenceEngine()
    stats = edge.benchmark_inference(input_shape=(1, 6, 64, 64), num_runs=5)
    assert stats["fps"] > 0
    assert "mean_latency_ms" in stats


def test_city_feature_resolver():
    """Verify universal city feature resolution across catalog and custom queries."""
    # Test catalog city (Mumbai)
    mumbai_res = CityFeatureResolver.resolve_city_features("Mumbai")
    assert mumbai_res["city"] == "Mumbai"
    assert mumbai_res["country"] == "India"
    assert "Mithi" in mumbai_res["basin"]
    assert len(mumbai_res["hotspots"]) >= 3
    assert len(mumbai_res["shelters"]) >= 2
    assert mumbai_res["danger_stage_m"] > 0

    # Test another catalog city (Tokyo)
    tokyo_res = CityFeatureResolver.resolve_city_features("Tokyo")
    assert tokyo_res["city"] == "Tokyo"
    assert tokyo_res["country"] == "Japan"
    assert len(tokyo_res["hotspots"]) >= 3

    # Test arbitrary procedural city
    custom_res = CityFeatureResolver.resolve_city_features("Zanzibar")
    assert custom_res["city"] == "Zanzibar"
    assert len(custom_res["hotspots"]) >= 2
    assert len(custom_res["shelters"]) >= 2
    assert -90 <= custom_res["center_lat"] <= 90
    assert -180 <= custom_res["center_lon"] <= 180


def test_flask_app_endpoints():
    """Verify Flask web dashboard routes, live datasets, and JSON endpoints."""
    from src.api.flask_app import app
    from src.data.dataset_registry import DatasetRegistry
    from src.data.realtime_weather import RealtimeWeatherEngine
    from src.data.realtime_hydrology import RealtimeHydrologyEngine
    from src.data.exposure_engine import ExposureEngine
    from src.data.crisis_social_feed import CrisisSocialFeedEngine

    app.config["TESTING"] = True
    client = app.test_client()

    # 1. Health endpoint
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.get_json()["status"] == "healthy"
    assert res_health.get_json()["datasets_count"] == 20

    # 2. Main HTML dashboard
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert b"Flood-AI" in res_index.data
    assert b"datasets" in res_index.data or b"Datasets" in res_index.data

    # 3. City features API (Live Mode)
    res_city = client.get("/api/city-features?city=Paris&mode=live")
    assert res_city.status_code == 200
    city_json = res_city.get_json()
    assert city_json["city"] == "Paris"
    assert "hotspots" in city_json
    assert len(city_json["hotspots"]) > 0
    assert "weather" in city_json
    assert "hydrology" in city_json
    assert "crisis_feed" in city_json
    assert "exposure_summary" in city_json

    # 4. Datasets API endpoint (All 20 Datasets)
    res_ds = client.get("/api/datasets")
    assert res_ds.status_code == 200
    ds_json = res_ds.get_json()
    assert ds_json["count"] == 20
    assert len(ds_json["datasets"]) == 20

    # 5. Siren audio endpoint
    res_audio = client.get("/api/siren-audio")
    assert res_audio.status_code == 200
    assert "data:audio/wav;base64," in res_audio.get_json()["audio_data_uri"]


def test_20_datasets_registry_and_models():
    """Verify 20 official multi-sensor datasets catalog and ingestion engines."""
    from src.data.dataset_registry import DatasetRegistry
    from src.data.exposure_engine import ExposureEngine
    from src.data.crisis_social_feed import CrisisSocialFeedEngine

    catalog = DatasetRegistry.load_catalog()
    assert len(catalog) == 20

    # Check key datasets
    names = [d["dataset_name"] for d in catalog]
    assert "Sen1Floods11" in names
    assert "xBD" in names
    assert "GPM IMERG" in names
    assert "USGS Water Data" in names
    assert "WorldPop" in names
    assert "CrisisMMD" in names
    assert "HumAID" in names

    # Check WorldPop & xBD Exposure calculations
    exp = ExposureEngine.calculate_hotspot_exposure("Dhaka", radius_m=1500, flood_probability=0.88, hotspot_name="Buriganga Basin")
    assert exp["exposed_population"] > 0
    assert exp["xbd_total_buildings"] > 0
    assert exp["xbd_damage_breakdown"]["destroyed_level3"] > 0

    # Check Crisis Social Feed
    feed = CrisisSocialFeedEngine.get_city_crisis_feed("Dhaka", 23.81, 90.41, count=3)
    assert len(feed) == 3
    assert feed[0]["verified"] is True
    assert "CrisisMMD" in feed[0]["source"]



