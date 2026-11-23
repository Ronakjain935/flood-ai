"""
Flood-AI: Universal Geospatial Early Warning & Siren Alarm Web Platform (Flask Edition).
Integrates the 20 official multi-sensor datasets: live satellite SAR & optical,
real-time precipitation (Open-Meteo, GPM IMERG, NEXRAD), live river streamgage telemetry
(USGS NWIS, BWDB, CWC, GRDC), high-resolution population exposure (WorldPop 100m),
structural damage assessment (DARPA xBD), and real-time citizen triage (CrisisMMD, HumAID).
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response

# Ensure project root is importable
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.city_feature_resolver import CityFeatureResolver, GLOBAL_CITIES_CATALOG
from src.data.dataset_registry import DatasetRegistry
from src.data.realtime_weather import RealtimeWeatherEngine, WeatherTelemetry
from src.data.realtime_hydrology import RealtimeHydrologyEngine, HydrologyTelemetry
from src.data.crisis_social_feed import CrisisSocialFeedEngine
from src.data.exposure_engine import ExposureEngine
from src.api.alarm_sound import generate_emergency_siren_wav
from src.inference.alert_generator import AlertSeverity, FloodAlertGenerator

# Configure Flask app with explicit template folder
template_dir = Path(__file__).resolve().parent / "templates"
app = Flask(__name__, template_folder=str(template_dir))


def compute_simulation_state(
    city_query: str = "Dhaka",
    sim_stage: float = 4.4,
    sim_rate: float = 0.35,
    sim_prob_boost: float = 1.0,
    sim_inundation: float = 650.0,
    mode: str = "live",
) -> Dict[str, Any]:
    """Computes complete hydro-risk state, red spots, exposure, and alert status for a given city."""
    region_info = CityFeatureResolver.resolve_city_features(city_query)
    city_name = region_info.get("city", city_query.title())
    country = region_info.get("country", "")
    danger_m = region_info.get("danger_stage_m", 4.0)
    emergency_m = region_info.get("emergency_stage_m", 4.8)
    center_lat = region_info.get("center_lat", 23.8103)
    center_lon = region_info.get("center_lon", 90.4125)

    is_live_mode = (str(mode).lower() == "live")

    # 1. Weather Ingestion (Live Open-Meteo & GPM IMERG vs Simulation)
    if is_live_mode:
        weather: WeatherTelemetry = RealtimeWeatherEngine.fetch_live_weather(
            city=city_name,
            lat=center_lat,
            lon=center_lon,
        )
        hydro: HydrologyTelemetry = RealtimeHydrologyEngine.fetch_live_hydrology(
            city=city_name,
            danger_m=danger_m,
            emergency_m=emergency_m,
            rain_rate_mm_h=weather.current_rain_mm_h,
        )
        effective_stage = hydro.stage_m
        effective_rate = hydro.rate_of_rise_m_hr
        effective_inundation = round(min(3200.0, max(120.0, 340.0 * ((effective_stage / danger_m) ** 1.8))), 1)
        sim_stage = effective_stage
        sim_rate = effective_rate
        sim_inundation = effective_inundation
    else:
        weather = RealtimeWeatherEngine._generate_climatological_baseline(city_name, center_lat, center_lon)
        weather.current_rain_mm_h = round(sim_rate * 42.0, 1)
        weather.source = "Simulated Climatological Stress Test (Manual Controls)"
        hydro = RealtimeHydrologyEngine.fetch_live_hydrology(
            city=city_name,
            danger_m=danger_m,
            emergency_m=emergency_m,
            rain_rate_mm_h=weather.current_rain_mm_h,
        )
        hydro.stage_m = sim_stage
        hydro.rate_of_rise_m_hr = sim_rate
        effective_stage = sim_stage
        effective_rate = sim_rate
        effective_inundation = sim_inundation

    # 2. Classify Alert Severity
    alert_gen = FloodAlertGenerator(stage_threshold_warning_m=danger_m, stage_threshold_emergency_m=emergency_m)
    peak_forecast_stage = float(effective_stage + (effective_rate * 4.0))

    alert = alert_gen.generate_advisory(
        inundation_hectares=float(effective_inundation),
        max_forecast_stage_m=peak_forecast_stage,
        rate_of_rise_m_hr=float(effective_rate),
        region_name=f"{city_name} ({region_info.get('basin', '')})",
    )

    # 3. Dynamic Hotspots with WorldPop (Dataset #13) and DARPA xBD Damage (Dataset #3)
    hotspots_raw = region_info.get("hotspots", [])
    processed_hotspots = []
    max_risk_prob = 0.0
    total_exposed_population = 0
    total_buildings_at_risk = 0
    total_estimated_loss_usd_m = 0.0

    for h in hotspots_raw:
        stage_ratio = effective_stage / h["danger_m"]
        calc_prob = min(0.99, max(0.15, h["base_prob"] * (stage_ratio ** 1.3) * sim_prob_boost))
        max_risk_prob = max(max_risk_prob, calc_prob)

        if calc_prob >= 0.75:
            category = "CRITICAL (RED)"
            color_hex = "#ef4444"
            fill_color = "#dc2626"
            radius_m = 1200 + int(calc_prob * 1800)
        elif calc_prob >= 0.50:
            category = "WARNING (ORANGE)"
            color_hex = "#f97316"
            fill_color = "#ea580c"
            radius_m = 900 + int(calc_prob * 1200)
        else:
            category = "SAFE (GREEN)"
            color_hex = "#22c55e"
            fill_color = "#16a34a"
            radius_m = 700

        # WorldPop & xBD Exposure Calculation
        exposure = ExposureEngine.calculate_hotspot_exposure(
            city=city_name,
            radius_m=radius_m,
            flood_probability=calc_prob,
            hotspot_name=h["name"],
        )
        total_exposed_population += exposure["exposed_population"]
        total_buildings_at_risk += exposure["xbd_total_buildings"]
        total_estimated_loss_usd_m += exposure["estimated_loss_usd_millions"]

        processed_hotspots.append({
            "name": h["name"],
            "lat": h["lat"],
            "lon": h["lon"],
            "river": h.get("river", region_info.get("basin", "River Channel")),
            "stage_m": round(effective_stage * (h["stage_m"] / danger_m), 2),
            "danger_m": h["danger_m"],
            "probability": round(calc_prob, 3),
            "prob_percent": f"{int(calc_prob * 100)}%",
            "risk_category": category,
            "population": exposure["exposed_population_formatted"],
            "exposed_population_num": exposure["exposed_population"],
            "buildings_at_risk": exposure["xbd_total_buildings"],
            "damage_breakdown": exposure["xbd_damage_breakdown"],
            "estimated_loss_usd_m": exposure["estimated_loss_usd_millions"],
            "color_hex": color_hex,
            "fill_color": fill_color,
            "radius_m": radius_m,
        })

    # 4. OpenStreetMap Emergency Shelters (Dataset #14)
    shelters_raw = region_info.get("shelters", [])
    processed_shelters = []
    for s in shelters_raw:
        processed_shelters.append({
            "name": s["name"],
            "lat": s["lat"],
            "lon": s["lon"],
            "capacity": s.get("capacity", 2500),
            "color_hex": "#10b981",
        })

    # 5. Live Crisis Social Media Feed (CrisisMMD #17 & HumAID #18)
    hotspot_names = [h["name"] for h in processed_hotspots]
    crisis_feed = CrisisSocialFeedEngine.get_city_crisis_feed(
        city=city_name,
        center_lat=center_lat,
        center_lon=center_lon,
        hotspot_names=hotspot_names,
    )

    # 6. Hydrograph: 0-72h Hydrograph Driven by Live Rain Forecast
    hours = list(range(1, 25))
    forecast_rain = weather.hourly_forecast_rain_mm or [weather.current_rain_mm_h] * 24
    
    # Model river hydrograph trajectory dynamically coupled to rainfall nowcast
    t_median = []
    accum = 0.0
    for idx, h in enumerate(hours):
        rain_val = forecast_rain[idx] if idx < len(forecast_rain) else forecast_rain[-1]
        accum += (rain_val * 0.018)
        drainage = (h * 0.012)
        stage_proj = round(float(effective_stage + accum - drainage + (effective_rate * 1.5 * np.sin(np.pi * (h / 24.0)))), 2)
        t_median.append(max(0.5, stage_proj))

    t_p10 = [round(max(0.0, m - 0.25), 2) for m in t_median]
    t_p90 = [round(m + 0.35, 2) for m in t_median]

    should_alarm = alert.level in [AlertSeverity.WARNING, AlertSeverity.EMERGENCY] or max_risk_prob >= 0.70

    return {
        "city": city_name,
        "country": country,
        "flag": region_info.get("flag", "📍"),
        "center_lat": center_lat,
        "center_lon": center_lon,
        "zoom": region_info.get("zoom", 12),
        "basin": region_info.get("basin", ""),
        "flood_type": region_info.get("flood_type", "Fluvial & Storm Surge Inundation"),
        "language": region_info.get("language", "English"),
        "danger_stage_m": danger_m,
        "emergency_stage_m": emergency_m,
        "mode": "live" if is_live_mode else "simulation",
        "is_live_mode": is_live_mode,
        "sim_stage": effective_stage,
        "sim_rate": effective_rate,
        "sim_prob_boost": sim_prob_boost,
        "sim_inundation": effective_inundation,
        "max_risk_prob": round(max_risk_prob, 3),
        "max_risk_percent": int(max_risk_prob * 100),
        "severity_level": alert.level.value,
        "headline": alert.headline,
        "english_advisory": alert.english_advisory,
        "local_advisory": region_info.get("local_advisory", alert.local_advisory),
        "evacuation_routes": region_info.get("evacuation_routes", alert.evacuation_routes),
        "hotspots": processed_hotspots,
        "shelters": processed_shelters,
        "crisis_feed": crisis_feed,
        "weather": {
            "current_rain_mm_h": weather.current_rain_mm_h,
            "accum_24h_mm": weather.precipitation_accum_24h_mm,
            "temperature_c": weather.temperature_c,
            "humidity_pct": weather.relative_humidity_pct,
            "pressure_hpa": weather.surface_pressure_hpa,
            "wind_kmh": weather.wind_speed_kmh,
            "condition": weather.weather_condition,
            "source": weather.source,
            "is_live": weather.is_live,
        },
        "hydrology": {
            "station_name": hydro.station_name,
            "station_id": hydro.station_id,
            "stage_m": hydro.stage_m,
            "discharge_cfs": hydro.discharge_cfs,
            "rate_of_rise": hydro.rate_of_rise_m_hr,
            "stage_status": hydro.stage_status,
            "source": hydro.source,
            "is_live": hydro.is_live,
        },
        "exposure_summary": {
            "total_exposed_population": f"{total_exposed_population:,}",
            "total_buildings_at_risk": f"{total_buildings_at_risk:,}",
            "total_estimated_loss_usd_m": round(total_estimated_loss_usd_m, 2),
            "exposure_provenance": "WorldPop (Dataset #13) & DARPA xBD (Dataset #3)",
        },
        "provenance": DatasetRegistry.get_provenance_summary(),
        "datasets_catalog": DatasetRegistry.load_catalog(),
        "hydrograph": {
            "hours": hours,
            "p10": t_p10,
            "p50": t_median,
            "p90": t_p90,
            "danger_line": [danger_m] * 24,
            "emergency_line": [emergency_m] * 24,
        },
        "should_alarm": should_alarm,
    }


@app.route("/")
def index():
    """Main dashboard page."""
    city_query = request.args.get("city", "Dhaka")
    mode = request.args.get("mode", "live")
    initial_data = compute_simulation_state(city_query=city_query, mode=mode)
    preset_cities = [
        "Dhaka, Bangladesh",
        "Mumbai, India",
        "Tokyo, Japan",
        "Miami, USA",
        "Paris, France",
        "London, UK",
        "Kolkata, India",
        "Sylhet, Bangladesh",
        "Jakarta, Indonesia",
        "Venice, Italy",
        "Houston, USA",
        "New Orleans, USA",
        "Cologne, Germany",
        "Porto Alegre, Brazil",
        "Cairo, Egypt",
        "Sydney, Australia",
        "Bangkok, Thailand",
    ]
    return render_template(
        "index.html",
        data=initial_data,
        preset_cities=preset_cities,
        datasets=DatasetRegistry.load_catalog(),
    )


@app.route("/api/city-features", methods=["GET", "POST"])
def api_city_features():
    """API endpoint to identify features for any city and recalculate state."""
    if request.method == "POST":
        req_data = request.get_json() or {}
        city_query = req_data.get("city", "Dhaka")
        sim_stage = float(req_data.get("sim_stage", 4.4))
        sim_rate = float(req_data.get("sim_rate", 0.35))
        sim_prob_boost = float(req_data.get("sim_prob_boost", 1.0))
        sim_inundation = float(req_data.get("sim_inundation", 650.0))
        mode = req_data.get("mode", "live")
    else:
        city_query = request.args.get("city", "Dhaka")
        sim_stage = float(request.args.get("sim_stage", 4.4))
        sim_rate = float(request.args.get("sim_rate", 0.35))
        sim_prob_boost = float(request.args.get("sim_prob_boost", 1.0))
        sim_inundation = float(request.args.get("sim_inundation", 650.0))
        mode = request.args.get("mode", "live")

    state = compute_simulation_state(
        city_query=city_query,
        sim_stage=sim_stage,
        sim_rate=sim_rate,
        sim_prob_boost=sim_prob_boost,
        sim_inundation=sim_inundation,
        mode=mode,
    )
    return jsonify(state)


@app.route("/api/datasets")
def api_datasets():
    """Returns the official 20-dataset architecture catalog as JSON."""
    return jsonify({
        "status": "success",
        "count": len(DatasetRegistry.load_catalog()),
        "datasets": DatasetRegistry.load_catalog(),
    })


@app.route("/api/siren-audio")
def api_siren_audio():
    """Returns synthesized base64 emergency warning audio."""
    wav_b64 = generate_emergency_siren_wav(duration_sec=2.5)
    return jsonify({"audio_data_uri": wav_b64})


@app.route("/health")
def health():
    """Service health check."""
    return jsonify({
        "status": "healthy",
        "service": "flood-ai-flask",
        "version": "2.1.0-realtime",
        "datasets_count": len(DatasetRegistry.load_catalog()),
    })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Flood-AI Flask Early Warning Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug reload")
    args = parser.parse_args()

    print("=" * 65)
    print("🌊 Flood-AI: Real-Time Early Warning & Siren Alarm System (Flask)")
    print(f"🚀 Running locally on: http://{args.host}:{args.port}")
    print(f"📦 Registered Multi-Sensor Datasets: {len(DatasetRegistry.load_catalog())}")
    print("=" * 65)
    app.run(host=args.host, port=args.port, debug=args.debug)
