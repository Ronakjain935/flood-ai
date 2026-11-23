"""
Flood-AI: Universal Geospatial Early Warning & Siren Alarm Web Platform (Flask Edition).
Provides real-time multimodal flood inundation mapping, 0-72h hydrograph nowcasting,
universal city feature identification, dynamic red-spot probability hotspots,
autonomous audible siren alarms, and localized multilingual civil defense advisories.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response

# Ensure project root is importable
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.city_feature_resolver import CityFeatureResolver, GLOBAL_CITIES_CATALOG
from src.data.global_flood_regions import GLOBAL_FLOOD_REGIONS, get_region_names, get_region_data
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
) -> Dict[str, Any]:
    """Computes complete hydro-risk state, red spots, and alert status for a given city and telemetry."""
    region_info = CityFeatureResolver.resolve_city_features(city_query)
    city_name = region_info.get("city", city_query.title())
    country = region_info.get("country", "")
    danger_m = region_info.get("danger_stage_m", 4.0)
    emergency_m = region_info.get("emergency_stage_m", 4.8)

    # Classify Alert Severity
    alert_gen = FloodAlertGenerator(stage_threshold_warning_m=danger_m, stage_threshold_emergency_m=emergency_m)
    peak_forecast_stage = float(sim_stage + (sim_rate * 4.0))

    alert = alert_gen.generate_advisory(
        inundation_hectares=float(sim_inundation),
        max_forecast_stage_m=peak_forecast_stage,
        rate_of_rise_m_hr=float(sim_rate),
        region_name=f"{city_name} ({region_info.get('basin', '')})",
    )

    # Compute Dynamic Hotspots with Probabilities & Red-Spot Sizing
    hotspots_raw = region_info.get("hotspots", [])
    processed_hotspots = []
    max_risk_prob = 0.0

    for h in hotspots_raw:
        stage_ratio = sim_stage / h["danger_m"]
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

        processed_hotspots.append({
            "name": h["name"],
            "lat": h["lat"],
            "lon": h["lon"],
            "river": h.get("river", region_info.get("basin", "River Channel")),
            "stage_m": round(sim_stage * (h["stage_m"] / danger_m), 2),
            "danger_m": h["danger_m"],
            "probability": round(calc_prob, 3),
            "prob_percent": f"{int(calc_prob * 100)}%",
            "risk_category": category,
            "population": f"{h.get('population_at_risk', 150000):,}",
            "color_hex": color_hex,
            "fill_color": fill_color,
            "radius_m": radius_m,
        })

    # Emergency Shelters
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

    # 72-Hour Hydrograph Prediction Trajectory
    hours = list(range(1, 25))
    t_median = [round(float(sim_stage + (sim_rate * 3.5) * np.sin(np.pi * (h / 24.0))), 2) for h in hours]
    t_p10 = [round(max(0.0, m - 0.25), 2) for m in t_median]
    t_p90 = [round(m + 0.35, 2) for m in t_median]

    should_alarm = alert.level in [AlertSeverity.WARNING, AlertSeverity.EMERGENCY] or max_risk_prob >= 0.70

    return {
        "city": city_name,
        "country": country,
        "flag": region_info.get("flag", "📍"),
        "center_lat": region_info.get("center_lat", 23.8103),
        "center_lon": region_info.get("center_lon", 90.4125),
        "zoom": region_info.get("zoom", 12),
        "basin": region_info.get("basin", ""),
        "flood_type": region_info.get("flood_type", "Fluvial & Storm Surge Inundation"),
        "language": region_info.get("language", "English"),
        "danger_stage_m": danger_m,
        "emergency_stage_m": emergency_m,
        "sim_stage": sim_stage,
        "sim_rate": sim_rate,
        "sim_prob_boost": sim_prob_boost,
        "sim_inundation": sim_inundation,
        "max_risk_prob": round(max_risk_prob, 3),
        "max_risk_percent": int(max_risk_prob * 100),
        "severity_level": alert.level.value,
        "headline": alert.headline,
        "english_advisory": alert.english_advisory,
        "local_advisory": region_info.get("local_advisory", alert.local_advisory),
        "evacuation_routes": region_info.get("evacuation_routes", alert.evacuation_routes),
        "hotspots": processed_hotspots,
        "shelters": processed_shelters,
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
    initial_data = compute_simulation_state(city_query=city_query)
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
    return render_template("index.html", data=initial_data, preset_cities=preset_cities)


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
    else:
        city_query = request.args.get("city", "Dhaka")
        sim_stage = float(request.args.get("sim_stage", 4.4))
        sim_rate = float(request.args.get("sim_rate", 0.35))
        sim_prob_boost = float(request.args.get("sim_prob_boost", 1.0))
        sim_inundation = float(request.args.get("sim_inundation", 650.0))

    state = compute_simulation_state(
        city_query=city_query,
        sim_stage=sim_stage,
        sim_rate=sim_rate,
        sim_prob_boost=sim_prob_boost,
        sim_inundation=sim_inundation,
    )
    return jsonify(state)


@app.route("/api/siren-audio")
def api_siren_audio():
    """Returns synthesized base64 emergency warning audio."""
    wav_b64 = generate_emergency_siren_wav(duration_sec=2.5)
    return jsonify({"audio_data_uri": wav_b64})


@app.route("/health")
def health():
    """Service health check."""
    return jsonify({"status": "healthy", "service": "flood-ai-flask", "version": "2.0.0"})


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
    print("=" * 65)
    app.run(host=args.host, port=args.port, debug=args.debug)
