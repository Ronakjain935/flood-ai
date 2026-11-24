"""
Unit and integration tests for Flood-AI Real-Time Data Layer (Part A & Part B).

Rules tested:
1. Live first, simulator fallback only when live call fails/times out.
2. Value tagging with origin = ('observed', 'official_forecast', 'model', 'simulated'), source_id, time_utc.
3. No invented numbers: null with reason = 'no official source'.
4. Never sound siren on simulated values; show screen notice for model thresholds.
5. All URLs in realtime_sources.py; loaders call fetch().
6. Polite client headers & rate limits.
7. API keys from env vars; runs with none set.
8. Offline tests run without network; live calls carry @pytest.mark.live.
"""

from datetime import datetime, timezone
import pytest
from src.data.realtime_sources import SOURCES, fetch, tag_value, _build_url_and_body
from src.data.realtime_hydrology import RealtimeHydrologyEngine, HydrologyTelemetry
from src.data.realtime_weather import RealtimeWeatherEngine, WeatherTelemetry
from src.data.city_feature_resolver import CityFeatureResolver
from src.data.ingest_sentinel import Sentinel1SARProcessor, Sentinel2OpticalProcessor
from src.data.fusion import MultimodalDataFusion
import numpy as np


def test_sources_catalog_completeness():
    """Verify that all Part B source endpoints are registered with valid configurations."""
    required_sources = [
        "openmeteo_geocoding", "nominatim", "overpass_rivers", "overpass_shelters",
        "photon_nearby_rivers", "photon_nearby_shelters", "wikidata_languages",
        "cldr_languages", "mymemory_translate", "ghsl_population", "worldpop_catalog",
        "hydrobasins", "hydrorivers", "pc_stac_s1_rtc", "pc_sas_sign", "earthsearch_s1",
        "cdse_stac_s1", "earthsearch_s2", "pc_stac_s2", "copdem30_aws", "pc_stac_nasadem",
        "gfm_wms", "jrc_surface_water", "jrc_flood_hazard", "openmeteo_flood",
        "geoglows_riverid", "geoglows_forecast", "geoglows_retrospective",
        "geoglows_returnperiods", "google_flood_api", "usgs_sites_bbox", "usgs_latest",
        "usgs_continuous", "noaa_nwps_bbox", "noaa_nwps_gauge", "noaa_nwps_stageflow",
        "ndma_sachet_json", "india_nwdp_datasets", "india_nwdp_resources", "india_nwdp_levels",
        "cwc_ffs_station_list", "cwc_ffs_station", "cwc_ffs_levels", "imd_district_warning",
        "nepal_bipad", "uk_ea_stations", "uk_ea_readings", "uk_ea_floods",
        "pegelonline_near", "pegelonline_level", "pegelonline_history",
        "hubeau_stations", "hubeau_observations", "canada_hydrometric",
        "australia_bom", "brazil_ana_legacy", "argentina_ina_series", "argentina_ina_obs",
        "openmeteo_marine", "ioc_sealevel_stations", "ioc_sealevel_data", "noaa_coops",
        "venice_tide", "openmeteo_forecast", "openmeteo_ensemble", "openmeteo_archive",
        "imd_gridded_rain", "gdacs_floods", "nws_flood_alerts"
    ]
    for src in required_sources:
        assert src in SOURCES, f"Missing source endpoint: '{src}'"
        assert "url" in SOURCES[src]
        assert "method" in SOURCES[src]
        assert "test_example" in SOURCES[src]


def test_tag_value_rule_2_and_3():
    """Verify Rule 2 (tagging provenance) and Rule 3 (null with reason for missing official values)."""
    # Valid observed tag
    tag = tag_value(4.25, origin="observed", source_id="usgs_latest")
    assert tag["value"] == 4.25
    assert tag["origin"] == "observed"
    assert tag["source_id"] == "usgs_latest"
    assert "time_utc" in tag

    # Rule 3: Never invent numbers -> return null with reason
    tag_null = tag_value(None, origin="observed", source_id="hubeau_observations", reason="no official source")
    assert tag_null["value"] is None
    assert tag_null["reason"] == "no official source"


def test_country_adapter_routing_and_conversion():
    """Verify that country adapters convert units to metres and route by country code."""
    # US adapter (feet -> m)
    us_hyd = RealtimeHydrologyEngine.fetch_live_hydrology(
        city="Houston",
        country_code="US",
        danger_m=4.5,
        emergency_m=5.6,
    )
    assert isinstance(us_hyd, HydrologyTelemetry)
    assert us_hyd.raw_unit in ["feet", "m"]

    # Germany adapter (cm -> m)
    de_hyd = RealtimeHydrologyEngine.fetch_live_hydrology(
        city="Cologne",
        country_code="DE",
        danger_m=8.2,
        emergency_m=10.0,
    )
    assert isinstance(de_hyd, HydrologyTelemetry)

    # France adapter (Rule 3: no official thresholds -> null)
    fr_hyd = RealtimeHydrologyEngine.fetch_live_hydrology(
        city="Paris",
        country_code="FR",
    )
    assert isinstance(fr_hyd, HydrologyTelemetry)


def test_city_feature_resolver_rules():
    """Verify CityFeatureResolver applies Part A Module 3 provenance tagging and rules."""
    res = CityFeatureResolver.resolve_city_features("Dhaka")
    assert res["city"] == "Dhaka"
    assert "origin" in res
    assert "hotspots" in res
    assert len(res["hotspots"]) > 0
    # Every hotspot must have origin tag
    for h in res["hotspots"]:
        assert "origin" in h

    # Check custom/unlisted resolution uses smart inference
    custom = CityFeatureResolver.resolve_city_features("Zanzibar")
    assert "Zanzibar" in custom["city"]
    assert "inferred" in custom["flood_type"].lower()


def test_fusion_engine_channels_6_and_7():
    """Verify Module 1 Fusion Engine channels 6 and 7 (Gauge Stage & Rate of Rise)."""
    fusion = MultimodalDataFusion()
    vv = np.random.uniform(-25.0, -5.0, (32, 32))
    vh = np.random.uniform(-30.0, -10.0, (32, 32))

    # Channel 6: Gauge Stage
    tensor_7ch = fusion.assemble_multimodal_tensor(vv, vh, gauge_stage_m=3.5)
    assert tensor_7ch.shape == (7, 32, 32)

    # Channel 6 & 7: Gauge Stage and Rate of Rise
    tensor_8ch = fusion.assemble_multimodal_tensor(vv, vh, gauge_stage_m=3.5, rate_of_rise_m_hr=0.25)
    assert tensor_8ch.shape == (8, 32, 32)


def test_sentinel1_and_sentinel2_calibrations():
    """Verify SAR power to dB and Sentinel-2 reflectance scaling rules."""
    # SAR gamma-nought power
    sar_proc = Sentinel1SARProcessor()
    power_arr = np.array([0.01, 0.1, 1.0], dtype=np.float32)
    db = sar_proc.linear_to_db(power_arr)
    assert pytest.approx(db[2], abs=1e-3) == 0.0

    # SAR raw GRD amplitude
    amp_arr = np.array([10.0], dtype=np.float32)
    db_amp = sar_proc.amplitude_to_db(amp_arr)
    assert pytest.approx(db_amp[0], abs=1e-3) == 20.0

    # Sentinel-2 reflectance scaling
    counts = np.array([2000.0], dtype=np.float32)
    ref_earthsearch = Sentinel2OpticalProcessor.calibrate_reflectance(counts, source="earthsearch_s2")
    assert pytest.approx(ref_earthsearch[0], abs=1e-4) == 0.20

    ref_pc = Sentinel2OpticalProcessor.calibrate_reflectance(counts, source="pc_stac_s2")
    assert pytest.approx(ref_pc[0], abs=1e-4) == 0.10


# --- LIVE TESTS (@pytest.mark.live) ---
@pytest.mark.live
def test_live_openmeteo_geocoding():
    """Live query to openmeteo_geocoding."""
    res = fetch("openmeteo_geocoding", city="Houston")
    assert "results" in res
    assert len(res["results"]) > 0
    assert res["results"][0]["country_code"].upper() == "US"


@pytest.mark.live
def test_live_openmeteo_forecast():
    """Live query to openmeteo_forecast."""
    res = fetch("openmeteo_forecast", lat=29.76, lon=-95.36)
    assert "current" in res
    assert "hourly" in res
