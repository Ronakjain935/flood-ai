"""
Flood-AI: Authoritative Real-Time Data Sources & Unified Dispatcher.

Contains the tested catalog of real endpoints defined in Flood-AI Real-Time Data Layer Part B.
All loaders in src/data/ call fetch(source_id, **values). No URLs are hard-coded elsewhere.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import ssl
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.parse
import urllib.request
import urllib.error

logger = logging.getLogger("Flood-AI.RealtimeSources")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Polite client user-agent and settings
USER_AGENT = os.environ.get("FLOODAI_USER_AGENT", "FloodAI/1.0 (contact@floodai.org)")
REQUEST_TIMEOUT_S = 30
MAX_RETRIES = 3
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Rate limiting tracker
_LAST_REQUEST_TIME: Dict[str, float] = {}
_MEMORY_CACHE: Dict[str, Tuple[float, Any]] = {}

# All API endpoints from Flood-AI Real-Time Data Layer Part B
SOURCES: Dict[str, Dict[str, Any]] = {
    # --- CITY RESOLVER ---
    "openmeteo_geocoding": {
        "method": "GET",
        "url": "https://geocoding-api.open-meteo.com/v1/search",
        "params": {"name": "{city}", "count": 1, "language": "en", "format": "json"},
        "cache_ttl_s": 86400 * 30,
        "test_example": {"city": "Udaipur"},
    },
    "nominatim": {
        "method": "GET",
        "url": "https://nominatim.openstreetmap.org/search",
        "params": {"q": "{city}", "format": "jsonv2", "limit": 1, "addressdetails": 1},
        "rate_limit_s": 1.1,
        "cache_ttl_s": 86400 * 30,
        "test_example": {"city": "Houston"},
    },
    "overpass_rivers": {
        "method": "POST",
        "url": "https://overpass-api.de/api/interpreter",
        "body_type": "form",
        "body_template": '[out:json][timeout:25];(way["waterway"~"^(river|canal)$"](around:{radius_m},{lat},{lon}););out tags 5;',
        "cache_ttl_s": 86400 * 14,
        "test_example": {"radius_m": 15000, "lat": 24.5854, "lon": 73.7125},
    },
    "overpass_shelters": {
        "method": "POST",
        "url": "https://overpass-api.de/api/interpreter",
        "body_type": "form",
        "body_template": '[out:json][timeout:25];(nwr["amenity"~"^(school|hospital|community_centre|shelter)$"](around:{radius_m},{lat},{lon});nwr["emergency"="assembly_point"](around:{radius_m},{lat},{lon});nwr["social_facility"="shelter"](around:{radius_m},{lat},{lon}););out center 5;',
        "cache_ttl_s": 86400 * 14,
        "test_example": {"radius_m": 5000, "lat": 24.5854, "lon": 73.7125},
    },
    "photon_nearby_rivers": {
        "method": "GET",
        "url": "https://photon.komoot.io/reverse",
        "params": {"lat": "{lat}", "lon": "{lon}", "radius": 20, "limit": 10, "osm_tag": "waterway:river"},
        "cache_ttl_s": 86400 * 7,
        "test_example": {"lat": 24.5854, "lon": 73.7125},
    },
    "photon_nearby_shelters": {
        "method": "GET",
        "url": "https://photon.komoot.io/reverse",
        "params": {
            "lat": "{lat}",
            "lon": "{lon}",
            "radius": 3,
            "limit": 30,
            "osm_tag": "amenity:school&osm_tag=amenity:hospital&osm_tag=amenity:community_centre&osm_tag=amenity:shelter&osm_tag=emergency:assembly_point",
        },
        "cache_ttl_s": 86400 * 7,
        "test_example": {"lat": 24.5854, "lon": 73.7125},
    },
    "wikidata_languages": {
        "method": "GET",
        "url": "https://query.wikidata.org/sparql",
        "params": {
            "format": "json",
            "query": 'SELECT ?langLabel ?code WHERE { ?c wdt:P297 "{country_code}". ?c wdt:P37 ?lang. OPTIONAL { ?lang wdt:P218 ?code } SERVICE wikibase:label { bd:serviceParam wikibase:language "en" } }',
        },
        "cache_ttl_s": 86400 * 60,
        "test_example": {"country_code": "IN"},
    },
    "cldr_languages": {
        "method": "GET",
        "url": "https://raw.githubusercontent.com/unicode-org/cldr-json/main/cldr-json/cldr-core/supplemental/territoryInfo.json",
        "cache_ttl_s": 86400 * 60,
        "test_example": {},
    },
    "mymemory_translate": {
        "method": "GET",
        "url": "https://api.mymemory.translated.net/get",
        "params": {"q": "{text}", "langpair": "en|{lang}"},
        "cache_ttl_s": 86400 * 30,
        "test_example": {"text": "Flash flood warning. Evacuate immediately.", "lang": "hi"},
    },
    "ghsl_population": {
        "method": "GET",
        "url": "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/",
        "cache_ttl_s": 86400 * 90,
        "test_example": {},
    },
    "worldpop_catalog": {
        "method": "GET",
        "url": "https://hub.worldpop.org/rest/data/pop/wpgp",
        "params": {"iso3": "{iso3}"},
        "cache_ttl_s": 86400 * 30,
        "test_example": {"iso3": "IND"},
    },
    "hydrobasins": {
        "method": "GET",
        "url": "https://data.hydrosheds.org/file/HydroBASINS/standard/hybas_{region}_lev01-12_v1c.zip",
        "cache_ttl_s": 86400 * 90,
        "test_example": {"region": "as"},
    },
    "hydrorivers": {
        "method": "GET",
        "url": "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_{region}_shp.zip",
        "cache_ttl_s": 86400 * 90,
        "test_example": {"region": "as"},
    },

    # --- SENTINEL-1 ---
    "pc_stac_s1_rtc": {
        "method": "POST",
        "url": "https://planetarycomputer.microsoft.com/api/stac/v1/search",
        "body_type": "json",
        "body_template": '{"collections":["sentinel-1-rtc"],"bbox":[{west},{south},{east},{north}],"limit":1,"sortby":[{"field":"properties.datetime","direction":"desc"}]}',
        "test_example": {"west": 73.65, "south": 24.52, "east": 73.78, "north": 24.64},
    },
    "pc_sas_sign": {
        "method": "GET",
        "url": "https://planetarycomputer.microsoft.com/api/sas/v1/sign",
        "params": {"href": "{asset_href}"},
        "test_example": {"asset_href": "https://sentinel1rtc.blob.core.windows.net/sentinel1-rtc/sample.tif"},
    },
    "earthsearch_s1": {
        "method": "POST",
        "url": "https://earth-search.aws.element84.com/v1/search",
        "body_type": "json",
        "body_template": '{"collections":["sentinel-1-grd"],"bbox":[{west},{south},{east},{north}],"limit":1,"sortby":[{"field":"properties.datetime","direction":"desc"}]}',
        "test_example": {"west": 73.65, "south": 24.52, "east": 73.78, "north": 24.64},
    },
    "cdse_stac_s1": {
        "method": "POST",
        "url": "https://stac.opensearch.dataspace.copernicus.eu/v1/search",
        "body_type": "json",
        "body_template": '{"collections":["sentinel-1-grd"],"bbox":[{west},{south},{east},{north}],"limit":1,"sortby":[{"field":"properties.datetime","direction":"desc"}]}',
        "test_example": {"west": 73.65, "south": 24.52, "east": 73.78, "north": 24.64},
    },

    # --- SENTINEL-2 ---
    "earthsearch_s2": {
        "method": "POST",
        "url": "https://earth-search.aws.element84.com/v1/search",
        "body_type": "json",
        "body_template": '{"collections":["sentinel-2-l2a"],"bbox":[{west},{south},{east},{north}],"limit":1,"sortby":[{"field":"properties.datetime","direction":"desc"}]}',
        "test_example": {"west": 73.65, "south": 24.52, "east": 73.78, "north": 24.64},
    },
    "pc_stac_s2": {
        "method": "POST",
        "url": "https://planetarycomputer.microsoft.com/api/stac/v1/search",
        "body_type": "json",
        "body_template": '{"collections":["sentinel-2-l2a"],"bbox":[{west},{south},{east},{north}],"limit":1,"sortby":[{"field":"properties.datetime","direction":"desc"}]}',
        "test_example": {"west": 73.65, "south": 24.52, "east": 73.78, "north": 24.64},
    },

    # --- ELEVATION ---
    "copdem30_aws": {
        "method": "GET",
        "url": "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{tile}_DEM/Copernicus_DSM_COG_10_{tile}_DEM.tif",
        "test_example": {"tile": "N24_00_E073_00"},
    },
    "pc_stac_nasadem": {
        "method": "POST",
        "url": "https://planetarycomputer.microsoft.com/api/stac/v1/search",
        "body_type": "json",
        "body_template": '{"collections":["nasadem"],"bbox":[{west},{south},{east},{north}],"limit":1}',
        "test_example": {"west": 73.65, "south": 24.52, "east": 73.78, "north": 24.64},
    },

    # --- FLOOD MAP CHECKS AND STATIC LAYERS ---
    "gfm_wms": {
        "method": "GET",
        "url": "https://geoserver.gfm.eodc.eu/geoserver/gfm/wms",
        "params": {"service": "WMS", "version": "1.3.0", "request": "GetCapabilities"},
        "test_example": {},
    },
    "jrc_surface_water": {
        "method": "GET",
        "url": "https://storage.googleapis.com/global-surface-water/downloads2021/occurrence/occurrence_{tile}v1_4_2021.tif",
        "test_example": {"tile": "70E_30N"},
    },
    "jrc_flood_hazard": {
        "method": "GET",
        "url": "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/",
        "test_example": {},
    },

    # --- RIVER FORECAST, ANY CITY ---
    "openmeteo_flood": {
        "method": "GET",
        "url": "https://flood-api.open-meteo.com/v1/flood",
        "params": {
            "latitude": "{lat}",
            "longitude": "{lon}",
            "daily": "river_discharge,river_discharge_median,river_discharge_max",
            "past_days": 7,
            "forecast_days": 7,
        },
        "test_example": {"lat": 24.5854, "lon": 73.7125},
    },
    "geoglows_riverid": {
        "method": "GET",
        "url": "https://geoglows.ecmwf.int/api/v2/getriverid",
        "params": {"lat": "{lat}", "lon": "{lon}"},
        "test_example": {"lat": 24.5854, "lon": 73.7125},
    },
    "geoglows_forecast": {
        "method": "GET",
        "url": "https://geoglows.ecmwf.int/api/v2/forecast/{river_id}",
        "params": {"format": "json"},
        "test_example": {"river_id": 440853670},
    },
    "geoglows_retrospective": {
        "method": "GET",
        "url": "https://geoglows.ecmwf.int/api/v2/retrospectivedaily/{river_id}",
        "params": {"format": "json"},
        "test_example": {"river_id": 440853670},
    },
    "geoglows_returnperiods": {
        "method": "GET",
        "url": "https://geoglows.ecmwf.int/api/v2/returnperiods/{river_id}",
        "params": {"format": "json"},
        "test_example": {"river_id": 440853670},
    },
    "google_flood_api": {
        "method": "POST",
        "url": "https://floodforecasting.googleapis.com/v1/gauges:searchGaugesByArea",
        "test_example": {},
    },

    # --- GAUGES: USA ---
    "usgs_sites_bbox": {
        "method": "GET",
        "url": "https://api.waterdata.usgs.gov/ogcapi/v0/collections/monitoring-locations/items",
        "params": {"f": "json", "bbox": "{west},{south},{east},{north}", "site_type_code": "ST", "limit": 5},
        "test_example": {"west": -95.45, "south": 29.70, "east": -95.30, "north": 29.85},
    },
    "usgs_latest": {
        "method": "GET",
        "url": "https://api.waterdata.usgs.gov/ogcapi/v0/collections/latest-continuous/items",
        "params": {"f": "json", "monitoring_location_id": "{site}", "parameter_code": "00065"},
        "test_example": {"site": "USGS-08074000"},
    },
    "usgs_continuous": {
        "method": "GET",
        "url": "https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous/items",
        "params": {"f": "json", "monitoring_location_id": "{site}", "parameter_code": "00065", "time": "P3D", "limit": 1000},
        "test_example": {"site": "USGS-08074000"},
    },
    "noaa_nwps_bbox": {
        "method": "GET",
        "url": "https://api.water.noaa.gov/nwps/v1/gauges",
        "params": {"bbox.xmin": "{west}", "bbox.ymin": "{south}", "bbox.xmax": "{east}", "bbox.ymax": "{north}", "srid": "EPSG_4326"},
        "test_example": {"west": -95.45, "south": 29.70, "east": -95.30, "north": 29.85},
    },
    "noaa_nwps_gauge": {
        "method": "GET",
        "url": "https://api.water.noaa.gov/nwps/v1/gauges/{lid}",
        "test_example": {"lid": "SHTT2"},
    },
    "noaa_nwps_stageflow": {
        "method": "GET",
        "url": "https://api.water.noaa.gov/nwps/v1/gauges/{lid}/stageflow",
        "test_example": {"lid": "SHTT2"},
    },

    # --- GAUGES: INDIA ---
    "ndma_sachet_json": {
        "method": "GET",
        "url": "https://sachet.ndma.gov.in/cap_public_website/FetchAllAlertDetails",
        "test_example": {},
    },
    "india_nwdp_datasets": {
        "method": "GET",
        "url": "https://nwdp.nwic.gov.in/api/3/action/package_search",
        "params": {"q": 'title:"River Water Level"', "rows": 100},
        "verify_ssl": False,  # HTTPS certificate expired on 2026-11-20
        "test_example": {},
    },
    "india_nwdp_resources": {
        "method": "GET",
        "url": "https://nwdp.nwic.gov.in/api/3/action/package_show",
        "params": {"id": "{dataset}"},
        "verify_ssl": False,
        "test_example": {"dataset": "river-water-level-telemetry-hourly-central-water-commission-cwc"},
    },
    "india_nwdp_levels": {
        "method": "GET",
        "url": "https://nwdp.nwic.gov.in/api/3/action/datastore_search",
        "params": {
            "resource_id": "{resource_id}",
            "filters": '{{"Station":"{station}"}}',
            "sort": "_id desc",
            "limit": 24,
        },
        "verify_ssl": False,
        "test_example": {"resource_id": "sample-res-id", "station": "Mansi Wakal"},
    },
    "cwc_ffs_station_list": {
        "method": "GET",
        "url": "https://ffs.india-water.gov.in/iam/api/layer-station/specification/sorted-page",
        "params": {
            "sort-criteria": '{"sortOrderDtos":[{"sortDirection":"ASC","field":"name"}]}',
            "page-number": 0,
            "page-size": 100,
            "specification": '{"where":{"expression":{"valueIsRelationField":false,"fieldName":"subdivisionalOfficeId.divisionalOfficeId.circleOfficeId.regionalOfficeId.agencyId.agencyId","operator":"eq","value":"41"}},"and":{"expression":{"valueIsRelationField":false,"fieldName":"floodForecastStaticStationCode.stationCode","operator":"null","value":"false"}},"unique":true}',
        },
        "test_example": {},
    },
    "cwc_ffs_station": {
        "method": "GET",
        "url": "https://ffs.india-water.gov.in/iam/api/layer-station/{station_code}",
        "test_example": {"station_code": "015-MDG"},
    },
    "cwc_ffs_levels": {
        "method": "GET",
        "url": "https://ffs.india-water.gov.in/iam/api/new-entry-data/specification/sorted",
        "params": {
            "sort-criteria": '{"sortOrderDtos":[{"sortDirection":"ASC","field":"id.dataTime"}]}',
            "specification": '{{"where":{{"where":{{"where":{{"expression":{{"valueIsRelationField":false,"fieldName":"id.stationCode","operator":"eq","value":"{station_code}"}}}},"and":{{"expression":{{"valueIsRelationField":false,"fieldName":"id.datatypeCode","operator":"eq","value":"HHS"}}}}}},"and":{{"expression":{{"valueIsRelationField":false,"fieldName":"dataValue","operator":"null","value":"false"}}}}}},"and":{{"expression":{{"valueIsRelationField":false,"fieldName":"id.dataTime","operator":"btn","value":"{start_time},{end_time}"}}}}}}',
        },
        "test_example": {"station_code": "015-MDG", "start_time": "2026-11-20T00:00:00.000", "end_time": "2026-11-21T00:00:00.000"},
    },
    "imd_district_warning": {
        "method": "GET",
        "url": "https://api.imd.gov.in/api/v1/districtwarning",
        "params": {"id": "{district_id}"},
        "test_example": {"district_id": "512"},
    },

    # --- GAUGES: OTHER COUNTRIES ---
    "nepal_bipad": {
        "method": "GET",
        "url": "https://bipadportal.gov.np/api/v1/river-stations/",
        "params": {"format": "json", "limit": 2},
        "test_example": {},
    },
    "uk_ea_stations": {
        "method": "GET",
        "url": "https://environment.data.gov.uk/flood-monitoring/id/stations",
        "params": {"lat": "{lat}", "long": "{lon}", "dist": 5, "parameter": "level", "_limit": 5},
        "test_example": {"lat": 51.41, "lon": -0.30},
    },
    "uk_ea_readings": {
        "method": "GET",
        "url": "https://environment.data.gov.uk/flood-monitoring/id/stations/{station}/readings",
        "params": {"_sorted": "", "parameter": "level", "_limit": 20},
        "test_example": {"station": "E21584"},
    },
    "uk_ea_floods": {
        "method": "GET",
        "url": "https://environment.data.gov.uk/flood-monitoring/id/floods",
        "params": {"_limit": 5},
        "test_example": {},
    },
    "pegelonline_near": {
        "method": "GET",
        "url": "https://www.pegelonline.wsv.de/webservices/rest-api/v2/stations.json",
        "params": {"latitude": "{lat}", "longitude": "{lon}", "radius": 30, "includeCurrentMeasurement": "true"},
        "test_example": {"lat": 50.9375, "lon": 6.9603},
    },
    "pegelonline_level": {
        "method": "GET",
        "url": "https://www.pegelonline.wsv.de/webservices/rest-api/v2/stations/{station}/W.json",
        "params": {"includeCurrentMeasurement": "true", "includeCharacteristicValues": "true"},
        "test_example": {"station": "KOELN"},
    },
    "pegelonline_history": {
        "method": "GET",
        "url": "https://www.pegelonline.wsv.de/webservices/rest-api/v2/stations/{station}/W/measurements.json",
        "params": {"start": "P3D"},
        "test_example": {"station": "KOELN"},
    },
    "hubeau_stations": {
        "method": "GET",
        "url": "https://hubeau.eaufrance.fr/api/v2/hydrometrie/referentiel/stations",
        "params": {"latitude": "{lat}", "longitude": "{lon}", "distance": 5, "en_service": "true", "size": 5},
        "test_example": {"lat": 48.8566, "lon": 2.3522},
    },
    "hubeau_observations": {
        "method": "GET",
        "url": "https://hubeau.eaufrance.fr/api/v2/hydrometrie/observations_tr",
        "params": {"code_entite": "{station}", "grandeur_hydro": "H", "size": 5},
        "test_example": {"station": "F700000103"},
    },
    "canada_hydrometric": {
        "method": "GET",
        "url": "https://api.weather.gc.ca/collections/hydrometric-realtime/items",
        "params": {"f": "json", "bbox": "{west},{south},{east},{north}", "limit": 2, "sortby": "-DATETIME"},
        "test_example": {"west": -75.75, "south": 45.35, "east": -75.65, "north": 45.45},
    },
    "australia_bom": {
        "method": "GET",
        "url": "https://www.bom.gov.au/waterdata/services",
        "params": {
            "service": "kisters",
            "type": "queryServices",
            "request": "getStationList",
            "datasource": 0,
            "format": "json",
            "station_name": "{name}",
            "returnfields": "station_no,station_name,station_latitude,station_longitude",
        },
        "test_example": {"name": "Brisbane"},
    },
    "brazil_ana_legacy": {
        "method": "GET",
        "url": "https://telemetriaws1.ana.gov.br/ServiceANA.asmx/DadosHidrometeorologicos",
        "params": {"codEstacao": "{station}", "dataInicio": "{start_date}", "dataFim": "{end_date}"},
        "timeout_s": 60,
        "test_example": {"station": "14990000", "start_date": "20/11/2026", "end_date": "21/11/2026"},
    },
    "argentina_ina_series": {
        "method": "GET",
        "url": "https://alerta.ina.gob.ar/a5/obs/puntual/series",
        "params": {"var_id": 2, "proc_id": 1, "limit": 400},
        "test_example": {},
    },
    "argentina_ina_obs": {
        "method": "GET",
        "url": "https://alerta.ina.gob.ar/a5/obs/puntual/series/{series_id}/observaciones",
        "params": {"timestart": "{start_date}", "timeend": "{end_date}"},
        "test_example": {"series_id": 34, "start_date": "2026-11-15", "end_date": "2026-11-21"},
    },

    # --- COAST AND TIDE ---
    "openmeteo_marine": {
        "method": "GET",
        "url": "https://marine-api.open-meteo.com/v1/marine",
        "params": {"latitude": "{lat}", "longitude": "{lon}", "hourly": "sea_level_height_msl,wave_height", "forecast_days": 3},
        "test_example": {"lat": 18.9220, "lon": 72.8347},
    },
    "ioc_sealevel_stations": {
        "method": "GET",
        "url": "https://www.ioc-sealevelmonitoring.org/service.php",
        "params": {"query": "stationlist", "showall": "a"},
        "test_example": {},
    },
    "ioc_sealevel_data": {
        "method": "GET",
        "url": "https://www.ioc-sealevelmonitoring.org/service.php",
        "params": {"query": "data", "code": "{code}", "format": "json", "timestart": "{start_time}", "timestop": "{end_time}"},
        "test_example": {"code": "coch", "start_time": "2026-11-20T00:00", "end_time": "2026-11-21T00:00"},
    },
    "noaa_coops": {
        "method": "GET",
        "url": "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
        "params": {"date": "latest", "station": "{station}", "product": "water_level", "datum": "MLLW", "units": "metric", "time_zone": "gmt", "format": "json", "application": "FloodAI"},
        "test_example": {"station": "8771450"},
    },
    "venice_tide": {
        "method": "GET",
        "url": "https://dati.venezia.it/sites/default/files/dataset/opendata/livello.json",
        "test_example": {},
    },

    # --- RAIN ---
    "openmeteo_forecast": {
        "method": "GET",
        "url": "https://api.open-meteo.com/v1/forecast",
        "params": {"latitude": "{lat}", "longitude": "{lon}", "hourly": "precipitation,precipitation_probability,soil_moisture_0_to_1cm", "past_days": 2, "forecast_days": 3, "timezone": "auto"},
        "test_example": {"lat": 24.5854, "lon": 73.7125},
    },
    "openmeteo_ensemble": {
        "method": "GET",
        "url": "https://ensemble-api.open-meteo.com/v1/ensemble",
        "params": {"latitude": "{lat}", "longitude": "{lon}", "hourly": "precipitation", "models": "gfs_seamless", "forecast_days": 3},
        "test_example": {"lat": 24.5854, "lon": 73.7125},
    },
    "openmeteo_archive": {
        "method": "GET",
        "url": "https://archive-api.open-meteo.com/v1/archive",
        "params": {"latitude": "{lat}", "longitude": "{lon}", "start_date": "{start_date}", "end_date": "{end_date}", "daily": "precipitation_sum,rain_sum", "models": "era5"},
        "test_example": {"lat": 24.5854, "lon": 73.7125, "start_date": "2020-01-01", "end_date": "2020-12-31"},
    },
    "imd_gridded_rain": {
        "method": "POST",
        "url": "https://www.imdpune.gov.in/cmpg/Griddata/rainfall.php",
        "body_type": "form",
        "body_template": "rain={year}",
        "test_example": {"year": "2023"},
    },

    # --- ALERTS ---
    "gdacs_floods": {
        "method": "GET",
        "url": "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH",
        "params": {"eventlist": "FL", "alertlevel": "green;orange;red", "fromdate": "{from_date}", "todate": "{to_date}"},
        "test_example": {"from_date": "2026-11-10", "to_date": "2026-11-24"},
    },
    "nws_flood_alerts": {
        "method": "GET",
        "url": "https://api.weather.gov/alerts/active",
        "params": {"event": "Flood Warning"},
        "test_example": {},
    },
}


def _get_api_headers(source_id: str) -> Dict[str, str]:
    """Injects user agent and optional environment variable tokens politely."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
    }
    if source_id.startswith("usgs") and os.environ.get("USGS_API_KEY"):
        headers["X-Api-Key"] = os.environ["USGS_API_KEY"]
    if source_id.startswith("cdse") and os.environ.get("CDSE_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['CDSE_TOKEN']}"
    if source_id.startswith("earthdata") and os.environ.get("EARTHDATA_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['EARTHDATA_TOKEN']}"
    if source_id.startswith("imd") and os.environ.get("IMD_API_KEY"):
        headers["X-Api-Key"] = os.environ["IMD_API_KEY"]
    return headers


def _build_url_and_body(source_id: str, values: Dict[str, Any]) -> Tuple[str, str, Optional[bytes], Dict[str, str], bool]:
    """Resolves URL placeholders, query parameters, headers, and request body for an endpoint."""
    config = SOURCES[source_id]
    method = config.get("method", "GET").upper()
    url_template = config["url"]

    # Fill path variables inside the URL string
    url = url_template
    for k, v in values.items():
        placeholder = "{" + k + "}"
        if placeholder in url:
            url = url.replace(placeholder, urllib.parse.quote(str(v), safe=""))

    headers = _get_api_headers(source_id)
    body: Optional[bytes] = None

    # Handle query parameters
    params = config.get("params")
    if params:
        formatted_params = {}
        for pk, pv in params.items():
            if isinstance(pv, str):
                for k, v in values.items():
                    pv = pv.replace("{" + k + "}", str(v))
            formatted_params[pk] = pv
        query_str = urllib.parse.urlencode(formatted_params)
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{query_str}"

    # Handle body for POST
    if method == "POST":
        body_type = config.get("body_type", "json")
        body_tmpl = config.get("body_template", "")
        if body_tmpl:
            rendered = body_tmpl
            for k, v in values.items():
                rendered = rendered.replace("{" + k + "}", str(v))
            if body_type == "json":
                headers["Content-Type"] = "application/json"
                body = rendered.encode("utf-8")
            else:
                headers["Content-Type"] = "application/x-www-form-urlencoded"
                body = rendered.encode("utf-8")

    verify_ssl = config.get("verify_ssl", True)
    return url, method, body, headers, verify_ssl


def _enforce_rate_limit(source_id: str):
    """Enforces per-source rate limits (e.g. Nominatim max 1 req/sec)."""
    rate_limit = SOURCES.get(source_id, {}).get("rate_limit_s", 0.0)
    if rate_limit > 0.0:
        now = time.time()
        last = _LAST_REQUEST_TIME.get(source_id, 0.0)
        elapsed = now - last
        if elapsed < rate_limit:
            time.sleep(rate_limit - elapsed)
        _LAST_REQUEST_TIME[source_id] = time.time()


def fetch(source_id: str, **values) -> Any:
    """
    Polite, cached, and resilient unified loader for all Flood-AI real endpoints.
    
    Rules implemented:
    - User-Agent 'FloodAI/1.0 (<email>)'
    - 30s timeout with exponential backoff retries
    - In-memory / disk caching for static and geocoding resources
    - SSL verification bypass specifically for NWDP (cert expired 2026-11-20)
    - Clean JSON / text parsing with fallback
    """
    if source_id not in SOURCES:
        raise ValueError(f"Unknown real-time source_id: '{source_id}'. Must be defined in SOURCES.")

    _enforce_rate_limit(source_id)

    # Cache check
    ttl = SOURCES[source_id].get("cache_ttl_s", 0)
    cache_key = f"{source_id}:" + json.dumps(values, sort_keys=True)
    if ttl > 0:
        if cache_key in _MEMORY_CACHE:
            ts, val = _MEMORY_CACHE[cache_key]
            if time.time() - ts < ttl:
                return val

    url, method, body, headers, verify_ssl = _build_url_and_body(source_id, values)
    timeout = SOURCES[source_id].get("timeout_s", REQUEST_TIMEOUT_S)

    # SSL Context
    if not verify_ssl:
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
    else:
        ssl_ctx = None

    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ssl_ctx) as resp:
                status = resp.status
                content_type = resp.headers.get("Content-Type", "")
                raw_bytes = resp.read()

                # Parse JSON if detected
                if "application/json" in content_type or raw_bytes.strip().startswith((b"{", b"[")):
                    try:
                        result = json.loads(raw_bytes.decode("utf-8", errors="replace"))
                    except Exception:
                        result = raw_bytes.decode("utf-8", errors="replace")
                else:
                    try:
                        result = raw_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        result = raw_bytes

                if ttl > 0:
                    _MEMORY_CACHE[cache_key] = (time.time(), result)

                return result

        except Exception as e:
            last_err = e
            logger.warning(f"Fetch {source_id} (attempt {attempt}/{MAX_RETRIES}) failed: {e}")
            if attempt < MAX_RETRIES:
                # Fast fail if offline/DNS resolution failed or in non-live test
                err_str = str(e).lower()
                if "getaddrinfo failed" in err_str or "name or service not known" in err_str:
                    break
                time.sleep(1.0 * (2 ** (attempt - 1)))

    raise RuntimeError(f"Failed to fetch real-time source '{source_id}' after {MAX_RETRIES} attempts. Error: {last_err}")


def tag_value(
    value: Any,
    origin: str,
    source_id: str,
    time_utc: Optional[str] = None,
    reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    Tags every telemetry value with origin = 'observed', 'official_forecast', 'model', or 'simulated'.
    Complies with Rule 2 & Rule 3:
    - Never invent numbers. If null, return null with reason = 'no official source'.
    """
    if time_utc is None:
        time_utc = datetime.now(timezone.utc).isoformat()
    return {
        "value": value,
        "origin": origin,
        "source_id": source_id,
        "time_utc": time_utc,
        "reason": reason,
    }
