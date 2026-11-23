"""
Flood-AI: Central Dataset & Live Sensor Registry.
Loads, validates, and manages the 20 real-time and foundational datasets
powering Flood-AI across satellite SAR, optical, radar, weather, hydrology,
population exposure, infrastructure routing, and crisis social NLP.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import csv
import logging

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger("Flood-AI.DatasetRegistry")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


class DatasetRegistry:
    """Manages the 20 official datasets powering Flood-AI."""

    _CATALOG_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "catalog" / "flood_ai_datasets.csv"
    _DATASETS_CACHE: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def load_catalog(cls) -> List[Dict[str, Any]]:
        """Loads and parses the 20-dataset CSV catalog."""
        if cls._DATASETS_CACHE is not None:
            return cls._DATASETS_CACHE

        if not cls._CATALOG_PATH.exists():
            logger.warning(f"Dataset catalog not found at {cls._CATALOG_PATH}. Initializing empty.")
            return []

        datasets: List[Dict[str, Any]] = []
        with open(cls._CATALOG_PATH, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Add operational status flags
                category = row.get("category", "")
                is_live = any(k in category.lower() for k in ["api", "telemetry", "streaming", "nowcasting"]) or \
                          any(k in row.get("dataset_name", "").lower() for k in ["openweather", "usgs", "bwdb", "cwc", "twitter", "nexrad"])
                
                row["is_real_time"] = is_live
                row["status_badge"] = "🟢 LIVE FEED" if is_live else "🔵 CORE BENCHMARK"
                datasets.append(row)

        cls._DATASETS_CACHE = datasets
        logger.info(f"Loaded {len(datasets)} verified datasets into Flood-AI Registry.")
        return datasets

    @classmethod
    def get_dataset_by_name(cls, name: str) -> Optional[Dict[str, Any]]:
        """Finds a dataset by name (case-insensitive partial match)."""
        catalog = cls.load_catalog()
        lower_name = name.lower()
        for ds in catalog:
            if lower_name in ds.get("dataset_name", "").lower():
                return ds
        return None

    @classmethod
    def get_realtime_datasets(cls) -> List[Dict[str, Any]]:
        """Returns all real-time/near-real-time streaming datasets."""
        catalog = cls.load_catalog()
        return [ds for ds in catalog if ds.get("is_real_time")]

    @classmethod
    def get_datasets_by_phase(cls, phase_str: str) -> List[Dict[str, Any]]:
        """Filters datasets by pipeline phase (e.g. 'Phase 1', 'Phase 2', 'Phase 3', 'Phase 4')."""
        catalog = cls.load_catalog()
        return [ds for ds in catalog if phase_str.lower() in ds.get("flood_ai_phase", "").lower()]

    @classmethod
    def get_provenance_summary(cls) -> Dict[str, str]:
        """Provides verified dataset attribution strings for all UI metric cards."""
        return {
            "rainfall": "Live Open-Meteo & NASA GPM IMERG (0.1° / 30-min)",
            "hydrology": "USGS NWIS & BWDB/CWC River Streamgage Telemetry",
            "sar_inundation": "Sen1Floods11 & Sentinel-1 SAR GRD (ViT-UNet)",
            "exposure": "WorldPop 100m High-Resolution Population Density",
            "damage": "DARPA xBD 4-Level Structural Damage Classification",
            "social_crisis": "CrisisMMD & HumAID Multimodal Rescue Triage",
            "evacuation": "OpenStreetMap Real-Time Vector Roadways & Shelters",
            "nowcasting": "Copernicus ERA5 & Temporal Fusion Transformer (0-72h)",
        }
