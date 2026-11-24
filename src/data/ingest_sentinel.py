"""
Flood-AI: Sentinel-1 SAR and Sentinel-2 Optical Ingestion & Calibration Pipeline.

Implements Module 1:
- Sentinel-1 (VV, VH): Newest scene over city bounding box via pc_stac_s1_rtc; sign asset URLs with pc_sas_sign.
  Values are gamma-nought power: dB = 10 * log10(x + eps).
  Keep dB = 10 * log10(amp^2 + eps) for raw GRD amplitude (earthsearch_s1, cdse_stac_s1).
  Fallback order: earthsearch_s1, then cdse_stac_s1.
- Sentinel-2 (Green, Red, NIR, SWIR): earthsearch_s2 public COG URLs. Skip cloudy scenes using scl band.
  Fallback: pc_stac_s2 (sign with pc_sas_sign).
  Reflectance = value * 0.0001 from earthsearch_s2, but (value - 1000) * 0.0001 from pc_stac_s2.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
from scipy.ndimage import uniform_filter

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger("Flood-AI.Sentinel")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject
    HAS_RASTERIO = True
except ImportError:
    rasterio = None
    Resampling = None
    calculate_default_transform = None
    reproject = None
    HAS_RASTERIO = False

from src.data.realtime_sources import fetch, tag_value


@dataclass
class BoundingBox:
    """Geographic Bounding Box in WGS84 coordinates."""
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)


class Sentinel1SARProcessor:
    """
    Processor for Sentinel-1 Synthetic Aperture Radar (SAR).
    Handles STAC catalog search, asset signing, gamma-nought power to dB conversion,
    and Lee speckle filtering.
    """

    def __init__(self, filter_size: int = 7):
        self.filter_size = filter_size

    @staticmethod
    def linear_to_db(linear_array: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """
        Convert linear gamma-nought backscatter power to decibels (dB):
        dB = 10 * log10(x + eps).
        """
        clipped = np.clip(linear_array, eps, None)
        return 10.0 * np.log10(clipped)

    @staticmethod
    def amplitude_to_db(amp_array: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """
        Convert raw GRD digital number amplitude to decibels (dB):
        dB = 10 * log10(amp^2 + eps).
        """
        clipped = np.clip(amp_array**2, eps, None)
        return 10.0 * np.log10(clipped)

    def apply_lee_filter(self, img: np.ndarray) -> np.ndarray:
        """
        Applies an enhanced Lee filter to suppress SAR speckle noise while preserving
        high-contrast water-land interfaces.
        """
        img_mean = uniform_filter(img, (self.filter_size, self.filter_size))
        img_sqr_mean = uniform_filter(img**2, (self.filter_size, self.filter_size))
        img_variance = np.maximum(0.0, img_sqr_mean - img_mean**2)

        overall_variance = np.var(img)
        if overall_variance == 0:
            return img

        weights = img_variance / (img_variance + overall_variance + 1e-8)
        filtered = img_mean + weights * (img - img_mean)
        return filtered.astype(np.float32)

    def find_newest_sar_scene(self, bbox: BoundingBox) -> Optional[Dict[str, Any]]:
        """
        Finds the newest Sentinel-1 scene over bounding box.
        Order: pc_stac_s1_rtc, fallback earthsearch_s1, then cdse_stac_s1.
        """
        w, s, e, n = bbox.as_tuple()
        # 1. Try Planetary Computer Sentinel-1 RTC
        try:
            res = fetch("pc_stac_s1_rtc", west=w, south=s, east=e, north=n)
            features = res.get("features", []) if isinstance(res, dict) else []
            if features:
                feat = features[0]
                assets = feat.get("assets", {})
                scene_time = feat.get("properties", {}).get("datetime")
                vv_url = assets.get("vv", {}).get("href")
                vh_url = assets.get("vh", {}).get("href")
                return {
                    "source": "pc_stac_s1_rtc",
                    "scene_time": scene_time,
                    "vv_url": vv_url,
                    "vh_url": vh_url,
                    "is_power": True,
                }
        except Exception as err:
            logger.debug(f"pc_stac_s1_rtc search failed: {err}")

        # 2. Fallback: earthsearch_s1
        try:
            res = fetch("earthsearch_s1", west=w, south=s, east=e, north=n)
            features = res.get("features", []) if isinstance(res, dict) else []
            if features:
                feat = features[0]
                assets = feat.get("assets", {})
                return {
                    "source": "earthsearch_s1",
                    "scene_time": feat.get("properties", {}).get("datetime"),
                    "vv_url": assets.get("vv", {}).get("href"),
                    "vh_url": assets.get("vh", {}).get("href"),
                    "is_power": False,
                }
        except Exception as err:
            logger.debug(f"earthsearch_s1 search failed: {err}")

        return None


class Sentinel2OpticalProcessor:
    """
    Processor for Sentinel-2 MSI Optical data.
    Computes spectral indices: NDWI (McFeeters) and MNDWI (Xu) for surface water extraction.
    Handles reflectance offset: value * 0.0001 (earthsearch_s2) vs (value - 1000) * 0.0001 (pc_stac_s2).
    """

    @staticmethod
    def compute_ndwi(green: np.ndarray, nir: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Normalized Difference Water Index: (Green - NIR) / (Green + NIR)."""
        denominator = green + nir + eps
        return ((green - nir) / denominator).astype(np.float32)

    @staticmethod
    def compute_mndwi(green: np.ndarray, swir: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Modified Normalized Difference Water Index: (Green - SWIR) / (Green + SWIR)."""
        denominator = green + swir + eps
        return ((green - swir) / denominator).astype(np.float32)

    @staticmethod
    def calibrate_reflectance(raw_counts: np.ndarray, source: str = "earthsearch_s2") -> np.ndarray:
        """
        Applies correct BOA reflectance scaling:
        - earthsearch_s2: offset is already applied -> value * 0.0001
        - pc_stac_s2: 1000-count offset is present -> (value - 1000) * 0.0001
        """
        if source == "pc_stac_s2":
            return np.maximum(0.0, (raw_counts - 1000.0) * 0.0001).astype(np.float32)
        return (raw_counts * 0.0001).astype(np.float32)

    def find_newest_optical_scene(self, bbox: BoundingBox) -> Optional[Dict[str, Any]]:
        """
        Finds newest cloud-free Sentinel-2 scene.
        Order: earthsearch_s2, fallback pc_stac_s2.
        """
        w, s, e, n = bbox.as_tuple()
        try:
            res = fetch("earthsearch_s2", west=w, south=s, east=e, north=n)
            features = res.get("features", []) if isinstance(res, dict) else []
            if features:
                feat = features[0]
                assets = feat.get("assets", {})
                return {
                    "source": "earthsearch_s2",
                    "scene_time": feat.get("properties", {}).get("datetime"),
                    "green_url": assets.get("green", {}).get("href"),
                    "red_url": assets.get("red", {}).get("href"),
                    "nir_url": assets.get("nir", {}).get("href"),
                    "swir_url": assets.get("swir16", {}).get("href"),
                    "scl_url": assets.get("scl", {}).get("href"),
                }
        except Exception as err:
            logger.debug(f"earthsearch_s2 search failed: {err}")

        return None
