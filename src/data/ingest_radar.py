"""
Flood-AI: Radar and Atmospheric Precipitation Ingestion Pipeline.
Handles GPM IMERG (Global Precipitation Measurement) half-hourly rasters
and ERA5 atmospheric reanalysis precipitation for flood forcing.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import logging

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger("Flood-AI.Radar")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    HAS_RASTERIO = True
except ImportError:
    rasterio = None
    HAS_RASTERIO = False


@dataclass
class RadarBBox:
    """Bounding box for precipitation sub-setting (WGS84)."""
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


class GPMPrecipitationIngestor:
    """
    Processor for NASA GPM IMERG (Integrated Multi-satellitE Retrievals for GPM).
    Ingests precipitation rate (mm/hr) and converts to accumulated surface runoff forcing.
    """

    def __init__(self, target_resolution_deg: float = 0.1):
        self.res = target_resolution_deg

    @staticmethod
    def rate_to_accumulation(precip_rate_mm_per_hr: np.ndarray, duration_hours: float = 0.5) -> np.ndarray:
        """
        Converts instantaneous precipitation rate (mm/hr) to cumulative depth (mm)
        over the measurement time step (default: 30 minutes = 0.5 hours).
        """
        return np.maximum(0.0, precip_rate_mm_per_hr * duration_hours).astype(np.float32)

    def generate_synthetic_storm(
        self,
        bbox: RadarBBox,
        shape: Tuple[int, int] = (64, 64),
        peak_intensity_mm_hr: float = 45.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generates a physically realistic 2D Gaussian rainfall cell for simulation and testing.
        """
        ny, nx = shape
        y = np.linspace(bbox.min_lat, bbox.max_lat, ny)
        x = np.linspace(bbox.min_lon, bbox.max_lon, nx)
        xv, yv = np.meshgrid(x, y)

        center_x = (bbox.min_lon + bbox.max_lon) / 2.0
        center_y = (bbox.min_lat + bbox.max_lat) / 2.0
        sigma_x = (bbox.max_lon - bbox.min_lon) / 5.0
        sigma_y = (bbox.max_lat - bbox.min_lat) / 5.0

        gaussian = np.exp(-(((xv - center_x) ** 2) / (2 * sigma_x ** 2) + ((yv - center_y) ** 2) / (2 * sigma_y ** 2)))
        noise = np.random.normal(0, 0.05, shape)
        rate = np.maximum(0.0, gaussian * peak_intensity_mm_hr + noise * 2.0).astype(np.float32)

        meta = {
            "bbox": (bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat),
            "crs": "EPSG:4326",
            "peak_intensity": float(np.max(rate)),
            "mean_intensity": float(np.mean(rate)),
        }
        return rate, meta

    def save_raster(
        self,
        precip_data: np.ndarray,
        bbox: RadarBBox,
        output_path: Union[str, Path],
    ) -> Path:
        """Saves precipitation grid as GeoTIFF."""
        if not HAS_RASTERIO:
            raise ImportError("rasterio is required to export precipitation GeoTIFFs.")

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        ny, nx = precip_data.shape
        transform = from_bounds(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat, nx, ny)

        with rasterio.open(
            out_p,
            "w",
            driver="GTiff",
            height=ny,
            width=nx,
            count=1,
            dtype=rasterio.float32,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(precip_data.astype(np.float32), 1)
            dst.set_band_description(1, "Precipitation_Rate_mm_hr")

        logger.info(f"Saved precipitation grid to: {out_p}")
        return out_p


class ERA5ReanalysisIngestor:
    """
    Ingestor for ECMWF ERA5 hourly atmospheric variables:
    Total precipitation (tp), runoff (ro), and volumetric soil water (swvl1).
    """

    @staticmethod
    def parse_era5_netcdf(nc_path: Union[str, Path], variable: str = "tp") -> Dict[str, Any]:
        """
        Parses NetCDF4 reanalysis file into structured numpy arrays.
        Handles meter to millimeter depth conversion for total precipitation.
        """
        logger.info(f"Loading ERA5 reanalysis from {nc_path} for variable: {variable}")
        try:
            import xarray as xr
            ds = xr.open_dataset(nc_path)
            data = ds[variable].values
            # ERA5 precipitation is accumulated in meters; convert to millimeters
            if variable == "tp":
                data = data * 1000.0
            return {
                "values": data,
                "latitudes": ds["latitude"].values,
                "longitudes": ds["longitude"].values,
                "time": ds["time"].values,
                "unit": "mm" if variable == "tp" else "dimensionless",
            }
        except ImportError:
            logger.warning("xarray not installed; using placeholder reader.")
            return {"status": "xarray required"}


if __name__ == "__main__":
    ingestor = GPMPrecipitationIngestor()
    dhaka_box = RadarBBox(min_lon=90.30, min_lat=23.65, max_lon=90.55, max_lat=23.95)
    rain_grid, meta = ingestor.generate_synthetic_storm(dhaka_box)
    acc = ingestor.rate_to_accumulation(rain_grid)
    print(f"Synthesized Storm: Max Rate = {meta['peak_intensity']:.2f} mm/hr, Max 30-min Acc = {np.max(acc):.2f} mm")
