"""
Flood-AI: Sentinel-1 SAR and Sentinel-2 Optical Ingestion & Calibration Pipeline.
Provides automated ingestion, radiometric calibration, Lee speckle filtering,
and water index calculation (NDWI, MNDWI) for high-resolution flood mapping.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import logging

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger("Flood-AI.Sentinel")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

from scipy.ndimage import uniform_filter

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
    Processor for Sentinel-1 Synthetic Aperture Radar (SAR) Ground Range Detected (GRD).
    Handles radiometric calibration to sigma0 (dB) and edge-preserving speckle reduction.
    """

    def __init__(self, filter_size: int = 7):
        self.filter_size = filter_size

    @staticmethod
    def linear_to_db(linear_array: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Convert linear backscatter intensity to decibels (dB)."""
        clipped = np.clip(linear_array, eps, None)
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

    def process_sar_scene(
        self,
        vv_path: Union[str, Path],
        vh_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Reads VV and VH polarizations, applies radiometric conversion to dB,
        computes speckle-filtered channels, and generates VV/VH ratio.
        """
        logger.info(f"Processing Sentinel-1 SAR scene: VV={vv_path}, VH={vh_path}")
        if not HAS_RASTERIO:
            raise ImportError("rasterio is required to process GeoTIFF files. Install via pip install rasterio.")

        with rasterio.open(vv_path) as src_vv, rasterio.open(vh_path) as src_vh:
            vv_raw = src_vv.read(1).astype(np.float32)
            vh_raw = src_vh.read(1).astype(np.float32)
            profile = src_vv.profile.copy()

        # Calibration: linear to dB
        vv_db = self.linear_to_db(vv_raw)
        vh_db = self.linear_to_db(vh_raw)

        # Speckle filtering
        vv_filtered = self.apply_lee_filter(vv_db)
        vh_filtered = self.apply_lee_filter(vh_db)

        # Dual-polarization difference / ratio in dB
        ratio_db = vv_filtered - vh_filtered

        processed_stack = {
            "vv_db": vv_filtered,
            "vh_db": vh_filtered,
            "ratio_db": ratio_db,
        }

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            profile.update(count=3, dtype=rasterio.float32)
            with rasterio.open(out_p, "w", **profile) as dst:
                dst.write(vv_filtered, 1)
                dst.write(vh_filtered, 2)
                dst.write(ratio_db, 3)
                dst.set_band_description(1, "VV_dB_Lee")
                dst.set_band_description(2, "VH_dB_Lee")
                dst.set_band_description(3, "VV_VH_Ratio_dB")
            logger.info(f"Saved processed SAR GeoTIFF to: {out_p}")

        return processed_stack


class Sentinel2OpticalProcessor:
    """
    Processor for Sentinel-2 MSI Optical data.
    Computes spectral indices: NDWI (McFeeters) and MNDWI (Xu) for surface water extraction.
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

    def process_optical_bands(
        self,
        green_path: Union[str, Path],
        nir_path: Union[str, Path],
        swir_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, np.ndarray]:
        """Loads spectral bands and computes NDWI and MNDWI rasters."""
        logger.info(f"Computing optical water indices: Green={green_path}, NIR={nir_path}")
        if not HAS_RASTERIO:
            raise ImportError("rasterio is required to process GeoTIFF files. Install via pip install rasterio.")

        with rasterio.open(green_path) as src_g, rasterio.open(nir_path) as src_nir:
            green = src_g.read(1).astype(np.float32)
            nir = src_nir.read(1).astype(np.float32)
            profile = src_g.profile.copy()

        ndwi = self.compute_ndwi(green, nir)
        results = {"ndwi": ndwi}

        if swir_path:
            with rasterio.open(swir_path) as src_swir:
                swir = src_swir.read(1).astype(np.float32)
            mndwi = self.compute_mndwi(green, swir)
            results["mndwi"] = mndwi

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            band_count = 2 if "mndwi" in results else 1
            profile.update(count=band_count, dtype=rasterio.float32)
            with rasterio.open(out_p, "w", **profile) as dst:
                dst.write(ndwi, 1)
                dst.set_band_description(1, "NDWI")
                if "mndwi" in results:
                    dst.write(results["mndwi"], 2)
                    dst.set_band_description(2, "MNDWI")
            logger.info(f"Saved optical indices GeoTIFF to: {out_p}")

        return results


def reproject_raster(
    input_raster_path: Union[str, Path],
    output_raster_path: Union[str, Path],
    target_crs: str = "EPSG:4326",
    resampling_method: Optional[Any] = None,
) -> Path:
    """Reprojects an input raster to a uniform target Coordinate Reference System (CRS)."""
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required for reproject_raster. Install via pip install rasterio.")

    if resampling_method is None:
        resampling_method = Resampling.bilinear

    with rasterio.open(input_raster_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height,
        })

        out_p = Path(output_raster_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_p, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=resampling_method,
                )
    logger.info(f"Reprojected {input_raster_path} to {target_crs} at {output_raster_path}")
    return out_p


if __name__ == "__main__":
    logger.info("Flood-AI Sentinel-1 / Sentinel-2 Data Ingestion Module Ready.")
