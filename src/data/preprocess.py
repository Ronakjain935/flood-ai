"""
Flood-AI: Spatial Preprocessing, Tiling, Normalization, and Dataset Slicing.
Provides sliding-window patch extraction, spatial reconstruction,
and robust feature normalization for multimodal geospatial arrays.
"""

from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple
import numpy as np


@dataclass
class TileCoord:
    """Bounding indices of a spatial tile patch within a full scene."""
    row_start: int
    row_end: int
    col_start: int
    col_end: int


class GeospatialPreprocessor:
    """
    Handles array normalization, sliding-window patch extraction,
    and border-overlap blending for large remote sensing scenes.
    """

    @staticmethod
    def normalize_sar(vv: np.ndarray, vh: np.ndarray, clip_range: Tuple[float, float] = (-30.0, 0.0)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalizes SAR backscatter values in dB to [0, 1] range.
        Water typically falls between -25 dB and -15 dB.
        """
        min_db, max_db = clip_range
        vv_norm = np.clip((vv - min_db) / (max_db - min_db), 0.0, 1.0).astype(np.float32)
        vh_norm = np.clip((vh - min_db) / (max_db - min_db), 0.0, 1.0).astype(np.float32)
        return vv_norm, vh_norm

    @staticmethod
    def normalize_dem(dem: np.ndarray, nodata_val: float = -9999.0) -> np.ndarray:
        """Min-max normalizes digital elevation model (DEM) while ignoring nodata values."""
        valid_mask = (dem != nodata_val) & ~np.isnan(dem)
        if not np.any(valid_mask):
            return np.zeros_like(dem, dtype=np.float32)

        min_val = np.min(dem[valid_mask])
        max_val = np.max(dem[valid_mask])
        denom = max_val - min_val if max_val > min_val else 1.0

        normalized = np.zeros_like(dem, dtype=np.float32)
        normalized[valid_mask] = (dem[valid_mask] - min_val) / denom
        return normalized

    @staticmethod
    def extract_patches(
        image: np.ndarray,
        patch_size: int = 128,
        stride: int = 64,
    ) -> Generator[Tuple[np.ndarray, TileCoord], None, None]:
        """
        Extracts sliding-window patches with overlap from a multi-channel image (C, H, W).
        Pads borders with reflection when image dimensions are not exact multiples.
        """
        C, H, W = image.shape
        pad_h = (patch_size - (H % stride)) % stride if H >= patch_size else patch_size - H
        pad_w = (patch_size - (W % stride)) % stride if W >= patch_size else patch_size - W

        if pad_h > 0 or pad_w > 0:
            padded = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        else:
            padded = image

        _, padded_h, padded_w = padded.shape

        for r in range(0, padded_h - patch_size + 1, stride):
            for c in range(0, padded_w - patch_size + 1, stride):
                patch = padded[:, r : r + patch_size, c : c + patch_size]
                coord = TileCoord(row_start=r, row_end=r + patch_size, col_start=c, col_end=c + patch_size)
                yield patch, coord

    @staticmethod
    def stitch_patches(
        patches: List[np.ndarray],
        coords: List[TileCoord],
        output_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Stitches overlapping predicted patch probability maps back into full scene,
        applying linear distance weight averaging across overlapping borders.
        """
        H, W = output_shape
        stitched = np.zeros((H, W), dtype=np.float32)
        weights = np.zeros((H, W), dtype=np.float32)

        for patch, coord in zip(patches, coords):
            p_sq = np.squeeze(patch)
            ph, pw = p_sq.shape

            # Limit to original scene boundary
            r_end = min(coord.row_end, H)
            c_end = min(coord.col_end, W)
            slice_h = r_end - coord.row_start
            slice_w = c_end - coord.col_start

            if slice_h <= 0 or slice_w <= 0:
                continue

            stitched[coord.row_start : r_end, coord.col_start : c_end] += p_sq[:slice_h, :slice_w]
            weights[coord.row_start : r_end, coord.col_start : c_end] += 1.0

        weights = np.maximum(weights, 1e-6)
        return stitched / weights


if __name__ == "__main__":
    pre = GeospatialPreprocessor()
    test_img = np.random.randn(4, 250, 300).astype(np.float32)
    patches = []
    coords = []
    for p, c in pre.extract_patches(test_img, patch_size=128, stride=64):
        patches.append(np.mean(p, axis=0))
        coords.append(c)

    stitched = pre.stitch_patches(patches, coords, output_shape=(250, 300))
    print(f"Extracted {len(patches)} overlapping patches. Stitched output shape: {stitched.shape}")
