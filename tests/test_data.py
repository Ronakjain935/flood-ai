"""
Unit tests for Flood-AI data ingestion and processing routines.
"""

import numpy as np
import pandas as pd
import pytest
from src.data.ingest_sentinel import Sentinel1SARProcessor, Sentinel2OpticalProcessor
from src.data.ingest_sensors import USGSSensorIngestor


def test_linear_to_db_conversion():
    """Verify linear backscatter to dB conversion."""
    processor = Sentinel1SARProcessor()
    linear = np.array([1.0, 10.0, 100.0], dtype=np.float32)
    db = processor.linear_to_db(linear)

    expected = np.array([0.0, 10.0, 20.0], dtype=np.float32)
    np.testing.assert_allclose(db, expected, atol=1e-5)


def test_lee_filter_shape_and_range():
    """Ensure Lee speckle filter preserves array dimensions and does not introduce NaNs."""
    processor = Sentinel1SARProcessor(filter_size=5)
    dummy_sar = np.random.normal(loc=-15.0, scale=3.0, size=(64, 64)).astype(np.float32)
    filtered = processor.apply_lee_filter(dummy_sar)

    assert filtered.shape == (64, 64)
    assert not np.isnan(filtered).any()
    assert not np.isinf(filtered).any()


def test_ndwi_mndwi_calculation():
    """Verify Normalized Difference Water Index calculations."""
    green = np.array([[0.2, 0.4], [0.1, 0.3]], dtype=np.float32)
    nir = np.array([[0.1, 0.1], [0.5, 0.3]], dtype=np.float32)

    ndwi = Sentinel2OpticalProcessor.compute_ndwi(green, nir)
    assert pytest.approx(ndwi[0, 0], rel=1e-4) == (0.2 - 0.1) / (0.2 + 0.1)
    assert pytest.approx(ndwi[1, 0], rel=1e-4) == (0.1 - 0.5) / (0.1 + 0.5)


def test_sensor_timeseries_preprocessing():
    """Verify time series regularization, linear interpolation, and rate-of-rise calculation."""
    timestamps = pd.date_range("2024-05-01 00:00:00", periods=5, freq="15min")
    raw_df = pd.DataFrame({
        "timestamp": timestamps,
        "site_id": ["01638500"] * 5,
        "gauge_height_m": [2.0, 2.1, np.nan, 2.3, 2.5],
    })

    processed = USGSSensorIngestor.preprocess_gauge_series(raw_df, freq="15min")

    # Missing value at step 2 should be linearly interpolated to 2.2
    assert pytest.approx(processed.loc[2, "gauge_height_m"], rel=1e-3) == 2.2
    # Rate of rise: delta 0.1m over 15min (0.25h) -> 0.4 m/hr
    assert pytest.approx(processed.loc[1, "rate_of_rise_m_per_hr"], rel=1e-3) == 0.4
