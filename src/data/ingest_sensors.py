"""
Flood-AI: USGS & IoT River Sensor Ingestion Pipeline.
Queries real-time and historical river stage and discharge measurements,
handles missing data interpolation, and computes rate-of-rise telemetry.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import requests
import logging

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger("Flood-AI.Sensors")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


@dataclass
class GaugeThresholds:
    """Standard National Weather Service (NWS) flood stage thresholds (in meters or feet)."""
    action_stage: float
    minor_flood: float
    moderate_flood: float
    major_flood: float


class USGSSensorIngestor:
    """
    Client for the USGS National Water Information System (NWIS) Instantaneous Values (IV) REST API.
    Retrieves gauge height (code 00065) and discharge streamflow (code 00060).
    """

    BASE_URL = "https://waterservices.usgs.gov/nwis/iv/"

    def __init__(self, timeout_sec: int = 30):
        self.timeout = timeout_sec

    def fetch_gauge_data(
        self,
        site_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        parameter_codes: Tuple[str, ...] = ("00065", "00060"),
    ) -> pd.DataFrame:
        """
        Retrieves time-series data for a specific USGS river gauge site.

        Args:
            site_id: 8-digit USGS gauge station ID (e.g., '01646500' for Potomac River).
            start_date: ISO 8601 string (e.g., '2024-05-01') or None (defaults to past 7 days).
            end_date: ISO 8601 string or None (defaults to now).
            parameter_codes: '00065' = Gage height (ft), '00060' = Streamflow (cfs).
        """
        if not start_date:
            start_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

        params = {
            "format": "json",
            "sites": site_id,
            "startDT": start_date,
            "parameterCd": ",".join(parameter_codes),
            "siteStatus": "all",
        }
        if end_date:
            params["endDT"] = end_date

        logger.info(f"Querying USGS NWIS for site {site_id} from {start_date}...")
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch USGS data for site {site_id}: {e}")
            return pd.DataFrame()

        records = []
        try:
            time_series = payload["value"]["timeSeries"]
            for ts in time_series:
                var_desc = ts["variable"]["variableDescription"]
                var_code = ts["variable"]["variableCode"][0]["value"]
                is_height = "00065" in var_code

                col_name = "gauge_height_ft" if is_height else "discharge_cfs"
                for pt in ts["values"][0]["value"]:
                    val = pt["value"]
                    timestamp = pt["dateTime"]
                    try:
                        f_val = float(val) if val not in ("-999999", "Ice") else np.nan
                    except ValueError:
                        f_val = np.nan

                    records.append({
                        "site_id": site_id,
                        "timestamp": pd.to_datetime(timestamp),
                        "variable": col_name,
                        "value": f_val,
                    })
        except (KeyError, IndexError) as err:
            logger.warning(f"Unexpected response structure for site {site_id}: {err}")
            return pd.DataFrame()

        if not records:
            logger.warning(f"No valid records retrieved for site {site_id}.")
            return pd.DataFrame()

        df_raw = pd.DataFrame(records)
        df_pivoted = df_raw.pivot_table(
            index=["site_id", "timestamp"],
            columns="variable",
            values="value",
            aggfunc="first",
        ).reset_index()

        # Sort chronologically
        df_pivoted = df_pivoted.sort_values(by="timestamp").reset_index(drop=True)

        # Convert feet to meters for international standard
        if "gauge_height_ft" in df_pivoted.columns:
            df_pivoted["gauge_height_m"] = df_pivoted["gauge_height_ft"] * 0.3048

        return df_pivoted

    @staticmethod
    def preprocess_gauge_series(
        df: pd.DataFrame,
        freq: str = "15min",
        max_gap_hours: int = 6,
    ) -> pd.DataFrame:
        """
        Regularizes time series to a uniform sampling frequency, performs linear interpolation
        for small gaps, and computes rate of rise (m/hr).
        """
        if df.empty or "timestamp" not in df.columns:
            return df

        df = df.copy()
        df = df.set_index("timestamp").asfreq(freq)

        # Forward/linear fill small gaps up to max_gap_hours
        limit_steps = int(pd.Timedelta(hours=max_gap_hours) / pd.Timedelta(freq))
        df["gauge_height_m"] = df["gauge_height_m"].interpolate(method="time", limit=limit_steps)

        # Rate of rise: delta height (meters) per hour
        # freq in minutes
        delta_hours = pd.Timedelta(freq).total_seconds() / 3600.0
        df["rate_of_rise_m_per_hr"] = df["gauge_height_m"].diff() / delta_hours

        return df.reset_index()


def save_processed_gauge_data(df: pd.DataFrame, output_dir: Union[str, Path], site_id: str) -> Path:
    """Exports structured gauge time-series to parquet or CSV."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"gauge_{site_id}.parquet"
    df.to_parquet(out_file, index=False)
    logger.info(f"Saved cleaned sensor series to: {out_file}")
    return out_file


if __name__ == "__main__":
    client = USGSSensorIngestor()
    # Sample site: Potomac River at Point of Rocks, MD
    sample_df = client.fetch_gauge_data(site_id="01638500")
    if not sample_df.empty:
        processed = client.preprocess_gauge_series(sample_df)
        print(processed.head())
