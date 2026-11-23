"""
Flood-AI: Multimodal Crisis Social Media Ingestion & Hotspot Clustering.
Processes geo-referenced incident reports, urgency classification,
and spatial density clustering following CrisisMMD specifications.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import logging

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger("Flood-AI.Social")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


@dataclass
class SocialReport:
    """Standardized multimodal citizen or first-responder incident report."""
    report_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    text: str
    urgency_score: float  # 0.0 (info) to 1.0 (life-critical emergency)
    image_url: Optional[str] = None
    water_depth_estimate_m: Optional[float] = None


class CrisisSocialIngestor:
    """
    Ingests and filters crowd-sourced disaster observations.
    Implements rule-based keyword scoring and spatial DBSCAN clustering
    to surface emerging localized flood inundation clusters.
    """

    CRISIS_KEYWORDS = {
        "stranded": 0.9,
        "trapped": 0.95,
        "submerged": 0.85,
        "drowning": 1.0,
        "rescue": 0.9,
        "overflow": 0.7,
        "evacuate": 0.8,
        "shelter": 0.6,
        "water level rising": 0.75,
        "road blocked": 0.65,
        "bridge collapsed": 0.95,
    }

    def __init__(self, urgency_threshold: float = 0.5):
        self.threshold = urgency_threshold

    def calculate_urgency(self, text: str) -> float:
        """Heuristic urgency scoring based on critical flood keywords."""
        lower_t = text.lower()
        score = 0.1
        for kw, weight in self.CRISIS_KEYWORDS.items():
            if kw in lower_t:
                score = max(score, weight)
        return min(1.0, score)

    def parse_reports(self, raw_reports: List[Dict[str, Any]]) -> pd.DataFrame:
        """Parses raw JSON/dictionary records into a validated pandas DataFrame."""
        parsed: List[Dict[str, Any]] = []
        for r in raw_reports:
            text = r.get("text", "")
            urgency = r.get("urgency_score") or self.calculate_urgency(text)
            parsed.append({
                "report_id": r.get("report_id", f"rep_{len(parsed)}"),
                "timestamp": pd.to_datetime(r.get("timestamp", datetime.utcnow())),
                "latitude": float(r["latitude"]),
                "longitude": float(r["longitude"]),
                "text": text,
                "urgency": float(urgency),
                "has_media": bool(r.get("image_url")),
            })
        df = pd.DataFrame(parsed)
        logger.info(f"Parsed {len(df)} social media reports.")
        return df

    def cluster_hotspots(
        self,
        df: pd.DataFrame,
        eps_km: float = 1.0,
        min_samples: int = 3,
    ) -> pd.DataFrame:
        """
        Clusters geo-referenced reports using spatial Euclidean/Haversine distance
        to identify geographic hotspots with multiple confirming citizen reports.
        """
        if df.empty or len(df) < min_samples:
            df["cluster_id"] = -1
            return df

        coords = df[["latitude", "longitude"]].values
        # Approximate 1 km in degrees ~ 0.009 deg
        eps_deg = eps_km / 111.0

        try:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=eps_deg, min_samples=min_samples).fit(coords)
            df = df.copy()
            df["cluster_id"] = clustering.labels_
            num_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            logger.info(f"Identified {num_clusters} high-density crisis hotspots.")
        except ImportError:
            logger.warning("scikit-learn not available. Setting single default cluster.")
            df["cluster_id"] = 0

        return df


if __name__ == "__main__":
    sample_feed = [
        {"report_id": "1", "latitude": 23.812, "longitude": 90.413, "text": "Water level rising rapidly near Mirpur road!"},
        {"report_id": "2", "latitude": 23.814, "longitude": 90.415, "text": "Families trapped on first floor, rescue needed!"},
        {"report_id": "3", "latitude": 23.811, "longitude": 90.411, "text": "Main road submerged, vehicles stranded."},
        {"report_id": "4", "latitude": 23.710, "longitude": 90.490, "text": "Light rain starting in Demra."},
    ]
    ingestor = CrisisSocialIngestor()
    df_reports = ingestor.parse_reports(sample_feed)
    clustered = ingestor.cluster_hotspots(df_reports, eps_km=1.5, min_samples=2)
    print(clustered[["report_id", "text", "urgency", "cluster_id"]])
