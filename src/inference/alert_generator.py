"""
Flood-AI: Multilingual Evacuation Advisory and Early Warning Generator.
Translates technical inundation polygons and hydrograph exceedances into
plain-language, high-impact advisories (English and Bengali for South Asia pilot).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class AlertSeverity(str, Enum):
    NORMAL = "NORMAL"
    ADVISORY = "ADVISORY"       # Minor ponding, be aware
    WATCH = "WATCH"             # Flood possible within 24-48h
    WARNING = "WARNING"         # Flooding imminent within 6-12h
    EMERGENCY = "EMERGENCY"     # Rapid rise, life-threatening, evacuate now


@dataclass
class AlertNotification:
    """Standardized emergency dispatch message."""
    level: AlertSeverity
    headline: str
    english_advisory: str
    local_advisory: str
    evacuation_routes: List[str]
    designated_shelters: List[str]


class FloodAlertGenerator:
    """
    Generates actionable early warning alerts. Combines rule-based hydrological
    threshold evaluation with multi-lingual localized message synthesis.
    """

    def __init__(self, stage_threshold_warning_m: float = 4.0, stage_threshold_emergency_m: float = 4.8):
        self.stage_warning = stage_threshold_warning_m
        self.stage_emergency = stage_threshold_emergency_m

    def classify_severity(
        self,
        inundation_hectares: float,
        max_forecast_stage_m: float,
        rate_of_rise_m_hr: float,
    ) -> AlertSeverity:
        """Determines alert severity level from multi-source physical parameters."""
        if max_forecast_stage_m >= self.stage_emergency or rate_of_rise_m_hr >= 0.5:
            return AlertSeverity.EMERGENCY
        elif max_forecast_stage_m >= self.stage_warning or inundation_hectares >= 500.0:
            return AlertSeverity.WARNING
        elif max_forecast_stage_m >= (self.stage_warning * 0.85) or rate_of_rise_m_hr >= 0.2:
            return AlertSeverity.WATCH
        elif inundation_hectares > 50.0:
            return AlertSeverity.ADVISORY
        return AlertSeverity.NORMAL

    def generate_advisory(
        self,
        inundation_hectares: float,
        max_forecast_stage_m: float,
        rate_of_rise_m_hr: float = 0.2,
        region_name: str = "Dhaka City",
    ) -> AlertNotification:
        """Synthesizes structured advisory notification in English and Bengali."""
        severity = self.classify_severity(inundation_hectares, max_forecast_stage_m, rate_of_rise_m_hr)

        shelters = [
            "Mirpur Government High School Relief Shelter",
            "Demra Riverbank Flood Center",
            "Uttara Sector 4 Multi-Purpose Cyclone/Flood Facility",
        ]
        routes = [
            "Use elevated Mirpur Flyover towards Airport Road",
            "Avoid low-lying Buriganga Embankment Road",
            "Take Dhaka-Mymensingh Highway northward",
        ]

        if severity == AlertSeverity.EMERGENCY:
            headline = f"CRITICAL FLOOD EMERGENCY: IMMEDIATE EVACUATION ORDER FOR {region_name.upper()}"
            eng_text = (
                f"River levels are forecasted to reach {max_forecast_stage_m:.2f} meters, rising rapidly at "
                f"{rate_of_rise_m_hr:.2f} m/hr. Severe flooding of {inundation_hectares:.0f} hectares is underway. "
                "MOVE TO HIGH GROUND IMMEDIATELY. Do not walk or drive through moving floodwaters."
            )
            ben_text = (
                f"জরুরি বন্যা সতর্কতা: {region_name} এলাকায় দ্রুত পানি বৃদ্ধি পাচ্ছে (উচ্চতা {max_forecast_stage_m:.2f} মিটার)। "
                "অবিলম্বে নিরাপদ আশ্রয়কেন্দ্রে বা উঁচু স্থানে সরে যান। পানিতে গাড়ি বা নৌকা চালানো থেকে বিরত থাকুন।"
            )
        elif severity == AlertSeverity.WARNING:
            headline = f"FLOOD WARNING: INUNDATION IMMINENT FOR {region_name.upper()}"
            eng_text = (
                f"Water levels are expected to exceed danger threshold ({max_forecast_stage_m:.2f}m) within 6-12 hours. "
                f"Projected inundation: {inundation_hectares:.0f} hectares. Prepare emergency kits and secure livestock."
            )
            ben_text = (
                f"বন্যা সতর্কতা: আগামী ৬-১২ ঘণ্টার মধ্যে {region_name} এলাকায় বিপৎসীমা অতিক্রম করতে পারে। "
                "প্রয়োজনীয় ওষুধ, শুকনো খাবার ও দলিলপত্র সংরক্ষণ করুন এবং সতর্ক থাকুন।"
            )
        elif severity == AlertSeverity.WATCH:
            headline = f"FLOOD WATCH: HEAVY RAINFALL & RISING WATERS FOR {region_name.upper()}"
            eng_text = (
                f"Sustained monsoon rain is causing river stages to rise. Forecast peak: {max_forecast_stage_m:.2f}m. "
                "Monitor updates and identify your nearest emergency evacuation shelter."
            )
            ben_text = f"বন্যা পর্যবেক্ষণ: ভারি বৃষ্টির কারণে {region_name} নদীতে পানি বৃদ্ধি পাচ্ছে। খবরাখবর নজরে রাখুন।"
        else:
            headline = f"FLOOD ADVISORY: NORMAL CONDITIONS IN {region_name.upper()}"
            eng_text = f"River levels ({max_forecast_stage_m:.2f}m) are currently within safe seasonal operational margins."
            ben_text = f"{region_name} এলাকার নদ-নদীর পানি স্বাভাবিক প্রবাহের মধ্যে রয়েছে।"

        return AlertNotification(
            level=severity,
            headline=headline,
            english_advisory=eng_text,
            local_advisory=ben_text,
            evacuation_routes=routes,
            designated_shelters=shelters,
        )


if __name__ == "__main__":
    gen = FloodAlertGenerator()
    alert = gen.generate_advisory(
        inundation_hectares=650.0,
        max_forecast_stage_m=4.95,
        rate_of_rise_m_hr=0.45,
        region_name="Turag Basin, Dhaka",
    )
    print("Headline:", alert.headline)
    print("Severity:", alert.level)
    print("English:", alert.english_advisory)
    print("Bengali:", alert.local_advisory)
