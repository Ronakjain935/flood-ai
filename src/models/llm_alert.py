"""
Flood-AI: Open-Source Small Language Model (LLM) for Evacuation Advisory Generation.
Wraps lightweight open models (e.g., Phi-3-mini, Gemma-2b, Mistral-7B) to translate
complex geospatial telemetry and probability rasters into plain-language advisories.
"""

from typing import Dict, List, Optional
import logging

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger("Flood-AI.LLMAlert")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


class OpenLLMAlertGenerator:
    """
    Interface for local quantized Small Language Models (SLMs).
    Constructs prompt context from hydrological telemetry and synthesizes clear guidance.
    """

    SYSTEM_PROMPT = (
        "You are an expert disaster early warning and civil protection AI. "
        "Your task is to analyze real-time flood monitoring data and generate a clear, "
        "compassionate, highly actionable, and urgent evacuation advisory for the public. "
        "Avoid overly technical jargon. Always state danger level, expected peak time, "
        "safe evacuation routes, and emergency shelter locations."
    )

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        load_in_4bit: bool = False,
    ):
        self.model_name = model_name
        self.pipeline = None
        self._load_model_if_available(load_in_4bit)

    def _load_model_if_available(self, load_in_4bit: bool):
        """Attempts to initialize transformers text generation pipeline."""
        try:
            from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
            logger.info(f"Loading LLM {self.model_name}...")
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device_map="auto",
                model_kwargs={"load_in_4bit": load_in_4bit} if load_in_4bit else {},
            )
            logger.info("Local LLM initialized successfully.")
        except Exception as e:
            logger.info(f"Operating in deterministic template fallback mode (LLM offline: {e}).")
            self.pipeline = None

    def synthesize_alert(
        self,
        region: str,
        inundated_hectares: float,
        current_stage_m: float,
        forecast_peak_stage_m: float,
        peak_eta_hours: int,
        shelters: List[str],
        routes: List[str],
    ) -> str:
        """Synthesizes human-interpretable evacuation instructions."""
        if self.pipeline is not None:
            prompt = (
                f"{self.SYSTEM_PROMPT}\n\n"
                f"Region: {region}\n"
                f"Flooded Area: {inundated_hectares:.1f} hectares\n"
                f"Current Water Height: {current_stage_m:.2f} meters\n"
                f"Forecast Peak Height: {forecast_peak_stage_m:.2f} meters\n"
                f"Estimated Time to Peak: {peak_eta_hours} hours\n"
                f"Available Shelters: {', '.join(shelters)}\n"
                f"Safe Evacuation Corridors: {', '.join(routes)}\n\n"
                "Generate the advisory:"
            )
            try:
                outputs = self.pipeline(prompt, max_new_tokens=200, temperature=0.3)
                return outputs[0]["generated_text"]
            except Exception as err:
                logger.warning(f"Inference error in LLM: {err}; using template fallback.")

        # High-reliability template fallback
        return (
            f"URGENT FLOOD ADVISORY for {region.upper()}:\n"
            f"River levels are currently at {current_stage_m:.2f}m and projected to surge to "
            f"{forecast_peak_stage_m:.2f}m within {peak_eta_hours} hours. An estimated {inundated_hectares:.0f} "
            f"hectares are at risk of inundation.\n"
            f"ACTION REQUIRED: Evacuate immediately if residing in low-lying zones.\n"
            f"Safe Shelters: {', '.join(shelters)}.\n"
            f"Designated Routes: {', '.join(routes)}."
        )


if __name__ == "__main__":
    generator = OpenLLMAlertGenerator()
    advisory = generator.synthesize_alert(
        region="Mirpur, Dhaka",
        inundated_hectares=320.0,
        current_stage_m=3.4,
        forecast_peak_stage_m=4.6,
        peak_eta_hours=8,
        shelters=["Mirpur Govt High School", "Sector 4 Community Hub"],
        routes=["Airport Elevated Flyover", "Mirpur-10 North Corridor"],
    )
    print("Generated Alert:\n", advisory)
