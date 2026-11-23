"""
Flood-AI: Edge Inference Engine for Raspberry Pi 5 and Google Coral TPU.
Optimized for low-power, disconnected frontline early warning operation
using quantized INT8 TFLite / ONNX Runtime with EdgeTPU acceleration.
"""

from pathlib import Path
import time
from typing import Dict, Optional, Tuple, Union
import numpy as np
import logging

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger("Flood-AI.Edge")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


class EdgeTPUInferenceEngine:
    """
    Inference driver for Google Coral Edge TPU USB Accelerators and Raspberry Pi 5.
    Falls back gracefully to ONNX Runtime CPU or NumPy when hardware is unavailable.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        use_edgetpu_delegate: bool = True,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.use_edgetpu = use_edgetpu_delegate
        self.runtime_type = "cpu_fallback"
        self.interpreter = None

        self._initialize_runtime()

    def _initialize_runtime(self):
        """Attempts to load Coral EdgeTPU delegate; falls back to CPU."""
        if self.use_edgetpu:
            try:
                # Try PyCoral / TFLite EdgeTPU delegate
                from pycoral.utils.edgetpu import load_edgetpu_delegate
                from tflite_runtime.interpreter import Interpreter
                delegate = load_edgetpu_delegate()
                if self.model_path and self.model_path.exists():
                    self.interpreter = Interpreter(
                        model_path=str(self.model_path),
                        experimental_delegates=[delegate],
                    )
                    self.interpreter.allocate_tensors()
                    self.runtime_type = "google_coral_edgetpu"
                    logger.info("Initialized Google Coral EdgeTPU delegate successfully.")
                    return
            except (ImportError, RuntimeError) as e:
                logger.info(f"Coral EdgeTPU not detected ({e}). Using optimized CPU runtime.")

        # Fallback to ONNX Runtime or standard CPU
        try:
            import onnxruntime as ort
            if self.model_path and self.model_path.suffix == ".onnx" and self.model_path.exists():
                self.interpreter = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
                self.runtime_type = "onnxruntime_cpu"
                logger.info("Loaded ONNX Runtime CPU execution provider.")
                return
        except ImportError:
            pass

        self.runtime_type = "emulated_cpu"
        logger.info("Operating in lightweight emulated edge mode.")

    def benchmark_inference(
        self,
        input_shape: Tuple[int, int, int, int] = (1, 6, 128, 128),
        num_runs: int = 50,
    ) -> Dict[str, float]:
        """
        Benchmarks inference latency (mean, p95) and FPS on edge hardware.
        """
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        latencies_ms = []

        # Warm-up runs
        for _ in range(5):
            _ = self.predict_tile(dummy_input)

        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = self.predict_tile(dummy_input)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        mean_lat = float(np.mean(latencies_ms))
        p95_lat = float(np.percentile(latencies_ms, 95))
        fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

        logger.info(f"Edge Benchmark ({self.runtime_type}): Mean = {mean_lat:.2f} ms, p95 = {p95_lat:.2f} ms, FPS = {fps:.1f}")
        return {
            "mean_latency_ms": mean_lat,
            "p95_latency_ms": p95_lat,
            "fps": fps,
            "runtime": self.runtime_type,
        }

    def predict_tile(self, tile_tensor: np.ndarray) -> np.ndarray:
        """
        Runs single-tile inference on edge device.
        Input: (1, C, H, W) float32 or int8
        Output: (1, 1, H, W) inundation probability mask
        """
        if self.runtime_type == "google_coral_edgetpu" and self.interpreter:
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            self.interpreter.set_tensor(input_details[0]["index"], tile_tensor)
            self.interpreter.invoke()
            out = self.interpreter.get_tensor(output_details[0]["index"])
            return out
        elif self.runtime_type == "onnxruntime_cpu" and self.interpreter:
            input_name = self.interpreter.get_inputs()[0].name
            out = self.interpreter.run(None, {input_name: tile_tensor})[0]
            return out
        else:
            # Emulated lightweight sigmoid response
            B, C, H, W = tile_tensor.shape
            # Simple heuristic for testing: low SAR VV (channel 0) indicates water
            vv = tile_tensor[:, 0:1, :, :]
            synthetic_prob = 1.0 / (1.0 + np.exp(vv * 8.0 - 4.0))
            return synthetic_prob.astype(np.float32)


if __name__ == "__main__":
    edge_engine = EdgeTPUInferenceEngine()
    stats = edge_engine.benchmark_inference(input_shape=(1, 6, 128, 128), num_runs=20)
    print("Edge Performance Profile:", stats)
