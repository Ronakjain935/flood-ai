# Flood-AI System Architecture & Specification

## 1. System Vision & Objective
Flood-AI is a multimodal, real-time disaster early warning and evacuation guidance system designed to provide high-resolution (10–30m) flood extent mapping and 0–72h multi-horizon hydrological nowcasting.

```
+-----------------------------------------------------------------------------------+
|                           Multimodal Ingestion Layer                              |
|   Sentinel-1 SAR (VV/VH) | Sentinel-2 Optical | GPM Precipitation | USGS Sensors  |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                        Spatio-Temporal Fusion & Tiling                            |
|             CRS Alignment | 7x7 Lee Filter | 6-Channel Tensor Assembly             |
+-----------------------------------------------------------------------------------+
                                        |
                   +--------------------+--------------------+
                   |                                         |
                   v                                         v
+-------------------------------------+   +-------------------------------------+
|        Segmentation Engine          |   |          Forecasting Engine         |
|   ViTUNet (Transformer Bottleneck)  |   |   Temporal Fusion Transformer (TFT) |
|   PINN Gravity Loss Regularizer     |   |   Spatio-Temporal River GNN         |
+-------------------------------------+   +-------------------------------------+
                   |                                         |
                   +--------------------+--------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                         Civil Protection & Alerting                               |
|        Dynamic Alert Classification | Multilingual LLM Advisory Synthesis         |
+-----------------------------------------------------------------------------------+
                                        |
                   +--------------------+--------------------+
                   |                                         |
                   v                                         v
+-------------------------------------+   +-------------------------------------+
|         Cloud Deployment            |   |           Edge Frontline            |
|     FastAPI REST & Streamlit GIS    |   |     Raspberry Pi 5 + Coral TPU      |
+-------------------------------------+   +-------------------------------------+
```

## 2. Core Modules
- **`src/data/`**: Ingestion of satellite imagery, weather radar, river gauge time-series, and crisis social reports.
- **`src/models/`**: Vision Transformer + U-Net for spatial extent and Temporal Fusion Transformer for multi-horizon stage forecasts.
- **`src/inference/`**: End-to-end streaming inference pipeline and quantized INT8 edge runner for Google Coral TPU.
- **`src/evaluation/`**: Critical Success Index (CSI), IoU, Nash-Sutcliffe Efficiency (NSE), and Kling-Gupta Efficiency (KGE).
- **`src/api/`**: Async FastAPI service and interactive Streamlit GIS dashboard.
