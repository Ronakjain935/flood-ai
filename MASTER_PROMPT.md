# 🌊 Flood-AI: Master System Prompt & Engineering Specification

> **System Designation**: `Flood-AI: Multimodal Earth Observation & Spatio-Temporal Hydro-Forecasting Early Warning Platform`  
> **Target Environment**: Python 3.10+, PyTorch 2.0+, Flask, FastAPI, Leaflet.js, Chart.js, NumPy, Pandas, Scipy, Matplotlib

---

## 🎯 Master Prompt Overview

You are an expert Geospatial Machine Learning Engineer, Hydrological Modeler, and Full-Stack AI Architect. Your task is to design, implement, and maintain **Flood-AI**, a real-time, global early warning and evacuation intelligence system. 

The system fuses Multimodal Earth Observation (Synthetic Aperture Radar Sentinel-1 and Optical Sentinel-2), Digital Elevation Models (SRTM DEM), and river gauge telemetry to provide:
1. **10-meter resolution flood inundation segmentation** using Vision Transformer UNet (**ViTUNet**).
2. **0-72 hour probabilistic river stage nowcasting** using **Temporal Fusion Transformers (TFT)** and **Spatio-Temporal Graph Neural Networks (River-GNN)**.
3. **Universal City Intelligence & Auto-Feature Resolver**: Mention ANY city in the world to instantly identify its coordinates, river basin, danger marks, red spots, shelters, and native language advisories.
4. **Geospatial Red-Spot Probability Hotspots**: Real-time visualization of high-probability flood inundation zones (🔴 Red Circles scaled by probability) alongside safe shelters (🟢 Green Circles).
5. **Real-Time Audible Siren Alarm System**: Automatic dual-tone browser-based audio alarms triggered when danger thresholds are exceeded.
6. **Localized Multi-Lingual Civil Defense Advisories**: High-impact actionable evacuation orders in regional languages (Bengali, Hindi, Urdu, Vietnamese, German, Portuguese, Spanish, etc.) and English.
7. **Production MLOps & Testing**: FastAPI REST endpoints, Streamlit dashboard, Edge TPU quantization, and a 100% automated pytest suite.

---

## 🏗️ System Architecture & Specifications

### 1. Multimodal Data Ingestion & Signal Processing (`src/data/`)
* **Sentinel-1 SAR Processing (`ingest_sentinel.py`)**:
  - Ingest dual-polarization SAR backscatter ($VV, VH$ bands).
  - Convert raw linear intensity to decibels: $\sigma^0_{\text{dB}} = 10 \cdot \log_{10}(\text{amplitude}^2 + \epsilon)$.
  - Apply a spatial **$7 \times 7$ Enhanced Lee Speckle Filter** utilizing local coefficient of variation ($C_v = \sigma / \mu$) and damping factor to preserve river channel edges while smoothing speckle noise.
  - Compute polarization cross-ratio: $\text{CR} = \sigma^0_{VH} / \sigma^0_{VV}$.
* **Sentinel-2 Optical Processing (`ingest_sentinel.py`)**:
  - Ingest Green (Band 3), Red (Band 4), NIR (Band 8), SWIR-1 (Band 11).
  - Compute **Normalized Difference Water Index**: $\text{NDWI} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}$.
  - Compute **Modified NDWI**: $\text{MNDWI} = \frac{\text{Green} - \text{SWIR}}{\text{Green} + \text{SWIR}}$.
  - Apply cloud masking with QA60 / SCL scene classification.
* **River Gauge Telemetry Ingestion (`ingest_sensors.py`)**:
  - Query live river gauges (USGS NWIS / CWC) via REST API.
  - Implement zero-dependency simulated offline fallback when offline or in isolated sandboxes.
  - Resample time-series to uniform 15-minute intervals with cubic spline interpolation.
* **Multimodal Tensor Fusion (`fusion.py`)**:
  - Assemble a unified 7-channel geospatial tensor: $[VV_{\text{filtered}}, VH_{\text{filtered}}, \text{CR}, \text{DEM}_{\text{elevation}}, \text{Slope}, \text{Stage}_{\text{telemetry}}, \text{RateOfRise}]$.
  - Normalize with per-channel robust z-score statistics.

---

### 2. Deep Learning Segmentation & Hydro-Nowcasting (`src/models/`)
* **ViTUNet: Hybrid Vision Transformer U-Net (`segmentation/vit_unet.py`)**:
  - **Encoder**: 4-stage convolutional downsampling with residual blocks, feeding into a 2-layer Multi-Head Self-Attention (MHSA) Vision Transformer bottleneck (embedding dimension: 128, 4 heads).
  - **Decoder**: Symmetric transposed convolution upsampling with skip connections from encoder stages.
  - **Loss Function**: Compound Hybrid Loss: $\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{BCEWithLogits}} + 0.5 \cdot (1 - \text{DiceCoefficient})$.
* **Temporal Fusion Transformer (`forecasting/tft_model.py`)**:
  - Multi-horizon quantile river stage forecasting (0 to 72 hours).
  - Variable Selection Networks (VSN) for past stage, rainfall, and upstream discharges.
  - Output quantiles: $p_{10}$ (optimistic lower bound), $p_{50}$ (median expectation), $p_{90}$ (pessimistic flash surge).
  - Quantile Pinball Loss: $\mathcal{L}_{q}(y, \hat{y}) = \max(q(y - \hat{y}), (q - 1)(y - \hat{y}))$.
* **Spatio-Temporal River-GNN (`forecasting/gnn_river.py`)**:
  - Directed Acyclic Graph (DAG) modeling river tributaries and mainstem confluence.
  - Graph Convolutional Networks (GCN) coupled with GRU cells for hydrodynamic flow propagation.

---

### 3. Universal City Intelligence & Feature Resolver (`src/data/city_feature_resolver.py`)
* When ANY city in the world is mentioned:
  - Automatically resolve **Latitude & Longitude**, country flag, and map zoom.
  - Identify primary **river system, canal network, bayou, or coastal lagoon**.
  - Identify **flood vulnerability regime** (Monsoon Riverine, Flash Flood, Coastal Storm Surge, Tidal Acqua Alta, Bayou Network Overflow).
  - Retrieve or calculate calibrated **danger stage marks** ($m$) and emergency limits.
  - Synthesize **🔴 Red Spot Hotspots** along the city's lowlands and watercourses with real population numbers and calibrated probabilities.
  - Designate **🟢 Certified Emergency Shelters** (stadiums, university high grounds, civic centers).
  - Generate **multi-lingual localized evacuation advisories** in the native language of that city.

---

### 4. Geospatial Visualization & Red Spot Hotspots (`src/api/flask_app.py`)
* **Interactive Leaflet.js & OpenStreetMap Geospatial Engine**:
  - **🔴 Red Spots**: Flood probability $\ge 75\%$, radius scaled dynamically from 1,200m to 3,000m based on inundation probability.
  - **🟠 Orange Spots**: Watch / elevated probability ($50\% - 74\%$).
  - **🟢 Green Circles**: Certified safe emergency high-ground shelters.
  - Rich interactive popups displaying location name, flood risk probability %, stage vs. danger mark, and population at risk.
* **Live Probability Radar Table**: Real-time table displaying all monitored river gauges, flood probabilities, status badges, and recommended actions.
* **Interactive Chart.js 72h Hydrograph**: Quantile prediction curves ($p_{10}, p_{50}, p_{90}$) and danger mark limit lines.

---

### 5. Autonomous Siren Alarm System (`src/api/alarm_sound.py`)
* **Dual-Tone Emergency Siren Synthesis**:
  - Synthesizes 16-bit PCM WAV dual-tone emergency sweep (800 Hz – 1200 Hz at 2 Hz modulation) without any external audio file downloads.
  - Returns base64 `data:audio/wav;base64,...` URI for direct browser `<audio>` playback.
* **HTML5 Web Audio API Synth**:
  - Injects dynamic `AudioContext` sawtooth oscillator that sweeps frequencies continuously during active alarms.
  - Includes an emergency pulsing banner and interactive `🔊 Sound Siren Now` / `🔇 Silence Siren` buttons.
  - Manual "🔔 Test Siren Sound" button in the sidebar.

---

### 6. Localized Evacuation Directives (`src/inference/alert_generator.py`)
* Evaluates hydraulic parameters: Inundation hectares, rate of rise ($m/\text{hr}$), and stage threshold exceedance.
* Classifies severity: `NORMAL`, `ADVISORY`, `WATCH`, `WARNING`, `EMERGENCY`.
* Generates actionable directives in English and the local language (Bengali, Hindi, Urdu, Vietnamese, German, Portuguese, Tagalog, Mandarin, etc.).

---

### 7. Deployment, MLOps, and Master Pipeline
* **Unified Master Pipeline**:
  - [`flood_ai_master_pipeline.ipynb`](file:///c:/Users/HP/OneDrive/Music/Desktop/flood_monitoring_system/flood-ai/notebooks/flood_ai_master_pipeline.ipynb) & [`flood_ai_master_pipeline.py`](file:///c:/Users/HP/OneDrive/Music/Desktop/flood_monitoring_system/flood-ai/notebooks/flood_ai_master_pipeline.py) covering all 6 stages from raw SAR ingestion to global red spots and siren audio synthesis.
* **REST API**:
  - [`src/api/fastapi_app.py`](file:///c:/Users/HP/OneDrive/Music/Desktop/flood_monitoring_system/flood-ai/src/api/fastapi_app.py) exposing `/health`, `/predict/inundation`, `/forecast/hydrograph`, `/alert/dispatch`.
* **Testing Suite**:
  - 15 automated pytest test cases in `tests/` passing 100%.
* **Launch Scripts**:
  - Root batch scripts `run_dashboard.bat`, `run_api.bat`, and `run_pipeline.bat` for 1-click execution on Windows.
