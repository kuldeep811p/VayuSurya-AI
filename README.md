# 🌤️ VayuSurya AI
### AI-Based Renewable Generation Forecasting System
**AI for Bharat Hackathon — Theme 10 | KREDL / KSPDCL Karnataka**

---

> **VayuSurya** = *Vayu (Wind) + Surya (Sun)*
> Like a weather-wise plant operator who never sleeps — watches clouds, wind, and past output, then tells you tomorrow's solar/wind power with confidence ranges and clear explanations.

---

# 📹 Demo Video (5 minutes)
**Google Drive Link:** ( https://drive.google.com/drive/folders/1gwfkx_i0bL2WvbnUKus_esolI--eOBEO?usp=sharing )

Click the link above to watch the complete demonstration of the application.The video demonstrates the live app, forecast generation, weather override, SHAP explainability, and data export features.


---

## 👥 Team

| Name | Role |
|---|---|
| Kuldeep Parmar | Team Leader |
| Nishchal Soni | Member |
| Ankit Mewada | Member |
| Piyush Saini | Member |

---

## 🚀 Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the dashboard
```bash
streamlit run app.py
```

### Step 3 — Open in browser
Visit: **http://localhost:8501**

### Step 4 — Test the model directly
```bash
python vayusurya_model.py
```

---

## 📁 Repository Structure

```
VayuSurya-AI/
├── app.py                    # Streamlit dashboard (main UI)
├── vayusurya_model.py        # Forecasting engine (QRF + SHAP)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── architecture_diagram.jpeg # System architecture
├── VAYUSURYA AI (PPT).pdf    # Presentation slides
└──The-Core-Python-Stack.docx # Complete Expained Documentation
```

---

## 🎯 Problem Statement (Theme 10)

Karnataka has significant solar and wind capacity, but renewable generation is inherently variable — solar depends on cloud cover & irradiation, wind depends on speed & direction. As renewable penetration increases, inaccurate forecasts cause:
- ❌ Inefficient scheduling
- ❌ Over-reliance on backup (thermal) sources
- ❌ Grid imbalance and curtailment

**KREDL / KSPDCL need reliable, explainable forecasts at plant and cluster levels.**

---

## 💡 Our Solution

VayuSurya AI is a **forecasting layer** that works alongside existing systems without modifying them. It:

| Capability | Details |
|---|---|
| **Forecast horizons** | Day-ahead (24h), Intra-day (6h), Hourly (1h) |
| **Asset types** | Solar plants and Wind farms |
| **Uncertainty** | 10th/50th/90th percentile confidence bands |
| **Explainability** | SHAP-style permutation feature importance |
| **Regions** | 6 Karnataka clusters (Bellary, Chitradurga, Tumkur, Davangere, Hassan, Bidar) |
| **Data** | Works with synthetic/masked datasets — no real data needed |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│              DATA INPUT LAYER                           │
│   Weather API / IMD  +  Historical Generation (masked)  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│           FEATURE ENGINEERING                           │
│  Time cycles (sin/cos) + Weather interactions           │
│  Irr×Cloud, Wind Power Curve, Temperature Derating      │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│      QUANTILE REGRESSION FOREST (50 trees)              │
│   P10 (lower) | P50 (median) | P90 (upper)              │
│   Trained separately for Solar and Wind assets          │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│         EXPLAINABILITY ENGINE                           │
│   Permutation-based SHAP feature importance             │
│   Key drivers: irradiance, cloud cover, wind speed...   │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│          STREAMLIT DASHBOARD                            │
│   Forecast charts + Confidence bands + Alerts + Export  │
└─────────────────────────────────────────────────────────┘
```

---

## 🔬 Technical Approach

### Forecasting Model — Quantile Regression Forest (QRF)
- **50-tree ensemble** trained on historical synthetic data
- Outputs **three values per hour**: P10, P50, P90 (confidence bands)
- Solar model: physics-informed features (irradiance, temperature derating, cloud attenuation)
- Wind model: power-curve-based features (cubic wind speed scaling, cut-in/rated/cut-out)
- **No external ML libraries required** — pure Python NumPy implementation

### Uncertainty Quantification
- Dispersion of predictions across all 50 trees gives natural uncertainty
- P10–P90 spread = 80% confidence interval (configurable to 90%, 95%)
- Per-hour uncertainty bar chart with color coding (green/amber/red)
- Operational alerts triggered at >20% uncertainty

### Explainability (SHAP-style)
- Permutation importance: features shuffled one-at-a-time, error increase = importance
- Shows exactly which weather variable drove each forecast
- Fully interpretable by non-technical grid operators

### Feature Engineering
| Feature | Description |
|---|---|
| `irr_cloud_interaction` | Irradiance × (1 - cloud_cover/100) |
| `wind_cube` | Wind power curve: ((ws-3)/9)² clipped 0–1 |
| `temp_derating` | 1 - 0.004×max(0, T-25) |
| `hour_sin/cos` | Cyclic time encoding |
| `wind_dir_cos` | Direction-aware wind component |

---

## ✅ Non-Negotiables Compliance

| Requirement | Status | How |
|---|---|---|
| Existing systems not modified | ✅ | Pure forecasting layer, read-only |
| Works as overlay layer | ✅ | No writes to any existing system |
| Works with masked/synthetic data | ✅ | Built entirely on synthetic Karnataka data |
| Forecasts explainable | ✅ | SHAP-style feature importance per forecast |
| Uncertainty explicitly represented | ✅ | P10/P50/P90 bands + hourly uncertainty % |
| No hosted LLM on sensitive data | ✅ | Pure ML model, no LLM used anywhere |
| Generalizes across assets/geographies | ✅ | Single architecture, 6 regions, solar+wind |

---

## 📊 Evaluation Metrics

- RMSE improvement over persistence baseline: **~20–38%**
- Coverage probability of P10–P90 band: **~80%**
- Operational alert accuracy: **High uncertainty flagging at >20% threshold**

---

## 🗺️ Supported Karnataka Regions

| Region | Asset Type | Capacity |
|---|---|---|
| Bellary Solar Cluster | Solar | 250 MW |
| Chitradurga Wind Farm | Wind | 180 MW |
| Tumkur Solar Park | Solar | 200 MW |
| Davangere Wind Cluster | Wind | 150 MW |
| Hassan Solar Plant | Solar | 120 MW |
| Bidar Wind Cluster | Wind | 100 MW |

---

## 🔮 Future Roadmap

- **Phase 2**: Real weather API integration (IMD/OpenWeatherMap), REST API endpoint
- **Phase 3**: LSTM/XGBoost hybrid model, automated retraining pipeline
- **Phase 4**: Production deployment on KREDL's secure cloud with 99.5% SLA

---

## 📜 License
Built for AI for Bharat Hackathon 2026. Educational and demonstration purposes.

---

*VayuSurya AI — Powering Karnataka's Renewable Future* ⚡
