# Bid Recommendation System

**Global Stat Solutions** | ML-powered bid fee + win probability prediction for commercial real estate appraisal engagements.

The system tells appraisers what to bid on each engagement by maximizing **Expected Value = P(Win) × BidFee**. Two independent LightGBM models — one regression, one classification — run on every prediction request and combine into a recommendation that balances winnability with revenue.

A $5,200 bid with 25% win chance (EV = $1,300) is worse than a $3,500 bid with 65% win chance (EV = $2,275). The system recommends the latter.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BID RECOMMENDATION SYSTEM                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────────────┐ │
│  │   FRONTEND   │    │    REST API      │    │      ML MODELS             │ │
│  │   (React)    │───▶│    (Flask)       │───▶│                            │ │
│  │              │    │                  │    │  ┌────────────────────┐    │ │
│  │  - Form UI   │    │  /api/v2/predict │    │  │ Phase 1A: BidFee   │    │ │
│  │  - Results   │    │  /api/v2/options │    │  │ LightGBM Regressor │    │ │
│  │  - Charts    │◀───│  /api/health     │◀───│  │ - 60 features      │    │ │
│  │              │    │                  │    │  │ - MAPE 3.9%, R² .94│    │ │
│  └──────────────┘    └──────────────────┘    │  └────────────────────┘    │ │
│        │                     │               │                            │ │
│        │                     │               │  ┌────────────────────┐    │ │
│        ▼                     ▼               │  │ Phase 1B: P(Win)   │    │ │
│   ┌─────────┐         ┌─────────────┐        │  │ LightGBM Classifier│    │ │
│   │ Vercel  │         │   Render    │        │  │ - v2 AUC 0.948     │    │ │
│   │ (Host)  │         │   (Host)    │        │  │ - v3 fee-elastic   │    │ │
│   └─────────┘         └─────────────┘        │  └────────────────────┘    │ │
│                                              │  EV = P(Win) × BidFee      │ │
│                                              └────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

For a full system map see **[ARCHITECTURE.md](./ARCHITECTURE.md)**. For model specifications see **[MODEL_CARD.md](./MODEL_CARD.md)**. For end-user documentation see **[docs/USER_GUIDE.md](./docs/USER_GUIDE.md)**.

---

## Features

- **Bid Fee Prediction** (Phase 1A): LightGBM regressor on 60 engineered features. v2 test MAPE 3.9%, R² 0.94, overfit ratio 1.49×.
- **Win Probability** (Phase 1B): LightGBM classifier with `BidFee` as a direct feature so the model learns fee-sensitivity from data. v2 test AUC 0.948, Brier 0.093.
- **EV Optimization**: `EV = P(Win) × BidFee`. The system recommends the fee that maximizes expected revenue, not just the market-typical fee.
- **Fee Sensitivity Curve**: 20-point P(Win) sweep across the fee range so the EV optimizer can find an interior maximum. When v2 saturates, falls back to the v3 fee-sensitive model (AUC 0.883, ~24pp elasticity span).
- **Confidence Intervals**: Empirical 80%-coverage quantile bands stratified by fee bucket.
- **Market Benchmarks**: Blended (segment + state + property-type) average for the comparison strip and recommendation text.
- **Zip Code Demographics**: 20,040-zip lookup table feeds per-zip features (population, median income, etc.) at inference.
- **Real-time API**: RESTful Flask endpoints. Live on Render with the React frontend on Vercel.

---

## Project Structure

```
BidRecommendationSystem/
├── README.md
├── ARCHITECTURE.md                # Full system map
├── MODEL_CARD.md                  # Model specs + limitations
├── LICENSE                        # MIT
│
├── api/                           # Flask REST API
│   ├── app.py                     # Routes, CORS, health, debug
│   ├── enhanced_prediction_service.py   # v2 + v3 prediction logic (production)
│   ├── prediction_service.py      # v1 service (legacy, /api/predict)
│   └── empirical_bands.py         # Confidence interval calculator
│
├── frontend/                      # React UI
│   └── src/components/
│       ├── BidForm.js             # Input form
│       └── ResultDisplay.js       # Results panel (fee + win prob + curve)
│
├── config/
│   └── model_config.py            # Hyperparameters, paths, exclusion lists
│
├── scripts/                       # Training & analysis pipeline (numbered sequentially)
│   ├── 02_data_preprocessing.py   # raw → cleaned
│   ├── 03_feature_engineering.py  # cleaned → features
│   ├── 04_model_lightgbm.py       # v1 Phase 1A
│   ├── 15_win_probability_baseline.py    # v1 Phase 1B
│   ├── 21_jobsdata_preprocessing.py      # v2 pipeline starts
│   ├── 22_combined_data_preprocessing.py
│   ├── 23_enhanced_feature_engineering.py
│   ├── 24_enhanced_bidfee_model.py       # v2 Phase 1A
│   ├── 25_enhanced_win_probability.py    # v2 Phase 1B
│   ├── 27_generate_api_stats_v2.py       # Runtime stats for API
│   ├── 29_fee_sensitive_win_prob.py      # v3 Phase 1B
│   └── 29b_v2_vs_v3_curve_comparison.py
│
├── outputs/
│   ├── models/                    # LightGBM .txt + metadata JSON
│   │   ├── lightgbm_bidfee_v2_model.txt
│   │   ├── lightgbm_win_probability_v2.txt
│   │   ├── lightgbm_win_probability_v3_fee_sensitive.txt   # saturation fallback
│   │   └── win_probability_v2_calibrator.pkl
│   ├── reports/                   # Precomputed runtime stats
│   │   ├── api_precomputed_stats_v2.json
│   │   ├── feature_defaults_v2.json
│   │   ├── empirical_bands.json
│   │   ├── win_rate_lookups_v3.json
│   │   ├── zip_demographics_lookup.json
│   │   └── ...
│   └── figures/                   # Training plots, evaluation charts
│
├── tests/
│   └── test_validation.py         # 41 tests (no data dependency)
│
├── docs/
│   ├── USER_GUIDE.md              # End-user guide
│   ├── V3_DEPLOYMENT_NOTES.md     # v3 runbook
│   ├── MODEL_VALIDATION_SUMMARY.md
│   ├── DATA_LEAKAGE_ANALYSIS.md
│   ├── FEATURE_ENGINEERING_DOCUMENTATION.md
│   ├── DEEP_TECHNICAL_MEETING_NOTES.md
│   └── BidRecommendationSystem_Report.{tex,pdf}
│
├── data/                          # NOT in repo (gitignored — IP)
│   ├── raw/                       # Original BidData.csv + JobsData.csv
│   ├── processed/                 # Cleaned intermediate files
│   └── features/                  # Feature-engineered training data
│
├── requirements.txt               # Full dev dependencies
├── requirements-api.txt           # Minimal API-only (10 packages, for Render)
├── render.yaml                    # Render deployment config
└── .python-version                # 3.11
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+

### 1. Clone & setup backend

```bash
git clone https://github.com/ujjawal40/BidRecommendationSystem.git
cd BidRecommendationSystem

python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the API

```bash
cd api && python app.py
# Serves on http://localhost:5001
```

### 3. Start the frontend (separate terminal)

```bash
cd frontend && npm install && npm start
# Serves on http://localhost:3000
```

### 4. Run tests

```bash
pytest tests/test_validation.py -v   # 41 tests, no data required
```

---

## API Reference

### Health
```
GET /api/health
```
Returns service status, v2 model load state, and v3 fee-sensitive load state.

### Debug
```
GET /api/debug
```
Returns per-file presence map for every runtime dependency. Use this on Render to diagnose missing-file deploy issues.

### Options (v2)
```
GET /api/v2/options
```
Returns dropdown values: business segments, property types, sub-types (cascading by property type), property states, office regions, office locations.

### Zip Lookup
```
GET /api/v2/zip/{5-digit-zip}
```
Returns whether the zip is in the demographics lookup table and what state it resolves to. Used by the frontend to auto-derive state from zip.

### Predict (v2 — production)
```
POST /api/v2/predict
Content-Type: application/json

{
  "business_segment":    "Financing",
  "property_type":       "Multifamily",
  "property_state":      "Illinois",
  "target_time":         30,
  "sub_property_type":   "Conventional",   // optional
  "office_location":     "Chicago",         // optional
  "delivery_days":       30,                // optional
  "zip_code":            "60601",           // optional
  "open_date":           "2026-05-14"       // optional, YYYY-MM-DD
}
```

**Response** (abbreviated):
```json
{
  "predicted_fee":       2706.65,
  "confidence_interval": { "low": 1678, "high": 3500 },
  "confidence_level":    "high",
  "segment_benchmark":   2863.35,
  "state_benchmark":     2248.63,
  "win_probability": {
    "probability_pct":   67.1,
    "confidence":        "high",
    "model_used":        "LightGBM Classifier v2 (AUC: 0.948)"
  },
  "expected_value":      1816.76,
  "ev_optimal_fee":      2767.40,
  "ev_capped_at_ceiling": false,
  "recommendation":      { "...": "..." },
  "fee_curve":           { "curve_points": [/* 20-point sweep */] },
  "metadata":            { "zip_info": "...", "data_coverage": "..." }
}
```

The `fee_curve.curve_points` array is the input to the EV optimizer in the UI — it's how the frontend draws the win-probability-vs-fee chart.

### Predict (v1 — legacy)
```
POST /api/predict
```
Older endpoint with a simpler response schema. Kept for backwards compatibility; new clients should use `/api/v2/predict`.

### Segment statistics
```
GET /api/segment/{segment_name}
```
Returns average fee, count, and win rate for a single business segment.

---

## Deployment

### Frontend → Vercel

1. Connect the repo to Vercel
2. Set root directory: `frontend`
3. Add environment variable:
   ```
   REACT_APP_API_URL=https://your-api.onrender.com
   ```

### Backend → Render

1. Connect the repo to Render
2. Render auto-detects `render.yaml` (Python 3.11, gunicorn, 120 s timeout, health probe on `/api/health`)
3. Uses `requirements-api.txt` (10 packages — no torch / shap / matplotlib)

The API loads models + precomputed stats lazily on the first `/api/predict` request, so cold start is ~3–5 s; subsequent requests are <100 ms.

---

## Model Performance

### Phase 1A — Bid Fee Regression (v2, production)

| Metric | Value |
|--------|-------|
| Algorithm | LightGBM with `log1p` target transform |
| Features | 60 |
| Test RMSE | $438 |
| Test MAPE | 3.9% |
| Test R² | 0.940 |
| Overfit ratio (train/test RMSE) | 1.49× |
| Within ±20% accuracy | 96.0% (5-fold CV) |

### Phase 1B — Win Probability Classification (v2, production)

| Metric | Value |
|--------|-------|
| Algorithm | LightGBM binary classifier with `BidFee` as a feature |
| Features | 44 |
| Calibration | Isotonic regression on validation set |
| Test AUC | 0.948 |
| Test Brier | 0.093 |
| Test accuracy | 86.7% |
| Test recall | 86.1% |

### Phase 1B Fallback — v3 fee-sensitive (saturation backup)

When v2 raw output saturates (≥ 0.93 for typical inputs — common for high-baseline segments), the API switches to v3 fee-sensitive so the EV optimizer has a real elasticity curve to maximize against. v3 is a worse ranker than v2 but produces fee elasticity v2 lacks.

| Metric | Value |
|--------|-------|
| Test AUC | 0.883 |
| Test Brier | 0.140 |
| Fee curve span (training) | ~24 pp |
| Fee curve span (API canary) | ~36 pp |

See [docs/V3_DEPLOYMENT_NOTES.md](./docs/V3_DEPLOYMENT_NOTES.md) for the full runbook.

### Top Bid Fee Features

| Rank | Feature | Importance |
|------|---------|-----------:|
| 1 | `segment_avg_fee` | ~64% |
| 2 | `state_avg_fee` | ~10% |
| 3 | `propertytype_avg_fee` | ~5% |
| 4 | `TargetTime` | ~4% |
| 5 | `subtype_avg_fee` (v2 addition) | ~3% |

The top 5 features explain ~90% of the variance — pricing is primarily a market-context problem, not an individual-deal problem.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18, plain CSS (no framework) |
| Backend | Flask 3, gunicorn, flask-cors |
| ML | LightGBM 4.x, scikit-learn (isotonic calibration), pandas, numpy |
| Plots / training | matplotlib, SHAP |
| Tests | pytest |
| CI | GitHub Actions (flake8, pytest, safety scan) |
| Deployment | Vercel (frontend) + Render (API) |
| Python | 3.11 |

---

## License

MIT — see [LICENSE](./LICENSE).

---

**Global Stat Solutions** | Bid Recommendation System v2.0 (production) + v3 fee-sensitive saturation fallback
