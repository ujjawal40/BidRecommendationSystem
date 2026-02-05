# Bid Recommendation System

**Global Stat Solutions** | AI-Powered Bid Fee Prediction Platform

A machine learning system that predicts optimal bid fees for commercial real estate appraisal services, helping appraisers make data-driven pricing decisions.

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
│  │  - Form UI   │    │  /api/predict    │    │  │ LightGBM Regressor │    │ │
│  │  - Results   │    │  /api/options    │    │  │ (Bid Fee Model)    │    │ │
│  │  - Charts    │◀───│  /api/health     │◀───│  │ - 500 trees        │    │ │
│  │              │    │                  │    │  │ - 68 features      │    │ │
│  └──────────────┘    └──────────────────┘    │  │ - MAE: $108        │    │ │
│        │                     │               │  └────────────────────┘    │ │
│        │                     │               │                            │ │
│        ▼                     ▼               │  ┌────────────────────┐    │ │
│   ┌─────────┐         ┌─────────────┐        │  │ Win Probability    │    │ │
│   │ Vercel  │         │   Render    │        │  │ Classifier         │    │ │
│   │ (Host)  │         │   (Host)    │        │  │ - AUC: 0.96        │    │ │
│   └─────────┘         └─────────────┘        └────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Bid Fee Prediction**: ML-powered predictions based on 68 engineered features
- **Win Probability**: Calibrated probability estimates (AUC: 0.96) for bid success
- **Confidence Intervals**: Empirical quantile bands (80% coverage) with heteroscedastic estimation
- **Market Benchmarks**: Compare against segment and state averages
- **Real-time API**: RESTful endpoints for integration

---

## Project Structure

```
BidRecommendationSystem/
├── api/                          # Flask REST API
│   ├── app.py                    # API endpoints
│   ├── prediction_service.py    # Core prediction logic
│   └── empirical_bands.py       # Confidence interval calculator
│
├── frontend/                     # React UI
│   ├── src/
│   │   ├── components/          # React components
│   │   │   ├── BidForm.js       # Input form
│   │   │   ├── ResultDisplay.js # Prediction results
│   │   │   └── Header.js        # Navigation
│   │   ├── services/            # API client
│   │   └── App.js               # Main app
│   └── package.json
│
├── config/                       # Configuration
│   └── model_config.py          # Model paths & settings
│
├── outputs/
│   ├── models/                  # Trained models
│   │   ├── lightgbm_bidfee_model.txt
│   │   └── lightgbm_win_probability.txt
│   └── reports/                 # Precomputed statistics
│       ├── empirical_bands.json
│       ├── feature_defaults.json
│       └── rolling_stats.json
│
├── scripts/                     # Training & analysis scripts
│
├── data/                        # Data directory (not in repo)
│   ├── raw/                     # Original data
│   ├── processed/               # Cleaned data
│   └── features/                # Feature-engineered data
│
├── requirements.txt             # Python dependencies
├── render.yaml                  # Render deployment config
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+

### 1. Clone & Setup Backend

```bash
git clone https://github.com/ujjawal40/BidRecommendationSystem.git
cd BidRecommendationSystem

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start API Server

```bash
cd api
python app.py
# API runs at http://localhost:5001
```

### 3. Start Frontend

```bash
cd frontend
npm install
npm start
# UI runs at http://localhost:3000
```

---

## API Reference

### Health Check
```
GET /api/health
```

### Get Options
```
GET /api/options
```

### Predict Bid Fee
```
POST /api/predict
Content-Type: application/json

{
  "business_segment": "Financing",
  "property_type": "Multifamily",
  "property_state": "Texas",
  "target_time": 30
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "predicted_fee": 3127.98,
    "confidence_interval": {"low": 3056.05, "high": 3188.72},
    "segment_benchmark": 3215.48,
    "win_probability": 0.72,
    "expected_value": 2252.15,
    "recommendation": "Predicted fee is within ±10% of segment average. Good competitive position."
  }
}
```

---

## Deployment

### Frontend → Vercel

1. Connect repo to Vercel
2. Set root directory: `frontend`
3. Add environment variable:
   ```
   REACT_APP_API_URL=https://your-api.onrender.com
   ```

### Backend → Render

1. Connect repo to Render
2. Uses `render.yaml` config automatically
3. Deploys at `https://bid-recommendation-api.onrender.com`

---

## Data & Training

- **Training Data**: 2023+ bids only (recent market conditions)
- **Split Strategy**: Time-based 60/20/20 (train/validation/test)
- **Total Samples**: ~52,000 bid records
- **Feature Engineering**: 68+ features including rolling averages, client history, market benchmarks

---

## Model Performance

### Bid Fee Prediction (Phase 1A)

| Metric | Value |
|--------|-------|
| Algorithm | LightGBM |
| Trees | 500 |
| Features | 68 |
| Test RMSE | $328.75 |
| Test MAE | $215.42 |
| Overfitting Ratio | 1.99x |

### Win Probability (Phase 1B)

| Metric | Value |
|--------|-------|
| Algorithm | LightGBM Classifier |
| Test AUC-ROC | 0.962 |
| Test Accuracy | 89.2% |
| Brier Score | 0.078 |

### Top Features (Bid Fee Model)

1. `segment_avg_fee` (63%) - Average fee for business segment
2. `state_avg_fee` (10%) - State-level market pricing
3. `propertytype_avg_fee` (5%) - Property type benchmarks
4. `TargetTime` (4%) - Delivery timeline
5. `rolling_avg_fee_segment` (3%) - Recent segment trends

### Top Features (Win Probability)

1. `JobCount` (47%) - Office workload capacity
2. `market_competitiveness` (11%) - Market competition level
3. `TargetTime_Original` (4%) - Delivery requirements
4. `PropertyState_frequency` (3%) - State market activity
5. `RooftopLongitude` (3%) - Geographic positioning

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18, Axios |
| Backend | Flask, Gunicorn, Python 3.11 |
| ML | LightGBM 4.6, scikit-learn, pandas |
| Deployment | Vercel (frontend), Render (API) |
| CI/CD | GitHub Actions |

---

## Anti-Overfitting Measures

- **L1/L2 Regularization**: reg_alpha=2.0, reg_lambda=2.0
- **Tree Constraints**: max_depth=8, num_leaves=18, min_child_samples=30
- **Sampling**: feature_fraction=0.8, bagging_fraction=0.8
- **Early Stopping**: 50 rounds patience on validation set
- **Recent Data Only**: Training on 2023+ data for better generalization

---

## License

Proprietary - Global Stat Solutions

---

**Global Stat Solutions** | Bid Recommendation System v2.0
