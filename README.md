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
│  └──────────────┘    └──────────────────┘    │  │ - RMSE: $328       │    │ │
│        │                     │               │  └────────────────────┘    │ │
│        │                     │               │                            │ │
│        ▼                     ▼               │  ┌────────────────────┐    │ │
│   ┌─────────┐         ┌─────────────┐        │  │ Win Probability    │    │ │
│   │ Vercel  │         │   Render    │        │  │ (Experimental)     │    │ │
│   │ (Host)  │         │   (Host)    │        │  └────────────────────┘    │ │
│   └─────────┘         └─────────────┘        └────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Bid Fee Prediction**: ML-powered predictions based on 68 features
- **Confidence Intervals**: Empirical quantile bands (80% coverage)
- **Win Probability**: Experimental competitive positioning indicator
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
    "segment_benchmark": 3215.48
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

## Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | LightGBM |
| Trees | 500 |
| Features | 68 |
| Test RMSE | $328.75 |
| Test MAE | $215.42 |

### Top Features

1. `segment_avg_fee` (63%)
2. `state_avg_fee` (10%)
3. `propertytype_avg_fee` (5%)
4. `TargetTime` (4%)
5. `rolling_avg_fee_segment` (3%)

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18 |
| Backend | Flask, Python 3.11 |
| ML | LightGBM, scikit-learn |
| Deployment | Vercel, Render |

---

## License

Proprietary - Global Stat Solutions

---

**Global Stat Solutions** | Bid Recommendation System v1.0
