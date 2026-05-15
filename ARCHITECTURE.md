# Architecture

## Business Problem

Commercial real estate appraisal firms decide what fee to bid on each engagement.
Bid too high → lose the job. Bid too low → win but leave revenue on the table.
This system solves that with **Expected Value optimization** using two
independent ML models:

- **Phase 1A (Regression)**: Predicts the optimal bid fee given market context — "What should we charge?"
- **Phase 1B (Classification)**: Predicts P(Win) at that fee — "Will we win at this price?"
- **Combined**: `EV = P(Win) × BidFee` — "What price maximizes expected revenue?"

A $5,200 bid with 25% win chance (EV=$1,300) is worse than a $3,500 bid with 65% win chance (EV=$2,275). The system recommends the latter.

## System Layers

```
┌──────────────────────────────────────────────────────────────────┐
│  React Frontend (frontend/)           Vercel deployment          │
│  └─ BidForm → POST /api/v2/predict → ResultDisplay              │
├──────────────────────────────────────────────────────────────────┤
│  Flask API (api/)                     Render deployment          │
│  └─ app.py → EnhancedBidPredictor                               │
│       ├─ _generate_features(): 6 raw inputs → 60+ features      │
│       ├─ LightGBM Regressor (log1p target) → expm1 → fee       │
│       │     └─ Benchmark-aware floor: max($500, 30% × seg_avg) │
│       ├─ Inject predicted fee as BidFee feature ──┐             │
│       ├─ LightGBM Classifier v2 → P(Win)         │              │
│       │     └─ Saturation check → fallback to v3 │              │
│       └─ EV = P(Win) × Fee → recommendation                    │
├──────────────────────────────────────────────────────────────────┤
│  ML Pipeline (scripts/)              Offline training            │
│  └─ raw → preprocess → feature engineer → train → validate      │
├──────────────────────────────────────────────────────────────────┤
│  Config (config/model_config.py)     Centralized hyperparams     │
└──────────────────────────────────────────────────────────────────┘
```

## Model Versions

| Version | Phase | Test AUC / R² | Status |
|---------|-------|---------------|--------|
| v1 Bid Fee | 1A regression | R² 0.938, MAPE 15.7% | superseded |
| v2 Bid Fee | 1A regression | R² 0.940, MAPE 3.9%, overfit 1.49× | **production** |
| v1 Win Prob | 1B classification | AUC 0.870 | superseded |
| v2 Win Prob | 1B classification | AUC 0.948, Brier 0.093 | **production (primary)** |
| v3 Win Prob | 1B classification | AUC 0.936 (saturated) | not deployed |
| v3 Fee-Sensitive | 1B classification | AUC 0.883, 24pp fee elasticity | **production (saturation fallback)** |

v2 produces a constant probability across the fee sweep (saturated at inference). v3 fee-sensitive replaces the hand-tuned heuristic sigmoid that previously patched this — see `docs/V3_DEPLOYMENT_NOTES.md`.

## Training Pipeline

```bash
# v1 (BidData only)
python scripts/02_data_preprocessing.py       # Raw → cleaned
python scripts/03_feature_engineering.py      # Cleaned → 68 features
python scripts/04_model_lightgbm.py           # Phase 1A: Bid Fee
python scripts/15_win_probability_baseline.py # Phase 1B: Win Probability

# v2 (JobsData + enriched BidData)
python scripts/21_jobsdata_preprocessing.py   # 215K-row JobsData → processed
python scripts/22_combined_data_preprocessing.py
python scripts/23_enhanced_feature_engineering.py
python scripts/24_enhanced_bidfee_model.py
python scripts/25_enhanced_win_probability.py
python scripts/27_generate_api_stats_v2.py    # Runtime stats for API

# v3 fee-sensitive win prob (new — saturation fallback)
V3_DROP_CONTEXT_FEE=1 python scripts/29_fee_sensitive_win_prob.py
```

## Running Locally

### Backend
```bash
pip install -r requirements.txt          # Full dev dependencies
pip install -r requirements-api.txt      # Minimal API-only
cd api && python app.py                  # localhost:5001
```

### Frontend
```bash
cd frontend && npm install && npm start  # localhost:3000
```

### Tests
```bash
pytest tests/test_validation.py -v       # 41 tests, no data needed
```

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| v2 trained on 2018+ data | Broader training data from JobsData (215K rows vs v1's 23K) |
| v1 trained on 2023+ only | Market regime shift; older BidData hurts generalization |
| Time-based 60/20/20 splits | Prevents temporal leakage |
| Leave-one-out aggregates | Prevents target leakage in segment/state averages |
| LightGBM `.txt` format | Portable, version-controllable (not pickle) |
| Log-transform on BidFee | MAPE 71%→16% improvement, better proportional accuracy |
| BidFee as win prob feature | Model learns fee-sensitivity from data, no post-hoc adjustment needed |
| Differential-imputation Jobs_* features excluded from win prob | Won bids get real JobsData values, lost get medians — model trivially distinguishes (data leakage) |

## Critical Conventions

- **Feature order must match** training metadata JSON exactly — silent failures otherwise
- **Win probability clipped** to [0.05, 0.95] — never report 0% or 100%
- **Win probability includes BidFee** as a feature — Phase 1A's output is injected before Phase 1B runs
- **State lookups use full names** ("Illinois" not "IL")
- **Benchmark-aware floor**: `max($500, 30% × segment_avg)` catches log-transform underprediction for rare segments
- **Precomputed stats** in `outputs/reports/` are runtime dependencies — must exist for the API to function

## File Layout

```
api/                         Flask service
  app.py                     Routes, CORS, health endpoints
  enhanced_prediction_service.py   v2 + v3 prediction logic
  empirical_bands.py         Confidence interval calculator
scripts/                     Training pipeline (numbered sequentially)
  02_data_preprocessing.py
  03_feature_engineering.py
  04_model_lightgbm.py       v1 Phase 1A
  15_win_probability_baseline.py    v1 Phase 1B
  21-27_*.py                 v2 pipeline
  29_fee_sensitive_win_prob.py      v3 win prob
  29b_v2_vs_v3_curve_comparison.py
frontend/src/                React app
config/model_config.py       Centralized hyperparams + paths
tests/test_validation.py     41 tests
outputs/
  models/                    LightGBM .txt boosters + metadata JSON
  reports/                   Precomputed stats, lookups, results JSON
  figures/                   Plots
docs/                        Full documentation
  V3_DEPLOYMENT_NOTES.md     v3 runbook
  MODEL_VALIDATION_SUMMARY.md
  DATA_LEAKAGE_ANALYSIS.md
  FEATURE_ENGINEERING_DOCUMENTATION.md
  DEEP_TECHNICAL_MEETING_NOTES.md
MODEL_CARD.md                Model specifications, limitations
README.md                    Quickstart
```

## Deployment

- **Frontend**: Vercel, auto-deploys `/frontend`. Env: `REACT_APP_API_URL`
- **Backend**: Render via `render.yaml`. `gunicorn api.app:app`, 120s timeout, health at `/api/health`
- **Python**: 3.11.0 (`.python-version` + `render.yaml`)

## CI

GitHub Actions (`.github/workflows/ci.yml`): Python 3.10+3.11, flake8 lint,
pytest, black/isort checks (non-blocking), `safety` security scan. Triggers
on push to `main`/`develop` and PRs to `main`.

## Full Documentation Map

| File | When to consult |
|------|-----------------|
| `MODEL_CARD.md` | Model specifications, intended use, limitations |
| `docs/V3_DEPLOYMENT_NOTES.md` | v3 fee-sensitive model runbook |
| `docs/USER_GUIDE.md` | End-user documentation |
| `docs/DEEP_TECHNICAL_MEETING_NOTES.md` | Why JobData was excluded, business context |
| `docs/FEATURE_ENGINEERING_DOCUMENTATION.md` | Feature creation strategy, leave-one-out methodology |
| `docs/DATA_LEAKAGE_ANALYSIS.md` | Leakage risks, time-based split rationale |
| `docs/MODEL_VALIDATION_SUMMARY.md` | Validation results, overfitting analysis, backtesting |
| `docs/FEATURE_SELECTION_RESULTS.md` | SHAP-based feature reduction results |
