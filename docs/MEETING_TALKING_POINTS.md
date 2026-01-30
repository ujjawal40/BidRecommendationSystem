# Bid Recommendation System - Meeting Talking Points
**Date**: January 2026
**Prepared for**: Supervisor Meeting

---

## 📊 EXECUTIVE SUMMARY

### Project Objective
Build a **Bid Recommendation System** for commercial real estate (CRE) appraisals that maximizes:

```
Expected Value = Win Probability × Bid Fee
```

### Current Status
| Phase | Task | Status | Performance |
|-------|------|--------|-------------|
| **Phase 1A** | Bid Fee Prediction | ✅ Complete | RMSE: $329, R²: 0.976 |
| **Phase 1B** | Win Probability | 🔜 Next | AUC: ~0.85 (preliminary) |

### Key Achievement
**Reduced overfitting from 3.6x to 1.99x** while maintaining prediction accuracy.

---

## 1️⃣ DATASET OVERVIEW

### Primary Data: BidData
| Attribute | Value |
|-----------|-------|
| **Total Records** | 114,503 bids |
| **Date Range** | January 2018 - December 2025 (8 years) |
| **Win Rate** | 43.34% |
| **Average Bid Fee** | $3,363 (Std: $2,131) |
| **Geographic Coverage** | 50 US states |

### Key Columns
- **Target Variable**: `BidFee` (what we predict)
- **Critical Driver**: `TargetTime` (turnaround days)
- **Categories**: Property Type, Business Segment, Office, State, Client

### Secondary Data: JobData (532K records)
- **Purpose**: Attempted enrichment with completed job data
- **Result**: ❌ **DEGRADED PERFORMANCE** (explained below)

---

## 2️⃣ DATA CLEANING & PREPROCESSING

### Issues Found & Fixed

| Issue | Count | Solution |
|-------|-------|----------|
| **Duplicate BidIds** | 41,592 (25%) | Aggregated Master/SubJob hierarchy |
| **Missing BidFee** | 3,072 (1.85%) | Removed rows |
| **Missing TargetTime** | 22% | Imputed with group medians |
| **Extreme Outliers** | 1% | Capped at 99th percentile ($15,000) |

### Cleaning Results
- **Before**: 166,372 records
- **After**: 114,503 records (68.8% retention)
- **Data Quality**: No missing values in final features

📊 **Chart**: `bidfee_distribution.png` - Shows target variable distribution before/after cleaning

---

## 3️⃣ FEATURE ENGINEERING

### Total Features Created: 58 engineered + 69 original = 127 total

### Feature Categories

| Category | Count | Examples |
|----------|-------|----------|
| **Rolling Time Series** | 7 | rolling_avg_fee_office, rolling_avg_fee_segment |
| **Client History** | 8 | prev_fee_same_client, win_rate_with_client |
| **Aggregations** | 15 | office_avg_fee, segment_avg_fee, state_avg_fee |
| **Interactions** | 4 | targettime_x_size, market_competitiveness |
| **Categorical Encodings** | 14 | Frequency & label encoding |
| **Temporal** | 10 | Year, Month_sin/cos, is_peak_season |

### Critical Fix: Data Leakage
- **Problem**: Aggregations included current row (cheating!)
- **Solution**: Changed to **leave-one-out** calculation
- **Impact**: R² dropped from 0.9996 to 0.98 (now honest)

📊 **Chart**: `correlation_analysis.png` - Feature correlation heatmap

---

## 4️⃣ WHY JOBDATA ENRICHMENT FAILED

### The Experiment
Added 22 features from 532K completed job records:
- office_avg_job_fee, office_job_volume, region aggregates, etc.

### Results

| Configuration | Test RMSE | Change |
|--------------|-----------|--------|
| Baseline (no JobData) | $237 | — |
| + Full JobData (22 features) | $297 | **-25% worse** |
| + Selective JobData (10 best) | $301 | **-27% worse** |

### Root Causes (Technical)

1. **Survivor Bias**
   ```
   JobData = ONLY winning bids (survivors)
   BidData = ALL bids (won + lost)
   ```
   Learning from winners doesn't predict all attempts.

2. **Multicollinearity**
   - `office_avg_job_fee` correlated 0.71 with `office_avg_fee`
   - Redundant signal = noise, not information

3. **Temporal Mismatch**
   - JobData: 2001-2026 (68% before BidData started)
   - Stale patterns don't apply to current market

4. **Feature Importance Evidence**
   - JobData features ranked **outside top 15**
   - Model itself determined they're not useful

### Decision
**Excluded all 27 JobData features** from final model.

---

## 5️⃣ HANDLING OVERFITTING

### The Problem
| Metric | Before Fix | After Fix |
|--------|------------|-----------|
| Train RMSE | $82 | $165 |
| Test RMSE | $384 | $329 |
| **Overfitting Ratio** | **3.6x** | **1.99x** |

### Solutions Applied

1. **Data Filtering**: Train on 2023+ only (removed stale 2018-2022 patterns)
2. **Regularization**: Increased L1/L2 from 0.1 to 1.0 (10x)
3. **Tree Complexity**: Reduced num_leaves 31→20, added max_depth=8
4. **Proper Validation**: 60/20/20 split (train/valid/test)
5. **Early Stopping**: On validation set, NOT test set

📊 **Chart**: `model_performance_over_time.png` - Shows temporal degradation

---

## 6️⃣ MODEL DETAILS

### Algorithm: LightGBM (Gradient Boosted Decision Trees)

### Why LightGBM?
- Handles categorical features well
- Fast training
- Built-in regularization
- Interpretable feature importance

### Final Hyperparameters
```python
{
    "num_leaves": 20,        # Simpler trees
    "max_depth": 8,          # Capped complexity
    "learning_rate": 0.05,
    "reg_alpha": 1.0,        # L1 regularization
    "reg_lambda": 1.0,       # L2 regularization
    "min_child_samples": 30, # Robust splits
    "num_boost_round": 500,
    "early_stopping": 50,
}
```

### Training Configuration
- **Data**: 2023+ only (52,308 records)
- **Features**: 68 (optimized from 84)
- **Split**: 60% train / 20% valid / 20% test

---

## 7️⃣ MODEL PERFORMANCE METRICS

### Final Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Test RMSE** | $328.75 | Average prediction error |
| **Test MAE** | $102.44 | Typical error |
| **Test R²** | 0.9761 | Explains 97.6% of variance |
| **Median AE** | $31.80 | Half of predictions within ±$32 |
| **Overfitting Ratio** | 1.99x | ✅ Good generalization |

### Metrics Explained

- **RMSE** (Root Mean Squared Error): Penalizes large errors more heavily
- **MAE** (Mean Absolute Error): Average magnitude of errors
- **R²** (R-squared): Proportion of variance explained (1.0 = perfect)
- **Median AE**: Robust to outliers, shows typical error
- **Overfitting Ratio**: Test/Train RMSE (target: <2.0x)

### Prediction Accuracy
- Within ±$100: ~45% of predictions
- Within ±$250: ~70% of predictions
- Within ±$500: ~90% of predictions

---

## 8️⃣ FEATURE IMPORTANCE (TOP 10)

| Rank | Feature | Importance | Business Meaning |
|------|---------|------------|------------------|
| 1 | **segment_avg_fee** | 63.8% | Business segment pricing |
| 2 | state_avg_fee | 8.5% | Geographic pricing |
| 3 | propertytype_avg_fee | 3.8% | Property type pricing |
| 4 | TargetTime | 3.0% | Turnaround complexity |
| 5 | rolling_avg_fee_segment | 2.2% | Recent segment trends |
| 6 | client_avg_fee | 1.9% | Client-specific pricing |
| 7 | rolling_avg_fee_proptype | 1.5% | Recent property trends |
| 8 | segment_std_fee | 1.4% | Segment price volatility |
| 9 | segment_win_rate | 1.4% | Segment competitiveness |
| 10 | office_avg_fee | 0.7% | Office pricing strategy |

### Key Insight
**Business Segment is the dominant predictor** (6 of top 10 features are segment-related).

📊 **Charts**:
- `lightgbm_feature_importance.png` - Bar chart of top features
- `lightgbm_shap_summary.png` - SHAP values showing feature impact

---

## 9️⃣ CHARTS & VISUALIZATIONS

### Available Charts (18 total)

| Chart | Purpose | Key Talking Point |
|-------|---------|-------------------|
| `bidfee_distribution.png` | Target distribution | "BidFee ranges $500-$15K, median $3,000" |
| `bidfee_comprehensive_analysis.png` | Complete EDA | "Shows outliers, trends, segments" |
| `correlation_analysis.png` | Feature correlations | "Identified multicollinearity issues" |
| `categorical_features_analysis.png` | Category breakdowns | "Multifamily 22%, Retail 21%" |
| `geographic_analysis.png` | State analysis | "IL 25%, FL 14%, TX 7%" |
| `targettime_analysis.png` | Critical driver | "22% missing, imputed carefully" |
| `time_series_analysis.png` | Temporal trends | "2024-2025 market shift detected" |
| `lightgbm_feature_importance.png` | Feature ranking | "segment_avg_fee dominates at 63.8%" |
| `lightgbm_shap_summary.png` | SHAP explainability | "Shows how features affect predictions" |
| `model_residual_analysis.png` | Error analysis | "No systematic bias detected" |
| `model_performance_over_time.png` | Temporal performance | "5x degradation on recent data fixed" |
| `optuna_optimization_history.png` | Hyperparameter tuning | "Tried 100+ configurations" |
| `office_fixed_effects.png` | Office pricing | "$10K spread between offices" |
| `win_probability_analysis.png` | Win rate patterns | "43% overall win rate" |

---

## 🔟 API & PRODUCTION READINESS

### Current State
- ✅ Model trained and serialized (`lightgbm_bidfee_model.txt`)
- ✅ Prediction demo script (`scripts/33_predict_demo.py`)
- ❌ No REST API yet (FastAPI recommended)
- ❌ No real-time serving infrastructure

### Demo Command
```bash
python scripts/33_predict_demo.py
```

### Sample Prediction Output
```
Commercial Office in Illinois:
  Input: segment_avg=$4500, state_avg=$3800, target_time=45 days
  Output: Recommended Bid = $3,847
  Confidence: Medium
```

### Production Roadmap
1. Create FastAPI endpoint
2. Add input validation
3. Implement model versioning
4. Set up monitoring/logging
5. Deploy to cloud (AWS/GCP)

---

## 📈 SUMMARY: WHAT WE ACHIEVED

| Goal | Status | Evidence |
|------|--------|----------|
| Predict bid fees | ✅ | RMSE $329, R² 0.976 |
| Handle overfitting | ✅ | Ratio: 3.6x → 1.99x |
| Fix data leakage | ✅ | R²: 0.9996 → 0.98 (honest) |
| Feature engineering | ✅ | 58 new features created |
| JobData analysis | ✅ | Proved it degrades performance |
| Model explainability | ✅ | SHAP analysis complete |

---

## ❓ ANTICIPATED QUESTIONS

### Q: Why not use neural networks?
**A**: LightGBM outperformed PyTorch NN in our experiments. Gradient boosting is better for tabular data with mixed feature types.

### Q: Can we improve further?
**A**: Yes, with:
- More data (competitor bids, property details)
- Phase 1B win probability model
- Ensemble methods

### Q: Is it production-ready?
**A**: Model is ready. Need API wrapper and monitoring infrastructure.

### Q: Why did JobData fail?
**A**: Survivor bias - JobData only contains winning bids, creating biased estimates that don't generalize to all bid attempts.

---

## 🎯 NEXT STEPS

1. **Short-term**: Deploy prediction API
2. **Medium-term**: Build Phase 1B (Win Probability)
3. **Long-term**: Complete Expected Value optimization system

---

*Document prepared by Bid Recommendation System Team*
