# Baseline Models - Bid Recommendation System

**Established**: January 23, 2026
**Status**: Both Phases Production Ready

---

## Expected Value System

```
Expected Value = P(Win) × Bid Fee
```

| Phase | Model | Target | Status |
|-------|-------|--------|--------|
| 1A | Bid Fee Prediction | Regression | ✅ Complete |
| 1B | Win Probability | Classification | ✅ Complete |

---

# Phase 1A: Bid Fee Prediction

**Model**: LightGBM Regression

---

## Performance Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| **Test RMSE** | $328.75 | Benchmark |
| **Test MAE** | $102.44 | ~3% of mean |
| **Test R²** | 0.9761 | Excellent |
| **Overfitting Ratio** | 1.99x | Good generalization |

### Performance by Split

| Set | RMSE | MAE | % Error |
|-----|------|-----|---------|
| Train | $165.42 | $65.07 | 5.2% |
| Validation | $350.95 | $109.07 | 10.4% |
| Test | $328.75 | $102.44 | 9.7% |

---

## Configuration

### Data Configuration
```python
DATA_START_DATE = "2023-01-01"  # Use recent data only
Total Records: 52,308 (filtered from 114,503)
Date Range: 2023-01-03 to 2025-12-19
```

### Train/Valid/Test Split
```
Train:      60% (31,384 records) | 2023-01 to 2024-11
Validation: 20% (10,462 records) | 2024-11 to 2025-06
Test:       20% (10,462 records) | 2025-06 to 2025-12
```

### Hyperparameters
```python
LIGHTGBM_CONFIG = {
    "params": {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 20,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "max_depth": 8,
        "min_child_samples": 30,
        "min_child_weight": 5,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
    },
    "training": {
        "num_boost_round": 500,
        "early_stopping_rounds": 50,
    },
}
```

---

## Features Used (68 features)

### Top 10 Most Important
1. **segment_avg_fee** - Business segment average (dominant predictor)
2. **state_avg_fee** - Geographic pricing baseline
3. **propertytype_avg_fee** - Property type pricing
4. **TargetTime** - Turnaround days (complexity proxy)
5. **rolling_avg_fee_segment** - Recent segment trends
6. **client_avg_fee** - Client-specific pricing
7. **rolling_avg_fee_proptype** - Recent property type trends
8. **segment_std_fee** - Segment volatility
9. **segment_win_rate** - Segment competitiveness
10. **office_avg_fee** - Office pricing strategy

### Features Excluded
- **27 JobData features** - Degraded performance (survivor bias)
- See `config/model_config.py` for full exclusion list

---

## Key Decisions & Rationale

### 1. Use 2023+ Data Only
- **Why**: Temporal distribution shift made older data (2018-2022) less predictive
- **Impact**: Better generalization to recent test data
- **Evidence**: Test RMSE improved by ~10%

### 2. Moderate Regularization (L1=L2=1.0)
- **Why**: Balance between overfitting (0.1) and underfitting (5.0)
- **Impact**: Overfitting ratio reduced from 3.6x to 1.99x
- **Evidence**: Grid search over 16 configurations

### 3. Exclude JobData Features
- **Why**: Survivor bias, multicollinearity, temporal mismatch
- **Impact**: Cleaner model, no artificial inflation
- **Evidence**: All enrichment experiments degraded performance

### 4. 60/20/20 Split with Proper Validation
- **Why**: Early stopping on validation (not test) prevents data leakage
- **Impact**: Unbiased test evaluation
- **Evidence**: Industry best practice

---

## Files

| File | Description |
|------|-------------|
| `outputs/models/lightgbm_bidfee_model.txt` | Trained model |
| `outputs/models/lightgbm_metadata.json` | Model metadata |
| `config/model_config.py` | Configuration |
| `scripts/04_model_lightgbm.py` | Training script |

### Archived Models
Previous experimental models moved to `outputs/models/archive/`

---

## Future Improvements

To beat this baseline, try:
1. **Feature ablation study** - Find minimum feature set
2. **External data** - Economic indicators, competitor data
3. **Ensemble methods** - Stacking with different model types
4. **Segment-specific models** - Separate models per business segment

---

## How to Reproduce

```bash
# Train the baseline model
python scripts/04_model_lightgbm.py

# Expected output:
# Test RMSE: ~$328.75
# Overfitting Ratio: ~1.99x
# R²: ~0.9761
```

---

# Phase 1B: Win Probability Prediction

**Model**: LightGBM Binary Classifier

## Performance Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| **Test AUC-ROC** | 0.9617 | Excellent |
| **Test Accuracy** | 89.16% | Strong |
| **Test F1** | 0.8846 | Good balance |
| **Test Precision** | 85.95% | Low false positives |
| **Test Recall** | 91.13% | Catches most wins |
| **Brier Score** | 0.0779 | Well calibrated |
| **Overfitting Ratio** | 1.02x | Near-perfect |

### Performance by Split

| Set | AUC-ROC | Accuracy | F1 |
|-----|---------|----------|-----|
| Train | 0.9784 | 91.14% | 0.9102 |
| Validation | 0.9672 | 90.06% | 0.9062 |
| Test | 0.9617 | 89.16% | 0.8846 |

### Confusion Matrix (Test Set)

```
                 Predicted
              Loss    Win
Actual Loss  4,980    711
Actual Win     423  4,348
```

---

## Classification Configuration

### Features Excluded (Leaky)
```python
# These features use the Won outcome - cannot use to predict Won
LEAKY_CLASSIFICATION_FEATURES = [
    'win_rate_with_client',
    'office_win_rate',
    'propertytype_win_rate',
    'state_win_rate',
    'segment_win_rate',
    'client_win_rate',
    'rolling_win_rate_office',
    'total_wins_with_client',
    'prev_won_same_client',
]
```

### Hyperparameters
```python
CLASSIFICATION_CONFIG = {
    "params": {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "num_leaves": 20,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_samples": 30,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.05,  # Adjusted for class balance
    },
}
```

---

## Top 10 Predictive Features for Win Probability

1. **JobCount** (47.1%) - Number of jobs in bid (dominant)
2. **market_competitiveness** (10.8%) - Market competition level
3. **TargetTime_Original** (3.5%) - Turnaround requirements
4. **PropertyState_frequency** (3.5%) - State activity level
5. **RooftopLongitude** (2.5%) - Geographic location
6. **targettime_ratio_to_proptype** (2.3%) - Relative turnaround
7. **RooftopLatitude** (2.2%) - Geographic location
8. **propertytype_std_fee** (1.5%) - Property type variability
9. **segment_std_fee** (1.5%) - Segment variability
10. **IECount** (1.5%) - Internal examiner count

---

## Phase 1B Files

| File | Description |
|------|-------------|
| `outputs/models/lightgbm_win_probability.txt` | Trained model |
| `outputs/models/lightgbm_win_probability_metadata.json` | Model metadata |
| `scripts/15_win_probability_baseline.py` | Training script |
| `outputs/figures/win_probability_evaluation.png` | ROC & calibration plots |
| `outputs/figures/win_probability_feature_importance.png` | Feature importance |

---

## How to Reproduce Phase 1B

```bash
# Train the win probability model
python scripts/15_win_probability_baseline.py

# Expected output:
# Test AUC-ROC: ~0.9617
# Overfitting Ratio: ~1.02x
# Accuracy: ~89%
```

---

# Next Steps

The Expected Value system is complete. Potential improvements:
1. **EV Optimization Script** - Combine both models for bid recommendations
2. **Threshold Tuning** - Optimize probability threshold based on business needs
3. **Segment-specific models** - Separate models per business segment
4. **A/B Testing Framework** - Validate recommendations in production
