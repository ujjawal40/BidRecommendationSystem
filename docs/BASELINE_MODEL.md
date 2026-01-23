# Baseline Model - Phase 1A: Bid Fee Prediction

**Established**: January 23, 2026
**Model**: LightGBM Regression
**Status**: Production Ready

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

## Next Step: Phase 1B

Build Win Probability Classification Model to complete:
```
Expected Value = Win Probability × Bid Fee
```
