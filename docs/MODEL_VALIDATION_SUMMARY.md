# Model Validation Summary
## Phase 1A: Bid Fee Prediction - Performance Analysis

**Date**: 2026-01-07
**Model**: LightGBM (Optuna Optimized)
**Status**: ⚠️ **OVERFITTING DETECTED - NOT PRODUCTION READY**

> **Update 2026-05-13**: A `## Phase 1B v3` section was appended to this
> file documenting the three v3 variants trained on 2026-05-13:
> v3 default (AUC 0.936), v3 leaky (AUC 0.9998 — archived), and the
> deployed v3_fee_sensitive variant (AUC 0.883 with a real ~24pp
> fee-elasticity curve). See bottom of file.

---

## Executive Summary

The model shows **significant overfitting** despite data leakage fixes and hyperparameter optimization. While test set R² = 0.98 appears strong, the model performs 4.3x worse on test data compared to training data, indicating it has memorized training patterns rather than learned generalizable relationships.

**Critical Issues**:
- Train RMSE: $70.26 vs Test RMSE: $305.13 (4.3x ratio)
- Performance degradation over time (RMSE CV: 70%)
- Model not production-ready in current state

---

## 1. How Are Results Looking?

### Test Set Performance (Last 20% of Data)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | $305.13 | Average prediction error of $305 |
| **MAE** | $90.40 | Typical error is $90 (2.7% of mean bid) |
| **R²** | 0.9800 | Explains 98% of variance |
| **Median Error** | $26.50 | Half of predictions within ±$26.50 |

**On the surface**, these look good. However, this masks serious underlying issues.

---

## 2. How Do We Know It's Not Overfitting?

### ❌ **WE DON'T - MODEL IS OVERFITTING SIGNIFICANTLY**

### Overfitting Indicators

| Metric | Train | Test | Ratio/Diff | Status |
|--------|-------|------|------------|--------|
| **RMSE** | $70.26 | $305.13 | **4.34x** | ✗ CRITICAL |
| **MAE** | $39.05 | $90.40 | **2.32x** | ✗ BAD |
| **R²** | 0.9989 | 0.9800 | **0.0189 diff** | ✓ OK |

### What This Means

**Train Performance (Too Good)**:
- RMSE of $70 on data with std dev of $2,131
- Only 3.3% error on training data
- R² = 0.9989 (99.89% of variance explained)
- **This is suspiciously perfect**

**Test Performance (Reality Check)**:
- RMSE jumps to $305 (4.3x worse)
- MAE doubles from $39 to $90
- Model learned training data patterns too well
- **Doesn't generalize to new data**

### Healthy Model Benchmarks

For a model that generalizes well, we expect:
- RMSE ratio (test/train): **< 1.3**
- MAE ratio (test/train): **< 1.3**
- R² difference: **< 0.05**

**Our model**: 4.34x, 2.32x, 0.0189 → **FAILS on RMSE/MAE ratios**

---

## 3. Train vs Test Results

### Dataset Split (Time-Based)

```
Train Set: 91,602 samples (80%)
  Period: Jan 2018 to Oct 2024
  Target mean: $3,363.41

Test Set: 22,901 samples (20%)
  Period: Oct 2024 to Dec 2025
  Target mean: $3,363.41
```

### Performance Comparison

```
TRAIN SET (80% - Jan 2018 to Oct 2024):
  RMSE: $70.26
  MAE:  $39.05
  R²:   0.9989

  ✓ Extremely accurate on training data
  ✗ TOO accurate - sign of overfitting

TEST SET (20% - Oct 2024 to Dec 2025):
  RMSE: $305.13
  MAE:  $90.40
  R²:   0.9800

  ✓ Still decent R² score
  ✗ 4.3x worse error than training
  ✗ Indicates poor generalization
```

### Visualization of Train vs Test Gap

```
Train Error: [$70]━━━━━
Test Error:  [$305]━━━━━━━━━━━━━━━━━━━━━
             ↑
             4.3x increase
```

---

## 4. What Metrics Are We Evaluating?

### Primary Metrics

1. **RMSE (Root Mean Squared Error)** - $305.13
   - Penalizes large errors heavily
   - In same units as target ($)
   - Most important metric for this use case
   - **Current**: Prediction errors average $305

2. **MAE (Mean Absolute Error)** - $90.40
   - Average absolute prediction error
   - More interpretable than RMSE
   - Less sensitive to outliers
   - **Current**: Typical prediction off by $90 (2.7% of mean)

3. **R² (R-squared)** - 0.9800
   - Proportion of variance explained
   - Range: 0 to 1 (higher is better)
   - **Current**: Explains 98% of variance
   - **Warning**: Can be misleading with overfitting

### Secondary Metrics

4. **Median Absolute Error** - $26.50
   - 50th percentile of errors
   - Robust to outliers
   - **Current**: Half of predictions within ±$26.50

5. **RMSE Ratio (Test/Train)** - 4.34
   - **Overfitting indicator**
   - Healthy: < 1.3
   - **Current**: 4.34 → SEVERE OVERFITTING

6. **RMSE Coefficient of Variation** - 70.47%
   - **Stability indicator**
   - Measures consistency across time periods
   - Healthy: < 15%
   - **Current**: 70.47% → UNSTABLE

### Residual Analysis

```
Mean Residual: -$8.56 (-0.25% of target)
  ✓ NO SYSTEMATIC BIAS
  Model doesn't consistently over/under predict

Residual Std Dev: $305.01
  Typical prediction error magnitude
  Matches RMSE as expected
```

---

## 5. Did We Do Backtesting?

### ✅ YES - Walk-Forward Backtesting Performed

We implemented **5-fold time series cross-validation** to simulate production deployment.

### Backtesting Methodology

**Approach**: Walk-Forward Expanding Window
- Simulates real-world scenario where model is trained once, then used over time
- Each fold uses an expanding training window
- Tests on future (unseen) data

**Configuration**:
- 5 time-based folds
- ~19,000 samples per test fold
- Preserves temporal ordering
- No data leakage between folds

### Backtesting Results

| Fold | Period | Samples | RMSE | MAE | R² |
|------|--------|---------|------|-----|-----|
| 1 | Jan 2020 - Sep 2021 | 19,083 | $65.06 | $36.45 | 0.9991 |
| 2 | Sep 2021 - Sep 2022 | 19,083 | $68.99 | $40.99 | 0.9989 |
| 3 | Sep 2022 - Nov 2023 | 19,083 | $76.96 | $40.85 | 0.9986 |
| 4 | Nov 2023 - Dec 2024 | 19,083 | $140.82 | $50.24 | 0.9959 |
| 5 | Dec 2024 - Dec 2025 | 19,083 | **$311.15** | **$91.82** | **0.9794** |

**Aggregate**:
- Mean RMSE: $132.60 ± $93.44
- Mean MAE: $52.07 ± $20.38
- Mean R²: 0.9944 ± 0.0076

### Critical Findings from Backtesting

#### ❌ **Performance Degrades Over Time**

```
Fold 1 (2020-2021): RMSE = $65   ████
Fold 2 (2021-2022): RMSE = $69   ████
Fold 3 (2022-2023): RMSE = $77   █████
Fold 4 (2023-2024): RMSE = $141  ████████████
Fold 5 (2024-2025): RMSE = $311  ████████████████████████████
                                 ↑
                            5x degradation!
```

**This is a red flag** indicating:
1. Model performance deteriorates on more recent data
2. Market dynamics may be changing
3. Training data from 2018-2022 doesn't represent 2024-2025 patterns
4. Model is not stable for production use

#### ❌ **High Variability (RMSE CV = 70%)**

- **RMSE Coefficient of Variation**: 70.47%
- **Interpretation**: RMSE varies by 70% across time periods
- **Healthy Threshold**: < 15%
- **Status**: UNSTABLE

**What this means**:
- Performance is unpredictable
- Error could be $65 or $311 depending on time period
- Cannot reliably forecast model accuracy
- Risk for business decisions

---

## 6. Why Is This Happening?

### Root Causes of Overfitting

1. **Model Complexity Too High**
   - 84 features (many may be redundant)
   - Model has too much capacity to memorize
   - Can fit noise rather than signal

2. **Temporal Distribution Shift**
   - Training data: 2018-2024
   - Test data: 2024-2025
   - Market conditions may have changed
   - COVID impact, economic shifts, etc.

3. **Insufficient Regularization**
   - Current reg_alpha: 0.697
   - Current reg_lambda: 0.106
   - May need stronger penalties

4. **Early Stopping at 999 iterations**
   - Model trained for almost full 1000 rounds
   - May have overfit to training data
   - Early stopping didn't trigger properly

---

## 7. What Should We Do?

### Recommended Fixes (Priority Order)

#### 🔴 **CRITICAL - Feature Selection**

**Problem**: 84 features, many likely redundant
**Solution**:
- Use SHAP to identify truly important features
- Remove features with < 0.1% importance
- Target: Reduce to 20-30 core features
- **Expected Impact**: 30-50% reduction in overfitting

#### 🔴 **CRITICAL - Stronger Regularization**

**Problem**: reg_alpha=0.697, reg_lambda=0.106 too weak
**Solution**:
- Increase L1 regularization: reg_alpha = 1.0 to 5.0
- Increase L2 regularization: reg_lambda = 1.0 to 5.0
- Add min_data_in_leaf constraint: 50-100
- **Expected Impact**: 20-40% reduction in overfitting

#### 🟡 **HIGH - Reduce Model Complexity**

**Problem**: num_leaves=62, max_depth=10 too complex
**Solution**:
- Reduce num_leaves: 20-30 (currently 62)
- Reduce max_depth: 5-7 (currently 10)
- Increase min_child_samples: 50-100 (currently 16)
- **Expected Impact**: 20-30% reduction in overfitting

#### 🟡 **HIGH - Ensemble with Simpler Models**

**Problem**: Single complex model overfits
**Solution**:
- Train 3-5 simpler models with different parameters
- Average predictions (ensemble)
- Each model: fewer features, higher regularization
- **Expected Impact**: 15-25% improvement in stability

#### 🟢 **MEDIUM - Rolling Window Training**

**Problem**: 2018 data may not represent 2025 patterns
**Solution**:
- Only use last 2-3 years of data for training
- Retrain model quarterly on recent data
- Discard oldest data
- **Expected Impact**: Better temporal stability

---

## 8. Performance Over Time Analysis

### Monthly Performance on Test Set

Test period: Oct 2024 to Dec 2025 (15 months)

```
Monthly MAE Statistics:
  Mean:   $90.24
  Min:    $78.62 (best month)
  Max:    $104.47 (worst month)
  Std:    $9.33
  Range:  $25.85
```

**Findings**:
- ✓ Relatively consistent month-to-month (±$10)
- ✓ No extreme outlier months
- ⚠ But overall performance worse than training

---

## 9. Business Impact Assessment

### Current Model Performance

**What the model can do**:
- Predict bid fees within ±$300 on average (test RMSE)
- Typical prediction within ±$90 (MAE)
- 50% of predictions within ±$27 (median)
- No systematic bias (doesn't consistently over/under predict)

**What the model cannot do reliably**:
- Generalize to new time periods (degrading performance)
- Maintain consistent accuracy (70% variation)
- Match training performance (4.3x worse on test)

### Risk Assessment

| Risk Factor | Level | Description |
|-------------|-------|-------------|
| Overfitting | **HIGH** | Model memorizes training data |
| Temporal Instability | **HIGH** | Performance degrades over time |
| Prediction Variance | **MEDIUM** | ±$90 error may impact bid decisions |
| Systematic Bias | **LOW** | No consistent over/under prediction |

### Production Readiness: ❌ **NOT READY**

**Recommendation**: DO NOT deploy current model to production

**Reasoning**:
1. Overfitting will lead to poor real-world performance
2. Unstable predictions create business risk
3. Performance gap (train vs test) indicates unreliable estimates
4. Model will likely perform worse than test metrics suggest

---

## 10. Next Steps

### Immediate Actions (This Week)

1. **Feature Selection Analysis**
   - Run SHAP importance analysis
   - Identify top 30 features
   - Retrain with reduced feature set

2. **Hyperparameter Re-optimization**
   - Focus on reducing overfitting
   - Stronger regularization
   - Simpler model architecture

3. **Validation Strategy Update**
   - Use only 2023-2025 data
   - More aggressive cross-validation

### Short-Term Actions (Next 2 Weeks)

4. **Ensemble Model Development**
   - Train 3-5 diverse models
   - Combine predictions
   - Improve stability

5. **Feature Engineering Review**
   - Remove correlated features
   - Simplify aggregation features
   - Focus on robust features

### Long-Term Strategy (Next Month)

6. **Phase 1B: Win Probability Model**
   - Build classification model
   - Combine with bid fee prediction
   - Optimize expected value

7. **Production Pipeline**
   - Model monitoring framework
   - Automated retraining schedule
   - Performance tracking dashboard

---

## 11. Conclusion

### Summary

The current LightGBM model shows **significant overfitting** despite:
- ✓ Data leakage fixes
- ✓ Hyperparameter optimization
- ✓ Time-based validation

**Key Metrics**:
- Test RMSE: $305.13 (4.3x worse than train)
- Test R²: 0.9800 (misleadingly high)
- RMSE CV: 70% (highly unstable)
- Production Ready: **NO**

### Positive Aspects

✓ No data leakage (after fixes)
✓ No systematic bias (-0.25%)
✓ Reasonable test R² (0.98)
✓ Proper time-based validation implemented
✓ Comprehensive backtesting performed

### Critical Issues

✗ Severe overfitting (4.3x train/test gap)
✗ Performance instability (70% CV)
✗ Temporal degradation (5x worse on recent data)
✗ Too many features (84)
✗ Insufficient regularization

### Recommendation

**DO NOT DEPLOY** - Implement feature selection and stronger regularization first.

Expected timeline to production-ready model: **1-2 weeks** with proper fixes.

---

**Generated**: 2026-01-07
**Model Version**: LightGBM Optimized Phase 1A
**Validation Method**: 5-Fold Walk-Forward Backtesting
**Dataset**: 114,503 bids (Jan 2018 - Dec 2025)

---

## Phase 1B v3 — Fee-Sensitive Win Probability (2026-05-13)

### Background
Phase 1B v2 (AUC 0.948) was found to be saturated at inference — raw output
~0.97 across realistic inputs, calibrator clipping to 0.95 ceiling. The EV
optimizer was always picking max fee because the win-probability curve was
flat. The hand-tuned heuristic sigmoid (in `enhanced_prediction_service.py`)
was a temporary fix.

v3 set out to fix this by replacing frequency-count features with
expanding-window LOO win-rate target encodings.

### Three Variants Trained

| Variant | Test AUC | Brier | Span at Inference | Status |
|---------|----------|-------|-------------------|--------|
| v3 default | 0.9359 | 0.097 | 0pp (still saturated) | not deployed |
| v3 leaky run | 0.9998 | 0.003 | 0pp | archived (target leakage via `land_acres_log`) |
| **v3_fee_sensitive** | **0.8830** | 0.140 | **~24pp training, 37pp inference canary** | **deployed as v2 saturation fallback** |

### Key Insight
LOO encoding alone was not enough to make the model fee-sensitive — even
with clean features, `subtype_avg_fee` and other context-fee aggregates
absorbed the signal. The `V3_DROP_CONTEXT_FEE` ablation removed 13 such
features, dropping AUC by 5.3pp but unlocking the fee curve.

### Deployment
v3_fee_sensitive does NOT replace v2 as the primary classifier (it would
hurt ranking accuracy). Instead, it replaces the hand-tuned heuristic
sigmoid in the saturation fallback path. When v2 raw output >= 0.93, the
API now calls v3_fee_sensitive instead of the heuristic.

### Files
- `outputs/models/lightgbm_win_probability_v3_fee_sensitive.txt`
- `outputs/models/lightgbm_win_probability_v3_fee_sensitive_metadata.json`
- `outputs/models/win_probability_v3_fee_sensitive_calibrator.pkl`
- `outputs/reports/win_rate_lookups_v3.json` (API-side category → win-rate)
- `outputs/reports/v3_api_integration_smoke_test.json`
- `outputs/reports/win_probability_v3_run{01,02,03,04_05}_results.json`
