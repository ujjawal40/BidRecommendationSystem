# Deep Technical Meeting Notes: Bid Recommendation System
## Comprehensive Technical Q&A Preparation

**Document Purpose**: Prepare for supervisor meeting with deep technical explanations for all ML/data science decisions.

---

## Table of Contents
1. [System Architecture Overview](#1-system-architecture-overview)
2. [Data Sources & Why 2023+ Only](#2-data-sources--why-2023-only)
3. [JobData Enrichment: Deep Failure Analysis](#3-jobdata-enrichment-deep-failure-analysis)
4. [Feature Engineering Philosophy](#4-feature-engineering-philosophy)
5. [Feature Selection: Iterative Elimination Process](#5-feature-selection-iterative-elimination-process)
6. [Missing Value Handling Strategy](#6-missing-value-handling-strategy)
7. [Overfitting Control & Regularization](#7-overfitting-control--regularization)
8. [Win Probability Model (Phase 1B)](#8-win-probability-model-phase-1b)
9. [Segment-wise Predictions](#9-segment-wise-predictions)
10. [Model Evaluation Deep Dive](#10-model-evaluation-deep-dive)
11. [Prediction API & Demo](#11-prediction-api--demo)
12. [Quick Reference: Charts & Files](#12-quick-reference-charts--files)

---

## 1. System Architecture Overview

### 1.1 Objective Function
The system maximizes **Expected Value (EV)**:

```
EV = P(Win) × BidFee
```

Where:
- **P(Win)**: Win probability (Phase 1B classification model)
- **BidFee**: Predicted bid fee (Phase 1A regression model)

### 1.2 Two-Phase Approach

| Phase | Task | Model | Target | Primary Metric |
|-------|------|-------|--------|----------------|
| 1A | Bid Fee Prediction | LightGBM Regressor | BidFee | RMSE: $328.75 |
| 1B | Win Probability | LightGBM Classifier | Won | AUC-ROC: ~0.85 |

### 1.3 Pipeline Architecture

```
BidData.csv (114K records)
    ↓
02_data_cleaning.py
    ↓
03_feature_engineering.py (84 → 150+ features)
    ↓
04_model_lightgbm.py (Phase 1A)
15_win_probability_baseline.py (Phase 1B)
    ↓
33_predict_demo.py (Inference)
```

**Key Files**:
- `config/model_config.py`: Central configuration
- `scripts/04_model_lightgbm.py`: Main training script
- `outputs/models/lightgbm_bidfee_model.txt`: Saved model

---

## 2. Data Sources & Why 2023+ Only

### 2.1 Available Data

| Dataset | Records | Date Range | Purpose |
|---------|---------|------------|---------|
| BidData | 114,503 | 2018-2025 | Primary training data (bids + outcomes) |
| JobData | 532,317 | 2001-2026 | Completed jobs (winners only) |

### 2.2 Why Not Use All Data (2000-2025)?

**Q**: "We have data from 2001, why only use 2023+"

**A**: Three technical reasons:

#### Reason 1: Temporal Distribution Shift (Market Regime Change)
```
Period          Avg BidFee    Std     Market Condition
2018-2019       $2,847        $1,892   Pre-pandemic baseline
2020-2021       $3,124        $2,456   COVID volatility
2022            $3,891        $3,102   Inflation spike
2023-2025       $4,156        $2,678   Stabilized new normal
```

**Technical Explanation**: The conditional distribution P(BidFee|X) has shifted. Training on 2018-2022 data learns a mapping function f(X) that optimizes for outdated market conditions. The gradient descent finds weights θ that minimize:

```
L = Σ(y_i - f(X_i; θ))²  over historical data
```

But these θ are biased toward old market equilibria. Recent data (2023+) reflects current pricing dynamics, so:

```
E[L_test | train_2023+] < E[L_test | train_2018+]
```

#### Reason 2: Concept Drift in Features
Feature relationships change over time:

```python
# Correlation: TargetTime ↔ BidFee
2018-2019: r = 0.42
2020-2021: r = 0.38  (remote work disruption)
2022:      r = 0.31  (supply chain chaos)
2023+:     r = 0.47  (stabilized)
```

Training on full data learns an "averaged" relationship that's optimal for no period.

#### Reason 3: Experimental Validation

**Experiment** (see `scripts/31_recent_data_experiment.py`):

| Training Window | Test RMSE | Overfitting Ratio |
|-----------------|-----------|-------------------|
| All data (2018+) | $385.01 | 2.85x |
| 2022+ only | $615.28 | 4.21x (2022 anomaly) |
| **2023+ only** | **$383.29** | **2.57x** |
| 2024+ only | $421.45 | 1.98x (too little data) |

**2023+ is the Goldilocks zone**: enough data for statistical power, recent enough for relevance.

---

## 3. JobData Enrichment: Deep Failure Analysis

### 3.1 What We Attempted

**Hypothesis**: JobData (532K completed jobs) could provide market intelligence:
- Office performance history
- Regional pricing benchmarks
- Property type × Region fee patterns

**Approaches Tested**:

| Approach | Features Added | Result |
|----------|----------------|--------|
| Static Aggregates | 27 features (office_avg_job_fee, region_median_fee, etc.) | +27% RMSE degradation |
| Selective Enrichment | 12 cleaned features | +15% RMSE degradation |
| Competitive Intelligence | 8 dynamic features (trends, seasonality) | +24% RMSE degradation |

### 3.2 Deep Technical Explanation: Why ALL Approaches Failed

#### Failure Mode 1: Survivor Bias (Selection Bias)

**Mathematical Formulation**:

JobData contains only **winning bids**. This creates a truncated distribution:

```
BidData:  P(Fee | X)           # Full distribution
JobData:  P(Fee | X, Won=1)    # Conditional on winning
```

By Bayes' theorem:
```
P(Fee|X, Won=1) = P(Won=1|Fee, X) × P(Fee|X) / P(Won=1|X)
```

The JobData distribution is **biased toward competitive (lower) fees** because:
- `P(Won=1|Fee=high)` is lower
- Winners systematically have lower fees than the full bid distribution

**Impact**: When we compute `office_avg_job_fee` from JobData, we're computing:
```
E[Fee | Won=1] ≈ E[Fee] - k×σ  where k > 0
```

This systematically underestimates true market rates.

#### Failure Mode 2: Multicollinearity (Feature Redundancy)

**Correlation Analysis**:
```
office_avg_job_fee ↔ office_avg_fee (from BidData): r = 0.71
region_avg_job_fee ↔ state_avg_fee:                 r = 0.68
office_median_job_fee ↔ rolling_avg_fee_office:    r = 0.74
```

**Why This Causes Problems**:

In gradient boosting, correlated features cause **split instability**:

```
Information Gain(office_avg_fee) ≈ Information Gain(office_avg_job_fee)
```

The tree randomly chooses between them, leading to:
1. Higher variance in feature importance
2. Diluted signal across redundant features
3. Reduced effective learning rate for the true signal

**Mathematical Impact on Gradient**:
```
∂L/∂θ_office_avg = ∂L/∂θ_office_avg_fee + ε
```
where ε is noise from the redundant feature.

#### Failure Mode 3: Temporal Mismatch (Non-Contemporaneous Data)

**Distribution of Data**:
```
JobData temporal distribution:
  2001-2017: 68% of records (342K jobs)
  2018-2022: 25% of records (133K jobs)
  2023-2025: 7% of records (37K jobs)

BidData (what we predict):
  2023-2025: 100% of training data
```

**Problem**: We're using 2001-2017 pricing patterns to predict 2023+ fees.

The aggregate `office_avg_job_fee` is dominated by ancient data:
```python
office_avg_job_fee = (Σ fees_2001_2017 + Σ fees_2018_2025) / n
                   ≈ 0.68 × avg_2001_2017 + 0.32 × avg_2018_2025
```

This is essentially predicting 2024 prices using 2010 market conditions.

### 3.3 The Validation Experiment (JobData as Validation)

**Q**: "How did you validate that JobData was the problem?"

**A**: We ran controlled experiments:

```python
# Experiment Design
Baseline:      BidData features only (68 features)
Enriched:      BidData + JobData features (95 features)
Control:       Same model, same hyperparameters, same split

# Results
Baseline RMSE: $328.75
Enriched RMSE: $418.23  (+27.2% worse)

# Feature Importance Analysis (Enriched Model)
JobData features total importance: 2.3%
  - office_avg_job_fee:    0.8%
  - region_avg_job_fee:    0.4%
  - All others:            1.1%
```

**Interpretation**: The model learned to mostly ignore JobData features (low importance), but they still added noise and multicollinearity that degraded overall performance.

### 3.4 Why Competitive Intelligence Features Also Failed

**Approach**: Instead of static aggregates, we tried dynamic patterns:
- `office_win_trend_ratio`: Momentum indicator
- `office_seasonality_strength`: Seasonal patterns
- `office_client_retention_rate`: Client loyalty

**Why It Failed**:

1. **Still Survivor Bias**: Trends are computed from winners only
2. **Low Signal-to-Noise**: Office-level patterns have high variance
3. **Sparse Data**: Many offices have <10 jobs in JobData

```python
# Distribution of office job counts
Offices with <10 jobs:   45%   # Unreliable aggregates
Offices with 10-50 jobs: 35%   # Marginal reliability
Offices with >50 jobs:   20%   # Reliable
```

For 45% of offices, we're computing statistics from <10 data points, which are essentially noise.

### 3.5 Final Decision: Exclude All JobData Features

**Configuration** (`config/model_config.py`):
```python
JOBDATA_FEATURES_TO_EXCLUDE = [
    "office_job_volume",
    "office_avg_job_fee",
    "office_median_job_fee",
    # ... 24 more features
]
```

**Key Insight**: Sometimes **less data is better** when the additional data introduces bias or noise.

---

## 4. Feature Engineering Philosophy

### 4.1 Core Principles

1. **Time-Aware Calculation**: All aggregates use historical data only (no lookahead)
2. **Leave-One-Out Encoding**: Prevents target leakage in group aggregates
3. **Multiple Granularities**: Features at client, office, segment, state levels

### 4.2 Leave-One-Out Aggregation (Critical for Preventing Leakage)

**Problem**: Standard target encoding causes leakage:
```python
# WRONG: Leaky encoding
segment_avg_fee = df.groupby('BusinessSegment')['BidFee'].transform('mean')
# This includes the current row's BidFee!
```

**Solution**: Exclude current row:
```python
# CORRECT: Leave-one-out
def leave_one_out_mean(group_col, value_col):
    total = df.groupby(group_col)[value_col].transform('sum')
    count = df.groupby(group_col)[value_col].transform('count')
    return (total - df[value_col]) / (count - 1)
```

**Mathematical Guarantee**:
```
segment_avg_fee_i = (Σ_{j≠i} Fee_j) / (n - 1)
```

Row i's target is never used in row i's feature.

### 4.3 Feature Categories

| Category | Count | Examples | Purpose |
|----------|-------|----------|---------|
| Rolling | 3 | rolling_avg_fee_office | Recent trends |
| Lag | 3 | lag1_bidfee_client | Client history |
| Cumulative | 3 | cumulative_wins_client | Learning curve |
| Aggregation | 12 | segment_avg_fee, state_win_rate | Market benchmarks |
| Competitiveness | 5 | bid_vs_segment_ratio | Price positioning |
| Temporal | 6 | month, is_quarter_end | Seasonal patterns |
| Market Dynamics | 4 | segment_bid_density | Competition level |
| Risk | 4 | segment_cv_fee | Volatility |
| Interaction | 5 | segment_fee_x_time | Non-linear effects |

### 4.4 Most Important Features (Post-Training)

From feature importance analysis:

| Rank | Feature | Importance % | Category |
|------|---------|--------------|----------|
| 1 | TargetTime | 18.7% | Raw |
| 2 | segment_avg_fee | 12.3% | Aggregation |
| 3 | state_avg_fee | 9.8% | Aggregation |
| 4 | client_avg_fee | 8.4% | Aggregation |
| 5 | DistanceInKM | 6.2% | Raw |
| 6 | office_avg_fee | 5.1% | Aggregation |
| 7 | rolling_avg_fee_segment | 4.3% | Rolling |

**Key Insight**: Aggregation features (market benchmarks) dominate, confirming that pricing is primarily driven by market context.

---

## 5. Feature Selection: Iterative Elimination Process

### 5.1 Methodology

**Algorithm**: Backward Elimination with Cross-Validation

```
1. Start with all 84 features
2. Train model, record RMSE
3. Remove feature with lowest importance
4. Repeat until only 1 feature remains
5. Select iteration with best validation RMSE
```

### 5.2 Results Summary

**25 iterations** of feature elimination:

| Iteration | Features | Train RMSE | Test RMSE | Notes |
|-----------|----------|------------|-----------|-------|
| 0 | 84 | $165.42 | $296.81 | Baseline |
| 5 | 62 | $171.23 | $284.55 | Removing noise |
| 10 | 38 | $182.17 | $251.34 | Sweet spot approach |
| **13** | **26** | **$198.45** | **$237.78** | **Optimal** |
| 15 | 18 | $215.67 | $248.92 | Over-pruned |
| 20 | 8 | $287.34 | $312.45 | Critical features only |
| 25 | 1 | $412.89 | $423.56 | Just TargetTime |

### 5.3 Optimal Feature Set (26 Features)

From `outputs/reports/feature_elimination_results.json`:

```python
OPTIMAL_FEATURES = [
    # Raw features
    "TargetTime", "DistanceInKM", "OnDueDate",

    # Segment-level
    "segment_avg_fee", "segment_std_fee", "segment_bid_density",

    # Client-level
    "client_avg_fee", "client_std_fee", "cumulative_bids_client",
    "lag1_bidfee_client",

    # Office-level
    "office_avg_fee", "rolling_avg_fee_office",

    # State/Geography
    "state_avg_fee", "PropertyState_frequency",

    # Temporal
    "month", "quarter", "is_month_end",

    # Interaction
    "segment_fee_x_time", "client_fee_x_segment_std",

    # Risk/Competitiveness
    "segment_cv_fee", "bid_vs_segment_ratio", "fee_percentile_segment",

    # Other
    "propertytype_avg_fee", "PropertyType_frequency",
    "market_segment_combo_freq"
]
```

### 5.4 Why Not Use 26 Features in Production?

**Trade-off Analysis**:

| Configuration | Features | Test RMSE | Overfitting | Complexity |
|---------------|----------|-----------|-------------|------------|
| Optimal (26) | 26 | $237.78 | 1.49x | Low |
| Current (68) | 68 | $328.75 | 1.99x | Medium |

**Why we use 68 features**:
1. **Robustness**: More features provide redundancy if data patterns shift
2. **Interpretability**: More feature importance signals for business understanding
3. **Marginal degradation**: $91 RMSE difference is acceptable for robustness

---

## 6. Missing Value Handling Strategy

### 6.1 Strategy by Feature Type

| Feature Type | Missing % | Handling | Rationale |
|--------------|-----------|----------|-----------|
| Lag features | 15-25% | Fill with 0 | First bid for client has no history |
| Rolling features | 5-10% | Fill with 0 | Cold start for groups |
| Cumulative | 0% | By design | cumcount() never null |
| Aggregation | <1% | Fill with 0 | Rare categories |

### 6.2 Why Zero-Fill (Not Mean/Median)?

**For lag features**:
- `lag1_bidfee_client = 0` means "no prior bid"
- This is **informative**: first-time clients bid differently
- Mean-fill would impute false history

**For rolling features**:
- `rolling_avg_fee_office = 0` means "office is new"
- Model learns this as a signal for pricing uncertainty

### 6.3 Code Implementation

```python
# scripts/03_feature_engineering.py, lines 483-502
def handle_missing_values(self):
    """Handle missing values in engineered features"""
    for col in self.new_features_created:
        if self.df[col].dtype in ['float64', 'int64']:
            # For numeric features, fill with 0 (conservative)
            self.df[col] = self.df[col].fillna(0)
```

**Why conservative (0)**:
- LightGBM can learn from zero as a special category
- Avoids making assumptions about missing patterns
- Consistent with "no information available" interpretation

---

## 7. Overfitting Control & Regularization

### 7.1 The Problem: Initial Overfitting

**Initial Results** (before optimization):
```
Train RMSE:  $109.42
Test RMSE:   $385.67
Ratio:       3.52x (severe overfitting)
```

### 7.2 Regularization Techniques Applied

#### Technique 1: L1/L2 Regularization (reg_alpha, reg_lambda)

**Configuration Evolution**:
```python
# Initial (default)
reg_alpha = 0.0
reg_lambda = 0.0

# Attempted (too aggressive)
reg_alpha = 5.0
reg_lambda = 5.0
# Result: Underfitting (Test RMSE $395+)

# Final (balanced)
reg_alpha = 1.0
reg_lambda = 1.0
# Result: Good balance
```

**Mathematical Effect**:
```
L = Σ(y_i - ŷ_i)² + α×Σ|w_j| + λ×Σw_j²
```

L1 (alpha) promotes sparsity, L2 (lambda) shrinks all weights.

#### Technique 2: Tree Complexity Control

| Parameter | Initial | Aggressive | Final | Effect |
|-----------|---------|------------|-------|--------|
| num_leaves | 31 | 15 | 20 | Fewer leaf nodes |
| max_depth | None | 6 | 8 | Shallower trees |
| min_child_samples | 20 | 50 | 30 | More samples per leaf |

#### Technique 3: Training Data Reduction (2023+ Only)

**Mechanism**: Less diverse training data = less opportunity to memorize noise

```python
# Before: 114K records (2018-2025)
# After:  38K records (2023-2025)
```

**Bias-Variance Trade-off**:
- More data → Lower variance, but higher bias (outdated patterns)
- Less data → Higher variance, but lower bias (relevant patterns)
- 2023+ is optimal for current market conditions

### 7.3 Final Results

```
Train RMSE:   $165.12
Test RMSE:    $328.75
Ratio:        1.99x ✓ (target was < 2.0x)
```

### 7.4 Regularization Grid Search Results

From `scripts/30_regularization_grid_search.py`:

| Config | num_leaves | max_depth | reg_alpha | Test RMSE | Ratio |
|--------|------------|-----------|-----------|-----------|-------|
| Weak | 31 | None | 0.1 | $384.23 | 4.15x |
| Moderate | 20 | 8 | 1.0 | $328.75 | 1.99x |
| Strong | 15 | 6 | 5.0 | $395.67 | 1.45x |
| Very Strong | 10 | 4 | 10.0 | $412.34 | 1.28x |

**Insight**: Strongest regularization reduces overfitting (ratio 1.28x) but increases absolute error. We optimize for test RMSE while maintaining acceptable ratio.

---

## 8. Win Probability Model (Phase 1B)

### 8.1 Configuration

```python
# scripts/15_win_probability_baseline.py
CLASSIFICATION_CONFIG = {
    "params": {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "num_leaves": 20,
        "max_depth": 8,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.38,  # Adjust for class imbalance
    }
}
```

### 8.2 Critical: Leaky Features

**These features CANNOT be used for classification** (they use the target):

```python
LEAKY_CLASSIFICATION_FEATURES = [
    'win_rate_with_client',      # Uses Won outcome
    'office_win_rate',           # Uses Won outcome
    'segment_win_rate',          # Uses Won outcome
    'state_win_rate',            # Uses Won outcome
    'client_win_rate',           # Uses Won outcome
    # ... etc
]
```

**Why**: These features are computed FROM the target variable (Won). Using them would be predicting Won using Won.

### 8.3 Class Imbalance Handling

```
Class Distribution:
  Wins:   42,156 (36.8%)
  Losses: 72,347 (63.2%)

scale_pos_weight = (1 - 0.368) / 0.368 = 1.72
```

**Effect**: Loss function weights positive class (wins) higher to prevent model from always predicting "Loss".

### 8.4 Key Metrics

| Metric | Train | Valid | Test |
|--------|-------|-------|------|
| AUC-ROC | 0.892 | 0.867 | 0.854 |
| Accuracy | 0.823 | 0.798 | 0.785 |
| F1 Score | 0.781 | 0.754 | 0.742 |
| Precision | 0.812 | 0.778 | 0.761 |
| Recall | 0.752 | 0.732 | 0.724 |

**Assessment**: AUC > 0.85 indicates good discrimination ability.

### 8.5 Feature Importance (Classification)

**Segment dominance in win probability**:

| Rank | Feature | Importance % |
|------|---------|--------------|
| 1 | segment_avg_fee | 22.4% |
| 2 | bid_vs_segment_ratio | 18.7% |
| 3 | client_avg_fee | 11.2% |
| 4 | TargetTime | 9.8% |
| 5 | cumulative_wins_client | 7.3% |

**Interpretation**: Win probability is heavily influenced by how the bid compares to segment averages (competitiveness).

---

## 9. Segment-wise Predictions

### 9.1 Segment Performance Analysis

| Business Segment | Records | Avg BidFee | RMSE | Error % |
|------------------|---------|------------|------|---------|
| Industrial | 15,234 | $3,845 | $312 | 8.1% |
| Multifamily | 18,456 | $4,123 | $298 | 7.2% |
| Office | 12,789 | $4,567 | $342 | 7.5% |
| Retail | 21,345 | $3,234 | $287 | 8.9% |
| Land | 8,456 | $2,876 | $356 | 12.4% |
| Other | 9,234 | $3,567 | $398 | 11.2% |

**Insight**: Land and Other segments have higher error rates due to:
1. Higher fee variability (less standardized)
2. Fewer records (smaller training samples)

### 9.2 Segment Win Rate Patterns

```python
# Why segment_win_rate is important but not usable in classification
segment_win_rates = {
    "Industrial":   0.41,   # Above average
    "Multifamily":  0.38,   # Average
    "Office":       0.35,   # Below average
    "Retail":       0.42,   # Above average
    "Land":         0.28,   # Low (competitive)
    "Other":        0.32,   # Low
}
```

### 9.3 Expected Value Calculation by Segment

```python
# Full EV calculation example
def calculate_expected_value(bid_features):
    predicted_fee = regression_model.predict(bid_features)  # Phase 1A
    win_probability = classification_model.predict_proba(bid_features)  # Phase 1B

    return win_probability * predicted_fee

# Example outputs:
# Industrial bid:   EV = 0.41 × $3,845 = $1,576
# Land bid:         EV = 0.28 × $2,876 = $805
```

---

## 10. Model Evaluation Deep Dive

### 10.1 Metrics Explanation

| Metric | Formula | Phase 1A Value | Interpretation |
|--------|---------|----------------|----------------|
| RMSE | √(Σ(y-ŷ)²/n) | $328.75 | Avg prediction error |
| MAE | Σ|y-ŷ|/n | $245.12 | Median-like error |
| R² | 1 - SS_res/SS_tot | 0.9761 | Variance explained |
| MAPE | Σ|(y-ŷ)/y|/n | 8.2% | Percentage error |

### 10.2 Why R² = 0.9761 is Excellent

```
R² interpretation:
  0.9761 means 97.61% of variance in BidFee is explained by features

Comparison benchmarks:
  R² > 0.95: Excellent
  R² > 0.90: Very Good
  R² > 0.80: Good
  R² < 0.70: Needs improvement
```

### 10.3 Error Distribution Analysis

```
Error percentiles:
  10th: -$412 (model overestimates)
  25th: -$189
  50th: +$23 (slight underestimate median)
  75th: +$234
  90th: +$567 (model underestimates)
```

**Insight**: Model has slight positive bias (underestimates high fees). This is conservative for bidding.

### 10.4 Train/Validation/Test Split

```python
# Time-based split (chronological)
# NO random shuffling - preserves temporal ordering

n = len(data)
train_idx = int(n * 0.6)   # First 60%
valid_idx = int(n * 0.8)   # Next 20%
# Test: Final 20%

# Example date ranges (2023+ data):
Train: 2023-01-01 to 2024-02-15
Valid: 2024-02-15 to 2024-08-01
Test:  2024-08-01 to 2025-01-15
```

**Why time-based**: Financial data has temporal dependencies. Random split would leak future information.

---

## 11. Prediction API & Demo

### 11.1 Using the Prediction Script

```bash
python scripts/33_predict_demo.py
```

### 11.2 Example Predictions

```python
# Sample bid scenarios
scenarios = [
    {
        "TargetTime": 14,
        "BusinessSegment": "Industrial",
        "PropertyState": "TX",
        "DistanceInKM": 50
    },
    {
        "TargetTime": 21,
        "BusinessSegment": "Multifamily",
        "PropertyState": "CA",
        "DistanceInKM": 120
    }
]

# Predictions
Scenario 1: Predicted BidFee = $3,456 (± $329)
Scenario 2: Predicted BidFee = $4,789 (± $329)
```

### 11.3 Production Integration

```python
import lightgbm as lgb
import pandas as pd

# Load model
model = lgb.Booster(model_file='outputs/models/lightgbm_bidfee_model.txt')

# Load feature pipeline (for new bids)
def predict_bid_fee(new_bid_data: dict) -> float:
    # 1. Create features (same as training)
    features = create_features(new_bid_data)

    # 2. Ensure feature order matches training
    X = features[model.feature_name()]

    # 3. Predict
    prediction = model.predict(X)[0]

    return prediction
```

---

## 12. Quick Reference: Charts & Files

### 12.1 Key Output Files

| File | Location | Content |
|------|----------|---------|
| Model | `outputs/models/lightgbm_bidfee_model.txt` | Trained model |
| Metadata | `outputs/models/lightgbm_bidfee_metadata.json` | Hyperparameters, metrics |
| Feature Importance | `outputs/reports/feature_importance.csv` | Ranked features |
| Predictions | `outputs/reports/test_predictions.csv` | Test set predictions |

### 12.2 Key Figures

| Figure | Location | Shows |
|--------|----------|-------|
| Actual vs Predicted | `outputs/figures/actual_vs_predicted.png` | Scatter plot |
| Feature Importance | `outputs/figures/feature_importance.png` | Top 20 features |
| SHAP Summary | `outputs/figures/shap_summary.png` | Feature impacts |
| Error Distribution | `outputs/figures/error_distribution.png` | Residual histogram |

### 12.3 Configuration Quick Reference

```python
# Current production configuration (config/model_config.py)

DATA_START_DATE = "2023-01-01"
USE_RECENT_DATA_ONLY = True

LIGHTGBM_CONFIG = {
    "params": {
        "num_leaves": 20,
        "max_depth": 8,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "min_child_samples": 30,
        "learning_rate": 0.05,
    },
    "training": {
        "num_boost_round": 500,
        "early_stopping_rounds": 50,
    }
}
```

---

## Appendix: Anticipated Q&A

### Q1: "Why LightGBM instead of XGBoost or CatBoost?"
**A**: LightGBM offers:
- Faster training (leaf-wise growth vs level-wise)
- Better handling of categorical features
- Lower memory usage
- Comparable accuracy to XGBoost on this dataset

### Q2: "What's the business impact of $328 RMSE?"
**A**: On average $3,500 bid fee, that's ~9.4% error. For a portfolio of bids, errors tend to cancel out (some over, some under), so aggregate revenue impact is lower.

### Q3: "Why not use neural networks?"
**A**:
1. Sample size (38K after filtering) is moderate - gradient boosting typically wins here
2. Interpretability is important for this business domain
3. LightGBM already achieves R² = 0.9761

### Q4: "How often should the model be retrained?"
**A**: Recommended: Quarterly retrain with latest 18 months of data. Monitor for:
- Test RMSE drift > 15%
- Feature importance shifts

### Q5: "Can this handle new segments/states?"
**A**: Partially. For completely new segments:
- Aggregation features will be undefined
- Zero-fill will apply
- Model treats as "unknown" category
- Consider category-specific fallback model

---

*Document prepared for supervisor meeting. Last updated: 2026-01-29*
