# Data Leakage Analysis Report
## Critical Issues Found in Feature Engineering

**Date**: 2026-01-06
**Severity**: CRITICAL
**Impact**: Model R² = 0.9996 is artificially inflated due to data leakage

---

## Executive Summary

The LightGBM model achieved an unrealistically high R² of 0.9996 (RMSE: $43.91) due to **severe data leakage** in engineered features. Multiple features include the target variable (BidFee) in their calculations, allowing the model to "cheat" by learning circular relationships.

**Status**: Model results are INVALID and cannot be used for production.

---

## Data Leakage Sources Identified

### 1. CRITICAL: Aggregation Features Include Current Row

**Location**: `scripts/03_feature_engineering.py:271-283`

**Affected Features** (5 groups × 3 stats = 15 features):
- `office_avg_fee`, `office_std_fee`
- `propertytype_avg_fee`, `propertytype_std_fee`
- `state_avg_fee`, `state_std_fee`
- `segment_avg_fee`, `segment_std_fee`
- `client_avg_fee`, `client_std_fee`

**Problem Code**:
```python
# Mean BidFee for this group
group_mean = self.df.groupby(col)['BidFee'].transform('mean')
self.df[f'{name.lower()}_avg_fee'] = group_mean

# Standard deviation for this group
group_std = self.df.groupby(col)['BidFee'].transform('std')
self.df[f'{name.lower()}_std_fee'] = group_std.fillna(0)
```

**Why This Leaks**:
- `transform('mean')` calculates the mean INCLUDING the current row's BidFee
- When predicting row i, the feature `propertytype_avg_fee` already contains information about BidFee[i]
- This is equivalent to saying "predict BidFee using the average that includes BidFee"

**Impact**:
- **propertytype_avg_fee** is the #4 most important feature (importance: 123 billion)
- **office_avg_fee** is the #5 most important feature (importance: 26 billion)

---

### 2. CRITICAL: Ratio Features Using Target Variable

**Location**: `scripts/03_feature_engineering.py:431-443`

**Affected Features**:
- `fee_ratio_to_proptype` = BidFee / propertytype_avg_fee
- `fee_ratio_to_rolling_office` = BidFee / rolling_avg_fee_office

**Problem Code**:
```python
self.df['fee_ratio_to_proptype'] = (
    self.df['BidFee'] / (self.df['propertytype_avg_fee'] + 1)
)
```

**Why This Leaks**:
- **fee_ratio_to_proptype**: Divides BidFee by an aggregate that includes BidFee
  - If BidFee = $3,000 and propertytype_avg = $3,000, ratio = 1.0
  - If BidFee = $6,000 and propertytype_avg = $3,200, ratio = 1.875
  - The ratio directly reveals information about the target!

- **fee_ratio_to_rolling_office**: While rolling_avg is properly shifted, the feature still uses BidFee directly in the calculation
  - This is borderline but acceptable IF only used with properly shifted aggregates

**Impact**:
- **fee_ratio_to_proptype** is the #1 most important feature (importance: 2.6 TRILLION!)
- This single feature has 10x more importance than all other features combined

---

### 3. CRITICAL: Fee Deviation Feature

**Location**: `scripts/03_feature_engineering.py:332-334`

**Affected Feature**:
- `fee_deviation_from_office_avg`

**Problem Code**:
```python
self.df['fee_deviation_from_office_avg'] = (
    self.df['BidFee'] - self.df['office_avg_fee']
) / (self.df['office_std_fee'] + 1)
```

**Why This Leaks**:
- Uses BidFee directly
- Both office_avg_fee and office_std_fee include the current row's BidFee
- This is a double-leakage: target in numerator, target in denominator

**Impact**:
- **#2 most important feature** (importance: 482 billion)

---

### 4. MINOR: Client Fee Ratio

**Location**: `scripts/03_feature_engineering.py:447-450`

**Affected Feature**:
- `client_fee_ratio_to_market`

**Problem Code**:
```python
market_avg = self.df['BidFee'].mean()
self.df['client_fee_ratio_to_market'] = (
    self.df['client_avg_fee'] / market_avg
)
```

**Why This Leaks**:
- `client_avg_fee` includes current row's BidFee
- `market_avg` is calculated on entire dataset including current row
- Less severe than others but still leaky

---

## Features WITHOUT Leakage (Properly Implemented)

### ✓ Rolling Features (Lines 102-160)
**Correct Implementation**:
```python
self.df['rolling_avg_fee_office'] = self.df.groupby('OfficeLocation')['BidFee'].transform(
    lambda x: x.rolling(window, min_periods=1).mean().shift(1)
)
```
- Uses `.shift(1)` to exclude current and future rows
- Only uses past data

### ✓ Lag Features (Lines 176-209)
**Correct Implementation**:
```python
self.df['prev_fee_same_client'] = self.df.groupby('BidCompanyName')['BidFee'].shift(1)
```
- Properly shifted

### ✓ Cumulative Features (Lines 213-242)
**Correct Implementation**:
```python
self.df['total_wins_with_client'] = self.df.groupby('BidCompanyName')['Won'].cumsum() - self.df['Won']
```
- Excludes current row by subtracting current value

---

## Evidence of Leakage Impact

### Model Performance (Too Good To Be True)
- **R² = 0.9996** (99.96% variance explained)
- **RMSE = $43.91** (on target with mean $3,363, std $2,131)
- **MAE = $16.09** (only $16 average error!)

### Top Features Are All Leaky
| Rank | Feature | Importance | Leaky? |
|------|---------|-----------|--------|
| 1 | fee_ratio_to_proptype | 2.6 trillion | **YES** ✗ |
| 2 | fee_deviation_from_office_avg | 482 billion | **YES** ✗ |
| 3 | fee_ratio_to_rolling_office | 151 billion | Borderline |
| 4 | propertytype_avg_fee | 124 billion | **YES** ✗ |
| 5 | office_avg_fee | 27 billion | **YES** ✗ |
| 6 | rolling_avg_fee_office | 18 billion | **NO** ✓ |
| 7 | office_std_fee | 15 billion | **YES** ✗ |
| 8 | rolling_std_fee_office | 9 billion | **NO** ✓ |
| 9 | propertytype_std_fee | 6 billion | **YES** ✗ |

**7 out of top 9 features contain data leakage.**

---

## Root Cause Analysis

### Why This Happened

1. **Confusion between global and time-aware aggregations**
   - Rolling features correctly use `.shift(1)`
   - Global aggregations incorrectly use `.transform('mean')` without exclusion

2. **Missing leave-one-out logic**
   - Global aggregations should exclude current row: `(sum - current) / (count - 1)`
   - Instead used: `sum / count` (includes current)

3. **Ratio features using target**
   - Should never divide/subtract the target variable directly
   - Should only use properly shifted/excluded aggregates

---

## Required Fixes

### Fix 1: Exclude Current Row from Aggregations

**Replace this**:
```python
group_mean = self.df.groupby(col)['BidFee'].transform('mean')
```

**With this (leave-one-out mean)**:
```python
# Calculate total sum and count
group_sum = self.df.groupby(col)['BidFee'].transform('sum')
group_count = self.df.groupby(col)['BidFee'].transform('count')

# Exclude current row: (total - current) / (count - 1)
group_mean = (group_sum - self.df['BidFee']) / (group_count - 1)
```

### Fix 2: Remove or Fix Ratio Features

**Option A - Remove entirely**:
```python
# DELETE these features:
# - fee_ratio_to_proptype
# - fee_deviation_from_office_avg
# - client_fee_ratio_to_market
```

**Option B - Use only shifted aggregates**:
```python
# ONLY use rolling features (already shifted):
self.df['fee_ratio_to_rolling_office'] = (
    self.df['BidFee'] / (self.df['rolling_avg_fee_office'] + 1)
)
```

**Note**: Even with shifted aggregates, using BidFee in the feature is questionable. Consider creating these as "deviation from expected" features instead.

### Fix 3: Verify All Features

Run this check after feature engineering:
```python
# Check for any feature that correlates too highly with target
correlations = df.corr()['BidFee'].abs().sort_values(ascending=False)
suspicious = correlations[correlations > 0.95]
print("Features with correlation > 0.95 to target:")
print(suspicious)
```

---

## Expected Performance After Fix

### Realistic Expectations
- **R² = 0.50 - 0.75** (typical for complex regression problems)
- **RMSE = $500 - $1,000** (15-30% of target std dev)
- **MAE = $300 - $700**

### Why Much Lower?
- Weak raw correlations (all < 0.012 from EDA)
- High cardinality categoricals
- Complex non-linear relationships
- Significant unexplained variance (market dynamics, negotiation, etc.)

---

## Action Items

### Immediate Actions
1. ✗ **DO NOT use current model for any predictions or business decisions**
2. ✗ **DO NOT share current results with stakeholders**
3. ✓ **Fix feature engineering script** (apply Fix 1, 2, 3 above)
4. ✓ **Retrain model with corrected features**
5. ✓ **Validate new model with cross-validation**

### Validation Checklist
- [ ] All aggregations exclude current row
- [ ] No features directly use BidFee in calculations
- [ ] All rolling features use `.shift(1)` or `.shift()`
- [ ] Cumulative features subtract current value
- [ ] Feature correlations with target all < 0.95
- [ ] Model R² is realistic (0.5-0.75 range)

---

## Lessons Learned

1. **Always be suspicious of perfect performance**
   - R² > 0.99 is almost always data leakage

2. **Never use target variable in feature engineering**
   - Exception: Properly shifted lag features in time series

3. **Global aggregations need leave-one-out logic**
   - `transform('mean')` includes current row by default

4. **Feature importance reveals leakage**
   - Extremely high importance (trillions vs billions) is a red flag
   - Top features should be interpretable business drivers

5. **Cross-validation would have caught this**
   - Train/test split still trains on leaky aggregates
   - CV would show impossibly low error on validation folds

---

## References

- Feature Engineering Script: `scripts/03_feature_engineering.py`
- Model Training Script: `scripts/04_model_lightgbm.py`
- Feature Importance Report: `outputs/reports/lightgbm_feature_importance.csv`
- Model Metadata: `outputs/models/lightgbm_metadata.json`

---

**Conclusion**: The current model is INVALID due to severe data leakage. All leaky features must be fixed before retraining. The true model performance is expected to be significantly lower (R² ~ 0.6) but will be honest and production-ready.
