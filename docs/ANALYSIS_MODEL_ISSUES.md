# Model Analysis: Why Predictions Aren't Varying

## Executive Summary

**Problem**: Win probability stays nearly constant regardless of input changes (distance, state, segment, target time).

**Root Cause**: The frontend uses a **hardcoded heuristic** instead of the actual trained model.

---

## Current Implementation Analysis

### Frontend Win Probability (ResultDisplay.js:165-174)

```javascript
function calculateExperimentalWinProb(predicted, benchmark) {
  const ratio = predicted / benchmark;

  if (ratio < 0.85) return 72;   // Very competitive bid
  if (ratio < 0.95) return 58;   // Competitive bid
  if (ratio < 1.05) return 45;   // Average bid
  if (ratio < 1.15) return 32;   // Above average bid
  return 22;                      // High bid
}
```

**This is the problem!** There are only 5 possible win probability values (22%, 32%, 45%, 58%, 72%), all based solely on how the predicted fee compares to segment benchmark.

### Why inputs don't affect win probability:

1. **Distance** → Only affects bid fee prediction slightly → Ratio stays similar → Same bucket
2. **State** → Affects both prediction and benchmark → Ratio stays similar → Same bucket
3. **Segment** → Affects both prediction and benchmark → Ratio stays similar → Same bucket
4. **Target Time** → Small effect on fee → Ratio stays similar → Same bucket

---

## The Math Behind Expected Value

### Correct Formula
```
Expected Value (EV) = P(Win) × Bid Fee
```

This is economically sound. To maximize revenue:
- **Higher bid** = Higher fee if won, but lower probability of winning
- **Lower bid** = Lower fee if won, but higher probability of winning

The optimal bid is where:
```
d(EV)/d(Bid) = 0
P'(Win) × Bid + P(Win) = 0
```

### Current Problems

1. **P(Win) is not responsive** - The hardcoded heuristic doesn't capture:
   - Market competitiveness by segment/state
   - Historical win patterns for similar bids
   - Client relationships
   - Timing factors

2. **Model exists but not used** - `lightgbm_win_probability.txt` is trained but not integrated into the API

---

## Trained Win Probability Model Analysis

The model was trained (script 15_win_probability_baseline.py) with:
- **Algorithm**: LightGBM Binary Classifier
- **Features**: 60+ features (excluding leaky win_rate features)
- **Target**: Won (binary)
- **Metric**: AUC-ROC

### Leaky Features Correctly Excluded
```python
LEAKY_CLASSIFICATION_FEATURES = [
    'win_rate_with_client',      # Uses Won outcome
    'office_win_rate',           # Uses Won outcome
    'propertytype_win_rate',     # Uses Won outcome
    'state_win_rate',            # Uses Won outcome
    'segment_win_rate',          # Uses Won outcome
    ...
]
```

---

## Proposed Solutions

### Option 1: Integrate Existing Win Probability Model (Quick Fix)

**Backend changes:**
1. Load `lightgbm_win_probability.txt` in prediction_service.py
2. Generate features for win probability prediction
3. Return actual model probability in API response

**Frontend changes:**
1. Use `win_probability` from API response
2. Remove `calculateExperimentalWinProb` heuristic

**Pros**: Fast to implement, uses already-trained model
**Cons**: Current model may not be well-calibrated

### Option 2: Retrain with XGBoost (Recommended)

Train both regression and classification models with XGBoost:

**For Bid Fee (Regression):**
```python
xgb.XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    objective='reg:squarederror'
)
```

**For Win Probability (Classification):**
```python
xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    objective='binary:logistic',
    scale_pos_weight=class_weight
)
```

### Option 3: Ensemble Approach (Best Performance)

Combine LightGBM and XGBoost:
```
Final Prediction = α × LightGBM + (1-α) × XGBoost
```

Where α is optimized on validation set.

---

## Recommended Git Branching Strategy

```
main
├── feature/xgboost-regression      # XGBoost bid fee model
├── feature/xgboost-classification  # XGBoost win probability
├── feature/win-prob-integration    # Integrate models into API
├── bugfix/prediction-variance      # Fix prediction variance issues
└── frontend/win-probability-ui     # Update frontend to use real model
```

---

## Action Items

1. **Immediate** (feature/win-prob-integration):
   - Integrate existing LightGBM win prob model into API
   - Update frontend to use actual predictions

2. **Short-term** (feature/xgboost-*):
   - Train XGBoost models for comparison
   - Benchmark against LightGBM

3. **Medium-term** (feature/ensemble):
   - Create ensemble if XGBoost performs better
   - Implement model versioning

---

## Feature Sensitivity Analysis Needed

To ensure predictions vary appropriately:

| Input Change | Expected Fee Change | Expected Win Prob Change |
|--------------|---------------------|--------------------------|
| Distance +50km | +2-5% | -1-3% |
| Segment: Financing → Consulting | +15-25% | Varies by market |
| State: IL → CA | +5-10% | Varies by competition |
| Target Time: 30 → 14 days | +5-15% (rush) | -5-10% (harder to win rush) |

If model doesn't show these sensitivities, features may need engineering.
