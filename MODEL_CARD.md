# Model Card: Bid Fee Prediction Model

## Model Details

| Property | Value |
|----------|-------|
| **Model Name** | BidFee Predictor v1.0 |
| **Model Type** | LightGBM Gradient Boosting Regressor |
| **Version** | 1.0.0 |
| **Developer** | Global Stat Solutions (GSS) |
| **Date** | January 2026 |
| **License** | Proprietary |

## Intended Use

### Primary Use Case
Predict optimal bid fees for commercial real estate appraisal engagements to maximize expected revenue while maintaining competitive win rates.

### Intended Users
- Bid managers at CRE valuation firms
- Pricing analysts
- Business development teams

### Out-of-Scope Uses
- Residential property appraisals
- Non-US markets
- Bids with TargetTime > 365 days
- Property types not in training data

## Training Data

### Dataset
- **Source**: Historical bid records (2023-2025)
- **Size**: ~52,000 records
- **Time Period**: January 2023 - January 2025
- **Geographic Coverage**: United States

### Features (68 total)
- **Temporal**: TargetTime, BidMonth, BidQuarter, BidDayOfWeek
- **Geographic**: State, Region, DistanceInKM, ZIP demographics
- **Property**: PropertyType, PropertySubType
- **Market**: segment_avg_fee, state_avg_fee, office_avg_fee
- **Client**: ClientCode, historical bid patterns

### Target Variable
- **BidFee**: Dollar amount of the bid (continuous, non-negative)

### Data Split
| Split | Percentage | Purpose |
|-------|------------|---------|
| Training | 60% | Model fitting |
| Validation | 20% | Early stopping |
| Test | 20% | Final evaluation |

## Model Performance

### Metrics (Test Set)

| Metric | Value |
|--------|-------|
| **RMSE** | $328.75 |
| **MAE** | $102.44 |
| **R²** | 0.9761 |
| **MAPE** | ~3.5% |

### Overfitting Analysis

| Metric | Train | Test | Ratio |
|--------|-------|------|-------|
| RMSE | $165.18 | $328.75 | 1.99x |

*Note: Overfitting ratio < 2.0x indicates acceptable generalization.*

## Model Architecture

### LightGBM Parameters
```python
{
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 20,
    'max_depth': 8,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 1.0,  # L1 regularization
    'reg_lambda': 1.0,  # L2 regularization
    'min_child_samples': 30,
    'verbose': -1
}
```

## Limitations

### Known Limitations
1. **Temporal Drift**: Model trained on 2023+ data; performance may degrade on patterns from earlier periods
2. **New Categories**: Cannot handle property types, states, or client codes not seen during training
3. **Extreme Values**: Less accurate for bids significantly outside the typical range ($500-$15,000)
4. **Market Shocks**: Does not account for sudden market disruptions

### Failure Modes
- Returns invalid predictions for missing required features
- May extrapolate poorly for extremely short (<3 days) or long (>180 days) turnaround times
- Confidence intervals assume normal residual distribution

## Ethical Considerations

### Fairness
- Model does not use protected characteristics (race, gender, etc.)
- Geographic features (state, ZIP) may correlate with demographic factors
- Recommend periodic fairness audits across regions

### Transparency
- SHAP values available for individual prediction explanations
- Feature importance rankings documented
- Prediction confidence intervals provided

### Privacy
- Model trained on aggregated business data
- No personal identifiable information (PII) in features
- Client codes are anonymized

## Validation & Testing

### Input Validation
The model includes input validation that checks:
- Required features present
- Feature values within expected ranges
- No negative fee values
- Missing value handling

### Confidence Intervals
95% confidence intervals provided with each prediction based on residual standard deviation.

## Maintenance

### Monitoring Recommendations
1. Track prediction accuracy monthly
2. Monitor for data drift in key features
3. Retrain quarterly with fresh data
4. Alert on prediction confidence interval width > $1,000

### Update Schedule
- **Quarterly**: Retrain with new data
- **Annually**: Full model review and potential architecture changes

## Contact

- **Developer**: Ujjawal Dwivedi
- **Organization**: Global Stat Solutions (GSS)
- **Repository**: BidRecommendationSystem

---

*Last Updated: January 2026*
