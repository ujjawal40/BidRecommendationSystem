# Bid Recommendation System

A production-grade machine learning system for optimizing commercial real estate appraisal bid recommendations.

## Project Overview

This project develops a data-driven recommendation engine for a CRE valuation firm to maximize expected revenue while maintaining high win probability on appraisal bids.

**Client**: Commercial Real Estate Valuation Firm
**Developer**: Global Stat Solutions (GSS)

## Business Objective

Recommend optimal bid fees that maximize:
```
Expected Value = Win Probability × Bid Fee
```

## Current Phase

**Phase 1**: Bid Fee Time Series Modeling
- Model bid fee behavior over time using historical bid data
- Incorporate critical drivers (TargetTime, property type, market, client, etc.)
- Build foundation for future win probability modeling

## Project Structure

```
BidRecommendationSystem/
├── data/
│   ├── raw/                    # Original data (read-only)
│   └── processed/              # Cleaned data with headers
├── notebooks/
│   └── 01_EDA.ipynb           # Exploratory Data Analysis
├── src/
│   └── utils/                 # Reusable utilities
├── outputs/
│   ├── figures/               # Visualizations
│   └── reports/               # Analysis reports
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Setup Instructions

### 1. Create Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate     # On Windows
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Setup
```bash
python -c "import pandas, numpy, sklearn, lightgbm, torch; print('Setup successful!')"
```

## Data Description

**Primary Dataset**: `data/processed/BidData_cleaned.csv`
- **166,372 rows** (bid records from 2018-2025)
- **55 columns** including:
  - **Target**: BidFee (fee amount)
  - **Key Driver**: TargetTime (turnaround days)
  - **Time Index**: BidDate
  - **Categories**: Property type, client, office, market
  - **Demographics**: Population, income, housing values by ZIP
  - **Outcomes**: BidStatusName (Won/Lost/Active/Declined/Placed)

**Data Dictionary**: `data/raw/BidDataDictionary.xlsx`

## Key Features

### Time Series Modeling
- Bid date as temporal index
- Rolling/lag features for historical patterns
- Seasonality and trend analysis

### Feature Engineering
- Categorical encoding (client, property type, office, market)
- Rolling aggregations (3-month avg fee by client/office/property type)
- Temporal features (day of week, month, quarter)
- Geographic features (location, distance)

### Model Explainability
- SHAP values for stakeholder communication
- Feature importance analysis
- Bid-level predictions with explanations

## Development Approach

**Iterative & Production-Focused**:
1. Start with EDA in notebooks
2. Extract reusable code to `src/` modules
3. Build incrementally with proper structure
4. Write tests for critical components
5. Maintain clean, documented code

## Technology Stack

- **Core ML**: pandas, numpy, scikit-learn, LightGBM, PyTorch
- **Visualization**: matplotlib, seaborn, plotly
- **Explainability**: SHAP
- **Code Quality**: black, flake8, pytest

## Current Model Performance (Phase 1A)

| Metric | Value |
|--------|-------|
| **Test RMSE** | $328.75 |
| **Test MAE** | $102.44 |
| **R²** | 0.9761 |
| **Overfitting Ratio** | 1.99x |

**Configuration**:
- Data: 2023+ (52,308 records)
- Features: 68 (optimized)
- Split: 60% train / 20% valid / 20% test
- Regularization: L1=1.0, L2=1.0

## Progress

- [x] Project setup
- [x] Exploratory Data Analysis
- [x] Feature engineering (58 engineered features)
- [x] Data leakage detection and fix
- [x] Baseline model (LightGBM)
- [x] Model optimization (regularization, 2023+ data)
- [ ] Feature ablation study
- [ ] Win probability modeling (Phase 1B)

## Contact

**Developer**: Ujjawal Dwivedi
**Organization**: Global Stat Solutions (GSS)

---

*Last Updated*: 2026-01-23
