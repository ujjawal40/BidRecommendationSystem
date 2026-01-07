# Feature Engineering Documentation
## Bid Recommendation System - Phase 1A

**Project**: Commercial Real Estate Bid Fee Prediction
**Dataset**: BidData (2018-2025)
**Total Engineered Features**: 58
**Final Feature Count**: 127 (69 original + 58 engineered)

---

## Table of Contents
1. [EDA Key Insights](#eda-key-insights)
2. [Complete Feature Mapping](#complete-mapping-engineered-features--original-features)
3. [Summary by Original Feature Usage](#summary-by-original-feature-usage)

---

## EDA Key Insights

### 1. Data Quality Issues
- **Duplicate BidIds (25%)**: 41,592 duplicate rows due to Master/SubJob hierarchy - we aggregated these to prevent data leakage
- **Missing Target (1.85%)**: Only 3,072 missing BidFee values - removed during preprocessing
- **Missing TargetTime (22%)**: Critical driver with 36,644 missing values - imputed using group medians

### 2. Target Variable (BidFee) Characteristics
- **Highly Right-Skewed**: Mean $4,335 vs Median $3,000
- **Extreme Outliers**: Max of $100M (unrealistic) - we capped at 99th percentile ($15,000)
- **Final Distribution**: After preprocessing, Mean $3,363, Std $2,131

### 3. Win Rate Patterns
- **Overall Win Rate**: 50.37% (raw) → 43.34% (after preprocessing)
- **Status Distribution**: Won (50.4%), Lost (40.1%), Placed (4.3%)
- **Imbalanced but workable** for classification modeling

### 4. Temporal Coverage
- **Date Range**: 2018-01-02 to 2025-12-20 (8 years)
- **Strong temporal patterns** exist for time series features

### 5. Geographic Concentration
- **Top States**: Illinois (25%), Florida (14%), Texas (7%)
- **Geographic features** likely predictive given concentration

### 6. Property Type Distribution
- **Top Types**: Multifamily (22%), Retail (21%), Office (15%), Industrial (14%)
- **Diverse portfolio** suggests property-type-specific patterns

### 7. Critical Finding: WEAK NUMERICAL CORRELATIONS
This is the most important finding:
- **BidFee correlation with TargetTime**: Only 0.0058 (essentially zero)
- **BidFee correlation with Distance**: Only 0.0077 (essentially zero)
- **BidFee correlation with Demographics**: All < 0.012 (negligible)

**Implication**: Raw numerical features alone won't predict BidFee well. We NEED:
- Categorical features (Office, PropertyType, Client, State)
- Aggregated historical patterns
- Interaction effects
- Time-based rolling features

This justified our heavy feature engineering approach.

---

## Complete Mapping: Engineered Features → Original Features

### **1. Rolling/Time Series Features (7)**

| Engineered Feature | Derived From | Calculation | Relevance |
|-------------------|--------------|-------------|-----------|
| `rolling_avg_fee_office` | `OfficeLocation` + `BidFee` + `BidDate` | 90-day rolling mean of BidFee per Office (shifted) | Captures office-specific pricing trends over time; offices may adjust pricing based on market conditions |
| `rolling_avg_fee_proptype` | `Bid_Property_Type` + `BidFee` + `BidDate` | 90-day rolling mean of BidFee per Property Type (shifted) | Property type markets fluctuate; retail vs office pricing changes over time |
| `rolling_avg_fee_state` | `PropertyState` + `BidFee` + `BidDate` | 90-day rolling mean of BidFee per State (shifted) | Geographic market dynamics; state-level economic conditions affect pricing |
| `rolling_avg_fee_segment` | `BusinessSegment` + `BidFee` + `BidDate` | 90-day rolling mean of BidFee per Segment (shifted) | Different business segments (residential, commercial) have different pricing patterns |
| `rolling_bid_count_office` | `OfficeLocation` + `BidDate` | Count of bids per Office in 90-day window (shifted) | Office capacity indicator; busy offices may price higher or lower based on workload |
| `rolling_win_rate_office` | `OfficeLocation` + `Won` + `BidDate` | Win rate per Office in 90-day window (shifted) | Office competitiveness; winning offices may command premium pricing |
| `rolling_std_fee_office` | `OfficeLocation` + `BidFee` + `BidDate` | Standard deviation of BidFee per Office in 90-day window | Fee volatility; stable vs variable pricing strategies by office |

---

### **2. Client Lag Features (4)**

| Engineered Feature | Derived From | Calculation | Relevance |
|-------------------|--------------|-------------|-----------|
| `prev_fee_same_client` | `BidCompanyName` + `BidFee` + `BidDate` | Previous BidFee to the same client (shifted by 1) | Pricing consistency; we may anchor to previous quotes for repeat clients |
| `prev_won_same_client` | `BidCompanyName` + `Won` + `BidDate` | Whether we won the last bid with this client (shifted) | Win/loss momentum affects next bid strategy; may bid lower after losing |
| `days_since_last_bid_client` | `BidCompanyName` + `BidDate` | Days between current bid and previous bid to same client | Relationship recency; frequent clients may get preferential pricing |
| `same_proptype_as_last_client` | `BidCompanyName` + `Bid_Property_Type` + `BidDate` | Binary: same property type as last bid to client | Client specialization; clients who consistently request same type may get better rates |

---

### **3. Cumulative Client Features (4)**

| Engineered Feature | Derived From | Calculation | Relevance |
|-------------------|--------------|-------------|-----------|
| `total_bids_to_client` | `BidCompanyName` + `BidDate` | Cumulative count of bids to each client (time-aware) | Relationship depth; long-term clients may negotiate volume discounts |
| `total_wins_with_client` | `BidCompanyName` + `Won` + `BidDate` | Cumulative wins with each client (time-aware) | Historical success; strong relationships may command premium or get loyalty pricing |
| `win_rate_with_client` | `total_wins_with_client` ÷ `total_bids_to_client` | Percentage of bids won with this client | Client-specific competitiveness; if we always win, we may be underpricing |
| `avg_historical_fee_client` | `BidCompanyName` + `BidFee` + `BidDate` | Running average of all past fees to this client | Client-specific pricing baseline; establishes what this client typically pays |

---

### **4. Aggregation Features (15)**

#### Office Aggregations (3)
| Engineered Feature | Derived From | Calculation | Relevance |
|-------------------|--------------|-------------|-----------|
| `office_avg_fee` | `OfficeLocation` + `BidFee` | Mean BidFee for each office (global average) | Office-specific baseline pricing; expensive vs budget offices |
| `office_std_fee` | `OfficeLocation` + `BidFee` | Std dev of BidFee for each office | Pricing consistency by office; some offices have standardized pricing |
| `office_win_rate` | `OfficeLocation` + `Won` | Win percentage for each office | Office competitiveness; high-performing offices may price differently |

#### Property Type Aggregations (3)
| Engineered Feature | Derived From | Calculation | Relevance |
|-------------------|--------------|-------------|-----------|
| `propertytype_avg_fee` | `Bid_Property_Type` + `BidFee` | Mean BidFee per property type | Property type baseline; retail ≠ office ≠ industrial pricing |
| `propertytype_std_fee` | `Bid_Property_Type` + `BidFee` | Std dev of BidFee per property type | Property type variability; some types have wide price ranges |
| `propertytype_win_rate` | `Bid_Property_Type` + `Won` | Win rate per property type | Property type competitiveness; may indicate market saturation |

#### State Aggregations (3)
| Engineered Feature | Derived From | Calculation | Relevance |
|-------------------|--------------|-------------|-----------|
| `state_avg_fee` | `PropertyState` + `BidFee` | Mean BidFee per state | Geographic pricing baseline; California ≠ Ohio pricing |
| `state_std_fee` | `PropertyState` + `BidFee` | Std dev of BidFee per state | Geographic variability; some markets more volatile |
| `state_win_rate` | `PropertyState` + `Won` | Win rate per state | Geographic competitiveness; competitive markets need lower bids |

#### Business Segment Aggregations (3)
| Engineered Feature | Derived From | Calculation | Relevance |
|-------------------|--------------|-------------|-----------|
| `segment_avg_fee` | `BusinessSegment` + `BidFee` | Mean BidFee per segment | Segment-specific pricing; commercial vs residential |
| `segment_std_fee` | `BusinessSegment` + `BidFee` | Std dev of BidFee per segment | Segment variability; standardized vs custom pricing |
| `segment_win_rate` | `BusinessSegment` + `Won` | Win rate per segment | Segment competitiveness; our strength areas |

#### Client Aggregations (3)
| Engineered Feature | Derived From | Calculation | Relevance |
|-------------------|--------------|-------------|-----------|
| `client_avg_fee` | `BidCompanyName` + `BidFee` | Mean BidFee per client (global) | Client-specific average; enterprise vs small clients |
| `client_std_fee` | `BidCompanyName` + `BidFee` | Std dev of BidFee per client | Client variability; consistent vs project-based pricing |
| `client_win_rate` | `BidCompanyName` + `Won` | Win rate per client (global) | Client competitiveness; sticky vs price-shopping clients |

---

### **5. Interaction Features (4)**

| Engineered Feature | Derived From | Calculation | Relevance |
|-------------------|--------------|-------------|-----------|
| `targettime_x_size` | `TargetTime` × `GrossBuildingAreaRange` | Product of time and size (numeric mapped) | Complexity indicator; large + lengthy = complex project requiring higher fee |
| `distance_x_volume` | `DistanceInMiles` × `rolling_bid_count_office` | Product of distance and office workload | Capacity constraint; distant jobs when busy are harder to staff |
| `client_relationship_strength` | `total_wins_with_client` × `win_rate_with_client` | Product of wins and win rate | Relationship quality; many wins + high rate = strong partnership |
| `market_competitiveness` | `fee_deviation_from_office_avg` × `office_win_rate` | Product of fee deviation and win rate | Pricing pressure; high-winning offices with deviant pricing indicates market power |

---

### **6. Categorical Encodings (14)**

#### Frequency Encoding (6)
| Engineered Feature | Derived From | Calculation | Relevance |
|-------------------|--------------|-------------|-----------|
| `Bid_Property_Type_frequency` | `Bid_Property_Type` | Count of each property type / total | Common property types may have standardized pricing |
| `PropertyState_frequency` | `PropertyState` | Count of each state / total | High-volume states may have economies of scale |
| `OfficeLocation_frequency` | `OfficeLocation` | Count of each office / total | Busy offices may have different pricing |
| `BusinessSegment_frequency` | `BusinessSegment` | Count of each segment / total | Common segments may have standardized rates |
| `BidCompanyType_frequency` | `BidCompanyType` | Count of each company type / total | Enterprise vs SMB frequency affects pricing power |
| `MarketOrientation_frequency` | `MarketOrientation` | Count of each orientation / total | Common market types may have competitive pricing |

#### Label Encoding (8)
| Engineered Feature | Derived From | Calculation | Relevance |
|-------------------|--------------|-------------|-----------|
| `Bid_Property_Type_encoded` | `Bid_Property_Type` | Integer mapping (0, 1, 2...) | Allows tree models to split on property type |
| `Bid_SubProperty_Type_encoded` | `Bid_SubProperty_Type` | Integer mapping | Finer property classification for tree splits |
| `PropertyState_encoded` | `PropertyState` | Integer mapping | Geographic splits in tree models |
| `PropertyCity_encoded` | `PropertyCity` | Integer mapping | City-level granularity for trees |
| `OfficeLocation_encoded` | `OfficeLocation` | Integer mapping | Office-specific patterns in trees |
| `BusinessSegment_encoded` | `BusinessSegment` | Integer mapping | Segment splits in tree models |
| `BidCompanyName_encoded` | `BidCompanyName` | Integer mapping | Client-specific patterns (high cardinality) |
| `BidCompanyType_encoded` | `BidCompanyType` | Integer mapping | Company type patterns in trees |

---

### **7. Temporal Trend Features (4)**

| Engineered Feature | Derived From | Calculation | Relevance |
|-------------------|--------------|-------------|-----------|
| `days_since_start` | `BidDate` | Days since 2018-01-02 (min date) | Linear time trend; pricing inflation over 8 years |
| `quarterly_avg_fee` | `BidDate` + `BidFee` | Mean BidFee per quarter | Seasonal pricing patterns; Q4 rush vs Q1 slow |
| `is_peak_season` | `BidDate` | Binary: 1 if Q2/Q3, else 0 | Peak workload season may affect pricing |
| `is_weekday` | `BidDate` | Binary: 1 if Mon-Fri, else 0 | Weekday vs weekend bid submission patterns |

---

### **8. Ratio Features (4)**

| Engineered Feature | Derived From | Calculation | Relevance |
|-------------------|--------------|-------------|-----------|
| `fee_ratio_to_rolling_office` | `BidFee` ÷ `rolling_avg_fee_office` | Current fee / recent office average | Relative pricing; bidding above/below recent office trend |
| `fee_ratio_to_proptype` | `BidFee` ÷ `propertytype_avg_fee` | Current fee / property type average | Property-specific relative pricing; expensive/cheap for this type |
| `client_fee_ratio_to_market` | `client_avg_fee` ÷ global avg fee | Current fee / overall average | Client premium indicator; high-value vs budget clients |
| `targettime_ratio_to_proptype` | `TargetTime` ÷ avg TargetTime per property type | Current time / property type average | Relative urgency; rushed vs normal timeline for this property type |

---

### **9. Utility Features (2)**

| Engineered Feature | Derived From | Calculation | Relevance |
|-------------------|--------------|-------------|-----------|
| `building_size_numeric` | `GrossBuildingAreaRange` | Mapping: Small=1, Medium=2, Large=3, XL=4 | Numeric size for interactions and correlations |
| `fee_deviation_from_office_avg` | (`BidFee` - `office_avg_fee`) / `office_std_fee` | Z-score of fee relative to office | Standardized deviation; how unusual is this bid for this office? |

---

## Summary by Original Feature Usage

### Most Used Original Features:
1. **BidFee** - Used in 28 engineered features (all rolling, aggregations, ratios)
2. **BidDate** - Used in 22 features (all time-based features)
3. **OfficeLocation** - Used in 11 features (office-level aggregations)
4. **BidCompanyName** - Used in 11 features (client relationship features)
5. **Bid_Property_Type** - Used in 8 features (property-level patterns)
6. **Won** - Used in 7 features (all win rate calculations)
7. **PropertyState** - Used in 6 features (geographic patterns)
8. **TargetTime** - Used in 4 features (complexity indicators)
9. **BusinessSegment** - Used in 6 features (segment patterns)
10. **GrossBuildingAreaRange** - Used in 2 features (size interactions)

---

## Key Takeaways for Modeling

1. **Weak raw correlations** mean ensemble methods (LightGBM, XGBoost) will be crucial
2. **Categorical features** (Office, Client, PropertyType, State) likely to be top predictors
3. **Historical patterns** (rolling, cumulative, lag features) should improve predictions
4. **Time-aware validation** is critical - must use TimeSeriesSplit
5. **Feature importance analysis** will be essential to understand what drives bids
6. **Client relationship features** may be the secret sauce for personalized pricing

### Feature Engineering Strategy

The feature engineering created a **127-column dataset** (69 original + 58 engineered) that should give the model rich context about:
- **Market conditions** (rolling averages, trends)
- **Relationship history** (client features)
- **Competitive positioning** (ratios, deviations)
- **Temporal patterns** (seasonality, trends)

### Critical Design Decision: Time-Aware Features

All rolling, lag, and cumulative features use proper **shifting** to avoid data leakage:
```python
# Example: Rolling average with shift to prevent leakage
df['rolling_avg_fee_office'] = df.groupby('OfficeLocation')['BidFee'].transform(
    lambda x: x.rolling('90D', min_periods=1).mean().shift(1)
)
```

The `.shift(1)` ensures that the current row's target (BidFee) is not included in the calculation, preventing data leakage during model training.

---

## Conclusion

**Key Insight**: The weak correlations in raw features are transformed into predictive power through:
- **Temporal aggregation** (rolling windows capture trends)
- **Grouping** (office/client/property-specific patterns)
- **Historical context** (cumulative and lag features)
- **Relative positioning** (ratios and deviations)

This transforms 69 weakly-correlated columns into 127 feature-rich columns with embedded domain knowledge about pricing dynamics, client relationships, and market conditions.

---

**Generated**: 2026-01-06
**Script**: `scripts/03_feature_engineering.py`
**Output**: `data/features/BidData_features.csv`
**Project**: Bid Recommendation System - Phase 1A
