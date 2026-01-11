"""
Diagnostic Analysis - Why Regression Fails
===========================================
Investigate why regression model shows no improvement

Author: Bid Recommendation System
Date: 2026-01-10
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from config.model_config import FEATURES_DATA, DATE_COLUMN

print("=" * 80)
print("DIAGNOSTIC ANALYSIS - WHY REGRESSION FAILS")
print("=" * 80)

# Load data
df = pd.read_csv(FEATURES_DATA)
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

# Filter to recent data
recent_cutoff = pd.Timestamp('2023-01-01')
df_recent = df[df[DATE_COLUMN] >= recent_cutoff].copy()

print(f"\nData: {len(df_recent):,} rows (2023-2025)\n")

# ============================================================================
# 1. TARGET VARIABLE ANALYSIS
# ============================================================================
print("=" * 80)
print("1. BidFee DISTRIBUTION")
print("=" * 80)

print(f"\nBasic stats:")
print(f"  Mean: ${df_recent['BidFee'].mean():,.2f}")
print(f"  Median: ${df_recent['BidFee'].median():,.2f}")
print(f"  Std Dev: ${df_recent['BidFee'].std():,.2f}")
print(f"  Min: ${df_recent['BidFee'].min():,.2f}")
print(f"  Max: ${df_recent['BidFee'].max():,.2f}")

print(f"\nPercentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    val = df_recent['BidFee'].quantile(p/100)
    print(f"  {p}th: ${val:,.2f}")

# Check coefficient of variation
cv = df_recent['BidFee'].std() / df_recent['BidFee'].mean()
print(f"\nCoefficient of Variation: {cv:.2f}")
print(f"Interpretation: {'HIGH variance' if cv > 0.5 else 'Moderate variance'}")

# ============================================================================
# 2. FEATURE-TARGET CORRELATION
# ============================================================================
print("\n" + "=" * 80)
print("2. FEATURE-TARGET CORRELATION")
print("=" * 80)

# Regression features
regression_features = [
    'segment_avg_fee', 'state_avg_fee', 'propertytype_avg_fee',
    'rolling_avg_fee_segment', 'segment_std_fee', 'client_avg_fee',
    'TargetTime', 'state_std_fee', 'propertytype_std_fee',
    'office_avg_fee', 'segment_win_rate'
]

correlations = []
for feat in regression_features:
    if feat in df_recent.columns:
        corr = df_recent[feat].corr(df_recent['BidFee'])
        correlations.append({'feature': feat, 'correlation': corr})

correlations_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False, key=abs)

print("\nTop features by correlation with BidFee:")
for i, row in correlations_df.iterrows():
    print(f"  {row['feature']:45s} {row['correlation']:7.4f}")

print(f"\nStrongest correlation: {correlations_df.iloc[0]['correlation']:.4f}")

# ============================================================================
# 3. COMPARE WITH CLASSIFICATION TASK
# ============================================================================
print("\n" + "=" * 80)
print("3. CLASSIFICATION vs REGRESSION")
print("=" * 80)

# Classification features
classification_features = [
    'JobCount', 'rolling_bid_count_office', 'propertytype_std_fee',
    'total_bids_to_client', 'office_avg_fee', 'PropertyState_encoded',
    'RooftopLongitude', 'BusinessSegment_frequency', 'office_std_fee',
    'DeliveryTotal', 'market_competitiveness'
]

win_rate = df_recent['Won'].mean()
print(f"\nWin Rate: {win_rate:.1%}")
print(f"Class balance: {'Balanced' if 0.4 < win_rate < 0.6 else 'Imbalanced'}")

# Check if classification features relate to BidFee
print(f"\nDo classification features predict BidFee?")
class_feat_corr = []
for feat in classification_features:
    if feat in df_recent.columns and df_recent[feat].dtype in [np.float64, np.int64]:
        corr = df_recent[feat].corr(df_recent['BidFee'])
        class_feat_corr.append({'feature': feat, 'correlation': corr})

if class_feat_corr:
    class_feat_df = pd.DataFrame(class_feat_corr).sort_values('correlation', ascending=False, key=abs)
    print("\nClassification features correlation with BidFee:")
    for _, row in class_feat_df.head(5).iterrows():
        print(f"  {row['feature']:45s} {row['correlation']:7.4f}")

# ============================================================================
# 4. PREDICTION DIFFICULTY
# ============================================================================
print("\n" + "=" * 80)
print("4. WHAT MAKES BidFee HARD TO PREDICT?")
print("=" * 80)

# Check if BidFee varies within groups
group_vars = []

if 'BusinessSegment' in df_recent.columns:
    segment_variance = df_recent.groupby('BusinessSegment')['BidFee'].agg(['mean', 'std', 'count'])
    segment_variance = segment_variance[segment_variance['count'] >= 10]
    avg_cv_segment = (segment_variance['std'] / segment_variance['mean']).mean()
    print(f"\nWithin-segment variation (avg CV): {avg_cv_segment:.2f}")
    group_vars.append(avg_cv_segment)

if 'PropertyState' in df_recent.columns:
    state_variance = df_recent.groupby('PropertyState')['BidFee'].agg(['mean', 'std', 'count'])
    state_variance = state_variance[state_variance['count'] >= 10]
    avg_cv_state = (state_variance['std'] / state_variance['mean']).mean()
    print(f"Within-state variation (avg CV): {avg_cv_state:.2f}")
    group_vars.append(avg_cv_state)

if 'PropertyType' in df_recent.columns:
    proptype_variance = df_recent.groupby('PropertyType')['BidFee'].agg(['mean', 'std', 'count'])
    proptype_variance = proptype_variance[proptype_variance['count'] >= 10]
    avg_cv_proptype = (proptype_variance['std'] / proptype_variance['mean']).mean()
    print(f"Within-property-type variation (avg CV): {avg_cv_proptype:.2f}")
    group_vars.append(avg_cv_proptype)

if group_vars:
    avg_within_group_cv = np.mean(group_vars)
    print(f"\nAverage within-group CV: {avg_within_group_cv:.2f}")
    print(f"Interpretation: BidFee varies {'HIGHLY' if avg_within_group_cv > 0.5 else 'moderately'} even within same groups")

# ============================================================================
# 5. BASELINE RMSE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("5. BASELINE RMSE CONTEXT")
print("=" * 80)

baseline_rmse = 237.57
mean_bidfee = df_recent['BidFee'].mean()

print(f"\nBaseline RMSE: ${baseline_rmse:,.2f}")
print(f"Mean BidFee: ${mean_bidfee:,.2f}")
print(f"RMSE as % of mean: {(baseline_rmse / mean_bidfee) * 100:.1f}%")

# Naive baseline: predict mean
naive_rmse = np.sqrt(((df_recent['BidFee'] - mean_bidfee) ** 2).mean())
print(f"\nNaive baseline (predict mean): ${naive_rmse:,.2f}")
print(f"Model improvement over naive: {((naive_rmse - baseline_rmse) / naive_rmse) * 100:.1f}%")

# ============================================================================
# 6. ROOT CAUSE
# ============================================================================
print("\n" + "=" * 80)
print("6. HYPOTHESIS: WHY NO IMPROVEMENT?")
print("=" * 80)

hypotheses = []

# H1: High inherent variance
if cv > 0.5:
    hypotheses.append("HIGH VARIANCE: BidFee has inherent randomness not captured by features")

# H2: Weak correlations
max_corr = correlations_df.iloc[0]['correlation']
if abs(max_corr) < 0.8:
    hypotheses.append(f"WEAK SIGNAL: Strongest feature correlation only {max_corr:.2f}")

# H3: Within-group variance
if group_vars and np.mean(group_vars) > 0.4:
    hypotheses.append("WITHIN-GROUP VARIANCE: BidFee varies highly even in same segment/state/type")

# H4: Features are averages
if any('avg' in f for f in regression_features[:5]):
    hypotheses.append("CIRCULAR FEATURES: Using avg_fee to predict fee creates dependency")

# H5: Missing key factors
if baseline_rmse > mean_bidfee * 0.15:
    hypotheses.append("MISSING FACTORS: Model RMSE > 15% of mean suggests key predictors missing")

print("\nLikely reasons for lack of improvement:\n")
for i, h in enumerate(hypotheses, 1):
    print(f"{i}. {h}")

# ============================================================================
# 7. RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("7. RECOMMENDATIONS")
print("=" * 80)

print("\n1. Check for BidFee determinants NOT in features:")
print("   - Client negotiation history")
print("   - Complexity indicators (GBA, property specifics)")
print("   - Competitive factors (# competitors)")
print("   - Time pressure (days until due date)")

print("\n2. Consider different approach:")
print("   - Predict BidFee CATEGORY (low/medium/high) instead of exact value")
print("   - Predict RANGE instead of point estimate")
print("   - Ensemble with win probability for bid optimization")

print("\n3. Feature engineering focus:")
print("   - Property complexity features")
print("   - Client relationship features")
print("   - Market competition features")
print("   - Urgency features")

print("\n" + "=" * 80)
