"""
Unused Features Exploration
============================
Test features NOT in current regression model

Current baseline: $237.57 RMSE using 11 features
Try: GBA, population, distance, delivery, property details

Author: Bid Recommendation System
Date: 2026-01-10
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings

from config.model_config import (
    FEATURES_DATA, TARGET_COLUMN, DATE_COLUMN,
    MODELS_DIR, REPORTS_DIR, RANDOM_SEED,
)

warnings.filterwarnings('ignore')

print("=" * 80)
print("UNUSED FEATURES EXPLORATION")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Baseline to beat: $237.57 RMSE\n")

# Load data
df = pd.read_csv(FEATURES_DATA)
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

# Filter to recent data
recent_cutoff = pd.Timestamp('2023-01-01')
df_recent = df[df[DATE_COLUMN] >= recent_cutoff].copy()

print(f"Data: {len(df_recent):,} rows (2023-2025)\n")

# ============================================================================
# IDENTIFY UNUSED FEATURES
# ============================================================================
print("=" * 80)
print("IDENTIFYING UNUSED FEATURES")
print("=" * 80)

# Current regression features
CURRENT_FEATURES = [
    'segment_avg_fee', 'state_avg_fee', 'propertytype_avg_fee',
    'rolling_avg_fee_segment', 'segment_std_fee', 'client_avg_fee',
    'TargetTime', 'state_std_fee', 'propertytype_std_fee',
    'office_avg_fee', 'segment_win_rate'
]

# Candidate unused features
UNUSED_CANDIDATES = [
    # Property characteristics
    'GrossBuildingAreaRange',
    'YearBuiltRange',
    'PropertyType',
    'SubType',

    # Distance/location
    'DistanceInMiles',
    'DistanceInKM',
    'RooftopLongitude',
    'RooftopLatitude',

    # Job/bid characteristics
    'JobCount',
    'IECount',
    'LeaseCount',
    'SaleCount',
    'DeliveryTotal',

    # Demographics/market
    'PopulationEstimate',
    'AverageHouseValue',
    'IncomePerHousehold',
    'MedianAge',
    'NumberofBusinesses',
    'NumberofEmployees',
    'ZipPopulation',

    # Business segment details
    'BusinessSegment',
    'BusinessSegmentDetail',
    'Market',
    'Submarket',

    # Office
    'OfficeId',
    'OfficeCode'
]

# Check which exist and have data
available_unused = []
for feat in UNUSED_CANDIDATES:
    if feat in df_recent.columns:
        non_null_pct = (1 - df_recent[feat].isna().mean()) * 100
        if non_null_pct > 20:  # At least 20% non-null
            available_unused.append({'feature': feat, 'non_null_pct': non_null_pct, 'dtype': str(df_recent[feat].dtype)})

available_df = pd.DataFrame(available_unused).sort_values('non_null_pct', ascending=False)

print(f"\nAvailable unused features ({len(available_df)}):")
print(f"\n{'Feature':<35} {'Non-null %':<12} {'Type'}")
print("-" * 60)
for _, row in available_df.iterrows():
    print(f"{row['feature']:<35} {row['non_null_pct']:<11.1f}% {row['dtype']}")

# ============================================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("ENCODING FEATURES")
print("=" * 80)

# Features to encode
features_to_use = []

# 1. Encode GrossBuildingAreaRange
if 'GrossBuildingAreaRange' in df_recent.columns:
    gba_mapping = {
        'Missing/Invalid': 0,
        '<10K': 1,
        '<25K': 2,
        '10K-25K': 3,
        '25K-50K': 4,
        '50K-100K': 5,
        '100K-250K': 6,
        '250K-500K': 7,
        '500K-1MM': 8,
        '>1MM': 9
    }
    df_recent['GBA_encoded'] = df_recent['GrossBuildingAreaRange'].map(gba_mapping).fillna(0)
    features_to_use.append('GBA_encoded')
    print(f"Encoded GrossBuildingAreaRange")

# 2. Encode YearBuiltRange
if 'YearBuiltRange' in df_recent.columns:
    year_mapping = {
        'Missing': 0,
        '<1950': 1,
        '1950 - 1964': 2,
        '1965 - 1979': 3,
        '1980 - 1994': 4,
        '1995 - 2009': 5,
        '>2009': 6
    }
    df_recent['YearBuilt_encoded'] = df_recent['YearBuiltRange'].map(year_mapping).fillna(0)
    features_to_use.append('YearBuilt_encoded')
    print(f"Encoded YearBuiltRange")

# 3. Numeric features - use as is
numeric_features = [
    'DistanceInMiles',
    'JobCount',
    'IECount',
    'DeliveryTotal',
    'PopulationEstimate',
    'AverageHouseValue',
    'IncomePerHousehold',
    'MedianAge',
    'NumberofBusinesses',
    'NumberofEmployees'
]

for feat in numeric_features:
    if feat in df_recent.columns:
        features_to_use.append(feat)

print(f"Using {len(numeric_features)} numeric features")

# 4. Categorical features - label encode
categorical_to_encode = ['PropertyType', 'SubType', 'BusinessSegment', 'Market']
for feat in categorical_to_encode:
    if feat in df_recent.columns:
        le = LabelEncoder()
        df_recent[f'{feat}_encoded'] = le.fit_transform(df_recent[feat].fillna('Unknown'))
        features_to_use.append(f'{feat}_encoded')

print(f"Encoded {len(categorical_to_encode)} categorical features")

print(f"\nTotal new features: {len(features_to_use)}")

# ============================================================================
# TEST 1: UNUSED FEATURES ONLY
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: UNUSED FEATURES ONLY")
print("=" * 80)

X_unused = df_recent[features_to_use].fillna(0).values
y = df_recent[TARGET_COLUMN].values

# 80/20 split
split_idx = int(len(X_unused) * 0.8)
X_train_unused, X_test_unused = X_unused[:split_idx], X_unused[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

train_data_unused = lgb.Dataset(X_train_unused, label=y_train, feature_name=features_to_use)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': RANDOM_SEED,
    'verbose': -1
}

print(f"Training with {len(features_to_use)} unused features...")
model_unused = lgb.train(
    params,
    train_data_unused,
    num_boost_round=1000,
    valid_sets=[train_data_unused],
    callbacks=[lgb.early_stopping(50)]
)

y_pred_train_unused = model_unused.predict(X_train_unused, num_iteration=model_unused.best_iteration)
y_pred_test_unused = model_unused.predict(X_test_unused, num_iteration=model_unused.best_iteration)

train_rmse_unused = np.sqrt(mean_squared_error(y_train, y_pred_train_unused))
test_rmse_unused = np.sqrt(mean_squared_error(y_test, y_pred_test_unused))
test_r2_unused = r2_score(y_test, y_pred_test_unused)
overfitting_unused = test_rmse_unused / train_rmse_unused

print(f"\nRESULTS:")
print(f"  Train RMSE: ${train_rmse_unused:,.2f}")
print(f"  Test RMSE: ${test_rmse_unused:,.2f}")
print(f"  Test R²: {test_r2_unused:.4f}")
print(f"  Overfitting: {overfitting_unused:.2f}x")

baseline_rmse = 237.57
improvement_unused = ((baseline_rmse - test_rmse_unused) / baseline_rmse) * 100
print(f"\nvs Baseline: {improvement_unused:+.1f}%")

# Feature importance
importance_unused = model_unused.feature_importance(importance_type='gain')
importance_df_unused = pd.DataFrame({
    'feature': features_to_use,
    'importance': importance_unused,
    'importance_pct': (importance_unused / importance_unused.sum() * 100)
}).sort_values('importance', ascending=False)

print(f"\nTop 10 unused features by importance:")
for i, row in enumerate(importance_df_unused.head(10).itertuples(), 1):
    print(f"  {i:2d}. {row.feature:35s} {row.importance_pct:6.2f}%")

# ============================================================================
# TEST 2: CURRENT + UNUSED FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: CURRENT + UNUSED FEATURES COMBINED")
print("=" * 80)

ALL_FEATURES = CURRENT_FEATURES + features_to_use

X_combined = df_recent[ALL_FEATURES].fillna(0).values
X_train_combined = X_combined[:split_idx]
X_test_combined = X_combined[split_idx:]

train_data_combined = lgb.Dataset(X_train_combined, label=y_train, feature_name=ALL_FEATURES)

print(f"Training with {len(ALL_FEATURES)} features ({len(CURRENT_FEATURES)} current + {len(features_to_use)} unused)...")
model_combined = lgb.train(
    params,
    train_data_combined,
    num_boost_round=1000,
    valid_sets=[train_data_combined],
    callbacks=[lgb.early_stopping(50)]
)

y_pred_train_combined = model_combined.predict(X_train_combined, num_iteration=model_combined.best_iteration)
y_pred_test_combined = model_combined.predict(X_test_combined, num_iteration=model_combined.best_iteration)

train_rmse_combined = np.sqrt(mean_squared_error(y_train, y_pred_train_combined))
test_rmse_combined = np.sqrt(mean_squared_error(y_test, y_pred_test_combined))
test_r2_combined = r2_score(y_test, y_pred_test_combined)
overfitting_combined = test_rmse_combined / train_rmse_combined

print(f"\nRESULTS:")
print(f"  Train RMSE: ${train_rmse_combined:,.2f}")
print(f"  Test RMSE: ${test_rmse_combined:,.2f}")
print(f"  Test R²: {test_r2_combined:.4f}")
print(f"  Overfitting: {overfitting_combined:.2f}x")

improvement_combined = ((baseline_rmse - test_rmse_combined) / baseline_rmse) * 100
print(f"\nvs Baseline: {improvement_combined:+.1f}%")

# Feature importance
importance_combined = model_combined.feature_importance(importance_type='gain')
importance_df_combined = pd.DataFrame({
    'feature': ALL_FEATURES,
    'importance': importance_combined,
    'importance_pct': (importance_combined / importance_combined.sum() * 100)
}).sort_values('importance', ascending=False)

print(f"\nTop 15 features (current + unused):")
for i, row in enumerate(importance_df_combined.head(15).itertuples(), 1):
    source = "CURRENT" if row.feature in CURRENT_FEATURES else "NEW"
    print(f"  {i:2d}. {row.feature:35s} {row.importance_pct:6.2f}%  [{source}]")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

print(f"\n{'Model':<35} {'Features':<12} {'Test RMSE':<15} {'vs Baseline':<12} {'Overfitting'}")
print("-" * 90)
print(f"{'Baseline (current features)':<35} {len(CURRENT_FEATURES):<12} ${baseline_rmse:<14,.2f} {'---':<12} 2.57x")
print(f"{'Unused features only':<35} {len(features_to_use):<12} ${test_rmse_unused:<14,.2f} {improvement_unused:+6.1f}%{'':6s} {overfitting_unused:.2f}x")
print(f"{'Current + Unused':<35} {len(ALL_FEATURES):<12} ${test_rmse_combined:<14,.2f} {improvement_combined:+6.1f}%{'':6s} {overfitting_combined:.2f}x")

# Determine best approach
best_rmse = min(baseline_rmse, test_rmse_unused, test_rmse_combined)
if best_rmse == test_rmse_combined:
    best_model = "Current + Unused"
    best_features = ALL_FEATURES
    best_improvement = improvement_combined
elif best_rmse == test_rmse_unused:
    best_model = "Unused features only"
    best_features = features_to_use
    best_improvement = improvement_unused
else:
    best_model = "Baseline (no improvement)"
    best_features = CURRENT_FEATURES
    best_improvement = 0.0

print(f"\nBEST: {best_model}")
if best_improvement > 0:
    print(f"Improvement: {best_improvement:.1f}%")
else:
    print("No improvement achieved")

# Save results
results = {
    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "baseline_rmse": float(baseline_rmse),
    "unused_features_count": len(features_to_use),
    "unused_features": features_to_use,
    "results": {
        "unused_only": {
            "test_rmse": float(test_rmse_unused),
            "test_r2": float(test_r2_unused),
            "overfitting": float(overfitting_unused),
            "improvement_pct": float(improvement_unused)
        },
        "current_plus_unused": {
            "test_rmse": float(test_rmse_combined),
            "test_r2": float(test_r2_combined),
            "overfitting": float(overfitting_combined),
            "improvement_pct": float(improvement_combined)
        }
    },
    "best_model": best_model,
    "best_features": best_features if best_improvement > 0 else None,
    "top_features": importance_df_combined.head(15).to_dict('records') if test_rmse_combined < baseline_rmse else None
}

results_path = REPORTS_DIR / "unused_features_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved: {results_path}")
print("\n" + "=" * 80)
