"""
Feature Engineering Round 2
============================
Create advanced features to improve regression baseline

Current baseline: $237.57 RMSE, 2.57x overfitting, 11 features

New features:
- Interaction features
- Ratio features
- Non-linear transformations
- Market position features
- Temporal features

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
import warnings

from config.model_config import (
    FEATURES_DATA, TARGET_COLUMN, DATE_COLUMN,
    MODELS_DIR, REPORTS_DIR, RANDOM_SEED,
)

warnings.filterwarnings('ignore')

print("=" * 80)
print("FEATURE ENGINEERING ROUND 2")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Baseline: $237.57 RMSE, 2.57x overfitting, 11 features\n")

# Load data
df = pd.read_csv(FEATURES_DATA)
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

# Filter to recent data
recent_cutoff = pd.Timestamp('2023-01-01')
df_recent = df[df[DATE_COLUMN] >= recent_cutoff].copy()

print(f"Data: {len(df_recent):,} rows (2023-2025)\n")

# Load existing regression features
REGRESSION_FEATURES_PATH = REPORTS_DIR / "regression_features.json"
with open(REGRESSION_FEATURES_PATH, 'r') as f:
    regression_config = json.load(f)
    EXISTING_FEATURES = regression_config['features']

print(f"Existing features: {len(EXISTING_FEATURES)}")

# ============================================================================
# CREATE NEW FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("CREATING NEW FEATURES")
print("=" * 80)

new_feature_count = 0

# 1. INTERACTION FEATURES
print("\n1. Interaction features...")
df_recent['segment_state_interaction'] = df_recent['segment_avg_fee'] * df_recent['state_avg_fee']
df_recent['segment_property_interaction'] = df_recent['segment_avg_fee'] * df_recent['propertytype_avg_fee']
df_recent['state_property_interaction'] = df_recent['state_avg_fee'] * df_recent['propertytype_avg_fee']
new_feature_count += 3

# 2. RATIO FEATURES
print("2. Ratio features...")
df_recent['segment_vs_state_ratio'] = df_recent['segment_avg_fee'] / (df_recent['state_avg_fee'] + 1)
df_recent['segment_vs_property_ratio'] = df_recent['segment_avg_fee'] / (df_recent['propertytype_avg_fee'] + 1)
df_recent['segment_vs_client_ratio'] = df_recent['segment_avg_fee'] / (df_recent['client_avg_fee'] + 1)
df_recent['rolling_vs_segment_ratio'] = df_recent['rolling_avg_fee_segment'] / (df_recent['segment_avg_fee'] + 1)
new_feature_count += 4

# 3. COEFFICIENT OF VARIATION (std / mean)
print("3. Coefficient of variation features...")
df_recent['segment_cv'] = df_recent['segment_std_fee'] / (df_recent['segment_avg_fee'] + 1)
df_recent['state_cv'] = df_recent['state_std_fee'] / (df_recent['state_avg_fee'] + 1)
df_recent['property_cv'] = df_recent['propertytype_std_fee'] / (df_recent['propertytype_avg_fee'] + 1)
new_feature_count += 3

# 4. NON-LINEAR TRANSFORMATIONS
print("4. Non-linear transformations...")
df_recent['segment_avg_fee_log'] = np.log1p(df_recent['segment_avg_fee'])
df_recent['state_avg_fee_log'] = np.log1p(df_recent['state_avg_fee'])
df_recent['segment_avg_fee_sqrt'] = np.sqrt(df_recent['segment_avg_fee'])
df_recent['segment_avg_fee_sq'] = df_recent['segment_avg_fee'] ** 2
new_feature_count += 4

# 5. MARKET POSITION FEATURES (percentile rankings)
print("5. Market position features...")
df_recent['segment_fee_percentile'] = df_recent.groupby('BusinessSegment_encoded')['segment_avg_fee'].rank(pct=True)
df_recent['state_fee_percentile'] = df_recent.groupby('PropertyState_encoded')['state_avg_fee'].rank(pct=True)
new_feature_count += 2

# 6. DEVIATION FROM AVERAGE
print("6. Deviation features...")
df_recent['segment_deviation'] = df_recent['segment_avg_fee'] - df_recent['state_avg_fee']
df_recent['rolling_deviation'] = df_recent['rolling_avg_fee_segment'] - df_recent['segment_avg_fee']
new_feature_count += 2

# 7. TIME-BASED FEATURES
print("7. Time-based features...")
df_recent['TargetTime_log'] = np.log1p(df_recent['TargetTime'])
df_recent['TargetTime_sqrt'] = np.sqrt(df_recent['TargetTime'])
new_feature_count += 2

print(f"\nCreated {new_feature_count} new features")

# ============================================================================
# COMBINE WITH EXISTING FEATURES
# ============================================================================

NEW_FEATURES = [
    # Interactions
    'segment_state_interaction',
    'segment_property_interaction',
    'state_property_interaction',
    # Ratios
    'segment_vs_state_ratio',
    'segment_vs_property_ratio',
    'segment_vs_client_ratio',
    'rolling_vs_segment_ratio',
    # Coefficient of variation
    'segment_cv',
    'state_cv',
    'property_cv',
    # Non-linear
    'segment_avg_fee_log',
    'state_avg_fee_log',
    'segment_avg_fee_sqrt',
    'segment_avg_fee_sq',
    # Market position
    'segment_fee_percentile',
    'state_fee_percentile',
    # Deviation
    'segment_deviation',
    'rolling_deviation',
    # Time
    'TargetTime_log',
    'TargetTime_sqrt'
]

# Combined feature set
ALL_FEATURES = EXISTING_FEATURES + NEW_FEATURES

print(f"\nTotal features: {len(ALL_FEATURES)} ({len(EXISTING_FEATURES)} existing + {len(NEW_FEATURES)} new)")

# ============================================================================
# PREPARE DATA
# ============================================================================

X = df_recent[ALL_FEATURES].fillna(0).values
y = df_recent[TARGET_COLUMN].values

# 80/20 split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

# ============================================================================
# TRAIN MODEL WITH ALL FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING MODEL WITH ALL FEATURES")
print("=" * 80)

train_data = lgb.Dataset(X_train, label=y_train, feature_name=ALL_FEATURES)

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

print("Training...")
model_all = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data],
    callbacks=[lgb.early_stopping(50)]
)

print(f"Best iteration: {model_all.best_iteration}\n")

# Predictions
y_pred_train_all = model_all.predict(X_train, num_iteration=model_all.best_iteration)
y_pred_test_all = model_all.predict(X_test, num_iteration=model_all.best_iteration)

# Evaluation
train_rmse_all = np.sqrt(mean_squared_error(y_train, y_pred_train_all))
test_rmse_all = np.sqrt(mean_squared_error(y_test, y_pred_test_all))
test_r2_all = r2_score(y_test, y_pred_test_all)
overfitting_all = test_rmse_all / train_rmse_all

print("RESULTS WITH ALL FEATURES:")
print(f"  Train RMSE: ${train_rmse_all:,.2f}")
print(f"  Test RMSE: ${test_rmse_all:,.2f}")
print(f"  Test R²: {test_r2_all:.4f}")
print(f"  Overfitting: {overfitting_all:.2f}x")

# ============================================================================
# FEATURE SELECTION ON NEW SET
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE SELECTION ON EXPANDED SET")
print("=" * 80)

# Get feature importance
importance_all = model_all.feature_importance(importance_type='gain')
importance_df_all = pd.DataFrame({
    'feature': ALL_FEATURES,
    'importance': importance_all,
    'importance_pct': (importance_all / importance_all.sum() * 100)
}).sort_values('importance', ascending=False)

# Select top features (95% cumulative importance)
importance_df_all['cumulative_pct'] = importance_df_all['importance_pct'].cumsum()
selected_features_v2 = importance_df_all[importance_df_all['cumulative_pct'] <= 95.0]['feature'].tolist()

# Always include at least top 15 features
if len(selected_features_v2) < 15:
    selected_features_v2 = importance_df_all.head(15)['feature'].tolist()

print(f"\nSelected {len(selected_features_v2)} features (95% importance)")
print("\nTop 20 features:")
for i, row in enumerate(importance_df_all.head(20).itertuples(), 1):
    marker = "✓" if row.feature in selected_features_v2 else " "
    print(f"{marker} {i:2d}. {row.feature:45s} {row.importance_pct:6.2f}%")

# ============================================================================
# TRAIN MODEL WITH SELECTED FEATURES V2
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING MODEL WITH SELECTED FEATURES V2")
print("=" * 80)

X_train_v2 = df_recent[selected_features_v2].iloc[:split_idx].fillna(0).values
X_test_v2 = df_recent[selected_features_v2].iloc[split_idx:].fillna(0).values

train_data_v2 = lgb.Dataset(X_train_v2, label=y_train, feature_name=selected_features_v2)

print(f"Training with {len(selected_features_v2)} features...")
model_v2 = lgb.train(
    params,
    train_data_v2,
    num_boost_round=1000,
    valid_sets=[train_data_v2],
    callbacks=[lgb.early_stopping(50)]
)

print(f"Best iteration: {model_v2.best_iteration}\n")

# Predictions
y_pred_train_v2 = model_v2.predict(X_train_v2, num_iteration=model_v2.best_iteration)
y_pred_test_v2 = model_v2.predict(X_test_v2, num_iteration=model_v2.best_iteration)

# Evaluation
train_rmse_v2 = np.sqrt(mean_squared_error(y_train, y_pred_train_v2))
train_mae_v2 = mean_absolute_error(y_train, y_pred_train_v2)
train_r2_v2 = r2_score(y_train, y_pred_train_v2)

test_rmse_v2 = np.sqrt(mean_squared_error(y_test, y_pred_test_v2))
test_mae_v2 = mean_absolute_error(y_test, y_pred_test_v2)
test_r2_v2 = r2_score(y_test, y_pred_test_v2)

overfitting_v2 = test_rmse_v2 / train_rmse_v2

print("RESULTS WITH SELECTED FEATURES V2:")
print(f"  Train RMSE: ${train_rmse_v2:,.2f}")
print(f"  Test RMSE: ${test_rmse_v2:,.2f}")
print(f"  Test R²: {test_r2_v2:.4f}")
print(f"  Overfitting: {overfitting_v2:.2f}x")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

baseline_rmse = 237.57
baseline_overfitting = 2.57
baseline_features = 11

improvement_rmse = ((baseline_rmse - test_rmse_v2) / baseline_rmse) * 100
improvement_overfitting = ((baseline_overfitting - overfitting_v2) / baseline_overfitting) * 100

print(f"\n{'Model':<35} {'Features':<12} {'Test RMSE':<15} {'Overfitting'}")
print("-" * 80)
print(f"{'Baseline':<35} {baseline_features:<12} ${baseline_rmse:<14,.2f} {baseline_overfitting:.2f}x")
print(f"{'All features':<35} {len(ALL_FEATURES):<12} ${test_rmse_all:<14,.2f} {overfitting_all:.2f}x")
print(f"{'Selected V2':<35} {len(selected_features_v2):<12} ${test_rmse_v2:<14,.2f} {overfitting_v2:.2f}x")

if test_rmse_v2 < baseline_rmse:
    print(f"\nIMPROVEMENT: {improvement_rmse:.1f}% better RMSE")
else:
    print(f"\nDEGRADATION: {-improvement_rmse:.1f}% worse RMSE")

if overfitting_v2 < baseline_overfitting:
    print(f"IMPROVEMENT: {improvement_overfitting:.1f}% less overfitting")
else:
    print(f"DEGRADATION: {-improvement_overfitting:.1f}% more overfitting")

# ============================================================================
# SAVE MODEL AND FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("SAVING MODEL AND FEATURES")
print("=" * 80)

model_path = MODELS_DIR / "lightgbm_bidfee_v2.txt"
model_v2.save_model(str(model_path))

# Save feature list
features_v2_config = {
    "task": "regression",
    "target": "BidFee",
    "version": 2,
    "num_features": len(selected_features_v2),
    "features": selected_features_v2,
    "new_features_added": NEW_FEATURES,
    "feature_importance": importance_df_all[importance_df_all['feature'].isin(selected_features_v2)].to_dict('records'),
    "metrics": {
        "train_rmse": float(train_rmse_v2),
        "test_rmse": float(test_rmse_v2),
        "test_r2": float(test_r2_v2),
        "overfitting_ratio": float(overfitting_v2)
    },
    "comparison": {
        "baseline_rmse": float(baseline_rmse),
        "baseline_overfitting": float(baseline_overfitting),
        "improvement_rmse_pct": float(improvement_rmse),
        "improvement_overfitting_pct": float(improvement_overfitting)
    },
    "creation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

features_v2_path = REPORTS_DIR / "regression_features_v2.json"
with open(features_v2_path, 'w') as f:
    json.dump(features_v2_config, f, indent=2)

# Save metadata
metadata = {
    "model_type": "LightGBM Regression (Feature Engineering V2)",
    "phase": "1A - Bid Fee Prediction",
    "target_variable": "BidFee",
    "optimization": "Feature engineering round 2 + feature selection",
    "num_features": len(selected_features_v2),
    "selected_features": selected_features_v2,
    "new_features_created": len(NEW_FEATURES),
    "new_features_list": NEW_FEATURES,
    "data_range": {
        "start_date": df_recent[DATE_COLUMN].min().strftime('%Y-%m-%d'),
        "end_date": df_recent[DATE_COLUMN].max().strftime('%Y-%m-%d'),
        "total_samples": int(len(df_recent)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test))
    },
    "best_iteration": int(model_v2.best_iteration),
    "parameters": params,
    "metrics": {
        "train": {
            "rmse": float(train_rmse_v2),
            "mae": float(train_mae_v2),
            "r2": float(train_r2_v2)
        },
        "test": {
            "rmse": float(test_rmse_v2),
            "mae": float(test_mae_v2),
            "r2": float(test_r2_v2)
        },
        "overfitting_ratio": float(overfitting_v2)
    },
    "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "comparison": {
        "baseline_rmse": float(baseline_rmse),
        "baseline_overfitting": float(baseline_overfitting),
        "improvement_rmse_pct": float(improvement_rmse),
        "improvement_overfitting_pct": float(improvement_overfitting)
    }
}

metadata_path = REPORTS_DIR / "feature_engineering_v2_results.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Model saved: {model_path}")
print(f"Features saved: {features_v2_path}")
print(f"Metadata saved: {metadata_path}")

print("\n" + "=" * 80)
print("FEATURE ENGINEERING V2 COMPLETE")
print("=" * 80)
print(f"Features: {len(selected_features_v2)} ({len(NEW_FEATURES)} new)")
print(f"Test RMSE: ${test_rmse_v2:,.2f}")
print(f"Overfitting: {overfitting_v2:.2f}x")
print(f"vs Baseline: ${baseline_rmse:,.2f}, {baseline_overfitting:.2f}x\n")
