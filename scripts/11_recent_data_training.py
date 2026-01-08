"""
Train Model with Recent Data Only (2023-2025)
==============================================
Address temporal drift by using only recent 2-3 years of data

Hypothesis: Older data (2018-2022) may hurt performance due to:
- Market changes (COVID, economic shifts)
- Different bidding patterns
- Outdated business relationships

Author: Bid Recommendation System
Date: 2026-01-08
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
    MODELS_DIR, RANDOM_SEED,
)

warnings.filterwarnings('ignore')

# Load selected features
with open(MODELS_DIR / "lightgbm_metadata_feature_selected.json", 'r') as f:
    metadata = json.load(f)
    SELECTED_FEATURES = metadata['selected_features']

print("=" * 80)
print("TRAINING WITH RECENT DATA ONLY (2023-2025)")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Strategy: Use only last 2-3 years to avoid temporal drift\n")

# Load data
df = pd.read_csv(FEATURES_DATA)
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

print(f"Full dataset: {len(df):,} rows")
print(f"  Date range: {df[DATE_COLUMN].min()} to {df[DATE_COLUMN].max()}")

# Filter to recent data only (2023-2025)
recent_cutoff = pd.Timestamp('2023-01-01')
df_recent = df[df[DATE_COLUMN] >= recent_cutoff].copy()

print(f"\nRecent data (2023+): {len(df_recent):,} rows")
print(f"  Date range: {df_recent[DATE_COLUMN].min()} to {df_recent[DATE_COLUMN].max()}")
print(f"  Reduction: {(1 - len(df_recent)/len(df)) * 100:.1f}% of data removed\n")

# Prepare data
X_recent = df_recent[SELECTED_FEATURES].fillna(0).values
y_recent = df_recent[TARGET_COLUMN].values

# 80/20 split on recent data
split_idx = int(len(X_recent) * 0.8)
X_train = X_recent[:split_idx]
X_test = X_recent[split_idx:]
y_train = y_recent[:split_idx]
y_test = y_recent[split_idx:]

print(f"Training set: {len(X_train):,} samples")
print(f"Test set: {len(X_test):,} samples\n")

# Compare with baseline (all data)
print("=" * 80)
print("BASELINE: MODEL TRAINED ON ALL DATA (2018-2025)")
print("=" * 80)

X_all = df[SELECTED_FEATURES].fillna(0).values
y_all = df[TARGET_COLUMN].values
split_all = int(len(X_all) * 0.8)

train_data_all = lgb.Dataset(X_all[:split_all], label=y_all[:split_all])
val_data_all = lgb.Dataset(X_all[split_all:], label=y_all[split_all:], reference=train_data_all)

baseline_params = {
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

model_baseline = lgb.train(
    baseline_params,
    train_data_all,
    num_boost_round=1000,
    valid_sets=[val_data_all],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

y_pred_base = model_baseline.predict(X_all[split_all:], num_iteration=model_baseline.best_iteration)
y_train_pred_base = model_baseline.predict(X_all[:split_all], num_iteration=model_baseline.best_iteration)

baseline_train_rmse = np.sqrt(mean_squared_error(y_all[:split_all], y_train_pred_base))
baseline_test_rmse = np.sqrt(mean_squared_error(y_all[split_all:], y_pred_base))
baseline_test_mae = mean_absolute_error(y_all[split_all:], y_pred_base)
baseline_test_r2 = r2_score(y_all[split_all:], y_pred_base)
baseline_overfitting = baseline_test_rmse / baseline_train_rmse

print(f"Baseline (all data 2018-2025):")
print(f"  Train RMSE: ${baseline_train_rmse:,.2f}")
print(f"  Test RMSE:  ${baseline_test_rmse:,.2f}")
print(f"  Test MAE:   ${baseline_test_mae:,.2f}")
print(f"  Test R²:    {baseline_test_r2:.4f}")
print(f"  Overfitting: {baseline_overfitting:.2f}x\n")

# Train on recent data only
print("=" * 80)
print("NEW MODEL: TRAINED ON RECENT DATA (2023-2025)")
print("=" * 80)

train_data_recent = lgb.Dataset(X_train, label=y_train, feature_name=SELECTED_FEATURES)
val_data_recent = lgb.Dataset(X_test, label=y_test, reference=train_data_recent)

print("Training with recent data...")
model_recent = lgb.train(
    baseline_params,
    train_data_recent,
    num_boost_round=1000,
    valid_sets=[train_data_recent, val_data_recent],
    valid_names=['train', 'valid'],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# Evaluate
y_pred_recent = model_recent.predict(X_test, num_iteration=model_recent.best_iteration)
y_train_pred_recent = model_recent.predict(X_train, num_iteration=model_recent.best_iteration)

recent_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_recent))
recent_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_recent))
recent_test_mae = mean_absolute_error(y_test, y_pred_recent)
recent_test_r2 = r2_score(y_test, y_pred_recent)
recent_overfitting = recent_test_rmse / recent_train_rmse

print(f"\n✓ Model trained on recent data")
print(f"\nRecent Model (2023-2025 only):")
print(f"  Train RMSE: ${recent_train_rmse:,.2f}")
print(f"  Test RMSE:  ${recent_test_rmse:,.2f}")
print(f"  Test MAE:   ${recent_test_mae:,.2f}")
print(f"  Test R²:    {recent_test_r2:.4f}")
print(f"  Overfitting: {recent_overfitting:.2f}x\n")

# Comparison
print("=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"\n{'Metric':<20} {'All Data':<15} {'Recent Data':<15} {'Change'}")
print("-" * 70)
print(f"{'Train RMSE':<20} ${baseline_train_rmse:<14,.2f} ${recent_train_rmse:<14,.2f} {(recent_train_rmse - baseline_train_rmse)/baseline_train_rmse*100:+.1f}%")
print(f"{'Test RMSE':<20} ${baseline_test_rmse:<14,.2f} ${recent_test_rmse:<14,.2f} {(recent_test_rmse - baseline_test_rmse)/baseline_test_rmse*100:+.1f}%")
print(f"{'Test MAE':<20} ${baseline_test_mae:<14,.2f} ${recent_test_mae:<14,.2f} {(recent_test_mae - baseline_test_mae)/baseline_test_mae*100:+.1f}%")
print(f"{'Test R²':<20} {baseline_test_r2:<15.4f} {recent_test_r2:<15.4f} {(recent_test_r2 - baseline_test_r2):+.4f}")
print(f"{'Overfitting Ratio':<20} {baseline_overfitting:<15.2f} {recent_overfitting:<15.2f} {(recent_overfitting - baseline_overfitting)/baseline_overfitting*100:+.1f}%")

# Determine if recent data approach is better
rmse_improvement = (baseline_test_rmse - recent_test_rmse) / baseline_test_rmse * 100
overfitting_improvement = (baseline_overfitting - recent_overfitting) / baseline_overfitting * 100

print(f"\nIMPROVEMENT SUMMARY:")
print(f"  Test RMSE: {rmse_improvement:+.1f}%")
print(f"  Overfitting: {overfitting_improvement:+.1f}%")

if recent_test_rmse < baseline_test_rmse and recent_overfitting < baseline_overfitting:
    print(f"\n✅ RECENT DATA MODEL IS BETTER - Use this for production!")
    best_model = model_recent
    best_type = "recent"
elif recent_test_rmse < baseline_test_rmse:
    print(f"\n⚠️  MIXED RESULTS - Recent model has better test performance but similar/worse overfitting")
    best_model = model_recent
    best_type = "recent"
else:
    print(f"\n❌ ALL DATA MODEL IS BETTER - Recent data alone doesn't help")
    best_model = model_baseline
    best_type = "all_data"

# Save the better model
if best_type == "recent":
    model_path = MODELS_DIR / "lightgbm_bidfee_model_recent_data.txt"
    best_model.save_model(str(model_path))

    metadata_save = {
        "model_type": "LightGBM (Recent Data 2023-2025)",
        "phase": "1A - Bid Fee Prediction",
        "target_variable": TARGET_COLUMN,
        "num_features": len(SELECTED_FEATURES),
        "selected_features": SELECTED_FEATURES,
        "data_range": {
            "start_date": df_recent[DATE_COLUMN].min().strftime('%Y-%m-%d'),
            "end_date": df_recent[DATE_COLUMN].max().strftime('%Y-%m-%d'),
            "total_samples": int(len(df_recent)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test))
        },
        "best_iteration": int(model_recent.best_iteration),
        "parameters": baseline_params,
        "metrics": {
            "train_rmse": float(recent_train_rmse),
            "test_rmse": float(recent_test_rmse),
            "test_mae": float(recent_test_mae),
            "test_r2": float(recent_test_r2),
            "overfitting_ratio": float(recent_overfitting)
        },
        "comparison_to_all_data": {
            "all_data_test_rmse": float(baseline_test_rmse),
            "recent_data_test_rmse": float(recent_test_rmse),
            "test_rmse_improvement_pct": float(rmse_improvement),
            "all_data_overfitting": float(baseline_overfitting),
            "recent_data_overfitting": float(recent_overfitting),
            "overfitting_improvement_pct": float(overfitting_improvement)
        },
        "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    metadata_path = MODELS_DIR / "lightgbm_metadata_recent_data.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata_save, f, indent=2)

    print(f"\n✓ Recent data model saved: {model_path}")
    print(f"✓ Metadata saved: {metadata_path}")

print("\n" + "=" * 80)
print("RECENT DATA TRAINING COMPLETE")
print("=" * 80)
