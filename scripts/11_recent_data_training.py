"""
Regression Model with Optimized Features (Phase 3)
===================================================
Train bid fee prediction model using 11 regression-optimized features
identified through dual feature selection

Features selected specifically for REGRESSION task (not classification)

Author: Bid Recommendation System
Date: 2026-01-09 (Phase 3 - Optimized)
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

# Load regression-optimized features from dual selection
REGRESSION_FEATURES_PATH = MODELS_DIR.parent / "reports" / "regression_features.json"
with open(REGRESSION_FEATURES_PATH, 'r') as f:
    regression_config = json.load(f)
    SELECTED_FEATURES = regression_config['features']

# REMOVE segment_avg_fee (circular/leakage feature - 67% importance)
SELECTED_FEATURES = [f for f in SELECTED_FEATURES if f != 'segment_avg_fee']

print("=" * 80)
print("REGRESSION MODEL - WITHOUT segment_avg_fee")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Using {len(SELECTED_FEATURES)} features (removed segment_avg_fee)")
print(f"Reason: segment_avg_fee is circular (uses avg fee to predict fee)\n")

# Load data
df = pd.read_csv(FEATURES_DATA)
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

# Filter to recent data only (2023-2025)
recent_cutoff = pd.Timestamp('2023-01-01')
df_recent = df[df[DATE_COLUMN] >= recent_cutoff].copy()

print(f"✓ Data loaded: {len(df_recent):,} rows (2023-2025)")
print(f"  Features: {len(SELECTED_FEATURES)}")
print(f"\nSelected features for REGRESSION:")
for i, feat in enumerate(SELECTED_FEATURES, 1):
    print(f"  {i:2d}. {feat}")

# Prepare data
X_recent = df_recent[SELECTED_FEATURES].fillna(0).values
y_recent = df_recent[TARGET_COLUMN].values

# 80/20 split on recent data
split_idx = int(len(X_recent) * 0.8)
X_train = X_recent[:split_idx]
X_test = X_recent[split_idx:]
y_train = y_recent[:split_idx]
y_test = y_recent[split_idx:]

print(f"\nTrain: {len(X_train):,} samples")
print(f"Test: {len(X_test):,} samples\n")

# Train model
print("=" * 80)
print("TRAINING LIGHTGBM REGRESSOR")
print("=" * 80)

train_data = lgb.Dataset(X_train, label=y_train, feature_name=SELECTED_FEATURES)

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
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data],
    callbacks=[lgb.early_stopping(50)]
)

print(f"✓ Model trained (best iteration: {model.best_iteration})\n")

# Predictions
y_pred_train = model.predict(X_train, num_iteration=model.best_iteration)
y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)

# Evaluation
print("=" * 80)
print("EVALUATION")
print("=" * 80)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_mae = mean_absolute_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print("\nTRAIN METRICS:")
print(f"  RMSE: ${train_rmse:,.2f}")
print(f"  MAE: ${train_mae:,.2f}")
print(f"  R²: {train_r2:.4f}")

print("\nTEST METRICS:")
print(f"  RMSE: ${test_rmse:,.2f}")
print(f"  MAE: ${test_mae:,.2f}")
print(f"  R²: {test_r2:.4f}")

overfitting_ratio = test_rmse / train_rmse
print(f"\nOVERFITTING RATIO: {overfitting_ratio:.2f}x")

# Feature importance
importance = model.feature_importance(importance_type='gain')
importance_df = pd.DataFrame({
    'feature': SELECTED_FEATURES,
    'importance': importance,
    'importance_pct': (importance / importance.sum() * 100)
}).sort_values('importance', ascending=False)

print(f"\nFEATURE IMPORTANCE:")
for i, row in enumerate(importance_df.itertuples(), 1):
    print(f"  {i:2d}. {row.feature:40s} {row.importance_pct:6.2f}%")

# Comparison with baselines
print("\n" + "=" * 80)
print("COMPARISON WITH PREVIOUS MODELS")
print("=" * 80)

baselines = {
    "Previous (12 features, all data)": {"test_rmse": 263.23, "overfitting": 2.64},
    "Previous (12 features, 2023+ data)": {"test_rmse": 238.03, "overfitting": 2.51},
    "Dual Selection (81 features)": {"test_rmse": 276.90, "overfitting": 1.68},
}

print(f"\n{'Model':<40} {'Test RMSE':<15} {'Overfitting':<15} {'Status'}")
print("-" * 80)
for name, metrics in baselines.items():
    print(f"{name:<40} ${metrics['test_rmse']:<14,.2f} {metrics['overfitting']:<14.2f}x")

print(f"{'NEW: Optimized (11 features)':<40} ${test_rmse:<14,.2f} {overfitting_ratio:<14.2f}x {'← CURRENT'}")

# Determine improvement
best_baseline_rmse = min(m['test_rmse'] for m in baselines.values())
improvement_pct = ((best_baseline_rmse - test_rmse) / best_baseline_rmse) * 100

if test_rmse < best_baseline_rmse:
    print(f"\n✓ IMPROVEMENT: {improvement_pct:.1f}% better than best baseline")
elif test_rmse < 250:
    print(f"\n✓ STRONG PERFORMANCE: Test RMSE under $250")
else:
    print(f"\n⚠ Test RMSE: ${test_rmse:.2f} vs best baseline ${best_baseline_rmse:.2f}")

# Save model
print("\n" + "=" * 80)
print("SAVING MODEL")
print("=" * 80)

model_path = MODELS_DIR / "lightgbm_bidfee_optimized.txt"
model.save_model(str(model_path))

metadata = {
    "model_type": "LightGBM Regression (Phase 3 - Optimized)",
    "phase": "1A - Bid Fee Prediction",
    "target_variable": "BidFee",
    "optimization": "Dual feature selection (regression-specific)",
    "num_features": len(SELECTED_FEATURES),
    "selected_features": SELECTED_FEATURES,
    "data_range": {
        "start_date": df_recent[DATE_COLUMN].min().strftime('%Y-%m-%d'),
        "end_date": df_recent[DATE_COLUMN].max().strftime('%Y-%m-%d'),
        "total_samples": int(len(df_recent)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test))
    },
    "best_iteration": int(model.best_iteration),
    "parameters": params,
    "metrics": {
        "train": {
            "rmse": float(train_rmse),
            "mae": float(train_mae),
            "r2": float(train_r2)
        },
        "test": {
            "rmse": float(test_rmse),
            "mae": float(test_mae),
            "r2": float(test_r2)
        },
        "overfitting_ratio": float(overfitting_ratio)
    },
    "feature_importance": importance_df.to_dict('records'),
    "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "comparison": {
        "improvement_vs_best_baseline_pct": float(improvement_pct),
        "best_baseline_rmse": float(best_baseline_rmse)
    }
}

metadata_path = MODELS_DIR / "lightgbm_bidfee_optimized_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Model saved: {model_path}")
print(f"✓ Metadata saved: {metadata_path}")

print("\n" + "=" * 80)
print("PHASE 3: REGRESSION MODEL COMPLETE")
print("=" * 80)
print(f"✓ Features: {len(SELECTED_FEATURES)} (regression-optimized)")
print(f"✓ Test RMSE: ${test_rmse:,.2f}")
print(f"✓ Test R²: {test_r2:.4f}")
print(f"✓ Overfitting: {overfitting_ratio:.2f}x\n")
