"""
Feature Ablation Study
======================
Systematically test model performance by removing low-importance features.

Current model uses 84 features, but some have zero or very low importance.
This study tests if we can simplify by removing bottom 10%, 20%, 30% features.

Goal: Find optimal feature set that maintains performance with fewer features.

Author: Bid Recommendation System
Date: 2026-01-21
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Import config
from config.model_config import (
    FEATURES_DATA, TARGET_COLUMN, DATE_COLUMN, EXCLUDE_COLUMNS,
    LIGHTGBM_CONFIG, REPORTS_DIR
)

print("=" * 80)
print("FEATURE ABLATION STUDY")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# LOAD DATA AND FEATURE IMPORTANCE
# ============================================================================
print("=" * 80)
print("STEP 1: LOAD DATA AND FEATURE IMPORTANCE")
print("=" * 80)

# Load data
df = pd.read_csv(FEATURES_DATA)
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

# Filter to 2023+ (same as current model)
df = df[df[DATE_COLUMN] >= '2023-01-01'].copy()
print(f"\n✓ Data loaded and filtered to 2023+")
print(f"  Records: {len(df):,}")
print(f"  Date range: {df[DATE_COLUMN].min()} to {df[DATE_COLUMN].max()}")

# Load feature importance first to know which features were actually used
fi_df = pd.read_csv(REPORTS_DIR / 'lightgbm_feature_importance.csv')
print(f"\n✓ Feature importance loaded: {len(fi_df)} features")

# Use ONLY the features that were in the trained model
feature_cols = fi_df['feature'].tolist()

# Filter to features that exist in dataset
available_features = [col for col in feature_cols if col in df.columns]

X = df[available_features].fillna(0)
y = df[TARGET_COLUMN]

print(f"\n  Features in importance file: {len(feature_cols)}")
print(f"  Features available in data: {len(available_features)}")
print(f"  Target mean: ${y.mean():,.2f}")

# ============================================================================
# SPLIT DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: TRAIN/VALIDATION/TEST SPLIT")
print("=" * 80)

# Same split as current model: 60/20/20
train_ratio = 0.6
valid_ratio = 0.2

train_idx = int(len(df) * train_ratio)
valid_idx = int(len(df) * (train_ratio + valid_ratio))

X_train = X.iloc[:train_idx]
X_valid = X.iloc[train_idx:valid_idx]
X_test = X.iloc[valid_idx:]

y_train = y.iloc[:train_idx]
y_valid = y.iloc[train_idx:valid_idx]
y_test = y.iloc[valid_idx:]

print(f"\nSplit: 60% train / 20% validation / 20% test")
print(f"  Train: {len(X_train):,} samples")
print(f"  Valid: {len(X_valid):,} samples")
print(f"  Test:  {len(X_test):,} samples")

# ============================================================================
# BASELINE: CURRENT MODEL WITH ALL FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: BASELINE - ALL FEATURES")
print("=" * 80)

print(f"\n📊 Training baseline model with ALL {len(available_features)} features...")

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

params = LIGHTGBM_CONFIG['params']
num_rounds = LIGHTGBM_CONFIG['training']['num_boost_round']
early_stop = LIGHTGBM_CONFIG['training']['early_stopping_rounds']

callbacks = [
    lgb.early_stopping(stopping_rounds=early_stop),
    lgb.log_evaluation(period=0),  # Silent
]

model_baseline = lgb.train(
    params,
    train_data,
    num_boost_round=num_rounds,
    valid_sets=[valid_data],
    callbacks=callbacks,
)

# Evaluate baseline
baseline_preds = model_baseline.predict(X_test, num_iteration=model_baseline.best_iteration)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
baseline_mae = mean_absolute_error(y_test, baseline_preds)

print(f"✓ Baseline trained")
print(f"  Features: {len(feature_cols)}")
print(f"  Test RMSE: ${baseline_rmse:,.2f}")
print(f"  Test MAE: ${baseline_mae:,.2f}")

# ============================================================================
# FEATURE ABLATION: REMOVE BOTTOM N% FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: FEATURE ABLATION EXPERIMENTS")
print("=" * 80)

# Test removing bottom 10%, 20%, 30%, 40%, 50% of features
removal_percentages = [10, 20, 30, 40, 50]
results = []

# Add baseline result
results.append({
    'removal_pct': 0,
    'num_features': len(available_features),
    'features_removed': 0,
    'test_rmse': baseline_rmse,
    'test_mae': baseline_mae,
    'rmse_change': 0.0,
    'rmse_change_pct': 0.0,
})

for removal_pct in removal_percentages:
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: Remove bottom {removal_pct}% features")
    print(f"{'='*80}")

    # Calculate how many features to keep
    n_features_to_remove = int(len(fi_df) * (removal_pct / 100))
    n_features_to_keep = len(fi_df) - n_features_to_remove

    # Get top features by importance
    top_features = fi_df.nlargest(n_features_to_keep, 'importance')['feature'].tolist()

    # Filter to features that exist in our dataset
    top_features_available = [f for f in top_features if f in X_train.columns]

    print(f"\n📊 Configuration:")
    print(f"  Features to remove: {n_features_to_remove} ({removal_pct}%)")
    print(f"  Features to keep: {len(top_features_available)}")

    # Select features
    X_train_subset = X_train[top_features_available]
    X_valid_subset = X_valid[top_features_available]
    X_test_subset = X_test[top_features_available]

    # Train model with reduced features
    train_data_subset = lgb.Dataset(X_train_subset, label=y_train)
    valid_data_subset = lgb.Dataset(X_valid_subset, label=y_valid, reference=train_data_subset)

    model_subset = lgb.train(
        params,
        train_data_subset,
        num_boost_round=num_rounds,
        valid_sets=[valid_data_subset],
        callbacks=callbacks,
    )

    # Evaluate
    preds_subset = model_subset.predict(X_test_subset, num_iteration=model_subset.best_iteration)
    rmse_subset = np.sqrt(mean_squared_error(y_test, preds_subset))
    mae_subset = mean_absolute_error(y_test, preds_subset)

    # Calculate change
    rmse_change = rmse_subset - baseline_rmse
    rmse_change_pct = (rmse_change / baseline_rmse) * 100

    print(f"\n✓ Results:")
    print(f"  Test RMSE: ${rmse_subset:,.2f}")
    print(f"  Test MAE: ${mae_subset:,.2f}")
    print(f"  RMSE change: ${rmse_change:+.2f} ({rmse_change_pct:+.2f}%)")

    # Store results
    results.append({
        'removal_pct': removal_pct,
        'num_features': len(top_features_available),
        'features_removed': n_features_to_remove,
        'test_rmse': rmse_subset,
        'test_mae': mae_subset,
        'rmse_change': rmse_change,
        'rmse_change_pct': rmse_change_pct,
    })

# ============================================================================
# SUMMARY RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("ABLATION STUDY RESULTS SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(results)

print(f"\n{results_df.to_string(index=False)}")

# Find optimal configuration
# Criterion: Smallest feature set with <1% RMSE degradation
acceptable_degradation = 1.0  # 1% RMSE increase
acceptable_results = results_df[results_df['rmse_change_pct'] <= acceptable_degradation]

if len(acceptable_results) > 1:  # More than just baseline
    optimal = acceptable_results.iloc[-1]  # Most aggressive reduction
    print(f"\n{'='*80}")
    print("OPTIMAL CONFIGURATION FOUND")
    print(f"{'='*80}")
    print(f"  Remove bottom {optimal['removal_pct']:.0f}% features")
    print(f"  Keep {optimal['num_features']:.0f} features (remove {optimal['features_removed']:.0f})")
    print(f"  Test RMSE: ${optimal['test_rmse']:,.2f}")
    print(f"  RMSE change: ${optimal['rmse_change']:+.2f} ({optimal['rmse_change_pct']:+.2f}%)")
    print(f"\n  ✓ Achieves <{acceptable_degradation}% performance degradation with fewer features")
else:
    print(f"\n⚠️  No configuration found with <{acceptable_degradation}% degradation")
    print(f"  Recommendation: Keep all features or accept higher degradation")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save CSV
csv_path = REPORTS_DIR / 'feature_ablation_results.csv'
results_df.to_csv(csv_path, index=False)
print(f"\n✓ Results saved: {csv_path}")

# Save detailed report
report = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'baseline': {
        'num_features': len(available_features),
        'test_rmse': float(baseline_rmse),
        'test_mae': float(baseline_mae),
    },
    'experiments': results_df.to_dict('records'),
    'optimal_config': {
        'removal_pct': int(optimal['removal_pct']) if len(acceptable_results) > 1 else 0,
        'num_features': int(optimal['num_features']) if len(acceptable_results) > 1 else len(available_features),
        'rmse_change_pct': float(optimal['rmse_change_pct']) if len(acceptable_results) > 1 else 0.0,
    } if len(acceptable_results) > 1 else None,
    'recommendation': 'Simplify model' if len(acceptable_results) > 1 else 'Keep all features',
}

report_path = REPORTS_DIR / 'feature_ablation_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"✓ Report saved: {report_path}")

print("\n" + "=" * 80)
print("ABLATION STUDY COMPLETE")
print("=" * 80)
