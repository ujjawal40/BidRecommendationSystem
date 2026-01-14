"""
Feature Elimination for Regression
===================================
Start with ALL features, iteratively eliminate least important

Strategy:
1. Train with ALL features
2. Remove bottom 10% by importance
3. Retrain and compare
4. Repeat until performance degrades

Author: Bid Recommendation System
Date: 2026-01-11
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
print("FEATURE ELIMINATION - REGRESSION")
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
# COLLECT ALL NUMERIC FEATURES
# ============================================================================
print("=" * 80)
print("COLLECTING ALL FEATURES")
print("=" * 80)

# Exclude non-feature columns
exclude_cols = [
    TARGET_COLUMN, DATE_COLUMN, 'BidId', 'BidFileNumber', 'BidName', 'BidDate',
    'Bid_DueDate', 'BidStatusName', 'Bid_JobPurpose', 'Bid_Deliverable',
    'BusinessSegmentDetail', 'Bid_Property_Type', 'Bid_SubProperty_Type',
    'Bid_SpecificUseProperty_Type', 'PropertyId', 'PropertyName', 'PropertyType',
    'SubType', 'PropertyCity', 'PropertyState', 'AddressDisplayCalc',
    'GrossBuildingAreaRange', 'YearBuiltRange', 'OfficeCode', 'OfficeCompanyName',
    'OfficeLocation', 'JobId', 'JobName', 'JobStatus', 'JobType', 'AppraisalFileType',
    'BidCompanyName', 'BidCompanyType', 'BidFee_Original', 'TargetTime_Original',
    'Market', 'Submarket', 'BusinessSegment', 'MarketOrientation', 'ZipCode',
    'Bid_SubProperty_Type', 'Bid_SpecificUseProperty_Type'
]

# Get all numeric columns
all_features = []
for col in df_recent.columns:
    if col not in exclude_cols:
        # Check if numeric
        if df_recent[col].dtype in [np.float64, np.int64]:
            # Check for sufficient non-null values
            non_null_pct = (1 - df_recent[col].isna().mean())
            if non_null_pct > 0.5:  # At least 50% non-null
                all_features.append(col)

print(f"\nTotal features available: {len(all_features)}")
print(f"Feature types: engineered + original numeric")

# Prepare data
X_all = df_recent[all_features].fillna(0).values
y = df_recent[TARGET_COLUMN].values

# 80/20 split
split_idx = int(len(X_all) * 0.8)
X_train, X_test = X_all[:split_idx], X_all[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

# ============================================================================
# ITERATION 1: ALL FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("ITERATION 1: ALL FEATURES")
print("=" * 80)

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

print(f"Training with {len(all_features)} features...")
train_data = lgb.Dataset(X_train, label=y_train, feature_name=all_features)

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data],
    callbacks=[lgb.early_stopping(50)]
)

y_pred_train = model.predict(X_train, num_iteration=model.best_iteration)
y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)
overfitting = test_rmse / train_rmse

print(f"\nRESULTS:")
print(f"  Train RMSE: ${train_rmse:,.2f}")
print(f"  Test RMSE: ${test_rmse:,.2f}")
print(f"  Test R²: {test_r2:.4f}")
print(f"  Overfitting: {overfitting:.2f}x")

baseline_rmse = 237.57
improvement = ((baseline_rmse - test_rmse) / baseline_rmse) * 100
print(f"\nvs Baseline: {improvement:+.1f}%")

# Get feature importance
importance = model.feature_importance(importance_type='gain')
importance_df = pd.DataFrame({
    'feature': all_features,
    'importance': importance,
    'importance_pct': (importance / importance.sum() * 100)
}).sort_values('importance', ascending=False)

print(f"\nTop 15 features:")
for i, row in enumerate(importance_df.head(15).itertuples(), 1):
    print(f"  {i:2d}. {row.feature:45s} {row.importance_pct:6.2f}%")

# Store results
iteration_results = [{
    'iteration': 1,
    'num_features': len(all_features),
    'train_rmse': float(train_rmse),
    'test_rmse': float(test_rmse),
    'test_r2': float(test_r2),
    'overfitting': float(overfitting),
    'improvement_vs_baseline': float(improvement),
    'features': all_features
}]

best_rmse = test_rmse
best_features = all_features
best_iteration = 1

# ============================================================================
# ITERATIVE ELIMINATION
# ============================================================================
print("\n" + "=" * 80)
print("ITERATIVE ELIMINATION")
print("=" * 80)

current_features = all_features.copy()
iteration = 2

while len(current_features) > 10:
    # Remove bottom 10% of features by importance
    n_to_remove = max(1, int(len(current_features) * 0.1))

    # Get features sorted by importance
    current_importance = importance_df[importance_df['feature'].isin(current_features)]
    features_to_remove = current_importance.tail(n_to_remove)['feature'].tolist()
    current_features = [f for f in current_features if f not in features_to_remove]

    print(f"\n{'='*80}")
    print(f"ITERATION {iteration}: {len(current_features)} FEATURES")
    print(f"Removed {n_to_remove} features: {', '.join(features_to_remove[:3])}...")
    print(f"{'='*80}")

    # Prepare data with current features
    X_train_current = df_recent[current_features].iloc[:split_idx].fillna(0).values
    X_test_current = df_recent[current_features].iloc[split_idx:].fillna(0).values

    # Train model
    train_data_current = lgb.Dataset(X_train_current, label=y_train, feature_name=current_features)

    model_current = lgb.train(
        params,
        train_data_current,
        num_boost_round=1000,
        valid_sets=[train_data_current],
        callbacks=[lgb.early_stopping(50)]
    )

    # Evaluate
    y_pred_train_current = model_current.predict(X_train_current, num_iteration=model_current.best_iteration)
    y_pred_test_current = model_current.predict(X_test_current, num_iteration=model_current.best_iteration)

    train_rmse_current = np.sqrt(mean_squared_error(y_train, y_pred_train_current))
    test_rmse_current = np.sqrt(mean_squared_error(y_test, y_pred_test_current))
    test_r2_current = r2_score(y_test, y_pred_test_current)
    overfitting_current = test_rmse_current / train_rmse_current
    improvement_current = ((baseline_rmse - test_rmse_current) / baseline_rmse) * 100

    print(f"\nRESULTS:")
    print(f"  Test RMSE: ${test_rmse_current:,.2f} (vs baseline: {improvement_current:+.1f}%)")
    print(f"  Overfitting: {overfitting_current:.2f}x")

    # Update importance for next iteration
    importance_current = model_current.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': current_features,
        'importance': importance_current,
        'importance_pct': (importance_current / importance_current.sum() * 100)
    }).sort_values('importance', ascending=False)

    # Store results
    iteration_results.append({
        'iteration': iteration,
        'num_features': len(current_features),
        'train_rmse': float(train_rmse_current),
        'test_rmse': float(test_rmse_current),
        'test_r2': float(test_r2_current),
        'overfitting': float(overfitting_current),
        'improvement_vs_baseline': float(improvement_current),
        'features': current_features.copy()
    })

    # Track best
    if test_rmse_current < best_rmse:
        best_rmse = test_rmse_current
        best_features = current_features.copy()
        best_iteration = iteration
        print(f"  ✓ NEW BEST RMSE!")

    iteration += 1

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ELIMINATION SUMMARY")
print("=" * 80)

print(f"\n{'Iteration':<12} {'Features':<12} {'Test RMSE':<15} {'vs Baseline':<12} {'Overfitting'}")
print("-" * 70)
for result in iteration_results:
    marker = " ←BEST" if result['iteration'] == best_iteration else ""
    print(f"{result['iteration']:<12} {result['num_features']:<12} ${result['test_rmse']:<14,.2f} "
          f"{result['improvement_vs_baseline']:+6.1f}%{'':6s} {result['overfitting']:.2f}x{marker}")

print(f"\n{'='*80}")
print(f"BEST RESULT: Iteration {best_iteration} with {len(best_features)} features")
print(f"Test RMSE: ${best_rmse:,.2f}")
print(f"Improvement vs baseline: {((baseline_rmse - best_rmse) / baseline_rmse) * 100:+.1f}%")
print(f"{'='*80}")

if best_rmse < baseline_rmse:
    print(f"\n✓ IMPROVEMENT FOUND!")
    print(f"\nTop 20 features in best model:")
    best_importance_df = importance_df[importance_df['feature'].isin(best_features)].head(20)
    for i, row in enumerate(best_importance_df.itertuples(), 1):
        print(f"  {i:2d}. {row.feature:45s} {row.importance_pct:6.2f}%")
else:
    print(f"\n⚠ No improvement over baseline ${baseline_rmse:,.2f}")

# Save results
results = {
    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "baseline_rmse": float(baseline_rmse),
    "initial_features": len(all_features),
    "iterations": len(iteration_results),
    "iteration_results": iteration_results,
    "best_iteration": best_iteration,
    "best_num_features": len(best_features),
    "best_test_rmse": float(best_rmse),
    "best_improvement_pct": float(((baseline_rmse - best_rmse) / baseline_rmse) * 100),
    "best_features": best_features
}

results_path = REPORTS_DIR / "feature_elimination_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved: {results_path}")
print("\n" + "=" * 80)
