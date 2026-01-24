"""
Recent Data Experiment
======================
Train on recent data only (2022+) to address temporal distribution shift.

Hypothesis: The 5x degradation on 2024-2025 data may be due to:
1. Market dynamics changed significantly post-2022
2. Older patterns (2018-2021) are now noise, not signal
3. Training on recent data = better generalization to test period

This script compares:
- Full data (2018-2025)
- Recent data (2022-2025)
- Very recent data (2023-2025)

Author: Bid Recommendation System
Date: 2026-01-20
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import json

from config.model_config import (
    FEATURES_DATA, TARGET_COLUMN, DATE_COLUMN,
    EXCLUDE_COLUMNS, REPORTS_DIR, RANDOM_SEED,
)

warnings.filterwarnings('ignore')

print("=" * 80)
print("RECENT DATA EXPERIMENT")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Hypothesis: Training on recent data improves generalization\n")

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading data...")
df = pd.read_csv(FEATURES_DATA)
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

print(f"Total records: {len(df):,}")
print(f"Date range: {df[DATE_COLUMN].min()} to {df[DATE_COLUMN].max()}")

# Prepare features
feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
print(f"Features: {len(numeric_features)}")

# ============================================================================
# DEFINE EXPERIMENTS
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT CONFIGURATIONS")
print("=" * 80)

# Use balanced regularization (middle ground from our analysis)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 1.0,       # Moderate regularization
    'reg_lambda': 1.0,
    'num_leaves': 25,
    'max_depth': 8,
    'min_child_samples': 30,
    'random_state': RANDOM_SEED,
    'verbose': -1,
}

experiments = [
    {'name': 'Full Data (2018-2025)', 'start_date': '2018-01-01'},
    {'name': 'Recent (2022-2025)', 'start_date': '2022-01-01'},
    {'name': 'Very Recent (2023-2025)', 'start_date': '2023-01-01'},
    {'name': 'Last 2 Years (2024-2025)', 'start_date': '2024-01-01'},
]

# Fixed test set: always last 20% chronologically
# This ensures fair comparison across experiments
test_cutoff_date = df[DATE_COLUMN].quantile(0.8)
print(f"\nFixed test set: after {test_cutoff_date.strftime('%Y-%m-%d')}")
print(f"Test set size: {len(df[df[DATE_COLUMN] >= test_cutoff_date]):,} records")

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================
print("\n" + "=" * 80)
print("RUNNING EXPERIMENTS")
print("=" * 80)

results = []

for exp in experiments:
    print(f"\n{'─' * 60}")
    print(f"Experiment: {exp['name']}")
    print(f"{'─' * 60}")

    # Filter training data by start date
    start_date = pd.Timestamp(exp['start_date'])
    df_train_pool = df[(df[DATE_COLUMN] >= start_date) & (df[DATE_COLUMN] < test_cutoff_date)].copy()
    df_test = df[df[DATE_COLUMN] >= test_cutoff_date].copy()

    if len(df_train_pool) < 1000:
        print(f"  ⚠ Skipping - only {len(df_train_pool)} training samples")
        continue

    # 80/20 split within training pool for validation
    train_valid_split = int(len(df_train_pool) * 0.8)
    df_train = df_train_pool.iloc[:train_valid_split]
    df_valid = df_train_pool.iloc[train_valid_split:]

    # Prepare data
    X_train = df_train[numeric_features].fillna(0).values
    y_train = df_train[TARGET_COLUMN].values
    X_valid = df_valid[numeric_features].fillna(0).values
    y_valid = df_valid[TARGET_COLUMN].values
    X_test = df_test[numeric_features].fillna(0).values
    y_test = df_test[TARGET_COLUMN].values

    print(f"  Train: {len(X_train):,} ({df_train[DATE_COLUMN].min().strftime('%Y-%m')} to {df_train[DATE_COLUMN].max().strftime('%Y-%m')})")
    print(f"  Valid: {len(X_valid):,}")
    print(f"  Test: {len(X_test):,}")

    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    # Train
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(30)]
    )

    # Predictions
    pred_train = model.predict(X_train, num_iteration=model.best_iteration)
    pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
    pred_test = model.predict(X_test, num_iteration=model.best_iteration)

    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
    valid_rmse = np.sqrt(mean_squared_error(y_valid, pred_valid))
    test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    test_r2 = r2_score(y_test, pred_test)

    overfit_ratio = test_rmse / train_rmse

    result = {
        'experiment': exp['name'],
        'start_date': exp['start_date'],
        'train_samples': len(X_train),
        'train_rmse': train_rmse,
        'valid_rmse': valid_rmse,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'overfit_ratio': overfit_ratio,
        'best_iter': model.best_iteration,
    }
    results.append(result)

    print(f"\n  Results:")
    print(f"    Train RMSE: ${train_rmse:,.2f}")
    print(f"    Valid RMSE: ${valid_rmse:,.2f}")
    print(f"    Test RMSE: ${test_rmse:,.2f}")
    print(f"    Test R²: {test_r2:.4f}")
    print(f"    Overfitting: {overfit_ratio:.2f}x")

# ============================================================================
# COMPARE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('test_rmse')

print("\n📊 Results Summary (sorted by Test RMSE):\n")
print(results_df[['experiment', 'train_samples', 'train_rmse', 'test_rmse', 'test_r2', 'overfit_ratio']].to_string(index=False))

# Best result
best = results_df.iloc[0]
print(f"\n🏆 BEST: {best['experiment']}")
print(f"   Test RMSE: ${best['test_rmse']:.2f}")
print(f"   Overfitting: {best['overfit_ratio']:.2f}x")

# Improvement over full data
full_data_result = results_df[results_df['experiment'].str.contains('Full')].iloc[0] if any(results_df['experiment'].str.contains('Full')) else None
if full_data_result is not None:
    improvement = (full_data_result['test_rmse'] - best['test_rmse']) / full_data_result['test_rmse'] * 100
    if improvement > 0:
        print(f"\n   Improvement vs Full Data: {improvement:.1f}% better")
    else:
        print(f"\n   vs Full Data: {-improvement:.1f}% worse")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

results_path = REPORTS_DIR / "recent_data_experiment_results.csv"
results_df.to_csv(results_path, index=False)
print(f"✓ Results saved: {results_path}")

# Save as JSON too
json_results = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'hypothesis': 'Training on recent data improves generalization',
    'test_cutoff': test_cutoff_date.strftime('%Y-%m-%d'),
    'params_used': params,
    'results': results_df.to_dict('records'),
    'best_experiment': best['experiment'],
    'conclusion': f"Best: {best['experiment']} with Test RMSE ${best['test_rmse']:.2f}",
}

json_path = REPORTS_DIR / "recent_data_experiment_results.json"
with open(json_path, 'w') as f:
    json.dump(json_results, f, indent=2, default=str)
print(f"✓ JSON results saved: {json_path}")

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)
print(f"\nConclusion:")
if best['experiment'] != 'Full Data (2018-2025)':
    print(f"  ✓ {best['experiment']} outperformed full data")
    print(f"  ✓ Temporal focus helps - older data is noise")
    print(f"\nRecommendation: Use {best['start_date']}+ for training")
else:
    print(f"  ✗ Full data still best - temporal shift may not be the issue")
    print(f"\nRecommendation: Focus on other improvements (feature selection, regularization)")
print()
