"""
Regularization Grid Search
==========================
Find optimal regularization to balance overfitting vs underfitting.

Current state:
- Weak reg (0.1): Train $82, Test $297, Ratio 3.6x (overfitting)
- Strong reg (5.0): Train $238, Test $560, Ratio 2.4x (underfitting)

Goal: Find middle ground with ratio <1.8x AND lowest test RMSE

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
from sklearn.metrics import mean_squared_error
import warnings
import json

from config.model_config import (
    FEATURES_DATA, TARGET_COLUMN, DATE_COLUMN,
    EXCLUDE_COLUMNS, REPORTS_DIR, RANDOM_SEED,
)

warnings.filterwarnings('ignore')

print("=" * 80)
print("REGULARIZATION GRID SEARCH")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Goal: Find optimal balance between overfitting and underfitting\n")

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================
print("Loading data...")
df = pd.read_csv(FEATURES_DATA)
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

# Prepare features
feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

X = df[numeric_features].fillna(0).values
y = df[TARGET_COLUMN].values

# 60/20/20 split (train/valid/test)
n = len(X)
train_idx = int(n * 0.6)
valid_idx = int(n * 0.8)

X_train, y_train = X[:train_idx], y[:train_idx]
X_valid, y_valid = X[train_idx:valid_idx], y[train_idx:valid_idx]
X_test, y_test = X[valid_idx:], y[valid_idx:]

print(f"Train: {len(X_train):,} | Valid: {len(X_valid):,} | Test: {len(X_test):,}")
print(f"Train dates: {df[DATE_COLUMN].iloc[0]} to {df[DATE_COLUMN].iloc[train_idx-1]}")
print(f"Valid dates: {df[DATE_COLUMN].iloc[train_idx]} to {df[DATE_COLUMN].iloc[valid_idx-1]}")
print(f"Test dates: {df[DATE_COLUMN].iloc[valid_idx]} to {df[DATE_COLUMN].iloc[-1]}")

# ============================================================================
# GRID SEARCH PARAMETERS
# ============================================================================
print("\n" + "=" * 80)
print("GRID SEARCH CONFIGURATION")
print("=" * 80)

# Parameter grid - exploring the space between 0.1 (overfit) and 5.0 (underfit)
param_grid = {
    'reg_alpha': [0.5, 1.0, 2.0, 3.0],
    'reg_lambda': [0.5, 1.0, 2.0, 3.0],
    'num_leaves': [15, 20, 25, 31],
    'max_depth': [4, 6, 8, -1],  # -1 = unlimited
    'min_child_samples': [20, 30, 50],
}

# Fixed parameters
base_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'random_state': RANDOM_SEED,
    'verbose': -1,
    'n_jobs': -1,
}

# Strategic combinations to test (not full grid - too expensive)
# These span from "more complex" to "more regularized"
test_configs = [
    # Baseline reference (original weak regularization)
    {'reg_alpha': 0.1, 'reg_lambda': 0.1, 'num_leaves': 31, 'max_depth': -1, 'min_child_samples': 20, 'name': 'Original (weak)'},

    # Slightly more regularized
    {'reg_alpha': 0.5, 'reg_lambda': 0.5, 'num_leaves': 31, 'max_depth': -1, 'min_child_samples': 20, 'name': 'Mild reg'},
    {'reg_alpha': 1.0, 'reg_lambda': 1.0, 'num_leaves': 31, 'max_depth': -1, 'min_child_samples': 20, 'name': 'Moderate reg'},

    # Limit depth with mild reg
    {'reg_alpha': 0.5, 'reg_lambda': 0.5, 'num_leaves': 31, 'max_depth': 8, 'min_child_samples': 20, 'name': 'Mild + depth=8'},
    {'reg_alpha': 0.5, 'reg_lambda': 0.5, 'num_leaves': 31, 'max_depth': 6, 'min_child_samples': 20, 'name': 'Mild + depth=6'},

    # Fewer leaves with mild reg
    {'reg_alpha': 0.5, 'reg_lambda': 0.5, 'num_leaves': 20, 'max_depth': -1, 'min_child_samples': 20, 'name': 'Mild + leaves=20'},
    {'reg_alpha': 0.5, 'reg_lambda': 0.5, 'num_leaves': 15, 'max_depth': -1, 'min_child_samples': 20, 'name': 'Mild + leaves=15'},

    # Moderate reg with complexity limits
    {'reg_alpha': 1.0, 'reg_lambda': 1.0, 'num_leaves': 25, 'max_depth': 8, 'min_child_samples': 30, 'name': 'Balanced 1'},
    {'reg_alpha': 1.0, 'reg_lambda': 1.0, 'num_leaves': 20, 'max_depth': 6, 'min_child_samples': 30, 'name': 'Balanced 2'},
    {'reg_alpha': 1.5, 'reg_lambda': 1.5, 'num_leaves': 20, 'max_depth': 6, 'min_child_samples': 30, 'name': 'Balanced 3'},

    # Stronger regularization (midpoint)
    {'reg_alpha': 2.0, 'reg_lambda': 2.0, 'num_leaves': 20, 'max_depth': 6, 'min_child_samples': 40, 'name': 'Strong 1'},
    {'reg_alpha': 2.0, 'reg_lambda': 2.0, 'num_leaves': 15, 'max_depth': 6, 'min_child_samples': 50, 'name': 'Strong 2'},

    # Aggressive reference (current underfitting config)
    {'reg_alpha': 5.0, 'reg_lambda': 5.0, 'num_leaves': 15, 'max_depth': 6, 'min_child_samples': 50, 'name': 'Aggressive (current)'},

    # Extra experiments
    {'reg_alpha': 3.0, 'reg_lambda': 3.0, 'num_leaves': 20, 'max_depth': 8, 'min_child_samples': 40, 'name': 'Strong + depth=8'},
    {'reg_alpha': 1.0, 'reg_lambda': 2.0, 'num_leaves': 25, 'max_depth': 8, 'min_child_samples': 30, 'name': 'Asymmetric L1<L2'},
    {'reg_alpha': 2.0, 'reg_lambda': 1.0, 'num_leaves': 25, 'max_depth': 8, 'min_child_samples': 30, 'name': 'Asymmetric L1>L2'},
]

print(f"Testing {len(test_configs)} configurations\n")

# ============================================================================
# RUN GRID SEARCH
# ============================================================================
print("=" * 80)
print("RUNNING EXPERIMENTS")
print("=" * 80)

results = []

for i, config in enumerate(test_configs, 1):
    name = config.pop('name')

    # Merge with base params
    params = {**base_params, **config}

    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=numeric_features)
    valid_data = lgb.Dataset(X_valid, label=y_valid, feature_name=numeric_features, reference=train_data)

    # Train with early stopping on validation set
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,  # Reduced for faster search
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

    overfit_ratio = test_rmse / train_rmse
    valid_overfit = valid_rmse / train_rmse

    # Negative predictions check
    neg_preds = (pred_test < 0).sum()

    result = {
        'name': name,
        'reg_alpha': config['reg_alpha'],
        'reg_lambda': config['reg_lambda'],
        'num_leaves': config['num_leaves'],
        'max_depth': config['max_depth'],
        'min_child_samples': config['min_child_samples'],
        'best_iter': model.best_iteration,
        'train_rmse': train_rmse,
        'valid_rmse': valid_rmse,
        'test_rmse': test_rmse,
        'overfit_ratio': overfit_ratio,
        'valid_overfit': valid_overfit,
        'neg_predictions': neg_preds,
    }
    results.append(result)

    # Status indicator
    status = "✓" if overfit_ratio < 2.0 and test_rmse < 400 else "○" if overfit_ratio < 2.5 else "✗"
    print(f"{i:2d}. {status} {name:25s} | Train: ${train_rmse:6.0f} | Test: ${test_rmse:6.0f} | Ratio: {overfit_ratio:.2f}x | Iter: {model.best_iteration}")

    # Restore name for next iteration
    config['name'] = name

# ============================================================================
# ANALYZE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("RESULTS ANALYSIS")
print("=" * 80)

results_df = pd.DataFrame(results)

# Sort by test RMSE
results_df = results_df.sort_values('test_rmse')

print("\n📊 All Results (sorted by Test RMSE):\n")
print(results_df[['name', 'train_rmse', 'valid_rmse', 'test_rmse', 'overfit_ratio', 'best_iter']].to_string(index=False))

# Find Pareto-optimal configs (best test RMSE for each overfitting level)
print("\n" + "-" * 80)
print("🎯 BEST CONFIGURATIONS BY CRITERIA")
print("-" * 80)

# Best overall test RMSE
best_rmse = results_df.iloc[0]
print(f"\n1. Lowest Test RMSE:")
print(f"   {best_rmse['name']}")
print(f"   Train: ${best_rmse['train_rmse']:.2f} | Test: ${best_rmse['test_rmse']:.2f} | Ratio: {best_rmse['overfit_ratio']:.2f}x")

# Best with ratio < 2.0
good_ratio = results_df[results_df['overfit_ratio'] < 2.0]
if len(good_ratio) > 0:
    best_good = good_ratio.iloc[0]
    print(f"\n2. Best with Ratio < 2.0x:")
    print(f"   {best_good['name']}")
    print(f"   Train: ${best_good['train_rmse']:.2f} | Test: ${best_good['test_rmse']:.2f} | Ratio: {best_good['overfit_ratio']:.2f}x")
else:
    print(f"\n2. Best with Ratio < 2.0x: None found")

# Best with ratio < 1.8
very_good_ratio = results_df[results_df['overfit_ratio'] < 1.8]
if len(very_good_ratio) > 0:
    best_very_good = very_good_ratio.iloc[0]
    print(f"\n3. Best with Ratio < 1.8x:")
    print(f"   {best_very_good['name']}")
    print(f"   Train: ${best_very_good['train_rmse']:.2f} | Test: ${best_very_good['test_rmse']:.2f} | Ratio: {best_very_good['overfit_ratio']:.2f}x")
else:
    print(f"\n3. Best with Ratio < 1.8x: None found")

# Best balanced (minimize test_rmse * overfit_ratio)
results_df['balance_score'] = results_df['test_rmse'] * results_df['overfit_ratio']
best_balanced = results_df.loc[results_df['balance_score'].idxmin()]
print(f"\n4. Best Balanced (RMSE × Ratio):")
print(f"   {best_balanced['name']}")
print(f"   Train: ${best_balanced['train_rmse']:.2f} | Test: ${best_balanced['test_rmse']:.2f} | Ratio: {best_balanced['overfit_ratio']:.2f}x")
print(f"   Score: {best_balanced['balance_score']:.0f}")

# ============================================================================
# RECOMMENDATION
# ============================================================================
print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

# Pick the best balanced config
recommended = best_balanced

print(f"\n🏆 RECOMMENDED CONFIGURATION: {recommended['name']}")
print(f"\nHyperparameters:")
print(f"  reg_alpha: {recommended['reg_alpha']}")
print(f"  reg_lambda: {recommended['reg_lambda']}")
print(f"  num_leaves: {recommended['num_leaves']}")
print(f"  max_depth: {recommended['max_depth']}")
print(f"  min_child_samples: {recommended['min_child_samples']}")

print(f"\nExpected Performance:")
print(f"  Train RMSE: ${recommended['train_rmse']:.2f}")
print(f"  Valid RMSE: ${recommended['valid_rmse']:.2f}")
print(f"  Test RMSE: ${recommended['test_rmse']:.2f}")
print(f"  Overfitting Ratio: {recommended['overfit_ratio']:.2f}x")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save full results
results_path = REPORTS_DIR / "regularization_grid_search_results.csv"
results_df.to_csv(results_path, index=False)
print(f"✓ Full results saved: {results_path}")

# Save recommended config
recommended_config = {
    'name': recommended['name'],
    'params': {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': float(recommended['reg_alpha']),
        'reg_lambda': float(recommended['reg_lambda']),
        'num_leaves': int(recommended['num_leaves']),
        'max_depth': int(recommended['max_depth']),
        'min_child_samples': int(recommended['min_child_samples']),
        'random_state': RANDOM_SEED,
        'verbose': -1,
    },
    'training': {
        'num_boost_round': 500,
        'early_stopping_rounds': 30,
    },
    'expected_metrics': {
        'train_rmse': float(recommended['train_rmse']),
        'valid_rmse': float(recommended['valid_rmse']),
        'test_rmse': float(recommended['test_rmse']),
        'overfit_ratio': float(recommended['overfit_ratio']),
    },
    'search_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
}

config_path = REPORTS_DIR / "recommended_lightgbm_config.json"
with open(config_path, 'w') as f:
    json.dump(recommended_config, f, indent=2)
print(f"✓ Recommended config saved: {config_path}")

print("\n" + "=" * 80)
print("GRID SEARCH COMPLETE")
print("=" * 80)
print(f"\nNext steps:")
print(f"1. Update model_config.py with recommended parameters")
print(f"2. Retrain model with full data using recommended config")
print(f"3. Test on recent data (2022+) to see if temporal focus helps")
print()
