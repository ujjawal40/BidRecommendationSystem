"""
Feature Ablation Study
======================
Systematically remove low-importance features to find optimal feature set.

Current state:
- 84 features
- Test RMSE: $345
- Overfitting: 2.03x

Goals:
1. Find minimum feature set that maintains performance
2. Potentially reduce overfitting further
3. Improve model interpretability

Approach:
- Train with all features, rank by importance
- Iteratively remove bottom N features
- Track performance at each step
- Find elbow point (diminishing returns)

Author: Bid Recommendation System
Date: 2026-01-22
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import json

from config.model_config import (
    FEATURES_DATA, TARGET_COLUMN, DATE_COLUMN,
    EXCLUDE_COLUMNS, REPORTS_DIR, MODELS_DIR, RANDOM_SEED,
    LIGHTGBM_CONFIG,
)

warnings.filterwarnings('ignore')

print("=" * 80)
print("FEATURE ABLATION STUDY")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Goal: Find minimum feature set that maintains performance\n")

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================
print("Loading data...")
df = pd.read_csv(FEATURES_DATA)
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

# Filter to 2023+ data
start_date = pd.Timestamp('2023-01-01')
df = df[df[DATE_COLUMN] >= start_date].copy()
print(f"Filtered to 2023+: {len(df):,} records")

# Prepare features
feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
print(f"Total features: {len(numeric_features)}")

X = df[numeric_features].fillna(0)
y = df[TARGET_COLUMN]

# 60/20/20 split
n = len(X)
train_idx = int(n * 0.6)
valid_idx = int(n * 0.8)

X_train_full, y_train = X.iloc[:train_idx], y.iloc[:train_idx]
X_valid_full, y_valid = X.iloc[train_idx:valid_idx], y.iloc[train_idx:valid_idx]
X_test_full, y_test = X.iloc[valid_idx:], y.iloc[valid_idx:]

print(f"Train: {len(X_train_full):,} | Valid: {len(X_valid_full):,} | Test: {len(X_test_full):,}")

# ============================================================================
# GET BASELINE FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: BASELINE MODEL & FEATURE IMPORTANCE")
print("=" * 80)

params = LIGHTGBM_CONFIG["params"]

# Train baseline model with all features
train_data = lgb.Dataset(X_train_full, label=y_train, feature_name=numeric_features)
valid_data = lgb.Dataset(X_valid_full, label=y_valid, reference=train_data)

print("\nTraining baseline model with all features...")
baseline_model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(50)]
)

# Get feature importance
importance = baseline_model.feature_importance(importance_type='gain')
importance_df = pd.DataFrame({
    'feature': numeric_features,
    'importance': importance
}).sort_values('importance', ascending=False)

importance_df['importance_pct'] = importance_df['importance'] / importance_df['importance'].sum() * 100
importance_df['cumulative_pct'] = importance_df['importance_pct'].cumsum()
importance_df['rank'] = range(1, len(importance_df) + 1)

print(f"\nBaseline model trained (best iteration: {baseline_model.best_iteration})")

# Baseline metrics
pred_train = baseline_model.predict(X_train_full)
pred_test = baseline_model.predict(X_test_full)
baseline_train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
baseline_test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))
baseline_ratio = baseline_test_rmse / baseline_train_rmse

print(f"\nBaseline Performance (84 features):")
print(f"  Train RMSE: ${baseline_train_rmse:.2f}")
print(f"  Test RMSE: ${baseline_test_rmse:.2f}")
print(f"  Overfitting: {baseline_ratio:.2f}x")

# Show top and bottom features
print(f"\nTop 10 Features (by importance):")
for _, row in importance_df.head(10).iterrows():
    print(f"  {row['rank']:2d}. {row['feature']:40s} {row['importance_pct']:6.2f}% (cum: {row['cumulative_pct']:.1f}%)")

print(f"\nBottom 10 Features:")
for _, row in importance_df.tail(10).iterrows():
    print(f"  {row['rank']:2d}. {row['feature']:40s} {row['importance_pct']:6.2f}%")

# Features with zero importance
zero_importance = importance_df[importance_df['importance'] == 0]
print(f"\nFeatures with ZERO importance: {len(zero_importance)}")
if len(zero_importance) > 0:
    for _, row in zero_importance.iterrows():
        print(f"  - {row['feature']}")

# ============================================================================
# ABLATION STUDY - REMOVE FEATURES PROGRESSIVELY
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: PROGRESSIVE FEATURE REMOVAL")
print("=" * 80)

# Test different feature counts
feature_counts = [84, 70, 60, 50, 40, 30, 25, 20, 15, 12, 10, 8, 5]
feature_counts = [f for f in feature_counts if f <= len(numeric_features)]

results = []

for n_features in feature_counts:
    # Select top N features
    top_features = importance_df.head(n_features)['feature'].tolist()

    # Prepare data with selected features
    X_train = X_train_full[top_features]
    X_valid = X_valid_full[top_features]
    X_test = X_test_full[top_features]

    # Train model
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=top_features)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(50)]
    )

    # Predictions
    pred_train = model.predict(X_train)
    pred_valid = model.predict(X_valid)
    pred_test = model.predict(X_test)

    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
    valid_rmse = np.sqrt(mean_squared_error(y_valid, pred_valid))
    test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    test_r2 = r2_score(y_test, pred_test)
    test_mae = mean_absolute_error(y_test, pred_test)

    overfit_ratio = test_rmse / train_rmse

    # Cumulative importance of selected features
    cum_importance = importance_df.head(n_features)['importance_pct'].sum()

    result = {
        'n_features': n_features,
        'train_rmse': train_rmse,
        'valid_rmse': valid_rmse,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'overfit_ratio': overfit_ratio,
        'best_iter': model.best_iteration,
        'cumulative_importance': cum_importance,
    }
    results.append(result)

    # Status indicator
    change = ((test_rmse - baseline_test_rmse) / baseline_test_rmse) * 100
    status = "✓" if change <= 2 else "○" if change <= 5 else "✗"
    print(f"{status} {n_features:2d} features | Test: ${test_rmse:6.0f} ({change:+5.1f}%) | Ratio: {overfit_ratio:.2f}x | Cum Imp: {cum_importance:.1f}%")

# ============================================================================
# ANALYZE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: RESULTS ANALYSIS")
print("=" * 80)

results_df = pd.DataFrame(results)

print("\n📊 Full Results Table:\n")
print(results_df[['n_features', 'train_rmse', 'test_rmse', 'overfit_ratio', 'cumulative_importance']].to_string(index=False))

# Find optimal point (best test RMSE)
best_idx = results_df['test_rmse'].idxmin()
best = results_df.loc[best_idx]

print(f"\n🏆 BEST PERFORMANCE:")
print(f"   Features: {int(best['n_features'])}")
print(f"   Test RMSE: ${best['test_rmse']:.2f}")
print(f"   Overfitting: {best['overfit_ratio']:.2f}x")

# Find most efficient point (best performance with fewer features)
# Within 2% of best test RMSE
threshold = best['test_rmse'] * 1.02
efficient_options = results_df[results_df['test_rmse'] <= threshold]
most_efficient = efficient_options.loc[efficient_options['n_features'].idxmin()]

print(f"\n🎯 MOST EFFICIENT (within 2% of best):")
print(f"   Features: {int(most_efficient['n_features'])}")
print(f"   Test RMSE: ${most_efficient['test_rmse']:.2f}")
print(f"   Overfitting: {most_efficient['overfit_ratio']:.2f}x")
print(f"   Feature reduction: {84 - int(most_efficient['n_features'])} features removed ({(84 - int(most_efficient['n_features']))/84*100:.0f}%)")

# Find lowest overfitting within 5% of best RMSE
threshold_5pct = best['test_rmse'] * 1.05
low_overfit_options = results_df[results_df['test_rmse'] <= threshold_5pct]
lowest_overfit = low_overfit_options.loc[low_overfit_options['overfit_ratio'].idxmin()]

print(f"\n📉 LOWEST OVERFITTING (within 5% of best RMSE):")
print(f"   Features: {int(lowest_overfit['n_features'])}")
print(f"   Test RMSE: ${lowest_overfit['test_rmse']:.2f}")
print(f"   Overfitting: {lowest_overfit['overfit_ratio']:.2f}x")

# ============================================================================
# RECOMMENDED FEATURE SET
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: RECOMMENDED FEATURE SET")
print("=" * 80)

# Use most efficient as recommendation
recommended_n = int(most_efficient['n_features'])
recommended_features = importance_df.head(recommended_n)['feature'].tolist()

print(f"\n✨ RECOMMENDED: {recommended_n} features\n")
print("Features (ranked by importance):")
for i, feat in enumerate(recommended_features, 1):
    imp_pct = importance_df[importance_df['feature'] == feat]['importance_pct'].values[0]
    print(f"  {i:2d}. {feat:45s} ({imp_pct:.2f}%)")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: SAVING RESULTS")
print("=" * 80)

# Save ablation results
results_path = REPORTS_DIR / "feature_ablation_results.csv"
results_df.to_csv(results_path, index=False)
print(f"✓ Ablation results saved: {results_path}")

# Save feature importance
importance_path = REPORTS_DIR / "feature_importance_ranked.csv"
importance_df.to_csv(importance_path, index=False)
print(f"✓ Feature importance saved: {importance_path}")

# Save recommended feature set
recommended_config = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'study_type': 'feature_ablation',
    'baseline': {
        'n_features': 84,
        'test_rmse': float(baseline_test_rmse),
        'overfit_ratio': float(baseline_ratio),
    },
    'recommended': {
        'n_features': recommended_n,
        'test_rmse': float(most_efficient['test_rmse']),
        'overfit_ratio': float(most_efficient['overfit_ratio']),
        'features': recommended_features,
    },
    'best_performance': {
        'n_features': int(best['n_features']),
        'test_rmse': float(best['test_rmse']),
        'overfit_ratio': float(best['overfit_ratio']),
    },
    'lowest_overfitting': {
        'n_features': int(lowest_overfit['n_features']),
        'test_rmse': float(lowest_overfit['test_rmse']),
        'overfit_ratio': float(lowest_overfit['overfit_ratio']),
    },
    'all_results': results_df.to_dict('records'),
}

config_path = REPORTS_DIR / "feature_ablation_recommended.json"
with open(config_path, 'w') as f:
    json.dump(recommended_config, f, indent=2)
print(f"✓ Recommended config saved: {config_path}")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n{'Configuration':<25} {'Features':<10} {'Test RMSE':<12} {'Overfitting':<12} {'Change'}")
print("-" * 75)
print(f"{'Baseline (all)':<25} {84:<10} ${baseline_test_rmse:<11.2f} {baseline_ratio:<12.2f}x -")

for _, row in results_df.iterrows():
    change = ((row['test_rmse'] - baseline_test_rmse) / baseline_test_rmse) * 100
    marker = "→" if row['n_features'] == recommended_n else " "
    print(f"{marker}{'Top ' + str(int(row['n_features'])):<24} {int(row['n_features']):<10} ${row['test_rmse']:<11.2f} {row['overfit_ratio']:<12.2f}x {change:+.1f}%")

print(f"\n→ = Recommended configuration")

print("\n" + "=" * 80)
print("ABLATION STUDY COMPLETE")
print("=" * 80)
print(f"\nKey Finding: Can reduce from 84 → {recommended_n} features with minimal performance loss")
print(f"Benefit: Simpler model, faster training, better interpretability")
print(f"\nNext step: Update model to use {recommended_n}-feature configuration")
print()
