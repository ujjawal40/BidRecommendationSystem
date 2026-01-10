"""
Final Model Comparison
======================
Compare all models from improvement experiments

Models:
1. Baseline LightGBM (11 features)
2. LightGBM Optuna (100 trials)
3. XGBoost Optuna (100 trials)
4. Neural Network Optuna (50 trials)
5. Feature Engineering V2 (15 features)

Author: Bid Recommendation System
Date: 2026-01-10
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
from datetime import datetime
from config.model_config import REPORTS_DIR, MODELS_DIR

print("=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Load all results
results = {}

# 1. Baseline
baseline_path = MODELS_DIR / "lightgbm_bidfee_optimized_metadata.json"
if baseline_path.exists():
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
        results['Baseline LightGBM'] = {
            'features': baseline['num_features'],
            'train_rmse': baseline['metrics']['train']['rmse'],
            'test_rmse': baseline['metrics']['test']['rmse'],
            'test_r2': baseline['metrics']['test']['r2'],
            'overfitting': baseline['metrics']['overfitting_ratio'],
            'approach': 'Original regression features'
        }

# 2. LightGBM Optuna
lgb_optuna_path = REPORTS_DIR / "lightgbm_optuna_results.json"
if lgb_optuna_path.exists():
    with open(lgb_optuna_path, 'r') as f:
        lgb_optuna = json.load(f)
        results['LightGBM Optuna'] = {
            'features': lgb_optuna['num_features'],
            'train_rmse': lgb_optuna['metrics']['train']['rmse'],
            'test_rmse': lgb_optuna['metrics']['test']['rmse'],
            'test_r2': lgb_optuna['metrics']['test']['r2'],
            'overfitting': lgb_optuna['metrics']['overfitting_ratio'],
            'approach': '100 trials, optimized for RMSE + overfitting'
        }

# 3. XGBoost Optuna
xgb_optuna_path = REPORTS_DIR / "xgboost_optuna_results.json"
if xgb_optuna_path.exists():
    with open(xgb_optuna_path, 'r') as f:
        xgb_optuna = json.load(f)
        results['XGBoost Optuna'] = {
            'features': xgb_optuna['num_features'],
            'train_rmse': xgb_optuna['metrics']['train']['rmse'],
            'test_rmse': xgb_optuna['metrics']['test']['rmse'],
            'test_r2': xgb_optuna['metrics']['test']['r2'],
            'overfitting': xgb_optuna['metrics']['overfitting_ratio'],
            'approach': '100 trials, optimized for RMSE + overfitting'
        }

# 4. Neural Network Optuna
nn_optuna_path = REPORTS_DIR / "pytorch_nn_optuna_results.json"
if nn_optuna_path.exists():
    with open(nn_optuna_path, 'r') as f:
        nn_optuna = json.load(f)
        results['Neural Network Optuna'] = {
            'features': nn_optuna['num_features'],
            'train_rmse': nn_optuna['metrics']['train']['rmse'],
            'test_rmse': nn_optuna['metrics']['test']['rmse'],
            'test_r2': nn_optuna['metrics']['test']['r2'],
            'overfitting': nn_optuna['metrics']['overfitting_ratio'],
            'approach': f"{nn_optuna['best_architecture']['n_layers']} layers, {nn_optuna['best_architecture']['activation']} activation"
        }
else:
    print("WARNING: Neural Network results not found (may still be training)\n")

# 5. Feature Engineering V2
fe_v2_path = REPORTS_DIR / "feature_engineering_v2_results.json"
if fe_v2_path.exists():
    with open(fe_v2_path, 'r') as f:
        fe_v2 = json.load(f)
        results['Feature Engineering V2'] = {
            'features': fe_v2['num_features'],
            'train_rmse': fe_v2['metrics']['train']['rmse'],
            'test_rmse': fe_v2['metrics']['test']['rmse'],
            'test_r2': fe_v2['metrics']['test']['r2'],
            'overfitting': fe_v2['metrics']['overfitting_ratio'],
            'approach': f"{fe_v2['new_features_created']} new features (percentiles, CV, interactions)"
        }

# Print table
print("=" * 120)
print(f"{'Model':<30} {'Features':<10} {'Test RMSE':<15} {'Test R²':<12} {'Overfitting':<12} {'vs Baseline'}")
print("=" * 120)

baseline_rmse = results['Baseline LightGBM']['test_rmse']

for name, metrics in results.items():
    improvement = ((baseline_rmse - metrics['test_rmse']) / baseline_rmse) * 100
    improvement_str = f"{improvement:+.1f}%" if name != 'Baseline LightGBM' else "---"

    print(f"{name:<30} {metrics['features']:<10} ${metrics['test_rmse']:<14,.2f} "
          f"{metrics['test_r2']:<11.4f} {metrics['overfitting']:<11.2f}x {improvement_str}")

print("=" * 120)

# Find best model by RMSE
best_model = min(results.items(), key=lambda x: x[1]['test_rmse'])
print(f"\nBEST MODEL BY TEST RMSE: {best_model[0]}")
print(f"  Test RMSE: ${best_model[1]['test_rmse']:,.2f}")
print(f"  Improvement: {((baseline_rmse - best_model[1]['test_rmse']) / baseline_rmse) * 100:.1f}%")
print(f"  Features: {best_model[1]['features']}")
print(f"  Approach: {best_model[1]['approach']}")

# Find best model by overfitting
best_overfitting_model = min(results.items(), key=lambda x: x[1]['overfitting'])
print(f"\nBEST MODEL BY OVERFITTING: {best_overfitting_model[0]}")
print(f"  Overfitting: {best_overfitting_model[1]['overfitting']:.2f}x")
print(f"  Test RMSE: ${best_overfitting_model[1]['test_rmse']:,.2f}")

# Detailed approach summary
print("\n" + "=" * 120)
print("APPROACH SUMMARY")
print("=" * 120)

for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  {metrics['approach']}")

# Save comparison
comparison_summary = {
    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "baseline_rmse": float(baseline_rmse),
    "models_compared": len(results),
    "results": {
        name: {
            "features": metrics['features'],
            "test_rmse": float(metrics['test_rmse']),
            "test_r2": float(metrics['test_r2']),
            "overfitting": float(metrics['overfitting']),
            "improvement_pct": float(((baseline_rmse - metrics['test_rmse']) / baseline_rmse) * 100),
            "approach": metrics['approach']
        }
        for name, metrics in results.items()
    },
    "best_model": {
        "name": best_model[0],
        "test_rmse": float(best_model[1]['test_rmse']),
        "improvement_pct": float(((baseline_rmse - best_model[1]['test_rmse']) / baseline_rmse) * 100),
        "features": best_model[1]['features'],
        "approach": best_model[1]['approach']
    },
    "best_overfitting_model": {
        "name": best_overfitting_model[0],
        "overfitting": float(best_overfitting_model[1]['overfitting']),
        "test_rmse": float(best_overfitting_model[1]['test_rmse'])
    }
}

comparison_path = REPORTS_DIR / "final_comparison_summary.json"
with open(comparison_path, 'w') as f:
    json.dump(comparison_summary, f, indent=2)

print("\n" + "=" * 120)
print(f"Comparison saved: {comparison_path}")
print("=" * 120)
