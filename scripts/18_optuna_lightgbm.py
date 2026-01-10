"""
LightGBM Hyperparameter Tuning with Optuna
===========================================
Optimize LightGBM hyperparameters to improve regression performance

Target: Beat baseline $237.57 RMSE and reduce 2.57x overfitting

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
import optuna
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

from config.model_config import (
    FEATURES_DATA, TARGET_COLUMN, DATE_COLUMN,
    MODELS_DIR, REPORTS_DIR, RANDOM_SEED,
)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Load regression-optimized features
REGRESSION_FEATURES_PATH = MODELS_DIR.parent / "reports" / "regression_features.json"
with open(REGRESSION_FEATURES_PATH, 'r') as f:
    regression_config = json.load(f)
    SELECTED_FEATURES = regression_config['features']

print("=" * 80)
print("LIGHTGBM HYPERPARAMETER TUNING (OPTUNA)")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Features: {len(SELECTED_FEATURES)}")
print(f"Baseline to beat: $237.57 RMSE, 2.57x overfitting\n")

# Load data
df = pd.read_csv(FEATURES_DATA)
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

# Filter to recent data
recent_cutoff = pd.Timestamp('2023-01-01')
df_recent = df[df[DATE_COLUMN] >= recent_cutoff].copy()

print(f"Data: {len(df_recent):,} rows (2023-2025)\n")

# Prepare data
X = df_recent[SELECTED_FEATURES].fillna(0).values
y = df_recent[TARGET_COLUMN].values

# 80/20 split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}\n")

# Optuna objective function
def objective(trial):
    # Suggest hyperparameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'random_state': RANDOM_SEED,
        'verbose': -1
    }

    # Train model
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=SELECTED_FEATURES)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(50)]
    )

    # Evaluate
    y_pred_train = model.predict(X_train, num_iteration=model.best_iteration)
    y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    overfitting_ratio = test_rmse / train_rmse

    # Multi-objective: minimize test RMSE and overfitting ratio
    # Weighted combination: 70% test RMSE, 30% overfitting penalty
    score = test_rmse + (overfitting_ratio - 1.0) * 100

    return score

# Run optimization
print("=" * 80)
print("RUNNING OPTUNA OPTIMIZATION")
print("=" * 80)
print("Trials: 100")
print("Objective: Minimize test RMSE + overfitting penalty\n")

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(objective, n_trials=100, show_progress_bar=False)

print(f"Best trial: {study.best_trial.number}")
print(f"Best score: {study.best_value:.2f}\n")

# Train final model with best params
print("=" * 80)
print("TRAINING FINAL MODEL WITH BEST PARAMS")
print("=" * 80)

best_params = study.best_params
best_params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'random_state': RANDOM_SEED,
    'verbose': -1
})

print("Best hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

train_data = lgb.Dataset(X_train, label=y_train, feature_name=SELECTED_FEATURES)

final_model = lgb.train(
    best_params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data],
    callbacks=[lgb.early_stopping(50)]
)

print(f"\nBest iteration: {final_model.best_iteration}\n")

# Predictions
y_pred_train = final_model.predict(X_train, num_iteration=final_model.best_iteration)
y_pred_test = final_model.predict(X_test, num_iteration=final_model.best_iteration)

# Evaluation
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_mae = mean_absolute_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

overfitting_ratio = test_rmse / train_rmse

print("=" * 80)
print("RESULTS")
print("=" * 80)

print("\nTRAIN METRICS:")
print(f"  RMSE: ${train_rmse:,.2f}")
print(f"  MAE: ${train_mae:,.2f}")
print(f"  R²: {train_r2:.4f}")

print("\nTEST METRICS:")
print(f"  RMSE: ${test_rmse:,.2f}")
print(f"  MAE: ${test_mae:,.2f}")
print(f"  R²: {test_r2:.4f}")

print(f"\nOVERFITTING RATIO: {overfitting_ratio:.2f}x")

# Compare with baseline
baseline_rmse = 237.57
baseline_overfitting = 2.57

improvement_rmse = ((baseline_rmse - test_rmse) / baseline_rmse) * 100
improvement_overfitting = ((baseline_overfitting - overfitting_ratio) / baseline_overfitting) * 100

print("\n" + "=" * 80)
print("COMPARISON WITH BASELINE")
print("=" * 80)

print(f"\nBaseline:  RMSE ${baseline_rmse:,.2f} | Overfitting {baseline_overfitting:.2f}x")
print(f"Optimized: RMSE ${test_rmse:,.2f} | Overfitting {overfitting_ratio:.2f}x")

if test_rmse < baseline_rmse:
    print(f"\nIMPROVEMENT: {improvement_rmse:.1f}% better RMSE")
else:
    print(f"\nDEGRADATION: {-improvement_rmse:.1f}% worse RMSE")

if overfitting_ratio < baseline_overfitting:
    print(f"IMPROVEMENT: {improvement_overfitting:.1f}% less overfitting")
else:
    print(f"DEGRADATION: {-improvement_overfitting:.1f}% more overfitting")

# Feature importance
importance = final_model.feature_importance(importance_type='gain')
importance_df = pd.DataFrame({
    'feature': SELECTED_FEATURES,
    'importance': importance,
    'importance_pct': (importance / importance.sum() * 100)
}).sort_values('importance', ascending=False)

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE")
print("=" * 80)
for i, row in enumerate(importance_df.itertuples(), 1):
    print(f"  {i:2d}. {row.feature:40s} {row.importance_pct:6.2f}%")

# Save model
print("\n" + "=" * 80)
print("SAVING MODEL")
print("=" * 80)

model_path = MODELS_DIR / "lightgbm_bidfee_optuna.txt"
final_model.save_model(str(model_path))

metadata = {
    "model_type": "LightGBM Regression (Optuna Optimized)",
    "phase": "1A - Bid Fee Prediction",
    "target_variable": "BidFee",
    "optimization": "Optuna hyperparameter tuning (100 trials)",
    "num_features": len(SELECTED_FEATURES),
    "selected_features": SELECTED_FEATURES,
    "data_range": {
        "start_date": df_recent[DATE_COLUMN].min().strftime('%Y-%m-%d'),
        "end_date": df_recent[DATE_COLUMN].max().strftime('%Y-%m-%d'),
        "total_samples": int(len(df_recent)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test))
    },
    "best_iteration": int(final_model.best_iteration),
    "best_hyperparameters": study.best_params,
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
        "baseline_rmse": float(baseline_rmse),
        "baseline_overfitting": float(baseline_overfitting),
        "improvement_rmse_pct": float(improvement_rmse),
        "improvement_overfitting_pct": float(improvement_overfitting)
    },
    "optuna_study": {
        "n_trials": len(study.trials),
        "best_trial": study.best_trial.number,
        "best_score": float(study.best_value)
    }
}

metadata_path = REPORTS_DIR / "lightgbm_optuna_results.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Model saved: {model_path}")
print(f"Metadata saved: {metadata_path}")

print("\n" + "=" * 80)
print("OPTUNA LIGHTGBM TUNING COMPLETE")
print("=" * 80)
print(f"Test RMSE: ${test_rmse:,.2f}")
print(f"Overfitting: {overfitting_ratio:.2f}x")
print(f"vs Baseline: ${baseline_rmse:,.2f}, {baseline_overfitting:.2f}x\n")
