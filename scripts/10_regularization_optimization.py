"""
Regularization-Focused Hyperparameter Optimization
===================================================
Re-optimize LightGBM with 12 selected features, focusing on reducing overfitting

Target: Reduce overfitting ratio from 2.64x to <1.5x

Strategy:
- Strong L1/L2 regularization
- Simpler tree structure
- More aggressive early stopping
- Conservative learning rate

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
import optuna
from optuna.samplers import TPESampler
import json
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

from config.model_config import (
    FEATURES_DATA, TARGET_COLUMN, DATE_COLUMN, EXCLUDE_COLUMNS,
    MODELS_DIR, REPORTS_DIR, RANDOM_SEED,
)

warnings.filterwarnings('ignore')

# Load the 12 selected features
with open(MODELS_DIR / "lightgbm_metadata_feature_selected.json", 'r') as f:
    metadata = json.load(f)
    SELECTED_FEATURES = metadata['selected_features']

print("=" * 80)
print("REGULARIZATION-FOCUSED HYPERPARAMETER OPTIMIZATION")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Goal: Reduce overfitting ratio from 2.64x to <1.5x")
print(f"Features: {len(SELECTED_FEATURES)} selected features\n")

# Load data
print("Loading data...")
df = pd.read_csv(FEATURES_DATA)
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

X = df[SELECTED_FEATURES].fillna(0).values
y = df[TARGET_COLUMN].values

print(f"✓ Data loaded: {len(X):,} samples\n")

# Baseline performance (current model)
print("=" * 80)
print("BASELINE: CURRENT FEATURE-SELECTED MODEL")
print("=" * 80)

split_idx = int(len(X) * 0.8)
X_train_base = X[:split_idx]
X_test_base = X[split_idx:]
y_train_base = y[:split_idx]
y_test_base = y[split_idx:]

train_data = lgb.Dataset(X_train_base, label=y_train_base)
val_data = lgb.Dataset(X_test_base, label=y_test_base, reference=train_data)

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

baseline_model = lgb.train(
    baseline_params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

y_pred_base = baseline_model.predict(X_test_base, num_iteration=baseline_model.best_iteration)
y_train_pred_base = baseline_model.predict(X_train_base, num_iteration=baseline_model.best_iteration)

baseline_train_rmse = np.sqrt(mean_squared_error(y_train_base, y_train_pred_base))
baseline_test_rmse = np.sqrt(mean_squared_error(y_test_base, y_pred_base))
baseline_overfitting = baseline_test_rmse / baseline_train_rmse

print(f"Baseline Performance:")
print(f"  Train RMSE: ${baseline_train_rmse:,.2f}")
print(f"  Test RMSE:  ${baseline_test_rmse:,.2f}")
print(f"  Overfitting: {baseline_overfitting:.2f}x")
print(f"  Target: <1.5x\n")

# Optuna optimization
class RegularizationOptimizer:
    """Optimize for low overfitting using strong regularization."""

    def __init__(self, X, y, n_trials=100, n_splits=3):
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.best_params = None
        self.best_overfitting = None

    def objective(self, trial):
        """Optuna objective: minimize overfitting while maintaining performance."""

        # Suggest hyperparameters with focus on regularization
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',

            # REGULARIZATION (main focus)
            'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 20.0),  # L1 penalty
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20.0),  # L2 penalty
            'min_child_samples': trial.suggest_int('min_child_samples', 50, 200),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 0.1),

            # TREE STRUCTURE (simpler trees)
            'num_leaves': trial.suggest_int('num_leaves', 10, 31),  # Reduced from 100
            'max_depth': trial.suggest_int('max_depth', 3, 8),  # Reduced from 12

            # LEARNING (conservative)
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),

            # SAMPLING (prevent overfitting)
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.8),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.8),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),

            'random_state': RANDOM_SEED,
            'verbose': -1,
        }

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        train_rmses = []
        test_rmses = []

        for train_idx, val_idx in tscv.split(self.X):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=2000,  # More rounds, let early stopping decide
                valid_sets=[train_data, val_data],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]  # More patient
            )

            # Evaluate on both train and validation
            y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
            y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)

            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

            train_rmses.append(train_rmse)
            test_rmses.append(val_rmse)

        # Calculate overfitting ratio
        avg_train_rmse = np.mean(train_rmses)
        avg_test_rmse = np.mean(test_rmses)
        overfitting_ratio = avg_test_rmse / avg_train_rmse

        # We want to minimize a weighted combination:
        # - Overfitting ratio (main priority)
        # - Test RMSE (secondary - maintain good performance)

        # Penalize heavily if overfitting is bad
        if overfitting_ratio > 2.0:
            penalty = (overfitting_ratio - 2.0) * 100
        else:
            penalty = 0

        # Objective: minimize test RMSE + overfitting penalty
        objective_value = avg_test_rmse + penalty

        # Store overfitting ratio for analysis
        trial.set_user_attr('overfitting_ratio', overfitting_ratio)
        trial.set_user_attr('train_rmse', avg_train_rmse)
        trial.set_user_attr('test_rmse', avg_test_rmse)

        return objective_value

    def optimize(self):
        """Run optimization."""
        print("=" * 80)
        print("OPTUNA OPTIMIZATION")
        print("=" * 80)
        print(f"Running {self.n_trials} trials with {self.n_splits}-fold CV")
        print("Focus: Minimize overfitting ratio")
        print("This will take approximately 20-30 minutes...\n")

        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=RANDOM_SEED)
        )

        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        self.best_params = study.best_params
        self.best_trial = study.best_trial

        # Get overfitting ratio for best trial
        best_overfitting = self.best_trial.user_attrs['overfitting_ratio']
        best_train_rmse = self.best_trial.user_attrs['train_rmse']
        best_test_rmse = self.best_trial.user_attrs['test_rmse']

        print(f"\n✓ Optimization complete!")
        print(f"\nBEST TRIAL RESULTS:")
        print(f"  Train RMSE: ${best_train_rmse:,.2f}")
        print(f"  Test RMSE:  ${best_test_rmse:,.2f}")
        print(f"  Overfitting: {best_overfitting:.2f}x")
        print(f"\nBest Hyperparameters:")
        for param, value in sorted(self.best_params.items()):
            if isinstance(value, float):
                print(f"  {param}: {value:.4f}")
            else:
                print(f"  {param}: {value}")

        self.study = study
        return study

# Run optimization
optimizer = RegularizationOptimizer(X, y, n_trials=50, n_splits=3)
study = optimizer.optimize()

# Train final model with best parameters
print("\n" + "=" * 80)
print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
print("=" * 80)

X_train = X[:split_idx]
X_test = X[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

train_data = lgb.Dataset(X_train, label=y_train, feature_name=SELECTED_FEATURES)
val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

final_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'random_state': RANDOM_SEED,
    'verbose': -1,
}
final_params.update(optimizer.best_params)

print("Training in progress...")
final_model = lgb.train(
    final_params,
    train_data,
    num_boost_round=2000,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'valid'],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
)

# Final evaluation
y_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration)
y_train_pred = final_model.predict(X_train, num_iteration=final_model.best_iteration)

final_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
final_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
final_mae = mean_absolute_error(y_test, y_pred)
final_r2 = r2_score(y_test, y_pred)
final_overfitting = final_test_rmse / final_train_rmse

print(f"\n✓ Final model trained")
print(f"\nFINAL PERFORMANCE:")
print(f"  Train RMSE: ${final_train_rmse:,.2f}")
print(f"  Test RMSE:  ${final_test_rmse:,.2f}")
print(f"  Test MAE:   ${final_mae:,.2f}")
print(f"  Test R²:    {final_r2:.4f}")
print(f"  Overfitting: {final_overfitting:.2f}x")

print(f"\nCOMPARISON TO BASELINE:")
print(f"  Baseline overfitting: {baseline_overfitting:.2f}x")
print(f"  Final overfitting:    {final_overfitting:.2f}x")
improvement = (baseline_overfitting - final_overfitting) / baseline_overfitting * 100
print(f"  Improvement: {improvement:.1f}%")

if final_overfitting < 1.5:
    print(f"\n✅ SUCCESS! Overfitting ratio < 1.5x - PRODUCTION READY!")
elif final_overfitting < 2.0:
    print(f"\n⚠️  Good progress, but still slightly high. Consider further tuning.")
else:
    print(f"\n❌ Overfitting still above 2.0x. May need more aggressive regularization.")

# Save model
model_path = MODELS_DIR / "lightgbm_bidfee_model_regularized.txt"
final_model.save_model(str(model_path))

metadata = {
    "model_type": "LightGBM (Regularization Optimized)",
    "phase": "1A - Bid Fee Prediction",
    "target_variable": TARGET_COLUMN,
    "num_features": len(SELECTED_FEATURES),
    "selected_features": SELECTED_FEATURES,
    "best_iteration": int(final_model.best_iteration),
    "best_parameters": {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
                       for k, v in optimizer.best_params.items()},
    "metrics": {
        "train_rmse": float(final_train_rmse),
        "test_rmse": float(final_test_rmse),
        "test_mae": float(final_mae),
        "test_r2": float(final_r2),
        "overfitting_ratio": float(final_overfitting)
    },
    "baseline_comparison": {
        "baseline_overfitting": float(baseline_overfitting),
        "final_overfitting": float(final_overfitting),
        "improvement_pct": float(improvement)
    },
    "optimization_config": {
        "n_trials": optimizer.n_trials,
        "n_cv_folds": optimizer.n_splits,
        "focus": "regularization to reduce overfitting"
    },
    "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "production_ready": final_overfitting < 1.5
}

metadata_path = MODELS_DIR / "lightgbm_metadata_regularized.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ Model saved: {model_path}")
print(f"✓ Metadata saved: {metadata_path}")

print("\n" + "=" * 80)
print("REGULARIZATION OPTIMIZATION COMPLETE")
print("=" * 80)
print(f"✓ Reduced overfitting: {baseline_overfitting:.2f}x → {final_overfitting:.2f}x")
print(f"✓ Model ready for production: {final_overfitting < 1.5}\n")
