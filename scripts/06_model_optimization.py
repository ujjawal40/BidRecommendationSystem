"""
Model Optimization with Cross-Validation and Hyperparameter Tuning
===================================================================
Phase 1A: Optimize LightGBM model for bid fee prediction

This script implements:
- TimeSeriesSplit cross-validation for robust evaluation
- Optuna hyperparameter optimization
- Final model training with best parameters
- Performance comparison with baseline

Author: Bid Recommendation System
Date: 2026-01-07
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

# Import configuration
from config.model_config import (
    FEATURES_DATA,
    TARGET_COLUMN,
    DATE_COLUMN,
    EXCLUDE_COLUMNS,
    LIGHTGBM_CONFIG,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    RANDOM_SEED,
)

warnings.filterwarnings('ignore')


class LightGBMOptimizer:
    """
    Optimize LightGBM model using cross-validation and Optuna.

    This class handles:
    - TimeSeriesSplit cross-validation
    - Optuna hyperparameter optimization
    - Model training with best parameters
    - Performance comparison
    """

    def __init__(self, n_trials=100, n_splits=5):
        """
        Initialize optimizer.

        Parameters
        ----------
        n_trials : int
            Number of Optuna optimization trials
        n_splits : int
            Number of TimeSeriesSplit folds
        """
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.best_params = None
        self.best_model = None
        self.cv_results = []
        self.baseline_score = None
        self.optimized_score = None

        print("=" * 80)
        print("LIGHTGBM MODEL OPTIMIZATION")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Optimization trials: {n_trials}")
        print(f"Cross-validation folds: {n_splits}\n")

    def load_data(self):
        """Load and prepare data."""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)

        # Load data
        df = pd.read_csv(FEATURES_DATA)
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

        print(f"✓ Data loaded: {df.shape[0]:,} rows, {df.shape[1]:,} columns")

        # Prepare features
        feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        X = df[numeric_features].fillna(0).values
        y = df[TARGET_COLUMN].values

        print(f"✓ Features prepared: {X.shape[1]} features")
        print(f"  Target mean: ${y.mean():,.2f}")
        print(f"  Target std: ${y.std():,.2f}\n")

        self.X = X
        self.y = y
        self.feature_names = numeric_features

        return X, y

    def baseline_cross_validation(self):
        """
        Evaluate baseline model with default parameters using cross-validation.

        Returns
        -------
        dict
            Cross-validation results
        """
        print("=" * 80)
        print("BASELINE MODEL CROSS-VALIDATION")
        print("=" * 80)

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        cv_scores = {'rmse': [], 'mae': [], 'r2': []}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X), 1):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            # Train model with baseline params
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            params = LIGHTGBM_CONFIG['params'].copy()

            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )

            # Evaluate
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            cv_scores['rmse'].append(rmse)
            cv_scores['mae'].append(mae)
            cv_scores['r2'].append(r2)

            print(f"  Fold {fold}/{self.n_splits}: RMSE=${rmse:.2f}, MAE=${mae:.2f}, R²={r2:.4f}")

        # Calculate mean scores
        baseline_results = {
            'rmse_mean': np.mean(cv_scores['rmse']),
            'rmse_std': np.std(cv_scores['rmse']),
            'mae_mean': np.mean(cv_scores['mae']),
            'mae_std': np.std(cv_scores['mae']),
            'r2_mean': np.mean(cv_scores['r2']),
            'r2_std': np.std(cv_scores['r2']),
        }

        self.baseline_score = baseline_results

        print(f"\n✓ Baseline CV Results:")
        print(f"  RMSE: ${baseline_results['rmse_mean']:.2f} ± ${baseline_results['rmse_std']:.2f}")
        print(f"  MAE:  ${baseline_results['mae_mean']:.2f} ± ${baseline_results['mae_std']:.2f}")
        print(f"  R²:   {baseline_results['r2_mean']:.4f} ± {baseline_results['r2_std']:.4f}\n")

        return baseline_results

    def objective(self, trial):
        """
        Optuna objective function for hyperparameter optimization.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object

        Returns
        -------
        float
            Mean RMSE across cross-validation folds
        """
        # Suggest hyperparameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'verbose': -1,
        }

        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        cv_rmse = []

        for train_idx, val_idx in tscv.split(self.X):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )

            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            cv_rmse.append(rmse)

        return np.mean(cv_rmse)

    def optimize_hyperparameters(self):
        """
        Run Optuna hyperparameter optimization.

        Returns
        -------
        dict
            Best hyperparameters found
        """
        print("=" * 80)
        print("OPTUNA HYPERPARAMETER OPTIMIZATION")
        print("=" * 80)
        print(f"Running {self.n_trials} trials with {self.n_splits}-fold CV...")
        print("This may take several minutes...\n")

        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=RANDOM_SEED)
        )

        # Optimize
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        self.best_params = study.best_params

        print(f"\n✓ Optimization complete!")
        print(f"  Best RMSE: ${study.best_value:.2f}")
        print(f"  Best parameters:")
        for param, value in self.best_params.items():
            print(f"    {param}: {value}")

        # Save optimization history
        self.save_optimization_history(study)

        return self.best_params

    def train_final_model(self):
        """
        Train final model with best parameters using full training data.

        Returns
        -------
        lgb.Booster
            Trained LightGBM model
        """
        print("\n" + "=" * 80)
        print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
        print("=" * 80)

        # 80/20 train/test split
        split_idx = int(len(self.X) * 0.8)
        X_train = self.X[:split_idx]
        X_test = self.X[split_idx:]
        y_train = self.y[:split_idx]
        y_test = self.y[split_idx:]

        # Prepare parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'verbose': -1,
        }
        params.update(self.best_params)

        # Train
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        print("Training in progress...")
        self.best_model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )

        # Evaluate
        y_pred = self.best_model.predict(X_test, num_iteration=self.best_model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.optimized_score = {'rmse': rmse, 'mae': mae, 'r2': r2}

        print(f"\n✓ Final model performance:")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAE:  ${mae:.2f}")
        print(f"  R²:   {r2:.4f}\n")

        return self.best_model

    def compare_performance(self):
        """Compare baseline vs optimized model performance."""
        print("=" * 80)
        print("PERFORMANCE COMPARISON")
        print("=" * 80)

        print("\nBaseline Model (Default Parameters):")
        print(f"  RMSE: ${self.baseline_score['rmse_mean']:.2f} ± ${self.baseline_score['rmse_std']:.2f}")
        print(f"  MAE:  ${self.baseline_score['mae_mean']:.2f} ± ${self.baseline_score['mae_std']:.2f}")
        print(f"  R²:   {self.baseline_score['r2_mean']:.4f} ± {self.baseline_score['r2_std']:.4f}")

        print("\nOptimized Model (Optuna Tuned):")
        print(f"  RMSE: ${self.optimized_score['rmse']:.2f}")
        print(f"  MAE:  ${self.optimized_score['mae']:.2f}")
        print(f"  R²:   {self.optimized_score['r2']:.4f}")

        # Calculate improvement
        rmse_improvement = (self.baseline_score['rmse_mean'] - self.optimized_score['rmse']) / self.baseline_score['rmse_mean'] * 100
        mae_improvement = (self.baseline_score['mae_mean'] - self.optimized_score['mae']) / self.baseline_score['mae_mean'] * 100
        r2_improvement = (self.optimized_score['r2'] - self.baseline_score['r2_mean']) / self.baseline_score['r2_mean'] * 100

        print("\nImprovement:")
        print(f"  RMSE: {rmse_improvement:+.2f}%")
        print(f"  MAE:  {mae_improvement:+.2f}%")
        print(f"  R²:   {r2_improvement:+.2f}%\n")

        return {
            'baseline': self.baseline_score,
            'optimized': self.optimized_score,
            'improvement': {
                'rmse_pct': rmse_improvement,
                'mae_pct': mae_improvement,
                'r2_pct': r2_improvement
            }
        }

    def save_optimization_history(self, study):
        """Save Optuna optimization history plot."""
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        save_path = FIGURES_DIR / "optuna_optimization_history.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Optimization history saved: {save_path}")

        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        save_path = FIGURES_DIR / "optuna_param_importances.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Parameter importances saved: {save_path}")

    def save_model(self):
        """Save optimized model and metadata."""
        print("=" * 80)
        print("SAVING OPTIMIZED MODEL")
        print("=" * 80)

        # Save model
        model_path = MODELS_DIR / "lightgbm_bidfee_model_optimized.txt"
        self.best_model.save_model(str(model_path))
        print(f"✓ Model saved: {model_path}")

        # Save metadata
        metadata = {
            "model_type": "LightGBM (Optuna Optimized)",
            "phase": "1A - Bid Fee Prediction",
            "target_variable": TARGET_COLUMN,
            "num_features": len(self.feature_names),
            "features": self.feature_names,
            "best_parameters": self.best_params,
            "best_iteration": int(self.best_model.best_iteration),
            "metrics": {
                "rmse": float(self.optimized_score['rmse']),
                "mae": float(self.optimized_score['mae']),
                "r2": float(self.optimized_score['r2'])
            },
            "baseline_comparison": {
                "baseline_rmse": float(self.baseline_score['rmse_mean']),
                "optimized_rmse": float(self.optimized_score['rmse']),
                "improvement_pct": float((self.baseline_score['rmse_mean'] - self.optimized_score['rmse']) / self.baseline_score['rmse_mean'] * 100)
            },
            "optimization_config": {
                "n_trials": self.n_trials,
                "n_cv_folds": self.n_splits
            },
            "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        metadata_path = MODELS_DIR / "lightgbm_metadata_optimized.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved: {metadata_path}\n")

    def run_full_pipeline(self):
        """Execute complete optimization pipeline."""
        # Load data
        self.load_data()

        # Baseline evaluation
        self.baseline_cross_validation()

        # Hyperparameter optimization
        self.optimize_hyperparameters()

        # Train final model
        self.train_final_model()

        # Compare performance
        comparison = self.compare_performance()

        # Save model
        self.save_model()

        print("=" * 80)
        print("OPTIMIZATION PIPELINE COMPLETE")
        print("=" * 80)
        print(f"✓ Model optimized and saved successfully\n")

        return comparison


def main():
    """Main execution function."""
    # Create optimizer (50 trials, 3-fold CV for efficiency)
    optimizer = LightGBMOptimizer(n_trials=50, n_splits=3)

    # Run optimization
    results = optimizer.run_full_pipeline()

    return optimizer, results


if __name__ == "__main__":
    optimizer, results = main()
