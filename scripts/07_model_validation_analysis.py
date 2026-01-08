"""
Model Validation and Backtesting Analysis
==========================================
Phase 1A: Comprehensive model validation to check for overfitting

This script implements:
- Train vs Test performance comparison
- Overfitting detection
- Walk-forward backtesting
- Time-based cross-validation analysis
- Residual analysis
- Performance stability over time

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
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings

# Import configuration
from config.model_config import (
    FEATURES_DATA,
    TARGET_COLUMN,
    DATE_COLUMN,
    EXCLUDE_COLUMNS,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
)

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ModelValidator:
    """
    Comprehensive model validation and backtesting.

    This class performs:
    - Train/test performance comparison
    - Overfitting analysis
    - Walk-forward backtesting
    - Residual analysis
    - Time-based validation
    """

    def __init__(self, model_path):
        """
        Initialize validator with trained model.

        Parameters
        ----------
        model_path : str
            Path to trained LightGBM model
        """
        self.model_path = model_path
        self.model = lgb.Booster(model_file=str(model_path))

        print("=" * 80)
        print("MODEL VALIDATION AND BACKTESTING ANALYSIS")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model: {model_path.name}\n")

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
        dates = df[DATE_COLUMN].values

        print(f"✓ Features prepared: {X.shape[1]} features")
        print(f"  Date range: {df[DATE_COLUMN].min()} to {df[DATE_COLUMN].max()}")
        print(f"  Target mean: ${y.mean():,.2f}")
        print(f"  Target std: ${y.std():,.2f}\n")

        self.X = X
        self.y = y
        self.dates = dates
        self.df = df

        return X, y, dates

    def train_test_comparison(self):
        """
        Compare performance on train and test sets to detect overfitting.

        Returns
        -------
        dict
            Train and test metrics
        """
        print("=" * 80)
        print("TRAIN VS TEST PERFORMANCE COMPARISON")
        print("=" * 80)

        # 80/20 split (same as model training)
        split_idx = int(len(self.X) * 0.8)

        X_train = self.X[:split_idx]
        X_test = self.X[split_idx:]
        y_train = self.y[:split_idx]
        y_test = self.y[split_idx:]

        # Get dates for splits
        train_dates = self.dates[:split_idx]
        test_dates = self.dates[split_idx:]

        print(f"Train set: {len(X_train):,} samples ({train_dates[0]} to {train_dates[-1]})")
        print(f"Test set:  {len(X_test):,} samples ({test_dates[0]} to {test_dates[-1]})\n")

        # Predict on both sets
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Calculate metrics for train set
        train_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'mae': mean_absolute_error(y_train, y_train_pred),
            'r2': r2_score(y_train, y_train_pred),
        }

        # Calculate metrics for test set
        test_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'r2': r2_score(y_test, y_test_pred),
        }

        # Display results
        print("TRAIN SET PERFORMANCE:")
        print(f"  RMSE: ${train_metrics['rmse']:,.2f}")
        print(f"  MAE:  ${train_metrics['mae']:,.2f}")
        print(f"  R²:   {train_metrics['r2']:.4f}")

        print("\nTEST SET PERFORMANCE:")
        print(f"  RMSE: ${test_metrics['rmse']:,.2f}")
        print(f"  MAE:  ${test_metrics['mae']:,.2f}")
        print(f"  R²:   {test_metrics['r2']:.4f}")

        # Calculate overfitting indicators
        rmse_ratio = test_metrics['rmse'] / train_metrics['rmse']
        mae_ratio = test_metrics['mae'] / train_metrics['mae']
        r2_diff = train_metrics['r2'] - test_metrics['r2']

        print("\nOVERFITTING INDICATORS:")
        print(f"  RMSE ratio (test/train): {rmse_ratio:.3f}")
        print(f"  MAE ratio (test/train):  {mae_ratio:.3f}")
        print(f"  R² difference (train-test): {r2_diff:.4f}")

        # Interpretation
        print("\nINTERPRETATION:")
        if rmse_ratio < 1.15 and mae_ratio < 1.15 and r2_diff < 0.05:
            print("  ✓ NO SIGNIFICANT OVERFITTING DETECTED")
            print("    - Test performance is close to train performance")
            print("    - Model generalizes well to unseen data")
        elif rmse_ratio < 1.3 and mae_ratio < 1.3 and r2_diff < 0.10:
            print("  ⚠ MILD OVERFITTING")
            print("    - Slight performance degradation on test set")
            print("    - Still acceptable for production use")
        else:
            print("  ✗ SIGNIFICANT OVERFITTING")
            print("    - Large performance gap between train and test")
            print("    - Model may not generalize well")

        print()

        self.train_metrics = train_metrics
        self.test_metrics = test_metrics

        return {
            'train': train_metrics,
            'test': test_metrics,
            'overfitting_indicators': {
                'rmse_ratio': rmse_ratio,
                'mae_ratio': mae_ratio,
                'r2_diff': r2_diff
            }
        }

    def walk_forward_backtest(self, n_splits=5):
        """
        Perform walk-forward backtesting to validate model stability.

        Parameters
        ----------
        n_splits : int
            Number of time-based splits for backtesting

        Returns
        -------
        dict
            Backtesting results
        """
        print("=" * 80)
        print("WALK-FORWARD BACKTESTING")
        print("=" * 80)
        print(f"Number of splits: {n_splits}")
        print("Simulating production-like scenario with expanding window...\n")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        backtest_results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(self.X), 1):
            X_test_fold = self.X[test_idx]
            y_test_fold = self.y[test_idx]
            test_dates_fold = self.dates[test_idx]

            # Make predictions
            y_pred_fold = self.model.predict(X_test_fold)

            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred_fold))
            mae = mean_absolute_error(y_test_fold, y_pred_fold)
            r2 = r2_score(y_test_fold, y_pred_fold)

            # Store results
            backtest_results.append({
                'fold': fold,
                'n_samples': len(test_idx),
                'date_start': test_dates_fold[0],
                'date_end': test_dates_fold[-1],
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            })

            print(f"Fold {fold}/{n_splits}:")
            print(f"  Period: {test_dates_fold[0]} to {test_dates_fold[-1]}")
            print(f"  Samples: {len(test_idx):,}")
            print(f"  RMSE: ${rmse:,.2f}, MAE: ${mae:,.2f}, R²: {r2:.4f}")

        # Calculate aggregate statistics
        rmse_scores = [r['rmse'] for r in backtest_results]
        mae_scores = [r['mae'] for r in backtest_results]
        r2_scores = [r['r2'] for r in backtest_results]

        print(f"\nBACKTEST SUMMARY:")
        print(f"  RMSE: ${np.mean(rmse_scores):,.2f} ± ${np.std(rmse_scores):,.2f}")
        print(f"  MAE:  ${np.mean(mae_scores):,.2f} ± ${np.std(mae_scores):,.2f}")
        print(f"  R²:   {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

        # Performance stability
        rmse_cv = (np.std(rmse_scores) / np.mean(rmse_scores)) * 100
        print(f"\nPERFORMANCE STABILITY:")
        print(f"  RMSE coefficient of variation: {rmse_cv:.2f}%")

        if rmse_cv < 15:
            print("  ✓ STABLE - Consistent performance across time periods")
        elif rmse_cv < 30:
            print("  ⚠ MODERATE - Some variation across time periods")
        else:
            print("  ✗ UNSTABLE - Significant variation across time periods")

        print()

        self.backtest_results = backtest_results

        return {
            'folds': backtest_results,
            'summary': {
                'rmse_mean': np.mean(rmse_scores),
                'rmse_std': np.std(rmse_scores),
                'mae_mean': np.mean(mae_scores),
                'mae_std': np.std(mae_scores),
                'r2_mean': np.mean(r2_scores),
                'r2_std': np.std(r2_scores),
                'rmse_cv': rmse_cv
            }
        }

    def residual_analysis(self):
        """
        Analyze prediction residuals to check for systematic errors.

        Returns
        -------
        dict
            Residual statistics
        """
        print("=" * 80)
        print("RESIDUAL ANALYSIS")
        print("=" * 80)

        # Use test set
        split_idx = int(len(self.X) * 0.8)
        X_test = self.X[split_idx:]
        y_test = self.y[split_idx:]

        # Get predictions
        y_pred = self.model.predict(X_test)

        # Calculate residuals
        residuals = y_test - y_pred

        # Residual statistics
        residual_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'median': np.median(residuals),
            'q25': np.percentile(residuals, 25),
            'q75': np.percentile(residuals, 75),
        }

        print("RESIDUAL STATISTICS:")
        print(f"  Mean: ${residual_stats['mean']:,.2f}")
        print(f"  Std Dev: ${residual_stats['std']:,.2f}")
        print(f"  Median: ${residual_stats['median']:,.2f}")
        print(f"  Min: ${residual_stats['min']:,.2f}")
        print(f"  Max: ${residual_stats['max']:,.2f}")
        print(f"  IQR: ${residual_stats['q75'] - residual_stats['q25']:,.2f}")

        # Check for bias
        mean_residual_pct = (residual_stats['mean'] / np.mean(y_test)) * 100

        print(f"\nBIAS ANALYSIS:")
        print(f"  Mean residual as % of target: {mean_residual_pct:.2f}%")

        if abs(mean_residual_pct) < 1:
            print("  ✓ NO SYSTEMATIC BIAS - Model predictions are unbiased")
        elif abs(mean_residual_pct) < 3:
            print("  ⚠ MILD BIAS - Small systematic error present")
        else:
            print("  ✗ SIGNIFICANT BIAS - Model has systematic over/under prediction")

        print()

        # Create residual plots
        self.plot_residuals(y_test, y_pred, residuals)

        return residual_stats

    def plot_residuals(self, y_true, y_pred, residuals):
        """Create residual diagnostic plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.3, s=10)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predicted Values ($)')
        axes[0, 0].set_ylabel('Residuals ($)')
        axes[0, 0].set_title('Residuals vs Predicted Values')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Histogram of residuals
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Residuals ($)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Residuals')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Actual vs Predicted
        axes[1, 0].scatter(y_true, y_pred, alpha=0.3, s=10)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[1, 0].set_xlabel('Actual Values ($)')
        axes[1, 0].set_ylabel('Predicted Values ($)')
        axes[1, 0].set_title('Actual vs Predicted')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normality Check)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = FIGURES_DIR / "model_residual_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Residual plots saved: {save_path}")

    def performance_over_time(self):
        """Analyze model performance stability over time."""
        print("=" * 80)
        print("PERFORMANCE OVER TIME")
        print("=" * 80)

        # Use test set
        split_idx = int(len(self.X) * 0.8)
        X_test = self.X[split_idx:]
        y_test = self.y[split_idx:]
        dates_test = self.dates[split_idx:]

        # Get predictions
        y_pred = self.model.predict(X_test)

        # Create DataFrame with results
        results_df = pd.DataFrame({
            'date': pd.to_datetime(dates_test),
            'actual': y_test,
            'predicted': y_pred,
            'error': np.abs(y_test - y_pred)
        })

        # Group by month and calculate metrics
        results_df['year_month'] = results_df['date'].dt.to_period('M')
        monthly_perf = results_df.groupby('year_month').agg({
            'error': ['mean', 'std'],
            'actual': 'count'
        }).reset_index()

        monthly_perf.columns = ['year_month', 'mae', 'std', 'count']

        print(f"Test period: {results_df['date'].min()} to {results_df['date'].max()}")
        print(f"Total months: {len(monthly_perf)}")
        print(f"\nMonthly MAE statistics:")
        print(f"  Mean: ${monthly_perf['mae'].mean():,.2f}")
        print(f"  Min:  ${monthly_perf['mae'].min():,.2f}")
        print(f"  Max:  ${monthly_perf['mae'].max():,.2f}")
        print(f"  Std:  ${monthly_perf['mae'].std():,.2f}\n")

        # Plot performance over time
        fig, ax = plt.subplots(figsize=(14, 6))

        x_labels = [str(ym) for ym in monthly_perf['year_month']]
        ax.plot(range(len(monthly_perf)), monthly_perf['mae'], marker='o', linewidth=2, markersize=8)
        ax.fill_between(range(len(monthly_perf)),
                        monthly_perf['mae'] - monthly_perf['std'],
                        monthly_perf['mae'] + monthly_perf['std'],
                        alpha=0.3)

        ax.set_xlabel('Month')
        ax.set_ylabel('Mean Absolute Error ($)')
        ax.set_title('Model Performance Over Time (Test Set)')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(monthly_perf)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')

        plt.tight_layout()
        save_path = FIGURES_DIR / "model_performance_over_time.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Performance over time plot saved: {save_path}")

        return monthly_perf

    def generate_validation_report(self, train_test_comp, backtest_results, residual_stats):
        """Generate comprehensive validation report."""
        print("=" * 80)
        print("GENERATING VALIDATION REPORT")
        print("=" * 80)

        report = {
            "model_info": {
                "model_path": str(self.model_path),
                "validation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "dataset_size": len(self.X),
                "train_size": int(len(self.X) * 0.8),
                "test_size": len(self.X) - int(len(self.X) * 0.8)
            },
            "train_test_comparison": train_test_comp,
            "backtesting": backtest_results,
            "residual_analysis": {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in residual_stats.items()
            },
            "evaluation_metrics": {
                "rmse_test": float(self.test_metrics['rmse']),
                "mae_test": float(self.test_metrics['mae']),
                "r2_test": float(self.test_metrics['r2'])
            },
            "validation_summary": {
                "overfitting_status": self._get_overfitting_status(train_test_comp['overfitting_indicators']),
                "performance_stability": self._get_stability_status(backtest_results['summary']['rmse_cv']),
                "bias_status": self._get_bias_status(residual_stats['mean'], np.mean(self.y)),
                "production_ready": self._is_production_ready(train_test_comp, backtest_results)
            }
        }

        # Save report
        report_path = REPORTS_DIR / "model_validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"✓ Validation report saved: {report_path}\n")

        return report

    def _get_overfitting_status(self, indicators):
        """Determine overfitting status from indicators."""
        rmse_ratio = indicators['rmse_ratio']
        r2_diff = indicators['r2_diff']

        if rmse_ratio < 1.15 and r2_diff < 0.05:
            return "NO_OVERFITTING"
        elif rmse_ratio < 1.3 and r2_diff < 0.10:
            return "MILD_OVERFITTING"
        else:
            return "SIGNIFICANT_OVERFITTING"

    def _get_stability_status(self, rmse_cv):
        """Determine performance stability status."""
        if rmse_cv < 15:
            return "STABLE"
        elif rmse_cv < 30:
            return "MODERATE"
        else:
            return "UNSTABLE"

    def _get_bias_status(self, mean_residual, mean_target):
        """Determine bias status."""
        bias_pct = abs(mean_residual / mean_target) * 100

        if bias_pct < 1:
            return "NO_BIAS"
        elif bias_pct < 3:
            return "MILD_BIAS"
        else:
            return "SIGNIFICANT_BIAS"

    def _is_production_ready(self, train_test_comp, backtest_results):
        """Determine if model is production ready."""
        overfitting_ok = train_test_comp['overfitting_indicators']['rmse_ratio'] < 1.3
        stability_ok = backtest_results['summary']['rmse_cv'] < 30
        performance_ok = train_test_comp['test']['r2'] > 0.95

        return overfitting_ok and stability_ok and performance_ok

    def run_full_validation(self):
        """Execute complete validation pipeline."""
        # Load data
        self.load_data()

        # Train/test comparison
        train_test_comp = self.train_test_comparison()

        # Walk-forward backtesting
        backtest_results = self.walk_forward_backtest(n_splits=5)

        # Residual analysis
        residual_stats = self.residual_analysis()

        # Performance over time
        self.performance_over_time()

        # Generate report
        report = self.generate_validation_report(train_test_comp, backtest_results, residual_stats)

        print("=" * 80)
        print("VALIDATION COMPLETE")
        print("=" * 80)
        print(f"✓ All validation checks completed")
        print(f"✓ Overfitting: {report['validation_summary']['overfitting_status']}")
        print(f"✓ Stability: {report['validation_summary']['performance_stability']}")
        print(f"✓ Bias: {report['validation_summary']['bias_status']}")
        print(f"✓ Production Ready: {report['validation_summary']['production_ready']}\n")

        return report


def main():
    """Main execution function."""
    # Validate optimized model
    model_path = MODELS_DIR / "lightgbm_bidfee_model_optimized.txt"

    if not model_path.exists():
        # Fall back to non-optimized model
        model_path = MODELS_DIR / "lightgbm_bidfee_model.txt"

    validator = ModelValidator(model_path)
    report = validator.run_full_validation()

    return validator, report


if __name__ == "__main__":
    validator, report = main()
