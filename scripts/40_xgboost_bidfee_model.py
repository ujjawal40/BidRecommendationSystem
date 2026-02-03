"""
XGBoost Bid Fee Regression Model
================================
Train XGBoost model for bid fee prediction to compare against LightGBM baseline.

XGBoost is known for:
- Strong regularization (L1, L2, tree pruning)
- Built-in handling of missing values
- Often performs better on structured/tabular data

Author: Bid Recommendation System
Date: 2026-02-03
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import xgboost as xgb
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings

from config.model_config import (
    FEATURES_DATA, DATE_COLUMN, TARGET_COLUMN,
    EXCLUDE_COLUMNS, JOBDATA_FEATURES_TO_EXCLUDE,
    DATA_START_DATE, USE_RECENT_DATA_ONLY,
    MODELS_DIR, REPORTS_DIR, FIGURES_DIR, RANDOM_SEED,
)

warnings.filterwarnings('ignore')

# ============================================================================
# XGBOOST CONFIGURATION
# ============================================================================

XGBOOST_REGRESSION_CONFIG = {
    "params": {
        "objective": "reg:squarederror",
        "eval_metric": ["rmse", "mae"],
        "booster": "gbtree",
        "max_depth": 8,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        "reg_alpha": 1.0,  # L1 regularization
        "reg_lambda": 1.0,  # L2 regularization
        "gamma": 0.1,  # Minimum loss reduction for split
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": 1,
    },
    "training": {
        "early_stopping_rounds": 50,
    },
}


def main():
    print("=" * 80)
    print("XGBOOST BID FEE REGRESSION MODEL")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Objective: Compare XGBoost vs LightGBM for bid fee prediction\n")

    # ========================================================================
    # LOAD AND FILTER DATA
    # ========================================================================
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    df = pd.read_csv(FEATURES_DATA)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    print(f"✓ Data loaded: {len(df):,} rows")
    print(f"  Date range: {df[DATE_COLUMN].min()} to {df[DATE_COLUMN].max()}")

    # Filter to recent data
    if USE_RECENT_DATA_ONLY:
        start_date = pd.Timestamp(DATA_START_DATE)
        original_count = len(df)
        df = df[df[DATE_COLUMN] >= start_date].copy()
        print(f"\n✓ Filtered to {DATA_START_DATE}+ data")
        print(f"  Removed: {original_count - len(df):,} older records")
        print(f"  Remaining: {len(df):,} records")

    # ========================================================================
    # PREPARE FEATURES
    # ========================================================================
    print("\n" + "=" * 80)
    print("PREPARING FEATURES")
    print("=" * 80)

    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    feature_cols = [col for col in feature_cols if col not in JOBDATA_FEATURES_TO_EXCLUDE]
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    print(f"Total features: {len(numeric_features)}")

    X = df[numeric_features].fillna(0)
    y = df[TARGET_COLUMN].values

    print(f"Target: {TARGET_COLUMN}")
    print(f"  Mean: ${y.mean():,.2f}")
    print(f"  Std: ${y.std():,.2f}")
    print(f"  Range: ${y.min():,.2f} - ${y.max():,.2f}")

    # ========================================================================
    # TIME-BASED SPLIT (60/20/20)
    # ========================================================================
    print("\n" + "=" * 80)
    print("TIME-BASED TRAIN/VALID/TEST SPLIT")
    print("=" * 80)

    n = len(X)
    train_idx = int(n * 0.6)
    valid_idx = int(n * 0.8)

    X_train, y_train = X.iloc[:train_idx], y[:train_idx]
    X_valid, y_valid = X.iloc[train_idx:valid_idx], y[train_idx:valid_idx]
    X_test, y_test = X.iloc[valid_idx:], y[valid_idx:]

    print(f"Split: 60% train / 20% valid / 20% test")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Valid: {len(X_valid):,} samples")
    print(f"  Test: {len(X_test):,} samples")

    # ========================================================================
    # TRAIN XGBOOST MODEL
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING XGBOOST REGRESSOR")
    print("=" * 80)

    params = XGBOOST_REGRESSION_CONFIG["params"]
    early_stopping = XGBOOST_REGRESSION_CONFIG["training"]["early_stopping_rounds"]

    print("Configuration:")
    for key, value in params.items():
        if key not in ['random_state', 'n_jobs', 'verbosity']:
            print(f"  {key}: {value}")

    # Create model with early stopping callback
    params_with_es = params.copy()
    params_with_es['early_stopping_rounds'] = early_stopping
    params_with_es['callbacks'] = [xgb.callback.EvaluationMonitor(period=100)]

    model = xgb.XGBRegressor(**params_with_es)

    print("\nTraining...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=False,
    )

    best_iter = getattr(model, 'best_iteration', model.n_estimators)
    print(f"\n✓ Training complete")
    print(f"  Best iteration: {best_iter}")

    # ========================================================================
    # EVALUATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)

    # Clamp to positive values
    y_train_pred = np.maximum(y_train_pred, 0)
    y_valid_pred = np.maximum(y_valid_pred, 0)
    y_test_pred = np.maximum(y_test_pred, 0)

    def calc_metrics(y_true, y_pred):
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100,
        }

    train_metrics = calc_metrics(y_train, y_train_pred)
    valid_metrics = calc_metrics(y_valid, y_valid_pred)
    test_metrics = calc_metrics(y_test, y_test_pred)

    print("\nPerformance Summary:")
    print(f"  {'Set':<12} {'RMSE':>12} {'MAE':>12} {'R²':>10} {'MAPE':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Train':<12} ${train_metrics['rmse']:>10,.2f} ${train_metrics['mae']:>10,.2f} {train_metrics['r2']:>10.4f} {train_metrics['mape']:>9.1f}%")
    print(f"  {'Valid':<12} ${valid_metrics['rmse']:>10,.2f} ${valid_metrics['mae']:>10,.2f} {valid_metrics['r2']:>10.4f} {valid_metrics['mape']:>9.1f}%")
    print(f"  {'Test':<12} ${test_metrics['rmse']:>10,.2f} ${test_metrics['mae']:>10,.2f} {test_metrics['r2']:>10.4f} {test_metrics['mape']:>9.1f}%")

    # Overfitting analysis
    overfit_ratio = train_metrics['rmse'] / test_metrics['rmse']
    print(f"\nOverfitting Analysis:")
    print(f"  Train/Test RMSE ratio: {1/overfit_ratio:.2f}x")
    if 1/overfit_ratio < 1.5:
        print(f"  Assessment: ✅ Good generalization")
    elif 1/overfit_ratio < 2.0:
        print(f"  Assessment: ⚠️ Slight overfitting")
    else:
        print(f"  Assessment: ❌ Overfitting detected")

    # Compare with LightGBM baseline
    print("\n" + "=" * 80)
    print("COMPARISON WITH LIGHTGBM BASELINE")
    print("=" * 80)
    print(f"  LightGBM Test RMSE: $328.75 (baseline)")
    print(f"  XGBoost Test RMSE:  ${test_metrics['rmse']:,.2f}")
    improvement = (328.75 - test_metrics['rmse']) / 328.75 * 100
    if improvement > 0:
        print(f"  Improvement: {improvement:.1f}% better ✅")
    else:
        print(f"  Difference: {-improvement:.1f}% worse")

    # ========================================================================
    # FEATURE IMPORTANCE
    # ========================================================================
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)

    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': numeric_features,
        'importance': importance
    }).sort_values('importance', ascending=False)

    importance_df['importance_pct'] = importance_df['importance'] / importance_df['importance'].sum() * 100

    print("\nTop 20 Most Important Features:")
    for i, (_, row) in enumerate(importance_df.head(20).iterrows()):
        print(f"  {i+1:2d}. {row['feature']:40s} {row['importance_pct']:6.2f}%")

    # ========================================================================
    # SAVE MODEL
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)

    model_path = MODELS_DIR / "xgboost_bidfee_model.json"
    model.save_model(str(model_path))
    print(f"✓ Model saved: {model_path}")

    # Save metadata
    metadata = {
        "model_type": "XGBoost Regressor",
        "target_variable": TARGET_COLUMN,
        "num_features": len(numeric_features),
        "features": numeric_features,
        "data_config": {
            "start_date": DATA_START_DATE,
            "total_samples": len(df),
            "train_samples": len(X_train),
            "valid_samples": len(X_valid),
            "test_samples": len(X_test),
        },
        "hyperparameters": {k: v for k, v in params.items() if k not in ['n_jobs', 'verbosity']},
        "training": {
            "best_iteration": model.best_iteration,
            "early_stopping_rounds": early_stopping,
        },
        "metrics": {
            "train": train_metrics,
            "valid": valid_metrics,
            "test": test_metrics,
        },
        "comparison": {
            "lightgbm_test_rmse": 328.75,
            "xgboost_test_rmse": test_metrics['rmse'],
            "improvement_pct": improvement,
        },
        "feature_importance": importance_df.head(20).to_dict('records'),
        "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    metadata_path = MODELS_DIR / "xgboost_bidfee_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"✓ Metadata saved: {metadata_path}")

    # Save feature importance
    importance_path = REPORTS_DIR / "xgboost_bidfee_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"✓ Feature importance saved: {importance_path}")

    # ========================================================================
    # SAVE PLOTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING PLOTS")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Actual vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_test_pred, alpha=0.5, s=10)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Bid Fee ($)')
    ax1.set_ylabel('Predicted Bid Fee ($)')
    ax1.set_title(f'XGBoost: Actual vs Predicted (Test Set)\nRMSE: ${test_metrics["rmse"]:,.2f}')
    ax1.grid(True, alpha=0.3)

    # Error distribution
    ax2 = axes[0, 1]
    errors = y_test_pred - y_test
    ax2.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', lw=2)
    ax2.set_xlabel('Prediction Error ($)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Error Distribution\nMean: ${errors.mean():,.2f}, Std: ${errors.std():,.2f}')
    ax2.grid(True, alpha=0.3)

    # Feature importance
    ax3 = axes[1, 0]
    top_n = 15
    top_features = importance_df.head(top_n)
    ax3.barh(range(len(top_features)), top_features['importance_pct'])
    ax3.set_yticks(range(len(top_features)))
    ax3.set_yticklabels(top_features['feature'])
    ax3.set_xlabel('Importance (%)')
    ax3.set_title(f'Top {top_n} Feature Importance')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')

    # Model comparison
    ax4 = axes[1, 1]
    models = ['LightGBM\n(Baseline)', 'XGBoost']
    rmse_values = [328.75, test_metrics['rmse']]
    colors = ['#3498db', '#e74c3c']
    bars = ax4.bar(models, rmse_values, color=colors, edgecolor='black')
    ax4.set_ylabel('Test RMSE ($)')
    ax4.set_title('Model Comparison: Test RMSE')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rmse_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'${val:,.0f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plot_path = FIGURES_DIR / 'xgboost_bidfee_evaluation.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Evaluation plots saved: {plot_path}")
    plt.close()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("XGBOOST BID FEE MODEL COMPLETE")
    print("=" * 80)

    print(f"\n📊 Model Performance:")
    print(f"   Test RMSE: ${test_metrics['rmse']:,.2f}")
    print(f"   Test MAE:  ${test_metrics['mae']:,.2f}")
    print(f"   Test R²:   {test_metrics['r2']:.4f}")

    if improvement > 0:
        assessment = f"✅ XGBoost is {improvement:.1f}% better than LightGBM"
    elif improvement > -5:
        assessment = f"⚠️ XGBoost is comparable to LightGBM ({-improvement:.1f}% worse)"
    else:
        assessment = f"❌ LightGBM performs better ({-improvement:.1f}% improvement)"

    print(f"\n📋 Assessment: {assessment}")

    print(f"\n📁 Files saved:")
    print(f"   - {model_path}")
    print(f"   - {metadata_path}")
    print(f"   - {importance_path}")
    print(f"   - {plot_path}")

    return model, test_metrics


if __name__ == "__main__":
    model, metrics = main()
