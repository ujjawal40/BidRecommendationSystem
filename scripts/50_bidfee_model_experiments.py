"""
Bid Fee Model Improvement Experiments
======================================
Tests multiple LightGBM configurations on the same train/valid/test split.
Does NOT modify production model files.

Usage: python scripts/50_bidfee_model_experiments.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

from config.model_config import (
    FEATURES_DATA, TARGET_COLUMN, DATE_COLUMN, EXCLUDE_COLUMNS,
    LIGHTGBM_CONFIG, MODELS_DIR, REPORTS_DIR, RANDOM_SEED,
    DATA_START_DATE, USE_RECENT_DATA_ONLY, JOBDATA_FEATURES_TO_EXCLUDE,
)

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

BASE_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 18,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": 8,
    "min_child_samples": 30,
    "min_child_weight": 5,
    "reg_alpha": 2.0,
    "reg_lambda": 2.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

EXPERIMENTS = {
    "A1_baseline": {
        "description": "Current production config (reproduction)",
        "target_transform": None,
        "params": {**BASE_PARAMS},
        "num_boost_round": 500,
        "early_stopping_rounds": 50,
    },
    "A2_huber_500": {
        "description": "Huber loss (alpha=500) — downweight large outliers",
        "target_transform": None,
        "params": {**BASE_PARAMS, "objective": "huber", "alpha": 500},
        "num_boost_round": 500,
        "early_stopping_rounds": 50,
    },
    "A3_huber_250": {
        "description": "Huber loss (alpha=250) — more aggressive outlier handling",
        "target_transform": None,
        "params": {**BASE_PARAMS, "objective": "huber", "alpha": 250},
        "num_boost_round": 500,
        "early_stopping_rounds": 50,
    },
    "A4_log_transform": {
        "description": "Log-transform target — proportional errors",
        "target_transform": "log1p",
        "params": {**BASE_PARAMS},
        "num_boost_round": 500,
        "early_stopping_rounds": 50,
    },
    "A5_log_huber": {
        "description": "Log-transform + Huber on log scale",
        "target_transform": "log1p",
        "params": {**BASE_PARAMS, "objective": "huber", "alpha": 0.5},
        "num_boost_round": 500,
        "early_stopping_rounds": 50,
    },
    "A6_lr002_1500": {
        "description": "Lower LR (0.02) + 1500 rounds",
        "target_transform": None,
        "params": {**BASE_PARAMS, "learning_rate": 0.02},
        "num_boost_round": 1500,
        "early_stopping_rounds": 100,
    },
    "A7_lr001_2500": {
        "description": "Lower LR (0.01) + 2500 rounds",
        "target_transform": None,
        "params": {**BASE_PARAMS, "learning_rate": 0.01},
        "num_boost_round": 2500,
        "early_stopping_rounds": 150,
    },
    "A8_1500_strong_reg": {
        "description": "1500 rounds + stronger reg (alpha=4, lambda=4, min_child=50)",
        "target_transform": None,
        "params": {**BASE_PARAMS, "reg_alpha": 4.0, "reg_lambda": 4.0, "min_child_samples": 50},
        "num_boost_round": 1500,
        "early_stopping_rounds": 100,
    },
    "A9_log_lr002_1500": {
        "description": "Log-transform + LR 0.02 + 1500 rounds",
        "target_transform": "log1p",
        "params": {**BASE_PARAMS, "learning_rate": 0.02},
        "num_boost_round": 1500,
        "early_stopping_rounds": 100,
    },
    # ==== PHASE B: Combinations based on Phase A insights ====
    "B1_log_lr001_3000": {
        "description": "Log-transform + LR 0.01 + 3000 rounds (push log further)",
        "target_transform": "log1p",
        "params": {**BASE_PARAMS, "learning_rate": 0.01},
        "num_boost_round": 3000,
        "early_stopping_rounds": 200,
    },
    "B2_log_lr005_5000": {
        "description": "Log-transform + LR 0.005 + 5000 rounds (very slow learning)",
        "target_transform": "log1p",
        "params": {**BASE_PARAMS, "learning_rate": 0.005},
        "num_boost_round": 5000,
        "early_stopping_rounds": 300,
    },
    "B3_log_huber_lr001_3000": {
        "description": "Log + Huber 0.5 + LR 0.01 + 3000 rounds",
        "target_transform": "log1p",
        "params": {**BASE_PARAMS, "objective": "huber", "alpha": 0.5, "learning_rate": 0.01},
        "num_boost_round": 3000,
        "early_stopping_rounds": 200,
    },
    "B4_log_more_leaves": {
        "description": "Log-transform + 31 leaves + LR 0.01 + 3000 rounds",
        "target_transform": "log1p",
        "params": {**BASE_PARAMS, "learning_rate": 0.01, "num_leaves": 31, "max_depth": 10},
        "num_boost_round": 3000,
        "early_stopping_rounds": 200,
    },
    "B5_log_dart": {
        "description": "Log-transform + DART boosting + LR 0.02 + 1500 rounds",
        "target_transform": "log1p",
        "params": {**BASE_PARAMS, "boosting_type": "dart", "learning_rate": 0.02, "drop_rate": 0.1},
        "num_boost_round": 1500,
        "early_stopping_rounds": 100,
    },
    "B6_raw_lr001_5000": {
        "description": "Raw scale + LR 0.01 + 5000 rounds (push baseline further)",
        "target_transform": None,
        "params": {**BASE_PARAMS, "learning_rate": 0.01},
        "num_boost_round": 5000,
        "early_stopping_rounds": 300,
    },
}


# ============================================================================
# DATA LOADING (identical to production pipeline)
# ============================================================================

def load_and_prepare_data():
    """Load data, filter, prepare features, split — exactly as 04_model_lightgbm.py."""
    print("=" * 80)
    print("LOADING AND PREPARING DATA")
    print("=" * 80)

    df = pd.read_csv(FEATURES_DATA)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    if USE_RECENT_DATA_ONLY:
        start_date = pd.Timestamp(DATA_START_DATE)
        df = df[df[DATE_COLUMN] >= start_date].copy()

    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    print(f"Records after filtering: {len(df):,}")

    # Prepare features (identical to production)
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    feature_cols = [col for col in feature_cols if col not in JOBDATA_FEATURES_TO_EXCLUDE]
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    X = df[numeric_features].fillna(0)
    y = df[TARGET_COLUMN].copy()

    # Select top 68 features (same as production)
    selected_features_path = REPORTS_DIR / 'selected_features_top68.csv'
    if selected_features_path.exists():
        selected = pd.read_csv(selected_features_path, header=None)[0].tolist()
        available = [f for f in selected if f in X.columns]
        X = X[available]
        print(f"Features selected: {len(available)} (top 68)")
    else:
        print(f"Using all {len(X.columns)} features (no selection file)")

    feature_names = X.columns.tolist()

    # Time-based split 60/20/20
    n = len(df)
    train_idx = int(n * 0.6)
    valid_idx = int(n * 0.8)

    X_train, X_valid, X_test = X.iloc[:train_idx], X.iloc[train_idx:valid_idx], X.iloc[valid_idx:]
    y_train, y_valid, y_test = y.iloc[:train_idx], y.iloc[train_idx:valid_idx], y.iloc[valid_idx:]

    print(f"Train: {len(X_train):,} | Valid: {len(X_valid):,} | Test: {len(X_test):,}")
    print(f"Target mean: ${y.mean():,.0f} | median: ${y.median():,.0f}")
    print()

    return X_train, X_valid, X_test, y_train, y_valid, y_test, feature_names


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(name, config, X_train, X_valid, X_test, y_train, y_valid, y_test, feature_names):
    """Run a single experiment and return metrics on original dollar scale."""
    transform = config["target_transform"]

    # Apply target transform
    if transform == "log1p":
        y_tr = np.log1p(y_train)
        y_va = np.log1p(y_valid)
    else:
        y_tr = y_train
        y_va = y_valid

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_tr, feature_name=feature_names)
    valid_data = lgb.Dataset(X_valid, label=y_va, reference=train_data)

    # Train
    model = lgb.train(
        config["params"],
        train_data,
        num_boost_round=config["num_boost_round"],
        valid_sets=[valid_data],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=config["early_stopping_rounds"]),
            lgb.log_evaluation(period=0),  # Silent
        ],
    )

    # Predict and inverse transform
    train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    valid_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)

    if transform == "log1p":
        train_pred = np.expm1(train_pred)
        valid_pred = np.expm1(valid_pred)
        test_pred = np.expm1(test_pred)

    # Clamp to 0
    train_pred = np.maximum(0, train_pred)
    valid_pred = np.maximum(0, valid_pred)
    test_pred = np.maximum(0, test_pred)

    # Metrics on original scale
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_median_ae = median_absolute_error(y_test, test_pred)

    non_zero = y_test != 0
    test_mape = np.mean(np.abs((y_test[non_zero] - test_pred[non_zero]) / y_test[non_zero])) * 100

    overfitting = test_rmse / train_rmse if train_rmse > 0 else 0

    return {
        "name": name,
        "description": config["description"],
        "train_rmse": round(train_rmse, 2),
        "valid_rmse": round(valid_rmse, 2),
        "test_rmse": round(test_rmse, 2),
        "test_mae": round(test_mae, 2),
        "test_r2": round(test_r2, 4),
        "test_mape": round(test_mape, 2),
        "test_median_ae": round(test_median_ae, 2),
        "overfitting": round(overfitting, 2),
        "best_iteration": model.best_iteration,
        "hit_cap": model.best_iteration == config["num_boost_round"],
        "target_transform": transform or "none",
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("BID FEE MODEL IMPROVEMENT EXPERIMENTS")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiments to run: {len(EXPERIMENTS)}\n")

    X_tr, X_va, X_te, y_tr, y_va, y_te, feat_names = load_and_prepare_data()

    results = []
    for name, config in EXPERIMENTS.items():
        print(f"Running {name}: {config['description']}...", end=" ", flush=True)
        result = run_experiment(name, config, X_tr, X_va, X_te, y_tr, y_va, y_te, feat_names)
        results.append(result)
        flag = "CAP" if result["hit_cap"] else f"iter {result['best_iteration']}"
        print(f"RMSE=${result['test_rmse']:.0f} | Overfit={result['overfitting']:.2f}x | {flag}")

    # ========================================================================
    # COMPARISON TABLE
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    # Sort by test RMSE
    results.sort(key=lambda r: r["test_rmse"])

    print(f"\n{'#':<4} {'Name':<22} {'RMSE':>8} {'MAE':>8} {'MAPE':>7} {'MedAE':>8} {'R²':>7} {'Overfit':>8} {'Iter':>6} {'Transform':>10}")
    print("-" * 105)

    for i, r in enumerate(results):
        overfit_flag = " *" if r["overfitting"] > 2.0 else ""
        cap_flag = "!" if r["hit_cap"] else ""
        print(
            f"{i+1:<4} {r['name']:<22} "
            f"${r['test_rmse']:>6,.0f} ${r['test_mae']:>6,.0f} "
            f"{r['test_mape']:>5.1f}% ${r['test_median_ae']:>6,.0f} "
            f"{r['test_r2']:>6.4f} "
            f"{r['overfitting']:>5.2f}x{overfit_flag:>2} "
            f"{r['best_iteration']:>4}{cap_flag:<1} "
            f"{r['target_transform']:>10}"
        )

    # Best within overfitting constraint
    valid_results = [r for r in results if r["overfitting"] <= 2.0]
    if valid_results:
        best = valid_results[0]  # Already sorted by RMSE
        print(f"\n{'='*80}")
        print(f"WINNER (lowest RMSE with overfitting <= 2.0x):")
        print(f"  {best['name']}: RMSE=${best['test_rmse']:.0f}, "
              f"MAPE={best['test_mape']:.1f}%, Overfit={best['overfitting']:.2f}x")
        print(f"  {best['description']}")
    else:
        print("\nNo experiments met the overfitting <= 2.0x constraint!")

    # Also show best MAPE
    best_mape = min(results, key=lambda r: r["test_mape"])
    print(f"\nBest MAPE: {best_mape['name']} at {best_mape['test_mape']:.1f}% "
          f"(RMSE=${best_mape['test_rmse']:.0f}, Overfit={best_mape['overfitting']:.2f}x)")

    # Save results
    results_path = REPORTS_DIR / "bidfee_experiment_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    return results


if __name__ == "__main__":
    results = main()
