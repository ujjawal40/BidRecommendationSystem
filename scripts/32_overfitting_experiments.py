"""
Bid Fee Overfitting Reduction Experiments
==========================================
Tests different hyperparameter combinations to reduce overfitting ratio
from 1.91x to <= 1.5x while maintaining MAPE < 6%.

Does NOT modify production model. Run experiments, pick best, then retrain.

Author: Ujjawal Dwivedi
Date: 2026-02-17
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import json
import warnings
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

from config.model_config import DATA_DIR, MIN_TRAINING_FEE, REPORTS_DIR

warnings.filterwarnings("ignore")

# ============================================================================
# CURRENT BASELINE (for comparison)
# ============================================================================
BASELINE = {
    "name": "current_v2",
    "params": {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 18,
        "learning_rate": 0.02,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "max_depth": 6,
        "min_child_samples": 50,
        "min_child_weight": 10,
        "reg_alpha": 5.0,
        "reg_lambda": 5.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
    "num_boost_round": 3000,
    "early_stopping_rounds": 150,
}

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================
EXPERIMENTS = [
    # --- Leaf reduction ---
    {
        "name": "fewer_leaves_14",
        "desc": "Reduce num_leaves 18 → 14",
        "changes": {"num_leaves": 14},
    },
    {
        "name": "fewer_leaves_10",
        "desc": "Reduce num_leaves 18 → 10",
        "changes": {"num_leaves": 10},
    },
    # --- Min child samples ---
    {
        "name": "min_child_80",
        "desc": "Increase min_child_samples 50 → 80",
        "changes": {"min_child_samples": 80},
    },
    {
        "name": "min_child_120",
        "desc": "Increase min_child_samples 50 → 120",
        "changes": {"min_child_samples": 120},
    },
    # --- Feature/row subsampling ---
    {
        "name": "strong_subsample",
        "desc": "Feature fraction 0.7 → 0.5, bagging 0.7 → 0.6",
        "changes": {"feature_fraction": 0.5, "bagging_fraction": 0.6},
    },
    # --- Stronger regularization ---
    {
        "name": "strong_reg",
        "desc": "reg_alpha/lambda 5.0 → 10.0",
        "changes": {"reg_alpha": 10.0, "reg_lambda": 10.0},
    },
    {
        "name": "very_strong_reg",
        "desc": "reg_alpha/lambda 5.0 → 20.0",
        "changes": {"reg_alpha": 20.0, "reg_lambda": 20.0},
    },
    # --- Min split gain ---
    {
        "name": "min_gain_01",
        "desc": "Add min_split_gain = 0.1",
        "changes": {"min_split_gain": 0.1},
    },
    {
        "name": "min_gain_05",
        "desc": "Add min_split_gain = 0.5",
        "changes": {"min_split_gain": 0.5},
    },
    # --- Depth reduction ---
    {
        "name": "depth_5",
        "desc": "Reduce max_depth 6 → 5",
        "changes": {"max_depth": 5},
    },
    {
        "name": "depth_4",
        "desc": "Reduce max_depth 6 → 4",
        "changes": {"max_depth": 4},
    },
    # --- Path smoothing ---
    {
        "name": "path_smooth_10",
        "desc": "Add path_smooth = 10",
        "changes": {"path_smooth": 10},
    },
    {
        "name": "path_smooth_50",
        "desc": "Add path_smooth = 50",
        "changes": {"path_smooth": 50},
    },
    # --- Combined approaches ---
    {
        "name": "combo_conservative",
        "desc": "leaves=14, min_child=80, reg=8.0",
        "changes": {"num_leaves": 14, "min_child_samples": 80, "reg_alpha": 8.0, "reg_lambda": 8.0},
    },
    {
        "name": "combo_aggressive",
        "desc": "leaves=12, min_child=100, reg=10, feat_frac=0.5, depth=5",
        "changes": {
            "num_leaves": 12, "min_child_samples": 100,
            "reg_alpha": 10.0, "reg_lambda": 10.0,
            "feature_fraction": 0.5, "max_depth": 5,
        },
    },
    {
        "name": "combo_balanced",
        "desc": "leaves=14, min_child=80, reg=10, min_gain=0.1, path_smooth=10",
        "changes": {
            "num_leaves": 14, "min_child_samples": 80,
            "reg_alpha": 10.0, "reg_lambda": 10.0,
            "min_split_gain": 0.1, "path_smooth": 10,
        },
    },
    {
        "name": "combo_max_regularize",
        "desc": "leaves=10, min_child=120, reg=20, feat=0.5, depth=5, gain=0.5, smooth=50",
        "changes": {
            "num_leaves": 10, "min_child_samples": 120,
            "reg_alpha": 20.0, "reg_lambda": 20.0,
            "feature_fraction": 0.5, "max_depth": 5,
            "min_split_gain": 0.5, "path_smooth": 50,
        },
    },
]


def load_data():
    """Load and prepare data."""
    from config.model_config import MODELS_DIR

    meta_path = MODELS_DIR / "lightgbm_bidfee_v2_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    feature_names = meta["features"]

    df = pd.read_csv(DATA_DIR / "features" / "JobsData_features_v2.csv", low_memory=False)
    df["StartDate"] = pd.to_datetime(df["StartDate"])
    df = df[df["NetFee"] >= MIN_TRAINING_FEE].copy()
    df = df.sort_values("StartDate").reset_index(drop=True)

    for f in feature_names:
        if f not in df.columns:
            df[f] = 0

    X = df[feature_names].values
    y = df["NetFee"].values
    n = len(X)

    # 60/20/20 split
    train_idx = int(n * 0.6)
    valid_idx = int(n * 0.8)

    return {
        "X_train": X[:train_idx], "y_train": y[:train_idx],
        "X_valid": X[train_idx:valid_idx], "y_valid": y[train_idx:valid_idx],
        "X_test": X[valid_idx:], "y_test": y[valid_idx:],
        "feature_names": feature_names,
        "n_total": n,
    }


def run_experiment(data, config):
    """Run a single experiment and return metrics."""
    params = BASELINE["params"].copy()
    params.update(config.get("changes", {}))

    y_train_log = np.log1p(data["y_train"])
    y_valid_log = np.log1p(data["y_valid"])

    train_data = lgb.Dataset(data["X_train"], label=y_train_log, feature_name=data["feature_names"])
    valid_data = lgb.Dataset(data["X_valid"], label=y_valid_log, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=BASELINE["num_boost_round"],
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(BASELINE["early_stopping_rounds"]),
            lgb.log_evaluation(0),  # Silent
        ],
    )

    results = {}
    for name, X, y in [("train", data["X_train"], data["y_train"]),
                        ("valid", data["X_valid"], data["y_valid"]),
                        ("test", data["X_test"], data["y_test"])]:
        pred_log = model.predict(X)
        pred = np.expm1(pred_log)
        pred = np.maximum(pred, 500)

        errors = np.abs(pred - y)
        pct_errors = errors / np.maximum(y, 1) * 100

        results[name] = {
            "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
            "mape": float(np.mean(pct_errors)),
            "median_ape": float(np.median(pct_errors)),
            "within_20pct": float(np.mean(pct_errors <= 20) * 100),
        }

    overfit = results["test"]["rmse"] / results["train"]["rmse"]
    results["overfitting_ratio"] = float(overfit)
    results["best_iteration"] = int(model.best_iteration)

    return results


def main():
    print("=" * 80)
    print("BID FEE OVERFITTING REDUCTION EXPERIMENTS")
    print(f"Target: overfit ratio <= 1.50x with MAPE < 6%")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    data = load_data()
    print(f"\nData: {data['n_total']:,} rows, {len(data['feature_names'])} features")
    print(f"Train: {len(data['X_train']):,}, Valid: {len(data['X_valid']):,}, Test: {len(data['X_test']):,}")

    # Run baseline first
    print(f"\n{'='*80}")
    print("Running baseline...")
    baseline_results = run_experiment(data, {"name": "baseline", "changes": {}})
    print(f"  Baseline: overfit={baseline_results['overfitting_ratio']:.2f}x, "
          f"test_MAPE={baseline_results['test']['mape']:.1f}%, "
          f"test_RMSE=${baseline_results['test']['rmse']:,.0f}")

    # Run experiments
    all_results = [{
        "name": "baseline (current v2)",
        "desc": "Current production config",
        "overfit": baseline_results["overfitting_ratio"],
        "test_mape": baseline_results["test"]["mape"],
        "test_rmse": baseline_results["test"]["rmse"],
        "test_within20": baseline_results["test"]["within_20pct"],
        "train_mape": baseline_results["train"]["mape"],
        "best_iter": baseline_results["best_iteration"],
        "results": baseline_results,
    }]

    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] {exp['name']}: {exp['desc']}")
        results = run_experiment(data, exp)

        overfit = results["overfitting_ratio"]
        mape = results["test"]["mape"]
        rmse = results["test"]["rmse"]

        # Color coding in text
        overfit_status = "OK" if overfit <= 1.5 else "HIGH" if overfit <= 1.7 else "BAD"
        mape_status = "OK" if mape < 6 else "HIGH" if mape < 10 else "BAD"

        print(f"  overfit={overfit:.2f}x [{overfit_status}], "
              f"test_MAPE={mape:.1f}% [{mape_status}], "
              f"test_RMSE=${rmse:,.0f}, "
              f"iters={results['best_iteration']}")

        all_results.append({
            "name": exp["name"],
            "desc": exp["desc"],
            "changes": exp["changes"],
            "overfit": overfit,
            "test_mape": mape,
            "test_rmse": rmse,
            "test_within20": results["test"]["within_20pct"],
            "train_mape": results["train"]["mape"],
            "best_iter": results["best_iteration"],
            "results": results,
        })

    # ========================================================================
    # RESULTS TABLE
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n  {'Config':<30} {'Overfit':>8} {'MAPE':>7} {'RMSE':>8} {'W/in 20%':>9} {'Iters':>6} {'Verdict':>10}")
    print("  " + "-" * 80)

    # Sort by overfit ratio
    for r in sorted(all_results, key=lambda x: x["overfit"]):
        meets_overfit = r["overfit"] <= 1.5
        meets_mape = r["test_mape"] < 6
        if meets_overfit and meets_mape:
            verdict = "** WINNER"
        elif meets_overfit:
            verdict = "MAPE HIGH"
        elif meets_mape:
            verdict = "OVERFIT"
        else:
            verdict = "BOTH BAD"

        marker = ">>>" if meets_overfit and meets_mape else "   "
        print(f"{marker}{r['name']:<30} {r['overfit']:>7.2f}x {r['test_mape']:>6.1f}% "
              f"${r['test_rmse']:>7,.0f} {r['test_within20']:>8.1f}% {r['best_iter']:>6} {verdict:>10}")

    # Best config that meets both criteria
    winners = [r for r in all_results if r["overfit"] <= 1.5 and r["test_mape"] < 6]
    if winners:
        best = min(winners, key=lambda x: x["test_mape"])
        print(f"\n  RECOMMENDED: {best['name']}")
        print(f"    Overfit: {best['overfit']:.2f}x (target <= 1.50x)")
        print(f"    MAPE:    {best['test_mape']:.1f}% (target < 6%)")
        print(f"    RMSE:    ${best['test_rmse']:,.0f}")
        print(f"    Within 20%: {best['test_within20']:.1f}%")
        if "changes" in best:
            print(f"    Changes: {best['changes']}")
    else:
        # Find best tradeoff
        candidates = [r for r in all_results if r["test_mape"] < 6]
        if candidates:
            best = min(candidates, key=lambda x: x["overfit"])
            print(f"\n  NO WINNER (none hit both targets). Closest:")
            print(f"    {best['name']}: overfit={best['overfit']:.2f}x, MAPE={best['test_mape']:.1f}%")

    # Save results
    output = {
        "run_date": datetime.now().isoformat(),
        "target_overfit": 1.5,
        "target_mape": 6.0,
        "baseline": all_results[0],
        "experiments": all_results[1:],
        "recommended": best["name"] if winners else None,
    }
    output_path = REPORTS_DIR / "overfitting_experiments.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {output_path}")


if __name__ == "__main__":
    main()
