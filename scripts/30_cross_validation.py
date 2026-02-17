"""
Time-Series Cross-Validation
==============================
5-fold expanding-window CV for both bid fee and win probability models.
Gives confidence intervals on metrics instead of single-point estimates.

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
from sklearn.metrics import roc_auc_score, brier_score_loss

from config.model_config import MODELS_DIR, REPORTS_DIR, MIN_TRAINING_FEE

warnings.filterwarnings("ignore")


def cv_bid_fee(n_folds=5):
    """Time-series CV for bid fee model."""
    print("=" * 80)
    print(f"TIME-SERIES CROSS-VALIDATION: BID FEE ({n_folds} folds)")
    print("=" * 80)

    # Load metadata for features and params
    meta_path = MODELS_DIR / "lightgbm_bidfee_v2_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    feature_names = meta["features"]

    df = pd.read_csv("data/features/JobsData_features_v2.csv", low_memory=False)
    df["StartDate"] = pd.to_datetime(df["StartDate"])
    df = df[df["NetFee"] >= MIN_TRAINING_FEE].copy()
    df = df.sort_values("StartDate").reset_index(drop=True)

    # Fill missing features
    for f in feature_names:
        if f not in df.columns:
            df[f] = 0

    X = df[feature_names].values
    y = df["NetFee"].values
    n = len(df)

    # Expanding window: each fold uses more training data
    # Fold 1: train on 40%, test on next 12%
    # Fold 2: train on 52%, test on next 12%
    # ...
    # Fold 5: train on 88%, test on last 12%
    fold_size = int(n * 0.12)
    min_train = int(n * 0.40)

    fold_results = []
    print(f"\n  Total rows: {n:,}, fold size: ~{fold_size:,}")

    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)

        if test_end <= test_start:
            break

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]

        # Train
        y_train_log = np.log1p(y_train)
        train_data = lgb.Dataset(X_train, label=y_train_log, feature_name=feature_names)

        params = meta["hyperparameters"].copy()
        params["verbose"] = -1

        model = lgb.train(
            params,
            train_data,
            num_boost_round=meta.get("best_iteration", 3000),
        )

        # Predict
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_pred = np.maximum(y_pred, 500)

        # Metrics
        errors = np.abs(y_pred - y_test)
        pct_errors = errors / np.maximum(y_test, 1) * 100

        fold_result = {
            "fold": fold + 1,
            "train_size": train_end,
            "test_size": test_end - test_start,
            "rmse": float(np.sqrt(np.mean((y_pred - y_test) ** 2))),
            "mae": float(np.mean(errors)),
            "mape": float(np.mean(pct_errors)),
            "median_ape": float(np.median(pct_errors)),
            "within_20pct": float(np.mean(pct_errors <= 20) * 100),
        }
        fold_results.append(fold_result)

        date_range = f"{df.iloc[test_start]['StartDate'].date()} to {df.iloc[test_end-1]['StartDate'].date()}"
        print(f"\n  Fold {fold+1}: train={train_end:,}, test={test_end-test_start:,} ({date_range})")
        print(f"    MAPE: {fold_result['mape']:.1f}%, Within 20%: {fold_result['within_20pct']:.1f}%, RMSE: ${fold_result['rmse']:,.0f}")

    # Summary statistics
    mapes = [r["mape"] for r in fold_results]
    within20s = [r["within_20pct"] for r in fold_results]
    rmses = [r["rmse"] for r in fold_results]

    print(f"\n  {'='*50}")
    print(f"  SUMMARY (across {len(fold_results)} folds):")
    print(f"    MAPE:       {np.mean(mapes):.1f}% +/- {np.std(mapes):.1f}% (range: {np.min(mapes):.1f}% - {np.max(mapes):.1f}%)")
    print(f"    Within 20%: {np.mean(within20s):.1f}% +/- {np.std(within20s):.1f}%")
    print(f"    RMSE:       ${np.mean(rmses):,.0f} +/- ${np.std(rmses):,.0f}")

    return fold_results


def cv_win_probability(n_folds=5):
    """Time-series CV for win probability model."""
    print("\n" + "=" * 80)
    print(f"TIME-SERIES CROSS-VALIDATION: WIN PROBABILITY ({n_folds} folds)")
    print("=" * 80)

    meta_path = MODELS_DIR / "lightgbm_win_probability_v2_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    feature_names = meta["features"]

    df = pd.read_csv("data/features/BidData_features_v2.csv", low_memory=False)
    date_col = "StartDate" if "StartDate" in df.columns else "BidDate"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    for f in feature_names:
        if f not in df.columns:
            df[f] = 0

    X = df[feature_names].values
    y = df["Won"].values.astype(int)
    n = len(df)

    fold_size = int(n * 0.12)
    min_train = int(n * 0.40)

    fold_results = []
    print(f"\n  Total rows: {n:,}, fold size: ~{fold_size:,}")

    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)

        if test_end <= test_start:
            break

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]

        # Train
        params = meta["hyperparameters"].copy()
        params["verbose"] = -1
        scale_pos_weight = (1 - y_train.mean()) / max(y_train.mean(), 0.01)
        params["scale_pos_weight"] = scale_pos_weight

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=meta.get("best_iteration", 2000),
        )

        # Predict
        y_prob = model.predict(X_test)
        y_prob = np.clip(y_prob, 0.05, 0.95)
        y_pred = (y_prob >= 0.5).astype(int)

        if y_test.sum() == 0 or y_test.sum() == len(y_test):
            continue

        fold_result = {
            "fold": fold + 1,
            "train_size": train_end,
            "test_size": test_end - test_start,
            "auc_roc": float(roc_auc_score(y_test, y_prob)),
            "accuracy": float(np.mean(y_pred == y_test)),
            "brier": float(brier_score_loss(y_test, y_prob)),
        }
        fold_results.append(fold_result)

        print(f"\n  Fold {fold+1}: train={train_end:,}, test={test_end-test_start:,}")
        print(f"    AUC: {fold_result['auc_roc']:.4f}, Accuracy: {fold_result['accuracy']*100:.1f}%, Brier: {fold_result['brier']:.4f}")

    # Summary
    aucs = [r["auc_roc"] for r in fold_results]
    accs = [r["accuracy"] for r in fold_results]
    briers = [r["brier"] for r in fold_results]

    print(f"\n  {'='*50}")
    print(f"  SUMMARY (across {len(fold_results)} folds):")
    print(f"    AUC-ROC:  {np.mean(aucs):.4f} +/- {np.std(aucs):.4f} (range: {np.min(aucs):.4f} - {np.max(aucs):.4f})")
    print(f"    Accuracy: {np.mean(accs)*100:.1f}% +/- {np.std(accs)*100:.1f}%")
    print(f"    Brier:    {np.mean(briers):.4f} +/- {np.std(briers):.4f}")

    return fold_results


def main():
    print(f"Time-Series Cross-Validation — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    fee_results = cv_bid_fee()
    wp_results = cv_win_probability()

    # Save
    report = {
        "run_date": datetime.now().isoformat(),
        "bid_fee_cv": fee_results,
        "win_probability_cv": wp_results,
        "summary": {
            "bid_fee_mape_mean": float(np.mean([r["mape"] for r in fee_results])),
            "bid_fee_mape_std": float(np.std([r["mape"] for r in fee_results])),
            "win_prob_auc_mean": float(np.mean([r["auc_roc"] for r in wp_results])),
            "win_prob_auc_std": float(np.std([r["auc_roc"] for r in wp_results])),
        },
    }

    output_path = REPORTS_DIR / "cross_validation_results.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
