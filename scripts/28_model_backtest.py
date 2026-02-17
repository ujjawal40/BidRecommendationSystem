"""
Model Backtest — Real-World Validation
========================================
Tests v2 models against held-out data to measure actual prediction quality.

Three backtests:
  1. Bid Fee: predicted vs actual fee on test set (most recent 20%)
  2. Win Probability: predicted vs actual outcome on enriched BidData test set
  3. Calibration: when model says X% win prob, does it actually win X%?
  4. Segment-level breakdown: which segments/states does the model fail on?

Author: Ujjawal Dwivedi
Date: 2026-02-16
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from config.model_config import MODELS_DIR, REPORTS_DIR, MIN_TRAINING_FEE

warnings.filterwarnings("ignore")


def load_model_and_features(model_path, meta_path):
    """Load a LightGBM model and its feature list."""
    model = lgb.Booster(model_file=str(model_path))
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return model, meta["features"], meta


def backtest_bid_fee():
    """Backtest bid fee model on the chronological test set."""
    print("=" * 80)
    print("BACKTEST 1: BID FEE PREDICTION")
    print("=" * 80)

    model, features, meta = load_model_and_features(
        MODELS_DIR / "lightgbm_bidfee_v2_model.txt",
        MODELS_DIR / "lightgbm_bidfee_v2_metadata.json",
    )

    df = pd.read_csv("data/features/JobsData_features_v2.csv", low_memory=False)
    df["StartDate"] = pd.to_datetime(df["StartDate"])
    df = df.sort_values("StartDate").reset_index(drop=True)

    # Filter anomaly fees (same as training)
    before = len(df)
    df = df[df["NetFee"] >= MIN_TRAINING_FEE].copy()
    df = df.sort_values("StartDate").reset_index(drop=True)
    print(f"  Filtered {before - len(df):,} rows with NetFee < ${MIN_TRAINING_FEE}")

    # Time-based split: last 20% is test
    n = len(df)
    test_start = int(n * 0.80)
    test = df.iloc[test_start:].copy()
    print(f"  Test set: {len(test):,} rows ({test['StartDate'].min().date()} to {test['StartDate'].max().date()})")

    # Build feature matrix
    available = [f for f in features if f in test.columns]
    missing = [f for f in features if f not in test.columns]
    if missing:
        print(f"  WARNING: {len(missing)} features missing: {missing[:5]}...")
        for f in missing:
            test[f] = 0

    X_test = test[features].values
    y_actual = test["NetFee"].values

    # Predict (log scale → expm1)
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(y_pred, 500)  # floor

    # Metrics
    errors = y_pred - y_actual
    abs_errors = np.abs(errors)
    pct_errors = abs_errors / np.maximum(y_actual, 1) * 100

    results = {
        "n_test": len(test),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(abs_errors)),
        "median_ae": float(np.median(abs_errors)),
        "mape": float(np.mean(pct_errors)),
        "median_ape": float(np.median(pct_errors)),
        "within_10pct": float(np.mean(pct_errors <= 10) * 100),
        "within_20pct": float(np.mean(pct_errors <= 20) * 100),
        "within_30pct": float(np.mean(pct_errors <= 30) * 100),
    }

    print(f"\n  Overall Results:")
    print(f"    RMSE:          ${results['rmse']:,.0f}")
    print(f"    MAE:           ${results['mae']:,.0f}")
    print(f"    Median AE:     ${results['median_ae']:,.0f}")
    print(f"    MAPE:          {results['mape']:.1f}%")
    print(f"    Median APE:    {results['median_ape']:.1f}%")
    print(f"    Within 10%:    {results['within_10pct']:.1f}% of predictions")
    print(f"    Within 20%:    {results['within_20pct']:.1f}% of predictions")
    print(f"    Within 30%:    {results['within_30pct']:.1f}% of predictions")

    # Segment breakdown
    test["predicted"] = y_pred
    test["abs_pct_error"] = pct_errors

    print(f"\n  By Business Segment:")
    print(f"    {'Segment':<25} {'Count':>6} {'MAPE':>8} {'Med APE':>8} {'Within 20%':>10}")
    print("    " + "-" * 59)
    for seg in sorted(test["BusinessSegment"].unique()):
        mask = test["BusinessSegment"] == seg
        seg_errors = test.loc[mask, "abs_pct_error"]
        within20 = (seg_errors <= 20).mean() * 100
        print(f"    {seg:<25} {mask.sum():>6,} {seg_errors.mean():>7.1f}% {seg_errors.median():>7.1f}% {within20:>9.1f}%")

    # Fee bucket breakdown
    print(f"\n  By Fee Range:")
    print(f"    {'Fee Range':<20} {'Count':>6} {'MAPE':>8} {'Med APE':>8} {'Within 20%':>10}")
    print("    " + "-" * 54)
    bins = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 5000), (5000, 10000), (10000, 999999)]
    labels = ["< $1K", "$1K-$2K", "$2K-$3K", "$3K-$5K", "$5K-$10K", "> $10K"]
    for (lo, hi), label in zip(bins, labels):
        mask = (y_actual >= lo) & (y_actual < hi)
        if mask.sum() == 0:
            continue
        bucket_errors = pct_errors[mask]
        within20 = (bucket_errors <= 20).mean() * 100
        print(f"    {label:<20} {mask.sum():>6,} {np.mean(bucket_errors):>7.1f}% {np.median(bucket_errors):>7.1f}% {within20:>9.1f}%")

    # Worst predictions
    print(f"\n  Top 10 Worst Predictions:")
    worst = test.nlargest(10, "abs_pct_error")
    print(f"    {'Segment':<20} {'Actual':>10} {'Predicted':>10} {'Error%':>8}")
    print("    " + "-" * 50)
    for _, row in worst.iterrows():
        print(f"    {row['BusinessSegment']:<20} ${row['NetFee']:>9,.0f} ${row['predicted']:>9,.0f} {row['abs_pct_error']:>7.1f}%")

    return results


def backtest_win_probability():
    """Backtest win probability model on enriched BidData test set."""
    print("\n" + "=" * 80)
    print("BACKTEST 2: WIN PROBABILITY")
    print("=" * 80)

    model, features, meta = load_model_and_features(
        MODELS_DIR / "lightgbm_win_probability_v2.txt",
        MODELS_DIR / "lightgbm_win_probability_v2_metadata.json",
    )

    df = pd.read_csv("data/features/BidData_features_v2.csv", low_memory=False)
    # BidData uses BidDate, not StartDate
    date_col = "StartDate" if "StartDate" in df.columns else "BidDate"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Time-based split: last 20% is test
    n = len(df)
    test_start = int(n * 0.80)
    test = df.iloc[test_start:].copy()
    print(f"  Test set: {len(test):,} rows ({test[date_col].min().date()} to {test[date_col].max().date()})")
    print(f"  Won: {test['Won'].sum():,}, Lost: {(~test['Won'].astype(bool)).sum():,}")

    # Build feature matrix
    for f in features:
        if f not in test.columns:
            test[f] = 0

    X_test = test[features].values
    y_actual = test["Won"].values.astype(int)

    # Predict
    y_prob = model.predict(X_test)
    y_prob = np.clip(y_prob, 0.05, 0.95)
    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss

    results = {
        "n_test": len(test),
        "accuracy": float(accuracy_score(y_actual, y_pred)),
        "precision": float(precision_score(y_actual, y_pred)),
        "recall": float(recall_score(y_actual, y_pred)),
        "f1": float(f1_score(y_actual, y_pred)),
        "auc_roc": float(roc_auc_score(y_actual, y_prob)),
        "brier_score": float(brier_score_loss(y_actual, y_prob)),
    }

    print(f"\n  Overall Results:")
    print(f"    AUC-ROC:       {results['auc_roc']:.4f}")
    print(f"    Accuracy:      {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)")
    print(f"    Precision:     {results['precision']:.4f}")
    print(f"    Recall:        {results['recall']:.4f}")
    print(f"    F1:            {results['f1']:.4f}")
    print(f"    Brier Score:   {results['brier_score']:.4f}")

    # Calibration analysis
    print(f"\n  Calibration (does predicted probability match reality?):")
    print(f"    {'Predicted Range':<20} {'Count':>6} {'Actual Win%':>12} {'Avg Predicted':>14} {'Gap':>8}")
    print("    " + "-" * 62)

    calibration = []
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    for (lo, hi), label in zip(bins, labels):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        actual_rate = y_actual[mask].mean() * 100
        pred_avg = y_prob[mask].mean() * 100
        gap = actual_rate - pred_avg
        calibration.append({"bin": label, "count": int(mask.sum()),
                           "actual_win_pct": round(actual_rate, 1),
                           "predicted_avg": round(pred_avg, 1),
                           "gap": round(gap, 1)})
        print(f"    {label:<20} {mask.sum():>6,} {actual_rate:>11.1f}% {pred_avg:>13.1f}% {gap:>+7.1f}%")

    results["calibration_bins"] = calibration

    # By segment
    test["predicted_prob"] = y_prob
    test["predicted_win"] = y_pred

    print(f"\n  By Business Segment:")
    print(f"    {'Segment':<25} {'Count':>6} {'AUC':>8} {'Accuracy':>10} {'Win Rate':>10}")
    print("    " + "-" * 61)
    for seg in sorted(test["BusinessSegment"].unique()):
        mask = test["BusinessSegment"] == seg
        sub = test[mask]
        if len(sub) < 20 or sub["Won"].nunique() < 2:
            continue
        seg_auc = roc_auc_score(sub["Won"], sub["predicted_prob"])
        seg_acc = accuracy_score(sub["Won"], sub["predicted_win"])
        win_rate = sub["Won"].mean() * 100
        print(f"    {seg:<25} {len(sub):>6,} {seg_auc:>7.3f} {seg_acc*100:>9.1f}% {win_rate:>9.1f}%")

    # Confidence vs accuracy
    test["prob_distance"] = np.abs(test["predicted_prob"] - 0.5)
    confident = test[test["prob_distance"] > 0.3]
    uncertain = test[test["prob_distance"] <= 0.15]

    if len(confident) > 0:
        conf_acc = accuracy_score(confident["Won"], confident["predicted_win"])
        print(f"\n  Confident predictions (>80% or <20%): {len(confident):,} bids, {conf_acc*100:.1f}% accuracy")
    if len(uncertain) > 0:
        unc_acc = accuracy_score(uncertain["Won"], uncertain["predicted_win"])
        print(f"  Uncertain predictions (35%-65%):       {len(uncertain):,} bids, {unc_acc*100:.1f}% accuracy")

    return results


def main():
    fee_results = backtest_bid_fee()
    winprob_results = backtest_win_probability()

    # Save results
    all_results = {
        "bid_fee_backtest": fee_results,
        "win_probability_backtest": winprob_results,
    }
    output_path = REPORTS_DIR / "model_backtest_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)
    print(f"  Bid Fee:  MAPE {fee_results['mape']:.1f}%, "
          f"Within 20%: {fee_results['within_20pct']:.1f}%, "
          f"Median error: ${fee_results['median_ae']:,.0f}")
    print(f"  Win Prob: AUC {winprob_results['auc_roc']:.4f}, "
          f"Accuracy {winprob_results['accuracy']*100:.1f}%, "
          f"Brier {winprob_results['brier_score']:.4f}")


if __name__ == "__main__":
    main()
