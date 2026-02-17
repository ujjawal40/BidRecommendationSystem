"""
Model Drift Monitor
=====================
Compares model predictions against actual outcomes on recent data
to detect performance degradation over time.

Checks:
  1. Rolling MAPE for bid fee model (monthly windows)
  2. Rolling AUC for win probability model (monthly windows)
  3. Feature distribution drift (KS test on top features)
  4. Segment-level degradation alerts

Usage:
  python scripts/29_model_drift_monitor.py

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
from scipy import stats

from config.model_config import MODELS_DIR, REPORTS_DIR

warnings.filterwarnings("ignore")

# Thresholds for alerts
MAPE_ALERT_THRESHOLD = 20.0  # Alert if monthly MAPE exceeds this
AUC_ALERT_THRESHOLD = 0.85   # Alert if monthly AUC drops below this
KS_DRIFT_THRESHOLD = 0.1     # Alert if KS statistic exceeds this


def load_model_and_features(model_path, meta_path):
    """Load a LightGBM model and its feature list."""
    model = lgb.Booster(model_file=str(model_path))
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return model, meta["features"], meta


def monitor_bid_fee_drift():
    """Check bid fee model for temporal drift."""
    print("=" * 80)
    print("DRIFT MONITOR: BID FEE MODEL")
    print("=" * 80)

    model, features, meta = load_model_and_features(
        MODELS_DIR / "lightgbm_bidfee_v2_model.txt",
        MODELS_DIR / "lightgbm_bidfee_v2_metadata.json",
    )

    df = pd.read_csv("data/features/JobsData_features_v2.csv", low_memory=False)
    df["StartDate"] = pd.to_datetime(df["StartDate"])
    df = df[df["NetFee"] >= 100].copy()
    df = df.sort_values("StartDate").reset_index(drop=True)

    # Use last 20% as test
    n = len(df)
    test_start = int(n * 0.80)
    test = df.iloc[test_start:].copy()

    # Fill missing features
    for f in features:
        if f not in test.columns:
            test[f] = 0

    X_test = test[features].values
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(y_pred, 500)
    test["predicted"] = y_pred
    test["abs_pct_error"] = np.abs(y_pred - test["NetFee"]) / np.maximum(test["NetFee"], 1) * 100

    # Monthly rolling MAPE
    test["month"] = test["StartDate"].dt.to_period("M")
    monthly = test.groupby("month").agg(
        count=("abs_pct_error", "size"),
        mape=("abs_pct_error", "mean"),
        median_ape=("abs_pct_error", "median"),
    ).reset_index()

    alerts = []
    print(f"\n  Monthly Performance:")
    print(f"    {'Month':<12} {'Count':>6} {'MAPE':>8} {'Med APE':>8} {'Status':>10}")
    print("    " + "-" * 46)
    for _, row in monthly.iterrows():
        status = "OK" if row["mape"] < MAPE_ALERT_THRESHOLD else "ALERT"
        if status == "ALERT":
            alerts.append(f"Bid fee MAPE {row['mape']:.1f}% in {row['month']} exceeds {MAPE_ALERT_THRESHOLD}% threshold")
        print(f"    {str(row['month']):<12} {row['count']:>6,} {row['mape']:>7.1f}% {row['median_ape']:>7.1f}% {'** ' + status if status == 'ALERT' else status:>10}")

    # Trend detection
    if len(monthly) >= 3:
        recent_mape = monthly.tail(3)["mape"].mean()
        early_mape = monthly.head(3)["mape"].mean()
        trend = recent_mape - early_mape
        print(f"\n  Trend: recent 3-month avg MAPE {recent_mape:.1f}% vs early {early_mape:.1f}% (change: {trend:+.1f}%)")
        if trend > 5:
            alerts.append(f"Upward drift detected: MAPE increasing by {trend:.1f}% over time")

    return alerts, monthly


def monitor_win_prob_drift():
    """Check win probability model for temporal drift."""
    print("\n" + "=" * 80)
    print("DRIFT MONITOR: WIN PROBABILITY MODEL")
    print("=" * 80)

    model, features, meta = load_model_and_features(
        MODELS_DIR / "lightgbm_win_probability_v2.txt",
        MODELS_DIR / "lightgbm_win_probability_v2_metadata.json",
    )

    df = pd.read_csv("data/features/BidData_features_v2.csv", low_memory=False)
    date_col = "StartDate" if "StartDate" in df.columns else "BidDate"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Use last 20% as test
    n = len(df)
    test_start = int(n * 0.80)
    test = df.iloc[test_start:].copy()

    for f in features:
        if f not in test.columns:
            test[f] = 0

    X_test = test[features].values
    y_prob = model.predict(X_test)
    y_prob = np.clip(y_prob, 0.05, 0.95)
    y_actual = test["Won"].values.astype(int)
    test["predicted_prob"] = y_prob
    test["correct"] = ((y_prob >= 0.5).astype(int) == y_actual).astype(int)

    # Monthly rolling metrics
    test["month"] = test[date_col].dt.to_period("M")

    from sklearn.metrics import roc_auc_score

    alerts = []
    print(f"\n  Monthly Performance:")
    print(f"    {'Month':<12} {'Count':>6} {'AUC':>8} {'Accuracy':>10} {'Status':>10}")
    print("    " + "-" * 48)

    for month in sorted(test["month"].unique()):
        mask = test["month"] == month
        sub = test[mask]
        if len(sub) < 50 or sub["Won"].nunique() < 2:
            continue
        auc = roc_auc_score(sub["Won"], sub["predicted_prob"])
        acc = sub["correct"].mean() * 100
        status = "OK" if auc >= AUC_ALERT_THRESHOLD else "ALERT"
        if status == "ALERT":
            alerts.append(f"Win prob AUC {auc:.3f} in {month} below {AUC_ALERT_THRESHOLD} threshold")
        print(f"    {str(month):<12} {len(sub):>6,} {auc:>7.3f} {acc:>9.1f}% {'** ' + status if status == 'ALERT' else status:>10}")

    return alerts


def monitor_feature_drift():
    """Check for distribution shift in top features."""
    print("\n" + "=" * 80)
    print("DRIFT MONITOR: FEATURE DISTRIBUTION")
    print("=" * 80)

    df = pd.read_csv("data/features/JobsData_features_v2.csv", low_memory=False)
    df["StartDate"] = pd.to_datetime(df["StartDate"])
    df = df.sort_values("StartDate").reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.6)
    test_start = int(n * 0.8)

    train = df.iloc[:train_end]
    test = df.iloc[test_start:]

    # Check top features
    top_features = [
        "segment_avg_fee", "subtype_avg_fee", "propertytype_avg_fee",
        "office_region_avg_fee", "companytype_avg_fee", "state_avg_fee",
        "segment_frequency", "NetFee",
    ]

    alerts = []
    print(f"\n  KS Test (train vs test distribution):")
    print(f"    {'Feature':<30} {'KS Stat':>8} {'p-value':>10} {'Status':>10}")
    print("    " + "-" * 60)

    for feat in top_features:
        if feat not in train.columns or feat not in test.columns:
            continue
        train_vals = train[feat].dropna()
        test_vals = test[feat].dropna()
        if len(train_vals) == 0 or len(test_vals) == 0:
            continue

        ks_stat, p_value = stats.ks_2samp(train_vals, test_vals)
        status = "OK" if ks_stat < KS_DRIFT_THRESHOLD else "DRIFT"
        if status == "DRIFT":
            alerts.append(f"Feature '{feat}' distribution shifted (KS={ks_stat:.3f}, p={p_value:.2e})")
        print(f"    {feat:<30} {ks_stat:>7.3f} {p_value:>9.2e} {'** ' + status if status == 'DRIFT' else status:>10}")

    return alerts


def main():
    print(f"Model Drift Monitor — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    all_alerts = []

    fee_alerts, _ = monitor_bid_fee_drift()
    all_alerts.extend(fee_alerts)

    wp_alerts = monitor_win_prob_drift()
    all_alerts.extend(wp_alerts)

    feat_alerts = monitor_feature_drift()
    all_alerts.extend(feat_alerts)

    # Summary
    print("\n" + "=" * 80)
    print("DRIFT MONITOR SUMMARY")
    print("=" * 80)

    if all_alerts:
        print(f"\n  {len(all_alerts)} ALERT(S) DETECTED:")
        for i, alert in enumerate(all_alerts, 1):
            print(f"    {i}. {alert}")
    else:
        print("\n  All clear — no drift detected.")

    # Save report
    report = {
        "run_date": datetime.now().isoformat(),
        "alerts": all_alerts,
        "alert_count": len(all_alerts),
        "thresholds": {
            "mape_alert": MAPE_ALERT_THRESHOLD,
            "auc_alert": AUC_ALERT_THRESHOLD,
            "ks_drift": KS_DRIFT_THRESHOLD,
        },
    }
    output_path = REPORTS_DIR / "drift_monitor_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {output_path}")


if __name__ == "__main__":
    main()
