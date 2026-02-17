"""
Model Validation Summary Report
=================================
Generates a consolidated report of all validation metrics for both models.
Designed for stakeholder review and documentation.

Author: Ujjawal Dwivedi
Date: 2026-02-17
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import json
import warnings

from config.model_config import MODELS_DIR, REPORTS_DIR

warnings.filterwarnings("ignore")


def main():
    print("=" * 80)
    print("MODEL VALIDATION SUMMARY REPORT")
    print("=" * 80)

    # Load all results
    backtest_path = REPORTS_DIR / "model_backtest_results.json"
    cv_path = REPORTS_DIR / "cross_validation_results.json"
    drift_path = REPORTS_DIR / "drift_monitor_report.json"

    bidfee_meta = MODELS_DIR / "lightgbm_bidfee_v2_metadata.json"
    winprob_meta = MODELS_DIR / "lightgbm_win_probability_v2_metadata.json"

    with open(bidfee_meta, "r") as f:
        bf_meta = json.load(f)
    with open(winprob_meta, "r") as f:
        wp_meta = json.load(f)

    print("\n1. MODEL SPECIFICATIONS")
    print("-" * 40)
    print(f"  Bid Fee Model:")
    print(f"    Type:           LightGBM Regression (log1p transform)")
    print(f"    Features:       {bf_meta['num_features']}")
    print(f"    Best iteration: {bf_meta['best_iteration']}")
    print(f"    Target:         {bf_meta['target_variable']}")
    print(f"  Win Probability Model:")
    print(f"    Type:           LightGBM Classifier + Isotonic Calibration")
    print(f"    Features:       {wp_meta['num_features']}")
    print(f"    Best iteration: {wp_meta['best_iteration']}")
    print(f"    Calibration:    {wp_meta['calibration']['method']}")

    # Training metrics
    print("\n2. TRAINING METRICS")
    print("-" * 40)
    bf_test = bf_meta["metrics"]["test"]
    wp_test = wp_meta["metrics"]["test"]
    print(f"  Bid Fee (test set):")
    print(f"    RMSE:   ${bf_test['rmse']:,.0f}")
    print(f"    MAPE:   {bf_test['mape']:.1f}%")
    print(f"    R²:     {bf_test['r2']:.4f}")
    print(f"  Win Probability (test set):")
    print(f"    AUC:    {wp_test['auc_roc']:.4f}")
    print(f"    Acc:    {wp_test['accuracy']*100:.1f}%")
    print(f"    F1:     {wp_test['f1']:.4f}")

    # Overfitting
    print("\n3. OVERFITTING CHECK")
    print("-" * 40)
    bf_overfit = bf_meta["metrics"]["overfitting_ratio"]
    wp_overfit = wp_meta["metrics"]["overfitting_ratio"]
    print(f"  Bid Fee:       {bf_overfit:.2f}x (target < 2.0x) {'PASS' if bf_overfit < 2.0 else 'FAIL'}")
    print(f"  Win Prob:      {wp_overfit:.2f}x (target < 1.15x) {'PASS' if wp_overfit < 1.15 else 'FAIL'}")

    # Backtest
    if backtest_path.exists():
        with open(backtest_path, "r") as f:
            bt = json.load(f)

        print("\n4. BACKTEST (held-out chronological test set)")
        print("-" * 40)
        bf_bt = bt["bid_fee_backtest"]
        wp_bt = bt["win_probability_backtest"]
        print(f"  Bid Fee:")
        print(f"    MAPE:        {bf_bt['mape']:.1f}%")
        print(f"    Within 20%:  {bf_bt['within_20pct']:.1f}%")
        print(f"    Median error: ${bf_bt['median_ae']:,.0f}")
        print(f"  Win Probability:")
        print(f"    AUC:         {wp_bt['auc_roc']:.4f}")
        print(f"    Accuracy:    {wp_bt['accuracy']*100:.1f}%")
        print(f"    Brier:       {wp_bt['brier_score']:.4f}")

    # Cross-validation
    if cv_path.exists():
        with open(cv_path, "r") as f:
            cv = json.load(f)
        summary = cv["summary"]

        print("\n5. CROSS-VALIDATION (5-fold time-series)")
        print("-" * 40)
        print(f"  Bid Fee MAPE:   {summary['bid_fee_mape_mean']:.1f}% +/- {summary['bid_fee_mape_std']:.1f}%")
        print(f"  Win Prob AUC:   {summary['win_prob_auc_mean']:.4f} +/- {summary['win_prob_auc_std']:.4f}")

    # Drift
    if drift_path.exists():
        with open(drift_path, "r") as f:
            drift = json.load(f)

        print("\n6. DRIFT MONITORING")
        print("-" * 40)
        print(f"  Alerts: {drift['alert_count']}")
        print(f"  Status: {'ALL CLEAR' if drift['alert_count'] == 0 else 'ALERTS DETECTED'}")

    # Calibration
    if backtest_path.exists():
        wp_bt = bt["win_probability_backtest"]
        if "calibration_bins" in wp_bt:
            print("\n7. CALIBRATION QUALITY")
            print("-" * 40)
            for bin_data in wp_bt["calibration_bins"]:
                gap = bin_data["gap"]
                status = "GOOD" if abs(gap) <= 5 else "FAIR" if abs(gap) <= 10 else "POOR"
                print(f"  {bin_data['bin']:<10} actual={bin_data['actual_win_pct']}% predicted={bin_data['predicted_avg']}% gap={gap:+.1f}% [{status}]")

    # Overall verdict
    print("\n" + "=" * 80)
    print("OVERALL VERDICT")
    print("=" * 80)
    issues = []
    if bf_overfit >= 2.0:
        issues.append("Bid fee overfitting exceeds 2.0x")
    if wp_overfit >= 1.15:
        issues.append("Win prob overfitting exceeds 1.15x")
    if backtest_path.exists() and bf_bt["mape"] > 15:
        issues.append(f"Bid fee MAPE {bf_bt['mape']:.1f}% exceeds 15% threshold")
    if backtest_path.exists() and wp_bt["auc_roc"] < 0.85:
        issues.append(f"Win prob AUC {wp_bt['auc_roc']:.4f} below 0.85 threshold")

    if issues:
        print(f"  {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  ALL CHECKS PASSED")
        print("  Models are validated and ready for production.")


if __name__ == "__main__":
    main()
