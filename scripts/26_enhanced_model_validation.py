"""
Enhanced Model Validation — v1 vs v2 Side-by-Side Comparison
=============================================================
Compares v1 (BidData-trained) and v2 (JobsData-trained) models across
key metrics and sample predictions.

Outputs:
  - outputs/reports/v1_vs_v2_comparison.json
  - Console report

Author: Ujjawal Dwivedi
Date: 2026-02-15
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import json
import warnings

import numpy as np

from config.model_config import MODELS_DIR, REPORTS_DIR

warnings.filterwarnings("ignore")


def load_metadata(name):
    path = MODELS_DIR / name
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def compare_models():
    """Compare v1 and v2 model metrics side by side."""
    print("=" * 80)
    print("MODEL COMPARISON: v1 vs v2")
    print("=" * 80)

    comparison = {}

    # Phase 1A: Bid Fee
    v1_bidfee = load_metadata("lightgbm_metadata.json")
    v2_bidfee = load_metadata("lightgbm_bidfee_v2_metadata.json")

    print("\n--- PHASE 1A: BID FEE PREDICTION ---")
    print(f"{'Metric':<30} {'v1':>12} {'v2':>12} {'Change':>12}")
    print("-" * 66)

    bidfee_comparison = {}

    if v1_bidfee and v2_bidfee:
        v1_test = v1_bidfee.get("metrics", {}).get("test", {})
        v2_test = v2_bidfee.get("metrics", {}).get("test", {})

        metrics = [
            ("RMSE", "rmse"),
            ("MAPE", "mape"),
            ("R²", "r2"),
            ("MAE", "mae"),
        ]

        for label, key in metrics:
            v1_val = v1_test.get(key, None)
            v2_val = v2_test.get(key, None)
            if v1_val is not None and v2_val is not None:
                change = ((v2_val - v1_val) / abs(v1_val)) * 100
                bidfee_comparison[key] = {
                    "v1": round(v1_val, 4),
                    "v2": round(v2_val, 4),
                    "change_pct": round(change, 1),
                }

                # Format for display
                if key in ("mape",):
                    print(f"  {label:<28} {v1_val*100:>11.1f}% {v2_val*100:>11.1f}% {change:>+11.1f}%")
                elif key in ("r2",):
                    print(f"  {label:<28} {v1_val:>12.4f} {v2_val:>12.4f} {change:>+11.1f}%")
                else:
                    print(f"  {label:<28} ${v1_val:>10,.0f} ${v2_val:>10,.0f} {change:>+11.1f}%")

        # Overfitting
        v1_of = v1_bidfee.get("metrics", {}).get("overfitting_ratio", None)
        v2_of = v2_bidfee.get("metrics", {}).get("overfitting_ratio", None)
        if v1_of and v2_of:
            bidfee_comparison["overfitting_ratio"] = {"v1": round(v1_of, 2), "v2": round(v2_of, 2)}
            print(f"  {'Overfitting Ratio':<28} {v1_of:>12.2f}x {v2_of:>12.2f}x")

        # Features
        v1_feats = v1_bidfee.get("num_features", 0)
        v2_feats = v2_bidfee.get("num_features", 0)
        bidfee_comparison["num_features"] = {"v1": v1_feats, "v2": v2_feats}
        print(f"  {'Features':<28} {v1_feats:>12} {v2_feats:>12}")

    comparison["bidfee"] = bidfee_comparison

    # Phase 1B: Win Probability
    v1_winprob = load_metadata("lightgbm_win_probability_metadata.json")
    v2_winprob = load_metadata("lightgbm_win_probability_v2_metadata.json")

    print("\n--- PHASE 1B: WIN PROBABILITY ---")
    print(f"{'Metric':<30} {'v1':>12} {'v2':>12} {'Change':>12}")
    print("-" * 66)

    winprob_comparison = {}

    if v1_winprob and v2_winprob:
        v1_test = v1_winprob.get("metrics", {}).get("test", {})
        v2_test = v2_winprob.get("metrics", {}).get("test", {})

        metrics = [
            ("AUC-ROC", "auc_roc"),
            ("Accuracy", "accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1", "f1"),
        ]

        for label, key in metrics:
            v1_val = v1_test.get(key, None)
            v2_val = v2_test.get(key, None)
            if v1_val is not None and v2_val is not None:
                change = ((v2_val - v1_val) / abs(v1_val)) * 100
                winprob_comparison[key] = {
                    "v1": round(v1_val, 4),
                    "v2": round(v2_val, 4),
                    "change_pct": round(change, 1),
                }
                print(f"  {label:<28} {v1_val:>12.4f} {v2_val:>12.4f} {change:>+11.1f}%")

        # Overfitting
        v1_of = v1_winprob.get("metrics", {}).get("overfitting_ratio", None)
        v2_of = v2_winprob.get("metrics", {}).get("overfitting_ratio", None)
        if v1_of and v2_of:
            winprob_comparison["overfitting_ratio"] = {"v1": round(v1_of, 2), "v2": round(v2_of, 2)}
            print(f"  {'Overfitting Ratio':<28} {v1_of:>12.2f}x {v2_of:>12.2f}x")

        # Brier score
        v1_brier = v1_winprob.get("calibration", {}).get("brier_score", None)
        v2_brier = v2_winprob.get("calibration", {}).get("brier_score", None)
        if v1_brier and v2_brier:
            change = ((v2_brier - v1_brier) / abs(v1_brier)) * 100
            winprob_comparison["brier_score"] = {"v1": round(v1_brier, 4), "v2": round(v2_brier, 4)}
            print(f"  {'Brier Score':<28} {v1_brier:>12.4f} {v2_brier:>12.4f} {change:>+11.1f}%")

        # Features
        v1_feats = v1_winprob.get("num_features", 0)
        v2_feats = v2_winprob.get("num_features", 0)
        winprob_comparison["num_features"] = {"v1": v1_feats, "v2": v2_feats}
        print(f"  {'Features':<28} {v1_feats:>12} {v2_feats:>12}")

    comparison["winprob"] = winprob_comparison

    # Sample predictions
    print("\n--- SAMPLE v2 PREDICTIONS ---")
    try:
        from api.enhanced_prediction_service import EnhancedBidPredictor
        ep = EnhancedBidPredictor()

        test_cases = [
            {"business_segment": "Financing", "property_type": "Multifamily", "property_state": "Illinois",
             "target_time": 30, "sub_property_type": "Conventional", "office_region": "Great Lakes"},
            {"business_segment": "Litigation", "property_type": "Industrial", "property_state": "California",
             "target_time": 45, "sub_property_type": "Distribution/Logistics", "office_region": "Pacific"},
            {"business_segment": "Tax", "property_type": "Office", "property_state": "New York",
             "target_time": 21, "sub_property_type": "CBD", "office_region": "Northeast"},
        ]

        print(f"  {'Case':<35} {'Fee':>10} {'Win%':>8} {'EV':>10} {'Conf':>8}")
        print("  " + "-" * 71)

        sample_results = []
        for tc in test_cases:
            result = ep.predict(**tc)
            label = f"{tc['business_segment']}/{tc['property_type'][:10]}/{tc['property_state'][:8]}"
            fee = result["predicted_fee"]
            win = result["win_probability"]["probability_pct"]
            ev = result["expected_value"]
            conf = result["confidence_level"]
            print(f"  {label:<35} ${fee:>8,.0f} {win:>7.1f}% ${ev:>8,.0f} {conf:>8}")
            sample_results.append({
                "inputs": tc,
                "fee": fee,
                "win_prob": win,
                "ev": ev,
                "confidence": conf,
            })

        comparison["sample_predictions"] = sample_results
    except Exception as e:
        print(f"  Could not generate sample predictions: {e}")

    # Save comparison
    output_path = REPORTS_DIR / "v1_vs_v2_comparison.json"
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nSaved: {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if bidfee_comparison.get("mape"):
        v1_mape = bidfee_comparison["mape"]["v1"]
        v2_mape = bidfee_comparison["mape"]["v2"]
        print(f"  Bid Fee MAPE: {v1_mape*100:.1f}% -> {v2_mape*100:.1f}% "
              f"({'improved' if v2_mape < v1_mape else 'regressed'})")
    if winprob_comparison.get("auc_roc"):
        v1_auc = winprob_comparison["auc_roc"]["v1"]
        v2_auc = winprob_comparison["auc_roc"]["v2"]
        print(f"  Win Prob AUC: {v1_auc:.4f} -> {v2_auc:.4f} "
              f"({'improved' if v2_auc > v1_auc else 'regressed'})")

    return comparison


if __name__ == "__main__":
    compare_models()
