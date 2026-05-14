"""
v2 vs v3 Fee Sensitivity Curve Comparison
==========================================
Plots win-probability vs bid-fee for both v2 and v3_fee_sensitive on the
same set of canonical inputs. Saved to outputs/figures/.

Usage:
    python scripts/29b_v2_vs_v3_curve_comparison.py

Reads:
    outputs/models/lightgbm_win_probability_v2.txt
    outputs/models/win_probability_v2_calibrator.pkl
    outputs/models/lightgbm_win_probability_v3_fee_sensitive.txt
    outputs/models/win_probability_v3_fee_sensitive_calibrator.pkl
    outputs/reports/win_rate_lookups_v3.json
    outputs/reports/api_precomputed_stats_v2.json

Writes:
    outputs/figures/v2_vs_v3_fee_curve.png
    outputs/reports/v2_vs_v3_fee_curve_data.json
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from api.enhanced_prediction_service import EnhancedBidPredictor


SCENARIOS = [
    {"label": "Financing / Multifamily / Illinois",
     "kwargs": {"business_segment": "Financing", "property_type": "Multifamily",
                "property_state": "Illinois", "sub_property_type": "Conventional",
                "office_location": "Chicago"}},
    {"label": "Litigation / Industrial / California",
     "kwargs": {"business_segment": "Litigation", "property_type": "Industrial",
                "property_state": "California", "sub_property_type": "Distribution/Logistics",
                "office_location": "Los Angeles"}},
    {"label": "Financial Reporting / Hospitality / Florida",
     "kwargs": {"business_segment": "Financial Reporting", "property_type": "Hospitality",
                "property_state": "Florida", "sub_property_type": "Hotel",
                "office_location": "Miami"}},
    {"label": "Consulting / Office / New York",
     "kwargs": {"business_segment": "Consulting", "property_type": "Office",
                "property_state": "New York", "sub_property_type": "Office",
                "office_location": "New York"}},
]


def main():
    p = EnhancedBidPredictor()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    data = {}

    for ax, scenario in zip(axes.ravel(), SCENARIOS):
        res = p.predict(target_time=30, delivery_days=30, **scenario["kwargs"])
        predicted_fee = res["predicted_fee"]
        seg_avg = res.get("segment_benchmark", predicted_fee)
        fees = np.geomspace(seg_avg * 0.3, seg_avg * 2.5, 30)

        # v2 curve (via raw booster, no saturation fallback)
        feats = p._generate_features(target_time=30, delivery_days=30, **{
            k: v for k, v in scenario["kwargs"].items()
            if k in ("business_segment", "property_type", "property_state",
                     "sub_property_type", "office_location")
        })
        p._populate_v3_features(
            feats,
            business_segment=scenario["kwargs"]["business_segment"],
            property_type=scenario["kwargs"]["property_type"],
            property_state=scenario["kwargs"]["property_state"],
            sub_property_type=scenario["kwargs"].get("sub_property_type"),
            office_region=None,
            company_location=scenario["kwargs"].get("office_location"),
        )

        v2_probs, v3_probs = [], []
        for fee in fees:
            # v2 raw
            f2 = dict(feats); f2["BidFee"] = float(fee)
            fee_seg = f2.get("segment_avg_fee", seg_avg) or seg_avg
            f2["fee_vs_segment_ratio"] = fee / fee_seg
            f2["fee_diff_from_segment"] = fee - fee_seg
            f2["bid_vs_state_ratio"] = fee / fee_seg
            f2["bid_vs_client_ratio"] = fee / fee_seg
            f2["fee_percentile_segment"] = min(1.0, max(0.0, fee / (2 * fee_seg)))
            fv = [f2.get(n, p.feature_defaults.get(n, {}).get("global_median", 0)) for n in p.win_prob_features]
            raw = float(p.win_prob_model.predict(np.array([fv]))[0])
            try:
                pr2 = float(p.win_prob_calibrator.predict([raw])[0]) if p.win_prob_calibrator else raw
            except Exception:
                pr2 = raw
            v2_probs.append(max(0.05, min(0.95, pr2)) * 100)

            # v3
            pr3 = p._predict_v3_fee_sensitive(dict(feats), float(fee),
                                              business_segment=scenario["kwargs"]["business_segment"])
            v3_probs.append((pr3 or 0.5) * 100)

        ax.plot(fees, v2_probs, "r-",  linewidth=2, label="v2 (AUC 0.948)")
        ax.plot(fees, v3_probs, "b-",  linewidth=2, label="v3_fee_sensitive (AUC 0.883)")
        ax.axvline(predicted_fee, color="gray", linestyle="--", alpha=0.5, label=f"predicted_fee ${predicted_fee:,.0f}")
        ax.set_xlabel("Bid Fee ($)")
        ax.set_ylabel("Win Probability (%)")
        ax.set_title(scenario["label"])
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        data[scenario["label"]] = {
            "predicted_fee": round(float(predicted_fee), 0),
            "v2_span_pp": round(max(v2_probs) - min(v2_probs), 1),
            "v3_span_pp": round(max(v3_probs) - min(v3_probs), 1),
        }

    plt.tight_layout()
    out_fig = Path("outputs/figures/v2_vs_v3_fee_curve.png")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_fig), dpi=150, bbox_inches="tight")
    print(f"Saved: {out_fig}")

    out_json = Path("outputs/reports/v2_vs_v3_fee_curve_data.json")
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {out_json}")

    print("\nSpan comparison (pp):")
    for label, d in data.items():
        print(f"  {label:<55s}  v2={d['v2_span_pp']:>5.1f}  v3={d['v3_span_pp']:>5.1f}")


if __name__ == "__main__":
    main()
