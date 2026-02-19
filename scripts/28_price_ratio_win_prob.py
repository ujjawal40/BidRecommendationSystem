"""
Option B Experiment: Stacked Price-Ratio Win Probability Model
==============================================================
Tests whether win probability can be derived cleanly from price positioning alone.

Supervisor's hypothesis: Phase 1A predicts market price P*. Win probability
should be a function of (your_bid / P*) — where you sit relative to market —
rather than 44+ independent features with leakage risks.

price_ratio = BidFee / blended_market_price
  where blended_market_price = 0.4*segment_avg + 0.3*state_avg + 0.3*propertytype_avg

We run three models for clear comparison:
  A. Logistic Regression on price_ratio alone  — "does the ratio explain it?"
  B. LightGBM on price_ratio + 7 context features — Option B (the real experiment)
  C. Current Phase 1B v2 AUC (0.948, 44 features) — production baseline

Nothing in production is modified. Outputs:
  outputs/models/price_ratio_win_prob_model.txt
  outputs/models/price_ratio_win_prob_calibrator.pkl
  outputs/models/price_ratio_win_prob_metadata.json
  outputs/reports/price_ratio_experiment_results.json

Run: python scripts/28_price_ratio_win_prob.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import json
import warnings
from datetime import datetime

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from config.model_config import DATA_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_START_DATE = "2018-01-01"
DATE_COLUMN = "BidDate"
TARGET = "Won"

# Blended market price weights (same as API's blended_benchmark)
BLEND_SEG = 0.4
BLEND_STATE = 0.3
BLEND_PROP = 0.3

# Current Phase 1B v2 baseline to beat
PHASE1B_V2_AUC = 0.948
PHASE1B_V2_FEATURES = 44

# LightGBM params — simple, deliberately constrained to avoid overfitting
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 16,           # Very shallow — we have few features
    "learning_rate": 0.02,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": 5,
    "min_child_samples": 50,
    "reg_alpha": 2.0,
    "reg_lambda": 2.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


# ============================================================================
# HELPERS
# ============================================================================

def sep(title=""):
    print("\n" + "=" * 70)
    if title:
        print(title)
        print("=" * 70)


def eval_model(y_true, y_proba, name):
    auc = roc_auc_score(y_true, y_proba)
    acc = accuracy_score(y_true, (y_proba >= 0.5).astype(int))
    brier = brier_score_loss(y_true, y_proba)
    print(f"  {name:<40s}  AUC={auc:.4f}  Acc={acc:.4f}  Brier={brier:.4f}")
    return {"auc": float(auc), "accuracy": float(acc), "brier": float(brier)}


# ============================================================================
# MAIN
# ============================================================================

def main():
    sep("OPTION B: PRICE-RATIO WIN PROBABILITY EXPERIMENT")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Hypothesis: win probability ~ f(price_ratio, segment_context)")
    print(f"Baseline to beat: Phase 1B v2 AUC = {PHASE1B_V2_AUC} ({PHASE1B_V2_FEATURES} features)")

    # ── Load data ────────────────────────────────────────────────────────────
    sep("LOADING DATA")

    df = pd.read_csv(DATA_DIR / "features" / "BidData_features_v2.csv", low_memory=False)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df[df[DATE_COLUMN] >= pd.Timestamp(DATA_START_DATE)].copy()
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    print(f"  Rows: {len(df):,}")
    print(f"  Date range: {df[DATE_COLUMN].min().date()} to {df[DATE_COLUMN].max().date()}")
    print(f"  Win rate: {df[TARGET].mean():.3f}")

    # ── Compute price ratio ──────────────────────────────────────────────────
    sep("COMPUTING PRICE RATIO")

    # Validate required columns are present
    required = ["BidFee", "segment_avg_fee", "state_avg_fee", "propertytype_avg_fee", TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Guard against zero denominators
    seg_avg = df["segment_avg_fee"].clip(lower=100)
    state_avg = df["state_avg_fee"].clip(lower=100)
    prop_avg = df["propertytype_avg_fee"].clip(lower=100)

    # Blended market price (same weights as API benchmark)
    blended = BLEND_SEG * seg_avg + BLEND_STATE * state_avg + BLEND_PROP * prop_avg
    df["price_ratio"] = df["BidFee"] / blended

    # Also compute component ratios for transparency
    df["ratio_vs_segment"] = df["BidFee"] / seg_avg
    df["ratio_vs_state"] = df["BidFee"] / state_avg

    print(f"  price_ratio  — median={df['price_ratio'].median():.3f}  "
          f"p10={df['price_ratio'].quantile(0.10):.3f}  "
          f"p90={df['price_ratio'].quantile(0.90):.3f}")

    # Win rate by price quartile — sanity check
    df["ratio_quartile"] = pd.qcut(df["price_ratio"], q=4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"])
    wr_by_quartile = df.groupby("ratio_quartile")[TARGET].mean()
    print(f"\n  Win rate by price quartile (higher ratio = more expensive bid):")
    for q, wr in wr_by_quartile.items():
        bar = "█" * int(wr * 30)
        print(f"    {q}: {wr:.3f}  {bar}")

    # ── Feature set for Option B ─────────────────────────────────────────────
    sep("OPTION B FEATURE SET")

    option_b_features = [
        # Core: price positioning (THE hypothesis feature)
        "price_ratio",           # BidFee / blended market benchmark
        "fee_vs_segment_ratio",  # BidFee / segment_avg (LOO)
        "bid_vs_state_ratio",    # BidFee / state_avg (LOO)

        # Segment/market context
        "subtype_frequency",     # How common this subtype is in training data
        "subtype_avg_fee",       # Average fee for this subtype (complement to frequency)
        "office_region_avg_fee", # Office region's average fee (market context)
        "segment_frequency",     # How common this segment is

        # Turnaround
        "targettime_log",        # Log of turnaround days

        # State market activity
        "state_frequency",       # How active this state is

        # Rolling market trends (time-aware, no leakage — computed on past bids only)
        "rolling_avg_fee_segment",  # Recent fee trend for this segment
        "rolling_avg_fee_state",    # Recent fee trend for this state
        "rolling_std_fee_segment",  # Fee volatility for this segment

        # Segment × state interaction (captures geo-segment market dynamics)
        "segment_x_state_fee",
    ]

    # Keep only columns that actually exist in the data
    available = [f for f in option_b_features if f in df.columns]
    missing_feat = [f for f in option_b_features if f not in df.columns]

    print(f"  Features requested : {len(option_b_features)}")
    print(f"  Features available : {len(available)}")
    if missing_feat:
        print(f"  Missing (skipped)  : {missing_feat}")
    print(f"\n  Feature set:")
    for i, f in enumerate(available, 1):
        print(f"    {i:2d}. {f}")

    # ── Time-based split (60/20/20) ──────────────────────────────────────────
    sep("TIME-BASED SPLIT (60/20/20)")

    n = len(df)
    train_end = int(n * 0.6)
    valid_end = int(n * 0.8)

    df_train = df.iloc[:train_end]
    df_valid = df.iloc[train_end:valid_end]
    df_test  = df.iloc[valid_end:]

    X_train = df_train[available].fillna(0).values
    X_valid = df_valid[available].fillna(0).values
    X_test  = df_test[available].fillna(0).values
    y_train = df_train[TARGET].values
    y_valid = df_valid[TARGET].values
    y_test  = df_test[TARGET].values

    # Price ratio column index (for logistic regression)
    ratio_idx = available.index("price_ratio")

    print(f"  Train: {len(df_train):,}  (win rate: {y_train.mean():.3f})")
    print(f"  Valid: {len(df_valid):,}  (win rate: {y_valid.mean():.3f})")
    print(f"  Test:  {len(df_test):,}  (win rate: {y_test.mean():.3f})")

    results = {}

    # ── Model A: Logistic Regression on price_ratio alone ────────────────────
    sep("MODEL A: LOGISTIC REGRESSION — price_ratio only")
    print("  Tests the simplest possible version of the hypothesis.")

    scaler_a = StandardScaler()
    X_train_a = scaler_a.fit_transform(X_train[:, [ratio_idx]])
    X_valid_a = scaler_a.transform(X_valid[:, [ratio_idx]])
    X_test_a  = scaler_a.transform(X_test[:, [ratio_idx]])

    lr_a = LogisticRegression(random_state=42, max_iter=500)
    lr_a.fit(X_train_a, y_train)

    proba_train_a = lr_a.predict_proba(X_train_a)[:, 1]
    proba_test_a  = lr_a.predict_proba(X_test_a)[:, 1]

    print(f"\n  Results:")
    results["logistic_price_ratio_train"] = eval_model(y_train, proba_train_a, "Train")
    results["logistic_price_ratio_test"]  = eval_model(y_test,  proba_test_a,  "Test")

    overfit_a = results["logistic_price_ratio_train"]["auc"] / results["logistic_price_ratio_test"]["auc"]
    print(f"  Overfit ratio: {overfit_a:.2f}×")

    # ── Model B: LightGBM on all Option B features ────────────────────────────
    sep("MODEL B: LIGHTGBM — price_ratio + context features")
    print(f"  {len(available)} features total (vs Phase 1B v2's {PHASE1B_V2_FEATURES})")

    scale_pos_weight = (1 - y_train.mean()) / y_train.mean()
    params = {**LGBM_PARAMS, "scale_pos_weight": scale_pos_weight}
    print(f"  scale_pos_weight: {scale_pos_weight:.3f}")

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=available)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    model_b = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(200),
        ],
    )

    print(f"\n  Best iteration: {model_b.best_iteration}")

    proba_train_b = model_b.predict(X_train, num_iteration=model_b.best_iteration)
    proba_valid_b = model_b.predict(X_valid, num_iteration=model_b.best_iteration)
    proba_test_b  = model_b.predict(X_test,  num_iteration=model_b.best_iteration)

    print(f"\n  Results (uncalibrated):")
    results["option_b_train"] = eval_model(y_train, proba_train_b, "Train")
    results["option_b_valid"] = eval_model(y_valid, proba_valid_b, "Valid")
    results["option_b_test"]  = eval_model(y_test,  proba_test_b,  "Test")

    overfit_b = results["option_b_train"]["auc"] / results["option_b_test"]["auc"]
    print(f"  Overfit ratio: {overfit_b:.2f}×  (target < 1.15×)")

    # ── Calibration ──────────────────────────────────────────────────────────
    sep("CALIBRATION (Isotonic Regression)")

    calibrator = IsotonicRegression(y_min=0.05, y_max=0.95, out_of_bounds="clip")
    calibrator.fit(proba_valid_b, y_valid)

    proba_test_cal = calibrator.predict(proba_test_b)

    brier_raw = brier_score_loss(y_test, proba_test_b)
    brier_cal = brier_score_loss(y_test, proba_test_cal)
    print(f"  Brier score raw:        {brier_raw:.4f}")
    print(f"  Brier score calibrated: {brier_cal:.4f}")
    print(f"  Improvement:            {(brier_raw - brier_cal) / brier_raw * 100:.1f}%")

    results["option_b_test_calibrated"] = {
        "auc":    float(roc_auc_score(y_test, proba_test_cal)),
        "brier":  float(brier_cal),
    }

    # Calibration quality check
    print(f"\n  Calibration bins:")
    print(f"    {'Bin':<12} {'Count':>6} {'Actual':>8} {'Raw':>8} {'Calibr':>8}")
    print("    " + "-" * 50)
    for lo, hi, label in [(0, .2, "0-20%"), (.2, .4, "20-40%"), (.4, .6, "40-60%"),
                          (.6, .8, "60-80%"), (.8, 1, "80-100%")]:
        mask = (proba_test_b >= lo) & (proba_test_b < hi)
        if mask.sum() == 0:
            continue
        actual   = y_test[mask].mean() * 100
        raw_avg  = proba_test_b[mask].mean() * 100
        cal_avg  = proba_test_cal[mask].mean() * 100
        print(f"    {label:<12} {mask.sum():>6,} {actual:>7.1f}% {raw_avg:>7.1f}% {cal_avg:>7.1f}%")

    # ── Feature importance ────────────────────────────────────────────────────
    sep("FEATURE IMPORTANCE")

    importance = pd.DataFrame({
        "feature": available,
        "importance": model_b.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    importance["pct"] = importance["importance"] / importance["importance"].sum() * 100

    for _, row in importance.iterrows():
        bar = "█" * max(1, int(row["pct"] / 2))
        print(f"  {row['feature']:35s} {row['pct']:6.2f}%  {bar}")

    # ── Comparison summary ────────────────────────────────────────────────────
    sep("COMPARISON SUMMARY")

    auc_logistic = results["logistic_price_ratio_test"]["auc"]
    auc_option_b = results["option_b_test"]["auc"]
    auc_option_b_cal = results["option_b_test_calibrated"]["auc"]

    print(f"  {'Model':<45}  {'AUC':>6}  {'Features':>8}  {'Overfit':>7}")
    print("  " + "-" * 75)
    print(f"  {'A. Logistic — price_ratio only':<45}  {auc_logistic:.4f}  {'1':>8}  {overfit_a:.2f}×")
    print(f"  {'B. Option B LightGBM (uncalibrated)':<45}  {auc_option_b:.4f}  {len(available):>8}  {overfit_b:.2f}×")
    print(f"  {'B. Option B LightGBM (calibrated)':<45}  {auc_option_b_cal:.4f}  {len(available):>8}  {overfit_b:.2f}×")
    print(f"  {'C. Phase 1B v2 (production, 44 features)':<45}  {PHASE1B_V2_AUC:.4f}  {PHASE1B_V2_FEATURES:>8}  1.04×")

    gap = PHASE1B_V2_AUC - auc_option_b
    gap_pct = gap / PHASE1B_V2_AUC * 100
    print(f"\n  AUC gap: Option B vs Phase 1B v2: {gap:+.4f} ({gap_pct:.1f}%)")

    # Verdict
    print("\n  VERDICT:")
    if auc_option_b >= PHASE1B_V2_AUC - 0.02:
        print("  ✓ Option B matches production AUC within 2pp — switch is justified.")
        verdict = "RECOMMEND_SWITCH"
    elif auc_option_b >= PHASE1B_V2_AUC - 0.05:
        print("  ~ Option B is within 5pp — meaningful loss, investigate feature set.")
        verdict = "INVESTIGATE"
    else:
        print(f"  ✗ Option B AUC is {gap_pct:.1f}% below production — extra features matter.")
        verdict = "KEEP_CURRENT"

    results["verdict"] = verdict
    results["auc_gap_vs_production"] = float(gap)
    results["option_b_features"] = available
    results["phase1b_v2_baseline"] = {"auc": PHASE1B_V2_AUC, "features": PHASE1B_V2_FEATURES}

    # ── Plots ─────────────────────────────────────────────────────────────────
    sep("SAVING PLOTS")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Win rate vs price ratio — the core relationship
    df_test_plot = df_test.copy()
    df_test_plot["proba"] = proba_test_cal
    df_test_plot["ratio_bin"] = pd.cut(
        df_test_plot["price_ratio"].clip(0.3, 3.0), bins=20
    )
    ratio_stats = df_test_plot.groupby("ratio_bin", observed=True).agg(
        actual_win_rate=(TARGET, "mean"),
        predicted_win_rate=("proba", "mean"),
        count=(TARGET, "count"),
    ).dropna()

    ax = axes[0]
    x_pos = range(len(ratio_stats))
    ax.bar(x_pos, ratio_stats["actual_win_rate"], alpha=0.6, label="Actual", color="steelblue")
    ax.plot(x_pos, ratio_stats["predicted_win_rate"], "r-o", markersize=3, label="Option B predicted")
    ax.axvline(x=len(ratio_stats) // 2, color="gray", linestyle="--", alpha=0.5, label="ratio=1.0 (market price)")
    ax.set_title("Win Rate vs Price Ratio (Test Set)")
    ax.set_xlabel("Price ratio bin (left=underbid, right=overbid)")
    ax.set_ylabel("Win Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Feature importance
    ax = axes[1]
    top = importance.head(len(available))
    ax.barh(range(len(top)), top["pct"], color="teal", alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=8)
    ax.set_xlabel("Importance (%)")
    ax.set_title(f"Option B Feature Importance ({len(available)} features)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    # AUC comparison bar chart
    ax = axes[2]
    models = [
        "Logistic\n(ratio only)", f"Option B\n({len(available)} features)", "Phase 1B v2\n(44 features)"
    ]
    aucs = [auc_logistic, auc_option_b_cal, PHASE1B_V2_AUC]
    colors = ["#fbbf24", "#0d9488", "#6366f1"]
    bars = ax.bar(models, aucs, color=colors, alpha=0.85)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{auc:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC-ROC (Test Set)")
    ax.set_title("AUC Comparison")
    ax.axhline(y=PHASE1B_V2_AUC, color="red", linestyle="--", alpha=0.5, label=f"Baseline {PHASE1B_V2_AUC}")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    plt.tight_layout()
    fig_path = FIGURES_DIR / "price_ratio_experiment.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {fig_path}")

    # ── Save model and results ────────────────────────────────────────────────
    sep("SAVING OUTPUTS")

    # LightGBM model (Option B)
    model_path = MODELS_DIR / "price_ratio_win_prob_model.txt"
    model_b.save_model(str(model_path))
    print(f"  Model:       {model_path}")

    # Calibrator
    cal_path = MODELS_DIR / "price_ratio_win_prob_calibrator.pkl"
    with open(cal_path, "wb") as f:
        pickle.dump(calibrator, f)
    print(f"  Calibrator:  {cal_path}")

    # Metadata
    metadata = {
        "model_type": "LightGBM Binary Classifier (Option B — Price Ratio)",
        "version": "experiment",
        "phase": "1B - Win Probability (Price Ratio Stacked Ensemble)",
        "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hypothesis": "Win probability is primarily a function of price_ratio (bid / market)",
        "features": available,
        "num_features": len(available),
        "feature_importance": importance[["feature", "pct"]].to_dict(orient="records"),
        "blend_weights": {"segment": BLEND_SEG, "state": BLEND_STATE, "propertytype": BLEND_PROP},
        "hyperparameters": params,
        "best_iteration": int(model_b.best_iteration),
        "metrics": results,
        "verdict": verdict,
        "production_baseline": {"model": "Phase 1B v2", "auc": PHASE1B_V2_AUC, "features": PHASE1B_V2_FEATURES},
        "note": "EXPERIMENT ONLY — production Phase 1B v2 is not modified.",
    }

    meta_path = MODELS_DIR / "price_ratio_win_prob_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Metadata:    {meta_path}")

    # Compact results JSON
    results_path = REPORTS_DIR / "price_ratio_experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results:     {results_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    sep("FINAL SUMMARY")
    print(f"  Logistic (ratio only):     AUC = {auc_logistic:.4f}")
    print(f"  Option B ({len(available)} features):       AUC = {auc_option_b_cal:.4f}  (calibrated)")
    print(f"  Phase 1B v2 (production):  AUC = {PHASE1B_V2_AUC:.4f}")
    print(f"\n  AUC gap:  {gap:+.4f}  ({gap_pct:.1f}% below production)")
    print(f"  Overfit:  {overfit_b:.2f}×")
    print(f"  Verdict:  {verdict}")
    print(f"\n  Note: price_ratio importance = {importance[importance['feature'] == 'price_ratio']['pct'].values[0]:.1f}%")
    print(f"  Production Phase 1B v2 is UNTOUCHED. Switch only after reviewing verdict.")

    return model_b, results


if __name__ == "__main__":
    model_b, results = main()
