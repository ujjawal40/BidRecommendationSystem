"""
Fee-Sensitive Win Probability Model (Phase 1B v3)
==================================================
Replaces raw frequency count features (subtype_frequency, company_location_frequency,
propertytype_frequency, office_region_frequency, state_frequency) with LOO target-encoded
win rates. These are more directly predictive than counts and make fee features relatively
more informative.

Background
----------
The v2 win prob model (AUC 0.948) has a nearly flat fee sensitivity curve within the
normal fee range. Feature importance shows:
  - Frequency/identity features: ~42% of weight (who you are)
  - Fee-relative features: ~8% of weight (what you charge)

The data genuinely supports this: Q1-Q3 bids have 55-57% win rates (nearly flat),
only Q4 (top quartile of fees) drops to 31.7%. Within normal fee range, firm
specialization and regional fit dominate.

However, frequency counts are a noisy proxy for specialization. Replacing them with LOO
win rates makes the model more interpretable and puts the fee signal in direct competition
with a better baseline. If fee features gain importance against cleaner alternatives, the
curve may become more responsive.

LOO Encoding Strategy
---------------------
Uses expanding-window historical encoding (correct for time-series):
  - Sort by BidDate (chronological)
  - For row at time T: win_rate = wins_before_T / count_before_T for this group
  - Smoothed with global prior: (wins + k * global_rate) / (count + k), k=30
  - Minimum 5 rows of history before using group rate; else fall back to global rate

This avoids leakage: no future information bleeds into a row's features.

Outputs
-------
  - outputs/models/lightgbm_win_probability_v3.txt
  - outputs/models/lightgbm_win_probability_v3_metadata.json
  - outputs/models/win_probability_v3_calibrator.pkl
  - outputs/reports/win_probability_v3_feature_importance.csv
  - outputs/figures/win_probability_v3_fee_sensitivity.png
  - outputs/reports/win_rate_lookups_v3.json   (for API inference)

Usage
-----
  python scripts/29_fee_sensitive_win_prob.py

Author: Ujjawal Dwivedi
Date: 2026-02-25
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import pickle
import warnings
from datetime import datetime

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    roc_auc_score,
)

from config.model_config import DATA_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================

DATA_START_DATE = "2018-01-01"
DATE_COLUMN = "BidDate"
LOO_SMOOTHING_K = 30        # Prior strength for LOO win rate smoothing
LOO_MIN_HISTORY = 5         # Min rows of history before using group win rate

V3_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 24,
    "learning_rate": 0.02,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": 8,
    "min_child_samples": 30,
    "reg_alpha": 2.0,
    "reg_lambda": 2.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

NUM_BOOST_ROUND = 2000
EARLY_STOPPING = 100

# Same exclusions as v2
EXCLUDE_COLUMNS = [
    "BidId", "BidFileNumber", "BidName",
    "BidDate", "Bid_DueDate",
    "Won",                                    # Target
    "BidStatusName",
    "Bid_JobPurpose", "Bid_Deliverable",
    "Market", "Submarket",
    "BusinessSegment", "BusinessSegmentDetail",
    "DistanceInKM",
    "Bid_Property_Type", "Bid_SubProperty_Type", "Bid_SpecificUseProperty_Type",
    "PropertyId", "PropertyName", "PropertyType", "SubType",
    "PropertyCity", "PropertyState",
    "ZipCode", "MarketOrientation",
    "AddressDisplayCalc", "GrossBuildingAreaRange", "YearBuiltRange",
    "OfficeId", "OfficeCode", "OfficeCompanyName", "OfficeLocation",
    "JobId", "JobName", "JobStatus", "JobType", "AppraisalFileType",
    "BidCompanyName", "BidCompanyType",
    "BidFee_Original", "TargetTime_Original",
    "Jobs_SubType", "Jobs_SpecificUse", "Jobs_CompanyLocation",
    "Jobs_Office_Region", "Jobs_PropertyType", "Jobs_MarketOrientation",
]

# Features nulled out for lost bids (differential imputation leakage).
# MUST stay in sync with scripts/25_enhanced_win_probability.py — the derivative
# features (*_log, fee_per_*, *_x_segment_fee) are computed from Jobs_* columns
# that have differential imputation and leak the target.
LEAKY_JOB_DERIVED_FEATURES = [
    "JobCount", "IECount", "LeaseCount", "SaleCount",
    "Jobs_PropertyID", "Jobs_OfficeID",
    "Jobs_PropertyType",
    "Jobs_GrossBuildingSF", "Jobs_GLARentableSF",
    "Jobs_GrossLandAreaAcres", "Jobs_JobLength_Days",
    "Jobs_YearBuilt", "Jobs_MarketOrientation",
    "Jobs_Zip_Population", "Jobs_Zip_PopDensity", "Jobs_Zip_HouseholdsPerZip",
    "Jobs_Zip_GrowthRank", "Jobs_Zip_AverageHouseValue", "Jobs_Zip_IncomePerHousehold",
    "Jobs_Zip_MedianAge", "Jobs_Zip_MedianIncome", "Jobs_Zip_NumberOfBusinesses",
    "Jobs_Zip_NumberOfEmployees", "Jobs_Zip_LandArea", "Jobs_Zip_PopulationEstimate",
    "Jobs_Zip_PopCount", "Jobs_Zip_DeliveryTotal", "Jobs_Zip_WorkersOutZip",
    # Derivatives of differentially-imputed Jobs_ columns — equally leaky.
    "land_acres_log",
    "building_sf_log",
    "fee_per_sqft",
    "fee_per_day",
    "joblength_log",
    "joblength_bucket",
    "joblength_x_segment_fee",
    "income_x_segment_fee",
    "pop_density_log",
]

# Pre-existing target-encoded columns in BidData_features_v2.csv that are NOT
# computed with LOO and therefore leak. v3 re-creates these with proper
# expanding-window LOO under the same column names, so the engineered values
# overwrite the leaky originals — but we keep this list as documentation.
PREEXISTING_LEAKY_RATE_COLS = [
    "client_win_rate",
    "office_win_rate",
    "cumulative_wins_client",
    "cumulative_winrate_client",
]

# Frequency count features replaced by LOO win rates in v3
FREQUENCY_FEATURES_REPLACED = [
    "subtype_frequency",
    "company_location_frequency",
    "propertytype_frequency",
    "office_region_frequency",
    "state_frequency",
    "segment_frequency",
]

# Ablation toggle — when True, drops absolute-fee context aggregates that
# dominate the model and crowd out BidFee as a signal. The hypothesis is
# that without subtype_avg_fee etc., the model is forced to weight BidFee
# more heavily, producing a steeper fee-sensitivity curve. Set via the
# V3_DROP_CONTEXT_FEE env var (any non-empty value enables).
import os as _os
DROP_CONTEXT_FEE_AGGREGATES = bool(_os.environ.get("V3_DROP_CONTEXT_FEE"))

CONTEXT_FEE_AGGREGATES = [
    "subtype_avg_fee",
    "propertytype_avg_fee",
    "office_region_avg_fee",
    "office_avg_fee",
    "client_avg_fee",
    "client_std_fee",
    "segment_avg_fee",
    "segment_std_fee",
    "state_avg_fee",
    "rolling_avg_fee_segment",
    "rolling_avg_fee_state",
    "rolling_std_fee_segment",
    "segment_x_state_fee",
]


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def compute_loo_win_rates(
    df: pd.DataFrame,
    group_col: str,
    win_col: str = "Won",
    k: int = LOO_SMOOTHING_K,
    min_history: int = LOO_MIN_HISTORY,
    output_col: str = None,
) -> pd.Series:
    """
    Expanding-window leave-one-out win rate for a categorical column.

    For each row at position i:
      history = all rows BEFORE i with the same group value
      win_rate = (sum(history wins) + k * global_rate) / (len(history) + k)
      If len(history) < min_history: use global prior only

    Parameters
    ----------
    df : sorted by date (ascending) before calling
    group_col : categorical column name (e.g. "SubType")
    k : smoothing strength (higher = more regularization toward global rate)
    min_history : rows of history required before using group rate

    Returns
    -------
    pd.Series with same index as df
    """
    if output_col is None:
        output_col = group_col.lower() + "_win_rate"

    global_rate = df[win_col].mean()
    wins = df[win_col].astype(float)

    # Expanding cum-sums per group, then LOO
    cum_wins = df.groupby(group_col)[win_col].transform(
        lambda x: x.cumsum().shift(1).fillna(0)
    )
    cum_count = df.groupby(group_col)[win_col].transform(
        lambda x: (x.expanding().count() - 1).clip(lower=0)
    )

    # Smoothed LOO rate
    rate = (cum_wins + k * global_rate) / (cum_count + k)

    # Fall back to global for rows with too little history
    rate[cum_count < min_history] = global_rate

    return rate


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the v3 feature set:
      - LOO win rates (replacing frequency counts)
      - All existing engineered features from v2 (market stats, fee ratios, demographics)
      - Same temporal / geographic features
    """
    print(f"\n[Features] Building v3 features on {len(df):,} rows...")

    # Sort chronologically (required for expanding-window LOO)
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    # ---- LOO Win Rates ---------------------------------------------------
    loo_cols = {
        "SubType":             "subtype_win_rate",
        "PropertyType":        "propertytype_win_rate",
        "PropertyState":       "state_win_rate",
        "BusinessSegment":     "segment_win_rate",
    }
    # Office region: use Jobs_Office_Region if available, else OfficeLocation
    if "Jobs_Office_Region" in df.columns and df["Jobs_Office_Region"].notna().any():
        loo_cols["Jobs_Office_Region"] = "office_region_win_rate"
    if "OfficeLocation" in df.columns and df["OfficeLocation"].notna().any():
        loo_cols["OfficeLocation"] = "company_location_win_rate"

    for col, out_col in loo_cols.items():
        if col in df.columns:
            df[out_col] = compute_loo_win_rates(df, col, output_col=out_col)
            print(f"  LOO win rate: {col} → {out_col}")
        else:
            df[out_col] = df["Won"].mean()
            print(f"  LOO win rate: {col} NOT FOUND — using global rate")

    # ---- Fee Percentile (empirical CDF) ----------------------------------
    # For each row: what fraction of historical bids in this segment had fee <= this fee?
    # More accurate than the approximation fee / (2 * seg_avg).
    def empirical_percentile_loo(grp):
        fees = grp["BidFee"].values
        dates = grp.index  # already sorted by date
        pct = np.zeros(len(fees))
        for i in range(len(fees)):
            historical = fees[:i]
            if len(historical) == 0:
                pct[i] = 0.5  # unknown → median prior
            else:
                pct[i] = np.mean(historical <= fees[i])
        return pd.Series(pct, index=dates)

    df["fee_percentile_empirical"] = (
        df.groupby("BusinessSegment", group_keys=False)
        .apply(empirical_percentile_loo)
    )

    return df


def build_market_stats(train_df: pd.DataFrame) -> dict:
    """
    Compute global win-rate lookups from training data only.
    Saved to outputs/reports/win_rate_lookups_v3.json for API inference.
    """
    stats = {}
    lookup_cols = {
        "BusinessSegment":     "segment_win_rate",
        "PropertyType":        "propertytype_win_rate",
        "PropertyState":       "state_win_rate",
        "SubType":             "subtype_win_rate",
        "Jobs_Office_Region":  "office_region_win_rate",
        "OfficeLocation":      "company_location_win_rate",
    }
    global_rate = float(train_df["Won"].mean())
    stats["global_win_rate"] = global_rate

    for col, key in lookup_cols.items():
        if col in train_df.columns:
            rates = train_df.groupby(col)["Won"].mean().to_dict()
            stats[key] = {str(k): round(float(v), 4) for k, v in rates.items()}
        else:
            stats[key] = {}

    # Fee percentile CDF per segment — the 10th/25th/50th/75th/90th percentiles
    # so the API can map fee → percentile using linear interpolation
    stats["segment_fee_cdf"] = {}
    if "BidFee" in train_df.columns:
        for seg, grp in train_df.groupby("BusinessSegment"):
            fees = grp["BidFee"].dropna().values
            if len(fees) >= 5:
                quantiles = np.percentile(fees, [10, 25, 50, 75, 90]).tolist()
                stats["segment_fee_cdf"][seg] = quantiles

    return stats


def select_features_v3(df: pd.DataFrame, exclude: list) -> list:
    """Select v3 feature columns."""
    exclude_set = set(exclude)
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()

    features = []
    for col in all_numeric:
        if col not in exclude_set and col != "Won":
            features.append(col)

    print(f"[Features] v3 feature count: {len(features)}")
    return features


# ============================================================================
# TRAIN / EVALUATE
# ============================================================================

def split_data(df: pd.DataFrame):
    """Time-based 60/20/20 split."""
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.60)
    val_end   = int(n * 0.80)
    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()
    print(f"\n[Split] Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    return train_df, val_df, test_df


def evaluate(y_true, y_pred_proba, label: str) -> dict:
    y_pred = (y_pred_proba >= 0.5).astype(int)
    return {
        "auc":      round(float(roc_auc_score(y_true, y_pred_proba)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "brier":    round(float(brier_score_loss(y_true, y_pred_proba)), 4),
        "n":        int(len(y_true)),
        "win_rate": round(float(y_true.mean()), 4),
    }


def fee_sensitivity_test(
    model: lgb.Booster,
    calibrator,
    feature_cols: list,
    base_features: dict,
    seg_avg: float,
    segment: str,
) -> None:
    """Print win prob at different fee multiples of segment average."""
    multiples = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    print(f"\n  Fee sensitivity for {segment} (seg_avg={seg_avg:,.0f}):")
    print(f"  {'Multiple':>8}  {'Fee':>8}  {'Win Prob':>10}")
    probs_collected = []
    for mult in multiples:
        fee = seg_avg * mult
        row = base_features.copy()
        row["BidFee"] = fee

        # Update fee-relative features
        row["fee_vs_segment_ratio"]   = fee / seg_avg
        row["fee_diff_from_segment"]  = fee - seg_avg
        row["bid_vs_state_ratio"]     = fee / seg_avg  # proxy
        row["bid_vs_client_ratio"]    = fee / seg_avg
        row["fee_percentile_segment"] = min(1.0, max(0.0, fee / (2 * seg_avg)))
        # empirical percentile — approximate with simple CDF approximation
        row["fee_percentile_empirical"] = min(1.0, max(0.0, fee / (2 * seg_avg)))

        X = np.array([[row.get(f, 0.0) for f in feature_cols]])
        raw = float(model.predict(X)[0])

        if calibrator is not None:
            try:
                prob = float(calibrator.predict([raw])[0])
            except Exception:
                prob = raw
        else:
            prob = raw

        prob = max(0.05, min(0.95, prob))
        probs_collected.append(prob)
        print(f"  {mult:>8.1f}×  ${fee:>7,.0f}  {prob*100:>9.1f}%")

    span_pp = (max(probs_collected) - min(probs_collected)) * 100 if probs_collected else 0.0
    print(f"  Curve span: {span_pp:.1f}pp  (healthy: >=20pp)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 65)
    print("FEE-SENSITIVE WIN PROBABILITY MODEL v3")
    print("=" * 65)

    # ---- Load data -------------------------------------------------------
    # NOTE: must read the FEATURE-ENGINEERED CSV (same source as v2 win prob,
    # script 25), not the raw enriched CSV. The processed/enriched_v2 file
    # has only raw columns; market-stat aggregates / fee ratios / client
    # history are added by script 23 and live in features/BidData_features_v2.csv.
    data_path = DATA_DIR / "features" / "BidData_features_v2.csv"
    print(f"\n[Data] Loading {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"[Data] Raw shape: {df.shape}")

    # Date filter
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.dropna(subset=[DATE_COLUMN])
    df = df[df[DATE_COLUMN] >= DATA_START_DATE]
    print(f"[Data] After {DATA_START_DATE} filter: {len(df):,} rows")

    # Target
    df["Won"] = pd.to_numeric(df["Won"], errors="coerce").fillna(0).astype(int)
    print(f"[Data] Win rate: {df['Won'].mean():.1%}  ({df['Won'].sum():,} wins)")

    # Fee anomaly filter (same as v2 — remove pro-bono/internal $1-$99 fees)
    original_count = len(df)
    df = df[df["BidFee"] >= 100].copy()
    removed = original_count - len(df)
    if removed > 0:
        print(f"[Data] Removed {removed:,} anomaly fees (<$100)")

    # Drop leaky job-derived features
    df.drop(columns=[c for c in LEAKY_JOB_DERIVED_FEATURES if c in df.columns], inplace=True)

    # ---- Feature engineering -------------------------------------------
    df = engineer_features(df)

    # ---- Split ---------------------------------------------------------
    train_df, val_df, test_df = split_data(df)

    # Build market-level win rate lookups from training data
    win_rate_stats = build_market_stats(train_df)
    lookup_path = REPORTS_DIR / "win_rate_lookups_v3.json"
    with open(lookup_path, "w") as f:
        json.dump(win_rate_stats, f, indent=2)
    print(f"\n[Stats] Win rate lookups saved: {lookup_path}")

    # ---- Feature selection ---------------------------------------------
    exclude = (
        list(EXCLUDE_COLUMNS)
        + list(LEAKY_JOB_DERIVED_FEATURES)
        + list(FREQUENCY_FEATURES_REPLACED)
        + list(PREEXISTING_LEAKY_RATE_COLS)
    )
    if DROP_CONTEXT_FEE_AGGREGATES:
        print(f"[Ablation] V3_DROP_CONTEXT_FEE=on — excluding {len(CONTEXT_FEE_AGGREGATES)} context-fee aggregates")
        exclude += list(CONTEXT_FEE_AGGREGATES)

    feature_cols = select_features_v3(df, exclude=exclude)

    # Confirm LOO win rate features are included
    print("\n[Features] LOO win rate features in feature set:")
    for f in feature_cols:
        if "win_rate" in f or "percentile_empirical" in f:
            print(f"  + {f}")

    print("\n[Features] Fee-relative features:")
    for f in feature_cols:
        if any(x in f for x in ["fee_vs", "bid_vs", "fee_diff", "percentile", "BidFee"]):
            print(f"  + {f}")

    # ---- Build datasets ------------------------------------------------
    def build_lgb_dataset(split_df, reference=None):
        X = split_df[feature_cols].fillna(0).values
        y = split_df["Won"].values
        if reference is None:
            return lgb.Dataset(X, label=y, feature_name=feature_cols)
        else:
            return lgb.Dataset(X, label=y, feature_name=feature_cols, reference=reference)

    train_set = build_lgb_dataset(train_df)
    val_set   = build_lgb_dataset(val_df, reference=train_set)

    # Class balance
    pos_weight = (train_df["Won"] == 0).sum() / max((train_df["Won"] == 1).sum(), 1)
    params = V3_PARAMS.copy()
    params["scale_pos_weight"] = round(float(pos_weight), 4)
    print(f"\n[Training] scale_pos_weight: {pos_weight:.3f}")

    # ---- Train ---------------------------------------------------------
    print(f"\n[Training] LightGBM v3 — {NUM_BOOST_ROUND} rounds, early stopping {EARLY_STOPPING}...")
    callbacks = [
        lgb.early_stopping(EARLY_STOPPING, verbose=False),
        lgb.log_evaluation(period=200),
    ]
    model = lgb.train(
        params,
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    print(f"[Training] Best iteration: {model.best_iteration}")

    # ---- Evaluate raw --------------------------------------------------
    X_train = train_df[feature_cols].fillna(0).values
    X_val   = val_df[feature_cols].fillna(0).values
    X_test  = test_df[feature_cols].fillna(0).values

    y_train_raw = model.predict(X_train)
    y_val_raw   = model.predict(X_val)
    y_test_raw  = model.predict(X_test)

    train_metrics = evaluate(train_df["Won"], y_train_raw, "train")
    val_metrics   = evaluate(val_df["Won"],   y_val_raw,   "val")
    test_metrics  = evaluate(test_df["Won"],  y_test_raw,  "test")

    print(f"\n[Results] Raw model:")
    print(f"  Train AUC: {train_metrics['auc']:.4f}  Brier: {train_metrics['brier']:.4f}")
    print(f"  Val   AUC: {val_metrics['auc']:.4f}  Brier: {val_metrics['brier']:.4f}")
    print(f"  Test  AUC: {test_metrics['auc']:.4f}  Brier: {test_metrics['brier']:.4f}")
    print(f"  Overfit ratio (train/test AUC): {train_metrics['auc']/max(test_metrics['auc'],1e-6):.2f}×")

    # ---- Calibrate -----------------------------------------------------
    print("\n[Calibration] Fitting isotonic regression on val set...")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_val_raw, val_df["Won"].values)

    y_test_cal  = calibrator.predict(y_test_raw)
    y_test_cal  = np.clip(y_test_cal, 0.05, 0.95)
    y_train_cal = calibrator.predict(y_train_raw)
    y_train_cal = np.clip(y_train_cal, 0.05, 0.95)

    cal_test  = evaluate(test_df["Won"],  y_test_cal,  "test_calibrated")
    cal_train = evaluate(train_df["Won"], y_train_cal, "train_calibrated")

    print(f"\n[Results] After calibration:")
    print(f"  Test  AUC: {cal_test['auc']:.4f}  Brier: {cal_test['brier']:.4f}")
    print(f"  v2 baseline: AUC=0.9475  Brier=0.0929")
    delta_auc = cal_test["auc"] - 0.9475
    print(f"  Delta vs v2: AUC {delta_auc:+.4f}")

    # ---- Feature importance --------------------------------------------
    importance = model.feature_importance(importance_type="gain")
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance,
        "importance_pct": 100 * importance / max(importance.sum(), 1e-6),
    }).sort_values("importance_pct", ascending=False).reset_index(drop=True)

    print(f"\n[Importance] Top 15 features:")
    for _, row in importance_df.head(15).iterrows():
        tag = " ← fee" if any(x in row["feature"] for x in ["fee_vs", "bid_vs", "fee_diff", "percentile", "BidFee"]) else ""
        tag = " ← win_rate" if "win_rate" in row["feature"] else tag
        print(f"  {row['importance_pct']:6.2f}%  {row['feature']}{tag}")

    fee_pct = importance_df[
        importance_df["feature"].str.contains("fee_vs|bid_vs|fee_diff|percentile|BidFee", regex=True)
    ]["importance_pct"].sum()
    rate_pct = importance_df[
        importance_df["feature"].str.contains("win_rate", regex=True)
    ]["importance_pct"].sum()
    print(f"\n  Fee-relative features: {fee_pct:.1f}%")
    print(f"  Win-rate features:     {rate_pct:.1f}%")
    print(f"  (v2 baseline: fee ~8%, frequency ~42%)")

    # ---- Segment-level AUC breakdown -----------------------------------
    print("\n[Segment AUC] Test AUC by business segment:")
    test_df_eval = test_df.copy()
    test_df_eval["pred"] = y_test_cal
    seg_auc_rows = []
    for seg, grp in test_df_eval.groupby("BusinessSegment"):
        if len(grp) < 50 or grp["Won"].nunique() < 2:
            continue
        try:
            seg_auc = roc_auc_score(grp["Won"], grp["pred"])
            seg_auc_rows.append((seg, seg_auc, len(grp), float(grp["Won"].mean())))
        except Exception:
            pass
    for seg, auc, n, wr in sorted(seg_auc_rows, key=lambda r: -r[2])[:10]:
        print(f"  {seg:<30s}  AUC={auc:.3f}  n={n:>5,}  win_rate={wr:.1%}")

    # ---- Fee sensitivity test ------------------------------------------
    print("\n[Sensitivity] Testing fee sensitivity curve...")
    # Build a representative base feature vector (median/mode values)
    base = {f: float(train_df[f].median()) for f in feature_cols if f in train_df.columns}
    base["BidFee"] = 0  # Will be overridden in the loop

    financing_seg_avg = float(train_df[train_df["BusinessSegment"] == "Financing"]["BidFee"].median())
    if financing_seg_avg > 0:
        # Override group-specific features for Financing / Multifamily context
        for f in feature_cols:
            if "segment_avg" in f:
                base[f] = financing_seg_avg
            if "segment_win_rate" in f:
                base[f] = float(win_rate_stats.get("segment_win_rate", {}).get("Financing", 0.45))
            if "propertytype_win_rate" in f:
                base[f] = float(win_rate_stats.get("propertytype_win_rate", {}).get("Multifamily", 0.45))
            if "subtype_win_rate" in f:
                base[f] = float(win_rate_stats.get("subtype_win_rate", {}).get("Conventional", 0.45))

        fee_sensitivity_test(model, calibrator, feature_cols, base, financing_seg_avg, "Financing")

    # ---- Fee sensitivity plot ------------------------------------------
    print("\n[Plot] Generating fee sensitivity comparison plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (seg_name, seg_label) in enumerate([("Financing", "Financing"), ("Consulting", "Consulting")]):
        seg_data = train_df[train_df["BusinessSegment"] == seg_name]
        if len(seg_data) < 10:
            continue
        seg_avg_fee = float(seg_data["BidFee"].median())

        base_seg = {f: float(seg_data[f].median()) for f in feature_cols if f in seg_data.columns}
        base_seg["BidFee"] = 0

        # Win rate feature overrides
        for wrf in ["segment_win_rate", "propertytype_win_rate", "subtype_win_rate"]:
            if wrf in feature_cols:
                base_seg[wrf] = float(win_rate_stats.get(wrf, {}).get(seg_name, win_rate_stats["global_win_rate"]))

        fees = np.linspace(seg_avg_fee * 0.3, seg_avg_fee * 2.5, 50)
        probs_v3 = []
        for fee in fees:
            row = base_seg.copy()
            row["BidFee"] = fee
            row["fee_vs_segment_ratio"] = fee / seg_avg_fee
            row["fee_diff_from_segment"] = fee - seg_avg_fee
            row["bid_vs_state_ratio"] = fee / seg_avg_fee
            row["bid_vs_client_ratio"] = fee / seg_avg_fee
            row["fee_percentile_segment"] = min(1.0, max(0.0, fee / (2 * seg_avg_fee)))
            row["fee_percentile_empirical"] = min(1.0, max(0.0, fee / (2 * seg_avg_fee)))

            X = np.array([[row.get(f, 0.0) for f in feature_cols]])
            raw = float(model.predict(X)[0])
            prob = float(calibrator.predict([raw])[0]) if calibrator else raw
            prob = max(0.05, min(0.95, prob))
            probs_v3.append(prob * 100)

        ax = axes[ax_idx]
        ax.plot(fees, probs_v3, "b-", linewidth=2, label="v3 (win rates)")
        ax.axvline(x=seg_avg_fee, color="gray", linestyle="--", alpha=0.6, label="Segment avg fee")
        ax.set_xlabel("Bid Fee ($)")
        ax.set_ylabel("Win Probability (%)")
        ax.set_title(f"Fee Sensitivity — {seg_label}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "win_probability_v3_fee_sensitivity.png"
    plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {fig_path}")

    # ---- Save model if AUC meets threshold -----------------------------
    AUC_THRESHOLD = 0.930   # Within ~2% of v2 baseline (0.9475)
    should_save = cal_test["auc"] >= AUC_THRESHOLD

    if should_save:
        print(f"\n[Save] AUC {cal_test['auc']:.4f} >= {AUC_THRESHOLD} threshold — SAVING as v3 model")

        model_path = MODELS_DIR / "lightgbm_win_probability_v3.txt"
        model.save_model(str(model_path))
        print(f"  Model: {model_path}")

        cal_path = MODELS_DIR / "win_probability_v3_calibrator.pkl"
        with open(cal_path, "wb") as f:
            pickle.dump(calibrator, f)
        print(f"  Calibrator: {cal_path}")

        imp_path = REPORTS_DIR / "win_probability_v3_feature_importance.csv"
        importance_df.to_csv(str(imp_path), index=False)
        print(f"  Importance: {imp_path}")

        meta = {
            "model_type": "LightGBM Classifier (v3 — LOO win rates)",
            "created_at": datetime.now().isoformat(),
            "features": feature_cols,
            "feature_count": len(feature_cols),
            "params": params,
            "best_iteration": int(model.best_iteration),
            "metrics": {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
                "test_calibrated": cal_test,
                "train_calibrated": cal_train,
            },
            "vs_v2": {
                "v2_test_auc": 0.9475,
                "v3_test_auc": cal_test["auc"],
                "delta_auc": round(float(delta_auc), 4),
            },
            "data": {
                "source": str(data_path),
                "start_date": DATA_START_DATE,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "global_win_rate": float(df["Won"].mean()),
            },
        }
        meta_path = MODELS_DIR / "lightgbm_win_probability_v3_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Metadata: {meta_path}")

        print(f"\n{'='*65}")
        print(f"  v3 model saved successfully")
        print(f"  AUC: {cal_test['auc']:.4f} (v2 baseline: 0.9475)")
        print(f"  To deploy: update enhanced_prediction_service.py to load")
        print(f"  lightgbm_win_probability_v3.txt and win_probability_v3_calibrator.pkl")
        print(f"  Also load win_rate_lookups_v3.json for API inference features.")
        print(f"{'='*65}")
    else:
        print(f"\n[Skip] AUC {cal_test['auc']:.4f} < {AUC_THRESHOLD} threshold — NOT saving as production model")
        print("  Results saved as experiment artifacts only (feature importance CSV + sensitivity plot).")
        imp_path = REPORTS_DIR / "win_probability_v3_feature_importance.csv"
        importance_df.to_csv(str(imp_path), index=False)

    # ---- Summary -------------------------------------------------------
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"  v2 baseline AUC: 0.9475")
    print(f"  v3 test AUC:     {cal_test['auc']:.4f}  ({delta_auc:+.4f} delta)")
    print(f"  Brier score:     {cal_test['brier']:.4f}  (v2: 0.0929)")
    print(f"  Fee features:    {fee_pct:.1f}%  (v2: ~8%)")
    print(f"  Win-rate feats:  {rate_pct:.1f}%  (v2 had ~0%)")
    print(f"\nNext step: run scripts/validate_end_to_end.py to compare curves")


if __name__ == "__main__":
    main()
