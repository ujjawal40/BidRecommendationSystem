"""
Enhanced Win Probability Model (Phase 1B v2) — LightGBM Classifier
=====================================================================
Trains on enriched BidData features with BidFee as a feature.
Compares to v1 baseline (AUC 0.870).

Output:
  - outputs/models/lightgbm_win_probability_v2.txt
  - outputs/models/lightgbm_win_probability_v2_metadata.json

Author: Ujjawal Dwivedi
Date: 2026-02-14
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
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config.model_config import (
    DATA_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
)

warnings.filterwarnings("ignore")

# ============================================================================
# V2 CLASSIFICATION CONFIG
# ============================================================================

V2_PARAMS = {
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
    "reg_alpha": 1.5,
    "reg_lambda": 1.5,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
    "is_unbalance": False,  # We'll set scale_pos_weight instead
}

NUM_BOOST_ROUND = 2000
EARLY_STOPPING = 100

# Date filtering
DATA_START_DATE = "2018-01-01"
DATE_COLUMN = "BidDate"

# Columns to exclude from features
EXCLUDE_COLUMNS = [
    "BidId", "BidFileNumber", "BidName",
    "BidDate", "Bid_DueDate",
    "BidFee",  # KEEP for win prob (injected as feature)
    "Won",  # Target
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
    # Jobs enrichment (categorical strings)
    "Jobs_SubType", "Jobs_SpecificUse", "Jobs_CompanyLocation",
    "Jobs_Office_Region", "Jobs_PropertyType", "Jobs_MarketOrientation",
]

# Leaky features to exclude (win_rate features derive from target)
LEAKY_CLASSIFICATION_FEATURES = [
    "segment_win_rate", "client_win_rate", "office_win_rate",
    "propertytype_win_rate", "state_win_rate",
    "cumulative_wins_client", "cumulative_winrate_client",
]

# Job-derived features that are NULL for lost bids → data leakage
LEAKY_JOB_DERIVED_FEATURES = [
    "JobCount", "IECount", "LeaseCount", "SaleCount",
    # Jobs_ enrichment IDs: populated for Won (from join), NaN for Lost → leakage
    "Jobs_PropertyID", "Jobs_OfficeID",
    "Jobs_PropertyType",  # redundant with BidData PropertyType
    # ALL Jobs_ numeric features with differential imputation quality:
    # Won bids get REAL values from JobsData join, Lost bids get MEDIAN fill.
    # Model trivially learns "real distribution → Won, median → Lost".
    "Jobs_GrossBuildingSF", "Jobs_GLARentableSF", "Jobs_GrossLandAreaAcres",
    "Jobs_YearBuilt", "Jobs_MarketOrientation",
    # Zip features also differentially imputed
    "Jobs_Zip_Population", "Jobs_Zip_PopDensity", "Jobs_Zip_HouseholdsPerZip",
    "Jobs_Zip_GrowthRank", "Jobs_Zip_AverageHouseValue", "Jobs_Zip_IncomePerHousehold",
    "Jobs_Zip_MedianAge", "Jobs_Zip_MedianIncome", "Jobs_Zip_NumberOfBusinesses",
    "Jobs_Zip_NumberOfEmployees", "Jobs_Zip_LandArea", "Jobs_Zip_PopulationEstimate",
    "Jobs_Zip_PopCount", "Jobs_Zip_DeliveryTotal", "Jobs_Zip_WorkersOutZip",
    # Derived features from differentially-imputed Jobs_ columns
    "land_acres_log",       # from Jobs_GrossLandAreaAcres
    "building_sf_log",      # from Jobs_GrossBuildingSF
    "fee_per_sqft",         # from Jobs_GrossBuildingSF
    "fee_per_day",          # from Jobs_JobLength_Days
    "joblength_log",        # from Jobs_JobLength_Days
    "joblength_bucket",     # from Jobs_JobLength_Days
    "joblength_x_segment_fee",  # from Jobs_JobLength_Days
    "Jobs_JobLength_Days",  # median by office for lost, real for won
    "income_x_segment_fee", # from Jobs_Zip_IncomePerHousehold
    "pop_density_log",      # from Jobs_Zip_PopDensity
]


def main():
    print("=" * 80)
    print("ENHANCED WIN PROBABILITY MODEL — Phase 1B v2")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ========================================================================
    # LOAD DATA
    # ========================================================================
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    df = pd.read_csv(DATA_DIR / "features" / "BidData_features_v2.csv", low_memory=False)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    # Filter to recent
    start = pd.Timestamp(DATA_START_DATE)
    df = df[df[DATE_COLUMN] >= start].copy()
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    print(f"  Rows: {len(df):,}")
    print(f"  Date range: {df[DATE_COLUMN].min()} to {df[DATE_COLUMN].max()}")

    # ========================================================================
    # PREPARE FEATURES
    # ========================================================================
    print("\n" + "=" * 80)
    print("PREPARING FEATURES")
    print("=" * 80)

    # Build exclusion list (keep BidFee!)
    exclude = set(EXCLUDE_COLUMNS + LEAKY_CLASSIFICATION_FEATURES + LEAKY_JOB_DERIVED_FEATURES)
    # Make sure BidFee is NOT excluded
    exclude.discard("BidFee")

    feature_cols = [c for c in df.columns if c not in exclude]
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # Ensure BidFee is in features
    if "BidFee" not in numeric_features:
        numeric_features.append("BidFee")

    X = df[numeric_features].fillna(0).copy()
    y = df["Won"].values

    win_rate = y.mean()
    print(f"  Features: {len(numeric_features)}")
    print(f"  Win rate: {win_rate:.3f}")
    print(f"  BidFee in features: {'BidFee' in numeric_features}")

    # ========================================================================
    # TIME-BASED SPLIT
    # ========================================================================
    print("\n" + "=" * 80)
    print("TIME-BASED SPLIT (60/20/20)")
    print("=" * 80)

    n = len(X)
    train_idx = int(n * 0.6)
    valid_idx = int(n * 0.8)

    X_train, X_valid, X_test = X.iloc[:train_idx], X.iloc[train_idx:valid_idx], X.iloc[valid_idx:]
    y_train, y_valid, y_test = y[:train_idx], y[train_idx:valid_idx], y[valid_idx:]

    print(f"  Train: {len(X_train):,} (win rate: {y_train.mean():.3f})")
    print(f"  Valid: {len(X_valid):,} (win rate: {y_valid.mean():.3f})")
    print(f"  Test:  {len(X_test):,} (win rate: {y_test.mean():.3f})")

    # ========================================================================
    # TRAIN
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING MODEL")
    print("=" * 80)

    # Set class weight
    params = V2_PARAMS.copy()
    scale_pos_weight = (1 - y_train.mean()) / y_train.mean()
    params["scale_pos_weight"] = scale_pos_weight
    print(f"  scale_pos_weight: {scale_pos_weight:.3f}")

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=numeric_features)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING),
            lgb.log_evaluation(200),
        ],
    )

    print(f"\n  Best iteration: {model.best_iteration}")

    # ========================================================================
    # EVALUATE
    # ========================================================================
    print("\n" + "=" * 80)
    print("EVALUATING MODEL")
    print("=" * 80)

    all_metrics = {}
    for name, X_split, y_split in [
        ("train", X_train, y_train),
        ("valid", X_valid, y_valid),
        ("test", X_test, y_test),
    ]:
        proba = model.predict(X_split, num_iteration=model.best_iteration)
        preds = (proba >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_split, preds)),
            "auc_roc": float(roc_auc_score(y_split, proba)),
            "precision": float(precision_score(y_split, preds)),
            "recall": float(recall_score(y_split, preds)),
            "f1": float(f1_score(y_split, preds)),
        }
        all_metrics[name] = metrics

        print(f"\n  {name.upper()}:")
        print(f"    AUC-ROC:   {metrics['auc_roc']:.4f}")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1:        {metrics['f1']:.4f}")

    # Overfitting
    overfit = all_metrics["train"]["auc_roc"] / all_metrics["test"]["auc_roc"]
    print(f"\n  Overfitting ratio (AUC): {overfit:.2f}× (target < 1.15×)")
    all_metrics["overfitting_ratio"] = float(overfit)

    # Brier score
    y_test_proba = model.predict(X_test, num_iteration=model.best_iteration)
    brier = brier_score_loss(y_test, y_test_proba)
    print(f"  Brier score: {brier:.4f}")
    all_metrics["brier_score"] = float(brier)

    # Confusion matrix
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    print(f"\n  Confusion Matrix:")
    print(f"    TP={tp:,}  FP={fp:,}")
    print(f"    FN={fn:,}  TN={tn:,}")

    # ========================================================================
    # CALIBRATION (Isotonic Regression)
    # ========================================================================
    print("\n" + "=" * 80)
    print("CALIBRATING PROBABILITIES (Isotonic Regression)")
    print("=" * 80)

    # Fit isotonic regression on validation set
    y_valid_proba = model.predict(X_valid, num_iteration=model.best_iteration)
    calibrator = IsotonicRegression(y_min=0.05, y_max=0.95, out_of_bounds="clip")
    calibrator.fit(y_valid_proba, y_valid)

    # Compare uncalibrated vs calibrated on test set
    y_test_raw = model.predict(X_test, num_iteration=model.best_iteration)
    y_test_calibrated = calibrator.predict(y_test_raw)

    brier_raw = brier_score_loss(y_test, y_test_raw)
    brier_cal = brier_score_loss(y_test, y_test_calibrated)

    print(f"  Brier score (raw):        {brier_raw:.4f}")
    print(f"  Brier score (calibrated): {brier_cal:.4f}")
    print(f"  Improvement:              {(brier_raw - brier_cal) / brier_raw * 100:.1f}%")

    # Calibration bins comparison
    print(f"\n  Calibration comparison (test set):")
    print(f"    {'Bin':<12} {'Count':>6} {'Actual':>8} {'Raw':>8} {'Calibr.':>8} {'Raw Gap':>8} {'Cal Gap':>8}")
    print("    " + "-" * 60)
    for lo, hi, label in [(0, 0.2, "0-20%"), (0.2, 0.4, "20-40%"), (0.4, 0.6, "40-60%"),
                           (0.6, 0.8, "60-80%"), (0.8, 1.0, "80-100%")]:
        mask = (y_test_raw >= lo) & (y_test_raw < hi)
        if mask.sum() == 0:
            continue
        actual = y_test[mask].mean() * 100
        raw_avg = y_test_raw[mask].mean() * 100
        cal_avg = y_test_calibrated[mask].mean() * 100
        print(f"    {label:<12} {mask.sum():>6,} {actual:>7.1f}% {raw_avg:>7.1f}% {cal_avg:>7.1f}% "
              f"{actual - raw_avg:>+7.1f}% {actual - cal_avg:>+7.1f}%")

    # Save calibrator
    calibrator_path = MODELS_DIR / "win_probability_v2_calibrator.pkl"
    with open(calibrator_path, "wb") as f:
        pickle.dump(calibrator, f)
    print(f"\n  Calibrator saved: {calibrator_path}")

    # ========================================================================
    # FEATURE IMPORTANCE
    # ========================================================================
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)

    importance_df = pd.DataFrame({
        "feature": numeric_features,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    importance_df["importance_pct"] = (
        importance_df["importance"] / importance_df["importance"].sum() * 100
    )

    print(f"\n  Top 20:")
    for _, row in importance_df.head(20).iterrows():
        print(f"    {row['feature']:40s} {row['importance_pct']:6.2f}%")

    importance_df.to_csv(REPORTS_DIR / "win_probability_v2_feature_importance.csv", index=False)

    # ========================================================================
    # PLOTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING PLOTS")
    print("=" * 80)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    axes[0].plot(fpr, tpr, "b-", label=f"AUC = {all_metrics['test']['auc_roc']:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Calibration
    prob_true, prob_pred = calibration_curve(y_test, y_test_proba, n_bins=10)
    axes[1].plot(prob_pred, prob_true, "bo-", label="Model")
    axes[1].plot([0, 1], [0, 1], "k--", label="Perfect")
    axes[1].set_xlabel("Predicted Probability")
    axes[1].set_ylabel("Actual Win Rate")
    axes[1].set_title("Calibration Plot")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Feature importance
    top = importance_df.head(15)
    axes[2].barh(range(len(top)), top["importance_pct"])
    axes[2].set_yticks(range(len(top)))
    axes[2].set_yticklabels(top["feature"], fontsize=8)
    axes[2].set_xlabel("Importance (%)")
    axes[2].set_title("Top 15 Features")
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "win_probability_v2_evaluation.png", dpi=150)
    plt.close()
    print(f"  Plots saved")

    # ========================================================================
    # SAVE MODEL
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)

    model_path = MODELS_DIR / "lightgbm_win_probability_v2.txt"
    model.save_model(str(model_path))
    print(f"  Model: {model_path}")

    metadata = {
        "model_type": "LightGBM Binary Classifier",
        "version": "v2",
        "phase": "1B - Win Probability Prediction (Enhanced)",
        "target_variable": "Won",
        "data_source": "BidData enriched with JobsData (2018+)",
        "num_features": len(numeric_features),
        "features": numeric_features,
        "bidfee_included": True,
        "excluded_leaky_features": LEAKY_CLASSIFICATION_FEATURES,
        "excluded_job_derived_features": LEAKY_JOB_DERIVED_FEATURES,
        "best_iteration": int(model.best_iteration),
        "hyperparameters": params,
        "metrics": all_metrics,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "calibration": {
            "brier_score_raw": float(brier_raw),
            "brier_score_calibrated": float(brier_cal),
            "method": "isotonic_regression",
            "calibrator_file": "win_probability_v2_calibrator.pkl",
        },
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    metadata_path = MODELS_DIR / "lightgbm_win_probability_v2_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Metadata: {metadata_path}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1B v2 COMPLETE")
    print("=" * 80)

    test = all_metrics["test"]
    print(f"\n  Test AUC-ROC:  {test['auc_roc']:.4f}")
    print(f"  Test Accuracy: {test['accuracy']:.4f}")
    print(f"  Test F1:       {test['f1']:.4f}")
    print(f"  Brier Score:   {brier:.4f}")
    print(f"  Overfit:       {overfit:.2f}×")

    # Compare to v1
    print(f"\n  v1 baseline: AUC=0.870, Accuracy=0.790, Overfit=1.09×")
    auc_diff = test["auc_roc"] - 0.870
    print(f"  v2 vs v1 AUC: {auc_diff:+.4f}")

    return model, all_metrics


if __name__ == "__main__":
    model, metrics = main()
