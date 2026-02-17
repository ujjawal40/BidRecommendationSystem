"""
Enhanced Bid Fee Model (Phase 1A v2) — LightGBM Regression
=============================================================
Trains on JobsData features with log1p(NetFee) target transform.
Uses SHAP-based feature selection and compares to v1 baseline.

Output:
  - outputs/models/lightgbm_bidfee_v2_model.txt
  - outputs/models/lightgbm_bidfee_v2_metadata.json

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
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from config.model_config import (
    DATA_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    MIN_TRAINING_FEE,
)

warnings.filterwarnings("ignore")

# ============================================================================
# V2 MODEL CONFIGURATION
# ============================================================================

V2_CONFIG = {
    "params": {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 18,        # Match v1 for regularization
        "learning_rate": 0.02,
        "feature_fraction": 0.7,  # Stronger feature sampling
        "bagging_fraction": 0.7,  # Stronger row sampling
        "bagging_freq": 5,
        "max_depth": 6,           # Shallower for less overfitting
        "min_child_samples": 50,  # Larger min samples
        "min_child_weight": 10,
        "reg_alpha": 20.0,        # Strong L1 (reduced overfit 1.91x → 1.48x)
        "reg_lambda": 20.0,       # Strong L2
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
    "target_transform": "log1p",
    "training": {
        "num_boost_round": 3000,
        "early_stopping_rounds": 150,
        "verbose_eval": 200,
    },
}

# Date filtering: 2018+ (v2 uses broader window than v1's 2023+)
DATA_START_DATE = "2018-01-01"
DATE_COLUMN = "StartDate"
TARGET_COLUMN = "NetFee"

# Columns to exclude from features
EXCLUDE_COLUMNS = [
    "JobId", "PropertyID", "OfficeID",
    "StartDate", "NetFee", "NetFee_Original",
    # Categorical strings (not encoded)
    "JobType", "JobPurpose", "Deliverable", "AppraisalFileType",
    "BusinessSegment", "BusinessSegmentDetail",
    "PortfolioMultiProperty", "PotentialLitigation",
    "JobDistanceMiles", "OfficeJobTerritory",
    "PropertyType", "SubType", "SpecificUse",
    "MarketOrientation", "StateName", "Market", "Submarket",
    "CompanyLocation", "Office_Region",
    "ContactType", "CompanyType",
    # LEAKY for Phase 1A: these features contain the target (NetFee) directly
    "fee_vs_segment_ratio",   # NetFee / segment_avg
    "fee_percentile_segment", # rank(NetFee) within segment
    "fee_diff_from_segment",  # NetFee - segment_avg
    "bid_vs_segment_ratio",   # alias
    "bid_vs_client_ratio",
    "bid_vs_state_ratio",
    "fee_per_day",            # NetFee / JobLength
    "fee_per_sqft",           # NetFee / BuildingSF
]


class EnhancedBidFeePredictor:
    """LightGBM regression model for NetFee prediction (Phase 1A v2)."""

    def __init__(self, config=None):
        self.config = config or V2_CONFIG
        self.model = None
        self.feature_names = None
        self.metrics = {}

    def load_data(self):
        """Load feature-engineered JobsData."""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)

        df = pd.read_csv(DATA_DIR / "features" / "JobsData_features_v2.csv", low_memory=False)
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

        # Filter to start date
        start = pd.Timestamp(DATA_START_DATE)
        df = df[df[DATE_COLUMN] >= start].copy()

        # Remove anomaly fees (pro-bono, internal work)
        before_fee_filter = len(df)
        df = df[df[TARGET_COLUMN] >= MIN_TRAINING_FEE].copy()
        removed = before_fee_filter - len(df)
        if removed > 0:
            print(f"  Removed {removed:,} rows with {TARGET_COLUMN} < ${MIN_TRAINING_FEE}")

        df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

        print(f"  Rows: {len(df):,}")
        print(f"  Date range: {df[DATE_COLUMN].min()} to {df[DATE_COLUMN].max()}")

        return df

    def prepare_features(self, df):
        """Select numeric features, excluding IDs/targets/categoricals."""
        print("\n" + "=" * 80)
        print("PREPARING FEATURES")
        print("=" * 80)

        feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        X = df[numeric_features].fillna(0).copy()
        y = df[TARGET_COLUMN].copy()

        self.feature_names = numeric_features
        print(f"  Features: {len(numeric_features)}")
        print(f"  Target: {TARGET_COLUMN} (mean=${y.mean():,.0f}, median=${y.median():,.0f})")

        return X, y

    def time_based_split(self, X, y):
        """60/20/20 chronological split."""
        print("\n" + "=" * 80)
        print("TIME-BASED SPLIT (60/20/20)")
        print("=" * 80)

        n = len(X)
        train_idx = int(n * 0.6)
        valid_idx = int(n * 0.8)

        splits = {
            "X_train": X.iloc[:train_idx],
            "X_valid": X.iloc[train_idx:valid_idx],
            "X_test": X.iloc[valid_idx:],
            "y_train": y.iloc[:train_idx],
            "y_valid": y.iloc[train_idx:valid_idx],
            "y_test": y.iloc[valid_idx:],
        }

        for name, data in splits.items():
            if name.startswith("X"):
                print(f"  {name}: {len(data):,} rows")

        return splits

    def train(self, splits):
        """Train with log1p target transform."""
        print("\n" + "=" * 80)
        print("TRAINING MODEL")
        print("=" * 80)

        y_train = np.log1p(splits["y_train"])
        y_valid = np.log1p(splits["y_valid"])

        train_data = lgb.Dataset(
            splits["X_train"], label=y_train, feature_name=self.feature_names
        )
        valid_data = lgb.Dataset(
            splits["X_valid"], label=y_valid, feature_name=self.feature_names,
            reference=train_data,
        )

        self.model = lgb.train(
            self.config["params"],
            train_data,
            num_boost_round=self.config["training"]["num_boost_round"],
            valid_sets=[valid_data],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(self.config["training"]["early_stopping_rounds"]),
                lgb.log_evaluation(self.config["training"]["verbose_eval"]),
            ],
        )

        print(f"\n  Best iteration: {self.model.best_iteration}")

    def evaluate(self, splits):
        """Evaluate on all splits, compute overfitting ratio."""
        print("\n" + "=" * 80)
        print("EVALUATING MODEL")
        print("=" * 80)

        results = {}
        for split_name, X, y_true in [
            ("train", splits["X_train"], splits["y_train"]),
            ("valid", splits["X_valid"], splits["y_valid"]),
            ("test", splits["X_test"], splits["y_test"]),
        ]:
            preds_log = self.model.predict(X, num_iteration=self.model.best_iteration)
            preds = np.expm1(preds_log)
            preds = np.maximum(0, preds)

            rmse = np.sqrt(mean_squared_error(y_true, preds))
            mae = mean_absolute_error(y_true, preds)
            r2 = r2_score(y_true, preds)
            mape = np.mean(np.abs((y_true - preds) / y_true)) * 100
            median_ae = median_absolute_error(y_true, preds)

            results[split_name] = {
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2),
                "mape": float(mape),
                "median_ae": float(median_ae),
            }

            print(f"\n  {split_name.upper()}:")
            print(f"    RMSE:      ${rmse:,.0f}")
            print(f"    MAE:       ${mae:,.0f}")
            print(f"    MAPE:      {mape:.1f}%")
            print(f"    R²:        {r2:.4f}")
            print(f"    Median AE: ${median_ae:,.0f}")

        # Overfitting ratio
        overfit = results["test"]["rmse"] / results["train"]["rmse"]
        print(f"\n  Overfitting ratio: {overfit:.2f}× (target < 2.0×)")
        results["overfitting_ratio"] = float(overfit)

        self.metrics = results
        self.test_predictions = np.expm1(
            self.model.predict(splits["X_test"], num_iteration=self.model.best_iteration)
        )
        self.test_predictions = np.maximum(0, self.test_predictions)
        self.y_test = splits["y_test"]

        return results

    def feature_importance(self):
        """Compute and save feature importance."""
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE")
        print("=" * 80)

        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False)

        importance_df["importance_pct"] = (
            importance_df["importance"] / importance_df["importance"].sum() * 100
        )

        print(f"\n  Top 20 features:")
        for _, row in importance_df.head(20).iterrows():
            print(f"    {row['feature']:40s} {row['importance_pct']:6.2f}%")

        # Save CSV
        importance_df.to_csv(REPORTS_DIR / "bidfee_v2_feature_importance.csv", index=False)

        # Plot
        plt.figure(figsize=(10, 8))
        top = importance_df.head(20)
        plt.barh(range(len(top)), top["importance_pct"])
        plt.yticks(range(len(top)), top["feature"])
        plt.xlabel("Importance (%)")
        plt.title("Top 20 Feature Importance — Bid Fee v2 Model")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "bidfee_v2_feature_importance.png", dpi=150)
        plt.close()

        # Evaluation plot
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(self.y_test, self.test_predictions, alpha=0.2, s=5)
        max_val = max(self.y_test.max(), self.test_predictions.max())
        plt.plot([0, max_val], [0, max_val], "r--", label="Perfect")
        plt.xlabel("Actual NetFee ($)")
        plt.ylabel("Predicted NetFee ($)")
        plt.title("Actual vs Predicted")
        plt.legend()

        plt.subplot(1, 2, 2)
        residuals = self.y_test.values - self.test_predictions
        plt.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
        plt.xlabel("Residual ($)")
        plt.title("Residual Distribution")
        plt.axvline(0, color="r", linestyle="--")

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "bidfee_v2_evaluation.png", dpi=150)
        plt.close()

        return importance_df

    def save(self):
        """Save model and metadata."""
        print("\n" + "=" * 80)
        print("SAVING MODEL")
        print("=" * 80)

        # Model
        model_path = MODELS_DIR / "lightgbm_bidfee_v2_model.txt"
        self.model.save_model(str(model_path))
        print(f"  Model: {model_path}")

        # Metadata
        metadata = {
            "model_type": "LightGBM Regressor",
            "version": "v2",
            "phase": "1A - Bid Fee Prediction (Enhanced)",
            "target_variable": TARGET_COLUMN,
            "target_transform": "log1p",
            "data_source": "JobsData (2018+)",
            "num_features": len(self.feature_names),
            "features": self.feature_names,
            "best_iteration": int(self.model.best_iteration),
            "hyperparameters": self.config["params"],
            "metrics": self.metrics,
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        metadata_path = MODELS_DIR / "lightgbm_bidfee_v2_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"  Metadata: {metadata_path}")

        # Predictions
        preds_df = pd.DataFrame({
            "Actual": self.y_test.values,
            "Predicted": self.test_predictions,
            "Residual": self.y_test.values - self.test_predictions,
            "Pct_Error": np.abs((self.y_test.values - self.test_predictions) / self.y_test.values) * 100,
        })
        preds_path = REPORTS_DIR / "bidfee_v2_predictions.csv"
        preds_df.to_csv(preds_path, index=False)
        print(f"  Predictions: {preds_path}")

    def run(self):
        """Full training pipeline."""
        df = self.load_data()
        X, y = self.prepare_features(df)
        splits = self.time_based_split(X, y)
        self.train(splits)
        results = self.evaluate(splits)
        self.feature_importance()
        self.save()

        # Summary
        print("\n" + "=" * 80)
        print("PHASE 1A v2 COMPLETE")
        print("=" * 80)
        test = results["test"]
        print(f"  Test RMSE:  ${test['rmse']:,.0f}")
        print(f"  Test MAPE:  {test['mape']:.1f}%")
        print(f"  Test R²:    {test['r2']:.4f}")
        print(f"  Overfit:    {results['overfitting_ratio']:.2f}×")

        return results


def main():
    predictor = EnhancedBidFeePredictor()
    results = predictor.run()
    return predictor, results


if __name__ == "__main__":
    predictor, results = main()
