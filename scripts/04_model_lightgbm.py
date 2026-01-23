"""
LightGBM Regression Model for Bid Fee Prediction
=================================================
Phase 1A: Predict optimal bid fees for commercial real estate appraisals

This script implements a LightGBM gradient boosting model to predict BidFee
based on engineered features including rolling averages, client history,
property characteristics, and market conditions.

Key Features:
- Time-aware train/test split (no data leakage)
- Feature importance analysis
- SHAP explainability
- Comprehensive evaluation metrics
- Model persistence

Author: Bid Recommendation System
Date: 2026-01-06
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
import shap

# Import configuration
from config.model_config import (
    FEATURES_DATA,
    TARGET_COLUMN,
    DATE_COLUMN,
    EXCLUDE_COLUMNS,
    TRAIN_TEST_SPLIT,
    LIGHTGBM_CONFIG,
    EVALUATION_METRICS,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    RANDOM_SEED,
    DATA_START_DATE,
    USE_RECENT_DATA_ONLY,
    JOBDATA_FEATURES_TO_EXCLUDE,
)


class LightGBMBidFeePredictor:
    """
    LightGBM model for predicting commercial real estate appraisal bid fees.

    This class handles the complete ML pipeline:
    - Data loading and preprocessing
    - Time-aware train/test splitting
    - Model training with early stopping
    - Feature importance analysis
    - SHAP explainability
    - Model evaluation and persistence
    """

    def __init__(self, config=None):
        """
        Initialize the LightGBM predictor.

        Parameters
        ----------
        config : dict, optional
            Model configuration dictionary. If None, uses default LIGHTGBM_CONFIG.
        """
        self.config = config or LIGHTGBM_CONFIG
        self.model = None
        self.feature_names = None
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predictions = None
        self.metrics = {}

        print("=" * 80)
        print("LIGHTGBM BID FEE PREDICTOR - PHASE 1A")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    def load_data(self, data_path=None):
        """
        Load feature-engineered dataset.

        Parameters
        ----------
        data_path : str or Path, optional
            Path to feature-engineered CSV file. If None, uses FEATURES_DATA from config.

        Returns
        -------
        pd.DataFrame
            Loaded dataset
        """
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)

        data_path = data_path or FEATURES_DATA
        df = pd.read_csv(data_path)

        print(f"✓ Data loaded from: {data_path}")
        print(f"  Rows: {df.shape[0]:,}")
        print(f"  Columns: {df.shape[1]:,}")

        # Convert BidDate to datetime
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

        # Filter to recent data if configured (improves generalization)
        if USE_RECENT_DATA_ONLY:
            original_count = len(df)
            start_date = pd.Timestamp(DATA_START_DATE)
            df = df[df[DATE_COLUMN] >= start_date].copy()
            print(f"✓ Filtered to {DATA_START_DATE}+ data")
            print(f"  Removed: {original_count - len(df):,} older records")
            print(f"  Remaining: {len(df):,} records")

        # Sort by date for time-aware split
        df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
        print(f"✓ Data sorted by {DATE_COLUMN}")
        print(f"  Date range: {df[DATE_COLUMN].min()} to {df[DATE_COLUMN].max()}\n")

        return df

    def prepare_features(self, df):
        """
        Prepare features and target variable.

        Excludes non-feature columns (IDs, dates, targets) and handles missing values.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset

        Returns
        -------
        tuple
            (X, y) feature matrix and target vector
        """
        print("=" * 80)
        print("PREPARING FEATURES")
        print("=" * 80)

        # Identify feature columns (exclude IDs, targets, dates)
        feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]

        # Exclude JobData features (they degrade performance - see documentation)
        jobdata_excluded = [col for col in feature_cols if col in JOBDATA_FEATURES_TO_EXCLUDE]
        feature_cols = [col for col in feature_cols if col not in JOBDATA_FEATURES_TO_EXCLUDE]

        if jobdata_excluded:
            print(f"✓ Excluded {len(jobdata_excluded)} JobData features (degrade performance)")

        # Remove any remaining non-numeric columns
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        print(f"Total columns: {df.shape[1]}")
        print(f"Excluded columns: {len(EXCLUDE_COLUMNS)}")
        print(f"JobData features excluded: {len(jobdata_excluded)}")
        print(f"Feature columns: {len(numeric_features)}")

        # Extract features and target
        X = df[numeric_features].copy()
        y = df[TARGET_COLUMN].copy()

        # Check for missing values
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            print(f"\n⚠ Warning: {missing_count} missing values found in features")
            print("  Filling with 0...")
            X = X.fillna(0)

        self.feature_names = numeric_features

        print(f"\n✓ Features prepared")
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Target variable: {TARGET_COLUMN}")
        print(f"  Target shape: {y.shape}")
        print(f"  Target mean: ${y.mean():,.2f}")
        print(f"  Target std: ${y.std():,.2f}\n")

        return X, y

    def time_based_split(self, df, X, y):
        """
        Perform time-based train/valid/test split to prevent data leakage.

        Uses 60/20/20 split for proper validation during training:
        - Train (60%): Model fitting
        - Valid (20%): Early stopping and hyperparameter tuning
        - Test (20%): Final unbiased evaluation

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset with date column
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable

        Returns
        -------
        None
            Sets self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test
        """
        print("=" * 80)
        print("TIME-BASED TRAIN/VALID/TEST SPLIT")
        print("=" * 80)

        # 60/20/20 split for train/valid/test
        n = len(df)
        train_idx = int(n * 0.6)
        valid_idx = int(n * 0.8)

        # Split data chronologically
        self.X_train = X.iloc[:train_idx].copy()
        self.X_valid = X.iloc[train_idx:valid_idx].copy()
        self.X_test = X.iloc[valid_idx:].copy()
        self.y_train = y.iloc[:train_idx].copy()
        self.y_valid = y.iloc[train_idx:valid_idx].copy()
        self.y_test = y.iloc[valid_idx:].copy()

        # Get date ranges
        train_dates = df[DATE_COLUMN].iloc[:train_idx]
        valid_dates = df[DATE_COLUMN].iloc[train_idx:valid_idx]
        test_dates = df[DATE_COLUMN].iloc[valid_idx:]

        print(f"Split ratio: 60% train / 20% valid / 20% test")
        print(f"\nTraining set:")
        print(f"  Rows: {len(self.X_train):,}")
        print(f"  Date range: {train_dates.min()} to {train_dates.max()}")
        print(f"  Target mean: ${self.y_train.mean():,.2f}")

        print(f"\nValidation set (for early stopping):")
        print(f"  Rows: {len(self.X_valid):,}")
        print(f"  Date range: {valid_dates.min()} to {valid_dates.max()}")
        print(f"  Target mean: ${self.y_valid.mean():,.2f}")

        print(f"\nTest set (final evaluation):")
        print(f"  Rows: {len(self.X_test):,}")
        print(f"  Date range: {test_dates.min()} to {test_dates.max()}")
        print(f"  Target mean: ${self.y_test.mean():,.2f}\n")

    def train_model(self):
        """
        Train LightGBM model with early stopping on validation set.

        IMPORTANT: Uses validation set (not test set) for early stopping.
        Test set is reserved for final unbiased evaluation only.

        Returns
        -------
        lgb.Booster
            Trained LightGBM model
        """
        print("=" * 80)
        print("TRAINING LIGHTGBM MODEL")
        print("=" * 80)

        # Create LightGBM datasets
        train_data = lgb.Dataset(
            self.X_train,
            label=self.y_train,
            feature_name=self.feature_names,
        )

        # Use VALIDATION set for early stopping (NOT test set!)
        valid_data = lgb.Dataset(
            self.X_valid,
            label=self.y_valid,
            feature_name=self.feature_names,
            reference=train_data,
        )

        # Training parameters
        params = self.config["params"]
        num_boost_round = self.config["training"]["num_boost_round"]
        early_stopping_rounds = self.config["training"]["early_stopping_rounds"]
        verbose_eval = self.config["training"]["verbose_eval"]

        print("Training configuration:")
        print(f"  Boosting rounds: {num_boost_round}")
        print(f"  Early stopping: {early_stopping_rounds} rounds")
        print(f"  Learning rate: {params['learning_rate']}")
        print(f"  Num leaves: {params['num_leaves']}")
        print(f"  Max depth: {params['max_depth']}")
        print(f"  Regularization: L1={params['reg_alpha']}, L2={params['reg_lambda']}")
        print("\nTraining in progress (early stopping on validation set)...\n")

        # Train model
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=verbose_eval),
        ]

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, valid_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        print(f"\n✓ Model training complete")
        print(f"  Best iteration: {self.model.best_iteration}")
        print(f"  Best score: {self.model.best_score['valid']['rmse']:.2f}\n")

        return self.model

    def evaluate_model(self):
        """
        Evaluate model performance on train, validation, and test sets.

        Calculates metrics for all three sets to assess overfitting:
        - RMSE (Root Mean Squared Error)
        - MAE (Mean Absolute Error)
        - R² (R-squared)
        - Overfitting Ratio (Test RMSE / Train RMSE)

        Returns
        -------
        dict
            Dictionary of evaluation metrics
        """
        print("=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)

        # Generate predictions for all sets
        pred_train = self.model.predict(self.X_train, num_iteration=self.model.best_iteration)
        pred_valid = self.model.predict(self.X_valid, num_iteration=self.model.best_iteration)
        self.predictions = self.model.predict(self.X_test, num_iteration=self.model.best_iteration)

        # Calculate metrics for all sets
        train_rmse = np.sqrt(mean_squared_error(self.y_train, pred_train))
        valid_rmse = np.sqrt(mean_squared_error(self.y_valid, pred_valid))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, self.predictions))

        test_mae = mean_absolute_error(self.y_test, self.predictions)
        test_r2 = r2_score(self.y_test, self.predictions)
        test_mape = np.mean(np.abs((self.y_test - self.predictions) / self.y_test)) * 100
        test_median_ae = median_absolute_error(self.y_test, self.predictions)

        # Overfitting ratio (key metric for generalization)
        overfitting_ratio = test_rmse / train_rmse

        self.metrics = {
            "train_rmse": train_rmse,
            "valid_rmse": valid_rmse,
            "test_rmse": test_rmse,
            "rmse": test_rmse,  # For backward compatibility
            "mae": test_mae,
            "r2": test_r2,
            "mape": test_mape,
            "median_ae": test_median_ae,
            "overfitting_ratio": overfitting_ratio,
        }

        # Display results with overfitting assessment
        print("\nPerformance Summary:")
        print(f"  {'Set':<12} {'RMSE':<12} {'Assessment'}")
        print(f"  {'-'*40}")
        print(f"  {'Train':<12} ${train_rmse:>8,.2f}    (fitting)")
        print(f"  {'Valid':<12} ${valid_rmse:>8,.2f}    (tuning)")
        print(f"  {'Test':<12} ${test_rmse:>8,.2f}    (final)")

        print(f"\nOverfitting Analysis:")
        print(f"  Overfitting Ratio: {overfitting_ratio:.2f}x (Test/Train)")
        if overfitting_ratio < 1.5:
            print(f"  Assessment: ✅ Excellent generalization")
        elif overfitting_ratio < 2.0:
            print(f"  Assessment: ✅ Good generalization")
        elif overfitting_ratio < 2.5:
            print(f"  Assessment: ⚠️ Moderate overfitting")
        else:
            print(f"  Assessment: ❌ Severe overfitting")

        print(f"\nTest Set Detailed Metrics:")
        print(f"  RMSE: ${test_rmse:,.2f}")
        print(f"  MAE: ${test_mae:,.2f}")
        print(f"  R²: {test_r2:.4f}")
        print(f"  MAPE: {test_mape:.2f}%")
        print(f"  Median AE: ${test_median_ae:,.2f}\n")

        # Prediction statistics
        print("Prediction Statistics:")
        print(f"  Mean prediction: ${self.predictions.mean():,.2f}")
        print(f"  Std prediction: ${self.predictions.std():,.2f}")
        print(f"  Min prediction: ${self.predictions.min():,.2f}")
        print(f"  Max prediction: ${self.predictions.max():,.2f}\n")

        return self.metrics

    def feature_importance_analysis(self):
        """
        Analyze and visualize feature importance.

        Generates feature importance plot showing top N most important features.

        Returns
        -------
        pd.DataFrame
            Feature importance dataframe sorted by importance
        """
        print("=" * 80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 80)

        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        top_n = self.config["feature_importance"]["save_top_n"]
        print(f"\nTop {top_n} Most Important Features:")
        print(importance_df.head(top_n).to_string(index=False))

        # Plot feature importance
        if self.config["feature_importance"]["plot"]:
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(top_n)

            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance (Gain)')
            plt.title(f'Top {top_n} Feature Importance - LightGBM Model')
            plt.gca().invert_yaxis()
            plt.tight_layout()

            save_path = FIGURES_DIR / "lightgbm_feature_importance.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Feature importance plot saved: {save_path}")
            plt.close()

        # Save feature importance
        importance_path = REPORTS_DIR / "lightgbm_feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"✓ Feature importance saved: {importance_path}\n")

        return importance_df

    def shap_analysis(self):
        """
        Perform SHAP (SHapley Additive exPlanations) analysis for model explainability.

        Generates SHAP summary plot showing feature impact on predictions.

        Returns
        -------
        shap.Explainer
            SHAP explainer object
        """
        if not self.config["shap"]["enabled"]:
            print("SHAP analysis disabled in config\n")
            return None

        print("=" * 80)
        print("SHAP EXPLAINABILITY ANALYSIS")
        print("=" * 80)

        sample_size = min(self.config["shap"]["sample_size"], len(self.X_test))
        X_sample = self.X_test.sample(n=sample_size, random_state=RANDOM_SEED)

        print(f"Computing SHAP values for {sample_size} samples...")

        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)

        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_sample,
            max_display=self.config["shap"]["plot_top_features"],
            show=False
        )
        plt.tight_layout()

        save_path = FIGURES_DIR / "lightgbm_shap_summary.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ SHAP summary plot saved: {save_path}\n")
        plt.close()

        return explainer

    def save_model(self):
        """
        Save trained model and metadata.

        Saves:
        - LightGBM model in text format
        - Model metadata (features, metrics, config)
        - Training report

        Returns
        -------
        None
        """
        if not self.config["save_model"]:
            print("Model saving disabled in config\n")
            return

        print("=" * 80)
        print("SAVING MODEL ARTIFACTS")
        print("=" * 80)

        # Save LightGBM model
        model_path = MODELS_DIR / self.config["model_filename"]
        self.model.save_model(str(model_path))
        print(f"✓ Model saved: {model_path}")

        # Save model metadata
        metadata = {
            "model_type": "LightGBM",
            "phase": "1A - Bid Fee Prediction",
            "target_variable": TARGET_COLUMN,
            "num_features": len(self.feature_names),
            "features": self.feature_names,
            "training_samples": len(self.X_train),
            "test_samples": len(self.X_test),
            "best_iteration": int(self.model.best_iteration),
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "hyperparameters": self.config["params"],
            "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        metadata_path = MODELS_DIR / "lightgbm_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved: {metadata_path}")

        # Save predictions
        predictions_df = pd.DataFrame({
            'Actual': self.y_test.values,
            'Predicted': self.predictions,
            'Residual': self.y_test.values - self.predictions,
            'Abs_Error': np.abs(self.y_test.values - self.predictions),
            'Pct_Error': np.abs((self.y_test.values - self.predictions) / self.y_test.values) * 100
        })

        predictions_path = REPORTS_DIR / "lightgbm_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        print(f"✓ Predictions saved: {predictions_path}\n")

    def run_full_pipeline(self):
        """
        Execute the complete modeling pipeline.

        Steps:
        1. Load data
        2. Prepare features
        3. Train/test split
        4. Train model
        5. Evaluate model
        6. Feature importance
        7. SHAP analysis
        8. Save model

        Returns
        -------
        dict
            Final evaluation metrics
        """
        # Load and prepare data
        df = self.load_data()
        X, y = self.prepare_features(df)

        # Train/test split
        self.time_based_split(df, X, y)

        # Train model
        self.train_model()

        # Evaluate
        metrics = self.evaluate_model()

        # Feature importance
        self.feature_importance_analysis()

        # SHAP analysis
        self.shap_analysis()

        # Save model
        self.save_model()

        print("=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"✓ LightGBM model trained and evaluated successfully")
        print(f"  Final RMSE: ${metrics['rmse']:,.2f}")
        print(f"  Final R²: {metrics['r2']:.4f}\n")

        return metrics


def main():
    """Main execution function."""
    # Initialize and run LightGBM predictor
    predictor = LightGBMBidFeePredictor()
    metrics = predictor.run_full_pipeline()

    return predictor, metrics


if __name__ == "__main__":
    predictor, metrics = main()
