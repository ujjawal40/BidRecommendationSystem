"""
Model Configuration File
=========================
Centralized configuration for all models in the Bid Recommendation System.

This file contains hyperparameters, paths, and settings for:
- LightGBM regression model (Phase 1A: Bid Fee Prediction)
- Win Probability classification model (Phase 1B - future)

Author: Bid Recommendation System
Date: 2026-01-06
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Root directory
ROOT_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = ROOT_DIR / "data"
FEATURES_DATA = DATA_DIR / "features" / "BidData_features.csv"  # Baseline: All JobData approaches failed (static, selective, competitive)
PROCESSED_DATA = DATA_DIR / "processed" / "BidData_processed.csv"

# ============================================================================
# DATA FILTERING CONFIGURATION
# ============================================================================
# Training on recent data (2023+) improves generalization
# See: outputs/reports/recent_data_experiment_results.json
DATA_START_DATE = "2023-01-01"  # Filter training data to 2023+ only
USE_RECENT_DATA_ONLY = True     # Set to False to use all historical data

# JobData features to EXCLUDE (they degrade performance - see jobdata_enrichment_final_results.txt)
JOBDATA_FEATURES_TO_EXCLUDE = [
    "office_job_volume",
    "office_avg_job_fee",
    "office_median_job_fee",
    "office_job_fee_std",
    "office_min_job_fee",
    "office_max_job_fee",
    "office_avg_appraisal_fee",
    "office_median_appraisal_fee",
    "office_master_job_pct",
    "office_fee_range",
    "office_fee_cv",
    "office_avg_profit_margin",
    "region_avg_job_fee",
    "region_median_job_fee",
    "region_job_volume",
    "office_vs_region_premium",
    "office_vs_region_ratio",
    "office_region_encoded",
    "office_primary_client_type_encoded",
    "office_market_tier_encoded",
    "property_region_avg_fee",
    "property_region_median_fee",
    "office_region",
    "office_primary_property_type",
    "office_primary_client_type",
    "office_market_tier",
    "PropertyType_enriched",
]

# Output paths
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Create directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Target variable
TARGET_COLUMN = "BidFee"

# Date column for time-based split
DATE_COLUMN = "BidDate"

# Columns to exclude from features (IDs, targets, dates)
EXCLUDE_COLUMNS = [
    "BidId",
    "BidName",
    "BidDate",
    "BidFee",  # Target for Phase 1A
    "Won",  # Target for Phase 1B
    "BidStatusName",
    "Bid_DueDate",
    "JobId",
    "PropertyId",
    "BidFee_Original",  # Keep only transformed BidFee
]

# Train/Test split configuration
TRAIN_TEST_SPLIT = {
    "method": "time_based",  # time_based or random
    "train_ratio": 0.8,  # 80% train, 20% test
    "validation_split": 0.2,  # 20% of training data for validation
    "random_state": 42,
}

# ============================================================================
# LIGHTGBM CONFIGURATION (Phase 1A: Bid Fee Prediction)
# ============================================================================

LIGHTGBM_CONFIG = {
    # Model hyperparameters - OPTIMIZED via experiment A9 (log-transform + lower LR)
    # MAPE improved from 71% to 16%, overfitting from 1.99x to 1.60x
    "params": {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 18,
        "learning_rate": 0.02,  # Lowered from 0.05 (experiment A9)
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "max_depth": 8,
        "min_child_samples": 30,
        "min_child_weight": 5,
        "reg_alpha": 2.0,  # Increased from 1.0 for stronger regularization
        "reg_lambda": 2.0,  # Increased from 1.0 for stronger regularization
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },

    # Target transform (applied before training, inverted after prediction)
    "target_transform": "log1p",  # log1p(BidFee) for proportional errors

    # Training parameters - More rounds needed with lower LR
    "training": {
        "num_boost_round": 1500,  # Increased from 500 (needed with LR 0.02)
        "early_stopping_rounds": 100,
        "verbose_eval": 100,
    },

    # Feature importance
    "feature_importance": {
        "save_top_n": 30,  # Save top 30 most important features
        "plot": True,
    },

    # SHAP explainability
    "shap": {
        "enabled": True,
        "sample_size": 1000,  # Sample size for SHAP calculation
        "plot_top_features": 20,
    },

    # Model artifacts
    "save_model": True,
    "model_filename": "lightgbm_bidfee_model.txt",
}

# ============================================================================
# PREDICTION SERVICE CONFIGURATION
# ============================================================================

PREDICTION_CONFIG = {
    # Fee-sensitivity adjustment for win probability
    # The classification model doesn't include BidFee features, so we apply a
    # post-prediction sigmoid: adjustment = 2 / (1 + exp(k * (ratio - 1)))
    "fee_sensitivity_k": 3.0,  # Sigmoid steepness (higher = more sensitive to fee)

    # Confidence thresholds for bid fee
    "confidence_segment_high": 1000,  # Min segment count for "high" data confidence
    "confidence_state_high": 500,     # Min state count for "high" data confidence
    "confidence_segment_medium": 100,
    "confidence_state_medium": 50,

    # Band width thresholds (ratio of band width to predicted fee)
    "band_ratio_high": 0.3,    # Below this = "high" band confidence
    "band_ratio_medium": 0.6,  # Below this = "medium" band confidence

    # Win probability clamp range
    "win_prob_min": 0.05,
    "win_prob_max": 0.95,

    # Minimum bid fee floor
    "min_fee": 500,
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================

EVALUATION_METRICS = {
    "regression": [
        "rmse",  # Root Mean Squared Error
        "mae",  # Mean Absolute Error
        "r2",  # R-squared
        "mape",  # Mean Absolute Percentage Error
        "median_ae",  # Median Absolute Error
    ],

    "classification": [  # For Phase 1B
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc_roc",
    ],
}

# ============================================================================
# CROSS-VALIDATION CONFIGURATION
# ============================================================================

CROSS_VALIDATION = {
    "enabled": False,  # Set to True to enable CV (takes longer)
    "method": "time_series_split",  # time_series_split or k_fold
    "n_splits": 5,
    "gap": 30,  # Gap between train and test in days (for time series)
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "save_logs": True,
    "log_file": OUTPUTS_DIR / "logs" / "model_training.log",
}

# Create logs directory
(OUTPUTS_DIR / "logs").mkdir(parents=True, exist_ok=True)

# ============================================================================
# RANDOM SEED FOR REPRODUCIBILITY
# ============================================================================

RANDOM_SEED = 42

# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

EXPERIMENT_CONFIG = {
    "track_experiments": True,
    "experiment_name": "Phase_1A_BidFee_Prediction",
    "save_predictions": True,
    "save_metrics": True,
    "save_plots": True,
}
