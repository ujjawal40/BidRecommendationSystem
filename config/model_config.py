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
    # Model hyperparameters
    "params": {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "max_depth": -1,
        "min_child_samples": 20,
        "reg_alpha": 0.1,  # L1 regularization
        "reg_lambda": 0.1,  # L2 regularization
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },

    # Training parameters
    "training": {
        "num_boost_round": 1000,
        "early_stopping_rounds": 50,
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
