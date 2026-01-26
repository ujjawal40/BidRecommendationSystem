"""
Prediction Confidence Intervals
===============================
Provides uncertainty estimates for bid fee predictions.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import lightgbm as lgb


class PredictionConfidence:
    """Calculate confidence intervals for LightGBM predictions."""

    def __init__(self, model: lgb.Booster, X_train: np.ndarray, y_train: np.ndarray):
        """
        Initialize with trained model and training data.

        Args:
            model: Trained LightGBM model
            X_train: Training features
            y_train: Training targets
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train

        # Calculate residual standard deviation from training
        train_preds = model.predict(X_train)
        self.residual_std = np.std(y_train - train_preds)

    def predict_with_interval(
        self,
        X: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with confidence intervals.

        Args:
            X: Features for prediction
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        from scipy import stats

        # Point predictions
        predictions = np.maximum(0, self.model.predict(X))

        # Z-score for confidence level
        z = stats.norm.ppf((1 + confidence) / 2)

        # Simple confidence interval based on residual std
        margin = z * self.residual_std

        lower_bound = np.maximum(0, predictions - margin)
        upper_bound = predictions + margin

        return predictions, lower_bound, upper_bound

    def get_prediction_summary(
        self,
        X: np.ndarray,
        confidence: float = 0.95
    ) -> pd.DataFrame:
        """
        Get prediction summary with confidence intervals.

        Args:
            X: Features for prediction
            confidence: Confidence level

        Returns:
            DataFrame with predictions and confidence intervals
        """
        preds, lower, upper = self.predict_with_interval(X, confidence)

        return pd.DataFrame({
            'Prediction': preds,
            'Lower_CI': lower,
            'Upper_CI': upper,
            'CI_Width': upper - lower,
            'Confidence': confidence
        })


def add_confidence_intervals(
    predictions: np.ndarray,
    residual_std: float,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple function to add confidence intervals to predictions.

    Args:
        predictions: Point predictions
        residual_std: Standard deviation of residuals
        confidence: Confidence level (default 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy import stats

    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * residual_std

    lower = np.maximum(0, predictions - margin)
    upper = predictions + margin

    return lower, upper
