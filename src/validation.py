"""
Input Validation Module
=======================
Validates input data before model prediction to ensure data quality.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class InputValidator:
    """Validates input features for bid fee prediction."""

    # Expected feature ranges (based on training data)
    FEATURE_RANGES = {
        'TargetTime': (1, 365),
        'DistanceInKM': (0, 5000),
        'DistanceInMiles': (0, 3000),
        'JobCount': (0, 10000),
        'PopulationEstimate': (0, 10_000_000),
        'AverageHouseValue': (0, 5_000_000),
        'IncomePerHousehold': (0, 500_000),
    }

    REQUIRED_FEATURES = [
        'TargetTime',
        'segment_avg_fee',
        'state_avg_fee',
        'office_avg_fee',
    ]

    def __init__(self):
        self.errors = []
        self.warnings = []

    def validate(self, X: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
        """
        Validate input features.

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        # Check for required features
        self._check_required_features(X)

        # Check for missing values
        self._check_missing_values(X)

        # Check feature ranges
        self._check_feature_ranges(X)

        # Check for negative values where not allowed
        self._check_negative_values(X)

        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings

    def _check_required_features(self, X: pd.DataFrame):
        """Check if required features are present."""
        missing = [f for f in self.REQUIRED_FEATURES if f not in X.columns]
        if missing:
            self.errors.append(f"Missing required features: {missing}")

    def _check_missing_values(self, X: pd.DataFrame):
        """Check for missing values."""
        missing_counts = X.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        if len(cols_with_missing) > 0:
            self.warnings.append(f"Features with missing values: {dict(cols_with_missing)}")

    def _check_feature_ranges(self, X: pd.DataFrame):
        """Check if features are within expected ranges."""
        for feature, (min_val, max_val) in self.FEATURE_RANGES.items():
            if feature in X.columns:
                below_min = (X[feature] < min_val).sum()
                above_max = (X[feature] > max_val).sum()
                if below_min > 0:
                    self.warnings.append(f"{feature}: {below_min} values below {min_val}")
                if above_max > 0:
                    self.warnings.append(f"{feature}: {above_max} values above {max_val}")

    def _check_negative_values(self, X: pd.DataFrame):
        """Check for negative values in fee-related features."""
        fee_features = [c for c in X.columns if 'fee' in c.lower() or 'avg' in c.lower()]
        for feature in fee_features:
            if feature in X.columns:
                neg_count = (X[feature] < 0).sum()
                if neg_count > 0:
                    self.warnings.append(f"{feature}: {neg_count} negative values")


def validate_prediction_input(X: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
    """
    Convenience function to validate prediction input.

    Args:
        X: Input DataFrame with features

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    validator = InputValidator()
    return validator.validate(X)
