"""
Empirical Confidence Bands Calculator
=====================================
Computes stratified empirical quantile bands from test set residuals.

Strategy:
- Stratify by fee bucket (low/medium/high)
- Compute 10th, 25th, 75th, 90th percentiles per stratum
- Wider bands for high fees (cap effect)

Author: Global Stat Solutions
Date: 2026-01-29
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import json
from typing import Dict, Tuple, Optional

from config.model_config import REPORTS_DIR, FEATURES_DATA


class EmpiricalBandCalculator:
    """Calculate empirical confidence bands from residuals."""

    # Fee bucket boundaries
    FEE_BUCKETS = {
        'low': (0, 2000),
        'medium': (2000, 4000),
        'high': (4000, 6000),
        'very_high': (6000, float('inf')),
    }

    def __init__(self):
        self.bands_by_bucket = {}
        self.bands_by_segment = {}
        self.bands_by_state = {}
        self.global_bands = {}
        self._computed = False

    def compute_bands_from_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        segments: Optional[np.ndarray] = None,
        states: Optional[np.ndarray] = None,
    ):
        """
        Compute empirical bands from predictions and actuals.

        Parameters:
        -----------
        y_true : array
            Actual bid fees
        y_pred : array
            Predicted bid fees
        segments : array, optional
            Business segments for each prediction
        states : array, optional
            States for each prediction
        """
        residuals = y_pred - y_true

        # Global bands
        self.global_bands = self._compute_quantiles(residuals)

        # Bands by fee bucket
        for bucket_name, (low, high) in self.FEE_BUCKETS.items():
            mask = (y_pred >= low) & (y_pred < high)
            if mask.sum() >= 20:  # Need minimum samples
                self.bands_by_bucket[bucket_name] = self._compute_quantiles(residuals[mask])
            else:
                self.bands_by_bucket[bucket_name] = self.global_bands

        # Bands by segment
        if segments is not None:
            for segment in np.unique(segments):
                mask = segments == segment
                if mask.sum() >= 30:
                    self.bands_by_segment[segment] = self._compute_quantiles(residuals[mask])

        # Bands by state
        if states is not None:
            for state in np.unique(states):
                mask = states == state
                if mask.sum() >= 50:  # States need more samples
                    self.bands_by_state[state] = self._compute_quantiles(residuals[mask])

        self._computed = True

    def _compute_quantiles(self, residuals: np.ndarray) -> Dict:
        """Compute quantile-based bands from residuals."""
        return {
            'p5': float(np.percentile(residuals, 5)),
            'p10': float(np.percentile(residuals, 10)),
            'p25': float(np.percentile(residuals, 25)),
            'p50': float(np.percentile(residuals, 50)),  # Median bias
            'p75': float(np.percentile(residuals, 75)),
            'p90': float(np.percentile(residuals, 90)),
            'p95': float(np.percentile(residuals, 95)),
            'std': float(np.std(residuals)),
            'n_samples': int(len(residuals)),
        }

    def get_confidence_interval(
        self,
        predicted_fee: float,
        segment: Optional[str] = None,
        state: Optional[str] = None,
        confidence_level: float = 0.80,
    ) -> Tuple[float, float, Dict]:
        """
        Get confidence interval for a prediction.

        Parameters:
        -----------
        predicted_fee : float
            The predicted bid fee
        segment : str, optional
            Business segment
        state : str, optional
            Property state
        confidence_level : float
            Confidence level (0.80 = 80% interval)

        Returns:
        --------
        tuple : (lower_bound, upper_bound, metadata)
        """
        if not self._computed:
            # Return default bands if not computed
            default_std = 350  # Approximate from model RMSE
            multiplier = 1.28 if confidence_level == 0.80 else 1.96
            return (
                max(500, predicted_fee - multiplier * default_std),
                predicted_fee + multiplier * default_std,
                {'source': 'default', 'method': 'normal_approximation'}
            )

        # Determine which bands to use (priority: segment > state > bucket > global)
        bands = None
        source = 'global'

        # Try segment-specific bands
        if segment and segment in self.bands_by_segment:
            bands = self.bands_by_segment[segment]
            source = f'segment:{segment}'

        # Try state-specific bands (if no segment bands)
        elif state and state in self.bands_by_state:
            bands = self.bands_by_state[state]
            source = f'state:{state}'

        # Try fee bucket bands
        if bands is None:
            for bucket_name, (low, high) in self.FEE_BUCKETS.items():
                if low <= predicted_fee < high:
                    bands = self.bands_by_bucket.get(bucket_name, self.global_bands)
                    source = f'fee_bucket:{bucket_name}'
                    break

        # Fallback to global
        if bands is None:
            bands = self.global_bands
            source = 'global'

        # Select quantiles based on confidence level
        if confidence_level >= 0.90:
            lower_q, upper_q = 'p5', 'p95'
        elif confidence_level >= 0.80:
            lower_q, upper_q = 'p10', 'p90'
        else:
            lower_q, upper_q = 'p25', 'p75'

        # Compute bounds (residual = pred - actual, so actual = pred - residual)
        # Lower bound: predicted - upper_residual_quantile
        # Upper bound: predicted - lower_residual_quantile
        lower_bound = predicted_fee - bands[upper_q]
        upper_bound = predicted_fee - bands[lower_q]

        # Ensure reasonable bounds
        lower_bound = max(500, lower_bound)  # Minimum $500
        upper_bound = max(lower_bound + 100, upper_bound)  # At least $100 range

        return (
            round(lower_bound, 2),
            round(upper_bound, 2),
            {
                'source': source,
                'method': 'empirical_quantiles',
                'confidence_level': confidence_level,
                'n_samples': bands.get('n_samples', 0),
                'median_bias': bands.get('p50', 0),
            }
        )

    def save_bands(self, filepath: Path = None):
        """Save computed bands to JSON."""
        if filepath is None:
            filepath = REPORTS_DIR / 'empirical_bands.json'

        data = {
            'global': self.global_bands,
            'by_fee_bucket': self.bands_by_bucket,
            'by_segment': self.bands_by_segment,
            'by_state': self.bands_by_state,
            'fee_bucket_boundaries': self.FEE_BUCKETS,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"[EmpiricalBands] Saved to {filepath}")

    def load_bands(self, filepath: Path = None):
        """Load precomputed bands from JSON."""
        if filepath is None:
            filepath = REPORTS_DIR / 'empirical_bands.json'

        if not filepath.exists():
            print(f"[EmpiricalBands] No saved bands found at {filepath}")
            return False

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.global_bands = data.get('global', {})
        self.bands_by_bucket = data.get('by_fee_bucket', {})
        self.bands_by_segment = data.get('by_segment', {})
        self.bands_by_state = data.get('by_state', {})
        self._computed = True

        print(f"[EmpiricalBands] Loaded from {filepath}")
        return True


def compute_and_save_bands():
    """Compute empirical bands from the trained model's test predictions."""
    import lightgbm as lgb
    from config.model_config import MODELS_DIR, EXCLUDE_COLUMNS, JOBDATA_FEATURES_TO_EXCLUDE, DATA_START_DATE

    print("=" * 60)
    print("COMPUTING EMPIRICAL CONFIDENCE BANDS")
    print("=" * 60)

    # Load model
    model_path = MODELS_DIR / "lightgbm_bidfee_model.txt"
    model = lgb.Booster(model_file=str(model_path))
    model_features = model.feature_name()

    # Load data
    df = pd.read_csv(FEATURES_DATA)
    df['BidDate'] = pd.to_datetime(df['BidDate'])
    df = df[df['BidDate'] >= pd.Timestamp(DATA_START_DATE)].copy()
    df = df.sort_values('BidDate').reset_index(drop=True)

    # Prepare features
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    feature_cols = [col for col in feature_cols if col not in JOBDATA_FEATURES_TO_EXCLUDE]
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # Get test set (last 20%)
    test_start = int(len(df) * 0.8)
    test_df = df.iloc[test_start:].copy()

    # Prepare features for prediction
    available_features = [f for f in model_features if f in numeric_features]
    X_test = test_df[available_features].fillna(0)
    y_test = test_df['BidFee'].values

    # Predict
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)

    print(f"Test set size: {len(y_test)}")
    print(f"RMSE: ${np.sqrt(np.mean((y_pred - y_test)**2)):,.2f}")

    # Compute bands
    calculator = EmpiricalBandCalculator()
    calculator.compute_bands_from_predictions(
        y_true=y_test,
        y_pred=y_pred,
        segments=test_df['BusinessSegment'].values if 'BusinessSegment' in test_df.columns else None,
        states=test_df['PropertyState'].values if 'PropertyState' in test_df.columns else None,
    )

    # Print summary
    print("\nGlobal Bands:")
    for k, v in calculator.global_bands.items():
        print(f"  {k}: {v}")

    print("\nBands by Fee Bucket:")
    for bucket, bands in calculator.bands_by_bucket.items():
        print(f"  {bucket}: p10={bands['p10']:.0f}, p90={bands['p90']:.0f}, n={bands['n_samples']}")

    print("\nBands by Segment (top 5):")
    for segment in list(calculator.bands_by_segment.keys())[:5]:
        bands = calculator.bands_by_segment[segment]
        print(f"  {segment}: p10={bands['p10']:.0f}, p90={bands['p90']:.0f}, n={bands['n_samples']}")

    # Save
    calculator.save_bands()

    return calculator


if __name__ == "__main__":
    compute_and_save_bands()
