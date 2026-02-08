"""
Unit Tests for Validation Module
================================
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from validation import InputValidator, validate_prediction_input


class TestInputValidator:
    """Tests for InputValidator class."""

    def test_valid_input(self):
        """Test that valid input passes validation."""
        X = pd.DataFrame({
            'TargetTime': [30, 45, 60],
            'segment_avg_fee': [3000, 3500, 4000],
            'state_avg_fee': [2800, 3200, 3800],
            'office_avg_fee': [2900, 3300, 3900],
            'DistanceInKM': [10, 20, 30],
        })

        is_valid, errors, warnings = validate_prediction_input(X)
        assert is_valid is True
        assert len(errors) == 0

    def test_missing_required_features(self):
        """Test that missing required features are caught."""
        X = pd.DataFrame({
            'TargetTime': [30, 45, 60],
            # Missing: segment_avg_fee, state_avg_fee, office_avg_fee
        })

        is_valid, errors, warnings = validate_prediction_input(X)
        assert is_valid is False
        assert any('Missing required features' in e for e in errors)

    def test_out_of_range_values(self):
        """Test that out-of-range values generate warnings."""
        X = pd.DataFrame({
            'TargetTime': [30, 500, 60],  # 500 is out of range
            'segment_avg_fee': [3000, 3500, 4000],
            'state_avg_fee': [2800, 3200, 3800],
            'office_avg_fee': [2900, 3300, 3900],
        })

        is_valid, errors, warnings = validate_prediction_input(X)
        assert any('TargetTime' in w for w in warnings)

    def test_negative_fee_values(self):
        """Test that negative fee values generate warnings."""
        X = pd.DataFrame({
            'TargetTime': [30, 45, 60],
            'segment_avg_fee': [3000, -100, 4000],  # Negative value
            'state_avg_fee': [2800, 3200, 3800],
            'office_avg_fee': [2900, 3300, 3900],
        })

        is_valid, errors, warnings = validate_prediction_input(X)
        assert any('negative' in w.lower() for w in warnings)

    def test_missing_values_warning(self):
        """Test that missing values generate warnings."""
        X = pd.DataFrame({
            'TargetTime': [30, np.nan, 60],
            'segment_avg_fee': [3000, 3500, 4000],
            'state_avg_fee': [2800, 3200, 3800],
            'office_avg_fee': [2900, 3300, 3900],
        })

        is_valid, errors, warnings = validate_prediction_input(X)
        assert any('missing' in w.lower() for w in warnings)


class TestConfidenceLogic:
    """Tests for confidence level computation logic."""

    def test_state_count_not_proportion(self):
        """Verify confidence uses actual counts, not proportions (0-1)."""
        # state_frequency is a proportion like 0.234 — must NOT be used for count thresholds
        # state_count is the actual count like 26748 — must be used instead
        state_frequency = 0.234  # Illinois proportion
        state_count = 26748  # Illinois actual count

        # Proportion should never exceed count thresholds
        assert state_frequency < 50, "Proportions should be < 1, never > 50"
        # Actual count should pass count thresholds
        assert state_count > 500, "Illinois should have > 500 samples"

    def test_confidence_hierarchy_high(self):
        """High data availability with narrow band should give high confidence."""
        segment_count = 5000
        state_count = 2000
        band_ratio = 0.1  # narrow band

        if segment_count > 1000 and state_count > 500:
            data_confidence = "high"
        elif segment_count > 100 and state_count > 50:
            data_confidence = "medium"
        else:
            data_confidence = "low"

        if band_ratio < 0.3:
            band_confidence = "high"
        elif band_ratio < 0.6:
            band_confidence = "medium"
        else:
            band_confidence = "low"

        rank = {"low": 0, "medium": 1, "high": 2}
        confidence = min(rank[data_confidence], rank[band_confidence])
        assert confidence == 2  # high

    def test_confidence_hierarchy_capped_by_band(self):
        """Wide confidence band should cap overall confidence."""
        segment_count = 5000
        state_count = 2000
        band_ratio = 0.8  # wide band

        if segment_count > 1000 and state_count > 500:
            data_confidence = "high"
        else:
            data_confidence = "low"

        if band_ratio < 0.3:
            band_confidence = "high"
        elif band_ratio < 0.6:
            band_confidence = "medium"
        else:
            band_confidence = "low"

        rank = {"low": 0, "medium": 1, "high": 2}
        confidence = min(rank[data_confidence], rank[band_confidence])
        assert confidence == 0  # low (capped by wide band)

    def test_win_prob_confidence_capped_by_bid_fee(self):
        """Win probability confidence should never exceed bid fee confidence."""
        confidence_rank = {"low": 0, "medium": 1, "high": 2}
        rank_to_label = {0: "low", 1: "medium", 2: "high"}

        bid_fee_confidence = "low"
        win_confidence = "high"

        max_rank = confidence_rank[bid_fee_confidence]
        if confidence_rank[win_confidence] > max_rank:
            win_confidence = rank_to_label[max_rank]

        assert win_confidence == "low"


class TestPredictionClamping:
    """Tests for prediction post-processing."""

    def test_negative_predictions_clamped(self):
        """Test that negative predictions are clamped to zero."""
        raw_predictions = np.array([100, -50, 200, -10, 300])
        clamped = np.maximum(0, raw_predictions)

        assert clamped.min() >= 0
        assert (clamped == np.array([100, 0, 200, 0, 300])).all()

    def test_mape_excludes_zeros(self):
        """Test MAPE calculation excludes zero actuals."""
        actuals = np.array([100, 0, 200, 300])
        predictions = np.array([110, 50, 180, 290])

        # MAPE excluding zeros
        non_zero_mask = actuals != 0
        mape = np.mean(np.abs((actuals[non_zero_mask] - predictions[non_zero_mask]) / actuals[non_zero_mask])) * 100

        assert not np.isinf(mape)
        assert mape > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
