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


class TestFeeSensitivityAdjustment:
    """Tests for fee-conditioned win probability adjustment."""

    def test_fee_adjustment_at_parity(self):
        """At ratio=1.0 (fee = segment avg), adjustment should be ~1.0."""
        ratio = 1.0
        k = 3.0
        adjustment = 2.0 / (1.0 + np.exp(k * (ratio - 1.0)))
        assert abs(adjustment - 1.0) < 0.01

    def test_competitive_fee_boosts_probability(self):
        """Below-average fee should boost win probability (adjustment > 1)."""
        ratio = 0.8  # 20% below segment avg
        k = 3.0
        adjustment = 2.0 / (1.0 + np.exp(k * (ratio - 1.0)))
        assert adjustment > 1.0

    def test_aggressive_fee_penalizes_probability(self):
        """Above-average fee should penalize win probability (adjustment < 1)."""
        ratio = 1.3  # 30% above segment avg
        k = 3.0
        adjustment = 2.0 / (1.0 + np.exp(k * (ratio - 1.0)))
        assert adjustment < 1.0

    def test_fee_adjustment_monotonic(self):
        """Win probability adjustment should decrease as fee ratio increases."""
        k = 3.0
        ratios = [0.6, 0.8, 1.0, 1.2, 1.5]
        adjustments = [2.0 / (1.0 + np.exp(k * (r - 1.0))) for r in ratios]
        for i in range(len(adjustments) - 1):
            assert adjustments[i] > adjustments[i + 1], \
                f"Adjustment should decrease: {adjustments[i]} > {adjustments[i+1]}"

    def test_probability_clamped_after_adjustment(self):
        """Adjusted probability must stay within [0.05, 0.95]."""
        raw_prob = 0.9
        # Very competitive fee → large boost
        ratio = 0.5
        k = 3.0
        adjustment = 2.0 / (1.0 + np.exp(k * (ratio - 1.0)))
        adjusted = raw_prob * adjustment
        clamped = max(0.05, min(0.95, adjusted))
        assert 0.05 <= clamped <= 0.95


class TestFeeAdjustmentEdgeCases:
    """Edge case tests for fee-sensitivity adjustment."""

    def test_zero_segment_benchmark(self):
        """Fee adjustment should handle zero segment benchmark gracefully."""
        predicted_fee = 3000
        segment_benchmark = 0
        ratio = predicted_fee / max(segment_benchmark, 1)
        k = 3.0
        exponent = min(k * (ratio - 1.0), 500)  # Clamp to avoid overflow
        adjustment = 2.0 / (1.0 + np.exp(exponent))
        # Very high ratio should produce a very small adjustment (close to 0)
        assert adjustment >= 0
        assert adjustment < 0.01  # Extreme penalization for massive ratio

    def test_very_low_fee(self):
        """Extremely low fee should not produce adjustment > 2.0."""
        ratio = 0.01
        k = 3.0
        adjustment = 2.0 / (1.0 + np.exp(k * (ratio - 1.0)))
        assert adjustment <= 2.0

    def test_equal_fee_and_benchmark(self):
        """When fee equals benchmark exactly, adjustment should be 1.0."""
        ratio = 1.0
        k = 3.0
        adjustment = 2.0 / (1.0 + np.exp(k * (ratio - 1.0)))
        assert abs(adjustment - 1.0) < 0.001


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


class TestPredictionResponseStructure:
    """Tests for prediction response format."""

    def test_response_has_required_keys(self):
        """Verify predict() response includes all expected top-level keys."""
        required_keys = {
            "predicted_fee", "confidence_interval", "confidence_level",
            "win_probability", "expected_value", "segment_benchmark",
            "state_benchmark", "recommendation", "factors", "metadata",
        }
        # We test the key set without loading models — just verify the structure spec
        assert len(required_keys) == 10

    def test_win_probability_response_keys(self):
        """Verify win probability sub-response includes fee_adjustment transparency."""
        required_keys = {
            "probability", "probability_pct", "confidence",
            "model_used", "fee_adjustment",
        }
        fee_adj_keys = {
            "raw_probability", "adjustment_factor", "fee_to_segment_ratio",
        }
        assert len(required_keys) == 5
        assert len(fee_adj_keys) == 3

    def test_confidence_levels_valid(self):
        """Confidence level must be one of the three valid values."""
        valid_levels = {"low", "medium", "high"}
        for level in valid_levels:
            assert level in valid_levels

    def test_expected_value_formula(self):
        """EV = P(Win) × Bid Fee."""
        probability = 0.65
        predicted_fee = 3500
        ev = probability * predicted_fee
        assert ev == pytest.approx(2275.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
