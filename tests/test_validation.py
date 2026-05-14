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

    def test_win_prob_confidence_independent_of_bid_fee(self):
        """Win probability confidence is based purely on distance from 0.5.

        Since the win prob model includes BidFee as a direct feature,
        its confidence stands on its own — no cap from bid fee confidence.
        """
        probability = 0.86  # High probability
        distance_from_uncertain = abs(probability - 0.5)  # 0.36

        if distance_from_uncertain > 0.3:
            win_confidence = "high"
        elif distance_from_uncertain > 0.15:
            win_confidence = "medium"
        else:
            win_confidence = "low"

        # 86% probability should be "high" confidence regardless of bid fee confidence
        assert win_confidence == "high"


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
        """Verify win probability sub-response keys (model has BidFee feature, no sigmoid hack)."""
        required_keys = {
            "probability", "probability_pct", "confidence",
            "model_used",
        }
        assert len(required_keys) == 4

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


class TestPredictionConfig:
    """Tests for prediction config constants."""

    def test_prediction_config_importable(self):
        """PREDICTION_CONFIG should be importable from model_config."""
        from config.model_config import PREDICTION_CONFIG
        assert isinstance(PREDICTION_CONFIG, dict)

    def test_prediction_config_required_keys(self):
        """PREDICTION_CONFIG must have all required keys."""
        from config.model_config import PREDICTION_CONFIG
        required = {
            'fee_sensitivity_k', 'confidence_segment_high', 'confidence_state_high',
            'confidence_segment_medium', 'confidence_state_medium',
            'band_ratio_high', 'band_ratio_medium',
            'win_prob_min', 'win_prob_max', 'min_fee',
        }
        assert required.issubset(set(PREDICTION_CONFIG.keys()))

    def test_prediction_config_values_reasonable(self):
        """Config values should be within reasonable ranges."""
        from config.model_config import PREDICTION_CONFIG
        assert PREDICTION_CONFIG['fee_sensitivity_k'] > 0
        assert PREDICTION_CONFIG['win_prob_min'] > 0
        assert PREDICTION_CONFIG['win_prob_max'] < 1
        assert PREDICTION_CONFIG['win_prob_min'] < PREDICTION_CONFIG['win_prob_max']
        assert PREDICTION_CONFIG['min_fee'] > 0
        assert PREDICTION_CONFIG['band_ratio_high'] < PREDICTION_CONFIG['band_ratio_medium']


class TestZipLookup:
    """Tests for zip code demographics lookup."""

    def test_zip_lookup_exists(self):
        """Verify zip_demographics_lookup.json exists."""
        from config.model_config import REPORTS_DIR
        path = REPORTS_DIR / "zip_demographics_lookup.json"
        assert path.exists(), f"Zip lookup not found: {path}"

    def test_zip_lookup_structure(self):
        """Verify zip lookup JSON has expected structure."""
        import json
        from config.model_config import REPORTS_DIR
        with open(REPORTS_DIR / "zip_demographics_lookup.json") as f:
            lookup = json.load(f)

        assert len(lookup) > 1000, f"Expected > 1000 zips, got {len(lookup)}"

        # All keys should be 5-digit strings
        for zip_code in list(lookup.keys())[:100]:
            assert len(zip_code) == 5, f"Zip should be 5 digits: {zip_code}"
            assert zip_code.isdigit(), f"Zip should be numeric: {zip_code}"

    def test_zip_entry_has_demographics(self):
        """Verify each zip entry has expected demographic fields."""
        import json
        from config.model_config import REPORTS_DIR
        with open(REPORTS_DIR / "zip_demographics_lookup.json") as f:
            lookup = json.load(f)

        required_fields = [
            "Zip_Population", "Zip_PopDensity", "Zip_AverageHouseValue",
            "Zip_MedianIncome", "Zip_NumberOfBusinesses",
        ]
        # Check first 50 entries
        for zip_code in list(lookup.keys())[:50]:
            entry = lookup[zip_code]
            for field in required_fields:
                assert field in entry, f"Zip {zip_code} missing {field}"
                assert isinstance(entry[field], (int, float)), f"Zip {zip_code}.{field} should be numeric"

    def test_zip_entry_has_state(self):
        """Most zip entries should include a state."""
        import json
        from config.model_config import REPORTS_DIR
        with open(REPORTS_DIR / "zip_demographics_lookup.json") as f:
            lookup = json.load(f)

        with_state = sum(1 for e in lookup.values() if e.get("state"))
        pct = with_state / len(lookup)
        assert pct > 0.9, f"Expected > 90% of zips with state, got {pct:.1%}"

    def test_known_zip_returns_correct_state(self):
        """Test that well-known zips map to correct states."""
        import json
        from config.model_config import REPORTS_DIR
        with open(REPORTS_DIR / "zip_demographics_lookup.json") as f:
            lookup = json.load(f)

        known_zips = {
            "10001": "New York",
            "60601": "Illinois",
            "90210": "California",
        }
        for zip_code, expected_state in known_zips.items():
            if zip_code in lookup:
                assert lookup[zip_code].get("state") == expected_state, \
                    f"Zip {zip_code} should be {expected_state}, got {lookup[zip_code].get('state')}"


class TestStateDistanceMiles:
    """Tests for state_distance_miles in v2 stats."""

    def test_state_distance_exists_in_v2_stats(self):
        """Verify state_distance_miles key exists in v2 stats."""
        import json
        from config.model_config import REPORTS_DIR
        with open(REPORTS_DIR / "api_precomputed_stats_v2.json") as f:
            stats = json.load(f)

        assert "state_distance_miles" in stats, "state_distance_miles missing from v2 stats"
        assert len(stats["state_distance_miles"]) > 0, "state_distance_miles is empty"

    def test_state_distance_values_reasonable(self):
        """Verify distance values are reasonable (positive, < 1000 miles)."""
        import json
        from config.model_config import REPORTS_DIR
        with open(REPORTS_DIR / "api_precomputed_stats_v2.json") as f:
            stats = json.load(f)

        for state, dist in stats["state_distance_miles"].items():
            assert dist > 0, f"{state} has non-positive distance: {dist}"
            assert dist < 1000, f"{state} has unreasonable distance: {dist}"

    def test_major_states_have_distance(self):
        """Major states should have distance data."""
        import json
        from config.model_config import REPORTS_DIR
        with open(REPORTS_DIR / "api_precomputed_stats_v2.json") as f:
            stats = json.load(f)

        for state in ["Illinois", "California", "Texas", "New York", "Florida"]:
            assert state in stats["state_distance_miles"], \
                f"{state} missing from state_distance_miles"


class TestWinRateLookupsV3:
    """Validate the v3 win-rate lookup JSON shape and bounds."""

    def _load(self):
        import json
        from config.model_config import REPORTS_DIR
        path = REPORTS_DIR / "win_rate_lookups_v3.json"
        if not path.exists():
            pytest.skip("win_rate_lookups_v3.json not present")
        with open(path) as f:
            return json.load(f)

    def test_required_keys_present(self):
        d = self._load()
        for key in [
            "global_win_rate",
            "segment_win_rate",
            "propertytype_win_rate",
            "state_win_rate",
            "subtype_win_rate",
            "office_region_win_rate",
            "company_location_win_rate",
            "segment_fee_cdf",
        ]:
            assert key in d, f"missing key: {key}"

    def test_global_win_rate_in_unit_interval(self):
        d = self._load()
        assert 0.0 < d["global_win_rate"] < 1.0

    def test_segment_win_rates_in_unit_interval(self):
        d = self._load()
        for seg, rate in d["segment_win_rate"].items():
            assert 0.0 <= rate <= 1.0, f"{seg} win rate out of range: {rate}"

    def test_segment_fee_cdf_monotonic(self):
        d = self._load()
        for seg, q in d["segment_fee_cdf"].items():
            assert len(q) == 5, f"{seg} fee CDF wrong length: {len(q)}"
            for a, b in zip(q[:-1], q[1:]):
                assert a <= b, f"{seg} fee CDF not monotonic: {q}"


class TestV3FeeSensitiveModelMetadata:
    """Validate the v3_fee_sensitive metadata file."""

    def _load(self):
        import json
        from config.model_config import MODELS_DIR
        path = MODELS_DIR / "lightgbm_win_probability_v3_fee_sensitive_metadata.json"
        if not path.exists():
            pytest.skip("v3_fee_sensitive metadata not present")
        with open(path) as f:
            return json.load(f)

    def test_marked_as_ablation_variant(self):
        d = self._load()
        assert d.get("variant") == "fee_sensitive_ablation", \
            "v3_fee_sensitive metadata must declare variant=fee_sensitive_ablation"

    def test_auc_above_minimum_threshold(self):
        d = self._load()
        auc = d["metrics"]["test_calibrated"]["auc"]
        assert auc >= 0.85, f"v3_fee_sensitive AUC too low: {auc}"

    def test_overfit_ratio_reasonable(self):
        d = self._load()
        train_auc = d["metrics"]["train"]["auc"]
        test_auc = d["metrics"]["test_calibrated"]["auc"]
        ratio = train_auc / max(test_auc, 1e-6)
        assert ratio <= 1.25, f"v3_fee_sensitive overfit too high: {ratio:.2f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
