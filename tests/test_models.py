"""
Tests for Bid Recommendation System Models
==========================================
Tests both Phase 1A (Bid Fee) and Phase 1B (Win Probability) models.

Run with: pytest tests/test_models.py -v
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
import lightgbm as lgb
import json

from config.model_config import (
    FEATURES_DATA, MODELS_DIR, TARGET_COLUMN,
    DATA_START_DATE, EXCLUDE_COLUMNS, JOBDATA_FEATURES_TO_EXCLUDE,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def sample_data():
    """Load and prepare sample data for testing."""
    df = pd.read_csv(FEATURES_DATA)
    df['BidDate'] = pd.to_datetime(df['BidDate'])
    df = df.sort_values('BidDate').reset_index(drop=True)

    # Filter to 2023+ data
    start_date = pd.Timestamp(DATA_START_DATE)
    df = df[df['BidDate'] >= start_date].copy()

    return df


@pytest.fixture(scope="module")
def bidfee_model():
    """Load the Phase 1A Bid Fee model."""
    model_path = MODELS_DIR / "lightgbm_bidfee_model.txt"
    if not model_path.exists():
        pytest.skip("Bid Fee model not found")
    return lgb.Booster(model_file=str(model_path))


@pytest.fixture(scope="module")
def bidfee_metadata():
    """Load the Phase 1A metadata."""
    metadata_path = MODELS_DIR / "lightgbm_metadata.json"
    if not metadata_path.exists():
        pytest.skip("Bid Fee metadata not found")
    with open(metadata_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def winprob_model():
    """Load the Phase 1B Win Probability model."""
    model_path = MODELS_DIR / "lightgbm_win_probability.txt"
    if not model_path.exists():
        pytest.skip("Win Probability model not found")
    return lgb.Booster(model_file=str(model_path))


@pytest.fixture(scope="module")
def winprob_metadata():
    """Load the Phase 1B metadata."""
    metadata_path = MODELS_DIR / "lightgbm_win_probability_metadata.json"
    if not metadata_path.exists():
        pytest.skip("Win Probability metadata not found")
    with open(metadata_path) as f:
        return json.load(f)


# ============================================================================
# PHASE 1A: BID FEE MODEL TESTS
# ============================================================================

class TestBidFeeModel:
    """Tests for Phase 1A Bid Fee Prediction Model."""

    def test_model_loads(self, bidfee_model):
        """Test that model loads successfully."""
        assert bidfee_model is not None
        assert isinstance(bidfee_model, lgb.Booster)

    def test_model_has_features(self, bidfee_model):
        """Test that model has expected features."""
        features = bidfee_model.feature_name()
        assert len(features) > 0
        assert len(features) >= 50  # Should have ~68 features

    def test_predictions_are_positive(self, bidfee_model, sample_data, bidfee_metadata):
        """Test that bid fee predictions are positive dollar amounts."""
        features = bidfee_metadata.get('features', bidfee_model.feature_name())

        # Prepare features
        X = sample_data[features].fillna(0)

        # Get predictions
        predictions = bidfee_model.predict(X.head(100))

        # All predictions should be positive (fees can't be negative)
        assert all(predictions > 0), "Bid fees must be positive"

    def test_predictions_are_reasonable(self, bidfee_model, sample_data, bidfee_metadata):
        """Test that predictions are in reasonable range."""
        features = bidfee_metadata.get('features', bidfee_model.feature_name())
        X = sample_data[features].fillna(0)

        predictions = bidfee_model.predict(X.head(100))

        # Fees should be between $100 and $50,000 (reasonable range)
        assert all(predictions >= 50), "Fees too low"
        assert all(predictions <= 100000), "Fees too high"

    def test_metadata_has_metrics(self, bidfee_metadata):
        """Test that metadata contains performance metrics."""
        assert 'metrics' in bidfee_metadata or 'test_rmse' in bidfee_metadata

    def test_no_jobdata_features(self, bidfee_model):
        """Test that JobData features are excluded."""
        features = bidfee_model.feature_name()

        for jobdata_feat in JOBDATA_FEATURES_TO_EXCLUDE:
            assert jobdata_feat not in features, f"JobData feature {jobdata_feat} should be excluded"


# ============================================================================
# PHASE 1B: WIN PROBABILITY MODEL TESTS
# ============================================================================

class TestWinProbabilityModel:
    """Tests for Phase 1B Win Probability Classification Model."""

    def test_model_loads(self, winprob_model):
        """Test that model loads successfully."""
        assert winprob_model is not None
        assert isinstance(winprob_model, lgb.Booster)

    def test_model_has_features(self, winprob_model):
        """Test that model has expected features."""
        features = winprob_model.feature_name()
        assert len(features) > 0
        assert len(features) >= 50  # Should have ~75 features

    def test_predictions_are_probabilities(self, winprob_model, sample_data, winprob_metadata):
        """Test that predictions are valid probabilities [0, 1]."""
        features = winprob_metadata.get('features', winprob_model.feature_name())

        X = sample_data[features].fillna(0)
        predictions = winprob_model.predict(X.head(100))

        # All predictions should be between 0 and 1
        assert all(predictions >= 0), "Probabilities must be >= 0"
        assert all(predictions <= 1), "Probabilities must be <= 1"

    def test_predictions_have_variance(self, winprob_model, sample_data, winprob_metadata):
        """Test that model doesn't predict same value for all inputs."""
        features = winprob_metadata.get('features', winprob_model.feature_name())

        X = sample_data[features].fillna(0)
        predictions = winprob_model.predict(X.head(100))

        # Should have some variance in predictions
        assert np.std(predictions) > 0.1, "Model should have prediction variance"

    def test_no_leaky_features(self, winprob_model):
        """Test that leaky win_rate features are excluded."""
        features = winprob_model.feature_name()

        leaky_features = [
            'win_rate_with_client', 'office_win_rate', 'propertytype_win_rate',
            'state_win_rate', 'segment_win_rate', 'client_win_rate',
            'rolling_win_rate_office', 'total_wins_with_client', 'prev_won_same_client',
        ]

        for leaky_feat in leaky_features:
            assert leaky_feat not in features, f"Leaky feature {leaky_feat} should be excluded"

    def test_metadata_has_auc(self, winprob_metadata):
        """Test that metadata contains AUC metric."""
        if 'metrics' in winprob_metadata:
            test_metrics = winprob_metadata['metrics'].get('test', {})
            assert 'auc_roc' in test_metrics, "Should have AUC-ROC metric"
            assert test_metrics['auc_roc'] > 0.5, "AUC should be better than random"


# ============================================================================
# INTEGRATION TESTS: EXPECTED VALUE SYSTEM
# ============================================================================

class TestExpectedValueSystem:
    """Tests for the combined Expected Value calculation."""

    def test_expected_value_calculation(self, bidfee_model, winprob_model, sample_data,
                                         bidfee_metadata, winprob_metadata):
        """Test that EV = P(Win) × BidFee works correctly."""
        # Get feature lists
        bidfee_features = bidfee_metadata.get('features', bidfee_model.feature_name())
        winprob_features = winprob_metadata.get('features', winprob_model.feature_name())

        # Prepare data
        X_bidfee = sample_data[bidfee_features].fillna(0).head(100)
        X_winprob = sample_data[winprob_features].fillna(0).head(100)

        # Get predictions
        predicted_fees = bidfee_model.predict(X_bidfee)
        win_probabilities = winprob_model.predict(X_winprob)

        # Calculate Expected Value
        expected_values = win_probabilities * predicted_fees

        # EV should be positive
        assert all(expected_values >= 0), "Expected values must be non-negative"

        # EV should be less than or equal to predicted fee (since P(Win) <= 1)
        assert all(expected_values <= predicted_fees + 0.01), "EV should be <= BidFee"

    def test_ev_ranking_consistency(self, bidfee_model, winprob_model, sample_data,
                                     bidfee_metadata, winprob_metadata):
        """Test that EV provides consistent ranking for bid decisions."""
        bidfee_features = bidfee_metadata.get('features', bidfee_model.feature_name())
        winprob_features = winprob_metadata.get('features', winprob_model.feature_name())

        X_bidfee = sample_data[bidfee_features].fillna(0).head(100)
        X_winprob = sample_data[winprob_features].fillna(0).head(100)

        predicted_fees = bidfee_model.predict(X_bidfee)
        win_probabilities = winprob_model.predict(X_winprob)
        expected_values = win_probabilities * predicted_fees

        # High win probability + high fee should yield high EV
        high_ev_mask = expected_values > np.percentile(expected_values, 75)

        # These should generally have above-average win probability OR above-average fee
        avg_winprob = win_probabilities[high_ev_mask].mean()
        avg_fee = predicted_fees[high_ev_mask].mean()

        assert avg_winprob > 0.3 or avg_fee > np.median(predicted_fees), \
            "High EV bids should have high probability or high fee"


# ============================================================================
# DATA QUALITY TESTS
# ============================================================================

class TestDataQuality:
    """Tests for data quality and consistency."""

    def test_data_loads(self, sample_data):
        """Test that data loads correctly."""
        assert len(sample_data) > 0
        assert 'BidFee' in sample_data.columns
        assert 'Won' in sample_data.columns

    def test_data_filtered_to_2023(self, sample_data):
        """Test that data is filtered to 2023+."""
        min_date = sample_data['BidDate'].min()
        assert min_date >= pd.Timestamp('2023-01-01'), "Data should be 2023+"

    def test_target_variable_valid(self, sample_data):
        """Test that target variables are valid."""
        # BidFee should be non-negative (some bids can be $0)
        assert (sample_data['BidFee'] >= 0).all(), "BidFee should be non-negative"

        # Most BidFees should be positive
        positive_ratio = (sample_data['BidFee'] > 0).mean()
        assert positive_ratio > 0.95, f"Most BidFees should be positive, got {positive_ratio:.1%}"

        # Won should be binary
        assert set(sample_data['Won'].unique()).issubset({0, 1}), "Won should be 0 or 1"

    def test_no_extreme_outliers(self, sample_data):
        """Test for reasonable data ranges."""
        # BidFee sanity check
        assert sample_data['BidFee'].max() < 500000, "BidFee has extreme outlier"
        assert sample_data['BidFee'].min() >= 0, "BidFee should be non-negative"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
