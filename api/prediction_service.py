"""
Prediction Service for Bid Recommendation System
=================================================
Production-ready prediction service that uses the trained LightGBM model.

This service:
1. Loads the trained model and feature statistics
2. Computes all necessary features from raw inputs
3. Returns predictions with confidence intervals

Usage:
    from api.prediction_service import BidPredictor

    predictor = BidPredictor()
    result = predictor.predict(
        business_segment="Financing",
        property_type="Multifamily",
        property_state="Illinois",
        target_time=30,
        ...
    )

Author: Global Stat Solutions
Date: 2026-01-29
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from config.model_config import (
    MODELS_DIR, FEATURES_DATA, REPORTS_DIR,
    DATA_START_DATE, USE_RECENT_DATA_ONLY,
    EXCLUDE_COLUMNS, JOBDATA_FEATURES_TO_EXCLUDE,
)

from api.empirical_bands import EmpiricalBandCalculator


class BidPredictor:
    """
    Production-ready bid fee prediction service.

    Loads the trained model and precomputes lookup tables for feature generation.
    """

    def __init__(self):
        """Initialize predictor with model and feature statistics."""
        self.model = None
        self.model_features = None
        self.feature_stats = {}
        self.segments = []
        self.property_types = []
        self.states = []
        self.offices = []
        self.band_calculator = EmpiricalBandCalculator()

        self._load_model()
        self._compute_feature_statistics()
        self._load_empirical_bands()

    def _load_model(self):
        """Load the trained LightGBM model."""
        model_path = MODELS_DIR / "lightgbm_bidfee_model.txt"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")

        self.model = lgb.Booster(model_file=str(model_path))
        self.model_features = self.model.feature_name()

        print(f"[BidPredictor] Model loaded: {self.model.num_trees()} trees, {len(self.model_features)} features")

    def _compute_feature_statistics(self):
        """Precompute feature statistics from training data for lookup."""
        print("[BidPredictor] Computing feature statistics...")

        df = pd.read_csv(FEATURES_DATA)
        df['BidDate'] = pd.to_datetime(df['BidDate'])

        # Filter to recent data
        if USE_RECENT_DATA_ONLY:
            df = df[df['BidDate'] >= pd.Timestamp(DATA_START_DATE)].copy()

        # Store unique values for dropdowns
        self.segments = sorted(df['BusinessSegment'].dropna().unique().tolist())
        self.property_types = sorted(df['PropertyType'].dropna().unique().tolist())
        self.states = sorted(df['PropertyState'].dropna().unique().tolist())
        self.offices = sorted(df['OfficeId'].dropna().unique().astype(int).tolist())

        # Compute aggregate statistics for feature generation
        # These are the lookup tables we'll use when a new bid comes in

        # Segment statistics
        self.feature_stats['segment'] = df.groupby('BusinessSegment').agg({
            'BidFee': ['mean', 'std', 'count'],
            'Won': 'mean',
            'TargetTime': 'mean',
        }).to_dict()

        # Flatten segment stats
        segment_stats = df.groupby('BusinessSegment')['BidFee'].agg(['mean', 'std', 'count'])
        segment_win_rate = df.groupby('BusinessSegment')['Won'].mean()
        self.feature_stats['segment_avg_fee'] = segment_stats['mean'].to_dict()
        self.feature_stats['segment_std_fee'] = segment_stats['std'].fillna(0).to_dict()
        self.feature_stats['segment_count'] = segment_stats['count'].to_dict()
        self.feature_stats['segment_win_rate'] = segment_win_rate.to_dict()

        # State statistics
        state_stats = df.groupby('PropertyState')['BidFee'].agg(['mean', 'std', 'count'])
        state_win_rate = df.groupby('PropertyState')['Won'].mean()
        self.feature_stats['state_avg_fee'] = state_stats['mean'].to_dict()
        self.feature_stats['state_std_fee'] = state_stats['std'].fillna(0).to_dict()
        self.feature_stats['state_frequency'] = state_stats['count'].to_dict()
        self.feature_stats['state_win_rate'] = state_win_rate.to_dict()

        # Office statistics
        office_stats = df.groupby('OfficeId')['BidFee'].agg(['mean', 'std'])
        office_win_rate = df.groupby('OfficeId')['Won'].mean()
        self.feature_stats['office_avg_fee'] = office_stats['mean'].to_dict()
        self.feature_stats['office_std_fee'] = office_stats['std'].fillna(0).to_dict()
        self.feature_stats['office_win_rate'] = office_win_rate.to_dict()

        # Property type statistics
        proptype_stats = df.groupby('PropertyType')['BidFee'].agg(['mean', 'std', 'count'])
        proptype_win_rate = df.groupby('PropertyType')['Won'].mean()
        self.feature_stats['propertytype_avg_fee'] = proptype_stats['mean'].to_dict()
        self.feature_stats['propertytype_std_fee'] = proptype_stats['std'].fillna(0).to_dict()
        self.feature_stats['PropertyType_frequency'] = proptype_stats['count'].to_dict()
        self.feature_stats['propertytype_win_rate'] = proptype_win_rate.to_dict()

        # Global statistics (for defaults)
        self.feature_stats['global_avg_fee'] = df['BidFee'].mean()
        self.feature_stats['global_std_fee'] = df['BidFee'].std()
        self.feature_stats['global_win_rate'] = df['Won'].mean()

        # State-segment combinations
        combo_stats = df.groupby(['PropertyState', 'BusinessSegment']).size()
        self.feature_stats['state_segment_combo_freq'] = combo_stats.to_dict()

        print(f"[BidPredictor] Statistics computed for {len(self.segments)} segments, {len(self.states)} states")

    def _load_empirical_bands(self):
        """Load precomputed empirical confidence bands."""
        bands_path = REPORTS_DIR / 'empirical_bands.json'
        if bands_path.exists():
            loaded = self.band_calculator.load_bands(bands_path)
            if loaded:
                print("[BidPredictor] Empirical bands loaded successfully")
            else:
                print("[BidPredictor] Warning: Could not load empirical bands, using defaults")
        else:
            print("[BidPredictor] Note: No empirical bands file found, using default intervals")
            print(f"         Run 'python api/empirical_bands.py' to compute bands")

    def _generate_features(
        self,
        business_segment: str,
        property_type: str,
        property_state: str,
        target_time: int,
        office_id: Optional[int] = None,
        distance_km: float = 0,
        on_due_date: int = 0,
        client_history: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Generate all features needed for prediction from raw inputs.

        This mimics what the feature engineering pipeline does, but for a single bid.
        """
        features = {}

        # Raw features
        features['TargetTime'] = target_time
        features['DistanceInKM'] = distance_km
        features['OnDueDate'] = on_due_date

        # Segment features
        features['segment_avg_fee'] = self.feature_stats['segment_avg_fee'].get(
            business_segment, self.feature_stats['global_avg_fee']
        )
        features['segment_std_fee'] = self.feature_stats['segment_std_fee'].get(
            business_segment, self.feature_stats['global_std_fee']
        )
        features['segment_win_rate'] = self.feature_stats['segment_win_rate'].get(
            business_segment, self.feature_stats['global_win_rate']
        )
        features['segment_bid_density'] = self.feature_stats['segment_count'].get(business_segment, 100)

        # State features
        features['state_avg_fee'] = self.feature_stats['state_avg_fee'].get(
            property_state, self.feature_stats['global_avg_fee']
        )
        features['state_win_rate'] = self.feature_stats['state_win_rate'].get(
            property_state, self.feature_stats['global_win_rate']
        )
        features['PropertyState_frequency'] = self.feature_stats['state_frequency'].get(property_state, 100)

        # Property type features
        features['propertytype_avg_fee'] = self.feature_stats['propertytype_avg_fee'].get(
            property_type, self.feature_stats['global_avg_fee']
        )
        features['propertytype_win_rate'] = self.feature_stats['propertytype_win_rate'].get(
            property_type, self.feature_stats['global_win_rate']
        )
        features['PropertyType_frequency'] = self.feature_stats['PropertyType_frequency'].get(property_type, 100)

        # Office features
        if office_id and office_id in self.feature_stats['office_avg_fee']:
            features['office_avg_fee'] = self.feature_stats['office_avg_fee'][office_id]
            features['office_std_fee'] = self.feature_stats['office_std_fee'].get(office_id, 0)
            features['office_win_rate'] = self.feature_stats['office_win_rate'].get(
                office_id, self.feature_stats['global_win_rate']
            )
        else:
            features['office_avg_fee'] = self.feature_stats['global_avg_fee']
            features['office_std_fee'] = self.feature_stats['global_std_fee']
            features['office_win_rate'] = self.feature_stats['global_win_rate']

        # Rolling features (use segment averages as proxy for new bids)
        features['rolling_avg_fee_segment'] = features['segment_avg_fee']
        features['rolling_avg_fee_office'] = features['office_avg_fee']
        features['rolling_std_fee_segment'] = features['segment_std_fee']

        # Client features (if history provided)
        if client_history:
            features['client_avg_fee'] = client_history.get('avg_fee', features['segment_avg_fee'])
            features['client_std_fee'] = client_history.get('std_fee', 0)
            features['client_win_rate'] = client_history.get('win_rate', self.feature_stats['global_win_rate'])
            features['cumulative_bids_client'] = client_history.get('total_bids', 0)
            features['cumulative_wins_client'] = client_history.get('total_wins', 0)
            features['cumulative_winrate_client'] = (
                features['cumulative_wins_client'] / max(features['cumulative_bids_client'], 1)
            )
            features['lag1_bidfee_client'] = client_history.get('last_bid_fee', 0)
            features['lag2_bidfee_client'] = client_history.get('second_last_bid_fee', 0)
            features['lag1_targettime_client'] = client_history.get('last_target_time', target_time)
            features['client_bid_count'] = client_history.get('total_bids', 0)
        else:
            # New client defaults
            features['client_avg_fee'] = features['segment_avg_fee']
            features['client_std_fee'] = 0
            features['client_win_rate'] = self.feature_stats['global_win_rate']
            features['cumulative_bids_client'] = 0
            features['cumulative_wins_client'] = 0
            features['cumulative_winrate_client'] = 0
            features['lag1_bidfee_client'] = 0
            features['lag2_bidfee_client'] = 0
            features['lag1_targettime_client'] = 0
            features['client_bid_count'] = 0

        # Competitiveness features (will be computed relative to segment)
        # These are computed AFTER we know the predicted fee
        features['bid_vs_segment_ratio'] = 1.0  # Placeholder
        features['bid_vs_client_ratio'] = 1.0
        features['bid_vs_state_ratio'] = 1.0
        features['fee_diff_from_segment'] = 0
        features['fee_percentile_segment'] = 0.5

        # Temporal features
        now = datetime.now()
        features['month'] = now.month
        features['quarter'] = (now.month - 1) // 3 + 1
        features['day_of_week'] = now.weekday()
        features['is_month_end'] = 1 if now.day > 25 else 0
        features['is_quarter_end'] = 1 if now.month in [3, 6, 9, 12] and now.day > 25 else 0

        # Market dynamics
        features['state_segment_combo_freq'] = self.feature_stats['state_segment_combo_freq'].get(
            (property_state, business_segment), 10
        )
        features['client_segment_specialization'] = 0.5  # Default for new clients
        features['office_workload_30d'] = 5  # Default workload
        features['days_since_last_bid_client'] = 30  # Default
        features['days_since_last_bid_segment'] = 1  # Segments always active
        features['bid_count_last_30d'] = 10  # Default market activity

        # Risk features
        features['segment_cv_fee'] = (
            features['segment_std_fee'] / max(features['segment_avg_fee'], 1)
        )
        features['client_fee_consistency'] = 1.0  # Default for new clients
        features['targettime_vs_segment'] = target_time / max(
            self.feature_stats.get('segment_avg_time', {}).get(business_segment, 30), 1
        )
        features['segment_fee_range'] = features['segment_std_fee'] * 3  # Approximate range

        # Interaction features
        features['segment_fee_x_time'] = features['segment_avg_fee'] * target_time
        features['client_fee_x_segment_std'] = features['client_avg_fee'] * features['segment_std_fee']
        features['office_fee_x_state_freq'] = features['office_avg_fee'] * features['PropertyState_frequency']
        features['competitiveness_x_winrate'] = features['bid_vs_segment_ratio'] * features['cumulative_winrate_client']
        features['time_x_density'] = target_time * features['segment_bid_density']

        # Categorical encodings
        features['BusinessSegment_frequency'] = features['segment_bid_density']

        return features

    def predict(
        self,
        business_segment: str,
        property_type: str,
        property_state: str,
        target_time: int,
        office_id: Optional[int] = None,
        distance_km: float = 0,
        on_due_date: int = 0,
        client_history: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Predict bid fee for a new opportunity.

        Parameters:
        -----------
        business_segment : str
            Business segment (e.g., "Financing", "Consulting")
        property_type : str
            Property type (e.g., "Multifamily", "Industrial")
        property_state : str
            State where property is located (e.g., "Illinois", "Texas")
        target_time : int
            Days to complete the appraisal
        office_id : int, optional
            Office handling the bid
        distance_km : float
            Distance to property in kilometers
        on_due_date : int
            Whether delivery is on due date (1) or before (0)
        client_history : dict, optional
            Client's historical data {avg_fee, total_bids, total_wins, last_bid_fee, etc.}

        Returns:
        --------
        dict : Prediction results
            {
                "predicted_fee": float,
                "confidence_interval": {"low": float, "high": float},
                "confidence_level": str,
                "factors": dict,
                "segment_benchmark": float,
                "recommendation": str,
            }
        """
        # Generate features
        features = self._generate_features(
            business_segment=business_segment,
            property_type=property_type,
            property_state=property_state,
            target_time=target_time,
            office_id=office_id,
            distance_km=distance_km,
            on_due_date=on_due_date,
            client_history=client_history,
        )

        # Create feature vector in correct order
        feature_vector = []
        for feat_name in self.model_features:
            if feat_name in features:
                feature_vector.append(features[feat_name])
            else:
                feature_vector.append(0)  # Default for missing features

        # Make prediction
        X = np.array([feature_vector])
        prediction = self.model.predict(X)[0]

        # Ensure positive prediction
        prediction = max(prediction, 500)  # Minimum $500 fee

        # Calculate confidence interval using empirical bands (stratified by fee bucket)
        low, high, band_metadata = self.band_calculator.get_confidence_interval(
            predicted_fee=prediction,
            segment=business_segment,
            state=property_state,
            confidence_level=0.80,  # 80% interval
        )

        # Confidence level based on data availability
        segment_count = self.feature_stats['segment_count'].get(business_segment, 0)
        state_freq = self.feature_stats['state_frequency'].get(property_state, 0)

        if segment_count > 1000 and state_freq > 500:
            confidence = "high"
        elif segment_count > 100 and state_freq > 50:
            confidence = "medium"
        else:
            confidence = "low"

        # Segment benchmark
        segment_avg = self.feature_stats['segment_avg_fee'].get(
            business_segment, self.feature_stats['global_avg_fee']
        )

        # Generate recommendation
        diff_pct = ((prediction - segment_avg) / segment_avg) * 100
        if diff_pct > 10:
            recommendation = f"Predicted fee is {diff_pct:.1f}% above segment average. Consider competitive positioning."
        elif diff_pct < -10:
            recommendation = f"Predicted fee is {abs(diff_pct):.1f}% below segment average. Good competitive position."
        else:
            recommendation = "Predicted fee is within normal range for this segment."

        return {
            "predicted_fee": round(prediction, 2),
            "confidence_interval": {
                "low": round(low, 2),
                "high": round(high, 2),
            },
            "confidence_level": confidence,
            "segment_benchmark": round(segment_avg, 2),
            "state_benchmark": round(
                self.feature_stats['state_avg_fee'].get(property_state, segment_avg), 2
            ),
            "recommendation": recommendation,
            "factors": {
                "segment_effect": round(features['segment_avg_fee'], 2),
                "state_effect": round(features['state_avg_fee'], 2),
                "office_effect": round(features['office_avg_fee'], 2),
                "time_factor": target_time,
                "client_history_effect": round(features['client_avg_fee'], 2) if client_history else None,
            },
            "metadata": {
                "model_version": "1.0",
                "prediction_date": datetime.now().isoformat(),
                "data_coverage": {
                    "segment_samples": segment_count,
                    "state_samples": state_freq,
                }
            }
        }

    def get_dropdown_options(self) -> Dict[str, List]:
        """Return options for UI dropdowns."""
        return {
            "segments": self.segments,
            "property_types": self.property_types,
            "states": self.states,
            "offices": self.offices[:100],  # Limit for UI
        }

    def get_segment_stats(self, segment: str) -> Dict:
        """Get statistics for a specific segment."""
        return {
            "avg_fee": self.feature_stats['segment_avg_fee'].get(segment, 0),
            "std_fee": self.feature_stats['segment_std_fee'].get(segment, 0),
            "win_rate": self.feature_stats['segment_win_rate'].get(segment, 0),
            "count": self.feature_stats['segment_count'].get(segment, 0),
        }


# Singleton instance for API
_predictor_instance = None

def get_predictor() -> BidPredictor:
    """Get or create the singleton predictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = BidPredictor()
    return _predictor_instance


if __name__ == "__main__":
    # Test the predictor
    print("\n" + "=" * 60)
    print("TESTING PREDICTION SERVICE")
    print("=" * 60)

    predictor = BidPredictor()

    # Test prediction
    result = predictor.predict(
        business_segment="Financing",
        property_type="Multifamily",
        property_state="Illinois",
        target_time=30,
        distance_km=50,
    )

    print(f"\nTest Prediction:")
    print(f"  Predicted Fee: ${result['predicted_fee']:,.2f}")
    print(f"  Confidence Interval: ${result['confidence_interval']['low']:,.2f} - ${result['confidence_interval']['high']:,.2f}")
    print(f"  Confidence Level: {result['confidence_level']}")
    print(f"  Segment Benchmark: ${result['segment_benchmark']:,.2f}")
    print(f"  Recommendation: {result['recommendation']}")

    # Test with client history
    result2 = predictor.predict(
        business_segment="Consulting",
        property_type="Office",
        property_state="Texas",
        target_time=45,
        client_history={
            "avg_fee": 5500,
            "total_bids": 25,
            "total_wins": 12,
            "last_bid_fee": 5200,
        }
    )

    print(f"\nTest Prediction with Client History:")
    print(f"  Predicted Fee: ${result2['predicted_fee']:,.2f}")
    print(f"  Confidence: {result2['confidence_level']}")
