"""
Prediction Service for Bid Recommendation System
=================================================
Production-ready prediction service that uses the trained LightGBM model.

This service:
1. Loads the trained model and feature statistics
2. Computes all necessary features from raw inputs
3. Returns predictions with confidence intervals
4. Predicts win probability with fee-sensitivity adjustment
5. Computes Expected Value (EV = P(Win) x Fee) for bid optimization

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
    PREDICTION_CONFIG,
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
        self.win_prob_model = None
        self.win_prob_features = None
        self.feature_stats = {}
        self.feature_defaults = {}  # Segment-specific defaults for missing features
        self.segments = []
        self.property_types = []
        self.states = []
        self.offices = []
        self.band_calculator = EmpiricalBandCalculator()

        self._load_model()
        self._load_win_probability_model()
        self._compute_feature_statistics()
        self._load_empirical_bands()
        self._load_feature_defaults()

    def _load_model(self):
        """Load the trained LightGBM model."""
        model_path = MODELS_DIR / "lightgbm_bidfee_model.txt"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")

        self.model = lgb.Booster(model_file=str(model_path))
        self.model_features = self.model.feature_name()

        print(f"[BidPredictor] Model loaded: {self.model.num_trees()} trees, {len(self.model_features)} features")

    def _load_win_probability_model(self):
        """Load the trained LightGBM win probability model."""
        model_path = MODELS_DIR / "lightgbm_win_probability.txt"
        metadata_path = MODELS_DIR / "lightgbm_win_probability_metadata.json"

        if not model_path.exists():
            print(f"[BidPredictor] Warning: Win probability model not found at {model_path}")
            print(f"[BidPredictor] Win probability predictions will use fallback heuristic")
            return

        self.win_prob_model = lgb.Booster(model_file=str(model_path))
        self.win_prob_features = self.win_prob_model.feature_name()

        # Load metadata for feature info
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.win_prob_metadata = json.load(f)
            print(f"[BidPredictor] Win probability model loaded: AUC={self.win_prob_metadata['metrics']['test']['auc_roc']:.4f}")
        else:
            print(f"[BidPredictor] Win probability model loaded: {self.win_prob_model.num_trees()} trees")

    def _compute_feature_statistics(self):
        """Load pre-computed feature statistics from JSON file."""
        print("[BidPredictor] Loading pre-computed feature statistics...")

        # Try to load from pre-computed JSON file (for production/Render deployment)
        precomputed_path = REPORTS_DIR / "api_precomputed_stats.json"

        if precomputed_path.exists():
            with open(precomputed_path, 'r') as f:
                stats = json.load(f)

            # Load dropdown options
            self.segments = stats['segments']
            self.property_types = stats['property_types']
            self.states = stats['states']
            self.offices = stats['offices']

            # Load global statistics
            self.feature_stats['global_avg_fee'] = stats['global_avg_fee']
            self.feature_stats['global_std_fee'] = stats['global_std_fee']
            self.feature_stats['global_win_rate'] = stats['global_win_rate']

            # Load segment statistics
            self.feature_stats['segment_avg_fee'] = stats['segment_avg_fee']
            self.feature_stats['segment_std_fee'] = stats['segment_std_fee']
            self.feature_stats['segment_count'] = {k: int(v) for k, v in stats['segment_count'].items()}
            self.feature_stats['segment_win_rate'] = stats['segment_win_rate']
            self.feature_stats['segment_frequency'] = stats['segment_frequency']

            # Load state statistics
            self.feature_stats['state_avg_fee'] = stats['state_avg_fee']
            self.feature_stats['state_std_fee'] = stats['state_std_fee']
            self.feature_stats['state_win_rate'] = stats['state_win_rate']
            self.feature_stats['state_frequency'] = stats['state_frequency']
            self.feature_stats['state_count'] = {k: int(v) for k, v in stats.get('state_count', {}).items()}

            # Load property type statistics
            self.feature_stats['propertytype_avg_fee'] = stats['propertytype_avg_fee']
            self.feature_stats['propertytype_std_fee'] = stats['propertytype_std_fee']
            self.feature_stats['propertytype_win_rate'] = stats['propertytype_win_rate']
            self.feature_stats['PropertyType_frequency'] = stats['PropertyType_frequency']

            # Load office statistics
            self.feature_stats['office_avg_fee'] = {int(k): v for k, v in stats['office_avg_fee'].items()}
            self.feature_stats['office_std_fee'] = {int(k): v for k, v in stats['office_std_fee'].items()}
            self.feature_stats['office_win_rate'] = {int(k): v for k, v in stats['office_win_rate'].items()}

            # Load state-segment combo frequency
            self.feature_stats['state_segment_combo_freq'] = {
                tuple(k.split('|')): v for k, v in stats['state_segment_combo_freq'].items()
            }

            print(f"[BidPredictor] Loaded stats for {len(self.segments)} segments, {len(self.states)} states")
            return

        # Fallback: compute from CSV if available (for local development)
        print("[BidPredictor] Pre-computed stats not found, trying to load from CSV...")

        if not FEATURES_DATA.exists():
            raise FileNotFoundError(
                f"Neither pre-computed stats nor feature data found. "
                f"Please run: python -c \"from api.prediction_service import *\" locally first, "
                f"or ensure {REPORTS_DIR / 'api_precomputed_stats.json'} exists."
            )

        df_all = pd.read_csv(FEATURES_DATA)
        df_all['BidDate'] = pd.to_datetime(df_all['BidDate'])
        df_full = df_all.copy()

        if USE_RECENT_DATA_ONLY:
            df = df_all[df_all['BidDate'] >= pd.Timestamp(DATA_START_DATE)].copy()
        else:
            df = df_all.copy()

        self.segments = sorted(df_full['BusinessSegment'].dropna().unique().tolist())
        self.property_types = sorted(df_full['PropertyType'].dropna().unique().tolist())
        self.states = sorted(df_full['PropertyState'].dropna().unique().tolist())
        self.offices = sorted(df_full['OfficeId'].dropna().unique().astype(int).tolist())

        self.feature_stats['global_avg_fee'] = df['BidFee'].mean()
        self.feature_stats['global_std_fee'] = df['BidFee'].std()
        self.feature_stats['global_win_rate'] = df['Won'].mean()

        total_count = len(df_full)
        segment_stats = df_full.groupby('BusinessSegment')['BidFee'].agg(['mean', 'std', 'count'])
        self.feature_stats['segment_avg_fee'] = segment_stats['mean'].to_dict()
        self.feature_stats['segment_std_fee'] = segment_stats['std'].fillna(0).to_dict()
        self.feature_stats['segment_count'] = segment_stats['count'].to_dict()
        self.feature_stats['segment_win_rate'] = df_full.groupby('BusinessSegment')['Won'].mean().to_dict()
        self.feature_stats['segment_frequency'] = (df_full.groupby('BusinessSegment').size() / total_count).to_dict()

        state_stats = df_full.groupby('PropertyState')['BidFee'].agg(['mean', 'std'])
        self.feature_stats['state_avg_fee'] = state_stats['mean'].to_dict()
        self.feature_stats['state_std_fee'] = state_stats['std'].fillna(0).to_dict()
        self.feature_stats['state_win_rate'] = df_full.groupby('PropertyState')['Won'].mean().to_dict()
        self.feature_stats['state_frequency'] = (df_full.groupby('PropertyState').size() / total_count).to_dict()
        self.feature_stats['state_count'] = df_full.groupby('PropertyState').size().to_dict()

        office_stats = df_full.groupby('OfficeId')['BidFee'].agg(['mean', 'std'])
        self.feature_stats['office_avg_fee'] = office_stats['mean'].to_dict()
        self.feature_stats['office_std_fee'] = office_stats['std'].fillna(0).to_dict()
        self.feature_stats['office_win_rate'] = df_full.groupby('OfficeId')['Won'].mean().to_dict()

        proptype_stats = df_full.groupby('PropertyType')['BidFee'].agg(['mean', 'std'])
        self.feature_stats['propertytype_avg_fee'] = proptype_stats['mean'].to_dict()
        self.feature_stats['propertytype_std_fee'] = proptype_stats['std'].fillna(0).to_dict()
        self.feature_stats['propertytype_win_rate'] = df_full.groupby('PropertyType')['Won'].mean().to_dict()
        self.feature_stats['PropertyType_frequency'] = (df_full.groupby('PropertyType').size() / total_count).to_dict()

        combo_stats = df.groupby(['PropertyState', 'BusinessSegment']).size()
        self.feature_stats['state_segment_combo_freq'] = combo_stats.to_dict()

        print(f"[BidPredictor] Computed stats for {len(self.segments)} segments, {len(self.states)} states")

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

    def _load_feature_defaults(self):
        """Load segment-specific default values for features."""
        defaults_path = REPORTS_DIR / 'feature_defaults.json'
        if defaults_path.exists():
            with open(defaults_path, 'r') as f:
                self.feature_defaults = json.load(f)
            print(f"[BidPredictor] Feature defaults loaded: {len(self.feature_defaults)} features")
        else:
            print("[BidPredictor] Warning: No feature defaults found, predictions may be inaccurate")
            print(f"         Run the feature defaults generation script first")

        # Load state-specific coordinates
        coords_path = REPORTS_DIR / 'state_coordinates.json'
        if coords_path.exists():
            with open(coords_path, 'r') as f:
                coords = json.load(f)
                self.feature_stats['state_latitude'] = coords.get('state_latitude', {})
                self.feature_stats['state_longitude'] = coords.get('state_longitude', {})
                self.feature_stats['state_distance_km'] = coords.get('state_distance_km', {})

        # Load rolling stats
        rolling_path = REPORTS_DIR / 'rolling_stats.json'
        if rolling_path.exists():
            with open(rolling_path, 'r') as f:
                rolling = json.load(f)
                self.feature_stats['office_rolling_bid_count'] = rolling.get('office_rolling_bid_count', {})
                self.feature_stats['office_rolling_std_fee'] = rolling.get('office_rolling_std_fee', {})
                self.feature_stats['office_rolling_win_rate'] = rolling.get('office_rolling_win_rate', {})
                self.feature_stats['segment_target_time'] = rolling.get('segment_target_time', {})
                self.feature_stats['proptype_target_time'] = rolling.get('proptype_target_time', {})
                self.feature_stats['global_rolling_bid_count'] = rolling.get('global_rolling_bid_count', 226)
                self.feature_stats['state_distance_miles'] = rolling.get('state_distance_miles', {})
                self.feature_stats['propertytype_std_fee'] = rolling.get('propertytype_std_fee', {})
                self.feature_stats['state_std_fee_median'] = rolling.get('state_std_fee', {})

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
        features['TargetTime_Original'] = target_time  # Model uses this too

        # Distance - use state median if not provided (0 is unrealistic)
        if distance_km > 0:
            features['DistanceInKM'] = distance_km
        else:
            features['DistanceInKM'] = self.feature_stats.get('state_distance_km', {}).get(property_state, 35)

        features['OnDueDate'] = on_due_date

        # State-specific location (for location-based features)
        features['RooftopLatitude'] = self.feature_stats.get('state_latitude', {}).get(property_state, 35.0)
        features['RooftopLongitude'] = self.feature_stats.get('state_longitude', {}).get(property_state, -95.0)

        # Distance in miles (model uses this for distance_x_volume)
        if distance_km > 0:
            features['DistanceInMiles'] = distance_km * 0.621371
        else:
            features['DistanceInMiles'] = self.feature_stats.get('state_distance_miles', {}).get(property_state, 25)

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
        # BusinessSegment_frequency is a PROPORTION (0-1), not raw count
        # Model was trained on proportions
        features['BusinessSegment_frequency'] = self.feature_stats['segment_frequency'].get(business_segment, 0.1)

        # State features
        features['state_avg_fee'] = self.feature_stats['state_avg_fee'].get(
            property_state, self.feature_stats['global_avg_fee']
        )
        features['state_win_rate'] = self.feature_stats['state_win_rate'].get(
            property_state, self.feature_stats['global_win_rate']
        )
        # PropertyState_frequency is a PROPORTION (0-1), not raw count
        # Model was trained on proportions
        features['PropertyState_frequency'] = self.feature_stats['state_frequency'].get(property_state, 0.05)

        # Property type features
        features['propertytype_avg_fee'] = self.feature_stats['propertytype_avg_fee'].get(
            property_type, self.feature_stats['global_avg_fee']
        )
        features['propertytype_win_rate'] = self.feature_stats['propertytype_win_rate'].get(
            property_type, self.feature_stats['global_win_rate']
        )
        # PropertyType_frequency is a PROPORTION (0-1), not raw count
        # Model was trained on proportions
        features['PropertyType_frequency'] = self.feature_stats['PropertyType_frequency'].get(property_type, 0.1)

        # Target time ratio to property type median
        proptype_median_time = self.feature_stats.get('proptype_target_time', {}).get(property_type, 21)
        features['targettime_ratio_to_proptype'] = target_time / max(proptype_median_time, 1)

        # Property type standard deviation
        features['propertytype_std_fee'] = self.feature_stats.get('propertytype_std_fee', {}).get(
            property_type, features['segment_std_fee']
        )

        # State standard deviation
        features['state_std_fee'] = self.feature_stats.get('state_std_fee_median', {}).get(
            property_state, features['segment_std_fee']
        )

        # Office features
        if office_id and office_id in self.feature_stats['office_avg_fee']:
            features['office_avg_fee'] = self.feature_stats['office_avg_fee'][office_id]
            features['office_std_fee'] = self.feature_stats['office_std_fee'].get(office_id, 0)
            features['office_win_rate'] = self.feature_stats['office_win_rate'].get(
                office_id, self.feature_stats['global_win_rate']
            )
            # Rolling stats for this office
            features['rolling_bid_count_office'] = self.feature_stats.get('office_rolling_bid_count', {}).get(
                office_id, self.feature_stats.get('global_rolling_bid_count', 226)
            )
            features['rolling_std_fee_office'] = self.feature_stats.get('office_rolling_std_fee', {}).get(
                office_id, features['office_std_fee']
            )
            features['rolling_win_rate_office'] = self.feature_stats.get('office_rolling_win_rate', {}).get(
                office_id, features['office_win_rate']
            )
        else:
            features['office_avg_fee'] = self.feature_stats['global_avg_fee']
            features['office_std_fee'] = self.feature_stats['global_std_fee']
            features['office_win_rate'] = self.feature_stats['global_win_rate']
            features['rolling_bid_count_office'] = self.feature_stats.get('global_rolling_bid_count', 226)
            features['rolling_std_fee_office'] = features['office_std_fee']
            features['rolling_win_rate_office'] = features['office_win_rate']

        # Computed feature: distance_x_volume = DistanceInMiles * rolling_bid_count_office
        features['distance_x_volume'] = features['DistanceInMiles'] * features['rolling_bid_count_office']

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
            # New client defaults - use segment statistics as reasonable defaults
            features['client_avg_fee'] = features['segment_avg_fee']
            features['client_std_fee'] = features['segment_std_fee'] * 0.7  # New clients typically have less variance
            features['client_win_rate'] = self.feature_stats['global_win_rate']
            features['cumulative_bids_client'] = 5  # Assume typical client
            features['cumulative_wins_client'] = 2  # Typical win rate
            features['cumulative_winrate_client'] = 0.4  # Typical win rate
            features['lag1_bidfee_client'] = features['segment_avg_fee']  # Assume previous bid was average
            features['lag2_bidfee_client'] = features['segment_avg_fee']
            features['lag1_targettime_client'] = target_time
            features['client_bid_count'] = 5

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

        # Win probability model features (with realistic defaults based on historical medians)
        features['IECount'] = 0  # Median is 0
        features['LeaseCount'] = 0  # Median is 0
        features['SaleCount'] = 0  # Median is 0
        features['market_competitiveness'] = 142  # Historical median
        features['building_size_numeric'] = 1  # Typical property
        features['client_relationship_strength'] = 65  # Historical median
        features['targettime_x_size'] = target_time * features['building_size_numeric']

        # Additional demographic features (use reasonable defaults)
        features['PopulationEstimate'] = 50000  # Typical metro area
        features['AverageHouseValue'] = 350000  # US median
        features['IncomePerHousehold'] = 75000  # US median
        features['MedianAge'] = 38  # US median
        features['DeliveryTotal'] = 1  # Standard delivery
        features['NumberofBusinesses'] = 500  # Typical area
        features['NumberofEmployees'] = 5000  # Typical area
        features['ZipPopulation'] = 30000  # Typical zip

        # Temporal encoding features
        features['Year'] = now.year
        features['Month'] = now.month
        features['Quarter'] = features['quarter']
        features['DayOfWeek'] = features['day_of_week']
        features['DayOfMonth'] = now.day
        features['WeekOfYear'] = now.isocalendar()[1]
        features['Month_sin'] = np.sin(2 * np.pi * now.month / 12)
        features['Month_cos'] = np.cos(2 * np.pi * now.month / 12)
        features['DayOfWeek_sin'] = np.sin(2 * np.pi * now.weekday() / 7)
        features['DayOfWeek_cos'] = np.cos(2 * np.pi * now.weekday() / 7)
        features['days_since_start'] = (now - datetime(2023, 1, 1)).days
        features['is_peak_season'] = 1 if now.month in [3, 4, 5, 9, 10, 11] else 0
        features['is_weekday'] = 1 if now.weekday() < 5 else 0

        # Note: Categorical frequency encodings (BusinessSegment_frequency,
        # PropertyState_frequency, PropertyType_frequency) are already set above
        # as proportions (0-1), which is what the model expects.

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

        # Create feature vector in correct order, using segment-specific defaults
        feature_vector = []
        for feat_name in self.model_features:
            if feat_name in features:
                feature_vector.append(features[feat_name])
            elif feat_name in self.feature_defaults:
                # Use segment-specific median if available, else global median
                defaults = self.feature_defaults[feat_name]
                segment_key = f'segment_{business_segment}_median'
                if segment_key in defaults:
                    feature_vector.append(defaults[segment_key])
                else:
                    feature_vector.append(defaults.get('global_median', 0))
            else:
                feature_vector.append(0)  # Last resort default

        # Make prediction
        X = np.array([feature_vector])
        prediction = self.model.predict(X)[0]

        # Ensure positive prediction
        prediction = max(prediction, PREDICTION_CONFIG['min_fee'])

        # Calculate confidence interval using empirical bands (stratified by fee bucket)
        low, high, band_metadata = self.band_calculator.get_confidence_interval(
            predicted_fee=prediction,
            segment=business_segment,
            state=property_state,
            confidence_level=0.80,  # 80% interval
        )

        # Confidence level based on data availability AND confidence band width
        segment_count = self.feature_stats['segment_count'].get(business_segment, 0)
        state_count = self.feature_stats.get('state_count', {}).get(property_state, 0)

        # Data availability score
        cfg = PREDICTION_CONFIG
        if segment_count > cfg['confidence_segment_high'] and state_count > cfg['confidence_state_high']:
            data_confidence = "high"
        elif segment_count > cfg['confidence_segment_medium'] and state_count > cfg['confidence_state_medium']:
            data_confidence = "medium"
        else:
            data_confidence = "low"

        # Band width score: narrow band relative to prediction = higher confidence
        band_width = high - low
        band_ratio = band_width / max(prediction, 1)
        if band_ratio < cfg['band_ratio_high']:
            band_confidence = "high"
        elif band_ratio < cfg['band_ratio_medium']:
            band_confidence = "medium"
        else:
            band_confidence = "low"

        # Overall bid fee confidence = minimum of data and band confidence
        confidence_rank = {"low": 0, "medium": 1, "high": 2}
        rank_to_label = {0: "low", 1: "medium", 2: "high"}
        confidence = rank_to_label[min(confidence_rank[data_confidence], confidence_rank[band_confidence])]

        # Segment benchmark
        segment_avg = self.feature_stats['segment_avg_fee'].get(
            business_segment, self.feature_stats['global_avg_fee']
        )

        # Generate recommendation - ±10% of benchmark is considered normal
        diff_pct = ((prediction - segment_avg) / segment_avg) * 100
        if diff_pct > 10:
            recommendation = f"Predicted fee is {diff_pct:.1f}% above segment average. May be priced high for this market."
        elif diff_pct < -10:
            recommendation = f"Predicted fee is {abs(diff_pct):.1f}% below segment average. May be underpriced - verify inputs."
        else:
            recommendation = f"Predicted fee is within ±10% of segment average. Good competitive position."

        # Predict win probability using classification model
        win_prob_result = self.predict_win_probability(
            features=features,
            predicted_fee=prediction,
            segment_benchmark=segment_avg,
            bid_fee_confidence=confidence,
        )

        # Calculate expected value: EV = P(Win) × Bid Fee
        expected_value = win_prob_result['probability'] * prediction

        return {
            "predicted_fee": round(prediction, 2),
            "confidence_interval": {
                "low": round(low, 2),
                "high": round(high, 2),
            },
            "confidence_level": confidence,
            "win_probability": win_prob_result,
            "expected_value": round(expected_value, 2),
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
                "model_version": "1.1",
                "prediction_date": datetime.now().isoformat(),
                "data_coverage": {
                    "segment_samples": segment_count,
                    "state_samples": state_count,
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

    def predict_win_probability(
        self,
        features: Dict[str, float],
        predicted_fee: float,
        segment_benchmark: float,
        bid_fee_confidence: str = "medium",
    ) -> Dict[str, Any]:
        """
        Predict win probability using the trained classification model.

        Parameters:
        -----------
        features : dict
            Generated features from _generate_features
        predicted_fee : float
            The predicted bid fee
        segment_benchmark : float
            Average fee for the segment (for fallback heuristic)
        bid_fee_confidence : str
            Confidence level of the bid fee prediction ("low", "medium", "high").
            Win probability confidence is capped at this level.

        Returns:
        --------
        dict : Win probability results
            {
                "probability": float (0-1),
                "confidence": str,
                "model_used": str,
            }
        """
        # If no win probability model, use improved heuristic
        if self.win_prob_model is None:
            return self._fallback_win_probability(predicted_fee, segment_benchmark)

        # Build feature vector for win probability model
        feature_vector = []
        for feat_name in self.win_prob_features:
            if feat_name in features:
                feature_vector.append(features[feat_name])
            elif feat_name in self.feature_defaults:
                defaults = self.feature_defaults[feat_name]
                feature_vector.append(defaults.get('global_median', 0))
            else:
                feature_vector.append(0)

        # Predict probability
        X = np.array([feature_vector])
        raw_probability = self.win_prob_model.predict(X)[0]

        # Fee-sensitivity adjustment: model doesn't know the predicted fee,
        # so we adjust based on how competitive the fee is vs segment average.
        # ratio < 1 means below-average fee (more competitive) → boost win prob
        # ratio > 1 means above-average fee (less competitive) → penalize win prob
        ratio = predicted_fee / max(segment_benchmark, 1)
        k = PREDICTION_CONFIG['fee_sensitivity_k']
        fee_adjustment = 2.0 / (1.0 + np.exp(k * (ratio - 1.0)))

        probability = raw_probability * fee_adjustment

        # Clamp to valid range
        probability = max(PREDICTION_CONFIG['win_prob_min'], min(PREDICTION_CONFIG['win_prob_max'], probability))

        # Confidence based on how close to 0.5 (more extreme = more confident)
        distance_from_uncertain = abs(probability - 0.5)
        if distance_from_uncertain > 0.3:
            win_confidence = "high"
        elif distance_from_uncertain > 0.15:
            win_confidence = "medium"
        else:
            win_confidence = "low"

        # Cap win probability confidence at bid fee confidence level
        confidence_rank = {"low": 0, "medium": 1, "high": 2}
        rank_to_label = {0: "low", 1: "medium", 2: "high"}
        max_rank = confidence_rank[bid_fee_confidence]
        if confidence_rank[win_confidence] > max_rank:
            win_confidence = rank_to_label[max_rank]

        return {
            "probability": round(probability, 4),
            "probability_pct": round(probability * 100, 1),
            "confidence": win_confidence,
            "model_used": "LightGBM Classifier + fee adjustment (AUC: 0.88)",
        }

    def _fallback_win_probability(
        self,
        predicted_fee: float,
        segment_benchmark: float,
    ) -> Dict[str, Any]:
        """
        Fallback heuristic when win probability model is not available.
        Uses fee-to-benchmark ratio with smoother curve.
        """
        ratio = predicted_fee / max(segment_benchmark, 1)

        # Smooth sigmoid-like function instead of hard buckets
        # Lower ratio (more competitive) = higher win probability
        # probability = 1 / (1 + exp(k * (ratio - 1)))
        k = 5  # Steepness
        probability = 1 / (1 + np.exp(k * (ratio - 1)))

        # Scale to realistic range (20% - 75%)
        probability = 0.20 + (probability * 0.55)

        return {
            "probability": round(probability, 4),
            "probability_pct": round(probability * 100, 1),
            "confidence": "low",
            "model_used": "Heuristic (model not loaded)",
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

    print(f"\nTest 1: Financing / Multifamily / Illinois (common)")
    print(f"  Predicted Fee: ${result['predicted_fee']:,.2f}")
    print(f"  Confidence Interval: ${result['confidence_interval']['low']:,.2f} - ${result['confidence_interval']['high']:,.2f}")
    print(f"  Bid Fee Confidence: {result['confidence_level']}")
    print(f"  Win Probability: {result['win_probability']['probability_pct']}%")
    print(f"  Win Prob Confidence: {result['win_probability']['confidence']}")
    print(f"  Win Prob Model: {result['win_probability']['model_used']}")
    print(f"  Expected Value: ${result['expected_value']:,.2f}")
    print(f"  Segment Benchmark: ${result['segment_benchmark']:,.2f}")
    print(f"  Data Coverage: {result['metadata']['data_coverage']}")
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

    print(f"\nTest 2: Consulting / Office / Texas (with client history)")
    print(f"  Predicted Fee: ${result2['predicted_fee']:,.2f}")
    print(f"  Bid Fee Confidence: {result2['confidence_level']}")
    print(f"  Win Probability: {result2['win_probability']['probability_pct']}%")
    print(f"  Win Prob Confidence: {result2['win_probability']['confidence']}")
    print(f"  Expected Value: ${result2['expected_value']:,.2f}")

    # Test with rare state
    result3 = predictor.predict(
        business_segment="Financing",
        property_type="Multifamily",
        property_state="Alaska",
        target_time=30,
    )

    print(f"\nTest 3: Financing / Multifamily / Alaska (rare state)")
    print(f"  Predicted Fee: ${result3['predicted_fee']:,.2f}")
    print(f"  Bid Fee Confidence: {result3['confidence_level']}")
    print(f"  Win Probability: {result3['win_probability']['probability_pct']}%")
    print(f"  Win Prob Confidence: {result3['win_probability']['confidence']}")
    print(f"  Data Coverage: {result3['metadata']['data_coverage']}")
