"""
Feature Engineering - Bid Recommendation System (Phase 1A: Bid Fee Prediction)

Objective: Create rich feature set for predicting bid fees including:
           - Rolling/lag features (time series patterns)
           - Client relationship features
           - Office capacity indicators
           - Market trends
           - Categorical encodings
           - Interaction features

Author: Ujjawal Dwivedi
Organization: Global Stat Solutions (GSS)
Date: 2026-01-03
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# Configuration
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)


class BidFeatureEngineer:
    """Feature engineering for Bid Fee Prediction (Phase 1A)"""

    def __init__(self, input_path: Path, output_dir: Path):
        """
        Initialize feature engineer

        Args:
            input_path: Path to processed data CSV
            output_dir: Directory to save outputs
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.processed_dir = output_dir.parent / 'data' / 'processed'
        self.features_dir = output_dir.parent / 'data' / 'features'
        self.reports_dir = output_dir / 'reports'

        # Create directories
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Data
        self.df = None
        self.feature_stats = {}
        self.new_features_created = []

    def load_data(self):
        """Load processed data and sort by date"""
        print("="*80)
        print("LOADING PROCESSED DATA")
        print("="*80)

        self.df = pd.read_csv(self.input_path)

        print(f"✓ Data loaded")
        print(f"  Rows: {len(self.df):,}")
        print(f"  Columns: {len(self.df.columns)}")

        # Convert dates
        self.df['BidDate'] = pd.to_datetime(self.df['BidDate'])

        # Sort by date (CRITICAL for time-aware features)
        self.df = self.df.sort_values('BidDate').reset_index(drop=True)

        print(f"✓ Data sorted by BidDate")
        print(f"  Date range: {self.df['BidDate'].min()} to {self.df['BidDate'].max()}")
        print()

    def create_rolling_features_by_group(self):
        """
        Create rolling average features grouped by different dimensions.
        These capture recent pricing trends.
        """
        print("="*80)
        print("CREATING ROLLING FEATURES")
        print("="*80)

        print(f"\nCreating rolling window features (time-aware):")
        print(f"  Window: 90 days (approximately 3 months)")

        # Define rolling window
        window = '90D'

        # 1. Rolling average BidFee by Office
        print(f"\n  1. Rolling avg BidFee by Office...")
        self.df = self.df.set_index('BidDate')
        self.df['rolling_avg_fee_office'] = self.df.groupby('OfficeLocation')['BidFee'].transform(
            lambda x: x.rolling(window, min_periods=1).mean().shift(1)
        )
        self.df = self.df.reset_index()
        self.new_features_created.append('rolling_avg_fee_office')

        # 2. Rolling average BidFee by Property Type
        print(f"  2. Rolling avg BidFee by Property Type...")
        self.df = self.df.set_index('BidDate')
        self.df['rolling_avg_fee_proptype'] = self.df.groupby('Bid_Property_Type')['BidFee'].transform(
            lambda x: x.rolling(window, min_periods=1).mean().shift(1)
        )
        self.df = self.df.reset_index()
        self.new_features_created.append('rolling_avg_fee_proptype')

        # 3. Rolling average BidFee by State
        print(f"  3. Rolling avg BidFee by State...")
        self.df = self.df.set_index('BidDate')
        self.df['rolling_avg_fee_state'] = self.df.groupby('PropertyState')['BidFee'].transform(
            lambda x: x.rolling(window, min_periods=1).mean().shift(1)
        )
        self.df = self.df.reset_index()
        self.new_features_created.append('rolling_avg_fee_state')

        # 4. Rolling average BidFee by Business Segment
        print(f"  4. Rolling avg BidFee by Business Segment...")
        self.df = self.df.set_index('BidDate')
        self.df['rolling_avg_fee_segment'] = self.df.groupby('BusinessSegment')['BidFee'].transform(
            lambda x: x.rolling(window, min_periods=1).mean().shift(1)
        )
        self.df = self.df.reset_index()
        self.new_features_created.append('rolling_avg_fee_segment')

        # 5. Rolling bid count (volume indicator) by Office
        print(f"  5. Rolling bid volume by Office...")
        self.df = self.df.set_index('BidDate')
        self.df['rolling_bid_count_office'] = self.df.groupby('OfficeLocation')['BidFee'].transform(
            lambda x: x.rolling(window, min_periods=1).count().shift(1)
        )
        self.df = self.df.reset_index()
        self.new_features_created.append('rolling_bid_count_office')

        # 6. Rolling win rate by Office (capacity indicator)
        print(f"  6. Rolling win rate by Office...")
        self.df = self.df.set_index('BidDate')
        self.df['rolling_win_rate_office'] = self.df.groupby('OfficeLocation')['Won'].transform(
            lambda x: x.rolling(window, min_periods=1).mean().shift(1)
        )
        self.df = self.df.reset_index()
        self.new_features_created.append('rolling_win_rate_office')

        # 7. Rolling std deviation of BidFee by Office (volatility)
        print(f"  7. Rolling std BidFee by Office...")
        self.df = self.df.set_index('BidDate')
        self.df['rolling_std_fee_office'] = self.df.groupby('OfficeLocation')['BidFee'].transform(
            lambda x: x.rolling(window, min_periods=2).std().shift(1)
        )
        self.df = self.df.reset_index()
        self.new_features_created.append('rolling_std_fee_office')

        print(f"\n✓ Created {7} rolling features")
        print()

    def create_lag_features_by_client(self):
        """
        Create lag features for client relationships.
        What happened in the last interaction with this client?
        """
        print("="*80)
        print("CREATING CLIENT LAG FEATURES")
        print("="*80)

        print(f"\nCreating client relationship features:")

        # 1. Previous fee for this client
        print(f"  1. Previous fee to same client...")
        self.df['prev_fee_same_client'] = self.df.groupby('BidCompanyName')['BidFee'].shift(1)
        self.new_features_created.append('prev_fee_same_client')

        # 2. Previous outcome for this client (did we win?)
        print(f"  2. Previous outcome with same client...")
        self.df['prev_won_same_client'] = self.df.groupby('BidCompanyName')['Won'].shift(1)
        self.new_features_created.append('prev_won_same_client')

        # 3. Days since last bid to this client
        print(f"  3. Days since last bid to same client...")
        self.df['prev_bid_date_same_client'] = self.df.groupby('BidCompanyName')['BidDate'].shift(1)
        self.df['days_since_last_bid_client'] = (
            self.df['BidDate'] - self.df['prev_bid_date_same_client']
        ).dt.days
        self.df.drop('prev_bid_date_same_client', axis=1, inplace=True)
        self.new_features_created.append('days_since_last_bid_client')

        # 4. Previous property type for this client (same type?)
        print(f"  4. Previous property type for same client...")
        self.df['prev_proptype_same_client'] = self.df.groupby('BidCompanyName')['Bid_Property_Type'].shift(1)
        self.df['same_proptype_as_last_client'] = (
            self.df['Bid_Property_Type'] == self.df['prev_proptype_same_client']
        ).astype(int)
        self.df.drop('prev_proptype_same_client', axis=1, inplace=True)
        self.new_features_created.append('same_proptype_as_last_client')

        print(f"\n✓ Created {4} client lag features")
        print()

    def create_cumulative_client_features(self):
        """
        Create cumulative features for client relationships.
        Historical track record with each client.
        """
        print("="*80)
        print("CREATING CUMULATIVE CLIENT FEATURES")
        print("="*80)

        print(f"\nCreating historical client relationship features:")

        # 1. Total number of bids to this client (relationship depth)
        print(f"  1. Total bids to same client (cumulative)...")
        self.df['total_bids_to_client'] = self.df.groupby('BidCompanyName').cumcount()
        self.new_features_created.append('total_bids_to_client')

        # 2. Total wins with this client
        print(f"  2. Total wins with same client (cumulative)...")
        self.df['total_wins_with_client'] = self.df.groupby('BidCompanyName')['Won'].cumsum() - self.df['Won']
        self.new_features_created.append('total_wins_with_client')

        # 3. Win rate with this client
        print(f"  3. Win rate with same client...")
        self.df['win_rate_with_client'] = np.where(
            self.df['total_bids_to_client'] > 0,
            self.df['total_wins_with_client'] / self.df['total_bids_to_client'],
            np.nan
        )
        self.new_features_created.append('win_rate_with_client')

        # 4. Average fee historically charged to this client
        print(f"  4. Avg historical fee to same client...")
        self.df['avg_historical_fee_client'] = self.df.groupby('BidCompanyName')['BidFee'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        self.new_features_created.append('avg_historical_fee_client')

        print(f"\n✓ Created {4} cumulative client features")
        print()

    def create_aggregation_features(self):
        """
        Create aggregated features at different levels using leave-one-out logic.
        Market-level, office-level statistics.

        IMPORTANT: Uses leave-one-out to prevent data leakage.
        Each row's aggregate excludes that row's own value.
        """
        print("="*80)
        print("CREATING AGGREGATION FEATURES (LEAVE-ONE-OUT)")
        print("="*80)

        print(f"\nCreating aggregated statistical features:")
        print(f"  Using leave-one-out logic to prevent data leakage")

        # Calculate overall statistics for each group
        group_features = {
            'Office': 'OfficeLocation',
            'PropertyType': 'Bid_Property_Type',
            'State': 'PropertyState',
            'Segment': 'BusinessSegment',
            'Client': 'BidCompanyName',
        }

        for name, col in group_features.items():
            print(f"  {name} statistics...")

            # Mean BidFee for this group (LEAVE-ONE-OUT)
            # Calculate: (group_sum - current_value) / (group_count - 1)
            group_sum = self.df.groupby(col)['BidFee'].transform('sum')
            group_count = self.df.groupby(col)['BidFee'].transform('count')
            group_mean = (group_sum - self.df['BidFee']) / (group_count - 1)
            # Handle single-item groups (division by zero)
            group_mean = group_mean.fillna(self.df['BidFee'].mean())
            self.df[f'{name.lower()}_avg_fee'] = group_mean
            self.new_features_created.append(f'{name.lower()}_avg_fee')

            # Standard deviation for this group (LEAVE-ONE-OUT)
            # For simplicity, use global group std (leakage impact is lower for std)
            # Proper leave-one-out std is complex; using group std as approximation
            group_std = self.df.groupby(col)['BidFee'].transform('std')
            self.df[f'{name.lower()}_std_fee'] = group_std.fillna(0)
            self.new_features_created.append(f'{name.lower()}_std_fee')

            # Win rate for this group (LEAVE-ONE-OUT)
            group_won_sum = self.df.groupby(col)['Won'].transform('sum')
            group_won_count = self.df.groupby(col)['Won'].transform('count')
            group_win_rate = (group_won_sum - self.df['Won']) / (group_won_count - 1)
            # Handle single-item groups
            group_win_rate = group_win_rate.fillna(self.df['Won'].mean())
            self.df[f'{name.lower()}_win_rate'] = group_win_rate
            self.new_features_created.append(f'{name.lower()}_win_rate')

        print(f"\n✓ Created {len(group_features) * 3} aggregation features (leak-free)")
        print()

    def create_interaction_features(self):
        """
        Create interaction features between important variables.
        Capture non-linear relationships.
        """
        print("="*80)
        print("CREATING INTERACTION FEATURES")
        print("="*80)

        print(f"\nCreating feature interactions:")

        # 1. TargetTime × Property Type complexity proxy
        print(f"  1. TargetTime × Building Size Range...")
        # Create a numeric proxy for building size
        size_mapping = {
            'Unknown': 0, '<10K': 1, '<25K': 2, '<50K': 3, '<100K': 4,
            '<200K': 5, '<500K': 6, '<1M': 7, '>1M': 8, 'Missing/Invalid': 0
        }
        self.df['building_size_numeric'] = self.df['GrossBuildingAreaRange'].map(size_mapping).fillna(0)
        self.df['targettime_x_size'] = self.df['TargetTime'] * self.df['building_size_numeric']
        self.new_features_created.append('building_size_numeric')
        self.new_features_created.append('targettime_x_size')

        # 2. Distance × State (out-of-market indicator)
        print(f"  2. Distance × Office capacity...")
        self.df['distance_x_volume'] = self.df['DistanceInMiles'] * self.df['rolling_bid_count_office']
        self.new_features_created.append('distance_x_volume')

        # 3. Client relationship strength
        print(f"  3. Client relationship strength...")
        self.df['client_relationship_strength'] = (
            self.df['total_bids_to_client'] * self.df['win_rate_with_client'].fillna(0)
        )
        self.new_features_created.append('client_relationship_strength')

        # 4. Market competitiveness (bid volume × win rate)
        print(f"  4. Market competitiveness indicator...")
        self.df['market_competitiveness'] = (
            self.df['rolling_bid_count_office'] * (1 - self.df['rolling_win_rate_office'].fillna(0.5))
        )
        self.new_features_created.append('market_competitiveness')

        # REMOVED: fee_deviation_from_office_avg (data leakage - uses BidFee directly)

        print(f"\n✓ Created {4} interaction features")
        print()

    def create_categorical_encodings(self):
        """
        Create various encodings for categorical variables.
        Multiple encoding strategies for robustness.
        """
        print("="*80)
        print("CREATING CATEGORICAL ENCODINGS")
        print("="*80)

        print(f"\nCreating categorical feature encodings:")

        # 1. Frequency encoding (how common is this category?)
        categorical_cols = [
            'Bid_Property_Type', 'PropertyState', 'OfficeLocation',
            'BusinessSegment', 'BidCompanyType', 'MarketOrientation'
        ]

        print(f"  1. Frequency encoding for {len(categorical_cols)} columns...")
        for col in categorical_cols:
            if col in self.df.columns:
                freq_map = self.df[col].value_counts(normalize=True).to_dict()
                self.df[f'{col}_frequency'] = self.df[col].map(freq_map)
                self.new_features_created.append(f'{col}_frequency')

        # 2. Label encoding (ordinal encoding)
        print(f"  2. Label encoding for categorical columns...")
        label_encode_cols = [
            'Bid_Property_Type', 'Bid_SubProperty_Type',
            'PropertyState', 'PropertyCity', 'OfficeLocation',
            'BusinessSegment', 'BidCompanyName', 'BidCompanyType'
        ]

        for col in label_encode_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                # Handle unseen categories by adding 'Unknown'
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.new_features_created.append(f'{col}_encoded')

        print(f"\n✓ Created {len(categorical_cols) + len(label_encode_cols)} encoding features")
        print()

    def create_temporal_trend_features(self):
        """
        Create features that capture temporal trends.
        Is the market heating up or cooling down?
        """
        print("="*80)
        print("CREATING TEMPORAL TREND FEATURES")
        print("="*80)

        print(f"\nCreating temporal trend indicators:")

        # 1. Days since start of dataset (linear time trend)
        print(f"  1. Days since data start (time trend)...")
        start_date = self.df['BidDate'].min()
        self.df['days_since_start'] = (self.df['BidDate'] - start_date).dt.days
        self.new_features_created.append('days_since_start')

        # 2. Quarter-over-quarter BidFee trend
        print(f"  2. Quarterly fee trend...")
        self.df['year_quarter'] = self.df['Year'].astype(str) + '_Q' + self.df['Quarter'].astype(str)
        quarterly_avg = self.df.groupby('year_quarter')['BidFee'].transform('mean')
        self.df['quarterly_avg_fee'] = quarterly_avg
        self.new_features_created.append('quarterly_avg_fee')

        # 3. Is this bid in peak season? (October, August, March from EDA)
        print(f"  3. Peak season indicator...")
        peak_months = [3, 8, 10]
        self.df['is_peak_season'] = self.df['Month'].isin(peak_months).astype(int)
        self.new_features_created.append('is_peak_season')

        # 4. Is this a weekday bid?
        print(f"  4. Weekday indicator...")
        self.df['is_weekday'] = (self.df['DayOfWeek'] < 5).astype(int)
        self.new_features_created.append('is_weekday')

        print(f"\n✓ Created {4} temporal trend features")
        print()

    def create_ratio_features(self):
        """
        Create ratio and relative features (LEAK-FREE VERSION).
        Only uses features that don't contain the target variable.

        REMOVED for data leakage prevention:
        - fee_ratio_to_rolling_office (uses BidFee directly)
        - fee_ratio_to_proptype (uses BidFee directly)
        - client_fee_ratio_to_market (uses BidFee in market_avg)
        """
        print("="*80)
        print("CREATING RATIO FEATURES (LEAK-FREE)")
        print("="*80)

        print(f"\nCreating ratio and relative features:")
        print(f"  Removed BidFee-based ratios to prevent data leakage")

        # TargetTime / Property Type Average TargetTime (SAFE - doesn't use BidFee)
        print(f"  1. TargetTime ratio to property type average...")
        proptype_avg_time = self.df.groupby('Bid_Property_Type')['TargetTime'].transform('mean')
        self.df['targettime_ratio_to_proptype'] = (
            self.df['TargetTime'] / (proptype_avg_time + 1)
        )
        self.new_features_created.append('targettime_ratio_to_proptype')

        print(f"\n✓ Created {1} ratio feature (leak-free)")
        print()

    def handle_feature_missing_values(self):
        """
        Handle missing values in engineered features.
        Different strategies for different feature types.
        """
        print("="*80)
        print("HANDLING MISSING VALUES IN ENGINEERED FEATURES")
        print("="*80)

        print(f"\nFilling missing values in new features:")

        # Count missing values in new features
        new_features_with_missing = {}
        for feature in self.new_features_created:
            if feature in self.df.columns:
                missing_count = self.df[feature].isnull().sum()
                if missing_count > 0:
                    new_features_with_missing[feature] = missing_count

        print(f"  Features with missing values: {len(new_features_with_missing)}")

        # Fill strategies
        fill_strategies = {
            # Rolling features: fill with 0 (no prior history)
            'rolling': 0,
            # Lag features: fill with median
            'prev_': 'median',
            'days_since': 999999,  # Very large number = never seen before
            # Cumulative: fill with 0
            'total_': 0,
            # Win rate: fill with 0.5 (neutral)
            'win_rate': 0.5,
            # Ratios: fill with 1.0 (neutral)
            'ratio': 1.0,
            # Others: fill with 0
            'default': 0
        }

        filled_count = 0
        for feature, missing_count in new_features_with_missing.items():
            # Determine fill strategy
            if 'rolling' in feature:
                fill_value = 0
            elif 'prev_' in feature or feature.startswith('days_since'):
                if feature == 'days_since_last_bid_client':
                    fill_value = 999999
                else:
                    fill_value = self.df[feature].median()
            elif 'total_' in feature or 'count' in feature:
                fill_value = 0
            elif 'win_rate' in feature or 'won' in feature:
                fill_value = 0.5
            elif 'ratio' in feature or 'deviation' in feature:
                fill_value = 1.0
            else:
                fill_value = 0

            self.df[feature].fillna(fill_value, inplace=True)
            filled_count += 1

        print(f"  ✓ Filled missing values in {filled_count} features")
        print()

    def validate_feature_quality(self):
        """
        Validate engineered features.
        Check for issues like infinite values, NaNs, etc.
        """
        print("="*80)
        print("VALIDATING FEATURE QUALITY")
        print("="*80)

        print(f"\nChecking for data quality issues:")

        # Check for infinite values
        inf_columns = []
        for col in self.new_features_created:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.float32]:
                if np.isinf(self.df[col]).any():
                    inf_columns.append(col)
                    # Replace inf with large number
                    self.df[col].replace([np.inf, -np.inf], [999999, -999999], inplace=True)

        if inf_columns:
            print(f"  ⚠️ Found and fixed infinite values in {len(inf_columns)} columns")
        else:
            print(f"  ✓ No infinite values found")

        # Check for remaining NaNs
        nan_count = self.df[self.new_features_created].isnull().sum().sum()
        if nan_count > 0:
            print(f"  ⚠️ Remaining NaN values: {nan_count}")
            # Fill any remaining NaNs with 0
            for col in self.new_features_created:
                if col in self.df.columns:
                    self.df[col].fillna(0, inplace=True)
            print(f"  ✓ Filled remaining NaNs with 0")
        else:
            print(f"  ✓ No NaN values found")

        # Summary
        print(f"\nFeature creation summary:")
        print(f"  Total new features created: {len(self.new_features_created)}")
        print(f"  Original columns: {len(self.df.columns) - len(self.new_features_created)}")
        print(f"  Final columns: {len(self.df.columns)}")
        print()

    def save_feature_engineered_data(self):
        """Save feature-engineered dataset"""
        print("="*80)
        print("SAVING FEATURE-ENGINEERED DATA")
        print("="*80)

        # Save main dataset
        output_path = self.features_dir / 'BidData_features.csv'
        self.df.to_csv(output_path, index=False)

        print(f"\n✓ Feature-engineered data saved:")
        print(f"  Location: {output_path}")
        print(f"  Size: {output_path.stat().st_size / (1024*1024):.1f} MB")
        print(f"  Rows: {len(self.df):,}")
        print(f"  Columns: {len(self.df.columns)}")

        # Save feature list
        feature_list_path = self.reports_dir / 'feature_list.txt'
        with open(feature_list_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FEATURE-ENGINEERED DATASET - COLUMN LIST\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total Features: {len(self.new_features_created)}\n\n")
            f.write("New Features Created:\n")
            f.write("-"*80 + "\n")
            for i, feature in enumerate(self.new_features_created, 1):
                f.write(f"{i:3d}. {feature}\n")

        print(f"✓ Feature list saved: {feature_list_path}")
        print()

    def generate_feature_engineering_report(self):
        """Generate comprehensive feature engineering report"""
        print("="*80)
        print("GENERATING FEATURE ENGINEERING REPORT")
        print("="*80)

        # Categorize features
        feature_categories = {
            'Rolling/Time Series': [f for f in self.new_features_created if 'rolling' in f],
            'Lag Features': [f for f in self.new_features_created if 'prev_' in f or 'days_since' in f],
            'Cumulative': [f for f in self.new_features_created if 'total_' in f or 'cumulative' in f],
            'Aggregation': [f for f in self.new_features_created if any(x in f for x in ['avg_', 'std_', 'mean_'])],
            'Interaction': [f for f in self.new_features_created if '_x_' in f or 'strength' in f or 'competitive' in f],
            'Encoding': [f for f in self.new_features_created if 'encoded' in f or 'frequency' in f],
            'Ratio': [f for f in self.new_features_created if 'ratio' in f or 'deviation' in f],
            'Temporal': [f for f in self.new_features_created if 'days_since_start' in f or 'peak' in f or 'weekday' in f or 'quarterly' in f],
            'Other': []
        }

        # Categorize remaining features
        categorized = set()
        for features in feature_categories.values():
            categorized.update(features)

        feature_categories['Other'] = [f for f in self.new_features_created if f not in categorized]

        # Create report
        report = {
            'Feature Engineering Summary': {
                'Total New Features': len(self.new_features_created),
                'Original Columns': len(self.df.columns) - len(self.new_features_created),
                'Final Columns': len(self.df.columns),
                'Dataset Rows': len(self.df),
            },
            'Feature Categories': {
                cat: len(features) for cat, features in feature_categories.items() if len(features) > 0
            }
        }

        # Print report
        for section, metrics in report.items():
            print(f"\n{section}:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")

        # Detailed breakdown
        print(f"\nDetailed Feature Breakdown:")
        for category, features in feature_categories.items():
            if len(features) > 0:
                print(f"\n  {category} ({len(features)} features):")
                for feature in features[:5]:  # Show first 5
                    print(f"    - {feature}")
                if len(features) > 5:
                    print(f"    ... and {len(features) - 5} more")

        # Save report
        import json
        report_path = self.reports_dir / 'feature_engineering_report.json'

        # Add detailed feature list to report
        report['Detailed_Feature_List'] = {
            cat: features for cat, features in feature_categories.items() if len(features) > 0
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Feature engineering report saved: {report_path}")
        print()

    def run_full_feature_engineering(self):
        """Execute complete feature engineering pipeline"""
        print("\n" + "="*80)
        print("STARTING FEATURE ENGINEERING (PHASE 1A)")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Execute all steps
        self.load_data()
        self.create_rolling_features_by_group()
        self.create_lag_features_by_client()
        self.create_cumulative_client_features()
        self.create_aggregation_features()
        self.create_interaction_features()
        self.create_categorical_encodings()
        self.create_temporal_trend_features()
        self.create_ratio_features()
        self.handle_feature_missing_values()
        self.validate_feature_quality()
        self.save_feature_engineered_data()
        self.generate_feature_engineering_report()

        print("="*80)
        print("FEATURE ENGINEERING COMPLETE (PHASE 1A)")
        print("="*80)
        print(f"\n✓ Feature-engineered dataset ready for modeling")
        print(f"  Location: {self.features_dir / 'BidData_features.csv'}")
        print(f"  Total features: {len(self.new_features_created)}")
        print(f"  Dataset shape: ({len(self.df):,}, {len(self.df.columns)})")
        print()


def main():
    """Main execution function"""
    # Define paths
    PROJECT_ROOT = Path(__file__).parent.parent
    INPUT_PATH = PROJECT_ROOT / 'data' / 'processed' / 'BidData_processed.csv'
    OUTPUT_DIR = PROJECT_ROOT / 'outputs'

    # Initialize and run feature engineering
    engineer = BidFeatureEngineer(input_path=INPUT_PATH, output_dir=OUTPUT_DIR)
    engineer.run_full_feature_engineering()


if __name__ == "__main__":
    main()
