"""
Feature Engineering - Bid Recommendation System (EXPANDED VERSION)

Objective: Create comprehensive feature set for BOTH tasks:
           - Phase 1A: Bid Fee Prediction (Regression)
           - Phase 1B: Win Probability Prediction (Classification)

Features expanded from 84 to ~150 including:
           - Original rolling/lag features
           - NEW: Competitiveness metrics
           - NEW: Temporal/behavioral patterns
           - NEW: Market dynamics
           - NEW: Risk/volatility indicators
           - NEW: Extended interaction features

Author: Bid Recommendation System
Date: 2026-01-08 (Expanded)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)


class BidFeatureEngineer:
    """Comprehensive feature engineering for bid prediction"""

    def __init__(self, input_path: Path, output_dir: Path):
        self.input_path = input_path
        self.output_dir = output_dir
        self.processed_dir = output_dir.parent / 'data' / 'processed'
        self.features_dir = output_dir.parent / 'data' / 'features'
        self.reports_dir = output_dir / 'reports'

        # Create directories
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

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
        """Create rolling average features grouped by different dimensions"""
        print("="*80)
        print("CREATING ROLLING FEATURES")
        print("="*80)

        print(f"\nCreating rolling window features (time-aware):")
        print(f"  Window: 90 days")

        window = '90D'

        # Use simpler rolling window approach (last N observations instead of time window)
        roll_n = 10  # Last 10 observations

        # Rolling average BidFee by segment
        print(f"\n  1. Rolling avg BidFee by BusinessSegment (last {roll_n} obs)...")
        self.df['rolling_avg_fee_segment'] = (
            self.df.groupby('BusinessSegment')['BidFee']
            .transform(lambda x: x.shift().rolling(window=roll_n, min_periods=1).mean())
        )
        self.new_features_created.append('rolling_avg_fee_segment')

        # Rolling average BidFee by Office
        print(f"  2. Rolling avg BidFee by Office (last {roll_n} obs)...")
        self.df['rolling_avg_fee_office'] = (
            self.df.groupby('OfficeId')['BidFee']
            .transform(lambda x: x.shift().rolling(window=roll_n, min_periods=1).mean())
        )
        self.new_features_created.append('rolling_avg_fee_office')

        # Rolling std BidFee by segment (NEW: volatility)
        print(f"  3. Rolling std BidFee by BusinessSegment (last {roll_n} obs)...")
        self.df['rolling_std_fee_segment'] = (
            self.df.groupby('BusinessSegment')['BidFee']
            .transform(lambda x: x.shift().rolling(window=roll_n, min_periods=1).std())
        )
        self.new_features_created.append('rolling_std_fee_segment')

        print(f"✓ Created {len([f for f in self.new_features_created if 'rolling' in f])} rolling features")

    def create_lag_features_by_client(self):
        """Create lag features for client history"""
        print("\n" + "="*80)
        print("CREATING LAG FEATURES")
        print("="*80)

        print(f"\nCreating lag features by Client:")

        # Lag 1: Previous bid fee for this client
        print(f"  1. Previous BidFee (lag 1)...")
        self.df['lag1_bidfee_client'] = (
            self.df.groupby('BidCompanyName')['BidFee'].shift(1)
        )
        self.new_features_created.append('lag1_bidfee_client')

        # Lag 2: Two bids ago
        print(f"  2. Two bids ago BidFee (lag 2)...")
        self.df['lag2_bidfee_client'] = (
            self.df.groupby('BidCompanyName')['BidFee'].shift(2)
        )
        self.new_features_created.append('lag2_bidfee_client')

        # Previous TargetTime for client
        print(f"  3. Previous TargetTime...")
        self.df['lag1_targettime_client'] = (
            self.df.groupby('BidCompanyName')['TargetTime'].shift(1)
        )
        self.new_features_created.append('lag1_targettime_client')

        print(f"✓ Created {len([f for f in self.new_features_created if 'lag' in f])} lag features")

    def create_cumulative_client_features(self):
        """Create cumulative/expanding features for clients"""
        print("\n" + "="*80)
        print("CREATING CUMULATIVE FEATURES")
        print("="*80)

        print(f"\nCreating cumulative features:")

        # Cumulative bid count per client
        print(f"  1. Cumulative bid count by Client...")
        self.df['cumulative_bids_client'] = (
            self.df.groupby('BidCompanyName').cumcount()
        )
        self.new_features_created.append('cumulative_bids_client')

        # Cumulative win count per client
        print(f"  2. Cumulative wins by Client...")
        self.df['cumulative_wins_client'] = (
            self.df.groupby('BidCompanyName')['Won'].cumsum().shift(1).fillna(0)
        )
        self.new_features_created.append('cumulative_wins_client')

        # Cumulative win rate per client
        print(f"  3. Cumulative win rate by Client...")
        self.df['cumulative_winrate_client'] = (
            self.df['cumulative_wins_client'] /
            (self.df['cumulative_bids_client'] + 1)
        )
        self.new_features_created.append('cumulative_winrate_client')

        print(f"✓ Created {len([f for f in self.new_features_created if 'cumulative' in f])} cumulative features")

    def create_aggregation_features(self):
        """Create aggregation features (mean, std, count) by various groups"""
        print("\n" + "="*80)
        print("CREATING AGGREGATION FEATURES")
        print("="*80)

        print(f"\nCreating aggregation features (leave-one-out):")

        # Helper function for leave-one-out
        def leave_one_out_mean(group_col, value_col):
            total = self.df.groupby(group_col)[value_col].transform('sum')
            count = self.df.groupby(group_col)[value_col].transform('count')
            return (total - self.df[value_col]) / (count - 1)

        def leave_one_out_std(group_col, value_col):
            return self.df.groupby(group_col)[value_col].transform(
                lambda x: x.shift().expanding().std()
            )

        # 1. BusinessSegment aggregations
        print(f"  1. BusinessSegment aggregations...")
        self.df['segment_avg_fee'] = leave_one_out_mean('BusinessSegment', 'BidFee')
        self.df['segment_std_fee'] = self.df.groupby('BusinessSegment')['BidFee'].transform(
            lambda x: x.shift().expanding().std()
        )
        self.df['segment_win_rate'] = leave_one_out_mean('BusinessSegment', 'Won')
        self.new_features_created.extend(['segment_avg_fee', 'segment_std_fee', 'segment_win_rate'])

        # 2. BidCompanyName aggregations
        print(f"  2. BidCompanyName aggregations...")
        self.df['client_avg_fee'] = leave_one_out_mean('BidCompanyName', 'BidFee')
        self.df['client_std_fee'] = self.df.groupby('BidCompanyName')['BidFee'].transform(
            lambda x: x.shift().expanding().std()
        )
        self.df['client_win_rate'] = leave_one_out_mean('BidCompanyName', 'Won')
        self.new_features_created.extend(['client_avg_fee', 'client_std_fee', 'client_win_rate'])

        # 3. Office aggregations
        print(f"  3. Office aggregations...")
        self.df['office_avg_fee'] = leave_one_out_mean('OfficeId', 'BidFee')
        self.df['office_std_fee'] = self.df.groupby('OfficeId')['BidFee'].transform(
            lambda x: x.shift().expanding().std()
        )
        self.df['office_win_rate'] = leave_one_out_mean('OfficeId', 'Won')
        self.new_features_created.extend(['office_avg_fee', 'office_std_fee', 'office_win_rate'])

        # 4. PropertyType aggregations
        print(f"  4. PropertyType aggregations...")
        self.df['propertytype_avg_fee'] = leave_one_out_mean('PropertyType', 'BidFee')
        self.df['propertytype_win_rate'] = leave_one_out_mean('PropertyType', 'Won')
        self.new_features_created.extend(['propertytype_avg_fee', 'propertytype_win_rate'])

        # 5. PropertyState aggregations
        print(f"  5. PropertyState aggregations...")
        self.df['state_avg_fee'] = leave_one_out_mean('PropertyState', 'BidFee')
        self.df['state_win_rate'] = leave_one_out_mean('PropertyState', 'Won')
        self.new_features_created.extend(['state_avg_fee', 'state_win_rate'])

        print(f"✓ Created aggregation features")

    def create_competitiveness_features(self):
        """NEW: Create features measuring bid competitiveness"""
        print("\n" + "="*80)
        print("CREATING COMPETITIVENESS FEATURES (NEW)")
        print("="*80)

        print(f"\nCreating competitiveness metrics:")

        # 1. Bid vs segment average ratio
        print(f"  1. Bid vs segment average ratio...")
        self.df['bid_vs_segment_ratio'] = (
            self.df['BidFee'] / (self.df['segment_avg_fee'] + 1)
        )
        self.new_features_created.append('bid_vs_segment_ratio')

        # 2. Bid vs client average ratio
        print(f"  2. Bid vs client average ratio...")
        self.df['bid_vs_client_ratio'] = (
            self.df['BidFee'] / (self.df['client_avg_fee'] + 1)
        )
        self.new_features_created.append('bid_vs_client_ratio')

        # 3. Bid vs state average ratio
        print(f"  3. Bid vs state average ratio...")
        self.df['bid_vs_state_ratio'] = (
            self.df['BidFee'] / (self.df['state_avg_fee'] + 1)
        )
        self.new_features_created.append('bid_vs_state_ratio')

        # 4. Fee above/below market (absolute difference)
        print(f"  4. Fee difference from segment average...")
        self.df['fee_diff_from_segment'] = (
            self.df['BidFee'] - self.df['segment_avg_fee']
        )
        self.new_features_created.append('fee_diff_from_segment')

        # 5. Fee percentile within segment (approximation using rank)
        print(f"  5. Fee percentile within segment...")
        self.df['fee_percentile_segment'] = (
            self.df.groupby('BusinessSegment')['BidFee']
            .rank(pct=True, method='average')
        )
        self.new_features_created.append('fee_percentile_segment')

        print(f"✓ Created {len([f for f in self.new_features_created if any(x in f for x in ['bid_vs', 'fee_diff', 'percentile'])])} competitiveness features")

    def create_temporal_behavioral_features(self):
        """NEW: Create temporal and behavioral pattern features"""
        print("\n" + "="*80)
        print("CREATING TEMPORAL/BEHAVIORAL FEATURES (NEW)")
        print("="*80)

        print(f"\nCreating temporal patterns:")

        # 1. Days since last bid for client
        print(f"  1. Days since last bid (client)...")
        self.df['days_since_last_bid_client'] = (
            self.df.groupby('BidCompanyName')['BidDate'].diff().dt.days
        )
        self.new_features_created.append('days_since_last_bid_client')

        # 2. Days since last bid for segment
        print(f"  2. Days since last bid (segment)...")
        self.df['days_since_last_bid_segment'] = (
            self.df.groupby('BusinessSegment')['BidDate'].diff().dt.days
        )
        self.new_features_created.append('days_since_last_bid_segment')

        # 3. Bid frequency (last 5 observations) - market activity
        print(f"  3. Bid count (rolling window)...")
        self.df['bid_count_last_30d'] = (
            self.df.groupby('BusinessSegment')['BidFee']
            .transform(lambda x: x.shift().rolling(window=5, min_periods=1).count())
        )
        self.new_features_created.append('bid_count_last_30d')

        # 4. Time-based features
        print(f"  4. Time-based features (month, quarter, day_of_week)...")
        self.df['month'] = self.df['BidDate'].dt.month
        self.df['quarter'] = self.df['BidDate'].dt.quarter
        self.df['day_of_week'] = self.df['BidDate'].dt.dayofweek
        self.df['is_month_end'] = (self.df['BidDate'].dt.day > 25).astype(int)
        self.df['is_quarter_end'] = (
            (self.df['BidDate'].dt.month.isin([3, 6, 9, 12])) &
            (self.df['BidDate'].dt.day > 25)
        ).astype(int)
        self.new_features_created.extend(['month', 'quarter', 'day_of_week', 'is_month_end', 'is_quarter_end'])

        # 5. Client loyalty metrics
        print(f"  5. Client bid count (total)...")
        self.df['client_bid_count'] = (
            self.df.groupby('BidCompanyName').cumcount()
        )
        self.new_features_created.append('client_bid_count')

        print(f"✓ Created temporal/behavioral features")

    def create_market_dynamics_features(self):
        """NEW: Create market dynamics and competition features"""
        print("\n" + "="*80)
        print("CREATING MARKET DYNAMICS FEATURES (NEW)")
        print("="*80)

        print(f"\nCreating market dynamics metrics:")

        # 1. Segment bid density (how crowded is this segment?)
        print(f"  1. Segment bid density...")
        segment_counts = self.df.groupby('BusinessSegment').size()
        self.df['segment_bid_density'] = self.df['BusinessSegment'].map(segment_counts)
        self.new_features_created.append('segment_bid_density')

        # 2. State-segment combo frequency
        print(f"  2. State-segment combo frequency...")
        combo_counts = self.df.groupby(['PropertyState', 'BusinessSegment']).size()
        self.df['state_segment_combo_freq'] = self.df.apply(
            lambda x: combo_counts.get((x['PropertyState'], x['BusinessSegment']), 0), axis=1
        )
        self.new_features_created.append('state_segment_combo_freq')

        # 3. Client segment specialization (does client focus on this segment?)
        print(f"  3. Client segment specialization...")
        client_segment_counts = self.df.groupby(['BidCompanyName', 'BusinessSegment']).size()
        client_total_counts = self.df.groupby('BidCompanyName').size()
        self.df['client_segment_specialization'] = self.df.apply(
            lambda x: client_segment_counts.get((x['BidCompanyName'], x['BusinessSegment']), 0) /
                     (client_total_counts.get(x['BidCompanyName'], 1) + 1), axis=1
        )
        self.new_features_created.append('client_segment_specialization')

        # 4. Office workload (rolling window)
        print(f"  4. Office workload (rolling window)...")
        self.df['office_workload_30d'] = (
            self.df.groupby('OfficeId')['BidFee']
            .transform(lambda x: x.shift().rolling(window=5, min_periods=1).count())
        )
        self.new_features_created.append('office_workload_30d')

        print(f"✓ Created market dynamics features")

    def create_risk_volatility_features(self):
        """NEW: Create risk and volatility indicators"""
        print("\n" + "="*80)
        print("CREATING RISK/VOLATILITY FEATURES (NEW)")
        print("="*80)

        print(f"\nCreating risk metrics:")

        # 1. Coefficient of variation for segment (volatility measure)
        print(f"  1. Segment fee coefficient of variation...")
        self.df['segment_cv_fee'] = (
            self.df['segment_std_fee'] / (self.df['segment_avg_fee'] + 1)
        )
        self.new_features_created.append('segment_cv_fee')

        # 2. Client fee consistency (inverse of CV)
        print(f"  2. Client fee consistency...")
        self.df['client_fee_consistency'] = (
            1 / (self.df['client_std_fee'] / (self.df['client_avg_fee'] + 1) + 0.01)
        )
        self.new_features_created.append('client_fee_consistency')

        # 3. TargetTime vs segment average
        print(f"  3. TargetTime vs segment average...")
        segment_avg_time = self.df.groupby('BusinessSegment')['TargetTime'].transform('mean')
        self.df['targettime_vs_segment'] = (
            self.df['TargetTime'] / (segment_avg_time + 1)
        )
        self.new_features_created.append('targettime_vs_segment')

        # 4. Fee range in segment (max - min as proxy for uncertainty)
        print(f"  4. Segment fee range...")
        segment_max = self.df.groupby('BusinessSegment')['BidFee'].transform('max')
        segment_min = self.df.groupby('BusinessSegment')['BidFee'].transform('min')
        self.df['segment_fee_range'] = segment_max - segment_min
        self.new_features_created.append('segment_fee_range')

        print(f"✓ Created risk/volatility features")

    def create_interaction_features(self):
        """Create interaction features (original + extended)"""
        print("\n" + "="*80)
        print("CREATING INTERACTION FEATURES (EXTENDED)")
        print("="*80)

        print(f"\nCreating key interaction terms:")

        # Original interactions
        print(f"  1. segment_avg_fee × TargetTime...")
        self.df['segment_fee_x_time'] = (
            self.df['segment_avg_fee'] * self.df['TargetTime']
        )
        self.new_features_created.append('segment_fee_x_time')

        print(f"  2. client_avg_fee × segment_std_fee...")
        self.df['client_fee_x_segment_std'] = (
            self.df['client_avg_fee'] * self.df['segment_std_fee']
        )
        self.new_features_created.append('client_fee_x_segment_std')

        # NEW interactions
        print(f"  3. office_avg_fee × PropertyState_frequency...")
        state_freq = self.df.groupby('PropertyState').size()
        self.df['PropertyState_frequency'] = self.df['PropertyState'].map(state_freq)
        self.df['office_fee_x_state_freq'] = (
            self.df['office_avg_fee'] * self.df['PropertyState_frequency']
        )
        self.new_features_created.extend(['PropertyState_frequency', 'office_fee_x_state_freq'])

        print(f"  4. bid_vs_segment_ratio × cumulative_winrate_client...")
        self.df['competitiveness_x_winrate'] = (
            self.df['bid_vs_segment_ratio'] * self.df['cumulative_winrate_client']
        )
        self.new_features_created.append('competitiveness_x_winrate')

        print(f"  5. TargetTime × segment_bid_density...")
        self.df['time_x_density'] = (
            self.df['TargetTime'] * self.df['segment_bid_density']
        )
        self.new_features_created.append('time_x_density')

        print(f"✓ Created interaction features")

    def create_categorical_encodings(self):
        """Create categorical encodings (frequency)"""
        print("\n" + "="*80)
        print("CREATING CATEGORICAL ENCODINGS")
        print("="*80)

        print(f"\nCreating frequency encodings:")

        # BusinessSegment frequency
        print(f"  1. BusinessSegment frequency...")
        segment_freq = self.df.groupby('BusinessSegment').size()
        self.df['BusinessSegment_frequency'] = self.df['BusinessSegment'].map(segment_freq)
        self.new_features_created.append('BusinessSegment_frequency')

        # PropertyType frequency
        print(f"  2. PropertyType frequency...")
        type_freq = self.df.groupby('PropertyType').size()
        self.df['PropertyType_frequency'] = self.df['PropertyType'].map(type_freq)
        self.new_features_created.append('PropertyType_frequency')

        print(f"✓ Created categorical encodings")

    def handle_missing_values(self):
        """Handle missing values in engineered features"""
        print("\n" + "="*80)
        print("HANDLING MISSING VALUES")
        print("="*80)

        # Count missing before
        missing_before = self.df[self.new_features_created].isnull().sum().sum()
        print(f"\nMissing values before: {missing_before:,}")

        # Fill missing values
        for col in self.new_features_created:
            if self.df[col].dtype in ['float64', 'int64']:
                # For numeric features, fill with 0 (conservative)
                self.df[col] = self.df[col].fillna(0)

        # Count missing after
        missing_after = self.df[self.new_features_created].isnull().sum().sum()
        print(f"Missing values after: {missing_after:,}")
        print(f"✓ Filled {missing_before - missing_after:,} missing values")

    def save_features(self):
        """Save engineered features"""
        print("\n" + "="*80)
        print("SAVING FEATURES")
        print("="*80)

        # Save to features directory
        output_path = self.features_dir / 'bid_features.csv'
        self.df.to_csv(output_path, index=False)

        print(f"✓ Features saved: {output_path}")
        print(f"  Total rows: {len(self.df):,}")
        print(f"  Total columns: {len(self.df.columns)}")
        print(f"  New features created: {len(self.new_features_created)}")

        # Save feature list
        feature_list_path = self.reports_dir / 'engineered_features_list.txt'
        with open(feature_list_path, 'w') as f:
            f.write(f"ENGINEERED FEATURES (Total: {len(self.new_features_created)})\n")
            f.write("=" * 80 + "\n\n")
            for i, feat in enumerate(self.new_features_created, 1):
                f.write(f"{i}. {feat}\n")

        print(f"✓ Feature list saved: {feature_list_path}")

        # Print feature categories
        print(f"\nFEATURE BREAKDOWN:")
        categories = {
            'Rolling': len([f for f in self.new_features_created if 'rolling' in f]),
            'Lag': len([f for f in self.new_features_created if 'lag' in f]),
            'Cumulative': len([f for f in self.new_features_created if 'cumulative' in f]),
            'Aggregation': len([f for f in self.new_features_created if any(x in f for x in ['avg', 'std', 'rate'])]),
            'Competitiveness': len([f for f in self.new_features_created if any(x in f for x in ['bid_vs', 'fee_diff', 'percentile'])]),
            'Temporal': len([f for f in self.new_features_created if any(x in f for x in ['days_since', 'month', 'quarter', 'day_of'])]),
            'Market': len([f for f in self.new_features_created if any(x in f for x in ['density', 'combo', 'specialization', 'workload'])]),
            'Risk': len([f for f in self.new_features_created if any(x in f for x in ['cv_', 'consistency', 'range'])]),
            'Interaction': len([f for f in self.new_features_created if '_x_' in f]),
            'Categorical': len([f for f in self.new_features_created if 'frequency' in f.lower()])
        }

        for cat, count in categories.items():
            print(f"  {cat}: {count}")

    def run(self):
        """Execute full feature engineering pipeline"""
        print("\n" + "="*80)
        print("COMPREHENSIVE FEATURE ENGINEERING PIPELINE")
        print("="*80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Load data
        self.load_data()

        # Create features (order matters for dependencies)
        self.create_aggregation_features()
        self.create_rolling_features_by_group()
        self.create_lag_features_by_client()
        self.create_cumulative_client_features()
        self.create_competitiveness_features()
        self.create_temporal_behavioral_features()
        self.create_market_dynamics_features()
        self.create_risk_volatility_features()
        self.create_interaction_features()
        self.create_categorical_encodings()

        # Handle missing values
        self.handle_missing_values()

        # Save
        self.save_features()

        print("\n" + "="*80)
        print("FEATURE ENGINEERING COMPLETE!")
        print("="*80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total features created: {len(self.new_features_created)}\n")


if __name__ == "__main__":
    # Paths
    from config.model_config import PROCESSED_DATA, OUTPUTS_DIR

    # Initialize and run
    engineer = BidFeatureEngineer(
        input_path=PROCESSED_DATA,
        output_dir=OUTPUTS_DIR
    )
    engineer.run()
