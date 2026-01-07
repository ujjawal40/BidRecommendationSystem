"""
Data Preprocessing - Bid Recommendation System

Objective: Clean and prepare data for modeling by handling:
           - Duplicate BidIds (aggregate Master/SubJob structure)
           - Missing values (imputation strategies)
           - Outliers (capping/removal)
           - Data type corrections
           - Feature validation

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
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

# Configuration
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)


class BidDataPreprocessor:
    """Comprehensive data preprocessing for Bid Recommendation System"""

    def __init__(self, input_path: Path, output_dir: Path):
        """
        Initialize preprocessor

        Args:
            input_path: Path to cleaned bid data CSV
            output_dir: Directory to save processed data and reports
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.processed_dir = output_dir.parent / 'data' / 'processed'
        self.reports_dir = output_dir / 'reports'

        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Data
        self.df_raw = None
        self.df_clean = None
        self.preprocessing_stats = {}

    def load_data(self):
        """Load raw data"""
        print("="*80)
        print("LOADING DATA")
        print("="*80)

        self.df_raw = pd.read_csv(self.input_path)

        print(f"✓ Raw data loaded")
        print(f"  Rows: {len(self.df_raw):,}")
        print(f"  Columns: {len(self.df_raw.columns)}")

        # Convert dates
        self.df_raw['BidDate'] = pd.to_datetime(self.df_raw['BidDate'], errors='coerce')
        self.df_raw['Bid_DueDate'] = pd.to_datetime(self.df_raw['Bid_DueDate'], errors='coerce')

        print(f"✓ Date columns converted")
        print()

        # Store initial stats
        self.preprocessing_stats['initial_rows'] = len(self.df_raw)
        self.preprocessing_stats['initial_columns'] = len(self.df_raw.columns)

    def handle_duplicate_bidids(self):
        """
        Aggregate duplicate BidIds to create one row per unique bid.
        This prevents data leakage (same bid appearing in both train and test).
        """
        print("="*80)
        print("HANDLING DUPLICATE BIDIDs")
        print("="*80)

        initial_rows = len(self.df_raw)
        duplicate_count = self.df_raw.duplicated(subset=['BidId']).sum()

        print(f"\nInitial state:")
        print(f"  Total rows: {initial_rows:,}")
        print(f"  Unique BidIds: {self.df_raw['BidId'].nunique():,}")
        print(f"  Duplicate rows: {duplicate_count:,} ({duplicate_count/initial_rows*100:.2f}%)")

        # Aggregation strategy for each column type
        agg_dict = {
            # Core bid information (take first/max)
            'BidFileNumber': 'first',
            'BidName': 'first',
            'BidDate': 'first',
            'Bid_DueDate': 'first',
            'BidFee': 'max',  # Use maximum fee (typically the Master bid)
            'TargetTime': 'max',  # Total time across all sub-jobs
            'BidStatusName': 'first',  # Status is same across all rows for a BidId
            'Bid_JobPurpose': 'first',
            'Bid_Deliverable': 'first',

            # Geographic (same for all rows of a BidId)
            'Market': 'first',
            'Submarket': 'first',
            'BusinessSegment': 'first',
            'BusinessSegmentDetail': 'first',
            'DistanceInKM': 'first',
            'DistanceInMiles': 'first',

            # Property information
            'Bid_Property_Type': 'first',
            'Bid_SubProperty_Type': 'first',
            'Bid_SpecificUseProperty_Type': 'first',
            'PropertyId': 'first',
            'PropertyName': 'first',
            'PropertyType': 'first',
            'SubType': 'first',
            'PropertyCity': 'first',
            'PropertyState': 'first',
            'RooftopLongitude': 'first',
            'RooftopLatitude': 'first',
            'ZipCode': 'first',
            'MarketOrientation': 'first',
            'AddressDisplayCalc': 'first',
            'GrossBuildingAreaRange': 'first',
            'YearBuiltRange': 'first',

            # Office information
            'OfficeId': 'first',
            'OfficeCode': 'first',
            'OfficeCompanyName': 'first',
            'OfficeLocation': 'first',

            # Job hierarchy (aggregate)
            'JobCount': 'max',  # Total number of jobs
            'IECount': 'sum',  # Sum across sub-jobs
            'LeaseCount': 'sum',
            'SaleCount': 'sum',
            'JobId': 'first',  # Keep first JobId
            'JobName': 'first',
            'JobStatus': 'first',
            'JobType': lambda x: 'Master' if 'Master' in x.values else x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
            'AppraisalFileType': 'first',

            # Demographics (same for all rows)
            'PopulationEstimate': 'first',
            'AverageHouseValue': 'first',
            'IncomePerHousehold': 'first',
            'MedianAge': 'first',
            'DeliveryTotal': 'first',
            'NumberofBusinesses': 'first',
            'NumberofEmployees': 'first',
            'ZipPopulation': 'first',

            # Client information
            'BidCompanyName': 'first',
            'BidCompanyType': 'first',
        }

        # Aggregate
        print(f"\nAggregating to BidId level...")
        self.df_clean = self.df_raw.groupby('BidId', as_index=False).agg(agg_dict)

        final_rows = len(self.df_clean)
        rows_removed = initial_rows - final_rows

        print(f"\n✓ Aggregation complete")
        print(f"  Final rows: {final_rows:,}")
        print(f"  Rows removed: {rows_removed:,} ({rows_removed/initial_rows*100:.2f}%)")
        print(f"  Now: 1 row per unique BidId (no duplicates)")
        print()

        self.preprocessing_stats['duplicate_rows_removed'] = rows_removed
        self.preprocessing_stats['rows_after_deduplication'] = final_rows

    def filter_relevant_bids(self):
        """
        Filter to bids relevant for modeling.
        Keep: Won, Lost, Placed (clear outcomes)
        Remove: Active (no outcome yet), Declined (didn't pursue), Approved (unclear)
        """
        print("="*80)
        print("FILTERING RELEVANT BIDS")
        print("="*80)

        initial_rows = len(self.df_clean)

        print(f"\nBid Status Distribution (before filtering):")
        status_counts = self.df_clean['BidStatusName'].value_counts()
        for status, count in status_counts.items():
            pct = count / initial_rows * 100
            print(f"  {status:>10}: {count:>7,} ({pct:>5.2f}%)")

        # Keep only Won, Lost, Placed for modeling
        relevant_statuses = ['Won', 'Lost', 'Placed']
        self.df_clean = self.df_clean[self.df_clean['BidStatusName'].isin(relevant_statuses)].copy()

        final_rows = len(self.df_clean)
        rows_removed = initial_rows - final_rows

        print(f"\n✓ Filtered to relevant bid statuses: {relevant_statuses}")
        print(f"  Rows retained: {final_rows:,}")
        print(f"  Rows removed: {rows_removed:,} ({rows_removed/initial_rows*100:.2f}%)")

        print(f"\nBid Status Distribution (after filtering):")
        status_counts_after = self.df_clean['BidStatusName'].value_counts()
        for status, count in status_counts_after.items():
            pct = count / final_rows * 100
            print(f"  {status:>10}: {count:>7,} ({pct:>5.2f}%)")
        print()

        self.preprocessing_stats['rows_after_filtering'] = final_rows
        self.preprocessing_stats['rows_filtered_out'] = rows_removed

    def handle_missing_target_variable(self):
        """Handle missing values in BidFee (target variable)"""
        print("="*80)
        print("HANDLING MISSING TARGET VARIABLE (BidFee)")
        print("="*80)

        missing_bidfee = self.df_clean['BidFee'].isnull().sum()
        missing_pct = missing_bidfee / len(self.df_clean) * 100

        print(f"\nBidFee missing values:")
        print(f"  Count: {missing_bidfee:,} ({missing_pct:.2f}%)")

        if missing_pct > 5:
            print(f"  ⚠️ Warning: >5% missing - this could impact model quality")

        # Remove rows with missing BidFee (can't train without target)
        initial_rows = len(self.df_clean)
        self.df_clean = self.df_clean[self.df_clean['BidFee'].notna()].copy()
        final_rows = len(self.df_clean)

        print(f"\n✓ Removed {initial_rows - final_rows:,} rows with missing BidFee")
        print(f"  Rows remaining: {final_rows:,}")
        print()

        self.preprocessing_stats['rows_missing_bidfee_removed'] = initial_rows - final_rows
        self.preprocessing_stats['rows_after_bidfee_filter'] = final_rows

    def handle_outliers_in_bidfee(self):
        """Handle extreme outliers in BidFee"""
        print("="*80)
        print("HANDLING OUTLIERS IN BidFee")
        print("="*80)

        print(f"\nBidFee distribution (before outlier treatment):")
        print(self.df_clean['BidFee'].describe())

        # Calculate percentiles
        p99 = self.df_clean['BidFee'].quantile(0.99)
        p999 = self.df_clean['BidFee'].quantile(0.999)
        max_val = self.df_clean['BidFee'].max()

        print(f"\nPercentile Analysis:")
        print(f"  99th percentile: ${p99:,.2f}")
        print(f"  99.9th percentile: ${p999:,.2f}")
        print(f"  Maximum: ${max_val:,.2f}")

        # Count extreme outliers
        outliers_above_p99 = (self.df_clean['BidFee'] > p99).sum()
        outliers_above_p999 = (self.df_clean['BidFee'] > p999).sum()

        print(f"\nOutlier counts:")
        print(f"  Above 99th percentile: {outliers_above_p99:,} ({outliers_above_p99/len(self.df_clean)*100:.2f}%)")
        print(f"  Above 99.9th percentile: {outliers_above_p999:,} ({outliers_above_p999/len(self.df_clean)*100:.2f}%)")

        # Cap at 99th percentile (keeps 99% of data intact)
        print(f"\n✓ Capping BidFee at 99th percentile: ${p99:,.2f}")
        self.df_clean['BidFee_Original'] = self.df_clean['BidFee'].copy()
        self.df_clean['BidFee'] = self.df_clean['BidFee'].clip(upper=p99)

        capped_count = (self.df_clean['BidFee_Original'] > p99).sum()
        print(f"  Capped {capped_count:,} values ({capped_count/len(self.df_clean)*100:.2f}%)")
        print()

        self.preprocessing_stats['bidfee_capped_at_p99'] = float(p99)
        self.preprocessing_stats['bidfee_values_capped'] = int(capped_count)

    def handle_missing_targettime(self):
        """Handle missing values in TargetTime (critical driver)"""
        print("="*80)
        print("HANDLING MISSING TargetTime")
        print("="*80)

        missing_count = self.df_clean['TargetTime'].isnull().sum()
        missing_pct = missing_count / len(self.df_clean) * 100

        print(f"\nTargetTime missing values:")
        print(f"  Count: {missing_count:,} ({missing_pct:.2f}%)")

        # Analyze missingness pattern
        print(f"\nMissingness by BidStatus:")
        for status in self.df_clean['BidStatusName'].unique():
            status_df = self.df_clean[self.df_clean['BidStatusName'] == status]
            status_missing = status_df['TargetTime'].isnull().sum()
            status_missing_pct = status_missing / len(status_df) * 100
            print(f"  {status:>10}: {status_missing_pct:>6.2f}% missing")

        # Imputation strategy: Group median by Property Type + BidStatus
        print(f"\nImputation Strategy: Group median (Property Type × Bid Status)")

        # Calculate group medians
        group_medians = self.df_clean.groupby(['Bid_Property_Type', 'BidStatusName'])['TargetTime'].median()

        # Impute missing values
        def impute_targettime(row):
            if pd.isna(row['TargetTime']):
                key = (row['Bid_Property_Type'], row['BidStatusName'])
                if key in group_medians.index:
                    return group_medians[key]
                else:
                    # Fallback to overall median
                    return self.df_clean['TargetTime'].median()
            else:
                return row['TargetTime']

        self.df_clean['TargetTime_Original'] = self.df_clean['TargetTime'].copy()
        self.df_clean['TargetTime'] = self.df_clean.apply(impute_targettime, axis=1)

        # Verify imputation
        remaining_missing = self.df_clean['TargetTime'].isnull().sum()
        imputed_count = missing_count - remaining_missing

        print(f"\n✓ Imputation complete")
        print(f"  Values imputed: {imputed_count:,}")
        print(f"  Remaining missing: {remaining_missing:,}")
        print()

        self.preprocessing_stats['targettime_values_imputed'] = int(imputed_count)
        self.preprocessing_stats['targettime_remaining_missing'] = int(remaining_missing)

    def handle_missing_categorical_features(self):
        """Handle missing values in categorical features"""
        print("="*80)
        print("HANDLING MISSING CATEGORICAL FEATURES")
        print("="*80)

        categorical_cols = [
            'Bid_Property_Type', 'Bid_SubProperty_Type', 'Bid_SpecificUseProperty_Type',
            'PropertyState', 'PropertyCity', 'Market', 'Submarket',
            'BusinessSegment', 'Bid_JobPurpose', 'Bid_Deliverable',
            'OfficeLocation', 'BidCompanyName', 'BidCompanyType',
            'GrossBuildingAreaRange', 'YearBuiltRange', 'MarketOrientation'
        ]

        print(f"\nFilling missing categorical values with 'Unknown':")

        imputation_summary = []

        for col in categorical_cols:
            if col in self.df_clean.columns:
                missing_before = self.df_clean[col].isnull().sum()
                if missing_before > 0:
                    self.df_clean[col] = self.df_clean[col].fillna('Unknown')
                    missing_after = self.df_clean[col].isnull().sum()
                    filled = missing_before - missing_after
                    pct = filled / len(self.df_clean) * 100
                    print(f"  {col:>35}: {filled:>7,} filled ({pct:>5.2f}%)")
                    imputation_summary.append({
                        'Column': col,
                        'Values_Filled': filled,
                        'Percent': pct
                    })

        print(f"\n✓ Categorical imputation complete")
        print()

        # Save imputation summary
        if imputation_summary:
            imputation_df = pd.DataFrame(imputation_summary)
            imputation_df.to_csv(self.reports_dir / 'categorical_imputation_report.csv', index=False)
            print(f"✓ Categorical imputation report saved")

    def handle_outliers_numerical_features(self):
        """Handle outliers in other numerical features"""
        print("="*80)
        print("HANDLING OUTLIERS IN NUMERICAL FEATURES")
        print("="*80)

        numerical_cols = [
            'DistanceInMiles', 'PopulationEstimate', 'AverageHouseValue',
            'IncomePerHousehold', 'MedianAge', 'ZipPopulation'
        ]

        print(f"\nCapping extreme outliers at 99th percentile:")

        for col in numerical_cols:
            if col in self.df_clean.columns:
                p99 = self.df_clean[col].quantile(0.99)
                outliers = (self.df_clean[col] > p99).sum()
                if outliers > 0:
                    self.df_clean[col] = self.df_clean[col].clip(upper=p99)
                    pct = outliers / len(self.df_clean) * 100
                    print(f"  {col:>25}: {outliers:>6,} values capped ({pct:>5.2f}%) at {p99:,.0f}")

        print(f"\n✓ Numerical outlier treatment complete")
        print()

    def create_temporal_features(self):
        """Create basic temporal features for modeling"""
        print("="*80)
        print("CREATING TEMPORAL FEATURES")
        print("="*80)

        print(f"\nExtracting temporal features from BidDate:")

        # Extract features
        self.df_clean['Year'] = self.df_clean['BidDate'].dt.year
        self.df_clean['Quarter'] = self.df_clean['BidDate'].dt.quarter
        self.df_clean['Month'] = self.df_clean['BidDate'].dt.month
        self.df_clean['DayOfWeek'] = self.df_clean['BidDate'].dt.dayofweek
        self.df_clean['DayOfMonth'] = self.df_clean['BidDate'].dt.day
        self.df_clean['WeekOfYear'] = self.df_clean['BidDate'].dt.isocalendar().week

        # Create cyclical encodings for periodic features
        self.df_clean['Month_sin'] = np.sin(2 * np.pi * self.df_clean['Month'] / 12)
        self.df_clean['Month_cos'] = np.cos(2 * np.pi * self.df_clean['Month'] / 12)
        self.df_clean['DayOfWeek_sin'] = np.sin(2 * np.pi * self.df_clean['DayOfWeek'] / 7)
        self.df_clean['DayOfWeek_cos'] = np.cos(2 * np.pi * self.df_clean['DayOfWeek'] / 7)

        print(f"  ✓ Year, Quarter, Month, DayOfWeek, DayOfMonth, WeekOfYear")
        print(f"  ✓ Cyclical encodings (Month_sin/cos, DayOfWeek_sin/cos)")
        print()

    def create_target_variable_for_classification(self):
        """Create binary target variable for Win Probability model (Phase 2)"""
        print("="*80)
        print("CREATING CLASSIFICATION TARGET")
        print("="*80)

        # Binary target: Won = 1, else = 0
        self.df_clean['Won'] = (self.df_clean['BidStatusName'] == 'Won').astype(int)

        won_count = self.df_clean['Won'].sum()
        lost_count = len(self.df_clean) - won_count
        won_pct = won_count / len(self.df_clean) * 100

        print(f"\nBinary classification target created:")
        print(f"  Won (1): {won_count:>7,} ({won_pct:>5.2f}%)")
        print(f"  Lost (0): {lost_count:>7,} ({100-won_pct:>5.2f}%)")
        print()

        self.preprocessing_stats['win_rate'] = float(won_pct)

    def validate_data_quality(self):
        """Final data quality validation"""
        print("="*80)
        print("DATA QUALITY VALIDATION")
        print("="*80)

        # Check for remaining missing values in critical columns
        critical_cols = ['BidFee', 'BidStatusName', 'BidDate', 'Bid_Property_Type', 'OfficeLocation']

        print(f"\nCritical columns - Missing values:")
        all_clean = True
        for col in critical_cols:
            if col in self.df_clean.columns:
                missing = self.df_clean[col].isnull().sum()
                pct = missing / len(self.df_clean) * 100
                status = "✓" if missing == 0 else "⚠️"
                print(f"  {status} {col:>25}: {missing:>7,} ({pct:>5.2f}%)")
                if missing > 0:
                    all_clean = False

        if all_clean:
            print(f"\n✓ All critical columns have no missing values")
        else:
            print(f"\n⚠️ Some critical columns still have missing values")

        # Data type validation
        print(f"\nData types:")
        print(f"  Numerical: {self.df_clean.select_dtypes(include=[np.number]).shape[1]}")
        print(f"  Categorical: {self.df_clean.select_dtypes(include=['object']).shape[1]}")
        print(f"  DateTime: {self.df_clean.select_dtypes(include=['datetime64']).shape[1]}")

        # Final row count
        print(f"\nFinal dataset:")
        print(f"  Rows: {len(self.df_clean):,}")
        print(f"  Columns: {len(self.df_clean.columns)}")
        print()

    def save_processed_data(self):
        """Save processed dataset"""
        print("="*80)
        print("SAVING PROCESSED DATA")
        print("="*80)

        # Save main processed dataset
        output_path = self.processed_dir / 'BidData_processed.csv'
        self.df_clean.to_csv(output_path, index=False)

        print(f"\n✓ Processed data saved:")
        print(f"  Location: {output_path}")
        print(f"  Size: {output_path.stat().st_size / (1024*1024):.1f} MB")
        print(f"  Rows: {len(self.df_clean):,}")
        print(f"  Columns: {len(self.df_clean.columns)}")
        print()

    def generate_preprocessing_report(self):
        """Generate comprehensive preprocessing report"""
        print("="*80)
        print("GENERATING PREPROCESSING REPORT")
        print("="*80)

        # Create detailed report
        report = {
            'Preprocessing Summary': {
                'Input File': str(self.input_path),
                'Output File': str(self.processed_dir / 'BidData_processed.csv'),
                'Processing Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            },
            'Row Transformations': {
                'Initial Rows': int(self.preprocessing_stats.get('initial_rows', 0)),
                'After Deduplication': int(self.preprocessing_stats.get('rows_after_deduplication', 0)),
                'After Status Filtering': int(self.preprocessing_stats.get('rows_after_filtering', 0)),
                'After BidFee Filter': int(self.preprocessing_stats.get('rows_after_bidfee_filter', 0)),
                'Final Rows': len(self.df_clean),
                'Total Rows Removed': int(self.preprocessing_stats.get('initial_rows', 0)) - len(self.df_clean),
                'Retention Rate': f"{len(self.df_clean) / self.preprocessing_stats.get('initial_rows', 1) * 100:.2f}%",
            },
            'Specific Transformations': {
                'Duplicate Rows Removed': int(self.preprocessing_stats.get('duplicate_rows_removed', 0)),
                'Rows Filtered Out (Status)': int(self.preprocessing_stats.get('rows_filtered_out', 0)),
                'Rows Missing BidFee Removed': int(self.preprocessing_stats.get('rows_missing_bidfee_removed', 0)),
                'BidFee Values Capped': int(self.preprocessing_stats.get('bidfee_values_capped', 0)),
                'BidFee Cap Value': f"${self.preprocessing_stats.get('bidfee_capped_at_p99', 0):,.2f}",
                'TargetTime Values Imputed': int(self.preprocessing_stats.get('targettime_values_imputed', 0)),
            },
            'Final Data Quality': {
                'Total Columns': len(self.df_clean.columns),
                'Numerical Columns': int(self.df_clean.select_dtypes(include=[np.number]).shape[1]),
                'Categorical Columns': int(self.df_clean.select_dtypes(include=['object']).shape[1]),
                'DateTime Columns': int(self.df_clean.select_dtypes(include=['datetime64']).shape[1]),
                'Win Rate': f"{self.preprocessing_stats.get('win_rate', 0):.2f}%",
            }
        }

        # Print report
        for section, metrics in report.items():
            print(f"\n{section}:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")

        # Save report
        import json
        report_path = self.reports_dir / 'preprocessing_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Preprocessing report saved to: {report_path}")
        print()

    def run_full_preprocessing(self):
        """Execute complete preprocessing pipeline"""
        print("\n" + "="*80)
        print("STARTING DATA PREPROCESSING")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Execute all preprocessing steps
        self.load_data()
        self.handle_duplicate_bidids()
        self.filter_relevant_bids()
        self.handle_missing_target_variable()
        self.handle_outliers_in_bidfee()
        self.handle_missing_targettime()
        self.handle_missing_categorical_features()
        self.handle_outliers_numerical_features()
        self.create_temporal_features()
        self.create_target_variable_for_classification()
        self.validate_data_quality()
        self.save_processed_data()
        self.generate_preprocessing_report()

        print("="*80)
        print("DATA PREPROCESSING COMPLETE")
        print("="*80)
        print(f"\n✓ Processed data ready for modeling")
        print(f"  Location: {self.processed_dir / 'BidData_processed.csv'}")
        print(f"  Rows: {len(self.df_clean):,}")
        print(f"  Columns: {len(self.df_clean.columns)}")
        print()


def main():
    """Main execution function"""
    # Define paths
    PROJECT_ROOT = Path(__file__).parent.parent
    INPUT_PATH = PROJECT_ROOT / 'data' / 'processed' / 'BidData_cleaned.csv'
    OUTPUT_DIR = PROJECT_ROOT / 'outputs'

    # Initialize and run preprocessing
    preprocessor = BidDataPreprocessor(input_path=INPUT_PATH, output_dir=OUTPUT_DIR)
    preprocessor.run_full_preprocessing()


if __name__ == "__main__":
    main()
