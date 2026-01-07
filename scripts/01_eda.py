"""
Exploratory Data Analysis - Bid Recommendation System

Objective: Comprehensive analysis of bid fee behavior, data quality, and patterns
           that will inform feature engineering and modeling.

Author: Ujjawal Dwivedi
Organization: Global Stat Solutions (GSS)
Date: 2026-01-02
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
from scipy import stats

# Configuration
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


class BidEDA:
    """Comprehensive Exploratory Data Analysis for Bid Recommendation System"""

    def __init__(self, data_path: Path, output_dir: Path):
        """
        Initialize EDA analyzer

        Args:
            data_path: Path to cleaned bid data CSV
            output_dir: Directory to save outputs (figures, reports)
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.figures_dir = output_dir / 'figures'
        self.reports_dir = output_dir / 'reports'

        # Create output directories
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Data
        self.df = None

    def load_data(self):
        """Load and perform initial data preparation"""
        print("="*80)
        print("LOADING DATA")
        print("="*80)

        self.df = pd.read_csv(self.data_path)

        print(f"✓ Dataset loaded successfully!")
        print(f"  Shape: {self.df.shape}")
        print(f"  Rows: {self.df.shape[0]:,}")
        print(f"  Columns: {self.df.shape[1]}")

        # Convert date columns
        self.df['BidDate'] = pd.to_datetime(self.df['BidDate'], errors='coerce')
        self.df['Bid_DueDate'] = pd.to_datetime(self.df['Bid_DueDate'], errors='coerce')

        # Extract temporal features
        self.df['Year'] = self.df['BidDate'].dt.year
        self.df['Month'] = self.df['BidDate'].dt.month
        self.df['Quarter'] = self.df['BidDate'].dt.quarter
        self.df['DayOfWeek'] = self.df['BidDate'].dt.dayofweek
        self.df['YearMonth'] = self.df['BidDate'].dt.to_period('M')

        print(f"✓ Date columns converted and temporal features extracted")
        print(f"  Date range: {self.df['BidDate'].min()} to {self.df['BidDate'].max()}")
        print(f"  Years covered: {self.df['Year'].min():.0f} - {self.df['Year'].max():.0f}")
        print()

    def analyze_data_quality(self):
        """Comprehensive data quality assessment"""
        print("="*80)
        print("DATA QUALITY ASSESSMENT")
        print("="*80)

        # Missing values
        missing_data = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum(),
            'Missing_Percent': (self.df.isnull().sum() / len(self.df) * 100).round(2)
        })

        missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values(
            'Missing_Percent', ascending=False
        )

        print(f"\nColumns with missing values: {len(missing_data)}/{len(self.df.columns)}")
        print("\nTop 10 columns by missing percentage:")
        print(missing_data.head(10).to_string(index=False))

        # Save full report
        missing_data.to_csv(self.reports_dir / 'missing_values_report.csv', index=False)
        print(f"\n✓ Full missing values report saved")

        # Key columns check
        key_columns = ['BidFee', 'TargetTime', 'BidDate', 'BidStatusName',
                      'Bid_Property_Type', 'OfficeLocation']

        print("\nKey columns missing data:")
        for col in key_columns:
            if col in self.df.columns:
                missing_pct = self.df[col].isnull().sum() / len(self.df) * 100
                print(f"  {col}: {missing_pct:.2f}%")

        # Duplicates
        duplicates = self.df.duplicated(subset=['BidId']).sum()
        print(f"\nDuplicate BidIds: {duplicates:,} ({duplicates/len(self.df)*100:.2f}%)")

        # Data types summary
        print(f"\nData types distribution:")
        print(self.df.dtypes.value_counts())
        print()

    def analyze_target_variable(self):
        """Comprehensive analysis of BidFee (target variable)"""
        print("="*80)
        print("TARGET VARIABLE ANALYSIS (BidFee)")
        print("="*80)

        # Summary statistics
        print("\nBidFee Summary Statistics:")
        print(self.df['BidFee'].describe())
        print(f"\nMissing: {self.df['BidFee'].isnull().sum():,} "
              f"({self.df['BidFee'].isnull().sum()/len(self.df)*100:.2f}%)")

        # Additional statistics
        print(f"\nSkewness: {self.df['BidFee'].skew():.2f}")
        print(f"Kurtosis: {self.df['BidFee'].kurtosis():.2f}")

        # Percentiles
        print("\nPercentile Analysis:")
        percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        for p in percentiles:
            val = self.df['BidFee'].quantile(p)
            print(f"  {p*100:5.1f}th percentile: ${val:,.2f}")

        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Original distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(self.df['BidFee'].dropna(), bins=50, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Bid Fee ($)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Bid Fee (Original Scale)')
        ax1.grid(True, alpha=0.3)

        # 2. Log-transformed distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(np.log10(self.df['BidFee'].dropna() + 1), bins=50,
                edgecolor='black', alpha=0.7, color='green')
        ax2.set_xlabel('Log10(Bid Fee + 1)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Bid Fee (Log Scale)')
        ax2.grid(True, alpha=0.3)

        # 3. Box plot (capped at 99th percentile for visibility)
        ax3 = fig.add_subplot(gs[0, 2])
        cap_value = self.df['BidFee'].quantile(0.99)
        capped_fees = self.df['BidFee'].clip(upper=cap_value)
        ax3.boxplot(capped_fees.dropna(), vert=True)
        ax3.set_ylabel('Bid Fee ($)')
        ax3.set_title(f'Box Plot (Capped at 99th percentile: ${cap_value:,.0f})')
        ax3.grid(True, alpha=0.3)

        # 4. BidFee by Status
        ax4 = fig.add_subplot(gs[1, 0])
        status_order = self.df.groupby('BidStatusName')['BidFee'].median().sort_values(ascending=False).index
        sns.boxplot(data=self.df, y='BidStatusName', x='BidFee', order=status_order, ax=ax4)
        ax4.set_xlim(0, self.df['BidFee'].quantile(0.95))
        ax4.set_xlabel('Bid Fee ($)')
        ax4.set_ylabel('Bid Status')
        ax4.set_title('Bid Fee Distribution by Status (95th percentile cap)')
        ax4.grid(True, alpha=0.3)

        # 5. BidFee over time (monthly median)
        ax5 = fig.add_subplot(gs[1, 1])
        monthly_fee = self.df.groupby('YearMonth')['BidFee'].median()
        monthly_fee.plot(ax=ax5, marker='o', linewidth=2)
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Median Bid Fee ($)')
        ax5.set_title('Median Bid Fee Over Time (Monthly)')
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='x', rotation=45)

        # 6. Q-Q Plot for normality check
        ax6 = fig.add_subplot(gs[1, 2])
        stats.probplot(np.log10(self.df['BidFee'].dropna() + 1), dist="norm", plot=ax6)
        ax6.set_title('Q-Q Plot (Log-Transformed BidFee)')
        ax6.grid(True, alpha=0.3)

        plt.savefig(self.figures_dir / 'bidfee_comprehensive_analysis.png',
                   dpi=300, bbox_inches='tight')
        print(f"\n✓ Comprehensive analysis plot saved")
        plt.close()
        print()

    def analyze_time_series_patterns(self):
        """Comprehensive temporal pattern analysis"""
        print("="*80)
        print("TIME SERIES PATTERNS")
        print("="*80)

        # Aggregate statistics
        bids_by_month = self.df.groupby('YearMonth').size()
        print(f"\nBid Volume Statistics:")
        print(f"  Bids per month - Min: {bids_by_month.min()}, Max: {bids_by_month.max()}, Avg: {bids_by_month.mean():.0f}")
        print(f"  Total months in data: {len(bids_by_month)}")

        # Year-over-year analysis
        print(f"\nYear-over-Year Bid Counts:")
        yearly_counts = self.df.groupby('Year').size()
        for year, count in yearly_counts.items():
            print(f"  {year:.0f}: {count:,} bids")

        # Win rate over time
        win_rate_monthly = self.df.groupby('YearMonth').apply(
            lambda x: (x['BidStatusName'] == 'Won').sum() / len(x) * 100
        )
        print(f"\nWin Rate Statistics:")
        print(f"  Overall: {(self.df['BidStatusName'] == 'Won').sum() / len(self.df) * 100:.2f}%")
        print(f"  Monthly - Min: {win_rate_monthly.min():.2f}%, Max: {win_rate_monthly.max():.2f}%, Avg: {win_rate_monthly.mean():.2f}%")

        # Seasonality check
        print(f"\nSeasonality Analysis (by Month):")
        monthly_avg = self.df.groupby('Month').size()
        print("  Bids by month of year:")
        for month, count in monthly_avg.items():
            month_name = pd.Timestamp(2024, int(month), 1).strftime('%B')
            print(f"    {month_name:>10}: {count:,}")

        # Create visualization
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Bid volume over time
        ax1 = fig.add_subplot(gs[0, :])
        bids_by_month.plot(ax=ax1, marker='o', linewidth=2, markersize=4)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Number of Bids')
        ax1.set_title('Bid Volume Over Time (Monthly)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. Average fee over time
        ax2 = fig.add_subplot(gs[1, 0])
        monthly_avg_fee = self.df.groupby('YearMonth')['BidFee'].mean()
        monthly_median_fee = self.df.groupby('YearMonth')['BidFee'].median()
        ax2.plot(monthly_avg_fee.index.to_timestamp(), monthly_avg_fee.values,
                marker='o', label='Mean', linewidth=2, markersize=4)
        ax2.plot(monthly_median_fee.index.to_timestamp(), monthly_median_fee.values,
                marker='s', label='Median', linewidth=2, markersize=4)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Bid Fee ($)')
        ax2.set_title('Average Bid Fee Over Time (Monthly)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        # 3. Win rate over time
        ax3 = fig.add_subplot(gs[1, 1])
        win_rate_monthly.plot(ax=ax3, marker='o', linewidth=2, markersize=4, color='green')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Win Rate (%)')
        ax3.set_title('Win Rate Over Time (Monthly)')
        ax3.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% baseline')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)

        # 4. Seasonality - Bids by month
        ax4 = fig.add_subplot(gs[2, 0])
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_avg.plot(kind='bar', ax=ax4, color='skyblue', edgecolor='black')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Total Bids')
        ax4.set_title('Bid Volume by Month of Year (Seasonality)')
        ax4.set_xticklabels(month_names, rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. Bids by day of week
        ax5 = fig.add_subplot(gs[2, 1])
        dow_counts = self.df.groupby('DayOfWeek').size()
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_counts.plot(kind='bar', ax=ax5, color='coral', edgecolor='black')
        ax5.set_xlabel('Day of Week')
        ax5.set_ylabel('Total Bids')
        ax5.set_title('Bid Volume by Day of Week')
        ax5.set_xticklabels(dow_names, rotation=45)
        ax5.grid(True, alpha=0.3, axis='y')

        plt.savefig(self.figures_dir / 'time_series_analysis.png',
                   dpi=300, bbox_inches='tight')
        print(f"\n✓ Time series analysis plot saved")
        plt.close()
        print()

    def analyze_target_time(self):
        """Deep dive into TargetTime (critical driver)"""
        print("="*80)
        print("TARGET TIME ANALYSIS")
        print("="*80)

        # Missing data analysis
        missing_count = self.df['TargetTime'].isnull().sum()
        missing_pct = missing_count / len(self.df) * 100
        print(f"\nMissing Data:")
        print(f"  Count: {missing_count:,} ({missing_pct:.2f}%)")

        # Summary statistics (non-null values)
        valid_target_time = self.df['TargetTime'].dropna()
        print(f"\nTargetTime Summary Statistics ({len(valid_target_time):,} non-null values):")
        print(valid_target_time.describe())

        # Relationship with BidFee
        correlation = self.df[['TargetTime', 'BidFee']].corr().iloc[0, 1]
        print(f"\nCorrelation with BidFee: {correlation:.4f}")

        # Missingness pattern analysis
        print(f"\nMissingness Pattern Analysis:")
        print(f"  Missing TargetTime by BidStatus:")
        missing_by_status = self.df.groupby('BidStatusName')['TargetTime'].apply(
            lambda x: x.isnull().sum() / len(x) * 100
        ).sort_values(ascending=False)
        for status, pct in missing_by_status.items():
            print(f"    {status:>10}: {pct:>6.2f}% missing")

        # Create visualization
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Distribution of TargetTime
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(valid_target_time, bins=50, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Target Time (days)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Target Time')
        ax1.grid(True, alpha=0.3)

        # 2. Box plot by BidStatus
        ax2 = fig.add_subplot(gs[0, 1])
        status_with_data = self.df[self.df['TargetTime'].notna()]
        sns.boxplot(data=status_with_data, y='BidStatusName', x='TargetTime', ax=ax2)
        ax2.set_xlabel('Target Time (days)')
        ax2.set_ylabel('Bid Status')
        ax2.set_title('Target Time by Bid Status')
        ax2.grid(True, alpha=0.3)

        # 3. Scatter: TargetTime vs BidFee
        ax3 = fig.add_subplot(gs[0, 2])
        sample_data = self.df[['TargetTime', 'BidFee']].dropna()
        if len(sample_data) > 10000:
            sample_data = sample_data.sample(10000, random_state=42)
        ax3.scatter(sample_data['TargetTime'], sample_data['BidFee'],
                   alpha=0.3, s=10)
        ax3.set_xlabel('Target Time (days)')
        ax3.set_ylabel('Bid Fee ($)')
        ax3.set_title(f'Target Time vs Bid Fee (corr={correlation:.3f})')
        ax3.set_ylim(0, self.df['BidFee'].quantile(0.95))
        ax3.grid(True, alpha=0.3)

        # 4. Missing rate over time
        ax4 = fig.add_subplot(gs[1, 0])
        missing_rate_monthly = self.df.groupby('YearMonth')['TargetTime'].apply(
            lambda x: x.isnull().sum() / len(x) * 100
        )
        missing_rate_monthly.plot(ax=ax4, marker='o', linewidth=2, color='red')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Missing Rate (%)')
        ax4.set_title('TargetTime Missing Rate Over Time')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)

        # 5. Average TargetTime over time
        ax5 = fig.add_subplot(gs[1, 1])
        avg_target_time = self.df.groupby('YearMonth')['TargetTime'].mean()
        avg_target_time.plot(ax=ax5, marker='o', linewidth=2, color='purple')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Average Target Time (days)')
        ax5.set_title('Average Target Time Over Time')
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='x', rotation=45)

        # 6. Missing pattern by PropertyType
        ax6 = fig.add_subplot(gs[1, 2])
        missing_by_proptype = self.df.groupby('Bid_Property_Type')['TargetTime'].apply(
            lambda x: x.isnull().sum() / len(x) * 100
        ).sort_values(ascending=False).head(10)
        missing_by_proptype.plot(kind='barh', ax=ax6, color='orange', edgecolor='black')
        ax6.set_xlabel('Missing Rate (%)')
        ax6.set_ylabel('Property Type')
        ax6.set_title('TargetTime Missing Rate by Property Type (Top 10)')
        ax6.grid(True, alpha=0.3, axis='x')

        plt.savefig(self.figures_dir / 'targettime_analysis.png',
                   dpi=300, bbox_inches='tight')
        print(f"\n✓ TargetTime analysis plot saved")
        plt.close()
        print()

    def analyze_categorical_features(self):
        """Comprehensive categorical feature analysis"""
        print("="*80)
        print("CATEGORICAL FEATURES ANALYSIS")
        print("="*80)

        # BidStatusName distribution
        print(f"\nBid Status Distribution:")
        status_counts = self.df['BidStatusName'].value_counts()
        for status, count in status_counts.items():
            pct = count / len(self.df) * 100
            print(f"  {status:>10}: {count:>7,} ({pct:>5.2f}%)")

        # Property Type distribution
        print(f"\nTop 10 Property Types:")
        prop_counts = self.df['Bid_Property_Type'].value_counts().head(10)
        for prop_type, count in prop_counts.items():
            pct = count / len(self.df) * 100
            print(f"  {prop_type:>20}: {count:>7,} ({pct:>5.2f}%)")

        # Office Location distribution
        print(f"\nTop 10 Office Locations:")
        office_counts = self.df['OfficeLocation'].value_counts().head(10)
        for office, count in office_counts.items():
            pct = count / len(self.df) * 100
            print(f"  {office:>30}: {count:>7,} ({pct:>5.2f}%)")

        # Business Segment
        print(f"\nTop Business Segments:")
        segment_counts = self.df['BusinessSegment'].value_counts().head(10)
        for segment, count in segment_counts.items():
            pct = count / len(self.df) * 100
            print(f"  {segment:>20}: {count:>7,} ({pct:>5.2f}%)")

        # Create visualization
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

        # 1. Bid Status
        ax1 = fig.add_subplot(gs[0, 0])
        status_counts.plot(kind='bar', ax=ax1, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Bid Status')
        ax1.set_ylabel('Count')
        ax1.set_title('Distribution of Bid Status')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(status_counts.values):
            ax1.text(i, v, f'{v:,}', ha='center', va='bottom')

        # 2. Property Type (top 15)
        ax2 = fig.add_subplot(gs[0, 1])
        prop_counts_top15 = self.df['Bid_Property_Type'].value_counts().head(15)
        prop_counts_top15.plot(kind='barh', ax=ax2, color='coral', edgecolor='black')
        ax2.set_xlabel('Count')
        ax2.set_ylabel('Property Type')
        ax2.set_title('Top 15 Property Types')
        ax2.grid(True, alpha=0.3, axis='x')

        # 3. Office Location (top 15)
        ax3 = fig.add_subplot(gs[1, 0])
        office_counts_top15 = self.df['OfficeLocation'].value_counts().head(15)
        office_counts_top15.plot(kind='barh', ax=ax3, color='lightgreen', edgecolor='black')
        ax3.set_xlabel('Count')
        ax3.set_ylabel('Office Location')
        ax3.set_title('Top 15 Office Locations')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. Business Segment
        ax4 = fig.add_subplot(gs[1, 1])
        segment_counts.plot(kind='barh', ax=ax4, color='gold', edgecolor='black')
        ax4.set_xlabel('Count')
        ax4.set_ylabel('Business Segment')
        ax4.set_title('Top 10 Business Segments')
        ax4.grid(True, alpha=0.3, axis='x')

        # 5. Win Rate by Property Type (top 15)
        ax5 = fig.add_subplot(gs[2, 0])
        top_props = self.df['Bid_Property_Type'].value_counts().head(15).index
        win_rate_by_prop = self.df[self.df['Bid_Property_Type'].isin(top_props)].groupby(
            'Bid_Property_Type'
        ).apply(lambda x: (x['BidStatusName'] == 'Won').sum() / len(x) * 100).sort_values(ascending=True)
        win_rate_by_prop.plot(kind='barh', ax=ax5, color='mediumseagreen', edgecolor='black')
        ax5.set_xlabel('Win Rate (%)')
        ax5.set_ylabel('Property Type')
        ax5.set_title('Win Rate by Property Type (Top 15 by volume)')
        ax5.axvline(x=50, color='r', linestyle='--', alpha=0.5)
        ax5.grid(True, alpha=0.3, axis='x')

        # 6. Average BidFee by Property Type (top 15)
        ax6 = fig.add_subplot(gs[2, 1])
        avg_fee_by_prop = self.df[self.df['Bid_Property_Type'].isin(top_props)].groupby(
            'Bid_Property_Type'
        )['BidFee'].median().sort_values(ascending=True)
        avg_fee_by_prop.plot(kind='barh', ax=ax6, color='mediumpurple', edgecolor='black')
        ax6.set_xlabel('Median Bid Fee ($)')
        ax6.set_ylabel('Property Type')
        ax6.set_title('Median Bid Fee by Property Type (Top 15 by volume)')
        ax6.grid(True, alpha=0.3, axis='x')

        plt.savefig(self.figures_dir / 'categorical_features_analysis.png',
                   dpi=300, bbox_inches='tight')
        print(f"\n✓ Categorical features analysis plot saved")
        plt.close()
        print()

    def analyze_geographic_patterns(self):
        """Geographic pattern analysis"""
        print("="*80)
        print("GEOGRAPHIC PATTERNS ANALYSIS")
        print("="*80)

        # State distribution
        print(f"\nTop 15 States by Bid Volume:")
        state_counts = self.df['PropertyState'].value_counts().head(15)
        for state, count in state_counts.items():
            pct = count / len(self.df) * 100
            print(f"  {state:>15}: {count:>7,} ({pct:>5.2f}%)")

        # Market distribution
        print(f"\nTop 10 Markets:")
        market_counts = self.df['Market'].value_counts().head(10)
        for market, count in market_counts.items():
            pct = count / len(self.df) * 100
            market_short = str(market)[:40]
            print(f"  {market_short:>40}: {count:>7,} ({pct:>5.2f}%)")

        # Distance analysis
        print(f"\nDistance Analysis:")
        print(self.df['DistanceInMiles'].describe())

        # Create visualization
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Top states by volume
        ax1 = fig.add_subplot(gs[0, 0])
        state_counts_top15 = self.df['PropertyState'].value_counts().head(15)
        state_counts_top15.plot(kind='barh', ax=ax1, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Number of Bids')
        ax1.set_ylabel('State')
        ax1.set_title('Top 15 States by Bid Volume')
        ax1.grid(True, alpha=0.3, axis='x')

        # 2. Win rate by state (top 15 states)
        ax2 = fig.add_subplot(gs[0, 1])
        top_states = state_counts_top15.index
        win_rate_by_state = self.df[self.df['PropertyState'].isin(top_states)].groupby(
            'PropertyState'
        ).apply(lambda x: (x['BidStatusName'] == 'Won').sum() / len(x) * 100).sort_values(ascending=True)
        win_rate_by_state.plot(kind='barh', ax=ax2, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Win Rate (%)')
        ax2.set_ylabel('State')
        ax2.set_title('Win Rate by State (Top 15 by volume)')
        ax2.axvline(x=50, color='r', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3, axis='x')

        # 3. Distance distribution
        ax3 = fig.add_subplot(gs[1, 0])
        valid_distance = self.df['DistanceInMiles'].dropna()
        # Cap at 95th percentile for better visualization
        cap_distance = valid_distance.quantile(0.95)
        capped_distance = valid_distance.clip(upper=cap_distance)
        ax3.hist(capped_distance, bins=50, edgecolor='black', alpha=0.7, color='green')
        ax3.set_xlabel('Distance (miles)')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Distance Distribution (Capped at 95th percentile: {cap_distance:.0f} miles)')
        ax3.grid(True, alpha=0.3)

        # 4. Median BidFee by state (top 15)
        ax4 = fig.add_subplot(gs[1, 1])
        median_fee_by_state = self.df[self.df['PropertyState'].isin(top_states)].groupby(
            'PropertyState'
        )['BidFee'].median().sort_values(ascending=True)
        median_fee_by_state.plot(kind='barh', ax=ax4, color='gold', edgecolor='black')
        ax4.set_xlabel('Median Bid Fee ($)')
        ax4.set_ylabel('State')
        ax4.set_title('Median Bid Fee by State (Top 15 by volume)')
        ax4.grid(True, alpha=0.3, axis='x')

        plt.savefig(self.figures_dir / 'geographic_analysis.png',
                   dpi=300, bbox_inches='tight')
        print(f"\n✓ Geographic analysis plot saved")
        plt.close()
        print()

    def analyze_correlations(self):
        """Correlation analysis for numerical features"""
        print("="*80)
        print("CORRELATION ANALYSIS")
        print("="*80)

        # Select numerical columns
        numerical_cols = ['BidFee', 'TargetTime', 'DistanceInKM', 'DistanceInMiles',
                         'PopulationEstimate', 'AverageHouseValue', 'IncomePerHousehold',
                         'MedianAge', 'DeliveryTotal', 'NumberofBusinesses',
                         'NumberofEmployees', 'ZipPopulation']

        # Filter to existing columns
        numerical_cols = [col for col in numerical_cols if col in self.df.columns]

        # Calculate correlation matrix
        corr_matrix = self.df[numerical_cols].corr()

        # BidFee correlations
        print(f"\nCorrelations with BidFee (sorted by absolute value):")
        bidfee_corr = corr_matrix['BidFee'].drop('BidFee').abs().sort_values(ascending=False)
        for feature, corr_val in bidfee_corr.items():
            actual_corr = corr_matrix.loc[feature, 'BidFee']
            print(f"  {feature:>25}: {actual_corr:>7.4f}")

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # 1. Full correlation heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=axes[0], cbar_kws={'label': 'Correlation'})
        axes[0].set_title('Correlation Matrix - Numerical Features', fontsize=14, fontweight='bold')

        # 2. BidFee correlations bar chart
        bidfee_corr_signed = corr_matrix['BidFee'].drop('BidFee').sort_values()
        colors = ['red' if x < 0 else 'green' for x in bidfee_corr_signed.values]
        bidfee_corr_signed.plot(kind='barh', ax=axes[1], color=colors, edgecolor='black')
        axes[1].set_xlabel('Correlation Coefficient')
        axes[1].set_ylabel('Feature')
        axes[1].set_title('Correlations with BidFee', fontsize=14, fontweight='bold')
        axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        axes[1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'correlation_analysis.png',
                   dpi=300, bbox_inches='tight')
        print(f"\n✓ Correlation analysis plot saved")
        plt.close()

        # Save correlation matrix
        corr_matrix.to_csv(self.reports_dir / 'correlation_matrix.csv')
        print(f"✓ Correlation matrix saved to CSV")
        print()

    def analyze_bid_hierarchy(self):
        """Investigate bid hierarchy and duplicate BidIds"""
        print("="*80)
        print("BID HIERARCHY INVESTIGATION")
        print("="*80)

        # Duplicate BidIds analysis
        dup_bid_ids = self.df[self.df.duplicated(subset=['BidId'], keep=False)].sort_values('BidId')
        unique_dup_bids = dup_bid_ids['BidId'].nunique()

        print(f"\nDuplicate BidId Analysis:")
        print(f"  Total rows with duplicate BidIds: {len(dup_bid_ids):,}")
        print(f"  Number of unique BidIds with duplicates: {unique_dup_bids:,}")
        print(f"  Average rows per duplicate BidId: {len(dup_bid_ids) / unique_dup_bids:.2f}")

        # JobType distribution (Master vs SubJob)
        print(f"\nJobType Distribution:")
        if 'JobType' in self.df.columns:
            job_type_counts = self.df['JobType'].value_counts()
            for job_type, count in job_type_counts.items():
                pct = count / len(self.df) * 100
                print(f"  {job_type:>15}: {count:>7,} ({pct:>5.2f}%)")

        # Job hierarchy pattern
        print(f"\nJob Hierarchy Pattern (JobCount distribution):")
        if 'JobCount' in self.df.columns:
            job_count_dist = self.df['JobCount'].value_counts().head(10).sort_index()
            for count, freq in job_count_dist.items():
                print(f"  JobCount={count:.0f}: {freq:,} bids")

        # Sample of duplicate BidIds
        print(f"\nSample Duplicate BidId Pattern (first 5 unique BidIds):")
        sample_dup_ids = dup_bid_ids['BidId'].unique()[:5]
        for bid_id in sample_dup_ids:
            bid_rows = self.df[self.df['BidId'] == bid_id]
            print(f"\n  BidId {bid_id}: {len(bid_rows)} rows")
            if 'JobType' in bid_rows.columns and 'JobName' in bid_rows.columns:
                for idx, row in bid_rows.iterrows():
                    job_type = row['JobType'] if pd.notna(row['JobType']) else 'N/A'
                    job_name = str(row['JobName'])[:40] if pd.notna(row['JobName']) else 'N/A'
                    print(f"    - JobType: {job_type:>10}, JobName: {job_name}")

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Distribution of rows per BidId
        rows_per_bidid = self.df.groupby('BidId').size()
        ax1 = axes[0, 0]
        rows_per_bidid.value_counts().sort_index().plot(kind='bar', ax=ax1,
                                                         color='steelblue', edgecolor='black')
        ax1.set_xlabel('Number of Rows per BidId')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Rows per BidId')
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. JobType distribution
        if 'JobType' in self.df.columns:
            ax2 = axes[0, 1]
            job_type_counts.plot(kind='bar', ax=ax2, color='coral', edgecolor='black')
            ax2.set_xlabel('Job Type')
            ax2.set_ylabel('Count')
            ax2.set_title('Job Type Distribution')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3, axis='y')

        # 3. JobCount distribution
        if 'JobCount' in self.df.columns:
            ax3 = axes[1, 0]
            self.df['JobCount'].dropna().hist(bins=30, ax=ax3,
                                              color='lightgreen', edgecolor='black')
            ax3.set_xlabel('Job Count')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of JobCount')
            ax3.grid(True, alpha=0.3)

        # 4. BidFee comparison: single vs multiple rows per BidId
        ax4 = axes[1, 1]
        single_row_bids = self.df[self.df['BidId'].isin(rows_per_bidid[rows_per_bidid == 1].index)]
        multi_row_bids = self.df[self.df['BidId'].isin(rows_per_bidid[rows_per_bidid > 1].index)]

        data_to_plot = [
            single_row_bids['BidFee'].dropna(),
            multi_row_bids['BidFee'].dropna()
        ]
        ax4.boxplot(data_to_plot, labels=['Single Row BidIds', 'Multi Row BidIds'])
        ax4.set_ylabel('Bid Fee ($)')
        ax4.set_title('BidFee: Single vs Multi-Row BidIds')
        ax4.set_ylim(0, self.df['BidFee'].quantile(0.95))
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'bid_hierarchy_analysis.png',
                   dpi=300, bbox_inches='tight')
        print(f"\n✓ Bid hierarchy analysis plot saved")
        plt.close()

        # Save detailed report
        hierarchy_report = pd.DataFrame({
            'Metric': [
                'Total Rows',
                'Unique BidIds',
                'BidIds with Duplicates',
                'Total Duplicate Rows',
                'Avg Rows per Duplicate BidId'
            ],
            'Value': [
                len(self.df),
                self.df['BidId'].nunique(),
                unique_dup_bids,
                len(dup_bid_ids),
                len(dup_bid_ids) / unique_dup_bids if unique_dup_bids > 0 else 0
            ]
        })
        hierarchy_report.to_csv(self.reports_dir / 'bid_hierarchy_report.csv', index=False)
        print(f"✓ Bid hierarchy report saved to CSV")
        print()

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("="*80)
        print("GENERATING SUMMARY REPORT")
        print("="*80)

        summary = {
            'Dataset Overview': {
                'Total Rows': len(self.df),
                'Total Columns': len(self.df.columns),
                'Date Range Start': str(self.df['BidDate'].min()),
                'Date Range End': str(self.df['BidDate'].max()),
                'Years Covered': f"{self.df['Year'].min():.0f} - {self.df['Year'].max():.0f}",
            },
            'Target Variable (BidFee)': {
                'Missing Count': int(self.df['BidFee'].isnull().sum()),
                'Missing Percent': f"{self.df['BidFee'].isnull().sum()/len(self.df)*100:.2f}%",
                'Mean': f"${self.df['BidFee'].mean():,.2f}",
                'Median': f"${self.df['BidFee'].median():,.2f}",
                'Std Dev': f"${self.df['BidFee'].std():,.2f}",
                'Min': f"${self.df['BidFee'].min():,.2f}",
                'Max': f"${self.df['BidFee'].max():,.2f}",
            },
            'Bid Status Distribution': {
                status: int(count)
                for status, count in self.df['BidStatusName'].value_counts().items()
            },
            'Critical Driver (TargetTime)': {
                'Missing Count': int(self.df['TargetTime'].isnull().sum()),
                'Missing Percent': f"{self.df['TargetTime'].isnull().sum()/len(self.df)*100:.2f}%",
                'Mean (days)': f"{self.df['TargetTime'].mean():.2f}",
                'Median (days)': f"{self.df['TargetTime'].median():.2f}",
            },
            'Top Property Types': {
                prop_type: int(count)
                for prop_type, count in self.df['Bid_Property_Type'].value_counts().head(5).items()
            },
            'Top States': {
                state: int(count)
                for state, count in self.df['PropertyState'].value_counts().head(5).items()
            },
            'Data Quality': {
                'Columns with Missing Data': int((self.df.isnull().sum() > 0).sum()),
                'Duplicate BidIds': int(self.df.duplicated(subset=['BidId']).sum()),
            }
        }

        # Print summary
        for section, metrics in summary.items():
            print(f"\n{section}:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")

        # Save to file
        import json
        with open(self.reports_dir / 'eda_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Summary report saved to JSON")
        print()

    def run_full_eda(self):
        """Execute complete comprehensive EDA pipeline"""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Run all analyses
        self.load_data()
        self.analyze_data_quality()
        self.analyze_target_variable()
        self.analyze_time_series_patterns()
        self.analyze_target_time()
        self.analyze_categorical_features()
        self.analyze_geographic_patterns()
        self.analyze_correlations()
        self.analyze_bid_hierarchy()
        self.generate_summary_report()

        print("="*80)
        print("COMPREHENSIVE EDA COMPLETE")
        print("="*80)
        print(f"\nOutputs saved to:")
        print(f"  Figures: {self.figures_dir}")
        print(f"  Reports: {self.reports_dir}")
        print(f"\nGenerated Visualizations:")
        for fig_file in sorted(self.figures_dir.glob('*.png')):
            print(f"  - {fig_file.name}")
        print(f"\nGenerated Reports:")
        for report_file in sorted(self.reports_dir.glob('*')):
            print(f"  - {report_file.name}")
        print()


def main():
    """Main execution function"""
    # Define paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'BidData_cleaned.csv'
    OUTPUT_DIR = PROJECT_ROOT / 'outputs'

    # Initialize and run EDA
    eda = BidEDA(data_path=DATA_PATH, output_dir=OUTPUT_DIR)
    eda.run_full_eda()


if __name__ == "__main__":
    main()
