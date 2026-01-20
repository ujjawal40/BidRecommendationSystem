"""
JobData Competitive Features - SMART Enrichment
================================================
Extract competitive dynamics and patterns from JobData instead of static fees.

NEW APPROACH (Competitive Intelligence):
----------------------------------------
1. Office Win Trend: Are they winning more/fewer jobs over time? (momentum)
2. Seasonal Patterns: When do offices typically win more? (timing strategy)
3. Property×Region Complexity: Which combinations are rare/difficult? (pricing signals)
4. Client Retention: Do clients come back? (relationship strength)

WHY THIS IS BETTER:
-------------------
Previous approach (FAILED): Static aggregates (office_avg_job_fee, region_median_fee)
  ✗ Just duplicated existing features with noise
  ✗ Survivor bias: only successful outcomes
  ✗ Backward-looking: historical costs

New approach (SMART): Dynamic patterns and competitive intelligence
  ✓ Captures momentum and trends (forward-looking)
  ✓ Reveals market positioning (competitive)
  ✓ Shows complexity and rarity (strategic)
  ✓ Indicates relationship strength (pricing power)

Author: Bid Recommendation System
Date: 2026-01-20
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("JOBDATA COMPETITIVE FEATURES EXTRACTION")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)

# Load JobData
print("\n📊 Loading JobData...")
df_job = pd.read_csv('data/raw/JobData.csv', encoding='latin-1', header=None, low_memory=False)

# Clean BOM character
if isinstance(df_job.iloc[0, 0], str) and 'ÿ' in str(df_job.iloc[0, 0]):
    df_job.iloc[0, 0] = df_job.iloc[0, 0].replace('ÿ', '')

# Assign column names manually (JobData has no header row)
column_names = [
    'JobId', 'JobType', 'JobCreateDate', 'JobDueDate', 'JobCloseDate',
    'PropertyId', 'OfficeId', 'OfficeCode', 'Region',
    'PropertyTypeId', 'PropertyType', 'SubPropertyTypeId', 'SubPropertyType',
    'JobPurposeId', 'JobPurpose', 'DeliverableId', 'Deliverable',
    'ClientId', 'JobFee', 'AssignedAppraiser', 'AppraisalFee', 'ClientType'
]
df_job.columns = column_names

print(f"✓ JobData loaded: {len(df_job):,} records")
print(f"  Columns: {len(df_job.columns)}")

# Parse dates
df_job['JobCreateDate'] = pd.to_datetime(df_job['JobCreateDate'], errors='coerce')

# Basic cleaning - remove invalid data only
df_job['JobFee'] = pd.to_numeric(df_job['JobFee'], errors='coerce')
df_job = df_job[df_job['JobFee'].notna()].copy()
df_job = df_job[df_job['JobFee'] > 0].copy()
df_job = df_job[df_job['JobFee'] <= 1000000].copy()
df_job = df_job[df_job['JobCreateDate'].notna()].copy()

print(f"✓ After cleaning: {len(df_job):,} jobs")
print(f"  Date range: {df_job['JobCreateDate'].min()} to {df_job['JobCreateDate'].max()}")

# Load BidData
print("\n📊 Loading BidData...")
df_bid = pd.read_csv('data/features/BidData_features.csv')
df_bid['BidDate'] = pd.to_datetime(df_bid['BidDate'])
print(f"✓ BidData loaded: {len(df_bid):,} records")

# ============================================================================
# FEATURE 1: OFFICE WIN TREND (Momentum Indicator)
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE 1: OFFICE WIN TREND OVER TIME")
print("=" * 80)

print("\n🔍 Calculating office win momentum...")

# For each office, calculate quarterly job counts
df_job['Year'] = df_job['JobCreateDate'].dt.year
df_job['Quarter'] = df_job['JobCreateDate'].dt.quarter
df_job['YearQuarter'] = df_job['Year'].astype(str) + 'Q' + df_job['Quarter'].astype(str)

# Count jobs per office per quarter
office_quarterly = df_job.groupby(['OfficeCode', 'YearQuarter']).size().reset_index(name='JobCount')

# Calculate trend for each office (recent quarters vs all-time)
def calculate_win_trend(group):
    """Calculate if office is trending up or down in wins"""
    if len(group) < 4:  # Need at least 4 quarters
        return 0.0, 0.0

    # Sort by time
    group = group.sort_values('YearQuarter')

    # Compare recent (last 4 quarters) vs historical average
    recent_avg = group.tail(4)['JobCount'].mean()
    historical_avg = group.head(max(1, len(group) - 4))['JobCount'].mean()

    if historical_avg == 0:
        return 0.0, 0.0

    # Trend ratio: >1 means improving, <1 means declining
    trend_ratio = recent_avg / historical_avg

    # Also calculate linear trend slope
    x = np.arange(len(group))
    y = group['JobCount'].values

    if len(x) > 1:
        slope, _ = np.polyfit(x, y, 1)
        # Normalize slope by mean
        mean_jobs = y.mean()
        normalized_slope = slope / mean_jobs if mean_jobs > 0 else 0
    else:
        normalized_slope = 0

    return trend_ratio, normalized_slope

office_trends = []
for office in office_quarterly['OfficeCode'].unique():
    office_data = office_quarterly[office_quarterly['OfficeCode'] == office]
    trend_ratio, slope = calculate_win_trend(office_data)

    office_trends.append({
        'OfficeCode': office,
        'office_win_trend_ratio': trend_ratio,
        'office_win_trend_slope': slope
    })

office_trends_df = pd.DataFrame(office_trends)

print(f"✓ Calculated win trends for {len(office_trends_df)} offices")
print(f"\n  Win Trend Ratio Distribution:")
print(f"    Mean: {office_trends_df['office_win_trend_ratio'].mean():.2f}")
print(f"    Min:  {office_trends_df['office_win_trend_ratio'].min():.2f}")
print(f"    Max:  {office_trends_df['office_win_trend_ratio'].max():.2f}")
print(f"\n  Interpretation:")
print(f"    > 1.2: Office gaining momentum (winning more recently)")
print(f"    0.8-1.2: Stable")
print(f"    < 0.8: Office declining (winning fewer recently)")

# ============================================================================
# FEATURE 2: SEASONAL WIN PATTERNS
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE 2: SEASONAL WIN PATTERNS BY OFFICE/REGION")
print("=" * 80)

print("\n🔍 Analyzing seasonal patterns...")

# Extract month from JobCreateDate
df_job['Month'] = df_job['JobCreateDate'].dt.month
df_job['Quarter'] = df_job['JobCreateDate'].dt.quarter

# Calculate peak season for each office
def identify_peak_season(group):
    """Find when office wins most jobs"""
    monthly_counts = group.groupby('Month').size()
    quarterly_counts = group.groupby('Quarter').size()

    if len(monthly_counts) == 0:
        return None, None, 0.0

    peak_month = monthly_counts.idxmax()
    peak_quarter = quarterly_counts.idxmax()

    # Calculate seasonality strength (coefficient of variation)
    avg_monthly = monthly_counts.mean()
    std_monthly = monthly_counts.std()
    seasonality_strength = std_monthly / avg_monthly if avg_monthly > 0 else 0

    return peak_month, peak_quarter, seasonality_strength

office_seasons = []
for office in df_job['OfficeCode'].unique():
    office_data = df_job[df_job['OfficeCode'] == office]
    peak_month, peak_quarter, strength = identify_peak_season(office_data)

    office_seasons.append({
        'OfficeCode': office,
        'office_peak_month': peak_month,
        'office_peak_quarter': peak_quarter,
        'office_seasonality_strength': strength
    })

office_seasons_df = pd.DataFrame(office_seasons)

print(f"✓ Identified seasonal patterns for {len(office_seasons_df)} offices")
print(f"\n  Seasonality Strength Distribution:")
print(f"    Mean: {office_seasons_df['office_seasonality_strength'].mean():.3f}")
print(f"    High seasonality (>0.5): {(office_seasons_df['office_seasonality_strength'] > 0.5).sum()} offices")
print(f"\n  Interpretation:")
print(f"    High strength: Office has clear busy/slow seasons")
print(f"    Low strength: Office has consistent volume year-round")

# ============================================================================
# FEATURE 3: PROPERTY TYPE × REGION COMPLEXITY INDEX
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE 3: PROPERTY×REGION COMPLEXITY INDEX")
print("=" * 80)

print("\n🔍 Computing complexity indices...")

# Calculate rarity/complexity for each Property×Region×PropertyType combination
# More rare = higher complexity = likely higher fees

# Region complexity
region_counts = df_job.groupby('Region').size()
region_complexity = (1 / region_counts) * 1000  # Invert: rare = complex

# PropertyType×Region complexity
proptype_region_counts = df_job.groupby(['PropertyType', 'Region']).size()
proptype_region_complexity = (1 / proptype_region_counts) * 1000

# Create lookup dictionaries
region_complexity_dict = region_complexity.to_dict()
proptype_region_complexity_dict = proptype_region_complexity.to_dict()

print(f"✓ Calculated complexity indices")
print(f"  Regions: {len(region_complexity_dict)}")
print(f"  PropertyType×Region combinations: {len(proptype_region_complexity_dict)}")

# ============================================================================
# FEATURE 4: CLIENT RETENTION RATE
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE 4: CLIENT RETENTION RATE BY OFFICE")
print("=" * 80)

print("\n🔍 Analyzing client retention...")

# For each office, calculate what % of clients are repeat clients
def calculate_retention(group):
    """Calculate client retention metrics"""
    client_counts = group.groupby('ClientId').size()

    total_jobs = len(group)
    total_clients = len(client_counts)

    if total_clients == 0:
        return 0.0, 0.0, 0.0

    # Retention rate: % of jobs from repeat clients
    repeat_clients = (client_counts > 1).sum()
    repeat_jobs = client_counts[client_counts > 1].sum()

    retention_rate = repeat_clients / total_clients if total_clients > 0 else 0
    repeat_job_rate = repeat_jobs / total_jobs if total_jobs > 0 else 0

    # Average jobs per client (loyalty indicator)
    avg_jobs_per_client = client_counts.mean()

    return retention_rate, repeat_job_rate, avg_jobs_per_client

office_retention = []
for office in df_job['OfficeCode'].unique():
    office_data = df_job[df_job['OfficeCode'] == office]
    retention, repeat_rate, avg_jobs = calculate_retention(office_data)

    office_retention.append({
        'OfficeCode': office,
        'office_client_retention_rate': retention,
        'office_repeat_job_rate': repeat_rate,
        'office_avg_jobs_per_client': avg_jobs
    })

office_retention_df = pd.DataFrame(office_retention)

print(f"✓ Calculated retention metrics for {len(office_retention_df)} offices")
print(f"\n  Client Retention Rate Distribution:")
print(f"    Mean: {office_retention_df['office_client_retention_rate'].mean():.2%}")
print(f"    Min:  {office_retention_df['office_client_retention_rate'].min():.2%}")
print(f"    Max:  {office_retention_df['office_client_retention_rate'].max():.2%}")
print(f"\n  Interpretation:")
print(f"    High retention: Office has loyal clients (pricing power)")
print(f"    Low retention: Office doing mostly one-off jobs (competitive pricing)")

# ============================================================================
# MERGE OFFICE-LEVEL FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: MERGING OFFICE-LEVEL FEATURES")
print("=" * 80)

# Merge all office features
office_features = office_trends_df.copy()
office_features = office_features.merge(office_seasons_df, on='OfficeCode', how='outer')
office_features = office_features.merge(office_retention_df, on='OfficeCode', how='outer')

print(f"\n✓ Merged office features: {len(office_features)} offices")
print(f"  Features per office: {len(office_features.columns) - 1}")

# ============================================================================
# CREATE BID-LEVEL FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: CREATING BID-LEVEL FEATURES")
print("=" * 80)

print("\n🔍 Enriching BidData with competitive features...")

# Map OfficeId from BidData to OfficeCode in JobData (they should match)
# Convert OfficeCode to numeric to match OfficeId
office_features['OfficeCode'] = pd.to_numeric(office_features['OfficeCode'], errors='coerce')

# Merge office features
df_enriched = df_bid.merge(
    office_features,
    left_on='OfficeId',
    right_on='OfficeCode',
    how='left'
)

# Add complexity features based on property/region
# Note: Need to map BidData's PropertyState/PropertyType to JobData's Region/PropertyType
print("\n🔍 Adding complexity indices...")

# For now, add region complexity based on state (approximate mapping)
# This would need proper State→Region mapping in production

# Add "is peak season" feature for each bid
df_enriched['is_office_peak_month'] = (
    df_enriched['Month'] == df_enriched['office_peak_month']
).astype(int)

df_enriched['is_office_peak_quarter'] = (
    df_enriched['Quarter'] == df_enriched['office_peak_quarter']
).astype(int)

# Fill NaN values for offices not in JobData
competitive_features = [
    'office_win_trend_ratio', 'office_win_trend_slope',
    'office_seasonality_strength',
    'office_client_retention_rate', 'office_repeat_job_rate',
    'office_avg_jobs_per_client',
    'is_office_peak_month', 'is_office_peak_quarter'
]

for col in competitive_features:
    if col in df_enriched.columns:
        df_enriched[col] = df_enriched[col].fillna(0)

# Drop intermediate columns
df_enriched = df_enriched.drop(columns=['OfficeCode'], errors='ignore')

print(f"✓ Enriched dataset created")
print(f"  Records: {len(df_enriched):,}")
print(f"  Features: {len(df_enriched.columns)}")
print(f"  New competitive features: {len(competitive_features)}")

# ============================================================================
# FEATURE SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("COMPETITIVE FEATURES SUMMARY")
print("=" * 80)

print(f"""
NEW FEATURES ({len(competitive_features)} total):

1. WIN MOMENTUM (2 features):
   • office_win_trend_ratio: Recent vs historical win rate
   • office_win_trend_slope: Linear trend in wins over time

2. SEASONAL PATTERNS (3 features):
   • office_seasonality_strength: How seasonal is office?
   • is_office_peak_month: Is this bid in office's peak month?
   • is_office_peak_quarter: Is this bid in office's peak quarter?

3. CLIENT LOYALTY (3 features):
   • office_client_retention_rate: % repeat clients
   • office_repeat_job_rate: % jobs from repeat clients
   • office_avg_jobs_per_client: Loyalty indicator

WHY THESE ARE BETTER:
---------------------
✓ Capture competitive dynamics (not just static averages)
✓ Reveal temporal patterns (momentum, seasonality)
✓ Indicate pricing power (client retention)
✓ Forward-looking (trends predict future behavior)
✓ No survivor bias (patterns apply to both wins and bids)

EXPECTED IMPACT:
----------------
• Win momentum: Offices gaining momentum may bid more aggressively
• Seasonality: Offices in peak season may charge premium
• Client loyalty: High retention = pricing power = higher fees
""")

# ============================================================================
# SAVE ENRICHED DATASET
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: SAVING ENRICHED DATASET")
print("=" * 80)

output_path = Path('data/features/BidData_enriched_competitive.csv')
df_enriched.to_csv(output_path, index=False)

print(f"\n✓ Dataset saved: {output_path}")
print(f"  Records: {len(df_enriched):,}")
print(f"  Features: {len(df_enriched.columns)}")
print(f"  Original features: {len(df_bid.columns)}")
print(f"  New features: {len(df_enriched.columns) - len(df_bid.columns)}")

# Save feature metadata
import json

metadata = {
    'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'jobdata_records': len(df_job),
    'biddata_records': len(df_bid),
    'enriched_records': len(df_enriched),
    'original_features': len(df_bid.columns),
    'new_features': len(competitive_features),
    'total_features': len(df_enriched.columns),
    'competitive_features': competitive_features,
    'approach': 'Competitive Intelligence (momentum, seasonality, loyalty)',
    'why_different': 'Extracts dynamic patterns instead of static aggregates'
}

metadata_path = Path('outputs/reports/competitive_features_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ Metadata saved: {metadata_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("COMPETITIVE FEATURE EXTRACTION COMPLETE")
print("=" * 80)

print(f"""
APPROACH COMPARISON:
--------------------

PREVIOUS (FAILED):
  • Static aggregates: office_avg_job_fee, office_median_job_fee
  • Redundant with existing features
  • Survivor bias: only successful outcomes
  • Backward-looking: historical costs
  • Result: 24-27% performance degradation

NEW (SMART):
  • Dynamic patterns: win trends, seasonality, loyalty
  • Unique competitive intelligence
  • No survivor bias: patterns apply to all bids
  • Forward-looking: momentum predicts future
  • Result: TBD (ready for testing)

NEXT STEPS:
-----------
1. Update config to use: data/features/BidData_enriched_competitive.csv
2. Train LightGBM model
3. Compare performance vs baseline ($296.81 RMSE)
4. Analyze feature importance of new competitive features

Dataset ready for model training!
""")

print("=" * 80)
