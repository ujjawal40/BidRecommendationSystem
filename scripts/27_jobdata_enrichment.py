"""
JobData Enrichment Script (ALL DATA VERSION)
=============================================
Enrich BidData with insights from JobData (571K completed job records).

Enrichment Strategy:
- Link via OfficeCode (100% overlap)
- Create office-level aggregate features from ALL historical data
- Add Region and ClientType dimensions
- Generate market intelligence features
- NO date filtering - use all 532K valid jobs (2001-2026)

Data Sources:
- BidData: 114K bid records (2018-2025)
- JobData: 571K job records → 532K after cleaning (2001-2026)

Cleaning: Only remove NULL, ≤$0, or >$1M JobFees (6.8%)
No date filter applied for maximum statistical power

Expected Impact: 4-8% RMSE improvement

Author: Bid Recommendation System
Date: 2026-01-15 (Updated)
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')

print("=" * 80)
print("JOBDATA ENRICHMENT - BID FEE PREDICTION")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)

# Load BidData
print("\n📊 Loading BidData...")
df_bid = pd.read_csv('data/features/BidData_features.csv')
print(f"✓ BidData loaded: {len(df_bid):,} records")
print(f"  Columns: {len(df_bid.columns)}")
print(f"  Date range: {df_bid['BidDate'].min()} to {df_bid['BidDate'].max()}")

# Load JobData
print("\n📊 Loading JobData...")
df_job = pd.read_csv('data/raw/JobData.csv', encoding='latin-1', header=None, low_memory=False)

# Clean BOM character
if isinstance(df_job.iloc[0, 0], str) and 'ÿ' in str(df_job.iloc[0, 0]):
    df_job.iloc[0, 0] = df_job.iloc[0, 0].replace('ÿ', '')

# Assign column names
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

# ============================================================================
# CLEAN JOBDATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: CLEANING JOBDATA")
print("=" * 80)

# Convert dates
df_job['JobCreateDate'] = pd.to_datetime(df_job['JobCreateDate'], errors='coerce')

# Clean JobFee (remove outliers)
df_job['JobFee'] = pd.to_numeric(df_job['JobFee'], errors='coerce')
initial_count = len(df_job)

# Remove extreme outliers (>$1M likely data errors)
df_job_clean = df_job[
    (df_job['JobFee'] > 0) &
    (df_job['JobFee'] < 1_000_000)
].copy()

print(f"\n✓ JobFee cleaned (removed NULL, ≤$0, >$1M):")
print(f"  Original records: {initial_count:,}")
print(f"  After cleaning: {len(df_job_clean):,}")
print(f"  Removed: {initial_count - len(df_job_clean):,} ({(initial_count - len(df_job_clean))/initial_count*100:.1f}%)")
print(f"  Mean JobFee: ${df_job_clean['JobFee'].mean():,.2f}")
print(f"  Median JobFee: ${df_job_clean['JobFee'].median():,.2f}")
print(f"  Date range: {df_job_clean['JobCreateDate'].min()} to {df_job_clean['JobCreateDate'].max()}")

# NO DATE FILTER - Using all historical data for maximum statistical power
print(f"\n✓ Using ALL historical data (no date filter)")
print(f"  Timespan: {(df_job_clean['JobCreateDate'].max() - df_job_clean['JobCreateDate'].min()).days / 365.25:.1f} years")
print(f"  Total jobs for aggregates: {len(df_job_clean):,}")

# ============================================================================
# CREATE OFFICE-LEVEL AGGREGATES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: CREATING OFFICE-LEVEL AGGREGATES")
print("=" * 80)

print("\n📊 Computing office statistics from JobData...")

# Group by office
office_agg = df_job_clean.groupby('OfficeCode').agg({
    # Volume metrics
    'JobId': 'count',

    # Fee metrics
    'JobFee': ['mean', 'median', 'std', 'min', 'max'],

    # Appraisal fee metrics
    'AppraisalFee': ['mean', 'median'],

    # Complexity metrics
    'JobType': lambda x: (x == 'Master').sum() / len(x),  # % Master jobs

    # Property type focus
    'PropertyType': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',

    # Client type
    'ClientType': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',

    # Region
    'Region': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
}).reset_index()

# Flatten column names
office_agg.columns = [
    'OfficeCode',
    'office_job_volume',
    'office_avg_job_fee',
    'office_median_job_fee',
    'office_job_fee_std',
    'office_min_job_fee',
    'office_max_job_fee',
    'office_avg_appraisal_fee',
    'office_median_appraisal_fee',
    'office_master_job_pct',
    'office_primary_property_type',
    'office_primary_client_type',
    'office_region'
]

print(f"✓ Created aggregates for {len(office_agg)} offices")
print(f"\nSample office statistics:")
print(office_agg.head(10).to_string())

# ============================================================================
# CREATE DERIVED FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: CREATING DERIVED FEATURES")
print("=" * 80)

# Fee range
office_agg['office_fee_range'] = office_agg['office_max_job_fee'] - office_agg['office_min_job_fee']

# Coefficient of variation (normalized volatility)
office_agg['office_fee_cv'] = (
    office_agg['office_job_fee_std'] / office_agg['office_avg_job_fee']
).fillna(0)

# Profit margin proxy
office_agg['office_avg_profit_margin'] = (
    office_agg['office_avg_job_fee'] - office_agg['office_avg_appraisal_fee']
).fillna(0)

# Market tier (based on avg fee)
office_agg['office_market_tier'] = pd.qcut(
    office_agg['office_avg_job_fee'],
    q=3,
    labels=['Budget', 'Mid', 'Premium'],
    duplicates='drop'
)

print(f"✓ Created derived features:")
print(f"  • office_fee_range")
print(f"  • office_fee_cv (coefficient of variation)")
print(f"  • office_avg_profit_margin")
print(f"  • office_market_tier (Budget/Mid/Premium)")

# ============================================================================
# CREATE REGIONAL AGGREGATES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: CREATING REGIONAL AGGREGATES")
print("=" * 80)

# Group by region
region_agg = df_job_clean.groupby('Region').agg({
    'JobFee': ['mean', 'median', 'std'],
    'JobId': 'count'
}).reset_index()

region_agg.columns = [
    'Region',
    'region_avg_job_fee',
    'region_median_job_fee',
    'region_job_fee_std',
    'region_job_volume'
]

print(f"✓ Created aggregates for {len(region_agg)} regions")
print(f"\nRegional statistics:")
print(region_agg.to_string())

# Merge region stats into office data
office_agg = office_agg.merge(
    region_agg[['Region', 'region_avg_job_fee', 'region_median_job_fee', 'region_job_volume']],
    left_on='office_region',
    right_on='Region',
    how='left'
).drop('Region', axis=1)

# Office vs region premium
office_agg['office_vs_region_premium'] = (
    office_agg['office_avg_job_fee'] - office_agg['region_avg_job_fee']
)

office_agg['office_vs_region_ratio'] = (
    office_agg['office_avg_job_fee'] / office_agg['region_avg_job_fee']
).fillna(1.0)

print(f"✓ Added regional comparison features:")
print(f"  • office_vs_region_premium")
print(f"  • office_vs_region_ratio")

# ============================================================================
# CREATE PROPERTY TYPE × REGION FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: CREATING PROPERTY TYPE × REGION FEATURES")
print("=" * 80)

# Property type by region benchmarks
prop_region_agg = df_job_clean.groupby(['PropertyType', 'Region']).agg({
    'JobFee': ['mean', 'median', 'count']
}).reset_index()

prop_region_agg.columns = [
    'PropertyType',
    'Region',
    'property_region_avg_fee',
    'property_region_median_fee',
    'property_region_volume'
]

# Filter to combinations with sufficient data (at least 10 jobs)
prop_region_agg = prop_region_agg[prop_region_agg['property_region_volume'] >= 10].copy()

print(f"✓ Created {len(prop_region_agg)} PropertyType × Region combinations")
print(f"\nSample combinations:")
print(prop_region_agg.head(15).to_string())

# ============================================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: ENCODING CATEGORICAL FEATURES")
print("=" * 80)

from sklearn.preprocessing import LabelEncoder

# Encode region
le_region = LabelEncoder()
office_agg['office_region_encoded'] = le_region.fit_transform(office_agg['office_region'].fillna('Unknown'))

# Encode primary client type
le_client = LabelEncoder()
office_agg['office_primary_client_type_encoded'] = le_client.fit_transform(
    office_agg['office_primary_client_type'].fillna('Unknown')
)

# Encode market tier
le_tier = LabelEncoder()
office_agg['office_market_tier_encoded'] = le_tier.fit_transform(office_agg['office_market_tier'].astype(str))

print(f"✓ Encoded categorical features:")
print(f"  • office_region: {len(le_region.classes_)} categories")
print(f"  • office_primary_client_type: {len(le_client.classes_)} categories")
print(f"  • office_market_tier: {len(le_tier.classes_)} categories")

# ============================================================================
# MERGE INTO BIDDATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: MERGING ENRICHED FEATURES INTO BIDDATA")
print("=" * 80)

# Count features before merge
features_before = len(df_bid.columns)

print(f"\n📊 BidData before enrichment:")
print(f"  Records: {len(df_bid):,}")
print(f"  Features: {features_before}")

# Merge office aggregates
df_bid_enriched = df_bid.merge(
    office_agg,
    on='OfficeCode',
    how='left'
)

# Fill missing values for offices not in JobData (if any)
numeric_cols = df_bid_enriched.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col.startswith('office_') or col.startswith('region_'):
        df_bid_enriched[col] = df_bid_enriched[col].fillna(df_bid_enriched[col].median())

features_after = len(df_bid_enriched.columns)
new_features = features_after - features_before

print(f"\n✓ Merge complete!")
print(f"\n📊 BidData after enrichment:")
print(f"  Records: {len(df_bid_enriched):,}")
print(f"  Features: {features_after}")
print(f"  New features added: {new_features}")

# ============================================================================
# ADD PROPERTY TYPE × REGION FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: ADDING PROPERTY TYPE × REGION BENCHMARKS")
print("=" * 80)

# Add property type column if not exists (use from JobData primary type)
if 'PropertyType' not in df_bid_enriched.columns:
    df_bid_enriched['PropertyType_enriched'] = df_bid_enriched['office_primary_property_type']
else:
    df_bid_enriched['PropertyType_enriched'] = df_bid_enriched['PropertyType']

# Merge property × region benchmarks
df_bid_enriched = df_bid_enriched.merge(
    prop_region_agg[['PropertyType', 'Region', 'property_region_avg_fee', 'property_region_median_fee']],
    left_on=['PropertyType_enriched', 'office_region'],
    right_on=['PropertyType', 'Region'],
    how='left'
).drop(['PropertyType', 'Region'], axis=1, errors='ignore')

# Fill missing with office averages
df_bid_enriched['property_region_avg_fee'] = df_bid_enriched['property_region_avg_fee'].fillna(
    df_bid_enriched['office_avg_job_fee']
)
df_bid_enriched['property_region_median_fee'] = df_bid_enriched['property_region_median_fee'].fillna(
    df_bid_enriched['office_median_job_fee']
)

print(f"✓ Added property type × region benchmarks")
print(f"  Total features now: {len(df_bid_enriched.columns)}")

# ============================================================================
# FEATURE SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: NEW FEATURES SUMMARY")
print("=" * 80)

new_feature_list = [col for col in df_bid_enriched.columns if col not in df_bid.columns]

print(f"\n✨ {len(new_feature_list)} new features created:\n")

# Group by category
volume_features = [f for f in new_feature_list if 'volume' in f.lower()]
fee_features = [f for f in new_feature_list if 'fee' in f.lower()]
region_features = [f for f in new_feature_list if 'region' in f.lower()]
market_features = [f for f in new_feature_list if 'market' in f.lower() or 'tier' in f.lower()]
other_features = [f for f in new_feature_list if f not in volume_features + fee_features + region_features + market_features]

print(f"📊 Volume Features ({len(volume_features)}):")
for f in volume_features:
    print(f"  • {f}")

print(f"\n💰 Fee Features ({len(fee_features)}):")
for f in fee_features:
    print(f"  • {f}")

print(f"\n🌎 Regional Features ({len(region_features)}):")
for f in region_features:
    print(f"  • {f}")

print(f"\n📈 Market Features ({len(market_features)}):")
for f in market_features:
    print(f"  • {f}")

if other_features:
    print(f"\n🔧 Other Features ({len(other_features)}):")
    for f in other_features:
        print(f"  • {f}")

# ============================================================================
# DATA QUALITY CHECKS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 11: DATA QUALITY CHECKS")
print("=" * 80)

print(f"\n🔍 Checking data quality...")

# Check missing values in new features
missing_summary = df_bid_enriched[new_feature_list].isnull().sum()
missing_features = missing_summary[missing_summary > 0]

if len(missing_features) > 0:
    print(f"\n⚠️ Features with missing values:")
    for feat, count in missing_features.items():
        pct = count / len(df_bid_enriched) * 100
        print(f"  • {feat}: {count:,} ({pct:.1f}%)")
else:
    print(f"\n✓ No missing values in new features!")

# Check feature statistics
print(f"\n📊 Feature Statistics (sample):")
sample_features = [
    'office_avg_job_fee',
    'office_job_volume',
    'office_vs_region_premium',
    'office_fee_cv'
]

for feat in sample_features:
    if feat in df_bid_enriched.columns:
        print(f"\n  {feat}:")
        print(f"    Mean: {df_bid_enriched[feat].mean():.2f}")
        print(f"    Median: {df_bid_enriched[feat].median():.2f}")
        print(f"    Min: {df_bid_enriched[feat].min():.2f}")
        print(f"    Max: {df_bid_enriched[feat].max():.2f}")

# ============================================================================
# SAVE ENRICHED DATASET
# ============================================================================
print("\n" + "=" * 80)
print("STEP 12: SAVING ENRICHED DATASET")
print("=" * 80)

# Create output directory
output_dir = Path('data/features')
output_dir.mkdir(parents=True, exist_ok=True)

# Save enriched data
output_path = output_dir / 'BidData_enriched_with_jobs.csv'
df_bid_enriched.to_csv(output_path, index=False)

print(f"\n✓ Enriched dataset saved:")
print(f"  Path: {output_path}")
print(f"  Records: {len(df_bid_enriched):,}")
print(f"  Features: {len(df_bid_enriched.columns)}")
print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

# Save office aggregates separately
office_agg_path = output_dir / 'office_aggregates_from_jobdata.csv'
office_agg.to_csv(office_agg_path, index=False)
print(f"\n✓ Office aggregates saved: {office_agg_path}")

# Save region aggregates
region_agg_path = output_dir / 'region_aggregates_from_jobdata.csv'
region_agg.to_csv(region_agg_path, index=False)
print(f"✓ Region aggregates saved: {region_agg_path}")

# ============================================================================
# GENERATE ENRICHMENT REPORT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 13: GENERATING ENRICHMENT REPORT")
print("=" * 80)

report = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'input_data': {
        'biddata_records': len(df_bid),
        'biddata_features': features_before,
        'jobdata_records_raw': len(df_job),
        'jobdata_records_cleaned': len(df_job_clean),
        'offices_matched': len(office_agg)
    },
    'output_data': {
        'enriched_records': len(df_bid_enriched),
        'enriched_features': features_after,
        'new_features_count': new_features
    },
    'new_features': {
        'volume_features': volume_features,
        'fee_features': fee_features,
        'region_features': region_features,
        'market_features': market_features,
        'other_features': other_features
    },
    'data_quality': {
        'missing_values': {feat: int(count) for feat, count in missing_features.items()}
    },
    'regional_breakdown': {
        row['office_region']: int(row['office_job_volume'])
        for _, row in office_agg.groupby('office_region')['office_job_volume'].sum().reset_index().iterrows()
    }
}

# Save report
reports_dir = Path('outputs/reports')
reports_dir.mkdir(parents=True, exist_ok=True)
report_path = reports_dir / 'jobdata_enrichment_report.json'

with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✓ Enrichment report saved: {report_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ENRICHMENT COMPLETE")
print("=" * 80)

print(f"\n✨ Summary:")
print(f"  • Input: {features_before} features")
print(f"  • Output: {features_after} features")
print(f"  • Added: {new_features} new features")
print(f"  • Source: {len(df_job_clean):,} job records")
print(f"  • Offices enriched: {len(office_agg)}")
print(f"  • Regions added: {df_bid_enriched['office_region'].nunique()}")

print(f"\n📁 Output files:")
print(f"  1. {output_path}")
print(f"  2. {office_agg_path}")
print(f"  3. {region_agg_path}")
print(f"  4. {report_path}")

print(f"\n🚀 Next Step:")
print(f"  Run regression model with enriched data:")
print(f"  → Use: data/features/BidData_enriched_with_jobs.csv")
print(f"  → Expected: 3-7% RMSE improvement")

print("\n" + "=" * 80)
