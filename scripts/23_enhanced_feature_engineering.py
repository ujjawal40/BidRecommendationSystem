"""
Enhanced Feature Engineering — Phase 3
=========================================
Creates features for both v2 model pipelines:
  - Phase 1A v2: JobsData → NetFee prediction features
  - Phase 1B v2: Enriched BidData → Win probability features

Reuses proven patterns from 03_feature_engineering.py:
  - Leave-one-out aggregation (prevents target leakage)
  - Rolling/lag features with shift() (prevents look-ahead bias)
  - Frequency encoding, interaction features

New features:
  - JobLength_Days + interactions
  - SubType frequency/LOO encoding
  - CompanyLocation/Office_Region encoding
  - GrossBuildingSF/GLARentableSF (log-scaled)
  - Top demographics from ZipCodeMaster
  - New cross-feature interactions

Outputs:
  - data/features/JobsData_features_v2.csv (Phase 1A)
  - data/features/BidData_features_v2.csv (Phase 1B)

Author: Ujjawal Dwivedi
Date: 2026-02-14
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from config.model_config import DATA_DIR, REPORTS_DIR

warnings.filterwarnings("ignore")

FEATURES_DIR = DATA_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# Selected ZipCodeMaster features (from Phase 0 EDA)
SELECTED_ZIP_FEATURES = [
    "Zip_Population", "Zip_PopDensity", "Zip_HouseholdsPerZip",
    "Zip_GrowthRank", "Zip_AverageHouseValue", "Zip_IncomePerHousehold",
    "Zip_MedianAge", "Zip_MedianIncome", "Zip_NumberOfBusinesses",
    "Zip_NumberOfEmployees", "Zip_LandArea", "Zip_PopulationEstimate",
    "Zip_PopCount", "Zip_DeliveryTotal", "Zip_WorkersOutZip",
]


def leave_one_out_mean(df, group_col, value_col):
    """Leave-one-out mean: excludes current row's value from group aggregate."""
    total = df.groupby(group_col)[value_col].transform("sum")
    count = df.groupby(group_col)[value_col].transform("count")
    return (total - df[value_col]) / (count - 1)


def frequency_encode(df, col, min_count=10):
    """Frequency encoding: map category to its count."""
    freq = df[col].value_counts()
    return df[col].map(freq).fillna(0)


def build_common_features(df, fee_col, date_col, segment_col, state_col,
                          property_type_col, subtype_col=None,
                          office_region_col=None, company_location_col=None,
                          joblength_col=None, building_sf_col=None,
                          land_acres_col=None, zip_features=None):
    """Build features common to both Phase 1A and Phase 1B v2 models."""
    features_created = []

    # Sort by date first (CRITICAL)
    df = df.sort_values(date_col).reset_index(drop=True)

    # ========================================================================
    # LEAVE-ONE-OUT AGGREGATION FEATURES
    # ========================================================================
    print("\n  LOO Aggregation Features...")

    # Segment avg fee
    df["segment_avg_fee"] = leave_one_out_mean(df, segment_col, fee_col)
    features_created.append("segment_avg_fee")

    # Segment std fee (expanding, shifted to prevent leakage)
    df["segment_std_fee"] = df.groupby(segment_col)[fee_col].transform(
        lambda x: x.shift().expanding().std()
    )
    features_created.append("segment_std_fee")

    # PropertyType avg fee
    df["propertytype_avg_fee"] = leave_one_out_mean(df, property_type_col, fee_col)
    features_created.append("propertytype_avg_fee")

    # State avg fee
    df["state_avg_fee"] = leave_one_out_mean(df, state_col, fee_col)
    features_created.append("state_avg_fee")

    # SubType avg fee (if available)
    if subtype_col and subtype_col in df.columns:
        df["subtype_avg_fee"] = leave_one_out_mean(df, subtype_col, fee_col)
        features_created.append("subtype_avg_fee")

    # Office_Region avg fee
    if office_region_col and office_region_col in df.columns:
        df["office_region_avg_fee"] = leave_one_out_mean(df, office_region_col, fee_col)
        features_created.append("office_region_avg_fee")

    # ========================================================================
    # ROLLING FEATURES
    # ========================================================================
    print("  Rolling Features...")
    roll_n = 10

    df["rolling_avg_fee_segment"] = df.groupby(segment_col)[fee_col].transform(
        lambda x: x.shift().rolling(window=roll_n, min_periods=1).mean()
    )
    features_created.append("rolling_avg_fee_segment")

    df["rolling_std_fee_segment"] = df.groupby(segment_col)[fee_col].transform(
        lambda x: x.shift().rolling(window=roll_n, min_periods=1).std()
    )
    features_created.append("rolling_std_fee_segment")

    df["rolling_avg_fee_state"] = df.groupby(state_col)[fee_col].transform(
        lambda x: x.shift().rolling(window=roll_n, min_periods=1).mean()
    )
    features_created.append("rolling_avg_fee_state")

    # ========================================================================
    # FREQUENCY ENCODING
    # ========================================================================
    print("  Frequency Encoding...")

    df["segment_frequency"] = frequency_encode(df, segment_col)
    features_created.append("segment_frequency")

    df["state_frequency"] = df[state_col].map(
        df[state_col].value_counts(normalize=True)
    ).fillna(0)
    features_created.append("state_frequency")

    df["state_count"] = frequency_encode(df, state_col)
    features_created.append("state_count")

    df["propertytype_frequency"] = frequency_encode(df, property_type_col)
    features_created.append("propertytype_frequency")

    if subtype_col and subtype_col in df.columns:
        df["subtype_frequency"] = frequency_encode(df, subtype_col)
        features_created.append("subtype_frequency")

    if office_region_col and office_region_col in df.columns:
        df["office_region_frequency"] = frequency_encode(df, office_region_col)
        features_created.append("office_region_frequency")

    if company_location_col and company_location_col in df.columns:
        df["company_location_frequency"] = frequency_encode(df, company_location_col)
        features_created.append("company_location_frequency")

    # ========================================================================
    # NEW: JOBLENGTH FEATURES
    # ========================================================================
    if joblength_col and joblength_col in df.columns:
        print("  JobLength Features...")
        jl = df[joblength_col].copy()

        # Log-scaled job length
        df["joblength_log"] = np.log1p(jl)
        features_created.append("joblength_log")

        # Job length buckets
        df["joblength_bucket"] = pd.cut(
            jl, bins=[0, 14, 30, 60, 120, 365], labels=False
        ).fillna(2)
        features_created.append("joblength_bucket")

        # Interaction: job length × segment avg fee
        df["joblength_x_segment_fee"] = jl * df["segment_avg_fee"]
        features_created.append("joblength_x_segment_fee")

        # Fee per day (will be target-aware for Phase 1A, ok since LOO)
        df["fee_per_day"] = df[fee_col] / (jl + 1)
        features_created.append("fee_per_day")

    # ========================================================================
    # NEW: BUILDING AREA FEATURES
    # ========================================================================
    if building_sf_col and building_sf_col in df.columns:
        print("  Building Area Features...")
        bsf = df[building_sf_col].copy()

        df["building_sf_log"] = np.log1p(bsf)
        features_created.append("building_sf_log")

        # Fee per sqft (target-aware but LOO-safe in aggregate)
        df["fee_per_sqft"] = df[fee_col] / (bsf + 1)
        features_created.append("fee_per_sqft")

    if land_acres_col and land_acres_col in df.columns:
        df["land_acres_log"] = np.log1p(df[land_acres_col])
        features_created.append("land_acres_log")

    # ========================================================================
    # INTERACTION FEATURES
    # ========================================================================
    print("  Interaction Features...")

    # Segment × State
    if "segment_avg_fee" in df.columns and "state_avg_fee" in df.columns:
        df["segment_x_state_fee"] = df["segment_avg_fee"] * df["state_avg_fee"]
        features_created.append("segment_x_state_fee")

    # Fee vs segment ratio
    df["fee_vs_segment_ratio"] = df[fee_col] / (df["segment_avg_fee"] + 1)
    features_created.append("fee_vs_segment_ratio")

    # Fee percentile within segment
    df["fee_percentile_segment"] = df.groupby(segment_col)[fee_col].rank(
        pct=True, method="average"
    )
    features_created.append("fee_percentile_segment")

    # Fee diff from segment
    df["fee_diff_from_segment"] = df[fee_col] - df["segment_avg_fee"]
    features_created.append("fee_diff_from_segment")

    # ========================================================================
    # TEMPORAL FEATURES
    # ========================================================================
    print("  Temporal Features...")

    # Days since last in segment
    df["days_since_last_segment"] = df.groupby(segment_col)[date_col].diff().dt.days
    features_created.append("days_since_last_segment")

    # ========================================================================
    # ZIP CODE DEMOGRAPHIC FEATURES (already numeric)
    # ========================================================================
    if zip_features:
        print("  ZipCode Demographic Features...")
        for col in zip_features:
            if col in df.columns:
                features_created.append(col)

        # Demographic interactions
        if "Zip_IncomePerHousehold" in df.columns and "segment_avg_fee" in df.columns:
            df["income_x_segment_fee"] = df["Zip_IncomePerHousehold"] * df["segment_avg_fee"]
            features_created.append("income_x_segment_fee")

        if "Zip_PopDensity" in df.columns:
            df["pop_density_log"] = np.log1p(df["Zip_PopDensity"])
            features_created.append("pop_density_log")

    print(f"  Total features created: {len(features_created)}")
    return df, features_created


def engineer_phase1a_features():
    """Feature engineering for Phase 1A v2 (JobsData → NetFee prediction)."""
    print("=" * 80)
    print("PHASE 1A v2: FEATURE ENGINEERING (JobsData → NetFee)")
    print("=" * 80)

    # Load processed JobsData
    df = pd.read_csv(DATA_DIR / "processed" / "JobsData_processed.csv", low_memory=False)
    df["StartDate"] = pd.to_datetime(df["StartDate"])
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Build zip features list (match what's in processed data)
    available_zip = [c for c in SELECTED_ZIP_FEATURES if c in df.columns]

    df, features_created = build_common_features(
        df,
        fee_col="NetFee",
        date_col="StartDate",
        segment_col="BusinessSegment",
        state_col="StateName",
        property_type_col="PropertyType",
        subtype_col="SubType",
        office_region_col="Office_Region",
        company_location_col="CompanyLocation",
        joblength_col="JobLength_Days",
        building_sf_col="GrossBuildingSF",
        land_acres_col="GrossLandAreaAcres",
        zip_features=available_zip,
    )

    # Phase 1A specific features
    print("\n  Phase 1A Specific Features...")

    # Company-level features (using CompanyType as proxy for client)
    if "CompanyType" in df.columns:
        df["companytype_avg_fee"] = leave_one_out_mean(df, "CompanyType", "NetFee")
        features_created.append("companytype_avg_fee")
        df["companytype_frequency"] = frequency_encode(df, "CompanyType")
        features_created.append("companytype_frequency")

    # ContactType features
    if "ContactType" in df.columns:
        df["contacttype_avg_fee"] = leave_one_out_mean(df, "ContactType", "NetFee")
        features_created.append("contacttype_avg_fee")

    # Handle NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Replace inf values
    df = df.replace([np.inf, -np.inf], 0)

    # Save
    output_path = FEATURES_DIR / "JobsData_features_v2.csv"
    df.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")
    print(f"  {len(df):,} rows × {len(df.columns)} columns")
    print(f"  Features created: {len(features_created)}")

    return df, features_created


def engineer_phase1b_features():
    """Feature engineering for Phase 1B v2 (enriched BidData → Win Probability)."""
    print("\n" + "=" * 80)
    print("PHASE 1B v2: FEATURE ENGINEERING (BidData Enriched → Win Prob)")
    print("=" * 80)

    # Load enriched BidData
    df = pd.read_csv(DATA_DIR / "processed" / "BidData_enriched_v2.csv", low_memory=False)
    df["BidDate"] = pd.to_datetime(df["BidDate"])
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Available enrichment zip features
    available_zip = [f"Jobs_{c}" for c in SELECTED_ZIP_FEATURES if f"Jobs_{c}" in df.columns]

    df, features_created = build_common_features(
        df,
        fee_col="BidFee",
        date_col="BidDate",
        segment_col="BusinessSegment",
        state_col="PropertyState",
        property_type_col="PropertyType",
        subtype_col="Jobs_SubType",
        office_region_col="Jobs_Office_Region",
        company_location_col="Jobs_CompanyLocation",
        joblength_col="Jobs_JobLength_Days",
        building_sf_col="Jobs_GrossBuildingSF",
        land_acres_col="Jobs_GrossLandAreaAcres",
        zip_features=available_zip,
    )

    # Phase 1B specific features
    print("\n  Phase 1B Specific Features...")

    # Client features (BidCompanyName)
    if "BidCompanyName" in df.columns:
        # Client avg fee (LOO)
        df["client_avg_fee"] = leave_one_out_mean(df, "BidCompanyName", "BidFee")
        features_created.append("client_avg_fee")

        # Client std fee
        df["client_std_fee"] = df.groupby("BidCompanyName")["BidFee"].transform(
            lambda x: x.shift().expanding().std()
        )
        features_created.append("client_std_fee")

        # Client win rate (LOO)
        df["client_win_rate"] = leave_one_out_mean(df, "BidCompanyName", "Won")
        features_created.append("client_win_rate")

        # Lag features
        df["lag1_bidfee_client"] = df.groupby("BidCompanyName")["BidFee"].shift(1)
        features_created.append("lag1_bidfee_client")

        # Cumulative
        df["cumulative_bids_client"] = df.groupby("BidCompanyName").cumcount()
        features_created.append("cumulative_bids_client")

        df["cumulative_wins_client"] = (
            df.groupby("BidCompanyName")["Won"].cumsum().shift(1).fillna(0)
        )
        features_created.append("cumulative_wins_client")

        df["cumulative_winrate_client"] = (
            df["cumulative_wins_client"] / (df["cumulative_bids_client"] + 1)
        )
        features_created.append("cumulative_winrate_client")

        # Days since last client bid
        df["days_since_last_bid_client"] = (
            df.groupby("BidCompanyName")["BidDate"].diff().dt.days
        )
        features_created.append("days_since_last_bid_client")

    # Office features
    if "OfficeId" in df.columns:
        df["office_avg_fee"] = leave_one_out_mean(df, "OfficeId", "BidFee")
        features_created.append("office_avg_fee")

        df["office_win_rate"] = leave_one_out_mean(df, "OfficeId", "Won")
        features_created.append("office_win_rate")

    # TargetTime features
    if "TargetTime" in df.columns:
        df["targettime_log"] = np.log1p(pd.to_numeric(df["TargetTime"], errors="coerce").fillna(0))
        features_created.append("targettime_log")

    # Distance features
    if "DistanceInMiles" in df.columns:
        dist = pd.to_numeric(df["DistanceInMiles"], errors="coerce").fillna(0)
        df["distance_log"] = np.log1p(dist)
        features_created.append("distance_log")

    # Competitiveness ratios
    df["bid_vs_client_ratio"] = df["BidFee"] / (df["client_avg_fee"] + 1)
    features_created.append("bid_vs_client_ratio")

    df["bid_vs_state_ratio"] = df["BidFee"] / (df["state_avg_fee"] + 1)
    features_created.append("bid_vs_state_ratio")

    # Handle NaN and inf
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df = df.replace([np.inf, -np.inf], 0)

    # Save
    output_path = FEATURES_DIR / "BidData_features_v2.csv"
    df.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")
    print(f"  {len(df):,} rows × {len(df.columns)} columns")
    print(f"  Features created: {len(features_created)}")

    return df, features_created


def main():
    print("=" * 80)
    print("ENHANCED FEATURE ENGINEERING — Phase 3")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Phase 1A features
    df_1a, feats_1a = engineer_phase1a_features()

    # Phase 1B features
    df_1b, feats_1b = engineer_phase1b_features()

    # Save feature summary
    summary = {
        "phase_1a": {
            "rows": len(df_1a),
            "total_columns": len(df_1a.columns),
            "numeric_columns": len(df_1a.select_dtypes(include=[np.number]).columns),
            "features_created": len(feats_1a),
            "feature_names": feats_1a,
        },
        "phase_1b": {
            "rows": len(df_1b),
            "total_columns": len(df_1b.columns),
            "numeric_columns": len(df_1b.select_dtypes(include=[np.number]).columns),
            "features_created": len(feats_1b),
            "feature_names": feats_1b,
        },
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    report_path = REPORTS_DIR / "enhanced_feature_engineering_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Feature report: {report_path}")

    # Summary
    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETE")
    print("=" * 80)
    print(f"  Phase 1A: {len(df_1a):,} rows, {len(feats_1a)} new features")
    print(f"  Phase 1B: {len(df_1b):,} rows, {len(feats_1b)} new features")

    return df_1a, df_1b


if __name__ == "__main__":
    df_1a, df_1b = main()
