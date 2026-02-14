"""
Combined Dataset for Win Probability — Phase 2
=================================================
Enriches BidData with JobsData features for Phase 1B v2 (Win Probability).

Strategy:
  - Won bids: LEFT JOIN BidData → JobsData on JobId
    → Pull SubType, GrossBuildingSF, Office_Region, JobLength_Days, demographics
  - Lost bids: Enrich via OfficeId for office features,
    PropertyId for property features, mode imputation for SubType
  - Output: data/processed/BidData_enriched_v2.csv

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

# Import column definitions
from scripts._jobsdata_columns import JOBSDATA_COLUMNS

# Paths
PROCESSED_DIR = DATA_DIR / "processed"
BIDDATA_CLEANED = PROCESSED_DIR / "BidData_cleaned.csv"
JOBSDATA_RAW = DATA_DIR / "raw" / "JobsData.csv"
OUTPUT_PATH = PROCESSED_DIR / "BidData_enriched_v2.csv"

# Features to pull from JobsData
JOBS_ENRICH_COLS = [
    "JobId",
    "SubType",          # Property sub-classification
    "SpecificUse",      # Specific property use
    "GrossBuildingSF",  # Building area
    "GLARentableSF",    # Rentable area
    "GrossLandAreaAcres",
    "JobLength_Days",   # Delivery time
    "CompanyLocation",  # Office location
    "Office_Region",    # Office region
    "OfficeID",         # For lost bid enrichment
    "PropertyID",       # For lost bid enrichment
    "PropertyType",     # For validation
    "YearBuilt",
    "MarketOrientation",
]

# ZipCodeMaster features (selected in Phase 0)
SELECTED_ZIP_FEATURES = [
    "Zip_Population", "Zip_PopDensity", "Zip_HouseholdsPerZip",
    "Zip_GrowthRank", "Zip_AverageHouseValue", "Zip_IncomePerHousehold",
    "Zip_MedianAge", "Zip_MedianIncome", "Zip_NumberOfBusinesses",
    "Zip_NumberOfEmployees", "Zip_LandArea", "Zip_PopulationEstimate",
    "Zip_PopCount", "Zip_DeliveryTotal", "Zip_WorkersOutZip",
]


def load_biddata():
    """Load cleaned BidData."""
    print("=" * 80)
    print("LOADING BIDDATA")
    print("=" * 80)

    df = pd.read_csv(BIDDATA_CLEANED, low_memory=False)
    df["BidDate"] = pd.to_datetime(df["BidDate"], errors="coerce")

    # Create Won flag
    df["Won"] = (df["BidStatusName"] == "Won").astype(int)

    print(f"  Loaded {len(df):,} bids")
    print(f"  Won: {df['Won'].sum():,}, Lost: {(df['Won'] == 0).sum():,}")
    print(f"  Date range: {df['BidDate'].min()} to {df['BidDate'].max()}")

    return df


def load_jobsdata():
    """Load raw JobsData with headers."""
    print("\n" + "=" * 80)
    print("LOADING JOBSDATA")
    print("=" * 80)

    df = pd.read_csv(
        JOBSDATA_RAW,
        header=None,
        names=JOBSDATA_COLUMNS,
        encoding="utf-8-sig",
        low_memory=False,
    )

    # Clean numeric columns
    for col in ["GrossBuildingSF", "GLARentableSF", "GrossLandAreaAcres",
                "JobLength_Days", "YearBuilt", "OfficeID", "PropertyID"]:
        df[col] = pd.to_numeric(df[col].replace("NULL", np.nan), errors="coerce")

    # Clean categoricals
    for col in ["SubType", "SpecificUse", "CompanyLocation", "Office_Region",
                "PropertyType", "MarketOrientation"]:
        df[col] = df[col].replace("NULL", np.nan)

    # Clean zip features
    for col in SELECTED_ZIP_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace("NULL", np.nan), errors="coerce")

    # Standardize CompanyLocation
    df["CompanyLocation"] = df["CompanyLocation"].str.strip().str.title()

    print(f"  Loaded {len(df):,} jobs × {len(df.columns)} columns")

    return df


def enrich_won_bids(bid_df, jobs_df):
    """Enrich Won bids by joining on JobId."""
    print("\n" + "=" * 80)
    print("ENRICHING WON BIDS (JOIN on JobId)")
    print("=" * 80)

    won_mask = bid_df["Won"] == 1
    won_bids = bid_df[won_mask].copy()
    print(f"  Won bids: {len(won_bids):,}")

    # Check JobId availability
    has_jobid = won_bids["JobId"].notna() & (won_bids["JobId"] != "NULL")
    print(f"  With JobId: {has_jobid.sum():,} ({has_jobid.mean() * 100:.1f}%)")

    # Convert JobId to numeric for join
    won_bids["JobId_num"] = pd.to_numeric(won_bids["JobId"], errors="coerce")

    # Prepare jobs lookup — deduplicate on JobId (keep first/highest fee)
    jobs_lookup = jobs_df[JOBS_ENRICH_COLS + SELECTED_ZIP_FEATURES].copy()
    jobs_lookup = jobs_lookup.drop_duplicates(subset=["JobId"], keep="first")
    jobs_lookup = jobs_lookup.rename(columns={
        "SubType": "Jobs_SubType",
        "SpecificUse": "Jobs_SpecificUse",
        "GrossBuildingSF": "Jobs_GrossBuildingSF",
        "GLARentableSF": "Jobs_GLARentableSF",
        "GrossLandAreaAcres": "Jobs_GrossLandAreaAcres",
        "JobLength_Days": "Jobs_JobLength_Days",
        "CompanyLocation": "Jobs_CompanyLocation",
        "Office_Region": "Jobs_Office_Region",
        "OfficeID": "Jobs_OfficeID",
        "PropertyID": "Jobs_PropertyID",
        "PropertyType": "Jobs_PropertyType",
        "YearBuilt": "Jobs_YearBuilt",
        "MarketOrientation": "Jobs_MarketOrientation",
    })
    # Also rename zip features
    zip_rename = {col: f"Jobs_{col}" for col in SELECTED_ZIP_FEATURES}
    jobs_lookup = jobs_lookup.rename(columns=zip_rename)

    # Join
    won_enriched = won_bids.merge(
        jobs_lookup,
        left_on="JobId_num",
        right_on="JobId",
        how="left",
        suffixes=("", "_jobs"),
    )

    # Check match rate
    matched = won_enriched["Jobs_SubType"].notna().sum()
    print(f"  Matched: {matched:,} ({matched / len(won_enriched) * 100:.1f}%)")

    # Drop helper columns
    won_enriched = won_enriched.drop(columns=["JobId_num", "JobId_jobs"], errors="ignore")

    return won_enriched


def enrich_lost_bids(bid_df, jobs_df):
    """Enrich Lost bids via OfficeId and PropertyId lookups."""
    print("\n" + "=" * 80)
    print("ENRICHING LOST BIDS (OfficeId + PropertyId lookup)")
    print("=" * 80)

    lost_mask = bid_df["Won"] == 0
    lost_bids = bid_df[lost_mask].copy()
    print(f"  Lost bids: {len(lost_bids):,}")

    # Initialize enrichment columns that won't be filled by merge
    for col in ["SubType", "SpecificUse", "GrossBuildingSF", "GLARentableSF",
                 "GrossLandAreaAcres", "OfficeID", "PropertyID", "PropertyType",
                 "YearBuilt", "MarketOrientation"]:
        lost_bids[f"Jobs_{col}"] = np.nan

    # --- Office-level enrichment ---
    # Build office lookup: mode/median per OfficeID
    print("  Building office-level lookup...")
    office_groups = jobs_df.groupby("OfficeID")

    office_lookup = pd.DataFrame({
        "Office_Region": office_groups["Office_Region"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan),
        "CompanyLocation": office_groups["CompanyLocation"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan),
        "JobLength_Days_median": office_groups["JobLength_Days"].median(),
    })

    # Zip features median by office
    for col in SELECTED_ZIP_FEATURES:
        if col in jobs_df.columns:
            office_lookup[f"{col}_median"] = office_groups[col].median()

    # Match lost bids to office lookup via vectorized merge
    lost_bids["OfficeId_num"] = pd.to_numeric(lost_bids["OfficeId"], errors="coerce")

    # Rename office_lookup columns for merge
    office_merge = office_lookup.copy()
    office_merge = office_merge.rename(columns={
        "Office_Region": "Jobs_Office_Region",
        "CompanyLocation": "Jobs_CompanyLocation",
        "JobLength_Days_median": "Jobs_JobLength_Days",
    })
    for col in SELECTED_ZIP_FEATURES:
        if f"{col}_median" in office_merge.columns:
            office_merge = office_merge.rename(columns={f"{col}_median": f"Jobs_{col}"})

    # Drop the pre-initialized NaN columns before merge
    drop_cols = [c for c in office_merge.columns if c in lost_bids.columns]
    lost_bids = lost_bids.drop(columns=drop_cols, errors="ignore")

    # Merge
    lost_bids = lost_bids.merge(
        office_merge,
        left_on="OfficeId_num",
        right_index=True,
        how="left",
    )

    matched_office = lost_bids["Jobs_Office_Region"].notna().sum()
    print(f"  Office match: {matched_office:,} ({matched_office / len(lost_bids) * 100:.1f}%)")

    # --- Property-level enrichment ---
    # SubType mode by PropertyType (from BidData's own PropertyType)
    print("  Building SubType mode lookup by PropertyType...")
    subtype_mode = jobs_df.groupby("PropertyType")["SubType"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
    )

    # Map SubType using BidData's PropertyType (vectorized)
    lost_bids["Jobs_SubType"] = lost_bids["PropertyType"].map(subtype_mode)

    subtype_filled = lost_bids["Jobs_SubType"].notna().sum()
    print(f"  SubType filled via PropertyType mode: {subtype_filled:,} ({subtype_filled / len(lost_bids) * 100:.1f}%)")

    # Drop helper columns
    lost_bids = lost_bids.drop(columns=["OfficeId_num"], errors="ignore")

    return lost_bids


def combine_and_finalize(won_enriched, lost_enriched):
    """Combine Won and Lost bids, finalize enrichment columns."""
    print("\n" + "=" * 80)
    print("COMBINING WON + LOST BIDS")
    print("=" * 80)

    # Ensure same columns
    all_cols = sorted(set(won_enriched.columns) | set(lost_enriched.columns))
    for col in all_cols:
        if col not in won_enriched.columns:
            won_enriched[col] = np.nan
        if col not in lost_enriched.columns:
            lost_enriched[col] = np.nan

    combined = pd.concat([won_enriched, lost_enriched], ignore_index=True)
    print(f"  Combined: {len(combined):,} bids")

    # Sort by date
    combined = combined.sort_values("BidDate").reset_index(drop=True)

    # Fill remaining NaN in enrichment columns
    for col in combined.columns:
        if col.startswith("Jobs_"):
            null_count = combined[col].isna().sum()
            if null_count > 0:
                if combined[col].dtype in ["float64", "int64"]:
                    median_val = combined[col].median()
                    combined[col] = combined[col].fillna(median_val)
                else:
                    combined[col] = combined[col].fillna("Unknown")

    # Coverage report
    print(f"\n  Enrichment coverage:")
    for col in ["Jobs_SubType", "Jobs_Office_Region", "Jobs_CompanyLocation",
                "Jobs_JobLength_Days", "Jobs_GrossBuildingSF"]:
        if col in combined.columns:
            non_null = combined[col].notna().sum()
            non_unknown = (combined[col] != "Unknown").sum() if combined[col].dtype == "object" else non_null
            print(f"    {col}: {non_unknown:,} ({non_unknown / len(combined) * 100:.1f}%)")

    return combined


def main():
    print("=" * 80)
    print("COMBINED DATASET FOR WIN PROBABILITY — Phase 2")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    bid_df = load_biddata()
    jobs_df = load_jobsdata()

    # Enrich separately
    won_enriched = enrich_won_bids(bid_df, jobs_df)
    lost_enriched = enrich_lost_bids(bid_df, jobs_df)

    # Combine
    combined = combine_and_finalize(won_enriched, lost_enriched)

    # Save
    print("\n" + "=" * 80)
    print("SAVING ENRICHED DATASET")
    print("=" * 80)

    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved {len(combined):,} rows × {len(combined.columns)} columns")
    print(f"  Path: {OUTPUT_PATH}")

    # Stats
    stats = {
        "total_bids": len(combined),
        "won_bids": int(combined["Won"].sum()),
        "lost_bids": int((combined["Won"] == 0).sum()),
        "columns": len(combined.columns),
        "enrichment_columns": [c for c in combined.columns if c.startswith("Jobs_")],
        "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    stats_path = REPORTS_DIR / "combined_data_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats: {stats_path}")

    # Summary
    print("\n" + "=" * 80)
    print("PHASE 2 COMPLETE")
    print("=" * 80)
    print(f"  Total: {len(combined):,} bids ({combined['Won'].sum():,} won, "
          f"{(combined['Won'] == 0).sum():,} lost)")
    print(f"  Enrichment columns: {len(stats['enrichment_columns'])}")

    return combined


if __name__ == "__main__":
    combined = main()
