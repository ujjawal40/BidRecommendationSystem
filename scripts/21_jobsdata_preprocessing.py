"""
JobsData Preprocessing — Phase 1
===================================
Cleans and prepares JobsData for Phase 1A v2 (NetFee prediction).

Steps:
  1. Filter to Closed status + 2018+
  2. Remove NetFee nulls/zeros
  3. Cap NetFee at 99th percentile
  4. Clean JobType (aggregate Master/SubJob → keep Master fee)
  5. Clean SubType, SpecificUse, CompanyLocation, Office_Region
  6. Parse JobLength_Days (cap 1-365)
  7. Select top ZipCodeMaster features
  8. Create temporal features
  9. Save to data/processed/JobsData_processed.csv

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

from config.model_config import DATA_DIR, REPORTS_DIR, ROOT_DIR

warnings.filterwarnings("ignore")

# Import column names
from scripts._jobsdata_columns import JOBSDATA_COLUMNS

# Output paths
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Data filtering
DATA_START_YEAR = 2018  # 7-year window for v2 (broader than v1's 2023+)

# ZipCodeMaster features selected by EDA (correlation + coverage)
SELECTED_ZIP_FEATURES = [
    "Zip_Population",
    "Zip_PopDensity",
    "Zip_HouseholdsPerZip",
    "Zip_GrowthRank",
    "Zip_AverageHouseValue",
    "Zip_IncomePerHousehold",
    "Zip_MedianAge",
    "Zip_MedianIncome",
    "Zip_NumberOfBusinesses",
    "Zip_NumberOfEmployees",
    "Zip_LandArea",
    "Zip_PopulationEstimate",
    "Zip_PopCount",
    "Zip_DeliveryTotal",
    "Zip_WorkersOutZip",
]

# SubType minimum count threshold (rare subtypes → "Other")
SUBTYPE_MIN_COUNT = 50


class JobsDataPreprocessor:
    """Preprocessing pipeline for JobsData.csv (Phase 1A v2 training data)."""

    def __init__(self):
        self.df = None
        self.stats = {}

    def load_data(self):
        """Load raw JobsData with assigned headers."""
        print("=" * 80)
        print("LOADING JOBSDATA")
        print("=" * 80)

        path = DATA_DIR / "raw" / "JobsData.csv"
        self.df = pd.read_csv(
            path,
            header=None,
            names=JOBSDATA_COLUMNS,
            encoding="utf-8-sig",
            low_memory=False,
        )
        self.stats["initial_rows"] = len(self.df)
        print(f"  Loaded {len(self.df):,} rows × {len(self.df.columns)} columns")

    def filter_closed_recent(self):
        """Filter to Closed jobs from 2018+."""
        print("\n" + "=" * 80)
        print("FILTERING: Closed + 2018+")
        print("=" * 80)

        before = len(self.df)

        # Filter status
        self.df = self.df[self.df["JobStatus"] == "Closed"].copy()
        after_status = len(self.df)
        print(f"  After Closed filter: {after_status:,} (removed {before - after_status:,})")

        # Filter year
        self.df["Year"] = pd.to_numeric(self.df["Year"], errors="coerce")
        self.df = self.df[self.df["Year"] >= DATA_START_YEAR].copy()
        after_year = len(self.df)
        print(f"  After {DATA_START_YEAR}+ filter: {after_year:,} (removed {after_status - after_year:,})")

        self.stats["after_closed_filter"] = after_status
        self.stats["after_year_filter"] = after_year

    def clean_netfee(self):
        """Clean and cap NetFee target variable."""
        print("\n" + "=" * 80)
        print("CLEANING NETFEE (Target)")
        print("=" * 80)

        # Convert to numeric
        self.df["NetFee"] = pd.to_numeric(
            self.df["NetFee"].replace("NULL", np.nan), errors="coerce"
        )

        before = len(self.df)
        # Remove nulls and zeros
        self.df = self.df[self.df["NetFee"].notna() & (self.df["NetFee"] > 0)].copy()
        after_null = len(self.df)
        print(f"  Removed null/zero NetFee: {before - after_null:,}")

        # Save original
        self.df["NetFee_Original"] = self.df["NetFee"].copy()

        # Cap at 99th percentile
        p99 = self.df["NetFee"].quantile(0.99)
        capped = (self.df["NetFee"] > p99).sum()
        self.df["NetFee"] = self.df["NetFee"].clip(upper=p99)
        print(f"  Capped at P99 (${p99:,.0f}): {capped:,} rows")

        print(f"  Final NetFee: mean=${self.df['NetFee'].mean():,.0f}, "
              f"median=${self.df['NetFee'].median():,.0f}")

        self.stats["netfee_p99_cap"] = float(p99)
        self.stats["netfee_capped_count"] = int(capped)
        self.stats["after_netfee_clean"] = len(self.df)

    def handle_job_types(self):
        """Handle Master/SubJob structure — keep max NetFee per JobId group."""
        print("\n" + "=" * 80)
        print("HANDLING JOB TYPES (Master/SubJob)")
        print("=" * 80)

        before = len(self.df)
        job_type_dist = self.df["JobType"].value_counts()
        print(f"  JobType distribution:")
        for jt, count in job_type_dist.items():
            print(f"    {jt}: {count:,}")

        # For duplicate JobIds, keep the row with max NetFee (typically Master)
        dup_count = self.df["JobId"].duplicated().sum()
        if dup_count > 0:
            self.df = self.df.sort_values("NetFee", ascending=False).drop_duplicates(
                subset=["JobId"], keep="first"
            )
            print(f"  Deduplicated JobIds: removed {before - len(self.df):,} duplicate rows")
        else:
            print(f"  No duplicate JobIds found")

        self.stats["after_dedup"] = len(self.df)

    def clean_categorical_features(self):
        """Clean categorical features: SubType, SpecificUse, CompanyLocation, Office_Region."""
        print("\n" + "=" * 80)
        print("CLEANING CATEGORICAL FEATURES")
        print("=" * 80)

        # Replace "NULL" strings with NaN
        null_cols = [
            "SubType", "SpecificUse", "CompanyLocation", "Office_Region",
            "BusinessSegment", "BusinessSegmentDetail", "PropertyType",
            "MarketOrientation", "Market", "Submarket", "Shape", "Topography",
            "BuildingClass", "PropertyCondition", "JobDistanceMiles",
            "OfficeJobTerritory", "ContactType", "CompanyType", "JobPurpose",
            "Deliverable", "AppraisalFileType", "PortfolioMultiProperty",
            "PotentialLitigation", "IRRRegion",
        ]
        for col in null_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].replace("NULL", np.nan)

        # --- SubType: collapse rare categories ---
        subtype_counts = self.df["SubType"].value_counts()
        rare_subtypes = subtype_counts[subtype_counts < SUBTYPE_MIN_COUNT].index.tolist()
        if rare_subtypes:
            self.df.loc[self.df["SubType"].isin(rare_subtypes), "SubType"] = "Other"
            print(f"  SubType: collapsed {len(rare_subtypes)} rare categories to 'Other'")
        self.df["SubType"] = self.df["SubType"].fillna("Unknown")
        print(f"  SubType: {self.df['SubType'].nunique()} unique values")

        # --- SpecificUse: collapse rare ---
        specuse_counts = self.df["SpecificUse"].value_counts()
        rare_specuse = specuse_counts[specuse_counts < SUBTYPE_MIN_COUNT].index.tolist()
        if rare_specuse:
            self.df.loc[self.df["SpecificUse"].isin(rare_specuse), "SpecificUse"] = "Other"
        self.df["SpecificUse"] = self.df["SpecificUse"].fillna("Unknown")
        print(f"  SpecificUse: {self.df['SpecificUse'].nunique()} unique values")

        # --- CompanyLocation: standardize case ---
        self.df["CompanyLocation"] = self.df["CompanyLocation"].str.strip().str.title()
        self.df["CompanyLocation"] = self.df["CompanyLocation"].fillna("Unknown")
        print(f"  CompanyLocation: {self.df['CompanyLocation'].nunique()} unique values")

        # --- Office_Region: fill missing ---
        self.df["Office_Region"] = self.df["Office_Region"].fillna("Unknown")
        print(f"  Office_Region: {self.df['Office_Region'].nunique()} unique values")

        # --- PropertyType: fill missing ---
        self.df["PropertyType"] = self.df["PropertyType"].fillna("Unknown")
        print(f"  PropertyType: {self.df['PropertyType'].nunique()} unique values")

        # --- BusinessSegment ---
        self.df["BusinessSegment"] = self.df["BusinessSegment"].fillna("Unknown")
        print(f"  BusinessSegment: {self.df['BusinessSegment'].nunique()} unique values")

        # --- MarketOrientation ---
        self.df["MarketOrientation"] = self.df["MarketOrientation"].fillna("Unknown")

    def clean_joblength(self):
        """Parse and cap JobLength_Days to reasonable range."""
        print("\n" + "=" * 80)
        print("CLEANING JOBLENGTH_DAYS")
        print("=" * 80)

        self.df["JobLength_Days"] = pd.to_numeric(
            self.df["JobLength_Days"].replace("NULL", np.nan), errors="coerce"
        )

        valid_before = self.df["JobLength_Days"].notna().sum()
        null_count = self.df["JobLength_Days"].isna().sum()
        print(f"  Valid: {valid_before:,}, Null: {null_count:,}")

        # Cap to [1, 365]
        valid = self.df["JobLength_Days"].notna()
        below_1 = (self.df.loc[valid, "JobLength_Days"] < 1).sum()
        above_365 = (self.df.loc[valid, "JobLength_Days"] > 365).sum()
        self.df["JobLength_Days"] = self.df["JobLength_Days"].clip(lower=1, upper=365)
        print(f"  Capped below 1: {below_1:,}, above 365: {above_365:,}")

        # Impute nulls with median by PropertyType
        group_median = self.df.groupby("PropertyType")["JobLength_Days"].transform("median")
        global_median = self.df["JobLength_Days"].median()
        self.df["JobLength_Days"] = self.df["JobLength_Days"].fillna(group_median).fillna(global_median)

        print(f"  After imputation: mean={self.df['JobLength_Days'].mean():.1f}, "
              f"median={self.df['JobLength_Days'].median():.0f}")

        self.stats["joblength_null_count"] = int(null_count)
        self.stats["joblength_capped_below"] = int(below_1)
        self.stats["joblength_capped_above"] = int(above_365)

    def clean_numeric_features(self):
        """Clean numeric features: building area, lat/lon, year built."""
        print("\n" + "=" * 80)
        print("CLEANING NUMERIC FEATURES")
        print("=" * 80)

        # Convert NULL strings to NaN for numeric columns
        numeric_cols = [
            "GrossBuildingSF", "GLARentableSF", "GrossLandAreaAcres",
            "GrossLandAreaSF", "RooftopLatitude", "RooftopLongitude",
            "YearBuilt", "PhotosCount",
        ]

        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(
                    self.df[col].replace("NULL", np.nan), errors="coerce"
                )

        # Building area: cap at 99th percentile, fill with 0
        area_cols = ["GrossBuildingSF", "GLARentableSF", "GrossLandAreaAcres", "GrossLandAreaSF"]
        for col in area_cols:
            valid = self.df[col].dropna()
            if len(valid) > 0:
                p99 = valid.quantile(0.99)
                self.df[col] = self.df[col].clip(upper=p99)
            self.df[col] = self.df[col].fillna(0)
            print(f"  {col}: filled {(self.df[col] == 0).sum():,} zeros")

        # YearBuilt: set invalid values to NaN, impute with median
        valid_yb = self.df["YearBuilt"].between(1800, 2026)
        self.df.loc[~valid_yb, "YearBuilt"] = np.nan
        self.df["YearBuilt"] = self.df["YearBuilt"].fillna(
            self.df.groupby("PropertyType")["YearBuilt"].transform("median")
        ).fillna(self.df["YearBuilt"].median()).fillna(2000)
        print(f"  YearBuilt: median={self.df['YearBuilt'].median():.0f}")

    def clean_zipcode_features(self):
        """Select and clean top ZipCodeMaster features."""
        print("\n" + "=" * 80)
        print("CLEANING ZIPCODE FEATURES")
        print("=" * 80)

        for col in SELECTED_ZIP_FEATURES:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(
                    self.df[col].replace("NULL", np.nan), errors="coerce"
                )
                null_pct = self.df[col].isna().mean() * 100
                # Fill with median
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
                print(f"  {col}: filled {null_pct:.1f}% nulls with median={median_val:.1f}")

    def create_temporal_features(self):
        """Create temporal features from StartDate."""
        print("\n" + "=" * 80)
        print("CREATING TEMPORAL FEATURES")
        print("=" * 80)

        # Parse StartDate
        self.df["StartDate"] = pd.to_datetime(self.df["StartDate"], errors="coerce")

        # Year already exists, create others
        self.df["Month"] = self.df["StartDate"].dt.month
        self.df["Quarter"] = self.df["StartDate"].dt.quarter
        self.df["DayOfWeek"] = self.df["StartDate"].dt.dayofweek
        self.df["WeekOfYear"] = self.df["StartDate"].dt.isocalendar().week.astype(int)

        # Cyclical encoding
        self.df["Month_sin"] = np.sin(2 * np.pi * self.df["Month"] / 12)
        self.df["Month_cos"] = np.cos(2 * np.pi * self.df["Month"] / 12)
        self.df["DayOfWeek_sin"] = np.sin(2 * np.pi * self.df["DayOfWeek"] / 7)
        self.df["DayOfWeek_cos"] = np.cos(2 * np.pi * self.df["DayOfWeek"] / 7)

        print(f"  Created: Month, Quarter, DayOfWeek, WeekOfYear + cyclical encodings")
        print(f"  Date range: {self.df['StartDate'].min()} to {self.df['StartDate'].max()}")

    def select_columns(self):
        """Select final columns for output."""
        print("\n" + "=" * 80)
        print("SELECTING FINAL COLUMNS")
        print("=" * 80)

        # Columns to keep
        keep_cols = [
            # IDs (needed for joins, excluded from features later)
            "JobId", "PropertyID", "OfficeID",
            # Target
            "NetFee", "NetFee_Original",
            # Date (for time-based split)
            "StartDate",
            # Job characteristics
            "JobType", "JobPurpose", "Deliverable", "AppraisalFileType",
            "BusinessSegment", "BusinessSegmentDetail",
            "PortfolioMultiProperty", "PotentialLitigation",
            "JobLength_Days",
            "JobDistanceMiles", "OfficeJobTerritory",
            # Property
            "PropertyType", "SubType", "SpecificUse",
            "GrossBuildingSF", "GLARentableSF",
            "GrossLandAreaAcres", "GrossLandAreaSF",
            "YearBuilt", "MarketOrientation",
            # Location
            "StateName", "Market", "Submarket",
            "RooftopLatitude", "RooftopLongitude",
            # Office
            "CompanyLocation", "Office_Region",
            # Client
            "ContactType", "CompanyType",
            # Temporal
            "Year", "Month", "Quarter", "DayOfWeek", "WeekOfYear",
            "Month_sin", "Month_cos", "DayOfWeek_sin", "DayOfWeek_cos",
        ] + SELECTED_ZIP_FEATURES

        # Only keep columns that exist
        available = [c for c in keep_cols if c in self.df.columns]
        missing = [c for c in keep_cols if c not in self.df.columns]
        if missing:
            print(f"  ⚠ Missing columns (skipped): {missing}")

        self.df = self.df[available].copy()
        print(f"  Selected {len(available)} columns")
        print(f"  Categorical: {list(self.df.select_dtypes(include=['object']).columns)}")

    def save(self):
        """Save processed data."""
        print("\n" + "=" * 80)
        print("SAVING PROCESSED DATA")
        print("=" * 80)

        # Sort by date for time-based splitting
        self.df = self.df.sort_values("StartDate").reset_index(drop=True)

        output_path = PROCESSED_DIR / "JobsData_processed.csv"
        self.df.to_csv(output_path, index=False)
        print(f"  Saved {len(self.df):,} rows × {len(self.df.columns)} columns")
        print(f"  Path: {output_path}")

        self.stats["final_rows"] = len(self.df)
        self.stats["final_columns"] = len(self.df.columns)

        # Save preprocessing stats
        self.stats["processing_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stats_path = REPORTS_DIR / "jobsdata_preprocessing_stats.json"
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2, default=str)
        print(f"  Stats: {stats_path}")

    def run(self):
        """Run full preprocessing pipeline."""
        self.load_data()
        self.filter_closed_recent()
        self.clean_netfee()
        self.handle_job_types()
        self.clean_categorical_features()
        self.clean_joblength()
        self.clean_numeric_features()
        self.clean_zipcode_features()
        self.create_temporal_features()
        self.select_columns()
        self.save()

        # Summary
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE")
        print("=" * 80)
        print(f"  {self.stats['initial_rows']:,} → {self.stats['final_rows']:,} rows")
        print(f"  {self.stats['final_columns']} columns selected")
        print(f"  NetFee: mean=${self.df['NetFee'].mean():,.0f}, "
              f"median=${self.df['NetFee'].median():,.0f}")

        return self.df


def main():
    preprocessor = JobsDataPreprocessor()
    df = preprocessor.run()
    return df


if __name__ == "__main__":
    df = main()
