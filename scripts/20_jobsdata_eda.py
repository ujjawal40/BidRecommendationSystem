"""
JobsData EDA & Column Validation — Phase 0
=============================================
Assigns headers to the headerless JobsData.csv (143 columns),
validates column positions against known value patterns,
profiles all columns, identifies top ZipCodeMaster features,
and analyzes SubType distribution.

Output: outputs/reports/jobsdata_eda_report.json

Author: Ujjawal Dwivedi
Date: 2026-02-14
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import csv
import json
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

from config.model_config import (
    DATA_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    ROOT_DIR,
)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

# ============================================================================
# COLUMN DEFINITIONS — 143 columns, validated against row samples
# ============================================================================

JOBSDATA_COLUMNS = [
    # --- Core Job Info (0-7) ---
    "JobId",                    # 0  - Numeric ID (e.g., 405033)
    "JobName",                  # 1  - Free text
    "JobType",                  # 2  - Standalone/SubJob/Master
    "JobStatus",                # 3  - Closed/Open/etc.
    "StartDate",                # 4  - YYYY-MM-DD
    "EndDate",                  # 5  - YYYY-MM-DD or NULL
    "DueDate",                  # 6  - YYYY-MM-DD
    "NetFee",                   # 7  - Dollar amount (target for Phase 1A v2)

    # --- Job Classification (8-14) ---
    "JobPurpose",               # 8  - Valuation, Consulting, etc.
    "Deliverable",              # 9  - Market Value, etc.
    "AppraisalFileType",        # 10 - Standard, Comprehensive, etc.
    "BusinessSegment",          # 11 - Financing, Litigation, etc.
    "BusinessSegmentDetail",    # 12 - Bank, REIT, etc.
    "PortfolioMultiProperty",   # 13 - Yes/No/Yes-Sub
    "PotentialLitigation",      # 14 - No Dispute Likely, etc.

    # --- Temporal Fields (15-27) ---
    "Year",                     # 15 - 4-digit year
    "WeekNumber",               # 16 - ISO week number
    "DayOfWeek",                # 17 - 0-6
    "YearWeek",                 # 18 - YYYY-Wnn
    "YearMonth",                # 19 - YYYY-Mn (with trailing space)
    "YearQuarter",              # 20 - YYYY-Qn
    "WeekOfYear",               # 21 - Week number
    "MonthNumber",              # 22 - 1-12 (with trailing space)
    "YearMonthNum",             # 23 - YYYY-n
    "YearWeekNum",              # 24 - YYYY-nn
    "YearWeekConcat",           # 25 - YYYYnn (concatenated)
    "WeekNum2",                 # 26 - Week number (duplicate)
    "MonthNum2",                # 27 - Month number (duplicate)

    # --- Job Geography & Duration (28-31) ---
    "JobDistanceMiles",         # 28 - Distance bucket (e.g., "10-29 Miles")
    "OfficeJobTerritory",       # 29 - Primary/Secondary
    "JobLength_Days",           # 30 - Integer days (can be NULL)
    "PropertyID",               # 31 - Numeric property ID

    # --- Property Classification (32-41) ---
    "PropertyType",             # 32 - Land, Retail, Multifamily, etc.
    "SubType",                  # 33 - Commercial, Shopping Center, etc.
    "SpecificUse",              # 34 - Retail, LIHTC, etc.
    "GrossLandAreaAcres",       # 35 - Decimal
    "GrossLandAreaSF",          # 36 - Square feet
    "Shape",                    # 37 - Rectangular, Irregular, etc.
    "Topography",               # 38 - Level, etc.
    "BuildingClass",            # 39 - A/B/C or NULL
    "YearBuilt",                # 40 - Integer or NULL
    "PropertyCondition",        # 41 - Good/Fair/etc. or NULL

    # --- Property Location (42-47) ---
    "CityMunicipality",        # 42 - City name
    "StateName",                # 43 - State name
    "Market",                   # 44 - Market area (e.g., "Wilmington, DE")
    "Submarket",                # 45 - Submarket area
    "CountyID",                 # 46 - FIPS code (e.g., 0500000US10003)
    "MarketOrientation",        # 47 - Rural/Suburban/Urban

    # --- Property Metrics (48-52) ---
    "PhotosCount",              # 48 - Integer
    "SaleCount",                # 49 - Integer or NULL
    "IECount",                  # 50 - Integer or NULL
    "LeaseCount",               # 51 - Integer or NULL
    "RentSurveyCount",          # 52 - Integer or NULL

    # --- Geography (53-57) ---
    "IRRRegion",                # 53 - East/West/South/Central
    "RooftopLatitude",          # 54 - Decimal
    "RooftopLongitude",         # 55 - Decimal
    "GrossBuildingSF",          # 56 - Square feet or NULL
    "GLARentableSF",            # 57 - Square feet or NULL

    # --- Office Info (58-61) ---
    "OfficeID",                 # 58 - Numeric
    "OfficeCode",               # 59 - "nnn - CityName, ST"
    "CompanyLocation",          # 60 - City, ST
    "Office_Region",            # 61 - Northeast/Southeast/Central/etc.

    # --- Client Info (62-66) ---
    "ClientCompanyID",          # 62 - Numeric
    "CompanyName",              # 63 - Client company name
    "ClientContactID",          # 64 - Numeric
    "ContactType",              # 65 - Management/Originator/etc.
    "CompanyType",              # 66 - Bank, Government, etc.

    # --- ZipCodeMaster Demographics (67-142) — 76 columns ---
    "ZipCode",                  # 67
    "Zip_Latitude",             # 68
    "Zip_Longitude",            # 69
    "Zip_StateAbbr",            # 70
    "Zip_StateName",            # 71
    "Zip_DecommissionedFlag",   # 72 - P (primary) or D
    "Zip_CityName",             # 73
    "Zip_CountyName",           # 74
    "Zip_CountyFIPS",           # 75
    "Zip_Congressional",        # 76
    "Zip_MSA",                  # 77 - NULL or number
    "Zip_CBSA",                 # 78
    "Zip_CBSA2",                # 79
    "Zip_Congressional2",       # 80
    "Zip_Congressional3",       # 81
    "Zip_MultiCounty",          # 82 - Y/N
    "Zip_FIPS",                 # 83
    "Zip_CityType",             # 84
    "Zip_CityAliasAbbr",        # 85 - P/N
    "Zip_PreferredFlag",        # 86 - Y/N
    "Zip_DecommFlag2",          # 87 - D or blank
    "Zip_CBSAName",             # 88
    "Zip_CBSAType",             # 89 - Metro/Micro
    "Zip_Population",           # 90
    "Zip_HousingUnits",         # 91
    "Zip_CSAName",              # 92
    "Zip_CSACode",              # 93
    "Zip_CBSACode2",            # 94
    "Zip_Combined_Code",        # 95
    "Zip_CensusTract",          # 96 - blank or tract ID
    "Zip_MSAName",              # 97
    "Zip_CMSAName",             # 98
    "Zip_PMSAName",             # 99
    "Zip_CensusRegion",         # 100 - South/Northeast/etc.
    "Zip_CensusDivision",       # 101 - South Atlantic, etc.
    "Zip_CensusDivisionCode",   # 102
    "Zip_CountyFIPS2",          # 103
    "Zip_PopulationEstimate",   # 104
    "Zip_HouseholdsPerZip",     # 105
    "Zip_AverageHouseValue",    # 106
    "Zip_IncomePerHousehold",   # 107
    "Zip_PersonsPerHousehold",  # 108
    "Zip_AverageHouseholdSize", # 109
    "Zip_MedianAge",            # 110
    "Zip_MedianAgeMale",        # 111
    "Zip_MedianAgeFemale",      # 112
    "Zip_DeliveryTotal",        # 113
    "Zip_SingleDelivery",       # 114
    "Zip_MultiDelivery",        # 115
    "Zip_GrowthRank",           # 116
    "Zip_GrowthIncrease",       # 117
    "Zip_MedianHouseValue",     # 118
    "Zip_AvgIncomePerHousehold",# 119
    "Zip_AvgHouseValue2",       # 120
    "Zip_MedianIncome",         # 121
    "Zip_NumberOfBusinesses",   # 122
    "Zip_NumberOfEmployees",    # 123
    "Zip_BusinessFirst8",       # 124
    "Zip_BusinessSecond8",      # 125
    "Zip_CompanyEmployees1k",   # 126
    "Zip_CompanyEmployees1kPct",# 127
    "Zip_CompanyEmployeesAll",  # 128
    "Zip_CompanyEmployeesAllPct",# 129
    "Zip_WorkersInZip",         # 130
    "Zip_WorkersOutZip",        # 131
    "Zip_WhiteCollar",          # 132 - NULL or numeric
    "Zip_BlueCollar",           # 133 - NULL or numeric
    "Zip_DeliveryResidential",  # 134
    "Zip_DeliveryBusiness",     # 135
    "Zip_DeliveryOther",        # 136
    "Zip_DeliveryAll",          # 137
    "Zip_LandArea",             # 138
    "Zip_WaterArea",            # 139
    "Zip_TotalDensity",         # 140
    "Zip_PopDensity",           # 141
    "Zip_PopCount",             # 142
]

assert len(JOBSDATA_COLUMNS) == 143, f"Expected 143 columns, got {len(JOBSDATA_COLUMNS)}"

# ZipCodeMaster columns start at index 67
ZIP_COLUMNS = [c for c in JOBSDATA_COLUMNS if c.startswith("Zip_")]
ZIP_NUMERIC_CANDIDATES = [
    "Zip_Latitude", "Zip_Longitude", "Zip_Population", "Zip_HousingUnits",
    "Zip_PopulationEstimate", "Zip_HouseholdsPerZip", "Zip_AverageHouseValue",
    "Zip_IncomePerHousehold", "Zip_PersonsPerHousehold", "Zip_AverageHouseholdSize",
    "Zip_MedianAge", "Zip_MedianAgeMale", "Zip_MedianAgeFemale",
    "Zip_DeliveryTotal", "Zip_SingleDelivery", "Zip_MultiDelivery",
    "Zip_GrowthRank", "Zip_GrowthIncrease", "Zip_MedianHouseValue",
    "Zip_AvgIncomePerHousehold", "Zip_AvgHouseValue2", "Zip_MedianIncome",
    "Zip_NumberOfBusinesses", "Zip_NumberOfEmployees",
    "Zip_LandArea", "Zip_WaterArea", "Zip_TotalDensity", "Zip_PopDensity",
    "Zip_PopCount", "Zip_WorkersInZip", "Zip_WorkersOutZip",
]


def load_jobsdata(nrows=None):
    """Load JobsData.csv with assigned headers."""
    path = DATA_DIR / "raw" / "JobsData.csv"
    print(f"Loading JobsData from {path}...")
    df = pd.read_csv(
        path,
        header=None,
        names=JOBSDATA_COLUMNS,
        encoding="utf-8-sig",
        low_memory=False,
        nrows=nrows,
    )
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def validate_columns(df):
    """Validate column positions against known value patterns."""
    print("\n" + "=" * 80)
    print("COLUMN VALIDATION")
    print("=" * 80)

    checks = []

    # [0] JobId — should be numeric
    jobid_numeric = pd.to_numeric(df["JobId"], errors="coerce").notna().mean()
    checks.append(("JobId (col 0) is numeric", jobid_numeric > 0.99))
    print(f"  JobId numeric: {jobid_numeric:.4f}")

    # [3] JobStatus — should contain 'Closed'
    has_closed = "Closed" in df["JobStatus"].unique()
    checks.append(("JobStatus (col 3) has 'Closed'", has_closed))
    print(f"  JobStatus has 'Closed': {has_closed}")

    # [7] NetFee — should be numeric dollar amounts
    netfee_numeric = pd.to_numeric(df["NetFee"], errors="coerce").notna().mean()
    checks.append(("NetFee (col 7) is numeric", netfee_numeric > 0.95))
    print(f"  NetFee numeric: {netfee_numeric:.4f}")

    # [15] Year — should be 4-digit years 2000-2026
    year_vals = pd.to_numeric(df["Year"], errors="coerce")
    year_valid = ((year_vals >= 2000) & (year_vals <= 2026)).mean()
    checks.append(("Year (col 15) is valid year", year_valid > 0.99))
    print(f"  Year in 2000-2026: {year_valid:.4f}")

    # [32] PropertyType — should contain known types
    known_ptypes = {"Land", "Retail", "Multifamily", "Office", "Industrial"}
    actual_ptypes = set(df["PropertyType"].dropna().unique())
    overlap = known_ptypes.intersection(actual_ptypes)
    checks.append(("PropertyType (col 32) has known types", len(overlap) >= 3))
    print(f"  PropertyType known overlap: {len(overlap)}/{len(known_ptypes)}")

    # [43] StateName — should contain US states
    known_states = {"California", "Texas", "Florida", "New York", "Illinois"}
    actual_states = set(df["StateName"].dropna().unique())
    state_overlap = known_states.intersection(actual_states)
    checks.append(("StateName (col 43) has known states", len(state_overlap) >= 3))
    print(f"  StateName known overlap: {len(state_overlap)}/{len(known_states)}")

    # [54] RooftopLatitude — should be US lat range
    lat = pd.to_numeric(df["RooftopLatitude"], errors="coerce")
    lat_valid = ((lat >= 18) & (lat <= 72)).mean()
    checks.append(("RooftopLatitude (col 54) in US range", lat_valid > 0.90))
    print(f"  Latitude in US range: {lat_valid:.4f}")

    # [61] Office_Region — known regions
    known_regions = {"Northeast", "Southeast", "Central", "West", "Southwest"}
    actual_regions = set(df["Office_Region"].dropna().unique())
    region_overlap = known_regions.intersection(actual_regions)
    checks.append(("Office_Region (col 61) has known regions", len(region_overlap) >= 3))
    print(f"  Office_Region known overlap: {len(region_overlap)}")

    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    print(f"\n  Validation: {passed}/{total} checks passed")

    if passed < total:
        for name, ok in checks:
            if not ok:
                print(f"  ⚠ FAILED: {name}")

    return {name: ok for name, ok in checks}


def profile_columns(df):
    """Profile all 143 columns: type, nulls, cardinality, top values."""
    print("\n" + "=" * 80)
    print("COLUMN PROFILING")
    print("=" * 80)

    profiles = {}
    for col in df.columns:
        series = df[col]
        null_count = series.isna().sum() + (series == "NULL").sum()
        null_pct = null_count / len(df) * 100

        # Try numeric
        numeric = pd.to_numeric(series.replace("NULL", np.nan), errors="coerce")
        is_numeric = numeric.notna().sum() > len(df) * 0.5

        profile = {
            "null_count": int(null_count),
            "null_pct": round(null_pct, 2),
            "unique": int(series.nunique()),
            "dtype": "numeric" if is_numeric else "categorical",
        }

        if is_numeric:
            clean = numeric.dropna()
            if len(clean) > 0:
                profile["min"] = float(clean.min())
                profile["max"] = float(clean.max())
                profile["mean"] = round(float(clean.mean()), 2)
                profile["median"] = round(float(clean.median()), 2)
                profile["std"] = round(float(clean.std()), 2)
        else:
            top5 = series.value_counts().head(5)
            profile["top_values"] = {str(k): int(v) for k, v in top5.items()}

        profiles[col] = profile

    # Print summary of high-null columns
    high_null = {k: v for k, v in profiles.items() if v["null_pct"] > 50}
    print(f"\n  Columns with >50% nulls: {len(high_null)}")
    for col, p in sorted(high_null.items(), key=lambda x: -x[1]["null_pct"])[:15]:
        print(f"    {col}: {p['null_pct']:.1f}% null")

    return profiles


def analyze_subtype(df):
    """Analyze SubType distribution for cardinality management."""
    print("\n" + "=" * 80)
    print("SUBTYPE ANALYSIS")
    print("=" * 80)

    subtype = df["SubType"].replace("NULL", np.nan)
    total = len(df)
    null_count = subtype.isna().sum()
    print(f"  Total rows: {total:,}")
    print(f"  SubType null: {null_count:,} ({null_count / total * 100:.1f}%)")
    print(f"  Unique SubTypes: {subtype.nunique()}")

    # Distribution
    dist = subtype.value_counts()
    print(f"\n  Top 20 SubTypes:")
    for st, count in dist.head(20).items():
        print(f"    {st:40s} {count:8,} ({count / total * 100:.2f}%)")

    # SubType by PropertyType
    print(f"\n  SubType counts by PropertyType (top 5 per):")
    for ptype in df["PropertyType"].value_counts().head(8).index:
        mask = df["PropertyType"] == ptype
        sub_dist = subtype[mask].value_counts()
        print(f"\n    {ptype} ({mask.sum():,} rows):")
        for st, count in sub_dist.head(5).items():
            print(f"      {st:36s} {count:8,}")

    # Minimum count thresholds
    thresholds = [10, 50, 100, 500]
    for t in thresholds:
        above = (dist >= t).sum()
        print(f"  SubTypes with >= {t:4d} rows: {above}")

    return {
        "unique_count": int(subtype.nunique()),
        "null_pct": round(null_count / total * 100, 2),
        "distribution": {str(k): int(v) for k, v in dist.head(40).items()},
        "by_property_type": {
            str(ptype): {str(k): int(v) for k, v in subtype[df["PropertyType"] == ptype].value_counts().head(10).items()}
            for ptype in df["PropertyType"].value_counts().head(10).index
        },
    }


def analyze_key_features(df):
    """Analyze features highlighted in the plan: JobLength_Days, Office, SubType, building area."""
    print("\n" + "=" * 80)
    print("KEY FEATURE ANALYSIS")
    print("=" * 80)

    report = {}

    # --- JobLength_Days ---
    print("\n--- JobLength_Days ---")
    jl = pd.to_numeric(df["JobLength_Days"].replace("NULL", np.nan), errors="coerce")
    valid = jl.dropna()
    null_pct = (len(df) - len(valid)) / len(df) * 100
    print(f"  Valid: {len(valid):,} ({100 - null_pct:.1f}%)")
    print(f"  Null: {null_pct:.1f}%")
    if len(valid) > 0:
        print(f"  Min: {valid.min():.0f}, Max: {valid.max():.0f}")
        print(f"  Mean: {valid.mean():.1f}, Median: {valid.median():.0f}")
        print(f"  P5: {valid.quantile(0.05):.0f}, P95: {valid.quantile(0.95):.0f}")
        report["JobLength_Days"] = {
            "valid_pct": round(100 - null_pct, 2),
            "min": float(valid.min()),
            "max": float(valid.max()),
            "mean": round(float(valid.mean()), 1),
            "median": float(valid.median()),
            "p5": float(valid.quantile(0.05)),
            "p95": float(valid.quantile(0.95)),
        }

    # --- CompanyLocation / Office_Region ---
    print("\n--- Office Features ---")
    for col in ["CompanyLocation", "Office_Region"]:
        vals = df[col].replace("NULL", np.nan)
        print(f"  {col}: {vals.nunique()} unique, {vals.isna().mean() * 100:.1f}% null")
        dist = vals.value_counts()
        for v, c in dist.head(8).items():
            print(f"    {v:30s} {c:8,}")
        report[col] = {
            "unique": int(vals.nunique()),
            "null_pct": round(vals.isna().mean() * 100, 2),
            "top_10": {str(k): int(v) for k, v in dist.head(10).items()},
        }

    # --- Building Area ---
    print("\n--- Building Area Features ---")
    for col in ["GrossBuildingSF", "GLARentableSF", "GrossLandAreaAcres", "GrossLandAreaSF"]:
        vals = pd.to_numeric(df[col].replace("NULL", np.nan), errors="coerce")
        valid = vals.dropna()
        null_pct = (len(df) - len(valid)) / len(df) * 100
        print(f"  {col}: {len(valid):,} valid ({100 - null_pct:.1f}%), "
              f"median={valid.median():.0f}" if len(valid) > 0 else f"  {col}: all null")
        if len(valid) > 0:
            report[col] = {
                "valid_pct": round(100 - null_pct, 2),
                "median": float(valid.median()),
                "p99": float(valid.quantile(0.99)),
            }

    # --- NetFee ---
    print("\n--- NetFee (Target) ---")
    netfee = pd.to_numeric(df["NetFee"].replace("NULL", np.nan), errors="coerce")
    valid_nf = netfee.dropna()
    valid_nf = valid_nf[valid_nf > 0]
    print(f"  Valid (>0): {len(valid_nf):,}")
    print(f"  Min: ${valid_nf.min():,.0f}, Max: ${valid_nf.max():,.0f}")
    print(f"  Mean: ${valid_nf.mean():,.0f}, Median: ${valid_nf.median():,.0f}")
    print(f"  P1: ${valid_nf.quantile(0.01):,.0f}, P99: ${valid_nf.quantile(0.99):,.0f}")
    report["NetFee"] = {
        "valid_count": int(len(valid_nf)),
        "min": float(valid_nf.min()),
        "max": float(valid_nf.max()),
        "mean": round(float(valid_nf.mean()), 0),
        "median": float(valid_nf.median()),
        "p01": float(valid_nf.quantile(0.01)),
        "p99": float(valid_nf.quantile(0.99)),
    }

    return report


def analyze_zipcode_features(df):
    """Identify top ZipCodeMaster features by correlation with NetFee."""
    print("\n" + "=" * 80)
    print("ZIPCODE FEATURE ANALYSIS")
    print("=" * 80)

    netfee = pd.to_numeric(df["NetFee"].replace("NULL", np.nan), errors="coerce")

    correlations = {}
    coverage = {}

    for col in ZIP_NUMERIC_CANDIDATES:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col].replace("NULL", np.nan), errors="coerce")
        valid_mask = vals.notna() & netfee.notna() & (netfee > 0)
        valid_count = valid_mask.sum()
        cov = valid_count / len(df) * 100
        coverage[col] = round(cov, 2)

        if valid_count > 1000:
            corr = vals[valid_mask].corr(netfee[valid_mask])
            if pd.notna(corr):
                correlations[col] = round(float(corr), 4)

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"\n  Top 15 ZipCodeMaster features by |correlation| with NetFee:")
    for col, corr in sorted_corr[:15]:
        print(f"    {col:40s} r={corr:+.4f}  coverage={coverage[col]:.1f}%")

    # Select top 10-15 with good coverage (>60%)
    top_features = [
        col for col, corr in sorted_corr
        if coverage[col] > 60
    ][:15]

    print(f"\n  Selected top features (coverage > 60%): {len(top_features)}")
    for col in top_features:
        print(f"    {col}")

    return {
        "correlations": dict(sorted_corr),
        "coverage": coverage,
        "selected_top_features": top_features,
    }


def analyze_date_distribution(df):
    """Analyze temporal distribution for data filtering decisions."""
    print("\n" + "=" * 80)
    print("TEMPORAL DISTRIBUTION")
    print("=" * 80)

    year = pd.to_numeric(df["Year"], errors="coerce")
    year_dist = year.value_counts().sort_index()

    print(f"\n  Year distribution:")
    for yr, count in year_dist.items():
        if pd.notna(yr):
            bar = "█" * (count // 5000)
            print(f"    {int(yr)}: {count:8,}  {bar}")

    # 2018+ filter stats
    mask_2018 = year >= 2018
    print(f"\n  2018+ rows: {mask_2018.sum():,} ({mask_2018.mean() * 100:.1f}%)")

    # JobStatus distribution
    print(f"\n  JobStatus distribution:")
    status_dist = df["JobStatus"].value_counts()
    for status, count in status_dist.items():
        print(f"    {status:20s} {count:8,}")

    # Closed + 2018+
    closed_2018 = (df["JobStatus"] == "Closed") & mask_2018
    print(f"\n  Closed + 2018+: {closed_2018.sum():,}")

    return {
        "year_distribution": {str(int(k)): int(v) for k, v in year_dist.items() if pd.notna(k)},
        "total_2018_plus": int(mask_2018.sum()),
        "closed_2018_plus": int(closed_2018.sum()),
    }


def analyze_join_fields(df):
    """Analyze fields needed for BidData join."""
    print("\n" + "=" * 80)
    print("JOIN FIELD ANALYSIS")
    print("=" * 80)

    report = {}

    # JobId uniqueness
    jobid = df["JobId"]
    print(f"  JobId unique: {jobid.nunique():,} / {len(df):,} ({jobid.nunique() / len(df) * 100:.1f}%)")
    dup_jobids = jobid[jobid.duplicated(keep=False)]
    print(f"  Duplicate JobId rows: {len(dup_jobids):,}")
    report["jobid_unique"] = int(jobid.nunique())
    report["jobid_duplicates"] = int(len(dup_jobids))

    # OfficeID coverage
    office_id = df["OfficeID"].replace("NULL", np.nan)
    print(f"  OfficeID unique: {office_id.nunique()}, null: {office_id.isna().mean() * 100:.1f}%")
    report["office_id_unique"] = int(office_id.nunique())

    # PropertyID coverage
    prop_id = df["PropertyID"].replace("NULL", np.nan)
    print(f"  PropertyID unique: {prop_id.nunique():,}, null: {prop_id.isna().mean() * 100:.1f}%")
    report["property_id_unique"] = int(prop_id.nunique())

    return report


def main():
    print("=" * 80)
    print("JOBSDATA EDA & COLUMN VALIDATION — Phase 0")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load all data
    df = load_jobsdata()

    # Run all analyses
    validation = validate_columns(df)
    profiles = profile_columns(df)
    subtype_analysis = analyze_subtype(df)
    key_features = analyze_key_features(df)
    zip_analysis = analyze_zipcode_features(df)
    temporal = analyze_date_distribution(df)
    join_fields = analyze_join_fields(df)

    # Compile full report
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "column_names": JOBSDATA_COLUMNS,
        "validation": validation,
        "subtype_analysis": subtype_analysis,
        "key_features": key_features,
        "zipcode_analysis": zip_analysis,
        "temporal_distribution": temporal,
        "join_fields": join_fields,
        "column_profiles": profiles,
    }

    # Save report
    report_path = REPORTS_DIR / "jobsdata_eda_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✓ EDA report saved: {report_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    all_passed = all(validation.values())
    print(f"  Column validation: {'✅ ALL PASSED' if all_passed else '⚠ SOME FAILED'}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Closed + 2018+: {temporal['closed_2018_plus']:,}")
    print(f"  SubType categories: {subtype_analysis['unique_count']}")
    print(f"  Top Zip features: {len(zip_analysis['selected_top_features'])}")
    print(f"  Report: {report_path}")

    return report


if __name__ == "__main__":
    report = main()
