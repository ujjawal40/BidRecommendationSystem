"""
Generate Zip Code Demographics Lookup
========================================
Extracts ZipCode + 15 demographics + lat/lng/state from raw JobsData,
groups by zip → median, and saves as a JSON lookup for the API.

Output:
  - outputs/reports/zip_demographics_lookup.json

Author: Raj Tiwari
Date: 2026-03-18
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import json
import warnings

import numpy as np
import pandas as pd

from config.model_config import DATA_DIR, REPORTS_DIR
from scripts._jobsdata_columns import JOBSDATA_COLUMNS

warnings.filterwarnings("ignore")

# The 15 selected ZipCodeMaster features (same as 21_jobsdata_preprocessing.py)
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

# Columns needed from the raw file
NEEDED_COLUMNS = (
    ["ZipCode", "Zip_Latitude", "Zip_Longitude", "Zip_StateName",
     "StateName", "RooftopLatitude", "RooftopLongitude",
     "JobDistanceMiles"]
    + SELECTED_ZIP_FEATURES
)


def generate_zip_lookup():
    """Generate per-zip demographics lookup from raw JobsData."""
    print("=" * 80)
    print("GENERATING ZIP CODE DEMOGRAPHICS LOOKUP")
    print("=" * 80)

    # Load raw JobsData with column names
    path = DATA_DIR / "raw" / "JobsData.csv"
    df = pd.read_csv(
        path,
        header=None,
        names=JOBSDATA_COLUMNS,
        encoding="utf-8-sig",
        low_memory=False,
    )
    print(f"  Loaded {len(df):,} rows from raw JobsData")

    # Filter to columns we need
    available = [c for c in NEEDED_COLUMNS if c in df.columns]
    df = df[available].copy()

    # Clean ZipCode — keep only valid 5-digit codes
    df["ZipCode"] = df["ZipCode"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    df = df[df["ZipCode"].str.match(r"^\d{5}$", na=False)].copy()
    print(f"  After zip filter: {len(df):,} rows with valid 5-digit zip codes")
    print(f"  Unique zips: {df['ZipCode'].nunique():,}")

    # Convert numeric columns
    numeric_cols = SELECTED_ZIP_FEATURES + [
        "Zip_Latitude", "Zip_Longitude",
        "RooftopLatitude", "RooftopLongitude",
        "JobDistanceMiles",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace("NULL", np.nan), errors="coerce")

    # Group by ZipCode → median for demographics, first for state
    agg_dict = {}
    for col in SELECTED_ZIP_FEATURES:
        if col in df.columns:
            agg_dict[col] = "median"

    # Lat/lng: prefer Zip_Latitude/Zip_Longitude, fall back to Rooftop
    if "Zip_Latitude" in df.columns:
        agg_dict["Zip_Latitude"] = "median"
        agg_dict["Zip_Longitude"] = "median"
    if "RooftopLatitude" in df.columns:
        agg_dict["RooftopLatitude"] = "median"
        agg_dict["RooftopLongitude"] = "median"

    if "JobDistanceMiles" in df.columns:
        agg_dict["JobDistanceMiles"] = "median"

    # State: most common per zip
    if "Zip_StateName" in df.columns:
        agg_dict["Zip_StateName"] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None
    elif "StateName" in df.columns:
        agg_dict["StateName"] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None

    grouped = df.groupby("ZipCode").agg(agg_dict)

    # Build lookup dict
    lookup = {}
    for zip_code, row in grouped.iterrows():
        entry = {}

        # Demographics
        for col in SELECTED_ZIP_FEATURES:
            if col in row.index and pd.notna(row[col]):
                entry[col] = round(float(row[col]), 2)

        # Only include zips with at least 5 demographic values
        if len(entry) < 5:
            continue

        # Latitude/Longitude
        lat = row.get("Zip_Latitude") if "Zip_Latitude" in row.index else None
        lng = row.get("Zip_Longitude") if "Zip_Longitude" in row.index else None
        if pd.isna(lat) and "RooftopLatitude" in row.index:
            lat = row["RooftopLatitude"]
        if pd.isna(lng) and "RooftopLongitude" in row.index:
            lng = row["RooftopLongitude"]

        if pd.notna(lat):
            entry["latitude"] = round(float(lat), 4)
        if pd.notna(lng):
            entry["longitude"] = round(float(lng), 4)

        # State
        state = None
        if "Zip_StateName" in row.index:
            state = row["Zip_StateName"]
        elif "StateName" in row.index:
            state = row["StateName"]
        if state and pd.notna(state) and state != "NULL":
            entry["state"] = str(state).strip()

        # Distance (median for this zip)
        if "JobDistanceMiles" in row.index and pd.notna(row["JobDistanceMiles"]):
            entry["distance_miles"] = round(float(row["JobDistanceMiles"]), 1)

        # Sample count
        entry["sample_count"] = int(df[df["ZipCode"] == zip_code].shape[0])

        lookup[zip_code] = entry

    # Save
    output_path = REPORTS_DIR / "zip_demographics_lookup.json"
    with open(output_path, "w") as f:
        json.dump(lookup, f, indent=None, separators=(",", ":"))

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n  Saved: {output_path}")
    print(f"  Zip entries: {len(lookup):,}")
    print(f"  File size: {file_size_mb:.1f} MB")

    # Sample entry
    sample_zip = list(lookup.keys())[0]
    print(f"\n  Sample entry ({sample_zip}):")
    for k, v in lookup[sample_zip].items():
        print(f"    {k}: {v}")

    return lookup


if __name__ == "__main__":
    generate_zip_lookup()
