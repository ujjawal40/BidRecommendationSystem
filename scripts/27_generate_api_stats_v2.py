"""
Generate API Precomputed Stats for v2 Models
===============================================
Creates the JSON files needed by EnhancedPredictionService at runtime.

Outputs:
  - outputs/reports/api_precomputed_stats_v2.json
  - outputs/reports/feature_defaults_v2.json

Author: Ujjawal Dwivedi
Date: 2026-02-15
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import json
import warnings

import numpy as np
import pandas as pd

from config.model_config import DATA_DIR, REPORTS_DIR

warnings.filterwarnings("ignore")


def generate_phase1a_stats():
    """Generate precomputed stats from JobsData features for Phase 1A v2."""
    print("=" * 80)
    print("GENERATING PHASE 1A v2 API STATS")
    print("=" * 80)

    df = pd.read_csv(DATA_DIR / "features" / "JobsData_features_v2.csv", low_memory=False)
    df["StartDate"] = pd.to_datetime(df["StartDate"])
    print(f"  Loaded {len(df):,} rows")

    stats = {}

    # Global
    stats["global_avg_fee"] = float(df["NetFee"].mean())
    stats["global_std_fee"] = float(df["NetFee"].std())

    # Dropdown options
    stats["segments"] = sorted(df["BusinessSegment"].dropna().unique().tolist())
    stats["property_types"] = sorted(df["PropertyType"].dropna().unique().tolist())
    stats["states"] = sorted(df["StateName"].dropna().unique().tolist())
    stats["subtypes"] = sorted(df["SubType"].dropna().unique().tolist())
    stats["office_regions"] = sorted(df["Office_Region"].dropna().unique().tolist())
    stats["company_locations"] = sorted(df["CompanyLocation"].dropna().unique().tolist())

    # Segment stats
    seg = df.groupby("BusinessSegment")
    stats["segment_avg_fee"] = seg["NetFee"].mean().to_dict()
    stats["segment_std_fee"] = seg["NetFee"].std().fillna(0).to_dict()
    stats["segment_count"] = seg.size().to_dict()
    stats["segment_frequency"] = seg.size().to_dict()  # raw count for frequency encoding

    # State stats
    st = df.groupby("StateName")
    stats["state_avg_fee"] = st["NetFee"].mean().to_dict()
    stats["state_count"] = st.size().to_dict()
    stats["state_frequency"] = (st.size() / len(df)).to_dict()  # proportion

    # PropertyType stats
    pt = df.groupby("PropertyType")
    stats["propertytype_avg_fee"] = pt["NetFee"].mean().to_dict()
    stats["propertytype_frequency"] = pt.size().to_dict()

    # SubType stats
    sub = df.groupby("SubType")
    stats["subtype_avg_fee"] = sub["NetFee"].mean().to_dict()
    stats["subtype_frequency"] = sub.size().to_dict()

    # Office_Region stats
    oreg = df.groupby("Office_Region")
    stats["office_region_avg_fee"] = oreg["NetFee"].mean().to_dict()
    stats["office_region_frequency"] = oreg.size().to_dict()

    # CompanyLocation stats
    cloc = df.groupby("CompanyLocation")
    stats["company_location_frequency"] = cloc.size().to_dict()

    # CompanyType stats
    if "CompanyType" in df.columns:
        ct = df.groupby("CompanyType")
        stats["companytype_avg_fee"] = ct["NetFee"].mean().to_dict()
        stats["companytype_frequency"] = ct.size().to_dict()

    # ContactType stats
    if "ContactType" in df.columns:
        ctype = df.groupby("ContactType")
        stats["contacttype_avg_fee"] = ctype["NetFee"].mean().to_dict()

    # Rolling averages (use last-period means as proxy)
    stats["rolling_avg_fee_segment"] = stats["segment_avg_fee"].copy()
    stats["rolling_avg_fee_state"] = stats["state_avg_fee"].copy()

    # State coordinates (lat/lon averages)
    state_coords = df.groupby("StateName").agg({
        "RooftopLatitude": "median",
        "RooftopLongitude": "median",
    })
    stats["state_latitude"] = state_coords["RooftopLatitude"].to_dict()
    stats["state_longitude"] = state_coords["RooftopLongitude"].to_dict()

    # SubType by PropertyType mapping (for cascading dropdown)
    subtype_by_proptype = {}
    for ptype in stats["property_types"]:
        mask = df["PropertyType"] == ptype
        subtypes = df.loc[mask, "SubType"].value_counts()
        subtype_by_proptype[ptype] = subtypes.head(20).index.tolist()
    stats["subtypes_by_property_type"] = subtype_by_proptype

    # Save
    output_path = REPORTS_DIR / "api_precomputed_stats_v2.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  Saved: {output_path}")

    return stats


def generate_feature_defaults():
    """Generate segment-specific default values for v2 model features."""
    print("\n" + "=" * 80)
    print("GENERATING FEATURE DEFAULTS v2")
    print("=" * 80)

    # Load model metadata to get feature list
    from config.model_config import MODELS_DIR
    meta_path = MODELS_DIR / "lightgbm_bidfee_v2_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    features = meta["features"]

    # Load feature data
    df = pd.read_csv(DATA_DIR / "features" / "JobsData_features_v2.csv", low_memory=False)

    defaults = {}
    for feat in features:
        if feat in df.columns:
            series = pd.to_numeric(df[feat], errors="coerce")
            if series.notna().sum() > 0:
                defaults[feat] = {
                    "global_median": float(series.median()),
                    "global_mean": float(series.mean()),
                }
                # Segment-specific medians
                for seg in df["BusinessSegment"].unique():
                    seg_vals = series[df["BusinessSegment"] == seg]
                    if seg_vals.notna().sum() > 0:
                        defaults[feat][f"segment_{seg}_median"] = float(seg_vals.median())

    output_path = REPORTS_DIR / "feature_defaults_v2.json"
    with open(output_path, "w") as f:
        json.dump(defaults, f, indent=2)
    print(f"  Saved: {output_path} ({len(defaults)} features)")

    return defaults


def main():
    stats = generate_phase1a_stats()
    defaults = generate_feature_defaults()

    print("\n" + "=" * 80)
    print("API STATS GENERATION COMPLETE")
    print("=" * 80)
    print(f"  Segments: {len(stats['segments'])}")
    print(f"  States: {len(stats['states'])}")
    print(f"  SubTypes: {len(stats['subtypes'])}")
    print(f"  Office Regions: {len(stats['office_regions'])}")
    print(f"  Feature defaults: {len(defaults)}")


if __name__ == "__main__":
    main()
