"""
Generate api_precomputed_stats.json for the prediction service.

This script reads the features CSV and produces the precomputed statistics
JSON that the API loads at startup. Run this after retraining or when the
feature data changes.

Usage:
    python scripts/generate_api_stats.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import pandas as pd
from config.model_config import FEATURES_DATA, REPORTS_DIR, DATA_START_DATE, USE_RECENT_DATA_ONLY


def generate_stats():
    """Generate precomputed stats from feature CSV."""
    if not FEATURES_DATA.exists():
        print(f"Error: Feature data not found at {FEATURES_DATA}")
        sys.exit(1)

    print(f"Loading features from {FEATURES_DATA}...")
    df_all = pd.read_csv(FEATURES_DATA)
    df_all['BidDate'] = pd.to_datetime(df_all['BidDate'])
    df_full = df_all.copy()

    if USE_RECENT_DATA_ONLY:
        df = df_all[df_all['BidDate'] >= pd.Timestamp(DATA_START_DATE)].copy()
    else:
        df = df_all.copy()

    total_count = len(df_full)
    stats = {}

    # Dropdown options
    stats['segments'] = sorted(df_full['BusinessSegment'].dropna().unique().tolist())
    stats['property_types'] = sorted(df_full['PropertyType'].dropna().unique().tolist())
    stats['states'] = sorted(df_full['PropertyState'].dropna().unique().tolist())
    stats['offices'] = sorted(df_full['OfficeId'].dropna().unique().astype(int).tolist())

    # Global statistics
    stats['global_avg_fee'] = df['BidFee'].mean()
    stats['global_std_fee'] = df['BidFee'].std()
    stats['global_win_rate'] = df['Won'].mean()

    # Segment statistics
    segment_stats = df_full.groupby('BusinessSegment')['BidFee'].agg(['mean', 'std', 'count'])
    stats['segment_avg_fee'] = segment_stats['mean'].to_dict()
    stats['segment_std_fee'] = segment_stats['std'].fillna(0).to_dict()
    stats['segment_count'] = {k: int(v) for k, v in segment_stats['count'].to_dict().items()}
    stats['segment_win_rate'] = df_full.groupby('BusinessSegment')['Won'].mean().to_dict()
    stats['segment_frequency'] = (df_full.groupby('BusinessSegment').size() / total_count).to_dict()

    # State statistics
    state_stats = df_full.groupby('PropertyState')['BidFee'].agg(['mean', 'std'])
    stats['state_avg_fee'] = state_stats['mean'].to_dict()
    stats['state_std_fee'] = state_stats['std'].fillna(0).to_dict()
    stats['state_win_rate'] = df_full.groupby('PropertyState')['Won'].mean().to_dict()
    stats['state_frequency'] = (df_full.groupby('PropertyState').size() / total_count).to_dict()
    stats['state_count'] = {k: int(v) for k, v in df_full.groupby('PropertyState').size().to_dict().items()}

    # Property type statistics
    proptype_stats = df_full.groupby('PropertyType')['BidFee'].agg(['mean', 'std'])
    stats['propertytype_avg_fee'] = proptype_stats['mean'].to_dict()
    stats['propertytype_std_fee'] = proptype_stats['std'].fillna(0).to_dict()
    stats['propertytype_win_rate'] = df_full.groupby('PropertyType')['Won'].mean().to_dict()
    stats['PropertyType_frequency'] = (df_full.groupby('PropertyType').size() / total_count).to_dict()

    # Office statistics
    office_stats = df_full.groupby('OfficeId')['BidFee'].agg(['mean', 'std'])
    stats['office_avg_fee'] = {str(int(k)): v for k, v in office_stats['mean'].to_dict().items()}
    stats['office_std_fee'] = {str(int(k)): v for k, v in office_stats['std'].fillna(0).to_dict().items()}
    stats['office_win_rate'] = {str(int(k)): v for k, v in df_full.groupby('OfficeId')['Won'].mean().to_dict().items()}

    # State-segment combo frequency
    combo_stats = df.groupby(['PropertyState', 'BusinessSegment']).size()
    stats['state_segment_combo_freq'] = {f"{k[0]}|{k[1]}": int(v) for k, v in combo_stats.to_dict().items()}

    # Write output
    output_path = REPORTS_DIR / 'api_precomputed_stats.json'
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Stats written to {output_path}")
    print(f"  Segments: {len(stats['segments'])}")
    print(f"  States: {len(stats['states'])} (with state_count)")
    print(f"  Property types: {len(stats['property_types'])}")
    print(f"  Offices: {len(stats['offices'])}")


if __name__ == '__main__':
    generate_stats()
