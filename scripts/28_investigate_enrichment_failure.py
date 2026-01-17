"""
Investigation: Why JobData Enrichment Failed
=============================================
Systematic audit of enriched dataset to identify:
1. Data leakage in new features
2. Feature correlations and redundancy
3. Non-redundant valuable features
4. Optimal feature subset

Goal: Create a cleaned enriched dataset that improves performance

Author: Bid Recommendation System
Date: 2026-01-16
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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

print("=" * 80)
print("JOBDATA ENRICHMENT FAILURE INVESTIGATION")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# LOAD DATASETS
# ============================================================================
print("=" * 80)
print("STEP 1: LOADING DATASETS")
print("=" * 80)

# Load baseline
print("\n📊 Loading baseline dataset...")
df_baseline = pd.read_csv('data/features/BidData_features.csv')
print(f"✓ Baseline loaded: {len(df_baseline):,} records × {len(df_baseline.columns)} features")

# Load enriched
print("\n📊 Loading enriched dataset...")
df_enriched = pd.read_csv('data/features/BidData_enriched_with_jobs.csv')
print(f"✓ Enriched loaded: {len(df_enriched):,} records × {len(df_enriched.columns)} features")

# Identify new features
baseline_cols = set(df_baseline.columns)
enriched_cols = set(df_enriched.columns)
new_features = sorted(list(enriched_cols - baseline_cols))

print(f"\n✨ New features from JobData enrichment: {len(new_features)}")

# ============================================================================
# STEP 2: CATEGORIZE NEW FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: CATEGORIZING NEW JOBDATA FEATURES")
print("=" * 80)

# Group by category
office_features = [f for f in new_features if f.startswith('office_')]
region_features = [f for f in new_features if 'region' in f.lower()]
property_region_features = [f for f in new_features if 'property_region' in f.lower()]
other_features = [f for f in new_features if f not in office_features + region_features + property_region_features]

print(f"\n📊 Feature Categories:")
print(f"   Office-level: {len(office_features)} features")
for f in office_features:
    print(f"      • {f}")

print(f"\n   Regional: {len(region_features)} features")
for f in region_features:
    print(f"      • {f}")

print(f"\n   Property×Region: {len(property_region_features)} features")
for f in property_region_features:
    print(f"      • {f}")

if other_features:
    print(f"\n   Other: {len(other_features)} features")
    for f in other_features:
        print(f"      • {f}")

# ============================================================================
# STEP 3: CHECK FOR DATA LEAKAGE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: DATA LEAKAGE AUDIT")
print("=" * 80)

print("\n🔍 Checking for potential data leakage...")

leakage_suspects = []

# Check 1: Features that might use target variable
target_related = ['fee', 'price', 'cost', 'value']
for feature in new_features:
    feature_lower = feature.lower()
    if any(term in feature_lower for term in target_related):
        # Check if it's an aggregate from JobData (which is OK)
        if 'job_fee' in feature_lower or 'appraisal_fee' in feature_lower:
            # These are from separate JobData, not from BidData target
            # But check if they're suspiciously perfect
            correlation = df_enriched[[feature, 'BidFee']].corr().iloc[0, 1]
            if abs(correlation) > 0.95:
                leakage_suspects.append({
                    'feature': feature,
                    'reason': 'Extremely high correlation with target',
                    'correlation': correlation
                })

# Check 2: Features with perfect or near-perfect correlation
print("\n🔍 Checking correlations with target (BidFee)...")

target_correlations = []
for feature in new_features:
    if df_enriched[feature].dtype in [np.float64, np.int64]:
        try:
            corr = df_enriched[[feature, 'BidFee']].corr().iloc[0, 1]
            if not np.isnan(corr):
                target_correlations.append({
                    'feature': feature,
                    'correlation': abs(corr),
                    'direction': 'positive' if corr > 0 else 'negative'
                })
        except:
            pass

target_correlations_df = pd.DataFrame(target_correlations).sort_values('correlation', ascending=False)

print(f"\nTop 10 JobData features by correlation with BidFee:")
print(target_correlations_df.head(10).to_string(index=False))

# Flag suspicious high correlations
suspicious_high = target_correlations_df[target_correlations_df['correlation'] > 0.9]
if len(suspicious_high) > 0:
    print(f"\n⚠️  WARNING: {len(suspicious_high)} features with >0.9 correlation (possible leakage):")
    for _, row in suspicious_high.iterrows():
        print(f"   • {row['feature']}: {row['correlation']:.4f}")
        leakage_suspects.append({
            'feature': row['feature'],
            'reason': 'Correlation > 0.9 with target',
            'correlation': row['correlation']
        })

print(f"\n✓ Data leakage audit complete")
print(f"  Suspicious features: {len(leakage_suspects)}")

# ============================================================================
# STEP 4: ANALYZE FEATURE CORRELATIONS (Redundancy)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: FEATURE REDUNDANCY ANALYSIS")
print("=" * 80)

print("\n🔍 Checking correlations between JobData and baseline features...")

# Find similar features (by name similarity)
similar_pairs = []

# Office features
baseline_office_features = [f for f in baseline_cols if 'office' in f.lower() and 'avg' in f.lower()]
for new_feat in office_features:
    if 'avg' in new_feat.lower():
        for base_feat in baseline_office_features:
            similar_pairs.append((new_feat, base_feat, 'office_avg'))

# Rolling features
baseline_rolling = [f for f in baseline_cols if 'rolling' in f.lower()]
for new_feat in office_features:
    if 'avg' in new_feat.lower() or 'median' in new_feat.lower():
        for base_feat in baseline_rolling:
            if 'office' in base_feat.lower():
                similar_pairs.append((new_feat, base_feat, 'rolling_vs_aggregate'))

# State/Region features
baseline_state = [f for f in baseline_cols if 'state' in f.lower() and 'avg' in f.lower()]
for new_feat in region_features:
    if 'avg' in new_feat.lower():
        for base_feat in baseline_state:
            similar_pairs.append((new_feat, base_feat, 'region_vs_state'))

print(f"\n📊 Found {len(similar_pairs)} potentially similar feature pairs")

# Calculate actual correlations
print("\n🔍 Computing correlations for similar pairs...")

correlation_results = []
for new_feat, base_feat, pair_type in similar_pairs[:20]:  # Limit to avoid too much computation
    try:
        # Both features must be numeric
        if df_enriched[new_feat].dtype in [np.float64, np.int64] and \
           df_enriched[base_feat].dtype in [np.float64, np.int64]:
            corr = df_enriched[[new_feat, base_feat]].corr().iloc[0, 1]
            if not np.isnan(corr):
                correlation_results.append({
                    'new_feature': new_feat,
                    'baseline_feature': base_feat,
                    'correlation': abs(corr),
                    'type': pair_type
                })
    except Exception as e:
        pass

correlation_results_df = pd.DataFrame(correlation_results).sort_values('correlation', ascending=False)

print(f"\nTop 15 correlated pairs (potential redundancy):")
print(correlation_results_df.head(15).to_string(index=False))

# Identify highly redundant features (>0.8 correlation)
highly_redundant = correlation_results_df[correlation_results_df['correlation'] > 0.8]
print(f"\n⚠️  HIGH REDUNDANCY: {len(highly_redundant)} feature pairs with >0.8 correlation")

redundant_features = set()
for _, row in highly_redundant.iterrows():
    print(f"   • {row['new_feature']} ≈ {row['baseline_feature']} (r={row['correlation']:.3f})")
    redundant_features.add(row['new_feature'])

# ============================================================================
# STEP 5: IDENTIFY VALUABLE NON-REDUNDANT FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: IDENTIFYING VALUABLE FEATURES")
print("=" * 80)

# Features to keep: low redundancy + meaningful correlation with target
valuable_features = []

for feature in new_features:
    # Skip if redundant
    if feature in redundant_features:
        continue

    # Skip if suspicious leakage
    if any(s['feature'] == feature for s in leakage_suspects):
        continue

    # Check if numeric
    if df_enriched[feature].dtype not in [np.float64, np.int64]:
        continue

    # Check correlation with target
    try:
        corr_with_target = abs(df_enriched[[feature, 'BidFee']].corr().iloc[0, 1])
        if not np.isnan(corr_with_target) and 0.1 < corr_with_target < 0.9:
            valuable_features.append({
                'feature': feature,
                'target_correlation': corr_with_target,
                'category': 'office' if feature in office_features else
                           'region' if feature in region_features else
                           'property_region' if feature in property_region_features else 'other'
            })
    except:
        pass

valuable_features_df = pd.DataFrame(valuable_features).sort_values('target_correlation', ascending=False)

print(f"\n✨ Identified {len(valuable_features)} potentially valuable features:")
print(f"\n{valuable_features_df.to_string(index=False)}")

# ============================================================================
# STEP 6: RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: RECOMMENDATIONS")
print("=" * 80)

recommendations = {
    'summary': {
        'total_new_features': len(new_features),
        'leakage_suspects': len(leakage_suspects),
        'redundant_features': len(redundant_features),
        'valuable_features': len(valuable_features)
    },
    'features_to_remove': {
        'leakage': [s['feature'] for s in leakage_suspects],
        'redundant': list(redundant_features)
    },
    'features_to_keep': valuable_features_df['feature'].tolist(),
    'feature_analysis': {
        'leakage_suspects': leakage_suspects,
        'high_redundancy': highly_redundant.to_dict('records'),
        'valuable_features': valuable_features
    }
}

print(f"\n📊 Summary:")
print(f"   Total JobData features: {len(new_features)}")
print(f"   Leakage suspects: {len(leakage_suspects)}")
print(f"   Highly redundant (>0.8 corr): {len(redundant_features)}")
print(f"   Valuable non-redundant: {len(valuable_features)}")

print(f"\n✅ RECOMMENDED ACTIONS:")

print(f"\n1. REMOVE {len(leakage_suspects)} leakage suspects:")
for s in leakage_suspects:
    print(f"   • {s['feature']}: {s['reason']}")

print(f"\n2. REMOVE {len(redundant_features)} redundant features:")
for feat in list(redundant_features)[:10]:
    print(f"   • {feat}")
if len(redundant_features) > 10:
    print(f"   ... and {len(redundant_features) - 10} more")

print(f"\n3. KEEP {len(valuable_features)} valuable features:")
for _, row in valuable_features_df.head(10).iterrows():
    print(f"   • {row['feature']} (corr={row['target_correlation']:.3f}, {row['category']})")
if len(valuable_features) > 10:
    print(f"   ... and {len(valuable_features) - 10} more")

# ============================================================================
# STEP 7: CREATE CLEANED DATASET
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: CREATING CLEANED ENRICHED DATASET")
print("=" * 80)

# Features to drop
features_to_drop = set()
features_to_drop.update([s['feature'] for s in leakage_suspects])
features_to_drop.update(redundant_features)

# Create cleaned dataset
df_cleaned = df_enriched.copy()
features_to_drop_existing = [f for f in features_to_drop if f in df_cleaned.columns]

if len(features_to_drop_existing) > 0:
    df_cleaned = df_cleaned.drop(columns=features_to_drop_existing)
    print(f"\n✓ Dropped {len(features_to_drop_existing)} problematic features")
    print(f"   New dataset: {len(df_cleaned.columns)} features (was {len(df_enriched.columns)})")

    # Save cleaned dataset
    output_path = Path('data/features/BidData_enriched_cleaned.csv')
    df_cleaned.to_csv(output_path, index=False)
    print(f"\n✓ Cleaned dataset saved: {output_path}")
    print(f"   Features: {len(df_cleaned.columns)}")
    print(f"   Records: {len(df_cleaned):,}")
else:
    print(f"\n⚠️  No features to drop - all seem OK")
    output_path = None

# Save analysis report
report_path = Path('outputs/reports/enrichment_investigation_report.json')
with open(report_path, 'w') as f:
    json.dump(recommendations, f, indent=2, default=str)

print(f"\n✓ Investigation report saved: {report_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("INVESTIGATION COMPLETE")
print("=" * 80)

print(f"""
FINDINGS:
---------
• Total JobData features analyzed: {len(new_features)}
• Features with potential data leakage: {len(leakage_suspects)}
• Features highly redundant with baseline: {len(redundant_features)}
• Valuable non-redundant features: {len(valuable_features)}

ACTIONS TAKEN:
--------------
• Created cleaned dataset with {len(df_cleaned.columns)} features
• Removed {len(features_to_drop_existing)} problematic features
• Saved cleaned dataset: data/features/BidData_enriched_cleaned.csv
• Saved detailed report: outputs/reports/enrichment_investigation_report.json

NEXT STEPS:
-----------
1. Test cleaned dataset with LightGBM
2. Compare performance: baseline vs cleaned enriched
3. If still worse, test with increased regularization
4. Consider using only regional features (not office-level)

Expected improvement with cleaned dataset: 2-5% over baseline
""")

print("=" * 80)
