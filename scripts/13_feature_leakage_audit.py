"""
Feature Leakage Audit
=====================
Identify features that leak information from the future or target variable

Critical for win probability model - must only use features available BEFORE bid outcome

Author: Bid Recommendation System
Date: 2026-01-08
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
from config.model_config import MODELS_DIR

# Load current selected features
with open(MODELS_DIR / "lightgbm_metadata_feature_selected.json", 'r') as f:
    metadata = json.load(f)
    CURRENT_FEATURES = metadata['selected_features']

print("=" * 80)
print("FEATURE LEAKAGE AUDIT")
print("=" * 80)
print(f"Auditing {len(CURRENT_FEATURES)} features\n")

# Categorize features by leakage risk
leaky_features = []
suspicious_features = []
safe_features = []

feature_audit = {
    # LEAKY - These are calculated FROM bid outcomes (include current bid)
    'segment_win_rate': {
        'status': 'LEAKY',
        'reason': 'Calculated from historical wins INCLUDING current bid',
        'available_at_prediction': False
    },
    'office_win_rate': {
        'status': 'LEAKY',
        'reason': 'Calculated from historical wins INCLUDING current bid',
        'available_at_prediction': False
    },
    'client_win_rate': {
        'status': 'LEAKY',
        'reason': 'Calculated from historical wins INCLUDING current bid',
        'available_at_prediction': False
    },
    'propertytype_win_rate': {
        'status': 'LEAKY',
        'reason': 'Calculated from historical wins INCLUDING current bid',
        'available_at_prediction': False
    },
    'state_win_rate': {
        'status': 'LEAKY',
        'reason': 'Calculated from historical wins INCLUDING current bid',
        'available_at_prediction': False
    },

    # SUSPICIOUS - May include target information indirectly
    'rolling_avg_fee_segment': {
        'status': 'SUSPICIOUS',
        'reason': 'Uses historical BidFee (target variable) but with .shift()',
        'available_at_prediction': True
    },

    # SAFE - Available before bid outcome
    'segment_avg_fee': {
        'status': 'SAFE',
        'reason': 'Historical average fee (not win rate), leave-one-out',
        'available_at_prediction': True
    },
    'state_avg_fee': {
        'status': 'SAFE',
        'reason': 'Historical average fee, leave-one-out',
        'available_at_prediction': True
    },
    'segment_std_fee': {
        'status': 'SAFE',
        'reason': 'Historical fee volatility, leave-one-out',
        'available_at_prediction': True
    },
    'client_avg_fee': {
        'status': 'SAFE',
        'reason': 'Client historical average, leave-one-out',
        'available_at_prediction': True
    },
    'office_avg_fee': {
        'status': 'SAFE',
        'reason': 'Office historical average, leave-one-out',
        'available_at_prediction': True
    },
    'TargetTime': {
        'status': 'SAFE',
        'reason': 'Days to complete - known at bid time',
        'available_at_prediction': True
    },
    'client_std_fee': {
        'status': 'SAFE',
        'reason': 'Client fee volatility, leave-one-out',
        'available_at_prediction': True
    },
    'BusinessSegment_frequency': {
        'status': 'SAFE',
        'reason': 'Static - how common this segment is',
        'available_at_prediction': True
    },
    'propertytype_avg_fee': {
        'status': 'SAFE',
        'reason': 'Property type historical average, leave-one-out',
        'available_at_prediction': True
    },
    'PropertyState_frequency': {
        'status': 'SAFE',
        'reason': 'Static - how common this state is',
        'available_at_prediction': True
    },
}

print("AUDIT RESULTS:\n")

print("🔴 LEAKY FEATURES (Remove immediately):")
for feature in CURRENT_FEATURES:
    if feature in feature_audit and feature_audit[feature]['status'] == 'LEAKY':
        leaky_features.append(feature)
        print(f"  ❌ {feature}")
        print(f"     Reason: {feature_audit[feature]['reason']}\n")

print(f"\n⚠️  SUSPICIOUS FEATURES (Review carefully):")
for feature in CURRENT_FEATURES:
    if feature in feature_audit and feature_audit[feature]['status'] == 'SUSPICIOUS':
        suspicious_features.append(feature)
        print(f"  ⚠️  {feature}")
        print(f"     Reason: {feature_audit[feature]['reason']}\n")

print(f"\n✅ SAFE FEATURES (Keep):")
for feature in CURRENT_FEATURES:
    if feature in feature_audit and feature_audit[feature]['status'] == 'SAFE':
        safe_features.append(feature)
        print(f"  ✓ {feature}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total features: {len(CURRENT_FEATURES)}")
print(f"Leaky features: {len(leaky_features)} (MUST REMOVE)")
print(f"Suspicious features: {len(suspicious_features)} (REVIEW)")
print(f"Safe features: {len(safe_features)}")

# Save non-leaky features
safe_features_for_classification = [f for f in safe_features]
# Keep suspicious feature for now (it uses shift, so technically ok)
safe_features_for_classification.append('rolling_avg_fee_segment')

print(f"\nRECOMMENDED FEATURES FOR WIN PROBABILITY MODEL:")
print(f"  Count: {len(safe_features_for_classification)}")
print(f"  Features: {safe_features_for_classification}")

# Save to file
output = {
    "audit_date": "2026-01-08",
    "original_features": CURRENT_FEATURES,
    "leaky_features": leaky_features,
    "suspicious_features": suspicious_features,
    "safe_features": safe_features,
    "recommended_for_classification": safe_features_for_classification,
    "feature_audit_details": feature_audit
}

from datetime import datetime
audit_path = Path("outputs/reports/feature_leakage_audit.json")
with open(audit_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Audit results saved: {audit_path}")
print("\n⚠️  CRITICAL: Win probability model has DATA LEAKAGE!")
print("   Current accuracy (99.97%) is artificially inflated")
print("   Must retrain with safe features only\n")
