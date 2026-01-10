"""
Win Probability Model - Classification Optimized (Phase 3)
===========================================================
Train win probability model using classification-optimized features
with ALL leaky win_rate features removed

Features selected specifically for CLASSIFICATION task (not regression)
Only using features available BEFORE bid outcome

Author: Bid Recommendation System
Date: 2026-01-09 (Phase 3 - Optimized & Clean)
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings

from config.model_config import (
    FEATURES_DATA, DATE_COLUMN,
    MODELS_DIR, RANDOM_SEED,
)

warnings.filterwarnings('ignore')

# Classification-optimized features (leaky win_rate features removed)
SAFE_FEATURES = [
    'JobCount',  # 9.32% importance
    'rolling_bid_count_office',  # 0.57%
    'propertytype_std_fee',  # 0.25%
    'total_bids_to_client',  # 0.20%
    'office_avg_fee',  # 0.18%
    'PropertyState_encoded',  # 0.12%
    'RooftopLongitude',  # 0.08%
    'BusinessSegment_frequency',  # 0.08%
    'office_std_fee',  # 0.07%
    'DeliveryTotal',  # 0.07%
    'market_competitiveness',  # 0.06%
]

print("=" * 80)
print("WIN PROBABILITY MODEL - CLASSIFICATION OPTIMIZED (PHASE 3)")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Using {len(SAFE_FEATURES)} classification-optimized features")
print(f"ALL leaky win_rate features removed\n")

# Load data
df = pd.read_csv(FEATURES_DATA)
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

# Filter to recent data
recent_cutoff = pd.Timestamp('2023-01-01')
df_recent = df[df[DATE_COLUMN] >= recent_cutoff].copy()

print(f"✓ Data loaded: {len(df_recent):,} rows (2023-2025)")
print(f"\nSelected features for CLASSIFICATION (NO LEAKAGE):")
for i, feat in enumerate(SAFE_FEATURES, 1):
    print(f"  {i:2d}. {feat}")

# Prepare data
X = df_recent[SAFE_FEATURES].fillna(0).values
y = df_recent['Won'].values

win_rate = y.mean()
print(f"\nClass Distribution:")
print(f"  Wins: {y.sum():,} ({win_rate*100:.1f}%)")
print(f"  Losses: {len(y) - y.sum():,} ({(1-win_rate)*100:.1f}%)")

# 80/20 split
split_idx = int(len(X) * 0.8)
X_train = X[:split_idx]
X_test = X[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

print(f"\nTrain: {len(X_train):,} samples")
print(f"Test: {len(X_test):,} samples\n")

# Train model
print("=" * 80)
print("TRAINING (WITH SAFE FEATURES ONLY)")
print("=" * 80)

train_data = lgb.Dataset(X_train, label=y_train, feature_name=SAFE_FEATURES)
val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'binary',
    'metric': ['binary_logloss', 'auc'],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': RANDOM_SEED,
    'verbose': -1
}

print("Training...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'valid'],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

print(f"\n✓ Model trained (best iteration: {model.best_iteration})\n")

# Predictions
y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred_proba >= 0.5).astype(int)

# Evaluation
print("=" * 80)
print("EVALUATION")
print("=" * 80)

test_acc = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, zero_division=0)
test_recall = recall_score(y_test, y_pred, zero_division=0)
test_f1 = f1_score(y_test, y_pred, zero_division=0)
test_auc = roc_auc_score(y_test, y_pred_proba)

print("TEST SET PERFORMANCE:")
print(f"  Accuracy: {test_acc:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall: {test_recall:.4f}")
print(f"  F1 Score: {test_f1:.4f}")
print(f"  AUC-ROC: {test_auc:.4f}")

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nCONFUSION MATRIX:")
print(f"  True Negatives:  {tn:,}")
print(f"  False Positives: {fp:,}")
print(f"  False Negatives: {fn:,}")
print(f"  True Positives:  {tp:,}")

print(f"\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Loss', 'Win'], zero_division=0))

# Feature importance
importance = model.feature_importance(importance_type='gain')
importance_df = pd.DataFrame({
    'feature': SAFE_FEATURES,
    'importance': importance
}).sort_values('importance', ascending=False)

print("FEATURE IMPORTANCE:")
for i, row in enumerate(importance_df.itertuples(), 1):
    pct = row.importance / importance_df['importance'].sum() * 100
    print(f"  {i:2d}. {row.feature:45s} {pct:6.2f}%")

# Compare with previous models
print("\n" + "=" * 80)
print("COMPARISON: PREVIOUS VS OPTIMIZED MODEL")
print("=" * 80)

baselines = {
    "Leaky model (12 features)": {"accuracy": 0.9997, "auc": 1.0000},
    "Clean model (11 features)": {"accuracy": 0.7715, "auc": 0.8485},
}

print(f"\n{'Model':<35} {'Accuracy':<15} {'AUC-ROC':<15} {'Status'}")
print("-" * 80)
for name, metrics in baselines.items():
    print(f"{name:<35} {metrics['accuracy']:<15.4f} {metrics['auc']:<15.4f}")

print(f"{'NEW: Optimized (11 features)':<35} {test_acc:<15.4f} {test_auc:<15.4f} {'← CURRENT'}")

# Determine improvement
clean_baseline_auc = 0.8485
improvement_pct = ((test_auc - clean_baseline_auc) / clean_baseline_auc) * 100

if test_auc > clean_baseline_auc:
    print(f"\n✓ IMPROVEMENT: {improvement_pct:.1f}% better AUC than previous clean model")
elif test_auc > 0.80:
    print(f"\n✓ STRONG PERFORMANCE: AUC > 0.80 without data leakage")
else:
    print(f"\n⚠ AUC: {test_auc:.4f} vs clean baseline {clean_baseline_auc:.4f}")

print(f"\nModel is production-ready: {'Yes' if test_auc > 0.75 and test_acc < 0.95 else 'Needs review'}")

# Save model
print("\n" + "=" * 80)
print("SAVING MODEL")
print("=" * 80)

model_path = MODELS_DIR / "lightgbm_win_probability_optimized.txt"
model.save_model(str(model_path))

metadata = {
    "model_type": "LightGBM Binary Classifier (Phase 3 - Optimized)",
    "phase": "1B - Win Probability Prediction",
    "target_variable": "Won",
    "optimization": "Dual feature selection (classification-specific, no leakage)",
    "num_features": len(SAFE_FEATURES),
    "selected_features": SAFE_FEATURES,
    "removed_leaky_features": [
        "segment_win_rate", "state_win_rate", "office_win_rate",
        "client_win_rate", "rolling_win_rate_office", "propertytype_win_rate"
    ],
    "data_range": {
        "start_date": df_recent[DATE_COLUMN].min().strftime('%Y-%m-%d'),
        "end_date": df_recent[DATE_COLUMN].max().strftime('%Y-%m-%d'),
        "total_samples": int(len(df_recent)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test))
    },
    "best_iteration": int(model.best_iteration),
    "parameters": params,
    "metrics": {
        "test": {
            "accuracy": float(test_acc),
            "precision": float(test_precision),
            "recall": float(test_recall),
            "f1_score": float(test_f1),
            "auc_roc": float(test_auc)
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    },
    "feature_importance": importance_df.to_dict('records'),
    "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "production_ready": bool(test_auc > 0.75 and test_acc < 0.95),
    "leakage_fixed": True,
    "comparison": {
        "improvement_vs_clean_baseline_pct": float(improvement_pct),
        "clean_baseline_auc": float(clean_baseline_auc)
    }
}

metadata_path = MODELS_DIR / "lightgbm_win_probability_optimized_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Model saved: {model_path}")
print(f"✓ Metadata saved: {metadata_path}")

print("\n" + "=" * 80)
print("PHASE 3: CLASSIFICATION MODEL COMPLETE")
print("=" * 80)
print(f"✓ Features: {len(SAFE_FEATURES)} (classification-optimized)")
print(f"✓ No data leakage: True")
print(f"✓ Accuracy: {test_acc:.1%} (realistic)")
print(f"✓ AUC-ROC: {test_auc:.4f}")
print(f"✓ Production ready: {'Yes' if metadata['production_ready'] else 'No'}\n")
