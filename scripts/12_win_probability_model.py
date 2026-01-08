"""
Phase 1B: Win Probability Classification Model
===============================================
Build binary classification model to predict probability of winning a bid

Uses the same 12 features from Phase 1A for consistency
Combines with bid fee prediction for optimal bid recommendation

Target: Predict Win (1) vs Loss (0)
Output: Probability score 0-1

Author: Bid Recommendation System
Date: 2026-01-08
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
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from config.model_config import (
    FEATURES_DATA, DATE_COLUMN,
    MODELS_DIR, REPORTS_DIR, FIGURES_DIR, RANDOM_SEED,
)

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# Load selected features from Phase 1A
with open(MODELS_DIR / "lightgbm_metadata_feature_selected.json", 'r') as f:
    metadata = json.load(f)
    SELECTED_FEATURES = metadata['selected_features']

print("=" * 80)
print("PHASE 1B: WIN PROBABILITY CLASSIFICATION MODEL")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Features: {len(SELECTED_FEATURES)} (same as Phase 1A)")
print(f"Target: Win (1) vs Loss (0)\n")

# Load data
print("Loading data...")
df = pd.read_csv(FEATURES_DATA)
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

# Filter to recent data (2023-2025) - best performing time range
recent_cutoff = pd.Timestamp('2023-01-01')
df_recent = df[df[DATE_COLUMN] >= recent_cutoff].copy()

print(f"✓ Data loaded: {len(df_recent):,} rows (2023-2025)")
print(f"  Date range: {df_recent[DATE_COLUMN].min()} to {df_recent[DATE_COLUMN].max()}")

# Prepare features and target
X = df_recent[SELECTED_FEATURES].fillna(0).values
y = df_recent['Won'].values  # Binary: 1 = Won, 0 = Lost

# Class distribution
win_rate = y.mean()
print(f"\nClass Distribution:")
print(f"  Wins (1): {y.sum():,} ({win_rate*100:.1f}%)")
print(f"  Losses (0): {len(y) - y.sum():,} ({(1-win_rate)*100:.1f}%)")

if win_rate < 0.1 or win_rate > 0.9:
    print(f"  ⚠️  Imbalanced dataset - will use class weights")
    is_balanced = False
else:
    print(f"  ✓ Reasonably balanced")
    is_balanced = True

# 80/20 split
split_idx = int(len(X) * 0.8)
X_train = X[:split_idx]
X_test = X[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

print(f"\nTrain set: {len(X_train):,} samples (Win rate: {y_train.mean()*100:.1f}%)")
print(f"Test set: {len(X_test):,} samples (Win rate: {y_test.mean()*100:.1f}%)\n")

# Train LightGBM classifier
print("=" * 80)
print("TRAINING LIGHTGBM CLASSIFIER")
print("=" * 80)

train_data = lgb.Dataset(X_train, label=y_train, feature_name=SELECTED_FEATURES)
val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Calculate class weights if imbalanced
if not is_balanced:
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    print(f"Imbalanced classes - using scale_pos_weight: {scale_pos_weight:.2f}")
else:
    scale_pos_weight = 1.0

params = {
    'objective': 'binary',
    'metric': ['binary_logloss', 'auc'],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'scale_pos_weight': scale_pos_weight,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': RANDOM_SEED,
    'verbose': -1
}

print("Training in progress...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'valid'],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

print(f"\n✓ Model trained")
print(f"  Best iteration: {model.best_iteration}")

# Predict probabilities
y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
y_pred_train_proba = model.predict(X_train, num_iteration=model.best_iteration)

# Convert probabilities to binary predictions (threshold = 0.5)
y_pred = (y_pred_proba >= 0.5).astype(int)
y_pred_train = (y_pred_train_proba >= 0.5).astype(int)

# Evaluation metrics
print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

# Train set metrics
train_acc = accuracy_score(y_train, y_pred_train)
train_auc = roc_auc_score(y_train, y_pred_train_proba)

# Test set metrics
test_acc = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, zero_division=0)
test_recall = recall_score(y_test, y_pred, zero_division=0)
test_f1 = f1_score(y_test, y_pred, zero_division=0)
test_auc = roc_auc_score(y_test, y_pred_proba)
test_ap = average_precision_score(y_test, y_pred_proba)

print("TRAIN SET:")
print(f"  Accuracy: {train_acc:.4f}")
print(f"  AUC-ROC: {train_auc:.4f}")

print("\nTEST SET:")
print(f"  Accuracy: {test_acc:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall: {test_recall:.4f}")
print(f"  F1 Score: {test_f1:.4f}")
print(f"  AUC-ROC: {test_auc:.4f}")
print(f"  Average Precision: {test_ap:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nCONFUSION MATRIX:")
print(f"  True Negatives:  {tn:,}")
print(f"  False Positives: {fp:,}")
print(f"  False Negatives: {fn:,}")
print(f"  True Positives:  {tp:,}")

# Detailed classification report
print(f"\nDETAILED CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Loss', 'Win'], zero_division=0))

# Probability calibration check
print("PROBABILITY CALIBRATION:")
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    precision_thresh = precision_score(y_test, y_pred_thresh, zero_division=0)
    recall_thresh = recall_score(y_test, y_pred_thresh, zero_division=0)
    print(f"  Threshold {threshold:.1f}: Precision={precision_thresh:.3f}, Recall={recall_thresh:.3f}")

# Feature importance
feature_importance = model.feature_importance(importance_type='gain')
importance_df = pd.DataFrame({
    'feature': SELECTED_FEATURES,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\nTOP 10 FEATURES FOR WIN PREDICTION:")
for i, row in enumerate(importance_df.head(10).itertuples(), 1):
    pct = row.importance / importance_df['importance'].sum() * 100
    print(f"  {i}. {row.feature}: {pct:.1f}%")

# Create visualizations
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# 1. Confusion Matrix Heatmap
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Predicted Loss', 'Predicted Win'],
            yticklabels=['Actual Loss', 'Actual Win'])
axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_ylabel('True Label')
axes[0, 0].set_xlabel('Predicted Label')

# 2. Probability Distribution
axes[0, 1].hist(y_pred_proba[y_test == 0], bins=50, alpha=0.5, label='Losses', color='red')
axes[0, 1].hist(y_pred_proba[y_test == 1], bins=50, alpha=0.5, label='Wins', color='green')
axes[0, 1].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
axes[0, 1].set_xlabel('Predicted Probability')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Predicted Probability Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Feature Importance
top_features = importance_df.head(10)
axes[1, 0].barh(range(len(top_features)), top_features['importance'])
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['feature'])
axes[1, 0].set_xlabel('Importance (Gain)')
axes[1, 0].set_title('Top 10 Features for Win Prediction')
axes[1, 0].grid(True, alpha=0.3)

# 4. Precision-Recall vs Threshold
thresholds = np.linspace(0, 1, 100)
precisions = []
recalls = []
for thresh in thresholds:
    y_pred_t = (y_pred_proba >= thresh).astype(int)
    precisions.append(precision_score(y_test, y_pred_t, zero_division=0))
    recalls.append(recall_score(y_test, y_pred_t, zero_division=0))

axes[1, 1].plot(thresholds, precisions, label='Precision', linewidth=2)
axes[1, 1].plot(thresholds, recalls, label='Recall', linewidth=2)
axes[1, 1].axvline(x=0.5, color='gray', linestyle='--', label='Default (0.5)')
axes[1, 1].set_xlabel('Probability Threshold')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Precision and Recall vs Threshold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
viz_path = FIGURES_DIR / "win_probability_analysis.png"
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Visualizations saved: {viz_path}")

# Save model
print("\n" + "=" * 80)
print("SAVING MODEL")
print("=" * 80)

model_path = MODELS_DIR / "lightgbm_win_probability_model.txt"
model.save_model(str(model_path))

metadata_save = {
    "model_type": "LightGBM Binary Classifier",
    "phase": "1B - Win Probability Prediction",
    "target_variable": "Won",
    "num_features": len(SELECTED_FEATURES),
    "selected_features": SELECTED_FEATURES,
    "data_range": {
        "start_date": df_recent[DATE_COLUMN].min().strftime('%Y-%m-%d'),
        "end_date": df_recent[DATE_COLUMN].max().strftime('%Y-%m-%d'),
        "total_samples": int(len(df_recent)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test))
    },
    "class_distribution": {
        "train_win_rate": float(y_train.mean()),
        "test_win_rate": float(y_test.mean()),
        "is_balanced": is_balanced,
        "scale_pos_weight": float(scale_pos_weight)
    },
    "best_iteration": int(model.best_iteration),
    "parameters": {k: float(v) if isinstance(v, (float, np.floating)) else v
                  for k, v in params.items()},
    "metrics": {
        "train": {
            "accuracy": float(train_acc),
            "auc_roc": float(train_auc)
        },
        "test": {
            "accuracy": float(test_acc),
            "precision": float(test_precision),
            "recall": float(test_recall),
            "f1_score": float(test_f1),
            "auc_roc": float(test_auc),
            "average_precision": float(test_ap)
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    },
    "feature_importance": importance_df.head(15).to_dict('records'),
    "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
}

metadata_path = MODELS_DIR / "lightgbm_win_probability_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata_save, f, indent=2)

print(f"✓ Model saved: {model_path}")
print(f"✓ Metadata saved: {metadata_path}")

print("\n" + "=" * 80)
print("PHASE 1B COMPLETE!")
print("=" * 80)
print(f"✓ Win probability model trained")
print(f"✓ Test AUC-ROC: {test_auc:.4f}")
print(f"✓ Test F1 Score: {test_f1:.4f}")
print(f"✓ Test Accuracy: {test_acc:.4f}")
print(f"\nReady to combine with Phase 1A (bid fee prediction) for bid optimization!\n")
