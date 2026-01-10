"""
Dual Feature Selection - Regression vs Classification
======================================================
Select optimal features for BOTH tasks independently:
- Regression: Bid Fee prediction (RMSE/MAE optimization)
- Classification: Win Probability (AUC/F1 optimization)

Comprehensive comparison showing:
- Which features work best for each task
- Feature overlap analysis
- Task-specific importance rankings

Author: Bid Recommendation System
Date: 2026-01-09
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from datetime import datetime
import warnings
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, f1_score, accuracy_score
)

from config.model_config import (
    FEATURES_DATA, DATE_COLUMN,
    MODELS_DIR, REPORTS_DIR, RANDOM_SEED
)

warnings.filterwarnings('ignore')

print("=" * 80)
print("DUAL FEATURE SELECTION: REGRESSION VS CLASSIFICATION")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Load data with all engineered features
df = pd.read_csv(FEATURES_DATA)
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

# Filter to recent data
recent_cutoff = pd.Timestamp('2023-01-01')
df_recent = df[df[DATE_COLUMN] >= recent_cutoff].copy()

print(f"✓ Data loaded: {len(df_recent):,} rows (2023-2025)")
print(f"  Total columns: {len(df_recent.columns)}")

# Identify feature columns (exclude ID, target, and categorical text columns)
exclude_cols = [
    DATE_COLUMN, 'BidId', 'BidFileNumber', 'BidName', 'Bid_DueDate',
    'BidStatusName', 'Bid_JobPurpose', 'Bid_Deliverable',
    'Market', 'Submarket', 'BusinessSegment', 'BusinessSegmentDetail',
    'Bid_Property_Type', 'Bid_SubProperty_Type', 'Bid_SpecificUseProperty_Type',
    'PropertyId', 'PropertyName', 'PropertyType', 'SubType',
    'PropertyCity', 'PropertyState', 'AddressDisplayCalc',
    'GrossBuildingAreaRange', 'YearBuiltRange',
    'OfficeId', 'OfficeCode', 'OfficeCompanyName', 'OfficeLocation',
    'JobId', 'JobName', 'JobStatus', 'JobType', 'AppraisalFileType',
    'BidCompanyName', 'BidCompanyType',
    'MarketOrientation',  # Categorical
    'BidFee', 'Won',  # Targets
    'BidFee_Original', 'TargetTime_Original'  # Original versions
]

feature_cols = [col for col in df_recent.columns if col not in exclude_cols]

# Further filter to only numeric columns
numeric_cols = df_recent[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
feature_cols = numeric_cols

print(f"  Available features: {len(feature_cols)} (numeric only)\n")

# Prepare data
X = df_recent[feature_cols].fillna(0).values
y_regression = df_recent['BidFee'].values
y_classification = df_recent['Won'].values

# 80/20 split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_reg_train, y_reg_test = y_regression[:split_idx], y_regression[split_idx:]
y_clf_train, y_clf_test = y_classification[:split_idx], y_classification[split_idx:]

print(f"Train: {len(X_train):,} samples")
print(f"Test: {len(X_test):,} samples\n")

# ============================================================================
# TASK 1: REGRESSION (Bid Fee Prediction)
# ============================================================================
print("=" * 80)
print("TASK 1: REGRESSION (BID FEE PREDICTION)")
print("=" * 80)

print("\nTraining LightGBM regressor on all features...")
reg_train_data = lgb.Dataset(X_train, label=y_reg_train, feature_name=feature_cols)

reg_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'random_state': RANDOM_SEED,
    'verbose': -1
}

reg_model = lgb.train(
    reg_params,
    reg_train_data,
    num_boost_round=200
)

# Get feature importance for regression
reg_importance = reg_model.feature_importance(importance_type='gain')
reg_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': reg_importance,
    'importance_pct': (reg_importance / reg_importance.sum() * 100)
}).sort_values('importance', ascending=False)

# Evaluate regression model
y_reg_pred_train = reg_model.predict(X_train)
y_reg_pred_test = reg_model.predict(X_test)

reg_train_rmse = np.sqrt(mean_squared_error(y_reg_train, y_reg_pred_train))
reg_test_rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred_test))
reg_test_r2 = r2_score(y_reg_test, y_reg_pred_test)

print(f"\nREGRESSION BASELINE (all {len(feature_cols)} features):")
print(f"  Train RMSE: ${reg_train_rmse:,.2f}")
print(f"  Test RMSE: ${reg_test_rmse:,.2f}")
print(f"  Test R²: {reg_test_r2:.4f}")
print(f"  Overfitting ratio: {reg_test_rmse / reg_train_rmse:.2f}x")

print(f"\nTop 20 features for REGRESSION:")
for i, row in enumerate(reg_importance_df.head(20).itertuples(), 1):
    print(f"  {i:2d}. {row.feature:40s} {row.importance_pct:6.2f}%")

# ============================================================================
# TASK 2: CLASSIFICATION (Win Probability)
# ============================================================================
print("\n" + "=" * 80)
print("TASK 2: CLASSIFICATION (WIN PROBABILITY)")
print("=" * 80)

print("\nTraining LightGBM classifier on all features...")
clf_train_data = lgb.Dataset(X_train, label=y_clf_train, feature_name=feature_cols)

clf_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'random_state': RANDOM_SEED,
    'verbose': -1
}

clf_model = lgb.train(
    clf_params,
    clf_train_data,
    num_boost_round=200
)

# Get feature importance for classification
clf_importance = clf_model.feature_importance(importance_type='gain')
clf_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': clf_importance,
    'importance_pct': (clf_importance / clf_importance.sum() * 100)
}).sort_values('importance', ascending=False)

# Evaluate classification model
y_clf_pred_train_proba = clf_model.predict(X_train)
y_clf_pred_test_proba = clf_model.predict(X_test)
y_clf_pred_test = (y_clf_pred_test_proba >= 0.5).astype(int)

clf_test_auc = roc_auc_score(y_clf_test, y_clf_pred_test_proba)
clf_test_f1 = f1_score(y_clf_test, y_clf_pred_test)
clf_test_acc = accuracy_score(y_clf_test, y_clf_pred_test)

print(f"\nCLASSIFICATION BASELINE (all {len(feature_cols)} features):")
print(f"  Test AUC: {clf_test_auc:.4f}")
print(f"  Test F1: {clf_test_f1:.4f}")
print(f"  Test Accuracy: {clf_test_acc:.4f}")

print(f"\nTop 20 features for CLASSIFICATION:")
for i, row in enumerate(clf_importance_df.head(20).itertuples(), 1):
    print(f"  {i:2d}. {row.feature:40s} {row.importance_pct:6.2f}%")

# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("COMPARATIVE ANALYSIS: REGRESSION VS CLASSIFICATION")
print("=" * 80)

# Merge importance rankings
comparison_df = reg_importance_df[['feature', 'importance_pct']].rename(
    columns={'importance_pct': 'reg_importance_pct'}
).merge(
    clf_importance_df[['feature', 'importance_pct']].rename(
        columns={'importance_pct': 'clf_importance_pct'}
    ),
    on='feature'
)

comparison_df['avg_importance'] = (
    comparison_df['reg_importance_pct'] + comparison_df['clf_importance_pct']
) / 2
comparison_df['importance_diff'] = abs(
    comparison_df['reg_importance_pct'] - comparison_df['clf_importance_pct']
)

comparison_df = comparison_df.sort_values('avg_importance', ascending=False)

print("\nTOP 15 FEATURES (by average importance):")
print(f"{'Rank':<6} {'Feature':<40} {'Regression %':<15} {'Classification %':<18} {'Avg %'}")
print("-" * 100)
for i, row in enumerate(comparison_df.head(15).itertuples(), 1):
    print(f"{i:<6} {row.feature:<40} {row.reg_importance_pct:<15.2f} {row.clf_importance_pct:<18.2f} {row.avg_importance:.2f}")

print("\nMOST DIFFERENT FEATURES (regression-specific vs classification-specific):")
top_diff = comparison_df.nlargest(10, 'importance_diff')
for i, row in enumerate(top_diff.itertuples(), 1):
    if row.reg_importance_pct > row.clf_importance_pct:
        task = "REGRESSION-specific"
    else:
        task = "CLASSIFICATION-specific"
    print(f"  {i:2d}. {row.feature:40s} - {task} (diff: {row.importance_diff:.2f}%)")

# ============================================================================
# FEATURE SELECTION
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE SELECTION FOR EACH TASK")
print("=" * 80)

# Select top features for each task (top 95% cumulative importance)
def select_features_by_cumulative_importance(importance_df, threshold=95):
    """Select features contributing to threshold% of total importance"""
    importance_df = importance_df.sort_values('importance', ascending=False).copy()
    importance_df['cumulative_pct'] = importance_df['importance_pct'].cumsum()
    selected = importance_df[importance_df['cumulative_pct'] <= threshold]['feature'].tolist()

    # Ensure at least 10 features
    if len(selected) < 10:
        selected = importance_df.head(10)['feature'].tolist()

    return selected, importance_df[importance_df['feature'].isin(selected)]

# Select features for regression
reg_selected_features, reg_selected_df = select_features_by_cumulative_importance(reg_importance_df, 95)
print(f"\nREGRESSION: Selected {len(reg_selected_features)} features (95% cumulative importance)")
print("Features:")
for i, row in enumerate(reg_selected_df.itertuples(), 1):
    print(f"  {i:2d}. {row.feature:40s} {row.importance_pct:6.2f}% (cumulative: {row.cumulative_pct:.1f}%)")

# Select features for classification
clf_selected_features, clf_selected_df = select_features_by_cumulative_importance(clf_importance_df, 95)
print(f"\nCLASSIFICATION: Selected {len(clf_selected_features)} features (95% cumulative importance)")
print("Features:")
for i, row in enumerate(clf_selected_df.itertuples(), 1):
    print(f"  {i:2d}. {row.feature:40s} {row.importance_pct:6.2f}% (cumulative: {row.cumulative_pct:.1f}%)")

# ============================================================================
# OVERLAP ANALYSIS (Venn Diagram Data)
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE OVERLAP ANALYSIS")
print("=" * 80)

reg_set = set(reg_selected_features)
clf_set = set(clf_selected_features)

shared_features = reg_set & clf_set
reg_only_features = reg_set - clf_set
clf_only_features = clf_set - reg_set

print(f"\nVenn Diagram Data:")
print(f"  Shared features: {len(shared_features)}")
print(f"  Regression-only: {len(reg_only_features)}")
print(f"  Classification-only: {len(clf_only_features)}")
print(f"  Total unique: {len(reg_set | clf_set)}")

if shared_features:
    print(f"\nShared features ({len(shared_features)}):")
    for i, feat in enumerate(sorted(shared_features), 1):
        print(f"  {i}. {feat}")

if reg_only_features:
    print(f"\nRegression-only features ({len(reg_only_features)}):")
    for i, feat in enumerate(sorted(reg_only_features), 1):
        print(f"  {i}. {feat}")

if clf_only_features:
    print(f"\nClassification-only features ({len(clf_only_features)}):")
    for i, feat in enumerate(sorted(clf_only_features), 1):
        print(f"  {i}. {feat}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save regression features
reg_features_path = REPORTS_DIR / "regression_features.json"
with open(reg_features_path, 'w') as f:
    json.dump({
        "task": "regression",
        "target": "BidFee",
        "num_features": len(reg_selected_features),
        "features": reg_selected_features,
        "feature_importance": reg_selected_df.to_dict('records'),
        "baseline_metrics": {
            "train_rmse": float(reg_train_rmse),
            "test_rmse": float(reg_test_rmse),
            "test_r2": float(reg_test_r2)
        },
        "selection_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }, f, indent=2)
print(f"✓ Regression features saved: {reg_features_path}")

# Save classification features
clf_features_path = REPORTS_DIR / "classification_features.json"
with open(clf_features_path, 'w') as f:
    json.dump({
        "task": "classification",
        "target": "Won",
        "num_features": len(clf_selected_features),
        "features": clf_selected_features,
        "feature_importance": clf_selected_df.to_dict('records'),
        "baseline_metrics": {
            "test_auc": float(clf_test_auc),
            "test_f1": float(clf_test_f1),
            "test_accuracy": float(clf_test_acc)
        },
        "selection_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }, f, indent=2)
print(f"✓ Classification features saved: {clf_features_path}")

# Save comprehensive comparison
comparison_path = REPORTS_DIR / "feature_comparison_report.json"
with open(comparison_path, 'w') as f:
    json.dump({
        "comparison_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "total_available_features": len(feature_cols),
        "regression": {
            "selected_features": len(reg_selected_features),
            "baseline_test_rmse": float(reg_test_rmse),
            "baseline_test_r2": float(reg_test_r2)
        },
        "classification": {
            "selected_features": len(clf_selected_features),
            "baseline_test_auc": float(clf_test_auc),
            "baseline_test_f1": float(clf_test_f1)
        },
        "overlap_analysis": {
            "shared_features": len(shared_features),
            "regression_only": len(reg_only_features),
            "classification_only": len(clf_only_features),
            "total_unique": len(reg_set | clf_set),
            "shared_feature_list": sorted(list(shared_features)),
            "regression_only_list": sorted(list(reg_only_features)),
            "classification_only_list": sorted(list(clf_only_features))
        },
        "full_comparison": comparison_df.to_dict('records')
    }, f, indent=2)
print(f"✓ Comparison report saved: {comparison_path}")

print("\n" + "=" * 80)
print("DUAL FEATURE SELECTION COMPLETE!")
print("=" * 80)
print(f"✓ Regression features: {len(reg_selected_features)}")
print(f"✓ Classification features: {len(clf_selected_features)}")
print(f"✓ Shared features: {len(shared_features)}")
print(f"✓ Total unique features: {len(reg_set | clf_set)}")
print(f"\nPhase 2: Dual feature selection complete!\n")
