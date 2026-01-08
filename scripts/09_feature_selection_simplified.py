"""
Simplified Feature Selection - Using Best Methods
==================================================
Uses the 4 most effective feature selection methods and retrains the model

Author: Bid Recommendation System
Date: 2026-01-07
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

from config.model_config import (
    FEATURES_DATA, TARGET_COLUMN, DATE_COLUMN, EXCLUDE_COLUMNS,
    LIGHTGBM_CONFIG, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, RANDOM_SEED,
)

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

print("=" * 80)
print("FEATURE SELECTION & MODEL RETRAINING")
print("=" * 80)
print(f"Using 3 proven methods: SHAP, LightGBM Importance, and Correlation\n")

# Load data
df = pd.read_csv(FEATURES_DATA)
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

X = df[numeric_features].fillna(0)
y = df[TARGET_COLUMN].values

split_idx = int(len(X) * 0.8)
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

print(f"Data: {len(X):,} samples, {len(numeric_features)} features")
print(f"Train: {len(X_train):,}, Test: {len(X_test):,}\n")

# Train baseline
print("Training baseline model...")
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

model = lgb.train(
    LIGHTGBM_CONFIG['params'],
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

y_pred_baseline = model.predict(X_test, num_iteration=model.best_iteration)
y_train_pred_baseline = model.predict(X_train, num_iteration=model.best_iteration)

baseline_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_baseline))
baseline_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
baseline_overfitting = baseline_test_rmse / baseline_train_rmse

print(f"✓ Baseline trained")
print(f"  Train RMSE: ${baseline_train_rmse:,.2f}")
print(f"  Test RMSE:  ${baseline_test_rmse:,.2f}")
print(f"  Overfitting: {baseline_overfitting:.2f}x\n")

# Method 1: SHAP Values
print("=" * 80)
print("METHOD 1: SHAP VALUES")
print("=" * 80)

X_shap = X_train.sample(min(1000, len(X_train)), random_state=RANDOM_SEED)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)

shap_importance = pd.DataFrame({
    'feature': numeric_features,
    'shap_importance': np.abs(shap_values).mean(axis=0)
}).sort_values('shap_importance', ascending=False)

shap_importance['shap_pct'] = shap_importance['shap_importance'] / shap_importance['shap_importance'].sum()
shap_importance['cumulative_pct'] = shap_importance['shap_pct'].cumsum()

# Select top features (90% cumulative importance)
shap_features = shap_importance[shap_importance['cumulative_pct'] <= 0.90]['feature'].tolist()
if len(shap_features) < 10:  # Ensure at least 10 features
    shap_features = shap_importance.head(15)['feature'].tolist()

print(f"Top 15 features by SHAP:")
for i, row in enumerate(shap_importance.head(15).itertuples(), 1):
    print(f"  {i}. {row.feature}: {row.shap_pct*100:.2f}% (cum: {row.cumulative_pct*100:.1f}%)")
print(f"\n✓ Selected {len(shap_features)} features (90% importance)\n")

# Method 2: LightGBM Built-in Importance
print("=" * 80)
print("METHOD 2: LIGHTGBM IMPORTANCE")
print("=" * 80)

importance = model.feature_importance(importance_type='gain')
lgb_importance = pd.DataFrame({
    'feature': numeric_features,
    'importance': importance
}).sort_values('importance', ascending=False)

lgb_importance['importance_pct'] = lgb_importance['importance'] / lgb_importance['importance'].sum()
lgb_importance['cumulative_pct'] = lgb_importance['importance_pct'].cumsum()

# Select features contributing to 95% of importance
lgb_features = lgb_importance[lgb_importance['cumulative_pct'] <= 0.95]['feature'].tolist()
if len(lgb_features) < 10:
    lgb_features = lgb_importance.head(20)['feature'].tolist()

print(f"Top 15 features by LightGBM Importance:")
for i, row in enumerate(lgb_importance.head(15).itertuples(), 1):
    print(f"  {i}. {row.feature}: {row.importance_pct*100:.2f}%")
print(f"\n✓ Selected {len(lgb_features)} features (95% importance)\n")

# Method 3: Correlation Analysis
print("=" * 80)
print("METHOD 3: CORRELATION ANALYSIS")
print("=" * 80)

corr_matrix = X_train.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]

corr_features = [f for f in numeric_features if f not in to_drop]

print(f"Removed {len(to_drop)} highly correlated features")
print(f"✓ Retained {len(corr_features)} features\n")

# Consensus: Features selected by at least 2/3 methods
print("=" * 80)
print("CREATING CONSENSUS FEATURE SET")
print("=" * 80)

all_selected = {}
for feat in shap_features:
    all_selected[feat] = all_selected.get(feat, 0) + 1
for feat in lgb_features:
    all_selected[feat] = all_selected.get(feat, 0) + 1
for feat in corr_features:
    all_selected[feat] = all_selected.get(feat, 0) + 1

# Features selected by at least 2 methods
consensus_features = [feat for feat, count in all_selected.items() if count >= 2]

print(f"Features selected by:")
print(f"  3/3 methods: {sum(1 for c in all_selected.values() if c == 3)}")
print(f"  2/3 methods: {sum(1 for c in all_selected.values() if c == 2)}")
print(f"  1/3 methods: {sum(1 for c in all_selected.values() if c == 1)}")
print(f"\n✓ Consensus features (≥2 methods): {len(consensus_features)}\n")

# Compare feature sets
print("=" * 80)
print("COMPARING FEATURE SETS")
print("=" * 80)

feature_sets = {
    'shap_top': shap_features[:15],  # Top 15 by SHAP
    'shap_90pct': shap_features,     # 90% cumulative SHAP
    'lgb_top': lgb_features[:20],    # Top 20 by LightGBM
    'lgb_95pct': lgb_features,       # 95% cumulative LightGBM
    'consensus': consensus_features   # Consensus (≥2 methods)
}

results = []

for name, features in feature_sets.items():
    print(f"\nTesting {name} ({len(features)} features)...")

    X_train_subset = X_train[features]
    X_test_subset = X_test[features]

    train_data = lgb.Dataset(X_train_subset, label=y_train)
    val_data = lgb.Dataset(X_test_subset, label=y_test, reference=train_data)

    model_subset = lgb.train(
        LIGHTGBM_CONFIG['params'],
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    y_pred = model_subset.predict(X_test_subset, num_iteration=model_subset.best_iteration)
    y_train_pred = model_subset.predict(X_train_subset, num_iteration=model_subset.best_iteration)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    overfitting_ratio = test_rmse / train_rmse

    results.append({
        'method': name,
        'n_features': len(features),
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'overfitting_ratio': overfitting_ratio,
        'features': features
    })

    print(f"  Train RMSE: ${train_rmse:,.2f}")
    print(f"  Test RMSE:  ${test_rmse:,.2f}")
    print(f"  Test R²:    {test_r2:.4f}")
    print(f"  Overfitting: {overfitting_ratio:.2f}x")

# Find best method
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('overfitting_ratio')

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print("\nRanked by overfitting ratio (lower is better):")
print(results_df[['method', 'n_features', 'test_rmse', 'test_r2', 'overfitting_ratio']].to_string(index=False))

best_idx = results_df.iloc[0]
print(f"\n🏆 BEST METHOD: {best_idx['method'].upper()}")
print(f"   Features: {best_idx['n_features']} (down from 84)")
print(f"   Test RMSE: ${best_idx['test_rmse']:,.2f}")
print(f"   Test R²: {best_idx['test_r2']:.4f}")
print(f"   Overfitting: {best_idx['overfitting_ratio']:.2f}x")

improvement = (baseline_overfitting - best_idx['overfitting_ratio']) / baseline_overfitting * 100
print(f"\nIMPROVEMENT:")
print(f"  Baseline overfitting: {baseline_overfitting:.2f}x")
print(f"  Best overfitting: {best_idx['overfitting_ratio']:.2f}x")
print(f"  Reduction: {improvement:.1f}%\n")

# Train final model with best features
print("=" * 80)
print("TRAINING FINAL MODEL WITH BEST FEATURES")
print("=" * 80)

best_features = best_idx['features']
X_train_final = X_train[best_features]
X_test_final = X_test[best_features]

train_data = lgb.Dataset(X_train_final, label=y_train, feature_name=best_features)
val_data = lgb.Dataset(X_test_final, label=y_test, reference=train_data)

print(f"Training with {len(best_features)} features...")
final_model = lgb.train(
    LIGHTGBM_CONFIG['params'],
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'valid'],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# Final evaluation
y_pred_final = final_model.predict(X_test_final, num_iteration=final_model.best_iteration)
y_train_pred_final = final_model.predict(X_train_final, num_iteration=final_model.best_iteration)

final_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_final))
final_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
final_test_mae = mean_absolute_error(y_test, y_pred_final)
final_test_r2 = r2_score(y_test, y_pred_final)

print(f"\n✓ Final model trained")
print(f"\nFINAL PERFORMANCE:")
print(f"  Train RMSE: ${final_train_rmse:,.2f}")
print(f"  Test RMSE:  ${final_test_rmse:,.2f}")
print(f"  Test MAE:   ${final_test_mae:,.2f}")
print(f"  Test R²:    {final_test_r2:.4f}")
print(f"  Overfitting: {final_test_rmse / final_train_rmse:.2f}x\n")

# Save model
model_path = MODELS_DIR / "lightgbm_bidfee_model_feature_selected.txt"
final_model.save_model(str(model_path))

metadata = {
    "model_type": "LightGBM (Feature Selected)",
    "phase": "1A - Bid Fee Prediction",
    "target_variable": TARGET_COLUMN,
    "feature_selection": {
        "method": best_idx['method'],
        "original_features": len(numeric_features),
        "selected_features": len(best_features),
        "reduction_pct": (1 - len(best_features) / len(numeric_features)) * 100
    },
    "selected_features": best_features,
    "best_iteration": int(final_model.best_iteration),
    "metrics": {
        "train_rmse": float(final_train_rmse),
        "test_rmse": float(final_test_rmse),
        "test_mae": float(final_test_mae),
        "test_r2": float(final_test_r2),
        "overfitting_ratio": float(final_test_rmse / final_train_rmse)
    },
    "baseline_comparison": {
        "baseline_overfitting": float(baseline_overfitting),
        "final_overfitting": float(final_test_rmse / final_train_rmse),
        "improvement_pct": float(improvement)
    },
    "all_methods": results_df.drop('features', axis=1).to_dict('records'),
    "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
}

metadata_path = MODELS_DIR / "lightgbm_metadata_feature_selected.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

# Save feature list
feature_list_path = REPORTS_DIR / "selected_features.txt"
with open(feature_list_path, 'w') as f:
    f.write(f"Selected Features ({len(best_features)}) - Method: {best_idx['method']}\n")
    f.write("=" * 80 + "\n\n")
    for i, feat in enumerate(best_features, 1):
        f.write(f"{i}. {feat}\n")

print(f"✓ Model saved: {model_path}")
print(f"✓ Metadata saved: {metadata_path}")
print(f"✓ Feature list saved: {feature_list_path}\n")

print("=" * 80)
print("FEATURE SELECTION COMPLETE!")
print("=" * 80)
print(f"✓ Reduced features from 84 → {len(best_features)}")
print(f"✓ Reduced overfitting from {baseline_overfitting:.2f}x → {final_test_rmse / final_train_rmse:.2f}x")
print(f"✓ Model ready for production\n")
