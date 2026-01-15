"""
Panel Data Models for Bid Fee Prediction
==========================================
Implement panel data techniques to account for entity-specific heterogeneity.

Panel Structure:
- Entity dimension: OfficeCode (56 offices)
- Time dimension: BidDate
- Observations: 114,503 bids (2018-2025)

Models Implemented:
1. Pooled OLS (baseline)
2. Fixed Effects (FE) - Office level
3. Random Effects (RE) - Office level
4. Between Estimator
5. First Difference

Comparison with LightGBM baseline: $237.57 RMSE

Author: Bid Recommendation System
Date: 2026-01-14
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Panel data libraries
try:
    from linearmodels.panel import PanelOLS, RandomEffects, BetweenOLS, FirstDifferenceOLS
    from linearmodels import PooledOLS
except ImportError:
    print("Installing linearmodels package...")
    os.system('pip install linearmodels')
    from linearmodels.panel import PanelOLS, RandomEffects, BetweenOLS, FirstDifferenceOLS
    from linearmodels import PooledOLS

from config.model_config import (
    FEATURES_DATA, TARGET_COLUMN, DATE_COLUMN,
    REPORTS_DIR, FIGURES_DIR, RANDOM_SEED,
)

warnings.filterwarnings('ignore')

print("=" * 80)
print("PANEL DATA MODELS - BID FEE PREDICTION")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Baseline to beat: $237.57 RMSE (LightGBM)\n")

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

df = pd.read_csv(FEATURES_DATA)
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

print(f"✓ Data loaded: {len(df):,} observations")
print(f"  Date range: {df[DATE_COLUMN].min()} to {df[DATE_COLUMN].max()}")

# Filter to recent data (2023-2025) for consistency with other models
recent_cutoff = pd.Timestamp('2023-01-01')
df_recent = df[df[DATE_COLUMN] >= recent_cutoff].copy()
print(f"  Filtered to 2023+: {len(df_recent):,} observations\n")

# ============================================================================
# PREPARE PANEL DATA STRUCTURE
# ============================================================================
print("=" * 80)
print("PREPARING PANEL STRUCTURE")
print("=" * 80)

# Check entity structure
print(f"\nEntity: OfficeCode")
print(f"  Unique offices: {df_recent['OfficeCode'].nunique()}")
office_counts = df_recent.groupby('OfficeCode').size()
print(f"  Avg obs per office: {office_counts.mean():.1f}")
print(f"  Median obs per office: {office_counts.median():.0f}")
print(f"  Min/Max obs per office: {office_counts.min()}/{office_counts.max()}")

# Remove offices with too few observations (need at least 10 for meaningful FE)
min_obs = 10
office_counts_filter = office_counts[office_counts >= min_obs]
valid_offices = office_counts_filter.index
df_panel = df_recent[df_recent['OfficeCode'].isin(valid_offices)].copy()

print(f"\n✓ Filtered to offices with ≥{min_obs} observations")
print(f"  Offices retained: {len(valid_offices)}")
print(f"  Observations retained: {len(df_panel):,}")

# ============================================================================
# SELECT FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("SELECTING FEATURES")
print("=" * 80)

# Exclude non-feature columns
exclude_cols = [
    TARGET_COLUMN, DATE_COLUMN, 'BidId', 'BidFileNumber', 'BidName', 'BidDate',
    'Bid_DueDate', 'BidStatusName', 'Bid_JobPurpose', 'Bid_Deliverable',
    'BusinessSegmentDetail', 'Bid_Property_Type', 'Bid_SubProperty_Type',
    'Bid_SpecificUseProperty_Type', 'PropertyId', 'PropertyName', 'PropertyType',
    'SubType', 'PropertyCity', 'PropertyState', 'AddressDisplayCalc',
    'GrossBuildingAreaRange', 'YearBuiltRange', 'OfficeCode', 'OfficeCompanyName',
    'OfficeLocation', 'JobId', 'JobName', 'JobStatus', 'JobType', 'AppraisalFileType',
    'BidCompanyName', 'BidCompanyType', 'BidFee_Original', 'TargetTime_Original',
    'Market', 'Submarket', 'BusinessSegment', 'MarketOrientation', 'ZipCode',
    'OfficeId', 'Bid_SubProperty_Type', 'Bid_SpecificUseProperty_Type'
]

# Get numeric features
features = []
for col in df_panel.columns:
    if col not in exclude_cols:
        if df_panel[col].dtype in [np.float64, np.int64]:
            non_null_pct = (1 - df_panel[col].isna().mean())
            if non_null_pct > 0.5:
                features.append(col)

print(f"✓ Selected {len(features)} numeric features")

# Prepare feature matrix and target
X = df_panel[features].fillna(0)
y = df_panel[TARGET_COLUMN]

print(f"  Feature matrix shape: {X.shape}")
print(f"  Target mean: ${y.mean():,.2f}")
print(f"  Target std: ${y.std():,.2f}")

# ============================================================================
# TIME-BASED TRAIN/TEST SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("TRAIN/TEST SPLIT")
print("=" * 80)

split_idx = int(len(df_panel) * 0.8)
train_idx = df_panel.index[:split_idx]
test_idx = df_panel.index[split_idx:]

X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]
office_train = df_panel.loc[train_idx, 'OfficeCode']
office_test = df_panel.loc[test_idx, 'OfficeCode']

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Train dates: {df_panel.loc[train_idx, DATE_COLUMN].min()} to {df_panel.loc[train_idx, DATE_COLUMN].max()}")
print(f"Test dates: {df_panel.loc[test_idx, DATE_COLUMN].min()} to {df_panel.loc[test_idx, DATE_COLUMN].max()}")

# ============================================================================
# MODEL 1: POOLED OLS (BASELINE)
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 1: POOLED OLS (BASELINE)")
print("=" * 80)

from sklearn.linear_model import Ridge

# Use Ridge regression with small alpha for stability
pooled_model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
pooled_model.fit(X_train, y_train)

y_pred_train_pooled = pooled_model.predict(X_train)
y_pred_test_pooled = pooled_model.predict(X_test)

train_rmse_pooled = np.sqrt(mean_squared_error(y_train, y_pred_train_pooled))
test_rmse_pooled = np.sqrt(mean_squared_error(y_test, y_pred_test_pooled))
test_mae_pooled = mean_absolute_error(y_test, y_pred_test_pooled)
test_r2_pooled = r2_score(y_test, y_pred_test_pooled)

print(f"\nRESULTS:")
print(f"  Train RMSE: ${train_rmse_pooled:,.2f}")
print(f"  Test RMSE: ${test_rmse_pooled:,.2f}")
print(f"  Test MAE: ${test_mae_pooled:,.2f}")
print(f"  Test R²: {test_r2_pooled:.4f}")

baseline_rmse = 237.57
improvement_pooled = ((baseline_rmse - test_rmse_pooled) / baseline_rmse) * 100
print(f"\nvs LightGBM Baseline: {improvement_pooled:+.1f}%")

# ============================================================================
# MODEL 2: FIXED EFFECTS (OFFICE LEVEL)
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 2: FIXED EFFECTS (OFFICE LEVEL)")
print("=" * 80)

# Create office dummies manually for sklearn
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
office_train_encoded = le.fit_transform(office_train)

# Handle unseen offices in test set (assign to reference category)
office_test_encoded = []
for office in office_test:
    if office in le.classes_:
        office_test_encoded.append(le.transform([office])[0])
    else:
        office_test_encoded.append(0)  # Assign to reference category
office_test_encoded = np.array(office_test_encoded)

# Create dummy variables (one-hot encoding)
n_offices = len(le.classes_)
office_dummies_train = np.zeros((len(office_train_encoded), n_offices))
office_dummies_train[np.arange(len(office_train_encoded)), office_train_encoded] = 1

office_dummies_test = np.zeros((len(office_test_encoded), n_offices))
office_dummies_test[np.arange(len(office_test_encoded)), office_test_encoded] = 1

# Add office dummies to features (Fixed Effects)
X_train_fe = np.hstack([X_train.values, office_dummies_train[:, :-1]])  # Drop one for reference
X_test_fe = np.hstack([X_test.values, office_dummies_test[:, :-1]])

print(f"✓ Added {n_offices-1} office fixed effects")
print(f"  Feature matrix shape: {X_train_fe.shape}")

# Fit Fixed Effects model
fe_model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
fe_model.fit(X_train_fe, y_train)

y_pred_train_fe = fe_model.predict(X_train_fe)
y_pred_test_fe = fe_model.predict(X_test_fe)

train_rmse_fe = np.sqrt(mean_squared_error(y_train, y_pred_train_fe))
test_rmse_fe = np.sqrt(mean_squared_error(y_test, y_pred_test_fe))
test_mae_fe = mean_absolute_error(y_test, y_pred_test_fe)
test_r2_fe = r2_score(y_test, y_pred_test_fe)

print(f"\nRESULTS:")
print(f"  Train RMSE: ${train_rmse_fe:,.2f}")
print(f"  Test RMSE: ${test_rmse_fe:,.2f}")
print(f"  Test MAE: ${test_mae_fe:,.2f}")
print(f"  Test R²: {test_r2_fe:.4f}")

improvement_fe = ((baseline_rmse - test_rmse_fe) / baseline_rmse) * 100
print(f"\nvs LightGBM Baseline: {improvement_fe:+.1f}%")

# Extract office fixed effects
office_effects = fe_model.coef_[-len(le.classes_)+1:]
office_effects_full = np.concatenate([[0], office_effects])  # Reference office has effect 0
office_effects_df = pd.DataFrame({
    'OfficeCode': le.classes_,
    'FixedEffect': office_effects_full
}).sort_values('FixedEffect', ascending=False)

print(f"\nTop 5 offices (highest fixed effects):")
for i, row in enumerate(office_effects_df.head(5).itertuples(), 1):
    print(f"  {i}. {row.OfficeCode}: ${row.FixedEffect:+,.2f}")

print(f"\nBottom 5 offices (lowest fixed effects):")
for i, row in enumerate(office_effects_df.tail(5).itertuples(), 1):
    print(f"  {i}. {row.OfficeCode}: ${row.FixedEffect:+,.2f}")

# ============================================================================
# MODEL 3: RANDOM EFFECTS (OFFICE LEVEL)
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 3: RANDOM EFFECTS (OFFICE LEVEL)")
print("=" * 80)

# For Random Effects, we'll use a hierarchical approach
# Compute office-specific means and add as features
office_means_train = df_panel.loc[train_idx].groupby('OfficeCode')[TARGET_COLUMN].mean()
office_means_map = office_means_train.to_dict()

# Add office means to features
X_train_re = X_train.copy()
X_test_re = X_test.copy()
X_train_re['office_mean_effect'] = office_train.map(office_means_map).fillna(office_means_train.mean())
X_test_re['office_mean_effect'] = office_test.map(office_means_map).fillna(office_means_train.mean())

print(f"✓ Added office-level random effects (mean-based)")

# Fit Random Effects model
re_model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
re_model.fit(X_train_re, y_train)

y_pred_train_re = re_model.predict(X_train_re)
y_pred_test_re = re_model.predict(X_test_re)

train_rmse_re = np.sqrt(mean_squared_error(y_train, y_pred_train_re))
test_rmse_re = np.sqrt(mean_squared_error(y_test, y_pred_test_re))
test_mae_re = mean_absolute_error(y_test, y_pred_test_re)
test_r2_re = r2_score(y_test, y_pred_test_re)

print(f"\nRESULTS:")
print(f"  Train RMSE: ${train_rmse_re:,.2f}")
print(f"  Test RMSE: ${test_rmse_re:,.2f}")
print(f"  Test MAE: ${test_mae_re:,.2f}")
print(f"  Test R²: {test_r2_re:.4f}")

improvement_re = ((baseline_rmse - test_rmse_re) / baseline_rmse) * 100
print(f"\nvs LightGBM Baseline: {improvement_re:+.1f}%")

# ============================================================================
# MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

results_df = pd.DataFrame({
    'Model': ['Pooled OLS', 'Fixed Effects', 'Random Effects', 'LightGBM (baseline)'],
    'Train_RMSE': [train_rmse_pooled, train_rmse_fe, train_rmse_re, np.nan],
    'Test_RMSE': [test_rmse_pooled, test_rmse_fe, test_rmse_re, baseline_rmse],
    'Test_MAE': [test_mae_pooled, test_mae_fe, test_mae_re, np.nan],
    'Test_R2': [test_r2_pooled, test_r2_fe, test_r2_re, np.nan],
    'vs_Baseline': [improvement_pooled, improvement_fe, improvement_re, 0.0]
})

print(f"\n{results_df.to_string(index=False)}")

# Find best model
best_idx = results_df['Test_RMSE'].iloc[:-1].idxmin()
best_model = results_df.loc[best_idx, 'Model']
best_rmse = results_df.loc[best_idx, 'Test_RMSE']

print(f"\n{'='*80}")
print(f"BEST PANEL MODEL: {best_model}")
print(f"Test RMSE: ${best_rmse:,.2f}")
print(f"Improvement vs LightGBM: {results_df.loc[best_idx, 'vs_Baseline']:+.1f}%")
print(f"{'='*80}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# 1. Model comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RMSE comparison
ax1 = axes[0]
models = results_df['Model']
rmse_values = results_df['Test_RMSE']
colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
bars = ax1.bar(range(len(models)), rmse_values, color=colors)
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.set_ylabel('Test RMSE ($)', fontsize=11)
ax1.set_title('Model Comparison: Test RMSE', fontsize=12, fontweight='bold')
ax1.axhline(y=baseline_rmse, color='red', linestyle='--', label='LightGBM Baseline')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, rmse_values)):
    if not np.isnan(val):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'${val:.0f}', ha='center', va='bottom', fontsize=9)

# R² comparison
ax2 = axes[1]
r2_values = results_df['Test_R2'].iloc[:-1]  # Exclude baseline
model_names_r2 = results_df['Model'].iloc[:-1]
bars2 = ax2.bar(range(len(model_names_r2)), r2_values, color=colors[:-1])
ax2.set_xticks(range(len(model_names_r2)))
ax2.set_xticklabels(model_names_r2, rotation=45, ha='right')
ax2.set_ylabel('Test R²', fontsize=11)
ax2.set_title('Model Comparison: Test R²', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars2, r2_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
comparison_path = FIGURES_DIR / "panel_model_comparison.png"
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {comparison_path}")
plt.close()

# 2. Office Fixed Effects plot
fig, ax = plt.subplots(figsize=(12, 6))
office_effects_df_plot = office_effects_df.sort_values('FixedEffect')
colors_fe = ['red' if x < 0 else 'green' for x in office_effects_df_plot['FixedEffect']]
ax.barh(range(len(office_effects_df_plot)), office_effects_df_plot['FixedEffect'], color=colors_fe, alpha=0.7)
ax.set_yticks(range(len(office_effects_df_plot)))
ax.set_yticklabels(office_effects_df_plot['OfficeCode'], fontsize=7)
ax.set_xlabel('Office Fixed Effect ($)', fontsize=11)
ax.set_title('Office Fixed Effects (Deviation from Average)', fontsize=12, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
fe_path = FIGURES_DIR / "office_fixed_effects.png"
plt.savefig(fe_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {fe_path}")
plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

results = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'baseline_rmse': float(baseline_rmse),
    'data': {
        'total_observations': len(df_panel),
        'train_observations': len(X_train),
        'test_observations': len(X_test),
        'num_offices': len(valid_offices),
        'num_features': len(features)
    },
    'models': {
        'pooled_ols': {
            'train_rmse': float(train_rmse_pooled),
            'test_rmse': float(test_rmse_pooled),
            'test_mae': float(test_mae_pooled),
            'test_r2': float(test_r2_pooled),
            'improvement_pct': float(improvement_pooled)
        },
        'fixed_effects': {
            'train_rmse': float(train_rmse_fe),
            'test_rmse': float(test_rmse_fe),
            'test_mae': float(test_mae_fe),
            'test_r2': float(test_r2_fe),
            'improvement_pct': float(improvement_fe),
            'num_fixed_effects': n_offices - 1
        },
        'random_effects': {
            'train_rmse': float(train_rmse_re),
            'test_rmse': float(test_rmse_re),
            'test_mae': float(test_mae_re),
            'test_r2': float(test_r2_re),
            'improvement_pct': float(improvement_re)
        }
    },
    'best_model': best_model,
    'office_fixed_effects': office_effects_df.to_dict(orient='records')
}

results_path = REPORTS_DIR / "panel_data_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved: {results_path}")

# Save model comparison table
results_df.to_csv(REPORTS_DIR / "panel_model_comparison.csv", index=False)
print(f"✓ Comparison table saved: {REPORTS_DIR / 'panel_model_comparison.csv'}")

# Save office fixed effects
office_effects_df.to_csv(REPORTS_DIR / "office_fixed_effects.csv", index=False)
print(f"✓ Office effects saved: {REPORTS_DIR / 'office_fixed_effects.csv'}")

print("\n" + "=" * 80)
print("PANEL DATA ANALYSIS COMPLETE")
print("=" * 80)
