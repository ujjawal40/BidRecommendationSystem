"""
Comprehensive Feature Selection Analysis
=========================================
Phase 1A: Apply multiple feature selection techniques and compare results

This script implements:
- SHAP values analysis
- Permutation importance
- Recursive Feature Elimination (RFE)
- Correlation analysis
- Variance Inflation Factor (VIF)
- Built-in LightGBM importance
- Feature subset comparison
- Model retraining with optimal features

Author: Bid Recommendation System
Date: 2026-01-07
"""

import sys
import os
from pathlib import Path

# Add project root to path
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
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

# Import configuration
from config.model_config import (
    FEATURES_DATA,
    TARGET_COLUMN,
    DATE_COLUMN,
    EXCLUDE_COLUMNS,
    LIGHTGBM_CONFIG,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    RANDOM_SEED,
)

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


class ComprehensiveFeatureSelector:
    """
    Apply and compare multiple feature selection techniques.

    Implements:
    - SHAP values
    - Permutation importance
    - RFE
    - Correlation analysis
    - VIF
    - LightGBM built-in importance
    """

    def __init__(self):
        """Initialize feature selector."""
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.model = None
        self.selection_results = {}

        print("=" * 80)
        print("COMPREHENSIVE FEATURE SELECTION ANALYSIS")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    def load_data(self):
        """Load and prepare data."""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)

        # Load data
        df = pd.read_csv(FEATURES_DATA)
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

        print(f"✓ Data loaded: {df.shape[0]:,} rows, {df.shape[1]:,} columns")

        # Prepare features
        feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        X = df[numeric_features].fillna(0)
        y = df[TARGET_COLUMN].values

        # 80/20 split
        split_idx = int(len(X) * 0.8)
        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y[:split_idx]
        self.y_test = y[split_idx:]
        self.feature_names = numeric_features

        print(f"✓ Features prepared: {len(numeric_features)} features")
        print(f"✓ Train set: {len(self.X_train):,} samples")
        print(f"✓ Test set: {len(self.X_test):,} samples\n")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_baseline_model(self):
        """Train baseline model with all features."""
        print("=" * 80)
        print("TRAINING BASELINE MODEL (ALL 84 FEATURES)")
        print("=" * 80)

        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        val_data = lgb.Dataset(self.X_test, label=self.y_test, reference=train_data)

        params = LIGHTGBM_CONFIG['params'].copy()

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        # Evaluate
        y_pred = self.model.predict(self.X_test, num_iteration=self.model.best_iteration)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        # Train performance
        y_train_pred = self.model.predict(self.X_train, num_iteration=self.model.best_iteration)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))

        self.baseline_metrics = {
            'train_rmse': train_rmse,
            'test_rmse': rmse,
            'test_mae': mae,
            'test_r2': r2,
            'overfitting_ratio': rmse / train_rmse
        }

        print(f"✓ Baseline model trained")
        print(f"  Train RMSE: ${train_rmse:,.2f}")
        print(f"  Test RMSE:  ${rmse:,.2f}")
        print(f"  Test MAE:   ${mae:,.2f}")
        print(f"  Test R²:    {r2:.4f}")
        print(f"  Overfitting ratio: {rmse / train_rmse:.2f}x\n")

        return self.model

    def method_1_correlation_analysis(self, threshold=0.9):
        """
        Remove highly correlated features.

        Parameters
        ----------
        threshold : float
            Correlation threshold for removal
        """
        print("=" * 80)
        print("METHOD 1: CORRELATION ANALYSIS")
        print("=" * 80)
        print(f"Removing features with correlation > {threshold}\n")

        # Calculate correlation matrix
        corr_matrix = self.X_train.corr().abs()

        # Find highly correlated pairs
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Select features to drop
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

        print(f"Features to remove: {len(to_drop)}")
        if to_drop:
            print("Highly correlated features:")
            for feat in to_drop[:10]:
                print(f"  - {feat}")
            if len(to_drop) > 10:
                print(f"  ... and {len(to_drop) - 10} more")

        selected_features = [f for f in self.feature_names if f not in to_drop]

        self.selection_results['correlation'] = {
            'selected_features': selected_features,
            'n_features': len(selected_features),
            'removed_features': to_drop,
            'n_removed': len(to_drop)
        }

        print(f"\n✓ Selected features: {len(selected_features)}/{len(self.feature_names)}\n")

        return selected_features

    def method_2_variance_inflation_factor(self, threshold=10):
        """
        Remove features with high VIF (multicollinearity).

        Parameters
        ----------
        threshold : float
            VIF threshold for removal
        """
        print("=" * 80)
        print("METHOD 2: VARIANCE INFLATION FACTOR (VIF)")
        print("=" * 80)
        print(f"Removing features with VIF > {threshold}\n")

        # Sample data for faster computation
        X_sample = self.X_train.sample(min(5000, len(self.X_train)), random_state=RANDOM_SEED)

        # Calculate VIF for each feature
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_sample.columns

        print("Calculating VIF scores (this may take a few minutes)...")
        vif_scores = []

        for i, col in enumerate(X_sample.columns):
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{len(X_sample.columns)} features")

            try:
                vif = variance_inflation_factor(X_sample.values, i)
                vif_scores.append(vif)
            except:
                vif_scores.append(0)

        vif_data["VIF"] = vif_scores
        vif_data = vif_data.sort_values('VIF', ascending=False)

        # Remove high VIF features
        high_vif = vif_data[vif_data['VIF'] > threshold]['feature'].tolist()
        selected_features = [f for f in self.feature_names if f not in high_vif]

        print(f"\n✓ Calculation complete")
        print(f"Features with VIF > {threshold}: {len(high_vif)}")
        if high_vif:
            print("High VIF features (top 10):")
            for feat in vif_data.head(10).itertuples():
                print(f"  - {feat.feature}: VIF = {feat.VIF:.2f}")

        self.selection_results['vif'] = {
            'selected_features': selected_features,
            'n_features': len(selected_features),
            'vif_scores': vif_data.to_dict('records'),
            'removed_features': high_vif,
            'n_removed': len(high_vif)
        }

        print(f"\n✓ Selected features: {len(selected_features)}/{len(self.feature_names)}\n")

        return selected_features

    def method_3_lightgbm_importance(self, threshold=0.001):
        """
        Use LightGBM built-in feature importance.

        Parameters
        ----------
        threshold : float
            Minimum importance threshold (as fraction of total)
        """
        print("=" * 80)
        print("METHOD 3: LIGHTGBM BUILT-IN IMPORTANCE")
        print("=" * 80)
        print(f"Keeping features with importance > {threshold*100}% of total\n")

        # Get feature importance
        importance = self.model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        # Calculate percentage
        total_importance = feature_importance['importance'].sum()
        feature_importance['importance_pct'] = (
            feature_importance['importance'] / total_importance
        )

        # Select features above threshold
        selected_features = feature_importance[
            feature_importance['importance_pct'] > threshold
        ]['feature'].tolist()

        print(f"Top 10 features by importance:")
        for feat in feature_importance.head(10).itertuples():
            print(f"  {feat.Index+1}. {feat.feature}: {feat.importance_pct*100:.2f}%")

        self.selection_results['lgb_importance'] = {
            'selected_features': selected_features,
            'n_features': len(selected_features),
            'feature_importance': feature_importance.to_dict('records')[:20],
            'n_removed': len(self.feature_names) - len(selected_features)
        }

        print(f"\n✓ Selected features: {len(selected_features)}/{len(self.feature_names)}\n")

        return selected_features

    def method_4_shap_values(self, threshold=0.01):
        """
        Use SHAP values for feature importance.

        Parameters
        ----------
        threshold : float
            Minimum SHAP importance threshold
        """
        print("=" * 80)
        print("METHOD 4: SHAP VALUES")
        print("=" * 80)
        print(f"Keeping features with SHAP importance > {threshold*100}% of total\n")

        # Sample for SHAP (it's computationally expensive)
        X_shap = self.X_train.sample(min(1000, len(self.X_train)), random_state=RANDOM_SEED)

        print("Calculating SHAP values (this may take 2-3 minutes)...")

        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_shap)

        # Calculate mean absolute SHAP values
        shap_importance = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('shap_importance', ascending=False)

        # Calculate percentage
        total_shap = shap_importance['shap_importance'].sum()
        shap_importance['shap_pct'] = shap_importance['shap_importance'] / total_shap

        # Cumulative importance
        shap_importance['cumulative_pct'] = shap_importance['shap_pct'].cumsum()

        # Select features
        selected_features = shap_importance[
            shap_importance['shap_pct'] > threshold
        ]['feature'].tolist()

        print(f"\n✓ SHAP calculation complete")
        print(f"Top 10 features by SHAP importance:")
        for feat in shap_importance.head(10).itertuples():
            print(f"  {feat.Index+1}. {feat.feature}: {feat.shap_pct*100:.2f}% (cumulative: {feat.cumulative_pct*100:.1f}%)")

        # Create SHAP summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=20)
        plt.tight_layout()
        shap_plot_path = FIGURES_DIR / "shap_feature_importance.png"
        plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ SHAP plot saved: {shap_plot_path}")

        self.selection_results['shap'] = {
            'selected_features': selected_features,
            'n_features': len(selected_features),
            'shap_importance': shap_importance.to_dict('records')[:30],
            'n_removed': len(self.feature_names) - len(selected_features)
        }

        print(f"\n✓ Selected features: {len(selected_features)}/{len(self.feature_names)}\n")

        return selected_features

    def method_5_permutation_importance(self, n_repeats=10, threshold=0.0):
        """
        Use permutation importance.

        Parameters
        ----------
        n_repeats : int
            Number of times to permute each feature
        threshold : float
            Minimum importance threshold
        """
        print("=" * 80)
        print("METHOD 5: PERMUTATION IMPORTANCE")
        print("=" * 80)
        print(f"Permutation repeats: {n_repeats}\n")

        print("Calculating permutation importance (this may take 3-5 minutes)...")

        # Calculate permutation importance
        perm_importance = permutation_importance(
            self.model,
            self.X_test,
            self.y_test,
            n_repeats=n_repeats,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )

        # Create DataFrame
        perm_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=False)

        # Select features with positive importance
        selected_features = perm_df[perm_df['importance'] > threshold]['feature'].tolist()

        print(f"\n✓ Permutation importance calculated")
        print(f"Top 10 features by permutation importance:")
        for feat in perm_df.head(10).itertuples():
            print(f"  {feat.Index+1}. {feat.feature}: {feat.importance:.2f} ± {feat.std:.2f}")

        self.selection_results['permutation'] = {
            'selected_features': selected_features,
            'n_features': len(selected_features),
            'perm_importance': perm_df.to_dict('records')[:30],
            'n_removed': len(self.feature_names) - len(selected_features)
        }

        print(f"\n✓ Selected features: {len(selected_features)}/{len(self.feature_names)}\n")

        return selected_features

    def method_6_recursive_feature_elimination(self, n_features_to_select=30):
        """
        Recursive Feature Elimination with cross-validation.

        Parameters
        ----------
        n_features_to_select : int
            Target number of features
        """
        print("=" * 80)
        print("METHOD 6: RECURSIVE FEATURE ELIMINATION (RFE)")
        print("=" * 80)
        print(f"Target number of features: {n_features_to_select}\n")

        print("Running RFE (this may take 5-10 minutes)...")

        # Create a simple model for RFE (faster than full model)
        estimator = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            random_state=RANDOM_SEED,
            verbose=-1
        )

        # Run RFE
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            step=5,
            verbose=1
        )

        rfe.fit(self.X_train, self.y_train)

        # Get selected features
        selected_features = [
            feat for feat, selected in zip(self.feature_names, rfe.support_) if selected
        ]

        # Get feature rankings
        rfe_ranking = pd.DataFrame({
            'feature': self.feature_names,
            'ranking': rfe.ranking_,
            'selected': rfe.support_
        }).sort_values('ranking')

        print(f"\n✓ RFE complete")
        print(f"Selected features (top 20):")
        for feat in rfe_ranking[rfe_ranking['selected']].head(20).itertuples():
            print(f"  - {feat.feature} (rank: {feat.ranking})")

        self.selection_results['rfe'] = {
            'selected_features': selected_features,
            'n_features': len(selected_features),
            'rfe_ranking': rfe_ranking.to_dict('records')[:50],
            'n_removed': len(self.feature_names) - len(selected_features)
        }

        print(f"\n✓ Selected features: {len(selected_features)}/{len(self.feature_names)}\n")

        return selected_features

    def compare_methods(self):
        """Compare results from all feature selection methods."""
        print("=" * 80)
        print("COMPARISON OF FEATURE SELECTION METHODS")
        print("=" * 80)

        comparison_df = pd.DataFrame({
            'Method': [],
            'Features Selected': [],
            'Features Removed': [],
            'Reduction %': []
        })

        for method, results in self.selection_results.items():
            comparison_df = pd.concat([comparison_df, pd.DataFrame({
                'Method': [method.upper()],
                'Features Selected': [results['n_features']],
                'Features Removed': [results['n_removed']],
                'Reduction %': [results['n_removed'] / len(self.feature_names) * 100]
            })], ignore_index=True)

        print(comparison_df.to_string(index=False))
        print()

        # Find consensus features (selected by majority of methods)
        all_selected = {}
        for method, results in self.selection_results.items():
            for feat in results['selected_features']:
                all_selected[feat] = all_selected.get(feat, 0) + 1

        # Features selected by at least 4/6 methods
        consensus_features = [feat for feat, count in all_selected.items() if count >= 4]

        print(f"CONSENSUS FEATURES (selected by ≥4/6 methods): {len(consensus_features)}")
        print()

        # Save comparison
        self.comparison_df = comparison_df
        self.consensus_features = consensus_features

        return comparison_df, consensus_features

    def evaluate_feature_subsets(self):
        """Evaluate model performance with each feature selection method."""
        print("=" * 80)
        print("EVALUATING FEATURE SUBSETS")
        print("=" * 80)
        print("Training models with each feature subset to compare performance...\n")

        evaluation_results = []

        for method, results in self.selection_results.items():
            print(f"Testing {method.upper()} ({results['n_features']} features)...")

            selected_features = results['selected_features']

            # Train model with selected features
            X_train_subset = self.X_train[selected_features]
            X_test_subset = self.X_test[selected_features]

            train_data = lgb.Dataset(X_train_subset, label=self.y_train)
            val_data = lgb.Dataset(X_test_subset, label=self.y_test, reference=train_data)

            params = LIGHTGBM_CONFIG['params'].copy()

            model_subset = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )

            # Evaluate
            y_pred = model_subset.predict(X_test_subset, num_iteration=model_subset.best_iteration)
            y_train_pred = model_subset.predict(X_train_subset, num_iteration=model_subset.best_iteration)

            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            test_mae = mean_absolute_error(self.y_test, y_pred)
            test_r2 = r2_score(self.y_test, y_pred)
            overfitting_ratio = test_rmse / train_rmse

            evaluation_results.append({
                'method': method,
                'n_features': results['n_features'],
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'overfitting_ratio': overfitting_ratio
            })

            print(f"  Train RMSE: ${train_rmse:,.2f}")
            print(f"  Test RMSE:  ${test_rmse:,.2f}")
            print(f"  Test MAE:   ${test_mae:,.2f}")
            print(f"  Test R²:    {test_r2:.4f}")
            print(f"  Overfitting: {overfitting_ratio:.2f}x")
            print()

        # Also evaluate consensus features
        print(f"Testing CONSENSUS FEATURES ({len(self.consensus_features)} features)...")

        X_train_consensus = self.X_train[self.consensus_features]
        X_test_consensus = self.X_test[self.consensus_features]

        train_data = lgb.Dataset(X_train_consensus, label=self.y_train)
        val_data = lgb.Dataset(X_test_consensus, label=self.y_test, reference=train_data)

        model_consensus = lgb.train(
            LIGHTGBM_CONFIG['params'],
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        y_pred = model_consensus.predict(X_test_consensus, num_iteration=model_consensus.best_iteration)
        y_train_pred = model_consensus.predict(X_train_consensus, num_iteration=model_consensus.best_iteration)

        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        test_mae = mean_absolute_error(self.y_test, y_pred)
        test_r2 = r2_score(self.y_test, y_pred)
        overfitting_ratio = test_rmse / train_rmse

        evaluation_results.append({
            'method': 'consensus',
            'n_features': len(self.consensus_features),
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'overfitting_ratio': overfitting_ratio
        })

        print(f"  Train RMSE: ${train_rmse:,.2f}")
        print(f"  Test RMSE:  ${test_rmse:,.2f}")
        print(f"  Test MAE:   ${test_mae:,.2f}")
        print(f"  Test R²:    {test_r2:.4f}")
        print(f"  Overfitting: {overfitting_ratio:.2f}x")
        print()

        # Create comparison DataFrame
        eval_df = pd.DataFrame(evaluation_results)
        eval_df = eval_df.sort_values('overfitting_ratio')

        print("=" * 80)
        print("PERFORMANCE COMPARISON")
        print("=" * 80)
        print("\nRanked by Overfitting Ratio (lower is better):")
        print(eval_df[['method', 'n_features', 'test_rmse', 'test_r2', 'overfitting_ratio']].to_string(index=False))
        print()

        # Find best method
        best_method = eval_df.iloc[0]['method']
        best_features = (self.consensus_features if best_method == 'consensus'
                        else self.selection_results[best_method]['selected_features'])

        print(f"🏆 BEST METHOD: {best_method.upper()}")
        print(f"   Features: {eval_df.iloc[0]['n_features']}")
        print(f"   Test RMSE: ${eval_df.iloc[0]['test_rmse']:,.2f}")
        print(f"   Test R²: {eval_df.iloc[0]['test_r2']:.4f}")
        print(f"   Overfitting ratio: {eval_df.iloc[0]['overfitting_ratio']:.2f}x")
        print()

        # Compare to baseline
        baseline_ratio = self.baseline_metrics['overfitting_ratio']
        best_ratio = eval_df.iloc[0]['overfitting_ratio']
        improvement = (baseline_ratio - best_ratio) / baseline_ratio * 100

        print(f"IMPROVEMENT OVER BASELINE:")
        print(f"  Baseline overfitting: {baseline_ratio:.2f}x")
        print(f"  Best method overfitting: {best_ratio:.2f}x")
        print(f"  Improvement: {improvement:.1f}%")
        print()

        self.evaluation_df = eval_df
        self.best_method = best_method
        self.best_features = best_features

        return eval_df, best_method, best_features

    def train_final_model_with_selected_features(self):
        """Train final model with the best selected features."""
        print("=" * 80)
        print("TRAINING FINAL MODEL WITH SELECTED FEATURES")
        print("=" * 80)
        print(f"Using features from: {self.best_method.upper()}")
        print(f"Number of features: {len(self.best_features)}\n")

        # Prepare data with selected features
        X_train_final = self.X_train[self.best_features]
        X_test_final = self.X_test[self.best_features]

        train_data = lgb.Dataset(X_train_final, label=self.y_train,
                                feature_name=self.best_features)
        val_data = lgb.Dataset(X_test_final, label=self.y_test, reference=train_data)

        # Train with default params
        params = LIGHTGBM_CONFIG['params'].copy()

        print("Training in progress...")
        final_model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )

        # Evaluate
        y_pred = final_model.predict(X_test_final, num_iteration=final_model.best_iteration)
        y_train_pred = final_model.predict(X_train_final, num_iteration=final_model.best_iteration)

        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)

        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        test_mae = mean_absolute_error(self.y_test, y_pred)
        test_r2 = r2_score(self.y_test, y_pred)

        print(f"\n✓ Final model trained")
        print(f"\nTRAIN SET PERFORMANCE:")
        print(f"  RMSE: ${train_rmse:,.2f}")
        print(f"  MAE:  ${train_mae:,.2f}")
        print(f"  R²:   {train_r2:.4f}")

        print(f"\nTEST SET PERFORMANCE:")
        print(f"  RMSE: ${test_rmse:,.2f}")
        print(f"  MAE:  ${test_mae:,.2f}")
        print(f"  R²:   {test_r2:.4f}")

        print(f"\nOVERFITTING RATIO: {test_rmse / train_rmse:.2f}x")
        print()

        self.final_model = final_model
        self.final_metrics = {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'overfitting_ratio': test_rmse / train_rmse
        }

        return final_model

    def save_results(self):
        """Save all results and the final model."""
        print("=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        # Save final model
        model_path = MODELS_DIR / "lightgbm_bidfee_model_feature_selected.txt"
        self.final_model.save_model(str(model_path))
        print(f"✓ Model saved: {model_path}")

        # Save metadata
        metadata = {
            "model_type": "LightGBM (Feature Selected)",
            "phase": "1A - Bid Fee Prediction",
            "target_variable": TARGET_COLUMN,
            "feature_selection": {
                "method": self.best_method,
                "original_features": len(self.feature_names),
                "selected_features": len(self.best_features),
                "reduction_pct": (1 - len(self.best_features) / len(self.feature_names)) * 100
            },
            "selected_features": self.best_features,
            "best_iteration": int(self.final_model.best_iteration),
            "metrics": {
                k: float(v) for k, v in self.final_metrics.items()
            },
            "baseline_comparison": {
                "baseline_overfitting": float(self.baseline_metrics['overfitting_ratio']),
                "final_overfitting": float(self.final_metrics['overfitting_ratio']),
                "improvement_pct": float(
                    (self.baseline_metrics['overfitting_ratio'] -
                     self.final_metrics['overfitting_ratio']) /
                    self.baseline_metrics['overfitting_ratio'] * 100
                )
            },
            "all_methods_comparison": self.evaluation_df.to_dict('records'),
            "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        metadata_path = MODELS_DIR / "lightgbm_metadata_feature_selected.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved: {metadata_path}")

        # Save detailed selection results
        selection_report_path = REPORTS_DIR / "feature_selection_report.json"
        with open(selection_report_path, 'w') as f:
            json.dump(self.selection_results, f, indent=2, default=str)
        print(f"✓ Selection report saved: {selection_report_path}")

        # Save comparison CSV
        comparison_csv_path = REPORTS_DIR / "feature_selection_comparison.csv"
        self.evaluation_df.to_csv(comparison_csv_path, index=False)
        print(f"✓ Comparison CSV saved: {comparison_csv_path}")

        # Create visualization
        self.create_comparison_plot()

        print()

    def create_comparison_plot(self):
        """Create visualization comparing all methods."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Number of features
        axes[0, 0].barh(self.evaluation_df['method'], self.evaluation_df['n_features'])
        axes[0, 0].axvline(x=len(self.feature_names), color='r', linestyle='--', label='Original (84)')
        axes[0, 0].set_xlabel('Number of Features')
        axes[0, 0].set_title('Features Selected by Each Method')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Test RMSE
        axes[0, 1].barh(self.evaluation_df['method'], self.evaluation_df['test_rmse'])
        axes[0, 1].axvline(x=self.baseline_metrics['test_rmse'], color='r',
                          linestyle='--', label=f"Baseline (${self.baseline_metrics['test_rmse']:.0f})")
        axes[0, 1].set_xlabel('Test RMSE ($)')
        axes[0, 1].set_title('Test Set Performance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Overfitting ratio
        colors = ['green' if x < 2.0 else 'orange' if x < 3.0 else 'red'
                 for x in self.evaluation_df['overfitting_ratio']]
        axes[1, 0].barh(self.evaluation_df['method'],
                       self.evaluation_df['overfitting_ratio'],
                       color=colors)
        axes[1, 0].axvline(x=self.baseline_metrics['overfitting_ratio'], color='r',
                          linestyle='--', label=f"Baseline ({self.baseline_metrics['overfitting_ratio']:.2f}x)")
        axes[1, 0].axvline(x=1.3, color='green', linestyle=':', label='Target (1.3x)')
        axes[1, 0].set_xlabel('Overfitting Ratio (Test/Train RMSE)')
        axes[1, 0].set_title('Overfitting Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. R² scores
        axes[1, 1].barh(self.evaluation_df['method'], self.evaluation_df['test_r2'])
        axes[1, 1].axvline(x=self.baseline_metrics['test_r2'], color='r',
                          linestyle='--', label=f"Baseline ({self.baseline_metrics['test_r2']:.4f})")
        axes[1, 1].set_xlabel('Test R²')
        axes[1, 1].set_title('R² Score Comparison')
        axes[1, 1].set_xlim(0.95, 1.0)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = FIGURES_DIR / "feature_selection_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Comparison plot saved: {save_path}")

    def run_full_pipeline(self):
        """Execute complete feature selection pipeline."""
        # Load data
        self.load_data()

        # Train baseline
        self.train_baseline_model()

        # Apply all feature selection methods
        print("\n" + "=" * 80)
        print("APPLYING FEATURE SELECTION METHODS")
        print("=" * 80)
        print("This will take approximately 15-20 minutes total...\n")

        self.method_1_correlation_analysis(threshold=0.9)
        self.method_2_variance_inflation_factor(threshold=10)
        self.method_3_lightgbm_importance(threshold=0.001)
        self.method_4_shap_values(threshold=0.01)
        self.method_5_permutation_importance(n_repeats=10)
        self.method_6_recursive_feature_elimination(n_features_to_select=30)

        # Compare methods
        self.compare_methods()

        # Evaluate each feature subset
        self.evaluate_feature_subsets()

        # Train final model with best features
        self.train_final_model_with_selected_features()

        # Save everything
        self.save_results()

        print("=" * 80)
        print("FEATURE SELECTION COMPLETE")
        print("=" * 80)
        print(f"✓ Best method: {self.best_method.upper()}")
        print(f"✓ Features reduced: {len(self.feature_names)} → {len(self.best_features)}")
        print(f"✓ Overfitting improved: {self.baseline_metrics['overfitting_ratio']:.2f}x → {self.final_metrics['overfitting_ratio']:.2f}x")
        print(f"✓ Model saved and ready for use\n")


def main():
    """Main execution function."""
    selector = ComprehensiveFeatureSelector()
    selector.run_full_pipeline()

    return selector


if __name__ == "__main__":
    selector = main()
