"""
XGBoost Win Probability Classification Model
=============================================
Train XGBoost classifier for win probability to compare against LightGBM.

Purpose: Predict P(Win) for Expected Value optimization: EV = P(Win) × BidFee

Author: Bid Recommendation System
Date: 2026-02-03
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import xgboost as xgb
import json
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import warnings

from config.model_config import (
    FEATURES_DATA, DATE_COLUMN, TARGET_COLUMN,
    EXCLUDE_COLUMNS, JOBDATA_FEATURES_TO_EXCLUDE,
    DATA_START_DATE, USE_RECENT_DATA_ONLY,
    MODELS_DIR, REPORTS_DIR, FIGURES_DIR, RANDOM_SEED,
)

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Features that LEAK future information for classification
LEAKY_CLASSIFICATION_FEATURES = [
    'win_rate_with_client',
    'office_win_rate',
    'propertytype_win_rate',
    'state_win_rate',
    'segment_win_rate',
    'client_win_rate',
    'rolling_win_rate_office',
    'total_wins_with_client',
    'prev_won_same_client',
]

XGBOOST_CLASSIFICATION_CONFIG = {
    "params": {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "booster": "gbtree",
        "max_depth": 8,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "gamma": 0.1,
        "scale_pos_weight": 1.0,  # Will adjust based on class imbalance
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": 1,
    },
    "training": {
        "early_stopping_rounds": 50,
    },
}


def main():
    print("=" * 80)
    print("XGBOOST WIN PROBABILITY CLASSIFICATION MODEL")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Objective: Predict P(Win) for Expected Value optimization\n")

    # ========================================================================
    # LOAD AND FILTER DATA
    # ========================================================================
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    df = pd.read_csv(FEATURES_DATA)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    print(f"✓ Data loaded: {len(df):,} rows")

    if USE_RECENT_DATA_ONLY:
        start_date = pd.Timestamp(DATA_START_DATE)
        original_count = len(df)
        df = df[df[DATE_COLUMN] >= start_date].copy()
        print(f"✓ Filtered to {DATA_START_DATE}+ data: {len(df):,} records")

    # ========================================================================
    # PREPARE FEATURES
    # ========================================================================
    print("\n" + "=" * 80)
    print("PREPARING FEATURES")
    print("=" * 80)

    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    feature_cols = [col for col in feature_cols if col not in JOBDATA_FEATURES_TO_EXCLUDE]
    feature_cols = [col for col in feature_cols if col not in LEAKY_CLASSIFICATION_FEATURES]
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [f for f in numeric_features if f != 'Won']

    print(f"Total features: {len(numeric_features)}")
    print(f"Excluded {len(LEAKY_CLASSIFICATION_FEATURES)} leaky features")

    X = df[numeric_features].fillna(0)
    y = df['Won'].values

    win_rate = y.mean()
    print(f"\nClass Distribution:")
    print(f"  Wins:   {y.sum():,} ({win_rate*100:.1f}%)")
    print(f"  Losses: {len(y) - y.sum():,} ({(1-win_rate)*100:.1f}%)")

    # Adjust for class imbalance
    scale_pos_weight = (1 - win_rate) / win_rate
    XGBOOST_CLASSIFICATION_CONFIG["params"]["scale_pos_weight"] = scale_pos_weight
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    # ========================================================================
    # TIME-BASED SPLIT
    # ========================================================================
    print("\n" + "=" * 80)
    print("TIME-BASED TRAIN/VALID/TEST SPLIT")
    print("=" * 80)

    n = len(X)
    train_idx = int(n * 0.6)
    valid_idx = int(n * 0.8)

    X_train, y_train = X.iloc[:train_idx], y[:train_idx]
    X_valid, y_valid = X.iloc[train_idx:valid_idx], y[train_idx:valid_idx]
    X_test, y_test = X.iloc[valid_idx:], y[valid_idx:]

    print(f"Split: 60% train / 20% valid / 20% test")
    print(f"  Train: {len(X_train):,} (win rate: {y_train.mean()*100:.1f}%)")
    print(f"  Valid: {len(X_valid):,} (win rate: {y_valid.mean()*100:.1f}%)")
    print(f"  Test: {len(X_test):,} (win rate: {y_test.mean()*100:.1f}%)")

    # ========================================================================
    # TRAIN MODEL
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING XGBOOST CLASSIFIER")
    print("=" * 80)

    params = XGBOOST_CLASSIFICATION_CONFIG["params"]
    early_stopping = XGBOOST_CLASSIFICATION_CONFIG["training"]["early_stopping_rounds"]

    print("Configuration:")
    for key, value in params.items():
        if key not in ['random_state', 'n_jobs', 'verbosity']:
            print(f"  {key}: {value}")

    # Add early stopping
    params_with_es = params.copy()
    params_with_es['early_stopping_rounds'] = early_stopping
    params_with_es['callbacks'] = [xgb.callback.EvaluationMonitor(period=100)]

    model = xgb.XGBClassifier(**params_with_es)

    print("\nTraining...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=False,
    )

    best_iter = getattr(model, 'best_iteration', model.n_estimators)
    print(f"\n✓ Training complete")
    print(f"  Best iteration: {best_iter}")

    # ========================================================================
    # EVALUATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    # Predictions
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_valid_proba = model.predict_proba(X_valid)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    y_train_pred = (y_train_proba >= 0.5).astype(int)
    y_valid_pred = (y_valid_proba >= 0.5).astype(int)
    y_test_pred = (y_test_proba >= 0.5).astype(int)

    def calc_metrics(y_true, y_pred, y_proba):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_proba),
            'avg_precision': average_precision_score(y_true, y_proba),
            'brier_score': brier_score_loss(y_true, y_proba),
        }

    train_metrics = calc_metrics(y_train, y_train_pred, y_train_proba)
    valid_metrics = calc_metrics(y_valid, y_valid_pred, y_valid_proba)
    test_metrics = calc_metrics(y_test, y_test_pred, y_test_proba)

    print("\nPerformance Summary:")
    print(f"  {'Set':<12} {'AUC-ROC':<10} {'Accuracy':<10} {'F1':<10} {'Brier':>10}")
    print(f"  {'-'*52}")
    print(f"  {'Train':<12} {train_metrics['auc_roc']:<10.4f} {train_metrics['accuracy']:<10.4f} {train_metrics['f1']:<10.4f} {train_metrics['brier_score']:>10.4f}")
    print(f"  {'Valid':<12} {valid_metrics['auc_roc']:<10.4f} {valid_metrics['accuracy']:<10.4f} {valid_metrics['f1']:<10.4f} {valid_metrics['brier_score']:>10.4f}")
    print(f"  {'Test':<12} {test_metrics['auc_roc']:<10.4f} {test_metrics['accuracy']:<10.4f} {test_metrics['f1']:<10.4f} {test_metrics['brier_score']:>10.4f}")

    # Overfitting check
    auc_overfit = train_metrics['auc_roc'] / test_metrics['auc_roc']
    print(f"\nOverfitting Analysis:")
    print(f"  Train/Test AUC ratio: {auc_overfit:.2f}x")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTest Set Confusion Matrix:")
    print(f"  {'':>15} {'Loss':>10} {'Win':>10}")
    print(f"  {'Actual Loss':>15} {tn:>10,} {fp:>10,}")
    print(f"  {'Actual Win':>15} {fn:>10,} {tp:>10,}")

    # Compare with LightGBM
    print("\n" + "=" * 80)
    print("COMPARISON WITH LIGHTGBM BASELINE")
    print("=" * 80)
    lightgbm_auc = 0.8815  # From metadata
    print(f"  LightGBM Test AUC: {lightgbm_auc:.4f} (baseline)")
    print(f"  XGBoost Test AUC:  {test_metrics['auc_roc']:.4f}")
    improvement = (test_metrics['auc_roc'] - lightgbm_auc) / lightgbm_auc * 100
    if improvement > 0:
        print(f"  Improvement: {improvement:.2f}% better ✅")
    else:
        print(f"  Difference: {-improvement:.2f}% worse")

    # ========================================================================
    # FEATURE IMPORTANCE
    # ========================================================================
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)

    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': numeric_features,
        'importance': importance
    }).sort_values('importance', ascending=False)

    importance_df['importance_pct'] = importance_df['importance'] / importance_df['importance'].sum() * 100

    print("\nTop 20 Most Important Features:")
    for i, (_, row) in enumerate(importance_df.head(20).iterrows()):
        print(f"  {i+1:2d}. {row['feature']:40s} {row['importance_pct']:6.2f}%")

    # ========================================================================
    # SAVE MODEL
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)

    model_path = MODELS_DIR / "xgboost_win_probability.json"
    model.save_model(str(model_path))
    print(f"✓ Model saved: {model_path}")

    # Save metadata
    metadata = {
        "model_type": "XGBoost Binary Classifier",
        "target_variable": "Won",
        "purpose": "Predict P(Win) for Expected Value optimization: EV = P(Win) × BidFee",
        "num_features": len(numeric_features),
        "features": numeric_features,
        "excluded_leaky_features": LEAKY_CLASSIFICATION_FEATURES,
        "data_config": {
            "start_date": DATA_START_DATE,
            "total_samples": len(df),
            "train_samples": len(X_train),
            "valid_samples": len(X_valid),
            "test_samples": len(X_test),
            "class_distribution": {
                "wins": int(y.sum()),
                "losses": int(len(y) - y.sum()),
                "win_rate": float(win_rate),
            }
        },
        "hyperparameters": {k: v for k, v in params.items() if k not in ['n_jobs', 'verbosity']},
        "training": {
            "best_iteration": model.best_iteration,
            "early_stopping_rounds": early_stopping,
        },
        "metrics": {
            "train": train_metrics,
            "valid": valid_metrics,
            "test": test_metrics,
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "comparison": {
            "lightgbm_test_auc": lightgbm_auc,
            "xgboost_test_auc": test_metrics['auc_roc'],
            "improvement_pct": improvement,
        },
        "feature_importance": importance_df.head(20).to_dict('records'),
        "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    metadata_path = MODELS_DIR / "xgboost_win_probability_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"✓ Metadata saved: {metadata_path}")

    importance_path = REPORTS_DIR / "xgboost_win_probability_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"✓ Feature importance saved: {importance_path}")

    # ========================================================================
    # SAVE PLOTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING PLOTS")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ROC Curve
    ax1 = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    ax1.plot(fpr, tpr, 'b-', label=f'XGBoost (AUC = {test_metrics["auc_roc"]:.4f})')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve - Win Probability')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Calibration plot
    ax2 = axes[0, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_test_proba, n_bins=10)
    ax2.plot(prob_pred, prob_true, 'bo-', label='XGBoost')
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfect')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Actual Win Rate')
    ax2.set_title('Calibration Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Feature importance
    ax3 = axes[1, 0]
    top_n = 15
    top_features = importance_df.head(top_n)
    ax3.barh(range(len(top_features)), top_features['importance_pct'])
    ax3.set_yticks(range(len(top_features)))
    ax3.set_yticklabels(top_features['feature'])
    ax3.set_xlabel('Importance (%)')
    ax3.set_title(f'Top {top_n} Feature Importance')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')

    # Model comparison
    ax4 = axes[1, 1]
    models = ['LightGBM\n(Baseline)', 'XGBoost']
    auc_values = [lightgbm_auc, test_metrics['auc_roc']]
    colors = ['#3498db', '#e74c3c']
    bars = ax4.bar(models, auc_values, color=colors, edgecolor='black')
    ax4.set_ylabel('Test AUC-ROC')
    ax4.set_title('Model Comparison: Test AUC')
    ax4.set_ylim(0.8, 1.0)
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, auc_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plot_path = FIGURES_DIR / 'xgboost_win_probability_evaluation.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plots saved: {plot_path}")
    plt.close()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("XGBOOST WIN PROBABILITY MODEL COMPLETE")
    print("=" * 80)

    print(f"\n📊 Model Performance:")
    print(f"   Test AUC-ROC:  {test_metrics['auc_roc']:.4f}")
    print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Brier Score:   {test_metrics['brier_score']:.4f}")

    if improvement > 0:
        assessment = f"✅ XGBoost is {improvement:.2f}% better than LightGBM"
    elif improvement > -2:
        assessment = f"⚠️ XGBoost is comparable to LightGBM"
    else:
        assessment = f"❌ LightGBM performs better"

    print(f"\n📋 Assessment: {assessment}")

    print(f"\n📁 Files saved:")
    print(f"   - {model_path}")
    print(f"   - {metadata_path}")
    print(f"   - {importance_path}")
    print(f"   - {plot_path}")

    return model, test_metrics


if __name__ == "__main__":
    model, metrics = main()
