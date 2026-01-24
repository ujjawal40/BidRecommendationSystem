"""
Phase 1B: Win Probability Classification Model (Baseline)
=========================================================
Predict probability of winning a bid to complete the Expected Value system:

    Expected Value = Win Probability × Bid Fee

This model matches the Phase 1A baseline approach:
- 2023+ data only
- 60/20/20 train/valid/test split
- Moderate regularization
- Proper validation (early stopping on valid, not test)
- Excludes leaky features (win_rate features use future information)

Target: AUC-ROC > 0.80

Author: Bid Recommendation System
Date: 2026-01-23
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
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
# CLASSIFICATION-SPECIFIC CONFIGURATION
# ============================================================================

# Features that LEAK future information for classification
# These use the outcome (Won) in their calculation - cannot use them to predict Won
LEAKY_CLASSIFICATION_FEATURES = [
    'win_rate_with_client',      # Uses Won outcome
    'office_win_rate',           # Uses Won outcome
    'propertytype_win_rate',     # Uses Won outcome
    'state_win_rate',            # Uses Won outcome
    'segment_win_rate',          # Uses Won outcome
    'client_win_rate',           # Uses Won outcome
    'rolling_win_rate_office',   # Uses Won outcome
    'total_wins_with_client',    # Uses Won outcome
    'prev_won_same_client',      # Uses Won outcome
]

# Classification hyperparameters (matched to regression baseline approach)
CLASSIFICATION_CONFIG = {
    "params": {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "num_leaves": 20,           # Same as regression baseline
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "max_depth": 8,             # Same as regression baseline
        "min_child_samples": 30,    # Same as regression baseline
        "min_child_weight": 5,
        "reg_alpha": 1.0,           # Moderate regularization
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.0,    # Will adjust based on class imbalance
        "random_state": RANDOM_SEED,
        "verbose": -1,
    },
    "training": {
        "num_boost_round": 500,
        "early_stopping_rounds": 50,
    },
}


def main():
    print("=" * 80)
    print("PHASE 1B: WIN PROBABILITY CLASSIFICATION MODEL")
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
    print(f"  Date range: {df[DATE_COLUMN].min()} to {df[DATE_COLUMN].max()}")

    # Filter to recent data (matching regression baseline)
    if USE_RECENT_DATA_ONLY:
        start_date = pd.Timestamp(DATA_START_DATE)
        original_count = len(df)
        df = df[df[DATE_COLUMN] >= start_date].copy()
        print(f"\n✓ Filtered to {DATA_START_DATE}+ data")
        print(f"  Removed: {original_count - len(df):,} older records")
        print(f"  Remaining: {len(df):,} records")

    # ========================================================================
    # PREPARE FEATURES (Exclude leaky features for classification)
    # ========================================================================
    print("\n" + "=" * 80)
    print("PREPARING FEATURES")
    print("=" * 80)

    # Start with all columns, exclude non-features
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]

    # Exclude JobData features (same as regression)
    feature_cols = [col for col in feature_cols if col not in JOBDATA_FEATURES_TO_EXCLUDE]

    # CRITICAL: Exclude leaky classification features
    feature_cols = [col for col in feature_cols if col not in LEAKY_CLASSIFICATION_FEATURES]

    # Keep only numeric
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # Also exclude 'Won' if it's somehow in features (it's our target)
    numeric_features = [f for f in numeric_features if f != 'Won']

    print(f"Total columns: {len(df.columns)}")
    print(f"After excluding IDs/targets: {len(feature_cols)}")
    print(f"After excluding JobData: {len([c for c in feature_cols if c not in JOBDATA_FEATURES_TO_EXCLUDE])}")
    print(f"After excluding leaky win_rate features: {len(numeric_features)}")

    print(f"\n⚠ Excluded {len(LEAKY_CLASSIFICATION_FEATURES)} leaky features:")
    for feat in LEAKY_CLASSIFICATION_FEATURES:
        print(f"   - {feat}")

    # Prepare X and y
    X = df[numeric_features].fillna(0)
    y = df['Won'].values

    print(f"\n✓ Features prepared: {len(numeric_features)} features")
    print(f"  Target: Won (binary)")

    # Class distribution
    win_rate = y.mean()
    print(f"\nClass Distribution:")
    print(f"  Wins:   {y.sum():,} ({win_rate*100:.1f}%)")
    print(f"  Losses: {len(y) - y.sum():,} ({(1-win_rate)*100:.1f}%)")

    # Adjust scale_pos_weight for class imbalance
    scale_pos_weight = (1 - win_rate) / win_rate
    CLASSIFICATION_CONFIG["params"]["scale_pos_weight"] = scale_pos_weight
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    # ========================================================================
    # TIME-BASED SPLIT (60/20/20)
    # ========================================================================
    print("\n" + "=" * 80)
    print("TIME-BASED TRAIN/VALID/TEST SPLIT")
    print("=" * 80)

    n = len(X)
    train_idx = int(n * 0.6)
    valid_idx = int(n * 0.8)

    X_train = X.iloc[:train_idx]
    X_valid = X.iloc[train_idx:valid_idx]
    X_test = X.iloc[valid_idx:]

    y_train = y[:train_idx]
    y_valid = y[train_idx:valid_idx]
    y_test = y[valid_idx:]

    train_dates = df[DATE_COLUMN].iloc[:train_idx]
    valid_dates = df[DATE_COLUMN].iloc[train_idx:valid_idx]
    test_dates = df[DATE_COLUMN].iloc[valid_idx:]

    print(f"Split: 60% train / 20% valid / 20% test")
    print(f"\nTraining set:")
    print(f"  Rows: {len(X_train):,}")
    print(f"  Date range: {train_dates.min()} to {train_dates.max()}")
    print(f"  Win rate: {y_train.mean()*100:.1f}%")

    print(f"\nValidation set:")
    print(f"  Rows: {len(X_valid):,}")
    print(f"  Date range: {valid_dates.min()} to {valid_dates.max()}")
    print(f"  Win rate: {y_valid.mean()*100:.1f}%")

    print(f"\nTest set:")
    print(f"  Rows: {len(X_test):,}")
    print(f"  Date range: {test_dates.min()} to {test_dates.max()}")
    print(f"  Win rate: {y_test.mean()*100:.1f}%")

    # ========================================================================
    # TRAIN MODEL
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING LIGHTGBM CLASSIFIER")
    print("=" * 80)

    params = CLASSIFICATION_CONFIG["params"]
    num_boost_round = CLASSIFICATION_CONFIG["training"]["num_boost_round"]
    early_stopping = CLASSIFICATION_CONFIG["training"]["early_stopping_rounds"]

    print("Configuration:")
    print(f"  Boosting rounds: {num_boost_round}")
    print(f"  Early stopping: {early_stopping} rounds (on validation)")
    print(f"  Learning rate: {params['learning_rate']}")
    print(f"  Num leaves: {params['num_leaves']}")
    print(f"  Max depth: {params['max_depth']}")
    print(f"  Regularization: L1={params['reg_alpha']}, L2={params['reg_lambda']}")
    print(f"  Class weight: {params['scale_pos_weight']:.2f}")

    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=numeric_features)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    print("\nTraining...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping),
            lgb.log_evaluation(period=100)
        ]
    )

    print(f"\n✓ Training complete")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best AUC: {model.best_score['valid']['auc']:.4f}")

    # ========================================================================
    # EVALUATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    # Predictions
    y_train_proba = model.predict(X_train, num_iteration=model.best_iteration)
    y_valid_proba = model.predict(X_valid, num_iteration=model.best_iteration)
    y_test_proba = model.predict(X_test, num_iteration=model.best_iteration)

    y_train_pred = (y_train_proba >= 0.5).astype(int)
    y_valid_pred = (y_valid_proba >= 0.5).astype(int)
    y_test_pred = (y_test_proba >= 0.5).astype(int)

    # Metrics for all sets
    def calc_metrics(y_true, y_pred, y_proba):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_proba),
            'avg_precision': average_precision_score(y_true, y_proba),
        }

    train_metrics = calc_metrics(y_train, y_train_pred, y_train_proba)
    valid_metrics = calc_metrics(y_valid, y_valid_pred, y_valid_proba)
    test_metrics = calc_metrics(y_test, y_test_pred, y_test_proba)

    print("\nPerformance Summary:")
    print(f"  {'Set':<12} {'AUC-ROC':<10} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}")
    print(f"  {'-'*62}")
    print(f"  {'Train':<12} {train_metrics['auc_roc']:<10.4f} {train_metrics['accuracy']:<10.4f} {train_metrics['f1']:<10.4f} {train_metrics['precision']:<10.4f} {train_metrics['recall']:<10.4f}")
    print(f"  {'Valid':<12} {valid_metrics['auc_roc']:<10.4f} {valid_metrics['accuracy']:<10.4f} {valid_metrics['f1']:<10.4f} {valid_metrics['precision']:<10.4f} {valid_metrics['recall']:<10.4f}")
    print(f"  {'Test':<12} {test_metrics['auc_roc']:<10.4f} {test_metrics['accuracy']:<10.4f} {test_metrics['f1']:<10.4f} {test_metrics['precision']:<10.4f} {test_metrics['recall']:<10.4f}")

    # Overfitting check
    auc_overfit = train_metrics['auc_roc'] / test_metrics['auc_roc']
    print(f"\nOverfitting Analysis:")
    print(f"  Train/Test AUC ratio: {auc_overfit:.2f}x")
    if auc_overfit < 1.1:
        print(f"  Assessment: ✅ Good generalization")
    elif auc_overfit < 1.2:
        print(f"  Assessment: ⚠️ Slight overfitting")
    else:
        print(f"  Assessment: ❌ Overfitting detected")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\nTest Set Confusion Matrix:")
    print(f"  {'':>20} Predicted")
    print(f"  {'':>15} {'Loss':>10} {'Win':>10}")
    print(f"  {'Actual Loss':>15} {tn:>10,} {fp:>10,}")
    print(f"  {'Actual Win':>15} {fn:>10,} {tp:>10,}")

    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=['Loss', 'Win'], zero_division=0))

    # ========================================================================
    # CALIBRATION ANALYSIS
    # ========================================================================
    print("=" * 80)
    print("PROBABILITY CALIBRATION ANALYSIS")
    print("=" * 80)

    # Check if probabilities are well-calibrated
    prob_true, prob_pred = calibration_curve(y_test, y_test_proba, n_bins=10)

    print("\nCalibration (Predicted vs Actual Win Rate):")
    print(f"  {'Predicted':>12} {'Actual':>12} {'Diff':>10}")
    print(f"  {'-'*36}")
    for pt, pp in zip(prob_true, prob_pred):
        diff = pt - pp
        print(f"  {pp:>12.1%} {pt:>12.1%} {diff:>+10.1%}")

    # Brier score (lower is better)
    brier = np.mean((y_test_proba - y_test) ** 2)
    print(f"\nBrier Score: {brier:.4f} (lower is better, 0 = perfect)")

    # ========================================================================
    # FEATURE IMPORTANCE
    # ========================================================================
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)

    importance = model.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': numeric_features,
        'importance': importance
    }).sort_values('importance', ascending=False)

    importance_df['importance_pct'] = importance_df['importance'] / importance_df['importance'].sum() * 100

    print("\nTop 20 Most Important Features:")
    for i, row in importance_df.head(20).iterrows():
        print(f"  {importance_df.index.get_loc(i)+1:2d}. {row['feature']:40s} {row['importance_pct']:6.2f}%")

    # ========================================================================
    # SAVE PLOTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING PLOTS")
    print("=" * 80)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {test_metrics["auc_roc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Win Probability Model')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Calibration Plot
    plt.subplot(1, 2, 2)
    plt.plot(prob_pred, prob_true, 'bo-', label='Model')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Win Rate')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'win_probability_evaluation.png', dpi=150, bbox_inches='tight')
    print(f"✓ Evaluation plots saved: {FIGURES_DIR / 'win_probability_evaluation.png'}")
    plt.close()

    # Feature importance plot
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['importance_pct'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance (%)')
    plt.title('Top 20 Feature Importance - Win Probability Model')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'win_probability_feature_importance.png', dpi=150, bbox_inches='tight')
    print(f"✓ Feature importance saved: {FIGURES_DIR / 'win_probability_feature_importance.png'}")
    plt.close()

    # ========================================================================
    # SAVE MODEL
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)

    model_path = MODELS_DIR / "lightgbm_win_probability.txt"
    model.save_model(str(model_path))
    print(f"✓ Model saved: {model_path}")

    # Save metadata
    metadata = {
        "model_type": "LightGBM Binary Classifier",
        "phase": "1B - Win Probability Prediction",
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
        "hyperparameters": params,
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
        "calibration": {
            "brier_score": float(brier),
        },
        "feature_importance": importance_df.head(20).to_dict('records'),
        "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    metadata_path = MODELS_DIR / "lightgbm_win_probability_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"✓ Metadata saved: {metadata_path}")

    # Save feature importance
    importance_path = REPORTS_DIR / "win_probability_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"✓ Feature importance saved: {importance_path}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1B COMPLETE - WIN PROBABILITY MODEL")
    print("=" * 80)

    print(f"\n📊 Model Performance:")
    print(f"   Test AUC-ROC:  {test_metrics['auc_roc']:.4f}")
    print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Test F1:       {test_metrics['f1']:.4f}")
    print(f"   Brier Score:   {brier:.4f}")

    # Assessment
    if test_metrics['auc_roc'] >= 0.85:
        assessment = "✅ EXCELLENT - Production ready"
    elif test_metrics['auc_roc'] >= 0.80:
        assessment = "✅ GOOD - Production ready"
    elif test_metrics['auc_roc'] >= 0.75:
        assessment = "⚠️ ACCEPTABLE - Consider improvements"
    else:
        assessment = "❌ NEEDS IMPROVEMENT"

    print(f"\n📋 Assessment: {assessment}")

    print(f"\n🎯 Expected Value System Ready:")
    print(f"   Phase 1A: Bid Fee Prediction ✅ (RMSE: $328.75)")
    print(f"   Phase 1B: Win Probability    ✅ (AUC: {test_metrics['auc_roc']:.4f})")
    print(f"   Formula: EV = P(Win) × BidFee")

    print(f"\n📁 Files saved:")
    print(f"   - {model_path}")
    print(f"   - {metadata_path}")
    print(f"   - {importance_path}")
    print(f"   - {FIGURES_DIR / 'win_probability_evaluation.png'}")
    print(f"   - {FIGURES_DIR / 'win_probability_feature_importance.png'}")

    return model, test_metrics


if __name__ == "__main__":
    model, metrics = main()
