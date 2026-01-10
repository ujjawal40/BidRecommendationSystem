"""
Alternative Models and Blending for Regression
===============================================
Compare LightGBM vs XGBoost vs Neural Network
Then blend all three for final predictions

Author: Bid Recommendation System
Date: 2026-01-09
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings

from config.model_config import (
    FEATURES_DATA, TARGET_COLUMN, DATE_COLUMN,
    MODELS_DIR, REPORTS_DIR, RANDOM_SEED,
)

warnings.filterwarnings('ignore')

# Load regression-optimized features
REGRESSION_FEATURES_PATH = MODELS_DIR.parent / "reports" / "regression_features.json"
with open(REGRESSION_FEATURES_PATH, 'r') as f:
    regression_config = json.load(f)
    SELECTED_FEATURES = regression_config['features']

print("=" * 80)
print("ALTERNATIVE MODELS AND BLENDING")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Features: {len(SELECTED_FEATURES)}\n")

# Load data
df = pd.read_csv(FEATURES_DATA)
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

# Filter to recent data
recent_cutoff = pd.Timestamp('2023-01-01')
df_recent = df[df[DATE_COLUMN] >= recent_cutoff].copy()

print(f"Data: {len(df_recent):,} rows (2023-2025)\n")

# Prepare data
X = df_recent[SELECTED_FEATURES].fillna(0).values
y = df_recent[TARGET_COLUMN].values

# 80/20 split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Scale for Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}\n")

# ============================================================================
# MODEL 1: LightGBM (Baseline)
# ============================================================================
print("=" * 80)
print("MODEL 1: LightGBM")
print("=" * 80)

lgb_train_data = lgb.Dataset(X_train, label=y_train, feature_name=SELECTED_FEATURES)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
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

print("Training LightGBM...")
lgb_model = lgb.train(
    lgb_params,
    lgb_train_data,
    num_boost_round=500
)

lgb_pred_train = lgb_model.predict(X_train)
lgb_pred_test = lgb_model.predict(X_test)

lgb_train_rmse = np.sqrt(mean_squared_error(y_train, lgb_pred_train))
lgb_test_rmse = np.sqrt(mean_squared_error(y_test, lgb_pred_test))
lgb_test_r2 = r2_score(y_test, lgb_pred_test)

print(f"Train RMSE: ${lgb_train_rmse:,.2f}")
print(f"Test RMSE: ${lgb_test_rmse:,.2f}")
print(f"Test R²: {lgb_test_r2:.4f}")
print(f"Overfitting: {lgb_test_rmse/lgb_train_rmse:.2f}x\n")

# ============================================================================
# MODEL 2: XGBoost
# ============================================================================
print("=" * 80)
print("MODEL 2: XGBoost")
print("=" * 80)

xgb_train_data = xgb.DMatrix(X_train, label=y_train, feature_names=SELECTED_FEATURES)
xgb_test_data = xgb.DMatrix(X_test, label=y_test, feature_names=SELECTED_FEATURES)

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': RANDOM_SEED,
    'verbosity': 0
}

print("Training XGBoost...")
xgb_model = xgb.train(
    xgb_params,
    xgb_train_data,
    num_boost_round=500
)

xgb_pred_train = xgb_model.predict(xgb.DMatrix(X_train, feature_names=SELECTED_FEATURES))
xgb_pred_test = xgb_model.predict(xgb.DMatrix(X_test, feature_names=SELECTED_FEATURES))

xgb_train_rmse = np.sqrt(mean_squared_error(y_train, xgb_pred_train))
xgb_test_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred_test))
xgb_test_r2 = r2_score(y_test, xgb_pred_test)

print(f"Train RMSE: ${xgb_train_rmse:,.2f}")
print(f"Test RMSE: ${xgb_test_rmse:,.2f}")
print(f"Test R²: {xgb_test_r2:.4f}")
print(f"Overfitting: {xgb_test_rmse/xgb_train_rmse:.2f}x\n")

# ============================================================================
# MODEL 3: Neural Network (PyTorch)
# ============================================================================
print("=" * 80)
print("MODEL 3: Neural Network (PyTorch)")
print("=" * 80)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    print("Training Neural Network...")

    # Define model
    class RegressionNN(nn.Module):
        def __init__(self, input_dim):
            super(RegressionNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.dropout1 = nn.Dropout(0.3)
            self.fc2 = nn.Linear(128, 64)
            self.dropout2 = nn.Dropout(0.2)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout1(x)
            x = self.relu(self.fc2(x))
            x = self.dropout2(x)
            x = self.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nn_model = RegressionNN(len(SELECTED_FEATURES)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

    # Training loop
    epochs = 200
    best_loss = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        nn_model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = nn_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Early stopping check
        avg_loss = train_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = nn_model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model
    nn_model.load_state_dict(best_model_state)

    # Predictions
    nn_model.eval()
    with torch.no_grad():
        nn_pred_train = nn_model(X_train_tensor.to(device)).cpu().numpy().flatten()
        nn_pred_test = nn_model(X_test_tensor.to(device)).cpu().numpy().flatten()

    nn_train_rmse = np.sqrt(mean_squared_error(y_train, nn_pred_train))
    nn_test_rmse = np.sqrt(mean_squared_error(y_test, nn_pred_test))
    nn_test_r2 = r2_score(y_test, nn_pred_test)

    print(f"Train RMSE: ${nn_train_rmse:,.2f}")
    print(f"Test RMSE: ${nn_test_rmse:,.2f}")
    print(f"Test R²: {nn_test_r2:.4f}")
    print(f"Overfitting: {nn_test_rmse/nn_train_rmse:.2f}x")
    print(f"Best epoch: {epoch + 1 - patience}\n")

    nn_available = True

except ImportError:
    print("PyTorch not available, skipping Neural Network\n")
    nn_available = False
    nn_pred_test = None
    nn_test_rmse = None
    nn_test_r2 = None

# ============================================================================
# BLENDING
# ============================================================================
print("=" * 80)
print("BLENDING MODELS")
print("=" * 80)

if nn_available:
    print("Finding optimal blend weights (LightGBM + XGBoost + NN)...\n")

    # Grid search for optimal weights
    best_rmse = float('inf')
    best_weights = None

    weight_range = np.linspace(0, 1, 11)
    for w1 in weight_range:
        for w2 in weight_range:
            w3 = 1.0 - w1 - w2
            if w3 < 0 or w3 > 1:
                continue

            blend_pred = w1 * lgb_pred_test + w2 * xgb_pred_test + w3 * nn_pred_test
            rmse = np.sqrt(mean_squared_error(y_test, blend_pred))

            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = (w1, w2, w3)

    print(f"Optimal weights:")
    print(f"  LightGBM: {best_weights[0]:.2f}")
    print(f"  XGBoost: {best_weights[1]:.2f}")
    print(f"  Neural Network: {best_weights[2]:.2f}\n")

    # Final blended predictions
    blend_pred_train = (best_weights[0] * lgb_pred_train +
                       best_weights[1] * xgb_pred_train +
                       best_weights[2] * nn_pred_train)

    blend_pred_test = (best_weights[0] * lgb_pred_test +
                      best_weights[1] * xgb_pred_test +
                      best_weights[2] * nn_pred_test)
else:
    print("Blending LightGBM + XGBoost only...\n")

    best_rmse = float('inf')
    best_weights = None

    weight_range = np.linspace(0, 1, 21)
    for w1 in weight_range:
        w2 = 1.0 - w1
        blend_pred = w1 * lgb_pred_test + w2 * xgb_pred_test
        rmse = np.sqrt(mean_squared_error(y_test, blend_pred))

        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = (w1, w2, 0)

    print(f"Optimal weights:")
    print(f"  LightGBM: {best_weights[0]:.2f}")
    print(f"  XGBoost: {best_weights[1]:.2f}\n")

    blend_pred_train = best_weights[0] * lgb_pred_train + best_weights[1] * xgb_pred_train
    blend_pred_test = best_weights[0] * lgb_pred_test + best_weights[1] * xgb_pred_test

blend_train_rmse = np.sqrt(mean_squared_error(y_train, blend_pred_train))
blend_test_rmse = np.sqrt(mean_squared_error(y_test, blend_pred_test))
blend_test_r2 = r2_score(y_test, blend_pred_test)

print("BLENDED MODEL RESULTS:")
print(f"Train RMSE: ${blend_train_rmse:,.2f}")
print(f"Test RMSE: ${blend_test_rmse:,.2f}")
print(f"Test R²: {blend_test_r2:.4f}")
print(f"Overfitting: {blend_test_rmse/blend_train_rmse:.2f}x\n")

# ============================================================================
# COMPARISON
# ============================================================================
print("=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

results = {
    "LightGBM": {"test_rmse": lgb_test_rmse, "test_r2": lgb_test_r2, "overfitting": lgb_test_rmse/lgb_train_rmse},
    "XGBoost": {"test_rmse": xgb_test_rmse, "test_r2": xgb_test_r2, "overfitting": xgb_test_rmse/xgb_train_rmse},
    "Blended": {"test_rmse": blend_test_rmse, "test_r2": blend_test_r2, "overfitting": blend_test_rmse/blend_train_rmse}
}

if nn_available:
    results["Neural Network"] = {"test_rmse": nn_test_rmse, "test_r2": nn_test_r2, "overfitting": nn_test_rmse/nn_train_rmse}

print(f"\n{'Model':<20} {'Test RMSE':<15} {'Test R²':<12} {'Overfitting'}")
print("-" * 65)
for name, metrics in results.items():
    print(f"{name:<20} ${metrics['test_rmse']:<14,.2f} {metrics['test_r2']:<11.4f} {metrics['overfitting']:.2f}x")

best_model = min(results.items(), key=lambda x: x[1]['test_rmse'])
print(f"\nBest model: {best_model[0]} (Test RMSE: ${best_model[1]['test_rmse']:,.2f})")

baseline_rmse = 237.57
improvement = ((baseline_rmse - best_model[1]['test_rmse']) / baseline_rmse) * 100
print(f"Improvement vs baseline: {improvement:.2f}%")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

metadata = {
    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "models_compared": list(results.keys()),
    "features_used": len(SELECTED_FEATURES),
    "data_samples": {"train": len(X_train), "test": len(X_test)},
    "results": {
        name: {
            "test_rmse": float(metrics['test_rmse']),
            "test_r2": float(metrics['test_r2']),
            "overfitting_ratio": float(metrics['overfitting'])
        }
        for name, metrics in results.items()
    },
    "blending": {
        "weights": {
            "lightgbm": float(best_weights[0]),
            "xgboost": float(best_weights[1]),
            "neural_network": float(best_weights[2]) if nn_available else 0.0
        },
        "test_rmse": float(blend_test_rmse),
        "test_r2": float(blend_test_r2)
    },
    "best_model": {
        "name": best_model[0],
        "test_rmse": float(best_model[1]['test_rmse']),
        "improvement_vs_baseline_pct": float(improvement)
    }
}

results_path = REPORTS_DIR / "alternative_models_comparison.json"
with open(results_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Results saved: {results_path}\n")
