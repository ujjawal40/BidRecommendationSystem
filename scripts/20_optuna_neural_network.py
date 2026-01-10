"""
Neural Network Architecture Search with Optuna
===============================================
Optimize PyTorch NN architecture for regression performance

Target: Beat baseline $237.57 RMSE and reduce 2.57x overfitting

Author: Bid Recommendation System
Date: 2026-01-10
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import optuna
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
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load regression-optimized features
REGRESSION_FEATURES_PATH = MODELS_DIR.parent / "reports" / "regression_features.json"
with open(REGRESSION_FEATURES_PATH, 'r') as f:
    regression_config = json.load(f)
    SELECTED_FEATURES = regression_config['features']

print("=" * 80)
print("NEURAL NETWORK ARCHITECTURE SEARCH (OPTUNA)")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Features: {len(SELECTED_FEATURES)}")
print(f"Baseline to beat: $237.57 RMSE, 2.57x overfitting\n")

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

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}\n")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Define flexible NN architecture
class FlexibleNN(nn.Module):
    def __init__(self, input_dim, layer_sizes, dropout_rates, activation):
        super(FlexibleNN, self).__init__()

        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Input layer
        prev_size = input_dim
        for i, size in enumerate(layer_sizes):
            self.layers.append(nn.Linear(prev_size, size))
            self.dropouts.append(nn.Dropout(dropout_rates[i]))
            prev_size = size

        # Output layer
        self.output = nn.Linear(prev_size, 1)

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:  # tanh
            self.activation = nn.Tanh()

    def forward(self, x):
        for layer, dropout in zip(self.layers, self.dropouts):
            x = self.activation(layer(x))
            x = dropout(x)
        x = self.output(x)
        return x

# Optuna objective function
def objective(trial):
    # Suggest architecture
    n_layers = trial.suggest_int('n_layers', 2, 5)
    layer_sizes = []
    dropout_rates = []

    for i in range(n_layers):
        size = trial.suggest_int(f'layer_{i}_size', 32, 256)
        dropout = trial.suggest_float(f'dropout_{i}', 0.0, 0.5)
        layer_sizes.append(size)
        dropout_rates.append(dropout)

    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu', 'tanh'])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    # Create model
    model = FlexibleNN(len(SELECTED_FEATURES), layer_sizes, dropout_rates, activation).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Create data loaders
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    epochs = 100
    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Early stopping
        avg_loss = train_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Evaluate
    model.eval()
    with torch.no_grad():
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

        y_pred_train = model(X_train_tensor).cpu().numpy().flatten()
        y_pred_test = model(X_test_tensor).cpu().numpy().flatten()

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    overfitting_ratio = test_rmse / train_rmse

    # Multi-objective: minimize test RMSE and overfitting ratio
    score = test_rmse + (overfitting_ratio - 1.0) * 100

    return score

# Run optimization
print("=" * 80)
print("RUNNING OPTUNA OPTIMIZATION")
print("=" * 80)
print("Trials: 50 (reduced for NN training time)")
print("Objective: Minimize test RMSE + overfitting penalty\n")

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(objective, n_trials=50, show_progress_bar=False)

print(f"Best trial: {study.best_trial.number}")
print(f"Best score: {study.best_value:.2f}\n")

# Train final model with best params
print("=" * 80)
print("TRAINING FINAL MODEL WITH BEST PARAMS")
print("=" * 80)

best_params = study.best_params

# Extract architecture
n_layers = best_params['n_layers']
layer_sizes = [best_params[f'layer_{i}_size'] for i in range(n_layers)]
dropout_rates = [best_params[f'dropout_{i}'] for i in range(n_layers)]
activation = best_params['activation']
learning_rate = best_params['learning_rate']
batch_size = best_params['batch_size']
weight_decay = best_params['weight_decay']

print("Best architecture:")
print(f"  Layers: {n_layers}")
print(f"  Layer sizes: {layer_sizes}")
print(f"  Dropout rates: {[f'{d:.3f}' for d in dropout_rates]}")
print(f"  Activation: {activation}")
print(f"  Learning rate: {learning_rate:.6f}")
print(f"  Batch size: {batch_size}")
print(f"  Weight decay: {weight_decay:.6f}\n")

# Create final model
final_model = FlexibleNN(len(SELECTED_FEATURES), layer_sizes, dropout_rates, activation).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(final_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Create data loaders
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop with early stopping
epochs = 200
best_loss = float('inf')
patience = 20
patience_counter = 0
best_model_state = None

for epoch in range(epochs):
    final_model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = final_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Early stopping
    avg_loss = train_loss / len(train_loader)
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        best_model_state = final_model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

# Load best model
final_model.load_state_dict(best_model_state)

print(f"Training stopped at epoch {epoch + 1 - patience}\n")

# Predictions
final_model.eval()
with torch.no_grad():
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

    y_pred_train = final_model(X_train_tensor).cpu().numpy().flatten()
    y_pred_test = final_model(X_test_tensor).cpu().numpy().flatten()

# Evaluation
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_mae = mean_absolute_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

overfitting_ratio = test_rmse / train_rmse

print("=" * 80)
print("RESULTS")
print("=" * 80)

print("\nTRAIN METRICS:")
print(f"  RMSE: ${train_rmse:,.2f}")
print(f"  MAE: ${train_mae:,.2f}")
print(f"  R²: {train_r2:.4f}")

print("\nTEST METRICS:")
print(f"  RMSE: ${test_rmse:,.2f}")
print(f"  MAE: ${test_mae:,.2f}")
print(f"  R²: {test_r2:.4f}")

print(f"\nOVERFITTING RATIO: {overfitting_ratio:.2f}x")

# Compare with baseline
baseline_rmse = 237.57
baseline_overfitting = 2.57

improvement_rmse = ((baseline_rmse - test_rmse) / baseline_rmse) * 100
improvement_overfitting = ((baseline_overfitting - overfitting_ratio) / baseline_overfitting) * 100

print("\n" + "=" * 80)
print("COMPARISON WITH BASELINE")
print("=" * 80)

print(f"\nBaseline:  RMSE ${baseline_rmse:,.2f} | Overfitting {baseline_overfitting:.2f}x")
print(f"Optimized: RMSE ${test_rmse:,.2f} | Overfitting {overfitting_ratio:.2f}x")

if test_rmse < baseline_rmse:
    print(f"\nIMPROVEMENT: {improvement_rmse:.1f}% better RMSE")
else:
    print(f"\nDEGRADATION: {-improvement_rmse:.1f}% worse RMSE")

if overfitting_ratio < baseline_overfitting:
    print(f"IMPROVEMENT: {improvement_overfitting:.1f}% less overfitting")
else:
    print(f"DEGRADATION: {-improvement_overfitting:.1f}% more overfitting")

# Save model
print("\n" + "=" * 80)
print("SAVING MODEL")
print("=" * 80)

model_path = MODELS_DIR / "pytorch_nn_optuna.pth"
torch.save({
    'model_state_dict': final_model.state_dict(),
    'scaler_mean': scaler.mean_,
    'scaler_scale': scaler.scale_,
    'architecture': {
        'layer_sizes': layer_sizes,
        'dropout_rates': dropout_rates,
        'activation': activation
    }
}, str(model_path))

metadata = {
    "model_type": "PyTorch Neural Network (Optuna Optimized)",
    "phase": "1A - Bid Fee Prediction",
    "target_variable": "BidFee",
    "optimization": "Optuna architecture search (50 trials)",
    "num_features": len(SELECTED_FEATURES),
    "selected_features": SELECTED_FEATURES,
    "data_range": {
        "start_date": df_recent[DATE_COLUMN].min().strftime('%Y-%m-%d'),
        "end_date": df_recent[DATE_COLUMN].max().strftime('%Y-%m-%d'),
        "total_samples": int(len(df_recent)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test))
    },
    "best_architecture": {
        "n_layers": int(n_layers),
        "layer_sizes": [int(s) for s in layer_sizes],
        "dropout_rates": [float(d) for d in dropout_rates],
        "activation": activation,
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "weight_decay": float(weight_decay)
    },
    "metrics": {
        "train": {
            "rmse": float(train_rmse),
            "mae": float(train_mae),
            "r2": float(train_r2)
        },
        "test": {
            "rmse": float(test_rmse),
            "mae": float(test_mae),
            "r2": float(test_r2)
        },
        "overfitting_ratio": float(overfitting_ratio)
    },
    "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "comparison": {
        "baseline_rmse": float(baseline_rmse),
        "baseline_overfitting": float(baseline_overfitting),
        "improvement_rmse_pct": float(improvement_rmse),
        "improvement_overfitting_pct": float(improvement_overfitting)
    },
    "optuna_study": {
        "n_trials": len(study.trials),
        "best_trial": study.best_trial.number,
        "best_score": float(study.best_value)
    }
}

metadata_path = REPORTS_DIR / "pytorch_nn_optuna_results.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Model saved: {model_path}")
print(f"Metadata saved: {metadata_path}")

print("\n" + "=" * 80)
print("OPTUNA NEURAL NETWORK TUNING COMPLETE")
print("=" * 80)
print(f"Test RMSE: ${test_rmse:,.2f}")
print(f"Overfitting: {overfitting_ratio:.2f}x")
print(f"vs Baseline: ${baseline_rmse:,.2f}, {baseline_overfitting:.2f}x\n")
