import sys
import os

# Add the parent directory so Python can find 'lfr_imn'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add the original IMN directory to your path (adjust if your IMN folder is elsewhere)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../IMN')))

import torch
import numpy as np
from sklearn.metrics import mean_squared_error

from lfr_imn.backbones import MLPBackbone, TabResNetBackbone
from lfr_imn.model import LFR_IMN

# ==========================================
# 0. CONFIGURATION TOGGLE
# ==========================================
# Set to True to use TabResNet, False to use MLP
USE_TABRESNET = True

# Hardcoded optimal hyperparameters from the paper
LAMBDA_SPARSE = 0.01
GAMMA_LFR = 10.0 if USE_TABRESNET else 100.0
EPOCHS = 800

# ==========================================
# 1. SYNTHETIC DATA GENERATION
# ==========================================
np.random.seed(42)
torch.manual_seed(42)

N_train, N_test, P = 2000, 500, 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_data(n_samples):
    X = np.random.normal(0, 1, size=(n_samples, P)).astype(np.float32)
    e = np.random.normal(0, 0.1, size=n_samples).astype(np.float32)
    y = 1.0 + X[:, 0] + X[:, 1] + X[:, 0]*X[:, 1] + e
    return torch.tensor(X).to(device), torch.tensor(y).to(device), X, y

X_train, y_train, _, _ = generate_data(N_train)
X_test, y_test, X_test_np, y_test_np = generate_data(N_test)

# ==========================================
# 2. MODEL INITIALIZATION HELPER
# ==========================================
def get_model():
    if USE_TABRESNET:
        from models.hypernetwork import HyperNet
        raw_hypernet = HyperNet(nr_features=P, nr_classes=1, nr_blocks=2, hidden_size=64)
        backbone = TabResNetBackbone(hypernet_model=raw_hypernet)
    else:
        backbone = MLPBackbone(num_features=P)
    return LFR_IMN(backbone, device=device)

# ==========================================
# 3. TRAINING
# ==========================================
backbone_name = "TabResNet" if USE_TABRESNET else "MLP"
print(f"Training using {backbone_name} backbone for {EPOCHS} epochs...\n")

# --- Train Baseline IMN ---
print("1. Training Baseline IMN (gamma=0)...")
imn_model = get_model()
imn_model.fit(X_train, y_train, lambda_sparse=LAMBDA_SPARSE, gamma_lfr=0.0, epochs=EPOCHS)
mse_base = mean_squared_error(y_test_np, imn_model.predict(X_test))

# --- Train LFR-IMN ---
print(f"2. Training Optimal LFR-IMN (gamma={GAMMA_LFR})...")
lfr_model = get_model()
lfr_model.fit(X_train, y_train, lambda_sparse=LAMBDA_SPARSE, gamma_lfr=GAMMA_LFR, epochs=EPOCHS)
mse_lfr = mean_squared_error(y_test_np, lfr_model.predict(X_test))

# ==========================================
# 4. EXPLAINABILITY COMPARISON (Patient 42)
# ==========================================
patient_idx = 42
X_patient = X_test[patient_idx].unsqueeze(0)
x1, x2 = X_test_np[patient_idx, 0], X_test_np[patient_idx, 1]

# Ground truth mathematical gradients: y = 1 + x1 + x2 + x1*x2
true_grad_x1 = 1.0 + x2
true_grad_x2 = 1.0 + x1
true_intercept = 1.0 - (x1 * x2)

w_base, w0_base = imn_model.explain(X_patient)
w_lfr, w0_lfr = lfr_model.explain(X_patient)

# ==========================================
# 5. RESULTS OUTPUT
# ==========================================
print("\n" + "="*80)
print(f" FINAL PERFORMANCE & LOCAL EXPLAINABILITY (Patient {patient_idx} | {backbone_name} Backbone)")
print("="*80)
print(f"Input Features: x1={x1:.4f}, x2={x2:.4f}")
print("Theoretical Optimal Test MSE : ~0.0100 (Based on injected noise)\n")

header_str = f"{'Metric':<14} | {'True':<8} | {'IMN Baseline':<16} | {'LFR-IMN':<20}"
print(header_str)
print("-" * len(header_str))

def print_row(metric_name, true_val, imn_val, slf_val):
    print(f"{metric_name:<14} | {true_val:<8.4f} | {imn_val:<16.4f} | {slf_val:<20.4f}")

print_row("Test MSE", 0.0100, mse_base, mse_lfr)
print_row("Intercept w_0", true_intercept, w0_base[0], w0_lfr[0])
print_row("Grad x1", true_grad_x1, w_base[0][0], w_lfr[0][0])
print_row("Grad x2", true_grad_x2, w_base[0][1], w_lfr[0][1])
print("="*80)