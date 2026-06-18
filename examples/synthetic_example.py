import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Import from our new package
from lfr_imn.backbones import MLPBackbone
from lfr_imn.model import LFR_IMN
from lfr_imn.metrics import compute_infidelity, compute_tuning_score

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
    return torch.tensor(X).to(device), torch.tensor(y).to(device)

X_train, y_train = generate_data(N_train)
X_test, y_test = generate_data(N_test)

# ==========================================
# 2. TRAINING BASELINE IMN (Unregularized)
# ==========================================
print("Training Baseline IMN (gamma=0)...")
backbone_base = MLPBackbone(num_features=P)
imn_model = LFR_IMN(backbone_base, device=device)

# Fit without LFR penalty
imn_model.fit(X_train, y_train, lambda_sparse=0.01, gamma_lfr=0.0, epochs=500)

# Evaluate Baseline
y_pred_base = imn_model.predict(X_test)
R2_base = r2_score(y_test.cpu().numpy(), y_pred_base)
I_base = compute_infidelity(imn_model.backbone, X_test)

print(f"Baseline R2: {R2_base:.4f} | Baseline Infidelity: {I_base:.4f}\n")

# ==========================================
# 3. TRAINING LFR-IMN & CALCULATING SCORE S
# ==========================================
gamma_val = 100.0  # Assume tuned optimal gamma
print(f"Training LFR-IMN (gamma={gamma_val})...")

backbone_lfr = MLPBackbone(num_features=P)
lfr_model = LFR_IMN(backbone_lfr, device=device)

# Fit WITH LFR penalty
lfr_model.fit(X_train, y_train, lambda_sparse=0.01, gamma_lfr=gamma_val, epochs=500)

# Evaluate LFR
y_pred_lfr = lfr_model.predict(X_test)
R2_lfr = r2_score(y_test.cpu().numpy(), y_pred_lfr)
I_lfr = compute_infidelity(lfr_model.backbone, X_test)

# Compute tuning score S
score_S, R_F = compute_tuning_score(D=R2_lfr, I_base=I_base, I_model=I_lfr, alpha_score=0.5)

print(f"LFR-IMN R2: {R2_lfr:.4f} | LFR Infidelity: {I_lfr:.4f}")
print(f"Fidelity Ratio (R_F): {R_F:.4f} | Final Score (S): {score_S:.4f}\n")

# ==========================================
# 4. EXPLAINABILITY COMPARISON (Patient 42)
# ==========================================
patient_idx = 42
X_patient = X_test[patient_idx].unsqueeze(0)

# Ground truth math from the data generating process: y = 1 + x1 + x2 + x1*x2
x1, x2 = X_patient[0, 0].item(), X_patient[0, 1].item()
true_grad_x1 = 1.0 + x2
true_grad_x2 = 1.0 + x1

w_base, _ = imn_model.explain(X_patient)
w_lfr, _ = lfr_model.explain(X_patient)

print("="*60)
print(f"Feature Attributions for Patient {patient_idx} (x1={x1:.4f}, x2={x2:.4f})")
print("="*60)
print(f"True Gradients : w1 = {true_grad_x1:7.4f}, w2 = {true_grad_x2:7.4f}")
print(f"Baseline IMN   : w1 = {w_base[0][0]:7.4f}, w2 = {w_base[0][1]:7.4f}")
print(f"LFR-IMN        : w1 = {w_lfr[0][0]:7.4f}, w2 = {w_lfr[0][1]:7.4f}")