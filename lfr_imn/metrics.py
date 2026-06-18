import torch
import torch.nn as nn

def compute_infidelity(model, X_val, perturb_std=0.1):
    """
    Computes the Local Fidelity Regularization (LFR) penalty (Infidelity) 
    on a validation set.
    """
    model.eval()
    with torch.no_grad():
        y_pred, w, w_0 = model(X_val)
        
        # Generate synthetic neighborhood
        X_neighbor = X_val + torch.randn_like(X_val) * perturb_std
        y_neighbor_true, _, _ = model(X_neighbor)
        
        # Extrapolate using original anchor weights
        y_neighbor_extrapolated = torch.sum(w * X_neighbor, dim=1) + w_0
        infidelity = nn.MSELoss()(y_neighbor_extrapolated, y_neighbor_true).item()
        
    return infidelity

def compute_tuning_score(D, I_base, I_model, alpha_score=0.5):
    """
    Computes the composite score S for hyperparameter tuning.
    
    Args:
        D (float): Bounded predictive performance metric (e.g., R2 or AUROC on [0,1]).
        I_base (float): Infidelity of the unregularized IMN model (gamma=0).
        I_model (float): Infidelity of the current LFR-IMN model (gamma=a).
        alpha_score (float): Trade-off parameter between performance and fidelity.
        
    Returns:
        float: The score S.
    """
    # Compute Fidelity Ratio (R_F)
    R_F = I_base / (I_model + I_base) if (I_model + I_base) > 0 else 1.0
    
    # Compute Final Score S
    S = alpha_score * max(0.0, D) + (1 - alpha_score) * R_F
    return S, R_F