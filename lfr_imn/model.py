import torch
import torch.nn as nn
import numpy as np

class LFR_IMN:
    """
    The Local Fidelity Regularized Interpretable Mesomorphic Neural Network.
    """
    def __init__(self, backbone, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone = backbone.to(self.device)
        
    def fit(self, X_train, y_train, lambda_sparse=0.01, gamma_lfr=0.0, epochs=200, lr=0.01, perturb_std=0.1):
        """
        Trains the LFR-IMN model.
        
        Args:
            X_train (torch.Tensor): Training features.
            y_train (torch.Tensor): Training targets.
            lambda_sparse (float): L1 sparsity penalty for the weights.
            gamma_lfr (float): LFR penalty (gamma) for local fidelity.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
            perturb_std (float): Standard deviation for neighborhood generation.
        """
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        
        optimizer = torch.optim.Adam(self.backbone.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.backbone.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            y_pred, w, w_0 = self.backbone(X_train)
            task_loss = criterion(y_pred, y_train)
            l1_tax = torch.mean(torch.abs(w))
            
            # Apply Local Fidelity Regularization (LFR)
            if gamma_lfr > 0:
                X_neighbor = X_train + torch.randn_like(X_train) * perturb_std
                y_neighbor_true, _, _ = self.backbone(X_neighbor)
                y_neighbor_extrapolated = torch.sum(w * X_neighbor, dim=1) + w_0
                fidelity_loss = criterion(y_neighbor_extrapolated, y_neighbor_true)
            else:
                fidelity_loss = 0.0
                
            loss = task_loss + (gamma_lfr * fidelity_loss) + (lambda_sparse * l1_tax)
            loss.backward()
            optimizer.step()
            
        return self

    def predict(self, X):
        """Returns predictions for the given features."""
        X = X.to(self.device)
        self.backbone.eval()
        with torch.no_grad():
            y_pred, _, _ = self.backbone(X)
        return y_pred.cpu().numpy()

    def explain(self, X):
        """
        Extracts the generated local linear weights for interpretation.
        Returns: 
            w (numpy.ndarray): The feature attributions / gradients.
            w_0 (numpy.ndarray): The local intercepts.
        """
        X = X.to(self.device)
        self.backbone.eval()
        with torch.no_grad():
            _, w, w_0 = self.backbone(X)
        return w.cpu().numpy(), w_0.cpu().numpy()