import torch
import torch.nn as nn

class MLPBackbone(nn.Module):
    """
    A standard Multilayer Perceptron backbone for IMN.
    """
    def __init__(self, num_features, hidden_dim=64):
        super().__init__()
        self.hypernet = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_features + 1) # Outputs weights + intercept
        )
        # Initialize final layer weights to zero for stability
        nn.init.zeros_(self.hypernet[-1].weight)
        nn.init.zeros_(self.hypernet[-1].bias)

    def forward(self, x):
        out = self.hypernet(x)
        w = out[:, :x.shape[1]]  
        w_0 = out[:, -1]         
        y_pred = torch.sum(w * x, dim=1) + w_0
        return y_pred, w, w_0


class TabResNetBackbone(nn.Module):
    """
    A wrapper for the TabResNet HyperNet from the original IMN codebase.
    Adapts the output to match the standard (y_pred, w, w_0) format.
    """
    def __init__(self, hypernet_model):
        super().__init__()
        self.hypernet = hypernet_model

    def forward(self, x):
        # Expects the HyperNet to be initialized with the same arguments as the original paper
        y_pred, weights_raw = self.hypernet(x, return_weights=True, simple_weights=True)
        y_pred = y_pred.squeeze()
        
        weights_raw = torch.squeeze(weights_raw, dim=2)
        w = weights_raw[:, :-1]
        w_0 = weights_raw[:, -1]
        
        return y_pred, w, w_0