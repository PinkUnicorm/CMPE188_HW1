import torch
import torch.nn as nn
import numpy as np
import sys

def get_task_metadata():
    return {"task_id": "poisson_regression", "description": "Poisson GLM for count-based prediction."}

def make_dataloaders():
    np.random.seed(42)
    X = np.random.randn(1000, 3)
    # y = Poisson(exp(Xw + b))
    rate = np.exp(X @ np.array([0.5, -0.3, 0.2]) + 1.0)
    y = np.random.poisson(rate).astype(float)
    X_t, y_t = torch.FloatTensor(X), torch.FloatTensor(y).view(-1, 1)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=64)

class PoissonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
    def forward(self, x):
        return torch.exp(self.linear(x)) # Exponential link function

def poisson_loss(y_pred, y_true):
    # Loss = mean(y_pred - y_true * log(y_pred))
    return torch.mean(y_pred - y_true * torch.log(y_pred + 1e-8))

if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    loader = make_dataloaders()
    model = PoissonModel()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    
    for _ in range(200):
        for bx, by in loader:
            opt.zero_grad()
            loss = poisson_loss(model(bx), by)
            loss.backward()
            opt.step()
            
    # Verification: Check if predicted mean is close to actual mean
    with torch.no_grad():
        X, y = next(iter(loader))
        y_p = model(X)
        diff = torch.abs(y_p.mean() - y.mean()).item()
        print(f"Poisson Mean Diff: {diff:.4f}")
        sys.exit(0 if diff < 1.0 else 1)