import torch
import torch.nn as nn
import numpy as np
import sys

def make_outlier_data():
    np.random.seed(42)
    X = np.linspace(-5, 5, 500).reshape(-1, 1)
    y = 2 * X + 1 + np.random.randn(*X.shape) * 0.5
    # Add heavy outliers
    y[::50] += 20 
    return torch.FloatTensor(X), torch.FloatTensor(y)

if __name__ == "__main__":
    X, y = make_outlier_data()
    # Model 1: Standard MSE
    model_mse = nn.Linear(1, 1)
    opt_mse = torch.optim.SGD(model_mse.parameters(), lr=0.01)
    # Model 2: Huber Loss
    model_huber = nn.Linear(1, 1)
    opt_huber = torch.optim.SGD(model_huber.parameters(), lr=0.01)
    criterion_huber = nn.HuberLoss(delta=1.0)

    for _ in range(500):
        # Train MSE
        opt_mse.zero_grad()
        l_mse = nn.MSELoss()(model_mse(X), y)
        l_mse.backward()
        opt_mse.step()
        # Train Huber
        opt_huber.zero_grad()
        l_huber = criterion_huber(model_huber(X), y)
        l_huber.backward()
        opt_huber.step()

    w_mse = model_mse.linear.weight.item() if hasattr(model_mse, 'linear') else model_mse.weight.item()
    w_huber = model_huber.weight.item()
    
    print(f"MSE Slope (Target 2.0): {w_mse:.4f}")
    print(f"Huber Slope (Target 2.0): {w_huber:.4f}")
    
    # Huber should be closer to 2.0 than MSE because it ignores outliers better
    huber_error = abs(w_huber - 2.0)
    mse_error = abs(w_mse - 2.0)
    sys.exit(0 if huber_error < mse_error else 1)