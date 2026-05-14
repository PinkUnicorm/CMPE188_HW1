import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error

# Constants
INPUT_DIM = 1  # Raw X
FOURIER_ORDER = 5 
TOTAL_FEATURES = 1 + (FOURIER_ORDER * 2) # X + sin/cos terms
TRAIN_SAMPLES = 800
VAL_SAMPLES = 200
# Updated Constants
EPOCHS = 300 
LR = 0.02
def get_task_metadata():
    return {
        "task_id": "linreg_fourier_features",
        "description": "Linear Regression with Fourier basis expansion for periodic data fitting."
    }

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dataloaders():
    set_seed()
    X = np.linspace(0, 10, TRAIN_SAMPLES + VAL_SAMPLES).reshape(-1, 1)
    y = 0.5 * X + np.sin(1.5 * X) + 0.1 * np.random.randn(*X.shape)
    
    X_expanded = [X]
    for i in range(1, FOURIER_ORDER + 1):
        X_expanded.append(np.sin(i * X))
        X_expanded.append(np.cos(i * X))
    X_final = np.hstack(X_expanded)

    # --- THE FIX: SHUFFLE THE INDICES ---
    indices = np.arange(len(X_final))
    np.random.shuffle(indices)
    X_final = X_final[indices]
    y = y[indices]

    X_train, X_val = X_final[:TRAIN_SAMPLES], X_final[TRAIN_SAMPLES:]
    y_train, y_val = y[:TRAIN_SAMPLES], y[TRAIN_SAMPLES:]

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    return DataLoader(train_ds, batch_size=32, shuffle=True), DataLoader(val_ds, batch_size=32), X_val, y_val
def build_model():
    return nn.Linear(TOTAL_FEATURES, 1)

def train(model, train_loader, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    for _ in range(EPOCHS):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

def evaluate(model, val_loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            preds.append(model(bx).cpu())
            targets.append(by)
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return {"mse": mean_squared_error(targets, preds), "r2": r2_score(targets, preds)}

def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X).to(device)).cpu().numpy()

def save_artifacts(metrics):
    os.makedirs("output", exist_ok=True)
    with open("output/metrics_fourier.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    device = get_device()
    train_loader, val_loader, X_v, y_v = make_dataloaders()
    model = build_model()
    train(model, train_loader, device)
    metrics = evaluate(model, val_loader, device)
    print(f"Fourier LinReg Metrics: {metrics}")
    save_artifacts(metrics)
    sys.exit(0 if metrics["r2"] > 0.9 else 1)