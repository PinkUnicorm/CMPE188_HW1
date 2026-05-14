import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# --- Constants ---
SAMPLES = 1000
FEAT_DIM = 50
INFORM_FEAT = 5
L1_LAMBDA = 0.1  # Strength of the sparsity penalty
EPOCHS = 100
BATCH_SIZE = 32
LR = 0.01

def get_task_metadata():
    return {
        "task_id": "sparse_logreg_l1",
        "description": "L1-regularized Logistic Regression for feature selection on high-dim noise data."
    }

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    np.random.seed(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dataloaders():
    set_seed()
    # Generate synthetic data
    X = np.random.randn(SAMPLES, FEAT_DIM)
    # Only the first 5 features determine the outcome
    weights = np.array([2, -2, 1.5, -1, 3])
    logits = X[:, :INFORM_FEAT] @ weights
    # Labels (0 or 1)
    y = (1 / (1 + np.exp(-logits)) > 0.5).astype(float)
    
    # Convert to Tensors
    X_train = torch.FloatTensor(X[:800])
    y_train = torch.FloatTensor(y[:800]).view(-1, 1)
    X_val = torch.FloatTensor(X[800:])
    y_val = torch.FloatTensor(y[800:]).view(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader

def build_model():
    # Simple linear layer (Logistic Regression)
    return nn.Linear(FEAT_DIM, 1)

def train(model, loader, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    for epoch in range(EPOCHS):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            
            output = model(bx)
            loss = criterion(output, by)
            
            # --- THE SPARSE MAGIC: L1 Regularization ---
            # We add the absolute value of weights to the loss
            l1_penalty = sum(p.abs().sum() for p in model.parameters())
            total_loss = loss + L1_LAMBDA * l1_penalty
            
            total_loss.backward()
            optimizer.step()

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            # Logits to probabilities to binary labels
            probs = torch.sigmoid(model(bx))
            preds = (probs > 0.5).float()
            all_preds.append(preds.cpu())
            all_targets.append(by)
            
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    acc = accuracy_score(all_targets, all_preds)
    
    # Check how many weights were "pushed" to zero (sparsity check)
    weights = model.weight.detach().cpu().numpy().flatten()
    zero_weights = np.sum(np.abs(weights) < 1e-2)
    
    return {"accuracy": float(acc), "zeroed_features": int(zero_weights)}

def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        return torch.sigmoid(model(X_t)).cpu().numpy()

def save_artifacts(metrics):
    os.makedirs("output", exist_ok=True)
    with open("output/sparse_metrics.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")
    
    train_loader, val_loader = make_dataloaders()
    model = build_model()
    
    print("Training Sparse Logistic Regression...")
    train(model, train_loader, device)
    
    print("Evaluating...")
    results = evaluate(model, val_loader, device)
    print(f"Results: {results}")
    
    save_artifacts(results)
    
    # Quality Checks
    # 1. Accuracy must be decent (> 80%)
    # 2. At least 30 of the 50 features should be effectively zero
    pass_acc = results["accuracy"] > 0.8
    pass_sparse = results["zeroed_features"] > 30
    
    if pass_acc and pass_sparse:
        print("PASS: Model is accurate and sparse!")
        sys.exit(0)
    else:
        print(f"FAIL: Accuracy Pass: {pass_acc}, Sparsity Pass: {pass_sparse}")
        sys.exit(1)