# CMPE 188 Machine Learning Tasks (PyTorch)

This repository contains the implementation of four distinct machine learning tasks using **PyTorch**. Each task is implemented as a self-contained script following the `pytorch_task_v1` protocol, featuring synthetic data generation, model training, evaluation, and automated quality checks.


## 📂 Project Structure

Each task is located in its own subfolder within the `tasks/` directory. Each `task.py` file contains the complete logic for training and validation.

```text
tasks/
├── fourier_linreg/       # Task 1: Linear Regression with Fourier Features
│   └── task.py
├── sparse_logreg/       # Task 2: Sparse Logistic Regression (L1 Regularization)
│   └── task.py
├── poisson_reg/         # Task 3: Poisson Regression for Count Data
│   └── task.py
└── huber_reg/           # Task 4: Robust Linear Regression (Huber Loss)
    └── task.py
