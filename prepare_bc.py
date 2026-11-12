"""
prepare_bc.py -- creates cache/features_bc.npz for the Wisconsin Diagnostic
Breast Cancer dataset. No download needed -- it's built into scikit-learn.

Usage:
    python prepare_bc.py
"""
import os
import numpy as np
from sklearn.datasets import load_breast_cancer

d = load_breast_cancer()
X, y = d.data, d.target
print(f"X shape: {X.shape}, classes: {(y==0).sum()} negative / {(y==1).sum()} positive")

os.makedirs("cache", exist_ok=True)
np.savez("cache/features_bc.npz", X=X, y=y)
print("Saved to cache/features_bc.npz")
