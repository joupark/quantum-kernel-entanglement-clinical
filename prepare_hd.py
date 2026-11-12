"""
prepare_hd.py -- builds cache/features_hd.npz for the UCI Cleveland Heart
Disease data (heart-disease results of the paper).

Source: the ISLR textbook copy of the Cleveland data (Heart.csv), which keeps
the original diagnosis labels (AHD = Yes/No, i.e. UCI num > 0). Categorical
strings are mapped back to the UCI numeric codes, and the six records with a
missing `ca` or `thal` value are dropped, leaving 297 records (137 disease).

A second file, cache/features_hd_onehot.npz, has the same records in the same
order but one-hot encodes the four nominal variables (chest-pain type, resting
ECG, slope of the ST segment, thal); it is used only for the sensitivity check
of the encoding (compare_quantum_kernels.py --dataset heart_disease_onehot).

Label convention: y = 1 means disease present (positive class).

Note: do NOT use the widely circulated Kaggle "heart.csv" copy of this data.
Its `target` column is inverted relative to UCI and its missing values are
coded as ordinary values (ca = 4, thal = 0).

Usage:
    python prepare_hd.py                      # downloads Heart.csv
    python prepare_hd.py --csv path/to/Heart.csv
"""
import argparse
import io
import os
import numpy as np
import pandas as pd

URL = ("https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/"
       "master/Notebooks/Data/Heart.csv")
FEATURES = ["Age", "Sex", "ChestPain", "RestBP", "Chol", "Fbs", "RestECG",
            "MaxHR", "ExAng", "Oldpeak", "Slope", "Ca", "Thal"]
NOMINAL = ["ChestPain", "RestECG", "Slope", "Thal"]


def load(csv_path=None, onehot=False):
    if csv_path:
        df = pd.read_csv(csv_path, index_col=0)
    else:
        import urllib.request
        with urllib.request.urlopen(URL, timeout=30) as r:
            df = pd.read_csv(io.BytesIO(r.read()), index_col=0)
    df["ChestPain"] = df["ChestPain"].map(
        {"typical": 1, "nontypical": 2, "nonanginal": 3, "asymptomatic": 4})
    df["Thal"] = df["Thal"].map({"normal": 3, "fixed": 6, "reversable": 7})
    df["y"] = (df["AHD"] == "Yes").astype(int)
    df = df.dropna(subset=["Ca", "Thal"])
    if onehot:
        Xdf = pd.get_dummies(df[FEATURES], columns=NOMINAL, dtype=float)
        X = Xdf.to_numpy(dtype=float)
    else:
        X = df[FEATURES].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)
    assert not np.isnan(X).any()
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="local copy of ISLR Heart.csv")
    args = ap.parse_args()
    X, y = load(args.csv)
    os.makedirs("cache", exist_ok=True)
    np.savez("cache/features_hd.npz", X=X, y=y)
    print(f"X {X.shape}; disease = {y.sum()}, no disease = {(1 - y).sum()}")
    print("saved cache/features_hd.npz")
    Xo, yo = load(args.csv, onehot=True)
    assert np.array_equal(yo, y)
    np.savez("cache/features_hd_onehot.npz", X=Xo, y=yo)
    print(f"one-hot: X {Xo.shape}; saved cache/features_hd_onehot.npz")


if __name__ == "__main__":
    main()
