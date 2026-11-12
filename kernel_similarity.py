"""
kernel_similarity.py -- how close are the tuned quantum kernels to a classical
RBF kernel?

For every outer training fold of compare_quantum_kernels.py (same splits,
same preprocessing, same hyperparameters as selected there), compute the
centered kernel alignment (CKA) between each quantum Gram matrix and
  (a) the RBF Gram matrix at the RBF's own selected gamma, and
  (b) the closest RBF Gram matrix over a dense gamma grid.
(c) the anisotropic Gaussian exp(-d^T M d) with M fitted to the quantum
    kernel itself by least squares on -log K (no labels used).
For contrast, (b) is also reported for each quantum kernel at a fixed,
untuned input scale s = 1.

CKA(K, L) = <HKH, HLH>_F / (||HKH||_F ||HLH||_F), H = I - 11^T/n; 1 = identical
up to scale and centering.

Usage: python kernel_similarity.py --dataset breast_cancer   (after compare_quantum_kernels.py)
Writes results/similarity_<dataset>.json
"""
import argparse
import json

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from compare_quantum_kernels import CACHE, gram, preprocess
from foldsafe_pipeline import build_balanced_subset

QUANTUM = ["fqk_cos2", "zz", "reup2", "reup3"]
DENSE_GAMMA = np.logspace(-5, 1, 61)


def center_unit(K):
    Kc = K - K.mean(0, keepdims=True) - K.mean(1, keepdims=True) + K.mean()
    return Kc / np.linalg.norm(Kc)


def cka(K, L):
    return float((center_unit(K) * center_unit(L)).sum())


def best_rbf(K, rbf_bank):
    Kc = center_unit(K)
    return max(float((Kc * R).sum()) for R in rbf_bank)


def aniso_gauss_fit(K, X):
    """Least-squares fit of -log K_ij ~ (x_i-x_j)^T M (x_i-x_j) over all pairs;
    returns (CKA(K, exp(-d^T M d)), min/max eigenvalue ratio of M)."""
    n, d = X.shape
    iu = np.triu_indices(n, 1)
    D = (X[:, None] - X[None])[iu]
    F = np.einsum("pa,pb->pab", D, D).reshape(len(D), -1)
    t = -np.log(np.clip(K[iu], 1e-300, None))
    m, *_ = np.linalg.lstsq(F, t, rcond=None)
    M = m.reshape(d, d); M = (M + M.T) / 2
    Dall = X[:, None] - X[None]
    G = np.exp(-np.einsum("ija,ab,ijb->ij", Dall, M, Dall))
    ev = np.linalg.eigvalsh(M)
    return cka(K, G), float(ev.min() / ev.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(CACHE))
    args = ap.parse_args()
    comp = json.load(open(f"results/compare_{args.dataset}.json"))
    path, per_class = CACHE[args.dataset]
    X, y = build_balanced_subset(path, per_class)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=comp["repeats"], random_state=0)

    rows = {k: {"cka_selected_rbf": [], "cka_best_rbf": [], "cka_best_rbf_s1": [],
                "cka_aniso_gauss": [], "metric_eig_ratio": []} for k in QUANTUM}
    for f, (tr, te) in enumerate(outer.split(X, y)):
        Xtr, _ = preprocess(X[tr], X[te])
        g_rbf = comp["kernels"]["rbf"]["params"][f][0]
        K_rbf = gram("rbf", Xtr, Xtr, g_rbf)
        bank = [center_unit(gram("rbf", Xtr, Xtr, g)) for g in DENSE_GAMMA]
        for k in QUANTUM:
            s = comp["kernels"][k]["params"][f][0]
            K = gram(k, Xtr, Xtr, s)
            rows[k]["cka_selected_rbf"].append(cka(K, K_rbf))
            rows[k]["cka_best_rbf"].append(best_rbf(K, bank))
            rows[k]["cka_best_rbf_s1"].append(best_rbf(gram(k, Xtr, Xtr, 1.0), bank))
            c, r = aniso_gauss_fit(K, Xtr)
            rows[k]["cka_aniso_gauss"].append(c); rows[k]["metric_eig_ratio"].append(r)
    out = {"dataset": args.dataset, "folds": len(rows[QUANTUM[0]]["cka_best_rbf"]), "kernels": {}}
    print(f"{args.dataset}: CKA with RBF (mean [min])")
    print(f"{'kernel':9s} {'vs selected RBF':>18s} {'vs closest RBF':>18s} {'closest RBF, s=1':>18s} {'aniso. Gaussian':>18s} {'eig ratio':>18s}")
    for k in QUANTUM:
        r = {m: (float(np.mean(v)), float(np.min(v))) for m, v in rows[k].items()}
        out["kernels"][k] = {m: {"mean": a, "min": b} for m, (a, b) in r.items()}
        out["kernels"][k]["n_folds_metric_positive_definite"] = int(np.sum(np.array(rows[k]["metric_eig_ratio"]) > 0))
        print(f"{k:9s} " + " ".join(f"{r[m][0]:10.4f} [{r[m][1]:.3f}]" for m in rows[k]))
    json.dump(out, open(f"results/similarity_{args.dataset}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
