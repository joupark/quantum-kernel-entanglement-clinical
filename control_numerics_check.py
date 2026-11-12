"""
control_numerics_check.py -- sensitivity of hyperparameter selection to
numerically equivalent computations of the same kernel.

The control kernel is computed in compare_quantum_kernels.py by statevector
simulation of V R(x). Mathematically it equals the closed form
prod_i cos^2(s (x_i - x'_i) / 2); the two agree to machine precision. This
script reruns the nested selection and outer evaluation for the control kernel
with the closed form and reports how many outer folds select a different
(s, C) pair and how the mean balanced accuracy changes.

Usage: python control_numerics_check.py --dataset heart_disease   (after compare_quantum_kernels.py)
Writes results/control_numerics_<dataset>.json
"""
import argparse
import json

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

import compare_quantum_kernels as cq
from foldsafe_pipeline import build_balanced_subset

_gram_statevector = cq.gram


def gram_closed_form(kind, Xa, Xb, p):
    if kind == "fqk_cos2":
        return np.prod(np.cos(p * (Xa[:, None, :] - Xb[None, :, :]) / 2) ** 2, axis=-1)
    return _gram_statevector(kind, Xa, Xb, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(cq.CACHE))
    args = ap.parse_args()
    stored = json.load(open(f"results/compare_{args.dataset}.json"))
    X, y = build_balanced_subset(*cq.CACHE[args.dataset])
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=stored["repeats"], random_state=0)

    cq.gram = gram_closed_form          # select_inner and fit_eval look up cq.gram at call time
    params, bal = [], []
    for f, (tr, te) in enumerate(outer.split(X, y)):
        Xtr, Xte = cq.preprocess(X[tr], X[te])
        p, C = cq.select_inner("fqk_cos2", X[tr], y[tr], seed=f)
        b, _ = cq.fit_eval("fqk_cos2", p, C, Xtr, y[tr], Xte, y[te])
        params.append([p, C]); bal.append(b)
    cq.gram = _gram_statevector

    ref = stored["kernels"]["fqk_cos2"]
    differ = [f for f, (a, b) in enumerate(zip(params, ref["params"])) if a != list(b)]
    out = {"dataset": args.dataset,
           "folds_with_different_selection": differ,
           "n_folds_different": len(differ),
           "bal_mean_statevector": ref["bal_mean"],
           "bal_mean_closed_form": float(np.mean(bal))}
    json.dump(out, open(f"results/control_numerics_{args.dataset}.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
