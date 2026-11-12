"""
gaussian_surrogate.py -- classical Gaussian kernels built from the small-scale
metric of each circuit, used as classifiers (added in revision).

For each quantum kernel, the classical kernel is
    k(x, x') = exp(-s^2 (x - x')^T g (x - x')),
where g is the Fubini-Study metric of the circuit's encoded state at x = 0
(entanglement_and_metric.metric_at_zero). This is the leading-order small-scale
form of the quantum kernel (Eq. 3 of the paper); it is computed from the circuit
once, is not fitted to the data, and needs no quantum circuit when classifying.
g is positive semidefinite, so k is a valid kernel.

s and C are selected by the protocol of compare_quantum_kernels.py unchanged
(same s grid, C grid, tie rule, inner and outer folds), on the same 100 outer
folds as the stored results, and each Gaussian kernel is compared fold by fold
with its quantum kernel and with the RBF baseline (Nadeau-Bengio test).

Usage: python gaussian_surrogate.py --dataset heart_disease
Writes results/gaussian_surrogate_<dataset>.json
"""
import argparse
import json

import numpy as np

import compare_quantum_kernels as cq
from entanglement_and_metric import metric_at_zero
from foldsafe_pipeline import build_balanced_subset
from robustness_checks import nb, run

QUANTUM = ["fqk_cos2", "reup2", "reup3", "zz"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["breast_cancer", "heart_disease"])
    args = ap.parse_args()
    log = lambda m: print(m, flush=True)

    G = {k: metric_at_zero(k) for k in QUANTUM}
    for k in QUANTUM:
        ev = np.linalg.eigvalsh(G[k])
        assert ev.min() > 0, (k, ev)                      # positive definite metric
        log(f"g(0) {k}: eigenvalues {np.round(ev, 4)}")

    exact = cq.gram

    def gram(kind, Xa, Xb, p):
        if kind.startswith("g0_"):
            d = Xa[:, None, :] - Xb[None, :, :]
            return np.exp(-p ** 2 * np.einsum("ija,ab,ijb->ij", d, G[kind[3:]], d))
        return exact(kind, Xa, Xb, p)

    cq.gram = gram                                         # read at call time by select_inner / fit_eval
    X, y = build_balanced_subset(*cq.CACHE[args.dataset])
    stored = json.load(open(f"results/compare_{args.dataset}.json"))["kernels"]
    res, n_tr, n_te = run(X, y, ["g0_" + k for k in QUANTUM], 20, log)
    cq.gram = exact

    rbf = np.array(stored["rbf"]["bal_per_fold"])
    out = {"dataset": args.dataset, "check": "gaussian_surrogate", "mean_train_size": n_tr, "mean_test_size": n_te,
           "kernels": {}}
    for k in QUANTUM:
        r = res["g0_" + k]; bal = np.array(r["bal"])
        out["kernels"][k] = {
            "metric_g0": G[k].tolist(), "metric_eigenvalues": np.linalg.eigvalsh(G[k]).tolist(),
            "bal_mean": float(bal.mean()), "auc_mean": float(np.mean(r["auc"])), "params": r["params"],
            "bal_per_fold": bal.tolist(), "quantum_bal_mean": stored[k]["bal_mean"],
            "n_constant_pred": int(sum(r["constant_pred"])),
            "vs_quantum": nb(bal - np.array(stored[k]["bal_per_fold"]), n_tr, n_te),
            "vs_rbf": nb(bal - rbf, n_tr, n_te)}
        log(f"{k}: Gaussian {bal.mean():.4f}  quantum {stored[k]['bal_mean']:.4f}  "
            f"diff {out['kernels'][k]['vs_quantum']['diff']:+.4f} p={out['kernels'][k]['vs_quantum']['p']:.3f}  "
            f"vs RBF {out['kernels'][k]['vs_rbf']['diff']:+.4f} p={out['kernels'][k]['vs_rbf']['p']:.3f}")
    fname = f"results/gaussian_surrogate_{args.dataset}.json"
    json.dump(out, open(fname, "w"), indent=1)
    log(f"saved {fname}")


if __name__ == "__main__":
    main()
