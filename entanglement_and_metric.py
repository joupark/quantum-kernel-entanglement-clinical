"""
entanglement_and_metric.py -- two diagnostics for the kernels of
compare_quantum_kernels.py.

1. Small-scale metric at x = 0: the Fubini-Study metric g of the encoded state,
   so that k = 1 - s^2 d^T g d + O(s^3) near x = 0.
   Reports the eigenvalue ratio min/max (1 = isotropic).
2. Entanglement of the encoded states at the input scales actually selected by
   nested cross-validation: the Meyer-Wallach measure
   Q = 2 (1 - (1/n) sum_k Tr rho_k^2)  (0 = product state, 1 = maximal),
   averaged over the outer training samples, for every outer fold.

Usage: python entanglement_and_metric.py                          (all datasets with results)
       python entanglement_and_metric.py --datasets heart_disease
Writes results/entanglement_and_metric[_<datasets>].json
"""
import argparse
import json
import os

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from compare_quantum_kernels import CACHE, NQ, gram, preprocess, states
from foldsafe_pipeline import build_balanced_subset

QUANTUM = ["fqk_cos2", "zz", "reup2", "reup3"]


def metric_at_zero(kind, h=1e-5):
    """Fubini-Study metric g_ab = Re[<d_a psi|d_b psi> - <d_a psi|psi><psi|d_b psi>]
    of the encoded state at x = 0 (s = 1; s enters the kernel only as s^2), with
    the state derivatives taken by central differences. Then
    k(x, x') = 1 - s^2 (x - x')^T g (x - x') + O(s^3) near x = 0."""
    x0 = np.zeros((1, NQ))
    psi = states(kind, x0, 1.0)[0]
    E = np.eye(NQ) * h
    dpsi = [(states(kind, E[a][None, :], 1.0)[0] - states(kind, -E[a][None, :], 1.0)[0]) / (2 * h)
            for a in range(NQ)]
    g = np.zeros((NQ, NQ))
    for a in range(NQ):
        for b in range(NQ):
            g[a, b] = np.real(np.vdot(dpsi[a], dpsi[b]) - np.vdot(dpsi[a], psi) * np.vdot(psi, dpsi[b]))
    return (g + g.T) / 2


def meyer_wallach(psi):
    """psi: (N, 2^NQ) -> Q per state."""
    N = psi.shape[0]
    T = psi.reshape((N,) + (2,) * NQ)
    purities = []
    for k in range(NQ):
        A = np.moveaxis(T, k + 1, 1).reshape(N, 2, -1)
        rho = np.einsum("nia,nja->nij", A, A.conj())
        purities.append(np.real(np.einsum("nij,nji->n", rho, rho)))
    return 2 * (1 - np.mean(purities, axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="default: every dataset with a results/compare_<dataset>.json")
    args = ap.parse_args()
    datasets = args.datasets or [d for d in CACHE if os.path.exists(f"results/compare_{d}.json")]
    out = {"metric_at_zero": {}, "entanglement": {}}
    print("small-scale metric at x=0: eigenvalue ratio min/max")
    for k in QUANTUM:
        ev = np.linalg.eigvalsh(metric_at_zero(k))
        out["metric_at_zero"][k] = {"eigenvalues": ev.tolist(), "ratio": float(ev.min() / ev.max())}
        print(f"  {k:9s} ratio {ev.min()/ev.max():.3f}  eigenvalues {np.round(ev, 3)}")

    for ds in datasets:
        comp = json.load(open(f"results/compare_{ds}.json"))
        X, y = build_balanced_subset(*CACHE[ds])
        outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=comp["repeats"], random_state=0)
        mw = {k: [] for k in QUANTUM}
        for f, (tr, te) in enumerate(outer.split(X, y)):
            Xtr, _ = preprocess(X[tr], X[te])
            for k in QUANTUM:
                s = comp["kernels"][k]["params"][f][0]
                mw[k].append(float(meyer_wallach(states(k, Xtr, s)).mean()))
        out["entanglement"][ds] = {k: {"mean": float(np.mean(v)), "min": float(np.min(v)),
                                       "max": float(np.max(v)), "per_fold": v} for k, v in mw.items()}
        print(f"{ds}: Meyer-Wallach entanglement at selected s (mean [min, max] over folds)")
        for k, v in mw.items():
            print(f"  {k:9s} {np.mean(v):.3f} [{np.min(v):.3f}, {np.max(v):.3f}]")
    tag = "" if args.datasets is None else "_" + "_".join(datasets)
    json.dump(out, open(f"results/entanglement_and_metric{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
