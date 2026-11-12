"""
boundary_ties.py -- are selections at the edge of the grids tie-break effects?

For every outer fold and kernel of the main analysis, recomputes the inner
cross-validation score of every (s or gamma, C) pair exactly as select_inner in
compare_quantum_kernels.py does, checks that the selected pair is the stored
one, and records whether the selected pair lies on the edge of the grid
(smallest s/gamma or largest C) and, if so, whether a pair inside the
grid (s/gamma above the smallest value and C below the largest) reaches the same
inner score within the selection tolerance (1e-12).

Usage: python boundary_ties.py --dataset heart_disease
Writes results/boundary_ties_<dataset>.json
"""
import argparse
import json

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

import compare_quantum_kernels as cq
from foldsafe_pipeline import build_balanced_subset

TOL = 1e-12


def inner_scores(kind, Xtr_raw, ytr, seed):
    grid = cq.GAMMA_GRID if kind == "rbf" else cq.S_GRID
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    splits = [(cq.preprocess(Xtr_raw[a], Xtr_raw[b]), ytr[a], ytr[b]) for a, b in inner.split(Xtr_raw, ytr)]
    scores = {}
    for p in grid:
        for C in cq.C_GRID:
            scores[(p, C)] = float(np.mean([cq.fit_eval(kind, p, C, A, ya, B, yb)[0] for (A, B), ya, yb in splits]))
    best, best_score = None, -1
    for p in grid:                       # same order and tie rule as select_inner
        for C in cq.C_GRID:
            if scores[(p, C)] > best_score + TOL:
                best, best_score = (p, C), scores[(p, C)]
    return scores, best, best_score, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(cq.CACHE))
    args = ap.parse_args()
    stored = json.load(open(f"results/compare_{args.dataset}.json"))["kernels"]
    X, y = build_balanced_subset(*cq.CACHE[args.dataset])
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
    out = {"dataset": args.dataset, "tolerance": TOL, "kernels": {}}
    rows = {k: [] for k in cq.KINDS}
    for f, (tr, te) in enumerate(outer.split(X, y)):
        for k in cq.KINDS:
            scores, best, bs, grid = inner_scores(k, X[tr], y[tr], f)
            assert [best[0], best[1]] == list(stored[k]["params"][f]), (k, f, best, stored[k]["params"][f])
            edge = best[0] == grid[0] or best[1] == cq.C_GRID[-1]
            interior = [sc for (p, C), sc in scores.items() if p > grid[0] and C < cq.C_GRID[-1]]
            tie_inside = edge and max(interior) >= bs - TOL
            rows[k].append({"edge": bool(edge), "interior_tie": bool(tie_inside),
                            "gap_to_best_interior": float(bs - max(interior))})
        if (f + 1) % 10 == 0:
            print(f"  fold {f+1}/100", flush=True)
    for k, r in rows.items():
        n_edge = sum(x["edge"] for x in r); n_tie = sum(x["interior_tie"] for x in r)
        gaps = [x["gap_to_best_interior"] for x in r if x["edge"] and not x["interior_tie"]]
        out["kernels"][k] = {"n_edge": n_edge, "n_edge_with_interior_tie": n_tie,
                             "max_gap_when_no_tie": float(max(gaps)) if gaps else 0.0,
                             "median_gap_when_no_tie": float(np.median(gaps)) if gaps else 0.0, "per_fold": r}
        print(k, n_edge, n_tie, out["kernels"][k]["max_gap_when_no_tie"])
    json.dump(out, open(f"results/boundary_ties_{args.dataset}.json", "w"), indent=1)
    print(f"saved results/boundary_ties_{args.dataset}.json")


if __name__ == "__main__":
    main()
