"""
partition_summary.py -- sizes of every data partition used in the paper.

Recreates the class-balanced subset and the outer and inner cross-validation
splits exactly as compare_quantum_kernels.py does, and records their sizes and
class counts. It also checks that the splits are the ones behind the stored
results: every outer test fold is disjoint from its training fold, each sample
is tested once per repetition, and outer fold 0 reproduces the stored balanced
accuracy of every kernel.

Usage: python partition_summary.py --dataset heart_disease
Writes results/partition_<dataset>.json
"""
import argparse
import json

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

import compare_quantum_kernels as cq
from foldsafe_pipeline import build_balanced_subset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(cq.CACHE))
    args = ap.parse_args()
    path, per_class = cq.CACHE[args.dataset]
    full = np.load(path)
    y_full = full["y"]
    X, y = build_balanced_subset(path, per_class)
    stored = json.load(open(f"results/compare_{args.dataset}.json"))

    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=stored["repeats"], random_state=0)
    out_tr, out_te, out_te_pos, inner_tr, inner_va = [], [], [], [], []
    tested = np.zeros((stored["repeats"], len(y)), int)
    for f, (tr, te) in enumerate(outer.split(X, y)):
        assert not set(tr) & set(te)
        tested[f // 5, te] += 1
        out_tr.append(len(tr)); out_te.append(len(te)); out_te_pos.append(int(y[te].sum()))
        inner = StratifiedKFold(3, shuffle=True, random_state=f)      # as in select_inner
        for a, b in inner.split(X[tr], y[tr]):
            inner_tr.append(len(a)); inner_va.append(len(b))
        if f == 0:                                                     # stored results use these splits
            Xtr, Xte = cq.preprocess(X[tr], X[te])
            for k, r in stored["kernels"].items():
                p, C = r["params"][0]
                b, _ = cq.fit_eval(k, p, C, Xtr, y[tr], Xte, y[te])
                assert abs(b - r["bal_per_fold"][0]) < 1e-12, (k, b, r["bal_per_fold"][0])
    assert np.all(tested == 1)                                         # each sample tested once per repetition

    out = {"dataset": args.dataset,
           "available": int(len(y_full)), "available_pos": int(y_full.sum()),
           "available_neg": int(len(y_full) - y_full.sum()),
           "subset": int(len(y)), "subset_per_class": int(per_class),
           "unused": int(len(y_full) - len(y)), "subset_seed": 0,
           "outer_train_sizes": sorted(set(out_tr)), "outer_test_sizes": sorted(set(out_te)),
           "outer_test_pos_counts": sorted(set(out_te_pos)),
           "inner_train_sizes": sorted(set(inner_tr)), "inner_val_sizes": sorted(set(inner_va)),
           "n_outer_folds": len(out_tr)}
    json.dump(out, open(f"results/partition_{args.dataset}.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
