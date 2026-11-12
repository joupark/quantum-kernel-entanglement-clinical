"""
robustness_checks.py -- robustness analyses added in revision.

Every check uses the protocol of compare_quantum_kernels.py unchanged (the same
preprocessing, nested selection, tie rule and Nadeau-Bengio test); only the
item named below differs.

  matched   Gate-matched controls. ctrl2 = V R R and ctrl3 = V V R R R use the
            same gates as reup2 = R V R and reup3 = R V R V R, with every CNOT
            chain after all rotation layers, where it cancels. Run on the same
            100 outer folds as the stored results and compared fold by fold
            with the re-uploading kernels.
  subset    A different class-balanced subset (--seed k instead of seed 0),
            evaluated with 5-fold CV repeated --repeats times (default 5).
  cgrid     SVM regularization grid extended to C = 1000 (s and gamma grids
            unchanged), on the same 100 outer folds. Records, per fold, whether
            the outer-test predictions are all one class and the fraction of
            training samples that are support vectors.

Usage:
    python robustness_checks.py matched --dataset heart_disease
    python robustness_checks.py subset  --dataset heart_disease --seed 1
    python robustness_checks.py cgrid   --dataset heart_disease
Writes results/matched_<dataset>.json, results/subset_seed<k>_<dataset>.json,
results/cgrid1000_<dataset>.json
"""
import argparse
import json
import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC

import compare_quantum_kernels as cq
from foldsafe_pipeline import build_balanced_subset
from nested_hybrid_test import nadeau_bengio_test


def nb(diff, n_tr, n_te):
    d, _, p = nadeau_bengio_test(np.asarray(diff), n_tr, n_te)
    lo, hi = cq.corrected_interval(diff, n_tr, n_te)
    return {"diff": float(d), "p": float(p), "ci95": [lo, hi]}


def fit_eval_diag(kind, p, C, Xtr, ytr, Xte, yte):
    """cq.fit_eval plus diagnostics (same computation)."""
    clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
    clf.fit(cq.gram(kind, Xtr, Xtr, p), ytr)
    Kte = cq.gram(kind, Xte, Xtr, p)
    pred = clf.predict(Kte); score = clf.decision_function(Kte)
    return (balanced_accuracy_score(yte, pred), roc_auc_score(yte, score),
            bool(len(set(pred)) == 1), float(len(clf.support_) / len(ytr)))


def run(X, y, kinds, repeats, log):
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=0)
    res = {k: {"bal": [], "auc": [], "params": [], "constant_pred": [], "sv_frac": []} for k in kinds}
    sizes = []
    t0 = time.time()
    for f, (tr, te) in enumerate(outer.split(X, y)):
        sizes.append((len(tr), len(te)))
        Xtr, Xte = cq.preprocess(X[tr], X[te])
        for k in kinds:
            p, C = cq.select_inner(k, X[tr], y[tr], seed=f)
            b, a, const, svf = fit_eval_diag(k, p, C, Xtr, y[tr], Xte, y[te])
            for key, v in zip(("bal", "auc", "params", "constant_pred", "sv_frac"), (b, a, [p, C], const, svf)):
                res[k][key].append(v)
        if (f + 1) % 5 == 0:
            log(f"  fold {f+1}/{5*repeats} ({time.time()-t0:.0f}s) " +
                " ".join(f"{k}={np.mean(res[k]['bal']):.3f}" for k in kinds))
    n_tr = float(np.mean([a for a, _ in sizes])); n_te = float(np.mean([b for _, b in sizes]))
    return res, n_tr, n_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["matched", "subset", "cgrid"])
    ap.add_argument("--dataset", required=True, choices=sorted(cq.CACHE))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    path, per_class = cq.CACHE[args.dataset]
    log = lambda m: print(m, flush=True)

    if args.check == "matched":
        X, y = build_balanced_subset(path, per_class)
        stored = json.load(open(f"results/compare_{args.dataset}.json"))["kernels"]
        res, n_tr, n_te = run(X, y, ["ctrl2", "ctrl3"], 20, log)
        out = {"dataset": args.dataset, "check": "matched", "mean_train_size": n_tr, "mean_test_size": n_te,
               "kernels": {k: {"bal_mean": float(np.mean(v["bal"])), "auc_mean": float(np.mean(v["auc"])),
                               "params": v["params"], "bal_per_fold": v["bal"]} for k, v in res.items()},
               "reuploading_vs_matched": {}}
        for r, c in (("reup2", "ctrl2"), ("reup3", "ctrl3")):
            d = np.array(stored[r]["bal_per_fold"]) - np.array(res[c]["bal"])
            out["reuploading_vs_matched"][r] = nb(d, n_tr, n_te)
        fname = f"results/matched_{args.dataset}.json"

    elif args.check == "subset":
        X, y = build_balanced_subset(path, per_class, seed=args.seed)
        res, n_tr, n_te = run(X, y, cq.KINDS, args.repeats, log)
        out = {"dataset": args.dataset, "check": "subset", "subset_seed": args.seed, "repeats": args.repeats,
               "mean_train_size": n_tr, "mean_test_size": n_te, "kernels": {}}
        for k in cq.KINDS:
            row = {"bal_mean": float(np.mean(res[k]["bal"])), "auc_mean": float(np.mean(res[k]["auc"])),
                   "params": res[k]["params"], "bal_per_fold": res[k]["bal"]}
            if k != "rbf":
                row["vs_rbf"] = nb(np.array(res[k]["bal"]) - np.array(res["rbf"]["bal"]), n_tr, n_te)
            if k in ("reup2", "reup3"):
                row["vs_control"] = nb(np.array(res[k]["bal"]) - np.array(res["fqk_cos2"]["bal"]), n_tr, n_te)
            out["kernels"][k] = row
        fname = f"results/subset_seed{args.seed}_{args.dataset}.json"

    else:
        cq.C_GRID = cq.C_GRID + [1000.0]          # select_inner reads cq.C_GRID at call time
        X, y = build_balanced_subset(path, per_class)
        res, n_tr, n_te = run(X, y, cq.KINDS, 20, log)
        out = {"dataset": args.dataset, "check": "cgrid", "C_grid": cq.C_GRID,
               "mean_train_size": n_tr, "mean_test_size": n_te, "kernels": {}}
        for k in cq.KINDS:
            row = {"bal_mean": float(np.mean(res[k]["bal"])), "auc_mean": float(np.mean(res[k]["auc"])),
                   "params": res[k]["params"], "bal_per_fold": res[k]["bal"],
                   "n_C1000": int(sum(p[1] == 1000.0 for p in res[k]["params"])),
                   "n_constant_pred": int(sum(res[k]["constant_pred"])),
                   "sv_frac_min": float(min(res[k]["sv_frac"])), "sv_frac_max": float(max(res[k]["sv_frac"]))}
            if k != "rbf":
                row["vs_rbf"] = nb(np.array(res[k]["bal"]) - np.array(res["rbf"]["bal"]), n_tr, n_te)
            if k in ("reup2", "reup3"):
                row["vs_control"] = nb(np.array(res[k]["bal"]) - np.array(res["fqk_cos2"]["bal"]), n_tr, n_te)
            out["kernels"][k] = row
        fname = f"results/cgrid1000_{args.dataset}.json"

    json.dump(out, open(fname, "w"), indent=1)
    log(f"saved {fname}")


if __name__ == "__main__":
    main()
