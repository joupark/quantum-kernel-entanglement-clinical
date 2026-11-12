"""
compare_quantum_kernels.py -- can a quantum kernel whose entanglement does NOT
cancel beat a tuned classical kernel on classical medical data?

Kernels (all 6 qubits, exact statevector, fidelity kernel |<psi(x')|psi(x)>|^2):
  rbf        classical RBF-SVM (gamma, C tuned)
  fqk_cos2   RY angle encoding + trailing CNOT chain  -> prod cos^2 (entanglement cancels)
  zz         ZZ feature map (Havlicek et al. 2019; Qiskit ZZFeatureMap, reps=2,
             linear entanglement): data-dependent two-qubit phases, nothing cancels
  reup2      RY(s x) - CNOT chain - RY(s x)        (entangler between data layers)
  reup3      RY - CNOT - RY - CNOT - RY            (three data layers)

Protocol:
  * standardization + PCA(6) fit inside each training fold only
  * every hyperparameter (input scale s or RBF gamma, and SVM C) chosen by an
    inner 3-fold CV on the outer training fold (nested CV)
  * outer 5-fold CV repeated R times; identical splits for all kernels
  * each quantum kernel vs RBF, and each re-uploading kernel vs the control:
    Nadeau-Bengio corrected paired t-test (mean train/test fold sizes) and 95% interval

Usage:
    python compare_quantum_kernels.py --dataset breast_cancer
    python compare_quantum_kernels.py --dataset heart_disease
Writes results/compare_<dataset>.json
"""
import argparse
import itertools
import json
import os
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from foldsafe_pipeline import build_balanced_subset
from nested_hybrid_test import nadeau_bengio_test

NQ = 6
CACHE = {
    "breast_cancer": ("cache/features_bc.npz", 150),
    "heart_disease": ("cache/features_hd.npz", 137),   # 137 disease cases after dropping incomplete records
    # same records, nominal variables one-hot encoded (sensitivity check; see prepare_hd.py)
    "heart_disease_onehot": ("cache/features_hd_onehot.npz", 137),
}
S_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
GAMMA_GRID = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
C_GRID = [0.1, 1.0, 10.0, 100.0]

# ----------------------------------------------------------------- states
BITS = np.array(list(itertools.product([0, 1], repeat=NQ)))       # qubit 0 = MSB
H1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
HN = H1
for _ in range(NQ - 1):
    HN = np.kron(HN, H1)


def _cnot_chain_perm():
    """CNOT_{0,1}, CNOT_{1,2}, ... applied in that order, as a basis permutation."""
    b = BITS.copy()
    for i in range(NQ - 1):
        b[:, i + 1] ^= b[:, i]
    idx = b @ (1 << np.arange(NQ - 1, -1, -1))
    P = np.zeros((2 ** NQ, 2 ** NQ))
    P[idx, np.arange(2 ** NQ)] = 1.0
    return P


V = _cnot_chain_perm()


def _ry_layer(theta):
    """Apply RY(theta_i) on each qubit to a batch of states. theta: (N, NQ); psi: (N, 64)."""
    def op(psi):
        N = psi.shape[0]
        psi = psi.reshape((N,) + (2,) * NQ)
        for q in range(NQ):
            c = np.cos(theta[:, q] / 2); s = np.sin(theta[:, q] / 2)
            a0 = np.take(psi, 0, axis=q + 1); a1 = np.take(psi, 1, axis=q + 1)
            shp = (N,) + (1,) * (NQ - 1)
            n0 = c.reshape(shp) * a0 - s.reshape(shp) * a1
            n1 = s.reshape(shp) * a0 + c.reshape(shp) * a1
            psi = np.stack([n0, n1], axis=q + 1)
        return psi.reshape(N, -1)
    return op


def states(kind, X, s):
    N = X.shape[0]
    th = s * X
    zero = np.zeros((N, 2 ** NQ), dtype=complex); zero[:, 0] = 1
    if kind == "fqk_cos2":
        return (V @ _ry_layer(th)(zero).T).T
    if kind in ("reup2", "reup3"):
        psi = _ry_layer(th)(zero)
        for _ in range(1 if kind == "reup2" else 2):
            psi = (V @ psi.T).T
            psi = _ry_layer(th)(psi)
        return psi
    if kind in ("ctrl2", "ctrl3"):
        # gate-matched controls (robustness check): the same gates as reup2 / reup3,
        # with every CNOT chain moved after all rotation layers, where it cancels
        psi = zero
        for _ in range(2 if kind == "ctrl2" else 3):
            psi = _ry_layer(th)(psi)
        for _ in range(1 if kind == "ctrl2" else 2):
            psi = (V @ psi.T).T
        return psi
    if kind == "zz":
        x = th
        phase = 2 * (x @ BITS.T)                                   # sum_i x_i b_i
        for i in range(NQ - 1):
            phi = (np.pi - x[:, i]) * (np.pi - x[:, i + 1])
            phase += 2 * np.outer(phi, BITS[:, i] ^ BITS[:, i + 1])
        D = np.exp(1j * phase)
        psi = zero
        for _ in range(2):                                         # reps = 2
            psi = (HN @ psi.T).T * D
        return psi
    raise ValueError(kind)


def gram(kind, Xa, Xb, p):
    if kind == "rbf":
        d = ((Xa[:, None, :] - Xb[None, :, :]) ** 2).sum(-1)
        return np.exp(-p * d)
    A, B = states(kind, Xa, p), states(kind, Xb, p)
    return np.abs(A.conj() @ B.T) ** 2


def fit_eval(kind, p, C, Xtr, ytr, Xte, yte):
    clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
    clf.fit(gram(kind, Xtr, Xtr, p), ytr)
    Kte = gram(kind, Xte, Xtr, p)
    pred = clf.predict(Kte); score = clf.decision_function(Kte)
    return balanced_accuracy_score(yte, pred), roc_auc_score(yte, score)


def preprocess(Xtr, Xte):
    sc = StandardScaler().fit(Xtr)
    a, b = sc.transform(Xtr), sc.transform(Xte)
    pca = PCA(n_components=NQ, random_state=0).fit(a)
    return pca.transform(a), pca.transform(b)


def select_inner(kind, Xtr_raw, ytr, seed):
    grid = GAMMA_GRID if kind == "rbf" else S_GRID
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    splits = [(preprocess(Xtr_raw[a], Xtr_raw[b]), ytr[a], ytr[b])
              for a, b in inner.split(Xtr_raw, ytr)]
    best, best_score = None, -1
    for p in grid:
        for C in C_GRID:
            sc = np.mean([fit_eval(kind, p, C, A, ya, B, yb)[0]
                          for (A, B), ya, yb in splits])
            if sc > best_score + 1e-12:   # ties keep the first pair: smallest s/gamma, then smallest C
                best, best_score = (p, C), sc
    return best


KINDS = ["rbf", "fqk_cos2", "zz", "reup2", "reup3"]


def corrected_interval(differences, n_train, n_test, level=0.95):
    """Interval of the Nadeau-Bengio corrected t-test: mean +/- t * sqrt((1/n + n_test/n_train) var)."""
    from scipy import stats
    d = np.asarray(differences); n = len(d)
    se = np.sqrt((1.0 / n + n_test / n_train) * d.var(ddof=1))
    h = stats.t.ppf(0.5 + level / 2, n - 1) * se
    return float(d.mean() - h), float(d.mean() + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(CACHE))
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    path, per_class = CACHE[args.dataset]
    X, y = build_balanced_subset(path, per_class)
    print(f"{args.dataset}: {len(y)} samples; 5-fold x {args.repeats} repeats, nested 3-fold")

    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats, random_state=0)
    res = {k: {"bal": [], "auc": [], "params": []} for k in KINDS}
    sizes = []
    t0 = time.time()
    for f, (tr, te) in enumerate(outer.split(X, y)):
        Xtr, Xte = preprocess(X[tr], X[te])
        sizes.append((len(tr), len(te)))
        for k in KINDS:
            p, C = select_inner(k, X[tr], y[tr], seed=f)
            b, a = fit_eval(k, p, C, Xtr, y[tr], Xte, y[te])
            res[k]["bal"].append(b); res[k]["auc"].append(a); res[k]["params"].append([p, C])
        if (f + 1) % 5 == 0:
            print(f"  fold {f+1}/{5*args.repeats}  ({time.time()-t0:.0f}s)  " +
                  "  ".join(f"{k}={np.mean(res[k]['bal']):.3f}" for k in KINDS))

    # Nadeau-Bengio correction uses the training/test sizes; with 274 samples the
    # folds are 219/55 or 220/54, so the mean sizes over all outer folds are used.
    n_tr = float(np.mean([a for a, _ in sizes])); n_te = float(np.mean([b for _, b in sizes]))
    out = {"dataset": args.dataset, "grids": {"s": S_GRID, "gamma": GAMMA_GRID, "C": C_GRID},
           "n": int(len(y)), "repeats": args.repeats, "mean_train_size": n_tr, "mean_test_size": n_te,
           "kernels": {}, "reuploading_vs_control": {}}
    print(f"\n{'kernel':9s} {'bal.acc':>8s} {'AUC':>6s}   vs RBF: diff   p(NB)")
    for k in KINDS:
        bal, auc = np.array(res[k]["bal"]), np.array(res[k]["auc"])
        row = {"bal_mean": float(bal.mean()), "bal_sd": float(bal.std(ddof=1)),
               "auc_mean": float(auc.mean()), "params": res[k]["params"],
               "bal_per_fold": bal.tolist(), "auc_per_fold": auc.tolist()}
        line = f"{k:9s} {bal.mean():8.3f} {auc.mean():6.3f}"
        if k != "rbf":
            d, t, p = nadeau_bengio_test(bal - np.array(res["rbf"]["bal"]), n_tr, n_te)
            lo, hi = corrected_interval(bal - np.array(res["rbf"]["bal"]), n_tr, n_te)
            row.update(diff_vs_rbf=float(d), p_vs_rbf=float(p), ci95_vs_rbf=[lo, hi])
            line += f"   {d:+.3f}  {p:.3f}"
        out["kernels"][k] = row
        print(line)
    ctrl = np.array(res["fqk_cos2"]["bal"])
    for k in ("reup2", "reup3"):
        diff = np.array(res[k]["bal"]) - ctrl
        d, t, p = nadeau_bengio_test(diff, n_tr, n_te)
        lo, hi = corrected_interval(diff, n_tr, n_te)
        out["reuploading_vs_control"][k] = {"diff": float(d), "p": float(p), "ci95": [lo, hi]}
        print(f"{k} - control: {d:+.4f}  p={p:.3f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(f"results/compare_{args.dataset}.json", "w"), indent=1)
    print(f"saved results/compare_{args.dataset}.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
