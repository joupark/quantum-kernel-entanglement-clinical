"""
noise_check.py -- simple test of finite measurement statistics and gate noise
(added in revision).

Three noise settings are applied to the four quantum kernels; the RBF baseline
is unaffected. The nested protocol of compare_quantum_kernels.py is unchanged:
every hyperparameter is re-selected under noise by the inner cross-validation.

  shots   Each kernel entry is estimated from a finite number of measurements:
          k_hat = Binomial(n_shots, k) / n_shots, with k the exact (noiseless)
          fidelity, as in the compute-uncompute estimate (default 1000 shots).
          Training diagonal entries are 1. The estimated training matrix is
          symmetric (each pair estimated once) and is passed to the SVM as it
          is, without projecting it onto positive semidefinite matrices; its
          smallest eigenvalue is recorded (train_gram_eig) for every training
          matrix built under shot noise, inner and outer.
  gate    Each encoded state is prepared by the circuit with a depolarizing
          channel after every gate (probability p1 for single-qubit gates, p2 on
          both qubits for CNOT gates; defaults 0.001 and 0.01), and the kernel is
          the state overlap Tr[rho(x) rho(x')]. Simulated with density matrices.
  both    gate noise, then the overlap estimated from n_shots measurements.

Circuits follow compare_quantum_kernels.py gate by gate; the ZZ feature map uses
Qiskit's decomposition (H, P(2 phi_i), and CNOT-P(2 phi_ij)-CNOT for each
neighboring pair). With p1 = p2 = 0 the density-matrix kernels equal the
statevector kernels (checked at start-up).

The test is run on the first 20 of the 100 outer folds of the main analysis
(the first four repetitions), and each quantum kernel is compared fold by fold
with its own noiseless result and with the RBF baseline on the same folds.

Usage: python noise_check.py --dataset heart_disease [--folds 20]
Writes results/noise_<dataset>.json
"""
import argparse
import hashlib
import json
import time

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

import compare_quantum_kernels as cq
from foldsafe_pipeline import build_balanced_subset
from nested_hybrid_test import nadeau_bengio_test

NQ = cq.NQ
QUANTUM = ["fqk_cos2", "reup2", "reup3", "zz"]


# ------------------------------------------------------------ density-matrix simulator
def apply_1q(rho, U, q):
    # ket side: U rho ; bra side: rho U^dagger
    k, b = 1 + q, 1 + NQ + q
    if U.ndim == 2:                                   # same gate for every sample
        rho = np.moveaxis(np.tensordot(U, rho, axes=([1], [k])), 0, k)
        rho = np.moveaxis(np.tensordot(U.conj(), rho, axes=([1], [b])), 0, b)
        return rho
    # per-sample gate U: (N, 2, 2)
    rho = np.moveaxis(rho, (k, b), (1, 2))            # (N, k, b, ...)
    rho = np.einsum("nij,njb...->nib...", U, rho)
    rho = np.einsum("nij,nkj...->nki...", U.conj(), rho)
    return np.moveaxis(rho, (1, 2), (k, b))


def apply_cnot(rho, c, t):
    for side in (0, NQ):                              # permute basis on ket and bra sides
        ci, ti = 1 + side + c, 1 + side + t
        rho = rho.copy()
        idx1 = [slice(None)] * rho.ndim; idx1[ci] = 1
        sub = rho[tuple(idx1)]                        # control = 1 block; target axis shifts by one if t > c
        tax = ti - 1 if ti > ci else ti
        sub = np.flip(sub, axis=tax)
        rho[tuple(idx1)] = sub
    return rho


def depolarize(rho, qubits, p):
    """rho -> (1 - p) rho + p * (I/d on `qubits`) ⊗ Tr_qubits(rho)."""
    if p == 0:
        return rho
    m = len(qubits); d = 2 ** m
    axes = [1 + q for q in qubits] + [1 + NQ + q for q in qubits]
    end = list(range(rho.ndim - 2 * m, rho.ndim))
    r = np.moveaxis(rho, axes, end)
    rest = r.shape[:-2 * m]
    red = np.trace(r.reshape(rest + (d, d)), axis1=-2, axis2=-1)
    mixed = (red[..., None, None] * (np.eye(d) / d)).reshape(rest + (2,) * (2 * m))
    return (1 - p) * rho + p * np.moveaxis(mixed, end, axes)


def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)       # (N, 2, 2)


def phase(phi):
    one = np.ones_like(phi)
    z = np.zeros_like(phi)
    return np.stack([np.stack([one, z], -1), np.stack([z, np.exp(1j * phi)], -1)], -2)


H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def noisy_states(kind, X, s, p1, p2):
    N = X.shape[0]
    rho = np.zeros((N,) + (2,) * (2 * NQ), dtype=complex)
    rho[(slice(None),) + (0,) * (2 * NQ)] = 1
    th = s * X

    def rlayer(r):
        for q in range(NQ):
            r = depolarize(apply_1q(r, ry(th[:, q]), q), [q], p1)
        return r

    def chain(r):
        for q in range(NQ - 1):
            r = depolarize(apply_cnot(r, q, q + 1), [q, q + 1], p2)
        return r

    if kind == "fqk_cos2":
        rho = chain(rlayer(rho))
    elif kind in ("reup2", "reup3"):
        rho = rlayer(rho)
        for _ in range(1 if kind == "reup2" else 2):
            rho = rlayer(chain(rho))
    elif kind == "zz":
        for _ in range(2):
            for q in range(NQ):
                rho = depolarize(apply_1q(rho, H, q), [q], p1)
            for q in range(NQ):
                rho = depolarize(apply_1q(rho, phase(2 * th[:, q]), q), [q], p1)
            for q in range(NQ - 1):
                phi = (np.pi - th[:, q]) * (np.pi - th[:, q + 1])
                rho = depolarize(apply_cnot(rho, q, q + 1), [q, q + 1], p2)
                rho = depolarize(apply_1q(rho, phase(2 * phi), q + 1), [q + 1], p1)
                rho = depolarize(apply_cnot(rho, q, q + 1), [q, q + 1], p2)
    else:
        raise ValueError(kind)
    return rho.reshape(N, -1)


# ------------------------------------------------------------ noisy gram with caching
_cache = {}
_eig = {}       # (setting, kind) -> [(smallest, largest eigenvalue) of each shot-noise training matrix]


def _key(kind, X, s):
    return (kind, float(s), hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest())


def make_gram(setting, p1, p2, shots, rng):
    def states_vec(kind, X, s):
        k = ("pure" if setting == "shots" else "noisy",) + _key(kind, X, s)
        if k not in _cache:     # cached per outer fold, shared by the "gate" and "both" settings
            if setting == "shots":
                _cache[k] = cq.states(kind, X, s)
            else:
                _cache[k] = noisy_states(kind, X, s, p1, p2).astype(np.complex64)
        return _cache[k]

    def gram(kind, Xa, Xb, p):
        if kind == "rbf":
            return cq._gram_exact(kind, Xa, Xb, p)
        A, B = states_vec(kind, Xa, p), states_vec(kind, Xb, p)
        if setting == "shots":
            K = np.abs(A.conj() @ B.T) ** 2
        else:
            K = np.real(A.conj() @ B.T).astype(float)
        if setting in ("shots", "both"):
            same = Xa is Xb or (Xa.shape == Xb.shape and np.array_equal(Xa, Xb))
            Kc = np.clip(K, 0, 1)
            if same:
                iu = np.triu_indices(len(Xa), 1)
                Ks = np.array(Kc)
                v = rng.binomial(shots, Kc[iu]) / shots
                Ks[iu] = v; Ks.T[iu] = v
                np.fill_diagonal(Ks, np.diag(Kc) if setting == "both" else 1.0)
                if setting == "both":   # diagonal (state purity) also estimated
                    np.fill_diagonal(Ks, rng.binomial(shots, np.diag(Kc)) / shots)
                K = Ks
                ev = np.linalg.eigvalsh(K)                 # diagnostic only; no random numbers used
                _eig.setdefault((setting, kind), []).append((float(ev[0]), float(ev[-1])))
            else:
                K = rng.binomial(shots, Kc) / shots
        return K
    return gram


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(cq.CACHE))
    ap.add_argument("--folds", type=int, default=20)
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--p1", type=float, default=0.001)
    ap.add_argument("--p2", type=float, default=0.01)
    args = ap.parse_args()

    # start-up check: noiseless density matrices reproduce the statevector kernels
    rng0 = np.random.default_rng(3); Xc = rng0.normal(size=(5, NQ))
    for kind in QUANTUM:
        for s in (0.05, 0.5, 2.0):
            Kd = np.real(noisy_states(kind, Xc, s, 0, 0).conj() @ noisy_states(kind, Xc, s, 0, 0).T)
            assert np.abs(Kd - cq.gram(kind, Xc, Xc, s)).max() < 1e-12, (kind, s)

    cq._gram_exact = cq.gram
    stored = json.load(open(f"results/compare_{args.dataset}.json"))["kernels"]
    X, y = build_balanced_subset(*cq.CACHE[args.dataset])
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
    splits = list(outer.split(X, y))[:args.folds]
    n_tr = float(np.mean([len(a) for a, _ in splits])); n_te = float(np.mean([len(b) for _, b in splits]))
    out = {"dataset": args.dataset, "folds": args.folds, "shots": args.shots, "p1": args.p1, "p2": args.p2,
           "settings": {}}
    t0 = time.time()
    settings = ("shots", "gate", "both")
    rngs = {st: np.random.default_rng(i) for i, st in enumerate(settings)}
    grams = {st: make_gram(st, args.p1, args.p2, args.shots, rngs[st]) for st in settings}
    res = {st: {k: {"bal": [], "params": []} for k in QUANTUM} for st in settings}
    for f, (tr, te) in enumerate(splits):
        _cache.clear()
        Xtr, Xte = cq.preprocess(X[tr], X[te])
        for st in settings:
            cq.gram = grams[st]
            for k in QUANTUM:
                p, C = cq.select_inner(k, X[tr], y[tr], seed=f)
                b, _ = cq.fit_eval(k, p, C, Xtr, y[tr], Xte, y[te])
                res[st][k]["bal"].append(b); res[st][k]["params"].append([p, C])
        print(f"  fold {f+1}/{args.folds} ({time.time()-t0:.0f}s) " +
              " ".join(f"{st}:{k}={np.mean(res[st][k]['bal']):.3f}" for st in settings for k in QUANTUM), flush=True)
    rbf = np.array(stored["rbf"]["bal_per_fold"][:args.folds])
    for st in settings:
        row = {}
        for k in QUANTUM:
            bal = np.array(res[st][k]["bal"]); clean = np.array(stored[k]["bal_per_fold"][:args.folds])
            d1, _, p1v = nadeau_bengio_test(bal - clean, n_tr, n_te)
            d2, _, p2v = nadeau_bengio_test(bal - rbf, n_tr, n_te)
            row[k] = {"bal_mean": float(bal.mean()), "noiseless_bal_mean": float(clean.mean()),
                      "rbf_bal_mean": float(rbf.mean()), "params": res[st][k]["params"], "bal_per_fold": bal.tolist(),
                      "vs_noiseless": {"diff": float(d1), "p": float(p1v)},
                      "vs_rbf": {"diff": float(d2), "p": float(p2v)}}
            if (st, k) in _eig:
                e = np.array(_eig[(st, k)])
                row[k]["train_gram_eig"] = {
                    "n_matrices": int(len(e)), "n_negative": int(np.sum(e[:, 0] < 0)),
                    "min_smallest": float(e[:, 0].min()), "min_ratio_smallest_to_largest": float((e[:, 0] / e[:, 1]).min())}
        out["settings"][st] = row
    cq.gram = cq._gram_exact
    json.dump(out, open(f"results/noise_{args.dataset}.json", "w"), indent=1)
    print(f"saved results/noise_{args.dataset}.json")


if __name__ == "__main__":
    main()
