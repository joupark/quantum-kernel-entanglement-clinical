"""
foldsafe_pipeline.py -- data helpers.

build_balanced_subset(path, per_class) loads a cached feature file (X, y) and
draws a class-balanced subset with a fixed seed. It returns raw features;
standardization and PCA are fit later, inside each cross-validation training
fold (see compare_quantum_kernels.preprocess).

The other functions in this module are not used by the scripts of the paper.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                              confusion_matrix)

SEED = 0


# ---------------------------------------------------------------------------
# Closed-form kernels (Corollary 1 of the paper)
# ---------------------------------------------------------------------------

def fqk_gram_closedform(Xa, Xb, s):
    """Exact FQK Gram matrix for angle encoding: prod_i cos^2(s*(x_i-x'_i)/2).
    Verified to agree with circuit simulation to machine precision."""
    diff = Xa[:, None, :] - Xb[None, :, :]
    return np.prod(np.cos(s * diff / 2) ** 2, axis=-1)


def amplitude_gram_closedform(Xa, Xb):
    """Exact amplitude-encoding fidelity kernel: squared cosine similarity
    of the (unpadded) feature vectors, (x . x' / (|x||x'|))^2."""
    Xa_n = Xa / np.linalg.norm(Xa, axis=1, keepdims=True)
    Xb_n = Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
    return (Xa_n @ Xb_n.T) ** 2


def pqk_gram_circuit(Xa, Xb, gamma, n_qubits, scale=1.0):
    """PQK Gram matrix via genuine circuit simulation (local Pauli
    expectation values) -- NOT subject to the FQK closed-form reduction.

    IMPORTANT: PQK re-uses the same angle-encoding circuit as FQK, U(x) =
    V R(x) with R(x) = tensor_i RY(scale * x_i) -- see the manuscript's
    Methods section ("Unlike the FQK, the entangling layer V does not
    cancel here... These expectation values are computed from the same
    U(x)"). scale MUST match whatever bandwidth s is being tested
    alongside FQK at that setting; omitting it (implicitly scale=1) gives
    identical PQK results regardless of which FQK bandwidth is under
    comparison, which is wrong.

    Implemented with plain numpy (explicit state-vector simulation) rather
    than a quantum-computing library such as PennyLane, to avoid fragile
    version dependencies (this project's original PennyLane installation
    became unusable on an older Python environment during revision). For
    n_qubits <= 10 or so this is fast and exact -- state vectors are at
    most 2^10 = 1024-dimensional, and the entangling layer (built once and
    reused across every sample) is the only expensive part.
    """
    n = n_qubits
    dim = 2 ** n

    def RY(theta):
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]])

    def kron_n(mats):
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    def cnot_chain(n):
        """Full dim x dim matrix for the nearest-neighbor CNOT chain used
        throughout this paper, built once and reused for every sample."""
        V = np.eye(dim)
        for i in range(n - 1):
            full = np.zeros((dim, dim))
            for basis in range(dim):
                bits = [(basis >> (n - 1 - k)) & 1 for k in range(n)]
                if bits[i] == 1:
                    bits[i + 1] ^= 1
                nb = 0
                for b in bits:
                    nb = (nb << 1) | b
                full[nb, basis] = 1
            V = full @ V
        return V

    PAULI = {
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    # Precompute the entangling layer and every single-qubit Pauli operator
    # (embedded in the full n-qubit space) ONCE, reused for every sample --
    # this is what keeps this fast even for hundreds of samples.
    V = cnot_chain(n)
    pauli_ops = []
    for i in range(n):
        for p in ("X", "Y", "Z"):
            ops = [np.eye(2)] * n
            ops[i] = PAULI[p]
            pauli_ops.append(kron_n(ops))

    def local_features(x):
        zero = np.zeros(dim); zero[0] = 1.0
        R = kron_n([RY(scale * xi) for xi in x])
        psi = V @ (R @ zero)
        return np.array([np.real(psi.conj() @ Op @ psi) for Op in pauli_ops])

    Pa = np.array([local_features(x) for x in Xa])
    Pb = np.array([local_features(x) for x in Xb])
    diff = Pa[:, None, :] - Pb[None, :, :]
    return np.exp(-gamma * np.sum(diff ** 2, axis=-1))


# ---------------------------------------------------------------------------
# Data loading: raw features only, no scaling/PCA (moved into cv_eval_foldsafe)
# ---------------------------------------------------------------------------

def build_balanced_subset(feature_cache_path, per_class, seed=SEED):
    """Load a cached npz (X, y) and return a class-balanced RAW subset.
    No StandardScaler, no PCA here -- that now happens per-fold."""
    data = np.load(feature_cache_path, allow_pickle=True)
    X, y = data["X"], data["y"]
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    k = min(per_class, len(pos), len(neg))
    sel = np.concatenate([rng.choice(pos, k, replace=False),
                          rng.choice(neg, k, replace=False)])
    rng.shuffle(sel)
    return X[sel], y[sel]


def sens_spec(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return sens, spec


# ---------------------------------------------------------------------------
# Fold-safe cross-validation
# ---------------------------------------------------------------------------

def cv_eval_foldsafe(X, y, n_qubits, method, folds=5, seed=SEED, **kwargs):
    """
    method: 'classical' | 'fqk' | 'amplitude' | 'pqk'
    kwargs: scale=... for fqk; gamma=... for pqk.

    Scaling and PCA are fit ONLY on each training fold, then applied to
    that fold's test data -- no information from the test fold enters the
    transformation. This is the corrected replacement for the old
    build_subset() pattern that fit these once on the whole subset.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    bal, auc, sens, spec = [], [], [], []

    for tr, te in cv.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        Xtr_s = scaler.transform(X[tr]); Xte_s = scaler.transform(X[te])
        pca = PCA(n_components=n_qubits, random_state=seed).fit(Xtr_s)
        Xtr_r = pca.transform(Xtr_s); Xte_r = pca.transform(Xte_s)

        if method == "classical":
            clf = SVC(kernel="rbf", class_weight="balanced")
            clf.fit(Xtr_r, y[tr])
            pred = clf.predict(Xte_r); score = clf.decision_function(Xte_r)

        else:
            if method == "fqk":
                s = kwargs["scale"]
                K_tr = fqk_gram_closedform(Xtr_r, Xtr_r, s)
                K_te = fqk_gram_closedform(Xte_r, Xtr_r, s)
            elif method == "amplitude":
                K_tr = amplitude_gram_closedform(Xtr_r, Xtr_r)
                K_te = amplitude_gram_closedform(Xte_r, Xtr_r)
            elif method == "pqk":
                gamma = kwargs["gamma"]
                pqk_scale = kwargs.get("scale", 1.0)
                K_tr = pqk_gram_circuit(Xtr_r, Xtr_r, gamma, n_qubits, scale=pqk_scale)
                K_te = pqk_gram_circuit(Xte_r, Xtr_r, gamma, n_qubits, scale=pqk_scale)
            else:
                raise ValueError(f"unknown method {method}")

            clf = SVC(kernel="precomputed", class_weight="balanced")
            clf.fit(K_tr, y[tr])
            pred = clf.predict(K_te); score = clf.decision_function(K_te)

        bal.append(balanced_accuracy_score(y[te], pred))
        auc.append(roc_auc_score(y[te], score))
        s_, sp_ = sens_spec(y[te], pred); sens.append(s_); spec.append(sp_)

    r = lambda a: (round(float(np.mean(a)), 3), round(float(np.std(a)), 3))
    return {"bal_acc": r(bal)[0], "bal_sd": r(bal)[1],
            "auc": r(auc)[0], "auc_sd": r(auc)[1],
            "sens": r(sens)[0], "spec": r(spec)[0]}


if __name__ == "__main__":
    print(__doc__)
