"""
nested_hybrid_test.py -- statistics helper.

nadeau_bengio_test(differences, n_train, n_test) is the corrected paired
t-test of Nadeau and Bengio (2003) for scores from repeated cross-validation:
the variance of the fold-level differences is inflated by (1/n + n_test/n_train)
to account for overlapping training sets. compare_quantum_kernels.py passes
the mean outer-fold training and test sizes.

The other functions in this module are not used by the scripts of the paper.
"""

import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from foldsafe_pipeline import fqk_gram_closedform, pqk_gram_circuit

SEED = 0


def nadeau_bengio_test(differences, n_train, n_test):
    """Corrected paired t-test for repeated/correlated k-fold CV scores
    (Nadeau & Bengio, 2003). Returns (mean_diff, t_stat, p_value)."""
    n = len(differences)
    d_bar = np.mean(differences)
    var_d = np.var(differences, ddof=1)
    if var_d == 0:
        return d_bar, np.inf if d_bar != 0 else 0.0, 0.0 if d_bar != 0 else 1.0
    correction = 1.0 / n + n_test / n_train
    t_stat = d_bar / np.sqrt(correction * var_d)
    df = n - 1
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    return d_bar, t_stat, p_value


def _fit_eval_hybrid(Xtr, ytr, Xte, yte, n_qubits, scale, gamma, alpha):
    """Fit hybrid/PQK/FQK on (Xtr,ytr), evaluate balanced accuracy on
    (Xte,yte). Assumes Xtr/Xte are already scaled+PCA-reduced."""
    K_fqk_tr = fqk_gram_closedform(Xtr, Xtr, scale)
    K_fqk_te = fqk_gram_closedform(Xte, Xtr, scale)
    K_pqk_tr = pqk_gram_circuit(Xtr, Xtr, gamma, n_qubits, scale=scale)
    K_pqk_te = pqk_gram_circuit(Xte, Xtr, gamma, n_qubits, scale=scale)

    results = {}
    for name, K_tr, K_te in [
        ("fqk", K_fqk_tr, K_fqk_te),
        ("pqk", K_pqk_tr, K_pqk_te),
        ("hybrid", alpha * K_fqk_tr + (1 - alpha) * K_pqk_tr,
                   alpha * K_fqk_te + (1 - alpha) * K_pqk_te),
    ]:
        clf = SVC(kernel="precomputed", class_weight="balanced")
        clf.fit(K_tr, ytr)
        results[name] = balanced_accuracy_score(yte, clf.predict(K_te))
    return results


def _select_alpha_inner(Xtr, ytr, n_qubits, scale, gamma, alpha_grid,
                         inner_folds=3, seed=SEED):
    """Inner CV on the outer-training data only, to pick the alpha that
    maximizes mean inner-validation balanced accuracy for the hybrid kernel."""
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True,
                                random_state=seed)
    scores = {a: [] for a in alpha_grid}
    for itr, ival in inner_cv.split(Xtr, ytr):
        Xi_tr, Xi_val = Xtr[itr], Xtr[ival]
        yi_tr, yi_val = ytr[itr], ytr[ival]
        K_fqk_tr = fqk_gram_closedform(Xi_tr, Xi_tr, scale)
        K_fqk_val = fqk_gram_closedform(Xi_val, Xi_tr, scale)
        K_pqk_tr = pqk_gram_circuit(Xi_tr, Xi_tr, gamma, n_qubits, scale=scale)
        K_pqk_val = pqk_gram_circuit(Xi_val, Xi_tr, gamma, n_qubits, scale=scale)
        for a in alpha_grid:
            K_tr = a * K_fqk_tr + (1 - a) * K_pqk_tr
            K_val = a * K_fqk_val + (1 - a) * K_pqk_val
            clf = SVC(kernel="precomputed", class_weight="balanced")
            clf.fit(K_tr, yi_tr)
            scores[a].append(balanced_accuracy_score(yi_val, clf.predict(K_val)))
    mean_scores = {a: np.mean(v) for a, v in scores.items()}
    return max(mean_scores, key=mean_scores.get), mean_scores


def nested_hybrid_significance_test(X, y, n_qubits, scale, gamma,
                                     alpha_grid=None, outer_folds=5,
                                     n_repeats=20, seed=SEED):
    """
    Fold-safe, nested-CV, correctly-corrected significance test of the
    hybrid kernel vs. pure PQK and pure FQK.

    Returns a dict with per-repeat/per-fold outer-test scores, the alphas
    selected by inner CV on each outer-training fold (for transparency --
    these need not all agree), and Nadeau-Bengio corrected (mean diff, t,
    p) for hybrid-vs-PQK and hybrid-vs-FQK.
    """
    if alpha_grid is None:
        alpha_grid = [round(a, 2) for a in np.arange(0.0, 1.01, 0.1)]

    all_fqk, all_pqk, all_hybrid, chosen_alphas = [], [], [], []
    n_train_sizes, n_test_sizes = [], []

    for rep in range(n_repeats):
        cv = StratifiedKFold(n_splits=outer_folds, shuffle=True,
                              random_state=seed + rep)
        for tr, te in cv.split(X, y):
            scaler = StandardScaler().fit(X[tr])
            Xtr_s = scaler.transform(X[tr]); Xte_s = scaler.transform(X[te])
            pca = PCA(n_components=n_qubits, random_state=seed).fit(Xtr_s)
            Xtr_r = pca.transform(Xtr_s); Xte_r = pca.transform(Xte_s)

            best_alpha, _ = _select_alpha_inner(
                Xtr_r, y[tr], n_qubits, scale, gamma, alpha_grid, seed=seed)
            chosen_alphas.append(best_alpha)

            res = _fit_eval_hybrid(Xtr_r, y[tr], Xte_r, y[te],
                                    n_qubits, scale, gamma, best_alpha)
            all_fqk.append(res["fqk"]); all_pqk.append(res["pqk"])
            all_hybrid.append(res["hybrid"])
            n_train_sizes.append(len(tr)); n_test_sizes.append(len(te))

    all_fqk = np.array(all_fqk); all_pqk = np.array(all_pqk)
    all_hybrid = np.array(all_hybrid)
    n_train = int(np.mean(n_train_sizes)); n_test = int(np.mean(n_test_sizes))

    diff_vs_pqk = all_hybrid - all_pqk
    diff_vs_fqk = all_hybrid - all_fqk
    d_pqk, t_pqk, p_pqk = nadeau_bengio_test(diff_vs_pqk, n_train, n_test)
    d_fqk, t_fqk, p_fqk = nadeau_bengio_test(diff_vs_fqk, n_train, n_test)
    _, w_p_pqk = stats.wilcoxon(diff_vs_pqk) if np.any(diff_vs_pqk != 0) else (0, 1.0)
    _, w_p_fqk = stats.wilcoxon(diff_vs_fqk) if np.any(diff_vs_fqk != 0) else (0, 1.0)

    return {
        "n_folds": len(all_hybrid),
        "chosen_alphas": chosen_alphas,
        "alpha_mode": max(set(chosen_alphas), key=chosen_alphas.count),
        "fqk_mean": np.mean(all_fqk), "fqk_sd": np.std(all_fqk),
        "pqk_mean": np.mean(all_pqk), "pqk_sd": np.std(all_pqk),
        "hybrid_mean": np.mean(all_hybrid), "hybrid_sd": np.std(all_hybrid),
        "diff_vs_pqk_mean": d_pqk, "diff_vs_pqk_t": t_pqk, "diff_vs_pqk_p": p_pqk,
        "diff_vs_fqk_mean": d_fqk, "diff_vs_fqk_t": t_fqk, "diff_vs_fqk_p": p_fqk,
        "diff_vs_pqk_wilcoxon_p": w_p_pqk, "diff_vs_fqk_wilcoxon_p": w_p_fqk,
    }
