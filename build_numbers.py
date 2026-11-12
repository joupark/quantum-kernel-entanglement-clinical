"""
build_numbers.py -- generate every number in the manuscript from the result files.

Reads
  results/compare_<dataset>.json            (compare_quantum_kernels.py)
  results/similarity_<dataset>.json         (kernel_similarity.py)
  results/entanglement_and_metric.json      (entanglement_and_metric.py)
  results/control_numerics_<dataset>.json   (control_numerics_check.py)
  results/partition_<dataset>.json          (partition_summary.py)
  results/matched_, subset_seed<k>_, cgrid1000_<dataset>.json (robustness_checks.py)
  results/noise_<dataset>.json              (noise_check.py)
and writes, into --out (default: paper/),
  numbers.tex      LaTeX macros used in the text
  table_perf.tex   Table: performance vs RBF
  table_reup.tex   Table: re-uploading vs control
  table_hyper.tex  Table: selected hyperparameters
  table_sim.tex    Table: kernel alignment with classical Gaussian kernels
  table_ent.tex    Table: Meyer-Wallach entanglement
  table_partition.tex  Table: data partitioning
  table_robust.tex Table: robustness checks

Every qualitative statement in the text that depends on the results is checked
here with an assertion; the script stops if any of them is false.
Bounds stated as "within x" or "at least x" are rounded outward so that the
statement stays true after rounding.

Usage: python build_numbers.py [--out ../paper2]
"""
import argparse
import json
import math
import os

import numpy as np

from nested_hybrid_test import nadeau_bengio_test

DATASETS = [("heart_disease", "Heart disease"), ("breast_cancer", "Breast cancer")]
QUANTUM = ["fqk_cos2", "reup2", "reup3", "zz"]
ROTATION = ["fqk_cos2", "reup2", "reup3"]
NAME = {"rbf": "RBF (classical)", "fqk_cos2": "Control ($V R$)", "reup2": "Re-uploading, 2 layers",
        "reup3": "Re-uploading, 3 layers", "zz": "ZZ feature map"}
ENT = {"rbf": "--", "fqk_cos2": "no", "reup2": "yes", "reup3": "yes", "zz": "yes"}
ALPHA = 0.05
S_MIN, GAMMA_MIN, C_MAX = 0.005, 0.0001, 100.0


# ------------------------------------------------------------ rounding helpers
def up(x, d):      # round up (outward for an upper bound)
    return math.ceil(x * 10 ** d - 1e-9) / 10 ** d


def down(x, d):    # round down (outward for a lower bound)
    return math.floor(x * 10 ** d + 1e-9) / 10 ** d


def rnd(x, d):
    """Round half away from zero (a value stored as an exact half, e.g. -0.0225, is shown as -0.023)."""
    return round(x + math.copysign(1e-12, x), d) + 0.0


def f(x, d):
    return f"{rnd(x, d):.{d}f}"


def sgn(x, d=3):
    r = rnd(x, d)
    if r == 0:
        return "$" + f(0.0, d) + "$"
    return ("$+" if r > 0 else "$-") + f(abs(r), d) + "$"


def sgn_plain(x, d=3):
    r = rnd(x, d)
    if r == 0:
        return f(0.0, d)
    return ("+" if r > 0 else "-") + f(abs(r), d)


def num(x, d=3):
    """Signed number for running text, minus sign typeset in math mode."""
    r = rnd(x, d)
    return (f"$-${f(abs(r), d)}" if r < 0 else f(r, d))


def pval(p):
    return f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"


def med(v):
    m = float(np.median(v))
    return f"{m:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper")
    out_dir = ap.parse_args().out
    os.makedirs(out_dir, exist_ok=True)

    C = {d: json.load(open(f"results/compare_{d}.json")) for d, _ in DATASETS}
    S = {d: json.load(open(f"results/similarity_{d}.json")) for d, _ in DATASETS}
    E = json.load(open("results/entanglement_and_metric.json"))
    NUM = {d: json.load(open(f"results/control_numerics_{d}.json")) for d, _ in DATASETS}
    P = {d: json.load(open(f"results/partition_{d}.json")) for d, _ in DATASETS}
    for d, _ in DATASETS:   # partition sizes agree with the comparison runs
        assert P[d]["subset"] == C[d]["n"] and P[d]["n_outer_folds"] == 100
        assert abs(np.mean(P[d]["outer_train_sizes"]) - C[d]["mean_train_size"]) < 0.51
        assert abs(np.mean(P[d]["outer_test_sizes"]) - C[d]["mean_test_size"]) < 0.51
    M = {}

    def macro(name, value):
        assert name not in M, name
        M[name] = str(value)

    for d, _ in DATASETS:
        assert C[d]["repeats"] == 20 and len(C[d]["kernels"]["rbf"]["bal_per_fold"]) == 100, d
        assert S[d]["folds"] == 100

    # ------------------------------------------------------------ performance
    diffs, ps, sds, cis = [], [], [], []
    for d, _ in DATASETS:
        K = C[d]["kernels"]
        for k in QUANTUM:
            diffs.append(K[k]["diff_vs_rbf"]); ps.append(K[k]["p_vs_rbf"]); cis.append(K[k]["ci95_vs_rbf"])
            # stored difference equals the per-fold mean difference
            dd = np.array(K[k]["bal_per_fold"]) - np.array(K["rbf"]["bal_per_fold"])
            assert abs(dd.mean() - K[k]["diff_vs_rbf"]) < 1e-12
        for k in ["rbf"] + QUANTUM:
            sds.append(K[k]["bal_sd"])
            assert abs(np.mean(K[k]["bal_per_fold"]) - K[k]["bal_mean"]) < 1e-12
        # claim: ZZ has the lowest mean balanced accuracy and the lowest mean AUC (all five kernels)
        assert min(K, key=lambda k: K[k]["bal_mean"]) == "zz", d
        assert min(K, key=lambda k: K[k]["auc_mean"]) == "zz", d
    # claim: no difference significant; every interval contains zero
    assert all(p > ALPHA for p in ps), ps
    assert all(lo < 0 < hi for lo, hi in cis), cis
    macro("AbsDiffMin", num(min(diffs))); macro("AbsDiffMax", num(max(diffs)) if max(diffs) < 0
                                                 else "$+$" + f(round(max(diffs), 3), 3))
    macro("AbsPMin", f(down(min(ps), 2), 2))
    macro("PMin", pval(min(ps))); macro("PMax", pval(max(ps)))
    macro("PerfMaxAbsDiff", f(up(max(abs(x) for x in diffs), 3), 3))
    macro("CIBound", f(up(max(max(abs(lo), abs(hi)) for lo, hi in cis), 2), 2))
    macro("SDMin", f(min(sds), 3)); macro("SDMax", f(max(sds), 3))

    # ------------------------------------------------------------ re-uploading vs control
    rd, rp, rci = [], [], []
    for d, _ in DATASETS:
        K = C[d]["kernels"]
        for k in ("reup2", "reup3"):
            r = C[d]["reuploading_vs_control"][k]
            dd = np.array(K[k]["bal_per_fold"]) - np.array(K["fqk_cos2"]["bal_per_fold"])
            assert abs(dd.mean() - r["diff"]) < 1e-12
            rd.append(r["diff"]); rp.append(r["p"]); rci.append(r["ci95"])
    assert all(p > ALPHA for p in rp), rp
    assert all(lo < 0 < hi for lo, hi in rci), rci
    macro("ReupCtrlMaxAbs", f(up(max(abs(x) for x in rd), 3), 3))
    macro("ReupCtrlPMin", pval(min(rp))); macro("ReupCtrlPMax", pval(max(rp)))
    macro("ReupCtrlCIBound", f(up(max(max(abs(lo), abs(hi)) for lo, hi in rci), 2), 2))
    macro("ReupCtrlPFloor", f(down(min(rp), 2), 2))     # used as "p >= x" in the abstract

    # ------------------------------------------------------------ hyperparameters
    rot_medians, zz_medians, c100 = [], [], []
    for d, _ in DATASETS:
        K = C[d]["kernels"]
        for k in QUANTUM:
            s = np.array([p[0] for p in K[k]["params"]])
            assert np.all(s < 1.0), (d, k)                     # claim: s >= 1 never chosen
            assert np.sum(s < 0.25) > 50, (d, k)              # claim: small scales (s < 0.25) in most folds
            (rot_medians if k in ROTATION else zz_medians).append(float(np.median(s)))
        for k in ["rbf"] + QUANTUM:
            c100.append(sum(p[1] == C_MAX for p in K[k]["params"]))
    macro("MedianSMax", f"{max(rot_medians):g}")
    assert len(set(zz_medians)) == 1, zz_medians               # text gives one ZZ median
    macro("ZZMedianS", f"{zz_medians[0]:g}")
    assert zz_medians[0] <= max(rot_medians)
    for d, tag in (("heart_disease", "HD"), ("breast_cancer", "BC")):
        K = C[d]["kernels"]
        macro(f"ZZAtMin{tag}", sum(p[0] == S_MIN for p in K["zz"]["params"]))
        macro(f"RBFAtMin{tag}", sum(p[0] == GAMMA_MIN for p in K["rbf"]["params"]))
    macro("CHundredMin", min(c100)); macro("CHundredMax", max(c100))

    # ------------------------------------------------------------ similarity
    def sim(d, k, m):
        return S[d]["kernels"][k][m]["mean"]
    g0 = {k: E["metric_at_zero"][k]["ratio"] for k in QUANTUM}
    macro("CtrlIsoHD", f(sim("heart_disease", "fqk_cos2", "cka_best_rbf"), 4))
    macro("CtrlIsoBC", f(sim("breast_cancer", "fqk_cos2", "cka_best_rbf"), 4))
    reup_iso = [sim(d, k, "cka_best_rbf") for d, _ in DATASETS for k in ("reup2", "reup3")]
    macro("ReupIsoMin", f(min(reup_iso), 4)); macro("ReupIsoMax", f(max(reup_iso), 4))
    macro("ZZIsoHD", f(sim("heart_disease", "zz", "cka_best_rbf"), 4))
    macro("ZZIsoBC", f(sim("breast_cancer", "zz", "cka_best_rbf"), 4))
    # claims: control essentially isotropic RBF; ZZ is not; re-uploading close to isotropic RBF
    assert min(sim(d, "fqk_cos2", "cka_best_rbf") for d, _ in DATASETS) > 0.99
    assert max(sim(d, "zz", "cka_best_rbf") for d, _ in DATASETS) < min(reup_iso)
    aniso = [sim(d, k, "cka_aniso_gauss") for d, _ in DATASETS for k in QUANTUM]
    macro("AnisoMin", f(min(aniso), 4)); macro("AnisoMax", f(max(aniso), 4))   # same rounding as Table 4
    eig = {k: [sim(d, k, "metric_eig_ratio") for d, _ in DATASETS] for k in QUANTUM}
    macro("CtrlEigMin", f(min(eig["fqk_cos2"]), 2)); macro("CtrlEigMax", f(max(eig["fqk_cos2"]), 2))
    reup_eig = eig["reup2"] + eig["reup3"]
    macro("ReupEigMin", f(min(reup_eig), 2)); macro("ReupEigMax", f(max(reup_eig), 2))
    macro("ZZEigMin", f(min(eig["zz"]), 2)); macro("ZZEigMax", f(max(eig["zz"]), 2))
    # claim ordering: control nearly isotropic > re-uploading moderate > ZZ strongly anisotropic
    assert min(eig["fqk_cos2"]) > max(reup_eig) and min(reup_eig) > max(eig["zz"])
    macro("GzeroCtrl", f(g0["fqk_cos2"], 2)); macro("GzeroReupTwo", f(g0["reup2"], 2))
    macro("GzeroReupThree", f(g0["reup3"], 2)); macro("GzeroZZ", f(g0["zz"], 3))
    # claim "fitted ratios are close to those of g(0)": same ordering, ZZ both < 0.05
    assert g0["fqk_cos2"] > g0["reup2"] > g0["reup3"] > g0["zz"]
    assert max(eig["zz"]) < 0.05 and g0["zz"] < 0.05
    for k in QUANTUM:
        for v in eig[k]:
            assert abs(v - g0[k]) < 0.15, (k, v, g0[k])
    npd = [(d, name, k, 100 - S[d]["kernels"][k]["n_folds_metric_positive_definite"])
           for d, name in DATASETS for k in QUANTUM]
    bad = [(name, k, n) for d, name, k, n in npd if n > 0]
    if not bad:
        macro("PDSentence", "The fitted $M$ was positive definite in every fold.")
    else:
        short = {"fqk_cos2": "the control", "reup2": "the two-layer re-uploading kernel",
                 "reup3": "the three-layer re-uploading kernel", "zz": "the ZZ feature map"}
        parts = [f"{n} of the 100 {name.lower().replace(' ', '-')} folds for {short[k]}" for name, k, n in bad]
        macro("PDSentence", "The fitted $M$ was positive definite in every fold except "
              + "; ".join(parts) + "; in those folds the eigenvalue ratio is negative and the fitted function is of Gaussian form "
              "but not a valid kernel, and they are included in the mean ratios and in the mean CKA.")

    # ------------------------------------------------------------ entanglement
    ent = E["entanglement"]
    assert set(ent) == {d for d, _ in DATASETS}, set(ent)
    qrot = [ent[d][k]["mean"] for d, _ in DATASETS for k in ROTATION]
    macro("QRotMin", f(min(qrot), 2)); macro("QRotMax", f(max(qrot), 2))
    macro("QZZHD", f(ent["heart_disease"]["zz"]["mean"], 2))
    macro("QZZBC", f(ent["breast_cancer"]["zz"]["mean"], 2))
    assert max(qrot) < 0.25                                   # "small on average"
    assert min(ent[d]["zz"]["min"] for d, _ in DATASETS) > 0.6  # "strongly entangled at every selected scale"
    for d, _ in DATASETS:
        for k in ROTATION:
            s = np.array([p[0] for p in C[d]["kernels"][k]["params"]])
            q = np.array(ent[d][k]["per_fold"])
            big = s >= 0.25
            # claims: s >= 0.25 is chosen in a minority of folds, and those folds give the larger values
            assert 0 < big.sum() < 50, (d, k, big.sum())
            assert q[big].min() > q[~big].max(), (d, k, q[big].min(), q[~big].max())

    # ------------------------------------------------------------ numerics check
    hd = NUM["heart_disease"]; bc = NUM["breast_cancer"]
    macro("CtrlTieFolds", hd["n_folds_different"])
    macro("CtrlBalSV", f(hd["bal_mean_statevector"], 3)); macro("CtrlBalCF", f(hd["bal_mean_closed_form"], 3))
    assert hd["n_folds_different"] > 0
    assert bc["n_folds_different"] == 0     # text: "none of the breast-cancer selections"
    assert abs(hd["bal_mean_statevector"] - C["heart_disease"]["kernels"]["fqk_cos2"]["bal_mean"]) < 1e-12

    # ------------------------------------------------------------ robustness checks (revision)
    MT = {d: json.load(open(f"results/matched_{d}.json")) for d, _ in DATASETS}
    SB = {d: [json.load(open(f"results/subset_seed{k}_{d}.json")) for k in range(1, 6)] for d, _ in DATASETS}
    CG = {d: json.load(open(f"results/cgrid1000_{d}.json")) for d, _ in DATASETS}
    NZ = {d: json.load(open(f"results/noise_{d}.json")) for d, _ in DATASETS}

    def rng_(vals):
        return min(vals), max(vals)

    # gate-matched controls
    md = [MT[d]["reuploading_vs_matched"][k] for d, _ in DATASETS for k in ("reup2", "reup3")]
    for d, _ in DATASETS:
        for k, c in (("reup2", "ctrl2"), ("reup3", "ctrl3")):
            dd = np.array(C[d]["kernels"][k]["bal_per_fold"]) - np.array(MT[d]["kernels"][c]["bal_per_fold"])
            assert abs(dd.mean() - MT[d]["reuploading_vs_matched"][k]["diff"]) < 1e-12
    assert all(r["p"] > ALPHA and r["ci95"][0] < 0 < r["ci95"][1] for r in md)
    macro("MatchedMin", sgn(min(r["diff"] for r in md))); macro("MatchedMax", sgn(max(r["diff"] for r in md)))
    macro("MatchedPFloor", f(down(min(r["p"] for r in md), 2), 2))
    macro("MatchedCIBound", f(up(max(max(abs(x) for x in r["ci95"]) for r in md), 2), 2))

    # other subsets
    sv = [r["kernels"][k]["vs_rbf"] for d, _ in DATASETS for r in SB[d] for k in QUANTUM]
    sc = [r["kernels"][k]["vs_control"] for d, _ in DATASETS for r in SB[d] for k in ("reup2", "reup3")]
    assert all(r["p"] > ALPHA for r in sv + sc)
    assert all(r["subset_seed"] == i + 1 and r["repeats"] == 5 for d, _ in DATASETS for i, r in enumerate(SB[d]))
    lo, hi = rng_([r["diff"] for r in sv])
    macro("SubDiffMin", num(lo)); macro("SubDiffMax", ("$+$" + f(round(hi, 3), 3)) if hi >= 0 else num(hi))
    macro("SubPFloor", f(down(min(r["p"] for r in sv), 2), 2))
    macro("SubCtrlMin", sgn(min(r["diff"] for r in sc))); macro("SubCtrlMax", sgn(max(r["diff"] for r in sc)))
    macro("SubCtrlPFloor", f(down(min(r["p"] for r in sc), 2), 2))

    # larger C grid
    cv = [CG[d]["kernels"][k]["vs_rbf"] for d, _ in DATASETS for k in QUANTUM]
    cc = [CG[d]["kernels"][k]["vs_control"] for d, _ in DATASETS for k in ("reup2", "reup3")]
    assert all(r["p"] > ALPHA for r in cv + cc)
    assert all(CG[d]["C_grid"][-1] == 1000.0 for d, _ in DATASETS)
    assert all(CG[d]["kernels"][k]["n_constant_pred"] == 0 for d, _ in DATASETS for k in CG[d]["kernels"])
    lo, hi = rng_([r["diff"] for r in cv])
    macro("CgridDiffMin", num(lo)); macro("CgridDiffMax", ("$+$" + f(round(hi, 3), 3)) if hi >= 0 else num(hi))
    macro("CgridPFloor", f(down(min(r["p"] for r in cv), 2), 2))
    macro("CgridCtrlMin", sgn(min(r["diff"] for r in cc))); macro("CgridCtrlMax", sgn(max(r["diff"] for r in cc)))
    macro("CgridCtrlPFloor", f(down(min(r["p"] for r in cc), 2), 2))
    n1000 = [CG[d]["kernels"][k]["n_C1000"] for d, _ in DATASETS for k in CG[d]["kernels"]]
    macro("CThousandMin", min(n1000)); macro("CThousandMax", max(n1000))
    assert min(n1000) > 0                                   # text: the boundary moved but did not disappear

    # noise
    for d, _ in DATASETS:
        assert NZ[d]["folds"] == 20 and NZ[d]["shots"] == 1000 and NZ[d]["p1"] == 0.001 and NZ[d]["p2"] == 0.01
    nn = [NZ[d]["settings"][st][k] for d, _ in DATASETS for st in ("shots", "gate", "both") for k in QUANTUM]
    assert all(r["vs_noiseless"]["p"] > ALPHA for r in nn)          # no significant change from noiseless
    macro("NoiseVsCleanPFloor", f(down(min(r["vs_noiseless"]["p"] for r in nn), 2), 2))
    # comparisons with the RBF baseline: none significantly better; report any significantly worse
    sig = [(d, name, st, k, NZ[d]["settings"][st][k]["vs_rbf"]) for d, name in DATASETS
           for st in ("shots", "gate", "both") for k in QUANTUM if NZ[d]["settings"][st][k]["vs_rbf"]["p"] <= ALPHA]
    assert all(r["diff"] < 0 for *_, r in sig)                       # never significantly better than RBF
    nonsig = [NZ[d]["settings"][st][k]["vs_rbf"] for d, _ in DATASETS for st in ("shots", "gate", "both") for k in QUANTUM
              if NZ[d]["settings"][st][k]["vs_rbf"]["p"] > ALPHA]
    setting_name = {"shots": "1000 shots", "gate": "gate errors", "both": "1000 shots and gate errors"}
    if not sig:
        macro("NoiseRBFSentence", "none differed significantly from the RBF baseline ($p\\geq" + f(down(min(r["p"] for r in nonsig), 2), 2) + "$)")
    else:
        groups = {}
        for d, name, st, k, r in sig:
            groups.setdefault((name, st), []).append((k, r))
        parts = []
        for (name, st), items in groups.items():
            ks = " and ".join(NAME[k].replace(" ($V R$)", "").replace("Re-uploading, 2 layers", "the two-layer re-uploading kernel")
                              .replace("Re-uploading, 3 layers", "the three-layer re-uploading kernel").replace("ZZ feature map", "the ZZ feature map")
                              .replace("Control", "the control") for k, _ in items)
            ps = "$ and $".join(f"{r['p']:.3f}" for _, r in items)
            parts.append(f"with {setting_name[st]} on {name.lower()}, {ks} {'were' if len(items) > 1 else 'was'} significantly below it ($p={ps}$)")
        body = "; ".join(parts)
        assert body.endswith("$)")
        macro("NoiseRBFSentence", "no quantum kernel was significantly better than the RBF baseline, and " + body[:-1]
              + f"; not adjusted for the {len(sig) + len(nonsig)} comparisons with the RBF baseline in this test)")
    macro("NoiseNSig", len(sig))
    gate_ch = [NZ[d]["settings"]["gate"][k]["vs_noiseless"]["diff"] for d, _ in DATASETS for k in QUANTUM]
    macro("NoiseGateMin", sgn(min(gate_ch))); macro("NoiseGateMax", sgn(max(gate_ch)))
    assert max(abs(x) for x in gate_ch) < 0.01                           # text: gate noise alone changed little
    shot_ch = [NZ[d]["settings"][st][k]["vs_noiseless"]["diff"] for d, _ in DATASETS for st in ("shots", "both") for k in QUANTUM]
    n_neg = sum(x < 0 for x in shot_ch)
    assert n_neg > len(shot_ch) / 2, n_neg                               # text: lowered accuracy for most kernels
    macro("NoiseShotNeg", n_neg); macro("NoiseShotTotal", len(shot_ch))
    shots_only = [NZ[d]["settings"]["shots"][k]["vs_noiseless"]["diff"] for d, _ in DATASETS for k in QUANTUM]
    macro("NoiseShotOnlyMaxDrop", f(max(-x for x in shots_only), 3))
    assert max(-x for x in shots_only) > 0
    NZC = {}
    for d, _ in DATASETS:            # re-uploading minus control under each noise setting (same folds)
        for st in ("shots", "gate", "both"):
            for k in ("reup2", "reup3"):
                dd = np.array(NZ[d]["settings"][st][k]["bal_per_fold"]) - np.array(NZ[d]["settings"][st]["fqk_cos2"]["bal_per_fold"])
                dm, _, pv = nadeau_bengio_test(dd, C[d]["mean_train_size"], C[d]["mean_test_size"])
                NZC[(d, st, k)] = {"diff": float(dm), "p": float(pv)}
    assert all(v["p"] > ALPHA for v in NZC.values())
    macro("NoiseCtrlPFloor", f(down(min(v["p"] for v in NZC.values()), 2), 2))
    for d, _ in DATASETS:               # text: ZZ lowest mean under every noise setting, as without noise
        for st in ("shots", "gate", "both"):
            row = NZ[d]["settings"][st]
            assert min(row, key=lambda k: row[k]["bal_mean"]) == "zz", (d, st)
    # shot noise moves selection to larger scales (medians over the 20 folds)
    for d, _ in DATASETS:
        for k in QUANTUM:
            clean = np.median([p[0] for p in C[d]["kernels"][k]["params"][:20]])
            shot = np.median([p[0] for p in NZ[d]["settings"]["shots"][k]["params"]])
            assert shot >= clean, (d, k, clean, shot)
    # same for "both" (shots on top of gate noise)
    for d, _ in DATASETS:
        for k in QUANTUM:
            clean = np.median([p[0] for p in C[d]["kernels"][k]["params"][:20]])
            both = np.median([p[0] for p in NZ[d]["settings"]["both"][k]["params"]])
            assert both >= clean, (d, k, clean, both)

    # shot-noise training matrices: positive semidefiniteness (diagnostic recorded by noise_check.py)
    ge = [NZ[d]["settings"][st][k]["train_gram_eig"] for d, _ in DATASETS for st in ("shots", "both") for k in QUANTUM]
    frac_neg = [100.0 * g["n_negative"] / g["n_matrices"] for g in ge]
    macro("NoisePSDNegMin", f"{min(frac_neg):.0f}"); macro("NoisePSDNegMax", f"{max(frac_neg):.0f}")
    worst = max(-g["min_ratio_smallest_to_largest"] for g in ge)
    assert 0 < worst < 0.02                                      # text: at most about 1% of the largest eigenvalue
    macro("NoisePSDWorstPct", f(up(100 * worst, 1), 1))
    assert all("train_gram_eig" not in NZ[d]["settings"]["gate"][k] for d, _ in DATASETS for k in QUANTUM)

    # classical Gaussian kernels from each circuit's g(0), used as classifiers (gaussian_surrogate.py)
    GS = {d: json.load(open(f"results/gaussian_surrogate_{d}.json")) for d, _ in DATASETS}
    for d, _ in DATASETS:
        assert GS[d]["mean_train_size"] == C[d]["mean_train_size"] and GS[d]["mean_test_size"] == C[d]["mean_test_size"]
        for k in QUANTUM:
            r = GS[d]["kernels"][k]
            ev = r["metric_eigenvalues"]
            assert min(ev) > 0 and abs(min(ev) / max(ev) - g0[k]) < 1e-9, (d, k)   # same g(0) as Table 5
            assert abs(r["quantum_bal_mean"] - C[d]["kernels"][k]["bal_mean"]) < 1e-12
            assert len(r["bal_per_fold"]) == 100 and r["n_constant_pred"] == 0
            dd = np.array(r["bal_per_fold"]) - np.array(C[d]["kernels"][k]["bal_per_fold"])
            assert abs(dd.mean() - r["vs_quantum"]["diff"]) < 1e-12
    sq = [GS[d]["kernels"][k]["vs_quantum"] for d, _ in DATASETS for k in QUANTUM]
    sr = [GS[d]["kernels"][k]["vs_rbf"] for d, _ in DATASETS for k in QUANTUM]
    assert all(r["p"] > ALPHA for r in sq + sr)
    macro("SurDiffMin", sgn(min(r["diff"] for r in sq))); macro("SurDiffMax", sgn(max(r["diff"] for r in sq)))
    macro("SurPFloor", f(down(min(r["p"] for r in sq), 2), 2))
    macro("SurCIBound", f(up(max(max(abs(x) for x in r["ci95"]) for r in sq), 2), 2))
    lo, hi = min(r["diff"] for r in sr), max(r["diff"] for r in sr)
    macro("SurRbfMin", num(lo)); macro("SurRbfMax", ("$+$" + f(round(hi, 3), 3)) if hi >= 0 else num(hi))
    macro("SurRbfPFloor", f(down(min(r["p"] for r in sr), 2), 2))

    # heart disease with the nominal variables one-hot encoded (compare_quantum_kernels.py --dataset heart_disease_onehot)
    OH = json.load(open("results/compare_heart_disease_onehot.json"))
    assert OH["n"] == C["heart_disease"]["n"] and OH["repeats"] == 20
    assert OH["mean_train_size"] == C["heart_disease"]["mean_train_size"]
    ov = [OH["kernels"][k] for k in QUANTUM]
    oc = [OH["reuploading_vs_control"][k] for k in ("reup2", "reup3")]
    assert all(r["p_vs_rbf"] > ALPHA for r in ov) and all(r["p"] > ALPHA for r in oc)
    lo, hi = min(r["diff_vs_rbf"] for r in ov), max(r["diff_vs_rbf"] for r in ov)
    macro("OnehotDiffMin", num(lo)); macro("OnehotDiffMax", ("$+$" + f(round(hi, 3), 3)) if rnd(hi, 3) > 0 else sgn(hi))
    macro("OnehotPFloor", f(down(min(r["p_vs_rbf"] for r in ov), 2), 2))
    macro("OnehotCtrlMin", sgn(min(r["diff"] for r in oc))); macro("OnehotCtrlMax", sgn(max(r["diff"] for r in oc)))
    macro("OnehotCtrlPFloor", f(down(min(r["p"] for r in oc), 2), 2))
    macro("OnehotNFeat", 22)

    # selections on the edge of the grid that tie with a pair inside the grid (boundary_ties.py)
    BT = {d: json.load(open(f"results/boundary_ties_{d}.json")) for d, _ in DATASETS}
    pct = []
    for d, _ in DATASETS:
        for k, r in BT[d]["kernels"].items():
            assert [x["edge"] for x in r["per_fold"]] == [
                (pp[0] == (GAMMA_MIN if k == "rbf" else S_MIN)) or pp[1] == C_MAX for pp in C[d]["kernels"][k]["params"]]
            pct.append(100.0 * r["n_edge_with_interior_tie"] / r["n_edge"])
    macro("EdgeTieMin", f"{min(pct):.0f}"); macro("EdgeTieMax", f"{max(pct):.0f}")
    # "an optimum reached along such a line is reported at its smallest s, often paired with the largest C":
    # among quantum-kernel selections with s <= 0.05, most have C = 100
    small = [pp for d, _ in DATASETS for k in QUANTUM for pp in C[d]["kernels"][k]["params"] if pp[0] <= 0.05]
    frac = sum(pp[1] == C_MAX for pp in small) / len(small)
    assert frac > 0.5, frac
    macro("SmallSLargeC", f"{100 * frac:.0f}")

    # same analysis with newer library versions (results/version_check_<dataset>.json)
    VC = {d: json.load(open(f"results/version_check_{d}.json")) for d, _ in DATASETS}
    vers = {VC[d]["library_versions"]["scikit-learn"] for d, _ in DATASETS}
    assert len(vers) == 1
    macro("VersionSkl", vers.pop())
    nd = sum(sum(a != b for a, b in zip(C[d]["kernels"][k]["params"], VC[d]["kernels"][k]["params"]))
             for d, _ in DATASETS for k in C[d]["kernels"])
    mx = max(abs(C[d]["kernels"][k]["bal_mean"] - VC[d]["kernels"][k]["bal_mean"]) for d, _ in DATASETS for k in C[d]["kernels"])
    assert all(VC[d]["kernels"][k]["p_vs_rbf"] > ALPHA for d, _ in DATASETS for k in QUANTUM)   # no conclusion changes
    macro("VersionDiffFolds", nd); macro("VersionMaxBalChange", f(up(mx, 4), 4))

    # ------------------------------------------------------------ write macros
    with open(os.path.join(out_dir, "numbers.tex"), "w") as fh:
        fh.write("% Generated by build_numbers.py from results/*.json -- do not edit by hand.\n")
        for k, v in M.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    # ------------------------------------------------------------ tables
    def write(name, text):
        open(os.path.join(out_dir, name), "w").write("% Generated by build_numbers.py -- do not edit by hand.\n" + text)

    rows = []
    for d, name in DATASETS:
        K = C[d]["kernels"]
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{name}}} ($N={C[d]['n']}$)}}\\\\")
        for k in ["rbf"] + QUANTUM:
            r = K[k]
            diff = "--" if k == "rbf" else sgn(r["diff_vs_rbf"])
            p = "--" if k == "rbf" else pval(r["p_vs_rbf"])
            rows.append(f"{NAME[k]} & {ENT[k]} & ${f(r['bal_mean'], 3)}\\pm{f(r['bal_sd'], 3)}$ & "
                        f"{f(r['auc_mean'], 3)} & {diff} & {p}\\\\")
        rows.append("\\hline")
    write("table_perf.tex", r"""\begin{table}[t]
\caption{Balanced accuracy (mean $\pm$ standard deviation over 100 outer folds), mean ROC-AUC, and mean difference in balanced accuracy from the RBF baseline with the Nadeau--Bengio corrected $p$-value. ``Ent.\ in kernel'': whether the entanglement of the encoded states reaches the kernel, that is, whether removing the entangling gates changes the kernel values (Sec.~\ref{sec:cancel}). Differences are computed before rounding. All hyperparameters selected by nested cross-validation.}
\label{tab:perf}
\centering
\footnotesize
\begin{tabular}{lccccc}
\hline
\textbf{Kernel} & \textbf{Ent.\ in kernel} & \textbf{Bal.\ acc.} & \textbf{AUC} & \textbf{Diff.} & $\boldsymbol{p}$\\
\hline
""" + "\n".join(rows) + "\n\\end{tabular}\n\\end{table}\n")

    rows = []
    for d, name in DATASETS:
        for k in ("reup2", "reup3"):
            r = C[d]["reuploading_vs_control"][k]
            lo, hi = r["ci95"]
            rows.append(f"{name} & {NAME[k]} & {sgn(r['diff'])} & $[{sgn_plain(lo)},\\ {sgn_plain(hi)}]$ & {pval(r['p'])}\\\\")
    write("table_reup.tex", r"""\begin{table}[t]
\caption{Re-uploading kernels (entanglement reaches the kernel) compared with the control (entanglement cancels): mean difference in balanced accuracy over 100 outer folds, corrected 95\% interval, and Nadeau--Bengio corrected $p$-value.}
\label{tab:reup}
\centering
\footnotesize
\begin{tabular}{llccc}
\hline
\textbf{Dataset} & \textbf{Kernel minus control} & \textbf{Diff.} & \textbf{95\% interval} & $\boldsymbol{p}$\\
\hline
""" + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n\\end{table}\n")

    rows = []
    for d, name in DATASETS:
        K = C[d]["kernels"]
        rows.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{name}}}}}\\\\")
        for k in ["rbf"] + QUANTUM:
            v = [p[0] for p in K[k]["params"]]
            at_min = sum(x == (GAMMA_MIN if k == "rbf" else S_MIN) for x in v)
            c = sum(p[1] == C_MAX for p in K[k]["params"])
            label = NAME[k].replace(" ($V R$)", "").replace(" (classical)", "") + (" ($\\gamma$)" if k == "rbf" else " ($s$)")
            rows.append(f"{label} & {med(v)} & {at_min} & {c}\\\\")
        rows.append("\\hline")
    write("table_hyper.tex", r"""\begin{table}[t]
\caption{Hyperparameters selected by the inner cross-validation over 100 outer folds: median input scale $s$ (RBF: median $\gamma$), number of folds at the smallest grid value ($s=0.005$; $\gamma=10^{-4}$), and number of folds with $C=100$, the largest grid value. With 100 folds, a median between two grid values is their average.}
\label{tab:hyper}
\centering
\footnotesize
\begin{tabular}{lccc}
\hline
\textbf{Kernel} & \textbf{Median} & \textbf{At min.} & $\boldsymbol{C{=}100}$\\
\hline
""" + "\n".join(rows) + "\n\\end{tabular}\n\\end{table}\n")

    rows = []
    short = {"fqk_cos2": "Control", "reup2": "Re-uploading, 2 layers", "reup3": "Re-uploading, 3 layers",
             "zz": "ZZ feature map"}
    for d, name in DATASETS:
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{name}}}}}\\\\")
        for k in QUANTUM:
            g = f(g0[k], 3) if k == "zz" else f(g0[k], 2)
            rows.append(f"{short[k]} & {f(sim(d, k, 'cka_best_rbf'), 4)} & {f(sim(d, k, 'cka_aniso_gauss'), 4)} & "
                        f"{f(sim(d, k, 'metric_eig_ratio'), 2)} & {S[d]['kernels'][k]['n_folds_metric_positive_definite']} & {g}\\\\")
        rows.append("\\hline")
    write("table_sim.tex", r"""\begin{table}[t]
\caption{Centered kernel alignment (CKA) between each tuned quantum kernel and classical kernels, mean over 100 outer training folds. ``Iso.\ RBF'': closest isotropic RBF kernel over 61 values of $\gamma$. ``Aniso.\ Gauss.'': $\exp(-(x-x')^\top M(x-x'))$ with $M$ fitted to the quantum kernel. ``Eig.\ ratio'': smallest over largest eigenvalue of $M$, mean over folds (1 = isotropic). ``PD'': number of folds in which the fitted $M$ is positive definite. ``$g(0)$'': eigenvalue ratio of the small-scale metric of Eq.~\eqref{eq:local}, computed from the circuit.}
\label{tab:sim}
\centering
\footnotesize
\begin{tabular}{lccccc}
\hline
\textbf{Kernel} & \textbf{Iso.\ RBF} & \textbf{Aniso.\ Gauss.} & \textbf{Eig.\ ratio} & \textbf{PD} & $\boldsymbol{g(0)}$\\
\hline
""" + "\n".join(rows) + "\n\\end{tabular}\n\\end{table}\n")

    rows = []
    for k in QUANTUM:
        cells = [f"{f(ent[d][k]['mean'], 2)} [{f(ent[d][k]['min'], 2)}, {f(ent[d][k]['max'], 2)}]" for d, _ in DATASETS]
        rows.append(f"{NAME[k]} & " + " & ".join(cells) + "\\\\")
    write("table_ent.tex", r"""\begin{table}[t]
\caption{Meyer--Wallach entanglement $Q$ of the encoded states at the input scale selected in each outer fold (mean over the training samples of the fold): mean over the 100 folds, with the minimum and maximum over folds in brackets.}
\label{tab:ent}
\centering
\footnotesize
\begin{tabular}{lcc}
\hline
\textbf{Kernel} & \textbf{Heart disease} & \textbf{Breast cancer}\\
\hline
""" + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n\\end{table}\n")

    def sizes(v):
        return " or ".join(str(x) for x in v)
    rows = []
    for d, name in DATASETS:
        q = P[d]
        rows.append(f"{name} & {q['available']} ({q['available_pos']}+{q['available_neg']}) & "
                    f"{q['subset']} ({q['subset_per_class']}+{q['subset_per_class']}) & {q['unused']} & "
                    f"{sizes(q['outer_train_sizes'])} & {sizes(q['outer_test_sizes'])} & "
                    f"{sizes(q['inner_train_sizes'])} & {sizes(q['inner_val_sizes'])}\\\\")
    write("table_partition.tex", r"""\begin{table}[t]
\caption{Data partitioning (numbers of samples). Class counts in parentheses: heart disease, disease + no disease; breast cancer, benign + malignant. One class-balanced subset is drawn from each dataset without replacement (seed 0); the remaining samples are not used. Outer splits: stratified five-fold cross-validation repeated 20 times (100 outer folds); in each repetition every sample of the subset is in exactly one test fold. Inner splits: stratified three-fold cross-validation within each outer training fold, used only for hyperparameter selection.}
\label{tab:partition}
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lccccccc}
\hline
 & & & & \multicolumn{2}{c}{\textbf{Outer fold}} & \multicolumn{2}{c}{\textbf{Inner fold}}\\
\textbf{Dataset} & \textbf{Available} & \textbf{Subset} & \textbf{Unused} & \textbf{Train} & \textbf{Test} & \textbf{Train} & \textbf{Val.}\\
\hline
""" + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n\\end{table}\n")

    def rng_cells(rows):
        lo = min(r["diff"] for r in rows); hi = max(r["diff"] for r in rows)
        rg = sgn(lo) if len(rows) == 1 else f"{sgn(lo)} to {sgn(hi)}"
        pm = min(r["p"] for r in rows)
        return rg + " & " + (f"{pval(pm)}" if len(rows) == 1 else f"$\\geq${f(down(pm, 2), 2)}")
    def line(label, per_ds):
        return f"{label} & " + " & ".join(rng_cells(per_ds[d]) for d, _ in DATASETS) + "\\\\"
    rows = ["\\multicolumn{5}{l}{\\emph{Gate-matched controls (100 outer folds)}}\\\\"]
    for k, lab in (("reup2", "Re-uploading, 2 layers $-$ matched control"), ("reup3", "Re-uploading, 3 layers $-$ matched control")):
        rows.append(line(lab, {d: [MT[d]["reuploading_vs_matched"][k]] for d, _ in DATASETS}))
    rows.append("\\multicolumn{5}{l}{\\emph{Five other subsets (25 outer folds each)}}\\\\")
    rows.append(line("Quantum $-$ RBF", {d: [r["kernels"][k]["vs_rbf"] for r in SB[d] for k in QUANTUM] for d, _ in DATASETS}))
    rows.append(line("Re-uploading $-$ control", {d: [r["kernels"][k]["vs_control"] for r in SB[d] for k in ("reup2", "reup3")] for d, _ in DATASETS}))
    rows.append("\\multicolumn{5}{l}{\\emph{$C$ up to 1000 (100 outer folds)}}\\\\")
    rows.append(line("Quantum $-$ RBF", {d: [CG[d]["kernels"][k]["vs_rbf"] for k in QUANTUM] for d, _ in DATASETS}))
    rows.append(line("Re-uploading $-$ control", {d: [CG[d]["kernels"][k]["vs_control"] for k in ("reup2", "reup3")] for d, _ in DATASETS}))
    rows.append("\\multicolumn{5}{l}{\\emph{Classical Gaussian kernel from the circuit's $g(0)$ (100 outer folds)}}\\\\")
    rows.append(line("Gaussian $-$ quantum kernel", {d: [GS[d]["kernels"][k]["vs_quantum"] for k in QUANTUM] for d, _ in DATASETS}))
    rows.append(line("Gaussian $-$ RBF", {d: [GS[d]["kernels"][k]["vs_rbf"] for k in QUANTUM] for d, _ in DATASETS}))
    rows.append("\\multicolumn{5}{l}{\\emph{Nominal variables one-hot encoded (100 outer folds)}}\\\\")
    ohr = [{"diff": OH["kernels"][k]["diff_vs_rbf"], "p": OH["kernels"][k]["p_vs_rbf"]} for k in QUANTUM]
    rows.append("Quantum $-$ RBF & " + rng_cells(ohr) + " & -- & --\\\\")
    rows.append("Re-uploading $-$ control & " + rng_cells([OH["reuploading_vs_control"][k] for k in ("reup2", "reup3")]) + " & -- & --\\\\")
    names = {"shots": "1000 shots", "gate": "gate errors", "both": "1000 shots and gate errors"}
    for st in ("shots", "gate", "both"):
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{Noise: {names[st]} (20 outer folds)}}}}\\\\")
        rows.append(line("Quantum $-$ same kernel without noise", {d: [NZ[d]["settings"][st][k]["vs_noiseless"] for k in QUANTUM] for d, _ in DATASETS}))
        rows.append(line("Quantum $-$ RBF", {d: [NZ[d]["settings"][st][k]["vs_rbf"] for k in QUANTUM] for d, _ in DATASETS}))
        rows.append(line("Re-uploading $-$ control", {d: [NZC[(d, st, k)] for k in ("reup2", "reup3")] for d, _ in DATASETS}))
    write("table_robust.tex", r"""\begin{table}[t]
\caption{Robustness checks (Sec.~\ref{sec:robust}): difference in mean balanced accuracy and Nadeau--Bengio corrected $p$-value. A range covers all four quantum kernels (or both re-uploading kernels) and, for the other subsets, all five subsets; the $p$ column then gives the smallest $p$. Each check uses the protocol of Sec.~\ref{sec:protocol}, with hyperparameters selected by nested cross-validation, and changes only the item named. The one-hot check applies only to heart disease, whose nominal variables are otherwise entered as their UCI numeric codes.}
\label{tab:robust}
\centering
\footnotesize
\begin{tabular}{lcccc}
\hline
 & \multicolumn{2}{c}{\textbf{Heart disease}} & \multicolumn{2}{c}{\textbf{Breast cancer}}\\
\textbf{Comparison} & \textbf{Diff.} & $\boldsymbol{p}$ & \textbf{Diff.} & $\boldsymbol{p}$\\
\hline
""" + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n\\end{table}\n")

    print(f"all checks passed; wrote numbers.tex and 7 tables to {out_dir}/")
    for k, v in M.items():
        print(f"  \\{k} = {v}")


if __name__ == "__main__":
    main()
