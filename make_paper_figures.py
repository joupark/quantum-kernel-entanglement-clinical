"""
make_paper_figures.py -- figures for "Does Entanglement Help Quantum Kernels on
Classical Data? A Controlled Comparison on Medical Datasets".

  Fig. 1  circuit schematics of the four quantum kernels (3 of 6 qubits shown)
  Fig. 2  difference in balanced accuracy from the RBF baseline, with the 95%
          interval of the Nadeau-Bengio corrected t-test (from per-fold scores)
  Fig. 3  input scale selected by nested cross-validation in each outer fold,
          and Meyer-Wallach entanglement at the selected scales

Reads results/compare_*.json and results/entanglement_and_metric*.json.
Usage: python make_paper_figures.py [--out figures]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8a8985"; GRID = "#e4e3df"
BLUE = "#2a78d6"     # quantum kernels (palette slot 1)
GRAY = "#8a8985"     # classical baseline

DATASETS = [("heart_disease", "Heart disease"), ("breast_cancer", "Breast cancer")]
KERNELS = [("fqk_cos2", "Control (cancels)"), ("reup2", "Re-uploading, 2 layers"),
           ("reup3", "Re-uploading, 3 layers"), ("zz", "ZZ feature map")]

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ------------------------------------------------------------------ Fig. 1
def gate(ax, x, y, text, w=0.9, h=0.5, fc="white"):
    ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, fc=fc, ec=INK, lw=0.7, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=7, zorder=4, color=INK)


_CNOTS = []


def cnot(ax, x, yc, yt):
    """Record a CNOT; drawn after layout so its circles stay round."""
    ax.plot([x, x], [yc, yt], color=INK, lw=0.7, zorder=2)
    _CNOTS.append((ax, x, yc, yt))


def _draw_cnots(fig):
    from matplotlib.patches import Ellipse
    fig.canvas.draw()
    for ax, x, yc, yt in _CNOTS:
        bb = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        k = ((x1 - x0) / bb.width) / ((y1 - y0) / bb.height)   # x-units per y-unit, same length
        r = 0.17
        ax.add_patch(Ellipse((x, yc), 2 * 0.07 * k, 2 * 0.07, color=INK, zorder=3))
        ax.add_patch(Ellipse((x, yt), 2 * r * k, 2 * r, fc="white", ec=INK, lw=0.7, zorder=3))
        ax.plot([x, x], [yt - r, yt + r], color=INK, lw=0.7, zorder=4)
        ax.plot([x - r * k, x + r * k], [yt, yt], color=INK, lw=0.7, zorder=4)
    _CNOTS.clear()


def fig1(out):
    fig, axes = plt.subplots(4, 1, figsize=(5.15, 4.6))
    ys = [2, 1, 0]
    rows = [
        ("(a) Control: $V\\,R(x)$ — the CNOT chain cancels in the fidelity kernel",
         [("R", 1), ("V", 2.2)], 3.4),
        ("(b) Re-uploading, 2 layers: $R(x)\\,V\\,R(x)$ — the chain does not cancel",
         [("R", 1), ("V", 2.2), ("R", 3.4)], 4.6),
        ("(c) Re-uploading, 3 layers: $R(x)\\,V\\,R(x)\\,V\\,R(x)$",
         [("R", 1), ("V", 2.2), ("R", 3.4), ("V", 4.6), ("R", 5.8)], 7.0),
        ("(d) ZZ feature map (two repetitions): $[D(x)\\,H^{\\otimes 6}]^2$",
         [("H", 1), ("D", 2.4), ("H", 3.8), ("D", 5.2)], 6.4),
    ]
    for ax, (title, ops, xend) in zip(axes, rows):
        for i, y in enumerate(ys):
            ax.plot([0.3, xend + 0.4], [y, y], color=INK2, lw=0.6, zorder=1)
            ax.text(0.1, y, f"$q_{i+1}$", ha="right", va="center", fontsize=7, color=INK2)
        for kind, x in ops:
            if kind == "R":
                for i, y in enumerate(ys):
                    gate(ax, x, y, f"$R_Y(sx_{i+1})$", w=1.0)
            elif kind == "H":
                for y in ys:
                    gate(ax, x, y, "$H$", w=0.45)
            elif kind == "V":
                cnot(ax, x - 0.25, ys[0], ys[1]); cnot(ax, x + 0.25, ys[1], ys[2])
            elif kind == "D":
                ax.add_patch(plt.Rectangle((x - 0.55, ys[2] - 0.3), 1.1, ys[0] - ys[2] + 0.6,
                                           fc="white", ec=INK, lw=0.7, zorder=3))
                ax.text(x, 1, "$D(x)$\nphases\n$\\phi_i,\\ \\phi_{i,i+1}$", ha="center",
                        va="center", fontsize=7, zorder=4, color=INK)
        if title.startswith("(a)"):
            ax.add_patch(plt.Rectangle((1.75, -0.45), 0.9, 2.9, fc="none", ec=BLUE, lw=0.9,
                                       ls=(0, (3, 2)), zorder=5))
            ax.text(2.8, 2.35, "$V$ cancels ($V^\\dagger V=I$)", fontsize=7, color=BLUE, va="center")
        ax.set_title(title, fontsize=7.5, loc="left", color=INK, pad=2)
        ax.set_xlim(-0.3, 7.6); ax.set_ylim(-0.6, 2.6); ax.axis("off")
    fig.tight_layout(h_pad=0.6)
    _draw_cnots(fig)
    fig.savefig(os.path.join(out, "fig1_circuits.pdf")); fig.savefig(os.path.join(out, "fig1_circuits.png"))
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 2
def fig2(out):
    fig, axes = plt.subplots(1, 2, figsize=(5.15, 2.1), sharey=True)
    for ax, (ds, name) in zip(axes, DATASETS):
        r = json.load(open(f"results/compare_{ds}.json"))["kernels"]
        for j, (k, label) in enumerate(KERNELS):
            d = r[k]["diff_vs_rbf"]
            lo, hi = r[k]["ci95_vs_rbf"]       # computed from the per-fold differences
            y = len(KERNELS) - 1 - j
            ax.plot([lo, hi], [y, y], color=BLUE, lw=1.4, solid_capstyle="round", zorder=2)
            ax.plot(d, y, "o", ms=4.5, color=BLUE, mec="white", mew=0.8, zorder=3)
        ax.axvline(0, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.set_title(name, fontsize=8, color=INK, pad=3)
        ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(-0.065, 0.065); ax.set_xticks([-0.05, 0, 0.05])
    axes[0].set_yticks(range(len(KERNELS))); axes[0].set_yticklabels([lab for _, lab in KERNELS][::-1])
    fig.supxlabel("Difference in balanced accuracy from RBF baseline (95% corrected interval)", fontsize=8, y=0.02)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(out, "fig2_differences.pdf")); fig.savefig(os.path.join(out, "fig2_differences.png"))
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 3
def load_entanglement():
    ent = {}
    for f in ("results/entanglement_and_metric.json",):
        if os.path.exists(f):
            ent.update(json.load(open(f))["entanglement"])
    return ent


def fig3(out):
    ent = load_entanglement()
    fig, axes = plt.subplots(2, 2, figsize=(5.15, 3.6), sharey="row",
                             gridspec_kw={"height_ratios": [1, 1]})
    rng = np.random.default_rng(0)
    for c, (ds, name) in enumerate(DATASETS):
        r = json.load(open(f"results/compare_{ds}.json"))["kernels"]
        ax = axes[0, c]
        for j, (k, label) in enumerate(KERNELS):
            s = np.array([p[0] for p in r[k]["params"]])
            y = len(KERNELS) - 1 - j + rng.uniform(-0.18, 0.18, len(s))
            ax.scatter(s * np.exp(rng.uniform(-0.08, 0.08, len(s))), y, s=8, color=BLUE,  # small jitter
                       alpha=0.35, lw=0, zorder=2)
            ax.plot(np.median(s), len(KERNELS) - 1 - j, "|", ms=11, mew=1.6, color=INK, zorder=3)
        ax.set_xscale("log"); ax.set_xlim(0.003, 3)
        ax.set_title(name, fontsize=8, color=INK, pad=3)
        ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)

        ax = axes[1, c]
        for j, (k, label) in enumerate(KERNELS):
            e = ent[ds][k]; y = len(KERNELS) - 1 - j
            ax.plot([e["min"], e["max"]], [y, y], color=BLUE, lw=1.4, alpha=0.45,
                    solid_capstyle="round", zorder=2)
            ax.plot(e["mean"], y, "o", ms=4.5, color=BLUE, mec="white", mew=0.8, zorder=3)
        ax.set_xlim(-0.02, 0.85)
        ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
    for row in (0, 1):
        axes[row, 0].set_yticks(range(len(KERNELS)))
        axes[row, 0].set_yticklabels([lab for _, lab in KERNELS][::-1])
    for c in range(2):
        axes[0, c].set_xlabel("selected input scale $s$")
        axes[1, c].set_xlabel("Meyer–Wallach $Q$ at selected $s$")
    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    for row, letter in ((0, "(a)"), (1, "(b)")):     # one panel letter per row, at the left edge
        top = axes[row, 0].get_position().y1
        fig.text(0.005, top + 0.01, letter, fontsize=8.5, fontweight="bold", color=INK, ha="left", va="bottom")
    fig.savefig(os.path.join(out, "fig3_scale_entanglement.pdf"))
    fig.savefig(os.path.join(out, "fig3_scale_entanglement.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    out = ap.parse_args().out
    os.makedirs(out, exist_ok=True)
    fig1(out); fig2(out); fig3(out)
    print(f"saved figures to {out}/")


if __name__ == "__main__":
    main()
