"""Figures for REPORT.md from results/runs and results/*.csv. Colours: blue/orange/aqua/yellow validated categorical slots."""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import head_rank, load_runs

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
                     "axes.titlesize": 9.5, "axes.titlecolor": INK, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
                     "savefig.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
NAMES = {"ag_L6": "AG News, 6 layers", "sst_L6": "SST-2, 6 layers", "cifar_L6": "CIFAR-10 ViT, 6 layers", "ag_L3": "AG News, 3", "ag_L12": "AG News, 12",
         "cifar_L3": "CIFAR-10, 3", "cifar_L12": "CIFAR-10, 12", "ag_L6_postln": "AG News, post-LN", "ag_L6_bigqk": "AG News, Q/K init ×3", "ag_L6_shuffled": "AG News, shuffled labels",
         "cifar_L6_shuffled": "CIFAR-10, shuffled labels", "bert_ag": "BERT fine-tune (AG News)"}
MAIN = ["ag_L6", "sst_L6", "cifar_L6", "ag_L6_bigqk", "bert_ag", "ag_L6_shuffled"]


def ramp(n):
    """Ordinal single-hue blue from light (first layer) to dark (last layer)."""
    a, b = np.array([.62, .79, .96]), np.array([.05, .22, .47])
    return [tuple(a + (b - a) * i / max(n - 1, 1)) for i in range(n)]


def by_cfg(runs):
    d = {}
    for r in runs:
        d.setdefault(r["cfg"], []).append(r)
    return d


def fig_rank(d):
    cs = [c for c in MAIN if c in d]
    fig, ax = plt.subplots(2, 3, figsize=(9.5, 5), sharex=True, sharey=True)
    for a, c in zip(ax.flat, cs):
        rs = d[c]
        v = np.mean([head_rank(r).mean(2) for r in rs], 0)
        for l, col in zip([0, len(v[0]) // 2, len(v[0]) - 1], [ramp(3)[i] for i in range(3)]):
            a.plot(rs[0]["frac"], v[:, l], color=col, lw=2, label=f"layer {l + 1}")
        lo, hi = np.min([head_rank(r) for r in rs], (0, 2, 3)), np.max([head_rank(r) for r in rs], (0, 2, 3))
        h = np.concatenate([head_rank(r) for r in rs], 2).reshape(len(rs[0]["frac"]), -1)
        a.fill_between(rs[0]["frac"], h.min(1), h.max(1), color=BLUE, alpha=.12, lw=0, label="all heads, range")
        a.axhline(.1, color=MUTED, ls="--", lw=1)
        a.axhline(.6, color=MUTED, ls=":", lw=1)
        a.axvline(.2, color=GRID, lw=1, zorder=0)
        a.set_title(NAMES[c])
        a.set_yscale("log")
        a.set_ylim(.015, 1)
        a.set_yticks([.02, .05, .1, .2, .6, 1])
        a.set_yticklabels(["0.02", "0.05", "0.1", "0.2", "0.6", "1"])
    ax[0, 0].legend(loc="upper left", fontsize=7.5)
    ax[0, 0].text(.98, .106, "low-rank limit 0.1", ha="right", color=MUTED, fontsize=7, transform=ax[0, 0].get_yaxis_transform())
    ax[0, 0].text(.98, .63, "high-rank limit 0.6", ha="right", color=MUTED, fontsize=7, transform=ax[0, 0].get_yaxis_transform())
    fig.supxlabel("fraction of training", color=MUTED, fontsize=9)
    fig.supylabel("rank_eff / n (log scale)", color=MUTED, fontsize=9)
    for a in ax.flat[len(cs):]:
        a.axis("off")
    fig.savefig("figures/fig1_rank.png")


def fig_hist(d):
    cs = [c for c in MAIN if c in d]
    fig, ax = plt.subplots(2, 3, figsize=(9.5, 4.6), sharex=True)
    bins = np.geomspace(.01, 1, 60)
    for a, c in zip(ax.flat, cs):
        rs = d[c]
        i = int(np.argmin(abs(rs[0]["frac"] - .2)))
        a.hist(np.concatenate([head_rank(r)[i].ravel() for r in rs]), bins, color=BLUE, label="20% of training")
        a.hist(np.concatenate([head_rank(r)[-1].ravel() for r in rs]), bins, histtype="step", color=ORANGE, lw=1.6, label="end of training")
        a.axvline(.1, color=MUTED, ls="--", lw=1)
        a.axvline(.6, color=MUTED, ls=":", lw=1)
        a.set_xscale("log")
        a.set_title(NAMES[c])
        a.set_xticks([.01, .1, .6, 1])
        a.set_xticklabels(["0.01", "0.1", "0.6", "1"])
    ax[0, 0].legend(fontsize=7.5)
    fig.supxlabel("head rank_eff / n", color=MUTED, fontsize=9)
    fig.supylabel("heads (all seeds)", color=MUTED, fontsize=9)
    for a in ax.flat[len(cs):]:
        a.axis("off")
    fig.savefig("figures/fig2_heads.png")


def fig_entropy(d):
    cs = [c for c in MAIN if c in d]
    fig, ax = plt.subplots(2, 3, figsize=(9.5, 5), sharex=True)
    for a, c in zip(ax.flat, cs):
        rs = d[c]
        L = rs[0]["L"]
        S = np.stack([r["m"][..., 0].mean((2, 3)) / np.log(r["n"]).mean() for r in rs])
        third = max(1, L // 3)
        for (lab, idx), col in zip((("first third of layers", range(third)), ("last third of layers", range(L - third, L))), (BLUE, ORANGE)):
            y = S[:, :, list(idx)].mean(2)
            a.plot(rs[0]["frac"], y.mean(0), color=col, lw=2, label=lab)
            a.fill_between(rs[0]["frac"], y.min(0), y.max(0), color=col, alpha=.15, lw=0)
        a.axhline(.3, color=MUTED, ls="--", lw=1)
        a.set_title(NAMES[c])
    ax[0, 0].legend(fontsize=7.5)
    fig.supxlabel("fraction of training", color=MUTED, fontsize=9)
    fig.supylabel("S / log n (band: seed range; dashed: pruning threshold 0.3)", color=MUTED, fontsize=9)
    for a in ax.flat[len(cs):]:
        a.axis("off")
    fig.savefig("figures/fig3_entropy.png")


def fig_prune():
    s = pd.read_csv("results/h3_sweep.csv")
    s = s[s.k >= 0]
    cs = [c for c in ["ag_L6", "sst_L6", "cifar_L6"] if c in set(s.cfg)]
    fig, ax = plt.subplots(1, len(cs), figsize=(9.5, 3.2), squeeze=False, constrained_layout=True)
    for a, c in zip(ax[0], cs):
        g = s[s.cfg == c].copy()
        g["bin"] = g.frac.round(2)
        for k, col, lab in (("low", BLUE, "lowest entropy first"), ("high", ORANGE, "highest entropy first"), ("random", AQUA, "random"), ("oracle", YELLOW, "ablation oracle")):
            m = g[g.bin.isin([0, .1, .25, .5, .75, 1])].groupby("bin")[k].mean()
            a.plot(m.index, m.values, color=col, lw=2, marker="o", ms=4, label=lab)
        a.axhline(2, color=MUTED, ls="--", lw=1)
        a.set_title(NAMES[c])
        a.set_yscale("symlog", linthresh=5)
    ax[0, 0].legend(fontsize=7.5)
    ax[0, 0].set_ylabel("test accuracy loss (pp, dashed 2 pp)")
    fig.supxlabel("fraction of heads removed", color=MUTED, fontsize=9)
    fig.savefig("figures/fig4_prune.png")


def fig_mi():
    if not os.path.exists("results/h4_long.csv"):
        return
    df = pd.read_csv("results/h4_long.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True, constrained_layout=True)
    eps = 2.0
    for a, t, name in zip(ax, ["order", "ag"], ["Synthetic order task", "AG News"]):
        g = df[(df.frac == 1) & (df.eps == eps) & (df.task == t)]
        last = g.layer.max()
        for n, col in zip(sorted(g.n.unique()), [BLUE, ORANGE, AQUA, YELLOW]):
            x = g[(g.n == n)].groupby("H")[["lo", "hi"]].mean()
            a.plot(x.index, x.lo, color=col, lw=2, marker="o", ms=4, label=f"n = {n:,}")
            a.plot(x.index, x.hi, color=col, lw=1.2, ls=":")
        h = np.array([0, df[df.task == t].H.max()])
        a.plot(h, h, color=INK, lw=1.4, ls="--", label="claimed bound, H(Y|X)")
        a.set_title(name)
        a.set_xlabel("H(Y|X) in nats")
    ax[0].set_ylabel(f"I(A;X) in nats, noise scale {eps}\n(solid lower, dotted upper bound)")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower center", ncol=5, fontsize=8)
    fig.savefig("figures/fig5_mi.png")


def fig_depth():
    d = pd.read_csv("results/h5.csv")
    cs = [["ag_L3", "ag_L6", "ag_L12"], ["cifar_L3", "cifar_L6", "cifar_L12"]]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2), sharey=True)
    for a, group in zip(ax, cs):
        for c, col in zip(group, [BLUE, ORANGE, AQUA]):
            g = d[d.cfg == c]
            if not len(g):
                continue
            m = g.groupby("layer").tmax.mean()
            a.plot((m.index + 1) / len(m), m.values, color=col, lw=2, marker="o", ms=4, label=NAMES[c])
        a.legend(fontsize=7.5)
        a.set_xlabel("relative depth (layer / L)")
    ax[0].set_ylabel("fraction of training to\nreach peak rank_eff")
    fig.savefig("figures/fig6_depth.png")


if __name__ == "__main__":
    runs = load_runs()
    d = by_cfg(runs)
    fig_rank(d)
    fig_hist(d)
    fig_entropy(d)
    fig_prune()
    fig_mi()
    fig_depth()
