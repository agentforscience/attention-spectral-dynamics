"""Evaluate H1-H5 by the criteria frozen in planning.md. Writes results/summary.json and results/*.csv.

Seed is the unit of inference. All randomness comes from a fixed generator; configs and tags are sorted.
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diptest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from spectral import METRICS

RNG = np.random.RandomState(0)
IX = {m: i for i, m in enumerate(METRICS)}
RANKDEFS = ["erank", "stable", "intrinsic", "rv", "numrank"]
LOW, HIGH = 0.1, 0.6
TAU = 0.3


def boot(x, B=10000):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = x[RNG.randint(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), np.quantile(m, .025), np.quantile(m, .975)


def load_runs():
    runs = []
    for f in sorted(glob.glob("results/runs/*.json")):
        if f.endswith(".ft.json"):
            continue
        j = json.load(open(f))
        z = np.load(f[:-5] + ".npz")
        n = z["lens"].astype(float)
        m = z["m"]  # [ck, layer, sample, head, metric]
        runs.append(dict(cfg=j["config"], seed=j["seed"], j=j, frac=z["frac"], n=n, m=m, L=m.shape[1], H=m.shape[3]))
    return runs


def head_rank(r, name="erank"):
    """[ck, layer, head] rank metric divided by n, averaged over probe inputs."""
    return (r["m"][..., IX[name]] / r["n"][None, None, :, None]).mean(2)


def head_S(r):
    return (r["m"][..., IX["S"]] / np.log(r["n"])[None, None, :, None]).mean(2)


def h1(runs):
    rows = []
    for r in runs:
        c20 = int(np.argmin(abs(r["frac"] - .2)))
        for name in RANKDEFS:
            for label, c in (("20%", c20), ("final", -1), ("init", 0)):
                v = head_rank(r, name)[c].ravel()
                hi, lo = (v > HIGH).mean(), (v < LOW).mean()
                dip, p = diptest.diptest(v.astype(float))
                rows.append(dict(cfg=r["cfg"], seed=r["seed"], rank_def=name, when=label, hi=hi, lo=lo, mid=1 - hi - lo,
                                 dip_p=p, med=np.median(v), min=v.min(), max=v.max(), sd=v.std(),
                                 ok=hi >= .1 and lo >= .1 and hi + lo >= .8 and p < .05))
        a, b = head_rank(r)[c20].ravel(), head_rank(r)[-1].ravel()
        rows.append(dict(cfg=r["cfg"], seed=r["seed"], rank_def="stability", when="20->final",
                         med=stats.spearmanr(a, b)[0] if a.std() > 0 and b.std() > 0 else np.nan))
    return pd.DataFrame(rows)


def classify(t, s, logn):
    """Pre-registered trajectory classes for one layer-mean S(t)."""
    rho = stats.spearmanr(t, s)[0]
    rng = s.max() - s.min()
    runmin = np.minimum.accumulate(s)
    rebound = (s - runmin).max()
    i = int(np.argmin(s))
    drop, rise = s[0] - s[i], s[-1] - s[i]
    mono_dec = rho <= -.8 and rebound <= .05 * rng
    u = .05 <= t[i] <= .8 and drop >= .02 * logn and rise >= .02 * logn and rise >= .25 * drop
    runmax = np.maximum.accumulate(s)
    mono_inc = rho >= .8
    return dict(rho=rho, mono_dec=mono_dec, u=u, mono_inc=mono_inc, net=s[-1] - s[0],
                peak_frac=t[int(np.argmax(s))], fallback=(runmax - s).max() / max(rng, 1e-9))


def h2(runs):
    rows = []
    for r in runs:
        S = r["m"][..., IX["S"]].mean((2, 3))
        logn = np.log(r["n"]).mean()
        third = max(1, r["L"] // 3)
        for l in range(r["L"]):
            band = "early" if l < third else "late" if l >= r["L"] - third else "mid"
            rows.append(dict(cfg=r["cfg"], seed=r["seed"], layer=l, band=band, S0=S[0, l], Send=S[-1, l],
                             S0_norm=S[0, l] / logn, Send_norm=S[-1, l] / logn, **classify(r["frac"], S[:, l], logn)))
    return pd.DataFrame(rows)


def h3(runs):
    rows, sweep = [], []
    for r in runs:
        p = r["j"]["pruning"]
        tot = r["L"] * r["H"]
        for row in p["rows"]:
            loss = lambda k: (p["base"] - row[k]) * 100
            sweep.append(dict(cfg=r["cfg"], seed=r["seed"], k=row["k"], frac=row["frac"], taus=row["taus"], base=p["base"] * 100,
                              low=loss("low_S"), high=loss("high_S"), oracle=loss("oracle"), random=loss("random")))
            if TAU in row["taus"]:
                rows.append(dict(cfg=r["cfg"], seed=r["seed"], k=row["k"], frac=row["frac"], base=p["base"] * 100,
                                 loss=loss("low_S"), high=loss("high_S"), oracle=loss("oracle"), random=loss("random")))
        s = np.array(p["S_norm"]).ravel()
        imp = np.array(p["importance"])
        # order in S_norm and importance is the same (layer, head) ordering
        sweep.append(dict(cfg=r["cfg"], seed=r["seed"], k=-1, frac=np.nan, taus=[], base=p["base"] * 100,
                          low=stats.spearmanr(s, imp)[0] if imp.std() > 0 else np.nan, high=np.nan, oracle=np.nan, random=np.nan))
    return pd.DataFrame(rows), pd.DataFrame(sweep)


def h5(runs):
    rows = []
    for r in runs:
        er = head_rank(r).mean(2)  # [ck, layer]
        t = r["frac"]
        for l in range(r["L"]):
            e = er[:, l]
            i = int(np.argmin(e))
            start = e[0]
            tol = .05 * (start - e[i])
            tmin = t[int(np.argmax(e <= e[i] + tol))]
            j = int(np.argmax(e))
            tol2 = .05 * (e[j] - start)
            tmax = t[int(np.argmax(e >= e[j] - tol2))]
            rows.append(dict(cfg=r["cfg"], seed=r["seed"], layer=l, L=r["L"], tmin=tmin, tmax=tmax, drop=(start - e[i]) / start,
                             rise=(e[j] - start) / start, rmin=e[i], rmax=e[j], r0=start, rend=e[-1]))
    return pd.DataFrame(rows)


def h5_ratios(df):
    rows = []
    for (c, s), g in df.groupby(["cfg", "seed"]):
        g = g.sort_values("layer")
        a, b = g.iloc[0], g.iloc[-1]
        rows.append(dict(cfg=c, seed=s, tmin_first=a.tmin, tmin_last=b.tmin, ratio_min=a.tmin / max(b.tmin, 1e-9) if b.tmin > 0 else np.nan,
                         genuine_fall=a["drop"] >= .1 and b["drop"] >= .1, tmax_first=a.tmax, tmax_last=b.tmax,
                         ratio_max=a.tmax / b.tmax if b.tmax > 0 else np.nan))
    return pd.DataFrame(rows)


def h4():
    rows = []
    for f in sorted(glob.glob("results/mi/*.json")):
        j = json.load(open(f))
        for lg in j["log"]:
            for r in lg["rows"]:
                rows.append(dict(task=j["task"], eta=j["eta"], n=j["n"], seed=j["seed"], H=j["H"], frac=lg["frac"],
                                 test_acc=lg["test_acc"], train_acc=lg["train_acc"], **r))
    return pd.DataFrame(rows)


def h4_tests(df):
    out = []
    last = df[(df.frac == 1)]
    for (task, eps), g in last.groupby(["task", "eps"]):
        g = g.groupby(["eta", "n", "seed", "H"], as_index=False)[["lo", "hi", "head_lo", "head_hi", "lo250", "lo_y", "hi_y"]].mean()
        for est in ["lo", "hi", "head_lo"]:
            X = sm.add_constant(np.c_[g.H, 1 / np.sqrt(g.n)])
            m = sm.OLS(g[est], X).fit(cov_type="HC3")
            ci = m.conf_int()
            out.append(dict(task=task, eps=eps, est=est, a=m.params.iloc[0], a_lo=ci.iloc[0, 0], a_hi=ci.iloc[0, 1],
                            b=m.params.iloc[1], b_lo=ci.iloc[1, 0], b_hi=ci.iloc[1, 1], c=m.params.iloc[2], c_lo=ci.iloc[2, 0], c_hi=ci.iloc[2, 1],
                            r2=m.rsquared, mean_lo=g.lo.mean(), mean_lo250=g.lo250.mean(),
                            viol_share=(g.lo > g.H + 1 / np.sqrt(g.n)).mean(), viol_share_c10=(g.lo > g.H + 10 / np.sqrt(g.n)).mean(),
                            ))
    return pd.DataFrame(out)


def main():
    runs = load_runs()
    acc = pd.DataFrame([dict(cfg=r["cfg"], seed=r["seed"], test=r["j"]["test"] * 100, n_mean=r["n"].mean(), n_min=r["n"].min(),
                             n_max=r["n"].max(), n_step=r["j"]["n_step"], minutes=r["j"]["log"][-1]["t"] / 60) for r in runs])
    acc.to_csv("results/accuracy.csv", index=False)
    print(acc.groupby("cfg")[["test", "n_mean", "minutes"]].agg(["mean", "std", "count"]).round(2).to_string())

    d1 = h1(runs)
    d1.to_csv("results/h1.csv", index=False)
    print("\nH1 by definition")
    t = d1[d1.when.isin(["20%", "final", "init"])].groupby(["cfg", "rank_def", "when"])[["hi", "lo", "mid", "med", "sd", "dip_p"]].mean().round(3)
    t["ok_share"] = d1[d1.when.isin(["20%", "final", "init"])].groupby(["cfg", "rank_def", "when"]).ok.mean().round(2)
    t.to_csv("results/h1_summary.csv")
    print(t.xs("20%", level="when").to_string())
    print(d1[d1.rank_def == "stability"].groupby("cfg").med.mean().round(3).to_string())

    d2 = h2(runs)
    d2.to_csv("results/h2.csv", index=False)
    s2 = d2.groupby(["cfg", "band"]).agg(S0=("S0_norm", "mean"), Send=("Send_norm", "mean"), net=("net", "mean"), mono_dec=("mono_dec", "mean"),
                                         u=("u", "mean"), mono_inc=("mono_inc", "mean"), rho=("rho", "mean"), peak=("peak_frac", "mean")).round(3)
    s2.to_csv("results/h2_summary.csv")
    print("\nH2\n", s2.to_string())

    d3, sw = h3(runs)
    d3.to_csv("results/h3.csv", index=False)
    sw.to_csv("results/h3_sweep.csv", index=False)
    rows = []
    for c, g in d3.groupby("cfg"):
        m, lo, hi = boot(g.loss)
        n = len(g)
        tt = stats.ttest_1samp(g.loss, 2, alternative="less") if n > 1 and g.loss.std() > 0 else None
        rows.append(dict(cfg=c, seeds=n, frac=g.frac.mean(), loss=m, lo=lo, hi=hi, random=g.random.mean(), high=g.high.mean(), oracle=g.oracle.mean(),
                         p_lt2=tt.pvalue if tt else np.nan, base=g.base.mean()))
    s3 = pd.DataFrame(rows)
    if len(s3):
        ok = s3.p_lt2.notna()
        s3.loc[ok, "p_holm"] = multipletests(s3.p_lt2[ok], method="holm")[1]
    s3.round(3).to_csv("results/h3_summary.csv", index=False)
    print("\nH3\n", s3.round(3).to_string())

    c = sw[sw.k == -1].groupby("cfg").low.agg(["mean", "std", "count"]).round(3)
    c.to_csv("results/h3_entropy_importance_corr.csv")
    print("Spearman S vs ablation importance\n", c.to_string())

    ft = [json.load(open(f)) for f in sorted(glob.glob("results/runs/*.ft.json"))]
    if ft:
        f = pd.DataFrame(ft)
        f["loss_before"], f["loss_after"] = (f.base - f.before) * 100, (f.base - f.after) * 100
        f.to_csv("results/h3_finetune.csv", index=False)
        s = f.groupby("config")[["frac", "loss_before", "loss_after"]].agg(["mean", "std", "count"]).round(3)
        s.to_csv("results/h3_finetune_summary.csv")
        print("fine-tune\n", s.to_string())

    d5 = h5(runs)
    d5.to_csv("results/h5.csv", index=False)
    r5 = h5_ratios(d5)
    r5.to_csv("results/h5_ratios.csv", index=False)
    s5 = [dict(cfg=c, tmin_first=g.tmin_first.mean(), tmin_last=g.tmin_last.mean(), genuine=g.genuine_fall.mean(),
               ratio_min=boot(g.ratio_min)[0], rmin_lo=boot(g.ratio_min)[1], rmin_hi=boot(g.ratio_min)[2],
               tmax_first=g.tmax_first.mean(), tmax_last=g.tmax_last.mean(), ratio_max=boot(g.ratio_max)[0],
               rmax_lo=boot(g.ratio_max)[1], rmax_hi=boot(g.ratio_max)[2]) for c, g in r5.groupby("cfg")]
    s5 = pd.DataFrame(s5).round(3)
    s5.to_csv("results/h5_summary.csv", index=False)
    print("\nH5\n", s5.to_string())

    df = h4()
    if len(df):
        df.to_csv("results/h4_long.csv", index=False)
        t4 = h4_tests(df)
        t4.round(3).to_csv("results/h4_regression.csv", index=False)
        print("\nH4\n", t4.round(3).to_string())
        fin = df[df.frac == 1].groupby(["task", "eta", "n", "H", "eps"])[["lo", "hi", "lo250", "head_lo", "lo_y", "test_acc", "train_acc"]].mean().round(3)
        fin.to_csv("results/h4_cells.csv")
        init = df[df.frac == 0].groupby(["task", "eps"])[["lo", "hi", "head_lo"]].mean().round(3)
        init.to_csv("results/h4_init.csv")
        print(init.to_string())
        # replacement bound I <= E||A - mean||^2 / (2 eps^2). True I lies in [lo, hi] so only lo can be held to it:
        # Jensen gives hi <= 2 * var_bound, so hi may legitimately exceed var_bound.
        df["var_bound"] = df["var"] / (2 * df.eps ** 2)
        chk = df[df.frac == 1].assign(holds=lambda d: d.lo <= d.var_bound * (1 + 1e-6) + 1e-6).groupby(["task", "eps"]).agg(
            lo=("lo", "mean"), hi=("hi", "mean"), var_bound=("var_bound", "mean"), holds=("holds", "mean")).round(3)
        chk.to_csv("results/h4_var_bound.csv")
        tr = df[df.eps == 2.0].groupby(["task", "eta", "n", "frac"])[["lo", "test_acc", "train_acc"]].mean().round(3).reset_index()
        tr.to_csv("results/h4_trajectory.csv", index=False)
        # does the excess over H(Y|X) fall with n at the rate an O(1/sqrt n) term needs?
        g = df[(df.frac == 1) & (df.eps == 2.0)].groupby(["task", "eta", "n", "seed", "H"], as_index=False).lo.mean()
        g["excess"] = g.lo - g.H
        p = g.groupby(["task", "eta", "n"]).excess.mean().unstack("n").round(3)
        p.to_csv("results/h4_excess.csv")
        print(p.to_string())
        # Pearson-free statement: implied C = excess * sqrt(n) must stay bounded
        q = (g.assign(C=g.excess * np.sqrt(g.n))).groupby(["task", "eta", "n"]).C.mean().unstack("n").round(1)
        q.to_csv("results/h4_implied_C.csv")
        print(q.to_string())
        print(chk.to_string())


if __name__ == "__main__":
    main()
