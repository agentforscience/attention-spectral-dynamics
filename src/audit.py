"""Check report statements that were read off figures or summaries against the raw logs."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from analyze import head_rank, load_runs

runs = load_runs()
by = {}
for r in runs:
    by.setdefault(r["cfg"], []).append(r)
for c, rs in sorted(by.items()):
    e = np.mean([head_rank(r).mean(2) for r in rs], 0)  # [ck, layer]
    order = (e[:, 0] > e[:, -1]).mean()
    mono = (np.diff(e, axis=1) < 0).all(1).mean()
    a = e.mean(1)
    i10 = int(np.argmin(abs(rs[0]["frac"] - .1)))
    share10 = (a[i10] - a[0]) / (a[-1] - a[0]) if a[-1] != a[0] else np.nan
    row = np.mean([r["m"][0, :, :, :, 6].mean() for r in rs])
    print(f"{c:18s} steps {rs[0]['j']['n_step']:6d} L1>Last share {order:.2f} strict depth order {mono:.2f} rise share by 10% {share10:.2f} init row_h {row:.3f}")
b = by["bert_ag"]
for name, c in (("init", 0), ("20%", 20), ("end", -1)):
    print("bert above .6n", name, np.mean([(head_rank(r)[c] > .6).mean() for r in b]), "by layer", np.mean([(head_rank(r)[c] > .6).mean(1) for r in b], 0).round(2))
print("bert head-rank change |end-init| median", np.median(np.abs(head_rank(b[0])[-1] - head_rank(b[0])[0])))
df = pd.read_csv("results/h4_long.csv")
f = df[df.frac == 1]
print("var bound exact hold", (f.lo <= f["var"] / (2 * f.eps ** 2) * (1 + 1e-6) + 1e-6).mean(), len(f))
print("all checkpoints", (df.lo <= df["var"] / (2 * df.eps ** 2) * (1 + 1e-6) + 1e-6).mean(), len(df))
print("max job minutes", pd.read_csv("results/accuracy.csv").groupby("cfg").minutes.max().round(1).to_dict())
h = pd.read_csv("results/h5.csv")
print(h[h.cfg == "cifar_L6_bigqk"].groupby("layer")["drop"].mean().round(3).tolist(), h[h.cfg == "ag_L6_bigqk"].groupby("layer")["drop"].mean().round(3).tolist())
