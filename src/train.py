"""Train one configuration, logging per-head spectral metrics at 41 checkpoints, then run the pruning sweep.

Exit code 75 means "no GPU room", which the queue treats as a retry rather than a failure.
"""
import argparse
import fcntl
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F

from model import Encoder, load_bert_weights
from spectral import METRICS, head_metrics

FRACS = [0.0] + [round(i * 0.01, 2) for i in range(1, 21)] + [round(0.2 + i * 0.04, 2) for i in range(1, 21)]
TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
BASE = dict(layers=6, pre_ln=True, epochs=6, bs=128, lr=5e-4, wd=0.05, shuffle=False, d=192, heads=8, mlp_ratio=2)
CONFIGS = {
    "ag_L6": dict(data="ag"),
    "ag_L3": dict(data="ag", layers=3),
    "ag_L12": dict(data="ag", layers=12),
    "ag_L6_postln": dict(data="ag", pre_ln=False),
    "ag_L6_shuffled": dict(data="ag", shuffle=True),
    "sst_L6": dict(data="sst"),
    "cifar_L6": dict(data="cifar", epochs=40, bs=256, lr=1e-3),
    "cifar_L3": dict(data="cifar", epochs=40, bs=256, lr=1e-3, layers=3),
    "cifar_L12": dict(data="cifar", epochs=40, bs=256, lr=1e-3, layers=12),
    "cifar_L6_shuffled": dict(data="cifar", epochs=40, bs=256, lr=1e-3, shuffle=True),
    # post-plan addition: start from sharply peaked attention, to test whether "no collapse" is an init artefact
    "ag_L6_bigqk": dict(data="ag", qk_gain=3.0),
    "cifar_L6_bigqk": dict(data="cifar", epochs=40, bs=256, lr=1e-3, qk_gain=3.0),
    "bert_ag": dict(data="ag_bert", layers=12, pre_ln=False, epochs=1, bs=32, lr=2e-5, wd=0.01, d=768, heads=12,
                    mlp_ratio=4, pretrained=True, n_train=40000),
}


def augment(x):
    """Random crop (pad 4) and flip on GPU, returns normalised floats."""
    B = x.size(0)
    p = F.pad(x.float() / 255, (4, 4, 4, 4), mode="reflect")
    ox, oy = torch.randint(0, 9, (2, B), device=x.device)
    g = torch.arange(32, device=x.device)
    i, j = oy[:, None] + g[None], ox[:, None] + g[None]
    out = p[torch.arange(B, device=x.device)[:, None, None, None], torch.arange(3, device=x.device)[None, :, None, None],
            i[:, None, :, None], j[:, None, None, :]]
    flip = torch.rand(B, device=x.device) < 0.5
    out[flip] = out[flip].flip(-1)
    return (out - 0.5) / 0.25


def prep(x, kind):
    return x if kind == "text" else (x.float() / 255 - 0.5) / 0.25


def lengths(x, kind, n=65):
    return (x > 0).sum(1) if kind == "text" else torch.full((len(x),), n, device=x.device)


@torch.no_grad()
def probe(model, x, kind):
    model.eval()
    out = [[] for _ in model.blocks]
    for xb in x.split(64):
        _, attns, _ = model(prep(xb, kind), want_attn=True)
        ln = lengths(xb, kind)
        for l, a in enumerate(attns):
            out[l].append(head_metrics(a, ln))
    return torch.stack([torch.cat(o) for o in out])  # [layers, samples, heads, metrics]


@torch.no_grad()
def accuracy(model, x, y, kind, masks=None, bs=512):
    model.eval()
    c = 0
    for xb, yb in zip(x.split(bs), y.split(bs)):
        c += (model(prep(xb, kind), masks).argmax(1) == yb).sum().item()
    return c / len(y)


def mk_masks(model, heads_off):
    H = model.blocks[0].attn.h
    m = [torch.ones(H, device=next(model.parameters()).device) for _ in model.blocks]
    for l, h in heads_off:
        m[l][h] = 0
    return m


def prune_sweep(model, cfg, seed, dev, test, kind, final_probe, lr):
    """Prune by normalised S under each threshold and against same-count baselines. Also single-head ablation on dev."""
    Ln = lengths(dev[0][:256], kind).float().log().cpu()
    s = (final_probe[..., METRICS.index("S")] / Ln[None, :, None]).mean(1).numpy()  # [layers, heads]
    L, H = s.shape
    all_heads = [(l, h) for l in range(L) for h in range(H)]
    order = sorted(all_heads, key=lambda t: s[t])
    x, y, xd, yd = *test, *dev
    base = accuracy(model, x, y, kind)
    # one-at-a-time ablation on dev gives the importance oracle
    base_dev = accuracy(model, xd, yd, kind)
    imp = {t: base_dev - accuracy(model, xd, yd, kind, mk_masks(model, [t])) for t in all_heads}
    oracle = sorted(all_heads, key=lambda t: imp[t])
    rng = np.random.RandomState(seed)
    rows = []
    counts = sorted({sum(s[t] < tau for t in all_heads) for tau in TAUS} | {round(f * L * H) for f in (.1, .25, .5, .75)})
    tau_of = {}
    for tau in TAUS:
        tau_of.setdefault(int(sum(s[t] < tau for t in all_heads)), []).append(tau)
    for k in counts:
        k = int(k)
        r = dict(k=k, frac=k / (L * H), taus=tau_of.get(k, []),
                 low_S=accuracy(model, x, y, kind, mk_masks(model, order[:k])),
                 high_S=accuracy(model, x, y, kind, mk_masks(model, order[::-1][:k])),
                 oracle=accuracy(model, x, y, kind, mk_masks(model, oracle[:k])))
        rand = []
        for _ in range(10):
            pick = [all_heads[i] for i in rng.choice(len(all_heads), k, replace=False)]
            rand.append(accuracy(model, x, y, kind, mk_masks(model, pick)))
        r["random"] = float(np.mean(rand))
        r["random_sd"] = float(np.std(rand))
        rows.append(r)
    return dict(base=base, base_dev=base_dev, S_norm=s.tolist(), importance=[imp[t] for t in all_heads], rows=rows)


def run(a):
    cfg = dict(BASE, **CONFIGS[a.config])
    if a.fast:
        cfg["epochs"] = 1
    rundir = "results/runs" + os.environ.get("RUN_SUFFIX", "")  # RUN_SUFFIX=_repro keeps reproducibility reruns apart
    os.makedirs(rundir, exist_ok=True)
    out = f"{rundir}/{a.config}_s{a.seed}"
    if os.path.exists(out + ".json") and not a.fast:
        print("done already")
        return
    lock = open(out + ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("LOCKED")
        return
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    D = torch.load(f"datasets/prepared/{cfg['data']}.pt", weights_only=False)
    kind = "vit" if cfg["data"] == "cifar" else "text"
    xtr, ytr = D["train"]
    if "n_train" in cfg:
        p = torch.randperm(len(xtr))[: cfg["n_train"]]
        xtr, ytr = xtr[p], ytr[p]
    if cfg["shuffle"]:
        ytr = ytr[torch.randperm(len(ytr))]
    kw = dict(classes=D["classes"], max_len=D["max_len"], d=cfg["d"], heads=cfg["heads"], layers=cfg["layers"],
              mlp_ratio=cfg["mlp_ratio"], pre_ln=cfg["pre_ln"])
    model = Encoder("vit", **kw) if kind == "vit" else Encoder("text", vocab=D["vocab"], **kw)
    if cfg.get("qk_gain"):
        with torch.no_grad():
            for b in model.blocks:
                b.attn.q.weight.mul_(cfg["qk_gain"])
                b.attn.k.weight.mul_(cfg["qk_gain"])
    if cfg.get("pretrained"):
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        snap = snapshot_download("google-bert/bert-base-uncased", local_files_only=True)
        load_bert_weights(model, load_file(snap + "/model.safetensors"))
    try:
        dev_gpu = torch.device("cuda")
        model.to(dev_gpu)
        xtr, ytr = xtr.to(dev_gpu), ytr.to(dev_gpu)
        dev = tuple(t.to(dev_gpu) for t in D["dev"])
        test = tuple(t.to(dev_gpu) for t in D["test"])
        # warm-up forward/backward to fail fast when the slice is too small
        model(prep(xtr[: cfg["bs"]], kind)).sum().backward()
        model.zero_grad()
    except RuntimeError as e:
        print("no room:", str(e)[:200])
        os._exit(75)
    xp = dev[0][:256]
    n_step = cfg["epochs"] * (len(xtr) // cfg["bs"])
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=cfg["wd"]), dict(params=nodecay, weight_decay=0)], lr=cfg["lr"])
    warm = max(1, int(0.05 * n_step))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min((t + 1) / warm, 0.5 * (1 + np.cos(np.pi * min(t / n_step, 1)))))
    ck = {int(round(f * n_step)): f for f in FRACS}
    log, mets, t0 = [], [], time.time()
    perm, pos = torch.randperm(len(xtr), device=dev_gpu), 0
    try:
        for step in range(n_step + 1):
            if step in ck:
                mets.append(probe(model, xp, kind).numpy().astype(np.float32))
                log.append(dict(step=step, frac=ck[step], dev=accuracy(model, *dev, kind), t=time.time() - t0))
                print(json.dumps(log[-1]), flush=True)
            if step == n_step:
                break
            if pos + cfg["bs"] > len(xtr):
                perm, pos = torch.randperm(len(xtr), device=dev_gpu), 0
            idx = perm[pos:pos + cfg["bs"]]
            pos += cfg["bs"]
            model.train()
            xb = augment(xtr[idx]) if kind == "vit" else xtr[idx]
            loss = F.cross_entropy(model(xb), ytr[idx], label_smoothing=0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        np.savez_compressed(out + ".npz", m=np.stack(mets), frac=np.array(sorted(ck.values())),
                            lens=lengths(xp, kind).cpu().numpy())
        pr = prune_sweep(model, cfg, a.seed, dev, test, kind, torch.from_numpy(mets[-1]), cfg["lr"])
        tail = dict(test=pr["base"], log=log, n_step=n_step, config=a.config, seed=a.seed, pruning=pr, cfg=cfg)
        if not cfg.get("pretrained"):
            torch.save(model.state_dict(), out + ".pt")
        with open(out + ".json", "w") as f:
            json.dump(tail, f)
        print("DONE", a.config, a.seed, "test", pr["base"], "time", time.time() - t0)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print("oom mid-run:", str(e)[:200])
            os._exit(75)
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true")
    run(ap.parse_args())
