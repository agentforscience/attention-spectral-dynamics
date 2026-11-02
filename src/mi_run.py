"""H4 grid: estimate I(A;X) under controlled H(Y|X).

Tasks: 'order' (synthetic, Y = does marker a precede marker b; true H(Y|X)=0 before noise) and 'ag'.
Symmetric label noise eta on training labels only gives H(Y|X) = h2(eta) + eta log(K-1). Clean test data measure X.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F

from model import Encoder
from spectral import mi_bounds

EPS = [0.2, 0.5, 1.0, 2.0, 4.0]
FRACS = [0, .05, .1, .2, .4, .6, .8, 1.0]
M = 2000


def order_task(n, rng, T=32, V=64):
    """Each sequence holds markers 3 and 4 once each among random tokens 5..V-1. CLS id 1 in front."""
    x = torch.randint(5, V, (n, T), generator=rng)
    pa = torch.rand(n, T, generator=rng).argsort(1)[:, :2]
    i = torch.arange(n)
    x[i, pa[:, 0]] = 3
    x[i, pa[:, 1]] = 4
    y = (pa[:, 0] < pa[:, 1]).long()
    return torch.cat([torch.ones(n, 1, dtype=torch.long), x], 1), y, V, 2


def hyx(eta, K):
    if eta == 0:
        return 0.0
    return -eta * math.log(eta) - (1 - eta) * math.log(1 - eta) + eta * math.log(K - 1)


@torch.no_grad()
def attn_stack(model, x):
    model.eval()
    out = [[] for _ in model.blocks]
    for xb in x.split(256):
        _, attns, pad = model(xb, want_attn=True)
        for l, a in enumerate(attns):
            a = a.masked_fill(pad[:, None, :, None], 0)  # pad rows carry no attention
            out[l].append(a.reshape(a.size(0), a.size(1), -1).cpu())
    return [torch.cat(o) for o in out]  # per layer [M, H, T*T]


@torch.no_grad()
def measure(model, x, y):
    rows = []
    for l, t in enumerate(attn_stack(model, x)):
        # d2 over all heads of a layer: squared Frobenius distance in the joint representation
        per = [torch.cdist(t[:, h].to(x.device), t[:, h].to(x.device)) ** 2 for h in range(t.size(1))]
        joint = sum(per)
        for eps in EPS:
            lo, hi, lo_y, hi_y = mi_bounds(joint, eps, y)
            sub = joint[:250][:, :250]
            lo250, hi250 = mi_bounds(sub, eps)
            ph = [mi_bounds(p, eps) for p in per]
            rows.append(dict(layer=l, eps=eps, var=joint.mean().item() / 2, lo=lo, hi=hi, lo_y=lo_y, hi_y=hi_y, lo250=lo250, hi250=hi250,
                             head_lo=float(np.mean([p[0] for p in ph])), head_hi=float(np.mean([p[1] for p in ph]))))
    return rows


def run(a):
    out = f"results/mi/{a.task}_e{a.eta}_n{a.n}_s{a.seed}.json"
    if os.path.exists(out):
        return
    torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(1000 + a.seed)
    if a.task == "order":
        xtr, ytr, V, K = order_task(a.n, g)
        xte, yte, _, _ = order_task(M, torch.Generator().manual_seed(7))
        ln = 33
    else:
        D = torch.load("datasets/prepared/ag.pt", weights_only=False)
        p = torch.randperm(len(D["train"][0]), generator=g)[: a.n]
        xtr, ytr = D["train"][0][p], D["train"][1][p]
        xte, yte = D["test"][0][:M], D["test"][1][:M]
        V, K, ln = D["vocab"], 4, 64
    # symmetric noise: with prob eta replace by a uniformly chosen other class
    flip = torch.rand(a.n, generator=g) < a.eta
    ytr = torch.where(flip, (ytr + torch.randint(1, K, (a.n,), generator=g)) % K, ytr)
    model = Encoder("text", classes=K, vocab=V, max_len=ln, d=128, heads=8, layers=4, drop=0.0, attn_drop=0.0)
    try:
        dev = torch.device("cuda")
        model.to(dev)
        xtr, ytr, xte, yte = xtr.to(dev), ytr.to(dev), xte.to(dev), yte.to(dev)
        measure(model, xte, yte)
    except RuntimeError as e:
        print("no room", str(e)[:200])
        os._exit(75)
    steps = a.steps
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min((t + 1) / 100, 0.5 * (1 + np.cos(np.pi * min(t / steps, 1)))))
    ck = {int(f * steps): f for f in FRACS}
    log = []
    t0 = time.time()
    try:
        for step in range(steps + 1):
            if step in ck:
                model.eval()
                with torch.no_grad():
                    test_acc = sum((model(xb).argmax(1) == yb).sum().item() for xb, yb in zip(xte.split(512), yte.split(512))) / M
                    train_acc = sum((model(xb).argmax(1) == yb).sum().item() for xb, yb in zip(xtr[:M].split(512), ytr[:M].split(512))) / min(M, a.n)
                log.append(dict(frac=ck[step], test_acc=test_acc, train_acc=train_acc, rows=measure(model, xte, yte)))
                print(step, test_acc, train_acc, time.time() - t0, flush=True)
            if step == steps:
                break
            model.train()
            idx = torch.randint(0, a.n, (64,), device=dev)
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    except RuntimeError as e:
        print("oom", str(e)[:200])
        os._exit(75)
    json.dump(dict(task=a.task, eta=a.eta, n=a.n, seed=a.seed, H=hyx(a.eta, K), log=log), open(out, "w"))
    print("DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    run(ap.parse_args())
