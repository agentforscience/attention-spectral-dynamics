"""Secondary H3 arm: after removing heads with S < tau*log n, fine-tune the rest for 5% of the training steps at lr/10."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F

from model import Encoder
from train import BASE, CONFIGS, accuracy, augment, mk_masks  # noqa


def main(a):
    cfg = dict(BASE, **CONFIGS[a.config])
    tag = f"results/runs/{a.config}_s{a.seed}"
    out = f"{tag}.ft.json"
    if os.path.exists(out):
        return
    j = json.load(open(tag + ".json"))
    S = np.array(j["pruning"]["S_norm"])
    off = [(int(l), int(h)) for l, h in zip(*np.where(S < a.tau))]
    torch.manual_seed(a.seed + 100)
    D = torch.load(f"datasets/prepared/{cfg['data']}.pt", weights_only=False)
    kind = "vit" if cfg["data"] == "cifar" else "text"
    kw = dict(classes=D["classes"], max_len=D["max_len"], d=cfg["d"], heads=cfg["heads"], layers=cfg["layers"],
              mlp_ratio=cfg["mlp_ratio"], pre_ln=cfg["pre_ln"])
    model = Encoder("vit", **kw) if kind == "vit" else Encoder("text", vocab=D["vocab"], **kw)
    model.load_state_dict(torch.load(tag + ".pt", map_location="cpu"))
    try:
        model.cuda()
        xtr, ytr = D["train"][0].cuda(), D["train"][1].cuda()
        xte, yte = D["test"][0].cuda(), D["test"][1].cuda()
        masks = mk_masks(model, off)
        before = accuracy(model, xte, yte, kind, masks)
        steps = int(.05 * j["n_step"])
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"] / 10, weight_decay=cfg["wd"])
        model.train()
        for _ in range(steps):
            idx = torch.randint(0, len(xtr), (cfg["bs"],), device=xtr.device)
            xb = augment(xtr[idx]) if kind == "vit" else xtr[idx]
            loss = F.cross_entropy(model(xb, masks), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        after = accuracy(model, xte, yte, kind, masks)
    except RuntimeError as e:
        print("no room", str(e)[:200])
        os._exit(75)
    json.dump(dict(config=a.config, seed=a.seed, tau=a.tau, frac=len(off) / S.size, base=j["test"], before=before, after=after, steps=steps), open(out, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tau", type=float, default=0.3)
    main(ap.parse_args())
