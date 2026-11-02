"""Sanity checks: metrics on known matrices, MI bounds against Monte Carlo, BERT port against HF."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from spectral import METRICS, head_metrics, mi_bounds
from model import Encoder, load_bert_weights

n, T = 8, 12
I = torch.zeros(T, T); I[:n, :n] = torch.eye(n)
U = torch.zeros(T, T); U[:n, :n] = 1 / n
P = torch.zeros(T, T); P[:n, 0] = 1
a = torch.stack([I, U, P])[None]
m = head_metrics(a, torch.tensor([n]))[0]
ix = METRICS.index
assert abs(m[0, ix("S")] - math.log(n)) < 1e-4 and abs(m[0, ix("erank")] - n) < 1e-3 and m[0, ix("row_h")] < 1e-6
assert m[1, ix("S")] < 1e-5 and abs(m[1, ix("row_h")] - 1) < 1e-5 and m[1, ix("numrank")] == 1
assert m[2, ix("S")] < 1e-5 and m[2, ix("row_h")] < 1e-6
print("metrics ok: identity S=log n, uniform S=0 row_h=1, pointer S=0 row_h=0")

torch.manual_seed(0)
M, d, eps = 200, 5, 0.5
c = torch.randn(M, d) * 0.6
d2 = torch.cdist(c, c) ** 2
lo, hi = mi_bounds(d2, eps)
S = 40000
z = c[torch.randint(0, M, (S,))] + eps * torch.randn(S, d)
lp = torch.logsumexp(-(torch.cdist(z, c) ** 2) / (2 * eps ** 2), 1) - math.log(M) - d / 2 * math.log(2 * math.pi * eps ** 2)
true = -lp.mean().item() - d / 2 * math.log(2 * math.pi * math.e * eps ** 2)
print(f"MI bounds lo {lo:.3f} <= MC {true:.3f} <= hi {hi:.3f}, ceiling {math.log(M):.3f}")
assert lo - .02 <= true <= hi + .02

if len(sys.argv) > 1:
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import BertModel
    p = snapshot_download("google-bert/bert-base-uncased", local_files_only=True)
    hf = BertModel.from_pretrained(p, attn_implementation="eager").eval()
    me = Encoder("text", classes=4, vocab=30522, max_len=64, d=768, heads=12, layers=12, mlp_ratio=4, pre_ln=False).eval()
    load_bert_weights(me, load_file(p + "/model.safetensors"))
    x = torch.randint(5, 30000, (2, 16)); x[:, 0] = 101
    with torch.no_grad():
        ho = hf(x, output_attentions=True)
        _, at, _ = me(x, want_attn=True)
    print([round((u - v).abs().max().item(), 5) for u, v in zip(ho.attentions, at)], ho.attentions[0].shape, at[0].shape, len(ho.attentions))
    err = max((u - v).abs().max().item() for u, v in zip(ho.attentions, at))
    print("BERT port max |A diff|", err)
    assert err < 0.01  # residual comes from LayerNorm eps 1e-5 here vs 1e-12 in HF
