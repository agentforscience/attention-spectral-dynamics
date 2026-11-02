"""Transformer encoder that exposes per-head attention and supports head masking."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d, heads, attn_drop):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.drop = nn.Dropout(attn_drop)

    def forward(self, x, pad, head_mask=None):
        B, T, _ = x.shape
        split = lambda z: z.view(B, T, self.h, self.dk).transpose(1, 2)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        s = q @ k.transpose(-1, -2) / math.sqrt(self.dk)
        if pad is not None:
            s = s.masked_fill(pad[:, None, None, :], float("-inf"))
        a = s.softmax(-1)
        y = self.drop(a) @ v  # [B, H, T, dk]
        if head_mask is not None:  # head_mask: [H], 1 keeps, 0 removes (no rescaling)
            y = y * head_mask.view(1, -1, 1, 1)
        return self.o(y.transpose(1, 2).reshape(B, T, -1)), a


class Block(nn.Module):
    def __init__(self, d, heads, mlp, drop, attn_drop, pre_ln, act):
        super().__init__()
        self.pre_ln = pre_ln
        self.attn = Attention(d, heads, attn_drop)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.fc1, self.fc2 = nn.Linear(d, mlp), nn.Linear(mlp, d)
        self.act, self.drop = act, nn.Dropout(drop)

    def mlp(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))

    def forward(self, x, pad, head_mask=None):
        if self.pre_ln:
            y, a = self.attn(self.n1(x), pad, head_mask)
            x = x + self.drop(y)
            return x + self.drop(self.mlp(self.n2(x))), a
        y, a = self.attn(x, pad, head_mask)
        x = self.n1(x + self.drop(y))
        return self.n2(x + self.drop(self.mlp(x))), a


class Encoder(nn.Module):
    """kind='text' takes token ids (0=pad, 1=CLS already placed); kind='vit' takes images."""

    def __init__(self, kind, classes, vocab=0, max_len=65, d=192, heads=8, layers=6, mlp_ratio=2,
                 drop=0.1, attn_drop=0.1, pre_ln=True, act="gelu", patch=4):
        super().__init__()
        self.kind = kind
        self.patch = patch
        if kind == "text":
            self.embed = nn.Embedding(vocab, d)
        else:
            self.embed = nn.Conv2d(3, d, patch, patch)
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.normal_(self.cls, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        nn.init.normal_(self.pos, std=0.02)
        self.in_norm = nn.LayerNorm(d) if not pre_ln else nn.Identity()
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList(
            Block(d, heads, d * mlp_ratio, drop, attn_drop, pre_ln, F.gelu) for _ in range(layers))
        self.final = nn.LayerNorm(d) if pre_ln else nn.Identity()
        self.head = nn.Linear(d, classes)

    def forward(self, x, head_masks=None, want_attn=False):
        if self.kind == "text":
            pad = x == 0
            z = self.embed(x)
        else:
            z = self.embed(x).flatten(2).transpose(1, 2)
            z = torch.cat([self.cls.expand(len(z), -1, -1), z], 1)
            pad = None
        z = self.drop(self.in_norm(z + self.pos[:, : z.size(1)]))
        attns = []
        for i, b in enumerate(self.blocks):
            z, a = b(z, pad, None if head_masks is None else head_masks[i])
            if want_attn:
                attns.append(a)
        logits = self.head(self.final(z)[:, 0])
        return (logits, attns, pad) if want_attn else logits


def load_bert_weights(model, state):
    """Copy google-bert/bert-base-uncased tensors into a post-LN Encoder of matching shape."""
    pfx = "bert." if any(k.startswith("bert.") for k in state) else ""
    g = lambda n: state[pfx + n]
    with torch.no_grad():
        model.embed.weight.copy_(g("embeddings.word_embeddings.weight"))
        # segment 0 embedding is a constant, so it can live inside the position table
        seg = g("embeddings.token_type_embeddings.weight")[0]
        model.pos.copy_((g("embeddings.position_embeddings.weight")[: model.pos.size(1)] + seg)[None])
        model.in_norm.weight.copy_(g("embeddings.LayerNorm.gamma"))
        model.in_norm.bias.copy_(g("embeddings.LayerNorm.beta"))
        for i, b in enumerate(model.blocks):
            p = f"encoder.layer.{i}."
            pairs = [(b.attn.q, "attention.self.query"), (b.attn.k, "attention.self.key"),
                     (b.attn.v, "attention.self.value"), (b.attn.o, "attention.output.dense"),
                     (b.n1, "attention.output.LayerNorm"), (b.fc1, "intermediate.dense"),
                     (b.fc2, "output.dense"), (b.n2, "output.LayerNorm")]
            for mod, name in pairs:
                ln = name.endswith("LayerNorm")  # checkpoint keeps the legacy gamma/beta names
                mod.weight.copy_(g(p + name + (".gamma" if ln else ".weight")))
                mod.bias.copy_(g(p + name + (".beta" if ln else ".bias")))
