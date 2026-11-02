"""Tokenise AG News / SST-2 and decode CIFAR-10 into tensors under datasets/prepared/.

Text token ids: 0 pad, 1 CLS, 2 UNK. Dev split comes from train (AG News, CIFAR); for SST-2, whose train
split holds overlapping phrases of the same sentences as nothing in validation, the 872 validation
sentences are halved into dev and test because the official test labels are hidden.
"""
import collections
import glob
import io
import os
import re
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

HF = os.environ.get("HF_HOME", "/data/transformers")
OUT = "datasets/prepared"
TOK = re.compile(r"[a-z0-9']+|[^\sa-z0-9]")


def table(name, split):
    f = glob.glob(f"{HF}/hub/datasets--{name}/snapshots/*/data/{split}-*.parquet")[0]
    return pq.read_table(f)


def encode(texts, vocab, max_len):
    out = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, t in enumerate(texts):
        ids = [1] + [vocab.get(w, 2) for w in TOK.findall(t.lower())][: max_len - 1]
        out[i, : len(ids)] = torch.tensor(ids)
    return out


def text_set(name, col, max_len, vocab_size):
    tr = table(name, "train")
    xtr, ytr = tr[col].to_pylist(), tr["label"].to_pylist()
    cnt = collections.Counter(w for t in xtr for w in TOK.findall(t.lower()))
    vocab = {w: i + 3 for i, (w, _) in enumerate(cnt.most_common(vocab_size - 3))}
    return xtr, ytr, vocab


def split_train(n, n_dev, seed=0):
    p = np.random.RandomState(seed).permutation(n)
    return torch.from_numpy(p[n_dev:]), torch.from_numpy(p[:n_dev])


def ag():
    xtr, ytr, vocab = text_set("fancyzhx--ag_news", "text", 64, 30000)
    te = table("fancyzhx--ag_news", "test")
    X, Y = encode(xtr, vocab, 64), torch.tensor(ytr)
    a, b = split_train(len(X), 2000)
    torch.save(dict(train=(X[a], Y[a]), dev=(X[b], Y[b]),
                    test=(encode(te["text"].to_pylist(), vocab, 64), torch.tensor(te["label"].to_pylist())),
                    vocab=len(vocab) + 3, classes=4, max_len=64), f"{OUT}/ag.pt")
    print("ag", X.shape, (X > 0).sum(1).float().mean().item())


def sst():
    xtr, ytr, vocab = text_set("stanfordnlp--sst2", "sentence", 48, 15000)
    va = table("stanfordnlp--sst2", "validation")
    X, Y = encode(va["sentence"].to_pylist(), vocab, 48), torch.tensor(va["label"].to_pylist())
    p = torch.from_numpy(np.random.RandomState(0).permutation(len(X)))
    d, t = p[: len(X) // 2], p[len(X) // 2:]
    torch.save(dict(train=(encode(xtr, vocab, 48), torch.tensor(ytr)), dev=(X[d], Y[d]), test=(X[t], Y[t]),
                    vocab=len(vocab) + 3, classes=2, max_len=48), f"{OUT}/sst.pt")
    print("sst", len(xtr), (X > 0).sum(1).float().mean().item())


def bert():
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    tr, te = table("fancyzhx--ag_news", "train"), table("fancyzhx--ag_news", "test")
    enc = lambda t: torch.tensor(tk(t, max_length=64, truncation=True, padding="max_length")["input_ids"])
    X, Y = enc(tr["text"].to_pylist()), torch.tensor(tr["label"].to_pylist())
    a, b = split_train(len(X), 2000)
    torch.save(dict(train=(X[a], Y[a]), dev=(X[b], Y[b]),
                    test=(enc(te["text"].to_pylist()), torch.tensor(te["label"].to_pylist())),
                    vocab=tk.vocab_size, classes=4, max_len=64), f"{OUT}/ag_bert.pt")
    print("ag_bert", X.shape)


def cifar():
    def load(s):
        t = pq.read_table(f"datasets/cifar10/{s}.parquet")
        cols = t.column_names
        imcol = "img" if "img" in cols else "image"
        ims = [np.asarray(Image.open(io.BytesIO(r["bytes"])).convert("RGB")) for r in t[imcol].to_pylist()]
        lab = "label" if "label" in cols else "labels"
        return torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2), torch.tensor(t[lab].to_pylist())
    X, Y = load("train")
    a, b = split_train(len(X), 2000)
    torch.save(dict(train=(X[a], Y[a]), dev=(X[b], Y[b]), test=load("test"), classes=10, max_len=65),
               f"{OUT}/cifar.pt")
    print("cifar", X.shape)


if __name__ == "__main__":
    for w in sys.argv[1:] or ["ag", "sst", "bert", "cifar"]:
        globals()[w]()
