"""Compare a same-seed rerun with the original: is the spectral trajectory reproducible?"""
import json
import numpy as np

a, b = "results/runs/ag_L6_s0", "results/runs_repro/ag_L6_s0"
ma, mb = np.load(a + ".npz")["m"], np.load(b + ".npz")["m"]
ja, jb = json.load(open(a + ".json")), json.load(open(b + ".json"))
ha, hb = ma[..., 1].mean(2), mb[..., 1].mean(2)
print("test", ja["test"], jb["test"], "identical m:", np.array_equal(ma, mb))
print("max relative head erank difference", np.abs(ha / hb - 1).max(), "at step 0:", np.abs(ha[0] / hb[0] - 1).max())
print("layer-mean relative difference at end", np.abs(ha[-1].mean(1) / hb[-1].mean(1) - 1).max())
