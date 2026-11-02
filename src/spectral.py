"""Spectral metrics of attention matrices and the pairwise-distance MI bounds."""
import math
import torch

METRICS = ["S", "erank", "stable", "intrinsic", "rv", "numrank", "row_h"]


@torch.no_grad()
def head_metrics(a, lens, chunk=1024):
    """a: [B, H, T, T] attention; lens: [B] true lengths. Returns [B, H, len(METRICS)] float32 on CPU.

    S, erank, row_h are computed on exact n. Rows/columns beyond n are zeroed so padding adds only zero singular values.
    row_h is mean row Shannon entropy divided by log n. erank etc. are unnormalised counts.
    """
    B, H, T, _ = a.shape
    idx = torch.arange(T, device=a.device)[None]
    valid = (idx < lens[:, None]).float()
    a = a.cpu()  # LAPACK is ~50x faster than the GPU driver here for these small matrices
    valid, lens = valid.cpu(), lens.cpu()
    m = a.double() * valid[:, None, :, None] * valid[:, None, None, :]
    flat = m.reshape(B * H, T, T)
    sv = torch.cat([torch.linalg.svdvals(c) for c in flat.split(chunk)]).reshape(B, H, T)
    p2 = sv ** 2 / (sv ** 2).sum(-1, keepdim=True)
    S = -(p2 * p2.clamp_min(1e-300).log()).sum(-1)
    p1 = sv / sv.sum(-1, keepdim=True)
    rv = (-(p1 * p1.clamp_min(1e-300).log()).sum(-1)).exp()
    stable = (sv ** 2).sum(-1) / sv[..., 0] ** 2
    intrinsic = sv.sum(-1) ** 2 / (sv ** 2).sum(-1)
    numrank = (sv > 0.01 * sv[..., :1]).sum(-1).double()
    row = -(m * m.clamp_min(1e-300).log()).sum(-1).sum(-1) / lens[:, None].double()
    row_h = row / lens[:, None].double().log().clamp_min(1e-9)
    return torch.stack([S, S.exp(), stable, intrinsic, rv, numrank, row_h], -1).float().cpu()


def mi_bounds(d2, eps, labels=None):
    """Kolchinsky-Tracey bounds on I(T+eps Z; X) for X uniform on the M points, from squared distances d2.

    Returns (I_lower_BD, I_upper_KL). Both are capped at log M by construction.
    If labels is given, also returns the same pair for I(T+eps Z; Y).
    """
    M = d2.size(0)
    logm = math.log(M)
    def h(alpha):  # entropy of the mixture minus entropy of one component
        return -(torch.logsumexp(-d2 / (alpha * eps ** 2), 1) - logm).mean().item()
    lo, hi = h(8), h(2)
    if labels is None:
        return lo, hi
    lo_y, hi_y = lo, hi
    for c in labels.unique():
        i = (labels == c).nonzero().squeeze(1)
        w = len(i) / M
        sub = d2[i][:, i]
        def hc(alpha):
            return -(torch.logsumexp(-sub / (alpha * eps ** 2), 1) - math.log(len(i))).mean().item()
        lo_y -= w * hc(8)
        hi_y -= w * hc(2)
    return lo, hi, lo_y, hi_y
