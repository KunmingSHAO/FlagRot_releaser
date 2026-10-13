"""GPTQ (Frantar et al., 2023) for every weight format of blockflag.formats, in the (selectively) rotated basis.

Columns are quantized left to right with error feedback through the Cholesky factor of H^-1 (lazy batch updates of
`blocksize` columns).  A group's scale is fixed when the loop reaches its first column, from the already-updated
weights of that group (group sizes 16 / 32 / 128 divide the block size, so the whole group is inside the current
block).  NVFP4's tensor-wide FP32 scale is taken from the initial weight.  Simplification: Hessians come from the
full-precision model (non-sequential GPTQ).
"""
from __future__ import annotations

import torch

from . import formats as FM


def _codes_with_scale(v: torch.Tensor, s: torch.Tensor, spec: str) -> torch.Tensor:
    name, _ = FM.parse(spec)
    t = v / s
    if name == "fp8":                                          # GPTQ updates can exceed the group's initial amax:
        return t.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()   # saturate (e4m3fn has no inf -> NaN)
    if name in ("mxfp4", "nvfp4"):
        return FM.e2m1(t)
    if name.startswith("int"):
        lim = 2 ** (int(name[3:]) - 1) - 1
        return torch.round(t).clamp(-lim, lim)
    raise ValueError(spec)


def _group_scale(Wg: torch.Tensor, spec: str, glob: torch.Tensor | None) -> torch.Tensor:
    """scale [N, 1] of one group of columns Wg [N, g]."""
    name, _ = FM.parse(spec)
    amax = Wg.abs().amax(1, keepdim=True)
    if name == "fp8":                                          # g x g block: one scale per block of g rows
        g = Wg.shape[1]
        N = Wg.shape[0]
        Np = (N + g - 1) // g * g
        a = torch.nn.functional.pad(amax, (0, 0, 0, Np - N)).view(-1, g).amax(1, keepdim=True)
        return (a.clamp_min(1e-12) / 448.0).repeat_interleave(g, 0)[:N]
    if name == "mxfp4":
        return torch.exp2(torch.floor(torch.log2(amax.clamp_min(1e-30))) - 2)
    if name == "nvfp4":
        # updated weights may exceed the tensor-wide amax behind glob: saturate the E4M3 scale instead of NaN
        return (amax / 6.0 / glob).clamp_max(448.0).to(torch.float8_e4m3fn).float().clamp_min(2.0 ** -9) * glob
    if name.startswith("int"):
        return amax.clamp_min(1e-12) / (2 ** (int(name[3:]) - 1) - 1)
    raise ValueError(spec)


@torch.no_grad()
def gptq_quantize(W: torch.Tensor, H: torch.Tensor, spec: str, blocksize: int = 128, percdamp: float = 0.01):
    """W [N, K], H [K, K] = sum x^T x (same basis as W) -> (codes [N, K], scales [N, K/g, 1])."""
    W = W.float().clone()
    H = H.float().clone()
    N, K = W.shape
    g = FM.group_of(spec, K)
    assert blocksize % g == 0 or g % blocksize == 0 and g <= blocksize, (g, blocksize)
    glob = W.abs().amax().clamp_min(1e-12) / (448.0 * 6.0) if FM.parse(spec)[0] == "nvfp4" else None
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += percdamp * torch.diag(H).mean() * torch.eye(K, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    codes = torch.zeros_like(W)
    scales = torch.zeros(N, K // g, 1, device=W.device)
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        W1, Hinv1 = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        Err1 = torch.zeros_like(W1)
        for i in range(i2 - i1):
            col = i1 + i
            if col % g == 0:
                s = _group_scale(W1[:, i:i + g], spec, glob)
                scales[:, col // g] = s
            w, d = W1[:, i], Hinv1[i, i]
            c = _codes_with_scale(w[:, None], s, spec)[:, 0]
            codes[:, col] = c
            err = (w - c * s[:, 0]) / d
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None]
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return codes, scales
