"""Block-diagonal Sylvester-Hadamard transforms with per-vector flags.

H_2n = [[H_n, H_n], [H_n, -H_n]] / sqrt(2) (natural order, symmetric, involutory: H H = I).  fwht(x, rb) applies
H_rb to every consecutive rb-vector of the last dim; sel_fwht only to the vectors whose flag is set.  Because
(x H)(W H)^T = x W^T, rotating the same vectors of an activation and of the weight's input dim is exact.
"""
from __future__ import annotations

import math

import torch


def fwht(x: torch.Tensor, rb: int) -> torch.Tensor:
    if rb == 1:
        return x
    shp, K = x.shape, x.shape[-1]
    assert K % rb == 0 and rb & (rb - 1) == 0, (K, rb)
    y = x.reshape(-1, K // rb, rb)
    h = 1
    while h < rb:
        y = y.reshape(-1, K // rb, rb // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2)
        h *= 2
    return y.reshape(shp) * (1.0 / math.sqrt(rb))


def sylvester(rb: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < rb:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(rb)).to(device=device, dtype=dtype)


def sel_fwht(x: torch.Tensor, mask: torch.Tensor | None, rb: int) -> torch.Tensor:
    """rotate the rb-vectors of the last dim whose flag is set; mask [..., K/rb] must broadcast against x's vectors."""
    if mask is None or not bool(mask.any()):
        return x
    y = fwht(x, rb)
    if bool(mask.all()):
        return y
    shp = x.shape
    m = mask.to(x.device)[..., None]
    return torch.where(m, y.reshape(*shp[:-1], -1, rb), x.reshape(*shp[:-1], -1, rb)).reshape(shp)


def pattern_mask(nvec: int, per_unit: int, p: int, device=None) -> torch.Tensor:
    """flag vector p of every unit of per_unit consecutive vectors."""
    return (torch.arange(nvec, device=device) % per_unit) == p
