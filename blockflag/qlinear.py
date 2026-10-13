"""Quantized linear with vector-level rotation flags, plus the greedy calibration of those flags.

y = Q_a(x R) Q_w(W R)^T with R = blockdiag over rb-vectors of (H_rb if flag else I).  The weight side is rotated and
quantized once (stored as BF16 codes + FP32 group scales); the activation side is rotated and fake-quantized per
call.  Calibration ('greedy'): the K dim is cut into scale-coupled units of U = min(max(g_a, g_w), 128) elements,
each holding U/rb vectors; pattern p rotates vector p of every unit at once, and vector p of unit j is flagged iff
pattern p lowers unit j's per-token relative output error  sum_t ||e_{t,j}||^2 / ||y_t||^2  (TRAIN windows).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import formats as FM
from .gptq import gptq_quantize
from .hadamard import pattern_mask, sel_fwht

SHARE = {"q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv", "gate_proj": "gu", "up_proj": "gu"}
STATE = {"quant": False, "calib": False, "hess": False}
HESS: dict[str, torch.Tensor] = {}                   # share_key -> sum x^T x (CPU, FP32) for GPTQ


def share_key(name: str) -> str:
    """'layers.3.self_attn.k_proj' -> 'layers.3.qkv': projections reading the same activation share one mask."""
    parts = name.split(".")
    return ".".join(parts[:2] + [SHARE.get(parts[-1], parts[-1])])


class SelLinear(nn.Module):
    def __init__(self, lin: nn.Linear, a_spec: str, w_spec: str, rb: int):
        super().__init__()
        K = lin.in_features
        if rb < 0:                                   # "full": largest power of two dividing K (QuaRot-style baseline)
            rb = min(4096, K & -K)
        self.lin, self.a_spec, self.w_spec, self.rb = lin, a_spec, w_spec, rb
        self.U = max(rb, min(max(FM.group_of(a_spec, K), FM.group_of(w_spec, K)), 128))
        self.nvec, self.per_unit = K // rb, self.U // rb
        self.mask: torch.Tensor | None = None
        self.calib_on = False
        self.acc: torch.Tensor | None = None          # [per_unit + 2, K / U]: none, positions, all
        self.hess_key: str | None = None              # set on the one module per shared input that accumulates H
        self.track_amax = False
        self.amax: torch.Tensor | None = None         # per input channel max |x| over the calibration windows
        self.e2: torch.Tensor | None = None           # per input channel sum of x^2 (rule-based flags)
        self.e2_n = 0

    def finalize(self, mask: torch.Tensor | None, H: torch.Tensor | None = None):
        """quantize the (rotated) weight once: RTN, or GPTQ if the Hessian H (original basis) is given."""
        self.mask = mask
        W = sel_fwht(self.lin.weight.detach().float(), mask, self.rb)
        if H is None:
            c, s = FM.quantize_weight(W, self.w_spec)
        else:                                             # R H R with R the (symmetric) selective rotation
            Hr = sel_fwht(sel_fwht(H.to(W.device).float(), mask, self.rb).t().contiguous(), mask, self.rb)
            c, s = gptq_quantize(W, Hr, self.w_spec)
        self.register_buffer("w_codes", c.to(torch.bfloat16))
        self.register_buffer("w_scales", s.float())

    def patterns(self):
        """rows of self.acc: no rotation, position p of every unit (p = 0..per_unit-1), every position."""
        return [-1, *range(self.per_unit), "all"]

    def pattern(self, p, device):
        if p == -1:
            return None
        if p == "all":
            return torch.ones(self.nvec, dtype=torch.bool, device=device)
        return pattern_mask(self.nvec, self.per_unit, p, device)

    @torch.no_grad()
    def calibrate(self, x: torch.Tensor):
        xf = x.float().reshape(-1, x.shape[-1])
        W = self.lin.weight.float()
        T, K = xf.shape
        J, U = K // self.U, self.U
        yn = (xf @ W.t()).square().sum(1).clamp_min(1e-30)
        xb, wb = xf.view(T, J, U), W.view(-1, J, U)
        chunk = max(1, int(1.5e9 // (4 * T * W.shape[0])))
        errs = []
        for p in self.patterns():
            m = self.pattern(p, xf.device)
            if p == "all" and self.per_unit == 1:          # identical to position 0
                errs.append(errs[1].clone())
                continue
            xq = FM.fake(sel_fwht(xf, m, self.rb), self.a_spec).view(T, J, U)
            wq = FM.fake_weight(sel_fwht(W, m, self.rb), self.w_spec).view(-1, J, U)
            e = torch.zeros(J, device=xf.device, dtype=torch.float64)
            for j0 in range(0, J, chunk):
                s = slice(j0, j0 + chunk)
                d = torch.einsum("tjk,njk->jtn", xq[:, s], wq[:, s]) - torch.einsum("tjk,njk->jtn", xb[:, s], wb[:, s])
                e[s] += (d.square().sum(2) / yn[None]).sum(1).double()
                del d
            errs.append(e)
        e = torch.stack(errs)
        self.acc = e if self.acc is None else self.acc + e

    @torch.no_grad()
    def gptq_pattern_errors(self, H: torch.Tensor):
        """GPTQ-aware proxy: per unit j and pattern p, tr(dW_j Hr_jj dW_j^T) with dW the GPTQ error of the rotated
        weight (pattern p) and Hr = R H R; cross-unit terms of Hr are dropped.  Stored in self.acc like calibrate()."""
        W = self.lin.weight.float()
        N, K = W.shape
        J, U = K // self.U, self.U
        Hd = H.to(W.device).float()
        errs = []
        for p in self.patterns():
            m = self.pattern(p, W.device)
            if p == "all" and self.per_unit == 1:
                errs.append(errs[1].clone())
                continue
            Wr = sel_fwht(W, m, self.rb)
            Hr = sel_fwht(sel_fwht(Hd, m, self.rb).t().contiguous(), m, self.rb)
            c, s = gptq_quantize(Wr, Hr, self.w_spec)
            dW = (FM.dequant(c, s) - Wr).view(N, J, U)
            Hb = torch.stack([Hr[j * U:(j + 1) * U, j * U:(j + 1) * U] for j in range(J)])
            errs.append(torch.einsum("nju,juv,njv->j", dW, Hb, dW).double())
            del Wr, Hr, c, s, dW, Hb
        self.acc = torch.stack(errs)

    def forward(self, x):
        if STATE["calib"] and self.calib_on:
            self.calibrate(x)
        if STATE["calib"] and self.track_amax:
            xa = x.detach().float().reshape(-1, x.shape[-1])
            a = xa.abs().amax(0)
            self.amax = a if self.amax is None else torch.maximum(self.amax, a)
            e2 = xa.square().sum(0)
            self.e2 = e2 if self.e2 is None else self.e2 + e2
            self.e2_n += xa.shape[0]
        if STATE["hess"] and self.hess_key is not None:
            xf = x.float().reshape(-1, x.shape[-1])
            h = (xf.t() @ xf).cpu()
            HESS[self.hess_key] = HESS[self.hess_key] + h if self.hess_key in HESS else h
        if not STATE["quant"]:
            return self.lin(x)
        shp = x.shape
        xq = FM.fake(sel_fwht(x.float().reshape(-1, shp[-1]), self.mask, self.rb), self.a_spec)
        y = xq @ FM.dequant(self.w_codes, self.w_scales).t()
        if self.lin.bias is not None:
            y = y + self.lin.bias.float()
        return y.reshape(*shp[:-1], -1).to(x.dtype)


def protected_vectors(m: "SelLinear", tau: float) -> torch.Tensor:
    """vectors holding a massive-activation input channel (max |x| > tau * median over channels): never rotated.
    Rotating such a vector smears e.g. Llama-3-8B's super weight (layers.1 down_proj [788, 2427]) over rb channels."""
    hot = m.amax > tau * m.amax.median()
    return hot.view(-1, m.rb).any(1)


@torch.no_grad()
def rule_mask(m: "SelLinear") -> torch.Tensor:
    """closed-form flags (ablation, no trial quantization). Integer weights: rotate a sub-vector iff the rotation lowers
    its crest factor averaged over output rows (data-free). Floating-point weights: rotate iff the joint energy
    concentration dFP = r * sum_i E[x_i^2] c_i / (sum_i E[x_i^2] * sum_i c_i) exceeds 1 (rotation drives it to 1)."""
    from .hadamard import fwht
    W = m.lin.weight.float()
    Wv = W.view(W.shape[0], -1, m.rb)
    if FM.parse(m.w_spec)[0].startswith("int"):
        crest = lambda t: (t.abs().amax(-1) / t.square().mean(-1).sqrt().clamp_min(1e-30)).mean(0)  # noqa: E731
        return crest(fwht(W, m.rb).view_as(Wv)) < crest(Wv)
    e2 = (m.e2 / max(m.e2_n, 1)).view(-1, m.rb)
    c = W.square().sum(0).view(-1, m.rb)
    dfp = m.rb * (e2 * c).sum(-1) / (e2.sum(-1) * c.sum(-1)).clamp_min(1e-30)
    return dfp > 1


def greedy_masks(mods: dict[str, SelLinear], margin: float = 0.0) -> dict[str, torch.Tensor]:
    """per-vector flags from the calibration errors; mask-sharing projections sum their errors first.
    A vector is flagged iff its pattern lowers the unit error by more than margin * (unit error without rotation)."""
    grp: dict[str, torch.Tensor] = {}
    for n, m in mods.items():
        k = share_key(n)
        grp[k] = grp[k] + m.acc if k in grp else m.acc.clone()
    out = {}
    for n in mods:
        e = grp[share_key(n)]
        out[n] = ((e[0][None] - e[1:-1]) > margin * e[0][None]).t().reshape(-1)   # vector index = unit * per_unit + p
    return out


def layer_masks(mods: dict[str, SelLinear]) -> dict[str, torch.Tensor]:
    """coarser ablation: one decision per shared input (layer group): rotate every vector iff that lowers the summed
    calibration error of the group (same proxy, same rb as the per-vector flags)."""
    grp: dict[str, torch.Tensor] = {}
    for n, m in mods.items():
        k = share_key(n)
        grp[k] = grp[k] + m.acc if k in grp else m.acc.clone()
    return {n: torch.full((m.nvec,), bool(grp[share_key(n)][-1].sum() < grp[share_key(n)][0].sum()),
                          dtype=torch.bool, device=m.acc.device) for n, m in mods.items()}
