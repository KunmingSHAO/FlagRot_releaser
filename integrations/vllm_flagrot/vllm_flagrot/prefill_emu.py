"""Phase 2f, Phase 1: numerical emulation of INT4-direct prefill attention designs (accuracy only, PyTorch fp32).

A kernel that reads the INT4 pages directly computes q.k in the STORED basis of each key (the FlagRot transform the
writer applied before quantization), so the emulation does too:
  static (rtn_a / static_a)  one basis per KV head:  q_s = B (H q / 8 if the head is whole-head rotated else q),
                              B = per-32-group H32 on the head's block-flagged groups (kernels' q prologue)
  konlya (key-only 2x2)      per key: H128 (head flag F) | per group H32 (group flag) | I; q is formed in all three
                              bases and every (key, group) uses the one its flags select (= the decode kernel k_cbk)
K is exact: the integer codes c - z (|c - z| <= 15, exact in E4M3) times the per-(token, 32-group) scale, in fp32.

variants (prefill_fp8 = "emu_<v>"):
  exact  q, K, V, P unrounded (fp32): must reproduce the BF16 FA path (pipeline check)
  q      only q as the FP8 hi + lo pair below (K, V, P exact): the cost of the q representation alone
  q1     q a single FP8 (hi only) in the stored basis, K, V, P exact (design A without the Hadamard)
  qtop1 / qtop2  q hi everywhere + lo on the 1 / 2 highest-energy 32-groups per KV head (fewer lo MMAs)
  ac1    ac with the lo part on the top-1 group only
  b      QK in BF16 on K dequantized + inverse-transformed in fp32 and rounded to BF16 (original basis, = the BF16
         path's K; q exact), V / P as ac: the "inverse-transform the K tile, then round" kernel
  ac     A+C: q in the stored basis as an FP8 hi + lo pair (per (token, head) power-of-two scale s: hi = e4m3(s q),
         lo = e4m3(s q - hi), q~ = (hi + lo) / s), K exact, V (stored-basis dequantized values) rounded to E4M3
         (scale 1), P = 2^8 exp(x - m) rounded to E4M3 (m = the final row max: the kernel's online max is <= it;
         the 2^8 = FA3 FP8's Max_offset 8, which keeps P out of the subnormal range)
  acv    as ac but V exact as well: V codes c - z exact in E4M3, the V scale folded into P' = P s_{t,g} / sref_g
         rounded to E4M3 (sref_g = max of s over each 64-key block and 32-channel group), o_g = sum P' sref (c - z)
The FP8 MMAs themselves (fp32-ish accumulation of exact products) are not emulated: Phase 2e measured FA3's FP8
compute at 8e-5 relative output MSE in total, including P's rounding, which is emulated here.
"""
from __future__ import annotations

import math

import torch

from . import kernels as K

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
D, RB, NG = 128, 32, 4
VARIANTS = ("exact", "q", "q1", "qtop1", "qtop2", "ac", "acv", "ac1", "b")
NLO = {"q1": 0, "qtop1": 1, "qtop2": 2, "ac1": 1}
PSCALE = 256.0      # P is rounded as 2^8 exp(x - m) (FA3 FP8's Max_offset = 8: keeps P out of E4M3's subnormals)


def fp8r(x: torch.Tensor) -> torch.Tensor:
    """round to E4M3 (RN, saturating) and back to fp32."""
    return x.clamp(-FP8_MAX, FP8_MAX).to(FP8).float()


def hilo(x: torch.Tensor, nlo: int = 4) -> torch.Tensor:
    """x fp32 [rows, 128] -> (hi + lo) / s, s = 2^floor(log2(448 / max|x_row|)) per row (exact in fp32).
    nlo < 4: the lo part only on the nlo 32-channel groups with the largest energy over all rows (one static choice
    per KV head, as a calibrated per-head group mask would be); nlo = 0: hi only (a single FP8 q)."""
    am = x.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    s = torch.exp2(torch.floor(torch.log2(FP8_MAX / am)))
    y = x * s
    hi = fp8r(y)
    lo = fp8r(y - hi)
    if nlo < 4:
        e = (x.view(-1, NG, RB) ** 2).sum((0, 2))
        keep = torch.zeros(NG, dtype=torch.bool, device=x.device)
        if nlo > 0:
            keep[e.topk(nlo).indices] = True
        lo = (lo.view(-1, NG, RB) * keep[None, :, None]).view_as(lo)
    return (hi + lo) / s


def _h32(dev):
    return K.hadamard(RB, dev, torch.float32)


def _h128(dev):
    return K.hadamard(D, dev, torch.float32)


def gather_codes(pool, block_table_row, n, asym):
    """first n tokens of a request: codes (c - z) fp32 [n, Hkv, 2, 128] (stored basis), scale magnitudes fp32
    [n, Hkv, 2, 4], raw scale words int16 [n, Hkv, 2, 4]."""
    P = 16
    npg = (n + P - 1) // P
    chunks = pool.index_select(0, block_table_row[:npg].long())
    codes, mag, sbits = K.parse_int4_pages(chunks, P, asym=asym)        # [Np, H, P, 2, ...]
    f = lambda t: t.transpose(1, 2).reshape(npg * P, *t.shape[1:2], *t.shape[3:])[:n]  # noqa: E731
    return f(codes).float(), f(mag), f(sbits)


@torch.no_grad()
def emu_attention(impl, st, q, pool, bt_row, nk, p0, shift, variant):
    """q bf16 [nq, Hq, 128] (original basis) at positions p0 .. p0 + nq - 1; keys 0 .. nk - 1 from the pages; key j is
    visible to query position i iff j <= i - shift.  Returns (o bf16 [nq, Hq, 128] original basis, lse fp32 [Hq, nq])."""
    assert variant in VARIANTS, variant
    dev = q.device
    nq, Hq, _ = q.shape
    Hkv = impl.num_kv_heads
    G = Hq // Hkv
    asym = getattr(impl, "v2fmt", None) == "int4a" or impl.kind == "konlya"
    codes, mag, sbits = gather_codes(pool, bt_row, nk, asym)
    kc, vc = codes[:, :, 0], codes[:, :, 1]                               # [nk, Hkv, 128]
    ks, vs = mag[:, :, 0], mag[:, :, 1]                                   # [nk, Hkv, 4]
    kx = (kc.view(nk, Hkv, NG, RB) * ks[..., None]).view(nk, Hkv, D)      # exact K, stored basis
    qf = q.float().view(nq, Hkv, G, D)
    if variant == "b":                 # K dequantized + inverse-transformed in fp32, rounded to BF16 (= BF16 path's K)
        npg = (nk + 15) // 16
        chunks = pool.index_select(0, bt_row[:npg].long())
        if impl.static:
            kb, _ = K.dequant_v2_pages(chunks, 16, torch.float32, fmt=impl.v2fmt)
            kb, _ = st.unrotate_cached(kb.reshape(-1, Hkv, D), kb.reshape(-1, Hkv, D))
        else:
            kb, _ = K.dequant_cb_pages(chunks, 16, impl.kind, torch.float32)
        kb = kb.reshape(-1, Hkv, D)[:nk].to(torch.bfloat16).float()
    h32, h128 = _h32(dev), _h128(dev)
    nlo = NLO.get(variant, 4)
    rnd = (lambda x: x) if variant == "exact" else (lambda x: hilo(x, nlo))
    pos = torch.arange(p0, p0 + nq, device=dev)
    allowed = torch.arange(nk, device=dev)[None, :] <= (pos[:, None] - shift)       # [nq, nk]
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    o = torch.empty(nq, Hkv, G, D, dtype=torch.float32, device=dev)
    lse = torch.empty(Hkv, G, nq, dtype=torch.float32, device=dev)
    try:
        if impl.static:
            kfull, kmask = st.kfull.bool(), st.kmask.bool()               # [Hkv], [Hkv, 4]
            vfull, vmask = st.vfull.bool(), st.vmask.bool()
        else:
            fk = ((sbits[:, :, 1, 0].to(torch.int32) >> 15) & 1).bool()   # [nk, Hkv] head flag (V group-0 word)
            lk = ((sbits[:, :, 0, :].to(torch.int32) >> 15) & 1).bool()   # [nk, Hkv, 4] group flags (K words)
        for h in range(Hkv):
            qh = qf[:, h].reshape(nq * G, D)                              # rows (token, head)
            kh = kx[:, h]                                                 # [nk, 128]
            if variant == "b":
                S = qh @ kb[:, h].t()
            elif impl.static:
                qs = qh @ (_hpm(dev) / 8) if bool(kfull[h]) else qh
                qs = qs.view(-1, NG, RB)
                qs = torch.where(kmask[h][None, :, None], qs @ h32, qs).view(-1, D)
                S = rnd(qs) @ kh.t()                                      # [nq G, nk]
            else:
                qI = rnd(qh)
                qH = rnd((qh.view(-1, NG, RB) @ h32).view(-1, D))
                qF = rnd(qh @ h128)
                S = torch.where(fk[None, :, h], qF @ kh.t(), torch.zeros((), device=dev))
                for g in range(NG):
                    sl = slice(RB * g, RB * (g + 1))
                    sg = torch.where(lk[None, :, h, g], qH[:, sl] @ kh[:, sl].t(), qI[:, sl] @ kh[:, sl].t())
                    S += torch.where(fk[None, :, h], torch.zeros((), device=dev), sg)
            S = S.view(nq, G, nk).transpose(0, 1) * impl.scale           # [G, nq, nk]
            S = S.masked_fill(~allowed[None], -math.inf)
            m = S.amax(-1, keepdim=True)
            Pm = torch.exp(S - m)
            l = Pm.sum(-1, keepdim=True)
            if variant in ("exact", "q", "q1", "qtop1", "qtop2"):
                oh = Pm @ (vc[:, h].view(nk, NG, RB) * vs[:, h, :, None]).view(nk, D)
            elif variant in ("ac", "ac1", "b"):
                v8 = fp8r((vc[:, h].view(nk, NG, RB) * vs[:, h, :, None]).view(nk, D))
                oh = (fp8r(Pm * PSCALE) @ v8) / PSCALE
            else:                                                         # acv
                nb = (nk + 63) // 64
                sv = torch.zeros(nb * 64, NG, device=dev)
                sv[:nk] = vs[:, h]
                sref = sv.view(nb, 64, NG).amax(1).clamp(min=1e-30)      # [nb, 4]
                srefk = sref.repeat_interleave(64, 0)[:nk]                # [nk, 4]
                w = vs[:, h] / srefk                                      # <= 1
                oh = torch.empty(G, nq, D, device=dev)
                for g in range(NG):
                    sl = slice(RB * g, RB * (g + 1))
                    oh[..., sl] = (fp8r(Pm * w[None, None, :, g] * PSCALE) @ (vc[:, h, sl] * srefk[:, g, None])) / PSCALE
            oh = oh / l                                                   # [G, nq, D] stored V basis
            o[:, h] = oh.transpose(0, 1)
            lse[h] = (m + torch.log(l)).squeeze(-1)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    o = o.view(nq, Hq, D)
    if impl.static:                                                       # V stored basis -> original
        ob = o.view(nq, Hkv, G, NG, RB)
        o = torch.where(vmask[None, :, None, :, None], ob @ h32, ob).view(nq, Hkv, G, D)
        o = torch.where(vfull[None, :, None, None], o @ (_hpm(dev) / 8), o).view(nq, Hq, D)
    return o.to(torch.bfloat16), lse.view(Hq, nq)


def _hpm(dev):
    return K.hadamard(D, dev, torch.float32) * math.sqrt(D)
