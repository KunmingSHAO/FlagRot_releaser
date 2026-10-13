"""PyTorch references for the selective-FWHT kernels (per-rb-vector masks).

Numerics follow selrot_ppl.py / selrot_kl.py (sel_fwht + fp8_codes), with the mask on rb-vectors instead of g-groups:
  * fwht: Sylvester natural order, stages h = 1, 2, 4, ... (a+b, a-b) in fp32, then * (1/sqrt(rb));
  * FP8 E4M3: scale = clamp_min(absmax, 1e-12) / 448 per (token, g-group), q = (y / scale).to(float8_e4m3fn);
  * NVFP4:  block 16, sf = e4m3(gs * (amax * (1/6))), bs = float(sf) / gs, v = y / bs, e2m1 RNE (sat 6);
            dequant = e2m1 * float(sf) / gs.  gs (global scale) is a calibrated fp32 scalar, e.g. 448*6/amax(tensor).
  * MXFP4:  block 32, shared_exp = clamp(floor(log2(amax)) - 2, -127, 127) (amax == 0 -> -127), E8M0 = e + 127,
            v = y * 2^-e, e2m1 RNE (sat 6).  (OCP MX v1.0)
The e2m1 rounding here uses an independent threshold implementation (ties to even code).
"""
from __future__ import annotations

import math

import torch

FP8_MAX = 448.0
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def fwht(x: torch.Tensor, B: int) -> torch.Tensor:
    """block-diagonal orthonormal Sylvester-Hadamard along the last dim (fp32; copy of rot_common.fwht)."""
    shp = x.shape
    K = shp[-1]
    y = x.reshape(-1, K // B, B)
    h = 1
    while h < B:
        y = y.reshape(-1, K // B, B // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2)
        h *= 2
    return y.reshape(shp) * (1.0 / math.sqrt(B))


def sel_fwht_rb(x: torch.Tensor, mask: torch.Tensor, rb: int) -> torch.Tensor:
    """x fp32 [..., K]; mask [K/rb] (or any shape with K/rb elements); element k rotated iff mask[k // rb]."""
    m = mask.reshape(-1).bool().to(x.device)
    if not bool(m.any()):
        return x
    y = fwht(x, rb)
    return torch.where(m.repeat_interleave(rb), y, x)


def fp8_quant(y: torch.Tensor, g: int):
    """y fp32 [T, K] -> q fp8 [T, K], s fp32 [K/g, T] (MN-major)."""
    T, K = y.shape
    blk = y.reshape(T, K // g, g)
    scale = blk.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / FP8_MAX
    q = (blk / scale).to(torch.float8_e4m3fn)
    return q.reshape(T, K), scale.reshape(T, K // g).t().contiguous()


def fp8_dequant(q, s, g):
    T, K = q.shape
    return (q.float().reshape(T, K // g, g) * s.t().reshape(T, K // g, 1)).reshape(T, K)


def selrot_quant_fp8(x, mask, rb, g):
    return fp8_quant(sel_fwht_rb(x.float(), mask, rb), g)


def rmsnorm_hf(x, w, eps):
    """HF LlamaRMSNorm / Qwen3RMSNorm (bf16 in, bf16 out)."""
    h = x.float()
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + eps)
    return w * h.to(x.dtype)


def rmsnorm_fp32(x, w, eps):
    h = x.float()
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def rmsnorm_selrot_quant_fp8(x, w, eps, mask, rb, g, hf_rounding=True):
    y = rmsnorm_hf(x, w, eps).float() if hf_rounding else rmsnorm_fp32(x, w, eps)
    return fp8_quant(sel_fwht_rb(y, mask, rb), g)


def swiglu_hf(gate, up):
    return torch.nn.functional.silu(gate) * up


def swiglu_selrot_quant_fp8(gate, up, mask, rb, g, hf_rounding=True):
    y = swiglu_hf(gate, up).float() if hf_rounding else torch.nn.functional.silu(gate.float()) * up.float()
    return fp8_quant(sel_fwht_rb(y, mask, rb), g)


# ------------------------------------------------------------------------------------------------ FP4
def e2m1_code(v: torch.Tensor) -> torch.Tensor:
    """fp32 -> 4-bit e2m1 code (uint8), RNE with ties to even code, saturating at 6; sign bit = signbit(v)."""
    a = v.abs()
    code = ((a > 0.25).int() + (a >= 0.75).int() + (a > 1.25).int() + (a >= 1.75).int()
            + (a > 2.5).int() + (a >= 3.5).int() + (a > 5.0).int())
    return (code | (torch.signbit(v).int() << 3)).to(torch.uint8)


def e2m1_value(code: torch.Tensor) -> torch.Tensor:
    mag = E2M1_VALUES.to(code.device)[(code & 7).long()]
    return torch.where((code & 8) != 0, -mag, mag)


def pack_nibbles(code: torch.Tensor) -> torch.Tensor:
    c = code.reshape(*code.shape[:-1], -1, 2)
    return (c[..., 0] | (c[..., 1] << 4)).to(torch.uint8)


def unpack_nibbles(p: torch.Tensor) -> torch.Tensor:
    return torch.stack((p & 15, p >> 4), dim=-1).reshape(*p.shape[:-1], -1)


def nvfp4_quant(y: torch.Tensor, gs: torch.Tensor):
    """y fp32 [..., D] -> packed codes [..., D/2], e4m3 scales [..., D/16]."""
    shp = y.shape
    b = y.reshape(*shp[:-1], shp[-1] // 16, 16)
    amax = b.abs().amax(-1, keepdim=True)
    sf = (gs * (amax * (1.0 / 6.0))).to(torch.float8_e4m3fn)
    bs = sf.float() / gs
    v = torch.where(bs != 0, b / torch.where(bs != 0, bs, torch.ones_like(bs)), torch.zeros_like(b))
    code = e2m1_code(v).reshape(shp)
    return pack_nibbles(code), sf.reshape(*shp[:-1], shp[-1] // 16)


def nvfp4_dequant(qp, sf, gs):
    code = unpack_nibbles(qp)
    shp = code.shape
    v = e2m1_value(code).reshape(*shp[:-1], shp[-1] // 16, 16)
    return (v * (sf.float() / gs).unsqueeze(-1)).reshape(shp)


def mxfp4_quant(y: torch.Tensor):
    """y fp32 [..., D] -> packed codes [..., D/2], e8m0 scales (uint8) [..., D/32]."""
    shp = y.shape
    b = y.reshape(*shp[:-1], shp[-1] // 32, 32)
    amax = b.abs().amax(-1, keepdim=True)
    _, ex = torch.frexp(amax)                    # amax = m * 2^ex, m in [0.5, 1)  ->  floor(log2 amax) = ex - 1
    e = torch.where(amax > 0, ex.int() - 1 - 2, torch.full_like(ex.int(), -127)).clamp(-127, 127)
    mul = torch.ldexp(torch.ones_like(amax), -e)  # exact power of two
    code = e2m1_code(b * mul).reshape(shp)
    return pack_nibbles(code), (e + 127).to(torch.uint8).reshape(*shp[:-1], shp[-1] // 32)


def mxfp4_dequant(qp, e8):
    code = unpack_nibbles(qp)
    shp = code.shape
    v = e2m1_value(code).reshape(*shp[:-1], shp[-1] // 32, 32)
    return (v * torch.ldexp(torch.ones_like(v[..., :1]), e8.int().unsqueeze(-1) - 127)).reshape(shp)


def int4_quant(y: torch.Tensor):
    """symmetric INT4 per 32 (blockflag 'int4:32', scale stored as fp16): s = fp16(amax * (1/7)),
    q = clamp(round(y / s), -7, 7) (s == 0 -> 0); returns packed codes (q + 8) [..., D/2] and fp16 scales [..., D/32]."""
    shp = y.shape
    b = y.reshape(*shp[:-1], shp[-1] // 32, 32)
    s = (b.abs().amax(-1, keepdim=True) * (1.0 / 7.0)).half()
    sf = s.float()
    q = torch.where(sf != 0, torch.round(b / torch.where(sf != 0, sf, torch.ones_like(sf))), torch.zeros_like(b))
    code = (q.clamp(-7, 7) + 8).to(torch.uint8).reshape(shp)
    return pack_nibbles(code), s.reshape(*shp[:-1], shp[-1] // 32)


def int4_dequant(qp, s):
    code = unpack_nibbles(qp).float() - 8
    shp = code.shape
    return (code.reshape(*shp[:-1], shp[-1] // 32, 32) * s.float().unsqueeze(-1)).reshape(shp)


def kv_selrot_quant(kv, mask, rb, fmt, global_scale=None):
    """kv bf16 [T, H, D]; mask [H, D/rb]."""
    T, H, D = kv.shape
    y = sel_fwht_rb(kv.float().reshape(T, H * D), mask, rb).reshape(T, H, D)
    if fmt == "fp8":
        scale = y.abs().amax(-1, keepdim=True).clamp_min(1e-12) / FP8_MAX
        return (y / scale).to(torch.float8_e4m3fn), scale.squeeze(-1)
    if fmt == "nvfp4":
        return nvfp4_quant(y, global_scale)
    if fmt == "mxfp4":
        return mxfp4_quant(y)
    if fmt == "int4":
        return int4_quant(y)
    raise ValueError(fmt)


def nvfp4_global_scale(y: torch.Tensor) -> torch.Tensor:
    """calibration rule used in the tests: gs = 448 * 6 / amax(y) (y = the rotated tensor)."""
    return (448.0 * 6.0 / y.abs().amax().float().clamp_min(1e-12)).reshape(1)


def int4a_quant(y: torch.Tensor):
    """asymmetric INT4 of the H20 cache (attn_decode_v2 'int4a'), groups of 32 along the last dim:
    blockflag int4ah:32 = int4a:32 with the range widened to include 0 (identical whenever min <= 0 <= max):
    s = max((max' - min') / 15, 1e-12) (CUDA: multiply by fp32(1/15)), z = rne(-min' / s), q = clamp(rne(y / s) + z, 0, 15).
    Returns codes long [..., D] (0..15) and the 16-bit scale words long [..., D/32] (bits 3..0 = z, the word read as
    FP16 = the stored scale; blockflag.formats.scale_word)."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import blockflag.formats as FM
    q, z, s, _ = FM.quant_asym_dep(y.float(), 32, False)
    wd = FM.scale_word(s, z)
    return q.reshape(y.shape).long(), wd.squeeze(-1).long() & 0xFFFF
