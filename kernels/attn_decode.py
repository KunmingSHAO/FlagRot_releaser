"""Quantized-KV decode attention (split-K, GQA 4, D = 128) for H20: JIT loader, KV-cache container, Python API and
PyTorch references.  Kernel: csrc/attn_decode.cu (design notes at the top of that file).

    cache = KVCache(fmt, B, Lmax, Hkv=8)                       fmt in FMTS
    cache.write(kv, pos, mask, rb, gs)                         kv [B, 2*Hkv, D] (k heads then v heads) at position pos
    cache.fill(kv_all, mask, rb, gs)                           kv_all [B, L, 2*Hkv, D]
    o = decode(q, cache, L, kmask=, vmask=, rb=, qrot="fused"|"pre"|None, vrot=False)   q [B, 32, 128] -> [B, 4096]
Cache layout = exactly what kv_selrot_quant writes, token-major per sequence: codes [B, Lmax, 2*Hkv, row bytes],
scales [B, Lmax, 2*Hkv, scale bytes]; K = heads 0..Hkv-1, V = heads Hkv..2Hkv-1.
FlagRot: K/V were rotated at write time with flags kv_mask[2*Hkv, D/rb] (k rows first).  qrot rotates q with the K
flags (fused in the kernel prologue, or a separate tiny kernel 'pre'); vrot rotates the output back with the V flags
(otherwise the V rotation is folded into W_o offline = the deployment variant: output in the rotated V basis).
"""
from __future__ import annotations

import math
import os

import torch

import ref as R
import selrot_cuda as S

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT = None
FMTS = ("bf16", "fp8", "nvfp4", "mxfp4", "int4")
FMT_ID = {f: i for i, f in enumerate(FMTS)}
ROW = {"bf16": 256, "fp8": 128, "nvfp4": 64, "mxfp4": 64, "int4": 64}  # bytes per (token, head) row
SCB = {"bf16": 0, "fp8": 4, "nvfp4": 8, "mxfp4": 4, "int4": 8}         # scale bytes per (token, head)
SDT = {"fp8": torch.float32, "nvfp4": torch.float8_e4m3fn, "mxfp4": torch.uint8, "int4": torch.float16}
VARIANTS = {0: (1, 3), 1: (1, 4), 2: (2, 3), 3: (2, 4), 4: (1, 6), 5: (2, 6), 6: (2, 2), 7: (4, 2),
            8: (4, 2), 9: (4, 3), 10: (4, 4), 11: (4, 6),   # 8..11: warpgroup (wgmma QK and PV) kernel
            12: (4, 2), 13: (4, 3), 14: (4, 4), 15: (4, 6)}  # 12..15: hybrid (wgmma QK, mma.sync PV)
WG_VARIANTS = (8, 9, 10, 11, 12, 13, 14, 15)  # 64-token CTA blocks  # (NT: 16-token tiles / warp step, stages)
DEFAULT_VARIANT = {"bf16": 6, "fp8": 2, "nvfp4": 2, "mxfp4": 2, "int4": 2}
NUM_SMS = None
LOG2E = 1.4426950408889634


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        os.environ["PATH"] = os.path.join(_HERE, ".deps", "bin") + os.pathsep + os.environ.get("PATH", "")
        os.environ.setdefault("CUDA_HOME", os.environ.get("FLAGROT_CUDA_HOME", "/usr/local/cuda-12.9"))
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
        os.environ.setdefault("MAX_JOBS", "8")
        bdir = os.path.join(_HERE, ".build_attn")
        os.makedirs(bdir, exist_ok=True)
        _EXT = load(
            name="attn_decode_ext",
            sources=[os.path.join(_HERE, "csrc", "attn_bind.cpp"), os.path.join(_HERE, "csrc", "attn_decode.cu")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-gencode", "arch=compute_90a,code=sm_90a", "-std=c++17",
                               "--expt-relaxed-constexpr", "-lineinfo"],
            build_directory=bdir,
            verbose=bool(int(os.environ.get("SELROT_VERBOSE", "0"))),
        )
    return _EXT


def kv_bytes_per_token(fmt, Hkv=8):
    """K + V bytes (codes + scales) per token per layer."""
    return 2 * Hkv * (ROW[fmt] + SCB[fmt])


class KVCache:
    def __init__(self, fmt, B, Lmax, Hkv=8, device="cuda"):
        self.fmt, self.B, self.Lmax, self.Hkv = fmt, B, Lmax, Hkv
        H2 = 2 * Hkv
        if fmt == "bf16":
            self.codes = torch.empty(B, Lmax, H2, 128, device=device, dtype=torch.bfloat16)
            self.c8 = self.codes.view(torch.uint8)
            self.scales = self.s8 = None
        else:
            self.c8 = torch.empty(B, Lmax, H2, ROW[fmt], device=device, dtype=torch.uint8)
            self.codes = self.c8.view(torch.float8_e4m3fn) if fmt == "fp8" else self.c8
            self.s8 = torch.empty(B, Lmax, H2, SCB[fmt], device=device, dtype=torch.uint8)
            st = self.s8.view(SDT[fmt])
            self.scales = st.squeeze(-1) if fmt == "fp8" else st  # the dtype / shape kv_selrot_quant writes
        self.gs = torch.tensor([448.0 * 6.0 / 8.0], device=device)  # nvfp4 global scale (static calibration)

    def nbytes_per_token(self):
        return kv_bytes_per_token(self.fmt, self.Hkv)

    def views(self):
        """(kc, vc, ks, vs) uint8 views [B, Lmax, Hkv, bytes] for the kernel."""
        H = self.Hkv
        kc, vc = self.c8[:, :, :H], self.c8[:, :, H:]
        if self.s8 is None:
            return kc, vc, None, None
        return kc, vc, self.s8[:, :, :H], self.s8[:, :, H:]

    def write(self, kv, pos, mask, rb):
        """kv [B, 2*Hkv, D] bf16 (rows may be strided, e.g. the k,v slice of the qkv GEMM output) -> position pos."""
        if self.fmt == "bf16":
            self.codes[:, pos].copy_(kv)
            return
        S.kv_selrot_quant(kv, mask, rb, self.fmt, self.gs if self.fmt == "nvfp4" else None,
                          out=(self.codes[:, pos], self.scales[:, pos]))

    def fill(self, kv_all, mask, rb, b=None, start=0):
        """kv_all [B, T, 2*Hkv, D] (or [T, 2*Hkv, D] for sequence b) -> positions start..start+T-1."""
        bs = range(self.B) if b is None else [b]
        for i, bb in enumerate(bs):
            x = kv_all[i] if b is None else kv_all
            T = x.shape[0]
            if self.fmt == "bf16":
                self.codes[bb, start:start + T].copy_(x)
            else:
                S.kv_selrot_quant(x, mask, rb, self.fmt, self.gs if self.fmt == "nvfp4" else None,
                                  out=(self.codes[bb, start:start + T], self.scales[bb, start:start + T]))

    def dequant(self, L):
        """fp32 [B, L, 2*Hkv, D] (rotated basis, as stored)."""
        f = self.fmt
        if f == "bf16":
            return self.codes[:, :L].float()
        q, s = self.codes[:, :L], self.scales[:, :L]
        if f == "fp8":
            return q.float() * s.unsqueeze(-1)
        if f == "nvfp4":
            return R.nvfp4_dequant(q, s, self.gs)
        if f == "mxfp4":
            return R.mxfp4_dequant(q, s)
        return R.int4_dequant(q, s)


class Workspace:
    """split-K partial buffers (reused across calls; the counters reset themselves)."""

    def __init__(self, B, Hkv, max_splits, device="cuda"):
        self.opart = torch.empty(B * Hkv * max_splits * 4 * 128, device=device, dtype=torch.float32)
        self.mlpart = torch.empty(B * Hkv * max_splits * 4 * 2, device=device, dtype=torch.float32)
        self.counter = torch.zeros(B * Hkv, device=device, dtype=torch.int32)
        self.max_splits = max_splits


def plan(B, L, fmt, Hkv=8, variant=None, qrot=False, waves=None, nsplit=None, c0=None, c1=None):
    """(variant, nsplit, chunk).  Split count from a wave-quantization cost model over the resident-CTA slots
    (occupancy x SMs): time ~ ceil(CTAs / slots) * (chunk + C0) + C1 * nsplit (C0: per-CTA fixed cost in
    token-equivalents, C1: merge cost per split); chunks are whole warp groups (64 * NT tokens)."""
    global NUM_SMS
    if NUM_SMS is None:
        NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    v = DEFAULT_VARIANT[fmt] if variant is None else variant
    nt = VARIANTS[v][0]
    gran = 64 if v in WG_VARIANTS else 16 * nt * 4  # whole 64-token blocks (WG) / whole warp groups (mma.sync)
    if nsplit is not None:
        chunk = math.ceil(L / nsplit)
        return v, math.ceil(L / chunk), chunk
    occ = max(1, ext().occupancy(FMT_ID[fmt], bool(qrot), v))
    slots = occ * NUM_SMS
    C0 = PLAN_C0 if c0 is None else c0
    C1 = PLAN_C1 if c1 is None else c1
    best = None
    for ns in range(1, min(512, math.ceil(L / gran)) + 1):
        chunk = math.ceil(L / ns / gran) * gran
        nse = math.ceil(L / chunk)
        if best is not None and nse == best[1]:
            continue
        cost = math.ceil(B * Hkv * nse / slots) * (chunk + C0) + (C1 * nse if nse > 1 else 0)
        if best is None or cost < best[0] - 1e-9:
            best = (cost, nse, chunk)
    return v, best[1], best[2]


PLAN_C0, PLAN_C1 = 128, 4
_WS = {}


def decode(q, cache: KVCache, L, kmask=None, vmask=None, rb=64, qrot=None, vrot=False, variant=None, nsplit=None,
           waves=None, out=None, ws=None, sm_scale=None):
    """q [B, 4*Hkv, 128] bf16 -> out [B, 4*Hkv*128] bf16.  qrot: None | 'fused' | 'pre'."""
    B, Hq, D = q.shape
    fmt = cache.fmt
    if qrot == "pre":
        q = ext().qrot(q, kmask, rb)
    fused = qrot == "fused"
    v, ns, chunk = plan(B, L, fmt, cache.Hkv, variant, fused, waves, nsplit)
    if ws is None:
        key = (B, cache.Hkv, q.device)
        ws = _WS.get(key)
        if ws is None or ws.max_splits < ns:
            ws = _WS[key] = Workspace(B, cache.Hkv, 512, q.device)
    if out is None:
        out = torch.empty(B, Hq * D, device=q.device, dtype=torch.bfloat16)
    kc, vc, ks, vs = cache.views()
    scale = (1.0 / math.sqrt(D) if sm_scale is None else sm_scale) * LOG2E
    ext().attn_decode(q, kc, vc, ks, vs, L, FMT_ID[fmt], ns, chunk, scale, cache.gs if fmt == "nvfp4" else None,
                      kmask, vmask, rb, fused, bool(vrot), v, ws.opart, ws.mlpart, ws.counter, out)
    return out


# ------------------------------------------------------------------------------------------------ references
def rotate_heads(x, mask, rb):
    """x [..., H, D] fp32; mask [H, D/rb]: FlagRot of every head with its own flags (ref.sel_fwht_rb per head)."""
    H, Dd = x.shape[-2], x.shape[-1]
    y = R.sel_fwht_rb(x.reshape(-1, H * Dd), mask.reshape(-1), rb)
    return y.reshape(x.shape)


def ref_attention(q, K, V, sm_scale=None):
    """fp32 GQA attention for one decode step: q [B, Hq, D], K/V [B, L, Hkv, D] -> [B, Hq, D] (float64 math)."""
    B, Hq, D = q.shape
    Hkv = K.shape[2]
    g = Hq // Hkv
    sc = 1.0 / math.sqrt(D) if sm_scale is None else sm_scale
    qd = q.double().reshape(B, Hkv, g, D)
    s = torch.einsum("bhgd,blhd->bhgl", qd, K.double()) * sc
    p = torch.softmax(s, dim=-1)
    o = torch.einsum("bhgl,blhd->bhgd", p, V.double())
    return o.reshape(B, Hq, D)
