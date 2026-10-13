"""JIT loader + Python API for the selective-FWHT CUDA kernels (H20 / sm_90a).

    q, s = selrot_quant_fp8(x, mask, rb, g)                       x bf16 [T,K] -> fp8 [T,K], fp32 [K/g, T]
    q, s = rmsnorm_selrot_quant_fp8(x, w, eps, mask, rb, g[, perm]) fused RMSNorm -> selective FWHT -> FP8 (perm: K gather)
    q, s = swiglu_selrot_quant_fp8(mask, rb, g, gate_up=..)       fused silu(gate)*up -> selective FWHT -> FP8
    q, s = kv_selrot_quant(kv, mask, rb, fmt, global_scale[, out])  kv [T,H,D]; fmt in {'fp8','nvfp4','mxfp4','int4'}
mask: uint8 [K/rb] (kv: [H, D/rb]); element k is rotated by H_rb/sqrt(rb) iff mask[k // rb].
FP8 MN-major scales s [K/g, T] are a view of a [K/g, align(T,4)] buffer: s.t() is DeepGEMM's TMA-aligned SFA (zero copy).
"""
from __future__ import annotations

import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        os.environ["PATH"] = os.path.join(_HERE, ".deps", "bin") + os.pathsep + os.environ.get("PATH", "")
        os.environ.setdefault("CUDA_HOME", os.environ.get("FLAGROT_CUDA_HOME", "/usr/local/cuda-12.9"))
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
        os.environ.setdefault("MAX_JOBS", "8")
        fast = os.environ.get("SELROT_FAST_DIV", "0") == "1"  # q = y * (1/s): ~2 FMA-pipe ops/elem cheaper, not bit-exact
        bdir = os.path.join(_HERE, ".build_fastdiv" if fast else ".build")
        os.makedirs(bdir, exist_ok=True)
        _EXT = load(
            name="selrot_cuda_ext_fastdiv" if fast else "selrot_cuda_ext",
            sources=[os.path.join(_HERE, "csrc", "selrot_bind.cpp"), os.path.join(_HERE, "csrc", "selrot_kernels.cu")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-gencode", "arch=compute_90a,code=sm_90a", "-std=c++17",
                               "--expt-relaxed-constexpr", "-lineinfo"] + (["-DSELROT_FAST_DIV"] if fast else []),
            build_directory=bdir,
            verbose=bool(int(os.environ.get("SELROT_VERBOSE", "0"))),
        )
    return _EXT


def _mask_u8(mask: torch.Tensor, n: int, device) -> torch.Tensor:
    m = mask.reshape(-1)
    if m.dtype != torch.uint8:
        m = (m != 0).to(torch.uint8)
    m = m.to(device).contiguous()
    assert m.numel() == n, (m.numel(), n)
    return m


def selrot_quant_fp8(x, mask, rb=64, g=128):
    x = x.contiguous()
    return tuple(ext().selrot_quant_fp8(x, _mask_u8(mask, x.shape[1] // rb, x.device), rb, g))


def rmsnorm_selrot_quant_fp8(x, rms_weight, eps, mask, rb=64, g=128, hf_rounding=True, perm=None):
    """perm (optional): int32 [K/8] chunk table, output chunk c (8 elements) = input chunk perm[c] (gather on load).
    rms_weight and mask are then given in OUTPUT (permuted) order; see perm.py."""
    if x.stride(-1) != 1:
        x = x.contiguous()
    if perm is not None:
        perm = perm.to(device=x.device, dtype=torch.int32).contiguous()
    return tuple(ext().rmsnorm_selrot_quant_fp8(x, rms_weight.contiguous(), float(eps),
                                                _mask_u8(mask, x.shape[1] // rb, x.device), rb, g, hf_rounding,
                                                perm))


def swiglu_selrot_quant_fp8(mask, rb=64, g=128, gate_up=None, gate=None, up=None, hf_rounding=True):
    """gate_up [T, 2I] (gate = first I columns, up = last I, vLLM MergedColumnParallel layout) or gate/up [T, I]."""
    if gate_up is not None:
        I = gate_up.shape[1] // 2
        gate, up = gate_up[:, :I], gate_up[:, I:]
    return tuple(ext().swiglu_selrot_quant_fp8(gate, up, _mask_u8(mask, gate.shape[1] // rb, gate.device),
                                               rb, g, hf_rounding))


_FMT = {"fp8": 0, "nvfp4": 2, "mxfp4": 3, "int4": 4}


def kv_selrot_quant(kv, mask, rb=64, fmt="fp8", global_scale=None, out=None):
    """kv [T, H, D] bf16 (rows [H, D] contiguous; the row stride may be arbitrary, e.g. the k,v slice of qkv).
         fp8   -> (q fp8 [T,H,D], s fp32 [T,H])
         nvfp4 -> (q uint8 [T,H,D/2] (e2m1 x2, low nibble = even element), s e4m3 [T,H,D/16]);
                  dequant = e2m1 * s / global_scale
         mxfp4 -> (q uint8 [T,H,D/2], s uint8 e8m0 [T,H,D/32]); dequant = e2m1 * 2^(s-127)
         int4  -> (q uint8 [T,H,D/2] (code = q + 8, q in [-7, 7]), s fp16 [T,H,D/32]); dequant = (code - 8) * s
    out = (q_dst, s_dst): write in place into views with contiguous rows and any row stride (e.g. cache[:, pos]);
    strided input / output needs H * D % 2048 == 0."""
    if kv.stride(-1) != 1 or kv.stride(-2) != kv.shape[-1]:
        kv = kv.contiguous()
    T, H, D = kv.shape
    if global_scale is not None and not torch.is_tensor(global_scale):
        global_scale = torch.tensor([float(global_scale)], device=kv.device, dtype=torch.float32)
    oq, os_ = (None, None) if out is None else out
    return tuple(ext().kv_selrot_quant(kv, _mask_u8(mask, H * D // rb, kv.device), rb, _FMT[fmt], global_scale,
                                       oq, os_))


def rmsnorm_bf16(x, w, eps):
    return ext().rmsnorm_bf16(x.contiguous(), w.contiguous(), float(eps))


def silu_mul_bf16(gate_up=None, gate=None, up=None):
    if gate_up is not None:
        I = gate_up.shape[1] // 2
        gate, up = gate_up[:, :I], gate_up[:, I:]
    return ext().silu_mul_bf16(gate, up)


def cast_bf16_fp8(x):
    """plain saturating bf16 -> fp8 e4m3 cast (bandwidth baseline, no scaling)."""
    return ext().cast_bf16_fp8(x.contiguous())
