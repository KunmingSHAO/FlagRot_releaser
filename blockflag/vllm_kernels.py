"""Triton ports of the scalar math of vLLM's KV-cache store kernels, for bit-exact simulation on CUDA.

vLLM computes scales, zero points and codes inside Triton kernels, where an FP32 division a / b lowers to the
approximate div.full.f32 (<= 2 ulp) and a * b + c may fuse into an FMA; a torch re-implementation therefore lands on
the other side of a rounding tie for a small fraction of elements (e.g. -min / scale == 7.5 exactly when
min == -max).  These kernels repeat vLLM's expressions verbatim, so on a GPU the codes match vLLM bit for bit; on CPU
(no Triton) formats.py falls back to IEEE torch arithmetic.

  int4ph_quant: vllm/v1/attention/ops/int4_per_token_head.py::_reshape_cache_int4_kernel (+ read-side dequant)
  tq_value_quant: vllm/v1/attention/ops/triton_turboquant_store.py::_store_quantized_value (+ decode dequant)
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:                                    # pragma: no cover
    triton = None


if triton is not None:
    @triton.jit
    def _int4ph_kernel(y_ptr, out_ptr, D: tl.constexpr, BLOCK_D: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK_D)
        m = offs < D
        y = tl.load(y_ptr + row * D + offs, mask=m, other=0.0)
        k_min = tl.min(tl.where(m, y, float("inf")))
        k_max = tl.max(tl.where(m, y, float("-inf")))
        k_scale = tl.maximum((k_max - k_min) / 15.0, 1e-6)
        k_zp_f = tl.clamp(
            tl.where(-k_min / k_scale >= 0, (-k_min / k_scale + 0.5).to(tl.int32),
                     (-k_min / k_scale - 0.5).to(tl.int32)).to(tl.float32), 0.0, 15.0)
        inv_k = 1.0 / k_scale
        s = y * inv_k + k_zp_f
        q = tl.clamp(tl.where(s >= 0, (s + 0.5).to(tl.int32), (s - 0.5).to(tl.int32)).to(tl.float32), 0.0, 15.0)
        k_zp_int = k_zp_f.to(tl.int32)
        packed = ((k_scale.to(tl.int32, bitcast=True) & -16) | (k_zp_int & 0xF)).to(tl.float32, bitcast=True)
        bits = packed.to(tl.int32, bitcast=True)                       # read side: split zero point and scale
        zp = (bits & 0xF).to(tl.float32)
        sc = (bits & -16).to(tl.float32, bitcast=True)
        tl.store(out_ptr + row * D + offs, (q - zp) * sc, mask=m)

    @triton.jit
    def _tq_value_kernel(v_ptr, out_ptr, D: tl.constexpr, BLOCK_D: tl.constexpr, LEVELS: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK_D)
        m = offs < D
        val_vec = tl.load(v_ptr + row * D + offs, mask=m, other=0.0)
        val_min = tl.min(tl.where(m, val_vec, float("inf")), axis=0)
        val_max = tl.max(tl.where(m, val_vec, -float("inf")), axis=0)
        v_scale = (val_max - val_min) / LEVELS
        v_scale = tl.where(v_scale > 1e-8, v_scale, 1e-8)
        q = tl.minimum(tl.maximum(((val_vec - val_min) / v_scale + 0.5).to(tl.int32), 0), LEVELS)
        sc = v_scale.to(tl.float16).to(tl.float32)
        zr = val_min.to(tl.float16).to(tl.float32)
        tl.store(out_ptr + row * D + offs, q.to(tl.float32) * sc + zr, mask=m)


def available(t: torch.Tensor) -> bool:
    return triton is not None and t.is_cuda


def _launch(kernel, x: torch.Tensor, **kw) -> torch.Tensor:
    D = x.shape[-1]
    x2 = x.float().contiguous().reshape(-1, D)
    out = torch.empty_like(x2)
    if x2.shape[0]:
        kernel[(x2.shape[0],)](x2, out, D=D, BLOCK_D=triton.next_power_of_2(D), **kw)
    return out.reshape(x.shape)


def int4ph_quant(y: torch.Tensor) -> torch.Tensor:
    """dequantized cache entry (q - zp) * (scale & -16) of the BF16-rounded rotated vectors y [..., D]."""
    return _launch(_int4ph_kernel, y)


def tq_value_quant(v: torch.Tensor, bits: int) -> torch.Tensor:
    return _launch(_tq_value_kernel, v, LEVELS=2 ** bits - 1)
