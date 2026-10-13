"""Phase 2f: paged INT4 prefill attention that reads the FlagRot INT4 pages directly (csrc/int4_prefill.cu, sm_90a).

Design A+C: K exact (asymmetric codes c - z in E4M3, one fresh fp32 accumulator per 32-channel
group, K scale in fp32), q an E4M3 hi + lo pair in the key's stored basis (prep_q), P = 2^8 exp2(x - m) and V = s (c - z)
rounded to E4M3, wgmma e4m3 for QK and PV.  No gather / dequantized workspace.  ctypes front end of
build/libflagrot_prefill.so (csrc/build_prefill.sh, rebuilt automatically when older than its sources).
"""
from __future__ import annotations

import ctypes
import fcntl
import os
import subprocess
from pathlib import Path

import torch

_PLUGIN = Path(__file__).resolve().parent.parent
D, RB, NG, PAGE = 128, 32, 4, 16
FP8 = torch.float8_e4m3fn
LOG2E = 1.4426950408889634
_LIB = None


def _lib_path() -> Path:
    return Path(os.environ.get("FLAGROT_PREFILL_LIB", _PLUGIN / "build" / "libflagrot_prefill.so"))


def _sources():
    c = _PLUGIN / "csrc"
    return [c / "int4_prefill.cu", c / "int4_prefill_wgmma.h", c / "int4_prefill_softmax_pv.inc",
            c / "int4_prefill_epilogue.inc", c / "build_prefill.sh"]


def lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    path = _lib_path()
    stale = (not path.exists()) or (not os.environ.get("FLAGROT_NO_REBUILD")
                                    and any(s.stat().st_mtime > path.stat().st_mtime for s in _sources()))
    if stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path) + ".lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            subprocess.run(["bash", str(_PLUGIN / "csrc" / "build_prefill.sh"), str(path.parent)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            fcntl.flock(lk, fcntl.LOCK_UN)
    L = ctypes.CDLL(str(path))
    P, I, LL, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_float
    L.frp_prep_q.argtypes = [P, LL, I, I, P, P, F, P, P, P, P]
    L.frp_int4_prefill.argtypes = [P, P, P, LL, P, LL, P, I, I, I, I, I, P, P, LL, P, I, P]
    L.frp_int4_prefill_kbf.argtypes = [P, LL, F, P, P, I, I, P, LL, P, I, I, I, I, I, P, LL, P, I, P]
    L.frp_int4_prefill_kbf.restype = I
    L.frp_prep_q.restype = I
    L.frp_int4_prefill.restype = I
    L.frp_last_error.restype = ctypes.c_char_p
    _LIB = L
    return L


def _ptr(t):
    return None if t is None else t.data_ptr()


def _check(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what}: launch failed ({lib().frp_last_error().decode()})")


def prep_q(q: torch.Tensor, scale: float, kfull=None, kmask=None):
    """q bf16 [nq, Hq, 128] (original basis, head rows contiguous) -> (q_hi, q_lo E4M3 [nq, Hq, 128] in the stored
    basis, qfac fp32 [nq, Hq] = scale * log2(e) / sigma).  kfull uint8 [Hkv] (whole-head H128 / 8 on the KV head's
    q heads), kmask uint8 [Hkv, 4] (orthonormal H32 on the group): the static FlagRot key transforms."""
    nq, Hq, _ = q.shape
    assert q.dtype == torch.bfloat16 and q.stride(2) == 1 and q.stride(1) == D
    qh = torch.empty(nq, Hq, D, dtype=FP8, device=q.device)
    ql = torch.empty_like(qh)
    qf = torch.empty(nq, Hq, dtype=torch.float32, device=q.device)
    _check(lib().frp_prep_q(q.data_ptr(), q.stride(0), nq, Hq, _ptr(kfull), _ptr(kmask), scale * LOG2E,
                            qh.data_ptr(), ql.data_ptr(), qf.data_ptr(), torch.cuda.current_stream().cuda_stream),
           "frp_prep_q")
    return qh, ql, qf


def int4_prefill(qh, ql, qf, pool, bt_row, nk, p0, shift, lomask=None, out=None, stages=2):
    """paged INT4 attention of nq = qh.shape[0] queries at positions p0 .. p0 + nq - 1 over keys 0 .. nk - 1 of the
    pages bt_row (int32), key j visible to position i iff j <= i - shift.  pool uint8 [pages, Hkv, 2304] (int4ah:32
    chunks).  lomask uint8 [Hkv] (bit g: group g gets the q lo wgmma; None = all).  Returns (o bf16 [nq, Hq, 128] in
    the stored V basis, lse fp32 [Hq, nq], natural log)."""
    nq, Hq, _ = qh.shape
    Hkv = pool.shape[1]
    assert pool.dtype == torch.uint8 and pool.stride(1) == 144 * PAGE and bt_row.dtype == torch.int32
    assert qh.stride(0) == Hq * D and qf.stride(0) == Hq
    if out is None:
        out = torch.empty(nq, Hq, D, dtype=torch.bfloat16, device=qh.device)
    assert out.stride(2) == 1 and out.stride(1) == D
    lse = torch.empty(Hq, nq, dtype=torch.float32, device=qh.device)
    _check(lib().frp_int4_prefill(qh.data_ptr(), ql.data_ptr(), qf.data_ptr(), qh.stride(0), pool.data_ptr(),
                                  pool.stride(0), bt_row.data_ptr(), nq, nk, p0, shift, Hkv, _ptr(lomask),
                                  out.data_ptr(), out.stride(0), lse.data_ptr(), stages,
                                  torch.cuda.current_stream().cuda_stream), "frp_int4_prefill")
    return out, lse


def prep_q_f16(q: torch.Tensor, kfull=None, kmask=None):
    """q bf16 [nq, Hq, 128] -> FP16 [nq, Hq, 128] in the keys' stored basis (static FlagRot transforms as prep_q)."""
    nq, Hq, _ = q.shape
    assert q.dtype == torch.bfloat16 and q.stride(2) == 1 and q.stride(1) == D
    qs = torch.empty(nq, Hq, D, dtype=torch.float16, device=q.device)
    _check(lib().frp_prep_q(q.data_ptr(), q.stride(0), nq, Hq, _ptr(kfull), _ptr(kmask), 0.0, qs.data_ptr(), None,
                            None, torch.cuda.current_stream().cuda_stream), "frp_prep_q")
    return qs


def int4_prefill_kbf(q, scale, pool, bt_row, nk, p0, shift, kmask=None, kfull=None, konly=False, out=None, stages=34):
    """v3, QK in FP16: q bf16 (original basis; converted to FP16 in the kernel) with K dequantized + inverse-transformed
    (static: kmask [Hkv, 4] / kfull [Hkv] uint8; konly: per-token key-only 2x2 flags from the scale sign bits) in packed
    FP16 in shared memory; or q FP16 already in the stored basis (prep_q_f16) with K kept in the stored basis (pass
    kmask = kfull = None).  PV as int4_prefill.  Returns (o [nq, Hq, 128] bf16 in the
    stored V basis, lse [Hq, nq])."""
    nq, Hq, _ = q.shape
    Hkv = pool.shape[1]
    assert q.dtype in (torch.bfloat16, torch.float16) and q.stride(2) == 1 and q.stride(1) == D
    assert pool.dtype == torch.uint8 and pool.stride(1) == 144 * PAGE and bt_row.dtype == torch.int32
    if out is None:
        out = torch.empty(nq, Hq, D, dtype=torch.bfloat16, device=q.device)
    lse = torch.empty(Hq, nq, dtype=torch.float32, device=q.device)
    _check(lib().frp_int4_prefill_kbf(q.data_ptr(), q.stride(0), scale * LOG2E, _ptr(kmask), _ptr(kfull), int(konly),
                                      int(q.dtype == torch.float16),
                                      pool.data_ptr(), pool.stride(0), bt_row.data_ptr(), nq, nk, p0, shift, Hkv,
                                      out.data_ptr(), out.stride(0), lse.data_ptr(), stages,
                                      torch.cuda.current_stream().cuda_stream), "frp_int4_prefill_kbf")
    return out, lse
