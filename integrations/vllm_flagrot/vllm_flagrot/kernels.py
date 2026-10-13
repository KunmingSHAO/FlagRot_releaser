"""ctypes front end of libflagrot_kernels.so (csrc/flagrot_capi.cu + kernels/csrc/{attn_decode_v2,attn_codebook,
attn_cb_write3}.cu).

Why ctypes + a plain shared library: the kernels need the CUDA 12.9 ptxas (12.4 serializes their wgmma) and the vLLM env runs torch 2.13 + CUDA 13.0, so a torch extension would have to be rebuilt against
torch 2.13 headers with a CUDA 13 toolkit.  The library links the 12.9 runtime statically, needs only the driver, and
takes raw device pointers + torch's current cudaStream_t: it is captured by CUDA graphs like any other launch.
The library is built on first use (csrc/build.sh, file lock) when it is missing or older than any kernel source.

Entry points (all asynchronous on the given stream):
  v2_write(kv, slots, mask, rb, pool, page, fmt)                  static FlagRot INT4 paged writer
  v2_attn(q, pool, bt, seq_lens, out, plan, ws, ..., fmt)         static FlagRot INT4 paged decode attention (+window)
                                                                  fmt "int4" (symmetric) | "int4a" (asymmetric int4a:32,
                                                                  stored as blockflag int4ah:32;
  cb_write(kv, slots, CT, co, kind, pool, page)                   per-token codebook writer (v1)
  cb_write3(kv, slots, CT, co, C1, beta, kind, pool, page)        per-token codebook writer v3 (same bytes; C1, beta =
                                                                  w3_calib(C_q), once per layer)
  w3_static_write(kv, slots, mask, hfull, asym, pool, page, path) static FlagRot INT4 writer v3 = head_fwht + v2_write
                                                                  fused (same bytes)
  cb_attn(q, pool, bt, seq_lens, out, plan, ws, kind, ...)        codebook paged decode attention (+window)
  plan_v2 / plan_cb                                               split-K planner (host side, no device sync)
  dequant_v2_pages / dequant_cb_pages                             PyTorch layout parsers (prefill FA path, tests)

GQA groups != 4 and head dim 256, ABI 5): the generalized static kernel (variant 264 | GEN
[| GT8] [| HD256]) reads static int4 / int4a pages and the key-only 2x2 codebook pages (fmt "konly" / "konlya");
D = 256 pages are two 128-d half chunks per (page, KV head); split-K partials are per q tile ("virtual head",
Hkv * ceil(G / GT) per row) and hold GT x D floats.
"""
from __future__ import annotations

import ctypes
import fcntl
import math
import os
import subprocess
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_PLUGIN = _HERE.parent                                   # integrations/vllm_flagrot
ABI = 9                                  # 7: kv_start of the attention entry points (Phase 2k); 8: INT2 (Phase 3a);
                                         # 9: static key centering (kmu, last argument of the four writers; int2c)
D, RB, NG = 128, 32, 4
# static v2 formats (kernels/csrc/attn_decode_v2.cuh AttnFmt2) and their default kernel variant / stages
# (kernels/attn_decode_v2.py DEFAULT_VARIANT: 264 = FP8 PV, conflict-free P', deferred PV wait; the only
# variant of int4a)
V2_FMT = {"int4": 4, "int4a": 5, "konly": 6, "konlya": 7,       # konly / konlya: generalized kernel only
          "int2a": 8, "konlya2": 9}                             # Phase 3a: INT2 (int2a:32 stored as int2ah:32)
V2_VARIANT = {"int4": 264, "int4a": 264, "konly": 264, "konlya": 264, "int2a": 264, "konlya2": 264}
INT2_FMTS = ("int2a", "konlya2")                                 # 2-bit page layout (80 B per token and 128-d half)
GEN, GT8, HD256 = 1024, 2048, 4096                                # variant bits of the generalized kernel


def qtile(G: int, Dh: int = 128) -> int:
    """q heads per CTA of the generalized kernel (= kernels/attn_decode_v2.qtile): 8 for G 5..8 at D 128, else 4."""
    return 8 if Dh == 128 and 5 <= G <= 8 else 4


def v2_variant(fmt: str, G: int = 4, Dh: int = 128) -> int:
    """kernel variant of a static / key-only cache for GQA group G and head dim Dh."""
    if G == 4 and Dh == 128 and fmt in ("int4", "int4a", "int2a"):
        return V2_VARIANT[fmt]
    return 264 | GEN | (GT8 if qtile(G, Dh) == 8 else 0) | (HD256 if Dh == 256 else 0)


def tile_of(variant: int) -> int:
    return 8 if (variant & GEN) and (variant & GT8) else 4


def vheads(Hkv: int, G: int, Dh: int = 128, fmt: str = "int4") -> int:
    """CTAs per (row, split) of the static / generalized kernels: Hkv * ceil(G / GT)"""
    return Hkv * -(-G // tile_of(v2_variant(fmt, G, Dh)))
STAGES = 4
MAX_CHUNK, MAX_SPLITS = 8192, 512
PLAN_C0, PLAN_C1 = 1024, 4                               # = kernels/attn_decode_v2.py / attn_codebook.py
# small-batch INT4 split planner of kernels/attn_decode_v2.py (_plan_int4)
PLAN_R = {1: 1.6, 2: 3.9, 3: 3.9, 4: 3.9}
PLAN_CFIX, PLAN_MERGE = 4.0, 2.0
PLAN_SPLITS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 39, 48, 64, 78)

# codebook kinds (kernels/csrc/attn_codebook.cuh CbKind) that the plugin supports: sign-flipped-Hadamard family
# (+ konly = key-only 2x2 'cq+f+konly': keys I/H32 per vector + H128 per head, flags in scale sign bits; values plain;
#  konlya = konly with asymmetric INT4 int4a:32: unsigned codes, zero point in the 4 low bits of the scale words)
CB_KINDS = {"final": 3, "head8": 4, "head16": 5, "q4x4": 7, "q4x4k": 8, "finalk": 9, "konly": 6, "konlya": 10,
            "konlya2": 11}                               # konlya2 (Phase 3a): konlya with INT2 (int2a:32 / int2ah:32)
KV_DYN_KIND = {"cq+f+S+Q5+FS+R5": "final", "cq+f+FS+R6": "head8", "cq+f+FS+R14": "head16",
               "cq+f+S+Q1+FS+R1": "q4x4", "cq+f+S+Q1+FS+R1+konly": "q4x4k", "cq+f+S+Q5+FS+R5+konly": "finalk",
               "cq+f+konly": "konly"}
CB_DEFAULT_VARIANT = {"final": 0, "head8": 0, "head16": 0, "q4x4": 0, "q4x4k": 0, "finalk": 0, "konly": 0,
                      "konlya": 0, "konlya2": 0}                      # konlya 0: value zero points on the CUDA cores
K2 = ("konly", "konlya", "konlya2")                      # key-only 2x2 kinds (k_cbk)
ASYM_KIND = {"konly": "konlya"}                          # codebook kinds with an asymmetric INT4 kernel
ASYM2_KIND = {"konly": "konlya2"}                        # ... with an asymmetric INT2 kernel (Phase 3a)
LBITS = {"final": 3, "head8": 1, "head16": 1, "q4x4": 2, "q4x4k": 2, "finalk": 3}
HBITS = {"final": 3, "head8": 4, "head16": 5, "q4x4": 2, "q4x4k": 2, "finalk": 3}
KONLY = ("q4x4k", "finalk")
SEEDS = (0,) + tuple(range(2, 16))                       # sign table entries (CB_NSG = 15)


def _src_dir() -> Path:
    env = os.environ.get("FLAGROT_KERNELS_SRC")
    return Path(env) if env else _PLUGIN.parent.parent / "kernels" / "csrc"


def _lib_path() -> Path:
    return Path(os.environ.get("FLAGROT_LIB", _PLUGIN / "build" / "libflagrot_kernels.so"))


def _sources() -> list:
    """the library's translation units + build script + every header they include (recursively, `#include "..."`):
    a change to any of them (e.g. attn_merge.cuh, attn_cb_common.cuh) triggers a rebuild."""
    import re
    src = _src_dir()
    todo = [_PLUGIN / "csrc" / "flagrot_capi.cu", _PLUGIN / "csrc" / "build.sh", src / "attn_decode_v2.cu",
            src / "attn_codebook.cu", src / "attn_cb_write3.cu"]
    seen = []
    while todo:
        f = todo.pop()
        if f in seen or not f.exists():
            continue
        seen.append(f)
        if f.suffix in (".cu", ".cuh", ".h"):
            for inc in re.findall(r'^\s*#include\s+"([^"]+)"', f.read_text(errors="replace"), re.M):
                todo.append(src / inc)
    return seen


def _stale(lib: Path) -> bool:
    if not lib.exists():
        return True
    if os.environ.get("FLAGROT_NO_REBUILD"):
        return False
    t = lib.stat().st_mtime
    return any(s.stat().st_mtime > t for s in _sources())


_LIB = None


def lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    path = _lib_path()
    if _stale(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path) + ".lock", "w") as lk:            # several vLLM workers may start at once
            fcntl.flock(lk, fcntl.LOCK_EX)
            if _stale(path):
                env = dict(os.environ, FLAGROT_KERNELS_SRC=str(_src_dir()))
                subprocess.run(["bash", str(_PLUGIN / "csrc" / "build.sh"), str(path.parent)], check=True, env=env,
                               stdout=subprocess.DEVNULL)
            fcntl.flock(lk, fcntl.LOCK_UN)
    L = ctypes.CDLL(str(path))
    P, I, LL, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_float
    L.fr_abi_version.restype = I
    if L.fr_abi_version() != ABI:
        raise RuntimeError(f"libflagrot_kernels ABI {L.fr_abi_version()} != {ABI}: rebuild ({path})")
    L.fr_v2_attn.argtypes = [I, P, LL, P, P, I, P, I, I, I, I, I, F, P, P, I, I, I, I, I, P, P, P, P, LL, P, P, P,
                             LL, LL, P, I, I]
    L.fr_v2_write.argtypes = [I, P, LL, P, I, I, I, P, I, P, P, I, P]
    L.fr_v2_occupancy.argtypes = [I, I, I, I]
    L.fr_v2_chunk_bytes.argtypes = [I, I]
    L.fr_cb_write.argtypes = [P, LL, P, I, I, I, I, I, P, P, P, I, P, P, I, P]
    L.fr_cb_attn.argtypes = [P, LL, P, P, I, P, I, I, I, I, I, F, P, I, I, I, P, P, P, P, LL, P, P, P, LL, LL, P, I]
    L.fr_cb_occupancy.argtypes = [I, I, I, I]
    L.fr_cb_chunk_bytes.argtypes = [I, I]
    L.fr_last_error.restype = ctypes.c_char_p
    L.fr_head_fwht.argtypes = [P, LL, P, LL, I, I, P, F, P, I]
    L.fr_cb_write3.argtypes = [P, LL, P, I, I, I, I, P, P, P, P, P, I, P, I, I, P, P]
    L.fr_w3_static_write.argtypes = [P, LL, P, I, I, I, P, P, I, P, I, P, P]
    for n in ("fr_v2_attn", "fr_v2_write", "fr_v2_occupancy", "fr_v2_chunk_bytes", "fr_cb_write", "fr_cb_attn",
              "fr_cb_occupancy", "fr_cb_chunk_bytes", "fr_cb_nsg", "fr_head_fwht", "fr_cb_write3",
              "fr_w3_static_write"):
        getattr(L, n).restype = I
    _LIB = L
    return L


def _check(rc: int, what: str):
    if rc != 0:
        raise RuntimeError(f"{what}: launch failed (code {rc}, {lib().fr_last_error().decode()})")


def _ptr(t):
    return None if t is None else t.data_ptr()


_DEV_IDX: list = []


def _stream():
    """raw handle of the current CUDA stream (captured by CUDA graphs like any launch).  torch.cuda.current_stream()
    costs ~16 us of host time per call (device-index lookups), i.e. per kernel launch of the eager mixed steps;
    the raw call with the device index cached per process (one device per vLLM worker) ~0.2 us."""
    if not _DEV_IDX:
        _DEV_IDX.append(torch.cuda.current_device())
    return torch._C._cuda_getCurrentRawStream(_DEV_IDX[0])


# ------------------------------------------------------------------ constants
def research_signs(seed: int) -> torch.Tensor:
    """= blockflag.qattention._signs(128, 'cpu', seed)."""
    g = torch.Generator().manual_seed(1234 + D + 7919 * seed)
    return (torch.randint(0, 2, (D,), generator=g) * 2 - 1).float()


_SG = None


def sign_words():
    """ctypes uint32[15 * 4]: bit d of word 4k + d // 32 set iff S_{SEEDS[k]}[d] < 0 (kernels/codebook_ref.sign_bits)."""
    global _SG
    if _SG is None:
        w = []
        for k in SEEDS:
            b = (research_signs(k) < 0).to(torch.int64)
            w += [int(sum(int(b[32 * j + i]) << i for i in range(32))) for j in range(4)]
        _SG = (ctypes.c_uint * len(w))(*w)
    return _SG


def v2_chunk_bytes(page: int, Dh: int = 128, fmt: str = "int4") -> int:
    return page * (80 if fmt in INT2_FMTS else 144) * Dh // 128


def cb_chunk_bytes(kind: str, page: int, Dh: int = 128) -> int:
    if kind in INT2_FMTS:
        return page * 80 * Dh // 128
    return page * (144 if kind in K2 else 146 if kind in KONLY else 148) * Dh // 128


# ------------------------------------------------------------------ planner (host only)
_OCC: dict = {}
_NUM_SMS = None


def _slots_per_gpu(key, occ_fn) -> int:
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if key not in _OCC:
        _OCC[key] = max(1, occ_fn())
    return _OCC[key] * _NUM_SMS


def _plan(B, Lmax, Hkv, page, slots, nsplit=None):
    """(nsplit, chunk) from the wave-quantization cost model of kernels/attn_decode_v2.plan (whole 64-token blocks)."""
    gran = max(64, page)
    Lmax = max(1, int(Lmax))
    if nsplit is not None:
        chunk = min(MAX_CHUNK, math.ceil(Lmax / nsplit / gran) * gran)
        return math.ceil(Lmax / chunk), chunk
    best = None
    for ns in range(max(1, math.ceil(Lmax / MAX_CHUNK)), min(MAX_SPLITS, math.ceil(Lmax / gran)) + 1):
        chunk = math.ceil(Lmax / ns / gran) * gran
        if chunk > MAX_CHUNK:
            continue
        nse = math.ceil(Lmax / chunk)
        if best is not None and nse == best[1]:
            continue
        cost = math.ceil(B * Hkv * nse / slots) * (chunk + PLAN_C0) + (PLAN_C1 * nse if nse > 1 else 0)
        if best is None or cost < best[0] - 1e-9:
            best = (cost, nse, chunk)
    return best[1], best[2]


def _plan_cost_int4(B, nse, chunk, Hkv, occ, sms):
    """modelled time (us) of B x Hkv x nse CTAs of chunk tokens (= kernels/attn_decode_v2.plan_cost_int4)"""
    full, rem = divmod(B * Hkv * nse, occ * sms)
    blk = chunk / 64
    t = full * (PLAN_CFIX + blk * occ / PLAN_R[min(occ, 4)])
    if rem:
        k = min(4, math.ceil(rem / sms))
        t += PLAN_CFIX + blk * k / PLAN_R[k]
    return t + PLAN_MERGE * math.ceil(nse / 16)


def _plan_int4(B, Lmax, Hkv, occ, sms, gran):
    """= kernels/attn_decode_v2._plan_int4: split count of the small-batch time model."""
    best = None
    waves = {max(1, (m * occ * sms) // (B * Hkv)) for m in range(1, 7)}
    for ns in sorted(set(PLAN_SPLITS) | waves | {max(1, math.ceil(Lmax / MAX_CHUNK))}):
        chunk = math.ceil(Lmax / ns / gran) * gran
        if chunk > MAX_CHUNK or ns > math.ceil(Lmax / gran):
            continue
        nse = math.ceil(Lmax / chunk)
        c = _plan_cost_int4(B, nse, chunk, Hkv, occ, sms)
        if best is None or c < best[0] - 1e-9:
            best = (c, nse, chunk)
    return best[1], best[2]


def plan_v2(B, Lmax, Hkv, page, nsplit=None, fmt="int4", G=4, Dh=128):
    """= kernels/attn_decode_v2.plan for the default variant: the time model when B Hkv < resident-CTA slots,
    else the wave-quantization model (C0 = 1024 token-equivalents per CTA).  GQA G != 4 / head dim 256: the
    generalized kernel's CTAs per (row, split) = vheads(Hkv, G) and its occupancy."""
    v = v2_variant(fmt, G, Dh)
    HV = vheads(Hkv, G, Dh, fmt)
    s = _slots_per_gpu(("v2", fmt, page, v), lambda: lib().fr_v2_occupancy(V2_FMT[fmt], v, STAGES, page))
    if nsplit is None and B * HV < s:
        return _plan_int4(B, max(1, int(Lmax)), HV, s // _NUM_SMS, _NUM_SMS, max(64, page))
    return _plan(B, Lmax, HV, page, s, nsplit)


def plan_cb(B, Lmax, Hkv, page, kind, nsplit=None, G=4, Dh=128):
    if kind in K2 and (G != 4 or Dh != 128):   # key-only 2x2 on the generalized static kernel
        return plan_v2(B, Lmax, Hkv, page, nsplit, fmt=kind, G=G, Dh=Dh)
    v = CB_DEFAULT_VARIANT[kind]
    s = _slots_per_gpu(("cb", kind, page), lambda: lib().fr_cb_occupancy(CB_KINDS[kind], v, STAGES, page))
    return _plan(B, Lmax, Hkv, page, s, nsplit)


class Workspace:
    """split-K partials for up to `parts` (row, kv head [generalized kernel: q tile], split) partials of GT q heads x
    Dh dims; counter: per (row, kv head / q tile) merge tickets (zeroed once, self-resetting).  Allocate once before
    CUDA graph capture (pointers are baked into graphs)."""

    def __init__(self, rows: int, Hkv: int, parts: int, device, GT: int = 4, Dh: int = 128):
        self.rows, self.parts, self.HV, self.GT, self.Dh = rows, parts, Hkv, GT, Dh
        self.opart = torch.empty(parts * GT * Dh, device=device, dtype=torch.float32)
        self.mlpart = torch.empty(parts * GT * 2, device=device, dtype=torch.float32)
        self.counter = torch.zeros(rows * Hkv, device=device, dtype=torch.int32)

    def fits(self, B, Hkv, nsplit_total, GT=4, Dh=128):
        return (B <= self.rows and B * Hkv * nsplit_total <= self.parts and Hkv <= self.HV and GT * Dh <= self.GT * self.Dh
                and B * Hkv * nsplit_total * GT * Dh <= self.parts * self.GT * self.Dh)


# ------------------------------------------------------------------ launches
def _kmu(kmu, kv):
    """static key centering vector of the K rows: fp32 [Hkv, Dh] contiguous on kv's device, or None (off)."""
    if kmu is None:
        return None
    Hkv, Dh = kv.shape[1] // 2, kv.shape[2]
    assert kmu.dtype == torch.float32 and kmu.is_contiguous() and tuple(kmu.shape) == (Hkv, Dh) \
        and kmu.device == kv.device, (kmu.dtype, tuple(kmu.shape), (Hkv, Dh))
    return kmu.data_ptr()


def _bt_args(block_table):
    return block_table.data_ptr(), block_table.stride(0)


def v2_write(kv, slots, mask, rb, pool, page, fmt="int4", kmu=None):
    """kv bf16 [T, 2*Hkv, Dh] (Dh 128 | 256, head rows contiguous, any token stride) -> pool [num_pages, Hkv,
    144 page Dh / 128] at slots int64 [T] (< 0 skipped); mask uint8 [2*Hkv, Dh/rb] (K rows first); fmt "int4" | "int4a"."""
    T = kv.shape[0]
    if T == 0:
        return
    Dh = kv.shape[2]
    assert kv.dtype == torch.bfloat16 and kv.stride(2) == 1 and kv.stride(1) == Dh and slots.dtype == torch.int64
    _check(lib().fr_v2_write(V2_FMT[fmt], kv.data_ptr(), kv.stride(0), slots.data_ptr(), T, kv.shape[1] // 2, page,
                             mask.data_ptr(), rb, pool.data_ptr(), _stream(), Dh, _kmu(kmu, kv)), "fr_v2_write")


def v2_attn(q, pool, block_table, seq_lens, out, nsplit, chunk, ws, qscale, kmask=None, vmask=None, rb=32,
            tail=None, tail_len=None, tail_idx=None, tail_bs=None, tail_ts=None, page=16, fmt="int4", kv_start=0):
    """q bf16 [B, G*Hkv, Dh] -> out bf16 [B, G*Hkv*Dh] (V back-rotated: vrot when vmask is given; Hkv = pool.shape[1],
    G != 4 or Dh = 256: the generalized kernel).  block_table int32 [B, *] (row stride may be 0), seq_lens int32
    [B] (cached tokens); tail: bf16 window rows (strides tail_bs / tail_ts elements) with tail_len int32 [B] (+ tail_idx
    int32 [B]).  qscale = softmax scale * log2(e).  fmt "int4" | "int4a" | "konly" | "konlya" (key-only: no masks).
    kv_start: tokens [0, kv_start) of every row are masked in the paged body (Phase 2k: BF16 sinks at real positions)."""
    B, Hq, Dh = q.shape
    Hkv = pool.shape[1]
    G = Hq // Hkv
    assert Hq == G * Hkv and q.stride(2) == 1 and q.stride(1) == Dh and out.stride(1) == 1
    v = v2_variant(fmt, G, Dh)
    GT = tile_of(v)
    assert ws.fits(B, vheads(Hkv, G, Dh, fmt), nsplit + (tail is not None), GT, Dh), "split-K workspace too small"
    bt, bts = _bt_args(block_table)
    qrot = kmask is not None
    vrot = vmask is not None
    _check(lib().fr_v2_attn(V2_FMT[fmt], q.data_ptr(), q.stride(0), pool.data_ptr(), bt, bts, seq_lens.data_ptr(), B, Hkv, page,
                            nsplit, chunk, qscale, _ptr(kmask), _ptr(vmask), rb, int(qrot), int(vrot), v,
                            STAGES, ws.opart.data_ptr(), ws.mlpart.data_ptr(), ws.counter.data_ptr(), out.data_ptr(),
                            out.stride(0), _ptr(tail), _ptr(tail_len), _ptr(tail_idx),
                            0 if tail is None else tail_bs, 0 if tail is None else tail_ts, _stream(), G,
                            int(kv_start)), "fr_v2_attn")


def cb_write(kv, slots, CT, co, kind, pool, page, tg=None, kmu=None):
    """kv bf16 [T, 2*Hkv, Dh] -> pool [num_pages, Hkv, chunk] at slots int64 [T]; CT [Hkv, Dh, Dh] = C_q^T, co
    [Hkv, Dh] fp32 (static per-layer calibration); Dh 256: key-only 2x2 kinds only."""
    T = kv.shape[0]
    if T == 0:
        return
    Dh = kv.shape[2]
    assert kv.dtype == torch.bfloat16 and kv.stride(2) == 1 and kv.stride(1) == Dh and slots.dtype == torch.int64
    if tg is None:  # kernels/attn_codebook.write_tg
        tg = 1 if T <= 16 else (2 if T <= 64 or kind not in K2 else 4)
    _check(lib().fr_cb_write(kv.data_ptr(), kv.stride(0), slots.data_ptr(), 0, T, kv.shape[1] // 2, page, tg,
                             CT.data_ptr(), co.data_ptr(), sign_words(), CB_KINDS[kind], pool.data_ptr(), _stream(), Dh,
                             _kmu(kmu, kv)), "fr_cb_write")


# ------------------------------------------------------------------ writer v3 (kernels/csrc/attn_cb_write3.cuh)
# codebook kinds with a v3 writer (all of CB_KINDS; kernels/attn_codebook.W3_KINDS) and its launch constants
W3_KINDS = ("final", "head8", "head16", "q4x4", "q4x4k", "finalk", "konly", "konlya", "konlya2")
W3_KAPPA = 2.0 ** -16
W3_SPLIT_T = int(os.environ.get("W3_SPLIT_T", "256"))   # T <= this: K / V kernels on two streams (decode steps)
W3_CTAS_PER_SM, W3_VCTAS_PER_SM = 2, 3                   # resident key / value CTAs per SM (persistent grids)
_SIDE: dict = {}


def side_stream(device=None):
    """the second stream of the v3 codebook writer (one per device; created before CUDA-graph capture by the
    backend's per-layer state)."""
    dev = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    if dev not in _SIDE:
        _SIDE[dev] = torch.cuda.Stream(dev)
    return _SIDE[dev]


def w3_nchunk(Hkv, per_sm):
    """persistent CTAs per kv head: one resident wave (= kernels/attn_codebook.w3_nchunk)."""
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return max(1, (per_sm * _NUM_SMS) // Hkv)


def w3_calib(Cq, kappa=None, nd=8):
    """host constants of the codebook writer v3 (= kernels/attn_codebook.w3_calib): C1 fp16
    [H, 168, 128] = rows 0..127 fp16(2^tau C_sym) (C_sym = (C + C^T) / 2), rows 128..135 the head correction F (rows
    D of E = 2^tau C_sym - C1, off-D entries doubled), rows 136 + 8 j .. the same for the diagonal 32-block j; aux
    fp32 [H, 48]: 0 head bound beta, 1..4 block bounds, 5 absolute slack, 8..15 the head's D, 16 + 8 j .. block j's D.
    beta = ||E_RR||_2 + ||F - fp16(F)||_2 + 2^-10 ||E||_2 + kappa rho(M), M = 2^tau (|C| + |C|^T) / 2, fp64, +1 %.
    Cq [H, 128, 128] (the writer's C_q, i.e. CT^T)."""
    kappa = W3_KAPPA if kappa is None else kappa
    C = Cq.double()
    H = C.shape[0]
    Cs = (C + C.transpose(1, 2)) / 2
    mx = Cs.abs().amax((1, 2))
    tau = torch.where(mx > 0, torch.floor(torch.log2(2.0 ** 14 / mx.clamp_min(1e-300))), torch.zeros_like(mx))
    sc = (2.0 ** tau)[:, None, None]
    Cp = Cs * sc
    C1 = Cp.float().half()
    E = Cp - C1.double()
    M = (C.abs() + C.abs().transpose(1, 2)) / 2 * sc
    spec = lambda X: torch.linalg.eigvalsh(X).abs().amax(-1)  # noqa: E731
    out = torch.zeros(H, 168, 128, dtype=torch.float16, device=C.device)
    out[:, :128] = C1
    aux = torch.zeros(H, 48, dtype=torch.float64, device=C.device)
    for bi, (lo, hi) in enumerate([(0, 128)] + [(32 * j, 32 * j + 32) for j in range(4)]):
        Eb, Mb = E[:, lo:hi, lo:hi], M[:, lo:hi, lo:hi]
        n = hi - lo
        Dd = torch.argsort(Eb.norm(dim=2), dim=1, descending=True)[:, :nd]                 # [H, nd] (block coords)
        inD = torch.zeros(H, n, dtype=torch.bool, device=C.device)
        inD.scatter_(1, Dd, True)
        Fm = torch.gather(Eb, 1, Dd[:, :, None].expand(-1, -1, n))
        Fm = torch.where(inD[:, None, :], Fm, 2 * Fm)
        Fh = Fm.float().half()
        Err = torch.where(inD[:, :, None] | inD[:, None, :], torch.zeros_like(Eb), Eb)
        beta = (spec(Err) + torch.linalg.matrix_norm(Fm - Fh.double(), 2) + 2.0 ** -10 * spec(Eb)
                + kappa * spec(Mb)) * 1.01
        row = 128 if bi == 0 else 136 + 8 * (bi - 1)
        out[:, row:row + nd, lo:hi] = Fh
        aux[:, bi] = beta
        aux[:, 8 + 8 * bi:16 + 8 * bi] = (Dd + lo).double()
    aux[:, 5] = 2.0 ** -40 * spec(M)
    return out.contiguous(), aux.float().contiguous()


def _w3_kv(kv):
    """the v3 writers read 16-byte rows with cp.async / vector loads: aligned rows, token stride % 8 == 0."""
    if kv.stride(2) != 1 or kv.stride(1) != D or kv.stride(0) % 8 or kv.data_ptr() % 16:
        kv = kv.contiguous()
    return kv


def cb_write3(kv, slots, CT, co, C1, beta, kind, pool, page, split=None, kmu=None):
    """= cb_write (same pool bytes) with the v3 writer: kv bf16 [T, 2*Hkv, 128] -> pool at slots int64 [T]; C1, beta
    from w3_calib(C_q) of the layer.  split (default: T <= W3_SPLIT_T): K kernel on the current stream, V kernel on
    the side stream (fork / join by stream waits; CUDA-graph capturable), else both on the current stream."""
    T = kv.shape[0]
    if T == 0:
        return
    assert kv.dtype == torch.bfloat16 and slots.dtype == torch.int64
    assert kv.shape[2] == D, f"writer v3: head dim {D} only (got {kv.shape[2]}; use the v1 writer)"
    kv = _w3_kv(kv)
    slots = slots.contiguous()
    Hkv = kv.shape[1] // 2
    L = lib()
    nk, nv = w3_nchunk(Hkv, W3_CTAS_PER_SM), w3_nchunk(Hkv, W3_VCTAS_PER_SM)
    km = _kmu(kmu, kv)
    args = (kv.data_ptr(), kv.stride(0), slots.data_ptr(), 0, T, Hkv, page, CT.data_ptr(), co.data_ptr(),
            C1.data_ptr(), beta.data_ptr(), sign_words(), CB_KINDS[kind], pool.data_ptr())
    if split is None:
        split = T <= W3_SPLIT_T
    if not split:
        _check(L.fr_cb_write3(*args, nk, nv, _stream(), km), "fr_cb_write3")
        return
    cur = torch.cuda.current_stream()
    side = side_stream(kv.device)
    side.wait_stream(cur)
    _check(L.fr_cb_write3(*args, nk, -1, cur.cuda_stream, km), "fr_cb_write3 (keys)")
    _check(L.fr_cb_write3(*args, -1, nv, side.cuda_stream, None), "fr_cb_write3 (values)")
    cur.wait_stream(side)   # joined at once: later work (and memory reuse) on the current stream follows both kernels


def w3_static_write(kv, slots, mask, hfull, pool, page, fmt="int4", path=0, kmu=None):
    """= head_fwht(kv, hfull, 1/16) + v2_write(.., mask, rb 32, ..) fused (same pool bytes): kv bf16 [T, 2*Hkv, 128]
    in the ORIGINAL basis of the whole-head heads; mask uint8 [2*Hkv, 4] (or None), hfull uint8 [2*Hkv] (or None),
    K rows first; path 0: by T, 1: prefill kernel (tiles of 32 tokens), 2: decode kernel (scattered slots)."""
    T = kv.shape[0]
    if T == 0:
        return
    assert kv.dtype == torch.bfloat16 and slots.dtype == torch.int64
    assert kv.shape[2] == D, f"writer v3: head dim {D} only (got {kv.shape[2]}; use the v1 writer)"
    kv = _w3_kv(kv)
    slots = slots.contiguous()
    _check(lib().fr_w3_static_write(kv.data_ptr(), kv.stride(0), slots.data_ptr(), T, kv.shape[1] // 2, page,
                                    _ptr(mask), _ptr(hfull), {"int4": 0, "int4a": 1, "int2a": 2}[fmt], pool.data_ptr(),
                                    path, _stream(), _kmu(kmu, kv)),
           "fr_w3_static_write")


def cb_attn(q, pool, block_table, seq_lens, out, nsplit, chunk, ws, qscale, kind, tail=None, tail_len=None,
            tail_idx=None, tail_bs=None, tail_ts=None, page=16, kv_start=0):
    B, Hq, Dh = q.shape
    if kind in K2 and (Hq != 4 * pool.shape[1] or Dh != 128):   # key-only 2x2, any GQA / head dim 256
        return v2_attn(q, pool, block_table, seq_lens, out, nsplit, chunk, ws, qscale, tail=tail, tail_len=tail_len,
                       tail_idx=tail_idx, tail_bs=tail_bs, tail_ts=tail_ts, page=page, fmt=kind, kv_start=kv_start)
    Hkv = Hq // 4
    assert Hq == 4 * pool.shape[1] and Dh == D, f"codebook {kind}: GQA 4, head dim 128 only"
    assert q.stride(2) == 1 and q.stride(1) == D and out.stride(1) == 1
    assert ws.fits(B, Hkv, nsplit + (tail is not None)), "split-K workspace too small"
    bt, bts = _bt_args(block_table)
    _check(lib().fr_cb_attn(q.data_ptr(), q.stride(0), pool.data_ptr(), bt, bts, seq_lens.data_ptr(), B, Hkv, page,
                            nsplit, chunk, qscale, sign_words(), CB_KINDS[kind], CB_DEFAULT_VARIANT[kind], STAGES,
                            ws.opart.data_ptr(), ws.mlpart.data_ptr(), ws.counter.data_ptr(), out.data_ptr(),
                            out.stride(0), _ptr(tail), _ptr(tail_len), _ptr(tail_idx),
                            0 if tail is None else tail_bs, 0 if tail is None else tail_ts, _stream(), int(kv_start)),
           "fr_cb_attn")


def head_fwht(x, flags, scale, out=None):
    """x bf16 [T, H, Dh] (Dh 128 | 256, head rows contiguous) -> out [T, H, Dh]: scale * H_Dh x (unnormalized +-1
    Sylvester) for the heads with flags[h] != 0 (uint8 [H]), a copy otherwise; out may alias x (default: new tensor)."""
    T, H, Dh = x.shape
    assert x.dtype == torch.bfloat16 and x.stride(2) == 1 and x.stride(1) == Dh
    if out is None:
        out = torch.empty(T, H, Dh, dtype=x.dtype, device=x.device)
    assert out.stride(2) == 1 and out.stride(1) == Dh
    if T:
        _check(lib().fr_head_fwht(x.data_ptr(), x.stride(0), out.data_ptr(), out.stride(0), T, H, flags.data_ptr(),
                                  scale, _stream(), Dh), "fr_head_fwht")
    return out


# ------------------------------------------------------------------ layout parsers (PyTorch; prefill FA path, tests)
_IDX: dict = {}


def _idx(page, device):
    key = (page, str(device))
    if key in _IDX:
        return _IDX[key]
    P = page
    tg = torch.arange(4, device=device).view(4, 1, 1)
    j = torch.arange(4, device=device).view(1, 4, 1)
    i = torch.arange(4, device=device).view(1, 1, 4)
    dA = (32 * j + 4 * tg + i).reshape(-1)
    dB = (32 * j + 16 + 4 * tg + i).reshape(-1)
    w = torch.arange(4, device=device).view(4, 1, 1, 1, 1)
    gid = torch.arange(8, device=device).view(1, 8, 1, 1, 1)
    c8 = torch.arange(P // 8, device=device).view(1, 1, P // 8, 1, 1)
    sub = torch.arange(4, device=device).view(1, 1, 1, 4, 1)
    ii = torch.arange(4, device=device).view(1, 1, 1, 1, 4)
    shp = (4, 8, P // 8, 4, 4)
    dd = (64 * (sub >> 1) + 16 * w + gid + 8 * (sub & 1)).expand(shp).reshape(-1)
    tA = (8 * c8 + 2 * ii).expand(shp).reshape(-1)
    perm8 = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=device)
    tok = (8 * torch.arange(P // 8, device=device).view(-1, 1) + perm8.view(1, -1)).reshape(-1)
    _IDX[key] = (dA, dB, dd, tA, tok)
    return _IDX[key]


def _codes(byt, asym=False):
    """INT4 element pairs of the v2 layout: [..., 4 bytes] -> (A [..., 4], B [..., 4]); symmetric: sign-magnitude
    (signed values); asym (int4a): unsigned nibbles, A in the low and B in the high half of the same byte."""
    if asym:
        return byt & 15, (byt >> 4) & 15
    sA, mA = (byt >> 7) & 1, byt & 7
    sB = (byt >> 3) & 1
    mB = (torch.roll(byt, shifts=1, dims=-1) >> 4) & 7
    return torch.where(sA == 1, -mA, mA), torch.where(sB == 1, -mB, mB)


def parse_int4_pages(chunks, page, asym=False, Dh=128):
    """chunks uint8 [Np, H, >= 144 page Dh / 128] (v2 INT4 layout; Dh 256: two 128-d half chunks) -> codes int16
    [Np, H, P, 2, Dh] (stored coordinates), scale magnitudes fp32 [Np, H, P, 2, Dh / 32], raw scale words
    [Np, H, P, 2, Dh / 32] (int16).
    asym (int4a / konlya): the returned codes are c - z with the zero point z = word & 15; the
    scale is the word read as FP16 (sign bit masked: konlya keeps its transform flags there), so value = codes * mag
    exactly as for the symmetric format."""
    if Dh == 256:
        Np, H = chunks.shape[:2]
        c, m, sb = parse_int4_pages(chunks.reshape(Np, 2 * H, -1)[..., :144 * page], page, asym)
        cat = lambda x: x.view(Np, H, 2, page, 2, x.shape[-1]).permute(0, 1, 3, 4, 2, 5).reshape(Np, H, page, 2, -1)  # noqa: E731
        return cat(c), cat(m), cat(sb)
    P = page
    Np, H = chunks.shape[:2]
    dA, dB, dd, tA, tok = _idx(P, chunks.device)
    c = chunks.to(torch.int16)
    kc = c[:, :, :64 * P].reshape(Np, H, P, 4, 4, 4)
    vt = c[:, :, 64 * P:128 * P].reshape(Np, H, 4, 8, P // 8, 4, 4)
    K = torch.empty(Np, H, P, 128, dtype=torch.int16, device=c.device)
    cA, cB = _codes(kc, asym)
    K[..., dA] = cA.reshape(Np, H, P, 64)
    K[..., dB] = cB.reshape(Np, H, P, 64)
    V = torch.empty(Np, H, P, 128, dtype=torch.int16, device=c.device)
    vA, vB = _codes(vt, asym)
    V[:, :, tA, dd] = vA.reshape(Np, H, -1)
    V[:, :, tA + 1, dd] = vB.reshape(Np, H, -1)
    ks = chunks[:, :, 128 * P:136 * P].contiguous().view(torch.int16).reshape(Np, H, P, 4)
    vs = chunks[:, :, 136 * P:144 * P].contiguous().view(torch.int16).reshape(Np, H, 4, P)
    vsb = torch.empty_like(vs)
    vsb[..., tok] = vs
    sbits = torch.stack([ks, vsb.transpose(2, 3)], 3)                                   # [Np, H, P, 2, 4] int16
    mag = (sbits & 0x7FFF).view(torch.float16).float()
    codes = torch.stack([K, V], 3)
    if asym:
        codes = codes - (sbits & 15).repeat_interleave(RB, -1)
    return codes, mag, sbits


_IDX2: dict = {}


def _idx2(page, device):
    """INT2 layout index maps (kernels/csrc/attn_decode_v2.cuh): K bytes [phys row, 8 tg + 4 jp + i] x field (h, b) ->
    (token, d); V bytes [w, gid, c8, m, i] x field (r, b) -> (token, d)."""
    key = (page, str(device))
    if key in _IDX2:
        return _IDX2[key]
    P = page
    ar = lambda n: torch.arange(n, device=device)  # noqa: E731
    t = ar(P)
    phys = t ^ ((t >> 1) & 2)                                            # token t's physical K row
    tg, jp, i, h, b = (x.reshape(-1) for x in torch.meshgrid(ar(4), ar(2), ar(4), ar(2), ar(2), indexing="ij"))
    kbyte = 8 * tg + 4 * jp + i
    kshift = 4 * h + 2 * b
    kd = 32 * (2 * jp + h) + 16 * b + 4 * tg + i
    w, gid, c8, m, i2, r, b2 = (x.reshape(-1) for x in torch.meshgrid(ar(4), ar(8), ar(P // 8), ar(2), ar(4), ar(2), ar(2),
                                                                         indexing="ij"))
    vbyte = (((w * 8 + gid) * (P // 8) + c8) * 8 + 4 * m + i2)
    vshift = 4 * r + 2 * b2
    vd = 64 * m + 16 * w + gid + 8 * r
    vt = 8 * c8 + 2 * i2 + b2
    perm8 = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=device)
    tok = (8 * torch.arange(P // 8, device=device).view(-1, 1) + perm8.view(1, -1)).reshape(-1)
    _IDX2[key] = (phys, kbyte, kshift, kd, vbyte, vshift, vd, vt, tok)
    return _IDX2[key]


def parse_int2_pages(chunks, page, Dh=128):
    """INT2 pages (fmt "int2a" / kind "konlya2", Phase 3a; chunks uint8 [Np, H, >= 80 page Dh / 128]) -> the same as
    parse_int4_pages(asym=True): codes c - z int16 [Np, H, P, 2, Dh], scale magnitudes fp32 [Np, H, P, 2, Dh / 32], raw
    scale words int16 (zero point = word & 3, flags in the sign bits)."""
    if Dh == 256:
        Np, H = chunks.shape[:2]
        c, m, sb = parse_int2_pages(chunks.reshape(Np, 2 * H, -1)[..., :80 * page], page)
        cat = lambda x: x.view(Np, H, 2, page, 2, x.shape[-1]).permute(0, 1, 3, 4, 2, 5).reshape(Np, H, page, 2, -1)  # noqa: E731
        return cat(c), cat(m), cat(sb)
    P = page
    Np, H = chunks.shape[:2]
    phys, kbyte, kshift, kd, vbyte, vshift, vd, vt, tok = _idx2(P, chunks.device)
    c = chunks.to(torch.int16)
    kc = c[:, :, :32 * P].reshape(Np, H, P, 32)[:, :, phys]                            # [Np, H, P (token), 32]
    K = torch.empty(Np, H, P, 128, dtype=torch.int16, device=c.device)
    K[..., kd] = ((kc[..., kbyte] >> kshift) & 3).to(torch.int16)
    vc = c[:, :, 32 * P:64 * P]
    V = torch.empty(Np, H, P, 128, dtype=torch.int16, device=c.device)
    V[:, :, vt, vd] = ((vc[..., vbyte] >> vshift) & 3).to(torch.int16)
    ks = chunks[:, :, 64 * P:72 * P].contiguous().view(torch.int16).reshape(Np, H, P, 4)
    vs = chunks[:, :, 72 * P:80 * P].contiguous().view(torch.int16).reshape(Np, H, 4, P)
    vsb = torch.empty_like(vs)
    vsb[..., tok] = vs
    sbits = torch.stack([ks, vsb.transpose(2, 3)], 3)                                   # [Np, H, P, 2, 4] int16
    mag = (sbits & 0x7FFF).view(torch.float16).float()
    codes = torch.stack([K, V], 3) - (sbits & 3).repeat_interleave(RB, -1)
    return codes, mag, sbits


def parse_pages(chunks, page, fmt, Dh=128):
    """codes (zero point applied), magnitudes, scale words of a static format / key-only 2x2 kind."""
    if fmt in INT2_FMTS:
        return parse_int2_pages(chunks, page, Dh)
    return parse_int4_pages(chunks, page, asym=fmt in ("int4a", "konlya"), Dh=Dh)


def dequant_v2_pages(chunks, page, dtype=torch.bfloat16, fmt="int4", Dh=128):
    """static INT4 / INT2 pages (fmt "int4" | "int4a" | "int2a") -> K, V [Np, P, H, Dh] in the STORED (FlagRot-rotated)
    basis."""
    codes, mag, _ = parse_pages(chunks, page, fmt, Dh)
    x = codes.float().reshape(*codes.shape[:-1], codes.shape[-1] // RB, RB) * mag[..., None]
    x = x.reshape(*codes.shape).to(dtype)                                                # [Np, H, P, 2, 128]
    return x[..., 0, :].transpose(1, 2), x[..., 1, :].transpose(1, 2)


_HM: dict = {}


def hadamard(n, device, dtype=torch.float32):
    """orthonormal Sylvester Hadamard (symmetric) = blockflag fwht on the identity."""
    key = (n, str(device), dtype)
    if key not in _HM:
        h = torch.ones(1, 1, dtype=torch.float64)
        while h.shape[0] < n:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _HM[key] = (h / math.sqrt(n)).to(device=device, dtype=dtype)
    return _HM[key]


_SGT: dict = {}


def _sign_table(device):
    if str(device) not in _SGT:
        _SGT[str(device)] = torch.stack([torch.ones(D)] + [research_signs(k) for k in SEEDS]).to(device)
    return _SGT[str(device)]


def dequant_cb_pages(chunks, page, kind, dtype=torch.bfloat16, Dh=128):
    """codebook pages (family kinds) -> K, V [Np, P, H, Dh] in the ORIGINAL basis (inverse transforms in fp32;
    = kernels/codebook_ref.reconstruct_final).  Dh 256: key-only 2x2 kinds."""
    P = page
    Np, H = chunks.shape[:2]
    codes, mag, sbits = parse_pages(chunks, P, kind, Dh=Dh)
    if kind in K2:        # K vector bit = sign of its K scale; K head bit (H_Dh) = sign of the V group-0 scale
        dev = chunks.device
        ng = Dh // RB
        d = codes.float().reshape(*codes.shape[:-1], ng, RB) * mag[..., None]            # [Np, H, P, 2, ng, 32]
        fk = ((sbits[..., 1, 0].to(torch.int32) >> 15) & 1).bool()                      # [Np, H, P]
        lk = ((sbits[..., 0, :].to(torch.int32) >> 15) & 1).bool()                      # [Np, H, P, ng]
        dk = d[..., 0, :, :]
        kl = torch.where(lk[..., None], dk @ hadamard(RB, dev), dk).reshape(Np, H, P, Dh)
        kf = dk.reshape(Np, H, P, Dh) @ hadamard(Dh, dev)
        k = torch.where(fk[..., None], kf, kl).to(dtype)
        v = d[..., 1, :, :].reshape(Np, H, P, Dh).to(dtype)
        return k.transpose(1, 2), v.transpose(1, 2)
    assert Dh == D, f"codebook {kind}: head dim 128 only"
    nsw = 1 if kind in KONLY else 2
    sw = chunks[:, :, 144 * P:(144 + 2 * nsw) * P].contiguous().view(torch.int16).reshape(Np, H, nsw, P)
    sw = sw.to(torch.int32) & 0xFFFF
    if nsw == 1:
        sw = torch.cat([sw, torch.zeros_like(sw)], 2)
    sw = sw.permute(0, 1, 3, 2)                                                          # [Np, H, P, 2]
    lb, hb = LBITS[kind], HBITS[kind]
    local = torch.stack([(sw >> (lb * j)) & ((1 << lb) - 1) for j in range(4)], -1).long()   # [Np, H, P, 2, 4]
    mode = ((sw >> (4 * lb)) & ((1 << hb) - 1)).long()                                   # [Np, H, P, 2]
    dev = chunks.device
    d = codes.float().reshape(*codes.shape[:-1], NG, RB) * mag[..., None]                # [Np, H, P, 2, 4, 32]
    sg = _sign_table(dev)                                                                # [16, 128]
    nrow = sg.shape[0]
    dH = d @ hadamard(RB, dev)
    li = (local - 1).clamp(0, nrow - 1)
    sl = sg.view(-1, NG, RB)[li, torch.arange(NG, device=dev).expand_as(li)]            # [..., 4, 32]
    loc = torch.where((local == 0)[..., None], d, sl * dH).reshape(*codes.shape)
    full = d.reshape(*codes.shape) @ hadamard(D, dev)
    sh = sg[(mode - 1).clamp(0, nrow - 1)]
    x = torch.where((mode == 0)[..., None], loc, sh * full).to(dtype)                    # [Np, H, P, 2, 128]
    return x[..., 0, :].transpose(1, 2), x[..., 1, :].transpose(1, 2)
