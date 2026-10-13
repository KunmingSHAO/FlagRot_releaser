"""kivi2: ctypes front end of libflagrot_kivi.so (csrc/kivi.cu) + PyTorch page parsers.

FlagRot-KIVI INT2 pages (simulator methods kivi_sel / kivi_c, kivi2:32 keys per channel over 32-token groups,
values per token, kv_dyn att+f); layout and semantics: csrc/kivi.cu header.  Entry points (asynchronous, current
stream, CUDA-graph capturable):
  v_write(kv, slots, co, M, pool)                    per-token value encoder (all tokens, at their real slots)
  k_encode(src, sidx, dslot, qs, lz, qscale, pool)   per-(32-token group, KV head) key encoder
  attn(q, pool, bt, cached, ...)                     paged split-K decode attention + BF16 sink / window CTA + merge
  plan(B, max_len, Hkv)                              (nsplit, chunk) of the decode kernel (host only)
  dequant(pool, bt_row, a, n, Hkv, D)                K, V [n - a, Hkv, D] fp32 of positions [a, n) (original basis)
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
_PLUGIN = _HERE.parent
ABI = 9
PAGE, GRP = 16, 32


def _lib_path() -> Path:
    return Path(os.environ.get("FLAGROT_KIVI_LIB", _PLUGIN / "build" / "libflagrot_kivi.so"))


def _sources():
    return [_PLUGIN / "csrc" / "kivi.cu", _PLUGIN / "csrc" / "build_kivi.sh"]


def _stale(lib: Path) -> bool:
    if not lib.exists():
        return True
    if os.environ.get("FLAGROT_NO_REBUILD"):
        return False
    t = lib.stat().st_mtime
    return any(s.exists() and s.stat().st_mtime > t for s in _sources())


_LIB = None


def lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    path = _lib_path()
    if _stale(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path) + ".lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if _stale(path):
                subprocess.run(["bash", str(_PLUGIN / "csrc" / "build_kivi.sh"), str(path.parent)], check=True,
                               stdout=subprocess.DEVNULL)
            fcntl.flock(lk, fcntl.LOCK_UN)
    L = ctypes.CDLL(str(path))
    P, I, LL, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_float
    L.kv_abi_version.restype = I
    if L.kv_abi_version() != ABI:
        raise RuntimeError(f"libflagrot_kivi ABI {L.kv_abi_version()} != {ABI}: rebuild ({path})")
    L.kv_last_error.restype = ctypes.c_char_p
    L.kv_chunk_bytes.argtypes = [I]
    L.kv_v_write.argtypes = [P, LL, I, I, I, P, P, P, P, I, P, I, P]
    L.kv_k_encode.argtypes = [P, LL, P, LL, P, P, I, I, I, P, P, I, F, P, I, P, I, P]
    L.kv_attn.argtypes = [P, LL, P, P, LL, P, I, I, I, I, I, I, I, F, P, P, P, I, P, P, P, LL, I, P]
    L.kv_dequant.argtypes = [P, P, I, I, I, I, I, I, P, P, I, P]
    L.kv_lse_merge.argtypes = [P, P, LL, P, P, LL, P, LL, I, I, I, P]
    for n in ("kv_chunk_bytes", "kv_v_write", "kv_k_encode", "kv_attn", "kv_dequant", "kv_lse_merge"):
        getattr(L, n).restype = I
    _LIB = L
    return L


def _check(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what}: failed (code {rc}, {lib().kv_last_error().decode()})")


def _p(t):
    return None if t is None else ctypes.c_void_p(t.data_ptr())


_DEV: list = []


def _stream():
    if not _DEV:
        _DEV.append(torch.cuda.current_device())
    return ctypes.c_void_p(torch._C._cuda_getCurrentRawStream(_DEV[0]))


def chunk_bytes(D: int, f8: bool = False) -> int:
    """bytes per (16-token page, KV head): 12 D + 16 (kivi2, FP16 scale / min) or 10 D (kivi2p8, FP8 scale / min)"""
    if f8:
        return 10 * D
    return 12 * D + 16


def cell_bytes(D: int, f8: bool = False) -> int:
    if f8:
        return chunk_bytes(D, True) // PAGE
    return chunk_bytes(D) // PAGE


# ------------------------------------------------------------------ writers
def v_write(kv, slots, co, M, pool, Hkv, D, ident=False, fix=None, f8=False):
    """fix: int32 [Hkv] frozen choices (kivi2s): bits 0..D/32-1 = H32 per sub-vector, bit 16 = whole-head H_D"""
    """kv [n, 2 Hkv, D] bf16 (row stride arbitrary, last two dims contiguous), slots [n] int64 (< 0 skipped)."""
    n = slots.shape[0]
    if n == 0:
        return
    assert kv.dtype == torch.bfloat16 and kv.stride(-1) == 1 and kv.stride(-2) == D and slots.dtype == torch.int64
    _check(lib().kv_v_write(_p(kv), kv.stride(0), n, Hkv, D, _p(slots), _p(co), _p(M), _p(pool), int(ident), _p(fix), int(f8), _stream()),
           "kv_v_write")


def k_encode(src, sidx, dslot, qs, lz, qscale, pool, Hkv, D, src2=None, ident=False, fix=None, f8=False):
    """src / src2 [rows, >= Hkv, D] bf16 rows (K row of head h = src[r, h]); sidx [N, 32] int64: r >= 0 src row,
    r <= -2 src2 row -r - 2, -1 not in the group; dslot [N, 32] int64 page slots (dslot[n, 0] < 0: item inactive)."""
    N = sidx.shape[0]
    if N == 0:
        return
    assert src.dtype == torch.bfloat16 and src.stride(-1) == 1 and src.stride(-2) == D
    if src2 is not None:
        assert src2.dtype == torch.bfloat16 and src2.stride(-1) == 1 and src2.stride(-2) == D
    assert sidx.dtype == torch.int64 and dslot.dtype == torch.int64 and sidx.is_contiguous() and dslot.is_contiguous()
    _check(lib().kv_k_encode(_p(src), src.stride(0), _p(src2), src2.stride(0) if src2 is not None else 0, _p(sidx),
                             _p(dslot), N, Hkv, D, _p(qs), _p(lz), qs.shape[1], float(qscale), _p(pool), int(ident),
                             _p(fix), int(f8), _stream()),
           "kv_k_encode")


# ------------------------------------------------------------------ decode attention
class Workspace:
    """split-K partials of up to `parts` (row, KV head, split) CTAs of G q heads x D (allocate before CUDA-graph
    capture: the pointers are baked into the graphs)."""

    def __init__(self, parts, G, D, device):
        self.parts, self.G, self.D = parts, G, D
        self.opart = torch.empty(parts * G * D, dtype=torch.float32, device=device)
        self.ml = torch.empty(parts * G * 2, dtype=torch.float32, device=device)


def fast_occ(D: int, G: int) -> int:
    """resident CTAs per SM of the kivi2p8f decode kernel (its launch bounds, csrc KA3F_MINB_*), for plan()"""
    e = int(os.environ.get("FLAGROT_KIVI_OCC_FAST", "0"))
    return e or (3 if D == 256 else 6)


def plan(B: int, max_len: int, Hkv: int, sms: int | None = None, occ: int | None = None):
    """(nsplit, chunk): chunk a multiple of 32 tokens (whole groups) >= 256, about 3 waves of `occ` (default 4)
    resident CTAs per SM over B Hkv (row, KV head) pairs; fixed per (B, max_len): CUDA-graph safe."""
    if sms is None:
        sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    max_len = max(1, int(max_len))
    occ = occ or int(os.environ.get("FLAGROT_KIVI_OCC", "4"))
    target = int(os.environ.get("FLAGROT_KIVI_WAVES", "3")) * occ * sms
    ns = max(1, math.ceil(target / max(1, B * Hkv)))
    chunk = max(256, math.ceil(max_len / ns / GRP) * GRP)
    return math.ceil(max_len / chunk), chunk


def attn(q, pool, bt, cached, out, nsplit, chunk, ws, qscale, kv_start, G, ring=None, tail_len=None, tail_idx=None,
         T=0, f8=False, fast=False):
    """q [B, Hkv G, D] bf16, out [B, Hkv G D] bf16 (row stride arbitrary); cached [B] int32 = page tokens read
    [kv_start, cached); tail: BF16 rows ring[(R + 1) T, 2 Hkv, D], slots [0, tail_len[b]) of row tail_idx[b]."""
    B, Hq, D = q.shape
    Hkv = Hq // G
    if B == 0:
        return
    assert q.stride(-1) == 1 and q.stride(-2) == D and bt.dtype == torch.int32 and cached.dtype == torch.int32
    assert ws.parts >= B * Hkv * (nsplit + 1) and ws.G >= G and ws.D == D
    _check(lib().kv_attn(_p(q), q.stride(0), _p(pool), _p(bt), bt.stride(0), _p(cached), int(kv_start), B, Hkv, G, D,
                         nsplit, chunk, float(qscale), _p(ring), _p(tail_len), _p(tail_idx), int(T), _p(ws.opart),
                         _p(ws.ml), _p(out), out.stride(0), int(f8) | (2 if fast else 0), _stream()), "kv_attn")


# ------------------------------------------------------------------ PyTorch page parsers
def _hada(n, device):
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(n)).to(device=device, dtype=torch.float32)


def _codes(b, D):
    """uint8 [..., D/4] -> float codes [..., D]"""
    sh = torch.tensor([0, 2, 4, 6], device=b.device, dtype=torch.uint8)
    return ((b[..., None] >> sh) & 3).reshape(*b.shape[:-1], D).float()


_KBIT = [2 * (i >> 1) + 16 * (i & 1) for i in range(16)]


def _kcodes(b, D):
    """key codes uint8 [..., D/4] (16-channel words, channel i of a word at bit kbit(i)) -> float codes [..., D]"""
    w = b.contiguous().view(torch.int32).long() & 0xFFFFFFFF                    # [..., D/16]
    sh = torch.tensor(_KBIT, device=b.device, dtype=torch.long)
    return ((w[..., None] >> sh) & 3).reshape(*b.shape[:-1], D).float()


def parse_chunks(ch, D, f8=False):
    """chunks [np, Hkv, CB] uint8 -> raw fields per (page, KV head, cell): kc, km [.., NV, 2] (scale with its sign,
    min; fp16 or fp32 from FP8), vc, vm [.., NV, 2], vf [np, Hkv, 16] (value whole-head flag)."""
    npg, Hkv, _ = ch.shape
    if f8:
        NV = D // 32
        e4 = lambda b: b.contiguous().view(torch.float8_e4m3fn).float()  # noqa: E731
        kc = ch[..., :4 * D].reshape(npg, Hkv, PAGE, D // 4)
        km = e4(ch[..., 4 * D:5 * D]).reshape(npg, Hkv, PAGE, NV, 2)
        vc = ch[..., 5 * D:9 * D].reshape(npg, Hkv, PAGE, D // 4)
        vm = e4(ch[..., 9 * D:10 * D]).reshape(npg, Hkv, PAGE, NV, 2)
        vf = torch.signbit(km[..., NV - 1, 0]).to(torch.uint8)
        return kc, km, vc, vm, vf
    kc = ch[..., :4 * D].reshape(npg, Hkv, PAGE, D // 4)
    km = ch[..., 4 * D:6 * D].reshape(npg, Hkv, PAGE, D // 8).contiguous().view(torch.float16).reshape(npg, Hkv, PAGE, D // 32, 2)
    vc = ch[..., 6 * D:10 * D].reshape(npg, Hkv, PAGE, D // 4)
    vm = ch[..., 10 * D:12 * D].reshape(npg, Hkv, PAGE, D // 8).contiguous().view(torch.float16).reshape(npg, Hkv, PAGE, D // 32, 2)
    vf = ch[..., 12 * D:12 * D + PAGE]
    return kc, km, vc, vm, vf


def dequant_groups(ch, D, basis="orig", f8=False):
    """chunks of WHOLE groups [2 ng, Hkv, CB] (pages 2 g, 2 g + 1 of each group, in order) -> K, V
    [2 ng 16, Hkv, D] fp32.  basis 'orig': inverse transforms applied (K per group, V per token)."""
    npg, Hkv, _ = ch.shape
    assert npg % 2 == 0
    ng = npg // 2
    kc, km, vc, vm, vf = parse_chunks(ch, D, f8)
    NV = D // 32
    # keys: group meta [ng, Hkv, 32 cells, NV, 2] -> per channel (cell-major)
    kmg = km.reshape(ng, 2, Hkv, PAGE, NV, 2).permute(0, 2, 1, 3, 4, 5).reshape(ng, Hkv, D, 2).float()
    s, mn = kmg[..., 0], kmg[..., 1]
    CHL = NV if f8 else 1                                      # key flag j at channel j CHL, whole head at NV CHL
    lbits = torch.signbit(s[..., 0:NV * CHL:CHL])              # [ng, Hkv, NV]
    fbit = torch.signbit(s[..., NV * CHL])                     # [ng, Hkv]
    s = s.abs()
    codes = _kcodes(kc, D).reshape(ng, 2, Hkv, PAGE, D).permute(0, 2, 1, 3, 4).reshape(ng, Hkv, 32, D)
    k = codes * s[:, :, None] + mn[:, :, None]                 # [ng, Hkv, 32, D]
    if basis == "orig":
        H32, HD = _hada(32, ch.device), _hada(D, ch.device)
        kb = k.reshape(ng, Hkv, 32, NV, 32)
        kh = torch.where((lbits & ~fbit[..., None])[:, :, None, :, None], kb @ H32, kb).reshape(ng, Hkv, 32, D)
        k = torch.where(fbit[:, :, None, None], k @ HD, kh)
    k = k.permute(0, 2, 1, 3).reshape(ng * 32, Hkv, D)
    # values per token
    vcode = _codes(vc, D)                                      # [np, Hkv, 16, D]
    vmf = vm.float()
    vs, vmn = vmf[..., 0], vmf[..., 1]                         # [np, Hkv, 16, NV]
    vl = torch.signbit(vs)
    vs = vs.abs()
    v = (vcode.reshape(npg, Hkv, PAGE, NV, 32) * vs[..., None] + vmn[..., None])
    if basis == "orig":
        H32, HD = _hada(32, ch.device), _hada(D, ch.device)
        vF = vf != 0                                           # [np, Hkv, 16]
        vh = torch.where(vl[..., None], v @ H32, v).reshape(npg, Hkv, PAGE, D)
        v = torch.where(vF[..., None], v.reshape(npg, Hkv, PAGE, D) @ HD, vh)
    v = v.reshape(npg, Hkv, PAGE, D).permute(0, 2, 1, 3).reshape(npg * PAGE, Hkv, D)
    return k, v


def dequant_cuda(pool, bt_row, a, n, Hkv, D, dtype=torch.bfloat16, f8=False, v2=False):
    """K, V [n - a, Hkv, D] (dtype bf16 | fp32) of positions [a, n) in the writers' basis (inverse H32 / H_D
    applied), one kernel launch (csrc/kivi.cu k_dequant); every group touched must be complete in the pages."""
    g0, g1 = a // GRP, (n + GRP - 1) // GRP
    o = a - g0 * GRP
    rows = n - g0 * GRP
    k = torch.empty(rows, Hkv, D, dtype=dtype, device=pool.device)
    v = torch.empty_like(k)
    if rows > 0:
        assert bt_row.dtype == torch.int32 and bt_row.is_contiguous()
        _check(lib().kv_dequant(_p(pool), _p(bt_row), g0, g1, rows, Hkv, D, int(dtype == torch.float32), _p(k), _p(v),
                                int(f8) | (2 if v2 else 0), _stream()), "kv_dequant")
    return k[o:], v[o:]


def lse_merge(oa, la, ob, lb, out):
    """kivi2p8ff: out [n, H, D] bf16 (row stride arbitrary) = the LSE merge of (oa [n, H, D], la [H, n]) and
    (ob [n, H, D], lb [H, n]), bit-identical to backend._merge (one kernel)."""
    n, H, D = oa.shape
    assert oa.is_contiguous() and ob.stride(1) == D and ob.stride(0) == H * D and oa.dtype == ob.dtype == torch.bfloat16
    assert la.dtype == lb.dtype == torch.float32 and la.stride(1) == 1 and lb.stride(1) == 1
    assert out.stride(1) == D and out.stride(2) == 1
    _check(lib().kv_lse_merge(_p(oa), _p(la), la.stride(0), _p(ob), _p(lb), lb.stride(0), _p(out), out.stride(0), n, H, D,
                              _stream()), "kv_lse_merge")


def dequant(pool, bt_row, a, n, Hkv, D, basis="orig", step=128, f8=False):
    """K, V [n - a, Hkv, D] fp32 of positions [a, n) of one request (block-table row bt_row); every group touched
    must be complete in the pages (positions < n are in complete groups: n <= L - W, W >= 31)."""
    g0, g1 = a // GRP, (n + GRP - 1) // GRP
    ks, vs = [], []
    for s0 in range(g0, g1, step):
        s1 = min(g1, s0 + step)
        ch = pool.index_select(0, bt_row[2 * s0:2 * s1].long())
        k, v = dequant_groups(ch, D, basis, f8)
        ks.append(k)
        vs.append(v)
    k = torch.cat(ks) if len(ks) > 1 else ks[0]
    v = torch.cat(vs) if len(vs) > 1 else vs[0]
    o = a - g0 * GRP
    return k[o:o + n - a], v[o:o + n - a]
