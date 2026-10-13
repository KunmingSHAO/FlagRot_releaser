"""Phase 2i: prefix-cached prefill attention straight from the FlagRot INT4 pages (csrc/mq_prefill.cu, sm_90).

For prefill chunks whose new segment is short relative to the cached prefix (multi-turn / agentic serving: a few
hundred new tokens over 10-30k cached ones), the "fa" path's PyTorch gather + dequantization of the whole prefix per
chunk dominates.  This module attends over the paged INT4 cache directly: split-K CTAs of 64 rows (64 / G query
tokens x the G q heads of a KV head) x 64-key blocks, K / V converted to FP16 tiles in shared memory (key-only 2x2:
per-token inverse transforms), mma.sync f16 with fp32 accumulation, partials merged per q tile.  Every prefill
request of a step that takes this path is served by one launch per layer (prep + attention + merge).

ctypes front end of build/libflagrot_mq.so (csrc/build_mq.sh; rebuilt automatically when older than its source,
FLAGROT_NO_REBUILD=1 disables, FLAGROT_MQ_LIB overrides the path).

  prep_q(q, qscale, G, idx, kfull, kmask)     q (bf16 / fp32 rows idx) -> FP16 [m, Hq, D], stored basis, * qscale
  build_plan(reqs, G, Hkv, shift, device)     host work list (tiles x key ranges), one int32 device buffer

Head size D = 128 (kernels 1 / 2 / 3) or 256 (Phase 2j: k_mq3, one producer + two consumer warpgroups with
setmaxnreg, 128 rows per CTA; the head size is taken from q / the pool, every other argument is the same).
  attn(q16, pool, bt, plan, konly)            -> partials (fp32 O, m, l)
  merge(plan, part, out=..., fp32=..., lse=...)  -> bf16 rows of the layer output (or fp32 + natural-log LSE)
"""
from __future__ import annotations

import ctypes
import fcntl
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_PLUGIN = Path(__file__).resolve().parent.parent
D, PAGE, BN = 128, 16, 64
TILE_INTS, ITEM_INTS = 9, 4
LOG2E = 1.4426950408889634
ABI = 7                        # 3: head-size argument (Phase 2j, head 256); 4: kmin (Phase 2k, sinks at real positions);  (7: INT2 pages, Phase 3a)
                               # 5: frmq_win_gather / frmq_bf_merge (Phase 2k BF16 parts); 6: tile ext
# kernel: 2 = k_mq2 (wgmma, producer / 2 consumer warpgroups, 128 rows per CTA, 1 CTA / SM); 1 = k_mq (mma.sync,
# 64 rows per CTA, 3 CTAs / SM).  Same numerics (FP16 operands, fp32 accumulation).
KERNEL = int(os.environ.get("FLAGROT_MQ_KERNEL", "3"))
KROWS = {1: 64, 2: 128, 3: 128}
KCTAS = {1: 3, 2: 1, 3: 1}                                    # resident CTAs per SM
_LIB = None


def _lib_path() -> Path:
    return Path(os.environ.get("FLAGROT_MQ_LIB", _PLUGIN / "build" / "libflagrot_mq.so"))


def _sources():
    c = _PLUGIN / "csrc"
    return [c / "mq_prefill.cu", c / "mq_wgmma.h", c / "build_mq.sh"]


def lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    path = _lib_path()
    stale = (not path.exists()) or (not os.environ.get("FLAGROT_NO_REBUILD")
                                    and any(s.exists() and s.stat().st_mtime > path.stat().st_mtime
                                            for s in _sources()))
    if stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path) + ".lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            subprocess.run(["bash", str(_PLUGIN / "csrc" / "build_mq.sh"), str(path.parent)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            fcntl.flock(lk, fcntl.LOCK_UN)
    L = ctypes.CDLL(str(path))
    P, I, LL, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_float
    L.frmq_abi_version.restype = I
    if L.frmq_abi_version() != ABI:
        raise RuntimeError(f"libflagrot_mq ABI {L.frmq_abi_version()} != {ABI}: rebuild ({path})")
    L.frmq_prep_q.argtypes = [P, I, LL, P, I, I, I, P, P, F, P, I, P]
    L.frmq_attn.argtypes = [P, P, LL, P, LL, P, P, I, I, I, I, I, P, P, I, I, P, I]
    L.frmq_merge.argtypes = [P, I, P, P, I, I, I, P, LL, I, P, LL, I, P]
    L.frmq_win_gather.argtypes = [P, I, P, P, P, LL, P, LL, I, P, P, I, P]
    L.frmq_bf_merge.argtypes = [I, I, I, P, LL, P, P, P, P, P, P, P, P, P, I, I, F, P, LL, I, P]
    for n in ("frmq_prep_q", "frmq_attn", "frmq_merge", "frmq_smem_bytes", "frmq_win_gather", "frmq_bf_merge"):
        getattr(L, n).restype = I
    L.frmq_last_error.restype = ctypes.c_char_p
    _LIB = L
    return L


def available() -> bool:
    try:
        lib()
        return True
    except Exception:  # noqa: BLE001
        return False


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


def _check(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what}: launch failed (code {rc}, {lib().frmq_last_error().decode()})")


_NSM: dict = {}


def num_sms(device) -> int:
    key = str(device)
    if key not in _NSM:
        _NSM[key] = torch.cuda.get_device_properties(device).multi_processor_count
    return _NSM[key]


@dataclass
class Plan:
    """host work list of one step (shared by all layers): tiles [nt, 9] = (block-table row, compact row, output row,
    tokens, position of the first token, page keys, first item, items, ext); items [ni, 4] = (tile, kb0, kb1, 0);
    idx [m] = output rows of the compact rows.  buf = the three on the device (one int32 tensor)."""
    ntiles: int
    nitems: int
    m: int
    buf: torch.Tensor
    reqs: list            # (r, q0, q1, L, s): the requests of the plan, compact rows in this order
    kernel: int = 2
    rows: int = 128       # q rows per tile (= per CTA)
    D: int = 128          # head size (256: k_mq3)
    kmin: int = 0         # keys [0, kmin) masked (the sink tokens, attended in BF16 outside the kernel)

    @property
    def tiles(self):
        return self.buf[:self.ntiles * TILE_INTS]

    @property
    def items(self):
        o = self.ntiles * TILE_INTS
        return self.buf[o:o + self.nitems * ITEM_INTS]

    @property
    def idx(self):
        o = self.ntiles * TILE_INTS + self.nitems * ITEM_INTS
        return self.buf[o:o + self.m]


WAVES = float(os.environ.get("FLAGROT_MQ_WAVES", "2"))        # grid target: WAVES x the resident CTAs of the GPU
MIN_BLOCKS = int(os.environ.get("FLAGROT_MQ_MIN_BLOCKS", "4"))  # >= 4 x 64 keys per work item


NO_EXT = -(1 << 30)


def build_plan(reqs, G: int, Hkv: int, shift: int, device, sink: int = 0, kernel: int | None = None,
               waves: float | None = None, min_blocks: int | None = None, D: int = D, ext: dict | None = None) -> Plan:
    """reqs: list of (block-table row r, q0, q1, L, s) (rows q0 .. q1 - 1 of the batch = positions s .. L - 1).
    Pages hold every position p at its real slot (Phase 2k); the query at position i sees the page keys
    sink <= j <= i - shift (sinks and the BF16 window are attended outside the kernel); ext {r: e}: the rows of
    request r also see the page keys sink <= j <= e (Phase 2k: window positions before a reused prefix's first
    computed token, which have no BF16 copy).  Key ranges are 64-key blocks; the split size is chosen so that the grid has about
    ctas_per_sm x SMs CTAs (at least min_blocks blocks per item; 3 CTAs of the kernel are resident per SM)."""
    kernel = KERNEL if kernel is None else kernel
    if D != 128:                     # head 256: k_mq3 only (128 rows per CTA, 1 CTA / SM, as kernel 3)
        assert D == 256, D
        kernel = 3
    waves = WAVES if waves is None else waves
    min_blocks = MIN_BLOCKS if min_blocks is None else min_blocks
    rows = KROWS[kernel]
    TQ = rows // G
    tiles, kends, idx = [], [], []
    m = 0
    for (r, q0, q1, L, s) in reqs:
        nq = q1 - q0
        e = ext.get(r, NO_EXT) if ext else NO_EXT
        nk = max(L - shift, e + 1, 0)
        for t0 in range(0, nq, TQ):
            nt = min(TQ, nq - t0)
            pos0 = s + t0
            tiles.append([r, m + t0, q0 + t0, nt, pos0, nk, 0, 0, e])
            kends.append(max(0, min(nk, max(pos0 + nt - shift, e + 1))))
        idx.append(np.arange(q0, q1, dtype=np.int32))
        m += nq
    nb = (np.asarray(kends, dtype=np.int64) + BN - 1) // BN
    target = int(waves * KCTAS[kernel] * num_sms(device))
    bpi = max(min_blocks, -(-int(nb.sum()) * Hkv // max(target, 1)))
    nitem = np.maximum(1, -(-nb // bpi))
    item0 = np.concatenate([[0], np.cumsum(nitem)[:-1]])
    T = np.asarray(tiles, dtype=np.int32).reshape(-1, TILE_INTS)
    T[:, 6] = item0
    T[:, 7] = nitem
    ti = np.repeat(np.arange(len(tiles), dtype=np.int32), nitem)
    sp = np.arange(int(nitem.sum()), dtype=np.int64) - np.repeat(item0, nitem)
    kb0 = (sp * bpi).astype(np.int32)
    kb1 = np.minimum(np.repeat(nb, nitem), kb0 + bpi).astype(np.int32)
    I_ = np.stack([ti, kb0, kb1, np.zeros_like(ti)], 1).astype(np.int32)
    host = np.concatenate([T.reshape(-1), I_.reshape(-1), np.concatenate(idx) if idx else np.zeros(0, np.int32)])
    buf = torch.from_numpy(host).to(device, non_blocking=False)
    return Plan(ntiles=T.shape[0], nitems=I_.shape[0], m=m, buf=buf, reqs=list(reqs), kernel=kernel, rows=rows, D=D,
                kmin=int(sink))


def prep_q(q: torch.Tensor, qscale: float, G: int, idx: torch.Tensor | None = None, kfull=None, kmask=None):
    """q [n, Hq, D] bf16 / fp32 (head rows contiguous, D = 128 | 256) -> FP16 [m, Hq, D] (m = len(idx) rows idx of q,
    or n): rotated into the keys' stored basis (static: whole-head H_D / (D / 16) on kfull[kv head], then orthonormal
    H32 on the kmask[kv head] groups) and multiplied by qscale (= softmax scale * log2 e)."""
    n, Hq, D = q.shape
    assert D in (128, 256)
    assert q.dtype in (torch.bfloat16, torch.float32) and q.stride(2) == 1 and q.stride(1) == D
    assert idx is None or idx.dtype == torch.int32
    m = idx.numel() if idx is not None else n
    q16 = torch.empty(m, Hq, D, dtype=torch.float16, device=q.device)
    _check(lib().frmq_prep_q(q.data_ptr(), int(q.dtype == torch.float32), q.stride(0), _ptr(idx), m, Hq, G,
                             _ptr(kfull), _ptr(kmask), float(qscale), q16.data_ptr(), D, _stream()), "frmq_prep_q")
    return q16


def attn(q16, pool, bt, plan: Plan, G: int, shift: int, konly: bool, int2: bool = False):
    """partial attention of every (item, KV head): returns (opart [ni, Hkv, rows, D] fp32, ml [ni, Hkv, rows, 2]).
    int2 (Phase 3a): INT2 pages (static int2a / konlya2, 80 B per token and 128-d half)."""
    Hkv = pool.shape[1]
    D = q16.shape[2]
    assert D == plan.D, (D, plan.D)
    assert pool.dtype == torch.uint8 and pool.stride(1) == (80 if int2 else 144) * PAGE * D // 128 and pool.stride(2) == 1
    assert bt.dtype == torch.int32 and bt.stride(1) == 1
    assert q16.dtype == torch.float16 and q16.is_contiguous() and q16.shape[1] == Hkv * G
    opart = torch.empty(plan.nitems, Hkv, plan.rows, D, dtype=torch.float32, device=q16.device)
    ml = torch.empty(plan.nitems, Hkv, plan.rows, 2, dtype=torch.float32, device=q16.device)
    _check(lib().frmq_attn(q16.data_ptr(), pool.data_ptr(), pool.stride(0), bt.data_ptr(), bt.stride(0),
                           plan.tiles.data_ptr(), plan.items.data_ptr(), plan.nitems, Hkv, G, shift,
                           int(konly) | (2 if int2 else 0),
                           opart.data_ptr(), ml.data_ptr(), plan.kernel, D, _stream(), plan.kmin), "frmq_attn")
    return opart, ml


def merge(plan: Plan, opart, ml, G: int, out=None, fp32: bool = False, want_lse: bool = False):
    """combine the items of each tile.  fp32=False: bf16 into `out` (2-D [n, Hq * 128] view of the layer output, rows
    = output rows of the plan); fp32=True: returns o [m, Hq, 128] fp32 in compact rows.  want_lse: also the natural-log
    LSE [Hq, m] (-inf: no visible key).  Returns (o or out, lse or None)."""
    Hkv, D = opart.shape[1], opart.shape[3]
    Hq = Hkv * G
    dev = opart.device
    lse = torch.empty(Hq, plan.m, dtype=torch.float32, device=dev) if want_lse else None
    if fp32:
        o = torch.empty(plan.m, Hq, D, dtype=torch.float32, device=dev)
        ots = Hq * D
    else:
        assert out is not None and out.dtype == torch.bfloat16 and out.stride(-1) == 1
        o = out
        ots = out.stride(0)
    _check(lib().frmq_merge(plan.tiles.data_ptr(), plan.ntiles, opart.data_ptr(), ml.data_ptr(), Hkv, G, plan.rows,
                            o.data_ptr(), ots, int(fp32), _ptr(lse), plan.m if lse is not None else 0, D, _stream()),
           "frmq_merge")
    return o, lse


def ref_partial_lse_merge(parts):
    """LSE merge of partial attentions [(o [n, H, D], lse [H, n] natural log, -inf / +inf = empty)], at least one
    part non-empty per row; fp32 result."""
    ls = [torch.where(torch.isposinf(lse), torch.full_like(lse, -math.inf), lse).transpose(0, 1)[..., None].float()
          for _, lse in parts]
    M = torch.stack(ls).amax(0)
    acc = None
    den = None
    for (o, _), l_ in zip(parts, ls):
        w = torch.exp(l_ - M)
        acc = o.float() * w if acc is None else acc + o.float() * w
        den = w if den is None else den + w
    return acc / den


# ---------------------------------------------------------------- Phase 2k: BF16 window / sink parts (batched)
def win_gather(map_, nk, sinkb, preb, kch, vch, Hkv, D):
    """K / V [nk, Hkv, D] bf16 (contiguous) of the window FlashAttention call: row d from map_[d] = (kind << 32) | row,
    kind 0 = sinkb [*, 2 Hkv, D], 1 = preb [*, 2 Hkv, D], 2 = the batch rows kch / vch ([n, Hkv * D] views, any row
    stride)."""
    dev = kch.device
    kw = torch.empty(nk, Hkv, D, dtype=torch.bfloat16, device=dev)
    vw = torch.empty_like(kw)
    for t in (sinkb, preb):
        assert t is None or (t.is_contiguous() and t.dtype == torch.bfloat16)
    assert kch.stride(-1) == 1 and vch.stride(-1) == 1 and kch.dtype == torch.bfloat16
    _check(lib().frmq_win_gather(map_.data_ptr(), nk, _ptr(sinkb), _ptr(preb), kch.data_ptr(), kch.stride(0),
                                 vch.data_ptr(), vch.stride(0), Hkv, kw.data_ptr(), vw.data_ptr(), D, _stream()),
           "frmq_win_gather")
    return kw, vw


def bf_merge(q, idx, o_p, lse_p, o_w, lse_w, sk, row_req, row_pos, sk_len, S, W, scale, out, Hkv):
    """out[idx[c]] = LSE merge of the page part (o_p fp32 [m, Hq, D], lse_p [Hq, m]), the window part (o_w bf16
    [m, Hq, D], lse_w [Hq, m]; None when W = 0) and the sink part (sk bf16 [nreq, S, 2 Hkv, D] original basis, keys
    j < min(S, len) with j <= pos - W; None when S = 0) of compact row c; q = the batch's q rows [n, Hq, D]."""
    m, Hq, D = o_p.shape
    assert o_p.is_contiguous() and lse_p.is_contiguous() and idx.dtype == torch.int64
    assert q.stride(-1) == 1 and q.stride(-2) == D and out.stride(-1) == 1
    _check(lib().frmq_bf_merge(m, Hq, Hkv, q.data_ptr(), q.stride(0), idx.data_ptr(), o_p.data_ptr(), lse_p.data_ptr(),
                               _ptr(o_w), _ptr(lse_w), _ptr(sk), _ptr(row_req), _ptr(row_pos), _ptr(sk_len), S, W,
                               float(scale), out.data_ptr(), out.stride(0), D, _stream()), "frmq_bf_merge")
