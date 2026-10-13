"""v2 paged decode attention over 4-bit KV caches (NVFP4 / MXFP4 / INT4) on H20: JIT loader, paged cache, writer,
reference dequantizer and Python API.  Kernel: csrc/attn_decode_v2.cu (design notes at its top).

    cache = PagedKVCache(fmt, num_pages, page=16, Hkv=8)
    cache.write(kv, slots, mask, rb)             kv [T, 2*Hkv, 128] bf16 -> slots [T] (int64 page*P+offset, -1 = skip),
                                                 FlagRot + quantization bit-identical to kv_selrot_quant (vLLM
                                                 reshape_and_cache semantics)
    o = decode(q, cache, block_table, seq_lens, kmask=, vmask=, rb=, qrot="fused"|None, vrot=False)
                                                 q [B, 32, 128] -> [B, 4096];  block_table int32 [B, max_pages],
                                                 seq_lens int32 [B] (variable lengths)
Contiguous caches are the special case of an identity block table (make_bench_cache).
Page layout (one contiguous chunk per (page, kv head): K codes, V^T codes, K scales, V scales) in attn_decode_v2.cuh.
"""
from __future__ import annotations

import functools
import math
import os

import torch

import ref as R

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT = None
# int4a: asymmetric INT4 (codes 0..15, per-(token, group) zero point in the low 4 bits of the fp16 scale word;
# blockflag int4ah:32 = int4a:32 with the range widened to include 0),
FMTS = ("nvfp4", "mxfp4", "int4", "int4a")
FMT_ID = {"nvfp4": 2, "mxfp4": 3, "int4": 4, "int4a": 5, "konly": 6, "konlya": 7}  # konly / konlya: the key-only 2x2
SB = {"nvfp4": 8, "mxfp4": 4, "int4": 8, "int4a": 8, "konly": 8, "konlya": 8}      # codebook caches of attn_codebook.py,
NG = {"nvfp4": 8, "mxfp4": 4, "int4": 4, "int4a": 4, "konly": 4, "konlya": 4}      # read by the generalized kernel
VARIANTS = {0: "FP8 PV (P' = p * V scale, hi/lo e4m3; MXFP4/INT4), bulk-copy loader",
            1: "f16 PV (per-element V scale), bulk-copy loader",
            2: "FP8 PV, cp.async loader", 3: "f16 PV, cp.async loader"}
DEFAULT_VARIANT = {"nvfp4": 1, "mxfp4": 0, "int4": 264, "int4a": 264}  # 264 = FP8 PV, conflict-free P', deferred PV wait; + SK_BIT = stream-K (use_sk)
DEFAULT_STAGES = 4
LOG2E = 1.4426950408889634
# generalized static kernel (csrc/attn_decode_v2g.cuh): variant 264 + GEN (+ GT8: q tiles of 8 heads,
# + HD256: head dim 256); required for GQA groups != 4 and head dim 256 (INT4 / INT4A only)
GEN, GT8, HD256 = 1024, 2048, 4096


def qtile(gqa, D=128):
    """q heads per CTA tile of the generalized kernel: 8 for groups 5..8 at D = 128 (one tile, wider wgmma N), else 4
    (groups of 12: 3 tiles of 4, no padding; D = 256: register budget)."""
    return 8 if D == 128 and 5 <= gqa <= 8 else 4


def gen_variant(gqa, D=128, gt=None):
    """variant id of the generalized kernel for a GQA group / head dim (gt: q tile, default qtile())."""
    gt = qtile(gqa, D) if gt is None else gt
    return 264 | GEN | (GT8 if gt == 8 else 0) | (HD256 if D == 256 else 0)


def tile_of(variant):
    return 8 if (variant & GEN) and (variant & GT8) else 4
NUM_SMS = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        os.environ["PATH"] = os.path.join(_HERE, ".deps", "bin") + os.pathsep + os.environ.get("PATH", "")
        # ptxas of CUDA 12.4 serializes every wgmma of this kernel (C7511-C7515: it cannot keep the async wgmma
        # pipeline open across the dequant / softmax code); the ptxas of CUDA 12.9 pipelines them.  The extension is
        # therefore built with a CUDA 12.9 toolkit: ATTN2_CUDA_HOME, else FLAGROT_CUDA_HOME, else /usr/local/cuda-12.9.
        cuda_home = os.environ.get("ATTN2_CUDA_HOME", os.environ.get("FLAGROT_CUDA_HOME", "/usr/local/cuda-12.9"))
        os.environ["CUDA_HOME"] = cuda_home
        import torch.utils.cpp_extension as CE
        CE.CUDA_HOME = cuda_home
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
        os.environ.setdefault("MAX_JOBS", "8")
        extra = os.environ.get("ATTN2_EXTRA_FLAGS", "")  # e.g. "-DATTN2_DBG_NOPV" (timing experiments, separate build)
        tag = "" if not extra else "_" + "".join(c for c in extra if c.isalnum())[:60]
        bdir = os.path.join(_HERE, ".build_v2" + ("" if cuda_home.endswith("12.9") else "_" + os.path.basename(cuda_home)) + tag)
        os.makedirs(bdir, exist_ok=True)
        _EXT = load(
            name="attn_decode_v2_ext" + tag,
            sources=[os.path.join(_HERE, "csrc", "attn_v2_bind.cpp"), os.path.join(_HERE, "csrc", "attn_decode_v2.cu")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-gencode", "arch=compute_90a,code=sm_90a", "-std=c++17",
                               "--expt-relaxed-constexpr", "-lineinfo"] + extra.split(),
            build_directory=bdir,
            verbose=bool(int(os.environ.get("SELROT_VERBOSE", "0"))),
        )
    return _EXT


def chunk_bytes(fmt, page):
    return page * (128 + 2 * SB[fmt])


def kv_bytes_per_token(fmt, Hkv=8, D=128):
    return 2 * Hkv * (64 + SB[fmt]) * D // 128


class PagedKVCache:
    """pool uint8 [num_pages, Hkv, chunk]: per (page, kv head) K codes | V^T codes | K scales | V scales."""

    def __init__(self, fmt, num_pages, page=16, Hkv=8, device="cuda", D=128):
        assert fmt in FMTS and page in (16, 32) and D in (128, 256)
        assert D == 128 or fmt in ("int4", "int4a"), "head dim 256: INT4 formats"
        self.fmt, self.num_pages, self.page, self.Hkv, self.D = fmt, num_pages, page, Hkv, D
        # D = 256: per (page, kv head) two 128-d half chunks of the v2 layout
        self.pool = torch.zeros(num_pages, Hkv, (D // 128) * chunk_bytes(fmt, page), device=device, dtype=torch.uint8)
        self.gs = torch.tensor([448.0 * 6.0 / 8.0], device=device)  # nvfp4 global scale (static calibration)

    def write(self, kv, slots, mask, rb):
        """kv [T, 2*Hkv, D] bf16 (row stride arbitrary) -> slots [T] int64 (page * P + offset; < 0 skipped)."""
        if kv.stride(-1) != 1 or kv.stride(-2) != kv.shape[-1]:
            kv = kv.contiguous()
        m = mask.reshape(-1).to(torch.uint8).contiguous()
        ext().paged_write(kv, slots.to(torch.int64).contiguous(), m, rb, FMT_ID[self.fmt],
                          self.gs if self.fmt == "nvfp4" else None, self.pool, self.page)

    # ------------------------------------------------------------------ reference dequantizer (layout parser)
    def _halves(self):
        """D = 256: the same pool as a cache of 2 Hkv half heads of 128 d (head h = half heads 2 h, 2 h + 1)"""
        c = PagedKVCache.__new__(PagedKVCache)
        c.fmt, c.num_pages, c.page, c.Hkv, c.D, c.gs = self.fmt, self.num_pages, self.page, 2 * self.Hkv, 128, self.gs
        c.pool = self.pool.view(self.num_pages, 2 * self.Hkv, -1)
        return c

    def parse_int4a(self):
        """int4a: codes long [Np, P, Hkv, 2 (K/V), D] (0..15) and scale words long [Np, P, Hkv, 2, D / 32] (bits 3..0 =
        zero point) parsed from the pages (independent of dequant_pages)."""
        if self.D == 256:
            c, w = self._halves().parse_int4a()                       # [Np, P, 2 Hkv, 2, 128] / [.., 4]
            Np, P = c.shape[:2]
            c = c.view(Np, P, self.Hkv, 2, 2, 128).permute(0, 1, 2, 4, 3, 5).reshape(Np, P, self.Hkv, 2, 256)
            w = w.view(Np, P, self.Hkv, 2, 2, 4).permute(0, 1, 2, 4, 3, 5).reshape(Np, P, self.Hkv, 2, 8)
            return c, w
        assert self.fmt == "int4a"
        P, H = self.page, self.Hkv
        pool = self.pool.long()
        Np = pool.shape[0]
        dev = pool.device
        kc = pool[:, :, :64 * P].reshape(Np, H, P, 4, 4, 4)                # [.., P, tg, j, i]
        K = torch.empty(Np, H, P, 128, dtype=torch.long, device=dev)
        tg = torch.arange(4, device=dev).view(4, 1, 1)
        j = torch.arange(4, device=dev).view(1, 4, 1)
        i = torch.arange(4, device=dev).view(1, 1, 4)
        K[..., (32 * j + 4 * tg + i).reshape(-1)] = (kc & 15).reshape(Np, H, P, 64)
        K[..., (32 * j + 16 + 4 * tg + i).reshape(-1)] = (kc >> 4).reshape(Np, H, P, 64)
        vt = pool[:, :, 64 * P:128 * P].reshape(Np, H, 4, 8, P // 8, 4, 4)
        V = torch.empty(Np, H, P, 128, dtype=torch.long, device=dev)
        w = torch.arange(4, device=dev).view(4, 1, 1, 1, 1)
        gid = torch.arange(8, device=dev).view(1, 8, 1, 1, 1)
        c8 = torch.arange(P // 8, device=dev).view(1, 1, P // 8, 1, 1)
        sub = torch.arange(4, device=dev).view(1, 1, 1, 4, 1)
        ii = torch.arange(4, device=dev).view(1, 1, 1, 1, 4)
        shp = (4, 8, P // 8, 4, 4)
        dd = (64 * (sub >> 1) + 16 * w + gid + 8 * (sub & 1)).expand(shp).reshape(-1)
        tA = (8 * c8 + 2 * ii).expand(shp).reshape(-1)
        V[:, :, tA, dd] = (vt & 15).reshape(Np, H, -1)
        V[:, :, tA + 1, dd] = (vt >> 4).reshape(Np, H, -1)
        kw = self.pool[:, :, 128 * P:136 * P].reshape(Np, H, P, 8).view(torch.int16).long() & 0xFFFF       # [Np,H,P,4]
        vw = self.pool[:, :, 136 * P:144 * P].reshape(Np, H, 4, 2 * P).view(torch.int16).long() & 0xFFFF  # [Np,H,4,P perm]
        perm = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=dev)
        tok = (8 * torch.arange(P // 8, device=dev).view(-1, 1) + perm.view(1, -1)).reshape(-1)
        vwt = torch.empty(Np, H, 4, P, dtype=torch.long, device=dev)
        vwt[..., tok] = vw
        codes = torch.stack([K, V], 3).permute(0, 2, 1, 3, 4)
        words = torch.stack([kw, vwt.permute(0, 1, 3, 2)], 3).permute(0, 2, 1, 3, 4)
        return codes, words

    def dequant_pages(self):
        """fp32 K, V [num_pages, P, Hkv, D] (rotated basis, exactly the stored values)."""
        if self.D == 256:
            K, V = self._halves().dequant_pages()                      # [Np, P, 2 Hkv, 128]
            Np, P = K.shape[:2]
            return K.reshape(Np, P, self.Hkv, 256), V.reshape(Np, P, self.Hkv, 256)
        f, P, H = self.fmt, self.page, self.Hkv
        pool = self.pool.long()
        Np = pool.shape[0]
        sb = SB[f]
        kc = pool[:, :, :64 * P].reshape(Np, H, P, 4, 4, 4)            # [.., P, tg, j, i] (byte 16 tg + 4 j + i)
        vt = pool[:, :, 64 * P:128 * P].reshape(Np, H, 4, 8, P // 8, 4, 4)  # [.., w, gid, c8, sub, i]
        ks = pool[:, :, 128 * P:128 * P + sb * P].reshape(Np, H, P, sb)
        vs = pool[:, :, 128 * P + sb * P:].reshape(Np, H, NG[f], P * sb // NG[f])

        def codes_of(word_bytes):  # [..., 4 bytes] -> (codeA [..., 4], codeB [..., 4]) as signed values (e2m1 code / int)
            byt = word_bytes
            if f == "int4a":                                           # unsigned nibbles: A low, B high
                return byt & 15, (byt >> 4) & 15
            if f == "int4":
                sA, mA = (byt >> 7) & 1, byt & 7
                sB = (byt >> 3) & 1
                mB = (torch.roll(byt, shifts=1, dims=-1) >> 4) & 7     # byte (i+3) % 4 = byte i-1
                return (sA << 3) | mA, (sB << 3) | mB
            sA, mA = (byt >> 7) & 1, (byt >> 2) & 7
            sB, mB = (byt >> 5) & 1, (((byt >> 6) & 1) << 2) | (byt & 3)
            return (sA << 3) | mA, (sB << 3) | mB

        def val(code):
            if f == "int4a":
                return code.float()
            if f == "int4":
                v = (code & 7).float()
                return torch.where((code & 8) != 0, -v, v)
            return R.e2m1_value(code.to(torch.uint8))

        # K: [Np, H, P, tg, j, i] -> d_A = 32 j + 4 tg + i, d_B = 32 j + 16 + 4 tg + i
        cA, cB = codes_of(kc)
        K = torch.empty(Np, H, P, 128, device=pool.device)
        tg = torch.arange(4, device=pool.device).view(4, 1, 1)
        j = torch.arange(4, device=pool.device).view(1, 4, 1)
        i = torch.arange(4, device=pool.device).view(1, 1, 4)
        dA = (32 * j + 4 * tg + i).reshape(-1)
        dB = (32 * j + 16 + 4 * tg + i).reshape(-1)
        K[..., dA] = val(cA).reshape(Np, H, P, 64)
        K[..., dB] = val(cB).reshape(Np, H, P, 64)
        # V^T: [Np, H, w, gid, c8, sub, i] -> d = 64 (sub >> 1) + 16 w + gid + 8 (sub & 1); token 8 c8 + 2 i (+1 for B)
        vA, vB = codes_of(vt)
        V = torch.empty(Np, H, P, 128, device=pool.device)
        w = torch.arange(4, device=pool.device).view(4, 1, 1, 1, 1)
        gid = torch.arange(8, device=pool.device).view(1, 8, 1, 1, 1)
        c8 = torch.arange(P // 8, device=pool.device).view(1, 1, P // 8, 1, 1)
        sub = torch.arange(4, device=pool.device).view(1, 1, 1, 4, 1)
        ii = torch.arange(4, device=pool.device).view(1, 1, 1, 1, 4)
        shp = (4, 8, P // 8, 4, 4)
        dd = (64 * (sub >> 1) + 16 * w + gid + 8 * (sub & 1)).expand(shp).reshape(-1)
        tA = (8 * c8 + 2 * ii).expand(shp).reshape(-1)
        V[:, :, tA, dd] = val(vA).reshape(Np, H, -1)
        V[:, :, tA + 1, dd] = val(vB).reshape(Np, H, -1)
        # scales -> per element
        if f == "nvfp4":
            skv = ks.to(torch.uint8).view(torch.float8_e4m3fn).float() / self.gs        # [Np, H, P, 8]
            svv = vs.to(torch.uint8).view(torch.float8_e4m3fn).float() / self.gs        # [Np, H, 8, P perm]
        elif f == "mxfp4":
            skv = torch.ldexp(torch.ones_like(ks, dtype=torch.float32), ks - 127)
            svv = torch.ldexp(torch.ones_like(vs, dtype=torch.float32), vs - 127)
        else:
            skv = ks.to(torch.uint8).reshape(Np, H, P, 8).view(torch.float16).float()   # [Np, H, P, 4]
            svv = vs.to(torch.uint8).reshape(Np, H, 4, 2 * P).view(torch.float16).float()
            if f == "int4a":                                           # zero points: low 4 bits of the scale words
                zkv = (ks.to(torch.uint8).reshape(Np, H, P, 8).view(torch.int16).long() & 15).float()
                zvv = (vs.to(torch.uint8).reshape(Np, H, 4, 2 * P).view(torch.int16).long() & 15).float()
        # V scale token order inside each 8-chunk: [A0..A3 B0..B3] = tokens 0 2 4 6 1 3 5 7
        perm = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=pool.device)
        tok = (8 * torch.arange(P // 8, device=pool.device).view(-1, 1) + perm.view(1, -1)).reshape(-1)
        svt = torch.empty(Np, H, NG[f], P, device=pool.device)
        svt[..., tok] = svv
        gsz = 128 // NG[f]
        if f == "int4a":                                               # (c - z) * s
            zvt = torch.empty(Np, H, NG[f], P, device=pool.device)
            zvt[..., tok] = zvv
            K = K - zkv.repeat_interleave(gsz, dim=-1)
            V = V - zvt.permute(0, 1, 3, 2).repeat_interleave(gsz, dim=-1)
        K = K * skv.repeat_interleave(gsz, dim=-1)
        V = V * svt.permute(0, 1, 3, 2).repeat_interleave(gsz, dim=-1)
        return K.permute(0, 2, 1, 3), V.permute(0, 2, 1, 3)

    def dequant(self, block_table, seq_lens):
        """fp32 K, V [B, Lmax, Hkv, D] gathered through the block table (positions >= seq_len are zero)."""
        Kp, Vp = self.dequant_pages()
        B = block_table.shape[0]
        Lmax = int(seq_lens.max().item())
        npg = (Lmax + self.page - 1) // self.page
        bt = block_table[:, :npg].long()
        K = Kp[bt].reshape(B, npg * self.page, self.Hkv, self.D)[:, :Lmax]
        V = Vp[bt].reshape(B, npg * self.page, self.Hkv, self.D)[:, :Lmax]
        valid = torch.arange(Lmax, device=K.device).view(1, -1) < seq_lens.view(-1, 1).long()
        return K * valid[..., None, None], V * valid[..., None, None]


def alloc_block_table(seq_lens, page, num_pages=None, shuffle=True, gen=None, device="cuda"):
    """vLLM-style allocation: each sequence gets ceil(len / page) pages from a (shuffled) free list.
    Returns block_table int32 [B, max_pages] (unused entries = 0) and the number of pages used."""
    lens = [int(x) for x in seq_lens]
    need = [(n + page - 1) // page for n in lens]
    tot = sum(need)
    num_pages = tot if num_pages is None else num_pages
    ids = torch.randperm(num_pages, generator=gen)[:tot] if shuffle else torch.arange(tot)
    mp = max(need) if need else 1
    bt = torch.zeros(len(lens), max(mp, 1), dtype=torch.int32)
    o = 0
    for b, n in enumerate(need):
        bt[b, :n] = ids[o:o + n].to(torch.int32)
        o += n
    return bt.to(device), num_pages


def slots_for(block_table, b, positions, page):
    """slot ids (int64) of the token positions of sequence b."""
    pos = torch.as_tensor(positions, device=block_table.device).long()
    return block_table[b].long()[pos // page] * page + pos % page


class Workspace:
    """split-K partials: B x Hkv (generalized kernel: virtual heads = q tiles) x max_splits slots of [GT][D] floats"""
    def __init__(self, B, Hkv, max_splits, device="cuda", GT=4, D=128):
        self.opart = torch.empty(B * Hkv * max_splits * GT * D, device=device, dtype=torch.float32)
        self.mlpart = torch.empty(B * Hkv * max_splits * GT * 2, device=device, dtype=torch.float32)
        self.counter = torch.zeros(B * Hkv, device=device, dtype=torch.int32)
        self.max_splits = max_splits
        self.GT, self.D = GT, D


PLAN_C0, PLAN_C1 = 1024, 4  # C0: per-CTA overhead in token-equivalents (split sweeps: ~4 us per CTA)

# INT4 split planner: a time model fitted to split sweeps over B 1..16 x L 4k..128k (sk_sweep_v2.py,
# results/sk_sweep_v2_smallb{,_wave}.json): CTAs run in waves over the occ x SMs resident slots; an SM with k resident
# CTAs moves PLAN_R[k] 64-token blocks per us (one CTA is latency-bound; 2..4 saturate the SM), every wave costs
# PLAN_CFIX us more (prologue, first loads, partial store), and the merge PLAN_MERGE us per 16 partials.  Candidates:
# the sweep's split counts PLAN_SPLITS and the counts that fill m = 1..6 waves of slots exactly; the model is flat
# between candidates and was validated on exactly these (mean regret vs the best measured split 0.4 %, worst 3.8 %).
PLAN_R = {1: 1.6, 2: 3.9, 3: 3.9, 4: 3.9}
PLAN_CFIX, PLAN_MERGE = 4.0, 2.0
PLAN_SPLITS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 39, 48, 64, 78)


def plan_cost_int4(B, nse, chunk, Hkv=8, occ=4, sms=78):
    """modelled time (us) of B x Hkv x nse CTAs of chunk tokens each """
    full, rem = divmod(B * Hkv * nse, occ * sms)
    blk = chunk / 64
    t = full * (PLAN_CFIX + blk * occ / PLAN_R[min(occ, 4)])  # (occupancy 4: the measured case)
    if rem:
        k = min(4, math.ceil(rem / sms))
        t += PLAN_CFIX + blk * k / PLAN_R[k]
    return t + PLAN_MERGE * math.ceil(nse / 16)


@functools.lru_cache(maxsize=4096)
def _plan_int4(B, Lmax, Hkv, occ, sms, gran, maxc):
    best = None
    waves = {max(1, (m * occ * sms) // (B * Hkv)) for m in range(1, 7)}  # split counts filling m waves of slots
    for ns in sorted(set(PLAN_SPLITS) | waves | {max(1, math.ceil(Lmax / maxc))}):
        chunk = math.ceil(Lmax / ns / gran) * gran
        if chunk > maxc or ns > math.ceil(Lmax / gran):
            continue
        nse = math.ceil(Lmax / chunk)
        c = plan_cost_int4(B, nse, chunk, Hkv, occ, sms)
        if best is None or c < best[0] - 1e-9:
            best = (c, nse, chunk)
    return best[1], best[2]


def plan(B, Lmax, fmt, Hkv=8, variant=None, stages=None, page=16, nsplit=None, c0=None, c1=None):
    """(variant, stages, nsplit, chunk): INT4 / INT4A with B Hkv < resident-CTA slots: the time model _plan_int4;
    otherwise (other formats, larger batches, c0 / c1 given): the wave-quantization cost model of v1
    (attn_decode.plan) over resident-CTA slots; chunks are whole 64-token blocks."""
    global NUM_SMS
    if NUM_SMS is None:
        NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    v = DEFAULT_VARIANT[fmt] if variant is None else variant
    st = DEFAULT_STAGES if stages is None else stages
    gran = max(64, page)
    MAXC = 8192  # MAX_CHUNK of csrc/attn_decode_v2.cu (page ids of a CTA staged in shared memory)
    if nsplit is not None:
        chunk = min(MAXC, math.ceil(Lmax / nsplit / gran) * gran)
        return v, st, math.ceil(Lmax / chunk), chunk
    occ = max(1, ext().occupancy(FMT_ID[fmt], v, st, page))
    if fmt in ("int4", "int4a") and c0 is None and c1 is None and B * Hkv < occ * NUM_SMS:
        # small batch (fewer (sequence, kv head) pairs than resident-CTA slots): the time model; larger batches
        # keep the token-equivalent model below (tuned on B 32 / 64, within 2 % of the best split count)
        nse, chunk = _plan_int4(B, Lmax, Hkv, occ, NUM_SMS, gran, MAXC)
        return v, st, nse, chunk
    slots = occ * NUM_SMS
    C0 = PLAN_C0 if c0 is None else c0
    C1 = PLAN_C1 if c1 is None else c1
    best = None
    for ns in range(max(1, math.ceil(Lmax / MAXC)), min(512, math.ceil(Lmax / gran)) + 1):
        chunk = math.ceil(Lmax / ns / gran) * gran
        if chunk > MAXC:
            continue
        nse = math.ceil(Lmax / chunk)
        if best is not None and nse == best[1]:
            continue
        cost = math.ceil(B * Hkv * nse / slots) * (chunk + C0) + (C1 * nse if nse > 1 else 0)
        if best is None or cost < best[0] - 1e-9:
            best = (cost, nse, chunk)
    return v, st, best[1], best[2]


SK_BIT = 512          # kernel MODE bit 9: stream-K schedule (csrc/attn_decode_v2.cu, sk_prefix),
SK_MAXBLK = 128       # blocks per stream-K CTA (page-address table in shared memory = MAX_CHUNK tokens)
SK_MAXB = 256


def plan_sk(B, Lmax, fmt, Hkv=8, variant=None, stages=None, page=16, grid=None):
    """stream-K schedule: (variant, stages, sk_grid).  CTA c owns the blocks [c T / G, (c + 1) T / G) of the flattened
    (sequence, kv head, 64-token block) space (T = Hkv * sum_b ceil(len_b / 64) on the device): equal work per CTA for
    any batch shape.  G = whole waves of the resident-CTA slots, as few as keep a CTA at <= SK_MAXBLK blocks (and
    never more CTAs than blocks)."""
    global NUM_SMS
    if NUM_SMS is None:
        NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    v = (DEFAULT_VARIANT[fmt] | SK_BIT) if variant is None else variant
    st = DEFAULT_STAGES if stages is None else stages
    tmax = B * Hkv * math.ceil(Lmax / 64)
    if grid is None:
        slots = max(1, ext().occupancy(FMT_ID[fmt], v, st, page)) * NUM_SMS
        grid = slots * max(1, math.ceil(tmax / (slots * SK_MAXBLK)))
        grid = max(1, min(grid, tmax))
    grid = max(Hkv, math.ceil(grid / Hkv) * Hkv)  # whole kv-head groups (CTA c: kv head c % Hkv)
    assert math.ceil(tmax / grid) <= SK_MAXBLK, "stream-K: more than SK_MAXBLK blocks per CTA"
    return v, st, grid


def use_sk(B, Lmax, fmt, variant):
    """schedule choice: stream-K only for a variant with SK_BIT, or INT4 with B <= SK_AUTO_MAXB"""
    if variant is not None:
        return bool(variant & SK_BIT)
    return fmt == "int4" and B <= SK_AUTO_MAXB


SK_AUTO_MAXB = 0  # stream-K is not chosen by default: not faster than the planned split grid
_WS = {}


def decode(q, cache: PagedKVCache, block_table=None, seq_lens=None, kmask=None, vmask=None, rb=64, qrot="fused",
           vrot=False, variant=None, stages=None, nsplit=None, out=None, ws=None, sm_scale=None, lmax=None, tail=None,
           tail_len=None, sk_grid=None):
    """q [B, G*Hkv, D] bf16 -> out [B, G*Hkv*D] bf16 (V rotation folded into W_o unless vrot); G != 4 or D = 256:
    the generalized kernel (gen_variant; variant=gen_variant(G, D, gt) picks the q tile).
    block_table / seq_lens default to cache.block_table / cache.seq_lens (make_bench_cache).  lmax: host-side max
    sequence length (avoids a device sync; defaults to cache.lmax or seq_lens.max()).
    tail / tail_len: BF16 staging window [B, W <= 256, 2*Hkv, 128] (original basis) / int32 [B]: the newest tail_len[b]
    tokens, not encoded yet, attended next to the cached ones by one extra split CTA (csrc/attn_tail.cuh; vmask is
    then needed whenever the cached values are FlagRot-rotated).
    Schedule: split-K (plan) or stream-K (plan_sk; INT4 with B <= SK_AUTO_MAXB by default, or a variant with SK_BIT;
    sk_grid overrides its CTA count)."""
    B, Hq, D = q.shape
    if block_table is None:
        block_table, seq_lens = cache.block_table, cache.seq_lens
    if lmax is None:
        lmax = getattr(cache, "lmax", None) if seq_lens is getattr(cache, "seq_lens", None) else None
        if lmax is None:
            lmax = int(seq_lens.max().item())
    gqa = Hq // cache.Hkv
    assert Hq == gqa * cache.Hkv and D == cache.D
    if variant is None and (gqa != 4 or D != 128):
        variant = gen_variant(gqa, D)                                  # generalized kernel
    GT = tile_of(variant) if variant is not None else 4
    HV = cache.Hkv * -(-gqa // GT)                                     # CTAs per (sequence, split): q tiles
    if (variant is not None and variant & SK_BIT) or (variant is None and nsplit is None and use_sk(B, lmax, cache.fmt, None)):
        v, st, grid = plan_sk(B, lmax, cache.fmt, cache.Hkv, variant, stages, cache.page, sk_grid)
        ns, chunk = 1, 64
        need = 2 * grid + B * cache.Hkv * (1 if tail is not None else 0)  # partial slots
    else:
        v, st, ns, chunk = plan(B, lmax, cache.fmt, HV, variant, stages, cache.page, nsplit)
        grid = 0
        need = B * HV * (ns + 1)
    slot = GT * D                                                      # floats per partial slot
    if ws is None:
        key = (B, HV, GT, D, q.device)
        ws = _WS.get(key)
        if ws is None or ws.max_splits < ns + 8 or ws.opart.numel() < need * slot:
            ws = _WS[key] = Workspace(B, HV, max(520, math.ceil(need / (B * HV))), q.device, GT, D)
    assert ws.opart.numel() >= need * slot, "workspace too small for this schedule"
    if out is None:
        out = torch.empty(B, Hq * D, device=q.device, dtype=torch.bfloat16)
    scale = (1.0 / math.sqrt(D) if sm_scale is None else sm_scale) * LOG2E
    ext().attn2(q, cache.pool, block_table, seq_lens, FMT_ID[cache.fmt], cache.page, ns, chunk, scale,
                cache.gs if cache.fmt == "nvfp4" else None, kmask, vmask, rb, qrot == "fused", bool(vrot), v, st,
                ws.opart, ws.mlpart, ws.counter, out, tail, tail_len, grid)
    return out


def make_bench_cache(fmt, B, L, mask, rb, page=16, src_tokens=4096, shuffle=False, gs=None, seed=1, D=128):
    """paged cache holding B sequences of length L: one block of src_tokens random tokens is written with the
    paged writer and its pages are replicated (contents do not affect timing).  shuffle=False: identity block table
    (= contiguous cache); True: random page order."""
    Hkv = mask.shape[0] // 2
    T = min(L, src_tokens)
    assert T % page == 0
    npg_seq = (L + page - 1) // page
    num_pages = B * npg_seq
    c = PagedKVCache(fmt, num_pages, page, Hkv, D=D)
    if gs is not None:
        c.gs = gs
    g = torch.Generator(device="cuda").manual_seed(seed)
    kv = torch.randn(T, 2 * Hkv, D, device="cuda", generator=g).bfloat16()
    c.write(kv, torch.arange(T, device="cuda"), mask, rb)  # pages 0 .. T/page - 1
    sp = T // page
    for s in range(sp, num_pages, sp):
        n = min(sp, num_pages - s)
        c.pool[s:s + n] = c.pool[:n]
    ids = torch.randperm(num_pages, generator=torch.Generator().manual_seed(seed)) if shuffle else torch.arange(num_pages)
    c.block_table = ids.view(B, npg_seq).to(torch.int32).cuda()
    c.seq_lens = torch.full((B,), L, dtype=torch.int32, device="cuda")
    c.lmax = L
    return c
