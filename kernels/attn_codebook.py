"""Per-token transform codebook ("FlagRot, dynamic codebook") for the INT4 paged KV cache on H20: JIT loader, paged
cache with the codebook writer, layout parser / reference reconstruction, decode attention API.
Kernels: csrc/attn_codebook.cu (+ attn_codebook_attn.cuh); semantics: codebook_ref.py;

    cache = CodebookKVCache("final" | "full" | "reduced", num_pages, page=16, Hkv=8)
    cache.set_calib(Cq [Hkv,128,128], co [Hkv,128])    static calibration constants of the layer (fp32)
    cache.write(kv, slots)                              kv [T, 2*Hkv, 128] bf16 (vLLM reshape_and_cache semantics)
    o = decode(q, cache, block_table, seq_lens)         q [B, 32, 128] bf16 -> [B, 4096] bf16 (original basis)
"""
from __future__ import annotations

import math
import os

import torch

import attn_decode_v2 as A2
import codebook_ref as CR

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT = None
# final = the locked method cq+f+S+Q5+FS+R5 (8 x 8); full = cq+S+FS+P+F (3 x 4); reduced = cq+f (2 x 2);
# head8 / head16 = cq+f+FS+R6 / cq+f+FS+R14 (1-bit I/H per vector x 8 / 16 whole-head transforms)
# konly = cq+f+konly (2 x 2 on the keys only, values plain INT4: the static v2 PV path)
# q4x4 = cq+f+S+Q1+FS+R1 (4 local x 4 head candidates, family kernel)
# q4x4k / finalk = the 4 x 4 / locked 8 x 8 on the keys only (values plain INT4, key-only family kernel)
# konlya = konly with asymmetric INT4 (blockflag int4a:32 choice, int4ah:32 storage)
KINDS = ("final", "full", "reduced", "head8", "head16", "konly", "q4x4", "q4x4k", "finalk", "konlya")
KIND_ID = {"final": 3, "full": 2, "reduced": 1, "head8": 4, "head16": 5, "konly": 6, "q4x4": 7, "q4x4k": 8, "finalk": 9,
           "konlya": 10}
FMT_NAME = {"final": "int4cb8", "full": "int4cb", "reduced": "int4cbr", "head8": "int4cbh8", "head16": "int4cbh16",
            "konly": "int4cbk", "q4x4": "int4cb4", "q4x4k": "int4cb4k", "finalk": "int4cb8k", "konlya": "int4cbka"}
KV_DYN = {"final": "cq+f+S+Q5+FS+R5", "full": "cq+f+S+FS+P", "reduced": "cq+f", "head8": "cq+f+FS+R6",
          "head16": "cq+f+FS+R14", "konly": "cq+f+konly", "q4x4": "cq+f+S+Q1+FS+R1", "q4x4k": "cq+f+S+Q1+FS+R1+konly",
          "finalk": "cq+f+S+Q5+FS+R5+konly", "konlya": "cq+f+konly (int4a:32)"}
QUANT = {"konlya": "asym"}  # codebook_ref quantizer of the writer (default: the cache's symmetric INT4)
SGW = CR.sign_bits()
DEFAULT_STAGES = 4
# 0 = software-pipelined wgmma, 1 = synchronous; reduced 2 = synchronous with prefetch (k_cbr2)
# konlya: variant 0 = value zero points on the CUDA cores (default), 1 = on the tensor cores
VARIANTS = {k: ((0, 1) if k in ("konly", "konlya") else (0,) if k in CR.KONLY else (0, 1, 2) if k == "reduced" else (0, 1))
            for k in KINDS}
DEFAULT_VARIANT = {"final": 0, "full": 1, "reduced": 2, "head8": 0, "head16": 0, "konly": 0, "q4x4": 0, "q4x4k": 0,
                   "finalk": 0, "konlya": 0}  # measured (attn_bench_cb.json)
LOG2E = 1.4426950408889634


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        os.environ["PATH"] = os.path.join(_HERE, ".deps", "bin") + os.pathsep + os.environ.get("PATH", "")
        cuda_home = os.environ.get("ATTN2_CUDA_HOME", os.environ.get("FLAGROT_CUDA_HOME", "/usr/local/cuda-12.9"))  # same toolkit as attn_decode_v2.py
        os.environ["CUDA_HOME"] = cuda_home
        import torch.utils.cpp_extension as CE
        CE.CUDA_HOME = cuda_home
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
        os.environ.setdefault("MAX_JOBS", "8")
        extra = os.environ.get("ATTNCB_EXTRA_FLAGS", "")
        tag = "" if not extra else "_" + "".join(c for c in extra if c.isalnum())[:60]
        bdir = os.path.join(_HERE, ".build_cb" + ("" if cuda_home.endswith("12.9") else "_" + os.path.basename(cuda_home)) + tag)
        os.makedirs(bdir, exist_ok=True)
        _EXT = load(
            name="attn_codebook_ext" + tag,
            sources=[os.path.join(_HERE, "csrc", "attn_cb_bind.cpp"), os.path.join(_HERE, "csrc", "attn_codebook.cu"),
                     os.path.join(_HERE, "csrc", "attn_cb_write3.cu")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-gencode", "arch=compute_90a,code=sm_90a", "-std=c++17",
                               "--expt-relaxed-constexpr", "-lineinfo"] + extra.split(),
            build_directory=bdir,
            verbose=bool(int(os.environ.get("SELROT_VERBOSE", "0"))),
        )
    return _EXT


def chunk_bytes(kind, page):
    return (144 * page + 16 if kind == "reduced" else 144 * page if kind in ("konly", "konlya") else
            146 * page if kind in ("q4x4k", "finalk") else 148 * page)


def kv_bytes_per_token(kind, Hkv=8):
    """K + V bytes per token per layer (codes, scales, side info; the reduced trailer is per page of 16)."""
    return Hkv * chunk_bytes(kind, 16) / 16


# writer v3 (csrc/attn_cb_write3.cuh): tensor-core candidate evaluation with an exact fallback; every
# codebook except 'full' (RoPE-pair candidate P: writer v1)
W3_KINDS = ("final", "reduced", "q4x4", "konly", "q4x4k", "finalk", "head8", "head16", "konlya")
# kappa: bound on (tensor-core accumulation + epilogue + second-order + reference fp32 rounding) / (rho(|C|) ||d'||^2),
#  (model 2^-16.6, measured <= 2^-19.4 on real data)
W3_KAPPA = 2.0 ** -16
W3_SPLIT_T = int(os.environ.get("W3_SPLIT_T", "256"))  # T <= this: K / V kernels on two streams
_SIDE = {}


def _side_stream(dev):
    if dev not in _SIDE:
        _SIDE[dev] = torch.cuda.Stream(dev)
    return _SIDE[dev]


W3_VCTAS_PER_SM = int(os.environ.get("W3_VCTAS", "3"))  # value kernel: resident CTAs per SM (persistent tiles of 8 tokens)
W3_CTAS_PER_SM = 2   # resident key CTAs per SM (smem / registers); persistent grid = Hkv x floor(2 SMs / Hkv)


def w3_nchunk(Hkv, per_sm=None):
    """persistent CTAs per kv head: one resident wave (a second partial wave would double the time)."""
    if A2.NUM_SMS is None:
        A2.NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    return max(1, ((W3_CTAS_PER_SM if per_sm is None else per_sm) * A2.NUM_SMS) // Hkv)


def w3_calib(Cq, kappa=None, nd=8):
    """host constants of writer v3 (csrc/attn_cb_write3.cuh): C1 fp16 [H, 168, 128] = rows 0..127 fp16(2^tau C_sym)
    (C_sym = (C + C^T) / 2), rows 128..135 the head correction F (rows D of E = 2^tau C_sym - C1, off-D entries
    doubled), rows 136 + 8 j .. the same for the diagonal 32-block j (zero outside the block); aux fp32 [H, 48]:
    0 head bound beta, 1..4 block bounds, 5 absolute slack, 8..15 the head's D, 16 + 8 j .. block j's D (column
    indices).  beta = ||E_RR||_2 + ||F - fp16(F)||_2 + 2^-10 ||E||_2 + kappa rho(M), M = 2^tau (|C| + |C|^T) / 2
    (rho(M) ||d'||^2 bounds |d'|^T |C| |d'|), all in fp64, +1 % for the fp32 evaluation in the kernel."""
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
    spec = lambda X: torch.linalg.eigvalsh(X).abs().amax(-1)
    out = torch.zeros(H, 168, 128, dtype=torch.float16, device=C.device)
    out[:, :128] = C1
    aux = torch.zeros(H, 48, dtype=torch.float64, device=C.device)
    for bi, (lo, hi) in enumerate([(0, 128)] + [(32 * j, 32 * j + 32) for j in range(4)]):
        Eb, Mb = E[:, lo:hi, lo:hi], M[:, lo:hi, lo:hi]
        n = hi - lo
        D = torch.argsort(Eb.norm(dim=2), dim=1, descending=True)[:, :nd]                  # [H, nd] (block coords)
        inD = torch.zeros(H, n, dtype=torch.bool, device=C.device)
        inD.scatter_(1, D, True)
        F = torch.gather(Eb, 1, D[:, :, None].expand(-1, -1, n))
        F = torch.where(inD[:, None, :], F, 2 * F)
        Fh = F.float().half()
        Err = torch.where(inD[:, :, None] | inD[:, None, :], torch.zeros_like(Eb), Eb)
        beta = (spec(Err) + torch.linalg.matrix_norm(F - Fh.double(), 2) + 2.0 ** -10 * spec(Eb)
                + kappa * spec(Mb)) * 1.01
        row = 128 if bi == 0 else 136 + 8 * (bi - 1)
        out[:, row:row + nd, lo:hi] = Fh
        aux[:, bi] = beta
        aux[:, 8 + 8 * bi:16 + 8 * bi] = (D + lo).double()
    aux[:, 5] = 2.0 ** -40 * spec(M)
    return out.contiguous(), aux.float().contiguous()


def write_tg(T, kind="full"):
    """tokens per writer CTA: small groups for decode steps (parallelism), larger for prefill (fewer C reloads); the
    final codebook keeps 8 head candidates per token in shared memory (at most 4 tokens per CTA there)."""
    return 1 if T <= 16 else (2 if T <= 64 or kind in CR.FAMILY else 4)  # measured (write_bench_cb.py sweep)


class CodebookKVCache:
    """pool uint8 [num_pages, Hkv, chunk]: per (page, kv head) the v2 INT4 chunk + side info (attn_codebook.cuh)."""

    def __init__(self, kind, num_pages, page=16, Hkv=8, device="cuda", D=128):
        assert kind in KINDS and page in (16, 32) and D in (128, 256)
        assert D == 128 or kind in GEN_KINDS, "head dim 256: key-only 2x2 kinds only "
        self.kind, self.num_pages, self.page, self.Hkv, self.D = kind, num_pages, page, Hkv, D
        self.fmt = FMT_NAME[kind]
        # D = 256: two 128-d half chunks per (page, kv head)
        self.pool = torch.zeros(num_pages, Hkv, (D // 128) * chunk_bytes(kind, page), device=device, dtype=torch.uint8)
        self.CT = self.co = None
        self.writer = "v3" if kind in W3_KINDS and D == 128 else "v1"  # (D = 256: writer v1, k_cb_write256)
        self.w3_stats = None   # int32 [4] tensor: += local items, local fallbacks, head items, head fallbacks (v3)

    def set_calib(self, Cq, co):
        """Cq [Hkv, 128, 128] (query second moment of the KV head's query heads), co [Hkv, 128] (o_proj column energy)."""
        self.Cq = Cq.float().contiguous()
        self.CT = self.Cq.transpose(1, 2).contiguous()
        self.co = co.float().contiguous()
        if self.writer == "v3":
            self.C1, self.beta = w3_calib(self.Cq)

    def write(self, kv, slots, tg=None, sort_window=0):
        """kv [T, 2*Hkv, 128] -> slots [T] (or per-(token, kv head) slots [T, Hkv]).
        sort_window = W > 0 (encoder-side reordering): the T tokens are T / W windows of W consecutive tokens of one
        sequence each (e.g. staging-window flushes); inside every window, per kv head, the tokens are stored in a
        stable order of their head-level choice (keys: base first, then the whole-head transforms; values as the
        secondary key), so that most 64-token blocks hold a single key head mode and the decode kernels skip the
        other variants.  Decode attention is order-invariant over cached tokens (RoPE is applied before caching), and
        K and V of a token move together.  Here the choice is computed twice (scratch encode, then placement)."""
        if kv.stride(-1) != 1 or kv.stride(-2) != kv.shape[-1]:
            kv = kv.contiguous()
        T = kv.shape[0]
        slots = slots.to(torch.int64)
        if sort_window:
            slots = self.sorted_slots(kv, slots, sort_window, tg)
        if self.writer == "v3":
            if kv.stride(0) % 8 or kv.data_ptr() % 16:      # cp.async prefetch: 16-byte aligned rows
                kv = kv.contiguous()
            args = (kv, slots.contiguous(), self.CT, self.co, self.C1, self.beta, SGW, KIND_ID[self.kind], self.pool,
                    self.page)
            if T <= W3_SPLIT_T:  # small writes (decode steps): K and V kernels side by
                cur = torch.cuda.current_stream()                # side on two streams (latency, not throughput)
                side = _side_stream(kv.device)
                side.wait_stream(cur)
                ext().cb_write3(*args, w3_nchunk(self.Hkv), -1, self.w3_stats)
                with torch.cuda.stream(side):
                    ext().cb_write3(*args, -1, w3_nchunk(self.Hkv, W3_VCTAS_PER_SM), self.w3_stats)
                cur.wait_stream(side)
            else:
                ext().cb_write3(*args, w3_nchunk(self.Hkv), w3_nchunk(self.Hkv, W3_VCTAS_PER_SM), self.w3_stats)
        else:
            ext().cb_write(kv, slots.contiguous(), self.CT, self.co, SGW, KIND_ID[self.kind], self.pool,
                           self.page, tg or write_tg(T, self.kind))

    def sorted_slots(self, kv, slots, W, tg=None):
        T, H = kv.shape[0], self.Hkv
        assert T % W == 0 and slots.dim() == 1
        npg = (T + self.page - 1) // self.page
        sc = CodebookKVCache(self.kind, npg, self.page, H, device=self.pool.device)
        sc.Cq, sc.CT, sc.co, sc.writer = self.Cq, self.CT, self.co, self.writer
        if self.kind in W3_KINDS:
            sc.C1, sc.beta = self.C1, self.beta
        sc.write(kv, torch.arange(T, device=kv.device, dtype=torch.int64), tg)
        mode = sc.parse_pages()["mode"].reshape(npg * self.page, H, 2)[:T]               # [T, H, (K, V)]
        key = (mode[..., 0] > 0).long() * 64 + mode[..., 0] * 2 + (mode[..., 1] > 0).long()  # K head mode, then V
        order = torch.sort(key.view(T // W, W, H), dim=1, stable=True).indices          # rank r -> token in window
        out = torch.empty(T // W, W, H, dtype=torch.int64, device=kv.device)
        out.scatter_(1, order, slots.view(T // W, W, 1).expand(-1, -1, H).contiguous())
        return out.view(T, H)

    # ------------------------------------------------------------------ layout parser (tests)
    def parse_pages(self):
        """-> per (page, token, kv head, K/V): codes int [Np, P, Hkv, 2, D] (stored coords), scales fp32 [.., D / 32]
        (magnitudes), local [.., D / 32], mode [..] (decoded from the side words / sign bits / head-bit planes)."""
        if getattr(self, "D", 128) == 256:  # two 128-d half chunks; the head bit lives in half 0
            hv = CodebookKVCache.__new__(CodebookKVCache)
            hv.kind, hv.num_pages, hv.page, hv.Hkv, hv.D = self.kind, self.num_pages, self.page, 2 * self.Hkv, 128
            hv.pool = self.pool.view(self.num_pages, 2 * self.Hkv, -1)
            r = hv.parse_pages()
            Np, P = r["codes"].shape[:2]
            cat = lambda x, n: x.view(Np, P, self.Hkv, 2, 2, n).permute(0, 1, 2, 4, 3, 5).reshape(Np, P, self.Hkv, 2, 2 * n)
            mode = r["mode"].view(Np, P, self.Hkv, 2, 2)[:, :, :, 0]                       # half 0 holds the bits
            ks = cat(r["scales"], 4)
            raw = cat(r["sbits"], 4) if "sbits" in r else None
            local = ((raw >> 15) & 1) * (mode == 0)[..., None] if raw is not None else cat(r["local"], 4)
            local[..., 1, :] = 0
            return dict(codes=cat(r["codes"], 128), scales=ks, local=local, mode=mode)
        P, H = self.page, self.Hkv
        pool = self.pool.long()
        Np = pool.shape[0]
        kc = pool[:, :, :64 * P].reshape(Np, H, P, 4, 4, 4)
        vt = pool[:, :, 64 * P:128 * P].reshape(Np, H, 4, 8, P // 8, 4, 4)
        ks = self.pool[:, :, 128 * P:136 * P].reshape(Np, H, P, 8).view(torch.int16).long() & 0xFFFF   # [Np,H,P,4]
        vs = self.pool[:, :, 136 * P:144 * P].reshape(Np, H, 4, 2 * P).view(torch.int16).long() & 0xFFFF  # [Np,H,4,P perm]

        def codes_of(byt):
            if self.kind == "konlya":                                  # unsigned nibbles: A low, B high
                return byt & 15, (byt >> 4) & 15
            sA, mA = (byt >> 7) & 1, byt & 7
            sB = (byt >> 3) & 1
            mB = (torch.roll(byt, shifts=1, dims=-1) >> 4) & 7
            return torch.where(sA == 1, -mA, mA), torch.where(sB == 1, -mB, mB)

        dev = pool.device
        K = torch.empty(Np, H, P, 128, dtype=torch.long, device=dev)
        tg = torch.arange(4, device=dev).view(4, 1, 1)
        j = torch.arange(4, device=dev).view(1, 4, 1)
        i = torch.arange(4, device=dev).view(1, 1, 4)
        cA, cB = codes_of(kc)
        K[..., (32 * j + 4 * tg + i).reshape(-1)] = cA.reshape(Np, H, P, 64)
        K[..., (32 * j + 16 + 4 * tg + i).reshape(-1)] = cB.reshape(Np, H, P, 64)
        V = torch.empty(Np, H, P, 128, dtype=torch.long, device=dev)
        w = torch.arange(4, device=dev).view(4, 1, 1, 1, 1)
        gid = torch.arange(8, device=dev).view(1, 8, 1, 1, 1)
        c8 = torch.arange(P // 8, device=dev).view(1, 1, P // 8, 1, 1)
        sub = torch.arange(4, device=dev).view(1, 1, 1, 4, 1)
        ii = torch.arange(4, device=dev).view(1, 1, 1, 1, 4)
        shp = (4, 8, P // 8, 4, 4)
        dd = (64 * (sub >> 1) + 16 * w + gid + 8 * (sub & 1)).expand(shp).reshape(-1)
        tA = (8 * c8 + 2 * ii).expand(shp).reshape(-1)
        vA, vB = codes_of(vt)
        V[:, :, tA, dd] = vA.reshape(Np, H, -1)
        V[:, :, tA + 1, dd] = vB.reshape(Np, H, -1)
        perm8 = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=dev)
        tok = (8 * torch.arange(P // 8, device=dev).view(-1, 1) + perm8.view(1, -1)).reshape(-1)
        vsb = torch.empty(Np, H, 4, P, dtype=torch.long, device=dev)
        vsb[..., tok] = vs
        vsb = vsb.permute(0, 1, 3, 2)                                                           # [Np, H, P, 4]
        sbits = torch.stack([ks, vsb], 3)                                                       # [Np, H, P, 2, 4]
        codes = torch.stack([K, V], 3)
        mag = (sbits & 0x7FFF).to(torch.int16).view(torch.float16).float()
        if self.kind in ("konly", "konlya"):                           # sign bits: K scale (vector), V group-0 scale (F)
            fk = (vsb[..., 0] >> 15) & 1                                                         # [Np, H, P]
            mode = 2 * torch.stack([fk, torch.zeros_like(fk)], -1)                               # [Np, H, P, 2]
            local = ((sbits >> 15) & 1) * (mode == 0)[..., None]
            local[..., 1, :] = 0
        elif self.kind != "reduced":
            nsw = 1 if self.kind in CR.KONLY else 2                                             # key-only: no V words
            sw = self.pool[:, :, 144 * P:(144 + 2 * nsw) * P].reshape(Np, H, nsw, 2 * P).view(torch.int16).long() & 0xFFFF
            if nsw == 1:
                sw = torch.cat([sw, torch.zeros_like(sw)], 2)                                    # [Np,H,2,P]
            sw = sw.permute(0, 1, 3, 2)                                                         # [Np, H, P, 2]
            lb, hb = (CR.LBITS[self.kind], CR.HBITS[self.kind]) if self.kind in CR.FAMILY else (2, 2)
            local = torch.stack([(sw >> (lb * jj)) & ((1 << lb) - 1) for jj in range(4)], -1)
            mode = (sw >> (4 * lb)) & ((1 << hb) - 1)
        else:
            tr = self.pool[:, :, 144 * P:144 * P + 8].reshape(Np, H, 2, 4).view(torch.int32).squeeze(-1).long() & 0xFFFFFFFF
            t = torch.arange(P, device=dev)
            fk = (tr[..., 0:1] >> t) & 1                                                         # [Np, H, P]
            vbit = 8 * (t // 8) + 4 * (t & 1) + ((t % 8) >> 1)
            fv = (tr[..., 1:2] >> vbit) & 1
            mode = 2 * torch.stack([fk, fv], -1)                                                 # [Np, H, P, 2]
            local = ((sbits >> 15) & 1) * (mode == 0)[..., None]
        p = lambda x: x.permute(0, 2, 1, 3, *range(4, x.dim())) if x.dim() > 3 else x
        return dict(codes=p(codes), scales=p(mag), local=p(local), mode=mode.permute(0, 2, 1, 3), sbits=p(sbits))

    def reconstruct_pages(self, dtype=torch.float64):
        """K, V [num_pages, P, Hkv, D] in the original basis (inverse transforms in `dtype`)."""
        r = self.parse_pages()
        fn = CR.reconstruct_final if self.kind in CR.FAMILY else CR.reconstruct
        codes = r["codes"]
        if self.kind == "konlya":                                       # (c - z) s
            codes = codes - CR.zero_points(r["scales"]).repeat_interleave(32, -1)
        rec = fn(codes, r["scales"], r["local"], r["mode"], dtype)                              # [Np, P, H, 2, 128]
        return rec[..., 0, :], rec[..., 1, :]

    def reconstruct(self, block_table, seq_lens, dtype=torch.float64):
        Kp, Vp = self.reconstruct_pages(dtype)
        B = block_table.shape[0]
        Lmax = int(seq_lens.max().item())
        npg = (Lmax + self.page - 1) // self.page
        bt = block_table[:, :npg].long()
        Dh = getattr(self, "D", 128)
        K = Kp[bt].reshape(B, npg * self.page, self.Hkv, Dh)[:, :Lmax]
        V = Vp[bt].reshape(B, npg * self.page, self.Hkv, Dh)[:, :Lmax]
        valid = torch.arange(Lmax, device=K.device).view(1, -1) < seq_lens.view(-1, 1).long()
        return K * valid[..., None, None], V * valid[..., None, None]


PLAN_C0, PLAN_C1 = int(os.environ.get("ATTNCB_PLAN_C0", 1024)), 4  # per-CTA overhead (token-equivalents), as attn_decode_v2


def plan(B, Lmax, kind, Hkv=8, stages=None, page=16, nsplit=None, variant=None):
    """(stages, nsplit, chunk): split count from the wave-quantization cost model of attn_decode_v2.plan."""
    st = DEFAULT_STAGES if stages is None else stages
    v = DEFAULT_VARIANT[kind] if variant is None else variant
    gran = max(64, page)
    MAXC = 8192
    if nsplit is not None:
        chunk = min(MAXC, math.ceil(Lmax / nsplit / gran) * gran)
        return st, math.ceil(Lmax / chunk), chunk
    if A2.NUM_SMS is None:
        A2.NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    occ = max(1, ext().occupancy(KIND_ID[kind], v, st, page))
    slots = occ * A2.NUM_SMS
    best = None
    for ns in range(max(1, math.ceil(Lmax / MAXC)), min(512, math.ceil(Lmax / gran)) + 1):
        chunk = math.ceil(Lmax / ns / gran) * gran
        if chunk > MAXC:
            continue
        nse = math.ceil(Lmax / chunk)
        if best is not None and nse == best[1]:
            continue
        cost = math.ceil(B * Hkv * nse / slots) * (chunk + PLAN_C0) + (PLAN_C1 * nse if nse > 1 else 0)
        if best is None or cost < best[0] - 1e-9:
            best = (cost, nse, chunk)
    return st, best[1], best[2]


_WS = {}


def decode(q, cache: CodebookKVCache, block_table=None, seq_lens=None, stages=None, nsplit=None, out=None, ws=None,
           sm_scale=None, lmax=None, variant=None, tail=None, tail_len=None):
    """q [B, 4*Hkv, 128] bf16 -> out [B, 4*Hkv*128] bf16 = attention over the reconstructions (original basis).
    tail / tail_len: BF16 staging window [B, W <= 256, 2*Hkv, 128] / int32 [B]: the newest tail_len[b] tokens, not
    encoded yet, attended in BF16 next to the seq_lens[b] cached tokens by one extra split CTA (csrc/attn_tail.cuh)."""
    B, Hq, Dd = q.shape
    if block_table is None:
        block_table, seq_lens = cache.block_table, cache.seq_lens
    if lmax is None:
        lmax = getattr(cache, "lmax", None) if seq_lens is getattr(cache, "seq_lens", None) else None
        if lmax is None:
            lmax = int(seq_lens.max().item())
    lmax = max(1, lmax)
    gqa = Hq // cache.Hkv
    if cache.kind in GEN_KINDS and (gqa != 4 or Dd != 128 or variant == "gen" or (isinstance(variant, int) and variant & A2.GEN)):
        # key-only 2x2 caches with any GQA group / head dim 256: the generalized static kernel
        gv = A2.gen_variant(gqa, Dd) if not isinstance(variant, int) else variant
        return A2.decode(q, _KOView(cache), block_table, seq_lens, rb=32, qrot=None, vrot=False, variant=gv,
                         stages=stages, nsplit=nsplit, out=out, sm_scale=sm_scale, lmax=lmax, tail=tail, tail_len=tail_len)
    v = DEFAULT_VARIANT[cache.kind] if variant is None else variant
    st, ns, chunk = plan(B, lmax, cache.kind, cache.Hkv, stages, cache.page, nsplit, v)
    if ws is None:
        key = (B, cache.Hkv, q.device)
        ws = _WS.get(key)
        if ws is None or ws.max_splits < ns + 8:
            ws = _WS[key] = A2.Workspace(B, cache.Hkv, 520, q.device)
    if out is None:
        out = torch.empty(B, Hq * Dd, device=q.device, dtype=torch.bfloat16)
    scale = (1.0 / math.sqrt(Dd) if sm_scale is None else sm_scale) * LOG2E
    ext().cb_attn(q, cache.pool, block_table, seq_lens, KIND_ID[cache.kind], cache.page, ns, chunk, scale, SGW, v, st,
                  ws.opart, ws.mlpart, ws.counter, out, tail, tail_len)
    return out


GEN_KINDS = ("konly", "konlya")  # kinds the generalized static kernel reads (A2 formats "konly" / "konlya")


class _KOView:
    """a key-only 2x2 CodebookKVCache seen as an attn_decode_v2 paged cache (same pool; fmt konly / konlya)"""

    def __init__(self, c):
        self.fmt, self.pool, self.page, self.Hkv, self.D, self.gs = c.kind, c.pool, c.page, c.Hkv, getattr(c, "D", 128), None
        for k in ("block_table", "seq_lens", "lmax"):
            if hasattr(c, k):
                setattr(self, k, getattr(c, k))


def random_calib(Hkv=8, nq=512, seed=0, device="cuda", D=128):
    """synthetic static calibration: C_q = sum of q q^T over nq LLM-like queries per KV head (a few dominant
    channels), c_o = positive o_proj column energies."""
    g = torch.Generator(device=device).manual_seed(seed)
    qs = torch.randn(Hkv, nq, D, device=device, generator=g)
    ch = torch.randint(0, D, (Hkv, 8 * D // 128), device=device, generator=g)
    for h in range(Hkv):
        qs[h][:, ch[h]] *= 4.0
    Cq = torch.einsum("hnd,hne->hde", qs, qs)
    co = torch.rand(Hkv, D, device=device, generator=g) + 0.1
    return Cq, co


def make_bench_cache(kind, B, L, Cq, co, page=16, src_tokens=4096, shuffle=False, seed=1, sort_window=0):
    """paged codebook cache holding B sequences of length L (one block of src_tokens random tokens written with the
    codebook writer, pages replicated; contents do not affect timing beyond the per-block variant mix: sort_window
    = W stores the tokens of every W-token window ordered by head mode, see CodebookKVCache.write)."""
    Hkv, Dh = Cq.shape[0], Cq.shape[-1]
    T = min(L, src_tokens)
    assert T % page == 0
    npg_seq = (L + page - 1) // page
    num_pages = B * npg_seq
    c = CodebookKVCache(kind, num_pages, page, Hkv, D=Dh)
    c.set_calib(Cq, co)
    g = torch.Generator(device="cuda").manual_seed(seed)
    kv = torch.randn(T, 2 * Hkv, Dh, device="cuda", generator=g)
    kv[:, :Hkv, :8] *= 6.0
    kv = kv.bfloat16()
    c.write(kv, torch.arange(T, device="cuda"), sort_window=sort_window)
    sp = T // page
    ids = torch.randperm(num_pages, generator=torch.Generator().manual_seed(seed)) if shuffle else torch.arange(num_pages)
    # logical page j of every sequence holds source page j % sp (physical placement shuffled), so the logical token
    # order inside a sequence (the encoder's windows) is preserved
    src = c.pool[:sp].clone()
    c.pool[ids.cuda()] = src[(torch.arange(num_pages, device="cuda") % npg_seq) % sp]
    c.block_table = ids.view(B, npg_seq).to(torch.int32).cuda()
    c.seq_lens = torch.full((B,), L, dtype=torch.int32, device="cuda")
    c.lmax = L
    return c
