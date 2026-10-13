"""FlagRot INT4 KV codec: reference PyTorch implementation of the research encoder (Phase 2a).

Semantics = blockflag.qattention.dyn_quant (rule "cq", INT4:32, rb 32) for a codebook given as a research kv_dyn
string (parsed exactly like blockflag.run.build).  Default = the locked method "cq+f+S+Q5+FS+R5":

  per (token, KV head) x in R^D (post-RoPE key or value, D = 128):
    per 32-value sub-vector u, 8 candidates: I: Q(u),  H: H32 Q(H32 u),  HS<s>: s * H32 Q(H32 (s u)) for the sign
        vectors of seeds s in {0, 2, 3, 4, 5, 6} (sign vector of length D, restricted to the sub-vector);
        the sub-vector keeps the candidate with the lowest weighted error (block-diagonal part of the weight)
    per head, 8 candidates: the per-vector choice ("base"), F: H_D Q(H_D x), FS<s>: s * H_D Q(H_D (s x)) for the
        same seeds; the head keeps the lowest full weighted error
  (the older codebook "cq+f+S+FS+P" = local {I, H, HS}, head {base, P, F, FS}, with P = the per-vector search in the
   RoPE-pair-interleaved order, is still supported.)
  Q = int4:32: symmetric, scale = absmax / 7 per 32 values, codes round(y / scale) in [-7, 7]; ties of the error
  argmin go to the first candidate in the research order.
  weights: keys   e^T C_q e  (C_q = calibrated query second moment of the KV head),
           values sum_i c_o,i e_i^2 (c_o = o_proj column energy of the KV head).

Storage per (token, KV head) and per side (K or V), 74 bytes for D = 128:
  codes  64 B  packed INT4 (code + 8, low nibble = even element), in the transform domain of the chosen candidate
  scales  8 B  4 x fp16 (one per 32 values; the encoder decides with the fp32 scale, the cache stores it as fp16)
  side    2 B  uint16: bits [b_l*j, b_l*(j+1)) = candidate of sub-vector j, bits [b_l*nv, b_l*nv + b_h) = head
               candidate; b_l = ceil(log2 #local), b_h = ceil(log2 #head) (3 + 3 bits: 4*3 + 3 = 15 bits);
               the sub-vector fields are 0 for whole-head candidates
A KV-cache cell holds K and V of one (token, KV head): [K codes | V codes | K scales | V scales | K side | V side]
= 148 bytes for D = 128 (4.625 bits per element).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

RB = 32           # rotation block = quantization group
LIM = 7           # int4 symmetric
DEFAULT_KV_DYN = "cq+f+S+Q5+FS+R5"


def side_bytes_per_kind(head_size: int) -> int:
    return head_size // 2 + 2 * (head_size // RB) + 2


def cell_bytes(head_size: int) -> int:
    """bytes of one (token, KV head) cell holding K and V."""
    return 2 * side_bytes_per_kind(head_size)


def cell_offsets(head_size: int) -> dict:
    nc, ns = head_size // 2, 2 * (head_size // RB)
    return dict(k_codes=0, v_codes=nc, k_scales=2 * nc, v_scales=2 * nc + ns, k_side=2 * nc + 2 * ns,
                v_side=2 * nc + 2 * ns + 2, end=2 * nc + 2 * ns + 4)


def research_signs(D: int, seed: int = 0) -> torch.Tensor:
    """= blockflag.qattention._signs (CPU generator, so identical across devices / torch versions checked)."""
    g = torch.Generator().manual_seed(1234 + D + 7919 * seed)
    return (torch.randint(0, 2, (D,), generator=g) * 2 - 1).float()


@dataclass(frozen=True)
class Codebook:
    """candidate lists in research order (blockflag.run.build parsing of kv_dyn)."""
    rule: str
    local: tuple            # e.g. ("H", "HS", "HS2", ..., "HS6"); identity is implicit candidate 0
    head: tuple             # e.g. ("F", "FS", "FS2", ..., "FS6") or ("P", "F", "FS"); "base" is implicit 0

    @classmethod
    def parse(cls, kv_dyn: str) -> "Codebook":
        parts = kv_dyn.split("+")
        cand = {x.upper() for x in parts[1:]}
        qk = max([int(c[1:]) for c in cand if c[:1] == "Q" and c[1:].isdigit()] or [0])
        rk = max([int(c[1:]) for c in cand if c[:1] == "R" and c[1:].isdigit()] or [0])
        local = ["H"] + (["HS"] if "S" in cand else []) + [f"HS{i}" for i in range(2, qk + 2)]
        head = [h for h in ("P", "F", "FS") if h in cand] + [f"FS{i}" for i in range(2, rk + 2)]
        if parts[0] != "cq":
            raise ValueError(f"only the cq rule is deployable (static weights), got {parts[0]!r}")
        return cls(parts[0], tuple(local), tuple(head))

    @property
    def n_local(self) -> int:
        return 1 + len(self.local)

    @property
    def n_head(self) -> int:
        return 1 + len(self.head)

    @property
    def bits_local(self) -> int:
        return math.ceil(math.log2(self.n_local))

    @property
    def bits_head(self) -> int:
        return math.ceil(math.log2(self.n_head)) if self.n_head > 1 else 0

    def side_bits(self, D: int) -> int:
        return self.bits_local * (D // RB) + self.bits_head


def _seed(name: str, prefix: str):
    """'H' / 'F' -> None (no signs); 'HS' -> 0, 'HS3' -> 3 (blockflag: int(t[2:] or 0))."""
    rest = name[len(prefix):]
    return None if rest == "" else int(rest[1:] or 0)


def fwht(x: torch.Tensor, rb: int) -> torch.Tensor:
    """orthonormal Sylvester-Hadamard on every consecutive rb-vector of the last dim (same butterfly order as
    blockflag.hadamard.fwht, so fp32 results match the research code)."""
    if rb == 1:
        return x
    shp, K = x.shape, x.shape[-1]
    y = x.reshape(-1, K // rb, rb)
    h = 1
    while h < rb:
        y = y.reshape(-1, K // rb, rb // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2)
        h *= 2
    return y.reshape(shp) * (1.0 / math.sqrt(rb))


def _q(y: torch.Tensor):
    """int4:32 of [..., D] -> codes (float, [-7, 7]) [..., D], fp32 scales [..., D/32] (blockflag.formats.quantize)."""
    b = y.reshape(*y.shape[:-1], -1, RB)
    s = b.abs().amax(-1, keepdim=True).clamp_min(1e-12) / LIM
    c = torch.round(b / s).clamp(-LIM, LIM)
    return c.reshape(y.shape), s.squeeze(-1)


def _deq(c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return (c.reshape(*c.shape[:-1], -1, RB) * s[..., None]).reshape(c.shape)


@dataclass
class KindWeights:
    """error weights of one (layer, kind) in both domains; kind 'mat' (keys) or 'diag' (values)."""
    kind: str
    full: torch.Tensor          # mat: [H, D, D]   diag: [H, D]
    blocks: torch.Tensor        # mat: [H, nv, RB, RB]  diag: [H, nv, RB]
    full_p: torch.Tensor        # same, RoPE-pair-permuted coordinates (only used by the 'P' head candidate)
    blocks_p: torch.Tensor


def make_weights(kind: str, w: torch.Tensor, perm: torch.Tensor) -> KindWeights:
    w = w.float()
    H, D = w.shape[0], w.shape[1]
    nv = D // RB
    if kind == "mat":
        wp = w.index_select(1, perm).index_select(2, perm)
        blk = lambda m: torch.stack([m[:, j * RB:(j + 1) * RB, j * RB:(j + 1) * RB] for j in range(nv)], 1)  # noqa
        return KindWeights("mat", w.contiguous(), blk(w).contiguous(), wp.contiguous(), blk(wp).contiguous())
    wp = w.index_select(1, perm)
    return KindWeights("diag", w.contiguous(), w.view(H, nv, RB).contiguous(), wp.contiguous(),
                       wp.view(H, nv, RB).contiguous())


def _vec_err(e: torch.Tensor, blocks: torch.Tensor, kind: str) -> torch.Tensor:
    """e [T, H, D] -> [T, H, nv] (blockflag.qattention._vec_err, block-diagonal part of the weight)."""
    T, H, D = e.shape
    ev = e.reshape(T, H, D // RB, RB)
    if kind == "diag":
        return (ev.square() * blocks[None]).sum(-1)
    return torch.einsum("thvi,hvij,thvj->thv", ev, blocks, ev)


def _head_err(e: torch.Tensor, full: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "diag":
        return (e.square() * full[None]).sum(-1)
    return torch.einsum("thi,hij,thj->th", e, full, e)


def _select(idx: torch.Tensor, cands: list) -> torch.Tensor:
    """idx [...] (broadcastable to the candidates) -> element of cands[idx]."""
    out = cands[0]
    for i in range(1, len(cands)):
        out = torch.where(idx == i, cands[i], out)
    return out


class FlagRotCodec:
    """encoder / decoder shared by all layers (holds the fixed transform constants)."""

    def __init__(self, codebook: Codebook, D: int, device, perm: torch.Tensor | None = None):
        self.cb, self.D = codebook, D
        assert codebook.side_bits(D) <= 16, (codebook, D)
        self.nv = D // RB
        if perm is None:
            h = torch.arange(D // 2)
            perm = torch.stack([h, h + D // 2], 1).reshape(-1)            # = blockflag.qattention._perm
        self.perm = perm.long().to(device)
        self.inv_perm = torch.argsort(self.perm)
        ones = torch.ones(D)
        sg = lambda name, pre: ones if _seed(name, pre) is None else research_signs(D, _seed(name, pre))  # noqa
        # sign tables indexed by candidate id: local id 0 = identity (unused row), 1.. = local list
        self.local_signs = torch.stack([ones] + [sg(t, "H") for t in codebook.local]).to(device)       # [nl, D]
        self.head_names = ("base",) + codebook.head
        self.whole = [i for i, h in enumerate(self.head_names) if h.startswith("F")]
        self.head_signs = torch.stack([ones if not h.startswith("F") else sg(h, "F")
                                       for h in self.head_names]).to(device)                          # [nh, D]
        bl, nv = codebook.bits_local, self.nv
        self.shifts = torch.arange(0, bl * nv, bl, device=device, dtype=torch.int32)
        self.lmask, self.hshift, self.hmask = (1 << bl) - 1, bl * nv, (1 << codebook.bits_head) - 1
        self.off = cell_offsets(D)

    # ------------------------------------------------------------------ encoder
    def _local_best(self, x: torch.Tensor, blocks: torch.Tensor, kind: str):
        """blockflag.qattention._local_best: per sub-vector best of identity and the local candidates.
        All candidates go through ONE batched butterfly (elementwise identical to the per-candidate research calls,
        so the fp32 results are bit-identical; batching only cuts the kernel-launch count)."""
        T, H, D = x.shape
        sg = self.local_signs[1:, None, None, :]                                      # [nl-1, 1, 1, D]
        y = torch.cat([x[None], fwht(x[None] * sg, RB)])                               # [nl, T, H, D]
        c, s = _q(y)
        r = _deq(c, s)
        r = torch.cat([r[:1], sg * fwht(r[1:], RB)])
        nl = y.shape[0]
        err = _vec_err((r - x[None]).reshape(nl * T, H, D), blocks, kind).view(nl, T, H, -1)
        ch = err.argmin(0)                                                                # first minimum wins
        chx = ch.repeat_interleave(RB, -1)[None]
        rec = r.gather(0, chx)[0]
        codes = c.gather(0, chx)[0]
        scales = s.gather(0, ch[None])[0]
        return rec, codes, scales, ch

    def encode(self, x: torch.Tensor, W: KindWeights, return_recon: bool = False):
        """x [T, H, D] -> codes int8 [T,H,D] in [-7, 7], scales fp32 [T,H,nv], side int32 [T,H] (+ recon fp32)."""
        x = x.float()
        T, H, D = x.shape
        r0, c0, s0, ch0 = self._local_best(x, W.blocks, W.kind)
        recs, codes, scales, chs = [r0[None]], [c0[None]], [s0[None]], [ch0[None]]
        if "P" in self.cb.head:                                    # earlier codebook: per-vector search, RoPE pairs
            xp = x.index_select(-1, self.perm)
            rp, cp, sp, chp = self._local_best(xp, W.blocks_p, W.kind)
            recs.append(rp.index_select(-1, self.inv_perm)[None]); codes.append(cp[None]); scales.append(sp[None])
            chs.append(chp[None])
        if self.whole:                                             # whole-head candidates, one batched butterfly
            sg = self.head_signs[self.whole][:, None, None, :]
            c, s = _q(fwht(x[None] * sg, D))
            recs.append(sg * fwht(_deq(c, s), D)); codes.append(c); scales.append(s)
            chs.append(torch.zeros_like(ch0).expand(len(self.whole), *ch0.shape))
        recs, codes, scales, chs = (torch.cat(t) for t in (recs, codes, scales, chs))   # research candidate order
        nh = recs.shape[0]
        if nh > 1:
            he = _head_err((recs - x[None]).reshape(nh * T, H, D), W.full, W.kind).view(nh, T, H)
            mode = he.argmin(0)
        else:
            mode = torch.zeros(T, H, dtype=torch.long, device=x.device)
        m = mode[None, ..., None]
        code = codes.gather(0, m.expand(1, T, H, D))[0]
        scale = scales.gather(0, m.expand(1, T, H, scales.shape[-1]))[0]
        ch = chs.gather(0, m.expand(1, T, H, chs.shape[-1]))[0]
        side = (ch.to(torch.int32) << self.shifts).sum(-1) | (mode.to(torch.int32) << self.hshift)
        out = (code.to(torch.int8), scale, side)
        if return_recon:
            out = out + (recs.gather(0, m.expand(1, T, H, D))[0],)
        return out

    def pack(self, codes: torch.Tensor, scales: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        """-> uint8 [T, H, 74] = [codes 64 | fp16 scales 8 | side 2] for one kind (side stored as 16 raw bits)."""
        u = (codes.to(torch.int16) + 8).to(torch.uint8)
        packed = u[..., 0::2] | (u[..., 1::2] << 4)
        sc = scales.to(torch.float16).contiguous().view(torch.uint8)
        sd = side.to(torch.int32).to(torch.int16)[..., None].contiguous().view(torch.uint8)
        return torch.cat([packed, sc, sd], -1)

    def encode_cells(self, k: torch.Tensor, v: torch.Tensor, Wk: KindWeights, Wv: KindWeights) -> torch.Tensor:
        """k, v [T, H, D] -> uint8 cells [T, H, cell_bytes(D)]."""
        pk = self.pack(*self.encode(k, Wk))
        pv = self.pack(*self.encode(v, Wv))
        nc, ns = self.D // 2, 2 * self.nv
        return torch.cat([pk[..., :nc], pv[..., :nc], pk[..., nc:nc + ns], pv[..., nc:nc + ns],
                          pk[..., nc + ns:], pv[..., nc + ns:]], -1)

    # ------------------------------------------------------------------ decoder
    def decode_kind(self, packed: torch.Tensor, scales_b: torch.Tensor, side_b: torch.Tensor) -> torch.Tensor:
        """packed uint8 [M,H,D/2], scales_b uint8 [M,H,2nv], side_b uint8 [M,H,2] -> fp32 [M,H,D] original basis."""
        M, H = packed.shape[:2]
        D = self.D
        lo = (packed & 15).to(torch.int8) - 8
        hi = (packed >> 4).to(torch.int8) - 8
        c = torch.stack([lo, hi], -1).reshape(M, H, D).float()
        s = scales_b.contiguous().view(torch.float16).float()                         # [M, H, nv]
        side = side_b.contiguous().view(torch.int16).to(torch.int32)[..., 0] & 0xFFFF  # [M, H]
        y = _deq(c, s)
        ch = ((side[..., None] >> self.shifts) & self.lmask).long().repeat_interleave(RB, -1)  # [M, H, D]
        mode = ((side >> self.hshift) & self.hmask).long()[..., None]                         # [M, H, 1]
        ar = torch.arange(D, device=y.device)
        h32 = fwht(y, RB)
        z = torch.where(ch == 0, y, h32 * self.local_signs[ch, ar])                     # per-vector domain
        hD = fwht(y, D) * self.head_signs[mode.squeeze(-1)]                              # whole-head candidates
        out = z
        for i, h in enumerate(self.head_names[1:], 1):
            out = torch.where(mode == i, z.index_select(-1, self.inv_perm) if h == "P" else hD, out)
        return out

    def decode_cells(self, cells: torch.Tensor, dtype=torch.bfloat16):
        """uint8 cells [M, H, C] -> K, V [M, H, D] in the original basis."""
        o = self.off
        nc, ns = self.D // 2, 2 * self.nv
        k = self.decode_kind(cells[..., o["k_codes"]:o["k_codes"] + nc], cells[..., o["k_scales"]:o["k_scales"] + ns],
                             cells[..., o["k_side"]:o["k_side"] + 2])
        v = self.decode_kind(cells[..., o["v_codes"]:o["v_codes"] + nc], cells[..., o["v_scales"]:o["v_scales"] + ns],
                             cells[..., o["v_side"]:o["v_side"] + 2])
        return k.to(dtype), v.to(dtype)
