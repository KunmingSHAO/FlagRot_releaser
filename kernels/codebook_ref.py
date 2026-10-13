"""PyTorch reference of the per-token transform codebook encoder of a 4-bit KV cache (FlagRot, dynamic codebook):
blockflag/qattention.py dyn_quant with rule 'cq' (D = 128, rb = 32, INT4 groups of 32), evaluated in a fixed fp32
operation order that the CUDA writer (csrc/attn_codebook.cu) reproduces bit-exactly.

Configurations (kv_dyn of blockflag/run.py):
  final    'cq+f+S+Q5+FS+R5' (the locked method)  KV["dyn"] = dict(rule="cq", local=["H", "HS", "HS2", .., "HS6"],
                          head=["F", "FS", "FS2", .., "FS6"]): 8 candidates per 32-vector x 8 per head, no P
  full     'cq+S+FS+P+F'  KV["dyn"] = dict(rule="cq", local=["H", "HS"], head=["P", "F", "FS"])  (earlier codebook)
  reduced  'cq+f'         KV["dyn"] = dict(rule="cq", local=["H"], head=["F"])

Per (token, kv head, K or V) row x [128] (fp32 of the bf16 input):
  local candidates per 32-vector j (z = x, or z = x[PERM] for the head candidate P):
      I : y = z                     rec = deq(Q(y))
      H : y = H32 z                 rec = H32 deq(Q(y))
      HSk: y = H32 (S_k * z)        rec = S_k * H32 deq(Q(y))         (S_k = _signs(128, seed k) (HS: k = 0), the
                                                                        32-slice of the 128-vector, in z coords)
  -> per vector the candidate with the lowest local error (first index on ties, in list order: I < H < HS < HS2 ..)
  head candidates: base (local choices on x), P (local choices on x[PERM], scattered back), F (y = H128 x), FSk
  (y = H128 (S_k * x), rec = S_k * H128 deq(Q(y))) -> lowest head error (first index on ties: base < P < F < FS ..).
  H_n = orthonormal Sylvester Hadamard (ref.fwht: butterflies (a+b, a-b), h = 1, 2, 4, ..., then * fp32(1/sqrt(n))).
  Q = the cache's INT4 quantizer (ref.int4_quant): per 32-group s = fp16(amax * fp32(1/7)), q = clamp(rne(y / s), -7, 7);
      deq = q * s (exact).  (blockflag's int4:32 keeps s = amax / 7 in fp32; quant='blockflag' selects it.)
      konlya (asymmetric; quant='asym'): the errors use blockflag int4a:32 (FP32 scale, unbounded zero point,
      deq = (q - z) s); the stored codes / scale word of the chosen vector are blockflag int4ah:32 (range widened to
      include 0, scale word = FP16 whose low 4 bits are z; formats.quant_asym_dep / scale_word): q in [0, 15], s = the
      word's value (z = its low 4 bits).
Errors (e = rec - z, fp32), with C = calibrated query second moment C_q [Hkv, 128, 128] (keys) and c_o [Hkv, 128]
(values), in the coordinates of z (C[PERM][:, PERM], c_o[PERM] for P):
  keys, local  err_j = tree( e_a * u_a ),  u_a = seq_{k=0..31} C_jj[a, k] e_k
  keys, head   err   = tree( e_i * u_i ),  u_i = ((U_0 + U_1) + U_2) + U_3,  U_b = seq_{k in [32b, 32b+32)} C[i, k] e_k
  values       err   = tree( (e_i * e_i) * c_i )   over the 32-vector (local) or the 128 elements (head)
  seq = sequential fp32 sum starting with the first product, every product and sum rounded separately (no FMA);
  tree = halving tree: v[:n/2] + v[n/2:] until one element.
Stored per row: the INT4 codes of the chosen transformed vector y (in the coordinates of the chosen candidate:
x[PERM] order for P), its 4 fp16 scales, and the choice:
  final:   16-bit side word  bits 3j..3j+2 = local choice of vector j (0 I, 1 H, 2 HS, 3..7 HS2..HS6; 0 unless base),
                             bits 12..14 = head mode (0 base, 1 F, 2 FS, 3..7 FS2..FS6)
  full:    16-bit side word  bits 2j..2j+1 = local choice of vector j (0 I, 1 H, 2 HS; 0 for F / FS),
                             bits 8..9 = head mode (0 base, 1 P, 2 F, 3 FS)
  reduced: local bit of vector j (H) = sign bit of fp16 scale j, head bit (F) in a per-page bit plane.
"""
from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import ref as R  # noqa: E402
from blockflag.qattention import _perm, _signs  # noqa: E402

D, RB, NG = 128, 32, 4
SEEDS = (0,) + tuple(range(2, 16))    # sign table: S_0 (HS / FS), S_2 .. S_15 (HSk / FSk); entry i <-> SEEDS[i]
SGS = {k: _signs(D, "cpu", k) for k in SEEDS}
SG = SGS[0]                           # float +-1 [128]
PERM = _perm(D, "cpu")                # long [128]: z = x[PERM] (RoPE pairs (i, i+64) interleaved)
IPERM = torch.argsort(PERM)
CFG = {"final": dict(local=["H", "HS"] + [f"HS{k}" for k in range(2, 7)], head=["F", "FS"] + [f"FS{k}" for k in range(2, 7)]),
       "head8": dict(local=["H"], head=["F", "FS"] + [f"FS{k}" for k in range(2, 8)]),     # cq+f+FS+R6
       "head16": dict(local=["H"], head=["F", "FS"] + [f"FS{k}" for k in range(2, 16)]),   # cq+f+FS+R14
       "full": dict(local=["H", "HS"], head=["P", "F", "FS"]), "reduced": dict(local=["H"], head=["F"]),
       "konly": dict(local=["H"], head=["F"]), "konlya": dict(local=["H"], head=["F"]),
       "konlya2": dict(local=["H"], head=["F"]),
       "q4x4": dict(local=["H", "HS", "HS2"], head=["F", "FS", "FS2"])}   # cq+f+S+Q1+FS+R1 (4 x 4)
CFG["q4x4k"], CFG["finalk"] = CFG["q4x4"], CFG["final"]   # '+konly': keys as 4 x 4 / 8 x 8, values without transform
KONLY = ("konly", "konlya", "konlya2", "q4x4k", "finalk")   # cq+f+konly: keys as reduced, values without transform (encode)
NOXF = dict(local=[], head=[])         # identity only (konly values)
FAMILY = ("final", "head8", "head16", "q4x4", "q4x4k", "finalk")  # sign-flipped-Hadamard codebooks (no P): side word LB / HB bits
MODES = {k: ["base"] + CFG[k]["head"] for k in FAMILY}
MODES.update({"full": ["base", "P", "F", "FS"], "reduced": ["base", "P", "F", "FS"], "konly": ["base", "P", "F", "FS"],
              "konlya": ["base", "P", "F", "FS"], "konlya2": ["base", "P", "F", "FS"]})
LBITS = {"final": 3, "head8": 1, "head16": 1, "q4x4": 2, "q4x4k": 2, "finalk": 3}       # bits per vector choice
HBITS = {"final": 3, "head8": 4, "head16": 5, "q4x4": 2, "q4x4k": 2, "finalk": 3}       # bits of the head mode (at bit 4 LB)


def seed_of(name):
    """'HS' / 'FS' -> 0, 'HS3' / 'FS3' -> 3."""
    return int(name[2:] or 0)


def sign_bits():
    """the sign vectors S_k (k in SEEDS) as 6 x 4 uint32 words (bit d set iff S_k[d] < 0) for the kernels."""
    out = []
    for k in SEEDS:
        b = (SGS[k] < 0).to(torch.int64)
        out += [int(sum(int(b[32 * w + i]) << i for i in range(32))) for w in range(4)]
    return out


def quant(y, kind="cache"):
    """y fp32 [..., D] -> (q int8 [..., D] in [-7, 7], scale fp32 [..., D / 32] (fp16 values for 'cache'), deq fp32).
    kind 'asym': deq of blockflag int4a:32 (the choice's error), q in [0, 15] / scale-word values of int4ah:32 (stored).
    (D = 128, or 256 for the key-only 2x2 kinds)"""
    b = y.reshape(*y.shape[:-1], y.shape[-1] // RB, RB)
    if kind in ("asym", "asym2"):   # asym2: int2a:32 choice / int2ah:32 storage (vLLM plugin Phase 3a, konlya2)
        import blockflag.formats as FM
        nb = 2 if kind == "asym2" else 4
        lv = 2 ** nb - 1
        mn, mx = b.amin(-1, keepdim=True), b.amax(-1, keepdim=True)
        s = ((mx - mn) / lv).clamp_min(1e-12)
        z = torch.round(-mn / s)
        qa = (torch.round(b / s) + z).clamp(0, lv)
        deq = ((qa - z) * s).reshape(y.shape)
        q, zs, ss, _ = FM.quant_asym_dep(y, RB, False, nb)
        sw = FM.word_value(FM.scale_word(ss, zs, nb))
        return q.to(torch.int8).reshape(y.shape), sw.squeeze(-1), deq
    if kind == "cache":
        s = (b.abs().amax(-1, keepdim=True) * (1.0 / 7.0)).half().float()
    else:  # blockflag int4:32 (FM.fake): fp32 scale amax / 7, clamp_min 1e-12
        s = b.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7.0
    q = torch.where(s != 0, torch.round(b / torch.where(s != 0, s, torch.ones_like(s))), torch.zeros_like(b)).clamp(-7, 7)
    return q.to(torch.int8).reshape(y.shape), s.squeeze(-1), (q * s).reshape(y.shape)


def tree(p):
    while p.shape[-1] > 1:
        n = p.shape[-1] // 2
        p = p[..., :n] + p[..., n:]
    return p[..., 0]


def seq_matvec(M, e, k0, k1):
    """u_i = seq_{k in [k0, k1)} M[..., i, k] * e[..., k] (M [H, n, K] broadcast over e [T, H, K])."""
    u = M[None, :, :, k0] * e[:, :, None, k0]
    for k in range(k0 + 1, k1):
        u = u + M[None, :, :, k] * e[:, :, None, k]
    return u


def local_err(e, W, kind):
    """e [T, H, D] (z coords) -> [T, H, D / 32] per 32-vector errors; W: C_z [H, D, D] (keys) or c_z [H, D]."""
    out = []
    for j in range(e.shape[-1] // RB):
        ej = e[..., RB * j:RB * (j + 1)]
        if kind == "k":
            u = seq_matvec(W[:, RB * j:RB * (j + 1), RB * j:RB * (j + 1)], ej, 0, RB)
            out.append(tree(ej * u))
        else:
            out.append(tree((ej * ej) * W[None, :, RB * j:RB * (j + 1)]))
    return torch.stack(out, -1)


def head_err(e, W, kind):
    if kind == "k":  # u = ((U_0 + U_1) + U_2) + ... (left fold over the D / 32 column blocks)
        U = [seq_matvec(W, e, RB * b, RB * (b + 1)) for b in range(e.shape[-1] // RB)]
        u = U[0]
        for b in range(1, len(U)):
            u = u + U[b]
        return tree(e * u)
    return tree((e * e) * W[None])


def first_argmin(errs):
    """errs [nc, ...] -> index of the first minimum (strict < in candidate order; = torch.argmin for finite values)."""
    best, idx = errs[0], torch.zeros_like(errs[0], dtype=torch.long)
    for c in range(1, errs.shape[0]):
        lt = errs[c] < best
        best = torch.where(lt, errs[c], best)
        idx = torch.where(lt, torch.full_like(idx, c), idx)
    return idx


def local_best(z, W, kind, local, qk):
    """z [T, H, 128] -> dict(q, s, rec, e, ch [T, H, 4]) of the per-vector best local candidate (z coords)."""
    cands = []
    q, s, d = quant(z, qk)
    cands.append((q, s, d))
    for t in local:
        if t == "H":
            q, s, d = quant(R.fwht(z, RB), qk)
            cands.append((q, s, R.fwht(d, RB)))
        elif t.startswith("HS"):
            sg = SGS[seed_of(t)].to(z.device)
            q, s, d = quant(R.fwht(z * sg, RB), qk)
            cands.append((q, s, sg * R.fwht(d, RB)))
    errs = torch.stack([local_err(rec - z, W, kind) for (_, _, rec) in cands])   # [nc, T, H, 4]
    ch = first_argmin(errs)
    sel = ch.repeat_interleave(RB, -1)
    q, s, rec = cands[0][0].clone(), cands[0][1].clone(), cands[0][2].clone()
    for i in range(1, len(cands)):
        q = torch.where(sel == i, cands[i][0], q)
        rec = torch.where(sel == i, cands[i][2], rec)
        s = torch.where(ch == i, cands[i][1], s)
    return dict(q=q, s=s, rec=rec, e=rec - z, ch=ch, errs=errs)


@torch.no_grad()
def encode(x, C, co, kind, cfg="full", qk="cache"):
    """x [T, H, 128] (bf16 or fp32) of one kind ('k' / 'v'); C [H, 128, 128] fp32 (keys), co [H, 128] fp32 (values).
    Returns dict: q int8 [T, H, 128] (stored coords), s fp32 [T, H, 4] (positive), local int [T, H, 4], mode int
    [T, H] (0 base, 1 P, 2 F, 3 FS), side int32 [T, H] (full side word), rec fp32 [T, H, 128] (original coords),
    head_errs [nh, T, H]."""
    c = NOXF if (cfg in KONLY and kind == "v") else CFG[cfg]
    x = x.float()
    dev = x.device
    sg, pm, ipm = SG.to(dev), PERM.to(dev), IPERM.to(dev)
    W = C.float() if kind == "k" else co.float()
    Wp = W[:, pm][:, :, pm] if kind == "k" else W[:, pm]
    base = local_best(x, W, kind, c["local"], qk)
    heads = [("base", base["q"], base["s"], base["rec"], base["ch"])]
    if "P" in c["head"]:
        P = local_best(x[..., pm], Wp, kind, c["local"], qk)
        heads.append(("P", P["q"], P["s"], P["rec"][..., ipm], P["ch"]))
    zero = torch.zeros_like(base["ch"])
    Dx = x.shape[-1]
    assert Dx == D or cfg in ("konly", "konlya", "konlya2"), "head dim 256: key-only 2x2 kinds only"
    if "F" in c["head"]:
        q, s, d = quant(R.fwht(x, Dx), qk)
        heads.append(("F", q, s, R.fwht(d, Dx), zero))
    for hn in [h for h in c["head"] if h.startswith("FS")]:
        sgk = SGS[seed_of(hn)].to(dev)
        q, s, d = quant(R.fwht(x * sgk, D), qk)
        heads.append((hn, q, s, sgk * R.fwht(d, D), zero))
    herr = torch.stack([head_err(h[3] - x, W, kind) for h in heads])         # [nh, T, H]
    hsel = first_argmin(herr)
    q, s, rec, ch = heads[0][1].clone(), heads[0][2].clone(), heads[0][3].clone(), heads[0][4].clone()
    mode = torch.zeros_like(hsel)
    for i in range(1, len(heads)):
        m = hsel == i
        q = torch.where(m[..., None], heads[i][1], q)
        s = torch.where(m[..., None], heads[i][2], s)
        rec = torch.where(m[..., None], heads[i][3], rec)
        ch = torch.where(m[..., None], heads[i][4], ch)
        mode = torch.where(m, torch.full_like(mode, MODES[cfg].index(heads[i][0])), mode)
    if cfg in FAMILY:
        lb = LBITS[cfg]
        side = (ch[..., 0] | (ch[..., 1] << lb) | (ch[..., 2] << 2 * lb) | (ch[..., 3] << 3 * lb) | (mode << 4 * lb)).to(torch.int32)
    elif Dx == D:
        side = (ch[..., 0] | (ch[..., 1] << 2) | (ch[..., 2] << 4) | (ch[..., 3] << 6) | (mode << 8)).to(torch.int32)
    else:  # (key-only kinds keep their choices in the scale sign bits: no side word)
        side = (mode << 8).to(torch.int32)
    return dict(q=q, s=s, local=ch, mode=mode, side=side, rec=rec, head_errs=herr)


def reconstruct_final(q, s, local, mode, dtype=torch.float64):
    """sign-flipped-Hadamard family (final / head8 / head16): local 0 I, 1 H, l >= 2 HS(seed SEEDS[l-2]);
    mode 0 base, 1 F, m >= 2 FS(seed SEEDS[m-2])."""
    dev = q.device
    d = (q.to(dtype).reshape(*q.shape[:-1], NG, RB) * s.to(dtype)[..., None])
    Hn = {n: R.fwht(torch.eye(n, dtype=dtype, device=dev), n) for n in (RB, D)}
    sg = torch.stack([torch.ones(D, dtype=dtype)] + [SGS[k].to(dtype) for k in SEEDS]).to(dev)   # [1 + n, 128]: 0 = none
    dH = d @ Hn[RB]
    nrow = len(SEEDS) + 1                                                  # (garbage rows of unused pages: clamped)
    li = (local - 1).clamp(0, nrow - 1)                                    # sign row: 0 for H, k-1 for HS..
    sl = sg.view(-1, NG, RB)[li, torch.arange(NG, device=dev).expand_as(li)]  # [..., 4, 32]
    loc = torch.where((local == 0)[..., None], d, sl * dH).reshape(*q.shape)
    full = d.reshape(*q.shape) @ Hn[D]
    sh = sg[(mode - 1).clamp(0, nrow - 1)]
    return torch.where((mode == 0)[..., None], loc, sh * full)


def zero_points(s, bits=4):
    """konlya: zero points [..., 4] = the low 4 bits of the stored scale words (s = their FP16 values); konlya2: 2 bits."""
    return (s.half().view(torch.int16).long() & ((1 << bits) - 1))


def reconstruct(q, s, local, mode, dtype=torch.float64):
    """inverse transforms of stored rows: q [..., 128] codes (stored coords), s [..., 4], local [..., 4] (0 I, 1 H,
    2 HS), mode [...] -> reconstruction [..., 128] in the original basis (dtype math; the attention reference).
    (Codebooks full / reduced; the final codebook: reconstruct_final.)"""
    dev = q.device
    Dq = q.shape[-1]
    d = (q.to(dtype).reshape(*q.shape[:-1], Dq // RB, RB) * s.to(dtype)[..., None])
    Hn = {n: R.fwht(torch.eye(n, dtype=dtype, device=dev), n) for n in (RB, Dq)}  # orthonormal Sylvester (symmetric)
    if Dq != D:  # key-only kinds at D = 256: modes 0 (base: I / H32 per vector) and 2 (F = H_D) only
        loc = torch.where((local == 1)[..., None], d @ Hn[RB], d).reshape(*q.shape)
        return torch.where((mode == 2)[..., None], d.reshape(*q.shape) @ Hn[Dq], loc)
    sg, ipm = SG.to(dev, dtype), IPERM.to(dev)
    dH = d @ Hn[RB]                                                       # per 32-block H32
    loc = torch.where((local == 0)[..., None], d, torch.where((local == 1)[..., None], dH, sg.view(NG, RB) * dH))
    loc = loc.reshape(*q.shape)
    full = d.reshape(*q.shape) @ Hn[D]
    out = torch.where((mode == 0)[..., None], loc, torch.zeros_like(loc))
    out = torch.where((mode == 1)[..., None], loc[..., ipm], out)
    out = torch.where((mode == 2)[..., None], full, out)
    out = torch.where((mode == 3)[..., None], sg * full, out)
    return out
