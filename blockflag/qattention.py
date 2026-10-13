"""Attention with a quantized KV cache and vector-level rotation flags (registered as attn_implementation='blockflag').

K and V are quantized after RoPE along head_dim, as a KV cache stores them.  Each (layer, kv-head, rb-vector) has a
static flag; a flagged K vector is rotated together with the same vector of every query head of its group
(Q K^T unchanged), a flagged V vector is rotated back after P V (i.e. H folds into o_proj offline).  Q, the scores
and P stay FP32, so only the KV-cache quantization error is measured.

Calibration ('greedy'), per unit of U = min(group, head_dim) elements holding U/rb vectors, pattern p as in qlinear:
  K error: sum_t e_t^T C_q e_t / (k_t^T C_q k_t),  C_q = sum of q q^T over the kv-head's query heads (score error)
  V error: sum_t e_t^T diag(c_o) e_t / ||v_t||^2,  c_o = column energy of o_proj for that head
(cross-unit terms of C_q are dropped).
"""
from __future__ import annotations

import re

import torch
from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

from . import formats as FM
from .hadamard import fwht, pattern_mask, sel_fwht, sylvester
from .modeling import attn_layers, head_dim, text_config
from .qlinear import STATE

KV: dict = {"k": "bf16", "v": "bf16", "rb": 64, "calib_on": False, "acc": {}, "kmask": None, "vmask": None, "co": {},
            "k_axis": "head", "resid": 0, "sink": 0, "flush": 0, "accfull": {}, "kfull": None, "vfull": None,
            "sel": None, "accx": {}, "measure": None, "meas": {}, "kx": None, "vx": None,
            "dyn": None, "cq": {}, "dynstat": {}, "qs": {}, "lz": {}, "qscale": 1.0, "dynnames": [],
            "xf": None, "xf_calib": None, "xs": {}, "skip": 0, "nl": 0, "bf16_layers": None, "resid_mode": "sliding",
            "kvq": None, "kvq_calib": False, "kvq_cb": {}, "kvq_thr": {}, "kvq_acc": {}, "kvqstat": {}, "rope": None,
            "kc": None, "kc_calib": False, "kc_acc": {}, "kc_meas": False, "kc_err": {}, "kc_frac": {},
            "rk": None, "rk_calib": False, "rk_acc": {}, "rk_perm": {}, "rk_k": {}, "rk_piv": {}, "rk_cur": None,
            "rk_stat": {}, "rk_W": {}, "rk_M": {}}
# RotateKV (kv_rotatekv; Su et al., IJCAI 2025): rk = dict(G=heads per grouped-head Hadamard,
# ro=channel reordering on, ma=top-n massive-activation sink tokens or 0); keys are quantized PRE-RoPE inside the model's
# own apply_rotary_pos_emb (rk_install) -- rk_k[layer] holds the dequantized keys after RoPE for the attention;
# rk_acc[layer] = calibration sum of the rotated pre-RoPE keys [Hkv * D] (FP64), rk_perm[layer] = argsort of it;
# rk_piv[layer] = bool [B, S] sink tokens of the current forward (top-n |x| of the layer's input residual stream)
# static key centering (kv_center='post'): kc = {layer: mu [Hkv, D]} mean post-RoPE calibration key; while kc_calib the
# attention accumulates kc_acc[layer] = (sum of keys [Hkv, D] FP64, count).  Every key of a quantized layer (and the keys
# seen by the calibration of flags / transforms) is replaced by k - mu: q.(k - mu) = q.k - q.mu shifts all logits of a
# query by the same amount, so attention is unchanged in exact arithmetic and only the quantization error changes.
# kvq = 'nuq' | 'nuq+ds': KVQuant-style cache (see run.py); kvq_acc[layer] collects the calibration pre-RoPE keys and
# values, kvq_thr[layer] = (lo, hi) [Hkv, D] key outlier thresholds, kvq_cb[layer] = {'k', 'v': 16 sorted levels in
# [-1, 1]}, kvqstat = outlier counts on the evaluation windows; rope = the model's rotary embedding module
# Token-axis (per-channel) key groups start after the kv_sink BF16 sink tokens, which are not part of any group.
# static calibrated transforms (kv_xform): xf = {layer: {"k": [Hkv, D, D], "q": [Hkv, D, D], "v": ..., "vinv": ...}}
# row-vector matrices: k' = k @ xf.k is quantized and q' = q @ xf.q (xf.q = xf.k^-T, so q'.k' = q.k exactly);
# v' = v @ xf.v is quantized and the attention output is mapped back with o = o' @ xf.vinv (xf.vinv = xf.v^-1).
# xf_calib = 'wush' | 'smooth' while the calibration windows run; xs[layer] accumulates the statistics.
# skip = n: the first and the last n layers keep a BF16 cache (vLLM TurboQuant's boundary-layer protection);
# bf16_layers = their layer_idx, counted over the KV-cache layers (the full-attention layers of a hybrid model); set by
# run.build; None = layers 0..skip-1 and nl-skip..nl-1.
# dynamic flags (kv_flags='dyn'): dyn = dict(rule='mse'|'cq', full=bool).  At cache-write time every rb-vector of every
# token is quantized with and without H_rb and the reconstruction with the lower error is kept (one bit per vector and
# token); with full=True the head may instead take the whole-head Hadamard for that token.  'cq' weights the key error
# with the calibrated query second moment cq[layer] [Hkv, D, D] (static; the query is unknown at write time) and the
# value error with the o_proj column energy.  dynstat[kind][layer] = (count of rotated (token, vector), tokens) and
# per-vector rotation counts for the consistency analysis (is the oracle bit the same for every token?)
# design-space mode (kv_flags='sel'): sel = dict(rbs=[...], proxy='score'|'mse'); pass 1 fills accx[kind][rb][layer]
# (per-position pattern errors at every candidate block size), pass 2 measures each candidate transform of a head
# (measure[kind][layer] = {name: {rb: mask}}) with the full head-level proxy into meas[kind][layer][name] [Hkv];
# kx / vx = {layer: {rb: mask [Hkv, D/rb]}} the chosen per-head transforms (rb = D: whole head)
# kfull / vfull: {layer: bool [Hkv]} heads rotated by the whole-head Hadamard H_D instead of their rb-block bits
# (decision space of kv_flags='headsel': per head none | rb-block bits | whole head; the two are never combined)


CHUNK = {"limit": 4096 * 4096, "elems": 1 << 26}   # chunk queries when S * Sk exceeds limit (tests lower it)


def _repeat(x, n):
    if n == 1:
        return x
    b, h = x.shape[:2]
    return x[:, :, None].expand(b, h, n, *x.shape[2:]).reshape(b, h * n, *x.shape[2:])


def rot(x: torch.Tensor, mask: torch.Tensor | None, full: torch.Tensor | None, rb: int, n_rep: int = 1) -> torch.Tensor:
    """x [B, H, S, D] with H = n_rep * Hkv: rb-block bits (mask [Hkv, D/rb]) and whole-head rotation (full [Hkv])."""
    y = sel_fwht(x, None if mask is None else mask.repeat_interleave(n_rep, 0)[None, :, None, :], rb)
    if full is not None and bool(full.any()):
        y = sel_fwht(y, full.repeat_interleave(n_rep, 0)[None, :, None, None], x.shape[-1])
    return y


def rotx(x: torch.Tensor, md: dict | None, n_rep: int = 1) -> torch.Tensor:
    """apply {rb: mask [Hkv, D/rb]} (per head at most one rb is active, so the order is irrelevant)."""
    for r, m in (md or {}).items():
        x = sel_fwht(x, m.repeat_interleave(n_rep, 0)[None, :, None, :], r)
    return x


def _perm(D: int, device) -> torch.Tensor:
    """RoPE-pair interleaving (HF rotate_half pairs channel i with i + D/2): blocks then hold both halves of each pair."""
    h = torch.arange(D // 2, device=device)
    return torch.stack([h, h + D // 2], 1).reshape(-1)


def _signs(D: int, device, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(1234 + D + 7919 * seed)
    return (torch.randint(0, 2, (D,), generator=g) * 2 - 1).float().to(device)


def _weights(kind: str, li: int, x: torch.Tensor, perm=None):
    """per-token weighting of the error under the dynamic rule, in the (optionally permuted) coordinates of x.
    mse: identity; cq: static query second moment (keys) / o_proj column energy (values);
    att: ScaNN-style score-aware weighting -- every calibration query q is weighted by the attention probability it
    would give this key, w_q(x) = exp(s q.x - logZ_q), so the key error is sum_q w_q (q.e)^2 and the value error is
    (sum_q w_q) * sum_i c_o,i e_i^2."""
    rule = KV["dyn"]["rule"]
    H = x.shape[1]
    pi = (lambda t, dim: t.index_select(dim, perm)) if perm is not None else (lambda t, dim: t)
    if kind == "k" and KV.get("oracle_ctx") is not None and perm is None:
        return ("oracle", KV["oracle_ctx"])    # diagnostic (default off): the real queries of this window, see below
    if rule == "mse":
        return ("mse", None)
    if rule == "cq":
        X = KV["xf"][li] if KV.get("xf") is not None and li in KV["xf"] else None
        if kind == "k":
            C = KV["cq"][li]
            if X is not None and "q" in X:                 # static transform first (e.g. WUSH-KV): weights follow q' = q X_q
                C = torch.einsum("hdi,hde,hej->hij", X["q"], C, X["q"])
            return ("mat", pi(pi(C, 1), 2))
        if X is not None and "vinv" in X:                  # value error e' maps back through X_vinv before W_o
            M = torch.einsum("hid,hd,hjd->hij", X["vinv"], KV["co"][li], X["vinv"])
            return ("mat", pi(pi(M, 1), 2))
        return ("diag", pi(KV["co"][li], 1)[None, :, None, :])
    qs, lz, sc = KV["qs"][li], KV["lz"][li], KV["qscale"]               # [H, nq, D], [H, nq]
    X = KV["xf"][li] if KV.get("xf") is not None and li in KV["xf"] else None
    if X is not None and "q" in X:      # static transform (kivi_c): x is k' = k X_k, so the queries become q' = q X_q
        qs = torch.einsum("hnd,hde->hne", qs, X["q"])                  # (q'.k' = q.k: logits and errors as untransformed)
    if KV.get("att_w"):                 # opt-in (calibration sensitivity): no cap | per-query softmax over the window
        lg = sc * torch.einsum("bhsd,hqd->bhsq", x if perm is None else x.index_select(-1, torch.argsort(perm)), qs)
        w = torch.exp(lg - lz[None, :, None, :]) if KV["att_w"] == "nocap" else torch.softmax(lg, 2)
    else:
        w = torch.exp(sc * torch.einsum("bhsd,hqd->bhsq", x if perm is None else x.index_select(-1, torch.argsort(perm)), qs)
                      - lz[None, :, None, :]).clamp_max(1.0)            # attention probability this key would get
    if kind == "k":
        return ("att", (w, pi(qs, 2)))
    if X is not None and "vinv" in X:   # value error e' maps back through X_vinv before W_o (as the cq rule)
        M = torch.einsum("hid,hd,hjd->hij", X["vinv"], KV["co"][li], X["vinv"])
        return ("wmat", (w.sum(-1, keepdim=True), pi(pi(M, 1), 2)))
    return ("diag", w.sum(-1, keepdim=True) * pi(KV["co"][li], 1)[None, :, None, :])


def _vec_err(e: torch.Tensor, W, rb: int) -> torch.Tensor:
    """e [B, H, S, D] -> per rb-vector error [B, H, S, D/rb] (block-diagonal part of the weighting)."""
    B, H, S, D = e.shape
    nv = D // rb
    ev = e.reshape(B, H, S, nv, rb)
    kindw, M = W
    if kindw == "mse":
        return ev.square().sum(-1)
    if kindw == "diag":
        return (ev.square() * M.reshape(*M.shape[:-1], nv, rb)).sum(-1)
    if kindw == "mat":
        Cb = torch.stack([M[:, j * rb:(j + 1) * rb, j * rb:(j + 1) * rb] for j in range(nv)], 1)
        return torch.einsum("bhsvi,hvij,bhsvj->bhsv", ev, Cb, ev)
    if kindw == "wmat":                                                 # per-token scale x static matrix
        ws, M_ = M
        Cb = torch.stack([M_[:, j * rb:(j + 1) * rb, j * rb:(j + 1) * rb] for j in range(nv)], 1)
        return ws * torch.einsum("bhsvi,hvij,bhsvj->bhsv", ev, Cb, ev)
    if kindw == "oracle":                                               # sum_i W_it (q_i[slot] . e_t[slot])^2
        q, Wt = M
        Wt = Wt[..., -S:]
        return torch.stack([(Wt * torch.einsum("bhrid,bhtd->bhrit", q[..., j * rb:(j + 1) * rb], ev[:, :, :, j]).square())
                            .sum((2, 3)) for j in range(nv)], -1)
    w, qs = M                                                           # att
    proj = torch.einsum("bhsvi,hqvi->bhsqv", ev, qs.reshape(H, -1, nv, rb))
    return (w[..., None] * proj.square()).sum(3)


def _head_err(e: torch.Tensor, W) -> torch.Tensor:
    kindw, M = W
    if kindw == "mse":
        return e.square().sum(-1)
    if kindw == "diag":
        return (e.square() * M).sum(-1)
    if kindw == "mat":
        return torch.einsum("bhsi,hij,bhsj->bhs", e, M, e)
    if kindw == "wmat":
        return M[0][..., 0] * torch.einsum("bhsi,hij,bhsj->bhs", e, M[1], e)
    if kindw == "oracle":
        q, Wt = M
        return (Wt[..., -e.shape[2]:] * torch.einsum("bhrid,bhtd->bhrit", q, e).square()).sum((2, 3))
    w, qs = M
    return (w * torch.einsum("bhsd,hqd->bhsq", e, qs).square()).sum(-1)


def _fake_tok(x: torch.Tensor, spec: str) -> torch.Tensor:
    """KIVI-style quantization of [B, H, S, D] along the token axis (per channel, groups of g consecutive tokens; the
    last partial group is padded by repeating its last token).  With kv_sink > 0 the sink tokens (kept in BF16) are left
    out and the groups start right after them."""
    ns = min(KV["sink"], x.shape[2] - 1) if KV["sink"] else 0
    if ns:
        return torch.cat([x[:, :, :ns], _fake_tok_(x[:, :, ns:], spec)], 2)
    return _fake_tok_(x, spec)


def _fake_tok_(x: torch.Tensor, spec: str) -> torch.Tensor:
    tt = x.transpose(2, 3)
    S = tt.shape[-1]
    g = FM.group_of(spec, S)
    pad = (-S) % g
    if pad:
        tt = torch.cat([tt, tt[..., -1:].expand(*tt.shape[:-1], pad)], -1)
    return FM.fake(tt, spec)[..., :S].transpose(2, 3)


def _pool_tokens(e: torch.Tensor, tg: int, dim: int = 3) -> torch.Tensor:
    """sum errors over groups of tg consecutive tokens (dim = token axis) and broadcast the sums back, so an argmin
    taken afterwards is one decision per token group (the unit a KIVI-style encoder quantizes at once)."""
    if tg <= 1:
        return e
    e = e.movedim(dim, -1)
    S = e.shape[-1]
    pad = (-S) % tg
    ep = torch.cat([e, e.new_zeros(*e.shape[:-1], pad)], -1) if pad else e
    sm = ep.reshape(*ep.shape[:-1], -1, tg).sum(-1).repeat_interleave(tg, -1)[..., :S]
    return sm.movedim(-1, dim)


def _local_best(x, spec, rb, W, local, Q=None, tg: int = 1):
    """per rb-vector the lowest-error candidate among identity and the local transforms ('H', 'HS'); returns the
    reconstruction (same coordinates as x), its per-vector choice index and the per-vector error."""
    D = x.shape[-1]
    Q = Q or (lambda y: FM.fake(y, spec))
    recs = [Q(x)]
    for t in local:
        if t == "H":
            recs.append(fwht(Q(fwht(x, rb)), rb))
        elif t.startswith("HS"):                          # HS, HS2, HS3, ...: H_rb after different fixed sign flips
            sg = _signs(D, x.device, int(t[2:] or 0))
            recs.append(sg * fwht(Q(fwht(x * sg, rb)), rb))
    errs = _pool_tokens(torch.stack([_vec_err(r - x, W, rb) for r in recs]), tg, 3)   # [nc, B, H, S, nv]
    ch = errs.argmin(0)
    rb_ = rb
    sel = ch[..., None].expand(*ch.shape, rb_).reshape(x.shape)
    r = recs[0]
    for i in range(1, len(recs)):
        r = torch.where(sel == i, recs[i], r)
    return r, ch


def dyn_quant(t: torch.Tensor, spec: str, rb: int, kind: str, li: int, axis: str = "head") -> torch.Tensor:
    """dynamic per-(token, vector) transform choice at cache-write time; returns the reconstruction in the original
    basis (what attention sees once queries / outputs use the same per-token choice).  Per vector: identity or the
    local transforms; per (token, head) additionally: the same in RoPE-pair-interleaved order ('P'), the whole-head
    Hadamard ('F') or the whole-head Hadamard after a fixed sign flip ('FS')."""
    if axis == "token" and KV["sink"] and t.shape[2] > KV["sink"]:
        return _after_sinks(lambda u: dyn_quant(u, spec, rb, kind, li, axis), t)
    x = t.float()
    D = x.shape[-1]
    cfg = KV["dyn"]
    # axis = token (KIVI keys): quantization groups run along tokens, so the choice is made once per group of g tokens
    # -- the encoding step a KIVI-style cache performs when a group leaves the full-precision residual window
    Q = (lambda y: _fake_tok(y, spec)) if axis == "token" else (lambda y: FM.fake(y, spec))
    tg = FM.group_of(spec, x.shape[2]) if axis == "token" else 1
    # packed encoding (cfg["group"] = g): the choice is shared by groups of g consecutive tokens, so that a decode tile
    # of g tokens has one transform per (head, sub-vector) -- the layout of the static kernel
    tg = max(tg, cfg.get("group", 1))
    W = _weights(kind, li, x)
    r, ch = _local_best(x, spec, rb, W, cfg["local"], Q, tg)
    if cfg.get("joint") and cfg["local"] == ["H"] and D // rb <= 8:
        # opt-in: best of the 2^(D/rb) joint {I, H_rb} slot combinations per token group under the full criterion
        # (the greedy rule picks each slot by its block-diagonal error); counts how often the pick differs
        nv = D // rb
        r0, r1 = Q(x), fwht(Q(fwht(x, rb)), rb)
        best_e, best_r, best_c = None, None, None
        for code in range(2 ** nv):
            sel = torch.tensor([(code >> j) & 1 for j in range(nv)], device=x.device, dtype=torch.bool)
            mask = sel.repeat_interleave(rb)
            rc = torch.where(mask, r1, r0)
            e = _pool_tokens(_head_err(rc - x, W)[None], tg, 3)[0]
            if best_e is None:
                best_e, best_r, best_c = e, rc, torch.zeros_like(e, dtype=torch.long)
            else:
                better = e < best_e
                best_e = torch.where(better, e, best_e)
                best_r = torch.where(better[..., None], rc, best_r)
                best_c = torch.where(better, torch.full_like(best_c, code), best_c)
        greedy_code = (ch.long() * (2 ** torch.arange(nv, device=x.device))).sum(-1)
        js = KV.setdefault("joint_stat", {}).setdefault(kind, [0.0, 0.0])
        js[0] += float((greedy_code != best_c).double().sum())
        js[1] += float(best_c.numel())
        ch = torch.stack([(best_c >> j) & 1 for j in range(nv)], -1)
        r = best_r
    heads = [r]
    names = ["base"]
    if "P" in cfg["head"]:
        pm = _perm(D, x.device)
        xp = x.index_select(-1, pm)
        rp, _ = _local_best(xp, spec, rb, _weights(kind, li, xp, pm), cfg["local"], Q, tg)
        heads.append(rp.index_select(-1, torch.argsort(pm))); names.append("P")
    if "F" in cfg["head"]:
        heads.append(fwht(Q(fwht(x, D)), D)); names.append("F")
    for hname in [h for h in cfg["head"] if h.startswith("FS")]:   # FS, FS2, ...: sign-flipped whole-head transforms
        sg = _signs(D, x.device, int(hname[2:] or 0))
        heads.append(sg * fwht(Q(fwht(x * sg, D)), D)); names.append(hname)
    hsel = None
    if len(heads) > 1:
        he = _pool_tokens(torch.stack([_head_err(h - x, W) for h in heads]), tg, 3)   # [nh, B, H, S]
        hsel = he.argmin(0)
        out = heads[0]
        for i in range(1, len(heads)):
            out = torch.where((hsel == i)[..., None], heads[i], out)
        r = out
    # statistics: per (layer, head, vector) count of tokens whose vector is transformed (base choice != identity and
    # the head kept the base candidate), tokens per head that took each head-level candidate
    st = KV["dynstat"].setdefault(kind, {})
    vec = (ch > 0) & ((hsel == 0)[..., None] if hsel is not None else True)
    cnt = vec.sum((0, 2)).double()
    ntok = ch.shape[0] * ch.shape[2]
    hc = torch.stack([(hsel == i).sum((0, 2)).double() for i in range(len(heads))]) if hsel is not None else \
        torch.full((1, ch.shape[1]), float(ntok), dtype=torch.double, device=ch.device)
    if li in st:
        st[li] = (st[li][0] + cnt, st[li][1] + ntok, st[li][2] + hc)
    else:
        st[li] = (cnt, ntok, hc)
    KV["dynnames"] = names
    if KV.get("dec_fn") is not None:          # diagnostic hook (default off): per-unit decisions (mechanism study)
        KV["dec_fn"](kind, li, ch, hsel, names, axis)
    if KV.get("crit_on"):                     # diagnostic: the criterion's error of the chosen vs the identity encoding
        cs = KV.setdefault("crit_stat", {}).setdefault(kind, {})
        e_id, e_sel = float(_head_err(Q(x) - x, W).double().sum()), float(_head_err(r - x, W).double().sum())
        o = cs.get(li, (0.0, 0.0))
        cs[li] = (o[0] + e_id, o[1] + e_sel)
    return r


def _after_sinks(fn, t: torch.Tensor) -> torch.Tensor:
    """apply a token-grouped encoder to the tokens after the kv_sink BF16 sinks (its groups, and the per-group
    decisions, then start right after them, as _fake_tok); the sink tokens are returned unquantized."""
    ns = KV["sink"]
    KV["sink"] = 0
    try:
        y = fn(t[:, :, ns:])
    finally:
        KV["sink"] = ns
    return torch.cat([t[:, :, :ns].float(), y], 2)


def dyn_quant_auto(t: torch.Tensor, spec: str, rb: int, li: int) -> torch.Tensor:
    """mode decision of the encoder (kv_k_axis='auto', keys only): per group of g tokens and KV head, keep either the
    per-token reconstruction (groups along head_dim, per-token transform choice) or the KIVI-style per-channel one
    (groups along tokens, per-group transform choice), whichever has the lower query-weighted error."""
    if KV["sink"] and t.shape[2] > KV["sink"]:
        return _after_sinks(lambda u: dyn_quant_auto(u, spec, rb, li), t)
    x = t.float()
    rt = dyn_quant(t, spec, rb, "k", li, "head")
    rc = dyn_quant(t, spec, rb, "k", li, "token")
    W = _weights("k", li, x)
    tg = FM.group_of(spec, x.shape[2])
    e = _pool_tokens(torch.stack([_head_err(rt - x, W), _head_err(rc - x, W)]), tg, 3)
    pick = (e[1] < e[0])[..., None]
    st = KV["dynstat"].setdefault("k_mode", {})
    a, n = st.get(li, (0.0, 0))
    st[li] = (a + float(pick.double().mean()), n + 1)
    return torch.where(pick, rc, rt)


def kv_quant(t: torch.Tensor, spec: str, mask: torch.Tensor | None, rb: int, axis: str = "head",
             full: torch.Tensor | None = None, md: dict | None = None) -> torch.Tensor:
    """t [B, Hkv, S, D] -> fake-quantized, in the rotated basis of the flagged vectors (mask [Hkv, D/rb]).
    axis = head: groups along head_dim per token; axis = token: per-channel groups of g consecutive tokens (KIVI keys);
    the last partial token group is padded by repeating its last token."""
    tf = rotx(t.float(), md) if md is not None else rot(t.float(), mask, full, rb)
    if axis == "head":
        return FM.fake(tf, spec)
    return _fake_tok(tf, spec)                                         # same as before; sinks excluded from groups


# decode memo for generation (lm-eval generate_until): the attention sees the whole BF16 cache at every step, and a
# per-token quantizer (head-axis groups, no whole-tensor scale) gives the past tokens the same codes as before, so
# only the new tokens are quantized.  A call without past tokens (prefill, KL / loglikelihood evaluation) resets it.
MEMO = {"on": True, "check": False, "store": {}}


def _memo(li, kind, t, fq, per_token, n_past):
    key = (li, kind)
    old = MEMO["store"].pop(key, None)
    if not (MEMO["on"] and per_token) or STATE["calib"]:
        return fq(t)
    if n_past == 0 or old is None or old.shape[0] != t.shape[0] or old.shape[2] != n_past:
        out = fq(t)
    else:
        out = torch.cat([old, fq(t[:, :, old.shape[2]:])], 2)
        if MEMO["check"]:
            ref = fq(t)
            assert torch.equal(out, ref), f"decode memo mismatch layer {li} {kind}: {(out - ref).abs().max().item()}"
    MEMO["store"][key] = out
    return out


def blockflag_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    li, n_rep, rb = module.layer_idx, module.num_key_value_groups, KV["rb"]
    dtype_in = query.dtype
    if KV.get("probe_fn") is not None and not STATE["calib"]:   # diagnostic hook (default off): raw post-RoPE keys
        KV["probe_fn"](li, key)
    if STATE["calib"] and KV["kc_calib"]:
        s_, n_ = key.double().sum((0, 2)), key.shape[0] * key.shape[2]
        old = KV["kc_acc"].get(li)
        KV["kc_acc"][li] = (s_, n_) if old is None else (old[0] + s_, old[1] + n_)
    if STATE["calib"] and KV.get("sacc"):
        _static_acc(li, key, value)
        return _plain_attention(query, key, value, attention_mask, scaling, module.num_key_value_groups, dtype_in)
    if STATE["calib"] and KV["kc_meas"]:
        _kc_measure(li, query, key, scaling)
        return _plain_attention(query, key, value, attention_mask, scaling, module.num_key_value_groups, dtype_in)
    if KV["kc"] is not None and li in KV["kc"] and (STATE["calib"] or STATE["quant"]) and not KV["kc_calib"]:
        bl_ = KV["bf16_layers"]
        if not (KV["skip"] and (li in bl_ if bl_ is not None else (li < KV["skip"] or li >= KV["nl"] - KV["skip"]))):
            key = key.float() - KV["kc"][li][None, :, None, :]       # static post-RoPE centering (exact for softmax)
    if STATE["calib"] and KV["calib_on"]:
        kv_calibrate(module, query, key, value)
    if STATE["calib"] and KV["measure"] is not None:
        kv_measure(module, query, key, value)
    if STATE["calib"] and KV["xf_calib"] is not None:
        xf_calibrate(module, query, key, value)
    if STATE["calib"] and KV["kvq_calib"]:
        kvq_calibrate(module, key, value, query)
    # during calibration the cache stays unquantized (joint calibration runs with quantized linears)
    quant = STATE["quant"] and not STATE["calib"] and (KV["k"] != "bf16" or KV["v"] != "bf16")
    bl = KV["bf16_layers"]
    if KV["skip"] and (li in bl if bl is not None else (li < KV["skip"] or li >= KV["nl"] - KV["skip"])):
        quant = False                                      # boundary layers keep a BF16 cache
    X = KV["xf"][li] if quant and KV["xf"] is not None else None
    if X is not None:                                      # static transform: exact in full precision
        if "k" in X:
            key = torch.einsum("bhsd,hde->bhse", key.float(), X["k"])
            query = torch.einsum("bhsd,hde->bhse", query.float(), X["q"].repeat_interleave(n_rep, 0))
        if "v" in X:
            value = torch.einsum("bhsd,hde->bhse", value.float(), X["v"])
    km = KV["kmask"][li] if quant and KV["kmask"] is not None else None
    vm = KV["vmask"][li] if quant and KV["vmask"] is not None else None
    kfl = KV["kfull"][li] if quant and KV["kfull"] is not None else None
    vfl = KV["vfull"][li] if quant and KV["vfull"] is not None else None
    kx = KV["kx"][li] if quant and KV["kx"] is not None else None
    vx = KV["vx"][li] if quant and KV["vx"] is not None else None
    RK = (lambda x, n=1: rotx(x, kx, n)) if kx is not None else (lambda x, n=1: rot(x, km, kfl, rb, n))
    RV = (lambda x, n=1: rotx(x, vx, n)) if vx is not None else (lambda x, n=1: rot(x, vm, vfl, rb, n))
    keep = keepv = None                                    # full-precision (query, token) pairs: keys / values
    dyn_on = KV["dyn"] is not None and (KV.get("dyn_layers") is None or li in KV["dyn_layers"])   # diagnostics hook
    if quant and dyn_on:                                   # dynamic bits: reconstructions in the original basis
        RK = RV = (lambda x, n=1: x)
        vm = vfl = vx = None
    if quant:
        if dyn_on:
            only = KV["dyn"].get("only")                   # ablation: per-token transforms on keys or values only
            if KV["k"] == "bf16":
                qk = lambda t: t.float()                    # noqa: E731
            elif only == "v":
                qk = lambda t: kv_quant(t, KV["k"], None, rb, "token" if KV["k_axis"] == "token" else "head")  # noqa: E731
            else:
                qk = lambda t: (dyn_quant_auto(t, KV["k"], rb, li) if KV["k_axis"] == "auto" else  # noqa: E731
                                dyn_quant(t, KV["k"], rb, "k", li, KV["k_axis"]))
            if KV["v"] == "bf16":
                qv = lambda t: t.float()                    # noqa: E731
            elif only == "k":
                qv = lambda t: kv_quant(t, KV["v"], None, rb)  # noqa: E731
            else:
                qv = lambda t: dyn_quant(t, KV["v"], rb, "v", li)  # noqa: E731
        elif KV["kvq"] is not None:                         # KVQuant-style cache (prefill / KL windows only)
            assert key.shape[2] == query.shape[2], "kv_kvq: pre-RoPE keys need positions 0..S-1 (no decode)"
            qk = lambda t: kvq_quant(t, "k", li)  # noqa: E731
            qv = lambda t: kvq_quant(t, "v", li)  # noqa: E731
        elif KV["rk"] is not None:                          # RotateKV (prefill / KL windows only)
            assert key.shape[2] == query.shape[2] and li in KV["rk_k"], "kv_rotatekv: prefill only"
            qk = lambda t: KV["rk_k"].pop(li)  # noqa: E731   pre-RoPE quantized in rk_install's RoPE wrapper
            qv = lambda t: rk_quant_v(t, KV["v"])  # noqa: E731
        else:
            qk = lambda t: kv_quant(t, KV["k"], km, rb, KV["k_axis"], kfl, kx)  # noqa: E731
            qv = lambda t: kv_quant(t, KV["v"], vm, rb, "head", vfl, vx)  # noqa: E731
        if KV.get("oracle_on") and dyn_on:     # diagnostic (default off): key errors weighted by the real queries
            KV["oracle_ctx"] = _oracle_ctx(query, key, attention_mask, scaling, n_rep, kwargs.get("softcap"))
        n_past = key.shape[2] - query.shape[2]
        nomemo = KV["kvq"] is not None or KV["rk"] is not None
        k, v = _memo(li, "k", key, qk, KV["k_axis"] == "head" and KV["k"] != "fp8t" and not nomemo, n_past), \
            _memo(li, "v", value, qv, KV["v"] != "fp8t" and not nomemo, n_past)
        KV["oracle_ctx"] = None
        if KV.get("true_on"):                 # diagnostic hook (default off): logit error of identity vs selected keys
            _true_stat(li, query, key, attention_mask, scaling, n_rep, rb, kwargs.get("softcap"))
        q = RK(query.float(), n_rep)
        piv = KV["rk_piv"].pop(li, None) if KV["rk"] is not None else None
        if KV["resid"] or KV["sink"] or piv is not None:
            # full-precision recent window (KIVI residual) and sink tokens (IntactKV-style), per query position;
            # mixed in the rotated basis so that the rotation stays exact
            S_, Sk_ = query.shape[2], key.shape[2]
            i = torch.arange(S_, device=key.device)[:, None] + (Sk_ - S_)
            j = torch.arange(Sk_, device=key.device)[None, :]
            if KV["flush"]:
                # WUSH-KV's cache policy (its Algorithm 2): tokens are processed in chunks of `flush`; before a chunk
                # starting at c0, every token outside the sink and the `resid` most recent ones has been quantized in
                # batches of `flush`, i.e. the quantized range is [sink, c0 - resid) (chunk starts and both windows are
                # multiples of `flush`); the current chunk itself is still in full precision
                c0 = (i // KV["flush"]) * KV["flush"]
                keep = (j < KV["sink"]) | (j >= torch.clamp(c0 - KV["resid"], min=KV["sink"]))
            else:
                keep = (j > i - KV["resid"]) & (j <= i) if KV["resid"] else torch.zeros(S_, Sk_, dtype=torch.bool, device=key.device)
                keep = keep | (j < KV["sink"])
            keepv = keep
            if KV["resid_mode"] == "kivi" and KV["resid"] and not KV["flush"]:
                # KIVI's own cache policy (models/llama_kivi.py, its Algorithm 1): keys stay in full precision until R
                # of them have accumulated and are then quantized together, i.e. query i sees full-precision keys
                # j in [R floor(i / R), i]; values slide, query i sees full-precision values j in [i - R, i]
                R_ = KV["resid"]
                keep = ((j >= (i // R_) * R_) & (j <= i)) | (j < KV["sink"])
                keepv = ((j >= i - R_) & (j <= i)) | (j < KV["sink"])
            if KV["resid_mode"] == "kgroup" and KV["resid"] and not KV["flush"]:
                # minimal causal per-channel key cache: the current incomplete group of R keys (j >= sink + R floor((i -
                # sink) / R)) is still in full precision, completed groups are quantized; values have no window
                R_, s_ = KV["resid"], KV["sink"]
                keep = ((j >= s_ + torch.div(i - s_, R_, rounding_mode="floor").clamp_min(0) * R_) & (j <= i)) | (j < s_)
                keepv = (j < s_) & (j <= i)
            if piv is not None:
                # RotateKV attention-sink-aware quantization: the sink tokens of this window stay in full precision
                # (keys and values) for every query; per batch element -> masks [B, 1, S, Sk]
                pj = piv[:, None, None, :Sk_].to(key.device)
                keep, keepv = keep | pj, keepv | pj
            kf = _repeat(RK(key.float()), n_rep)
            vf = _repeat(RV(value.float()), n_rep)
    else:
        q, k, v = query.float(), key.float(), value.float()
    k, v = _repeat(k, n_rep), _repeat(v, n_rep)
    S, Sk = q.shape[2], k.shape[2]
    # queries are processed in blocks so that long contexts (RULER / LongBench) fit: the score block is
    # [B, H, CHUNK, Sk] instead of [B, H, S, Sk]; results are identical to the unchunked computation
    chunk = S if S * Sk <= CHUNK["limit"] else max(1, CHUNK["elems"] // (q.shape[1] * Sk))
    outs = []
    for c0 in range(0, S, chunk):
        c1 = min(S, c0 + chunk)
        qc = q[:, :, c0:c1]
        scores = torch.matmul(qc, k.transpose(2, 3)) * scaling
        kc = None if keep is None else keep[..., c0:c1, :]
        kv_ = None if keepv is None else keepv[..., c0:c1, :]
        if kc is not None:
            scores = torch.where(kc, torch.matmul(qc, kf.transpose(2, 3)) * scaling, scores)
        if kwargs.get("softcap"):                                  # Gemma-2 attention logit soft-capping
            scores = torch.tanh(scores / kwargs["softcap"]) * kwargs["softcap"]
        if attention_mask is not None and attention_mask.dtype == torch.bool:
            # finite fill: a fully masked row (a left-padding query) must stay finite, or its NaN output would reach
            # the next layer's keys/values at that position and poison every row through 0 * NaN
            scores = scores.masked_fill(~attention_mask[..., c0:c1, :Sk], torch.finfo(scores.dtype).min)
        elif attention_mask is not None:
            scores = scores + attention_mask[..., c0:c1, :Sk].float()
        else:
            i = torch.arange(c0, c1, device=scores.device)[:, None] + (Sk - S)
            scores = scores.masked_fill(torch.arange(Sk, device=scores.device)[None, :] > i, float("-inf"))
        prob = torch.softmax(scores, -1)
        del scores
        if kc is not None:
            outs.append(torch.matmul(prob * ~kv_, v) + torch.matmul(prob * kv_, vf))
        else:
            outs.append(torch.matmul(prob, v))
        del prob
    out = torch.cat(outs, 2)
    if vm is not None or vfl is not None or vx is not None:
        out = RV(out, n_rep)
    if X is not None and "v" in X:
        out = torch.einsum("bhsd,hde->bhse", out, X["vinv"].repeat_interleave(n_rep, 0))
    return out.transpose(1, 2).contiguous().to(dtype_in), None


@torch.no_grad()
@torch.no_grad()
def _oracle_ctx(query, key, attention_mask, scaling, n_rep, softcap=None):
    """diagnostic (KV['oracle_on'], Gemma-3 round 2): the selection criterion with the real queries of the window --
    per key t the error sum_i W_it (q_i . e_t)^2, W_it = s^2 p_it over the (query, key) pairs the cache quantizes (BF16
    attention p with the model mask / soft-cap; sinks and the BF16 recent window excluded).  Uses future queries, so it
    is an upper-bound reference for the criterion's query model, not an encoder."""
    q = query.float()
    B, H, S, D = q.shape
    kr = key.float()
    Sk = kr.shape[2]
    i = torch.arange(S, device=q.device)[:, None] + (Sk - S)
    j = torch.arange(Sk, device=q.device)[None, :]
    keep = ((j > i - KV["resid"]) & (j <= i) if KV["resid"] else torch.zeros(S, Sk, dtype=torch.bool, device=q.device)) | (j < KV["sink"])
    qm = ((j <= i) & ~keep).float()
    sc = scaling * torch.matmul(q, _repeat(kr, n_rep).transpose(2, 3))
    if softcap:
        sc = torch.tanh(sc / softcap) * softcap
    if attention_mask is not None and attention_mask.dtype == torch.bool:
        sc = sc.masked_fill(~attention_mask[..., :Sk], torch.finfo(sc.dtype).min)
    elif attention_mask is not None:
        sc = sc + attention_mask[..., :Sk].float()
    else:
        sc = sc.masked_fill(j > i, float("-inf"))
    Wt = torch.softmax(sc, -1) * qm * scaling ** 2
    del sc
    Hkv = H // n_rep
    return q.reshape(B, Hkv, n_rep, S, D), Wt.reshape(B, Hkv, n_rep, S, Sk)


def _true_stat(li, query, key, attention_mask, scaling, n_rep, rb, softcap=None):
    """diagnostic (KV['true_on']): on the same keys of this layer, the plain-cache reconstruction (identity) and the
    dynamically selected one; with the real queries and the BF16 attention probabilities p (model mask, soft-cap) over
    the quantized (query, key) pairs: sum p n^2 (uncentered, what an independent-error criterion sees) and
    sum p n^2 - (sum p n)^2 (centered: the softmax-relevant part; a shift common to all keys of a query cancels), n =
    s q.(k_hat - k); plus sum p n^2 and attention mass by key distance (< 1024 / >= 1024 beyond the BF16 window)."""
    q = query.float()
    H = q.shape[1]
    kr = key.float()
    rec = {"id": kv_quant(key, KV["k"], None, rb, KV["k_axis"]),
           "sel": dyn_quant(key, KV["k"], rb, "k", li, KV["k_axis"])}
    S, Sk = q.shape[2], kr.shape[2]
    i = torch.arange(S, device=q.device)[:, None] + (Sk - S)
    j = torch.arange(Sk, device=q.device)[None, :]
    keep = ((j > i - KV["resid"]) & (j <= i) if KV["resid"] else torch.zeros(S, Sk, dtype=torch.bool, device=q.device)) | (j < KV["sink"])
    qm = ((j <= i) & ~keep).float()
    far = ((i - j) >= 1024).float()
    sc = scaling * torch.matmul(q, _repeat(kr, n_rep).transpose(2, 3))
    if softcap:
        sc = torch.tanh(sc / softcap) * softcap
    if attention_mask is not None and attention_mask.dtype == torch.bool:
        sc = sc.masked_fill(~attention_mask[..., :Sk], torch.finfo(sc.dtype).min)
    elif attention_mask is not None:
        sc = sc + attention_mask[..., :Sk].float()
    else:
        sc = sc.masked_fill(j > i, float("-inf"))
    p = torch.softmax(sc, -1)
    del sc
    st = KV.setdefault("true_stat", {}).setdefault(li, {})
    for name, kh in rec.items():
        n = scaling * torch.matmul(q, _repeat(kh.float() - kr, n_rep).transpose(2, 3)) * qm
        pn = p * n
        unc = float((pn * n).sum())
        cen = float(((pn * n).sum(-1) - pn.sum(-1).square()).sum())
        unc_far = float((pn * n * far).sum())
        o = st.get(name, [0.0, 0.0, 0.0])
        st[name] = [o[0] + unc, o[1] + cen, o[2] + unc_far]
        del n, pn
    m = st.get("mass", [0.0, 0.0])
    st["mass"] = [m[0] + float((p * qm).sum()), m[1] + float((p * qm * far).sum())]


def _plain_attention(query, key, value, attention_mask, scaling, n_rep, dtype_in):
    """BF16-cache attention (FP32 scores) for calibration passes that only collect statistics."""
    q, k, v = query.float(), _repeat(key.float(), n_rep), _repeat(value.float(), n_rep)
    S, Sk = q.shape[2], k.shape[2]
    if S * Sk > CHUNK["limit"]:            # long calibration windows: query blocks (2k windows: unchanged path below)
        cb = max(1, CHUNK["elems"] // (q.shape[1] * Sk))
        outs = []
        for c0 in range(0, S, cb):
            c1 = min(S, c0 + cb)
            sc = torch.matmul(q[:, :, c0:c1], k.transpose(2, 3)) * scaling
            if attention_mask is not None and attention_mask.dtype == torch.bool:
                sc = sc.masked_fill(~attention_mask[..., c0:c1, :Sk], torch.finfo(sc.dtype).min)
            elif attention_mask is not None:
                sc = sc + attention_mask[..., c0:c1, :Sk].float()
            else:
                i = torch.arange(c0, c1, device=sc.device)[:, None] + (Sk - S)
                sc = sc.masked_fill(torch.arange(Sk, device=sc.device)[None, :] > i, float("-inf"))
            outs.append(torch.matmul(torch.softmax(sc, -1), v))
            del sc
        return torch.cat(outs, 2).transpose(1, 2).contiguous().to(dtype_in), None
    sc = torch.matmul(q, k.transpose(2, 3)) * scaling
    if attention_mask is not None and attention_mask.dtype == torch.bool:
        sc = sc.masked_fill(~attention_mask[..., :Sk], torch.finfo(sc.dtype).min)
    elif attention_mask is not None:
        sc = sc + attention_mask[..., :Sk].float()
    else:
        i = torch.arange(S, device=sc.device)[:, None] + (Sk - S)
        sc = sc.masked_fill(torch.arange(Sk, device=sc.device)[None, :] > i, float("-inf"))
    return torch.matmul(torch.softmax(sc, -1), v).transpose(1, 2).contiguous().to(dtype_in), None


# ---------------------------------------------------------------------------------------------------------------------
# static selection (kv_dyn ...+static, the kivi2s control): sacc = 'local' | 'head' while the calibration windows run
# STATIC_PAD: positions prepended (copies, zero error weight) before the first non-sink key so that the key groups of
# the statistics start at an absolute multiple of 32 (set by integrations/.../sim_kivi_align.py in A-aligned runs only)
STATIC = {"acc": {}, "pad": 0}


@torch.no_grad()
def _static_acc(li, key, value):
    """summed candidate errors of the dynamic rule over the calibration tokens after the sinks: stage 'local' -> per
    (kind, head, slot) the errors of I and H_rb; stage 'head' -> per (kind, head) the head errors of the frozen slot
    choice and of the whole-head transform.  K: groups along tokens (as dyn_quant(axis=token)); V: per token."""
    cfg, rb, ns = KV["dyn"], KV["rb"], KV["sink"]
    for kind, t in (("k", key), ("v", value)):
        spec = KV[kind]
        if spec == "bf16":
            continue
        x = t.float()[:, :, ns:] if ns else t.float()
        pad = STATIC["pad"] if kind == "k" else 0
        if pad:
            x = torch.cat([x[:, :, :1].expand(-1, -1, pad, -1), x], 2)
        D = x.shape[-1]
        sink0 = KV["sink"]
        KV["sink"] = 0
        try:
            W = _weights(kind, li, x)
            Q = (lambda y: _fake_tok(y, spec)) if kind == "k" else (lambda y: FM.fake(y, spec))
            recs = [Q(x), fwht(Q(fwht(x, rb)), rb)]
            keep = torch.ones(x.shape[2], device=x.device)
            if pad:
                keep[:pad] = 0
            a = STATIC["acc"].setdefault(("local", kind, li), 0)
            if KV["sacc"] == "local":
                ev = torch.stack([_vec_err(r - x, W, rb) for r in recs])            # [2, B, H, S, nv]
                STATIC["acc"][("local", kind, li)] = a + (ev * keep[None, None, None, :, None]).double().sum((1, 3))
            else:
                ch = STATIC["choice"][(kind, li)]                                   # [H, nv] bool
                sel = ch[None, :, None, :, None].expand(*recs[0].shape[:-1], ch.shape[-1], rb).reshape(recs[0].shape)
                r = torch.where(sel, recs[1], recs[0])
                F = fwht(Q(fwht(x, D)), D)
                he = torch.stack([_head_err(h - x, W) for h in (r, F)])            # [2, B, H, S]
                b = STATIC["acc"].get(("head", kind, li), 0)
                STATIC["acc"][("head", kind, li)] = b + (he * keep[None, None, None, :]).double().sum((1, 3))
        finally:
            KV["sink"] = sink0


def static_finalize(stage):
    """stage local: slot choice = argmin(I, H) of the summed errors (identity on ties); stage head: whole-head choice =
    argmin(base, F) (base on ties) -> KV kmask / kfull / vmask / vfull, and the dynamic rule is switched off."""
    if stage == "local":
        STATIC["choice"] = {}
        for (st, kind, li), e in STATIC["acc"].items():
            if st == "local":
                STATIC["choice"][(kind, li)] = e[1] < e[0]
        return
    out = {"k": ({}, {}), "v": ({}, {})}
    for (st, kind, li), e in STATIC["acc"].items():
        if st != "head":
            continue
        full = e[1] < e[0]                                                          # [H]
        out[kind][0][li] = STATIC["choice"][(kind, li)] & ~full[:, None]
        out[kind][1][li] = full
    KV["kmask"], KV["kfull"] = out["k"]
    KV["vmask"], KV["vfull"] = out["v"]
    KV["dyn"] = None
    KV["static_summary"] = {kind: dict(slot_h=float(torch.stack([m.double().mean() for m in out[kind][0].values()]).mean()),
                                       full=float(torch.stack([f.double().mean() for f in out[kind][1].values()]).mean()))
                            for kind in "kv"}
    STATIC["acc"], STATIC["choice"] = {}, {}


def _kc_group(D: int) -> int:
    return min(FM.group_of(KV["k"], D), D)


@torch.no_grad()
def _kc_measure(li, query, key, scaling):
    """kv_center='sel', pass 2: per (KV head, key scale group) the attention-weighted variance of the logit error of a
    per-token RTN cache without / with the mean, sum_i [sum_t p_it n_it^2 - (sum_t p_it n_it)^2] with
    n_it = s q_i[g] . e_t[g] (error of group g of key t) over the (query, key) pairs the cache quantizes (the kv_sink /
    kv_resid tokens are exact), p the BF16 attention probabilities.  Unlike a per-token C_q weighting it sees the keys
    that hold most of the attention (sinks), whose error decides the softmax."""
    if li not in KV["kc"]:
        return
    x = key.float()
    B, Hkv, S, D = x.shape
    H = query.shape[1]
    q = query.float()
    g = _kc_group(D)
    ng = D // g
    i = torch.arange(S, device=x.device)[:, None]
    j = torch.arange(S, device=x.device)[None, :]
    keep = (j > i - KV["resid"]) if KV["resid"] else torch.zeros(S, S, dtype=torch.bool, device=x.device)
    keep = keep | (j < KV["sink"])
    qm = ((j <= i) & ~keep).float()                                       # quantized (query, key) pairs
    out = torch.zeros(2, Hkv, ng, dtype=torch.float64, device=x.device)
    if S * S > CHUNK["limit"]:
        # long calibration windows (e.g. WUSH-KV's 32k FineWeb sequences): the same sums over query blocks, so that the
        # [B, H, S, S] logits are never materialised (2k windows keep the unchunked path below, bit for bit)
        xr = _repeat(x, H // Hkv)
        es = [_repeat(FM.fake(y, KV["k"]) - y, H // Hkv) for y in (x, x - KV["kc"][li][None, :, None, :])]
        cb = max(1, CHUNK["elems"] // (H * S))
        for c0 in range(0, S, cb):
            c1 = min(S, c0 + cb)
            qc = q[:, :, c0:c1]
            lg = scaling * torch.einsum("bhid,bhtd->bhit", qc, xr)
            p = torch.softmax(lg.masked_fill((j > i)[c0:c1], float("-inf")), -1)
            del lg
            for ci, e in enumerate(es):
                for gi in range(ng):
                    sl = slice(gi * g, (gi + 1) * g)
                    n = scaling * torch.einsum("bhid,bhtd->bhit", qc[..., sl], e[..., sl]) * qm[c0:c1]
                    v = (p * n.square()).sum(-1) - (p * n).sum(-1).square()
                    out[ci, :, gi] += v.sum((0, 2)).double().view(Hkv, H // Hkv).sum(1)
                    del n, v
            del p
        KV["kc_err"][li] = KV["kc_err"][li] + out if li in KV["kc_err"] else out
        return
    lg = scaling * torch.einsum("bhid,bhtd->bhit", q, _repeat(x, H // Hkv))
    p = torch.softmax(lg.masked_fill(j > i, float("-inf")), -1)
    del lg
    for ci, y in enumerate((x, x - KV["kc"][li][None, :, None, :])):
        e = _repeat(FM.fake(y, KV["k"]) - y, H // Hkv)
        for gi in range(ng):
            sl = slice(gi * g, (gi + 1) * g)
            n = scaling * torch.einsum("bhid,bhtd->bhit", q[..., sl], e[..., sl]) * qm
            v = (p * n.square()).sum(-1) - (p * n).sum(-1).square()          # [B, H, S]
            out[ci, :, gi] = v.sum((0, 2)).double().view(Hkv, H // Hkv).sum(1)
            del n, v
    KV["kc_err"][li] = KV["kc_err"][li] + out if li in KV["kc_err"] else out


def kc_select():
    """keep the mean on the (head, group) blocks where it lowers the calibrated error, 0 elsewhere."""
    for li, e in KV["kc_err"].items():
        mu = KV["kc"][li]
        use = (e[1] < e[0])                                               # [Hkv, ng]
        KV["kc"][li] = mu * use.repeat_interleave(_kc_group(mu.shape[-1]), -1).to(mu.dtype)
        KV["kc_frac"][li] = float(use.double().mean())
    KV["kc_err"] = {}


def kc_summary() -> dict:
    out = dict(mu_norm={str(li): float(v.norm()) for li, v in sorted(KV["kc"].items())})
    if KV["kc_frac"]:
        out["frac_centered"] = {str(li): f for li, f in sorted(KV["kc_frac"].items())}
        out["frac_centered_mean"] = sum(KV["kc_frac"].values()) / len(KV["kc_frac"])
    return out


AttentionInterface.register("blockflag", blockflag_attention)
# without a registered mask function transformers passes no mask to a custom attention, so left padding (batched
# generation in lm-eval) would be attended; the SDPA boolean mask is what blockflag_attention expects
AttentionMaskInterface.register("blockflag", sdpa_mask)


def _cq(query, Hkv, D):
    B, H, S, _ = query.shape
    q = query.float().reshape(B, Hkv, H // Hkv, S, D)
    if (KV["sel"] or {}).get("proxy") == "mse" or KV.get("cq_eye"):     # unweighted MSE proxy (design-space ablation)
        return torch.eye(D, device=query.device).expand(Hkv, D, D).contiguous()
    return torch.einsum("bhrsd,bhrse->hde", q, q)


@torch.no_grad()
def kv_calibrate(module, query, key, value):
    li, n_rep = module.layer_idx, module.num_key_value_groups
    D = key.shape[-1]
    Cq = _cq(query, key.shape[1], D)
    if KV["dyn"] is not None:                              # dynamic bits only need static query statistics
        KV["cq"][li] = KV["cq"][li] + Cq if li in KV["cq"] else Cq
        if KV["dyn"]["rule"] == "att":
            if KV.get("att_nq"):                           # opt-in: nq queries per KV head, even over the windows
                nq = KV["att_nq"]
                _sample_queries(module, query, key, per_window=-(-nq // (max(1, KV["att_nw"]) * (query.shape[1] // key.shape[1]))),
                                keep=nq)
            else:
                _sample_queries(module, query, key)
        return
    _pattern_errors(li, key, value, Cq, KV["rb"], KV["acc"])
    for r in (KV["sel"] or {}).get("rbs", []):
        _pattern_errors(li, key, value, Cq, r, KV["accx"].setdefault(r, {}))


def _sample_queries(module, query, key, per_window: int = 64, keep: int = 512):
    """ScaNN-style rule: store up to `keep` calibration queries per KV head (all its query heads) with
    logZ_q = logsumexp_t s q.k_t over the causal context, so that exp(s q.x - logZ_q) is the attention probability a
    new key x would receive from q."""
    li = module.layer_idx
    B, H, S, D = query.shape
    Hkv = key.shape[1]
    sc = float(getattr(module, "scaling", D ** -0.5))
    g = torch.Generator().manual_seed(li)
    pos = torch.randint(S // 4, S, (per_window,), generator=g).to(query.device)
    q = query.float()[:, :, pos]                                          # [B, H, n, D]
    lg = sc * torch.einsum("bhnd,bhtd->bhnt", q, _repeat(key.float(), H // Hkv))
    lg = lg.masked_fill(torch.arange(S, device=q.device)[None, None, None, :] > pos[None, None, :, None], float("-inf"))
    sw = getattr(module, "sliding_window", None)
    if sw and (KV["dyn"] or {}).get("sw"):     # opt-in: sliding-window layers attend only to the last `sw` keys
        lg = lg.masked_fill(torch.arange(S, device=q.device)[None, None, None, :] <= pos[None, None, :, None] - sw, float("-inf"))
    lz = torch.logsumexp(lg, -1)                                          # [B, H, n]
    q = q.reshape(B, Hkv, H // Hkv, per_window, D).transpose(0, 1).reshape(Hkv, -1, D)
    lz = lz.reshape(B, Hkv, H // Hkv, per_window).transpose(0, 1).reshape(Hkv, -1)
    if li in KV["qs"]:
        q, lz = torch.cat([KV["qs"][li], q], 1)[:, -keep:], torch.cat([KV["lz"][li], lz], 1)[:, -keep:]
    KV["qs"][li], KV["lz"][li], KV["qscale"] = q[:, -keep:], lz[:, -keep:], sc


def _pattern_errors(li, key, value, Cq, rb, store):
    B, Hkv, S, D = key.shape
    for kind in ("k", "v"):
        spec = KV[kind]
        if spec == "bf16":
            continue
        t = (key if kind == "k" else value).float()
        U = max(rb, min(FM.group_of(spec, D), D))
        pu, nvec, nu = U // rb, D // rb, D // U
        if kind == "k":
            Cb = torch.stack([Cq[:, u * U:(u + 1) * U, u * U:(u + 1) * U] for u in range(nu)], 1)   # [Hkv, nu, U, U]
            den = torch.einsum("bhsd,hde,bhse->bhs", t, Cq, t).clamp_min(1e-30)
        else:
            co = KV["co"][li].view(Hkv, nu, U)
            den = t.square().sum(-1).clamp_min(1e-30)
        if (KV["sel"] or {}).get("proxy") == "abs":
            den = torch.ones_like(den)
        errs = []
        for p in [-1, *range(pu), "all"]:
            if p == "all" and pu == 1:
                errs.append(errs[1].clone())
                continue
            m = None if p == -1 else (torch.ones(Hkv, nvec, dtype=torch.bool, device=t.device) if p == "all"
                                      else pattern_mask(nvec, pu, p, t.device).expand(Hkv, nvec))
            tq = kv_quant(t, spec, m, rb, KV["k_axis"] if kind == "k" else "head")
            e = (sel_fwht(tq, None if m is None else m[None, :, None, :], rb) - t).reshape(B, Hkv, S, nu, U)
            if kind == "k":
                num = torch.einsum("bhsuk,hukl,bhsul->bhsu", e, Cb, e)
            else:
                num = (e.square() * co[None, :, None]).sum(-1)
            errs.append((num / den[..., None]).sum((0, 2)).double())      # [Hkv, nu]
        e = torch.stack(errs)                                             # [pu + 2, Hkv, nu]: none, positions, all
        acc = store.setdefault(kind, {})
        acc[li] = acc[li] + e if li in acc else e
        if store is not KV["acc"]:
            continue
        # whole-head Hadamard (one more candidate per head for kv_flags='headsel'), same per-unit error
        fm = torch.ones(Hkv, dtype=torch.bool, device=t.device)
        tq = kv_quant(t, spec, None, rb, KV["k_axis"] if kind == "k" else "head", fm)
        e = (rot(tq, None, fm, rb) - t).reshape(B, Hkv, S, nu, U)
        num = torch.einsum("bhsuk,hukl,bhsul->bhsu", e, Cb, e) if kind == "k" else (e.square() * co[None, :, None]).sum(-1)
        ef = (num / den[..., None]).sum((0, 2)).double()                  # [Hkv, nu]
        af = KV["accfull"].setdefault(kind, {})
        af[li] = af[li] + ef if li in af else ef


def greedy_kv_masks(kind: str, margin: float = 0.0) -> dict[int, torch.Tensor]:
    out = {}
    for li, e in KV["acc"][kind].items():
        gain = e[0][None] - e[1:-1] - margin * e[0][None]                 # [pu, Hkv, nu]
        out[li] = (gain > 0).permute(1, 2, 0).reshape(e.shape[1], -1)      # [Hkv, nu * pu]
    return out


@torch.no_grad()
def kv_measure(module, query, key, value):
    """pass 2 of kv_flags='sel': head-level proxy error of every candidate transform (full C_q, no unit truncation)."""
    li = module.layer_idx
    B, Hkv, S, D = key.shape
    Cq = _cq(query, Hkv, D)
    for kind in ("k", "v"):
        if KV[kind] == "bf16":
            continue
        t = (key if kind == "k" else value).float()
        if kind == "k":
            den = torch.einsum("bhsd,hde,bhse->bhs", t, Cq, t).clamp_min(1e-30)
        else:
            co = KV["co"][li]                                             # [Hkv, D]
            den = t.square().sum(-1).clamp_min(1e-30)
        if (KV["sel"] or {}).get("proxy") == "abs":                       # absolute error, no per-token normalization
            den = torch.ones_like(den)
        for name, md in KV["measure"][kind][li].items():
            tq = kv_quant(t, KV[kind], None, KV["rb"], KV["k_axis"] if kind == "k" else "head", None, md)
            e = rotx(tq, md) - t
            num = torch.einsum("bhsd,hde,bhse->bhs", e, Cq, e) if kind == "k" else (e.square() * co[None, :, None]).sum(-1)
            err = (num / den).sum((0, 2)).double()                        # [Hkv]
            d = KV["meas"].setdefault(kind, {}).setdefault(li, {})
            d[name] = d[name] + err if name in d else err


def sel_candidates(kind: str, cands: list[str], Hkv: int, D: int, device) -> dict[int, dict[str, dict]]:
    """candidate transforms per layer: none, b<r> (greedy bits at block size r), a<r> (every block of size r), f (H_D)."""
    out = {}
    for li in KV["acc"][kind]:
        c = {"none": {}}
        for nm in cands:
            if nm == "f":
                c["f"] = {D: torch.ones(Hkv, 1, dtype=torch.bool, device=device)}
            elif nm[0] == "a":
                r = int(nm[1:])
                c[nm] = {r: torch.ones(Hkv, D // r, dtype=torch.bool, device=device)}
            elif nm[0] == "b":
                r = int(nm[1:])
                e = (KV["accx"][r] if r in KV["accx"] else KV["acc"])[kind][li]
                gain = e[0][None] - e[1:-1]
                c[nm] = {r: (gain > 0).permute(1, 2, 0).reshape(Hkv, -1)}
        out[li] = c
    return out


def sel_choose(kind: str, gran: str) -> dict[int, dict]:
    """per head / layer / model the candidate with the lowest measured error -> {layer: {rb: mask [Hkv, D/rb]}}."""
    meas, cand = KV["meas"][kind], KV["measure"][kind]
    names = list(next(iter(cand.values())).keys())
    E = {li: torch.stack([meas[li][n] for n in names]) for li in meas}   # [ncand, Hkv]
    if gran == "model":
        tot = sum(e.sum(1) for e in E.values())
        pick = {li: torch.full((e.shape[1],), int(tot.argmin()), device=e.device) for li, e in E.items()}
    elif gran == "layer":
        pick = {li: torch.full((e.shape[1],), int(e.sum(1).argmin()), device=e.device) for li, e in E.items()}
    else:
        pick = {li: e.argmin(0) for li, e in E.items()}
    out = {}
    for li, pk in pick.items():
        md = {}
        for ci, n in enumerate(names):
            for r, m in cand[li][n].items():
                sel = (pk == ci).to(m.device)[:, None]
                md[r] = (md[r] | (m & sel)) if r in md else (m & sel)
        out[li] = {r: m for r, m in md.items() if bool(m.any())}
    return out


def headsel_kv_masks(kind: str) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """per (layer, KV head) the candidate with the lowest calibrated error: no rotation, the greedy rb-block bits
    (error estimated as none minus the positive per-position gains, the additivity the greedy rule assumes) or the
    whole-head Hadamard.  Returns (block masks with zero rows for the other choices, whole-head flags)."""
    blocks, full = greedy_kv_masks(kind), {}
    for li, e in KV["acc"][kind].items():
        e0 = e[0].sum(-1)                                                 # [Hkv]
        eb = e0 - (e[0][None] - e[1:-1]).clamp_min(0).sum((0, 2))
        ef = KV["accfull"][kind][li].sum(-1)
        choice = torch.stack([e0, eb, ef]).argmin(0)                      # 0 none, 1 blocks, 2 whole head
        blocks[li] = blocks[li] & (choice == 1)[:, None]
        full[li] = choice == 2
    return blocks, full


def head_kv_masks(kind: str) -> dict[int, torch.Tensor]:
    """coarser ablation: one decision per (layer, KV head): rotate the whole head iff that lowers its summed error."""
    out = {}
    for li, e in KV["acc"][kind].items():
        pu = e.shape[0] - 2
        rot = e[-1].sum(-1) < e[0].sum(-1)                                # [Hkv]
        out[li] = rot[:, None].expand(-1, e.shape[2] * pu).clone()
    return out


# ---------------------------------------------------------------------------------------------------------------------
# static calibrated KV transforms (kv_xform): WUSH-KV and QServe's SmoothAttention
# ---------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def xf_calibrate(module, query, key, value):
    """WUSH-KV statistics per (layer, KV head) on post-RoPE keys / queries (FP64 sums over calibration tokens):
    M_K = sum k k^T, H_K = 2 sum_g sum q_g q_g^T (all query heads of the GQA group), M_V = sum v v^T.
    (SmoothAttention collects pre-RoPE key maxima with forward hooks instead, see smooth_hooks.)"""
    li = module.layer_idx
    if KV["xf_calib"] == "smoothpost":                  # SmoothAttention on post-RoPE key maxima (QServe's code)
        a = key.detach().float().abs().amax((0, 2))                   # [Hkv, D]
        st = KV["xs"].setdefault(li, {})
        st["kmax"] = torch.maximum(st["kmax"], a) if "kmax" in st else a
        return
    if KV["xf_calib"] != "wush":
        return
    B, Hkv, S, D = key.shape
    # per-window sums in FP32 (H20's FP64 rate makes FP64 einsums over 32k-token windows take hours), accumulated
    # across windows in FP64
    k, v = key.float(), value.float()
    q = query.float().reshape(B, Hkv, -1, S, D)
    st = KV["xs"].setdefault(li, {})
    for name, val in (("MK", torch.einsum("bhsd,bhse->hde", k, k)),
                      ("HK", 2 * torch.einsum("bhrsd,bhrse->hde", q, q)),
                      ("MV", torch.einsum("bhsd,bhse->hde", v, v))):
        val = val.double()
        st[name] = st[name] + val if name in st else val


def smooth_hooks(model) -> list:
    """forward hooks recording max |K_i| per (layer, KV head, channel) of the pre-RoPE keys (output of k_norm when the
    model has one, else of k_proj), as QServe calibrates SmoothAttention."""
    hs = []
    cfg = text_config(model)
    for li, at in attn_layers(model).items():
        Hkv = cfg.num_key_value_heads
        mod = getattr(at, "k_norm", None) or getattr(at, "k_proj", None)
        ksl = None
        if mod is None:                                    # fused QKV projection (Phi-3/4): slice the key part
            mod = at.qkv_proj
            D = head_dim(cfg)
            q = cfg.num_attention_heads * D
            ksl = slice(q, q + Hkv * D)

        def hook(m, inp, out, li=li, Hkv=Hkv, ksl=ksl):
            if ksl is not None:
                out = out[..., ksl]
            a = out.detach().float().reshape(-1, Hkv, out.shape[-1] if out.dim() == 4 else out.shape[-1] // Hkv)
            a = a.abs().amax(0)
            st = KV["xs"].setdefault(li, {})
            st["kmax"] = torch.maximum(st["kmax"], a) if "kmax" in st else a
        hs.append(mod.register_forward_hook(hook))
    return hs


def wush(M: torch.Tensor, H: torch.Tensor, gamma: float = 1e-2) -> torch.Tensor:
    """WUSH transform (WUSH-KV eq. 1), batched over leading dims, FP64:
    L L^T = H + gamma/d tr(H) I,  U Lam U^T = L^T (M + gamma/d tr(M) I) L,
    c = (1 + gamma)^(1/2) tr(M)^(1/2) tr(Lam^(1/2))^(-1/2),  T = c Had Lam^(-1/4) U^T L^T  (column vectors: x' = T x)."""
    d = M.shape[-1]
    I = torch.eye(d, dtype=M.dtype, device=M.device)
    tr = lambda A: A.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]  # noqa: E731
    L = torch.linalg.cholesky(H + gamma / d * tr(H) * I)
    lam, U = torch.linalg.eigh(L.transpose(-1, -2) @ (M + gamma / d * tr(M) * I) @ L)
    lam = lam.clamp_min(1e-300)
    c = (1 + gamma) ** 0.5 * tr(M).sqrt() / lam.sqrt().sum(-1)[..., None, None].sqrt()
    had = sylvester(d, M.device, M.dtype)
    return c * had @ torch.diag_embed(lam.pow(-0.25)) @ U.transpose(-1, -2) @ L.transpose(-1, -2)


def _blockdiag_wush(M: torch.Tensor, H: torch.Tensor, g: int) -> torch.Tensor:
    """[Hkv, D, D] block-diagonal WUSH with blocks of width g (WUSH-KV: block width = quantization group size)."""
    Hkv, D, _ = M.shape
    T = torch.zeros_like(M)
    for j in range(0, D, g):
        T[:, j:j + g, j:j + g] = wush(M[:, j:j + g, j:j + g], H[:, j:j + g, j:j + g])
    return T


@torch.no_grad()
def xf_finalize(model, mode: str, kspec: str, vspec: str) -> dict[int, dict]:
    """per layer the row-vector matrices of the static transform (see KV['xf'])."""
    cfg = text_config(model)
    D = head_dim(cfg)
    Hkv, H = cfg.num_key_value_heads, cfg.num_attention_heads
    out = {}
    for li, at in attn_layers(model).items():
        st, X = KV["xs"][li], {}
        if mode.startswith("smooth"):
            alpha = float(mode.split(":")[1])
            m = st["kmax"].double()                                   # [Hkv, D], pre-RoPE
            m = torch.maximum(m, m.roll(D // 2, -1))                  # RoPE pairs (i, i + D/2) share lambda
            lam = m.clamp_min(1e-5).pow(alpha)
            if kspec != "bf16":
                X["k"], X["q"] = torch.diag_embed(1.0 / lam), torch.diag_embed(lam)
        else:
            if kspec != "bf16":
                T = _blockdiag_wush(st["MK"], st["HK"], min(FM.group_of(kspec, D), D))
                X["k"], X["q"] = T.transpose(-1, -2), torch.linalg.inv(T)
            if vspec != "bf16":
                op = at.o_proj
                Wo = getattr(op, "lin", op).weight.double()           # [hidden, H * D] (BF16 weight)
                Ws = Wo.reshape(Wo.shape[0], H, D).permute(1, 2, 0)   # [H, D, hidden] = W_O(h, g)
                HV = 2 * torch.einsum("hdo,heo->hde", Ws, Ws).reshape(Hkv, H // Hkv, D, D).sum(1)
                T = _blockdiag_wush(st["MV"], HV, min(FM.group_of(vspec, D), D))
                X["v"], X["vinv"] = T.transpose(-1, -2), torch.linalg.inv(T).transpose(-1, -2)
        out[li] = {k: v.float().contiguous() for k, v in X.items()}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# KVQuant-style cache (kv_kvq): pre-RoPE per-channel keys, per-token values, non-uniform 4-bit codes, optional 1 %
# dense-and-sparse outliers.  Not a reproduction of KVQuant (Hooper et al., 2024): scales are min/max per group of g
# (as every other 5-bit row), the codebook is an unweighted k-means (no Fisher weighting), no Q-norm, no special
# treatment of the first token beyond kv_sink.
# ---------------------------------------------------------------------------------------------------------------------
def _rot_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], -1)


def _rope_cs(S: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos, sin [1, 1, S, D] of positions 0..S-1 from the model's rotary embedding (HF rotate_half convention)."""
    re_ = KV["rope"]
    assert re_ is not None and float(getattr(re_, "attention_scaling", 1.0)) == 1.0, "kv_kvq needs plain RoPE"
    pos = torch.arange(S, device=device)[None]
    cos, sin = re_(torch.zeros(1, dtype=torch.float32, device=device), pos)
    return cos.float()[:, None], sin.float()[:, None]


def _unrope(k: torch.Tensor) -> torch.Tensor:
    cos, sin = _rope_cs(k.shape[2], k.device)
    return k * cos - _rot_half(k) * sin                 # inverse rotation: angle -theta


def _rerope(k: torch.Tensor) -> torch.Tensor:
    cos, sin = _rope_cs(k.shape[2], k.device)
    return k * cos + _rot_half(k) * sin


def _kvq_layout(x: torch.Tensor, kind: str, spec: str):
    """grouped view [..., n, g] of the quantized tokens (keys: per channel along tokens after the sinks, the last partial
    group padded by repeating its last token; values: per token along head_dim) and the inverse map."""
    if kind == "v":
        g = min(FM.group_of(spec, x.shape[-1]), x.shape[-1])
        return x.reshape(*x.shape[:-1], -1, g), (lambda y: y.reshape(x.shape))
    ns = min(KV["sink"], x.shape[2] - 1) if KV["sink"] else 0
    tt = x[:, :, ns:].transpose(2, 3)                                    # [B, H, D, S']
    S = tt.shape[-1]
    g = FM.group_of(spec, S)
    pad = (-S) % g
    if pad:
        tt = torch.cat([tt, tt[..., -1:].expand(*tt.shape[:-1], pad)], -1)
    grp = tt.reshape(*tt.shape[:-1], -1, g)

    def back(y):
        y = y.reshape(*tt.shape)[..., :S].transpose(2, 3)
        return torch.cat([x[:, :, :ns], y], 2) if ns else y
    return grp, back


def _kvq_outliers(x: torch.Tensor, kind: str, li: int, basis: str = "I") -> torch.Tensor | None:
    """dense-and-sparse outlier mask in the layout of x [B, H, S, D] (keys pre-RoPE, in the given basis)."""
    if "ds" not in KV["kvq"].split("+"):
        return None
    if kind == "k":                                      # offline per-(head, channel) thresholds
        lo, hi = KV["kvq_thr"][li] if basis == "I" else KV["kvq_thr"][(li, basis)]
        return (x < lo[None, :, None, :]) | (x > hi[None, :, None, :])
    B, H, S, D = x.shape                                 # online: the 1 % largest |v| of each token over all KV heads
    a = x.abs().transpose(1, 2).reshape(B, S, H * D)
    kk = max(1, int(round(0.01 * H * D)))
    thr = a.topk(kk, -1).values[..., -1:]
    return (a >= thr).reshape(B, S, H, D).transpose(1, 2)


def _kvq_norm(grp: torch.Tensor, out: torch.Tensor | None):
    """min / max of the inliers of every group -> (u in [-1, 1], mn, rng)."""
    if out is None:
        mn, mx = grp.amin(-1, keepdim=True), grp.amax(-1, keepdim=True)
    else:
        big = torch.finfo(grp.dtype).max
        mn = torch.where(out, big, grp).amin(-1, keepdim=True)
        mx = torch.where(out, -big, grp).amax(-1, keepdim=True)
        allout = mn > mx                                 # a group made only of outliers: nothing dense to code
        mn, mx = torch.where(allout, 0.0, mn), torch.where(allout, 0.0, mx)
    rng = mx - mn
    u = (2 * (grp - mn) / torch.where(rng > 0, rng, torch.ones_like(rng)) - 1).clamp(-1, 1)
    return u, mn, rng


KVQ_RB = 32                                             # sub-vector of the selection on KVQuant-style keys


def _kvq_basis(x: torch.Tensor, basis: str) -> torch.Tensor:
    """I: identity; H: H_32 on every 32-channel block; F: H_D on the whole head (all involutory)."""
    return x if basis == "I" else fwht(x, KVQ_RB if basis == "H" else x.shape[-1])


def _kvq_core(x: torch.Tensor, kind: str, li: int, basis: str = "I", count: bool = True):
    """non-uniform (+ dense-and-sparse) quantization of x (pre-RoPE keys / values) in the given basis; returns the
    reconstruction in that basis and the outlier mask (None without +ds)."""
    spec = KV[kind]
    out = _kvq_outliers(x, kind, li, basis)
    grp, back = _kvq_layout(x, kind, spec)
    og = None if out is None else _kvq_layout(out, kind, spec)[0]
    u, mn, rng = _kvq_norm(grp, og)
    cb = KV["kvq_cb"][li][kind if basis == "I" else f"{kind}_{basis}"]
    mid = (cb[1:] + cb[:-1]) / 2
    q = cb[torch.bucketize(u.contiguous(), mid)]
    y = torch.where(rng > 0, (q + 1) / 2 * rng + mn, mn)
    if og is not None:
        y = torch.where(og, grp, y)
        if count and not STATE["calib"]:                 # outliers among the quantized (non-sink) tokens
            ns = min(KV["sink"], x.shape[2] - 1) if KV["sink"] else 0
            st = KV["kvqstat"].setdefault(kind, [0, 0])
            st[0] += int(out[:, :, ns:].sum())
            st[1] += out[:, :, ns:].numel()
    return back(y), out


def kvq_quant(t: torch.Tensor, kind: str, li: int) -> torch.Tensor:
    x = t.float()
    if kind == "k":
        x = _unrope(x)
        if "sel" in KV["kvq"].split("+"):
            return _rerope(_kvq_sel_keys(x, li))
    y, _ = _kvq_core(x, kind, li)
    return _rerope(y) if kind == "k" else y


def _kvq_sel_keys(x: torch.Tensor, li: int) -> torch.Tensor:
    """KVQuant-style keys + FlagRot selection (kv_kvq '+sel'): every pre-RoPE key group of 32 tokens is coded, per
    32-channel block, either as is (I) or after H_32 (H), or for the whole head after H_128 (F); each basis has its own
    calibrated codebook (and +ds thresholds).  Per-channel quantization makes the block choice exactly separable.  The
    choice minimizes the post-RoPE query-weighted error sum_t R_t(e_t)^T C_q R_t(e_t) over the group (C_q the
    calibrated query second moment, as FlagRot), cross-block terms dropped for the block choice."""
    B, H, S, D = x.shape
    nb = D // KVQ_RB
    ns = min(KV["sink"], S - 1) if KV["sink"] else 0
    C = KV["cq"][li]
    rec, outm = {}, {}
    for b in ("I", "H", "F"):
        y, o = _kvq_core(_kvq_basis(x, b), "k", li, b, count=False)
        rec[b] = _kvq_basis(y, b)
        outm[b] = o

    def qerr(e):                                          # [B, H, S] post-RoPE C_q-weighted error
        ep = _rerope(e)
        return torch.einsum("bhsi,hij,bhsj->bhs", ep, C, ep)

    blk = torch.arange(D, device=x.device) // KVQ_RB

    def blockerr(y):                                      # [B, H, S, nb]
        e = y - x
        return torch.stack([qerr(e * (blk == j)) for j in range(nb)], -1)

    def pool(e):                                          # sums over the 32-token groups that start after the sinks
        return torch.cat([e[:, :, :ns], _pool_tokens(e[:, :, ns:], 32, 2)], 2)
    ch = pool(blockerr(rec["H"])) < pool(blockerr(rec["I"]))                            # [B, H, S, nb]
    chx = ch.repeat_interleave(KVQ_RB, -1)
    base = torch.where(chx, rec["H"], rec["I"])
    full = (pool(qerr(rec["F"] - x)) < pool(qerr(base - x)))[..., None]                  # [B, H, S, 1]
    y = torch.where(full, rec["F"], base)
    if ns:
        y = torch.cat([x[:, :, :ns], y[:, :, ns:]], 2)
    if not STATE["calib"]:
        st = KV["kvqstat"].setdefault("sel", [0.0, 0.0, 0])
        st[0] += float((ch & ~full)[:, :, ns:].float().mean())
        st[1] += float(full[:, :, ns:].float().mean())
        st[2] += 1
        if outm["I"] is not None:                        # outliers of the chosen coding
            o = torch.where(full, outm["F"], torch.where(chx, outm["H"], outm["I"]))[:, :, ns:]
            so = KV["kvqstat"].setdefault("k", [0, 0])
            so[0] += int(o.sum())
            so[1] += o.numel()
    return y


@torch.no_grad()
def kvq_calibrate(module, key, value, query=None):
    """store the calibration pre-RoPE keys and values (FP32, [Hkv, tokens, D]) of every layer; with '+sel' also the
    post-RoPE query second moment C_q (as FlagRot's calibration)."""
    li = module.layer_idx
    if query is not None and "sel" in KV["kvq"].split("+"):
        Cq = _cq(query, key.shape[1], key.shape[-1])
        KV["cq"][li] = KV["cq"][li] + Cq if li in KV["cq"] else Cq
    k = _unrope(key.float())
    acc = KV["kvq_acc"].setdefault(li, {"k": [], "v": []})
    acc["k"].append(k)
    acc["v"].append(value.float())


def _lloyd(u: torch.Tensor, n: int = 16, iters: int = 60) -> torch.Tensor:
    """1-D k-means of the samples u in [-1, 1], initialized at the uniform grid (so the MSE never exceeds uniform)."""
    cb = torch.linspace(-1, 1, n, device=u.device, dtype=torch.float64)
    u = u.double()
    for _ in range(iters):
        idx = torch.bucketize(u, (cb[1:] + cb[:-1]) / 2)
        s_ = torch.zeros(n, dtype=torch.float64, device=u.device).index_add_(0, idx, u)
        c_ = torch.zeros(n, dtype=torch.float64, device=u.device).index_add_(0, idx, torch.ones_like(u))
        new = torch.where(c_ > 0, s_ / c_.clamp_min(1), cb)
        new, _ = new.sort()
        if torch.allclose(new, cb, atol=1e-9):
            break
        cb = new
    return cb.float()


@torch.no_grad()
def kvq_finalize(max_samples: int = 4_000_000):
    """key outlier thresholds (0.5 / 99.5 % quantiles per head and channel) and per-(layer, K/V) codebooks."""
    g = torch.Generator(device="cpu").manual_seed(0)
    for li, acc in KV["kvq_acc"].items():
        k, v = torch.cat(acc["k"], 2), torch.cat(acc["v"], 2)             # [B, Hkv, S_total, D]
        bases = ("I", "H", "F") if "sel" in KV["kvq"].split("+") else ("I",)
        if "ds" in KV["kvq"].split("+"):
            for b in bases:
                kk = _kvq_basis(k, b).permute(1, 3, 0, 2).reshape(k.shape[1], k.shape[3], -1)   # [Hkv, D, tokens]
                thr = (torch.quantile(kk, 0.005, dim=-1), torch.quantile(kk, 0.995, dim=-1))
                KV["kvq_thr"][li if b == "I" else (li, b)] = thr
        cbs = {}
        for kind, x, b in [("k", k, "I"), ("v", v, "I")] + [("k", _kvq_basis(k, b), b) for b in bases[1:]]:
            # normalize exactly as at evaluation time, per calibration window (the windows are concatenated along S, so
            # quantize each window's slice separately)
            us = []
            for xs in x.split(int(x.shape[2] // len(acc[kind])), 2):
                out = _kvq_outliers(xs, kind, li, b)
                grp, _ = _kvq_layout(xs, kind, KV[kind])
                og = None if out is None else _kvq_layout(out, kind, KV[kind])[0]
                u, _, rng = _kvq_norm(grp, og)
                keep = (rng > 0).expand_as(u) if og is None else ((rng > 0) & ~og)
                us.append(u[keep])
            u = torch.cat(us)
            if u.numel() > max_samples:
                u = u[torch.randint(0, u.numel(), (max_samples,), generator=g).to(u.device)]
            # 2^b levels of the int<b>a spec (16 for the INT4 rows, unchanged; 4 for int2a)
            nlev = 2 ** int(re.match(r"int(\d)", KV[kind]).group(1)) if re.match(r"int\d", KV[kind]) else 16
            cbs[kind if b == "I" else f"{kind}_{b}"] = _lloyd(u, n=nlev)
        KV["kvq_cb"][li] = cbs
    KV["kvq_acc"] = {}


def kvq_summary() -> dict:
    """measured dense-and-sparse outlier fractions (evaluation windows) and the codebooks' mean |level| spacing."""
    out = {k: (v[0] / v[1] if v[1] else 0.0) for k, v in KV["kvqstat"].items() if k in ("k", "v")}
    sel = KV["kvqstat"].get("sel")
    if sel:
        out = dict(outlier_frac=out, sel=dict(block_H=sel[0] / sel[2], head_F=sel[1] / sel[2]))
        return dict(mode=KV["kvq"], **out)
    cbs = {f"cb_{kd}_layer0": KV["kvq_cb"][min(KV["kvq_cb"])][kd].tolist() for kd in ("k", "v")} if KV["kvq_cb"] else {}
    return dict(mode=KV["kvq"], outlier_frac=out, **cbs)


# ---------------------------------------------------------------------------------------------------------------------
# RotateKV (kv_rotatekv; Su et al., "RotateKV: Accurate and Robust 2-Bit KV Cache Quantization for LLMs via
# Outlier-Aware Adaptive Rotations", IJCAI 2025; official code github.com/ZunhaiSu/RotateKV @3055e2b).  Keys: pre-RoPE, grouped-head Walsh-Hadamard over G consecutive KV heads (G * D
# channels, fused into k_proj in their code), then one calibrated permutation of all Hkv * D channels per layer (argsort
# of the calibration channel sums of the rotated keys), per-token groups of g of the permuted channels, inverse
# permutation and rotation, then RoPE.  Values: per-head H_D (fused into v_proj / o_proj), per-token groups of g.
# Attention-sink-aware quantization: the tokens holding the top-n |x| entries of the layer's input residual stream stay
# in full precision.  Prefill only (the KL / PPL protocol); the quantizer is the kv_k / kv_v spec (int<b>ac<clip>:<g>).
# ---------------------------------------------------------------------------------------------------------------------
def rk_group(Hkv: int, G: int) -> int:
    """heads per grouped-head rotation: G if it divides Hkv, else the largest power of two <= G dividing Hkv (Phi-4:
    10 KV heads -> 2, GLM-4-9B: 2 KV heads -> 2)."""
    while G > 1 and Hkv % G:
        G //= 2
    return max(G, 1)


def _rk_rot(x: torch.Tensor, G: int) -> torch.Tensor:
    """[B, Hkv, S, D] -> [B, S, Hkv * D] after the grouped-head Hadamard (block-diagonal H_{G D}; involutory)."""
    B, H, S, D = x.shape
    return fwht(x.transpose(1, 2).reshape(B, S, H * D), G * D)


def _rk_unrot(y: torch.Tensor, G: int, H: int) -> torch.Tensor:
    B, S, HD = y.shape
    return fwht(y, G * (HD // H)).reshape(B, S, H, HD // H).transpose(1, 2)


def rk_quant_k(k: torch.Tensor, li: int) -> torch.Tensor:
    """pre-RoPE keys [B, Hkv, S, D] -> fake-quantized pre-RoPE keys (original basis, FP32)."""
    B, H, S, D = k.shape
    G = KV["rk"]["G"]
    y = _rk_rot(k.float(), G)                                            # [B, S, H D]
    perm = KV["rk_perm"].get(li) if KV["rk"]["ro"] else None
    if perm is not None:
        y = y.index_select(-1, perm)
    xs = y.reshape(B, S, H, D).transpose(1, 2)
    yq = (_rk_select(xs, li) if KV["rk"].get("sel") else FM.fake(xs, KV["k"])).transpose(1, 2).reshape(B, S, H * D)
    if perm is not None:
        yq = yq.index_select(-1, torch.argsort(perm))
    return _rk_unrot(yq, G, H)


def rk_quant_v(v: torch.Tensor, spec: str) -> torch.Tensor:
    """values: per-head Walsh-Hadamard H_D (offline in v_proj / o_proj), per-token groups of g, rotated back."""
    D = v.shape[-1]
    return fwht(FM.fake(fwht(v.float(), D), spec), D)


def rk_install(model) -> list:
    """wrap the apply_rotary_pos_emb of every attention module's modeling file once (inert unless KV['rk'] is set and
    an attention forward is running) and register the per-layer hooks: attention pre / post hooks set rk_cur, decoder
    layer pre-hooks find the massive-activation sink tokens.  Returns the hook handles."""
    import sys
    hs = []
    layers = None
    for li, at in attn_layers(model).items():
        mod = sys.modules[type(at).__module__]
        f = getattr(mod, "apply_rotary_pos_emb")
        if not getattr(f, "_rk_wrapped", False):
            mod.apply_rotary_pos_emb = _rk_wrap(f)
        hs.append(at.register_forward_pre_hook(lambda m, a, li=li: KV.__setitem__("rk_cur", li)))
        hs.append(at.register_forward_hook(lambda m, a, o: KV.__setitem__("rk_cur", None)))
        if KV["rk"] and KV["rk"]["ma"]:
            if layers is None:
                from .modeling import decoder_layers
                layers = decoder_layers(model)
            hs.append(layers[li].register_forward_pre_hook(
                lambda m, a, kw, li=li: _rk_sinks(li, a[0] if a else kw["hidden_states"]), with_kwargs=True))
    return hs


def _rk_sinks(li: int, h: torch.Tensor):
    """official modeling_llama_RotateKV.py 1226-1232: token index of the top-n |x| entries of the decoder layer's input
    hidden states (flattened over tokens x hidden), per batch element; only while the quantized cache is evaluated."""
    if KV["rk"] is None or not STATE["quant"] or STATE["calib"]:
        return
    B, S, Hd = h.shape
    n = KV["rk"]["ma"]
    idx = torch.topk(h.detach().abs().reshape(B, -1), min(n, S * Hd), dim=-1).indices // Hd    # [B, n]
    m = torch.zeros(B, S, dtype=torch.bool, device=h.device)
    m.scatter_(1, idx, True)
    KV["rk_piv"][li] = m
    st = KV["rk_stat"]
    st["n"] = st.get("n", 0) + B
    st["tokens"] = st.get("tokens", 0) + float(m.sum())
    st["seq"] = st.get("seq", 0) + B * S
    st["first"] = st.get("first", 0) + float(m[:, 0].sum())


def _rk_wrap(orig):
    def wrapped(q, k, cos, sin, *a, **kw):
        qe, ke = orig(q, k, cos, sin, *a, **kw)
        li = KV.get("rk_cur")
        if KV["rk"] is None or li is None:
            return qe, ke
        bl = KV["bf16_layers"]
        if KV["skip"] and (li in bl if bl is not None else (li < KV["skip"] or li >= KV["nl"] - KV["skip"])):
            return qe, ke
        if STATE["calib"] and KV["rk_calib"]:                       # Algorithm 1: channel sums of the rotated keys
            s_ = _rk_rot(k.float(), KV["rk"]["G"]).double().sum((0, 1))
            KV["rk_acc"][li] = KV["rk_acc"][li] + s_ if li in KV["rk_acc"] else s_
            if KV["rk"]["sel"]:
                _rk_sel_calib(li, qe, k, cos, sin, orig, a, kw)
        elif STATE["quant"] and not STATE["calib"] and KV["k"] != "bf16":
            _, kq = orig(q, rk_quant_k(k, li), cos, sin, *a, **kw)
            KV["rk_k"][li] = kq.float()
        return qe, ke
    wrapped._rk_wrapped = True
    wrapped.__wrapped__ = orig
    return wrapped


def rk_finalize():
    """reordering indices P_l = argsort(channel_sum_l) (ascending, as Algorithm 1 / k_reordering_calibration.py);
    with +sel also the selection weights of every layer in the stored (rotated, permuted) basis."""
    KV["rk_perm"] = {li: torch.argsort(s_).to(torch.long) for li, s_ in KV["rk_acc"].items()} if KV["rk"]["ro"] else {}
    KV["rk_acc"] = {}
    if KV["rk"]["sel"]:
        for li, W in KV["rk_W"].items():
            KV["rk_M"][li] = _rk_sel_matrix(W, li)
        KV["rk_W"] = {}


# RotateKV + FlagRot selection (kv_rotatekv '+sel'): FlagRot's per-token key encoder (cq+f+konly:
# per 32-channel sub-vector identity or H_32, per head slot additionally H_D on the whole slot, lowest weighted error)
# applied to the stored RotateKV representation, i.e. after the grouped-head rotation and the channel reordering,
# before RoPE.  Weight: FlagRot's calibrated post-RoPE query second moment C_q (per KV head) mapped to the pre-RoPE
# basis and averaged over the key positions of a calibration window, W = mean_t R_t^T C_q R_t (R_t = the model's own
# RoPE at position t, read from its apply_rotary_pos_emb call, so partial / interleaved RoPE need nothing special),
# then to the stored basis, M = Pi H_G blockdiag(W) H_G Pi^T; as in FlagRot the cross-block terms of M are dropped
# for the sub-vector choice and the slot-diagonal block decides the whole-slot option.
RK_RB = 32


def _rk_sel_calib(li, qe, k, cos, sin, orig, a, kw):
    B, H, S, D = qe.shape
    Hkv = k.shape[1]
    q = qe.float().reshape(B, Hkv, H // Hkv, S, D)
    C = torch.einsum("bhrsd,bhrse->hde", q, q)                         # post-RoPE C_q of this window [Hkv, D, D]
    eye = torch.eye(D, device=k.device, dtype=torch.float32)[None, :, None, :].expand(1, D, S, D).contiguous()
    _, Ob = orig(eye, eye, cos[:1], sin[:1], *a, **kw)                  # Ob[0, i, t] = R_t e_i
    O = Ob[0].float().transpose(0, 1)                                   # [S, D, D], O[t, i] = R_t e_i
    W = torch.zeros(Hkv, D, D, dtype=torch.float64, device=k.device)
    for t0 in range(0, S, 256):
        Ot = O[t0:t0 + 256]
        W += torch.einsum("tac,hcd,tbd->hab", Ot, C, Ot).double()
    W /= S
    KV["rk_W"][li] = KV["rk_W"][li] + W if li in KV["rk_W"] else W


def _rk_sel_matrix(W: torch.Tensor, li: int) -> torch.Tensor:
    """slot-diagonal blocks [Hkv, D, D] of M = Pi H_G blockdiag(W) H_G Pi^T (row / column vectors alike: H symmetric)."""
    Hkv, D, _ = W.shape
    G = KV["rk"]["G"]
    A = torch.block_diag(*W.unbind(0))                                  # [Hkv D, Hkv D]
    A = fwht(fwht(A, G * D).transpose(0, 1).contiguous(), G * D).transpose(0, 1)
    perm = KV["rk_perm"].get(li)
    if perm is not None:
        A = A.index_select(0, perm).index_select(1, perm)
    return torch.stack([A[h * D:(h + 1) * D, h * D:(h + 1) * D] for h in range(Hkv)]).float().contiguous()


def _rk_select(x: torch.Tensor, li: int) -> torch.Tensor:
    """FlagRot's per-token choice on the stored RotateKV slots x [B, Hkv, S, D] -> reconstruction (same basis)."""
    spec, D = KV["k"], x.shape[-1]
    Wm = ("mat", KV["rk_M"][li])
    r, ch = _local_best(x, spec, RK_RB, Wm, ["H"])
    hf = fwht(FM.fake(fwht(x, D), spec), D)
    full = (_head_err(hf - x, Wm) < _head_err(r - x, Wm))[..., None]
    st = KV["rk_stat"]
    st["sel_n"] = st.get("sel_n", 0) + 1
    st["sel_block_H"] = st.get("sel_block_H", 0.0) + float(((ch > 0) & ~full).float().mean())
    st["sel_slot_F"] = st.get("sel_slot_F", 0.0) + float(full.float().mean())
    return torch.where(full, hf, r)


def rk_summary() -> dict:
    st = KV["rk_stat"]
    out = dict(G=KV["rk"]["G"], reorder=KV["rk"]["ro"], sinks_topn=KV["rk"]["ma"])
    if KV["rk"].get("sel") and st.get("sel_n"):
        out.update(sel=True, sel_block_H=st["sel_block_H"] / st["sel_n"], sel_slot_F=st["sel_slot_F"] / st["sel_n"])
    if st.get("n"):
        out.update(sink_tokens_per_layer=st["tokens"] / st["n"], sink_frac=st["tokens"] / st["seq"],
                   sink_first_token_frac=st["first"] / st["n"])
    return out
