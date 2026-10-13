"""kivi2: FlagRot-KIVI INT2 pages (csrc/kivi.cu, vllm_flagrot/kivi.py) vs the research simulation.

  CUDA_VISIBLE_DEVICES=2 python integrations/vllm_flagrot/tests/test_kivi.py [--calib F]

Checks:
  1. key group encoder (kv_k_encode) + page parser == blockflag qattention.dyn_quant(kivi2:32, axis token, att+f)
     on whole 32-token groups and on the first group [S, 32) (28 tokens, sinks excluded = the plugin's option-A
     alignment), with the real exported calibration queries (qs, lz, qscale) of a model; head dim 128 / 256.
     Criterion per (group, KV head): ||rec_vllm - rec_sim|| / ||rec_sim - x|| (the plugin stores FP16 scale / min,
     the simulation keeps them in FP32: ~1e-3 when codes and choices agree; a flipped choice is O(1)).
  2. value writer (kv_v_write) == dyn_quant(kivi2:32, axis head, att+f) per token (diag c_o; KIVI-C: matrix M).
  3. decode attention (kv_attn) vs float64 attention over the parsed pages + BF16 sinks / window rows
     (kv_start = S, cached = L - W, tail = min(L, S + W) slots), GQA 4 / 6 / 8, head 128 / 256.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE.parents[2]))
from vllm_flagrot import kivi as KV  # noqa: E402
from blockflag import qattention as QA  # noqa: E402

DEV = torch.device("cuda")
F8 = False          # --f8: kivi2p8 (kivi2f8:32: FP8 scale / min)
SPEC = "kivi2:32"
PAGE, GRP = 16, 32


def sim_state(qs, lz, qscale, co, X=None):
    QA.KV.update(dyn=dict(rule="att", local=["H"], head=["F"], group=1, only=None), sink=0, qs={0: qs}, lz={0: lz},
                 qscale=qscale, co={0: co}, xf=({0: X} if X is not None else None), dynstat={})


def keys(T, Hkv, D, g):
    """post-RoPE-like keys: per-channel offsets (large on a few channels), per-token noise, a few outlier tokens"""
    mu = torch.randn(Hkv, D, generator=g) * 2.0
    mu[:, torch.randint(0, D, (6,), generator=g)] *= 6
    x = mu[None] + torch.randn(T, Hkv, D, generator=g) * (0.5 + torch.rand(Hkv, D, generator=g))
    x[torch.randint(0, T, (3,), generator=g)] *= 3
    return x.bfloat16()


def alloc(npg_needed, Hkv, D, g):
    nb = npg_needed + 7
    pool = torch.zeros(nb, Hkv, KV.chunk_bytes(D, F8), dtype=torch.uint8, device=DEV)
    perm = torch.randperm(nb, generator=g)[:npg_needed].int().to(DEV)
    return pool, perm


def sim_choice_k(xf, D):
    """per 32-token group of xf [1, H, T, D] (T a multiple of 32 or one partial group): (local bits [ng, H, D/32],
    whole-head bit [ng, H]) of the simulation's choice (= dyn_quant(axis token, att+f) internals)"""
    from blockflag.hadamard import fwht
    W = QA._weights("k", 0, xf)
    Q = lambda y: QA._fake_tok(y, SPEC)  # noqa: E731
    recs = [Q(xf), fwht(Q(fwht(xf, 32)), 32)]
    errs = QA._pool_tokens(torch.stack([QA._vec_err(r - xf, W, 32) for r in recs]), 32, 3)
    ch = errs.argmin(0)
    sel = ch[..., None].expand(*ch.shape, 32).reshape(xf.shape)
    r = torch.where(sel == 1, recs[1], recs[0])
    F = fwht(Q(fwht(xf, D)), D)
    he = QA._pool_tokens(torch.stack([QA._head_err(h - xf, W) for h in (r, F)]), 32, 3)
    hs = he.argmin(0)
    idx = torch.arange(0, xf.shape[2], 32, device=xf.device)
    return ch[0][:, idx].permute(1, 0, 2).bool(), hs[0][:, idx].permute(1, 0).bool()


def page_flags_k(pool, bt, D, Hkv):
    ch = pool.index_select(0, bt.long())
    _, km, _, _, _ = KV.parse_chunks(ch, D, f8=F8)
    NV = D // 32
    ng = ch.shape[0] // 2
    kmg = km.reshape(ng, 2, Hkv, PAGE, NV, 2).permute(0, 2, 1, 3, 4, 5).reshape(ng, Hkv, D, 2).float()
    CHL = NV if F8 else 1
    return torch.signbit(kmg[..., 0][..., 0:NV * CHL:CHL]), torch.signbit(kmg[..., 0][..., NV * CHL])


def rel_err_groups(rec, ref, x, dims):
    d = (rec - ref).double().pow(2).sum(dims).sqrt()
    n = (ref - x).double().pow(2).sum(dims).sqrt().clamp_min(1e-12)
    return d / n


def test_keys(cal, li, D, Hkv, g, S=4, ng=12):
    qs, lz = cal["qs"][li].to(DEV), cal["lz"][li].to(DEV)
    if "xq" in cal:
        pass                                  # qs already in the base's basis (kivi_c); keys = any primed vectors
    qscale = float(cal["qscale"])
    co = cal["co"][li].to(DEV)
    T = ng * GRP
    x = keys(T, Hkv, D, g)                      # [T, Hkv, D] bf16
    xs = x.to(DEV)
    pool, bt = alloc(T // PAGE, Hkv, D, g)
    pos = torch.arange(T, device=DEV)
    slots = bt[pos // PAGE].long() * PAGE + pos % PAGE
    sidx = pos.view(ng, GRP).clone()
    sidx[0, :S] = -1                           # sinks are not part of group 0
    dslot = slots.view(ng, GRP).contiguous()
    src = torch.cat([xs, xs], 1).contiguous()  # rows [T, 2 Hkv, D] (K rows first)
    KV.k_encode(src, sidx.contiguous(), dslot, qs, lz, qscale, pool, Hkv, D)
    torch.cuda.synchronize()
    k, _ = KV.dequant(pool, bt, 0, T, Hkv, D)  # [T, Hkv, D]
    # simulation: each group on its own (sink=0); group 0 = tokens [S, 32); its per-group choices from the same
    # functions dyn_quant uses (qattention._weights / _vec_err / _head_err / _pool_tokens)
    sim_state(qs, lz, qscale, co)
    xf = xs.float().permute(1, 0, 2)[None]     # [1, Hkv, T, D]
    ref = torch.empty_like(xf)
    ref[:, :, :S] = xf[:, :, :S]
    mism = 0
    ngh = 0
    for a, b_ in ((S, GRP), (GRP, T)):
        ref[:, :, a:b_] = QA.dyn_quant(xf[:, :, a:b_], SPEC, 32, "k", 0, "token")
        lb, fb = sim_choice_k(xf[:, :, a:b_], D)
        mlb, mfb = page_flags_k(pool, bt, D, Hkv)
        g0 = a // GRP
        for gi in range(lb.shape[0]):
            for h in range(Hkv):
                ngh += 1
                sf = bool(fb[gi, h]); mf = bool(mfb[g0 + gi, h])
                if sf != mf or (not sf and lb[gi, h].tolist() != mlb[g0 + gi, h].tolist()):
                    mism += 1
    kk = k.permute(1, 0, 2)[None]
    kk = torch.cat([xf[:, :, :S], kk[:, :, S:]], 2)
    e_v = (kk - xf).double().pow(2).sum()
    e_s = (ref - xf).double().pow(2).sum()
    r = rel_err_groups(kk.reshape(1, Hkv, ng, GRP, D), ref.reshape(1, Hkv, ng, GRP, D),
                       xf.reshape(1, Hkv, ng, GRP, D), (3, 4))
    print(f"keys  D {D} layer {li}: groups x heads {ngh}, choice mismatches {mism}; total error vLLM / sim "
          f"{float(e_v / e_s):.4f}; ||rec_vllm - rec_sim|| / ||rec_sim - x|| median {float(r.median()):.2e} "
          f"max {float(r.max()):.2e}")
    bad = mism
    return bad <= max(1, r.numel() // 50)


def test_values(cal, li, D, Hkv, g, T=300):
    qs, lz = cal["qs"][li].to(DEV), cal["lz"][li].to(DEV)
    qscale = float(cal["qscale"])
    co = cal["co"][li].to(DEV)
    M = cal["vM"][li].to(DEV).contiguous() if "vM" in cal else None
    x = keys(T, Hkv, D, g).float() * 0.3
    x = x.bfloat16().to(DEV)
    npg = (T + PAGE - 1) // PAGE
    npg += npg % 2
    pool, bt = alloc(npg, Hkv, D, g)
    pos = torch.arange(T, device=DEV)
    slots = bt[pos // PAGE].long() * PAGE + pos % PAGE
    src = torch.cat([x, x], 1).contiguous()
    KV.v_write(src, slots.contiguous(), co.contiguous(), M, pool, Hkv, D)
    torch.cuda.synchronize()
    _, v = KV.dequant(pool, bt, 0, T, Hkv, D) if T % GRP == 0 else _dq_v(pool, bt, T, Hkv, D)
    X = None
    if M is not None:
        X = {"vinv": cal["xvinv"][li].to(DEV), "q": cal["xq"][li].to(DEV)}
    sim_state(qs if X is None else qs, lz, qscale, co, X)
    if X is not None:
        # _weights maps qs by X['q'] itself: hand it the ORIGINAL-basis queries (the file stores mapped ones)
        QA.KV["qs"] = {0: torch.einsum("hnd,hde->hne", qs, torch.linalg.inv(X["q"]))}
    xf = x.float().permute(1, 0, 2)[None]
    ref = QA.dyn_quant(xf, SPEC, 32, "v", 0)
    vv = v[:T].permute(1, 0, 2)[None]
    r = rel_err_groups(vv, ref, xf, (3,))
    bad = int((r > 0.2).sum())
    e_v = (vv - xf).double().pow(2).sum()
    e_s = (ref - xf).double().pow(2).sum()
    print(f"values D {D} layer {li}{' (M)' if M is not None else ''}: tokens x heads {r.numel()}, differing "
          f"(rel > 0.2) {bad}, total error vLLM / sim {float(e_v / e_s):.4f}, rel median {float(r.median()):.2e}")
    return bad <= max(1, r.numel() // 200)


def _dq_v(pool, bt, T, Hkv, D):
    n = (T + GRP - 1) // GRP * GRP
    return KV.dequant(pool, bt, 0, n, Hkv, D)


def test_attn(cal, li, D, Hkv, G, g, S=4, W=128, Ls=(1, 5, 100, 132, 133, 161, 300, 1000, 2077)):
    qs, lz = cal["qs"][li].to(DEV), cal["lz"][li].to(DEV)
    qscale = float(cal["qscale"])
    co = cal["co"][li].to(DEV)
    if qs.shape[0] != Hkv or qs.shape[-1] != D:
        qs = torch.randn(Hkv, 512, D, device=DEV) * 0.3
        lz = torch.full((Hkv, 512), 5.0, device=DEV)
        co = torch.rand(Hkv, D, device=DEV)
    B = len(Ls)
    Lmax = max(Ls)
    npgr = ((Lmax + GRP - 1) // GRP) * 2
    T = S + W
    nb = B * npgr + 5
    pool = torch.zeros(nb, Hkv, KV.chunk_bytes(D, F8), dtype=torch.uint8, device=DEV)
    perm = torch.randperm(nb, generator=g)[:B * npgr].view(B, npgr).int().to(DEV)
    ring = torch.zeros(B + 1, T, 2 * Hkv, D, dtype=torch.bfloat16, device=DEV)
    q = (torch.randn(B, Hkv * G, D, generator=g) * 0.5).bfloat16().to(DEV)
    scale = 1 / math.sqrt(D)
    outs_ref = []
    kvs = []
    for b, L in enumerate(Ls):
        k = keys(L, Hkv, D, g).to(DEV)
        v = (keys(L, Hkv, D, g).float() * 0.3).bfloat16().to(DEV)
        kv = torch.cat([k, v], 1).contiguous()
        pos = torch.arange(L, device=DEV)
        slots = perm[b][pos // PAGE].long() * PAGE + pos % PAGE
        KV.v_write(kv, slots, co, None, pool, Hkv, D)
        ngc = L // GRP                                      # complete groups
        if ngc:
            gp = torch.arange(ngc * GRP, device=DEV).view(ngc, GRP)
            sidx = torch.where(gp < S, torch.full_like(gp, -1), gp).contiguous()
            dslot = (perm[b][gp // PAGE].long() * PAGE + gp % PAGE).contiguous()
            KV.k_encode(kv, sidx, dslot, qs, lz, qscale, pool, Hkv, D)
        # ring: sinks [0, S), window ring over the newest W tokens >= S
        for p in range(L):
            if p < S:
                ring[b, p] = kv[p]
            elif p >= L - W:
                ring[b, S + (p - S) % W] = kv[p]
        kvs.append(kv)
    torch.cuda.synchronize()
    cached = torch.tensor([L - W if L - W > S else 0 for L in Ls], dtype=torch.int32, device=DEV)
    tl = torch.tensor([min(L, T) for L in Ls], dtype=torch.int32, device=DEV)
    ti = torch.arange(B, dtype=torch.int32, device=DEV)
    ns, chunk = KV.plan(B, Lmax, Hkv)
    for ns_, ch_ in ((ns, chunk), (max(1, (Lmax + 255) // 256), 256)):
        ws = KV.Workspace(B * Hkv * (ns_ + 1), G, D, DEV)
        out = torch.zeros(B, Hkv * G * D, dtype=torch.bfloat16, device=DEV)
        KV.attn(q, pool, perm, cached, out, ns_, ch_, ws, scale * 1.4426950408889634, S, G, ring=ring.view(-1, 2 * Hkv, D),
                tail_len=tl, tail_idx=ti, T=T)
        torch.cuda.synchronize()
        worst = 0.0
        for b, L in enumerate(Ls):
            kv = kvs[b].double()
            kq, vq = kv[:, :Hkv].clone(), kv[:, Hkv:].clone()
            c = int(cached[b])
            if c > S:
                kd, vd = KV.dequant(pool, perm[b], S, c, Hkv, D)
                kq[S:c], vq[S:c] = kd.double(), vd.double()
            qb = q[b].double().view(Hkv, G, D)
            sc = torch.einsum("hgd,thd->hgt", qb, kq) * scale
            p = torch.softmax(sc, -1)
            o = torch.einsum("hgt,thd->hgd", p, vq).reshape(-1)
            e = float((out[b].double() - o).abs().max() / o.abs().max().clamp_min(1e-9))
            worst = max(worst, e)
        print(f"attn  D {D} G {G} nsplit {ns_} chunk {ch_}: worst max|o - ref| / max|ref| = {worst:.2e}")
        if worst > 1e-2:
            return False
    return True


def test_dequant(cal, D, Hkv, g, T=640):
    """CUDA page dequantization (kv_dequant) == the PyTorch parser (kivi.dequant), bf16 / fp32, partial ranges"""
    li = 0
    qs, lz = cal["qs"][li].to(DEV), cal["lz"][li].to(DEV)
    co = cal["co"][li].to(DEV)
    x = keys(T, Hkv, D, g).to(DEV)
    v = (keys(T, Hkv, D, g).float() * 0.3).bfloat16().to(DEV)
    kv = torch.cat([x, v], 1).contiguous()
    pool, bt = alloc(T // PAGE, Hkv, D, g)
    pos = torch.arange(T, device=DEV)
    slots = bt[pos // PAGE].long() * PAGE + pos % PAGE
    KV.v_write(kv, slots, co, None, pool, Hkv, D)
    gp = pos.view(-1, GRP)
    KV.k_encode(kv, torch.where(gp < 4, -1, gp).contiguous(), slots.view(-1, GRP).contiguous(), qs, lz,
                float(cal["qscale"]), pool, Hkv, D)
    worst = 0.0
    for a, n in ((0, T), (4, 300), (96, 97), (37, T)):
        k0, v0 = KV.dequant(pool, bt, a, n, Hkv, D)
        for dt in (torch.float32, torch.bfloat16):
            k1, v1 = KV.dequant_cuda(pool, bt, a, n, Hkv, D, dt)
            e = max(float((k1.float() - k0).abs().max() / k0.abs().max()), float((v1.float() - v0).abs().max() / v0.abs().max()))
            if dt == torch.float32:
                worst = max(worst, e)
            else:
                worst = max(worst, e / 50)                     # bf16 rounding of the output
    print(f"dequant D {D}: CUDA vs PyTorch parser worst rel {worst:.2e}")
    return worst < 1e-4


def test_ident(cal, D, Hkv, g, ng=8, S=4):
    """kivi2id: identity-only writers == the simulation's plain KIVI (research config 'KIVI': kv_quant axis token /
    head, no transform): keys = qattention._fake_tok_ per 32-token group (group 0 = [S, 32)), values =
    formats.fake(kivi2:32) per token; every flag 0."""
    from blockflag import formats as FM
    T = ng * GRP
    x = keys(T, Hkv, D, g).to(DEV)
    v = (keys(T, Hkv, D, g).float() * 0.3).bfloat16().to(DEV)
    kv = torch.cat([x, v], 1).contiguous()
    pool, bt = alloc(T // PAGE, Hkv, D, g)
    pos = torch.arange(T, device=DEV)
    slots = bt[pos // PAGE].long() * PAGE + pos % PAGE
    qs, lz, co = cal["qs"][0].to(DEV), cal["lz"][0].to(DEV), cal["co"][0].to(DEV)
    KV.v_write(kv, slots, co, None, pool, Hkv, D, ident=True)
    gp = pos.view(ng, GRP)
    KV.k_encode(kv, torch.where(gp < S, -1, gp).contiguous(), slots.view(ng, GRP).contiguous(), qs, lz,
                float(cal["qscale"]), pool, Hkv, D, ident=True)
    torch.cuda.synchronize()
    k, vv = KV.dequant(pool, bt, 0, T, Hkv, D)
    xf = x.float().permute(1, 0, 2)[None]
    ref = torch.cat([QA._fake_tok_(xf[:, :, S:GRP], SPEC), QA._fake_tok_(xf[:, :, GRP:], SPEC)], 2)
    kk = k.permute(1, 0, 2)[None][:, :, S:]
    rk = rel_err_groups(kk, ref, xf[:, :, S:], (2, 3))
    vref = FM.fake(v.float(), SPEC)
    rv = rel_err_groups(vv, vref, v.float(), (1, 2))
    ch = pool.index_select(0, bt.long())
    _, km, _, vm, vf = KV.parse_chunks(ch, D, f8=F8)
    flags = int(torch.signbit(km[..., 0].float()).sum() + torch.signbit(vm[..., 0].float()).sum() + (vf != 0).sum())
    print(f"ident D {D}: keys rel {float(rk.max()):.2e}, values rel {float(rv.max()):.2e}, flags set {flags}")
    return float(rk.max()) < 1e-2 and float(rv.max()) < 1e-2 and flags == 0


def test_fixed(cal, D, Hkv, g, ng=8, S=4):
    """kivi2s: writers with frozen choices (fix words: bits 0..D/32-1 = H32 per sub-vector, bit 16 = H_D) == the
    simulation's static path (qattention.kv_quant with kmask / kfull: rot -> _fake_tok / fake -> rot back)"""
    T = ng * GRP
    NV = D // 32
    x = keys(T, Hkv, D, g).to(DEV)
    v = (keys(T, Hkv, D, g).float() * 0.3).bfloat16().to(DEV)
    kv = torch.cat([x, v], 1).contiguous()
    pool, bt = alloc(T // PAGE, Hkv, D, g)
    pos = torch.arange(T, device=DEV)
    slots = bt[pos // PAGE].long() * PAGE + pos % PAGE
    qs, lz, co = cal["qs"][0].to(DEV), cal["lz"][0].to(DEV), cal["co"][0].to(DEV)
    gg = torch.Generator().manual_seed(5)
    masks, fulls, fix = {}, {}, {}
    for kind in "kv":
        fulls[kind] = (torch.rand(Hkv, generator=gg) < 0.3).to(DEV)
        masks[kind] = (torch.rand(Hkv, NV, generator=gg) < 0.5).to(DEV) & ~fulls[kind][:, None]
        fix[kind] = ((masks[kind].int() << torch.arange(NV, device=DEV, dtype=torch.int32)).sum(-1) |
                     (fulls[kind].int() << 16)).int().contiguous()
    KV.v_write(kv, slots, co, None, pool, Hkv, D, fix=fix["v"])
    gp = pos.view(ng, GRP)
    KV.k_encode(kv, torch.where(gp < S, -1, gp).contiguous(), slots.view(ng, GRP).contiguous(), qs, lz,
                float(cal["qscale"]), pool, Hkv, D, fix=fix["k"])
    torch.cuda.synchronize()
    k, vv = KV.dequant(pool, bt, 0, T, Hkv, D)
    xf = x.float().permute(1, 0, 2)[None]
    vf = v.float().permute(1, 0, 2)[None]
    rk = QA.rot(xf, masks["k"], fulls["k"], 32)
    QA.KV["sink"] = 0
    kq = torch.cat([QA._fake_tok_(rk[:, :, S:GRP], SPEC), QA._fake_tok_(rk[:, :, GRP:], SPEC)], 2)
    kref = QA.rot(QA.rot(kq, None, fulls["k"], 32), masks["k"], None, 32)
    from blockflag import formats as FM
    vq = FM.fake(QA.rot(vf, masks["v"], fulls["v"], 32), SPEC)
    vref = QA.rot(QA.rot(vq, None, fulls["v"], 32), masks["v"], None, 32)
    kk = k.permute(1, 0, 2)[None][:, :, S:]
    ek = rel_err_groups(kk, kref, xf[:, :, S:], (2, 3))
    ev = rel_err_groups(vv.permute(1, 0, 2)[None], vref, vf, (2, 3))
    lb, fb = page_flags_k(pool, bt, D, Hkv)
    flag_ok = bool((lb == masks["k"][None]).all() and (fb == fulls["k"][None]).all())
    print(f"fixed D {D}: keys rel {float(ek.max()):.2e}, values rel {float(ev.max()):.2e}, key flags as fixed {flag_ok}")
    return float(ek.max()) < 1e-2 and float(ev.max()) < 1e-2 and flag_ok


def test_vf_preserve(cal, D, Hkv, g, ng=6, S=4):
    """kivi2p8: the value whole-head flag lives in the sign bit of a key scale (cell i, slot NV - 1); the key group
    encoder writes those bytes later and must keep it.  Interleaved order: V of group 0, K of group 0, V of groups
    1.., K of group 1 then K of group 0 again (re-encode), V of the last group after everything else."""
    if not F8:
        return True
    T = ng * GRP
    x = keys(T, Hkv, D, g).to(DEV)
    v = (keys(T, Hkv, D, g).float() * 0.3).bfloat16().to(DEV)
    kv = torch.cat([x, v], 1).contiguous()
    pool, bt = alloc(T // PAGE, Hkv, D, g)
    pos = torch.arange(T, device=DEV)
    slots = bt[pos // PAGE].long() * PAGE + pos % PAGE
    qs, lz, co = cal["qs"][0].to(DEV), cal["lz"][0].to(DEV), cal["co"][0].to(DEV)
    gp = pos.view(ng, GRP)
    sid = torch.where(gp < S, -1, gp).contiguous()
    ds = slots.view(ng, GRP).contiguous()

    def vflags():
        ch = pool.index_select(0, bt.long())
        return KV.parse_chunks(ch, D, f8=True)[4].clone()            # [np, Hkv, 16]

    def venc(a, b):
        KV.v_write(kv[a:b], slots[a:b].contiguous(), co, None, pool, Hkv, D)

    def kenc(gs):
        KV.k_encode(kv, sid[gs].contiguous(), ds[gs].contiguous(), qs, lz, float(cal["qscale"]), pool, Hkv, D)
    venc(0, GRP)
    kenc([0])
    venc(GRP, T - GRP)
    kenc(list(range(1, ng - 1)))
    kenc([0])
    venc(T - GRP, T)
    kenc([ng - 1])
    torch.cuda.synchronize()
    after = vflags()
    pool.zero_()                                                     # reference: values only
    venc(0, T)
    torch.cuda.synchronize()
    ref = vflags()
    n = int(ref.sum())
    ok = bool(torch.equal(after, ref)) and n > 0
    print(f"vf preserve D {D}: value whole-head flags set {n} of {ref.numel()}, unchanged by key encoding: {torch.equal(after, ref)}")
    return ok


FAST = False


def _wrap_f8():
    import functools
    for n in ("v_write", "k_encode", "attn", "dequant", "dequant_cuda"):
        f = getattr(KV, n)

        def g(*a, _f=f, _n=n, **k):
            k.setdefault("f8", F8)
            if _n == "attn":
                k.setdefault("fast", FAST)
            return _f(*a, **k)
        setattr(KV, n, functools.wraps(f)(g))


def main():
    global F8, SPEC, FAST
    ap = argparse.ArgumentParser()
    ap.add_argument("--f8", action="store_true", help="kivi2p8 pages (kivi2f8:32) instead of kivi2")
    ap.add_argument("--fast", action="store_true", help="the kivi2p8f / kivi2f fast decode-attention kernel")
    ap.add_argument("--calib", default=os.path.join(os.environ.get("FLAGROT_CALIB_DIR", str(HERE.parents[2] / "calib")), "flagrot_kivi_llama8b.pt"))
    ap.add_argument("--calib256", default="")
    a = ap.parse_args()
    if a.f8:
        F8, SPEC = True, "kivi2f8:32"
    FAST = a.fast
    _wrap_f8()
    torch.backends.cuda.matmul.allow_tf32 = False
    g = torch.Generator().manual_seed(7)
    cal = torch.load(a.calib)
    D, Hkv = cal["meta"]["head_dim"], cal["meta"]["num_kv_heads"]
    nl = cal["qs"].shape[0]
    ok = True
    for li in (0, nl // 2, nl - 1):
        ok &= test_keys(cal, li, D, Hkv, g)
        ok &= test_values(cal, li, D, Hkv, g)
    ok &= test_attn(cal, 0, D, Hkv, 4, g)
    for G in (6, 8):
        ok &= test_attn(cal, 0, D, Hkv, G, g, Ls=(7, 200, 1500))
    if a.calib256:
        c2 = torch.load(a.calib256)
        D2, H2 = c2["meta"]["head_dim"], c2["meta"]["num_kv_heads"]
        for li in (0, c2["qs"].shape[0] - 1):
            ok &= test_keys(c2, li, D2, H2, g, ng=6)
            ok &= test_values(c2, li, D2, H2, g)
        ok &= test_attn(c2, 0, D2, H2, c2["meta"]["num_heads"] // H2, g, Ls=(9, 140, 700, 1300))
    else:
        cal256 = {"qs": torch.randn(1, 4, 512, 256) * 0.2, "lz": torch.full((1, 4, 512), 4.0),
                  "qscale": torch.tensor(1 / 16), "co": torch.rand(1, 4, 256)}
        ok &= test_keys(cal256, 0, 256, 4, g, ng=6)
        ok &= test_values(cal256, 0, 256, 4, g)
        ok &= test_attn(cal256, 0, 256, 4, 6, g, Ls=(9, 140, 700, 1300))
    ok &= test_dequant(cal, D, Hkv, g)
    ok &= test_ident(cal, D, Hkv, g)
    ok &= test_fixed(cal, D, Hkv, g)
    ok &= test_vf_preserve(cal, D, Hkv, g)
    print("ALL OK" if ok else "FAILED")


if __name__ == "__main__":
    main()
