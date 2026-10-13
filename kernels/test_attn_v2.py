#!/usr/bin/env python3
"""Correctness of the v2 paged 4-bit decode attention (attn_decode_v2.py / csrc/attn_decode_v2.cu) and of the paged
KV writer.

1. Writer: kv [T, 2*Hkv, 128] bf16 -> pages via slot_mapping (random page order, variable lengths, page 16 / 32) must
   hold exactly the values kv_selrot_quant produces (same FlagRot, same codes and scales): the paged pool is parsed
   by PagedKVCache.dequant (independent Python layout parser) and compared bit-exactly with the dequantized
   contiguous kv_selrot_quant cache (attn_decode.KVCache).
2. Attention: reference = float64 attention on the dequantized pages, original basis (K_ref = R_k Kdeq,
   V_ref = R_v Vdeq), per sequence with its own length; vrot=False compared with R_v ref (V rotation folded into W_o).
   Yardstick and tolerance as test_attn.py: torch SDPA in BF16 on the same dequantized K/V vs the same reference;
   pass iff max|err| / max|ref| <= max(2 x SDPA, 4e-3) for every case.
   Cases: formats x kernel variants (0 FP8-PV for MXFP4/INT4, 1 f16-PV) x stages {2, 4} x page {16, 32} x
   q rotation fused / none (unrotated cache) x V back-rotation on / off x splits auto / 1 / 3 (stream-K variants,
   CTA counts auto / minimum / 7 / 97) x batches with variable lengths (incl. 1, partial pages, partial
   64-token blocks, 33000).
usage: CUDA_VISIBLE_DEVICES=6 python test_attn_v2.py [--quick]  -> results/test_attn_v2_log.txt, .json
       (TEST_V2_FMTS=int4,int4a restricts the formats, TEST_V2_TAG=_x suffixes the result files)
"""
from __future__ import annotations

import argparse
import json
import sys
import os

import torch
import torch.nn.functional as Fn

import attn_decode as A
import attn_decode_v2 as A2
import ref as R
from test_attn import errs

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:  # blockflag (int4a reference)
    sys.path.insert(0, os.path.dirname(HERE))
DEV = "cuda"
HKV, D = 8, 128
RBF = {"nvfp4": 16, "mxfp4": 32, "int4": 32, "int4a": 32}


def make_kv(lens, seed, outliers=True):
    """per-sequence kv [L_b, 2Hkv, D] bf16 (LLM-like keys: outlier channels, per-token magnitude spread), q."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    ch = torch.randint(0, D, (HKV, 4), device=DEV, generator=g)
    kvs = []
    for L in lens:
        k = torch.randn(L, HKV, D, device=DEV, generator=g)
        v = torch.randn(L, HKV, D, device=DEV, generator=g)
        if outliers:
            for h in range(HKV):
                k[:, h, ch[h]] *= 6.0
            k = k * (0.5 + torch.rand(L, 1, 1, device=DEV, generator=g))
            v = v * (0.25 + 1.5 * torch.rand(L, 1, 1, device=DEV, generator=g))  # per-token V scale spread
        kvs.append(torch.cat([k, v], 1).bfloat16())
    q = (torch.randn(len(lens), 4 * HKV, D, device=DEV, generator=g) * 0.6).bfloat16()
    return q, kvs


def build(fmt, page, lens, kvs, mask, rb, gs, seed, extra_pages=7):
    lens_t = torch.tensor(lens, dtype=torch.int32)
    need = sum((L + page - 1) // page for L in lens)
    bt, npages = A2.alloc_block_table(lens_t, page, num_pages=need + extra_pages,
                                      gen=torch.Generator().manual_seed(seed))
    c = A2.PagedKVCache(fmt, npages, page, HKV)
    c.pool.random_(0, 256)  # garbage in unused slots / pages (the kernel must not read it)
    c.gs = gs
    for b, L in enumerate(lens):
        c.write(kvs[b], A2.slots_for(bt, b, range(L), page), mask, rb)
    return c, bt, lens_t.to(DEV)


def edge_groups(kvs, seed):
    """int4a writer edge cases: overwrite some 32-groups with all-positive / all-negative (zero point outside [0, 15]
    in blockflag int4a:32), all-zero, tiny (FP16-subnormal scales) and large groups."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    out = []
    for kv in kvs:
        x = kv.float().clone()
        L = x.shape[0]
        flat = x.view(L, -1, 32)                                   # [L, 2Hkv * 4, 32]
        sel = torch.randint(0, 6, flat.shape[:2], device=DEV, generator=g)
        r = torch.rand(flat.shape, device=DEV, generator=g)
        flat[sel == 1] = (r[sel == 1] + 2.0)                       # all positive, away from 0
        flat[sel == 2] = -(r[sel == 2] * 0.5 + 0.25)               # all negative
        flat[sel == 3] = 0.0
        flat[sel == 4] = (r[sel == 4] - 0.5) * 1e-5                # scale below the FP16 normal range
        flat[sel == 5] = (r[sel == 5] - 0.3) * 300.0
        out.append(x.bfloat16())
    return out


def check_writer_int4a(page, lens, kvs, mask, rb, cache, bt, lines):
    """int4a: parsed codes / zero points / scale words == the PyTorch reference (FlagRot + ref.int4a_quant) bit for
    bit; also counts the groups whose codes differ from blockflag int4a:32 (range not containing 0)."""
    import blockflag.formats as FM
    codes, words = cache.parse_int4a()
    bad = n = outside = 0
    for b, L in enumerate(lens):
        y = R.sel_fwht_rb(kvs[b].float().reshape(L, -1), mask, rb).reshape(L, 2 * HKV, D)
        qr, wr = R.int4a_quant(y)
        slots = A2.slots_for(bt, b, range(L), page)
        pg, off = slots // page, slots % page
        cq = codes[pg, off].permute(0, 2, 1, 3).reshape(L, 2 * HKV, D)    # [L, (K heads, V heads), D]
        cw = words[pg, off].permute(0, 2, 1, 3).reshape(L, 2 * HKV, 4)
        bad += int(((cq != qr).reshape(L, 2 * HKV, 4, 32).any(-1) | (cw != wr)).sum())
        n += L * 2 * HKV * 4
        qa = FM._grp(y, 32)                                       # blockflag int4a:32 codes (unbounded zero point)
        mn, mx = qa.amin(-1, keepdim=True), qa.amax(-1, keepdim=True)
        sa = ((mx - mn) / 15).clamp_min(1e-12)
        za = torch.round(-mn / sa)
        ca = (torch.round(qa / sa) + za).clamp(0, 15)
        outside += int((ca != qr.reshape(ca.shape).float()).any(-1).sum())
    lines.append(f"writer int4a  page {page} lens {lens}: groups differing from the reference (codes, zero point, "
                 f"scale word) {bad} of {n}; groups whose codes differ from blockflag int4a:32 {outside}")
    print(lines[-1], flush=True)
    return bad == 0


def check_writer(fmt, page, lens, kvs, mask, rb, gs, cache, bt, lines):
    if fmt == "int4a":
        ok = check_writer_int4a(page, lens, kvs, mask, rb, cache, bt, lines)
        K2, V2 = cache.dequant(bt, torch.tensor(lens, device=DEV))
        return ok, K2, V2
    K2, V2 = cache.dequant(bt, torch.tensor(lens, device=DEV))
    worst = 0.0
    for b, L in enumerate(lens):
        c1 = A.KVCache(fmt, 1, L, HKV)
        if fmt == "nvfp4":
            c1.gs = gs
        c1.fill(kvs[b].unsqueeze(0), mask, rb)
        d1 = c1.dequant(L)[0]
        worst = max(worst, (K2[b, :L] - d1[:, :HKV]).abs().max().item(), (V2[b, :L] - d1[:, HKV:]).abs().max().item())
    lines.append(f"writer {fmt:6s} page {page} lens {lens}: max |paged - kv_selrot_quant| = {worst:.1e}")
    print(lines[-1], flush=True)
    return worst == 0.0, K2, V2


def run(quick):
    torch.manual_seed(0)
    rows, lines = [], []
    batches = [[1], [77, 5], [300, 1000, 64, 17], [4101, 33]] if quick else \
        [[1], [77, 5], [300, 1000, 64, 17], [4101, 33], [33000], [2048, 1, 129, 640, 63, 4096, 191, 1]]
    writer_ok = True
    for fmt in A2.FMTS:
        rb = RBF[fmt]
        variants = [1, 3] if fmt == "nvfp4" else [264] if fmt == "int4a" else [0, 1, 2, 3]
        if os.environ.get("TEST_V2_VARIANTS") and fmt == "int4":  # extra INT4 kernel variants (e.g. "0,4,8,12")
            variants = [int(x) for x in os.environ["TEST_V2_VARIANTS"].split(",")]
        if os.environ.get("TEST_V2_FMTS") and fmt not in os.environ["TEST_V2_FMTS"].split(","):
            continue
        for page in (16, 32):
            for bi, lens in enumerate(batches):
                seed = 100 * bi + page + len(fmt)
                q, kvs = make_kv(lens, seed)
                if fmt == "int4a" and bi % 2 == 1:   # writer edge cases (groups not containing 0, zero, tiny, large)
                    kvs = edge_groups(kvs, seed)
                g = torch.Generator(device="cpu").manual_seed(seed)
                mask = (torch.rand(2 * HKV, D // rb, generator=g) < 0.6).to(torch.uint8).to(DEV)
                gs = R.nvfp4_global_scale(torch.cat([A.rotate_heads(x.float(), mask, rb) for x in kvs])).to(DEV)
                cache, bt, lens_d = build(fmt, page, lens, kvs, mask, rb, gs, seed)
                ok, K2, V2 = check_writer(fmt, page, lens, kvs, mask, rb, gs, cache, bt, lines)
                writer_ok &= ok
                # references per sequence (rotated-back basis) + SDPA yardstick
                refs, refs_rot, sd = [], [], []
                for b, L in enumerate(lens):
                    kd = A.rotate_heads(K2[b:b + 1, :L], mask[:HKV], rb)
                    vd = A.rotate_heads(V2[b:b + 1, :L], mask[HKV:], rb)
                    r = A.ref_attention(q[b:b + 1].float(), kd, vd)
                    refs.append(r)
                    refs_rot.append(A.rotate_heads(r.float(), mask[HKV:].repeat_interleave(4, 0), rb).double())
                    s = Fn.scaled_dot_product_attention(q[b:b + 1].unsqueeze(2), kd.bfloat16().transpose(1, 2),
                                                        vd.bfloat16().transpose(1, 2), enable_gqa=True).squeeze(2)
                    sd.append(errs(s.float(), r)["max_rel"])
                km, vm = mask[:HKV].contiguous(), mask[HKV:].contiguous()
                for v in variants:
                    for st in ((3, 4) if not quick else (4,)):
                        for vrot in (False, True):
                            sk = bool(v & A2.SK_BIT)  # stream-K variants: CTA counts instead of split counts
                            tmax = HKV * len(lens) * -(-max(lens) // 64)
                            gmin = -(-tmax // A2.SK_MAXBLK)
                            for ns in ((None, max(1, gmin), max(7, gmin), max(97, gmin)) if sk else (None, 1, 3)):
                                if not sk and ns is not None and ns > -(-max(lens) // 64):
                                    continue
                                o = A2.decode(q, cache, bt, lens_d, kmask=km, vmask=vm, rb=rb, qrot="fused",
                                              vrot=vrot, variant=v, stages=st, nsplit=None if sk else ns,
                                              sk_grid=ns if sk else None, lmax=max(lens))
                                o = o.view(len(lens), 4 * HKV, D).float()
                                for b, L in enumerate(lens):
                                    r = refs[b] if vrot else refs_rot[b]
                                    e = errs(o[b:b + 1], r)
                                    tol = max(2 * sd[b], 4e-3)
                                    rows.append(dict(fmt=fmt, page=page, lens=lens, b=b, L=L, variant=v, stages=st,
                                                     vrot=vrot, nsplit=ns, **e, sdpa_bf16=sd[b], tol=tol,
                                                     passed=bool(e["max_rel"] <= tol)))
                worst = max((r for r in rows if r["fmt"] == fmt and r["page"] == page and r["lens"] == lens),
                            key=lambda r: r["max_rel"] / r["tol"])
                lines.append(f"attn   {fmt:6s} page {page} lens {str(lens)[:40]:40s}: worst max|err|/max|ref| "
                             f"{worst['max_rel']:.2e} (v{worst['variant']}, tol {worst['tol']:.2e}, SDPA {worst['sdpa_bf16']:.2e})")
                print(lines[-1], flush=True)
        # unrotated cache (zero mask), q rotation off
        lens = [300, 4101]
        q, kvs = make_kv(lens, 7)
        z = torch.zeros(2 * HKV, D // rb, dtype=torch.uint8, device=DEV)
        gs = R.nvfp4_global_scale(torch.cat([x.float() for x in kvs])).to(DEV)
        cache, bt, lens_d = build(fmt, 16, lens, kvs, z, rb, gs, 7)
        K2, V2 = cache.dequant(bt, lens_d)
        for v in variants:
            o = A2.decode(q, cache, bt, lens_d, rb=rb, qrot=None, variant=v, lmax=max(lens)).view(len(lens), 4 * HKV, D).float()
            for b, L in enumerate(lens):
                r = A.ref_attention(q[b:b + 1].float(), K2[b:b + 1, :L], V2[b:b + 1, :L])
                s = Fn.scaled_dot_product_attention(q[b:b + 1].unsqueeze(2), K2[b:b + 1, :L].bfloat16().transpose(1, 2),
                                                    V2[b:b + 1, :L].bfloat16().transpose(1, 2), enable_gqa=True).squeeze(2)
                sdv = errs(s.float(), r)["max_rel"]
                e = errs(o[b:b + 1], r)
                rows.append(dict(fmt=fmt, page=16, lens=lens, b=b, L=L, variant=v, stages=None, vrot=None,
                                 nsplit=None, qrot=None, **e, sdpa_bf16=sdv, tol=max(2 * sdv, 4e-3),
                                 passed=bool(e["max_rel"] <= max(2 * sdv, 4e-3))))
    summ = []
    for fmt in A2.FMTS:
        for v in sorted({r["variant"] for r in rows if r["fmt"] == fmt}):
            ks = [r for r in rows if r["fmt"] == fmt and r["variant"] == v]
            if not ks:
                continue
            summ.append(dict(fmt=fmt, variant=v, cases=len(ks), max_abs=max(r["max_abs"] for r in ks),
                             max_rel=max(r["max_rel"] for r in ks), rel_l2=max(r["rel_l2"] for r in ks),
                             sdpa_bf16_max_rel=max(r["sdpa_bf16"] for r in ks), passed=all(r["passed"] for r in ks)))
    lines.append("")
    lines.append(f"writer bit-exact vs kv_selrot_quant: {writer_ok}")
    lines.append("| fmt | variant | cases | max abs err | max abs err / max abs ref | max rel-L2 | bf16 SDPA max abs err / max abs ref | pass |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in summ:
        lines.append(f"| {s['fmt']} | {s['variant']} | {s['cases']} | {s['max_abs']:.2e} | {s['max_rel']:.2e} | "
                     f"{s['rel_l2']:.2e} | {s['sdpa_bf16_max_rel']:.2e} | {s['passed']} |")
    txt = "\n".join(lines)
    print(txt)
    if not quick:
        open(os.path.join(HERE, "results", f"test_attn_v2{os.environ.get('TEST_V2_TAG', '')}_log.txt"), "w").write(txt + "\n")
        json.dump(dict(rows=rows, summary=summ, writer_bit_exact=writer_ok),
                  open(os.path.join(HERE, "results", f"test_attn_v2{os.environ.get('TEST_V2_TAG', '')}.json"), "w"), indent=1)


# ------------------------------------------------------------------ generalized kernel
# GQA groups G != 4 and head dim 256: csrc/attn_decode_v2g.cuh (variant 264 + GEN, q tiles of GT = 4 / 8 heads).
# Same references and tolerance rule as run(); configurations (G, D, GT, Hkv) cover the target models (Qwen3-14B G 5,
# Qwen3-32B G 8, Qwen3.8-27B G 6 / D 256 / Hkv 4, GLM-4.5-Air G 12) plus the generalized kernel at G 4 (= variant 264).
GEN_CFGS = [(4, 128, 4, 8), (5, 128, 8, 8), (5, 128, 4, 8), (6, 128, 8, 8), (8, 128, 8, 8), (8, 128, 4, 8),
            (12, 128, 4, 8), (12, 128, 8, 8), (6, 256, 4, 4), (4, 256, 4, 2), (1, 128, 4, 8)]


def make_kv_g(lens, seed, G, D, Hkv, outliers=True):
    """make_kv for any GQA group G, head dim D and Hkv: q [B, G*Hkv, D], kv per sequence [L_b, 2Hkv, D] bf16."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    ch = torch.randint(0, D, (Hkv, 4 * D // 128), device=DEV, generator=g)
    kvs = []
    for L in lens:
        k = torch.randn(L, Hkv, D, device=DEV, generator=g)
        v = torch.randn(L, Hkv, D, device=DEV, generator=g)
        if outliers:
            for h in range(Hkv):
                k[:, h, ch[h]] *= 6.0
            k = k * (0.5 + torch.rand(L, 1, 1, device=DEV, generator=g))
            v = v * (0.25 + 1.5 * torch.rand(L, 1, 1, device=DEV, generator=g))
        kvs.append(torch.cat([k, v], 1).bfloat16())
    q = (torch.randn(len(lens), G * Hkv, D, device=DEV, generator=g) * 0.6 * (128 / D) ** 0.25).bfloat16()
    return q, kvs


def ref_attention_x(q, K, V):
    """float64 GQA decode attention with an explicit softmax (test_codebook.ref_attention; any G, D)."""
    B, Hq, Dd = q.shape
    Hkv = K.shape[2]
    qd = q.double().reshape(B, Hkv, Hq // Hkv, Dd)
    s = torch.einsum("bhgd,blhd->bhgl", qd, K.double()) / Dd ** 0.5
    e = torch.exp(s - s.amax(-1, keepdim=True))
    p = e / e.sum(-1, keepdim=True)
    return torch.einsum("bhgl,blhd->bhgd", p, V.double()).reshape(B, Hq, Dd)


def build_g(fmt, page, lens, kvs, mask, rb, seed, Hkv, D, extra_pages=7):
    lens_t = torch.tensor(lens, dtype=torch.int32)
    need = sum((L + page - 1) // page for L in lens)
    bt, npages = A2.alloc_block_table(lens_t, page, num_pages=need + extra_pages, gen=torch.Generator().manual_seed(seed))
    c = A2.PagedKVCache(fmt, npages, page, Hkv, D=D)
    c.pool.random_(0, 256)
    for b, L in enumerate(lens):
        c.write(kvs[b], A2.slots_for(bt, b, range(L), page), mask, rb)
    return c, bt, lens_t.to(DEV)


def check_writer_g(fmt, page, lens, kvs, mask, rb, cache, bt, Hkv, D, lines):
    """D = 256: the pool must equal the 128-d writer's pool over 2 Hkv half heads (byte for byte, written pages);
    int4a additionally: codes / scale words == the PyTorch reference (FlagRot + ref.int4a_quant), any D."""
    ok = True
    if D == 256:
        h = A2.PagedKVCache(fmt, cache.num_pages, page, 2 * Hkv, D=128)
        h.pool.copy_(cache.pool.view(cache.num_pages, 2 * Hkv, -1))
        mh = mask.reshape(4 * Hkv, -1)                                   # [2 Hkv, 256 / rb] = [4 Hkv, 128 / rb]
        for b, L in enumerate(lens):
            h.write(kvs[b].reshape(L, 4 * Hkv, 128), A2.slots_for(bt, b, range(L), page), mh, rb)
        nd = int((h.pool.view(cache.num_pages, Hkv, -1) != cache.pool).sum())
        ok &= nd == 0
        lines.append(f"writer {fmt:6s} D 256 page {page} lens {lens}: bytes differing from the 128-d writer over half heads: {nd}")
        print(lines[-1], flush=True)
    if fmt == "int4a":
        codes, words = cache.parse_int4a()
        bad = n = 0
        for b, L in enumerate(lens):
            y = R.sel_fwht_rb(kvs[b].float().reshape(L, -1), mask, rb).reshape(L, 2 * Hkv, D)
            qr, wr = R.int4a_quant(y)
            slots = A2.slots_for(bt, b, range(L), page)
            pg, off = slots // page, slots % page
            cq = codes[pg, off].permute(0, 2, 1, 3).reshape(L, 2 * Hkv, D)
            cw = words[pg, off].permute(0, 2, 1, 3).reshape(L, 2 * Hkv, D // 32)
            bad += int(((cq != qr).reshape(L, 2 * Hkv, D // 32, 32).any(-1) | (cw != wr)).sum())
            n += L * 2 * Hkv * D // 32
        ok &= bad == 0
        lines.append(f"writer int4a  D {D} page {page} lens {lens}: groups differing from the reference {bad} of {n}")
        print(lines[-1], flush=True)
    return ok


def run_gen(quick, cfgs=None, fmts=("int4", "int4a")):
    torch.manual_seed(0)
    rows, lines = [], []
    cfgs = GEN_CFGS if cfgs is None else cfgs
    batches = [[1], [77, 5], [300, 1000, 64, 17], [4101, 33]] if quick else \
        [[1], [77, 5], [300, 1000, 64, 17], [4101, 33], [33000], [2048, 1, 129, 640, 63, 4096, 191, 1], [513, 777, 1023]]
    writer_ok = True
    rb = 32
    for (G, D, GT, Hkv) in cfgs:
        v = A2.gen_variant(G, D, GT)
        for fmt in fmts:
            for page in ((16,) if quick else (16, 32)):
                for bi, lens in enumerate(batches):
                    seed = 100 * bi + page + 7 * G + D + len(fmt)
                    q, kvs = make_kv_g(lens, seed, G, D, Hkv)
                    if fmt == "int4a" and bi % 2 == 1:
                        kvs = edge_groups(kvs, seed)
                    gc = torch.Generator(device="cpu").manual_seed(seed)
                    mask = (torch.rand(2 * Hkv, D // rb, generator=gc) < 0.6).to(torch.uint8).to(DEV)
                    cache, bt, lens_d = build_g(fmt, page, lens, kvs, mask, rb, seed, Hkv, D)
                    if bi < 4:
                        writer_ok &= check_writer_g(fmt, page, lens, kvs, mask, rb, cache, bt, Hkv, D, lines)
                    K2, V2 = cache.dequant(bt, lens_d)
                    refs, refs_rot, sd = [], [], []
                    for b, L in enumerate(lens):
                        kd = A.rotate_heads(K2[b:b + 1, :L], mask[:Hkv], rb)
                        vd = A.rotate_heads(V2[b:b + 1, :L], mask[Hkv:], rb)
                        r = ref_attention_x(q[b:b + 1].float(), kd, vd)
                        refs.append(r)
                        refs_rot.append(A.rotate_heads(r.float(), mask[Hkv:].repeat_interleave(G, 0), rb).double())
                        s = Fn.scaled_dot_product_attention(q[b:b + 1].unsqueeze(2), kd.bfloat16().transpose(1, 2),
                                                            vd.bfloat16().transpose(1, 2), enable_gqa=True).squeeze(2)
                        sd.append(errs(s.float(), r)["max_rel"])
                    km, vm = mask[:Hkv].contiguous(), mask[Hkv:].contiguous()
                    for st in ((4,) if quick else (3, 4)):
                        for vrot in (False, True):
                            for ns in (None, 1, 3):
                                if ns is not None and ns > -(-max(lens) // 64):
                                    continue
                                o = A2.decode(q, cache, bt, lens_d, kmask=km, vmask=vm, rb=rb, qrot="fused", vrot=vrot,
                                              variant=v, stages=st, nsplit=ns, lmax=max(lens))
                                o = o.view(len(lens), G * Hkv, D).float()
                                for b, L in enumerate(lens):
                                    r = refs[b] if vrot else refs_rot[b]
                                    e = errs(o[b:b + 1], r)
                                    tol = max(2 * sd[b], 4e-3)
                                    rows.append(dict(fmt=fmt, G=G, D=D, GT=GT, Hkv=Hkv, page=page, lens=lens, b=b, L=L,
                                                     variant=v, stages=st, vrot=vrot, nsplit=ns, **e, sdpa_bf16=sd[b],
                                                     tol=tol, passed=bool(e["max_rel"] <= tol)))
                    cur = [r for r in rows if r["fmt"] == fmt and r["G"] == G and r["D"] == D and r["GT"] == GT
                           and r["page"] == page and r["lens"] == lens]
                    worst = max(cur, key=lambda r: r["max_rel"] / r["tol"])
                    lines.append(f"attn   {fmt:6s} G {G:2d} D {D} GT {GT} page {page} lens {str(lens)[:40]:40s}: worst "
                                 f"max|err|/max|ref| {worst['max_rel']:.2e} (tol {worst['tol']:.2e}) "
                                 f"{'ok' if all(r['passed'] for r in cur) else 'FAIL'}")
                    print(lines[-1], flush=True)
            # BF16 staging window (k_tail_g): bodies 0 .. 4101 tokens, windows 0 .. 256
            for page in (16,):
                for ci, (lens, tls) in enumerate([([0, 77, 1000, 4101], [5, 256, 0, 129])] if quick else
                                                 [([0, 77, 1000, 4101], [5, 256, 0, 129]), ([1, 33000], [255, 1])]):
                    seed = 900 + 10 * ci + G + D
                    q, kvs = make_kv_g(lens, seed, G, D, Hkv)
                    gc = torch.Generator(device="cpu").manual_seed(seed)
                    mask = (torch.rand(2 * Hkv, D // rb, generator=gc) < 0.6).to(torch.uint8).to(DEV)
                    cache, bt, lens_d = build_g(fmt, page, lens, kvs, mask, rb, seed, Hkv, D)
                    K2, V2 = cache.dequant(bt, lens_d)
                    K2 = A.rotate_heads(K2, mask[:Hkv], rb)
                    V2 = A.rotate_heads(V2, mask[Hkv:], rb)
                    gt_ = torch.Generator(device=DEV).manual_seed(seed)
                    tail = (torch.randn(len(lens), 256, 2 * Hkv, D, device=DEV, generator=gt_) * 1.5).bfloat16()
                    tl = torch.tensor(tls, dtype=torch.int32, device=DEV)
                    km, vm = mask[:Hkv].contiguous(), mask[Hkv:].contiguous()
                    for vrot in (False, True):
                        for ns in (None, 3):
                            o = A2.decode(q, cache, bt, lens_d, kmask=km, vmask=vm, rb=rb, vrot=vrot, nsplit=ns, variant=v,
                                          lmax=max(max(lens), 1), tail=tail, tail_len=tl).view(len(lens), G * Hkv, D).float()
                            if not vrot:
                                o = A.rotate_heads(o, vm.repeat_interleave(G, 0), rb).float()
                            for b, (L, T) in enumerate(zip(lens, tls)):
                                kd = torch.cat([K2[b:b + 1, :L].double(), tail[b:b + 1, :T, :Hkv].double()], 1)
                                vd = torch.cat([V2[b:b + 1, :L].double(), tail[b:b + 1, :T, Hkv:].double()], 1)
                                r = ref_attention_x(q[b:b + 1].float(), kd, vd)
                                s = Fn.scaled_dot_product_attention(q[b:b + 1].unsqueeze(2), kd.bfloat16().transpose(1, 2),
                                                                    vd.bfloat16().transpose(1, 2), enable_gqa=True).squeeze(2)
                                sdv = errs(s.float(), r)["max_rel"]
                                e = errs(o[b:b + 1], r)
                                tol = max(2 * sdv, 4e-3)
                                rows.append(dict(fmt=fmt, G=G, D=D, GT=GT, Hkv=Hkv, page=page, lens=lens, b=b, L=L, tail=T,
                                                 variant=v, stages=4, vrot=vrot, nsplit=ns, **e, sdpa_bf16=sdv, tol=tol,
                                                 passed=bool(e["max_rel"] <= tol)))
                    cur = [r for r in rows if r.get("tail") is not None and r["fmt"] == fmt and r["G"] == G and
                           r["D"] == D and r["GT"] == GT and r["lens"] == lens]
                    worst = max(cur, key=lambda r: r["max_rel"] / r["tol"])
                    lines.append(f"tail   {fmt:6s} G {G:2d} D {D} GT {GT} lens {lens} tails {tls}: worst "
                                 f"{worst['max_rel']:.2e} (tol {worst['tol']:.2e}) {'ok' if all(r['passed'] for r in cur) else 'FAIL'}")
                    print(lines[-1], flush=True)
    summ = []
    for (G, D, GT, Hkv) in cfgs:
        for fmt in fmts:
            ks = [r for r in rows if r["fmt"] == fmt and r["G"] == G and r["D"] == D and r["GT"] == GT]
            if not ks:
                continue
            summ.append(dict(fmt=fmt, G=G, D=D, GT=GT, Hkv=Hkv, cases=len(ks), max_abs=max(r["max_abs"] for r in ks),
                             max_rel=max(r["max_rel"] for r in ks), rel_l2=max(r["rel_l2"] for r in ks),
                             sdpa_bf16_max_rel=max(r["sdpa_bf16"] for r in ks),
                             worst_err_over_tol=max(r["max_rel"] / r["tol"] for r in ks),
                             passed=all(r["passed"] for r in ks)))
    lines.append("")
    lines.append(f"writer checks (D 256 == 128-d writer over half heads; int4a == reference): {writer_ok}")
    lines.append("| fmt | G | D | q tile | Hkv | cases | max abs err | max err / max ref | max rel-L2 | bf16 SDPA max err / max ref | worst err/tol | pass |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summ:
        lines.append(f"| {s['fmt']} | {s['G']} | {s['D']} | {s['GT']} | {s['Hkv']} | {s['cases']} | {s['max_abs']:.2e} | "
                     f"{s['max_rel']:.2e} | {s['rel_l2']:.2e} | {s['sdpa_bf16_max_rel']:.2e} | {s['worst_err_over_tol']:.2f} | {s['passed']} |")
    txt = "\n".join(lines)
    print(txt)
    if not quick:
        tag = os.environ.get("TEST_V2_TAG", "")
        open(os.path.join(HERE, "results", f"test_attn_v2_gen{tag}_log.txt"), "w").write(txt + "\n")
        json.dump(dict(rows=rows, summary=summ, writer_ok=writer_ok),
                  open(os.path.join(HERE, "results", f"test_attn_v2_gen{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--gen", action="store_true", help="generalized kernel (GQA != 4, head dim 256)")
    ap.add_argument("--cfgs", default=None, help="G:D:GT:Hkv,... (default GEN_CFGS)")
    ap.add_argument("--fmts", default="int4,int4a")
    a = ap.parse_args()
    if a.gen:
        cfgs = None if a.cfgs is None else [tuple(int(x) for x in c.split(":")) for c in a.cfgs.split(",")]
        run_gen(a.quick, cfgs, tuple(a.fmts.split(",")))
    else:
        run(a.quick)
