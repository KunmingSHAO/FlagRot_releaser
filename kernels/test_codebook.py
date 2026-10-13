#!/usr/bin/env python3
"""Correctness of the per-token transform codebook INT4 KV cache (attn_codebook.py, csrc/attn_codebook.cu).

0. Reference: codebook_ref.encode (fixed fp32 order) == blockflag.qattention.dyn_quant (rule 'cq', the full and the
   reduced codebook) on the reconstruction, with blockflag's int4:32 quantizer (fp32 scale).
1. Writer: kv [T, 2*Hkv, 128] bf16 -> pages via slot_mapping (random slots, page 16 / 32, tokens per CTA 1 / 3 / 8,
   T up to 4097, garbage in the pool): codes (stored coordinates), fp16 scales, per-vector choices and head mode parsed
   from the page (independent Python layout parser) must equal codebook_ref.encode with the cache quantizer
   (fp16 scales) bit for bit.
2. Attention: reference = float64 attention over the reconstructions of the cached rows (inverse transforms of the
   parsed codes / scales / choices in float64), per sequence with its own length.  Yardstick and tolerance as
   test_attn_v2.py: torch SDPA in BF16 on the same reconstructions vs the same reference; pass iff
   max|err| / max|ref| <= max(2 x SDPA, 4e-3) for every case.  Cases: kind (full / reduced) x page {16, 32} x stages
   {3, 4} x kernel variant (0 pipelined, 1 synchronous) x splits auto / 1 / 3 x the batches of test_attn_v2 (variable lengths 1 .. 33000, partial pages / blocks)
   + one batch of odd lengths in 513 .. 1023, shuffled block tables.  The float64 softmax is explicit (torch 2.6 +
   cu124 torch.softmax on CUDA float64 is wrong for odd lengths 513 .. 1023; attn_decode.ref_attention uses it).
usage: CUDA_VISIBLE_DEVICES=7 python test_codebook.py [--quick]  -> results/test_codebook_log.txt, .json
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as Fn

import attn_codebook as AC
import attn_decode_v2 as A2
import codebook_ref as CR
from test_attn import errs
from test_attn_v2 import edge_groups, make_kv

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"
HKV, D = 8, 128
TAIL_KINDS = AC.KINDS  # kinds whose kernels take the BF16 staging window (all: extra split CTA)


def ref_attention(q, K, V):
    """float64 GQA decode attention (= attn_decode.ref_attention with an explicit softmax: torch 2.6 + cu124
    torch.softmax on CUDA float64 returns wrong values for odd lengths 513 .. 1023, e.g. non-uniform p for zero scores)."""
    B, Hq, Dd = q.shape
    Hkv = K.shape[2]
    qd = q.double().reshape(B, Hkv, Hq // Hkv, Dd)
    s = torch.einsum("bhgd,blhd->bhgl", qd, K.double()) / Dd ** 0.5
    e = torch.exp(s - s.amax(-1, keepdim=True))
    p = e / e.sum(-1, keepdim=True)
    return torch.einsum("bhgl,blhd->bhgd", p, V.double()).reshape(B, Hq, Dd)


def check_reference(lines, kinds=AC.KINDS):
    """codebook_ref (blockflag quantizer) vs blockflag dyn_quant: identical reconstructions."""
    import blockflag.qattention as QA
    torch.manual_seed(0)
    T = 3000
    Cq, co = AC.random_calib(HKV, seed=11)
    q, kvs = make_kv([T], 5)
    kv = kvs[0]
    ok = True
    for cfg in kinds:
        for kind, x in (("k", kv[:, :HKV]), ("v", kv[:, HKV:])):
            c = CR.NOXF if (cfg in CR.KONLY and kind == "v") else CR.CFG[cfg]   # konly values: plain quantization
            QA.KV.update(dyn=dict(rule="cq", local=list(c["local"]), head=list(c["head"])), cq={0: Cq}, co={0: co},
                         dynstat={})
            asym = cfg == "konlya"   # asymmetric: the choice by the int4a:32 error (dyn_quant with spec int4a:32)
            r = CR.encode(x, Cq, co, kind, cfg, qk="asym" if asym else "blockflag")
            ref = QA.dyn_quant(x.transpose(0, 1)[None], "int4a:32" if asym else "int4:32", 32, kind, 0).squeeze(0).transpose(0, 1)
            nd = int((r["rec"] != ref).any(-1).sum())
            ok &= nd == 0
            modes = torch.bincount(r["mode"].flatten(), minlength=8).tolist()
            loc = torch.bincount(r["local"][r["mode"] == 0].flatten(), minlength=8).tolist()
            lines.append(f"reference {cfg:7s} {kind}: rows whose reconstruction differs from blockflag dyn_quant: {nd} "
                         f"of {T * HKV}  (head modes {modes}, vector choices {loc}; index order of codebook_ref.MODES / local list)")
            print(lines[-1], flush=True)
    return ok


def check_writer(lines, quick, kinds=AC.KINDS):
    Cq, co = AC.random_calib(HKV, seed=3)
    ok, n = True, 0
    for kind in kinds:
        for page in (16, 32):
            for T in ((1, 5, 64, 1000) if quick else (1, 5, 64, 1000, 4097)):
                q, kvs = make_kv([T], 1000 + T + page)
                kv = kvs[0]
                if kind == "konlya" and T >= 1000:  # asymmetric writer edge cases: groups not containing 0, zero, tiny, large
                    kv = edge_groups([kv], T + page)[0]
                npages = (T + page - 1) // page + 5
                c = AC.CodebookKVCache(kind, npages, page, HKV)
                c.pool.random_(0, 256)
                c.set_calib(Cq, co)
                slots = torch.randperm(npages * page, device=DEV, generator=torch.Generator(device=DEV).manual_seed(T))[:T]
                qk = AC.QUANT.get(kind, "cache")
                refs = [CR.encode(kv[:, :HKV], Cq, co, "k", kind, qk), CR.encode(kv[:, HKV:], Cq, co, "v", kind, qk)]
                for tg in [t for t in ((1, 3, 8) if T > 1 else (1,))
                           if AC.ext().write_smem(AC.KIND_ID[kind], t) <= 200 * 1024]:  # (head16: 17 head candidates)
                    c.write(kv, slots, tg=tg)
                    r = c.parse_pages()
                    pg, off = slots // page, slots % page
                    bad = 0
                    for kk in range(2):
                        ref = refs[kk]
                        b = ((r["codes"][pg, off, :, kk] != ref["q"].long()).any(-1)
                             | (r["scales"][pg, off, :, kk] != ref["s"]).any(-1)
                             | (r["local"][pg, off, :, kk] != ref["local"]).any(-1)
                             | (r["mode"][pg, off, :, kk] != ref["mode"]))
                        bad += int(b.sum())
                    ok &= bad == 0
                    n += 1
                    lines.append(f"writer {kind:7s} page {page} T {T:5d} tokens/CTA {tg}: rows differing from "
                                 f"codebook_ref (codes, scales, choices) {bad} of {2 * T * HKV}")
                    print(lines[-1], flush=True)
    return ok, n


def build(kind, page, lens, kvs, Cq, co, seed, extra_pages=7):
    lens_t = torch.tensor(lens, dtype=torch.int32)
    need = sum((L + page - 1) // page for L in lens)
    bt, npages = A2.alloc_block_table(lens_t, page, num_pages=need + extra_pages, gen=torch.Generator().manual_seed(seed))
    c = AC.CodebookKVCache(kind, npages, page, HKV)
    c.pool.random_(0, 256)
    c.set_calib(Cq, co)
    for b, L in enumerate(lens):
        c.write(kvs[b], A2.slots_for(bt, b, range(L), page))
    return c, bt, lens_t.to(DEV)


def tail_cases(quick):
    """(sequence lengths of the paged body, BF16 staging-window lengths) per case; window buffer of 256 tokens"""
    return ([([0, 77, 1000, 4101], [5, 256, 0, 129]), ([1, 33000], [255, 1]), ([300, 64], [32, 64])] if not quick
            else [([0, 77, 1000], [5, 256, 0])])


def tail_check_rows(rows, lines, kind, page, lens, tls, outs, K2, V2, q, tail):
    """outs: {variant label: output [B, 32, 128]}; reference = float64 attention over body (K2 / V2, original basis)
    + the tail tokens"""
    for v, o in outs.items():
        for b, (L, T) in enumerate(zip(lens, tls)):
            kb = K2[b:b + 1, :L] if K2 is not None else torch.zeros(1, 0, HKV, D, dtype=torch.float64, device=DEV)
            vb = V2[b:b + 1, :L] if V2 is not None else torch.zeros(1, 0, HKV, D, dtype=torch.float64, device=DEV)
            kd = torch.cat([kb, tail[b:b + 1, :T, :HKV].double()], 1)
            vd = torch.cat([vb, tail[b:b + 1, :T, HKV:].double()], 1)
            r = ref_attention(q[b:b + 1].float(), kd, vd)
            s = Fn.scaled_dot_product_attention(q[b:b + 1].unsqueeze(2), kd.bfloat16().transpose(1, 2),
                                                vd.bfloat16().transpose(1, 2), enable_gqa=True).squeeze(2)
            sd = errs(s.float(), r)["max_rel"]
            e = errs(o[b:b + 1], r)
            tol = max(2 * sd, 4e-3)
            rows.append(dict(kind=kind, variant=v, page=page, lens=lens, b=b, L=L, tail=T, stages=4,
                             nsplit=None, **e, sdpa_bf16=sd, tol=tol, passed=bool(e["max_rel"] <= tol)))
    cur = [r for r in rows if r.get("tail") is not None and r["kind"] == kind and r["page"] == page and r["lens"] == lens]
    worst = max(cur, key=lambda r: r["max_rel"] / r["tol"])
    lines.append(f"tail   {kind:7s} page {page} lens {lens} tails {tls}: worst max|err|/max|ref| "
                 f"{worst['max_rel']:.2e} (tol {worst['tol']:.2e}) {'ok' if all(r['passed'] for r in cur) else 'FAIL'}")
    print(lines[-1], flush=True)


def check_tail(lines, rows, quick, kinds=AC.KINDS):
    """BF16 staging window (all codebook kernels; one extra split CTA, csrc/attn_tail.cuh): attention over the paged
    body (reconstructions) + the BF16 tail tokens; bodies of length 0 (everything in the tail), windows 0 .. 256."""
    Cq, co = AC.random_calib(HKV, seed=9)
    for kind in [k for k in kinds if k in TAIL_KINDS]:
        for page in (16, 32):
            for ci, (lens, tls) in enumerate(tail_cases(quick)):
                seed = 500 + 10 * ci + page
                q, kvs = make_kv(lens, seed)
                cache, bt, lens_d = build(kind, page, lens, kvs, Cq, co, seed)
                g = torch.Generator(device=DEV).manual_seed(seed)
                tail = (torch.randn(len(lens), 256, 2 * HKV, D, device=DEV, generator=g) * 1.5).bfloat16()
                tl = torch.tensor(tls, dtype=torch.int32, device=DEV)
                K2, V2 = cache.reconstruct(bt, lens_d) if max(lens) > 0 else (None, None)
                outs = {}
                for v in AC.VARIANTS[kind]:
                    for ns in (None, 3):
                        if ns is not None and ns > -(-max(max(lens), 1) // 64):
                            continue
                        o = AC.decode(q, cache, bt, lens_d, lmax=max(lens), variant=v, nsplit=ns, tail=tail, tail_len=tl)
                        outs[f"{v}/ns{ns}"] = o.view(len(lens), 4 * HKV, D).float()
                tail_check_rows(rows, lines, kind, page, lens, tls, outs, K2, V2, q, tail)


def check_tail_v2(lines, rows, quick):
    """BF16 staging window on the static FlagRot INT4 v2 kernel (attn_decode_v2.py): the tail CTA rotates its partial
    into the V basis of the cache; q rotation fused, V back-rotation on / off."""
    import attn_decode as A
    import test_attn_v2 as T2
    rb = 64
    for page in (16, 32):
        for ci, (lens, tls) in enumerate(tail_cases(quick)):
            seed = 700 + 10 * ci + page
            q, kvs = make_kv(lens, seed)
            g = torch.Generator(device="cpu").manual_seed(seed)
            mask = (torch.rand(2 * HKV, D // rb, generator=g) < 0.6).to(torch.uint8).to(DEV)
            cache, bt, lens_d = T2.build("int4", page, lens, kvs, mask, rb, None, seed)
            K2, V2 = cache.dequant(bt, lens_d)
            K2 = A.rotate_heads(K2, mask[:HKV], rb)                           # original basis
            V2 = A.rotate_heads(V2, mask[HKV:], rb)
            gt = torch.Generator(device=DEV).manual_seed(seed)
            tail = (torch.randn(len(lens), 256, 2 * HKV, D, device=DEV, generator=gt) * 1.5).bfloat16()
            tl = torch.tensor(tls, dtype=torch.int32, device=DEV)
            km, vm = mask[:HKV].contiguous(), mask[HKV:].contiguous()
            outs = {}
            for vrot in (False, True):
                # split-K auto / 3 splits; stream-K schedule (variant 776) auto / 7 CTAs
                for ns, var, grid in ((None, 264, None), (3, 264, None), (None, 776, None), (None, 776, 7)):
                    if ns is not None and ns > -(-max(max(lens), 1) // 64):
                        continue
                    if grid is not None and -(-HKV * len(lens) * -(-max(max(lens), 1) // 64) // grid) > A2.SK_MAXBLK:
                        continue
                    o = A2.decode(q, cache, bt, lens_d, kmask=km, vmask=vm, rb=rb, vrot=vrot, nsplit=ns, variant=var,
                                  sk_grid=grid, lmax=max(max(lens), 1), tail=tail, tail_len=tl).view(len(lens), 4 * HKV, D).float()
                    if not vrot:  # output in the V-rotated basis (W_o folded): rotate back for the comparison
                        o = A.rotate_heads(o, vm.repeat_interleave(4, 0), rb).float()
                    outs[f"vrot{int(vrot)}/v{var}/ns{ns}/g{grid}"] = o
            tail_check_rows(rows, lines, "static", page, lens, tls, outs, K2, V2, q, tail)


def run(quick, kinds=AC.KINDS, out_tag=""):
    torch.manual_seed(0)
    rows, lines = [], []
    kcb = tuple(k for k in kinds if k in AC.KINDS)
    ref_ok = check_reference(lines, kcb)
    writer_ok, nw = check_writer(lines, quick, kcb)
    batches = [[1], [77, 5], [300, 1000, 64, 17], [4101, 33]] if quick else \
        [[1], [77, 5], [300, 1000, 64, 17], [4101, 33], [33000], [2048, 1, 129, 640, 63, 4096, 191, 1], [513, 777, 1023]]
    Cq, co = AC.random_calib(HKV, seed=7)
    for kind in [k for k in kinds if k in AC.KINDS]:
        for page in (16, 32):
            for bi, lens in enumerate(batches):
                seed = 100 * bi + page + len(kind)
                q, kvs = make_kv(lens, seed)
                cache, bt, lens_d = build(kind, page, lens, kvs, Cq, co, seed)
                K2, V2 = cache.reconstruct(bt, lens_d)                       # float64, original basis
                refs, sd = [], []
                for b, L in enumerate(lens):
                    kd, vd = K2[b:b + 1, :L], V2[b:b + 1, :L]
                    r = ref_attention(q[b:b + 1].float(), kd, vd)
                    refs.append(r)
                    s = Fn.scaled_dot_product_attention(q[b:b + 1].unsqueeze(2), kd.bfloat16().transpose(1, 2),
                                                        vd.bfloat16().transpose(1, 2), enable_gqa=True).squeeze(2)
                    sd.append(errs(s.float(), r)["max_rel"])
                for v in AC.VARIANTS[kind]:
                    for st in ((4,) if quick else (3, 4)):
                        for ns in (None, 1, 3):
                            if ns is not None and ns > -(-max(lens) // 64):
                                continue
                            o = AC.decode(q, cache, bt, lens_d, stages=st, nsplit=ns, lmax=max(lens), variant=v)
                            o = o.view(len(lens), 4 * HKV, D).float()
                            for b, L in enumerate(lens):
                                e = errs(o[b:b + 1], refs[b])
                                tol = max(2 * sd[b], 4e-3)
                                rows.append(dict(kind=kind, variant=v, page=page, lens=lens, b=b, L=L, stages=st,
                                                 nsplit=ns, **e, sdpa_bf16=sd[b], tol=tol,
                                                 passed=bool(e["max_rel"] <= tol)))
                cur = [r for r in rows if r["kind"] == kind and r["page"] == page and r["lens"] == lens]
                worst = max(cur, key=lambda r: r["max_rel"] / r["tol"])
                lines.append(f"attn   {kind:7s} page {page} lens {str(lens)[:40]:40s}: worst max|err|/max|ref| "
                             f"{worst['max_rel']:.2e} (tol {worst['tol']:.2e}, SDPA {worst['sdpa_bf16']:.2e}) "
                             f"{'ok' if all(r['passed'] for r in cur) else 'FAIL'}")
                print(lines[-1], flush=True)
    check_tail(lines, rows, quick, kinds)
    if "static" in kinds or kinds == AC.KINDS:
        check_tail_v2(lines, rows, quick)
    summ = []
    for kind, v in [(k, v) for k in kinds if k in AC.KINDS for v in AC.VARIANTS[k]]:
        ks = [r for r in rows if r["kind"] == kind and r["variant"] == v]
        summ.append(dict(kind=kind, variant=v, cases=len(ks), max_abs=max(r["max_abs"] for r in ks),
                         max_rel=max(r["max_rel"] for r in ks), rel_l2=max(r["rel_l2"] for r in ks),
                         sdpa_bf16_max_rel=max(r["sdpa_bf16"] for r in ks),
                         worst_err_over_tol=max(r["max_rel"] / r["tol"] for r in ks),
                         passed=all(r["passed"] for r in ks)))
    tr = [r for r in rows if r.get("tail") is not None]
    lines.append("")
    if tr:
        lines.append(f"BF16 staging window (0 .. 256 tokens, extra split CTA): {len(tr)} cases over "
                     f"{sorted({r['kind'] for r in tr})}, worst err/tol {max(r['max_rel'] / r['tol'] for r in tr):.2f}, "
                     f"all passed: {all(r['passed'] for r in tr)}")
    lines.append(f"reference == blockflag dyn_quant: {ref_ok}")
    lines.append(f"writer bit-exact vs codebook_ref ({nw} configurations): {writer_ok}")
    lines.append("| codebook | kernel variant | cases | max abs err | max abs err / max abs ref | max rel-L2 | bf16 SDPA max abs err / max abs ref | worst err/tol | pass |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in summ:
        lines.append(f"| {s['kind']} | {s['variant']} | {s['cases']} | {s['max_abs']:.2e} | {s['max_rel']:.2e} | {s['rel_l2']:.2e} | "
                     f"{s['sdpa_bf16_max_rel']:.2e} | {s['worst_err_over_tol']:.2f} | {s['passed']} |")
    txt = "\n".join(lines)
    print(txt)
    if not quick:
        open(os.path.join(HERE, "results", f"test_codebook{out_tag}_log.txt"), "w").write(txt + "\n")
        json.dump(dict(rows=rows, summary=summ, writer_bit_exact=writer_ok, reference_equals_dyn_quant=ref_ok),
                  open(os.path.join(HERE, "results", f"test_codebook{out_tag}.json"), "w"), indent=1)


# ------------------------------------------------------------------ generalized kernel
# key-only 2x2 caches (konly / konlya) read by the generalized static kernel (csrc/attn_decode_v2g.cuh) for GQA groups
# != 4 and head dim 256; reference = float64 attention over the reconstructions of the parsed codes / choices (as run()).
GEN_CFGS = [(4, 128, 4, 8), (5, 128, 8, 8), (6, 128, 8, 8), (8, 128, 8, 8), (8, 128, 4, 8), (12, 128, 4, 8),
            (6, 256, 4, 4), (1, 128, 4, 8)]


def check_reference_g(lines, kinds, D=256, Hkv=4):
    """D = 256: codebook_ref.encode (blockflag quantizer) == blockflag dyn_quant ('cq+f+konly', rb 32, int4:32 /
    int4a:32) on the reconstruction, keys and values."""
    import blockflag.qattention as QA
    import test_attn_v2 as T2
    T = 2000
    Cq, co = AC.random_calib(Hkv, seed=11, D=D)
    _, kvs = T2.make_kv_g([T], 5, 4, D, Hkv)
    kv = kvs[0]
    ok = True
    for cfg in kinds:
        for kind, x in (("k", kv[:, :Hkv]), ("v", kv[:, Hkv:])):
            c = CR.NOXF if kind == "v" else CR.CFG[cfg]
            QA.KV.update(dyn=dict(rule="cq", local=list(c["local"]), head=list(c["head"])), cq={0: Cq}, co={0: co},
                         dynstat={})
            asym = cfg == "konlya"
            r = CR.encode(x, Cq, co, kind, cfg, qk="asym" if asym else "blockflag")
            ref = QA.dyn_quant(x.transpose(0, 1)[None], "int4a:32" if asym else "int4:32", 32, kind, 0).squeeze(0).transpose(0, 1)
            nd = int((r["rec"] != ref).any(-1).sum())
            ok &= nd == 0
            lines.append(f"reference {cfg:7s} D {D} {kind}: rows differing from blockflag dyn_quant {nd} of {T * Hkv} "
                         f"(head F {int((r['mode'] == 2).sum())}, vectors H32 {int(r['local'][r['mode'] == 0].sum())})")
            print(lines[-1], flush=True)
    return ok


def check_writer_g(lines, kinds, quick, D=256, Hkv=4):
    """D = 256 writer (k_cb_write256): codes, scales, vector / head choices parsed from the pages == codebook_ref.encode
    with the cache quantizer, bit for bit (random slots, page 16 / 32, tokens per CTA 1 / 3 / 8, garbage pools)."""
    import test_attn_v2 as T2
    Cq, co = AC.random_calib(Hkv, seed=3, D=D)
    ok, n = True, 0
    for kind in kinds:
        for page in (16, 32):
            for T in ((1, 5, 64, 1000) if quick else (1, 5, 64, 1000, 4097)):
                _, kvs = T2.make_kv_g([T], 1000 + T + page, 4, D, Hkv)
                kv = kvs[0]
                if kind == "konlya" and T >= 1000:
                    kv = T2.edge_groups([kv], T + page)[0]
                npages = (T + page - 1) // page + 5
                c = AC.CodebookKVCache(kind, npages, page, Hkv, D=D)
                c.pool.random_(0, 256)
                c.set_calib(Cq, co)
                slots = torch.randperm(npages * page, device=DEV, generator=torch.Generator(device=DEV).manual_seed(T))[:T]
                qk = AC.QUANT.get(kind, "cache")
                refs = [CR.encode(kv[:, :Hkv], Cq, co, "k", kind, qk), CR.encode(kv[:, Hkv:], Cq, co, "v", kind, qk)]
                for tg in [t for t in ((1, 3, 8) if T > 1 else (1,)) if AC.ext().write_smem(AC.KIND_ID[kind], t, D) <= 200 * 1024]:
                    c.write(kv, slots, tg=tg)
                    r = c.parse_pages()
                    pg, off = slots // page, slots % page
                    bad = 0
                    for kk in range(2):
                        ref = refs[kk]
                        b = ((r["codes"][pg, off, :, kk] != ref["q"].long()).any(-1)
                             | (r["scales"][pg, off, :, kk] != ref["s"]).any(-1)
                             | (r["local"][pg, off, :, kk] != ref["local"]).any(-1)
                             | (r["mode"][pg, off, :, kk] != ref["mode"]))
                        bad += int(b.sum())
                    ok &= bad == 0
                    n += 1
                    lines.append(f"writer {kind:7s} D {D} page {page} T {T:5d} tokens/CTA {tg}: rows differing from "
                                 f"codebook_ref (codes, scales, choices) {bad} of {2 * T * Hkv}")
                    print(lines[-1], flush=True)
    return ok, n


def run_gen(quick, kinds=("konly", "konlya"), cfgs=None, out_tag=""):
    import test_attn_v2 as T2
    torch.manual_seed(0)
    rows, lines = [], []
    cfgs = GEN_CFGS if cfgs is None else cfgs
    ref_ok = writer_ok = True
    nw = 0
    if any(c[1] == 256 for c in cfgs):
        ref_ok = check_reference_g(lines, kinds)
        writer_ok, nw = check_writer_g(lines, kinds, quick)
    batches = [[1], [77, 5], [300, 1000, 64, 17], [4101, 33]] if quick else \
        [[1], [77, 5], [300, 1000, 64, 17], [4101, 33], [33000], [2048, 1, 129, 640, 63, 4096, 191, 1], [513, 777, 1023]]
    for (G, D, GT, Hkv) in cfgs:
        v = A2.gen_variant(G, D, GT)
        Cq, co = AC.random_calib(Hkv, seed=7, D=D)
        for kind in kinds:
            for page in ((16,) if quick else (16, 32)):
                for bi, lens in enumerate(batches):
                    seed = 100 * bi + page + len(kind) + 7 * G + D
                    q, kvs = T2.make_kv_g(lens, seed, G, D, Hkv)
                    lens_t = torch.tensor(lens, dtype=torch.int32)
                    need = sum((L + page - 1) // page for L in lens)
                    bt, npages = A2.alloc_block_table(lens_t, page, num_pages=need + 7, gen=torch.Generator().manual_seed(seed))
                    cache = AC.CodebookKVCache(kind, npages, page, Hkv, D=D)
                    cache.pool.random_(0, 256)
                    cache.set_calib(Cq, co)
                    for b, L in enumerate(lens):
                        cache.write(kvs[b], A2.slots_for(bt, b, range(L), page))
                    lens_d = lens_t.to(DEV)
                    K2, V2 = cache.reconstruct(bt, lens_d)
                    refs, sd = [], []
                    for b, L in enumerate(lens):
                        kd, vd = K2[b:b + 1, :L], V2[b:b + 1, :L]
                        r = ref_attention(q[b:b + 1].float(), kd, vd)
                        refs.append(r)
                        s = Fn.scaled_dot_product_attention(q[b:b + 1].unsqueeze(2), kd.bfloat16().transpose(1, 2),
                                                            vd.bfloat16().transpose(1, 2), enable_gqa=True).squeeze(2)
                        sd.append(errs(s.float(), r)["max_rel"])
                    for st in ((4,) if quick else (3, 4)):
                        for ns in (None, 1, 3):
                            if ns is not None and ns > -(-max(lens) // 64):
                                continue
                            o = AC.decode(q, cache, bt, lens_d, stages=st, nsplit=ns, lmax=max(lens), variant=v)
                            o = o.view(len(lens), G * Hkv, D).float()
                            for b, L in enumerate(lens):
                                e = errs(o[b:b + 1], refs[b])
                                tol = max(2 * sd[b], 4e-3)
                                rows.append(dict(kind=kind, G=G, D=D, GT=GT, Hkv=Hkv, page=page, lens=lens, b=b, L=L,
                                                 stages=st, nsplit=ns, **e, sdpa_bf16=sd[b], tol=tol,
                                                 passed=bool(e["max_rel"] <= tol)))
                    cur = [r for r in rows if r["kind"] == kind and r["G"] == G and r["D"] == D and r["GT"] == GT and
                           r["page"] == page and r["lens"] == lens]
                    worst = max(cur, key=lambda r: r["max_rel"] / r["tol"])
                    lines.append(f"attn   {kind:7s} G {G:2d} D {D} GT {GT} page {page} lens {str(lens)[:40]:40s}: worst "
                                 f"{worst['max_rel']:.2e} (tol {worst['tol']:.2e}) {'ok' if all(r['passed'] for r in cur) else 'FAIL'}")
                    print(lines[-1], flush=True)
            # BF16 staging window
            for ci, (lens, tls) in enumerate([([0, 77, 1000, 4101], [5, 256, 0, 129])] if quick else
                                             [([0, 77, 1000, 4101], [5, 256, 0, 129]), ([1, 33000], [255, 1])]):
                page, seed = 16, 950 + 10 * ci + G + D
                q, kvs = T2.make_kv_g(lens, seed, G, D, Hkv)
                lens_t = torch.tensor(lens, dtype=torch.int32)
                need = sum((L + page - 1) // page for L in lens)
                bt, npages = A2.alloc_block_table(lens_t, page, num_pages=need + 7, gen=torch.Generator().manual_seed(seed))
                cache = AC.CodebookKVCache(kind, npages, page, Hkv, D=D)
                cache.pool.random_(0, 256)
                cache.set_calib(Cq, co)
                for b, L in enumerate(lens):
                    cache.write(kvs[b], A2.slots_for(bt, b, range(L), page))
                lens_d = lens_t.to(DEV)
                K2, V2 = cache.reconstruct(bt, lens_d) if max(lens) > 0 else (None, None)
                gt_ = torch.Generator(device=DEV).manual_seed(seed)
                tail = (torch.randn(len(lens), 256, 2 * Hkv, D, device=DEV, generator=gt_) * 1.5).bfloat16()
                tl = torch.tensor(tls, dtype=torch.int32, device=DEV)
                for ns in (None, 3):
                    o = AC.decode(q, cache, bt, lens_d, lmax=max(lens), variant=v, nsplit=ns, tail=tail, tail_len=tl)
                    o = o.view(len(lens), G * Hkv, D).float()
                    for b, (L, T) in enumerate(zip(lens, tls)):
                        kd = torch.cat([K2[b:b + 1, :L], tail[b:b + 1, :T, :Hkv].double()], 1)
                        vd = torch.cat([V2[b:b + 1, :L], tail[b:b + 1, :T, Hkv:].double()], 1)
                        r = ref_attention(q[b:b + 1].float(), kd, vd)
                        s = Fn.scaled_dot_product_attention(q[b:b + 1].unsqueeze(2), kd.bfloat16().transpose(1, 2),
                                                            vd.bfloat16().transpose(1, 2), enable_gqa=True).squeeze(2)
                        sdv = errs(s.float(), r)["max_rel"]
                        e = errs(o[b:b + 1], r)
                        tol = max(2 * sdv, 4e-3)
                        rows.append(dict(kind=kind, G=G, D=D, GT=GT, Hkv=Hkv, page=page, lens=lens, b=b, L=L, tail=T,
                                         stages=4, nsplit=ns, **e, sdpa_bf16=sdv, tol=tol, passed=bool(e["max_rel"] <= tol)))
                cur = [r for r in rows if r.get("tail") is not None and r["kind"] == kind and r["G"] == G and
                       r["D"] == D and r["GT"] == GT and r["lens"] == lens]
                worst = max(cur, key=lambda r: r["max_rel"] / r["tol"])
                lines.append(f"tail   {kind:7s} G {G:2d} D {D} GT {GT} lens {lens} tails {tls}: worst {worst['max_rel']:.2e} "
                             f"(tol {worst['tol']:.2e}) {'ok' if all(r['passed'] for r in cur) else 'FAIL'}")
                print(lines[-1], flush=True)
    summ = []
    for (G, D, GT, Hkv) in cfgs:
        for kind in kinds:
            ks = [r for r in rows if r["kind"] == kind and r["G"] == G and r["D"] == D and r["GT"] == GT]
            if ks:
                summ.append(dict(kind=kind, G=G, D=D, GT=GT, Hkv=Hkv, cases=len(ks), max_abs=max(r["max_abs"] for r in ks),
                                 max_rel=max(r["max_rel"] for r in ks), rel_l2=max(r["rel_l2"] for r in ks),
                                 sdpa_bf16_max_rel=max(r["sdpa_bf16"] for r in ks),
                                 worst_err_over_tol=max(r["max_rel"] / r["tol"] for r in ks),
                                 passed=all(r["passed"] for r in ks)))
    lines.append("")
    if nw:
        lines.append(f"reference (D 256) == blockflag dyn_quant: {ref_ok}")
        lines.append(f"writer (D 256) bit-exact vs codebook_ref ({nw} configurations): {writer_ok}")
    lines.append("| codebook | G | D | q tile | Hkv | cases | max abs err | max err / max ref | max rel-L2 | bf16 SDPA max err / max ref | worst err/tol | pass |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summ:
        lines.append(f"| {s['kind']} | {s['G']} | {s['D']} | {s['GT']} | {s['Hkv']} | {s['cases']} | {s['max_abs']:.2e} | "
                     f"{s['max_rel']:.2e} | {s['rel_l2']:.2e} | {s['sdpa_bf16_max_rel']:.2e} | {s['worst_err_over_tol']:.2f} | {s['passed']} |")
    txt = "\n".join(lines)
    print(txt)
    if not quick:
        open(os.path.join(HERE, "results", f"test_codebook_gen{out_tag}_log.txt"), "w").write(txt + "\n")
        json.dump(dict(rows=rows, summary=summ, reference_equals_dyn_quant=ref_ok, writer_bit_exact=writer_ok, writer_configs=nw),
                  open(os.path.join(HERE, "results", f"test_codebook_gen{out_tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--kinds", default=",".join(AC.KINDS))
    ap.add_argument("--tag", default="", help="suffix of the result files (results/test_codebook<tag>_log.txt, .json)")
    ap.add_argument("--gen", action="store_true", help="key-only caches on the generalized kernel ")
    ap.add_argument("--cfgs", default=None, help="G:D:GT:Hkv,... (default GEN_CFGS)")
    a = ap.parse_args()
    if a.gen:
        cfgs = None if a.cfgs is None else [tuple(int(x) for x in c.split(":")) for c in a.cfgs.split(",")]
        kinds = tuple(k for k in a.kinds.split(",") if k in ("konly", "konlya")) or ("konly", "konlya")
        run_gen(a.quick, kinds, cfgs, a.tag)
    else:
        run(a.quick, tuple(a.kinds.split(",")), a.tag)
