#!/usr/bin/env python3
"""Correctness of the codebook cache writer v3 (csrc/attn_cb_write3.cuh).

1. Synthetic (random_calib, LLM-like rows): for every v3 codebook, page 16 / 32, T in {1, 5, 64, 1000, 4097}, random
   slots in a garbage-filled pool, plus per-(token, kv head) slots and a strided kv input: codes (stored coordinates),
   fp16 scales, per-vector choices and head modes parsed from the pages == codebook_ref.encode (cache quantizer), bit
   for bit; and the whole pool == the pool written by writer v1 (k_cb_write), byte for byte.
2. Real rows (capture_kv_cb.py: Llama-3.1-8B post-RoPE keys / values of 7 layers on WikiText-2 test windows, the
   layers' own C_q / c_o): pool v3 == pool v1 byte for byte for every codebook and layer (all tokens); the fraction of
   (token, vector) / (token) choices that needed the exact fallback (the tensor-core bound did not separate the
   candidates), and -- with a build without the fallback (-DW3_NOFB=1) -- how many choices the bare tensor-core
   ranking would have flipped (tie-level flips) and their relative error excess.
usage: CUDA_VISIBLE_DEVICES=3 python test_writer3.py --kv /path/kv_llama8b.pt [--quick]
-> results/test_writer3_log.txt, results/test_writer3.json
"""
from __future__ import annotations

import argparse
import json
import os

import torch

import attn_codebook as AC
import codebook_ref as CR
from test_attn_v2 import edge_groups, make_kv

HERE = os.path.dirname(os.path.abspath(__file__))
HKV = 8


def pool_of(kind, kv, slots, Cq, co, page, npages, writer, stats=None):
    c = AC.CodebookKVCache(kind, npages, page, HKV)
    c.pool.random_(0, 256, generator=torch.Generator(device="cuda").manual_seed(7))
    c.set_calib(Cq, co)
    c.writer = writer
    c.w3_stats = stats
    c.write(kv, slots)
    torch.cuda.synchronize()
    return c


def const_groups(kv, seed):
    """asymmetric writer edge case on top of edge_groups: some 32-groups constant (s = 1e-12 in int4a:32: huge quotients,
    the writer's rintf path) or a single non-zero element."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = kv.float().clone()
    flat = x.view(x.shape[0], -1, 32)
    sel = torch.randint(0, 12, flat.shape[:2], device="cuda", generator=g)
    flat[sel == 0] = 1.5
    flat[sel == 1] = -0.375
    one = torch.zeros(32, device="cuda")
    one[7] = 2.0
    flat[sel == 2] = one
    return x.bfloat16()


def compare_ref(c, kv, slots, Cq, co, kind, page):
    r = c.parse_pages()
    bad = 0
    sl = slots if slots.dim() == 1 else None
    for kk, x in ((0, kv[:, :HKV]), (1, kv[:, HKV:])):
        ref = CR.encode(x, Cq, co, "k" if kk == 0 else "v", kind, AC.QUANT.get(kind, "cache"))
        if sl is not None:
            pg, off = sl // page, sl % page
            get = lambda a: a[pg, off, :, kk]
        else:   # per-(token, head) slots
            pg, off = slots // page, slots % page                                  # [T, H]
            hh = torch.arange(HKV, device=slots.device).expand_as(pg)
            get = lambda a: a[pg, off, hh, kk]
        b = ((get(r["codes"]) != ref["q"].long()).any(-1) | (get(r["scales"]) != ref["s"]).any(-1)
             | (get(r["local"]) != ref["local"]).any(-1) | (get(r["mode"]) != ref["mode"]))
        bad += int(b.sum())
    return bad


def synthetic(lines, quick, kinds):
    Cq, co = AC.random_calib(HKV, seed=3)
    ok, n = True, 0
    for kind in kinds:
        for page in (16, 32):
            for T in ((1, 64, 1000) if quick else (1, 5, 64, 1000, 4097)):
                q, kvs = make_kv([T], 1000 + T + page)
                kv = kvs[0]
                if kind == "konlya" and T >= 1000:   # asymmetric edge cases (test_codebook): + constant groups
                    kv = const_groups(edge_groups([kv], T + page)[0], T)
                npages = (T + page - 1) // page + 5
                g = torch.Generator(device="cuda").manual_seed(T)
                variants = [("slots", torch.randperm(npages * page, device="cuda", generator=g)[:T], kv)]
                if T >= 64:
                    ph = torch.stack([torch.randperm(npages * page, device="cuda", generator=g)[:T] for _ in range(HKV)], 1)
                    variants.append(("per-head slots", ph.contiguous(), kv))
                    big = torch.zeros(T, 3 * HKV, 128, device="cuda", dtype=torch.bfloat16)   # strided rows (qkv slice)
                    big[:, HKV:] = kv
                    variants.append(("strided kv", variants[0][1], big[:, HKV:]))
                for name, slots, x in variants:
                    c3 = pool_of(kind, x, slots, Cq, co, page, npages, "v3")
                    c1 = pool_of(kind, x, slots, Cq, co, page, npages, "v1")
                    bad = compare_ref(c3, kv, slots, Cq, co, kind, page)
                    nd = int((c3.pool != c1.pool).sum())
                    ok &= bad == 0 and nd == 0
                    n += 1
                    lines.append(f"synthetic {kind:7s} page {page} T {T:5d} {name:15s}: rows != codebook_ref {bad} of "
                                 f"{2 * T * HKV}; pool bytes != writer v1 {nd}")
                    print(lines[-1], flush=True)
    return ok, n


def real(lines, path, quick, rec, kinds):
    dat = torch.load(path)
    layers = dat["layers"][:3] if quick else dat["layers"]
    nofb = None
    try:   # second build without the exact fallback: the bare tensor-core ranking
        os.environ["ATTNCB_EXTRA_FLAGS"] = "-DW3_NOFB=1"
        AC._EXT = None
        nofb = AC.ext()
    finally:
        os.environ.pop("ATTNCB_EXTRA_FLAGS", None)
        AC._EXT = None
    ok = True
    page = 16
    for kind in kinds:
        agg = dict(rows=0, lf=0, li=0, hf=0, hi=0, flip_local=0, flip_head=0, rows_changed=0)
        for li in layers:
            K, V = dat["K"][li], dat["V"][li]
            T = K.shape[0] if not quick else 2048
            kv = torch.cat([K[:T], V[:T]], 1).cuda().contiguous()
            Cq, co = dat["Cq"][li].cuda().float(), dat["co"][li].cuda().float()
            npages = (T + page - 1) // page
            slots = torch.arange(T, device="cuda")
            st = torch.zeros(64, dtype=torch.int32, device="cuda")
            c3 = pool_of(kind, kv, slots, Cq, co, page, npages, "v3", st)
            c1 = pool_of(kind, kv, slots, Cq, co, page, npages, "v1")
            nd = int((c3.pool != c1.pool).sum())
            ok &= nd == 0
            s = st[:4].tolist()
            agg["rows"] += T * HKV
            agg["li"] += s[0]; agg["lf"] += s[1]; agg["hi"] += s[2]; agg["hf"] += s[3]
            # bare tensor-core ranking (no fallback)
            AC._EXT, keep = nofb, AC._EXT
            try:
                c0 = pool_of(kind, kv, slots, Cq, co, page, npages, "v3")
            finally:
                AC._EXT = keep
            r0, r1 = c0.parse_pages(), c3.parse_pages()
            fl = (r0["local"][..., 0, :] != r1["local"][..., 0, :]) & (r1["mode"][..., 0:1] == 0)
            fh = r0["mode"][..., 0] != r1["mode"][..., 0]
            agg["flip_local"] += int(fl.sum()); agg["flip_head"] += int(fh.sum())
            agg["rows_changed"] += int((fl.any(-1) | fh).sum())
            lines.append(f"real {kind:7s} layer {li:2d} T {T}: pool bytes != writer v1 {nd}; exact fallback: local "
                         f"{s[1] / max(1, s[0]):.4f}, head {s[3] / max(1, s[2]):.4f}; bare tensor-core ranking: "
                         f"{int(fl.sum())} vector / {int(fh.sum())} head-mode flips of {T * HKV} key rows")
            print(lines[-1], flush=True)
        rec[kind] = agg
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", default=None, help="capture_kv_cb.py output (real rows); skipped if absent")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--kinds", default=",".join(AC.W3_KINDS))
    ap.add_argument("--tag", default="", help="suffix of the result files (e.g. _asym)")
    a = ap.parse_args()
    lines, rec = [], {}
    kinds = a.kinds.split(",")
    syn_ok, n = synthetic(lines, a.quick, kinds)
    real_ok = None
    if a.kv and os.path.exists(a.kv):
        real_ok = real(lines, a.kv, a.quick, rec, kinds)
    lines.append("")
    lines.append(f"synthetic: {n} configurations, all bit-exact vs codebook_ref and byte-identical to writer v1: {syn_ok}")
    if real_ok is not None:
        lines.append(f"real rows: pools byte-identical to writer v1 for every codebook and layer: {real_ok}")
        for kind, g in rec.items():
            lines.append(f"  {kind:7s}: {g['rows']} key rows; exact fallback on {g['lf'] / max(1, g['li']):.4f} of the "
                         f"(token, vector) and {g['hf'] / max(1, g['hi']):.4f} of the (token) choices; without it the "
                         f"tensor-core ranking would flip {g['flip_local']} vector and {g['flip_head']} head choices "
                         f"({g['rows_changed'] / max(1, g['rows']):.2e} of the rows)")
    txt = "\n".join(lines)
    print(txt)
    if not a.quick:
        open(os.path.join(HERE, "results", f"test_writer3{a.tag}_log.txt"), "w").write(txt + "\n")
        json.dump(dict(synthetic_ok=syn_ok, synthetic_configs=n, real_ok=real_ok, real=rec),
                  open(os.path.join(HERE, "results", f"test_writer3{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
