#!/usr/bin/env python3
"""Correctness of the static FlagRot INT4 / INT4A writer in the v3 style (csrc/attn_cb_write3.cuh k_w3s).

Reference = the vLLM plugin's v1 static write path: whole-head rotation of the flagged heads (fr_head_fwht: bf16(H128 x
/ 16) with the unnormalized Sylvester butterflies in fp32, emulated here operation for operation) followed by the v2
paged writer (attn_decode_v2.PagedKVCache.write, rb 32, A2_INT4 / A2_INT4A).  Both write into copies of the same
garbage-filled pool; the pools must be byte-identical.
Sweep: INT4 / INT4A x page 16 / 32 x T in {1, 5, 64, 1000, 4097} x slot patterns (random slots = decode steps,
consecutive page-aligned runs = prefill, runs with -1 holes, offset runs not 8-aligned) x both kernels (k_w3s tiles,
k_w3s_dec warp per row) x flags (none = rtn_a, random
block flags, random block + whole-head flags = static_a) x inputs (LLM-like rows; for INT4A + test_attn_v2.edge_groups
and constant groups); plus real Llama-3.1-8B rows (capture_kv_cb.py) if --kv is given.
usage: CUDA_VISIBLE_DEVICES=5 python test_w3_static.py [--kv kv_llama8b.pt] [--quick]
-> results/test_w3_static_log.txt, results/test_w3_static.json
"""
from __future__ import annotations

import argparse
import json
import os

import torch

import attn_codebook as AC
import attn_decode_v2 as A2
from test_attn_v2 import edge_groups, make_kv
from test_writer3 import const_groups

HERE = os.path.dirname(os.path.abspath(__file__))
HKV = 8


def head_fwht_ref(kv, hfull, scale=1.0 / 16):
    """fr_head_fwht (integrations/vllm_flagrot/csrc/flagrot_capi.cu) in torch: rows of flagged heads -> bf16(scale *
    H128 x), butterflies (a + b, a - b) for h = 1, 2, .., 64 in fp32 (same operations, same order)."""
    x = kv.float().clone()
    T, H2, D = x.shape
    y = x
    h = 1
    while h < D:
        y = y.view(T, H2, D // (2 * h), 2, h)
        a, b = y[:, :, :, 0], y[:, :, :, 1]
        y = torch.stack([a + b, a - b], 3).reshape(T, H2, D)
        h *= 2
    y = (y * scale).bfloat16()
    return torch.where(hfull.bool().view(1, H2, 1), y, kv)


def slot_patterns(T, page, npages, g):
    perm = torch.randperm(npages, device="cuda", generator=g)
    out = [("random", torch.randperm(npages * page, device="cuda", generator=g)[:T].long())]
    np_ = (T + page - 1) // page
    run = (perm[:np_, None] * page + torch.arange(page, device="cuda")[None]).reshape(-1)[:T].long()
    out.append(("page runs", run))
    if T >= 5:
        holes = run.clone()
        holes[torch.randperm(T, device="cuda", generator=g)[:max(1, T // 7)]] = -1
        out.append(("runs with -1", holes))
        off = (perm[:np_ + 1, None] * page + torch.arange(page, device="cuda")[None]).reshape(-1)[3:3 + T].long()
        out.append(("runs offset 3", off))   # consecutive within pages but not 8-aligned
    return out


def check(kv, slots, mask, hfull, fmt, page, npages, path=0):
    asym = fmt == "int4a"
    c1 = A2.PagedKVCache(fmt, npages, page, HKV)
    c1.pool.random_(0, 256, generator=torch.Generator(device="cuda").manual_seed(11))
    pool0 = c1.pool.clone()
    x = head_fwht_ref(kv, hfull) if hfull is not None else kv
    c1.write(x, slots, mask if mask is not None else torch.zeros(2 * HKV, 4, dtype=torch.uint8, device="cuda"), 32)
    p3 = pool0.clone().view(npages, HKV, -1)
    AC.ext().w3_static_write(kv, slots, mask, hfull, int(asym), p3, page, path)
    torch.cuda.synchronize()
    return int((p3.view_as(c1.pool) != c1.pool).sum())


def flag_sets(g):
    km = (torch.rand(2 * HKV, 4, device="cuda", generator=g) < 0.5).to(torch.uint8)
    hf = (torch.rand(2 * HKV, device="cuda", generator=g) < 0.4).to(torch.uint8)
    hf[0], hf[HKV] = 1, 1
    return [("rtn (no flags)", None, None), ("block flags", km, None), ("block + whole-head", km, hf)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    lines, ok, n = [], True, 0
    for fmt in ("int4", "int4a"):
        for page in (16, 32):
            for T in ((1, 64, 1000) if a.quick else (1, 5, 64, 1000, 4097)):
                q, kvs = make_kv([T], 2000 + T + page)
                inputs = [("llm-like", kvs[0])]
                if fmt == "int4a" and T >= 64:
                    inputs.append(("edge groups", const_groups(edge_groups([kvs[0]], T + page)[0], T)))
                npages = (T + page - 1) // page + 9
                g = torch.Generator(device="cuda").manual_seed(T + page)
                for fname, mask, hfull in flag_sets(g):
                    for iname, kv in inputs:
                        for sname, slots in slot_patterns(T, page, npages, g):
                            for path in (1, 2):   # k_w3s (tiles), k_w3s_dec (warp per row)
                                nd = check(kv, slots, mask, hfull, fmt, page, npages, path)
                                ok &= nd == 0
                                n += 1
                                lines.append(f"{fmt:5s} page {page} T {T:5d} {fname:18s} {iname:11s} {sname:13s} "
                                             f"{'tiles' if path == 1 else 'rows '}: pool bytes != v1 path {nd}")
                                print(lines[-1], flush=True)
    real_ok = None
    if a.kv and os.path.exists(a.kv):
        dat = torch.load(a.kv)
        real_ok = True
        g = torch.Generator(device="cuda").manual_seed(5)
        for li in dat["layers"]:
            kv = torch.cat([dat["K"][li], dat["V"][li]], 1).cuda().contiguous()
            T = kv.shape[0]
            npages = T // 16 + 4
            for fmt in ("int4", "int4a"):
                for fname, mask, hfull in flag_sets(g):
                    for sname, slots in slot_patterns(T, 16, npages, g)[:2]:
                        nd = check(kv, slots, mask, hfull, fmt, 16, npages, 1) + check(kv, slots, mask, hfull, fmt, 16, npages, 2)
                        real_ok &= nd == 0
                        n += 1
                        lines.append(f"real layer {li:2d} {fmt:5s} T {T} {fname:18s} {sname:10s}: pool bytes != v1 path {nd}")
                        print(lines[-1], flush=True)
    lines.append(f"\n{n} configurations; synthetic byte-identical: {ok}; real rows byte-identical: {real_ok}")
    print(lines[-1])
    if not a.quick:
        open(os.path.join(HERE, "results", "test_w3_static_log.txt"), "w").write("\n".join(lines) + "\n")
        json.dump(dict(configs=n, synthetic_ok=ok, real_ok=real_ok), open(os.path.join(HERE, "results", "test_w3_static.json"), "w"))


if __name__ == "__main__":
    main()
