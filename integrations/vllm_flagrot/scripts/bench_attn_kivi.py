"""kivi2 (+ Phase 3a): decode-attention kernel throughput (fmt kivi2 = libflagrot_kivi kv_attn, 97 B per token and
head, random codes, FP16 scale / min words in a sane range, random transform flags), INT2 vs INT4 vs FP8 (one layer's attention launch, Llama-3.1-8B /
Qwen3-8B shapes: 32 q / 8 KV heads, head 128; no BF16 window).  Each configuration is captured in a CUDA graph that
cycles through N distinct caches (pages never in L2 across launches), median over reps of the graph replay.

  formats  int4a    static v2 variant 264, INT4 pages (144 B per token and head)      = rtn_a / static_a
           int2a    the same kernel, INT2 pages (80 B)                                = rtn_a2 / static_a2
           konlya   k_cbk (key-only 2x2, per-token selection), INT4 pages             = konlya / konlya_c
           konlya2  k_cbk, INT2 pages                                                 = konlya2 / konlya_c2
           fp8      vLLM FlashAttention-3 decode on an FP8 E4M3 paged cache (256 B), = vLLM kv_cache_dtype=fp8
           bf16     vLLM FlashAttention-3 decode on a BF16 paged cache (512 B)

  CUDA_VISIBLE_DEVICES=7 python integrations/vllm_flagrot/scripts/bench_attn_kivi.py --out results/bench/attn_kivi.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vllm_flagrot import kernels as K  # noqa: E402
from vllm_flagrot import kivi as KIVI  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, nargs="+", default=[32, 64])
ap.add_argument("--L", type=int, nargs="+", default=[16384, 32768, 65536])
ap.add_argument("--fmts", default="kivi2p8,kivi2,int2a,konlya2,fp8,bf16")
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--mem_gb", type=float, default=24.0, help="cache bytes cycled per configuration (cap)")
ap.add_argument("--hq", type=int, default=32)
ap.add_argument("--hkv", type=int, default=8)
ap.add_argument("--D", type=int, default=128)
ap.add_argument("--occ", type=int, default=0, help="KIVI.plan resident CTAs per SM (0: the format default)")
ap.add_argument("--out", default="")
a = ap.parse_args()
dev = "cuda"
HQ, HKV, D, P = a.hq, a.hkv, a.D, 16
CELL = {"kivi2p8": 10 * D // 16, "kivi2p8f": 10 * D // 16, "kivi2f": (12 * D + 16) // 16, "kivi2": (12 * D + 16) // 16, "int4a": 144, "int2a": 80, "konlya": 144, "konlya2": 80, "fp8": 256, "bf16": 512}


def bench_graph(fn, ncopy):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(ncopy):
            fn(i)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(a.iters):
            for i in range(ncopy):
                fn(i)
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(a.reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / (a.iters * ncopy))
    return statistics.median(ts), min(ts)


def run(fmt, B, L):
    g = torch.Generator().manual_seed(0)
    npg = (L + P - 1) // P
    tok_bytes = HKV * CELL[fmt]
    one = B * npg * P * tok_bytes
    ncopy = max(1, min(8, int(a.mem_gb * 1e9 // one)))
    q = torch.randn(B, HQ, D, generator=g).bfloat16().to(dev)
    out = torch.empty(B, HQ * D, dtype=torch.bfloat16, device=dev)
    seq = torch.full((B,), L, dtype=torch.int32, device=dev)
    scale = D ** -0.5
    if fmt in ("fp8", "bf16"):
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        dt = torch.float8_e4m3fn if fmt == "fp8" else torch.bfloat16
        caches = []
        for _ in range(ncopy):
            kc = (torch.randn(B * npg, P, HKV, D, device=dev) * 0.5).to(dt)
            vc = (torch.randn(B * npg, P, HKV, D, device=dev) * 0.5).to(dt)
            bt = torch.randperm(B * npg, device=dev).view(B, npg).int()
            caches.append((kc, vc, bt))
        o3 = out.view(B, HQ, D)
        ds = torch.ones(B, HKV, dtype=torch.float32, device=dev) if fmt == "fp8" else None
        qf = q.to(torch.float8_e4m3fn) if fmt == "fp8" else q      # (vLLM's fp8 call: q quantized, descales)
        cuq = torch.arange(B + 1, dtype=torch.int32, device=dev)

        def fn(i):
            kc, vc, bt = caches[i]
            kw = dict(q_descale=ds, k_descale=ds, v_descale=ds) if fmt == "fp8" else {}
            flash_attn_varlen_func(qf, kc, vc, 1, cuq, L, seqused_k=seq, block_table=bt, softmax_scale=scale,
                                   causal=True, out=o3, fa_version=3, **kw)
        med, mn = bench_graph(fn, ncopy)
        return dict(fmt=fmt, B=B, L=L, us=med, us_min=mn, ncopy=ncopy, kv_bytes=one, nsplit=None)
    if fmt in ("kivi2p8", "kivi2p8f"):
        chunk = KIVI.chunk_bytes(D, True)
        caches = []
        for _ in range(ncopy):
            pool = torch.randint(0, 256, (B * npg, HKV, chunk), dtype=torch.uint8, generator=g)
            for o0, o1 in ((4 * D, 5 * D), (9 * D, 10 * D)):          # (s8 | flag, m8) byte pairs, sane E4M3 values
                n = (o1 - o0) // 2
                sb = torch.randint(0x18, 0x30, (B * npg, HKV, n), generator=g, dtype=torch.int32) | \
                    (torch.randint(0, 2, (B * npg, HKV, n), generator=g, dtype=torch.int32) << 7)
                mb = torch.randint(0x20, 0x38, (B * npg, HKV, n), generator=g, dtype=torch.int32)
                pool[:, :, o0:o1] = torch.stack([sb, mb], -1).reshape(B * npg, HKV, -1).to(torch.uint8)
            caches.append((pool.to(dev), torch.randperm(B * npg, generator=g).view(B, npg).int().to(dev)))
        fast = fmt == "kivi2p8f"
        nsplit, ch = KIVI.plan(B, L, HKV, occ=a.occ or (KIVI.fast_occ(D, HQ // HKV) if fast else None))
        ws = KIVI.Workspace(B * HKV * (nsplit + 1), HQ // HKV, D, dev)
        cached = seq.clone()

        def fn(i):
            pool, bt = caches[i]
            KIVI.attn(q, pool, bt, cached, out, nsplit, ch, ws, scale * 1.4426950408889634, 0, HQ // HKV, f8=True,
                      fast=fmt == "kivi2p8f")
        med, mn = bench_graph(fn, ncopy)
        return dict(fmt=fmt, B=B, L=L, us=med, us_min=mn, ncopy=ncopy, kv_bytes=one, nsplit=nsplit, chunk=ch)
    if fmt in ("kivi2", "kivi2f"):
        chunk = KIVI.chunk_bytes(D)
        caches = []
        for _ in range(ncopy):
            pool = torch.randint(0, 256, (B * npg, HKV, chunk), dtype=torch.uint8, generator=g)
            for o0, o1 in ((4 * D, 6 * D), (10 * D, 12 * D)):          # (scale, min) half2 words
                n = (o1 - o0) // 4
                sw = torch.randint(0x1C00, 0x2C00, (B * npg, HKV, n), generator=g, dtype=torch.int32) | \
                    (torch.randint(0, 2, (B * npg, HKV, n), generator=g, dtype=torch.int32) << 15) | \
                    (torch.randint(0x2000, 0x2400, (B * npg, HKV, n), generator=g, dtype=torch.int32) << 16)
                pool[:, :, o0:o1] = sw.view(torch.uint8).view(B * npg, HKV, -1)[:, :, :o1 - o0]
            caches.append((pool.to(dev), torch.randperm(B * npg, generator=g).view(B, npg).int().to(dev)))
        fast = fmt == "kivi2f"
        nsplit, ch = KIVI.plan(B, L, HKV, occ=a.occ or (KIVI.fast_occ(D, HQ // HKV) if fast else None))
        ws = KIVI.Workspace(B * HKV * (nsplit + 1), HQ // HKV, D, dev)
        cached = seq.clone()

        def fn(i):
            pool, bt = caches[i]
            KIVI.attn(q, pool, bt, cached, out, nsplit, ch, ws, scale * 1.4426950408889634, 0, HQ // HKV,
                      fast=fmt == "kivi2f")
        med, mn = bench_graph(fn, ncopy)
        return dict(fmt=fmt, B=B, L=L, us=med, us_min=mn, ncopy=ncopy, kv_bytes=one, nsplit=nsplit, chunk=ch)
    chunk = K.v2_chunk_bytes(P, D, fmt) if fmt in ("int4a", "int2a") else K.cb_chunk_bytes(fmt, P, D)
    caches = []
    for _ in range(ncopy):
        pool = torch.randint(0, 256, (B * npg, HKV, chunk), dtype=torch.uint8, generator=g).to(dev)
        # sane scales: random FP16 scale words in [2^-8, 2^-4] (sign bits = random flags for konlya*), zero points random
        ns = chunk - (64 if fmt in ("int2a", "konlya2") else 128) * P
        sw = (torch.randint(0x1C00, 0x2C00, (B * npg, HKV, ns // 2), generator=g, dtype=torch.int32)
              | (torch.randint(0, 2, (B * npg, HKV, ns // 2), generator=g, dtype=torch.int32) << 15))
        pool[:, :, chunk - ns:] = sw.to(torch.int16).view(torch.uint8).to(dev).view(B * npg, HKV, ns)
        bt = torch.randperm(B * npg, generator=g).view(B, npg).int().to(dev)
        caches.append((pool, bt))
    if fmt in ("int4a", "int2a"):
        nsplit, ch = K.plan_v2(B, L, HKV, P, fmt=fmt)
        HV, GT = K.vheads(HKV, 4, D, fmt), K.tile_of(K.v2_variant(fmt, 4, D))
    else:
        nsplit, ch = K.plan_cb(B, L, HKV, P, fmt)
        HV, GT = HKV, 4
    ws = K.Workspace(B, HV, B * HV * (nsplit + 1), dev, GT, D)
    km = torch.zeros(HKV, 4, dtype=torch.uint8, device=dev)

    def fn(i):
        pool, bt = caches[i]
        if fmt in ("int4a", "int2a"):
            K.v2_attn(q, pool, bt, seq, out, nsplit, ch, ws, scale * 1.4426950408889634, kmask=km, vmask=km, rb=32, fmt=fmt)
        else:
            K.cb_attn(q, pool, bt, seq, out, nsplit, ch, ws, scale * 1.4426950408889634, fmt)
    med, mn = bench_graph(fn, ncopy)
    return dict(fmt=fmt, B=B, L=L, us=med, us_min=mn, ncopy=ncopy, kv_bytes=one, nsplit=nsplit, chunk=ch)


res = []
for B in a.B:
    for L in a.L:
        for fmt in a.fmts.split(","):
            try:
                r = run(fmt, B, L)
            except Exception as ex:  # noqa: BLE001
                r = dict(fmt=fmt, B=B, L=L, error=repr(ex)[:300])
            r["GBps"] = r.get("kv_bytes", 0) / max(r.get("us", 1e-9), 1e-9) / 1e3 if "us" in r else None
            res.append(r)
            print(json.dumps(r), flush=True)
            torch.cuda.empty_cache()
if a.out:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), results=res), indent=1))
