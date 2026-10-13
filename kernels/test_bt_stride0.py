#!/usr/bin/env python3
"""Regression test: a block table whose rows are broadcast (stride 0, e.g. block_table[:1].expand(B, -1), as the vLLM
plugin's kernel-prefill path passes it) must give exactly the same output as the same table materialized row by row.
Covers static v2 INT4 and asymmetric INT4 (default variants and the split planner), sequence lengths from 1 to 32768.
usage: CUDA_VISIBLE_DEVICES=<gpu> python test_bt_stride0.py"""
import torch
import attn_decode_v2 as AD

ok = True
for fmt in ("int4", "int4a"):
    for L in (64, 1008, 4096, 32768):
        Hkv, D, rb = 8, 128, 32
        mask = torch.zeros(2 * Hkv, D // rb, dtype=torch.bool, device="cuda")
        cache = AD.make_bench_cache(fmt, 1, L, mask, rb, shuffle=True)
        lens = sorted({1, 17, min(L, 777), max(1, L - 1), L})
        B = len(lens)
        bt = cache.block_table[:1].expand(B, -1)
        sl = torch.tensor(lens, dtype=torch.int32, device="cuda")
        q = torch.randn(B, 32, D, device="cuda", dtype=torch.bfloat16)
        km = torch.zeros(Hkv, D // rb, dtype=torch.uint8, device="cuda")
        o_b = AD.decode(q, cache, bt, sl, kmask=km, vmask=km, rb=rb)
        o_c = AD.decode(q, cache, bt.contiguous(), sl, kmask=km, vmask=km, rb=rb)
        same = torch.equal(o_b, o_c)
        ok &= same
        print(f"{fmt:6s} L={L:6d} lens={lens}: {'identical' if same else 'MISMATCH max|d|=%.3g' % (o_b.float() - o_c.float()).abs().max().item()}")
print("PASS" if ok else "FAIL")
