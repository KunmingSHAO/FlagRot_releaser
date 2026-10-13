#!/usr/bin/env python3
"""Correctness of the quantized-KV decode attention kernel (attn_decode.py / csrc/attn_decode.cu).

Reference = float64 attention on the DEQUANTIZED cache (ref.py dequantizers, exactly the bytes the kernel reads),
in the original basis: K_ref = R_k Kdeq, V_ref = R_v Vdeq (R = FlagRot with the cache's flags, R = R^-1).
  vrot=True  -> kernel output must equal  attn(q, K_ref, V_ref)
  vrot=False -> kernel output must equal  R_v attn(q, K_ref, V_ref)   (V rotation folded into W_o: rotated basis)
Errors: max |err|, max |err| / max |ref|, rel-L2.  Tolerance yardstick: torch SDPA in BF16 on the same dequantized
K/V vs the same float64 reference (what a BF16 attention kernel gets).
Also reported: end-to-end error vs attention on the ORIGINAL bf16 K/V (= quantization error of each format, with and
without FlagRot) for K with outlier channels.
usage: CUDA_VISIBLE_DEVICES=7 python test_attn.py [--quick]  -> results/test_attn_log.txt, results/test_attn.json
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch
import torch.nn.functional as Fn

import attn_decode as A
import ref as R

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"
HKV, D = 8, 128
RBF = {"bf16": 64, "fp8": 64, "nvfp4": 16, "mxfp4": 32, "int4": 32}


def make_kv(B, L, seed, outliers=True):
    g = torch.Generator(device=DEV).manual_seed(seed)
    k = torch.randn(B, L, HKV, D, device=DEV, generator=g)
    v = torch.randn(B, L, HKV, D, device=DEV, generator=g)
    if outliers:  # LLM-like keys: a few large channels per head, plus a per-token magnitude spread
        ch = torch.randint(0, D, (HKV, 4), device=DEV, generator=g)
        for h in range(HKV):
            k[:, :, h, ch[h]] *= 6.0
        k = k * (0.5 + torch.rand(B, L, 1, 1, device=DEV, generator=g))
    q = torch.randn(B, 4 * HKV, D, device=DEV, generator=g) * 0.6
    return q.bfloat16(), torch.cat([k, v], 2).bfloat16()  # kv [B, L, 2Hkv, D]


def errs(o, ref):
    e = (o.double() - ref.double())
    return dict(max_abs=e.abs().max().item(), max_rel=(e.abs().max() / ref.abs().max()).item(),
                rel_l2=(e.norm() / ref.norm()).item())


def run(quick):
    torch.manual_seed(0)
    rows, lines = [], []
    shapes = [(1, 1), (1, 77), (3, 300), (2, 4101)] if quick else [(1, 1), (1, 77), (3, 300), (2, 4101), (5, 1000), (1, 33000)]
    worst = {}
    for fmt in A.FMTS:
        rb = RBF[fmt]
        for (B, L) in shapes:
            q, kv = make_kv(B, L, 1000 + B * 7 + L)
            g = torch.Generator(device="cpu").manual_seed(L)
            mask = (torch.rand(2 * HKV, D // rb, generator=g) < 0.6).to(torch.uint8).to(DEV)
            if fmt == "bf16":
                mask.zero_()  # the bf16 cache is not rotated
            cache = A.KVCache(fmt, B, L + 5, HKV)
            if fmt == "nvfp4":  # calibrated global scale (448 * 6 / amax of the rotated tensor), as in the tests of
                cache.gs = R.nvfp4_global_scale(A.rotate_heads(kv.float(), mask, rb)).to(DEV)
            cache.fill(kv, mask, rb)
            deq = cache.dequant(L)  # [B, L, 2Hkv, D] rotated basis
            kd = A.rotate_heads(deq[:, :, :HKV], mask[:HKV], rb)
            vd = A.rotate_heads(deq[:, :, HKV:], mask[HKV:], rb)
            ref = A.ref_attention(q.float(), kd, vd)                                   # original basis
            ref_rot = A.rotate_heads(ref.float(), mask[HKV:].repeat_interleave(4, 0), rb).double()  # rotated V basis
            # BF16 SDPA yardstick on the same dequantized K/V
            sd = Fn.scaled_dot_product_attention(q.unsqueeze(2), kd.bfloat16().transpose(1, 2),
                                                 vd.bfloat16().transpose(1, 2), enable_gqa=True).squeeze(2)
            ys = errs(sd.float(), ref)
            worst.setdefault((fmt, "sdpa_bf16"), []).append(ys["max_rel"])
            variants = [v for v in A.VARIANTS if A.ext().occupancy(A.FMT_ID[fmt], False, v) > 0]
            for v in variants if not quick else variants[:2]:
                for qrot in ([None, "fused", "pre"] if fmt != "bf16" else [None]):
                    for vrot in ([False, True] if fmt != "bf16" else [False]):
                        for ns in (None, 1, 7):
                            if ns is not None and ns > L:
                                continue
                            if fmt != "bf16" and qrot is None:
                                continue  # K was rotated: q must be rotated too
                            o = A.decode(q, cache, L, kmask=mask[:HKV].contiguous(), vmask=mask[HKV:].contiguous(),
                                         rb=rb, qrot=qrot, vrot=vrot, variant=v, nsplit=ns)
                            o = o.view(B, 4 * HKV, D).float()
                            r = ref if (vrot or fmt == "bf16") else ref_rot
                            e = errs(o, r)
                            row = dict(fmt=fmt, B=B, L=L, variant=v, qrot=qrot, vrot=vrot, nsplit=ns, **e,
                                       sdpa_bf16=ys)
                            rows.append(row)
                            worst.setdefault((fmt, "kernel"), []).append(e["max_rel"])
            # end-to-end quantization error (kernel, deployment variant + back-rotation) vs the ORIGINAL K/V
            kvf = kv.float()
            ref0 = A.ref_attention(q.float(), kvf[:, :, :HKV], kvf[:, :, HKV:])
            o = A.decode(q, cache, L, kmask=mask[:HKV].contiguous(), vmask=mask[HKV:].contiguous(), rb=rb,
                         qrot="fused" if fmt != "bf16" else None, vrot=fmt != "bf16").view(B, 4 * HKV, D).float()
            e0 = errs(o, ref0)
            # same format without FlagRot (zero mask) for comparison
            z = torch.zeros_like(mask)
            c2 = A.KVCache(fmt, B, L + 5, HKV)
            if fmt == "nvfp4":
                c2.gs = R.nvfp4_global_scale(kv.float()).to(DEV)
            c2.fill(kv, z, rb)
            o2 = A.decode(q, c2, L).view(B, 4 * HKV, D).float()
            e2 = errs(o2, ref0)
            rows.append(dict(fmt=fmt, B=B, L=L, kind="vs_original", flagrot=e0, norot=e2))
            lines.append(f"{fmt:6s} B={B} L={L:6d}: kernel vs dequant-ref worst max|err|/max|ref| = "
                         f"{max(r['max_rel'] for r in rows if r.get('fmt') == fmt and r.get('B') == B and r.get('L') == L and 'max_rel' in r):.2e}"
                         f" (bf16 SDPA {ys['max_rel']:.2e}); vs original K/V rel-L2: FlagRot {e0['rel_l2']:.3e}, no rot {e2['rel_l2']:.3e}")
            print(lines[-1], flush=True)
    summ = []
    for fmt in A.FMTS:
        ks = [r for r in rows if r.get("fmt") == fmt and "max_rel" in r]
        summ.append(dict(fmt=fmt, cases=len(ks), max_abs=max(r["max_abs"] for r in ks),
                         max_rel=max(r["max_rel"] for r in ks), rel_l2=max(r["rel_l2"] for r in ks),
                         sdpa_bf16_max_rel=max(worst[(fmt, "sdpa_bf16")]),
                         passed=all(r["max_rel"] <= max(2 * r["sdpa_bf16"]["max_rel"], 4e-3) for r in ks)))
    lines.append("")
    lines.append("| fmt | cases | max abs err | max abs err / max abs ref | max rel-L2 | bf16 SDPA max abs err / max abs ref | pass (<= max(2x SDPA, 4e-3)) |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in summ:
        lines.append(f"| {s['fmt']} | {s['cases']} | {s['max_abs']:.2e} | {s['max_rel']:.2e} | {s['rel_l2']:.2e} | "
                     f"{s['sdpa_bf16_max_rel']:.2e} | {s['passed']} |")
    txt = "\n".join(lines)
    print(txt)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    if not quick:
        open(os.path.join(HERE, "results", "test_attn_log.txt"), "w").write(txt + "\n")
        json.dump(dict(rows=rows, summary=summ), open(os.path.join(HERE, "results", "test_attn.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    run(ap.parse_args().quick)
