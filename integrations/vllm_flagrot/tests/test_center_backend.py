"""int2c: backend-level checks of the key centering (FlagRotAttentionImpl without an engine; vLLM env).

  CUDA_VISIBLE_DEVICES=2 python integrations/vllm_flagrot/tests/test_center_backend.py \
      [--models llama8b,qwen8b] [--modes konlya2,rtn_a2,static_a2,konlya_c2] [--out f.json]

Per (model, mode, layer 0 / middle / last) with the real calibration and centering files: keys = real-scale rows
(the layer's mu + Gaussian noise of the same scale), pages written through the impl's own write path (_store with
center=True, _write with the per-layer writer vector) and read by
  * the "fa" prefill (dequantize + BF16 FlashAttention) and the INT2-direct prefix-cached prefill (prefill_mq),
    both vs float64 attention over the dequantized centered pages (original basis);
  * shift invariance in float64: attention over (dequantized + mu) == over the dequantized centered pages;
  * consistency of the BF16-slot rows: _unstore(_center_rows(_store(kv))) == _center_key(k) up to bf16 rounding;
  * quantization error vs the exact attention over the raw keys (float64), centered and uncentered cache.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "scripts"))
sys.path.insert(0, str(HERE.parents[0]))
import bench_mq_prefill as BM  # noqa: E402
from common import FLAGROT, KV_DYN, QUANT, calib_path, center_path  # noqa: E402


def make_impl(model, mode, center, li_pick=0):
    os.environ["FLAGROT_KV"] = FLAGROT[mode]
    os.environ["FLAGROT_CALIB"] = calib_path(model, mode, center)
    os.environ["FLAGROT_QUANT"] = QUANT.get(mode, "int4")
    os.environ["FLAGROT_PREFILL_MQ"] = "all"
    if center:
        os.environ["FLAGROT_CENTER"] = center_path(model)
    else:
        os.environ.pop("FLAGROT_CENTER", None)
    if mode in KV_DYN:
        os.environ["FLAGROT_KV_DYN"] = KV_DYN[mode]
    else:
        os.environ.pop("FLAGROT_KV_DYN", None)
    from vllm_flagrot.attention import backend as B
    B._LAST.clear()
    meta = B.load_calib(os.environ["FLAGROT_CALIB"])["meta"]
    Hkv = meta["num_kv_heads"]
    Hq = meta.get("num_heads", 4 * Hkv)
    D = meta.get("head_dim", 128)
    layers = meta.get("layers") or list(range(meta["num_layers"]))
    li = layers[li_pick]
    impl = B.FlagRotAttentionImpl(Hq, D, 1 / math.sqrt(D), num_kv_heads=Hkv)
    assert impl.center == center, (impl.center, center)
    layer = SimpleNamespace(layer_name=f"model.layers.{li}.self_attn.attn")
    st = impl._state(layer, torch.device("cuda"))
    return B, impl, st, li, D


def run(model, mode, li_pick, seed=0):
    dev = torch.device("cuda")
    out = {}
    B, implc, stc, li, D = make_impl(model, mode, True, li_pick)
    BM.D = D
    mu = implc._kmu_orig(stc)                                                       # [Hkv, D] fp32 (device)
    Hkv, Hq = implc.num_kv_heads, implc.num_heads
    g = torch.Generator().manual_seed(seed)
    Ls, nqs = [3000, 517], [300, 517]
    sc = float(mu.abs().mean()) + 0.5
    kvs = []
    for L in Ls:
        kv = torch.randn(L, 2 * Hkv, D, generator=g).to(dev)
        kv[:, :Hkv] = kv[:, :Hkv] * sc + mu
        kv[:, :Hkv, :4] *= 1.5
        kvs.append(kv.to(torch.bfloat16))
    n = sum(nqs)
    q = (torch.randn(n, Hq, D, generator=g) * 2).to(dev).to(torch.bfloat16)
    qsl = [0]
    for nq in nqs:
        qsl.append(qsl[-1] + nq)
    res = {}
    for tag in ("centered", "uncentered"):
        if tag == "uncentered":
            B, impl, st, li2, _ = make_impl(model, mode, False, li_pick)
            assert li2 == li
        else:
            impl, st = implc, stc
        CBY = 80 if impl.kind in B.INT2_KINDS else 144
        BM.CBY = CBY
        pool, bt, slots = BM.make_pool(Ls, Hkv, dev, seed=seed)
        for i, kv in enumerate(kvs):
            impl._write(st, impl._store(st, kv, center=impl.center), slots[i], pool)
        md = SimpleNamespace(num_decodes=0, num_reqs=len(Ls), query_start_loc_cpu=torch.tensor(qsl, dtype=torch.int32),
                             seq_lens_cpu=torch.tensor(Ls, dtype=torch.int32), block_table=bt, req_idx=None)
        o_fa = BM.run_path(impl, st, q, pool, md, "fa").float()
        o_mq = BM.run_path(impl, st, q, pool, BM.fresh_md(md), "mq").float() if impl.mq else None
        c, rows = 0, []
        for r, (L, nq) in enumerate(zip(Ls, nqs)):
            k, v = impl._cached_kv_impl(st, pool, md, r, L, dtype=torch.float32)       # original basis (centered)
            ro, _ = BM.ref_attention(q[c:c + nq], k, v, L - nq, 0, impl.scale)
            ro = ro.reshape(nq, -1)
            raw = kvs[r].double()
            rx, _ = BM.ref_attention(q[c:c + nq], raw[:, :Hkv], raw[:, Hkv:], L - nq, 0, impl.scale)   # exact, no quant
            rx = rx.reshape(nq, -1)
            rec = dict(L=L, nq=nq, fa_vs_f64=BM.rel(o_fa[c:c + nq], ro),
                       quant_err_vs_exact=BM.rel(ro, rx))
            if o_mq is not None:
                rec.update(mq_vs_f64=BM.rel(o_mq[c:c + nq], ro), mq_vs_fa=BM.rel(o_mq[c:c + nq], o_fa[c:c + nq]))
            if tag == "centered":
                ru, _ = BM.ref_attention(q[c:c + nq], k.double() + mu.double(), v, L - nq, 0, impl.scale)
                rec["shift_invariance_f64"] = BM.rel(ru.reshape(nq, -1), ro)
            rows.append(rec)
            c += nq
        res[tag] = rows
    # BF16-slot rows vs the centered chunk keys (the two ways the attention sees BF16 keys)
    kv = kvs[0][:256]
    a = implc._unstore(stc, implc._center_rows(stc, implc._store(stc, kv, center=True)))[:, :Hkv].float()
    b = implc._center_key(stc, kv[:, :Hkv].reshape(256, -1), 256).reshape(256, Hkv, D).float()
    res["slot_vs_key_rel"] = ((a - b).norm() / b.norm()).item()
    res["mu_norm"] = float(mu.norm())
    return dict(model=model, mode=mode, layer=li, **res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama8b,qwen8b")
    ap.add_argument("--modes", default="konlya2,rtn_a2,static_a2,konlya_c2")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out, ok = [], True
    for model in a.models.split(","):
        for mode in a.modes.split(","):
            if not Path(calib_path(model, mode, True)).exists():
                print(f"skip {model} {mode}: no centered calibration {calib_path(model, mode, True)}")
                continue
            for lp in (0, -1):
                r = run(model, mode, lp)
                out.append(r)
                c, u = r["centered"], r["uncentered"]
                good = (all(x["fa_vs_f64"] < 1e-2 and x.get("mq_vs_f64", 0) < 5e-3 and x["shift_invariance_f64"] < 1e-9
                            for x in c) and r["slot_vs_key_rel"] < 4e-3)
                ok &= good
                print(f"{'ok ' if good else 'FAIL'} {model} {mode} layer {r['layer']} |mu| {r['mu_norm']:.1f}: "
                      + "; ".join(f"L{x['L']}: fa {x['fa_vs_f64']:.1e} mq {x.get('mq_vs_f64', float('nan')):.1e} "
                                  f"shift {x['shift_invariance_f64']:.0e} quant err {x['quant_err_vs_exact']:.3f} "
                                  f"(uncentered {y['quant_err_vs_exact']:.3f})" for x, y in zip(c, u))
                      + f"; slot vs key {r['slot_vs_key_rel']:.1e}", flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    print("ALL OK" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
