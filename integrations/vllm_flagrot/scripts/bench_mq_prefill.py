"""Phase 2i: correctness + speed of the INT4-direct prefix-cached prefill (vllm_flagrot/prefill_mq.py,
csrc/mq_prefill.cu) against float64 references and against the "fa" path (PyTorch gather + dequant + BF16 FA).

  python bench_mq_prefill.py --check [--D 256] [--out f.json]
      kernel level: synthetic multi-request paged caches written by the plugin's writers (rtn, static block + whole-
      head flags on K and V, konlya per-token flags), shuffled block tables with poisoned unused entries, garbage free
      pages, partial pages / blocks, W (shift) and sink offsets, several requests per launch, G 4 / 5 / 8; float64
      reference on the dequantized pages (exact INT4 values) and the BF16 FA path for scale.  --D 256 (Phase 2j):
      head size 256, 4 KV heads, the same 18 request configurations with GQA 4 -> 6 (Qwen3.8-27B) + G 4 / 12.
  python bench_mq_prefill.py --backend --model qwen8b [--out f.json]
      backend level (W = S = 0): FlagRotAttentionImpl._prefill_fa vs ._prefill_mq on the same pages written by the
      backend's own writer, every mode (rtn_a, static_a, konlya, konlya_c, real calibrations): output difference and
      both against float64.
  python bench_mq_prefill.py --bench --model qwen8b [--out f.json]
      per-layer prefill attention time (CUDA events, median) of the two backend paths over a grid of (new segment,
      cached prefix) and for batches of several short requests (one launch for all of them vs the per-request loop).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vllm_flagrot import kernels as K  # noqa: E402
from vllm_flagrot import prefill_mq as MQ  # noqa: E402

D, PAGE = 128, 16          # D: head size of the checks / benchmarks (--D; 256 = Phase 2j)
INT2 = False               # --int2 (Phase 3a): INT2 pages (static int2a, konlya2) in the kernel-level check
SFMT, CBK, CBY = "int4a", "konlya", 144                   # static format, key-only kind, bytes per token and 128 dims
LOG2E = 1.4426950408889634


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def hpm(n, dev, dtype=torch.float64):
    return K.hadamard(n, dev, dtype) * math.sqrt(n)


def make_pool(Ls, Hkv, dev, seed=0, shuffle=True):
    """pages for sequences of lengths Ls: block table [R, maxpg] int32 (unused entries poisoned with 2^30: an
    over-read faults), pool with garbage-filled free pages; returns (pool, bt, slots per sequence)."""
    g = torch.Generator().manual_seed(seed)
    npg = [(L + PAGE - 1) // PAGE for L in Ls]
    total = sum(npg) + 11
    perm = torch.randperm(total, generator=g) if shuffle else torch.arange(total)
    maxpg = max(npg) + 3
    bt = torch.full((len(Ls), maxpg), 1 << 30, dtype=torch.int32)
    o = 0
    for i, n in enumerate(npg):
        bt[i, :n] = perm[o:o + n].to(torch.int32)
        o += n
    pool = torch.randint(0, 256, (total, Hkv, CBY * PAGE * D // 128), generator=g, dtype=torch.uint8).to(dev)
    bt = bt.to(dev)
    slots = [bt[i].long()[torch.arange(L, device=dev) // PAGE] * PAGE + torch.arange(L, device=dev) % PAGE
             for i, L in enumerate(Ls)]
    return pool, bt, slots


def rand_kv(L, Hkv, g, dev):
    k = torch.randn(L, Hkv, D, generator=g)
    v = torch.randn(L, Hkv, D, generator=g)
    k[..., 5] *= 8
    k[..., 70] += 4
    v[..., 3] += 2
    if D > 128:
        k[..., 200] *= 6
        v[..., 131] += 2
    return torch.cat([k, v], 1).to(dev)                   # [L, 2 Hkv, D] fp32, K rows first


def ref_attention(q, kk, vv, p0, shift, scale):
    """float64: q [nq, Hq, D] at positions p0 ..; keys kk / vv [nk, Hkv, D]; key j visible iff j <= p - shift."""
    nq, Hq, _ = q.shape
    nk, Hkv, _ = kk.shape
    G = Hq // Hkv
    kk = kk.double().repeat_interleave(G, 1)
    vv = vv.double().repeat_interleave(G, 1)
    s = torch.einsum("qhd,khd->hqk", q.double(), kk) * scale
    pos = torch.arange(p0, p0 + nq, device=q.device)
    ok = torch.arange(nk, device=q.device)[None, :] <= (pos[:, None] - shift)
    s = s.masked_fill(~ok[None], -math.inf)
    lse = torch.logsumexp(s, -1)
    p = torch.exp(s - lse[..., None]).nan_to_num(0.0)
    return torch.einsum("hqk,khd->qhd", p, vv), lse


# --------------------------------------------------------------------------------------------- kernel-level check
CASES = [  # (requests [(L, nq)], shift W, sink S, G, shuffle)
    ([(1000, 1000)], 0, 0, 4, True),
    ([(777, 300)], 0, 0, 4, True),
    ([(10145, 145)], 0, 0, 4, True),
    ([(4100, 517)], 0, 0, 4, False),
    ([(3000, 1)], 0, 0, 4, True),
    ([(17, 17)], 0, 0, 4, True),
    ([(16, 5)], 0, 0, 4, True),
    ([(64, 64)], 0, 0, 4, True),
    ([(9000, 145), (12000, 300), (5003, 17), (700, 700), (20000, 64)], 0, 0, 4, True),
    ([(2048, 2048)], 128, 0, 4, True),
    ([(5000, 2500)], 256, 0, 4, True),
    ([(130, 70)], 64, 0, 4, True),
    ([(8191, 300), (3000, 3000)], 128, 4, 4, True),
    ([(140, 140)], 128, 4, 4, True),
    ([(33000, 1200)], 0, 0, 4, True),
    ([(6000, 200), (300, 300)], 0, 0, 5, True),
    ([(6000, 200), (300, 300)], 0, 0, 8, True),
    ([(6000, 200)], 0, 0, 1, True),
]


# head 256 (Phase 2j): the same configurations with GQA 4 -> 6 (Qwen3.8-27B), plus G 4 and G 12
CASES256 = [(r, W, S, 6 if G == 4 else G, sh) for (r, W, S, G, sh) in CASES] + [
    ([(6000, 200), (300, 300)], 0, 0, 4, True),
    ([(9000, 145), (700, 300)], 0, 0, 12, True),
]


def check(a):
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    from vllm_flagrot.attention.backend import _Static
    dev = "cuda"
    Hkv = 8 if D == 128 else 4
    ng = D // 32
    scale = 1 / math.sqrt(D)
    out = []
    g = torch.Generator().manual_seed(1)
    CT = None
    calp = os.environ.get("FLAGROT_CB_CALIB")
    kernels = KERNELS if D == 128 and not INT2 else (3,)
    if calp and Path(calp).exists():
        cal = torch.load(calp, map_location="cpu", weights_only=True)
        CT = cal["cq"][0][:Hkv].float().transpose(1, 2).contiguous().to(dev)
        co = cal["co"][0][:Hkv].float().contiguous().to(dev)
    else:                                                   # synthetic SPD C_q (any SPD weighting is a valid writer)
        A = torch.randn(Hkv, D, D, generator=g) / math.sqrt(D)
        CT = (A @ A.transpose(1, 2) + 0.1 * torch.eye(D)).contiguous().to(dev)
        co = torch.ones(Hkv, D).to(dev)
    for ci, (reqs, W, S, G, shuffle) in enumerate(CASES if D == 128 else CASES256):
        Hq = Hkv * G
        for mname in ("rtn", "flags", "konlya"):
            gg = torch.Generator().manual_seed(100 * ci + len(mname))
            Ls = [L for L, _ in reqs]
            pool, bt, slots = make_pool([max(L - W, 1) for L in Ls], Hkv, dev, seed=ci, shuffle=shuffle)
            st = None
            if mname == "flags":
                cal = dict(kmask=(torch.rand(1, Hkv, ng, generator=gg) < 0.5), vmask=(torch.rand(1, Hkv, ng, generator=gg) < 0.5),
                           kfull=(torch.rand(1, Hkv, generator=gg) < 0.4), vfull=(torch.rand(1, Hkv, generator=gg) < 0.4))
                st = _Static(cal, 0, torch.arange(Hkv), dev, G, D)
            refs, qs, mqreqs, row = [], [], [], 0
            for i, (L, nq) in enumerate(reqs):
                npage = max(L - W, 0)                       # Phase 2k: pages hold positions [0, L - W) at their real
                                                            # slots; the kernel masks the S sink positions (kmin)
                kv = rand_kv(max(npage, 1), Hkv, gg, dev)
                if mname == "flags":
                    kvs = _Static.apply(kv.to(torch.bfloat16), st.m_kv)
                    K.v2_write(kvs, slots[i], st.wmask, 32, pool, PAGE, fmt=SFMT)
                elif mname == "konlya":
                    K.cb_write(kv.to(torch.bfloat16), slots[i], CT, co, CBK, pool, PAGE)
                else:
                    K.v2_write(kv.to(torch.bfloat16), slots[i], torch.zeros(2 * Hkv, ng, dtype=torch.uint8, device=dev),
                               32, pool, PAGE, fmt=SFMT)
                npg = (max(npage, 1) + PAGE - 1) // PAGE
                ch = pool.index_select(0, bt[i, :npg].long())
                if mname == "konlya":
                    kk, vv = K.dequant_cb_pages(ch, PAGE, CBK, torch.float64, Dh=D)
                else:
                    kk, vv = K.dequant_v2_pages(ch, PAGE, torch.float64, fmt=SFMT, Dh=D)
                kk, vv = kk.reshape(-1, Hkv, D)[S:npage], vv.reshape(-1, Hkv, D)[S:npage]   # visible: [S, L - W)
                if mname == "flags":                        # stored basis -> original: x = H (B x_s) / (D / 16)
                    h32 = K.hadamard(32, dev, torch.float64)
                    for nm in ("k", "v"):
                        x = kk if nm == "k" else vv
                        msk = (st.kmask if nm == "k" else st.vmask).bool()
                        full = (st.kfull if nm == "k" else st.vfull).bool()
                        xb = x.reshape(-1, Hkv, ng, 32)
                        x = torch.where(msk[None, :, :, None], xb @ h32, xb).reshape(-1, Hkv, D)
                        x = torch.where(full[None, :, None], x @ (hpm(D, dev) * (16 / D)), x)
                        if nm == "k":
                            kk = x
                        else:
                            vv = x
                q = (torch.randn(nq, Hq, D, generator=gg) * 2)
                q[..., 5] *= 4
                q = q.to(dev).to(torch.bfloat16)
                s = L - nq
                ro, rl = ref_attention(q, kk, vv, s - S, W, scale)
                refs.append((ro, rl, kk, vv, s))
                qs.append(q)
                mqreqs.append((i, row, row + nq, L, s))
                row += nq
            qall = torch.cat(qs)
            for kern in kernels:
                plan = MQ.build_plan(mqreqs, G, Hkv, W, dev, sink=S, kernel=kern, D=D)
                kf = st.kfull_u8 if (st is not None and st.any_kfull) else None
                km = st.kmask if (st is not None and st.any_block[0]) else None
                q16 = MQ.prep_q(qall, scale * LOG2E, G, idx=plan.idx, kfull=kf, kmask=km)
                opart, ml = MQ.attn(q16, pool, bt, plan, G, W, konly=mname == "konlya", int2=INT2)
                o, lse = MQ.merge(plan, opart, ml, G, fp32=True, want_lse=True)
                if st is not None:
                    o = st.o_stored_to_orig(o)
                torch.cuda.synchronize()
                ob = None
                if mname == "rtn" and W == 0 and S == 0:     # the direct bf16 merge into a layer-output view
                    ob = torch.zeros(row + 3, Hq * D, dtype=torch.bfloat16, device=dev)
                    MQ.merge(plan, opart, ml, G, out=ob)
                c = 0
                for (L, nq), (ro, rl, kk, vv, s) in zip(reqs, refs):
                    oo, ll = o[c:c + nq], lse[:, c:c + nq]
                    fin = torch.isfinite(rl)
                    rec = dict(case=ci, D=D, kernel=kern, mode=mname, G=G, W=W, S=S, L=L, nq=nq, nreq=len(reqs),
                               rel_l2_vs_f64=rel(oo, ro) if fin.any() else 0.0,
                               max_err_over_max=((oo.double() - ro).abs().max() / ro.abs().max().clamp_min(1e-30)).item(),
                               lse_max_abs=(ll.double()[fin] - rl[fin]).abs().max().item() if fin.any() else 0.0,
                               lse_empty_ok=bool(torch.isneginf(ll[~fin]).all()) if (~fin).any() else True)
                    if ob is not None:
                        rec["direct_bf16_vs_fp32"] = rel(ob[c:c + nq].view(nq, Hq, D), oo)
                    nk = kk.shape[0]
                    if W == 0 and S == 0 and nk > 0:            # BF16 FA on the dequantized pages (= the fa path), for scale
                        qq = qall[c:c + nq]
                        cq = torch.tensor([0, nq], dtype=torch.int32, device=dev)
                        ck = torch.tensor([0, nk], dtype=torch.int32, device=dev)
                        of = flash_attn_varlen_func(qq, kk.to(torch.bfloat16).contiguous(),
                                                    vv.to(torch.bfloat16).contiguous(), nq, cq, nk, ck,
                                                    softmax_scale=scale, causal=True, fa_version=3)
                        rec["bf16_fa_rel_l2_vs_f64"] = rel(of, ro)
                    out.append(rec)
                    print(json.dumps(rec), flush=True)
                    c += nq
    worst = max(r["rel_l2_vs_f64"] for r in out)
    worst_fa = max((r.get("bf16_fa_rel_l2_vs_f64", 0) for r in out))
    summ = dict(cases=len(out), worst_rel_l2_vs_f64=worst, worst_bf16_fa_rel_l2_vs_f64=worst_fa,
                worst_lse_abs=max(r["lse_max_abs"] for r in out), all_empty_rows_ok=all(r["lse_empty_ok"] for r in out))
    print(json.dumps(summ))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(dict(summary=summ, cases=out), indent=1))


# --------------------------------------------------------------------------------------------- backend level
MODES = ("rtn_a", "static_a", "konlya", "konlya_c")
KERNELS = tuple(int(k) for k in os.environ.get("FLAGROT_MQ_CHECK_KERNELS", "1,2,3").split(","))


def make_impl(model, mode, prefill_mq="auto"):
    """a FlagRotAttentionImpl for layer 0 of `model` (no vLLM config: options from the environment, as in a worker
    before the config is set) + its per-layer state."""
    from common import FLAGROT, KV_DYN, QUANT, calib_path
    os.environ["FLAGROT_KV"] = FLAGROT[mode]
    os.environ["FLAGROT_CALIB"] = calib_path(model, mode)
    os.environ["FLAGROT_QUANT"] = QUANT.get(mode, "int4")
    os.environ["FLAGROT_PREFILL_MQ"] = prefill_mq
    if mode in KV_DYN:
        os.environ["FLAGROT_KV_DYN"] = KV_DYN[mode]
    else:
        os.environ.pop("FLAGROT_KV_DYN", None)
    from vllm_flagrot.attention import backend as B
    global D
    B._LAST.clear()
    meta = B.load_calib(os.environ["FLAGROT_CALIB"])["meta"]
    Hkv = meta["num_kv_heads"]
    Hq = meta.get("num_heads", 4 * Hkv)
    D = meta.get("head_dim", 128)                           # Qwen3.8-27B: 256 (Phase 2j)
    li = (meta.get("layers") or [0])[0]                     # hybrid models: the first full-attention layer
    impl = B.FlagRotAttentionImpl(Hq, D, 1 / math.sqrt(D), num_kv_heads=Hkv)
    layer = SimpleNamespace(layer_name=f"model.layers.{li}.self_attn.attn")
    st = impl._state(layer, torch.device("cuda"))
    return B, impl, st


def backend_setup(B, impl, st, Ls, nqs, seed=0):
    """pages of len(Ls) requests written through the impl's own write path; q rows of the new segments."""
    dev = torch.device("cuda")
    Hkv, Hq = impl.num_kv_heads, impl.num_heads
    global CBY
    CBY = 80 if impl.kind in B.INT2_KINDS else 144              # Phase 3a: INT2 pages
    pool, bt, slots = make_pool(Ls, Hkv, dev, seed=seed)
    g = torch.Generator().manual_seed(seed + 7)
    for i, L in enumerate(Ls):
        for c0 in range(0, L, 16384):                       # writer calls of <= 16k tokens (chunked prefill)
            c1 = min(L, c0 + 16384)
            kv = rand_kv(c1 - c0, Hkv, g, dev).to(torch.bfloat16)
            impl._write(st, impl._store(st, kv), slots[i][c0:c1], pool)
    n = sum(nqs)
    q = (torch.randn(n, Hq, D, generator=g) * 2).to(dev).to(torch.bfloat16)
    qsl = [0]
    for nq in nqs:
        qsl.append(qsl[-1] + nq)
    md = SimpleNamespace(num_decodes=0, num_reqs=len(Ls), query_start_loc_cpu=torch.tensor(qsl, dtype=torch.int32),
                         seq_lens_cpu=torch.tensor(Ls, dtype=torch.int32), block_table=bt, req_idx=None)
    return pool, md, q


def fresh_md(md):
    return SimpleNamespace(**{k: v for k, v in vars(md).items() if k != "_flagrot_mq"})


def run_path(impl, st, q, pool, md, path):
    out = torch.zeros(q.shape[0], impl.num_heads * D, dtype=torch.bfloat16, device=q.device)
    if path == "fa":
        impl._prefill_fa(st, q, None, None, pool, md, out)
    else:
        rest = impl._prefill_mq(st, q, None, None, pool, md, out)
        assert rest == [], rest
    return out


def backend_check(a):
    res = []
    for mode in a.modes.split(","):
        B, impl, st = make_impl(a.model, mode, "all")
        for (Ls, nqs) in (([10145], [145]), ([3000], [3000]), ([20000, 9000, 517], [64, 300, 517])):
            pool, md, q = backend_setup(B, impl, st, Ls, nqs, seed=len(Ls))
            o_fa = run_path(impl, st, q, pool, md, "fa").float()
            o_mq = run_path(impl, st, q, pool, fresh_md(md), "mq").float()
            # float64 reference on the same pages: the fa path's own dequantization in float64 (no BF16 rounding)
            c, errs = 0, []
            for r, (L, nq) in enumerate(zip(Ls, nqs)):
                k, v = impl._cached_kv_impl(st, pool, md, r, L, dtype=torch.float32)
                ro, _ = ref_attention(q[c:c + nq], k, v, L - nq, 0, impl.scale)
                ro = ro.reshape(nq, -1)
                errs.append(dict(L=L, nq=nq, fa_vs_f64=rel(o_fa[c:c + nq], ro), mq_vs_f64=rel(o_mq[c:c + nq], ro),
                                 mq_vs_fa=rel(o_mq[c:c + nq], o_fa[c:c + nq])))
                c += nq
            rec = dict(model=a.model, mode=mode, reqs=errs)
            res.append(rec)
            print(json.dumps(rec), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


def timed(fn, iters):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ts.sort()
    return ts[len(ts) // 2]


def bench(a):
    res = []
    grid_new = [int(x) for x in a.new.split(",")]
    grid_pre = [int(x) for x in a.prefix.split(",")]
    for mode in a.modes.split(","):
        B, impl, st = make_impl(a.model, mode, "all")
        for pre in grid_pre:
            for nq in grid_new:
                pool, md, q = backend_setup(B, impl, st, [pre + nq], [nq], seed=pre + nq)
                t_fa = timed(lambda: run_path(impl, st, q, pool, md, "fa"), a.iters)
                md_c = fresh_md(md)
                run_path(impl, st, q, pool, md_c, "mq")             # plan cached on md_c (as for layers 2 .. N)
                t_mq = timed(lambda: run_path(impl, st, q, pool, md_c, "mq"), a.iters)
                t_mq_plan = timed(lambda: run_path(impl, st, q, pool, fresh_md(md), "mq"), a.iters)
                rec = dict(model=a.model, mode=mode, prefix=pre, new=nq, reqs=1, fa_ms=t_fa, mq_ms=t_mq,
                           mq_with_plan_ms=t_mq_plan, speedup=t_fa / t_mq)
                res.append(rec)
                print(json.dumps(rec), flush=True)
                del pool
        # batches of short requests (tau2-like steps): one launch for all of them vs the per-request loop
        for nb in (2, 4, 8):
            Ls = [a.batch_prefix + 1000 * i + 145 for i in range(nb)]
            pool, md, q = backend_setup(B, impl, st, Ls, [145] * nb, seed=nb)
            t_fa = timed(lambda: run_path(impl, st, q, pool, md, "fa"), a.iters)
            md_c = fresh_md(md)
            run_path(impl, st, q, pool, md_c, "mq")
            t_mq = timed(lambda: run_path(impl, st, q, pool, md_c, "mq"), a.iters)
            rec = dict(model=a.model, mode=mode, prefix=a.batch_prefix, new=145, reqs=nb, fa_ms=t_fa, mq_ms=t_mq,
                       speedup=t_fa / t_mq)
            res.append(rec)
            print(json.dumps(rec), flush=True)
        # host cost of the per-step plan (built once per step, shared by all layers)
        Ls = [a.batch_prefix + 145] * 4
        _, md, _ = backend_setup(B, impl, st, Ls, [145] * 4, seed=3)
        t0 = time.perf_counter()
        for _ in range(50):
            impl._mq_select(fresh_md(md), q.device)
        torch.cuda.synchronize()
        res.append(dict(model=a.model, mode=mode, plan_build_ms=(time.perf_counter() - t0) / 50 * 1e3))
        print(json.dumps(res[-1]), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(dict(device=torch.cuda.get_device_name(), rows=res), indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--backend", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--model", default="qwen8b")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--new", default="16,64,145,256,512,1024,2048,4096")
    ap.add_argument("--prefix", default="2048,8192,16384,32768")
    ap.add_argument("--batch_prefix", type=int, default=10000)
    ap.add_argument("--iters", type=int, default=9)
    ap.add_argument("--out", default="")
    ap.add_argument("--D", type=int, default=128, help="head size of --check (128 | 256)")
    ap.add_argument("--int2", action="store_true", help="--check on INT2 pages (Phase 3a: int2a / konlya2)")
    a = ap.parse_args()
    D = a.D
    if a.int2:
        INT2, SFMT, CBK, CBY = True, "int2a", "konlya2", 80
    if a.check:
        check(a)
    if a.backend:
        backend_check(a)
    if a.bench:
        bench(a)
