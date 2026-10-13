"""End-to-end decode throughput of vLLM with each KV cache (Python API, offline engine, CUDA graphs on).

Per (KV cache, context length) one engine (max_model_len = ctx + 1024, max_num_seqs = 64, gpu_memory_utilization
0.90, chunked prefill 16384 tokens, prefix caching off).  B_max = min(64, sequences that fit) requests with distinct
random prompts of exactly ctx tokens (greedy, ignore_eos, no detokenization) are submitted at once and the engine is
stepped by hand:
  1. prefill phase until every request has its first token (prompt tok/s reported);
  2. decode phases at batch sizes B in {64, 32, 8, 1} (those <= B_max): `steps` pure-decode engine steps each, then
     requests are aborted to reach the next batch size (exact phases, no prefill in between).
TPOT = wall time / tokens generated per request over the timed span of a phase (after 3 warm-up tokens);
output tok/s = B / TPOT.
KV capacity = vLLM's num_gpu_blocks x block size (tokens); max concurrent sequences = capacity // (ctx + 1024).
--cap_vllm additionally caps it by vLLM's own kv_cache_max_concurrency (group-aware on hybrid models; used for the
capacity / latency curves), and a phase that makes no progress for --stall_s seconds is recorded as stalled.

  CUDA_VISIBLE_DEVICES=0 python integrations/vllm_flagrot/scripts/bench_e2e.py --model qwen8b --mode kivi2p8ff:s4w128 \
      --ctx 32768 --batches 256 128 64 32 16 8 4 1 --max_seqs 256 --cap_vllm --out results/bench/qwen8b_kivi2p8ff.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # engine core in-process: step() = one engine step

from common import make_llm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="llama8b")
ap.add_argument("--mode", default="", help="mode string as in serve_mode.sh / launch.py (e.g. konlya2:s4w128:ca, "
                "kivi2p8ff:s4w128 or an alias); overrides --kv / --window / --sink")
ap.add_argument("--kv", default="")
ap.add_argument("--window", type=int, default=0)
ap.add_argument("--sink", type=int, default=0)
ap.add_argument("--ctx", type=int, nargs="+", default=[8192, 16384, 32768])
ap.add_argument("--batches", type=int, nargs="+", default=[64, 32, 8, 1])
ap.add_argument("--steps", type=int, default=48)
ap.add_argument("--margin", type=int, default=1024)
ap.add_argument("--gpu_mem", type=float, default=0.90)
ap.add_argument("--chunk", type=int, default=16384)
ap.add_argument("--eager", action="store_true")
ap.add_argument("--tp", type=int, default=1, help="tensor parallel size")
ap.add_argument("--prefill_only", action="store_true", help="batch prefill phase only (no decode phases)")
ap.add_argument("--prefill_fp8", default=None, choices=["had", "unit", "amax", "off", "emu_exact", "emu_ac", "emu_acv", "int4", "int4ac"],
                help="FlagRot fa prefill attention: FP8 (K/V scale 1 | per-head amax) or BF16; default = plugin default")
ap.add_argument("--max_seqs", type=int, default=0, help="engine max_num_seqs (default: the largest batch)")
ap.add_argument("--cap_vllm", action="store_true", help="cap the batch by vLLM's kv_cache_max_concurrency")
ap.add_argument("--stall_s", type=float, default=0, help="end a phase without progress after this many seconds (0: off)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
CENTER = False
if a.mode:
    from common import parse_mode, resolve_alias, split_center
    a.mode = resolve_alias(a.mode)
    _, CENTER = split_center(a.mode)
    a.kv, a.sink, a.window = parse_mode(a.mode)
assert a.kv, "--mode or --kv"

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vllm import SamplingParams  # noqa: E402

results = []
outp = Path(a.out)
outp.parent.mkdir(parents=True, exist_ok=True)
if outp.exists():
    results = json.loads(outp.read_text()).get("runs", [])
for ctx in a.ctx:
    if any(r["ctx"] == ctx for r in results):
        continue
    bmax_cfg = a.max_seqs or max(a.batches)
    llm = make_llm(a.model, a.kv, max_model_len=ctx + a.margin, max_num_batched_tokens=a.chunk,
                   gpu_memory_utilization=a.gpu_mem, window=a.window, sink=a.sink, enforce_eager=a.eager, max_num_seqs=bmax_cfg,
                   prefill_fp8=a.prefill_fp8, tp=a.tp, center=CENTER)
    eng = llm.llm_engine
    cc = eng.vllm_config.cache_config
    cap_tokens = int(cc.num_gpu_blocks) * int(cc.block_size)
    fit = cap_tokens // (ctx + a.margin)
    if a.cap_vllm and getattr(cc, "kv_cache_max_concurrency", None):
        fit = min(fit, int(cc.kv_cache_max_concurrency))
    bmax = min(bmax_cfg, fit)
    rec = dict(kv=a.kv, window=a.window, sink=a.sink, tp=a.tp, ctx=ctx, kv_capacity_tokens=cap_tokens, num_gpu_blocks=int(cc.num_gpu_blocks),
               max_concurrent_seqs=fit, b_run=bmax, phases={},
               vllm_kv_cache_size_tokens=getattr(cc, "kv_cache_size_tokens", None),
               vllm_max_concurrency=getattr(cc, "kv_cache_max_concurrency", None))
    print(json.dumps({k: v for k, v in rec.items() if k != "phases"}), flush=True)
    if bmax < 1:
        results.append(rec)
        continue
    rng = np.random.default_rng(ctx)
    vocab = 128000
    sp = SamplingParams(temperature=0, max_tokens=a.margin - 16, ignore_eos=True, detokenize=False)
    ids = [str(i) for i in range(bmax)]
    for i in ids:
        eng.add_request(i, {"prompt_token_ids": rng.integers(1000, vocab, ctx).tolist()}, sp)
    first = set()
    ntok = {i: 0 for i in ids}
    t0 = time.perf_counter()
    while len(first) < bmax:
        for o in eng.step():
            if o.outputs and len(o.outputs[0].token_ids) > 0:
                first.add(o.request_id)
                ntok[o.request_id] = len(o.outputs[0].token_ids)
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0
    rec["prefill_seconds"] = round(t_prefill, 2)
    rec["prefill_tok_s"] = round(bmax * ctx / t_prefill, 1)
    rec["tokens_during_prefill_max"] = max(ntok.values())
    print(json.dumps(dict(ctx=ctx, B=bmax, prefill_seconds=rec["prefill_seconds"], prefill_tok_s=rec["prefill_tok_s"])),
          flush=True)
    live = list(ids)
    phases = [b for b in a.batches if b <= bmax]
    if bmax not in phases:
        phases = [bmax] + phases
    phases = [] if a.prefill_only else sorted(set(phases), reverse=True)
    class Stall(Exception):
        pass
    last = [0, time.perf_counter()]

    def progress(live):
        if a.stall_s and time.perf_counter() - last[1] > a.stall_s:
            raise Stall()
        for o in eng.step():
            if o.request_id in ntok and o.outputs:
                ntok[o.request_id] = len(o.outputs[0].token_ids)
        m = min(ntok[i] for i in live)
        if m != last[0]:
            last[0], last[1] = m, time.perf_counter()
        return m

    for b in phases:
        drop = live[b:]
        if drop:
            eng.abort_request(drop)
            live = live[:b]
            for i in drop:
                ntok.pop(i, None)
        # warm-up: 3 tokens per live request, then time `steps` tokens (every live request decodes once per step)
        last[0], last[1] = -1, time.perf_counter()
        try:
            p0 = progress(live)
            while progress(live) < p0 + 3:
                pass
            p1 = min(ntok[i] for i in live)
            stamps = [(p1, time.perf_counter())]
            while stamps[-1][0] < p1 + a.steps:
                p = progress(live)
                if p > stamps[-1][0]:
                    stamps.append((p, time.perf_counter()))
        except Stall:
            rec["phases"][str(b)] = dict(stalled=True)
            print(json.dumps(dict(ctx=ctx, B=b, stalled=True)), flush=True)
            break
        dts = [(t2 - t1) / (q2 - q1) for (q1, t1), (q2, t2) in zip(stamps, stamps[1:])]
        tpot = (stamps[-1][1] - stamps[0][1]) / (stamps[-1][0] - stamps[0][0])
        rec["phases"][str(b)] = dict(tpot_ms=round(1e3 * tpot, 3), out_tok_s=round(b / tpot, 1),
                                     tpot_median_ms=round(1e3 * statistics.median(dts), 3),
                                     tokens_timed=stamps[-1][0] - stamps[0][0])
        print(json.dumps(dict(ctx=ctx, B=b, **rec["phases"][str(b)])), flush=True)
    eng.abort_request(live)
    while eng.has_unfinished_requests():
        eng.step()
    results.append(rec)
    outp.write_text(json.dumps(dict(model=a.model, mode=a.mode, kv=a.kv, window=a.window, sink=a.sink, center=CENTER,
                                prefill_fp8=a.prefill_fp8, runs=results), indent=1))
    del llm, eng
    import gc
    gc.collect()
    torch.cuda.empty_cache()
outp.write_text(json.dumps(dict(model=a.model, mode=a.mode, kv=a.kv, window=a.window, sink=a.sink, center=CENTER,
                                prefill_fp8=a.prefill_fp8, runs=results), indent=1))
print("BENCH_DONE")
