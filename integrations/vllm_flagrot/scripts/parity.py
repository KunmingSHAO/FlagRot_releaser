"""Accuracy parity of the flagrot_int4 vLLM KV cache with the research simulation (WikiText-2 test windows).

KL(BF16 || Q) needs both full next-token distributions.  vLLM's prompt_logprobs returns top-k lists only, so the
script captures the final-norm hidden states of every prompt position with a forward hook (in-process engine,
VLLM_ENABLE_V1_MULTIPROCESSING=0) and applies lm_head itself (BF16 GEMM, then FP32 log-softmax - the research
eval_kl protocol).  Both distributions come from ONE engine with the FlagRot backend:
  reference  backend MODE = "ref"   : attention over the unquantized K / V of the window (the stock FLASH_ATTN
                                      computation for a fresh prefill; checked against the stock backend below)
  quantized  backend MODE = "quant" : K / V written to and read back from the flagrot_int4 cache
every window runs as one 2048-token prefill (max_num_batched_tokens >= 2048, prefix caching off), reference first,
exactly like blockflag.run.eval_kl.  vLLM's own prompt_logprobs (logprob of the target token) is recorded too, as a
check of the hook path (its NLL must equal the hook NLL).

usage (vLLM env, repo root):
  CUDA_VISIBLE_DEVICES=6 python integrations/vllm_flagrot/scripts/parity.py --model llama8b --windows 20 \
      --kv codebook|static [--window W] [--prefill fa|kernel] [--max_num_batched_tokens 4096] \
      [--save_lp DIR --save_n 2] [--compare_lp DIR] \
      [--kv bf16]  (stock vLLM backend + BF16 cache: hook NLL / log-probs only, for the reference-path check)
      --out results/vllm_flagrot/parity_<...>.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common import ROOT, make_llm  # noqa: E402

CAP = {"on": False, "h": []}


def _lm(model):
    """(final norm, lm_head) of a causal LM or of a multimodal wrapper (Qwen3.5 / 3.8: model.language_model)"""
    m = getattr(model, "language_model", model)
    return m.model.norm, m.lm_head


def _install_hooks(llm):
    def setup(model):
        norm, head = _lm(model)
        CAP["lm_head"] = head.weight

        def hook(mod, inp, out):
            if CAP["on"]:
                h = out[0] if isinstance(out, tuple) else out
                CAP["h"].append(h.detach().clone())
        norm.register_forward_hook(hook)
        return True
    llm.apply_model(setup)


def _run(llm, ids, sp, hooks=True):
    CAP["h"], CAP["on"] = [], True
    out = llm.generate([{"prompt_token_ids": ids.tolist()}], sp, use_tqdm=False)[0]
    CAP["on"] = False
    plp = out.prompt_logprobs                                     # [None, {tok: Logprob}, ...]
    tgt = ids[1:].tolist()
    nll_vllm = -sum(plp[i + 1][t].logprob for i, t in enumerate(tgt)) / len(tgt)
    if not hooks:
        return None, nll_vllm
    h = torch.cat(CAP["h"], 0)
    # with chunked prefill the last call(s) may be the 1-token decode; keep the prompt positions only
    h = h[: ids.numel()]
    assert h.shape[0] == ids.numel(), (h.shape, ids.numel())
    CAP["last_h"] = h
    lp = F.log_softmax(F.linear(h[:-1], CAP["lm_head"]).float(), -1)
    return lp, nll_vllm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama8b", help="scripts/common.MODELS key")
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--kv", default="codebook", choices=["codebook", "static", "konly", "static_a", "konlya", "konlya_c", "rtn_a", "flagrot", "bf16", "fp8", "int4pth",
                             "rtn_a2", "static_a2", "konlya2", "konlya_c2", "kivi2", "kivi2c", "kivi2id", "kivi2s", "kivi2p8", "kivi2p8id", "kivi2p8f", "kivi2p8fid", "kivi2f", "kivi2p8ff", "kivi2p8ffid", "kivi2ff", "kivi2ffid", "kivi2h", "kivi2hid", "kivi2p8ffs"])
    ap.add_argument("--window", type=int, default=0, help="FlagRot BF16 staging window W")
    ap.add_argument("--sink", type=int, default=0, help="FlagRot BF16 sink tokens S (research kv_sink)")
    ap.add_argument("--prefill", default="fa", choices=["fa", "kernel"],
                    help="FlagRot prefill path: 'kernel' runs every prompt position through the decode kernel")
    ap.add_argument("--graphs", action="store_true")
    ap.add_argument("--center", nargs="?", const="post", default="", choices=["post", "auto"],
                    help="int2c: static post-RoPE key centering (plugin option 'center' = flagrot_center_<model>.pt; "
                    "FlagRot-S / FlagRot-C: the centered calibrations); 'auto': the research kv_center = auto vector")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size (> 1: NLL from vLLM's prompt_logprobs only; "
                    "the hidden-state hook needs the in-process single-GPU worker)")
    ap.add_argument("--max_num_batched_tokens", type=int, default=4096)
    ap.add_argument("--save_lp", default="")
    ap.add_argument("--save_n", type=int, default=2)
    ap.add_argument("--compare_lp", default="")
    ap.add_argument("--save_h", default="", help="save the final-norm hidden states of every window (stock bf16 run)")
    ap.add_argument("--ref_h", default="", help="stock kv (e.g. fp8): reference = the hidden states saved by a bf16 "
                    "run with --save_h, so KL(BF16 || this cache) is measured in the same harness")
    ap.add_argument("--prefill_fp8", default=None, choices=["had", "unit", "amax", "off", "emu_exact", "emu_q", "emu_q1", "emu_qtop1", "emu_qtop2", "emu_ac", "emu_acv", "emu_ac1", "emu_b", "int4", "int4ac"],
                help="FlagRot fa prefill attention: FP8 (K/V scale 1 | per-head amax) or BF16; default = plugin default")
    ap.add_argument("--token_kl", default="", help="also save the per-token KL of every window [n, L - 1] (torch file)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from vllm import SamplingParams
    from blockflag.modeling import windows
    ids_all = torch.load(ROOT / "cache" / f"{a.model}_test_ids.pt")
    wins = windows(ids_all, a.windows)
    from common import FLAGROT
    flag = a.kv in FLAGROT
    llm = make_llm(a.model, a.kv, max_model_len=4096, max_num_batched_tokens=a.max_num_batched_tokens,
                   window=a.window, prefill=a.prefill, enforce_eager=not a.graphs, max_num_seqs=16,
                   prefill_fp8=a.prefill_fp8, sink=a.sink, tp=a.tp, center=a.center)
    hooks = a.tp == 1
    if hooks:
        _install_hooks(llm)
    if flag:
        from vllm_flagrot.attention import backend as B
    sp = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0)
    per = []
    tok_kl = []
    if flag:
        B.FP8_STATS.acc.clear()                 # drop the profiling / warm-up runs (dummy batches, empty caches)
    t0 = time.time()
    for wi, w in enumerate(wins):
        tgt = w[1:].cuda()
        rec = dict(window=wi, tokens=int(tgt.numel()))
        if flag:
            B.MODE = "ref"
            if a.ref_h and hooks:          # chunked prefill (no single-chunk reference): a stock BF16 run's states
                href = torch.load(Path(a.ref_h) / f"h_{wi}.pt").cuda()
                lr = F.log_softmax(F.linear(href[:-1], CAP["lm_head"]).float(), -1)
                nll_v_ref = float("nan")
            elif a.max_num_batched_tokens >= w.numel() and hooks:
                lr, nll_v_ref = _run(llm, w, sp)
            else:
                lr, nll_v_ref = None, float("nan")
            B.MODE = "quant"
            lq, nll_v_q = _run(llm, w, sp, hooks)
        else:
            lx, nll_x = _run(llm, w, sp, hooks)
            if a.ref_h:
                href = torch.load(Path(a.ref_h) / f"h_{wi}.pt").cuda()
                lr = F.log_softmax(F.linear(href[:-1], CAP["lm_head"]).float(), -1)
                nll_v_ref = float("nan")
                lq, nll_v_q = lx, nll_x
            else:
                lr, nll_v_ref = lx, nll_x
                lq, nll_v_q = None, float("nan")
        if a.save_h and hooks:
            Path(a.save_h).mkdir(parents=True, exist_ok=True)
            torch.save(CAP["last_h"].cpu(), Path(a.save_h) / f"h_{wi}.pt")
        if lr is not None:
            rec.update(nll_ref=float(-lr.gather(1, tgt[:, None]).mean()))
        if lq is not None:
            rec.update(nll_q=float(-lq.gather(1, tgt[:, None]).mean()))
        if not math.isnan(nll_v_ref):
            rec.update(nll_ref_vllm=nll_v_ref)
        if not math.isnan(nll_v_q):
            rec.update(nll_q_vllm=nll_v_q)
        if lr is not None and lq is not None:
            if a.token_kl:
                tok_kl.append((lr.exp() * (lr - lq)).sum(-1).cpu())
            rec.update(kl=float((lr.exp() * (lr - lq)).sum(-1).mean()),
                       agree=float((lr.argmax(-1) == lq.argmax(-1)).float().mean()))
        if a.save_lp and wi < a.save_n:
            d = Path(a.save_lp)
            d.mkdir(parents=True, exist_ok=True)
            for name, t in (("ref", lr), ("q", lq)):
                if t is not None:
                    torch.save(t.half().cpu(), d / f"{name}_{wi}.pt")
        if a.compare_lp and wi < a.save_n:
            d = Path(a.compare_lp)
            base = torch.load(d / f"ref_{wi}.pt").cuda().float() if (d / f"ref_{wi}.pt").exists() else None
            if base is not None:
                for name, t in (("ref", lr), ("q", lq)):
                    if t is not None:
                        rec[f"kl_saved_ref||{name}"] = float((base.exp() * (base - t)).sum(-1).mean())
            if (d / f"q_{wi}.pt").exists() and lq is not None:
                bq = torch.load(d / f"q_{wi}.pt").cuda().float()
                rec["kl_saved_q||q"] = float((bq.exp() * (bq - lq)).sum(-1).mean())
        per.append(rec)
        print(json.dumps(rec), flush=True)
        del lr, lq
    N = sum(r["tokens"] for r in per)
    agg = {}
    for k in ("kl", "nll_ref", "nll_q", "nll_ref_vllm", "nll_q_vllm", "agree",
              "kl_saved_ref||ref", "kl_saved_ref||q", "kl_saved_q||q"):
        rs = [r for r in per if k in r and not math.isnan(r[k])]
        if rs:
            agg[k] = sum(r[k] * r["tokens"] for r in rs) / sum(r["tokens"] for r in rs)
            if len(rs) < len(per):
                agg[k + "_windows"] = len(rs)
    for k in ("nll_ref", "nll_q"):
        if k in agg:
            agg["ppl_" + k[4:]] = math.exp(agg[k])
    res = dict(model=a.model, kv=a.kv, center=a.center, window=a.window, sink=a.sink, prefill=a.prefill, prefill_fp8=a.prefill_fp8, graphs=a.graphs,
               windows=a.windows, tp=a.tp, max_num_batched_tokens=a.max_num_batched_tokens, ref_h=a.ref_h or None,
               seconds=round(time.time() - t0, 1), **agg, per_window=per)
    if flag and B.FP8_STATS.on:
        res["fp8_stats"] = B.FP8_STATS.read()
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_window", "fp8_stats")}))
    if a.token_kl and tok_kl:
        torch.save(torch.stack(tok_kl), a.token_kl)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
