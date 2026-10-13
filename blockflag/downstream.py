"""Zero-shot downstream accuracy with lm-evaluation-harness on the in-memory (fake-quantized) model.

The calibrated model object is handed to lm-eval's HFLM directly.  For every task the per-document correctness of
the reported metric (acc_norm where the harness defines it, else acc) is stored in doc_id order, so two configs of the
same model can be compared with a paired bootstrap over documents (blockflag.stats.paired_acc).
"""
from __future__ import annotations

import os

NORM_TASKS = {"arc_challenge", "arc_easy", "hellaswag", "piqa"}     # tasks reported with length-normalized accuracy
FEWSHOT_TASKS = {"gsm8k"}                                            # generative tasks run with the harness default shots
FILTER = {"gsm8k": "flexible-extract"}                                # one answer-extraction filter per task


def eval_lm(model, tok, tasks: list[str], limit: int | None, batch_size: int = 16, max_len: int = 0,
            pretrained: str = "") -> dict:
    """max_len > 0: long-context tasks (RULER: niah_*, ruler_*) built at that sequence length."""
    for k in ("all_proxy", "ALL_PROXY"):                    # httpx cannot use the SOCKS proxy for dataset downloads
        os.environ.pop(k, None)
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    long = max_len > 0
    if set(tasks) <= FEWSHOT_TASKS:                  # generation: the per-step quantizer cost is amortized over the batch
        batch_size = 32
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=1 if long else batch_size, max_length=max_len or 2048)
    kw = dict(metadata={"max_seq_lengths": [max_len], "pretrained": pretrained}) if long else {}
    shots = None if long or set(tasks) <= FEWSHOT_TASKS else 0
    assert shots is None or not set(tasks) & FEWSHOT_TASKS, "few-shot and zero-shot tasks go in separate runs"
    out = lm_eval.simple_evaluate(model=lm, tasks=tasks, limit=limit, num_fewshot=shots, log_samples=True,
                                  bootstrap_iters=0, random_seed=0, numpy_random_seed=0, torch_random_seed=0, **kw)
    res = {}
    for t in tasks:
        samples = [d for d in out["samples"][t] if t not in FILTER or d.get("filter") == FILTER[t]]
        samples = sorted(samples, key=lambda d: d["doc_id"])
        metric = "acc_norm" if t in NORM_TASKS else ("acc" if "acc" in samples[0] else
                                                       next(k for k in (str(max_len), "score", "exact_match") if k in samples[0]))
        per = [float(d[metric]) for d in samples]
        res[t] = dict(metric=metric, acc=sum(per) / len(per), n=len(per), per_doc=per)
    res["mean"] = dict(metric="mean", acc=sum(v["acc"] for v in res.values()) / len(tasks), n=len(tasks))
    return dict(lm=res)
