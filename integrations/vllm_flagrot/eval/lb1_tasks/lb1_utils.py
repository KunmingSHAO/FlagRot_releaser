"""LongBench v1 for lm-eval with the official prompts, data and middle truncation (THUDM/LongBench LongBench/pred.py).

Data: the official LongBench release (THUDM/LongBench data.zip, sha256 cb45b11a…; <task>.jsonl in `lb1_dir`, env LB1_DIR,
default <repo>/data/longbench1). Prompt = official/dataset2prompt.json[task].format(**item); then, as pred.py,

    ids = tokenizer(prompt, truncation=False).input_ids     # model tokenizer, special tokens as in pred.py
    if len(ids) > max_length: prompt = decode(ids[:max_length//2]) + decode(ids[-max_length//2:])   (skip_special_tokens)

with max_length = metadata `max_prompt_tokens` (pred.py / model2maxlen.json: 31500 for the 32k-context models). The chat
template (none for trec / triviaqa / samsum / lsht / lcc / repobench-p, as pred.py) is applied by the server afterwards.
Metrics: lm-eval's tasks/longbench/metrics.py (taken from the official metrics), imported unchanged. The docs keep all
item fields except `context` (the prompt is logged once, in each sample's arguments) and add `n_prompt_tokens` (before
truncation) and `truncated`.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import lm_eval.tasks

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "lb1_official_metrics", Path(lm_eval.tasks.__file__).parent / "longbench" / "metrics.py")
_M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_M)
for _n in dir(_M):
    if _n.startswith("get_") and _n.endswith("_with_score"):
        globals()[_n] = getattr(_M, _n)

_TEMPLATES = json.loads((HERE / "official/dataset2prompt.json").read_text())
_PROMPTS: dict[tuple[str, str], str] = {}
_TOK = {}


def _tokenizer(path):
    if path not in _TOK:
        from transformers import AutoTokenizer
        _TOK[path] = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return _TOK[path]


def _load(task: str, pretrained: str = "", max_prompt_tokens: int = 31500, lb1_dir: str = "", **kw):
    import datasets
    if not pretrained:
        raise ValueError("lb1: --metadata needs 'pretrained' (model tokenizer for the middle truncation)")
    d = Path(lb1_dir or os.environ.get("LB1_DIR", str(HERE.parents[3] / "data" / "longbench1")))
    tok = _tokenizer(pretrained)
    half = int(max_prompt_tokens / 2)
    docs = []
    for line in open(d / f"{task}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        prompt = _TEMPLATES[task].format(**r)
        ids = tok(prompt, truncation=False).input_ids
        trunc = len(ids) > max_prompt_tokens
        if trunc:
            prompt = (tok.decode(ids[:half], skip_special_tokens=True)
                      + tok.decode(ids[-half:], skip_special_tokens=True))
        key = (pretrained, max_prompt_tokens, task, r["_id"])
        assert key not in _PROMPTS or _PROMPTS[key] == prompt, key
        _PROMPTS[key] = prompt
        r.pop("context")
        r.update(n_prompt_tokens=len(ids), truncated=trunc, lb1_task=task, lb1_tok=pretrained, lb1_max=max_prompt_tokens)
        docs.append(r)
    return {"test": datasets.Dataset.from_list(docs)}


def doc_to_text(doc) -> str:
    return _PROMPTS[(doc["lb1_tok"], doc["lb1_max"], doc["lb1_task"], doc["_id"])]


for _t in _TEMPLATES:
    globals()["load_" + _t.replace("-", "_")] = (lambda t: (lambda **kw: _load(t, **kw)))(_t)
