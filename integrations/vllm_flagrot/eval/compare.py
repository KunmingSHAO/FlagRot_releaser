"""Paired comparison of KV-cache modes on the task suites of launch.py (LongBench v1, RULER-128k, AIME).

  python integrations/vllm_flagrot/eval/compare.py --suite longbench1 --model qwen8b
  python integrations/vllm_flagrot/eval/compare.py --suite ruler128k --model llama8b_it
  python integrations/vllm_flagrot/eval/compare.py --suite aime --model qwen8b [--pairs kivi2p8ff_s4_w128-bf16,...]

Reads results/tasks/<suite>/<model>/<mode dir>/<task>/ (done.json + lm-eval samples or reasoning jsonl) and prints, per
pair (a - b), the score of a and b and the difference with a 95 % paired bootstrap CI (numpy rng 0, B = 10000):
  * LongBench v1: macro average over the 21 tasks (mean of per-task means), items resampled within each task;
  * RULER: macro average over the 13 subtasks, stratified the same way;
  * AIME: accuracy per problem averaged over the seeds, AIME24 + AIME25 pooled (60 problems), flat bootstrap.
Only items present in both modes are paired.  Scores are in percent.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
B = 10000
LB1_TASKS = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", "dureader",
             "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", "passage_count",
             "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
DEFAULT_PAIRS = [("kivi2p8ff_s4_w128", "kivi2p8ffid_s4_w128"), ("kivi2p8ffs_s4_w128", "kivi2p8ffid_s4_w128"),
                 ("kivi2p8ff_s4_w128", "konlya2_s4_w128_ca"), ("kivi2_s4_w128", "kivi2id_s4_w128"),
                 ("kivi2p8ff_s4_w128", "bf16"), ("kivi2p8ffid_s4_w128", "bf16"), ("kivi2p8ffs_s4_w128", "bf16"),
                 ("konlya2_s4_w128_ca", "bf16"), ("kivi2_s4_w128", "bf16"), ("kivi2id_s4_w128", "bf16"),
                 ("konlya_s4_w128", "bf16")]


def items_lmeval(tdir: Path) -> dict:
    """{subtask: {doc_id: score}} from lm-eval samples_*.jsonl (latest file per subtask)."""
    latest = {}
    for f in sorted(glob.glob(str(tdir / "*" / "samples_*.jsonl"))):
        latest[Path(f).name[len("samples_"):].rsplit("_", 1)[0]] = f
    out = {}
    for sub, f in latest.items():
        d = {}
        for line in open(f):
            s = json.loads(line)
            v = None
            for m in s.get("metrics") or []:       # RULER lists every configured length; -1 marks lengths not run
                if isinstance(s.get(m), (int, float)) and s[m] >= 0:
                    v = float(s[m])
                    break
            if v is None:
                for k, x in s.items():
                    if k.isdigit() and isinstance(x, (int, float)) and x >= 0:
                        v = float(x)
                        break
            if v is not None and s["doc_id"] not in d:
                d[s["doc_id"]] = v
        out[sub] = d
    return out


def done(d: Path) -> bool:
    return (d / "done.json").exists()


def load_mode(suite: str, model: str, mdir: Path):
    """per-mode items: {stratum: {item: score}} (LongBench, RULER) or {item: score} (AIME); None if incomplete."""
    if suite == "longbench1":
        if not (done(mdir / "lb1_chat") and done(mdir / "lb1_raw")):
            return None
        it = {}
        for part in ("lb1_chat", "lb1_raw"):
            for sub, d in items_lmeval(mdir / part).items():
                it[sub[len("lb1_"):] if sub.startswith("lb1_") else sub] = d
        return it
    if suite == "ruler128k":
        return items_lmeval(mdir / "ruler_128k") if done(mdir / "ruler_128k") else None
    if suite == "aime":
        per = {}
        for t in ("aime24", "aime25"):
            f = mdir / t / f"{t}.jsonl"
            if not (done(mdir / t) and f.exists()):
                return None
            acc = defaultdict(list)
            for line in open(f):
                s = json.loads(line)
                acc[s["id"]].append(1.0 if s.get("correct") else 0.0)
            per.update({(t, k): sum(v) / len(v) for k, v in acc.items()})
        return per
    raise ValueError(suite)


def strat(a: dict, b: dict, strata: list):
    rng = np.random.default_rng(0)
    bs = np.zeros(B)
    ma = mb = 0.0
    for t in strata:
        ks = sorted(set(a[t]) & set(b[t]))
        da, db = np.array([a[t][k] for k in ks]), np.array([b[t][k] for k in ks])
        ma += da.mean() / len(strata)
        mb += db.mean() / len(strata)
        d = da - db
        bs += d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1) / len(strata)
    return ma, mb, np.sort(bs)


def flat(a: dict, b: dict):
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks])
    bs = np.sort(d[np.random.default_rng(0).integers(0, len(d), size=(B, len(d)))].mean(axis=1))
    return np.mean([a[k] for k in ks]), np.mean([b[k] for k in ks]), bs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=["longbench1", "ruler128k", "aime"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="", help="launch.py --tag of the runs (results/tasks/<suite>_<tag>)")
    ap.add_argument("--results", default=str(ROOT / "results/tasks"))
    ap.add_argument("--pairs", default="", help="comma list a-b of mode directories (default: the paper's contrasts)")
    a = ap.parse_args()
    base = Path(a.results) / (a.suite + (f"_{a.tag}" if a.tag else "")) / a.model
    M = {}
    for mdir in sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []:
        it = load_mode(a.suite, a.model, mdir)
        if it:
            M[mdir.name] = it
    print(f"## {a.suite} / {a.model}: complete modes {sorted(M)}")
    pairs = [tuple(p.split("-", 1)) for p in a.pairs.split(",")] if a.pairs else DEFAULT_PAIRS
    for x, y in pairs:
        if x not in M or y not in M:
            continue
        if a.suite == "aime":
            ma, mb, bs = flat(M[x], M[y])
        else:
            strata = LB1_TASKS if a.suite == "longbench1" else sorted(set(M[x]) & set(M[y]))
            ma, mb, bs = strat(M[x], M[y], strata)
        print(f"   {x} - {y}: {100 * (ma - mb):+.2f} [{100 * bs[int(.025 * B)]:+.2f}, {100 * bs[int(.975 * B) - 1]:+.2f}]"
              f"  ({100 * ma:.2f} vs {100 * mb:.2f})")


if __name__ == "__main__":
    main()
