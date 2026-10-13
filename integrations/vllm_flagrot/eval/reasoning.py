"""Long-generation reasoning tasks against an OpenAI-compatible server (evaltools venv):
AIME 2024 / 2025, MATH-500, GPQA-Diamond; several seeds per problem, resumable, per-item tokens and wall time.

  python reasoning.py --base-url http://127.0.0.1:8602/v1 --model qwen8b --task aime24 --seeds 0,1,2 \
      --out results/tasks/reasoning/qwen8b/konlya_w0/aime24.jsonl [--limit 2] [--concurrency 64]

Sampling follows Qwen3's thinking-mode recommendation (temperature 0.6, top_p 0.95, top_k 20, min_p 0) unless
--temperature 0 (greedy, then one seed). The answer is the last \\boxed{...} of the final content (the server's
reasoning parser moves the thinking into `reasoning_content`; without it the whole text is searched).
Scoring: AIME integer equality; MATH-500 math_verify (falls back to normalized string equality); GPQA letter.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time

for _k in ("all_proxy", "ALL_PROXY"):
    os.environ.pop(_k, None)
os.environ["no_proxy"] = os.environ["NO_PROXY"] = "127.0.0.1,localhost," + os.environ.get("no_proxy", "")

MATH_PROMPT = "{q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
GPQA_PROMPT = ("Answer the following multiple choice question. Think step by step, then give the letter of the correct "
               "option within \\boxed{{}}.\n\n{q}\n\nA) {a}\nB) {b}\nC) {c}\nD) {d}")


def load(task: str):
    from datasets import load_dataset
    if task == "aime24":
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        return [dict(id=str(r["ID"]), prompt=MATH_PROMPT.format(q=r["Problem"]), answer=str(r["Answer"])) for r in ds]
    if task == "aime25":
        ds = load_dataset("math-ai/aime25", split="test")
        return [dict(id=str(r.get("id", i)), prompt=MATH_PROMPT.format(q=r["problem"]), answer=str(r["answer"]))
                for i, r in enumerate(ds)]
    if task == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [dict(id=r["unique_id"], prompt=MATH_PROMPT.format(q=r["problem"]), answer=r["answer"]) for r in ds]
    if task == "gpqa_diamond":
        try:
            ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        except Exception:     # gated: the CSV pre-downloaded into the HF hub cache (offline / no token)
            import glob as _g
            f = sorted(_g.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--Idavidrein--gpqa/snapshots/*/"
                                                  "gpqa_diamond.csv")))[-1]
            ds = load_dataset("csv", data_files=f, split="train")
        items = []
        for i, r in enumerate(ds):
            opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            perm = list(range(4))
            random.Random(i).shuffle(perm)                      # fixed per question, identical for every KV mode
            o = [opts[p].strip() for p in perm]
            items.append(dict(id=r.get("Record ID", str(i)), answer="ABCD"[perm.index(0)],
                              prompt=GPQA_PROMPT.format(q=r["Question"].strip(), a=o[0], b=o[1], c=o[2], d=o[3])))
        return items
    raise ValueError(task)


def last_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k].strip()
        k += 1
    return None


def norm(s: str) -> str:
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("$", "").replace(" ", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").replace("^\\circ", "").replace("^{\\circ}", "")
    return s.rstrip(".")


def score(task: str, pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    if task.startswith("aime"):
        m = re.findall(r"-?\d+", pred.replace(",", ""))
        return bool(m) and int(m[-1]) == int(gold)
    if task == "gpqa_diamond":
        m = re.findall(r"[ABCD]", pred.upper())
        return bool(m) and m[-1] == gold
    try:
        from math_verify import parse, verify
        return bool(verify(parse("$" + gold + "$"), parse("$" + pred + "$")))
    except Exception:
        return norm(pred) == norm(gold)


async def one(client, a, item, seed, sem, fh, lock):
    async with sem:
        t0 = time.time()
        kw = dict(model=a.model, messages=[{"role": "user", "content": item["prompt"]}], max_tokens=a.max_tokens)
        extra = {"chat_template_kwargs": {"enable_thinking": bool(a.thinking)}}
        if a.temperature > 0:
            kw.update(temperature=a.temperature, top_p=a.top_p, seed=seed)
            extra.update(top_k=a.top_k, min_p=0.0)
        else:
            kw.update(temperature=0.0)
        rec = dict(id=item["id"], seed=seed, answer=item["answer"])
        try:
            r = await client.chat.completions.create(**kw, extra_body=extra)
            msg = r.choices[0].message
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
            pred = last_boxed(content) or (last_boxed(reasoning) if not content else None)
            rec.update(pred=pred, correct=score(a.task, pred, item["answer"]), finish=r.choices[0].finish_reason,
                       prompt_tokens=r.usage.prompt_tokens, completion_tokens=r.usage.completion_tokens,
                       reasoning_tokens=getattr(r.usage.completion_tokens_details, "reasoning_tokens", None),
                       reasoning_chars=len(reasoning), content=content[-2000:], wall_s=round(time.time() - t0, 2))
        except Exception as e:
            rec.update(error=f"{type(e).__name__}: {str(e)[:300]}", wall_s=round(time.time() - t0, 2))
        async with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
        print(json.dumps({k: rec.get(k) for k in ("id", "seed", "pred", "answer", "correct", "completion_tokens",
                                                     "finish", "wall_s", "error")}), flush=True)


async def main_async(a):
    from openai import AsyncOpenAI
    items = load(a.task)[a.offset:]
    if a.limit:
        items = items[:a.limit]
    seeds = [int(s) for s in a.seeds.split(",")] if a.temperature > 0 else [0]
    done = set()
    if os.path.exists(a.out):
        for ln in open(a.out):
            r = json.loads(ln)
            if "error" not in r:
                done.add((r["id"], r["seed"]))
    todo = [(it, s) for it in items for s in seeds if (it["id"], s) not in done]
    print(f"{a.task}: {len(items)} items x {len(seeds)} seeds, {len(todo)} to run", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    client = AsyncOpenAI(base_url=a.base_url, api_key="EMPTY", timeout=a.timeout, max_retries=2)
    sem, lock = asyncio.Semaphore(a.concurrency), asyncio.Lock()
    t0 = time.time()
    with open(a.out, "a") as fh:
        await asyncio.gather(*(one(client, a, it, s, sem, fh, lock) for it, s in todo))
    summarize(a, items, seeds, time.time() - t0)


def summarize(a, items, seeds, wall):
    ids = {it["id"] for it in items}
    recs = {}
    for ln in open(a.out):
        r = json.loads(ln)
        if r["id"] in ids and "error" not in r:
            recs[(r["id"], r["seed"])] = r
    rs = list(recs.values())
    per_q = {}
    for r in rs:
        per_q.setdefault(r["id"], []).append(r["correct"])
    toks = [r["completion_tokens"] for r in rs]
    s = dict(task=a.task, model=a.model, n_items=len(per_q), n_samples=len(rs), seeds=seeds,
             accuracy=(sum(sum(v) / len(v) for v in per_q.values()) / len(per_q) if per_q else None),
             mean_completion_tokens=(sum(toks) / len(toks) if toks else None), max_completion_tokens=max(toks or [0]),
             truncated=sum(r["finish"] == "length" for r in rs), wall_s_this_call=round(wall, 1),
             mean_item_wall_s=(sum(r["wall_s"] for r in rs) / len(rs) if rs else None),
             per_question={k: sum(v) / len(v) for k, v in per_q.items()})
    json.dump(s, open(a.out.replace(".jsonl", "") + ".summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in s.items() if k != "per_question"}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("SERVED_NAME", "qwen8b"))
    ap.add_argument("--task", required=True, choices=["aime24", "aime25", "math500", "gpqa_diamond"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--thinking", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--out", required=True)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
