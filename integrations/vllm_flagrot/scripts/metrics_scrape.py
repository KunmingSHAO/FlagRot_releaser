"""Scrape a vLLM server's Prometheus /metrics into JSON (stdlib only; any python3).

  metrics_scrape.py --port P --out serving.json [--log server.log] [--samples samples.jsonl] [--since base.json]
      one snapshot: counters / gauges, histograms with count, sum, mean and p50 / p90 / p99 (bucket interpolation),
      plus the KV-cache capacity from the server log and the peak gauges of the sampler file; with --since, counters
      and histograms are reported as differences to an earlier snapshot (one suite on a long-lived server)
  metrics_scrape.py --port P --sample 5 --samples samples.jsonl
      sampler: every 5 s append {t, kv_cache_usage, running, waiting, preemptions, gen_tokens} until the server is gone
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import urllib.request

LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)')
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
DROP_LABELS = {"model_name", "engine"}


def fetch(port: int, timeout: float = 10.0) -> str:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never through the http proxy
    with opener.open(f"http://127.0.0.1:{port}/metrics", timeout=timeout) as r:
        return r.read().decode()


def parse(text: str) -> dict:
    """{'name|lab=val,...': value}; model_name / engine labels dropped (one model per server)."""
    out: dict[str, float] = {}
    for ln in text.splitlines():
        if not ln or ln[0] == "#":
            continue
        m = LINE.match(ln)
        if not m:
            continue
        name, labels, val = m.groups()
        try:
            v = float(val)
        except ValueError:
            continue
        labs = [(k, x) for k, x in LABEL.findall(labels or "") if k not in DROP_LABELS]
        key = name + ("|" + ",".join(f"{k}={x}" for k, x in labs) if labs else "")
        out[key] = out.get(key, 0.0) + v
    return out


def summarize(flat: dict, base: dict | None = None) -> dict:
    """counters / gauges as values; histograms (…_bucket / _sum / _count) as count, sum, mean, quantiles."""
    base = base or {}
    res, hist = {}, {}
    for key, v in flat.items():
        name, _, labs = key.partition("|")
        if not name.startswith("vllm:") or name.endswith("_created"):
            continue
        delta = v - base.get(key, 0.0) if (name.endswith(("_total", "_bucket", "_sum", "_count"))) else v
        if name.endswith("_bucket"):
            le = dict(x.split("=", 1) for x in labs.split(",") if x).get("le")
            rest = ",".join(x for x in labs.split(",") if x and not x.startswith("le="))
            hist.setdefault((name[:-7], rest), {}).setdefault("b", []).append(
                (math.inf if le == "+Inf" else float(le), delta))
        elif name.endswith("_sum") and any(k.startswith(name[:-4] + "_bucket") for k in flat):
            hist.setdefault((name[:-4], labs), {})["sum"] = delta
        elif name.endswith("_count") and any(k.startswith(name[:-6] + "_bucket") for k in flat):
            hist.setdefault((name[:-6], labs), {})["count"] = delta
        else:
            res[key] = delta
    for (name, labs), h in hist.items():
        b = sorted(h.get("b", []))
        n = h.get("count", b[-1][1] if b else 0.0)
        d = dict(count=n, sum=h.get("sum", 0.0), mean=(h.get("sum", 0.0) / n if n else None))
        for q in (0.5, 0.9, 0.99):
            d[f"p{int(q * 100)}"] = _quantile(b, q * n) if n else None
        res[name + ("|" + labs if labs else "")] = d
    return res


def _quantile(buckets, target):
    prev_le, prev_c = 0.0, 0.0
    for le, c in buckets:
        if c >= target:
            if math.isinf(le):
                return prev_le
            return prev_le + (le - prev_le) * ((target - prev_c) / (c - prev_c) if c > prev_c else 0.0)
        prev_le, prev_c = le, c
    return prev_le


def capacity(log: str) -> dict:
    out = {}
    try:
        txt = open(log, errors="replace").read()
    except OSError:
        return out
    m = re.findall(r"GPU KV cache size: ([\d,]+) tokens", txt)
    if m:
        out["kv_cache_tokens"] = int(m[-1].replace(",", ""))
    m = re.findall(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", txt)
    if m:
        out["max_concurrency"] = {"tokens_per_request": int(m[-1][0].replace(",", "")), "x": float(m[-1][1])}
    return out


def pick(flat: dict, prefix: str) -> float:
    return sum(v for k, v in flat.items() if k.split("|")[0] == prefix)


def sampler(port: int, period: float, path: str):
    fails = 0
    while fails < 5:
        try:
            f = parse(fetch(port, timeout=5))
            fails = 0
        except Exception:
            fails += 1
            time.sleep(period)
            continue
        rec = dict(t=round(time.time(), 1), kv_cache_usage=pick(f, "vllm:kv_cache_usage_perc"),
                   running=pick(f, "vllm:num_requests_running"), waiting=pick(f, "vllm:num_requests_waiting"),
                   preemptions=pick(f, "vllm:num_preemptions_total"), gen_tokens=pick(f, "vllm:generation_tokens_total"),
                   prompt_tokens=pick(f, "vllm:prompt_tokens_total"))
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        time.sleep(period)


def peaks(path: str, t0: float | None = None) -> dict:
    try:
        rows = [json.loads(x) for x in open(path) if x.strip()]
    except OSError:
        return {}
    if t0:
        rows = [r for r in rows if r["t"] >= t0]
    if not rows:
        return {}
    busy = [r for r in rows if r["running"] > 0]
    return dict(n_samples=len(rows), peak_kv_cache_usage=max(r["kv_cache_usage"] for r in rows),
                peak_running=max(r["running"] for r in rows), peak_waiting=max(r["waiting"] for r in rows),
                mean_running_busy=(sum(r["running"] for r in busy) / len(busy) if busy else 0.0),
                busy_seconds=(len(busy) * (rows[1]["t"] - rows[0]["t"]) if len(rows) > 1 else 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out")
    ap.add_argument("--raw", help="also save the raw flat metrics (for --since)")
    ap.add_argument("--since", help="raw flat metrics of an earlier snapshot: report differences")
    ap.add_argument("--log")
    ap.add_argument("--samples")
    ap.add_argument("--sample", type=float, default=0.0)
    ap.add_argument("--meta", default="{}", help="JSON merged into the output")
    a = ap.parse_args()
    if a.sample:
        return sampler(a.port, a.sample, a.samples)
    flat = parse(fetch(a.port))
    if a.raw:
        json.dump(dict(t=time.time(), flat=flat), open(a.raw, "w"))
    base, t0 = None, None
    if a.since and os.path.exists(a.since):
        b = json.load(open(a.since))
        base, t0 = b["flat"], b["t"]
    if a.out:
        res = dict(json.loads(a.meta), port=a.port, scraped=time.strftime("%Y-%m-%d %H:%M:%S"),
                   window_start=t0, metrics=summarize(flat, base))
        if a.log:
            res.update(capacity(a.log))
        if a.samples:
            res["sampler"] = peaks(a.samples, t0)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
