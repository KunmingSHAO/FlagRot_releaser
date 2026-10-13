#!/usr/bin/env python3
"""Simulated KL sweeps of the paper (KL(BF16 || quantized-KV model) on all 2048-token WikiText-2 test windows).

Every job is one `python -m blockflag.run` configuration; its result file is results/runs/<config id>.json (the id is a
hash of the canonical configuration, so finished jobs are skipped and shared rows are run once).

  python scripts/sim/sweep.py list   --table unified                       # print the job list (name, id, config)
  python scripts/sim/sweep.py run    --table unified --gpus 0,1,2,3        # run the missing jobs, one per GPU at a time
  python scripts/sim/sweep.py report --table unified                       # geometric-mean table from the run files

  --models llama3b,llama8b   restrict the model suite (default: the 12-model suite)
  --bits 2 / --methods rtn,flagrot / --windows 8   restrict bit widths / methods; fewer evaluation windows (smoke tests;
                             a different `windows` gives different config ids)

Tables
  unified   bit-width table: INT2 / INT3 / INT4 codes, groups of 32, S4W128 (4 sink + 128 recent tokens in BF16):
            per-token RTN and FlagRot, each without / with automatic key centering (+auto), and KIVI (per-channel keys)
            without / with selection in two layouts: FP8 scale + minimum (b + 0.5 bits, *_p8) and FP16 (b + 1 bits);
            plus the RotateKV simulation at INT2 (g32, clip 0.8 / 1.0)
  int2      per-model INT2 comparison (12 models, S4W128): the unified INT2 rows, fixed selection after centering
            (fixed+auto) and RotateKV (g32 clip 0.8 / 1.0, g128 clip 0.8)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from blockflag import paths  # noqa: E402
from blockflag.run import canonical, config_id  # noqa: E402

BASE = dict(windows=1000, calib_windows=4)           # windows = 1000 is clipped to the available test windows
S4W128 = dict(kv_sink=4, kv_resid=128)
SUITE12 = ("llama8b", "qwen8b", "mistral7b", "qwen14b", "qwen38_27b", "gemma3_27b", "llama2_7b", "llama3b", "gemma2_9b",
           "phi4", "nemo12b", "glm4_9b")
DEV = SUITE12[:4]                                     # development models; the other 8 are held out
NAMES = {"llama8b": "Llama-3.1-8B", "qwen8b": "Qwen3-8B", "mistral7b": "Mistral-7B-v0.3", "qwen14b": "Qwen3-14B",
         "qwen38_27b": "Qwen3.8-27B", "gemma3_27b": "Gemma-3-27B", "llama2_7b": "Llama-2-7B", "llama3b": "Llama-3.2-3B",
         "gemma2_9b": "Gemma-2-9B", "phi4": "Phi-4", "nemo12b": "Mistral-NeMo-12B", "glm4_9b": "GLM-4-9B"}
FLAGROT = dict(kv_rb=32, kv_flags="dyn", kv_dyn="cq+f+konly")     # per-token key selection (I / H32 per sub-vector, H_D)
KIVI_SEL = dict(kv_rb=32, kv_flags="dyn", kv_dyn="att+f")         # KIVI + per-group selection
AUTO = dict(kv_center="auto")                                     # static post-RoPE key centering, chosen by calib. KL


def methods(bits: int) -> dict:
    Q = dict(kv_k=f"int{bits}a:32", kv_v=f"int{bits}a:32")
    K16 = dict(kv_k=f"kivi{bits}:32", kv_v=f"kivi{bits}:32", kv_k_axis="token")
    K8 = dict(kv_k=f"kivi{bits}f8:32", kv_v=f"kivi{bits}f8:32", kv_k_axis="token")
    m = {"rtn": dict(Q), "rtn+auto": dict(Q, **AUTO), "flagrot": dict(Q, **FLAGROT), "flagrot+auto": dict(Q, **FLAGROT, **AUTO),
         "fixed+auto": dict(Q, kv_rb=32, kv_flags="headsel", **AUTO),
         "kivi_p8": dict(K8), "kivi_sel_p8": dict(K8, **KIVI_SEL), "kivi": dict(K16), "kivi_sel": dict(K16, **KIVI_SEL)}
    if bits == 2:   # RotateKV (grouped-head Hadamard over 4 KV heads + calibrated channel reordering, pre-RoPE keys)
        m.update({"rotatekv_g32_c080": dict(kv_k="int2ac80:32", kv_v="int2ac80:32", kv_rotatekv="gh4+ro"),
                  "rotatekv_g32_c100": dict(kv_k="int2ac100:32", kv_v="int2ac100:32", kv_rotatekv="gh4+ro"),
                  "rotatekv_g128_c080": dict(kv_k="int2ac80:128", kv_v="int2ac80:128", kv_rotatekv="gh4+ro")})
    return m


UNIFIED = ("rtn", "rtn+auto", "flagrot", "flagrot+auto", "kivi_p8", "kivi_sel_p8", "kivi", "kivi_sel")
TABLES = {
    "unified": [(b, k) for b in (2, 3, 4) for k in UNIFIED] + [(2, "rotatekv_g32_c080"), (2, "rotatekv_g32_c100")],
    "int2": [(2, k) for k in ("rtn", "flagrot", "rtn+auto", "fixed+auto", "flagrot+auto", "kivi_p8", "kivi_sel_p8", "kivi",
                              "kivi_sel", "rotatekv_g32_c080", "rotatekv_g32_c100", "rotatekv_g128_c080")],
}
# selection rows and the row without selection they are compared with (paired over windows)
PAIRS = {"flagrot": "rtn", "flagrot+auto": "rtn+auto", "kivi_sel_p8": "kivi_p8", "kivi_sel": "kivi"}


def jobs(a) -> list:
    models = a.models.split(",") if a.models else list(SUITE12)
    want_m = set(a.methods.split(",")) if a.methods else None
    want_b = {int(x) for x in a.bits.split(",")} if a.bits else None
    out = []
    for bits, meth in TABLES[a.table]:
        if (want_m and meth not in want_m) or (want_b and bits not in want_b):
            continue
        for m in models:
            c = canonical(dict(BASE, model=m, **S4W128, **methods(bits)[meth], **({"windows": a.windows} if a.windows else {})))
            out.append(dict(name=f"int{bits}/{m}/{meth}", bits=bits, model=m, method=meth, id=config_id(c), cfg=c))
    return out


def run_file(j) -> Path:
    return paths.RESULTS / "runs" / f"{j['id']}.json"


def cmd_list(a):
    for j in jobs(a):
        print(json.dumps(dict(name=j["name"], id=j["id"], done=run_file(j).exists(), cfg=j["cfg"])))


def cmd_run(a):
    todo = [j for j in jobs(a) if not run_file(j).exists()]
    todo = list({j["id"]: j for j in todo}.values())          # shared configurations run once
    print(f"{len(todo)} jobs to run on GPUs {a.gpus}", flush=True)
    if a.dry_run:
        for j in todo:
            print(j["name"], j["id"])
        return
    logs = paths.RESULTS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    free, running, failed = a.gpus.split(","), {}, []
    env0 = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                TOKENIZERS_PARALLELISM="false")
    while todo or running:
        while todo and free:
            j, g = todo.pop(0), free.pop(0)
            lf = open(logs / f"{j['name'].replace('/', '__')}.log", "a")
            p = subprocess.Popen([sys.executable, "-m", "blockflag.run", "--config", json.dumps(j["cfg"]),
                                  "--out", str(run_file(j))], cwd=ROOT, env=dict(env0, CUDA_VISIBLE_DEVICES=g),
                                 stdout=lf, stderr=subprocess.STDOUT)
            running[p] = (j, g, time.time())
            print(f"start {j['name']} on GPU {g}", flush=True)
        time.sleep(5)
        for p in [p for p in running if p.poll() is not None]:
            j, g, t0 = running.pop(p)
            free.append(g)
            ok = p.returncode == 0 and run_file(j).exists()
            print(f"{'done' if ok else 'FAILED'} {j['name']} ({time.time() - t0:.0f} s)", flush=True)
            if not ok:
                failed.append(j["name"])
    if failed:
        sys.exit(f"failed: {failed} (logs in {logs})")


def gm(v):
    return math.exp(sum(math.log(x) for x in v) / len(v))


def cmd_report(a):
    from blockflag.stats import paired_delta
    J = jobs(a)
    R = {}
    for j in J:
        f = run_file(j)
        if f.exists():
            R[(j["bits"], j["model"], j["method"])] = json.loads(f.read_text())
    models = a.models.split(",") if a.models else list(SUITE12)
    rows = list(dict.fromkeys(m for _, m in TABLES[a.table] if not a.methods or m in a.methods.split(",")))
    bits_l = sorted({b for b, _ in TABLES[a.table] if not a.bits or str(b) in a.bits.split(",")})
    out = {}
    print(f"KL x 1e3, S4W128, groups of 32; geometric mean over {len(models)} models (held-out: models outside {DEV})")
    print("| method | " + " | ".join(f"INT{b}" for b in bits_l) + " |")
    print("|---|" + "---:|" * len(bits_l))
    for meth in rows:
        cells = []
        for b in bits_l:
            v = [1e3 * R[(b, m, meth)]["kl"] for m in models if (b, m, meth) in R]
            vh = [1e3 * R[(b, m, meth)]["kl"] for m in models if m not in DEV and (b, m, meth) in R]
            if not v:
                cells.append("–")
                continue
            c = f"{gm(v):.1f}" + ("" if len(v) == len(models) else f" ({len(v)}/{len(models)} models)")
            if meth in PAIRS:          # models on which the selection row is significantly lower / higher
                lo = hi = 0
                for m in models:
                    x, y = R.get((b, m, meth)), R.get((b, m, PAIRS[meth]))
                    if x and y:
                        d = paired_delta(x, y, n_boot=5000, alpha=0.05, seed=0)
                        lo, hi = lo + (d["hi"] < 0), hi + (d["lo"] > 0)
                c += f" [{lo}/{hi}]"
            cells.append(c)
            out.setdefault(meth, {})[f"INT{b}"] = dict(gmean=gm(v), gmean_heldout=gm(vh) if vh else None, n_models=len(v),
                                                       per_model={m: 1e3 * R[(b, m, meth)]["kl"] for m in models if (b, m, meth) in R})
        print(f"| {meth} | " + " | ".join(cells) + " |")
    print("[L/H] = models with a significantly lower / higher KL than the row without selection (paired 95 % bootstrap "
          "over windows)")
    if a.table == "int2" or a.per_model:
        for b in bits_l:
            print(f"\nINT{b} per model (KL x 1e3)\n| model | " + " | ".join(rows) + " |\n|---|" + "---:|" * len(rows))
            for m in models:
                print(f"| {NAMES.get(m, m)} | " + " | ".join(
                    f"{1e3 * R[(b, m, k)]['kl']:.1f}" if (b, m, k) in R else "–" for k in rows) + " |")
    if a.json:
        Path(a.json).write_text(json.dumps(dict(table=a.table, rows=out), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["list", "run", "report"])
    ap.add_argument("--table", default="unified", choices=sorted(TABLES))
    ap.add_argument("--models", default="")
    ap.add_argument("--methods", default="")
    ap.add_argument("--bits", default="")
    ap.add_argument("--windows", type=int, default=0, help="evaluation windows (default: all test windows)")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--per-model", action="store_true", help="report: per-model tables also for --table unified")
    ap.add_argument("--json", default="", help="report: also write the table as JSON")
    a = ap.parse_args()
    {"list": cmd_list, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
