"""Suite runner: servers in parallel on the given GPUs, tasks against them, resumable (eval env with lm-eval).

  python integrations/vllm_flagrot/eval/launch.py --suite aime --gpus 6,7
      [--models qwen8b] [--modes kivi2p8ff:s4w128,bf16] [--tasks aime24] [--limit 2] [--dry-run] [--tag smoke]

* jobs = models x modes of configs/<suite>.yaml; one job = one `scripts/serve_mode.sh` server on one GPU, its tasks
  run sequentially; jobs are scheduled on the free GPUs of --gpus (one server per GPU) in config order.
* outputs: results/tasks/<suite>[_<tag>]/<model>/<mode>/<task>/ (tool output + done.json with wall time and
  the command), serving_<task>.json (server /metrics difference over the task + sampler peaks) and serving.json
  (whole job) in .../<model>/<mode>/.  Mode directories: bf16, fp8, int4pth, rtn_a_w0, konlya_w128, ...
* resumable: a task with done.json is skipped; a job whose tasks are all done starts no server; reasoning outputs
  are additionally resumed per (problem, seed); lm-eval reruns a task from scratch (use --limit for checks).
* env: EVALTOOLS = prefix of a separate eval venv (EVALTOOLS/bin/lm_eval, EVALTOOLS/bin/python); default: `lm_eval` on
  PATH and this interpreter.  The servers are started by scripts/serve_mode.sh (env VLLM = vllm binary of the vLLM env).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SERVE = ROOT / "integrations/vllm_flagrot/scripts/serve_mode.sh"
SCRAPE = ROOT / "integrations/vllm_flagrot/scripts/metrics_scrape.py"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integrations/vllm_flagrot/scripts"))
from blockflag.paths import MODELS as MODEL_PATHS  # noqa: E402  (model key -> local directory or HF id)

_EV = os.environ.get("EVALTOOLS")
LM_EVAL = str(Path(_EV) / "bin/lm_eval") if _EV else "lm_eval"
EVPY = str(Path(_EV) / "bin/python") if _EV else sys.executable
ALIASES = {"flagrot-int4": "konlya:s4w128", "flagrot-int2-auto": "konlya2:s4w128:ca",    # = scripts/common.py ALIASES
           "flagrot-kivi-3.0": "kivi2:s4w128", "kivi-3.0": "kivi2id:s4w128",
           "flagrot-kivi-2.5": "kivi2p8ff:s4w128", "kivi-2.5": "kivi2p8ffid:s4w128",
           "flagrot-kivi-2.5-fixed": "kivi2p8ffs:s4w128"}
# thinking-mode sampling per model (reasoning suite): the model's generation_config.json (Qwen3-8B: 0.6 / 0.95 / 20)
SAMPLING = {"qwen38_27b": dict(temperature=1.0, top_p=0.95, top_k=20)}
PORT_BASE = int(os.environ.get("PORT_BASE", "8600"))
LOCK = threading.Lock()


def log(*a):
    with LOCK:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def mode_dir(mode: str) -> str:
    """bf16 | fp8 | int4pth; plugin modes <kv>_w<W> (suffix ':W' / ':wW') or <kv>_s<S>_w<W> (':sS', ':sSwW');
    a trailing ':nopc' (prefix caching off, e.g. bf16:nopc) adds '_nopc'."""
    if mode.endswith(":nopc"):
        return mode_dir(mode[:-len(":nopc")]) + "_nopc"
    if mode.endswith(":mqoff"):
        return mode_dir(mode[:-len(":mqoff")]) + "_mqoff"
    for fl in ("c", "ca"):              # int2c: static key centering (post | auto; serve_mode.sh handles them)
        if mode.endswith(":" + fl):
            return mode_dir(mode[:-len(fl) - 1]) + "_" + fl
        if f":{fl}:" in mode:           # before other flags: same directory as with the flag last
            return mode_dir(mode.replace(f":{fl}:", ":", 1) + ":" + fl)
    kv, _, suf = mode.partition(":")
    if kv in ("bf16", "fp8", "int4pth"):
        return kv
    if not suf or suf.isdigit():
        return f"{kv}_w{suf or 0}"
    m = re.fullmatch(r"(?:s(\d+))?(?:w(\d+))?", suf)
    if not m:
        raise ValueError(f"bad mode suffix {suf!r}")
    s, w = int(m.group(1) or 0), int(m.group(2) or 0)
    return f"{kv}_s{s}_w{w}" if s else f"{kv}_w{w}"


def env_clean():
    e = dict(os.environ)
    for k in ("all_proxy", "ALL_PROXY"):
        e.pop(k, None)
    e["no_proxy"] = e["NO_PROXY"] = "127.0.0.1,localhost," + e.get("no_proxy", "")
    return e


def host_id(gpu) -> dict:
    """host and GPU model of a job (results from different hardware are kept apart in the serving tables)."""
    name = ""
    try:
        name = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        pass
    return dict(host=os.uname().nodename, gpu_name=name, gpu_class=os.environ.get("GPU_CLASS", ""))


def sys_load() -> dict:
    """host CPU load and free memory when a task starts (CPU contention distorts throughput / server start)."""
    d = {}
    try:
        d["loadavg_1_5_15"] = [round(x, 1) for x in os.getloadavg()]
        d["n_cpu"] = os.cpu_count()
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                d["mem_available_gb"] = round(int(line.split()[1]) / 2**20, 1)
    except Exception:
        pass
    return d


def kernel_so_id() -> dict:
    """md5 + mtime of the plugin kernel library this host serves with (provenance of plugin-mode results)."""
    import hashlib
    f = ROOT / "integrations/vllm_flagrot/build/libflagrot_kernels.so"
    try:
        d = dict(md5=hashlib.md5(f.read_bytes()).hexdigest(), mtime=time.strftime("%F %T", time.localtime(f.stat().st_mtime)))
    except OSError:
        return {}
    mq = f.with_name("libflagrot_mq.so")       # Phase 2i INT4 prefix-cached prefill library (prefill_mq=auto)
    if mq.exists():
        d["mq_md5"] = hashlib.md5(mq.read_bytes()).hexdigest()
    return d


def healthy(port: int) -> bool:
    try:
        op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return op.open(f"http://127.0.0.1:{port}/health", timeout=5).status == 200
    except Exception:
        return False


def start_server(model, mode, gpu, port, srv: dict, run_dir: Path) -> bool:
    env = env_clean()
    parts = mode.split(":")
    center = next((f for f in ("c", "ca") if f in parts[1:]), "")   # int2c: ':c' / ':ca' anywhere among the flags
    if center:
        mode = ":".join(x for i, x in enumerate(parts) if i == 0 or x != center)
    if mode.endswith(":mqoff"):         # plugin modes on the old INT4 prefix-cached prefill path (Phase 2i off)
        mode = mode[:-len(":mqoff")]
        env["FLAGROT_PREFILL_MQ"] = "off"
    if mode.endswith(":nopc"):          # control modes with prefix caching off (like-for-like with the window modes)
        mode = mode[:-len(":nopc")]
        env["PREFIX_CACHE"] = "0"
    env.update(MAX_MODEL_LEN=str(srv.get("max_model_len", 32768)), GPU_MEM=str(srv.get("gpu_mem", 0.9)),
               MAX_NUM_SEQS=str(srv.get("max_num_seqs", 128)), RUN_DIR=str(run_dir),
               TOOLS=str(srv.get("tools", 0)), REASONING=str(srv.get("reasoning", 0)))
    if "think" in srv:
        env["THINK"] = str(srv["think"])
    if "max_batched" in srv:
        env["MAX_BATCHED"] = str(srv["max_batched"])
    if srv.get("cg_sizes"):      # fewer CUDA-graph sizes: faster startup (accuracy suites; keep default for throughput)
        env["CG_SIZES"] = " ".join(str(x) for x in srv["cg_sizes"])
    run_dir.mkdir(parents=True, exist_ok=True)
    if center:
        mode += ":" + center
    p = subprocess.run(["bash", str(SERVE), model, mode, str(gpu), str(port)], env=env, capture_output=True, text=True)
    log(f"[gpu {gpu}] {model} {mode}: {p.stdout.strip()[-300:]}")
    return p.returncode == 0


def stop_server(port):
    subprocess.run(["bash", str(SERVE), "stop", str(port)], env=env_clean())


def scrape(port, out=None, raw=None, since=None, log_=None, samples=None, meta=None):
    cmd = [sys.executable, str(SCRAPE), "--port", str(port)]
    for k, v in (("--out", out), ("--raw", raw), ("--since", since), ("--log", log_), ("--samples", samples)):
        if v:
            cmd += [k, str(v)]
    if meta:
        cmd += ["--meta", json.dumps(meta)]
    subprocess.run(cmd, env=env_clean(), capture_output=True)


# ---------------------------------------------------------------------------------------------------- task commands
def task_cmd(t: dict, model: str, port: int, out: Path, a, user_port=None) -> list[str] | str:
    url = f"http://127.0.0.1:{port}/v1"
    lim = a.limit or t.get("limit")
    k = t["kind"]
    if k == "lmeval":
        chat = t.get("api", "chat") == "chat"
        margs = (f"model={model},base_url={url}/{'chat/completions' if chat else 'completions'},"
                 f"num_concurrent={t.get('concurrency', 32)},max_retries=3,tokenized_requests=False,"
                 f"tokenizer_backend=huggingface,tokenizer={MODEL_PATHS[model]},timeout=3600,"
                 f"max_length={a.max_model_len}")
        cmd = [LM_EVAL, "--model", "local-chat-completions" if chat else "local-completions",
               "--model_args", margs, "--tasks", ",".join(t["tasks"]), "--output_path", str(out), "--log_samples",
               "--seed", "0"]
        if chat:
            cmd += ["--apply_chat_template"]
        if t.get("fewshot_as_multiturn"):
            cmd += ["--fewshot_as_multiturn"]
        if "num_fewshot" in t:
            cmd += ["--num_fewshot", str(t["num_fewshot"])]
        if t.get("include_path"):        # extra lm-eval task configs (e.g. lmeval_tasks/ruler_fit, lb1_tasks), relative to eval/
            cmd += ["--include_path", str(HERE / t["include_path"])]
        if t.get("gen_kwargs"):
            cmd += ["--gen_kwargs", t["gen_kwargs"]]
        md = dict(t.get("metadata", {}))
        if md:
            md.setdefault("pretrained", MODEL_PATHS[model])
            cmd += ["--metadata", json.dumps(md)]
        if lim:
            cmd += ["--limit", str(lim)]
        return cmd
    if k == "reasoning":
        cmd = [EVPY, str(HERE / "reasoning.py"), "--base-url", url, "--model", model,
               "--task", t["task"], "--seeds", str(a.seeds or t.get("seeds", "0,1,2")),
               "--max-tokens", str(t.get("max_tokens", 32768)), "--concurrency", str(t.get("concurrency", 64)),
               "--out", str(out / f"{t['task']}.jsonl")]
        for k_, v_ in SAMPLING.get(model, {}).items():
            cmd += [f"--{k_.replace('_', '-')}", str(v_)]
        if lim:
            cmd += ["--limit", str(lim)]
        return cmd
    raise ValueError(k)


def task_done(t: dict, jdir: Path, a) -> bool:
    """done.json exists and, for reasoning tasks, was produced with the seeds the config asks for now."""
    f = jdir / t["name"] / "done.json"
    if not f.exists():
        return False
    if t.get("kind") == "reasoning":
        try:
            cmd = json.loads(f.read_text()).get("cmd", [])
            want = str(a.seeds or t.get("seeds", "0,1,2"))
            if "--seeds" in cmd and cmd[cmd.index("--seeds") + 1] != want:
                return False
        except Exception:
            pass
    return True


def run_job(job, gpu, a, cfg, user_port):
    model, mode, tasks, jdir = job["model"], job["mode"], job["tasks"], job["dir"]
    todo = [t for t in tasks if not task_done(t, jdir, a)]
    if (jdir / "SKIP").exists():     # job moved to another launcher / host (rebalancing)
        log(f"[gpu {gpu}] {model} {mode}: SKIP file, skipped")
        return
    if not todo:
        log(f"[gpu {gpu}] {model} {mode}: all done")
        return
    if a.task_claims and not a.dry_run:
        # task-level sharing: several launchers may serve the same (model, mode) on different GPUs and split its tasks;
        # a job-level .claim (launcher without --task-claims) owns the whole job
        if (jdir / ".claim").exists():
            log(f"[gpu {gpu}] {model} {mode}: job claimed by a job-level launcher, skipped")
            return
        jdir.mkdir(parents=True, exist_ok=True)
        # claim the first open task BEFORE starting the server (otherwise several launchers reaching the job together
        # each start a server and all but one find nothing left to run)
        pre = None
        for t in todo:
            try:
                (jdir / f".claim_{t['name']}").mkdir()
                (jdir / f".claim_{t['name']}" / "owner").write_text(
                    f"{os.uname().nodename} pid {os.getpid()} gpu {gpu} {time.strftime('%F %T')}")
                pre = t["name"]
                break
            except FileExistsError:
                continue
        if pre is None:
            log(f"[gpu {gpu}] {model} {mode}: all open tasks claimed, skipped")
            return
        try:
            _run_job(job, gpu, a, cfg, user_port, todo, pre)
        finally:
            import shutil
            shutil.rmtree(jdir / f".claim_{pre}", ignore_errors=True)
        return
    if not a.dry_run:     # several launchers may share one job list on one host: first to claim the job runs it
        try:
            jdir.mkdir(parents=True, exist_ok=True)
            (jdir / ".claim").mkdir()
            (jdir / ".claim/owner").write_text(f"{os.uname().nodename} pid {os.getpid()} gpu {gpu} {time.strftime('%F %T')}")
        except FileExistsError:
            log(f"[gpu {gpu}] {model} {mode}: claimed by another launcher, skipped")
            return
    try:
        _run_job(job, gpu, a, cfg, user_port, todo)
    finally:
        if not a.dry_run:
            import shutil
            shutil.rmtree(jdir / ".claim", ignore_errors=True)


def _run_job(job, gpu, a, cfg, user_port, todo, pre=None):
    model, mode, jdir = job["model"], job["mode"], job["dir"]
    port = PORT_BASE + int(gpu)
    srv = dict(cfg.get("server", {}))
    if a.dry_run:
        for t in todo:
            log(f"[dry] gpu {gpu} {model} {mode} {t['name']}: " + " ".join(
                shlex.quote(x) for x in task_cmd(t, model, port, jdir / t["name"], a, user_port)))
        return
    sdir = jdir / (f"server_gpu{gpu}" if a.task_claims else "server")
    t_srv = time.time()
    if not start_server(model, mode, gpu, port, srv, sdir):
        (jdir / "FAILED_SERVER").write_text(time.strftime("%F %T"))
        return
    srv_up = time.time() - t_srv
    scrape(port, raw=sdir / "metrics_start.json")
    try:
        for t in todo:
            tdir = jdir / t["name"]
            tclaim = jdir / f".claim_{t['name']}"
            if a.task_claims and t["name"] != pre:
                if task_done(t, jdir, a):
                    continue
                try:
                    tclaim.mkdir()
                    (tclaim / "owner").write_text(f"{os.uname().nodename} pid {os.getpid()} gpu {gpu} {time.strftime('%F %T')}")
                except FileExistsError:
                    continue
            tdir.mkdir(parents=True, exist_ok=True)
            cmd = task_cmd(t, model, port, tdir, a, user_port)
            raw0 = tdir / "metrics_before.json"
            scrape(port, raw=raw0)
            load0 = sys_load()
            t0 = time.time()
            log(f"[gpu {gpu}] {model} {mode} {t['name']}: start")
            env = env_clean()
            env.update(OPENAI_API_KEY="EMPTY", HF_DATASETS_TRUST_REMOTE_CODE="1")
            with open(tdir / "run.log", "a") as fh:
                rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=str(tdir)).returncode
            wall = time.time() - t0
            scrape(port, out=jdir / f"serving_{t['name']}.json", since=raw0, log_=sdir / "server.log",
                   samples=sdir / "samples.jsonl",
                   meta=dict(suite=cfg["suite"], model=model, mode=mode, task=t["name"], wall_s=wall, sys_at_start=load0, sys_at_end=sys_load(),
                             **host_id(gpu)))
            if rc == 0:
                (tdir / "done.json").write_text(json.dumps(dict(
                    wall_s=round(wall, 1), server_startup_s=round(srv_up, 1), gpu=gpu, limit=a.limit or t.get("limit"),
                    cmd=cmd, finished=time.strftime("%F %T"), kernel_so=kernel_so_id(), **host_id(gpu)), indent=1))
            log(f"[gpu {gpu}] {model} {mode} {t['name']}: rc={rc} wall {wall:.0f} s")
            if a.task_claims:
                import shutil
                shutil.rmtree(tclaim, ignore_errors=True)
    finally:
        scrape(port, out=jdir / ("serving.json" if not a.task_claims else f"serving_job_gpu{gpu}.json"),
               since=sdir / "metrics_start.json", log_=sdir / "server.log", samples=sdir / "samples.jsonl",
               meta=dict(suite=cfg["suite"], model=model, mode=mode, server_startup_s=srv_up, **host_id(gpu)))
        stop_server(port)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--config", help="config file stem in configs/ (default: --suite); results go to the config's suite dir")
    ap.add_argument("--gpus", required=True, help="comma list of free GPUs (one server per GPU)")
    ap.add_argument("--only-gpus", default="", help="queue lines for particular pairs: exit unless --gpus is in this list "
                    "(a third throughput run goes to a pair that point has not used yet)")
    ap.add_argument("--models")
    ap.add_argument("--modes")
    ap.add_argument("--tasks")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seeds", help="override the seeds of reasoning tasks")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mode-major", action="store_true", help="job order: modes outer, models inner")
    ap.add_argument("--task-claims", action="store_true",
                    help="claim single tasks instead of whole jobs (several GPUs can split one mode's tasks)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(HERE / "configs" / f"{a.config or a.suite}.yaml"))
    a.max_model_len = cfg.get("server", {}).get("max_model_len", 32768)
    models = a.models.split(",") if a.models else cfg["models"]
    modes = [ALIASES.get(m, m) for m in (a.modes.split(",") if a.modes else [str(m) for m in cfg["modes"]])]
    tasks = [t for t in cfg["tasks"] if not a.tasks or t["name"] in a.tasks.split(",")]
    base = ROOT / "results/tasks" / (cfg["suite"] + (f"_{a.tag}" if a.tag else ""))
    pairs = [(m, k) for k in modes for m in models] if a.mode_major else [(m, k) for m in models for k in modes]
    jobs = [dict(model=m, mode=k, tasks=tasks, dir=base / m / mode_dir(k)) for m, k in pairs]
    gpus = a.gpus.split(",")
    if a.only_gpus and not set(gpus) & set(a.only_gpus.split(",")):
        return
    user_port = None
    for j in jobs:
        j["dir"].mkdir(parents=True, exist_ok=True)
    free = list(gpus)
    cv = threading.Condition()
    threads = []

    def worker(job, gpu):
        try:
            run_job(job, gpu, a, cfg, user_port)
        except Exception as e:
            log(f"[gpu {gpu}] {job['model']} {job['mode']} failed: {e!r}")
        finally:
            with cv:
                free.append(gpu)
                cv.notify_all()

    for job in jobs:
        with cv:
            while not free:
                cv.wait()
            gpu = free.pop(0)
        th = threading.Thread(target=worker, args=(job, gpu), daemon=False)
        th.start()
        threads.append(th)
        if a.dry_run:
            th.join()
    for th in threads:
        th.join()
    log("SUITE_DONE", cfg["suite"])


if __name__ == "__main__":
    main()
