"""Phase 2k: vLLM prefix caching for the FlagRot caches with BF16 sink tokens / a BF16 window (S + W > 0).

Data layout (attention/backend.py): the INT4 pages hold every token at its real position (written when the token is
computed, as with W = S = 0), so every full block is a valid prefix-cache block.  The per-request BF16 row holds the
S sinks and the newest W tokens; the attention kernels read the pages over [S, i - W] (kv_start / kmin masks) and the
BF16 row for the rest.  A request that reuses a cached prefix lacks two things, which this module provides:

  window  the BF16 copies of the last W tokens before its first computed token.  The scheduler hook caps every
          prefix-cache hit (KVCacheCoordinator.find_longest_cache_hit with a reduced max length, so the block /
          hybrid-group alignment is kept): option `pc_recompute` = "min" (default) - the reused prefix ends at <=
          num_tokens - W, i.e. every request computes at least its last W prompt tokens (nothing changes when the
          new segment is >= W tokens); an integer R - recompute >= R tokens before the original hit; 0 - keep the
          whole hit.  The window positions before the request's first computed token rv are read from the INT4 pages:
          by the INT4-direct kernel through a per-tile extension (every row also sees the page keys <= rv - 1), or
          dequantized into the BF16 rows for "fa"-path chunks and a first decode window reaching below rv.  Only the
          queries of the first <= W computed tokens read them - the one approximation of the design; with "min" the last prompt token and every generated token see their whole window in BF16.  Hybrid models (Mamba groups keep states at sparse positions only: a capped hit finds none)
          default to 0.
  sinks   the BF16 sinks depend on the first S token ids only.  A small GPU pool of BF16 sink rows per layer
          (`sink_pool` slots, LRU, keyed by (LoRA id, first S token ids)) is written by every request that computes
          its sinks and read into the BF16 row of a request whose first computed token is past the sinks.  The pool's
          index is updated by the same deterministic rule in the scheduler (a mirror, used to decide whether a hit
          may keep its prefix) and in every worker (used to address the pool): the events of a step are derived from
          its SchedulerOutput alone (reads: new requests with num_computed_tokens >= S; writes: requests whose
          scheduled chunk covers position S - 1), applied once per step in a canonical order (reads before writes,
          keys sorted).  A hit whose key is not in the pool keeps no prefix (computed from scratch): correctness
          never depends on the pool.

Hooks (installed by vllm_flagrot.register() in every process; inactive unless the FlagRot options have S + W > 0 and
prefix caching is on):
  KVCacheManager.get_computed_blocks   window cap + sink-pool check (scheduler)
  Scheduler.__init__ / schedule        mirror index, per-step update, statistics (log + FLAGROT_PC_STATS json file)
  GPUModelRunner.execute_model / update_requests (model runner V2)   per-step pool reads / writes and the
                                       requests that reuse a prefix (their first computed position) (worker)
"""
from __future__ import annotations

import json
import os
import time
from collections import OrderedDict


SINK_POOL_DEFAULT = 256
# pc_recompute = "min": the reused prefix ends at <= num_tokens - W, i.e. every request computes at least its last W
# prompt tokens (recomputing the shortfall when the new segment is shorter than W): the last prompt token and every
# generated token see their whole window in BF16; prompt tokens within W of the reuse point read the positions
# before it from the pages.  Default for models without Mamba groups.
MIN_W = -1


def pc_config(vllm_config) -> dict | None:
    """{"S", "W", "R", "N"} when the FlagRot backend with BF16 slots runs with prefix caching, else None."""
    try:
        add = getattr(vllm_config, "additional_config", None) or {}
        fr = add.get("flagrot") if isinstance(add, dict) else None
        if not fr:
            return None
        if not getattr(vllm_config.cache_config, "enable_prefix_caching", False):
            return None
        from .attention.backend import flagrot_options
        o = flagrot_options(vllm_config)
    except Exception:  # noqa: BLE001
        return None
    S, W = o["sink"], o["window"]
    if S + W == 0:
        return None
    # tokens recomputed before a reused prefix: None = auto (W; 0 on hybrid models, decided by the scheduler hook,
    # whose Mamba groups keep states only at sparse positions); an explicit value (0 = window rows from the pages)
    R = o.get("pc_recompute")
    R = None if R in (None, "", "auto") else (MIN_W if R in ("min", "last") else int(R))
    if R is not None and R < 0 and R != MIN_W:
        raise ValueError(f"FlagRot: pc_recompute must be 'auto', 'min' or >= 0 (got {R})")
    # kivi2: KIVI key groups [32 g, 32 g + 32) are encoded once complete, so a reused prefix must end at a group
    # boundary (a multiple of 32): the request then never needs the BF16 keys of a group it did not compute
    return dict(S=S, W=W, R=R, N=int(o.get("sink_pool") or SINK_POOL_DEFAULT),
                align=32 if o.get("kv_cache_dtype") == "flagrot_kivi" else 1)


# ------------------------------------------------------------------------------------------------ sink-pool index
def _lora_id(lr) -> int:
    return int(getattr(lr, "lora_int_id", 0) or 0) if lr is not None else 0


def key_of(prompt_token_ids, mm_features, lora_request, S: int):
    """pool key of a request: (LoRA id, first S prompt token ids); None = never shared (no token ids, a prompt
    shorter than S, multimodal inputs whose embeddings the token ids do not determine)."""
    if S <= 0 or prompt_token_ids is None or len(prompt_token_ids) < S or mm_features:
        return None
    return (_lora_id(lora_request),) + tuple(int(t) for t in prompt_token_ids[:S])


class SinkIndex:
    """LRU map key -> pool slot.  apply() = the one state transition per step (identical in the scheduler mirror and
    in the workers)."""

    def __init__(self, nslots: int):
        self.n = int(nslots)
        self.map: dict = {}
        self.lru: OrderedDict = OrderedDict()
        self.free = list(range(self.n))[::-1]

    def slot(self, key):
        return self.map.get(key)

    def apply(self, reads, writes) -> dict:
        """reads: keys read this step (LRU touch), writes: keys written this step -> {key: slot} of the writes, in
        sorted key order (the order the workers copy in).  A write never evicts another write of the same step."""
        for k in sorted(set(reads)):
            if k in self.lru:
                self.lru.move_to_end(k)
        out: dict = {}
        for k in sorted(set(writes)):
            if k in self.map:
                self.lru.move_to_end(k)
            elif self.free:
                self.map[k] = self.free.pop()
                self.lru[k] = None
            else:
                victim = next((v for v in self.lru if v not in out), None)
                if victim is None:
                    continue
                del self.lru[victim]
                self.map[k] = self.map.pop(victim)
                self.lru[k] = None
            out[k] = self.map[k]
        return out


class StepEvents:
    """per-process bookkeeping of the step events (request id -> key; the index)."""

    def __init__(self, S: int, N: int):
        self.S, self.N = S, N
        self.keys: dict = {}
        self.index = SinkIndex(N)

    def events(self, so):
        """SchedulerOutput -> (reads [(req_id, key, num_computed)], writes [(req_id, key)])."""
        S = self.S
        reads, writes = [], []
        for nr in so.scheduled_new_reqs:
            key = key_of(nr.prompt_token_ids, nr.mm_features, nr.lora_request, S)
            self.keys[nr.req_id] = key
            if key is None or S == 0:
                continue
            nct = int(nr.num_computed_tokens)
            n = int(so.num_scheduled_tokens.get(nr.req_id, 0))
            if nct >= S:
                reads.append((nr.req_id, key, nct))
            elif nct < S <= nct + n:
                writes.append((nr.req_id, key))
        cr = so.scheduled_cached_reqs
        if S > 0:
            for req_id, nct in zip(cr.req_ids, cr.num_computed_tokens):
                key = self.keys.get(req_id)
                n = int(so.num_scheduled_tokens.get(req_id, 0))
                if key is not None and nct < S <= nct + n:
                    writes.append((req_id, key))
        for req_id in list(so.finished_req_ids or ()) + list(getattr(so, "preempted_req_ids", None) or ()):
            self.keys.pop(req_id, None)
        return reads, writes

    def step(self, so):
        """events of this step + the index transition: (reads [(req_id, slot, num_computed)] addressed with the index
        state BEFORE this step, writes [(req_id, slot)] in copy order, missing reads [req_id])."""
        reads, writes = self.events(so)
        r_out, miss = [], []
        for req_id, key, nct in reads:
            sl = self.index.slot(key)
            (r_out.append((req_id, sl, nct)) if sl is not None else miss.append(req_id))
        wmap = self.index.apply([k for _, k, _ in reads], [k for _, k in writes])
        w_out = [(req_id, wmap[k]) for req_id, k in sorted(writes, key=lambda x: x[1]) if k in wmap]
        return r_out, w_out, miss


# ------------------------------------------------------------------------------------------------ scheduler side
class SchedulerPC:
    def __init__(self, cfg: dict, block_size: int):
        self.cfg = cfg
        self.S, self.W, self.R = cfg["S"], cfg["W"], cfg["R"]
        self.ev = StepEvents(self.S, cfg["N"])
        self.block_size = block_size
        # lookups: per get_computed_blocks call (a waiting request may be looked up several times);
        # sched_*: per scheduled request (counted once, when it is admitted)
        self.stats = dict(lookups=0, hits=0, hit_tokens=0, kept_tokens=0, capped_tokens=0, sink_lookups=0,
                          sink_hits=0, sink_misses=0, no_key=0, pool_reads=0, pool_writes=0, steps=0,
                          sched_reqs=0, sched_hit_reqs=0, sched_hit_tokens=0, sched_kept_tokens=0,
                          sched_sink_lookups=0, sched_sink_hits=0, sched_prompt_tokens=0)
        self.path = os.environ.get("FLAGROT_PC_STATS")
        self._t = 0.0

    def dump(self, force=False):
        now = time.time()
        if not force and now - self._t < 30:
            return
        self._t = now
        s = dict(self.stats, S=self.S, W=self.W, R=self.R, N=self.ev.N, pool_used=len(self.ev.index.map),
                 sink_hit_rate=(self.stats["sched_sink_hits"] / self.stats["sched_sink_lookups"])
                 if self.stats["sched_sink_lookups"] else None, time=now)
        if self.path:
            try:
                tmp = self.path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(s, f)
                os.replace(tmp, self.path)
            except OSError:
                pass
        from vllm.logger import init_logger
        init_logger("vllm.plugins.flagrot").info(
            "FlagRot prefix cache (S %d, W %d): %d of %d scheduled requests hit the cache, tokens reused %d of %d "
            "hit; sink pool %d / %d hits; pool %d / %d slots", self.S, self.W, self.stats["sched_hit_reqs"],
            self.stats["sched_reqs"], self.stats["sched_kept_tokens"], self.stats["sched_hit_tokens"],
            self.stats["sched_sink_hits"], self.stats["sched_sink_lookups"], len(self.ev.index.map), self.ev.N)


def _wrap_get_computed_blocks(orig):
    def get_computed_blocks(self, request):
        res = orig(self, request)
        pc = getattr(self, "_flagrot_pc", None)
        if pc is None:
            return res
        blocks, n, spb = res
        if n <= 0:
            return res
        st = pc.stats
        st["lookups"] += 1
        st["hit_tokens"] += n
        empty = (self.empty_kv_cache_blocks, 0, 0)
        if pc.R == MIN_W:                               # compute at least the last W tokens of the request
            cap = min(n, request.num_tokens - pc.W)
        else:                                           # recompute R tokens before the hit (0: none)
            cap = n - pc.R + 1 if pc.R > 0 else n
        al = pc.cfg.get("align", 1)
        if al > 1:                                      # kivi2: reuse points at key-group boundaries
            cap = cap // al * al
        if 0 < cap < pc.S:                              # a prefix ending inside the sinks: nothing kept
            cap = 0
        sink = None                                     # sink-pool outcome (None: not needed)
        if pc.S > 0 and cap >= pc.S:                   # a kept prefix past the sinks needs the pooled BF16 sinks
            key = key_of(request.prompt_token_ids, request.mm_features, request.lora_request, pc.S)
            st["sink_lookups"] += 1
            sink = key is not None and pc.ev.index.slot(key) is not None
            if not sink:
                st["sink_misses"] += 1
                st["no_key"] += int(key is None)
                cap = 0
            else:
                st["sink_hits"] += 1
        if cap <= 0:
            st["capped_tokens"] += n
            request._flagrot_pc = (n, 0, sink)
            return empty
        if cap >= n:
            st["hits"] += 1
            st["kept_tokens"] += n
            request._flagrot_pc = (n, n, sink)
            return res
        computed, n2, unc = self.coordinator.find_longest_cache_hit(request.block_hashes, cap)
        while al > 1 and n2 % al and n2 > 0:           # block-aligned hit inside a key group: shorter hit
            computed, n2, unc = self.coordinator.find_longest_cache_hit(request.block_hashes, n2 // al * al)
        if n2 < max(pc.S, 1):
            st["capped_tokens"] += n
            request._flagrot_pc = (n, 0, sink)
            return empty
        st["hits"] += 1
        st["kept_tokens"] += n2
        st["capped_tokens"] += n - n2
        request._flagrot_pc = (n, n2, sink)
        return self.create_kv_cache_blocks(computed), n2, (n2 + unc if unc else 0)
    get_computed_blocks._flagrot = True
    return get_computed_blocks


def _wrap_scheduler_init(orig):
    def __init__(self, *args, **kw):
        orig(self, *args, **kw)
        cfg = pc_config(getattr(self, "vllm_config", None))
        if cfg is None:
            return
        hybrid = bool(getattr(getattr(self.kv_cache_manager, "kv_cache_config", None), "has_mamba_layers", False))
        if cfg["R"] is None:
            # hybrid models (Qwen3.5 / 3.8): the Mamba groups keep states at sparse positions only (the end of a
            # request's computed blocks, junctions); a hit capped below them finds no state and would drop the whole
            # prefix (measured: 5 % of the hit kept), so the window rows come from the pages instead (R = 0)
            cfg["R"] = 0 if hybrid else MIN_W
        cfg["hybrid"] = hybrid
        bs = getattr(getattr(self, "cache_config", None), "block_size", 16)
        self._flagrot_pc = SchedulerPC(cfg, bs)
        self.kv_cache_manager._flagrot_pc = self._flagrot_pc
        from vllm.logger import init_logger
        init_logger("vllm.plugins.flagrot").info(
            "FlagRot prefix caching with BF16 slots (Phase 2k): sinks %d, window %d, %s%s, sink pool %d slots",
            cfg["S"], cfg["W"], "every request computes >= its last W tokens" if cfg["R"] == MIN_W else
            f"recompute >= {cfg['R']} tokens before a reused prefix", " (hybrid model)" if cfg["hybrid"] else "",
            cfg["N"])
    __init__._flagrot = True
    return __init__


def _wrap_schedule(orig):
    def schedule(self, *args, **kw):
        so = orig(self, *args, **kw)
        pc = getattr(self, "_flagrot_pc", None)
        if pc is not None:
            reads, writes, miss = pc.ev.step(so)
            if miss:   # cannot happen: hits were checked against the same index state
                raise RuntimeError(f"FlagRot prefix cache: sink pool miss for scheduled hits {miss}")
            for nr in so.scheduled_new_reqs:
                req = self.requests.get(nr.req_id)
                info = getattr(req, "_flagrot_pc", None)
                pc.stats["sched_reqs"] += 1
                pc.stats["sched_prompt_tokens"] += int(getattr(req, "num_prompt_tokens", 0) or 0)
                if info is None:
                    continue
                n, n2, sink = info
                req._flagrot_pc = None
                pc.stats["sched_hit_reqs"] += 1
                pc.stats["sched_hit_tokens"] += n
                pc.stats["sched_kept_tokens"] += n2
                if sink is not None:
                    pc.stats["sched_sink_lookups"] += 1
                    pc.stats["sched_sink_hits"] += int(bool(sink))
            pc.stats["pool_reads"] += len(reads)
            pc.stats["pool_writes"] += len(writes)
            pc.stats["steps"] += 1
            pc.dump()
        return so
    schedule._flagrot = True
    return schedule


# ------------------------------------------------------------------------------------------------ worker side
class WorkerPC:
    """the current step's sink-pool copies and requests that reuse a cached prefix (read by the backend)."""

    def __init__(self, cfg: dict, max_slots: int):
        self.cfg = cfg
        self.S, self.W = cfg["S"], cfg["W"]
        self.ev = StepEvents(self.S, cfg["N"])
        self.step_id = 0
        self.reads: list = []                                 # [(req slot, pool slot)]
        self.writes: list = []                                # [(req slot, pool slot)] in copy order
        self.hits: dict = {}                                  # req slot -> (first computed position, prefill tokens)
        # per request slot: first position whose BF16 row exists (0 = all; a request reusing a cached prefix: its
        # first computed position, until the backend fills the earlier window rows from the pages)
        self.rv = [0] * (max_slots + 1)
        self.total = [0] * (max_slots + 1)                    # prefill tokens of the slot's request
        self.stats = dict(reads=0, writes=0, hit_reqs=0, hit_tokens=0)

    def clear(self):
        self.reads, self.writes, self.hits = [], [], {}

    def update(self, so, req_states):
        idx = req_states.req_id_to_index
        S, W = self.S, self.W
        reads, writes, miss = self.ev.step(so)
        if miss:
            raise RuntimeError("FlagRot prefix cache: a request reuses a cached prefix past the sink tokens but the "
                               f"sink pool has no entry for it ({miss}); is the scheduler-side hook active "
                               "(vllm_flagrot.register)?")
        hits = {}
        for nr in so.scheduled_new_reqs:
            sl = idx.get(nr.req_id)
            nct = int(nr.num_computed_tokens)
            if sl is not None:
                self.rv[sl] = nct
                self.total[sl] = len(nr.prefill_token_ids) if nr.prefill_token_ids is not None else nr.prompt_len
            if sl is None or nct <= 0:
                continue
            if 0 < nct < S:
                raise RuntimeError(f"FlagRot prefix cache: request {nr.req_id} reuses {nct} tokens, fewer than the "
                                   f"{S} sinks (scheduler-side hook inactive?)")
            total = len(nr.prefill_token_ids) if nr.prefill_token_ids is not None else nr.prompt_len
            hits[sl] = (nct, total)   # its window positions before nct: read from the pages (backend)
            self.stats["hit_reqs"] += 1
            self.stats["hit_tokens"] += nct
        self.hits = hits
        self.reads = [(idx[r], p) for r, p, _ in reads if r in idx]
        self.writes = [(idx[r], p) for r, p in writes if r in idx]
        self.stats["reads"] += len(self.reads)
        self.stats["writes"] += len(self.writes)
        self.step_id += 1


WORKER: WorkerPC | None = None


def _wrap_execute_model(orig):
    def execute_model(self, *args, **kw):
        if WORKER is not None:
            WORKER.clear()
        return orig(self, *args, **kw)
    execute_model._flagrot = True
    return execute_model


def _wrap_update_requests(orig):
    def update_requests(self, scheduler_output, *args, **kw):
        out = orig(self, scheduler_output, *args, **kw)
        global WORKER
        if WORKER is None and not getattr(self, "_flagrot_pc_checked", False):
            self._flagrot_pc_checked = True
            cfg = pc_config(getattr(self, "vllm_config", None))
            if cfg is not None:
                WORKER = WorkerPC(cfg, self.req_states.max_num_reqs)
        if WORKER is not None:
            WORKER.update(scheduler_output, self.req_states)
        return out
    update_requests._flagrot = True
    return update_requests


def _patch(cls, name, wrap):
    f = getattr(cls, name, None)
    if f is None or getattr(f, "_flagrot", False):
        return
    setattr(cls, name, wrap(f))


def install() -> None:
    """scheduler-side hooks (engine core: vllm_flagrot.register() runs there before the scheduler is built);
    idempotent, inactive unless pc_config() of the scheduler's config is set."""
    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        _patch(KVCacheManager, "get_computed_blocks", _wrap_get_computed_blocks)
        from vllm.v1.core.sched.scheduler import Scheduler
        _patch(Scheduler, "__init__", _wrap_scheduler_init)
        _patch(Scheduler, "schedule", _wrap_schedule)
        try:
            from vllm.v1.core.sched.async_scheduler import AsyncScheduler
            if "schedule" in AsyncScheduler.__dict__:
                _patch(AsyncScheduler, "schedule", _wrap_schedule)
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        from vllm.logger import init_logger
        init_logger("vllm.plugins.flagrot").warning("FlagRot: scheduler hooks not installed (%s)", e)


def install_worker() -> None:
    """worker-side hooks (model runner V2); called by the attention impl's constructor (worker process, before the
    first step).  Idempotent."""
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    _patch(GPUModelRunner, "execute_model", _wrap_execute_model)
    _patch(GPUModelRunner, "update_requests", _wrap_update_requests)
