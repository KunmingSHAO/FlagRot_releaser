"""FlagRot INT4 KV-cache attention backend for vLLM v1: CUDA kernels, CUDA graphs, BF16 sinks / window (prefix caching:
Phase 2k).

Selection (plugin route; vLLM's CacheDType is a closed Literal, see README):
    --attention-backend CUSTOM --kv-cache-dtype auto --additional-config '{"flagrot": {
        "kv_cache_dtype": "flagrot_int4" | "flagrot_int4_static",   codebook (default) | static FlagRot transforms
        "calib": "/path/flagrot_calib_<m>.pt" | "/path/flagrot_static_<m>.pt",
        "window": 0 | 32 | 64 | 128 | 256,                          BF16 staging window W (newest W tokens in BF16)
        "prefill": "fa" | "kernel",                                 prefill attention path (see below)
        "prefill_fp8": "off" | "had" | "unit" | "amax" | "int4" | "int4ac",   "fa" path prefill attention: BF16
                                                                    (default) | FP8 (Phase 2e) | INT4-direct (2f)
        "quant": "int4" | "int4a" | "int2a"}}'                      INT4 codes: symmetric (default) | asymmetric
                                                                    int4a:32 (static and key-only 2x2 only) |
                                                                    asymmetric INT2 int2a:32 (Phase 3a, idem)

Formats (one contiguous chunk per (page, KV head), the layouts of kernels/csrc; vLLM layout LBHNC, page = 16 tokens):
  flagrot_int4_static  v2 INT4 chunk, 144 B / (token, KV head): per (layer, KV head, K/V) the static choice of the
                       research config kv_flags=headsel, kv_rb=32 (none | rb-32 block Hadamards | whole-head H128).
                       Block flags are applied by the kernels (writer: before quantization; attention: q rotated in
                       the prologue, output back-rotated at the split merge).  Whole-head H128 heads are rotated
                       outside the kernels with exact power-of-two scaled +-1 matrices: k, v -> H k / 16 before the
                       writer, q -> H q / 8 and o -> H o / 8 (q'.k' = q.k and o = v exactly; INT4 is scale-invariant).
  flagrot_int4         per-token codebook (locked "cq+f+S+Q5+FS+R5": 8 x 8 candidates, 15 side bits), 148 B: writer
                       = kernels/csrc/attn_codebook.cu (bit-identical to the research encoder up to the fp16 scale),
                       attention applies the inverse transforms in the kernel (original basis in and out).
  quant = "int4a"      asymmetric INT4 (research int4a:32: per 32-value group scale + zero point), stored as
                       blockflag int4ah:32 (unsigned codes, the zero point in the 4 low bits of the FP16 scale word;
                       same chunk layout and size): static -> v2 format A2_INT4A (variant 264),
                       key-only 2x2 cq+f+konly -> codebook kind konlya.  The 8x8 codebook has no asymmetric kernel.
  quant = "int2a"      Phase 3a: asymmetric INT2 (research int2a:32), stored as blockflag int2ah:32 (2-bit unsigned codes,
                       the zero point in the 2 low bits of the FP16 scale word, flags in the sign bits): 80 B per
                       (token, KV head) at head size 128 = 2.5 bits per element (kernels/csrc/attn_decode_v2.cuh INT2
                       layout).  static -> kind static_a2 (v2 format int2a; static calibration made for int2a:32, or an
                       all-zero RTN calibration), key-only 2x2 -> codebook kind konlya2 (same C_q / c_o / FlagRot-C file).

Data flow per layer (forward_includes_kv_cache_update = False):
  do_kv_cache_update   new tokens -> paged writer kernel at slot_mapping, i.e. at their real positions (since Phase
                       2k also with the BF16 slots).  W > 0 / S > 0: per request a BF16 row (indexed by the request slot
                       of the runner, CommonAttentionMetadata.req_idx) additionally keeps the S sink tokens and a ring of
                       the newest W tokens; the attention reads them there and the pages over [S, i - W] (the research
                       semantics kv_sink = S + kv_resid = W: every query sees its sinks and its last W keys in BF16).
                       Every full block is a valid prefix-cache block; prefix_cache.py recomputes >= W tokens before a
                       reused prefix and restores the sinks from a pool of BF16 sink rows.
  forward, decode rows the paged split-K decode kernel (static v2 / codebook) on the cached tokens [S, L - W) (kv_start =
                       S) + the BF16 sinks / window (one extra split CTA reading the row through tail_idx).  Fixed grid: the split plan comes
                       from max_model_len, so the step is CUDA-graph capturable (UNIFORM_SINGLE_TOKEN_DECODE).
  forward, prefill     "fa" (default): gather + dequantize the request's pages (PyTorch) to BF16 in the original
                       basis + FlashAttention varlen (W > 0: quantized keys j <= i - W and BF16 keys i - W < j <= i
                       as two FA calls merged by LSE).  prefill_fp8 (Phase 2e): the dequantized prefix goes to FP8
                       E4M3 workspaces instead of BF16 and FA3 runs with FP8 inputs + descales, vLLM's
                       kv_cache_dtype=fp8 call (q -> FP8 with vLLM's static per-tensor q scale 1.0):
                       "had" = q -> H q / 8 and K -> H K / 16 (whole-head Hadamard, q'.k' = q.k) before
                       the FP8 cast, K / V scale 1.0; "unit" = no rotation, K / V scale 1.0 (vLLM's default k / v
                       scale); "amax" = no rotation, dynamic per-(request, KV head) scale amax / 448; "off" = BF16.
                       The FP8 error is in q.k (E4M3's relative rounding on the few large channels that carry the
                       big logits); the Hadamard spreads them over the head.  The BF16 window
                       call of W > 0 stays BF16.  Default "off" (BF16): "had" costs +10 ... +35 % prompt KL.
                       Phase 2f, optional: "int4" / "int4ac" = CUDA kernels reading the INT4 pages directly (no
                       gather / dequantized workspace; prefill_int4.py, csrc/int4_prefill.cu): QK in FP16 (int4;
                       FlagRot: per-key inverse transform of the K tile in shared memory) or design A+C (int4ac:
                       exact E4M3 codes, q E4M3 hi + lo), PV in FP8.  KL +1 ... +7 % over the BF16 path, but on H20
                       only ~0.8x the BF16 path's attention time.
                       "kernel": every prefill token is a row of the decode kernel
                       (the request's block-table row repeated, per-row lengths, per-row window offsets via tail_idx):
                       exact decode numerics for every prompt position (parity of the decode kernels), O(L^2) bytes
                       (short prompts).
                       Phase 2i, default (prefill_mq = "auto"): requests whose new segment has at most mq_max_new =
                       2048 tokens (multi-turn turns over a cached prefix) attend over the INT4 pages directly, all of
                       a step's in one launch per layer (prefill_mq.py, csrc/mq_prefill.cu: split-K, FP16 wgmma tiles
                       converted in shared memory, fp32 softmax; at least as precise as the BF16 "fa" path); the
                       others take the "fa" path.  "off" = Phase 2b-2h behaviour, "all" = every prefill request.
                       Phase 2j: head size 256 too (k_mq3; Qwen3.5 / 3.8 full-attention layers, any GQA group).

GQA groups and head sizes (Phase B): the static formats and the key-only 2x2 codebook support any
GQA group G (q tiles of 4 / 8 heads in the generalized kernel) and head_size 128 or 256 (pages = two 128-d half chunks;
whole-head transforms H_D); the other codebooks keep G = 4, head_size 128.  Hybrid models (Qwen3.5 / 3.8: Gated
DeltaNet + every 4th layer full attention): only the attention layers reach this backend; the calibration file then
lists the model layer indices it covers (meta["layers"]).  Tensor parallelism: each rank owns a contiguous slice of
the KV heads (vLLM's QKVParallelLinear) and takes the matching slice of the per-head calibration.

Cache writers (option "writer" / env FLAGROT_WRITER): "v3" (default) = the fused writers of kernels/csrc/
attn_cb_write3.cuh): codebook kinds -> fr_cb_write3 (tensor-core candidate evaluation +
exact fallback; writes of T <= W3_CB_V1_MAX_T = 64 tokens (decode batches up to 64) keep the v1 kernel, which is
faster there inside vLLM; T <= 256: the K and V kernels on two streams), static -> fr_w3_static_write (whole-head H128 +
block flags + quantization in one kernel; with no BF16 slots (S = W = 0) the whole-head rotation of k / v is fused
into it instead of a separate fr_head_fwht launch).  "v1" = the previous writers (fr_cb_write; fr_head_fwht +
fr_v2_write).  Both write the same page bytes.  The v3 writers
are head_size 128 only: head_size 256 layers keep the v1 writers.

Static key centering (option "center" / env FLAGROT_CENTER = flagrot_center_<model>.pt, int2c; opt-in, default off):
the research kv_center = post (blockflag.run.build): mu = per-(layer, KV head, channel) mean of the post-RoPE keys of
the calibration windows (scripts/export_calib.py --center), and every cached key becomes k - mu.  q.(k - mu) = q.k -
q.mu shifts all logits of a query by the same amount, so attention is unchanged in exact arithmetic; only the
quantization error changes.  Writers: the raw bf16 key and mu (stored basis: whole-head H128 heads H mu / 16) go to
the kernels, which subtract in fp32 before encoding (bit-identical to the simulation's key.float() - mu); the BF16
sink / window slots hold bf16(k - mu) (stored basis); the attention reads the centered keys everywhere (pages, BF16
slots, sink pool, and the prefill's current-chunk keys, centered at the start of forward); values are untouched.
FlagRot-C: the base transform applies after centering ((k - mu) xk in fp32, with a base calibrated on centered keys,
flagrot_calib_cc_<model>.pt); FlagRot-S: headsel calibrated on centered keys (flagrot_static_a2c_<model>.pt).

kivi2 (FlagRot-KIVI, kv_cache_dtype "flagrot_kivi", calibration flagrot_kivi_<model>.pt / flagrot_kivi_c_<model>.pt):
simulator methods kivi_sel / kivi_c (kivi2:32 keys per channel over 32-token groups + values per token, kv_dyn
att+f; kivi_c: + the calibrated WUSH base, calib_rot 32768), pages of vllm_flagrot/kivi.py / csrc/kivi.cu (97 B per
(token, KV head) at head 128).  Values are encoded per token at their real slots as every format; a key group
[32 g, 32 g + 32) (sinks excluded; layout option A: absolute alignment) is encoded once its last token is written -
decode: by the step that writes position 32 g + 31 (from the BF16 window ring, CUDA-graph safe: inactive rows exit),
prefill: by the chunk that contains it (chunk rows + the ring rows of earlier positions).  Needs sink + window with
W >= 32 (every key read from the pages lies in a completed group) and prefix-cache reuse points at multiples of 32
(prefix_cache.py).  Decode: kivi.attn (q . k = sum_c (q'_c s_c) code_c + q' . min per group); prefill: the "fa" path
on the dequantized pages (no INT2-direct prefill for this format).

Debug switch (env / module global): FLAGROT_MODE=ref attends to the unquantized key / value of a fresh prefill
(= the stock FLASH_ATTN result), so one engine produces both distributions for KL(BF16 || Q).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout

from .. import kernels as K
from .. import kivi as KIVI

logger = init_logger("vllm.plugins.flagrot")

MODE = os.environ.get("FLAGROT_MODE", "quant")
BT_COPY = os.environ.get("FLAGROT_BT_COPY") == "1"      # prefill="kernel": materialized block-table rows (old path)
# debug / parity diagnostic: the "fa" prefill calls (and FLAGROT_MODE=ref) in fp32 eager attention instead of BF16
# FlashAttention (the research simulation's attention numerics); BF16-input inputs, fp32 scores / softmax / PV
FA_FP32 = os.environ.get("FLAGROT_FA_FP32") == "1"
# debug (Phase 2k parity diagnostic): the BF16 window rows of the current chunk go through the stored basis and back,
# as in Phase 2b-2j (static whole-head H rotations / FlagRot-C base: two extra BF16 roundings); default off = the
# model's BF16 K / V straight
WIN_ROUNDTRIP = os.environ.get("FLAGROT_WIN_ROUNDTRIP") == "1"
# check switch: the batched BF16 window / sink parts of the INT4-direct path in PyTorch ops instead of the
# k_win_gather / k_bf_merge kernels (prefill_mq.py)
BF_TORCH = os.environ.get("FLAGROT_BF_TORCH") == "1"
# kivi2 check switch: the "fa" prefill's page dequantization in PyTorch (kivi.dequant) instead of the CUDA kernel
KIVI_DQ_TORCH = os.environ.get("FLAGROT_KIVI_DQ_TORCH") == "1"
# kivi2 check switch (parity of the decode kernel): fresh prefills (no cached prefix) attend through the decode kernel
# k_attn3, one kernel row per prompt token (pages [S, i + 1 - W) + BF16 sinks and window), instead of the "fa" path
KIVI_DEC_PREFILL = os.environ.get("FLAGROT_KIVI_DEC_PREFILL") == "1"
KIVI_DEC_CHUNK = int(os.environ.get("FLAGROT_KIVI_DEC_CHUNK", "256"))
# kivi2p8f check (with KIVI_DEC_PREFILL and kivi_fast): every decode-kernel call is repeated with the k_attn3 kernel of
# kivi2p8 on the same (real) cache; max |fast - old| and max |old| per layer call are written to this JSON at exit
KIVI_FAST_CHECK = os.environ.get("FLAGROT_KIVI_FAST_CHECK", "")
KIVI_FAST_STATS: dict = {"calls": 0, "max_abs": 0.0, "max_rel": 0.0, "sum_abs": 0.0, "n": 0, "bitexact": 0, "elems": 0}
# kivi2p8ff check: every k_dequant2 call is repeated with k_dequant and every fused merge with _merge (same inputs);
# the counts of differing bits are written to this JSON at exit (expected 0)
KIVI_DQ_CHECK = os.environ.get("FLAGROT_KIVI_DQ_CHECK", "")
KIVI_DQ_STATS: dict = {"dequant_calls": 0, "dequant_elems": 0, "dequant_diff": 0, "merge_calls": 0, "merge_elems": 0,
                       "merge_diff": 0}
if KIVI_DQ_CHECK:
    import atexit
    import json as _json2

    def _dump_dq_check():
        with open(KIVI_DQ_CHECK, "w") as f:
            _json2.dump(KIVI_DQ_STATS, f)
    atexit.register(_dump_dq_check)
if KIVI_FAST_CHECK:
    import atexit
    import json as _json

    def _dump_fast_check():
        with open(KIVI_FAST_CHECK, "w") as f:
            _json.dump(dict(KIVI_FAST_STATS, mean_abs=KIVI_FAST_STATS["sum_abs"] / max(1, KIVI_FAST_STATS["n"])), f)
    atexit.register(_dump_fast_check)


class _OpTimer:
    """op-level timer of the eager impl paths (FLAGROT_OPTIMER=1 or OPTIMER.on = True): CUDA events around the cache
    update (encoding: whole-head rotations + writer + window bookkeeping), the prefill attention and its PyTorch
    dequantization.  Inactive during CUDA-graph capture (decode graphs are not timed).  read() synchronizes and
    returns {name: (ms, calls)}; reset() clears."""

    def __init__(self):
        self.on = os.environ.get("FLAGROT_OPTIMER") == "1"
        self.ev = []

    def start(self, name):
        if not self.on or torch.cuda.is_current_stream_capturing():
            return None
        e0 = torch.cuda.Event(enable_timing=True)
        e0.record()
        return (name, e0)

    def stop(self, tok):
        if tok is None:
            return
        e1 = torch.cuda.Event(enable_timing=True)
        e1.record()
        self.ev.append((tok[0], tok[1], e1))

    def reset(self):
        self.ev = []

    def read(self):
        torch.cuda.synchronize()
        out = {}
        for name, e0, e1 in self.ev:
            ms, n = out.get(name, (0.0, 0))
            out[name] = (ms + e0.elapsed_time(e1), n + 1)
        return out


OPTIMER = _OpTimer()
FMT_CB, FMT_STATIC = "flagrot_int4", "flagrot_int4_static"
FMT_KIVI = "flagrot_kivi"                # kivi2: FlagRot-KIVI INT2 pages (vllm_flagrot/kivi.py)
PREFILL_INT4_STAGES = int(os.environ.get("FLAGROT_INT4_STAGES", "14133"))   # A+C v2 variant (prefill_int4.py)
PREFILL_KBF_STAGES = int(os.environ.get("FLAGROT_KBF_STAGES", "34"))        # FP16-QK v3 variant
PREFILL_FP8_DEFAULT = "off"      # BF16 prefill attention (default pending the Phase 2f decision); "had" = Phase 2e FP8
# prefill attention modes of the "fa" path: BF16 FA ("off"), FA3 FP8 on a dequantized workspace (Phase 2e), the
# Phase 2f accuracy emulations of INT4-direct designs (prefill_emu.py, debug / parity only)
PREFILL_MODES = ("off", "unit", "amax", "had", "int4", "int4ac", "emu_exact", "emu_q", "emu_q1", "emu_qtop1", "emu_qtop2", "emu_ac", "emu_acv", "emu_ac1", "emu_b")
FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
WRITER_DEFAULT = "v3"            # cache writers: fused v3 (Phase 2h) | "v1" (Phase 2b-2g); same page bytes
# writer v3, codebook kinds: writes of at most this many tokens (decode steps of B <= 64, window evictions) keep the v1
# codebook writer (same bytes).  Inside a decode step the v3 key kernel has one CTA per 16-token tile and kv head (8
# CTAs at B = 8) and first loads the head's 43 KB correction matrix from DRAM (the weight GEMMs flush L2): in vLLM
# (nsys) its key kernel takes 17.2 / 24.2 us per layer at B = 8 / 64 vs v1 16.4 / 21.5, while
# from B = 96 on v1 switches to 4 tokens per CTA and v3 is faster (B = 96 / 128 / 256: 24.1 / 24.6 / 26.0 vs v1
# 28.6 / 32.0 / 57.2 us).  FLAGROT_W3_CB_V1_MAX_T=0: v3 for every write.
W3_CB_V1_MAX_T = int(os.environ.get("FLAGROT_W3_CB_V1_MAX_T", "64"))
# Phase 2i: prefill chunks served straight from the INT4 pages (prefill_mq.py, csrc/mq_prefill.cu) instead of the
# "fa" path's PyTorch gather + dequantization of the whole prefix: "auto" = requests whose new segment has at most
# mq_max_new tokens (multi-turn / agentic turns over a cached prefix), "all" =
# every prefill request (parity checks), "off" = the Phase 2b-2h path.  Only with prefill="fa", prefill_fp8="off",
# int4a caches (static_a / rtn_a / konlya / konlya_c); head size 128 or (Phase 2j) 256.
PREFILL_MQ_DEFAULT = "auto"
MQ_MAX_NEW_DEFAULT = 2048
# head size 256 (Phase 2j, H20): the INT4-direct path is >= the fa path up to 4096 new tokens at every prefix length
# and even at 8192; hybrid models cache prefixes in ~2.8k-token blocks, so a turn computes up to
# ~3k tokens.  Used when mq_max_new is not set.
MQ_MAX_NEW_DEFAULT_256 = 4096
# per-process counters (prefill requests / tokens per path, counted once per step and KV-cache group; checks)
MQ_STATS: dict = {}
# debug switch (module global, in-process engines; like MODE): None = the configured prefill_mq, "off" / "all" force
MQ_OVERRIDE: str | None = None
PAGE = 16
LOG2E = 1.4426950408889634
D = 128                     # default head size (the codebook kinds other than the key-only 2x2 support only this)
HEAD_SIZES = (128, 256)


def _patch_hybrid_req_idx() -> None:
    """Model runner V2, hybrid (Mamba / Gated DeltaNet) models: MambaHybridModelState.prepare_attn builds the
    CommonAttentionMetadata without req_idx (DefaultModelState passes input_batch.idx_mapping_np), but the BF16 sink /
    window slots are indexed by the request slot.  Wrap it so that its build_attn_metadata call gets the same req_idx
    as the default model state (a no-op once vLLM passes it itself: setdefault)."""
    try:
        from vllm.v1.worker.gpu.model_states import mamba_hybrid as MH
    except Exception:  # noqa: BLE001
        return
    if getattr(MH, "_flagrot_req_idx", False):
        return
    cur: dict = {}
    build = MH.build_attn_metadata
    prepare = MH.MambaHybridModelState.prepare_attn

    def build_attn_metadata(*args, **kw):
        if cur.get("req_idx") is not None:
            kw.setdefault("req_idx", cur["req_idx"])
        return build(*args, **kw)

    def prepare_attn(self, input_batch, *args, **kw):
        cur["req_idx"] = getattr(input_batch, "idx_mapping_np", None)
        try:
            return prepare(self, input_batch, *args, **kw)
        finally:
            cur["req_idx"] = None

    MH.build_attn_metadata = build_attn_metadata
    MH.MambaHybridModelState.prepare_attn = prepare_attn
    MH._flagrot_req_idx = True


def _vllm_config():
    try:
        from vllm.config.vllm import get_current_vllm_config_or_none
        return get_current_vllm_config_or_none()
    except Exception:  # noqa: BLE001
        return None


_LAST: dict = {}


def flagrot_options(cfg=None) -> dict:
    """{"kv_cache_dtype", "calib", "window", "prefill", "kv_dyn", "quant", "plan_len"} from --additional-config
    '{"flagrot": {...}}' (env FLAGROT_CALIB / FLAGROT_WINDOW / FLAGROT_PREFILL / FLAGROT_KV / FLAGROT_QUANT /
    FLAGROT_KV_DYN).  Without a current vLLM config (e.g. while the KV-cache spec is built) the options last read from
    a config are used."""
    opts = {}
    cfg = cfg if cfg is not None else _vllm_config()
    add = getattr(cfg, "additional_config", None) if cfg is not None else None
    if isinstance(add, dict) and add.get("flagrot"):
        opts.update(add["flagrot"])
        _LAST.clear()
        _LAST.update(opts)
    elif _LAST:
        opts.update(_LAST)
    for key, env in (("calib", "FLAGROT_CALIB"), ("window", "FLAGROT_WINDOW"), ("prefill", "FLAGROT_PREFILL"),
                     ("sink", "FLAGROT_SINK"),
                     ("kv_cache_dtype", "FLAGROT_KV"), ("quant", "FLAGROT_QUANT"), ("kv_dyn", "FLAGROT_KV_DYN"),
                     ("prefill_fp8", "FLAGROT_PREFILL_FP8"), ("writer", "FLAGROT_WRITER"),
                     ("prefill_mq", "FLAGROT_PREFILL_MQ"), ("mq_max_new", "FLAGROT_MQ_MAX_NEW"),
                     ("sink_pool", "FLAGROT_SINK_POOL"), ("pc_recompute", "FLAGROT_PC_RECOMPUTE"),
                     ("center", "FLAGROT_CENTER"), ("kivi_id", "FLAGROT_KIVI_ID"),
                     ("kivi_static", "FLAGROT_KIVI_STATIC"), ("kivi_f8", "FLAGROT_KIVI_F8"),
                     ("kivi_fast", "FLAGROT_KIVI_FAST"), ("kivi_ff", "FLAGROT_KIVI_FF")):
        if os.environ.get(env) and key not in opts:
            opts[key] = os.environ[env]
    opts.setdefault("kv_cache_dtype", FMT_CB)
    opts["window"] = int(opts.get("window", 0) or 0)
    opts["sink"] = int(opts.get("sink", 0) or 0)
    opts.setdefault("prefill", "fa")
    opts["prefill_fp8"] = str(opts.get("prefill_fp8", PREFILL_FP8_DEFAULT) or "off").lower()
    if opts["prefill_fp8"] in ("0", "false", "none", "bf16"):
        opts["prefill_fp8"] = "off"
    if opts["prefill_fp8"] not in PREFILL_MODES:
        raise ValueError(f"FlagRot backend: prefill_fp8 must be one of {PREFILL_MODES}, got "
                         f"{opts['prefill_fp8']!r}")
    opts.setdefault("quant", "int4")
    opts["writer"] = str(opts.get("writer") or WRITER_DEFAULT).lower()
    if opts["writer"] not in ("v1", "v3"):
        raise ValueError(f"FlagRot backend: writer must be 'v1' or 'v3', got {opts['writer']!r}")
    if opts["quant"] not in ("int4", "int4a", "int2a"):
        raise ValueError(f"FlagRot backend: quant must be 'int4', 'int4a' or 'int2a', got {opts['quant']!r}")
    if opts["kv_cache_dtype"] not in (FMT_CB, FMT_STATIC, FMT_KIVI):
        raise ValueError(f"FlagRot backend: unknown kv format {opts['kv_cache_dtype']!r}")
    if not 0 <= opts["window"] <= 256:
        raise ValueError("FlagRot backend: window must be in [0, 256]")
    if opts["sink"] < 0 or opts["sink"] + opts["window"] > 256:
        raise ValueError("FlagRot backend: sink >= 0 and sink + window <= 256 (one BF16 tail CTA per sequence)")
    if opts["prefill"] not in ("fa", "kernel"):
        raise ValueError(f"FlagRot backend: prefill must be 'fa' or 'kernel', got {opts['prefill']!r}")
    opts["prefill_mq"] = str(opts.get("prefill_mq") or PREFILL_MQ_DEFAULT).lower()
    if opts["prefill_mq"] in ("0", "false", "none", "fa"):
        opts["prefill_mq"] = "off"
    if opts["prefill_mq"] not in ("auto", "off", "all"):
        raise ValueError(f"FlagRot backend: prefill_mq must be 'auto', 'off' or 'all', got {opts['prefill_mq']!r}")
    opts["mq_max_new"] = int(opts.get("mq_max_new") or 0)       # 0: the head-size default (MQ_MAX_NEW_DEFAULT[_256])
    opts["kivi_id"] = str(opts.get("kivi_id") or "0").lower() in ("1", "true", "yes")
    opts["kivi_static"] = str(opts.get("kivi_static") or "0").lower() in ("1", "true", "yes")
    # kivi2p8: the research kivi2f8:32 storage (FP8 E4M3 scale + minimum, 80 B per token and KV head at head 128)
    opts["kivi_f8"] = str(opts.get("kivi_f8") or "0").lower() in ("1", "true", "yes")
    # kivi2p8f / kivi2f: the fast decode-attention variant of k_attn3 (same math; csrc/kivi.cu FAST)
    opts["kivi_fast"] = str(opts.get("kivi_fast") or "0").lower() in ("1", "true", "yes")
    # kivi2p8ff: kivi_fast + the bandwidth-bound prefix dequantization (k_dequant2, bit-identical K / V) and the fused
    # LSE merge of the "fa" prefill (bit-identical to _merge)
    opts["kivi_ff"] = str(opts.get("kivi_ff") or "0").lower() in ("1", "true", "yes")
    if opts["kivi_ff"]:
        opts["kivi_fast"] = True
    opts["center"] = str(opts.get("center") or "")
    if opts["center"].lower() in ("0", "false", "none", "off"):
        opts["center"] = ""
    if opts["center"] and not os.path.exists(opts["center"]):
        raise ValueError(f"FlagRot backend: center file {opts['center']!r} not found (scripts/export_calib.py --center)")
    return opts


_CALIB: dict = {}


def load_calib(path: str) -> dict:
    if path not in _CALIB:
        _CALIB[path] = torch.load(path, map_location="cpu", weights_only=True)
    return _CALIB[path]


STATIC_KINDS = {"static": "int4", "static_a": "int4a", "static_a2": "int2a"}   # static kind -> v2 kernel format
INT2_KINDS = ("static_a2", "konlya2")                       # Phase 3a: INT2 pages
ASYM_KINDS = ("static_a", "konlya")                          # INT4A kinds (INT4-direct prefill kernels)


def codebook_kind(opts) -> str:
    """'static' | 'static_a' (asymmetric) | a codebook kind of kernels.CB_KINDS (konlya = asymmetric key-only 2x2)."""
    q = opts.get("quant", "int4")
    asym = q in ("int4a", "int2a")
    if opts["kv_cache_dtype"] == FMT_KIVI:
        return "kivi2"
    if opts["kv_cache_dtype"] == FMT_STATIC:
        return "static_a2" if q == "int2a" else "static_a" if asym else "static"
    kv_dyn = opts.get("kv_dyn")
    if not kv_dyn and opts.get("calib"):
        kv_dyn = load_calib(opts["calib"])["meta"].get("kv_dyn")
    kv_dyn = kv_dyn or "cq+f+S+Q5+FS+R5"
    if kv_dyn not in K.KV_DYN_KIND:
        raise ValueError(f"FlagRot backend: codebook {kv_dyn!r} has no kernel (supported: {list(K.KV_DYN_KIND)})")
    kind = K.KV_DYN_KIND[kv_dyn]
    if asym:
        if kind not in K.ASYM_KIND:
            raise ValueError(f"FlagRot backend: codebook {kv_dyn!r} has no asymmetric INT4 kernel (quant=int4a: "
                             "static or cq+f+konly only)")
        kind = (K.ASYM2_KIND if q == "int2a" else K.ASYM_KIND)[kind]
    return kind


def is_static(kind: str) -> bool:
    return kind in STATIC_KINDS


def gen_ok(kind: str) -> bool:
    """formats the generalized kernel serves (any GQA group, head size 256): static and key-only 2x2 (and kivi2:
    GQA <= 8, head 128 / 256)"""
    return is_static(kind) or kind in K.K2 or kind == "kivi2"


def cell_bytes(kind: str, head_size: int = D) -> int:
    """bytes per (token, KV head) of the chunk layouts (page 16): static 144 (int4 and int4a), codebook 148 (key-only
    4x4 / 8x8: 146, key-only 2x2: 144); head size 256: twice (static / key-only 2x2 only)."""
    if head_size != D and not gen_ok(kind):
        raise ValueError(f"FlagRot KV: head_size {head_size} needs a static or key-only 2x2 format (kind {kind})")
    if kind == "kivi2":
        return KIVI.cell_bytes(head_size, flagrot_options().get("kivi_f8", False))
    return (K.v2_chunk_bytes(PAGE, head_size, STATIC_KINDS[kind]) if is_static(kind)
            else K.cb_chunk_bytes(kind, PAGE, head_size)) // PAGE


class FlagRotAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["FlagRotAttentionImpl"]:
        return FlagRotAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["FlagRotMetadataBuilder"]:
        return FlagRotMetadataBuilder

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """uint8 pages, one head slot per KV head; C = bytes per token of the per-(page, KV head) chunk."""
        if spec.state_content_bytes is not None:
            return spec
        assert spec.head_size == spec.head_size_v and spec.head_size in HEAD_SIZES, \
            f"FlagRot KV kernels: head_size {HEAD_SIZES} only (got {spec.head_size})"
        return replace(spec, dtype=torch.uint8,
                       state_content_bytes=cell_bytes(codebook_kind(flagrot_options()), spec.head_size))

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBHNC,)        # [blocks, Hkv, N, C]: one contiguous chunk per (page, KV head)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list:
        return [PAGE]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return list(HEAD_SIZES)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False

    @classmethod
    def supports_combination(cls, head_size, dtype, kv_cache_dtype, block_size, use_mla, has_sink, use_sparse,
                             use_mm_prefix, device_capability) -> str | None:
        if use_mla or has_sink or use_sparse or use_mm_prefix:
            return "FlagRot KV: MLA / sinks / sparse / mm-prefix not supported"
        if head_size not in HEAD_SIZES:
            return "FlagRot KV: head_size 128 / 256 only"
        if device_capability is not None and (device_capability.major, device_capability.minor) != (9, 0):
            return "FlagRot KV kernels: sm_90 (Hopper) only"
        return None


@dataclass
class FlagRotMetadata(AttentionMetadata):
    seq_lens: torch.Tensor                # [n] context incl. the new tokens
    query_start_loc: torch.Tensor         # [n + 1]
    block_table: torch.Tensor             # [n, max_blocks]
    slot_mapping: torch.Tensor
    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    num_decodes: int                      # rows [0, num_decodes) have one query token (fast decode kernel)
    query_start_loc_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor | None     # host, exact for prefill rows
    # decode plan (fixed per batch size: CUDA-graph safe)
    nsplit: int = 1
    chunk: int = 64
    # decode rows, persistent device buffers (sliced [:num_decodes])
    dec_cached: torch.Tensor | None = None     # int32 page tokens read: [0, L - W) with [0, S) masked (kv_start = S);
                                               # 0 when L - W <= S (= seq_lens if S = W = 0)
    dec_tail_len: torch.Tensor | None = None   # int32 BF16 slots read: min(L, S + W)
    dec_tail_idx: torch.Tensor | None = None   # int32 ring row (request slot)
    dec_ring_pos: torch.Tensor | None = None   # int64 flat BF16 slot of the new token (req * (S + W) + slot)
    req_idx: np.ndarray | None = None          # host request slots of all rows (prefill window bookkeeping)
    kivi_sidx: torch.Tensor | None = None      # kivi2 decode rows: [nd, 32] int64 ring rows of the key group the
    kivi_dslot: torch.Tensor | None = None     # step completes (-1 sinks) and its page slots ([:, 0] < 0: none)


class FlagRotMetadataBuilder(AttentionMetadataBuilder[FlagRotMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        self.opts = flagrot_options(vllm_config)
        self.kind = codebook_kind(self.opts)
        self.W = self.opts["window"]
        self.S = self.opts["sink"]
        self.Hkv = kv_cache_spec.num_kv_heads
        self.Dh = kv_cache_spec.head_size
        self.G = gqa_group(vllm_config, self.Hkv)
        self.max_len = int(self.opts.get("plan_len") or vllm_config.model_config.max_model_len)
        sc = vllm_config.scheduler_config
        cc = getattr(vllm_config, "compilation_config", None)
        caps = list(getattr(cc, "cudagraph_capture_sizes", None) or [])
        self.max_rows = max([sc.max_num_seqs] + caps)
        self.R = sc.max_num_seqs                         # ring rows (request slots); row R = padding sink
        n = self.max_rows
        i32 = dict(dtype=torch.int32, device=device)
        i64 = dict(dtype=torch.int64, device=device)
        self._cached = torch.zeros(n, **i32)
        self._tail_len = torch.zeros(n, **i32)
        self._tail_idx = torch.full((n,), self.R, **i32)
        self._ring_pos = torch.zeros(n, **i64)
        self._plans: dict = {}
        if self.kind == "kivi2":
            self._ksidx = torch.full((n, 32), -1, **i64)
            self._kdslot = torch.full((n, 32), -1, **i64)
            self._k32 = torch.arange(32, **i64)

    def plan(self, B: int):
        if B not in self._plans:
            self._plans[B] = decode_plan(self.kind, B, self.max_len, self.Hkv, self.G, self.Dh)
        return self._plans[B]

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> FlagRotMetadata:
        cam = common_attn_metadata
        n = cam.num_reqs
        nd, _, _, _ = split_decodes_and_prefills(cam, decode_threshold=1)
        seq = cam.seq_lens[:n]
        bt = cam.block_table_tensor[:n]
        md = FlagRotMetadata(seq_lens=seq, query_start_loc=cam.query_start_loc[:n + 1], block_table=bt,
                             slot_mapping=cam.slot_mapping, num_reqs=n, num_actual_tokens=cam.num_actual_tokens,
                             max_query_len=cam.max_query_len, max_seq_len=cam.max_seq_len, num_decodes=nd,
                             query_start_loc_cpu=cam.query_start_loc_cpu[:n + 1],
                             seq_lens_cpu=None if cam.seq_lens_cpu_upper_bound is None
                             else cam.seq_lens_cpu_upper_bound[:n])
        req = getattr(cam, "req_idx", None)
        if req is not None:
            req = np.asarray(req)[:n]
        md.req_idx = req
        if nd > 0:
            md.nsplit, md.chunk = self.plan(nd)
            L = seq[:nd]
            if self.W == 0 and self.S == 0:
                md.dec_cached = L
            else:
                W, S = self.W, self.S
                T = S + W                    # BF16 slots per request row: S sink slots, then the W-slot window ring
                # request slots (padding rows and dummy runs without req_idx -> the sink row R)
                self._tail_idx[:nd].fill_(self.R)
                if req is not None:
                    k = min(len(req), nd)
                    # non-blocking host->device copy: a blocking copy would wait for the GPU queue every decode step
                    # and serialize the host-side preparation of the next step with the running one (the source is
                    # pageable: the driver stages it before returning, so the temporary may be freed)
                    self._tail_idx[:k].copy_(torch.from_numpy(np.ascontiguousarray(req[:k], dtype=np.int32)),
                                             non_blocking=True)
                # Phase 2k: the pages hold every position at its real slot (the new token is written to its page in
                # this step, like W = 0); the kernel reads [S, L - W) of them (kv_start = S masks the sinks) and the
                # BF16 slots [0, min(L, T)) = the sinks [0, min(L, S)) + the window [max(S, L - W), L)
                torch.where(L - W > S, L - W, torch.zeros_like(L), out=self._cached[:nd])
                torch.clamp(L, max=T, out=self._tail_len[:nd])
                p = L.long() - 1                                            # the new token
                row = self._tail_idx[:nd].long() * T
                if W > 0:
                    # slot of p: p < S -> sink slot p, else S + (p - S) % W (it held token p - W, already in its page)
                    slot = torch.where(p < S, p, S + torch.remainder(p - S, W))
                else:
                    # sinks only: p < S -> sink slot p; else no BF16 copy (dummy slot of the padding row R)
                    slot = torch.where(p < S, p, torch.full_like(p, -1))
                torch.where(slot >= 0, row + slot, torch.full_like(p, self.R * T), out=self._ring_pos[:nd])
                md.dec_cached = self._cached[:nd]
                md.dec_tail_len = self._tail_len[:nd]
                md.dec_tail_idx = self._tail_idx[:nd]
                md.dec_ring_pos = self._ring_pos[:nd]
                if self.kind == "kivi2":
                    self._kivi_decode_groups(md, L, bt[:nd], nd)
        return md

    def _kivi_decode_groups(self, md, L, bt, nd):
        """kivi2: the key group [32 g, 32 g + 32) completed by a decode step (new token p = L - 1 = 32 g + 31) is
        encoded from the BF16 ring in the same step (device ops on persistent buffers: CUDA-graph safe)."""
        S, W, T = self.S, self.W, self.S + self.W
        Ll = L.long()
        act = (Ll % 32 == 0) & (Ll >= 32) & (self._tail_idx[:nd] < self.R)
        pos = ((Ll - 1) // 32 * 32)[:, None] + self._k32[None]                          # [nd, 32]
        row = self._tail_idx[:nd].long()[:, None] * T
        slot = torch.where(pos < S, pos, S + torch.remainder(pos - S, W))
        torch.where((pos < S) | ~act[:, None], torch.full_like(pos, -1), row + slot, out=self._ksidx[:nd])
        pg = torch.clamp(pos // PAGE, min=0, max=bt.shape[1] - 1)
        ds = bt.gather(1, pg).long() * PAGE + pos % PAGE
        torch.where(act[:, None], ds, torch.full_like(ds, -1), out=self._kdslot[:nd])
        md.kivi_sidx = self._ksidx[:nd]
        md.kivi_dslot = self._kdslot[:nd]


# ---------------------------------------------------------------------------------------------- per-device shared state
def plan_bucket(B: int) -> int:
    """decode batch size the split plan is made for: vLLM's default CUDA-graph capture sizes (1, 2, 4, then multiples
    of 8 up to 256, of 16 beyond), so a padded graph batch and the same batch run eagerly use the same splits (the
    split count changes the summation order, hence the bits of the output)."""
    if B <= 4:
        return 1 << (B - 1).bit_length()
    return -(-B // 8) * 8 if B <= 256 else -(-B // 16) * 16


def gqa_group(vllm_config, Hkv: int) -> int:
    """q heads per KV head on this rank (TP shards q and KV heads alike; KV heads replicated when TP > Hkv)"""
    try:
        mc, pc = vllm_config.model_config, vllm_config.parallel_config
        hq = mc.get_num_attention_heads(pc)
        return max(1, hq // max(1, Hkv))
    except Exception:  # noqa: BLE001
        return 4


def tile_layout(kind: str, Hkv: int, G: int, Dh: int):
    """(CTAs per (row, split), q heads per partial slot, head dim) of the decode kernel of `kind`."""
    if is_static(kind) or (kind in K.K2 and (G != 4 or Dh != D)):
        fmt = STATIC_KINDS.get(kind, kind)
        return K.vheads(Hkv, G, Dh, fmt), K.tile_of(K.v2_variant(fmt, G, Dh)), Dh
    return Hkv, 4, D


def decode_plan(kind: str, B: int, max_len: int, Hkv: int, G: int = 4, Dh: int = D, fast: bool | None = None):
    """(nsplit, chunk) of a decode batch of B rows: fixed per (batch bucket, max_model_len), no device sync."""
    Bp = plan_bucket(B)
    if kind == "kivi2":
        if fast is None:
            fast = flagrot_options().get("kivi_fast", False)
        return KIVI.plan(Bp, max_len, Hkv, occ=KIVI.fast_occ(Dh, G) if fast else None)
    if is_static(kind):
        return K.plan_v2(Bp, max_len, Hkv, PAGE, fmt=STATIC_KINDS[kind], G=G, Dh=Dh)
    return K.plan_cb(Bp, max_len, Hkv, PAGE, kind, G=G, Dh=Dh)


_WS: dict = {}
_KWS: dict = {}


_KWS_PARTS: dict = {}


def kivi_workspace(device, rows, Hkv, G, Dh, plan_fn, pkey=None):
    """kivi2 split-K workspace, sized once for every decode batch size <= rows (CUDA graphs bake the pointers in).
    pkey: a hashable key of the plan (kivi2p8ff: the parts are computed once per key instead of every call)."""
    key = (str(device), Hkv, G, Dh)
    ws = _KWS.get(key)
    parts = _KWS_PARTS.get((key, rows, pkey)) if pkey is not None else None
    if parts is None:
        parts = max(plan_bucket(B) * Hkv * (plan_fn(B)[0] + 1) for B in range(1, rows + 1))
        if pkey is not None:
            _KWS_PARTS[(key, rows, pkey)] = parts
    if ws is None or ws.parts < parts:
        ws = _KWS[key] = KIVI.Workspace(parts, G, Dh, device)
    return ws


def decode_workspace(device, layout, rows, plan_fn):
    """one split-K workspace per device and tile layout (vheads, GT, Dh), sized once for every decode batch size
    <= rows (CUDA graphs bake the pointers in; layers run sequentially so they share it)."""
    HV, GT, Dh = layout
    key = (str(device), HV, GT, Dh)
    ws = _WS.get(key)
    if ws is None or ws.rows < rows:
        parts = max(plan_bucket(B) * HV * (plan_fn(B)[0] + 1) for B in range(1, rows + 1))
        ws = _WS[key] = K.Workspace(rows, HV, parts, device, GT, Dh)
    return ws


_PWS: dict = {}


def prefill_workspace(device, layout, rows, parts):
    HV, GT, Dh = layout
    key = (str(device), HV, GT, Dh)
    ws = _PWS.get(key)
    if ws is None or ws.rows < rows or ws.parts < parts:
        ws = _PWS[key] = K.Workspace(max(rows, ws.rows if ws else 1), HV, max(parts, ws.parts if ws else 1), device,
                                     GT, Dh)
    return ws


def _hpm(device, n: int = D) -> torch.Tensor:
    """unnormalized Sylvester Hadamard (+-1, symmetric, H H = n I), fp32."""
    return K.hadamard(n, device, torch.float32) * math.sqrt(n)


class _Static:
    """per-layer static FlagRot transforms (kv_flags=headsel): block flags for the kernels + whole-head matrices."""

    def __init__(self, cal, li, heads, device, Hq_per_kv, Dh=D):
        self.Dh = Dh
        km, vm = cal["kmask"][li][heads], cal["vmask"][li][heads]          # [Hkv, Dh / 32] bool
        kf, vf = cal["kfull"][li][heads], cal["vfull"][li][heads]          # [Hkv] bool
        assert km.shape[-1] == Dh // 32, (km.shape, Dh)
        self.kmask = km.to(torch.uint8).contiguous().to(device)
        self.vmask = vm.to(torch.uint8).contiguous().to(device)
        self.wmask = torch.cat([km, vm]).to(torch.uint8).contiguous().to(device)    # [2 Hkv, 4]: K rows first
        self.kfull, self.vfull = kf.to(device), vf.to(device)
        self.any_full = bool(kf.any() or vf.any())
        # writer v3: the whole-head flags (K rows first) for the fused H128 (None: no whole-head head in this layer)
        self.hfull = torch.cat([kf, vf]).to(torch.uint8).contiguous().to(device) if self.any_full else None
        # attention-kernel flags: None when the layer has no K (V) block flag, i.e. no q rotation (o back-rotation)
        # in the kernel (all-zero calibration = rtn_a: plain INT4, no transform anywhere)
        self.kmask_attn = self.kmask if bool(km.any()) else None
        self.vmask_attn = self.vmask if bool(vm.any()) else None
        self.any_block = (bool(km.any()), bool(vm.any()))
        self.any_kfull, self.any_vfull = bool(kf.any()), bool(vf.any())
        self.kfull_u8 = kf.to(torch.uint8).contiguous().to(device)

        def sel(flags, scale):  # (uint8 head flags, power-of-two scale) for kernels.head_fwht
            return flags.to(torch.uint8).contiguous().to(device), scale

        rep = lambda f: f.repeat_interleave(Hq_per_kv)  # noqa: E731
        self.vfull_q = rep(vf).to(device)                                          # [Hq] bool (Phase 2i post)
        # unnormalized H_Dh (H H = Dh I): k, v -> H x / 16; then q -> H q / (Dh / 16), o -> H o / (Dh / 16) keep
        # q'.k' = q.k and o = v exactly (power-of-two scales: 1/16 and 1/8 at Dh 128, 1/16 and 1/16 at Dh 256)
        sq = 16 / Dh
        self.m_kv = sel(torch.cat([kf, vf]), 1 / 16) if self.any_full else None     # writer input  H x / 16
        self.m_kv_inv = sel(torch.cat([kf, vf]), sq) if self.any_full else None     # (H x / 16 -> x)
        self.m_q = sel(rep(kf), sq) if bool(kf.any()) else None                     # q -> H q / (Dh / 16)
        self.m_o = sel(rep(vf), sq) if bool(vf.any()) else None                     # o -> H o / (Dh / 16)
        self.h32 = K.hadamard(32, device, torch.float32)

    @staticmethod
    def apply(x, m, out=None):
        """x [T, H, Dh] bf16 -> scale * H_Dh x on the flagged heads (one warp per row, exact +-1 butterflies)."""
        if x.stride(-1) != 1 or x.stride(-2) != x.shape[-1]:
            x = x.contiguous()
        return K.head_fwht(x, m[0], m[1], out=out)

    def o_stored_to_orig(self, o):
        """attention output in the stored V basis, fp32 [n, Hq, Dh] -> original basis (block H32 per flagged V group,
        then H o / (Dh / 16) on the whole-head V heads), fp32 (Phase 2i)."""
        n, Hq, Dh = o.shape
        if self.any_block[1]:
            Hkv = self.vmask.shape[0]
            ob = o.reshape(n, Hkv, Hq // Hkv, Dh // 32, 32)
            o = torch.where(self.vmask.bool()[None, :, None, :, None], ob @ self.h32, ob).reshape(n, Hq, Dh)
        if self.any_vfull:
            o = torch.where(self.vfull_q[None, :, None], o @ (_hpm(o.device, Dh) * (16 / Dh)), o)
        return o

    def unrotate_cached(self, k, v, whole=True):
        """dequantized pages (stored basis) [L, Hkv, Dh] fp32 -> original basis (whole=False: the block flags only,
        i.e. the BF16 rows' basis with the whole-head H still applied)."""
        out = []
        Dh = self.Dh
        for x, mask, full, anyb in ((k, self.kmask, self.kfull, self.any_block[0]),
                                    (v, self.vmask, self.vfull, self.any_block[1])):
            if anyb:
                xb = x.reshape(*x.shape[:-1], Dh // 32, 32)
                x = torch.where(mask.bool()[None, :, :, None], xb @ self.h32, xb).reshape(x.shape)
            if whole and bool(full.any()):
                x = torch.where(full[None, :, None], x @ (_hpm(x.device, Dh) * (16 / Dh)), x)
            out.append(x)
        return out


class _Base:
    """per-layer calibrated static base transform under the per-token codebook (FlagRot-C = research wush_k2:
    kv_xform = wush, block-diagonal with 32-wide blocks = the int4a:32 scale groups).  Row-vector matrices of the
    calibration file (qattention.KV['xf']): k' = k xk, v' = v xv are what the writer selects / quantizes (C_q in
    the file is already xq^T C_q xq); decode q' = q xq (q'.k' = q.k), output o = o' xvinv.  The cache, the BF16 slots
    and the kernels all live in the primed basis; the "fa" prefill maps dequantized / BF16 rows back with
    k = k' xq^T, v = v' xvinv.  fp32 batched GEMMs (block zeros included: 128 x 128 per head)."""

    def __init__(self, cal, li, heads, device, G):
        f = lambda n: cal[n][li].index_select(0, heads).float()  # noqa: E731   [Hkv, Dh, Dh]
        xk, xq, xv, xvinv = f("xk"), f("xq"), f("xv"), f("xvinv")
        self.G = G
        self.m_kv = torch.cat([xk, xv]).contiguous().to(device)                            # [2 Hkv, D, D]
        self.m_kv_inv = torch.cat([xq.transpose(1, 2), xvinv]).contiguous().to(device)
        self.m_q = xq.contiguous().to(device)
        self.m_o = xvinv.contiguous().to(device)

    @staticmethod
    def _heads(x, m, mu=None):
        """x [n, H, 128] (any dtype / strides) -> bf16 [n, H, 128] contiguous, row h times m[h]; mu [H0, 128] fp32:
        rows h < H0 are centered first (x - mu in fp32, static key centering: the K rows of a K | V batch)."""
        xf = x.float()
        if mu is not None:
            xf = torch.cat([xf[:, :mu.shape[0]] - mu, xf[:, mu.shape[0]:]], 1)
        y = torch.bmm(xf.transpose(0, 1), m)                                            # [H, n, 128]
        return y.transpose(0, 1).to(torch.bfloat16).contiguous()

    def kv(self, kv, mu=None):
        return self._heads(kv, self.m_kv, mu)

    def kv_inv(self, kv):
        return self._heads(kv, self.m_kv_inv)

    def _grouped(self, x, m, out=None):
        """x [n, Hkv * G, 128]: query / output head h uses m[h // G]."""
        n, H, _ = x.shape
        Hkv = H // self.G
        Dh = x.shape[-1]
        xg = x.float().view(n, Hkv, self.G, Dh).transpose(0, 1).reshape(Hkv, n * self.G, Dh)
        y = torch.bmm(xg, m).view(Hkv, n, self.G, Dh).transpose(0, 1).reshape(n, H, Dh)
        if out is None:
            return y.to(torch.bfloat16)
        out.copy_(y)
        return out

    def q(self, q):
        return self._grouped(q, self.m_q)

    def _grouped32(self, x, m):
        n, H, Dh = x.shape
        Hkv = H // self.G
        xg = x.float().view(n, Hkv, self.G, Dh).transpose(0, 1).reshape(Hkv, n * self.G, Dh)
        return torch.bmm(xg, m).view(Hkv, n, self.G, Dh).transpose(0, 1).reshape(n, H, Dh)

    def q32(self, q):
        """q' = q xq in fp32 (Phase 2i: rounded once, to FP16, by the prep kernel)"""
        return self._grouped32(q, self.m_q)

    def o32(self, o):
        """o = o' xvinv, fp32 in and out"""
        return self._grouped32(o, self.m_o)

    def o(self, o, out=None):
        return self._grouped(o, self.m_o, out=out)

    def kv_inv_split(self, k, v):
        """dequantized pages fp32 [n, Hkv, 128] (primed basis) -> original basis, fp32."""
        Hkv = k.shape[1]
        k = torch.bmm(k.transpose(0, 1), self.m_kv_inv[:Hkv]).transpose(0, 1)
        v = torch.bmm(v.transpose(0, 1), self.m_kv_inv[Hkv:]).transpose(0, 1)
        return k, v


class FlagRotAttentionImpl(AttentionImpl[FlagRotMetadata]):

    def __init__(self, num_heads, head_size, scale, num_kv_heads=None, alibi_slopes=None, sliding_window=None,
                 kv_cache_dtype="auto", logits_soft_cap=None, attn_type=AttentionType.DECODER,
                 kv_sharing_target_layer_name=None, **kwargs):
        if alibi_slopes is not None or sliding_window is not None or logits_soft_cap:
            raise NotImplementedError("FlagRot KV: alibi / sliding window / softcap not supported")
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("FlagRot KV: cross-layer KV sharing not supported")
        self.num_heads, self.head_size, self.scale = num_heads, head_size, float(scale)
        self.num_kv_heads = num_kv_heads or num_heads
        self.D = head_size
        self.G = self.num_heads // self.num_kv_heads
        assert self.num_heads == self.G * self.num_kv_heads, (num_heads, num_kv_heads)
        self.kv_cache_dtype = kv_cache_dtype
        cfg = _vllm_config()
        self.opts = flagrot_options(cfg)
        self.kind = codebook_kind(self.opts)
        self.static = is_static(self.kind)
        self.v2fmt = STATIC_KINDS.get(self.kind)              # static: v2 kernel format "int4" | "int4a"
        if (self.G != 4 or self.D != D) and not gen_ok(self.kind):
            raise ValueError(f"FlagRot backend: GQA group {self.G} / head size {self.D} need a static or key-only 2x2 "
                             f"format (got {self.kind}; the other codebooks: GQA 4, head size 128 only)")
        if self.D not in HEAD_SIZES:
            raise ValueError(f"FlagRot backend: head size {self.D} not supported ({HEAD_SIZES})")
        self.W = self.opts["window"]
        self.S = self.opts["sink"]
        self.T = self.S + self.W                              # BF16 slots per request: S sinks + W-token window
        self.prefill_mode = self.opts["prefill"]
        self.writer = self.opts["writer"]
        # v3 static writer without BF16 slots: k / v enter the writer unrotated, the whole-head H128 is fused into it
        # (with S + W > 0 the BF16 slots hold the rotated rows the window CTA reads, so the rotation stays outside)
        # The v3 writers are head_size 128 only: head_size 256 (e.g. Qwen3.8-27B) keeps the v1 writers
        # (fr_head_fwht + fr_v2_write; fr_cb_write).
        self.w3_static = self.static and self.writer == "v3" and self.D == D
        self.fuse_h = self.w3_static and self.T == 0
        if not self.opts.get("calib"):
            raise ValueError("FlagRot backend: calibration file required (--additional-config "
                             "'{\"flagrot\": {\"calib\": \"...pt\"}}' or FLAGROT_CALIB)")
        self.calib_path = self.opts["calib"]
        meta = load_calib(self.calib_path)["meta"]
        assert meta["head_dim"] == head_size and meta["fmt"] in ("int4:32", "int4a:32", "int2a:32", "kivi2:32") and meta["rb"] == 32, meta
        self.kivi = self.kind == "kivi2"
        if self.kivi != (meta["fmt"] == "kivi2:32"):
            raise ValueError(f"FlagRot backend: calibration {self.calib_path} ({meta['fmt']}) does not match the cache "
                             f"{self.opts['kv_cache_dtype']} (kivi2 needs export_calib.py --kivi kivi | kivi_c)")
        if self.kivi:
            if self.S + self.W == 0 or self.W < 32 or self.prefill_mode != "fa" or self.opts["prefill_fp8"] != "off":
                raise ValueError("FlagRot kivi2: needs a BF16 window W >= 32 (KIVI key groups are encoded when "
                                 "complete; the research config is S4W128), prefill='fa', prefill_fp8='off'")
            if self.G > 8:
                raise ValueError(f"FlagRot kivi2: GQA group {self.G} > 8 not supported")
            if self.opts["center"]:
                raise ValueError("FlagRot kivi2: key centering ('center') not supported")
        # kivi2id: the KIVI-only control (research config 'KIVI': kivi2:32 K per channel / V per token, no selection):
        # the writers encode the identity candidate only; the calibration file only supplies its meta (shapes)
        self.kivi_id = self.kivi and self.opts["kivi_id"]
        # kivi2s: static selection control (kv_dyn att+f+static): the writers encode with the frozen per-(layer, KV
        # head, slot) choices of the calibration file (kmask / kfull / vmask / vfull)
        self.kivi_static = self.kivi and self.opts["kivi_static"]
        self.kivi_f8 = self.kivi and self.opts["kivi_f8"]
        self.kivi_fast = self.kivi and self.opts["kivi_fast"]
        self.kivi_ff = self.kivi and self.opts["kivi_ff"]
        if self.kivi_f8 and meta.get("kv_xform"):
            raise ValueError("FlagRot kivi2p8: the FP8 page format is validated for kivi_sel / KIVI only (no KIVI-C base)")
        if self.kivi_static and "kmask" not in load_calib(self.calib_path):
            raise ValueError(f"FlagRot kivi2s needs a static-selection calibration (export_calib.py --kivi kivi_s), "
                             f"got {self.calib_path}")
        if self.kivi_static:
            # the frozen choice is made under one page format: kivi2s (kivi2:32) vs kivi2p8ffs (kivi2f8:32,
            # export_calib.py --kivi kivi_sp8); a file made for the other format would select differently
            want = "kivi2f8:32" if self.kivi_f8 else "kivi2:32"
            got = meta.get("kivi_fmt", "kivi2:32")
            if got != want:
                raise ValueError(f"FlagRot kivi static selection: calibration {self.calib_path} was frozen under {got}, "
                                 f"the cache is {want} (kivi2p8ffs: flagrot_kivi_sp8_<model>.pt)")
        if self.kivi_id and meta.get("kv_xform"):
            raise ValueError("FlagRot kivi2id (plain KIVI) takes the flagrot_kivi_<model>.pt file, not a KIVI-C base")
        cal_all = load_calib(self.calib_path)
        no_flags = self.static and not any(bool(cal_all[k].any()) for k in ("kmask", "vmask", "kfull", "vfull")
                                           if k in cal_all)
        # (an all-zero static calibration = plain RTN (rtn_a / rtn_a2): no quantizer-dependent choice in it)
        if self.static and not no_flags and meta["fmt"] != {"int4a": "int4a:32", "int2a": "int2a:32"}.get(self.v2fmt, "int4:32"):
            # the headsel choice depends on the quantizer's error: calibrate with the cache's format
            raise ValueError(f"FlagRot backend: static calibration {self.calib_path} was made for {meta['fmt']}, the "
                             f"cache is {self.v2fmt} (scripts/export_calib.py --static --fmt ...)")
        if self.static != (meta.get("kv_flags") == "headsel"):
            raise ValueError(f"FlagRot backend: calibration {self.calib_path} does not match {self.opts['kv_cache_dtype']}"
                             " (static needs scripts/export_calib.py --static)")
        try:
            from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
            tp, rank = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
        except AssertionError:
            tp, rank = 1, 0
        hkv = meta["num_kv_heads"]
        if tp <= hkv:
            assert hkv % tp == 0 and self.num_kv_heads == hkv // tp, (hkv, tp, self.num_kv_heads)
            self.kv_heads = list(range(rank * self.num_kv_heads, (rank + 1) * self.num_kv_heads))
        else:
            assert self.num_kv_heads == 1
            self.kv_heads = [rank * hkv // tp]
        # calibration rows: per model layer index (meta["layers"]: the layers the file covers, e.g. the full-attention
        # layers of a hybrid model; default: every layer, row = layer index)
        self.calib_rows = {li: i for i, li in enumerate(meta["layers"])} if meta.get("layers") is not None else None
        self.layout = tile_layout(self.kind, self.num_kv_heads, self.G, self.D)
        self.qscale = self.scale * LOG2E
        self.max_rows = 1
        self.max_len = 1
        if cfg is not None:
            sc = cfg.scheduler_config
            caps = list(getattr(getattr(cfg, "compilation_config", None), "cudagraph_capture_sizes", None) or [])
            self.max_rows = max([sc.max_num_seqs] + caps)
            self.max_len = int(self.opts.get("plan_len") or cfg.model_config.max_model_len)
        self.ring = None
        if self.T > 0:
            _patch_hybrid_req_idx()                           # hybrid models: req_idx for the BF16 slots
        # Phase 2k: prefix caching with BF16 slots (prefix_cache.py): every token is in its page at its real position,
        # the scheduler hook recomputes >= W tokens before every reused prefix and keeps a prefix past the sinks only
        # when the sink pool holds their BF16 rows; the worker hook tells the impl which pool rows to copy per step
        from .. import prefix_cache as PCM
        self.pc = PCM.pc_config(cfg) if (self.T > 0 and cfg is not None) else None
        self.sink_pool = None
        if self.pc is not None and self.kind == "kivi2":
            bs = int(getattr(getattr(cfg, "cache_config", None), "block_size", 0) or 0)
            if bs % 32:
                raise ValueError(f"FlagRot kivi2 with prefix caching needs a block size that is a multiple of 32 (got "
                                 f"{bs}): a cached block must hold whole KIVI key groups (--block-size 32)")
        if self.pc is not None:
            if self.prefill_mode != "fa":
                raise ValueError("FlagRot: prefix caching with BF16 slots needs prefill='fa'")
            PCM.install_worker()
        if self.T > 0:
            # BF16 rows, one per request slot (+1 padding row); allocated here, before vLLM profiles the memory, so
            # the KV-cache budget accounts for them.  ring[r, p] = K | V of sink token p < S; ring[r, S + (p - S) % W]
            # = K | V of window token p >= S (a ring over the newest W tokens past the sinks).
            R = cfg.scheduler_config.max_num_seqs if cfg is not None else 64
            self.ring = torch.zeros(R + 1, self.T, 2 * self.num_kv_heads, self.D, dtype=torch.bfloat16,
                                    device=torch.device("cuda", torch.cuda.current_device()))
            if self.pc is not None and self.S > 0:
                # BF16 sink rows shared between requests with the same first S tokens (prefix_cache.SinkIndex slots)
                self.sink_pool = torch.zeros(self.pc["N"], self.S, 2 * self.num_kv_heads, self.D,
                                             dtype=torch.bfloat16, device=self.ring.device)
        self._fa = None
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version, is_flash_attn_varlen_func_available
        if is_flash_attn_varlen_func_available():
            from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
            self._fa = flash_attn_varlen_func
            self.fa_version = get_flash_attn_version(head_size=head_size)
        self._pre_orig, self._pre_map = None, {}
        self.prefill_fp8 = self.opts["prefill_fp8"]
        if self.prefill_fp8 in ("int4", "int4ac") and self.kind not in ("static_a", "konlya"):
            raise ValueError(f"FlagRot: prefill_fp8={self.prefill_fp8} needs an int4a static or konlya cache")
        if (self.prefill_fp8 in ("int4", "int4ac") or self.prefill_fp8.startswith("emu_")) and \
                (self.D != D or self.G != 4):
            raise ValueError(f"FlagRot: prefill_fp8={self.prefill_fp8} has GQA-4 / head-128 kernels only "
                             f"(G {self.G}, head size {self.D}); use 'off'")
        if self.prefill_fp8 == "int4ac" and not self.static:
            raise ValueError("FlagRot: prefill_fp8=int4ac (A+C) has no per-token-transform (konlya) kernel; use int4")
        meta_x = meta.get("kv_xform", "")
        self.has_base = bool(meta_x)
        # static key centering (int2c): mu per (layer, KV head, channel) of the post-RoPE calibration keys
        self.center_path = self.opts["center"]
        self.center = bool(self.center_path)
        ckey = meta.get("kv_center", "") or ""                 # calibration made on centered keys (static / base)
        key_dep = self.has_base or (self.static and not no_flags)
        if self.center_path:
            cm = load_calib(self.center_path)["meta"]
            if (cm["head_dim"] != head_size or cm["num_kv_heads"] != meta["num_kv_heads"]
                    or cm.get("layers") != meta.get("layers") or cm.get("model") != meta.get("model")):
                raise ValueError(f"FlagRot backend: center file {self.center_path} ({cm.get('model')}, layers "
                                 f"{cm.get('layers')}, Hkv {cm['num_kv_heads']}, head {cm['head_dim']}) does not match the "
                                 f"calibration {self.calib_path} ({meta.get('model')}, {meta.get('layers')}, "
                                 f"{meta['num_kv_heads']}, {meta['head_dim']})")
            if not bool(load_calib(self.center_path)["mu"].any()):
                # kv_center = auto chose no centering for this (model, method): mu = 0, i.e. exactly the plain cache
                logger.info_once("FlagRot: center file %s holds mu = 0 (auto: no centering); centering off",
                                 self.center_path)
                self.center = False
            if key_dep and ckey != cm["kv_center"]:
                raise ValueError(f"FlagRot backend: {self.calib_path} was calibrated on {'centered (' + ckey + ')' if ckey else 'uncentered'} "
                                 f"keys, the center file is kv_center={cm['kv_center']}: use the matching calibration "
                                 "(export_calib.py --static / --wush with --kv_center)")
        elif key_dep and ckey:
            raise ValueError(f"FlagRot backend: {self.calib_path} was calibrated on centered keys (kv_center={ckey}); "
                             "it needs the option 'center' (flagrot_center_<model>.pt)")
        if self.has_base and (self.static or self.kind not in ("konlya", "konlya2", "kivi2")):
            raise ValueError(f"FlagRot: calibration {self.calib_path} has a static base transform ({meta_x}); it "
                             "is implemented for the asymmetric key-only 2x2 codebook (konlya) only")
        if (self.has_base or self.S > 0) and self.prefill_fp8 != "off":
            raise ValueError("FlagRot: the base transform (FlagRot-C) and sink tokens support prefill_fp8='off' "
                             "(the BF16 prefill path) only")
        if self.S > 0 and self.prefill_mode != "fa":
            raise ValueError("FlagRot: sink tokens need prefill='fa'")
        if self.prefill_fp8 in ("unit", "amax", "had") and (self._fa is None or self.fa_version != 3):
            logger.warning_once("FlagRot: FP8 prefill attention needs FlashAttention 3; using BF16")
            self.prefill_fp8 = "off"
        self._unit_descale = None
        self._allq = None
        self._krot = None
        # Phase 2i: prefix-cached prefill straight from the INT4 pages (prefill_mq.py); the eligible configurations,
        # else the "fa" path for every request
        self.mq_mode = self.opts["prefill_mq"]
        self.mq_max_new = self.opts["mq_max_new"] or (MQ_MAX_NEW_DEFAULT if self.D == D else MQ_MAX_NEW_DEFAULT_256)
        self.mq = (self.mq_mode != "off" and self.prefill_mode == "fa" and self.prefill_fp8 == "off"
                   and self.kind in ASYM_KINDS + INT2_KINDS and self.D in HEAD_SIZES and 1 <= self.G <= 64
                   and MODE != "ref" and not FA_FP32)
        logger.info_once("FlagRot KV %s (%s, %s%s%s), sink %d, window %d, prefill %s (attention %s; INT4-direct "
                         "prefix-cached prefill %s), writer %s, kernels %s", self.opts["kv_cache_dtype"], self.kind,
                         ("kivi2:32, identity only (plain KIVI)" if self.opts.get("kivi_id") else
                          "kivi2:32, static selection" if self.opts.get("kivi_static") else "kivi2:32")
                         .replace("kivi2:32", "kivi2f8:32 (p8)" if self.opts.get("kivi_f8") else "kivi2:32")
                         .replace(":32", (":32, fast decode + fast prefix dequant" if self.opts.get("kivi_ff") else
                                          ":32, fast decode") if self.opts.get("kivi_fast") else ":32")
                         if self.kind == "kivi2" else self.opts["quant"], f", base {meta_x}" if self.has_base else "",
                         f", key centering {self.center_path}" if self.center else "", self.S, self.W,
                         self.prefill_mode, "bf16" if self.prefill_fp8 == "off" else f"fp8 e4m3, {self.prefill_fp8}",
                         (f"{self.mq_mode}, new segment <= {self.mq_max_new}" if self.mq_mode == "auto"
                          else self.mq_mode) if self.mq else "off", self.writer,
                         KIVI._lib_path() if self.kivi else K._lib_path())

    # ------------------------------------------------------------------ per-layer constants
    def _state(self, layer, device):
        st = getattr(layer, "_flagrot_state", None)
        if st is not None:
            return st
        from vllm.model_executor.models.utils import extract_layer_index
        li = extract_layer_index(layer.layer_name)
        if self.calib_rows is not None:
            if li not in self.calib_rows:
                raise ValueError(f"FlagRot backend: layer {li} ({layer.layer_name}) is not in the calibration file "
                                 f"{self.calib_path} (layers {sorted(self.calib_rows)})")
            li = self.calib_rows[li]
        cal = load_calib(self.calib_path)
        hs = torch.tensor(self.kv_heads)
        if self.kivi:
            f = lambda n: cal[n][li].index_select(0, hs).float().contiguous().to(device)  # noqa: E731
            fix = {}
            if self.kivi_static:
                for kind in ("k", "v"):
                    m = cal[f"{kind}mask"][li].index_select(0, hs).int()            # [Hkv, D/32]
                    fl = cal[f"{kind}full"][li].index_select(0, hs).int()           # [Hkv]
                    w = (m << torch.arange(m.shape[-1], dtype=torch.int32)).sum(-1) | (fl << 16)
                    fix[kind] = w.int().contiguous().to(device)
            st = dict(kfix=fix.get("k"), vfix=fix.get("v"),
                      qs=f("qs"), lz=f("lz"), co=f("co"), vM=f("vM") if "vM" in cal else None,
                      qscale=float(cal["qscale"]), base=_Base(cal, li, hs, device, self.G) if self.has_base else None,
                      kmu=None, kmu_orig=None)
            layer._flagrot_state = st
            self._stats_layer = li
            KIVI.lib()
            return st
        if self.static:
            st = _Static(cal, li, hs, device, self.G, self.D)
        else:
            cq = cal["cq"][li].index_select(0, hs).float()
            st = dict(CT=cq.transpose(1, 2).contiguous().to(device),
                      co=cal["co"][li].index_select(0, hs).float().contiguous().to(device),
                      base=_Base(cal, li, hs, device, self.G) if self.has_base
                      else None, w3=None)
            if self.writer == "v3" and self.kind in K.W3_KINDS and self.D == D:
                # writer v3 constants of this layer's C_q (host fp64, ~43 KB fp16 per layer); FlagRot-C: the file's
                # C_q is already in the base-transformed basis the writer selects in.  head_size 128 only (the v3
                # kernels); head_size 256 keeps the v1 codebook writer
                C1, beta = K.w3_calib(cq)
                st["w3"] = (C1.to(device), beta.to(device))
                K.side_stream(device)                       # created outside CUDA-graph capture
        if self.center:
            mu = load_calib(self.center_path)["mu"][li].index_select(0, hs).double()       # [Hkv, Dh] (original basis)
            mu_orig = mu.float().contiguous().to(device)
            if self.static and st.any_kfull:
                # stored basis of the whole-head H heads: bf16(H k / 16) - H mu / 16 (the writer subtracts after its H)
                hm = _hpm(torch.device("cpu"), self.D).double()
                kf = st.kfull.cpu()
                mu = torch.where(kf[:, None], (mu @ hm) / 16, mu)
            kmu = mu.float().contiguous().to(device)
            if self.static:
                st.kmu, st.kmu_orig = kmu, mu_orig
            else:
                st["kmu"] = None if st["base"] is not None else kmu     # FlagRot-C: centered before the base (_store)
                st["kmu_orig"] = mu_orig
        elif self.static:
            st.kmu = st.kmu_orig = None
        else:
            st["kmu"] = st["kmu_orig"] = None
        layer._flagrot_state = st
        self._stats_layer = li
        K.lib()
        return st

    def _plan_fn(self, B):
        # kivi2p8ff: memoized per batch size (kivi_workspace evaluates every B <= max_rows; uncached, that was
        # thousands of plan / option lookups per eager decode step on the host)
        c = self.__dict__.setdefault("_plan_cache", {})
        r = c.get(B)
        if r is None:
            r = c[B] = decode_plan(self.kind, B, self.max_len, self.num_kv_heads, self.G, self.D,
                                   fast=getattr(self, "kivi_fast", False) if self.kind == "kivi2" else None)
        return r

    def _pool(self, kv_cache):
        nb = kv_cache.shape[0]
        assert kv_cache.is_contiguous() and kv_cache.dtype == torch.uint8
        return kv_cache.view(nb, self.num_kv_heads, -1)

    def _kv(self, key, value, n):
        """K | V rows of n tokens as [n, 2 Hkv, 128] bf16 (a strided view of the qkv output when K and V are
        adjacent in it, else a copy)."""
        Hkv = self.num_kv_heads
        k, v = key[:n], value[:n]
        if k.dim() == 2:
            k, v = k.view(n, Hkv, self.D), v.view(n, Hkv, self.D)
        es = k.element_size()
        if (k.stride(-1) == 1 and k.stride(-2) == self.D and v.stride() == k.stride()
                and v.data_ptr() == k.data_ptr() + Hkv * self.D * es):
            return k.as_strided((n, 2 * Hkv, self.D), (k.stride(0), self.D, 1))
        return torch.cat([k, v], 1)

    def _base(self, st):
        return None if self.static else st["base"]

    def _kmu(self, st):
        """writer centering vector (stored basis, fp32 [Hkv, Dh]) or None"""
        return st.kmu if self.static else st["kmu"]

    def _kmu_orig(self, st):
        return st.kmu_orig if self.static else st["kmu_orig"]

    def _center_rows(self, st, kv):
        """K | V rows in the stored basis -> the BF16-slot rows: K rows bf16(k - mu_stored) (fp32 subtraction), V rows
        unchanged.  Identity without centering (and for FlagRot-C, whose _store already centered)."""
        kmu = self._kmu(st)
        if kmu is None:
            return kv
        Hkv = self.num_kv_heads
        return torch.cat([(kv[:, :Hkv].float() - kmu).to(torch.bfloat16), kv[:, Hkv:]], 1)

    def _center_key(self, st, key, n):
        """the step's keys (original basis) -> bf16(k - mu), for every attention path that reads the current chunk's
        keys in BF16 (prefill: FA window / MQ BF16 parts / kernel windows): consistent with the centered pages / slots."""
        mu = self._kmu_orig(st)
        k = key[:n]
        kc = (k.reshape(n, self.num_kv_heads, self.D).float() - mu).to(torch.bfloat16)
        return kc.reshape(k.shape)

    def _store(self, st, kv, center=False):
        """K | V rows [n, 2 Hkv, 128] -> the basis the writer / BF16 slots / kernels take (static: whole-head H128
        heads rotated; FlagRot-C: the base transform).  center (cache update of raw keys, FlagRot-C with key centering):
        (k - mu) xk in fp32 (the other formats center in the writer kernels / _center_rows)."""
        if self.static:
            return _Static.apply(kv, st.m_kv) if st.m_kv is not None and not self.fuse_h else kv
        b = st["base"]
        return b.kv(kv, st["kmu_orig"] if center else None) if b is not None else kv

    def _unstore(self, st, kv):
        """inverse of _store (BF16 slots -> original basis, for the "fa" prefill)."""
        if self.static:
            return _Static.apply(kv, st.m_kv_inv) if st.m_kv_inv is not None else kv
        b = st["base"]
        return b.kv_inv(kv) if b is not None else kv

    def _write(self, st, kv, slots, pool, dec=False):
        """kv (the _store basis; v3 static with S = W = 0: unrotated) -> pages at slots.  dec: decode-step rows
        (one token per sequence at scattered slots: the v3 static writer's decode kernel).  Key centering: the
        kernels subtract _kmu (stored basis) from the K rows in fp32 before encoding."""
        kmu = self._kmu(st)
        if self.static:
            if self.w3_static:
                K.w3_static_write(kv, slots, st.wmask, st.hfull if self.fuse_h else None, pool, PAGE,
                                  fmt=self.v2fmt, path=2 if dec else 0, kmu=kmu)
            else:
                K.v2_write(kv, slots, st.wmask, 32, pool, PAGE, fmt=self.v2fmt, kmu=kmu)
        elif st["w3"] is not None and kv.shape[0] > W3_CB_V1_MAX_T:
            K.cb_write3(kv, slots, st["CT"], st["co"], st["w3"][0], st["w3"][1], self.kind, pool, PAGE, kmu=kmu)
        else:
            K.cb_write(kv, slots, st["CT"], st["co"], self.kind, pool, PAGE, kmu=kmu)

    def _attn(self, st, q, pool, bt, cached, out, nsplit, chunk, ws, tail=None, tail_len=None, tail_idx=None,
              tail_bs=None, tail_ts=None, kv_start=0):
        tkw = dict(tail=tail, tail_len=tail_len, tail_idx=tail_idx, tail_bs=tail_bs, tail_ts=tail_ts, page=PAGE,
                   kv_start=kv_start)
        if self.static:
            K.v2_attn(q, pool, bt, cached, out, nsplit, chunk, ws, self.qscale, kmask=st.kmask_attn,
                      vmask=st.vmask_attn, rb=32, fmt=self.v2fmt, **tkw)
        else:
            K.cb_attn(q, pool, bt, cached, out, nsplit, chunk, ws, self.qscale, self.kind, **tkw)

    # ------------------------------------------------------------------ write path
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping) -> None:
        tok = OPTIMER.start("write")
        self._kv_cache_update(layer, key, value, kv_cache, slot_mapping)
        OPTIMER.stop(tok)

    def _kv_cache_update(self, layer, key, value, kv_cache, slot_mapping) -> None:
        n = slot_mapping.shape[0]
        if n == 0 or kv_cache.numel() == 0:
            return
        st = self._state(layer, key.device)
        pool = self._pool(kv_cache)
        if self.kivi:
            return self._kivi_update(st, layer, key, value, pool, slot_mapping, n)
        kv = self._store(st, self._kv(key, value, n), center=self.center)   # static whole-head H128 / FlagRot-C base
        from vllm.model_executor.layers.attention.attention import get_attention_context
        if self.T == 0:
            # v3 static writer: a pure decode batch (incl. CUDA-graph padding rows, slot -1) takes the decode kernel
            dec = False
            if self.w3_static:
                md = get_attention_context(layer.layer_name)[0]
                dec = md is not None and md.num_decodes == md.num_reqs
            self._write(st, kv, slot_mapping, pool, dec=dec)
            return
        md = get_attention_context(layer.layer_name)[0]
        if md is None:                                          # profiling / dummy run without metadata
            self._write(st, kv, torch.full_like(slot_mapping, -1), pool)
            return
        # Phase 2k: every token goes to its page at its real position now (one writer launch, as with W = 0); the
        # BF16 row additionally keeps the sinks and the newest W tokens.  Codes are those of an encoding at window
        # exit (the writer is deterministic per token), and every full block is a valid prefix-cache block.
        nd = md.num_decodes
        self._write(st, kv, slot_mapping, pool, dec=nd == md.num_reqs)
        ring = self.ring.view(-1, 2 * self.num_kv_heads, self.D)    # [(R + 1) T, 2 Hkv, Dh]
        kvr = self._center_rows(st, kv) if self.center else kv     # BF16 slots: k - mu (stored basis)
        if nd > 0:
            ring.index_copy_(0, md.dec_ring_pos, kvr[:nd])         # the new token's BF16 slot (sink or window)
        self._pre_orig, self._pre_map = None, {}
        if md.num_reqs > nd:
            self._prefill_update(st, md, kvr, pool, ring, nd)

    def _kivi_update(self, st, layer, key, value, pool, slot_mapping, n):
        """kivi2 cache update: every token's value at its slot; the BF16 ring (sinks + window) as every format; the key
        groups completed in this step encoded from the chunk rows / the ring (csrc/kivi.cu)."""
        from vllm.model_executor.layers.attention.attention import get_attention_context
        kv = self._store(st, self._kv(key, value, n))                   # kivi_c: primed basis (base transform)
        KIVI.v_write(kv, slot_mapping, st["co"], st["vM"], pool, self.num_kv_heads, self.D, ident=self.kivi_id,
                     fix=st["vfix"], f8=self.kivi_f8)
        md = get_attention_context(layer.layer_name)[0]
        if md is None:                                                  # profiling / dummy run
            return
        nd = md.num_decodes
        ring = self.ring.view(-1, 2 * self.num_kv_heads, self.D)
        if nd > 0:
            ring.index_copy_(0, md.dec_ring_pos, kv[:nd])
        self._pre_orig, self._pre_map = None, {}
        if md.num_reqs > nd:
            kp = self._kivi_prefill_groups(md, kv.device)
            if kp is not None:                                          # before the ring rows are overwritten
                KIVI.k_encode(kv, kp[0], kp[1], st["qs"], st["lz"], st["qscale"], pool, self.num_kv_heads, self.D,
                              src2=ring, ident=self.kivi_id, fix=st["kfix"], f8=self.kivi_f8)
        if nd > 0 and md.kivi_sidx is not None:
            KIVI.k_encode(ring, md.kivi_sidx, md.kivi_dslot, st["qs"], st["lz"], st["qscale"], pool,
                          self.num_kv_heads, self.D, ident=self.kivi_id, fix=st["kfix"], f8=self.kivi_f8)
        if md.num_reqs > nd:
            self._prefill_update(st, md, kv, pool, ring, nd)

    def _kivi_prefill_groups(self, md, device):
        """(sidx, dslot) [N, 32] int64 of the key groups completed by this step's prefill chunks (host plan shared by
        all layers, cached on the metadata object): group g of request r with s <= 32 g + 31 < L; position p of it
        comes from the chunk row q0 + p - s (p >= s), the ring row of p (S <= p < s, encoded as -2 - row) or is a
        sink (-1).  A chunk never starts inside a group of a reused prefix (prefix_cache.py: reuse points are
        multiples of 32), so the ring holds every earlier position of a straddling group."""
        c = getattr(md, "_flagrot_kivi", None)
        if c is not None:
            return c[0]
        S, W, T = self.S, self.W, self.T
        qsl = md.query_start_loc_cpu.tolist()
        seq = md.seq_lens_cpu.tolist()
        from .. import prefix_cache as PCM
        w = PCM.WORKER
        sid, rows, pos0 = [], [], []
        ar = np.arange(32)
        for r in range(md.num_decodes, md.num_reqs):
            q0, q1 = qsl[r], qsl[r + 1]
            L = int(seq[r])
            s = L - (q1 - q0)
            g0, g1 = s // 32, L // 32
            if g1 <= g0:
                continue
            base = int(md.req_idx[r]) * T
            rv = w.rv[base // T] if w is not None else 0
            if rv > 0 and rv % 32 and s < (rv // 32 + 1) * 32:
                raise RuntimeError(f"FlagRot kivi2: a reused prefix ends at {rv}, inside a key group (prefix_cache "
                                   "must align reuse points to 32)")
            pos = (np.arange(g0, g1) * 32)[:, None] + ar[None]
            ring_row = base + S + (pos - S) % W
            sidx = np.where(pos < S, -1, np.where(pos >= s, q0 + pos - s, -2 - ring_row))
            sid.append(sidx)
            rows.append(np.full(g1 - g0, r))
            pos0.append(np.arange(g0, g1) * 32)
        if not sid:
            md._flagrot_kivi = (None,)
            return None
        sidx = torch.from_numpy(np.concatenate(sid).astype(np.int64)).to(device, non_blocking=True)
        rr = torch.from_numpy(np.concatenate(rows).astype(np.int64)).to(device, non_blocking=True)
        p0 = torch.from_numpy(np.concatenate(pos0).astype(np.int64)).to(device, non_blocking=True)
        pos = p0[:, None] + torch.arange(32, device=device)[None]
        dslot = (md.block_table[rr[:, None], pos // PAGE].long() * PAGE + pos % PAGE).contiguous()
        out = (sidx.contiguous(), dslot)
        md._flagrot_kivi = (out,)
        return out

    def _pc_step(self, md, device):
        """(read rows, read slots, write rows, write slots) of the sink pool in this step (prefix_cache.WORKER), as
        device index tensors cached on the metadata object; None when nothing is copied."""
        c = getattr(md, "_flagrot_pc", None)
        if c is not None:
            return c[0]
        from .. import prefix_cache as PCM
        w = PCM.WORKER
        out = None
        if w is not None and self.sink_pool is not None and (w.reads or w.writes):
            nr_, nw_ = len(w.reads), len(w.writes)
            buf = torch.tensor([r for r, _ in w.reads] + [p for _, p in w.reads] + [r for r, _ in w.writes]
                               + [p for _, p in w.writes], dtype=torch.int64).to(device, non_blocking=True)
            out = (buf[:nr_], buf[nr_:2 * nr_], buf[2 * nr_:2 * nr_ + nw_], buf[2 * nr_ + nw_:], nr_, nw_)
        md._flagrot_pc = (out,)
        return out

    def _bf_plan(self, md, nd, device):
        """host plan of the BF16 row updates of this step's prefill requests, shared by all layers (cached on the
        metadata object; one host->device copy): per request row base = req * T, chunk positions p < S -> BF16 slot
        p, the newest W positions >= S of the chunk -> window slot S + (p - S) % W; the BF16 rows of [p0, s), p0 =
        max(S, s - W + 1), that the chunk's queries still read in BF16 are gathered first (their slots are
        overwritten).  A request that reuses a cached prefix has BF16 rows from its first computed position rv on only
        (prefix_cache.WORKER.rv): its INT4-direct chunks read the window positions [p0, rv) from the pages through the
        kernel (ext = rv - 1 of the mq plan; ws = the first BF16 window position); chunks of the "fa" path, and the
        last chunk when the first decode window reaches below rv (fewer than W tokens computed: hybrid models), fill
        those window slots from the pages (dequantized)."""
        c = getattr(md, "_flagrot_bf", None)
        if c is not None and c[0] == (self.S, self.W):
            return c[1]
        from .. import prefix_cache as PCM
        S, W, T = self.S, self.W, self.T
        qsl = md.query_start_loc_cpu.tolist()
        seq = md.seq_lens_cpu.tolist()
        if md.req_idx is None:
            raise RuntimeError("FlagRot sink / window: prefill needs CommonAttentionMetadata.req_idx (model runner V2)")
        w = PCM.WORKER
        mq_on = self.mq and MQ_OVERRIDE != "off"
        pre_src, put_dst, put_src = [], [], []
        pre = {}            # prefill row r -> (offset in the gathered rows, first position, count)
        hit_rows = []       # (r, a, b, base): window rows [a, b) of a reused prefix filled from the pages
        rows = []           # (r, q0, q1, L, s, base)
        ext, ws = {}, {}    # mq plan: r -> last page key every row sees (rv - 1); first BF16 window position
        for r in range(nd, md.num_reqs):
            q0, q1 = qsl[r], qsl[r + 1]
            L = int(seq[r])
            s = L - (q1 - q0)
            base = int(md.req_idx[r]) * T
            rows.append((r, q0, q1, L, s, base))
            if W > 0:
                p0 = max(S, s - W + 1)
                rv = w.rv[base // T] if w is not None else 0
                a = p0                                       # first window position read from the ring
                if rv > p0 and s > p0:
                    if mq_on and self._mq_take(q1 - q0, L):
                        ext[r] = rv - 1                     # [p0, rv) through the INT4-direct kernel (pages)
                        ws[r] = rv
                        a = rv
                    else:
                        hit_rows.append((r, p0, min(rv, s), base))
                if rv > 0 and L == w.total[base // T] and rv > max(S, L - W + 1):
                    # the first decode window reaches below rv: its rows from the pages (none of this chunk's slots)
                    hit_rows.append((r, max(S, L - W + 1), rv, base))
                if s > a:
                    pre[r] = (len(pre_src), a, s - a)
                    pre_src.extend(base + S + (p - S) % W for p in range(a, s))
            n_s = min(max(S - s, 0), q1 - q0)
            if n_s > 0:
                put_dst.extend(range(base + s, base + s + n_s))
                put_src.extend(range(q0, q0 + n_s))
            if W > 0:
                b0 = max(s + n_s, L - W)
                if L > b0:
                    put_dst.extend(base + S + (p - S) % W for p in range(b0, L))
                    put_src.extend(range(q0 + b0 - s, q1))
        n1, n2 = len(pre_src), len(put_dst)
        buf = torch.tensor(pre_src + put_dst + put_src, dtype=torch.int64).to(device, non_blocking=True)
        plan = SimpleNamespace(pre=pre, hit_rows=hit_rows, rows=rows, pre_src=buf[:n1], put_dst=buf[n1:n1 + n2],
                               put_src=buf[n1 + n2:], n_pre=n1, n_put=n2, ext=ext, ws=ws)
        md._flagrot_bf = ((self.S, self.W), plan)
        return plan

    def _prefill_update(self, st, md, kv, pool, ring, nd):
        """prefill chunks (the pages were written by the caller): the BF16 rows of _bf_plan in a few batched ops per
        layer (+ the sink pool copies of prefix_cache.WORKER)."""
        S = self.S
        dev = kv.device
        pcs = self._pc_step(md, dev)
        plan = self._bf_plan(md, nd, dev)
        T3 = self.ring.shape[1:]
        if pcs is not None and pcs[4]:                          # sink pool -> BF16 rows of requests reusing a prefix
            self.ring.view(-1, *T3)[pcs[0], :S] = self.sink_pool[pcs[1]]
        for (r, a, b, base) in plan.hit_rows:                   # reused prefix: window rows [a, b) from the pages
            pos = torch.arange(a, b, device=dev)
            ring.index_copy_(0, base + S + (pos - S) % self.W, self._page_rows_store(st, pool, md, r, a, b))
        # window rows the chunks' queries still read in BF16, gathered before their slots are overwritten (original
        # basis, one conversion for all requests)
        self._pre_orig = self._unstore(st, ring.index_select(0, plan.pre_src)) if plan.n_pre else None
        self._pre_map = plan.pre
        if plan.n_put:
            ring.index_copy_(0, plan.put_dst, kv.index_select(0, plan.put_src))
        if pcs is not None and pcs[5]:                          # BF16 sinks of this step -> sink pool
            self.sink_pool[pcs[3]] = self.ring.view(-1, *T3)[pcs[2], :S]

    # ------------------------------------------------------------------ read path
    def forward(self, layer, query, key, value, kv_cache, attn_metadata: FlagRotMetadata, output=None,
                output_scale=None, output_block_scale=None) -> torch.Tensor:
        assert output is not None
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("FlagRot KV: fused output quantization not supported")
        if attn_metadata is None:
            return output.fill_(0)
        md = attn_metadata
        n = md.num_actual_tokens
        if n == 0:
            return output
        st = self._state(layer, query.device)
        out = output[:n].view(n, self.num_heads * self.D)
        q = query[:n].view(n, self.num_heads, self.D)
        if MODE == "ref":
            return self._forward_ref(q, key, value, md, out, output)
        pool = self._pool(kv_cache)
        nd = md.num_decodes
        if self.center and md.num_reqs > nd:
            key = self._center_key(st, key, n)                 # prefill paths read the chunk's keys: k - mu
        if nd > 0:
            self._decode(st, q[:nd], pool, md, out[:nd])
        if md.num_reqs > nd:
            tok = OPTIMER.start("prefill_attn")
            if self.kivi and KIVI_DEC_PREFILL:
                self._prefill_kivi_dec(st, q, key, value, pool, md, out)
            elif self.prefill_mode == "kernel":
                self._prefill_kernel(st, q, key, value, pool, md, out)
            else:
                rows = self._prefill_mq(st, q, key, value, pool, md, out) if self.mq and MQ_OVERRIDE != "off" else None
                if rows is None or rows:
                    self._prefill_fa(st, q, key, value, pool, md, out, rows=rows)
            OPTIMER.stop(tok)
        return output

    def _decode(self, st, q, pool, md, out):
        nd = q.shape[0]
        if self.kivi:
            ws = kivi_workspace(q.device, self.max_rows, self.num_kv_heads, self.G, self.D, self._plan_fn,
                                pkey=(self.kind, self.max_len, self.kivi_fast))
            base = st["base"]
            q = base.q(q) if base is not None else (q if q.stride(1) == self.D and q.stride(2) == 1 else q.contiguous())
            KIVI.attn(q, pool.view(pool.shape[0], -1), md.block_table[:nd], md.dec_cached, out, md.nsplit, md.chunk,
                      ws, self.qscale, self.S, self.G, ring=self.ring, tail_len=md.dec_tail_len,
                      tail_idx=md.dec_tail_idx, T=self.T, f8=self.kivi_f8, fast=self.kivi_fast)
            if base is not None:
                ov = out.view(nd, self.num_heads, self.D)
                base.o(ov, out=ov)
            return
        ws = decode_workspace(q.device, self.layout, self.max_rows, self._plan_fn)
        static = self.static
        base = self._base(st)
        if static and st.m_q is not None:
            q = _Static.apply(q, st.m_q)
        elif base is not None:
            q = base.q(q)                                      # q' = q xq (FlagRot-C)
        elif q.stride(1) != self.D or q.stride(2) != 1:
            q = q.contiguous()
        tkw = {}
        if self.T > 0:
            r = self.ring
            tkw = dict(tail=r, tail_len=md.dec_tail_len, tail_idx=md.dec_tail_idx, tail_bs=r.stride(0),
                       tail_ts=r.stride(1), kv_start=self.S)     # the sinks are in the pages too, read in BF16
        self._attn(st, q, pool, md.block_table[:nd], md.dec_cached, out, md.nsplit, md.chunk, ws, **tkw)
        if static and st.m_o is not None:                      # whole-head V heads: o -> H o / 8, in place
            ov = out.view(nd, self.num_heads, self.D)
            _Static.apply(ov, st.m_o, out=ov)
        elif base is not None:                                 # o = o' xvinv, in place
            ov = out.view(nd, self.num_heads, self.D)
            base.o(ov, out=ov)

    # -- prefill through the decode kernel: one kernel row per prompt token
    def _prefill_kernel(self, st, q, key, value, pool, md, out):
        nd = md.num_decodes
        qsl = md.query_start_loc_cpu.tolist()
        seq = md.seq_lens_cpu.tolist()
        dev = q.device
        static = self.static
        W = self.W
        for r in range(nd, md.num_reqs):
            q0, q1 = qsl[r], qsl[r + 1]
            nq = q1 - q0
            L = int(seq[r])
            s = L - nq
            pos = torch.arange(s, L, device=dev, dtype=torch.int32)
            qr = q[q0:q1]
            qr = _Static.apply(qr, st.m_q) if static and st.m_q is not None else qr.contiguous()
            if self._base(st) is not None:
                qr = self._base(st).q(qr)
            # one block-table row per prompt token: a stride-0 broadcast of the request's row (master's static v2
            # kernel handles bt_stride 0 since a later fix; the codebook kernels always
            # did); FLAGROT_BT_COPY=1 passes a materialized copy (nq x row width int32) as before
            bt = md.block_table[r:r + 1].expand(nq, -1)
            if BT_COPY:
                bt = bt.contiguous()
            Lmax = L
            ns, chunk = (K.plan_v2(nq, Lmax, self.num_kv_heads, PAGE, fmt=self.v2fmt, G=self.G, Dh=self.D) if static
                         else K.plan_cb(nq, Lmax, self.num_kv_heads, PAGE, self.kind, G=self.G, Dh=self.D))
            ws = prefill_workspace(dev, self.layout, nq, nq * self.layout[0] * (ns + 1))
            o = torch.empty(nq, self.num_heads * self.D, dtype=q.dtype, device=dev)
            tkw = {}
            if W > 0:
                cached = torch.clamp(pos + 1 - W, min=0)
                # window rows: positions [p0, L) as one contiguous BF16 buffer; row i reads W tokens from i + 1 - W
                win, p0 = self._window_buffer(st, md, r, key, value, q0, q1, s, orig=False)
                tl = torch.clamp(pos + 1, max=W).int()
                ti = (cached - p0).int()
                tkw = dict(tail=win, tail_len=tl, tail_idx=ti, tail_bs=win.stride(0), tail_ts=win.stride(0))
            else:
                cached = pos + 1
            self._attn(st, qr, pool, bt, cached.int(), o, ns, chunk, ws, **tkw)
            if static and st.m_o is not None:
                o = _Static.apply(o.view(nq, self.num_heads, self.D), st.m_o).view(nq, -1)
            elif self._base(st) is not None:
                o = self._base(st).o(o.view(nq, self.num_heads, self.D)).view(nq, -1)
            out[q0:q1].copy_(o)

    def _prefill_kivi_dec(self, st, q, key, value, pool, md, out):
        """kivi2 check path (FLAGROT_KIVI_DEC_PREFILL=1): every prompt token i of a fresh prefill attends through the
        decode kernel as a decode row would: pages [S, i + 1 - W) (kv_start = S) and the BF16 rows of the sinks
        [0, min(S, i + 1)) and the window [max(S, i + 1 - W), i + 1) as its tail (chunks of KIVI_DEC_CHUNK rows)."""
        nd = md.num_decodes
        qsl = md.query_start_loc_cpu.tolist()
        seq = md.seq_lens_cpu.tolist()
        dev = q.device
        S, W, Hkv, G, Dh = self.S, self.W, self.num_kv_heads, self.G, self.D
        T = S + W
        assert st["base"] is None, "FLAGROT_KIVI_DEC_PREFILL: kivi2 without a base transform only"
        j = torch.arange(T, device=dev)
        npg = pool.shape[0]
        for r in range(nd, md.num_reqs):
            q0, q1 = qsl[r], qsl[r + 1]
            nq = q1 - q0
            L = int(seq[r])
            assert L == nq, "FLAGROT_KIVI_DEC_PREFILL: fresh prefills only (no cached prefix, one chunk)"
            kv = self._kv(key, value, q1)[q0:q1]                           # [nq, 2 Hkv, Dh] original basis
            for c0 in range(0, nq, KIVI_DEC_CHUNK):
                c1 = min(nq, c0 + KIVI_DEC_CHUNK)
                nc = c1 - c0
                pos = torch.arange(c0, c1, device=dev)
                lo = torch.clamp(pos + 1 - W, min=S)
                idx = torch.where(j[None] < S, j[None], lo[:, None] + (j[None] - S))
                valid = idx <= pos[:, None]
                tl = valid.sum(1).int()
                ring = kv[torch.where(valid, idx, torch.zeros_like(idx)).reshape(-1)].contiguous()   # [nc T, 2Hkv, Dh]
                cached = torch.where(pos + 1 - W > S, pos + 1 - W, torch.zeros_like(pos)).int()
                bt = md.block_table[r:r + 1].expand(nc, -1)
                ns, chunk = KIVI.plan(nc, L, Hkv, occ=KIVI.fast_occ(Dh, G) if self.kivi_fast else None)
                ws = KIVI.Workspace(nc * Hkv * (ns + 1), G, Dh, dev)
                o = torch.empty(nc, self.num_heads * Dh, dtype=q.dtype, device=dev)
                ti = torch.arange(nc, device=dev, dtype=torch.int32)
                KIVI.attn(q[q0 + c0:q0 + c1].contiguous(), pool.view(npg, -1), bt, cached, o, ns, chunk, ws,
                          self.qscale, S, G, ring=ring, tail_len=tl, tail_idx=ti, T=T, f8=self.kivi_f8, fast=self.kivi_fast)
                if KIVI_FAST_CHECK and self.kivi_fast:
                    ns2, chunk2 = KIVI.plan(nc, L, Hkv)
                    ws2 = KIVI.Workspace(nc * Hkv * (ns2 + 1), G, Dh, dev)
                    o2 = torch.empty_like(o)
                    KIVI.attn(q[q0 + c0:q0 + c1].contiguous(), pool.view(npg, -1), bt, cached, o2, ns2, chunk2, ws2,
                              self.qscale, S, G, ring=ring, tail_len=tl, tail_idx=ti, T=T, f8=self.kivi_f8, fast=False)
                    d = (o.float() - o2.float()).abs()
                    st_ = KIVI_FAST_STATS
                    st_["calls"] += 1
                    st_["max_abs"] = max(st_["max_abs"], d.max().item())
                    st_["max_rel"] = max(st_["max_rel"], (d.max() / o2.float().abs().max().clamp_min(1e-30)).item())
                    st_["sum_abs"] += d.sum().item()
                    st_["n"] += d.numel()
                    st_["bitexact"] += int((d == 0).sum().item())
                    st_["elems"] += d.numel()
                out[q0 + c0:q0 + c1].copy_(o)

    def _window_buffer(self, st, md, r, key, value, q0, q1, s, orig=True):
        """BF16 K | V rows of positions [p0, L) for the window part of the prefill attention, in the ORIGINAL basis
        (orig=False: the kernels' input basis, for the decode-kernel / Phase 2f prefill paths): the saved part [p0, s)
        (_pre_orig / _pre_map of the cache update: ring rows, filled from the pages for a reused prefix), the sink slots of positions
        [max(0, s - W + 1), min(S, s)) and the chunk.  Returns ([L - p0, 2 Hkv, Dh], p0)."""
        kv = self._kv(key, value, q1)[q0:q1]
        if WIN_ROUNDTRIP:      # debug: the chunk's rows through the stored basis and back (the Phase 2b-2j numerics)
            kv = self._unstore(st, self._store(st, kv.contiguous()))
        pre = self._pre_map.get(r)
        parts, p0 = [kv], s
        if pre is not None:
            off, p0, cnt = pre
            parts.insert(0, self._pre_orig[off:off + cnt])
        if self.S > 0:
            ps = max(0, s - self.W + 1)
            if ps < min(self.S, s):
                parts.insert(0, self._unstore(st, self._sink_rows(md, r, ps, min(self.S, s))))
                p0 = ps
        out = parts[0].contiguous() if len(parts) == 1 else torch.cat(parts)
        return (out if orig else self._store(st, out)), p0

    def _sink_rows(self, md, r, a, b):
        """BF16 slots of sink positions [a, b) of request r (stored basis) [b - a, 2 Hkv, 128]."""
        base = int(md.req_idx[r]) * self.T
        return self.ring.view(-1, 2 * self.num_kv_heads, self.D)[base + a:base + b]

    def _quant_prefix(self, st, pool, md, r, n):
        """K, V [n, Hkv, Dh] bf16 (original basis) of positions [0, n) as the quantized part of the "fa" prefill
        sees them: sink positions < S from the BF16 slots, the rest dequantized from the pages (real positions)."""
        k, v = self._cached_kv(st, pool, md, r, n)
        S = min(self.S, n)
        if S > 0:
            Hkv = self.num_kv_heads
            sk = self._unstore(st, self._sink_rows(md, r, 0, S))
            k[:S], v[:S] = sk[:, :Hkv], sk[:, Hkv:]
        return k, v

    def _page_rows_store(self, st, pool, md, r, a, b):
        """K | V [b - a, 2 Hkv, Dh] bf16 of positions [a, b) dequantized from request r's pages, in the basis of the
        BF16 rows (_store: static whole-head H applied, block flags not; FlagRot-C primed): the window rows of a
        reused prefix (Phase 2k)."""
        k, v = self._cached_kv_impl(st, pool, md, r, b, start=a, basis="store")
        return torch.cat([k, v], 1)

    # -- prefill: dequantize + FlashAttention
    def _cached_kv(self, st, pool, md, r, n, fp8="off"):
        """K, V [n, Hkv, 128] of the first n cached tokens of request r: BF16 in the original basis (fp8 = "off"), or
        FP8 E4M3 workspaces + their descales [1, Hkv] fp32:
          "had"   K -> H K / 16 (whole-head Hadamard, fp32, on the dequantized slices; q gets H q / 8 in _fp8_q, so
                  q'.k' = q.k exactly), V in the original basis; scale 1.0 (vLLM's default k / v scale); the fp32
                  slices are rounded to FP8 directly
          "unit"  original basis, scale 1.0
          "amax"  original basis, BF16 dequantization, then per KV head scale = amax / 448 over the n tokens"""
        tok = OPTIMER.start("prefill_dequant")
        if fp8 == "off":
            k, v = self._cached_kv_impl(st, pool, md, r, n)
            OPTIMER.stop(tok)
            return k, v
        if fp8 in ("unit", "had"):
            krot = self._hk(pool.device) if fp8 == "had" else None
            k, v = self._cached_kv_impl(st, pool, md, r, n, dtype=FP8, krot=krot)
            kd = vd = self._unit(pool.device)
        else:
            k, v = self._cached_kv_impl(st, pool, md, r, n)
            k, kd = _fp8_amax(k)
            v, vd = _fp8_amax(v)
        OPTIMER.stop(tok)
        return k, v, kd, vd

    def _unit(self, device):
        """descale 1.0, shape (num_seqs = 1, Hkv) as vLLM passes it to FA3 (layer._q_scale.expand(...))."""
        if self._unit_descale is None:
            self._unit_descale = torch.ones(1, self.num_kv_heads, dtype=torch.float32, device=device)
        return self._unit_descale

    @staticmethod
    def _q8(q):
        """q -> FP8 E4M3 as vLLM's attention layer does for kv_cache_dtype=fp8 (QuantFP8, static per-tensor scale
        layer._q_scale = 1.0: x * (1 / scale), clamp to +-448, cast)."""
        return q.clamp(-FP8_MAX, FP8_MAX).to(FP8)

    def _hk(self, device):
        """H / 16 (unnormalized +-1 Sylvester Hadamard, fp32): k -> H k / 16 for the "had" FP8 workspace."""
        if self._krot is None:
            self._krot = _hpm(device, self.D) / 16
        return self._krot

    def _fp8_q(self, q, fp8):
        """prefill queries of the FP8 path: "had" -> H q / 8 (fr_head_fwht on every head, exact +-1 butterflies),
        then vLLM's FP8 cast (scale 1)."""
        if fp8 == "had":
            if self._allq is None:
                self._allq = torch.ones(self.num_heads, dtype=torch.uint8, device=q.device)
            q = _Static.apply(q, (self._allq, 16 / self.D))     # H q / (Dh / 16): q'.k' = q.k
        return self._q8(q)

    def _cached_kv_impl(self, st, pool, md, r, n, dtype=torch.bfloat16, krot=None, start=0, basis="orig"):
        if self.kivi:
            base = st["base"] if basis == "orig" else None
            if KIVI_DQ_TORCH:
                kk, vv = KIVI.dequant(pool, md.block_table[r], start, n, self.num_kv_heads, self.D, f8=self.kivi_f8)
            else:
                kk, vv = KIVI.dequant_cuda(pool, md.block_table[r], start, n, self.num_kv_heads, self.D,
                                           torch.float32 if base is not None else dtype, f8=self.kivi_f8,
                                           v2=self.kivi_ff)
                if KIVI_DQ_CHECK and self.kivi_ff:
                    k1, v1 = KIVI.dequant_cuda(pool, md.block_table[r], start, n, self.num_kv_heads, self.D,
                                               torch.float32 if base is not None else dtype, f8=self.kivi_f8)
                    it = torch.int16 if k1.dtype == torch.bfloat16 else torch.int32
                    KIVI_DQ_STATS["dequant_calls"] += 1
                    KIVI_DQ_STATS["dequant_elems"] += 2 * k1.numel()
                    KIVI_DQ_STATS["dequant_diff"] += int((k1.contiguous().view(it) != kk.contiguous().view(it)).sum().item()
                                                         + (v1.contiguous().view(it) != vv.contiguous().view(it)).sum().item())
            if base is not None:                                     # KIVI-C: primed basis -> original
                kk, vv = base.kv_inv_split(kk.float(), vv.float())
            return kk.to(dtype), vv.to(dtype)
        pg0 = start // PAGE                                         # positions [start, n) (pages pg0 ..)
        npg = (n + PAGE - 1) // PAGE - pg0
        Hkv = self.num_kv_heads
        k = torch.empty(npg * PAGE, Hkv, self.D, dtype=dtype, device=pool.device)
        v = torch.empty_like(k)
        sat = dtype == FP8                                          # E4M3 has no inf: saturate before the cast
        step = 256                                                  # pages per slice: bounds the fp32 temporaries
        for p0 in range(0, npg, step):
            p1 = min(npg, p0 + step)
            chunks = pool.index_select(0, md.block_table[r, pg0 + p0:pg0 + p1].long())
            if self.static:
                kk, vv = K.dequant_v2_pages(chunks, PAGE, torch.float32, fmt=self.v2fmt, Dh=self.D)
                kk, vv = st.unrotate_cached(kk.reshape(-1, Hkv, self.D), vv.reshape(-1, Hkv, self.D),
                                            whole=basis == "orig")
            else:
                kk, vv = K.dequant_cb_pages(chunks, PAGE, self.kind, torch.float32, Dh=self.D)
                if st["base"] is not None and basis == "orig":      # FlagRot-C: primed basis -> original
                    kk, vv = st["base"].kv_inv_split(kk.reshape(-1, Hkv, self.D), vv.reshape(-1, Hkv, self.D))
            if krot is not None:
                kk = kk.reshape(-1, self.D) @ krot
            if sat:
                kk, vv = kk.clamp(-FP8_MAX, FP8_MAX), vv.clamp(-FP8_MAX, FP8_MAX)
            k[p0 * PAGE:p1 * PAGE] = kk.reshape(-1, Hkv, self.D)
            v[p0 * PAGE:p1 * PAGE] = vv.reshape(-1, Hkv, self.D)
        o = start - pg0 * PAGE
        return k[o:o + n - start], v[o:o + n - start]

    def _fa_call(self, q, k, v, cu_q, cu_k, mq, mk, out=None, window=None, lse=False, descale=None):
        if FA_FP32 and descale is None:
            return self._fa_fp32(q, k, v, cu_q, cu_k, out, window, lse)
        kw = dict(fa_version=self.fa_version) if self.fa_version is not None else {}
        if window is not None:
            kw["window_size"] = window
        if descale is not None:                 # FP8 inputs: FA3 with q / k / v descales [num_seqs, Hkv] (vLLM fp8)
            kw.update(q_descale=descale[0], k_descale=descale[1], v_descale=descale[2])
        return self._fa(q=q, k=k, v=v, out=out, cu_seqlens_q=cu_q, max_seqlen_q=mq, cu_seqlens_k=cu_k,
                        max_seqlen_k=mk, softmax_scale=self.scale, causal=True, return_softmax_lse=lse, **kw)

    def _fa_fp32(self, q, k, v, cu_q, cu_k, out, window, lse):
        """FLAGROT_FA_FP32=1: fp32 attention with FA's varlen semantics (bottom-right causal, local window (wl, 0)),
        returns (o, lse [Hq, nq]) like FA when lse.  Debug only (host syncs, O(nq nk) memory per sequence)."""
        cq, ck = cu_q.tolist(), cu_k.tolist()
        G = self.num_heads // self.num_kv_heads
        o_all = torch.empty(q.shape[0], self.num_heads, self.D, dtype=q.dtype, device=q.device) if out is None else \
            out.view(q.shape[0], self.num_heads, self.D)
        l_all = torch.empty(self.num_heads, q.shape[0], dtype=torch.float32, device=q.device)
        for b in range(len(cq) - 1):
            qs, ks = q[cq[b]:cq[b + 1]].float(), k[ck[b]:ck[b + 1]].float()
            vs = v[ck[b]:ck[b + 1]].float()
            nq, nk = qs.shape[0], ks.shape[0]
            for h0 in range(0, self.num_heads, 8):          # 8 query heads at a time: bounds the score memory
                kvh = torch.arange(h0, min(h0 + 8, self.num_heads), device=q.device) // G     # KV head of each q head
                kh = ks[:, kvh]                                                  # [nk, 8, Dh]
                vh = vs[:, kvh]
                sc = torch.einsum("qhd,khd->hqk", qs[:, h0:h0 + 8], kh) * self.scale
                i = torch.arange(nq, device=q.device)[:, None] + (nk - nq)
                j = torch.arange(nk, device=q.device)[None, :]
                m = j <= i
                if window is not None and window[0] >= 0:
                    m = m & (j >= i - window[0])
                sc = sc.masked_fill(~m, float("-inf"))
                ls = torch.logsumexp(sc, -1)                                     # [8, nq]
                p = torch.exp(sc - ls[..., None]).nan_to_num(0.0)
                o_all[cq[b]:cq[b + 1], h0:h0 + 8] = torch.einsum("hqk,khd->qhd", p, vh).to(q.dtype)
                l_all[h0:h0 + 8, cq[b]:cq[b + 1]] = torch.where(torch.isneginf(ls), torch.full_like(ls, math.inf), ls)
        return (o_all, l_all) if lse else o_all

    def _prefill_fa(self, st, q, key, value, pool, md, out, rows=None):
        """the "fa" path for the prefill requests `rows` (None: all)."""
        nd = md.num_decodes
        qsl = md.query_start_loc_cpu.tolist()
        seq = md.seq_lens_cpu.tolist()
        dev = q.device
        W = self.W
        Hq = self.num_heads
        for r in (range(nd, md.num_reqs) if rows is None else rows):
            q0, q1 = qsl[r], qsl[r + 1]
            nq = q1 - q0
            L = int(seq[r])
            s = L - nq
            qr = q[q0:q1]
            o = out[q0:q1].view(nq, Hq, self.D)
            fp8 = self.prefill_fp8
            if fp8 in ("int4", "int4ac"):
                self._prefill_int4(st, qr, key, value, pool, md, r, q0, q1, L, s, o)
                continue
            if fp8.startswith("emu_"):
                self._prefill_emu(st, qr, key, value, pool, md, r, q0, q1, L, s, o)
                continue
            if W == 0:
                cq = torch.tensor([0, nq], dtype=torch.int32, device=dev)
                ck = torch.tensor([0, L], dtype=torch.int32, device=dev)
                if FP8_STATS.on and s == 0:
                    FP8_STATS.collect(self, st, pool, md, r, qr, key, value, q0, q1, cq, ck)
                if fp8 == "off":
                    k, v = self._quant_prefix(st, pool, md, r, L)
                    self._fa_call(qr, k, v, cq, ck, nq, L, out=o)
                else:
                    k, v, kd, vd = self._cached_kv(st, pool, md, r, L, fp8)
                    self._fa_call(self._fp8_q(qr, fp8), k, v, cq, ck, nq, L, out=o,
                                  descale=(self._unit(dev), kd, vd))
                continue
            # (b) BF16 window: keys [p0, L), local window (W - 1, 0): j in [i - W + 1, i]
            kvo, p0 = self._window_buffer(st, md, r, key, value, q0, q1, s)
            kb, vb = kvo[:, :self.num_kv_heads], kvo[:, self.num_kv_heads:]
            cu = torch.tensor([0, nq, 0, L - p0, 0, L - max(s, W), 0, L - W], dtype=torch.int32).to(dev, non_blocking=True)
            cq, ck = cu[0:2], cu[2:4]
            ob, lb = self._fa_call(qr, kb.contiguous(), vb.contiguous(), cq, ck, nq, L - p0, window=[W - 1, 0],
                                   lse=True)
            # (a) quantized keys [0, L - W) for the queries i >= W (j <= i - W)
            a0 = max(s, W)
            if L - W <= 0 or a0 >= L:
                o.copy_(ob)
                continue
            na = L - a0
            cqa, cka = cu[4:6], cu[6:8]
            if fp8 == "off":
                k, v = self._quant_prefix(st, pool, md, r, L - W)     # (sinks: BF16 rows for j < S)
                oa, la = self._fa_call(qr[a0 - s:], k, v, cqa, cka, na, L - W, lse=True)
            else:                                # quantized part in FP8 (the BF16 window call above stays BF16)
                k, v, kd, vd = self._cached_kv(st, pool, md, r, L - W, fp8)
                oa, la = self._fa_call(self._fp8_q(qr[a0 - s:], fp8), k, v, cqa, cka, na, L - W, lse=True,
                                       descale=(self._unit(dev), kd, vd))
            o.copy_(ob)
            if self.kivi_ff and oa.dtype == torch.bfloat16 and oa.is_contiguous():
                KIVI.lse_merge(oa, la, ob[a0 - s:], lb[:, a0 - s:], o[a0 - s:])
                if KIVI_DQ_CHECK:
                    ref = _merge(oa, la, ob[a0 - s:], lb[:, a0 - s:])
                    KIVI_DQ_STATS["merge_calls"] += 1
                    KIVI_DQ_STATS["merge_elems"] += ref.numel()
                    KIVI_DQ_STATS["merge_diff"] += int((ref.view(torch.int16) != o[a0 - s:].contiguous().view(torch.int16)).sum().item())
            else:
                o[a0 - s:] = _merge(oa, la, ob[a0 - s:], lb[:, a0 - s:])

    # -- Phase 2i: prefix-cached prefill straight from the INT4 pages (prefill_mq.py, csrc/mq_prefill.cu)
    def _mq_take(self, nq: int, L: int) -> bool:
        """dispatcher: the INT4-direct path for short new segments ("auto": nq <= mq_max_new; the crossover against
        the "fa" path's gather + dequant + FA is at several hundred to a few thousand new tokens), every request with "all"."""
        return (MQ_OVERRIDE or self.mq_mode) == "all" or nq <= self.mq_max_new

    def _mq_select(self, md, device):
        """(plan, fa_rows) of this step, built once and shared by all layers (cached on the metadata object)."""
        key = (self.G, self.num_kv_heads, self.D, self.W, self.S, self.mq_mode, self.mq_max_new, MQ_OVERRIDE)
        c = getattr(md, "_flagrot_mq", None)
        if c is not None and c[0] == key:
            return c[1], c[2]
        from .. import prefill_mq as MQ
        qsl = md.query_start_loc_cpu.tolist()
        seq = md.seq_lens_cpu.tolist()
        sel, rest = [], []
        for r in range(md.num_decodes, md.num_reqs):
            q0, q1 = qsl[r], qsl[r + 1]
            L = int(seq[r])
            if q1 > q0 and self._mq_take(q1 - q0, L):
                sel.append((r, q0, q1, L, L - (q1 - q0)))
            else:
                rest.append(r)
        ext = self._bf_plan(md, md.num_decodes, device).ext if (self.T > 0 and sel) else None
        plan = MQ.build_plan(sel, self.G, self.num_kv_heads, self.W, device, sink=self.S, D=self.D,
                             ext=ext) if sel else None
        md._flagrot_mq = (key, plan, rest)
        for k, v in (("mq_reqs", len(sel)), ("mq_tokens", sum(q1 - q0 for _, q0, q1, _, _ in sel)),
                     ("fa_reqs", len(rest)), ("fa_tokens", sum(qsl[r + 1] - qsl[r] for r in rest))):
            MQ_STATS[k] = MQ_STATS.get(k, 0) + v
        return plan, rest

    def _prefill_mq(self, st, q, key, value, pool, md, out):
        """INT4-direct attention of the selected prefill requests (one prep / attention / merge launch for all of
        them); returns the request rows left to the "fa" path.  Static formats: q rotated into the keys' stored
        basis, output rotated back from the V basis; FlagRot-C: q' = q xq (fp32), o = o' xvinv; window / sinks: the
        page part (keys v <= i - S - W) merged by LSE with the BF16 window FA call and the BF16 sinks, as the "fa"
        path's research mask."""
        from .. import prefill_mq as MQ
        if not getattr(self, "_mq_ready", False):
            try:
                MQ.lib()
                self._mq_ready = True
            except Exception as e:  # noqa: BLE001   library missing / not loadable: the "fa" path for everything
                logger.warning("FlagRot: INT4-direct prefill unavailable (%s); using the fa path", e)
                self.mq = False
                return None
        plan, rest = self._mq_select(md, q.device)
        if plan is None:
            return rest
        tok = OPTIMER.start("prefill_mq")
        W, S, G = self.W, self.S, self.G
        base = self._base(st)
        if base is not None:
            qsrc, idx = base.q32(q.index_select(0, plan.idx.long())), None
        else:
            qsrc, idx = q, plan.idx
        kf = km = None
        if self.static:
            kf = st.kfull_u8 if st.any_kfull else None
            km = st.kmask if st.any_block[0] else None
        q16 = MQ.prep_q(qsrc, self.qscale, G, idx=idx, kfull=kf, kmask=km)
        opart, ml = MQ.attn(q16, pool, md.block_table, plan, G, W, konly=not self.static, int2=self.kind in INT2_KINDS)
        post = base is not None or (self.static and (st.any_block[1] or st.any_vfull))
        if W == 0 and S == 0 and not post:
            MQ.merge(plan, opart, ml, G, out=out)
        else:
            o, lse = MQ.merge(plan, opart, ml, G, fp32=True, want_lse=W + S > 0)
            if base is not None:
                o = base.o32(o)
            elif post:
                o = st.o_stored_to_orig(o)
            if W == 0 and S == 0:
                out.index_copy_(0, plan.idx.long(), o.to(out.dtype).view(plan.m, -1))
            else:
                self._mq_bf16_parts(st, q, key, value, md, plan, o, lse, out)
        OPTIMER.stop(tok)
        return rest

    def _mq_bf_plan(self, md, plan, device):
        """host plan of the BF16 window / sink attention of the INT4-direct requests of this step (compact row
        order of the mq plan; cached on the metadata object, one host->device copy): window keys of request r =
        positions [ps, L), ps = max(0, s - W + 1), assembled from the sink slots ([ps, min(S, s))), the gathered
        window rows ([max(S, ps), s), _pre_orig) and the chunk rows ([s, L)); sink keys = the S sink slots."""
        c = getattr(md, "_flagrot_mqbf", None)
        if c is not None and c[0] == (self.S, self.W):
            return c[1]
        S, W, T = self.S, self.W, self.T
        bf = self._bf_plan(md, md.num_decodes, device)
        base_of = {r: base for (r, _, _, _, _, base) in bf.rows}
        ks_src, ks_dst, kp_src, kp_dst, kc_src, kc_dst = [], [], [], [], [], []
        cu_q, cu_k, row_req, row_pos, sk_src, sk_len = [0], [0], [], [], [], []
        ks_row = []                 # window-sink keys: row in the sink buffer (request i's sinks at i * S ..)
        mq = mk = 0
        for i, (r, q0, q1, L, s) in enumerate(plan.reqs):
            nq = q1 - q0
            base = base_of[r]
            if W > 0:
                ps = max(0, s - W + 1, bf.ws.get(r, 0))          # first BF16 window key (rv for a reused prefix)
                k0 = cu_k[-1]
                a1 = min(S, s)
                if a1 > ps:
                    ks_src.extend(range(base + ps, base + a1))
                    ks_dst.extend(range(k0, k0 + a1 - ps))
                    ks_row.extend(range(i * S + ps, i * S + a1))
                pre = bf.pre.get(r)
                if pre is not None:
                    off, p0, cnt = pre
                    assert p0 >= ps, (p0, ps)
                    kp_src.extend(range(off, off + cnt))
                    kp_dst.extend(range(k0 + p0 - ps, k0 + p0 - ps + cnt))
                kc_src.extend(range(q0, q1))
                kc_dst.extend(range(k0 + s - ps, k0 + L - ps))
                cu_k.append(k0 + L - ps)
                mk = max(mk, L - ps)
            cu_q.append(cu_q[-1] + nq)
            mq = max(mq, nq)
            if S > 0:
                # sinks not in the BF16 window part: j < max(ws, i - W + 1); stored as lim + W (the kernels' j <= pos - W)
                wsr = max(0, bf.ws.get(r, 0))
                row_req.extend([i] * nq)
                row_pos.extend(max(wsr, i_ - W + 1) - 1 + W for i_ in range(s, L))
                sk_src.extend(range(base, base + S))
                sk_len.append(min(S, L))
        # window-key map of the gather kernel: (kind << 32) | row, kind 0 = sink rows (sk_src order), 1 = gathered
        # window rows, 2 = batch rows
        nk = cu_k[-1]
        kmap = [0] * nk
        for r_, d_ in zip(ks_row, ks_dst):
            kmap[d_] = r_                                    # row in the sink buffer (sk_src order)
        for r_, d_ in zip(kp_src, kp_dst):
            kmap[d_] = (1 << 32) | r_
        for r_, d_ in zip(kc_src, kc_dst):
            kmap[d_] = (2 << 32) | r_
        parts = [ks_src, ks_dst, kp_src, kp_dst, kc_src, kc_dst, row_req, row_pos, sk_src, sk_len, kmap]
        sizes = [len(x) for x in parts]
        flat = [v for x in parts for v in x]
        buf = torch.tensor(flat + cu_q + cu_k, dtype=torch.int64).to(device, non_blocking=True)
        out, o = [], 0
        for n_ in sizes:
            out.append(buf[o:o + n_])
            o += n_
        cuq = buf[o:o + len(cu_q)].to(torch.int32)
        cuk = buf[o + len(cu_q):].to(torch.int32)
        rows = [x for (_, q0, q1, _, _) in plan.reqs for x in (q0, q1)]
        contig = all(rows[2 * j + 1] == rows[2 * j + 2] for j in range(len(plan.reqs) - 1))
        p = SimpleNamespace(ks_src=out[0], ks_dst=out[1], kp_src=out[2], kp_dst=out[3], kc_src=out[4], kc_dst=out[5],
                            row_req=out[6], row_pos=out[7], sk_src=out[8], sk_len=out[9], kmap=out[10], cu_q=cuq, cu_k=cuk,
                            max_q=mq, max_k=mk, nk=cu_k[-1], nreq=len(plan.reqs), sizes=sizes,
                            q_rows=(rows[0], rows[-1]) if contig else None)
        md._flagrot_mqbf = ((self.S, self.W), p)
        return p

    def _mq_bf16_parts(self, st, q, key, value, md, plan, o, lse, out):
        """W / S > 0: the BF16 window (one FlashAttention varlen call for all INT4-direct requests of the step, local
        window (W - 1, 0), keys gathered by k_win_gather) and sink (j < S, j <= i - W) parts, merged with the page
        part by LSE and written to the output rows by k_bf_merge (prefill_mq.py; FLAGROT_BF_TORCH=1: the same in
        PyTorch ops)."""
        if BF_TORCH:
            return self._mq_bf16_parts_torch(st, q, key, value, md, plan, o, lse, out)
        from .. import prefill_mq as MQ
        W, S = self.W, self.S
        Hkv, Dh = self.num_kv_heads, self.D
        dev = q.device
        bp = self._mq_bf_plan(md, plan, dev)
        idx = plan.idx.long()
        ring = self.ring.view(-1, 2 * Hkv, Dh)
        o_w = l_w = sk = None
        if S > 0:                                   # the sinks of every request, original basis (also window keys)
            sk = self._unstore(st, ring.index_select(0, bp.sk_src)).contiguous()
        if W > 0:
            n = q.shape[0]
            kch, vch = key[:n].view(n, -1), value[:n].view(n, -1)
            kw, vw = MQ.win_gather(bp.kmap, bp.nk, sk if bp.sizes[0] else None,
                                   self._pre_orig if bp.sizes[2] else None, kch, vch, Hkv, Dh)
            qc = q[bp.q_rows[0]:bp.q_rows[1]] if bp.q_rows is not None else q.index_select(0, idx)
            if qc.stride(-1) != 1 or qc.stride(-2) != Dh:
                qc = qc.contiguous()
            o_w, l_w = self._fa_call(qc, kw, vw, bp.cu_q, bp.cu_k, bp.max_q, bp.max_k, window=[W - 1, 0], lse=True)
            o_w = o_w.contiguous()
            l_w = l_w.contiguous()
        MQ.bf_merge(q, idx, o.contiguous(), lse.contiguous(), o_w, l_w, sk, bp.row_req if S > 0 else None,
                    bp.row_pos if S > 0 else None, bp.sk_len if S > 0 else None, S, W, self.scale, out, Hkv)

    def _mq_bf16_parts_torch(self, st, q, key, value, md, plan, o, lse, out):
        """W / S > 0: the BF16 window (one FlashAttention varlen call for all requests, local window (W - 1, 0)) and
        sink (j < S, j <= i - W; batched fp32) parts of every INT4-direct request, merged with the page part by LSE
        (original basis), written to the output rows of the plan."""
        from ..prefill_mq import ref_partial_lse_merge
        W, S = self.W, self.S
        Hkv, G, Dh = self.num_kv_heads, self.G, self.D
        dev = q.device
        bp = self._mq_bf_plan(md, plan, dev)
        idx = plan.idx.long()
        qc = q.index_select(0, idx)                                 # [m, Hq, Dh] bf16, compact rows
        m = qc.shape[0]
        parts = [(o, lse)]
        if W > 0:
            kvo = self._kv(key, value, q.shape[0])                  # original basis K | V rows of the batch
            ring = self.ring.view(-1, 2 * Hkv, Dh)
            kw = torch.empty(bp.nk, 2 * Hkv, Dh, dtype=torch.bfloat16, device=dev)
            if bp.sizes[0]:
                kw.index_copy_(0, bp.ks_dst, self._unstore(st, ring.index_select(0, bp.ks_src)))
            if bp.sizes[2]:
                kw.index_copy_(0, bp.kp_dst, self._pre_orig.index_select(0, bp.kp_src))
            kw.index_copy_(0, bp.kc_dst, kvo.index_select(0, bp.kc_src))
            parts.append(self._fa_call(qc, kw[:, :Hkv].contiguous(), kw[:, Hkv:].contiguous(), bp.cu_q, bp.cu_k,
                                       bp.max_q, bp.max_k, window=[W - 1, 0], lse=True))
        if S > 0:
            ring = self.ring.view(-1, 2 * Hkv, Dh)
            sk = self._unstore(st, ring.index_select(0, bp.sk_src)).float().view(bp.nreq, S, 2 * Hkv, Dh)
            kr = sk.index_select(0, bp.row_req)                     # [m, S, 2 Hkv, Dh]
            sc = torch.einsum("mkgd,mskd->mkgs", qc.float().view(m, Hkv, G, Dh), kr[:, :, :Hkv]) * self.scale
            j = torch.arange(S, device=dev)[None, :]
            ok = (j <= (bp.row_pos - W)[:, None]) & (j < bp.sk_len.index_select(0, bp.row_req)[:, None])
            sc = sc.masked_fill(~ok[:, None, None, :], float("-inf"))
            ls = torch.logsumexp(sc, -1)                            # [m, Hkv, G]
            pr = torch.exp(sc - ls[..., None]).nan_to_num(0.0)
            os_ = torch.einsum("mkgs,mskd->mkgd", pr, kr[:, :, Hkv:])
            parts.append((os_.reshape(m, Hkv * G, Dh), ls.reshape(m, Hkv * G).transpose(0, 1)))
        out.index_copy_(0, idx, ref_partial_lse_merge(parts).to(out.dtype).view(m, -1))

    def _sink_attn(self, qr, k, v, s, W):
        """fp32 attention of the queries at positions s .. s + nq - 1 over the BF16 sink keys j (< S) with
        j <= i - W; returns (o [nq, Hq, D], lse [Hq, nq], -inf = no visible sink)."""
        G = self.G
        kh, vh = k.repeat_interleave(G, 1), v.repeat_interleave(G, 1)
        sc = torch.einsum("qhd,khd->hqk", qr.float(), kh) * self.scale
        i = torch.arange(s, s + qr.shape[0], device=qr.device)[:, None]
        j = torch.arange(k.shape[0], device=qr.device)[None, :]
        sc = sc.masked_fill(~(j <= i - W)[None], float("-inf"))
        ls = torch.logsumexp(sc, -1)
        p = torch.exp(sc - ls[..., None]).nan_to_num(0.0)
        return torch.einsum("hqk,khd->qhd", p, vh), ls

    def _int4_attn(self, st, q, pool, bt, nk, p0, shift, out=None):
        """Phase 2f INT4-direct prefill kernels (prefill_int4.py), no gather / dequantized workspace:
          "int4"   static modes: QK in FP16 with q rotated into the keys' stored basis (prep_q_f16) and K = (c - z) s
                   in FP16 (one rounding, 2^-12) straight from the codes; key-only 2x2 codebook (konlya): QK in FP16
                   on the K tile dequantized + inverse-transformed per key in shared memory; PV in FP8 (V, P E4M3)
          "int4ac" static modes: design A+C as specified (exact E4M3 K codes, per-group fp32 scale, q E4M3 hi + lo)
        Returns (o [n, Hq, 128] in the original basis, lse [Hq, n])."""
        from .. import prefill_int4 as PI
        kf = st.kfull.to(torch.uint8) if self.static and bool(st.kfull.any()) else None
        km = st.kmask if self.static and (st.any_block[0] or kf is not None) else None
        if self.static and self.prefill_fp8 == "int4ac":
            qh, ql, qf = PI.prep_q(q, self.scale, kfull=kf, kmask=km)
            o, lse = PI.int4_prefill(qh, ql, qf, pool, bt, nk, p0, shift, out=out, stages=PREFILL_INT4_STAGES)
        elif self.static:
            qs = PI.prep_q_f16(q, kfull=kf, kmask=km) if (kf is not None or km is not None) else q
            o, lse = PI.int4_prefill_kbf(qs, self.scale, pool, bt, nk, p0, shift, out=out, stages=PREFILL_KBF_STAGES)
        else:
            assert self.kind == "konlya", f"INT4 prefill kernel: codebook kind {self.kind} not supported"
            o, lse = PI.int4_prefill_kbf(q, self.scale, pool, bt, nk, p0, shift, konly=True, out=out,
                                         stages=PREFILL_KBF_STAGES)
        if self.static and (st.any_block[1] or bool(st.vfull.any())):    # V stored basis -> original
            n = o.shape[0]
            ob = o.float().view(n, self.num_kv_heads, -1, 4, 32)
            vm = st.vmask.bool()[None, :, None, :, None]
            ob = torch.where(vm, ob @ st.h32, ob).view(n, -1, D).to(torch.bfloat16)
            if st.m_o is not None:
                ob = _Static.apply(ob, st.m_o)
            if out is not None:
                out.copy_(ob)
                o = out
            else:
                o = ob
        return o, lse

    def _prefill_int4(self, st, qr, key, value, pool, md, r, q0, q1, L, s, o):
        """prefill_fp8 = "int4" / "int4ac": the INT4-direct kernels on the cached keys; W > 0: + the BF16 window FA
        call merged by LSE (as _prefill_fa)."""
        W = self.W
        bt = md.block_table[r]
        tok = OPTIMER.start("prefill_int4")
        if W == 0:
            self._int4_attn(st, qr, pool, bt, L, s, 0, out=o)
            OPTIMER.stop(tok)
            return
        nq = q1 - q0
        dev = qr.device
        win, p0 = self._window_buffer(st, md, r, key, value, q0, q1, s)       # original basis
        kb, vb = win[:, :self.num_kv_heads], win[:, self.num_kv_heads:]
        cq = torch.tensor([0, nq], dtype=torch.int32, device=dev)
        ck = torch.tensor([0, L - p0], dtype=torch.int32, device=dev)
        ob, lb = self._fa_call(qr, kb.contiguous(), vb.contiguous(), cq, ck, nq, L - p0, window=[W - 1, 0], lse=True)
        a0 = max(s, W)
        if L - W <= 0 or a0 >= L:
            o.copy_(ob)
            OPTIMER.stop(tok)
            return
        oa, la = self._int4_attn(st, qr[a0 - s:], pool, bt, L - W, a0, W)
        o.copy_(ob)
        o[a0 - s:] = _merge(oa, la, ob[a0 - s:], lb[:, a0 - s:])
        OPTIMER.stop(tok)

    def _prefill_emu(self, st, qr, key, value, pool, md, r, q0, q1, L, s, o):
        """Phase 2f emulation (prefill_emu.py) of the quantized-key attention; W > 0: + the BF16 window FA call merged
        by LSE, as in _prefill_fa."""
        from ..prefill_emu import emu_attention
        var = self.prefill_fp8[4:]
        W = self.W
        bt = md.block_table[r]
        if W == 0:
            oq, _ = emu_attention(self, st, qr, pool, bt, L, s, 0, var)
            o.copy_(oq)
            return
        nq = q1 - q0
        dev = qr.device
        win, p0 = self._window_buffer(st, md, r, key, value, q0, q1, s)       # original basis
        kb, vb = win[:, :self.num_kv_heads], win[:, self.num_kv_heads:]
        cq = torch.tensor([0, nq], dtype=torch.int32, device=dev)
        ck = torch.tensor([0, L - p0], dtype=torch.int32, device=dev)
        ob, lb = self._fa_call(qr, kb.contiguous(), vb.contiguous(), cq, ck, nq, L - p0, window=[W - 1, 0], lse=True)
        a0 = max(s, W)
        if L - W <= 0 or a0 >= L:
            o.copy_(ob)
            return
        oa, la = emu_attention(self, st, qr[a0 - s:], pool, bt, L - W, a0, W, var)
        o.copy_(ob)
        o[a0 - s:] = _merge(oa, la, ob[a0 - s:], lb[:, a0 - s:])

    def _forward_ref(self, q, key, value, md, out, output):
        """FLAGROT_MODE=ref: attention over the unquantized key / value of a fresh prefill (stock FLASH_ATTN)."""
        n = q.shape[0]
        qcpu = md.query_start_loc_cpu
        ql = (qcpu[1:] - qcpu[:-1]).to(torch.int64)
        if md.seq_lens_cpu is None or not torch.equal(ql, md.seq_lens_cpu.to(torch.int64)):
            raise RuntimeError("FLAGROT_MODE=ref needs batches without cached context")
        k = key[:n].view(n, self.num_kv_heads, self.D)
        v = value[:n].view(n, self.num_kv_heads, self.D)
        self._fa_call(q, k, v, md.query_start_loc, md.query_start_loc, md.max_query_len, md.max_query_len,
                      out=out.view(n, self.num_heads, self.D))
        return output


def _fp8_amax(x):
    """x [n, Hkv, 128] bf16 -> (FP8 E4M3 x / s, descale s [1, Hkv] fp32), s = amax over (tokens, dims) / 448 per KV
    head (vLLM's dynamic FP8 scale formula, QuantFP8: (amax / 448).clamp(min=1 / (448 * 512)))."""
    s = (x.abs().amax(dim=(0, 2)).float() / FP8_MAX).clamp(min=1.0 / (FP8_MAX * 512.0))
    x8 = (x.float() * s.reciprocal()[None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return x8, s.view(1, -1)


class _Fp8Stats:
    """FLAGROT_FP8_STATS=1 (parity harness, fresh W = 0 prefills only): per layer the squared errors of the
    dequantized INT4 K / V against the unquantized K / V (INT4 error), of their FP8 E4M3 rounding (scale 1.0 and
    per-head amax) against the dequantized values, of q's FP8 rounding, and of the attention output of the prefill
    variants (BF16 FA on the dequantized K / V, FP8 unit, FP8 amax, and FP8 on the unquantized K / V = vLLM's fp8
    cache) against BF16 FA on the unquantized K / V and against each other.  Sums of squares over all windows."""

    def __init__(self):
        self.on = os.environ.get("FLAGROT_FP8_STATS") == "1"
        self.acc = {}

    def _add(self, li, name, num, den):
        a = self.acc.setdefault(li, {})
        n0, d0 = a.get(name, (0.0, 0.0))
        a[name] = (n0 + float(num), d0 + float(den))

    @torch.no_grad()
    def collect(self, impl, st, pool, md, r, qr, key, value, q0, q1, cq, ck):
        li = getattr(impl, "_stats_layer", -1)
        L = q1 - q0
        Hkv = impl.num_kv_heads
        kr = key[q0:q1].reshape(L, Hkv, impl.D).contiguous()
        vr = value[q0:q1].reshape(L, Hkv, impl.D).contiguous()
        k0, v0 = kr.float(), vr.float()
        kb, vb = impl._cached_kv_impl(st, pool, md, r, L)                       # BF16 dequantized
        ku, vu = impl._cached_kv_impl(st, pool, md, r, L, dtype=FP8)            # FP8, scale 1
        (ka, kd), (va, vd) = _fp8_amax(kb), _fp8_amax(vb)
        kaf, vaf = ka.float() * kd.view(1, -1, 1), va.float() * vd.view(1, -1, 1)
        sq = lambda t: (t.float() ** 2).sum()  # noqa: E731
        for nm, x0, xb, xu, xa in (("k", k0, kb, ku, kaf), ("v", v0, vb, vu, vaf)):
            den = sq(x0)
            self._add(li, f"{nm}_int4", sq(xb.float() - x0), den)
            self._add(li, f"{nm}_fp8unit", sq(xu.float() - xb.float()), den)
            self._add(li, f"{nm}_fp8amax", sq(xa - xb.float()), den)
            self._add(li, f"{nm}_int4+fp8unit", sq(xu.float() - x0), den)
            self._add(li, f"{nm}_int4+fp8amax", sq(xa - x0), den)
            self._add(li, f"{nm}_fp8unit_subnormal_frac", ((xu.float().abs() < 2.0 ** -6) & (xu.float() != 0)).sum(),
                      xu.numel())
        q8 = impl._q8(qr)
        self._add(li, "q_fp8", sq(q8.float() - qr.float()), sq(qr))
        one = impl._unit(qr.device)
        o_ref = impl._fa_call(qr, kr, vr, cq, ck, L, L).float()
        o_bf = impl._fa_call(qr, kb, vb, cq, ck, L, L).float()
        o_u = impl._fa_call(q8, ku, vu, cq, ck, L, L, descale=(one, one, one)).float()
        o_a = impl._fa_call(q8, ka, va, cq, ck, L, L, descale=(one, kd, vd)).float()
        o_f = impl._fa_call(q8, impl._q8(kr), impl._q8(vr), cq, ck, L, L, descale=(one, one, one)).float()
        # BF16 FA on the FP8-rounded inputs (exact in BF16): splits the FP8 path's error into input rounding
        # (o_bf16round vs o_bf16deq) and FA3's FP8 compute (o_fp8unit vs o_bf16round: P in FP8 for the PV GEMM)
        bf = torch.bfloat16
        o_r = impl._fa_call(q8.to(bf), ku.to(bf), vu.to(bf), cq, ck, L, L).float()
        self._add(li, "o_bf16round_vs_bf16deq", sq(o_r - o_bf), sq(o_bf))
        self._add(li, "o_fp8unit_vs_bf16round", sq(o_u - o_r), sq(o_r))
        # which input's rounding matters: BF16 FA with only q / k / v rounded to FP8
        for nm, qq, kk, vv in (("q", q8.to(bf), kb, vb), ("k", qr, ku.to(bf), vb), ("v", qr, kb, vu.to(bf))):
            o_x = impl._fa_call(qq, kk, vv, cq, ck, L, L).float()
            self._add(li, f"o_bf16round_{nm}only_vs_bf16deq", sq(o_x - o_bf), sq(o_bf))
        # whole-head Hadamard on q and K before the FP8 cast (q -> H q / 8, k -> H k / 16: q'.k' = q.k exactly)
        allh = torch.ones(Hkv, dtype=torch.uint8, device=qr.device)
        allq = torch.ones(impl.num_heads, dtype=torch.uint8, device=qr.device)
        qh = K.head_fwht(qr.contiguous(), allq, 16 / impl.D)
        kh = K.head_fwht(kb.contiguous(), allh, 1 / 16)
        self._add(li, "qh_fp8", sq(impl._q8(qh).float() - qh.float()), sq(qh))
        self._add(li, "kh_fp8", sq(kh.clamp(-FP8_MAX, FP8_MAX).to(FP8).float() - kh.float()), sq(kh))
        o_h = impl._fa_call(impl._q8(qh), kh.clamp(-FP8_MAX, FP8_MAX).to(FP8), vu, cq, ck, L, L,
                            descale=(one, one, one)).float()
        self._add(li, "o_fp8hadqk_vs_bf16deq", sq(o_h - o_bf), sq(o_bf))
        self._add(li, "o_fp8hadqk_vs_ref", sq(o_h - o_ref), sq(o_ref))
        dref, dbf = sq(o_ref), sq(o_bf)
        for nm, o in (("o_bf16deq", o_bf), ("o_fp8unit", o_u), ("o_fp8amax", o_a), ("o_vllmfp8", o_f)):
            self._add(li, f"{nm}_vs_ref", sq(o - o_ref), dref)
        self._add(li, "o_fp8unit_vs_bf16deq", sq(o_u - o_bf), dbf)
        self._add(li, "o_fp8amax_vs_bf16deq", sq(o_a - o_bf), dbf)

    def read(self):
        """relative MSE (sum of squared errors / sum of squares of the reference): layer mean, pooled, per layer."""
        per = {li: {k: n / max(d, 1e-30) for k, (n, d) in a.items()} for li, a in self.acc.items()}
        tot = {}
        for a in self.acc.values():
            for k, (n, d) in a.items():
                n0, d0 = tot.get(k, (0.0, 0.0))
                tot[k] = (n0 + n, d0 + d)
        mean = {k: sum(per[li][k] for li in per) / len(per) for k in sorted(tot)}
        return dict(layer_mean=mean, pooled={k: n / max(d, 1e-30) for k, (n, d) in tot.items()},
                    per_layer={str(li): per[li] for li in sorted(per)})


FP8_STATS = _Fp8Stats()


def _merge(oa, la, ob, lb):
    """LSE merge of two partial attentions: o* [n, H, D], l* [H, n] (FA's +inf = empty row)."""
    la = torch.where(torch.isposinf(la), torch.full_like(la, -math.inf), la).transpose(0, 1)[..., None].float()
    lb = torch.where(torch.isposinf(lb), torch.full_like(lb, -math.inf), lb).transpose(0, 1)[..., None].float()
    m = torch.maximum(la, lb)
    wa, wb = torch.exp(la - m), torch.exp(lb - m)
    return ((oa.float() * wa + ob.float() * wb) / (wa + wb)).to(ob.dtype)
