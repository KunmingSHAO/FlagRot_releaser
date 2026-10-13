"""Shared helpers of the vllm_flagrot evaluation scripts (vLLM env)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integrations/vllm_flagrot"))        # this checkout's plugin before the installed one
os.environ.setdefault("PYTHONPATH", "")
if str(ROOT / "integrations/vllm_flagrot") not in os.environ["PYTHONPATH"].split(os.pathsep):   # engine processes
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "integrations/vllm_flagrot")] + [p for p in os.environ["PYTHONPATH"].split(os.pathsep) if p])
# this env has no nvcc / ninja on PATH for FlashInfer's JIT sampler; the PyTorch sampler is equivalent for us
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from blockflag.paths import MODELS  # noqa: E402  (model key -> local directory or HF id; env BLOCKFLAG_MODEL_DIR)
# KV-cache configurations: stock vLLM dtypes and the FlagRot plugin formats
STOCK = {"bf16": "auto", "fp8": "fp8", "int4pth": "int4_per_token_head"}
FLAGROT = {"static": "flagrot_int4_static", "codebook": "flagrot_int4", "flagrot": "flagrot_int4", "konly": "flagrot_int4",
           "static_a": "flagrot_int4_static", "konlya": "flagrot_int4", "rtn_a": "flagrot_int4_static",
           "konlya_c": "flagrot_int4",
           # Phase 3a: INT2 pages (int2a:32 stored as int2ah:32)
           "rtn_a2": "flagrot_int4_static", "static_a2": "flagrot_int4_static", "konlya2": "flagrot_int4",
           "konlya_c2": "flagrot_int4",
           # kivi2: FlagRot-KIVI INT2 pages (simulator kivi_sel; kivi2c = kivi_c + calib_rot 32768)
           "kivi2": "flagrot_kivi", "kivi2c": "flagrot_kivi",
           "kivi2id": "flagrot_kivi",   # kivi2id: plain KIVI (no selection), the matched control of kivi2
           "kivi2s": "flagrot_kivi",    # kivi2s: static selection (frozen per layer / head / slot), control of kivi2
           # kivi2p8 / kivi2p8id: the 2.5-bit packing kivi2f8:32 (FP8 scale + minimum), selection / identity
           "kivi2p8": "flagrot_kivi", "kivi2p8id": "flagrot_kivi",
           # kivi2p8f / kivi2p8fid / kivi2f: the same caches with the fast decode-attention kernel (kivi_fast)
           "kivi2p8f": "flagrot_kivi", "kivi2p8fid": "flagrot_kivi", "kivi2f": "flagrot_kivi",
           # kivi2p8ff / kivi2p8ffid: + fast prefix dequantization / fused merge / encoder (kivi_ff)
           "kivi2p8ff": "flagrot_kivi", "kivi2p8ffid": "flagrot_kivi", "kivi2ff": "flagrot_kivi",
           "kivi2ffid": "flagrot_kivi", "kivi2h": "flagrot_kivi", "kivi2hid": "flagrot_kivi",
           "kivi2p8ffs": "flagrot_kivi"}   # kivi2p8ffs: static selection frozen under the p8 format
KV_DYN = {"konly": "cq+f+konly", "konlya": "cq+f+konly", "konlya_c": "cq+f+konly",   # key-only 2x2 codebook
          "konlya2": "cq+f+konly", "konlya_c2": "cq+f+konly"}
QUANT = {"static_a": "int4a", "konlya": "int4a", "rtn_a": "int4a", "konlya_c": "int4a",   # asymmetric INT4 (int4a:32)
         "static_a2": "int2a", "konlya2": "int2a", "rtn_a2": "int2a", "konlya_c2": "int2a"}  # Phase 3a: INT2
# konlya_c = FlagRot-C (research wush_k2): the key-only 2x2 codebook on top of the calibrated block-diagonal WUSH base
# transform; calibration file flagrot_calib_c_<model>.pt (scripts/export_calib.py --wush) holds the base matrices


def split_center(mode: str):
    """'<mode>:c' (int2c: static key centering, plugin option "center") -> (mode without ':c', "post");
    '<mode>:ca' (the research kv_center = auto: per-group selective centering or none, chosen per (model, method))
    -> (mode, "auto"); else (mode, False)."""
    parts = mode.split(":")
    for flag, kind in (("c", "post"), ("ca", "auto")):
        if flag in parts[1:]:
            return ":".join(p for i, p in enumerate(parts) if i == 0 or p != flag), kind
    return mode, False


def parse_mode(mode: str):
    """'<kv>[:<suffix>]' -> (kv, sink, window): suffix '128' (window, Phase 2b-2f syntax) | 'w128' | 's4' | 's4w128'.
    (a ':c' centering flag is ignored here: split_center)"""
    import re
    mode, _ = split_center(mode)
    kv, _, suf = mode.partition(":")
    if not suf:
        return kv, 0, 0
    if suf.isdigit():
        return kv, 0, int(suf)
    m = re.fullmatch(r"(?:s(\d+))?(?:w(\d+))?", suf)
    if not m or not suf:
        raise ValueError(f"bad mode suffix {suf!r} (expected W, wW, sS or sSwW)")
    return kv, int(m.group(1) or 0), int(m.group(2) or 0)
# rtn_a = plain asymmetric INT4 (RTN, no rotation) in static_a's kernels and layout: a static calibration file with
# all-zero masks (scripts/make_rtn_calib.py); writer flags 0, no q / o transform, no whole-head launches


CALIB_DIR = Path(os.environ.get("FLAGROT_CALIB_DIR", os.environ.get("BLOCKFLAG_CALIB_DIR", str(ROOT / "calib"))))
# friendly aliases of the paper's configurations (README, mode table) -> canonical mode strings
ALIASES = {"flagrot-int4": "konlya:s4w128", "flagrot-int2-auto": "konlya2:s4w128:ca",
           "flagrot-kivi-3.0": "kivi2:s4w128", "kivi-3.0": "kivi2id:s4w128",
           "flagrot-kivi-2.5": "kivi2p8ff:s4w128", "kivi-2.5": "kivi2p8ffid:s4w128",
           "flagrot-kivi-2.5-fixed": "kivi2p8ffs:s4w128"}


def resolve_alias(mode: str) -> str:
    return ALIASES.get(mode, mode)


def calib_path(model: str, kv: str = "codebook", center: bool = False) -> str:
    name = {"static": "flagrot_static", "static_a": "flagrot_static_a", "rtn_a": "flagrot_static_rtn_a",
            "konlya_c": "flagrot_calib_c", "static_a2": "flagrot_static_a2", "rtn_a2": "flagrot_static_rtn_a",
            "konlya_c2": "flagrot_calib_c", "kivi2": "flagrot_kivi", "kivi2c": "flagrot_kivi_c", "kivi2id": "flagrot_kivi",
            "kivi2s": "flagrot_kivi_s", "kivi2p8": "flagrot_kivi", "kivi2p8id": "flagrot_kivi",
            "kivi2p8f": "flagrot_kivi", "kivi2p8fid": "flagrot_kivi", "kivi2f": "flagrot_kivi",
            "kivi2p8ff": "flagrot_kivi", "kivi2p8ffid": "flagrot_kivi", "kivi2ff": "flagrot_kivi",
            "kivi2ffid": "flagrot_kivi", "kivi2h": "flagrot_kivi", "kivi2hid": "flagrot_kivi",
            "kivi2p8ffs": "flagrot_kivi_sp8"}.get(kv, "flagrot_calib")
    if center and center != "auto":   # int2c: key-dependent calibrations on the centered keys (--kv_center post)
        name = {"flagrot_static_a2": "flagrot_static_a2c", "flagrot_static_a": "flagrot_static_ac",
                "flagrot_calib_c": "flagrot_calib_cc"}.get(name, name)
    return str(CALIB_DIR / f"{name}_{model}.pt")


AUTO_METHOD = {"konlya2": "flagrot", "rtn_a2": "rtn"}   # the INT2 auto choice (the simulator sweep ran auto for INT2)


def center_path(model: str, kind="post", kv: str = "", sink: int = 0, window: int = 0) -> str:
    """int2c: the static key-centering vector of the model (export_calib.py --center); kind "auto": the vector of the
    research kv_center = auto for (method of kv, protection) (export_calib.py --center auto --method --prot)."""
    name = "flagrot_center"
    if kind == "auto":
        if kv not in AUTO_METHOD or (sink, window) not in ((4, 128), (0, 0)):
            raise ValueError(f"kv_center auto: exported for rtn_a2 / konlya2 at S4W128 / W0 only (got {kv}, s{sink} w{window})")
        name = f"flagrot_center_auto_{AUTO_METHOD[kv]}_{'s4w128' if window else 'w0'}"
    return str(CALIB_DIR / f"{name}_{model}.pt")


def make_llm(model: str, kv: str, max_model_len: int = 4096, max_num_batched_tokens: int = 4096,
             gpu_memory_utilization: float | None = None, enable_prefix_caching: bool = False, window: int = 0,
             prefill: str = "fa", enforce_eager: bool = True, prefill_fp8: str | None = None, sink: int = 0, tp: int = 1,
             center: bool = False, **kw):
    """kv = 'bf16' | 'fp8' | 'int4pth' (stock vLLM backends) | 'static' | 'codebook' | 'konly' | 'static_a' | 'konlya'
    (FlagRot plugin, CUSTOM backend; 'static_a' / 'konlya': asymmetric INT4); window / prefill: FlagRot BF16 staging
    window and prefill path; prefill_fp8: FlagRot "fa" prefill attention in FP8 ("unit" | "amax") or BF16 ("off"),
    None = the plugin default."""
    from vllm import LLM
    if gpu_memory_utilization is None:
        gpu_memory_utilization = float(os.environ.get("FLAGROT_GPU_MEM", "0.45"))
    args = dict(model=MODELS[model], dtype="bfloat16", max_model_len=max_model_len, enforce_eager=enforce_eager,
                gpu_memory_utilization=gpu_memory_utilization, max_num_batched_tokens=max_num_batched_tokens,
                enable_prefix_caching=enable_prefix_caching, seed=0, tensor_parallel_size=tp, **kw)
    if model == "qwen38_27b":  # text only (the vision tower is not exercised); MTP head unused
        args.setdefault("limit_mm_per_prompt", {"image": 0, "video": 0})
    if os.environ.get("FLAGROT_KV_BYTES"):
        # fixed KV-cache size: vLLM skips its free-memory profiling, so several correctness runs can share a GPU
        # (the profiling assertion fails when another process frees memory meanwhile); not for speed runs
        args.setdefault("kv_cache_memory_bytes", int(float(os.environ["FLAGROT_KV_BYTES"])))
    if FLAGROT.get(kv) == "flagrot_kivi":
        # kivi2: a prefix-cache block must hold whole KIVI key groups (a full block is then always encoded); hybrid
        # models raise the attention block to a multiple of max(16, block_size) = 32 (checked by the backend)
        args.setdefault("block_size", 32)
    if kv in FLAGROT:
        args.update(attention_backend="CUSTOM", kv_cache_dtype="auto",
                    additional_config={"flagrot": {"kv_cache_dtype": FLAGROT[kv], "calib": calib_path(model, kv, center),
                                                   "window": window, "prefill": prefill, "sink": sink}})
        if center:
            kind = "auto" if center == "auto" else "post"
            args["additional_config"]["flagrot"]["center"] = center_path(model, kind, kv, sink, window)
        if kv in KV_DYN:
            args["additional_config"]["flagrot"]["kv_dyn"] = KV_DYN[kv]
        if prefill_fp8 is not None:
            args["additional_config"]["flagrot"]["prefill_fp8"] = prefill_fp8
        if kv in QUANT:
            args["additional_config"]["flagrot"]["quant"] = QUANT[kv]
        if kv in ("kivi2id", "kivi2p8id", "kivi2p8fid", "kivi2p8ffid", "kivi2ffid", "kivi2hid"):
            args["additional_config"]["flagrot"]["kivi_id"] = 1
        if kv in ("kivi2p8", "kivi2p8id", "kivi2p8f", "kivi2p8fid", "kivi2p8ff", "kivi2p8ffid", "kivi2p8ffs"):
            args["additional_config"]["flagrot"]["kivi_f8"] = 1
        if kv in ("kivi2p8f", "kivi2p8fid", "kivi2f"):
            args["additional_config"]["flagrot"]["kivi_fast"] = 1
        if kv in ("kivi2p8ff", "kivi2p8ffid", "kivi2ff", "kivi2ffid", "kivi2p8ffs"):
            args["additional_config"]["flagrot"]["kivi_ff"] = 1
        if kv in ("kivi2s", "kivi2p8ffs"):
            args["additional_config"]["flagrot"]["kivi_static"] = 1
    elif kv in STOCK:
        assert not center, "key centering is a FlagRot plugin option"
        args.update(kv_cache_dtype=STOCK[kv])
    else:
        raise ValueError(kv)
    return LLM(**args)


def in_process():
    """the debug toggle of the backend (MODE) and the logits hooks need the worker in this process."""
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
