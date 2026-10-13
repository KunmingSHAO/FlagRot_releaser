"""vllm-flagrot: out-of-tree vLLM plugin for FlagRot.

Phase 2b: the FlagRot INT4 KV caches "flagrot_int4" (per-token codebook) and "flagrot_int4_static" (static
FlagRot transforms), optional BF16 staging window, as attention backend CUSTOM
(vllm_flagrot.attention.backend.FlagRotAttentionBackend) on the CUDA kernels of kernels/ (vllm_flagrot.kernels,
ctypes), CUDA-graph capturable decode.  See the top-level README.md.

Design stubs for later phases (NOT imported by register(), never run against vLLM): config.py (linear-layer
quantization config), ops.py (custom-op schemas of the CUDA kernels), fusion.py, converter.py, format.py.

register() is the `vllm.general_plugins` entry point; vLLM calls it in every process (front end, engine core,
workers), possibly several times, so it is idempotent and does not touch CUDA.
"""
from __future__ import annotations

_REGISTERED = False


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    # dotted class path (resolved lazily with resolve_obj_by_qualname); selected with --attention-backend CUSTOM
    register_backend(AttentionBackendEnum.CUSTOM, "vllm_flagrot.attention.backend.FlagRotAttentionBackend")
    # Phase 2k: prefix caching with BF16 sinks / window - scheduler-side hooks (inactive unless configured)
    from .prefix_cache import install
    install()
    _REGISTERED = True
