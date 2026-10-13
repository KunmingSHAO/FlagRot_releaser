"""Plain asymmetric INT4 (RTN, no rotation) for the static v2 kernel: a copy of the static int4a calibration file with
every flag cleared (kmask / vmask / kfull / vfull all 0).  With it the plugin runs static_a's kernels and layout
(writer A2_INT4A, attention variant 264, 144 B per token-head) without any transform: the writer's block flags are 0,
the attention kernel gets no q rotation / o back-rotation (kmask = vmask = None), no whole-head H128 launches.

  python3 integrations/vllm_flagrot/scripts/make_rtn_calib.py [--model llama8b]
  -> calib/flagrot_static_rtn_a_<model>.pt        (bench kv name: rtn_a)

--from_config (a model without a research-env calibration): the all-zero file is built from the
model's config.json (KV-cache layers, KV heads, head dim; hybrid models: the full-attention layers) instead.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from common import MODELS, calib_path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="llama8b")
ap.add_argument("--from_config", action="store_true")
a = ap.parse_args()
dst = calib_path(a.model, "rtn_a")
if a.from_config:
    if Path(MODELS[a.model]).is_dir():
        cfg = json.loads((Path(MODELS[a.model]) / "config.json").read_text())
    else:                                   # HF hub id
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(MODELS[a.model]).to_dict()
    tc = cfg.get("text_config", cfg)
    L, Hkv, Hq = tc["num_hidden_layers"], tc["num_key_value_heads"], tc["num_attention_heads"]
    D = tc.get("head_dim") or tc["hidden_size"] // Hq
    types = tc.get("layer_types")
    layers = [i for i, t in enumerate(types) if t == "full_attention"] if types else list(range(L))
    n = len(layers)
    out = dict(kmask=torch.zeros(n, Hkv, D // 32, dtype=torch.bool), vmask=torch.zeros(n, Hkv, D // 32, dtype=torch.bool),
               kfull=torch.zeros(n, Hkv, dtype=torch.bool), vfull=torch.zeros(n, Hkv, dtype=torch.bool))
    out["meta"] = dict(model=a.model, model_path=MODELS[a.model], num_layers=L, num_kv_heads=Hkv, num_heads=Hq,
                       head_dim=D, layers=None if layers == list(range(L)) else layers, rb=32, fmt="int4a:32",
                       kv_flags="headsel", rtn=True, derived_from="config.json",
                       created=time.strftime("%Y-%m-%d %H:%M:%S"))
else:
    src = calib_path(a.model, "static_a")
    cal = torch.load(src, map_location="cpu", weights_only=True)
    out = {k: torch.zeros_like(cal[k]) for k in ("kmask", "vmask", "kfull", "vfull")}
    out["meta"] = dict(cal["meta"], rtn=True, derived_from=src, created=time.strftime("%Y-%m-%d %H:%M:%S"))
assert out["meta"]["fmt"] == "int4a:32" and out["meta"]["kv_flags"] == "headsel"
Path(dst).parent.mkdir(parents=True, exist_ok=True)
torch.save(out, dst)
print(dst, {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)})
