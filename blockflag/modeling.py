"""Model / data loading.  WikiText-2 protocol: raw split joined with '\\n\\n', tokenized once, cut into 2048-token
windows (the GPTQ convention); TRAIN windows are used for calibration, TEST windows for evaluation."""
from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn

from . import paths

SEQLEN = 2048
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_model(key: str, attn_implementation: str = "sdpa", device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = str(paths.MODELS[key])
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=attn_implementation)
    return model.to(device).eval(), tok


# models whose predictions collapse without a BOS token at the start of the context (Gemma-2: WikiText-2 window
# perplexity 196-415 without, 6-12 with): every SEQLEN window of their stream starts with BOS + SEQLEN-1 text tokens
BOS_EVERY_WINDOW = {"gemma2_9b", "gemma3_27b"}


def token_stream(key: str, tok, split: str) -> torch.Tensor:
    paths.CACHE.mkdir(parents=True, exist_ok=True)
    cp = paths.CACHE / f"{key}_{split}_ids{'_bos' if key in BOS_EVERY_WINDOW else ''}.pt"
    if cp.exists():
        return torch.load(cp)
    if split == "fineweb":                          # calibration stream: documents in file order, ~6M tokens
        docs, n = [], 0
        for t in pd.read_parquet(paths.DATA[split], columns=["text"])["text"]:
            docs.append(t)
            n += len(t) // 3                            # rough token count; the stream is cut by windows()
            if n > 9_000_000:
                break
        text = "\n\n".join(docs)
    else:
        text = "\n\n".join(pd.read_parquet(paths.DATA[split])["text"].tolist())
    ids = tok(text, return_tensors="pt").input_ids[0]
    if key in BOS_EVERY_WINDOW:
        body = ids[1:] if ids[0] == tok.bos_token_id else ids
        n = body.numel() // (SEQLEN - 1)
        ids = torch.cat([torch.full((n, 1), tok.bos_token_id, dtype=ids.dtype), body[: n * (SEQLEN - 1)].view(n, -1)], 1).reshape(-1)
    torch.save(ids, cp)
    return ids


def windows(ids: torch.Tensor, n: int | None = None, seqlen: int = SEQLEN) -> torch.Tensor:
    nw = ids.numel() // seqlen
    nw = nw if n is None else min(nw, n)
    return ids[: nw * seqlen].view(nw, seqlen)


def text_config(model):
    """the language model's config (multimodal wrappers such as Gemma-3 / Qwen3.5 keep it in config.text_config)."""
    return model.config.get_text_config()


def decoder_layers(model) -> nn.ModuleList:
    """decoder layers: model.model.layers, or model.model.language_model.layers in multimodal wrappers (Gemma-3)."""
    m = model.model
    return m.layers if hasattr(m, "layers") else m.language_model.layers


def attn_layers(model) -> dict[int, nn.Module]:
    """{layer_idx: attention module} of the layers that own a KV cache, i.e. have self_attn (hybrid models such as
    Qwen3.5 interleave linear-attention layers without one); keyed by the module's layer_idx, as the attention sees it."""
    out = {}
    for li, layer in enumerate(decoder_layers(model)):
        at = getattr(layer, "self_attn", None)
        if at is not None:
            assert getattr(at, "layer_idx", li) == li, (li, at.layer_idx)
            out[li] = at
    return out


def head_dim(cfg) -> int:
    return getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads


def decoder_linears(model) -> list[tuple[str, nn.Module, str]]:
    """(name, parent, attribute) of every q/k/v/o/gate/up/down projection, name = 'layers.<i>.<mlp|self_attn>.<proj>'
    (linear-attention layers of hybrid models contribute their MLP only)."""
    out = []
    for li, layer in enumerate(decoder_layers(model)):
        for mod_name in ("self_attn", "mlp"):
            parent = getattr(layer, mod_name, None)
            if parent is None:
                continue
            for pn in PROJ:
                if hasattr(parent, pn):
                    out.append((f"layers.{li}.{mod_name}.{pn}", parent, pn))
    return out
