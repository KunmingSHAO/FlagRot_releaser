"""One experiment = one config -> one result JSON (+ flag masks).

Config keys (all optional except model): model, w, a, rb, flags, kv_k, kv_v, kv_rb, kv_flags, calib_windows, windows.
  flags / kv_flags: none | all | greedy | greedy:<m> | proj:<a+b> (linears only) | layer (linears) | head (KV) | headsel (KV: per head none, rb-block bits or whole head)
    layer / head = one calibrated rotate-or-not decision per layer group / per (layer, KV head) (granularity ablation)
    rule = closed-form per-vector flags without trial quantization (crest rule for INT weights, dFP > 1 otherwise)
    greedy:<m> flags a vector only if it lowers its unit's calibration error by more than m (relative)
  rb / kv_rb: rotation unit; -1 = "full" (largest power of two dividing K, <= 4096; whole head_dim for KV)
  wq: rtn | gptq (weights; GPTQ Hessians from gptq_windows TRAIN windows, flags still chosen with the RTN proxy)
  flags = file + mask_file = <path to a torch file {linear name: bool mask [K/rb]}>: explicit per-vector flags (probes)
  protect: tau > 0 -> vectors holding an input channel with max |x| > tau * median (massive activations, super
    weights) are never rotated, whatever the flag mode (greedy / all / proj)
  proxy: rtn | gptq -> greedy flags from the RTN per-token relative error, or from the GPTQ error tr(dW H dW^T)
  joint_calib: true -> KV flags are calibrated with the (already quantized) linears in place
  kv_k_axis: head | token (per-channel key groups over tokens, KIVI); kv_resid: R most recent tokens kept in full
    precision (KIVI residual); kv_sink: first n tokens kept in full precision (IntactKV-style)
  kv_resid_mode: sliding (default: query i reads keys and values j > i - R in full precision) | kivi (KIVI's own
    policy: keys j >= R floor(i / R), values j >= i - R)
  eval_split: test | valid (WikiText-2 split of the KL evaluation; hyper-parameters are selected on valid)
  kv_xform: "" | wush | smooth[:alpha]  static per-(layer, KV head) transform calibrated on calib_windows TRAIN windows:
    WUSH-KV (keys post-RoPE with T_K, queries T_K^-T; values T_V folded into W_V / W_O; block width = the spec's scale
    group, at most head_dim) or QServe SmoothAttention (keys / lambda, queries * lambda, lambda_i = max|K_i|^alpha
    over pre-RoPE keys, tied on RoPE pairs; default alpha 0.5).  Combines with kv_flags none | all.
    smoothpost[:alpha] = SmoothAttention with max|K_i| over POST-RoPE keys, as QServe's calibration code collects
    them (deepcompressor / lmquant: k_rotary_emb outputs, pairs (i, i + D/2) tied by their max).
    kv_xform = file:<path>: fixed transform loaded from a torch file {layer: {"k", "q", "v", "vinv": [Hkv, D, D]}}
    (row vectors, same convention as QA.KV['xf']; no calibration), e.g. SpinQuant's learned R2 on values and the
    Hadamard R3 on keys / queries exported from the official SpinQuant code.
  kv_skip: n -> the first and last n layers keep a BF16 cache (vLLM TurboQuant's boundary-layer protection, n = 2)
  KV baselines are kv_k / kv_v specs (formats.py): int4ph (vLLM int4_per_token_head), tqk4n / tqk3n / tqk8 + tqv4 /
    tqv3 (vLLM turboquant_*), nvfp4_46 (Four Over Six), int<b>p<k> (WUSH-KV's percentile-clipped quantizer)
  kv_static_only: "" | k | v  static flag modes (headsel / greedy / head / all) on one tensor only, the other keeps plain
    quantization (criterion ablation: static keys-only vs per-token keys-only)
  kv_kvq: "" | nuq | nuq+ds  KVQuant-style cache (not a reproduction): keys quantized PRE-RoPE per channel (groups of g
    consecutive tokens, as KIVI), values per token (groups of g along head_dim), both with a per-(layer, K/V) 16-level
    non-uniform codebook (unweighted 1-D k-means on min/max-normalized calibration values, initialized at the uniform
    grid; KVQuant weights by Fisher information); +ds = dense-and-sparse: keys outside per-(layer, head, channel)
    0.5 / 99.5 % calibration quantiles and the 1 % largest |v| of every token (all KV heads) stay exact
  kv_resid_mode kgroup: only the current incomplete key group (j >= R floor(i / R)) stays in full precision, values
    have no window -- the minimum a per-channel key cache needs to stay causal (R = group size)
  kv_center: "" | post | sel  static key centering (exact for attention): mu = per-(layer, KV head, channel) mean of the
    post-RoPE calibration keys (calib_windows TRAIN windows); every key of a quantized layer, the BF16 sink / window keys
    included, is replaced by k - mu before the cache stores it (and in the calibration of flags / transforms), so each
    query's logits shift by the same q . mu and the softmax is unchanged; 0 extra bits (one static vector per head);
    sel: per (layer, KV head, key scale group) the mean or 0, by the calibrated attention-weighted variance of the logit
    error of a per-token RTN cache with the run's sink / window protection (qattention._kc_measure);
    auto: sel or nothing, one decision per run by the KL of the complete cache on the calibration windows
  kc_windows: 0 | <n>     opt-in: the kv_center statistics (mean pass and sel measurement)
    use only the first n calibration windows (e.g. 4 of WUSH-KV's 128 FineWeb sequences); 0 = all calib_windows.
  calib_rot: "" | <L>   RoPE-phase-covered calibration (FlagRot-C long-context calibration):
    every calibration window is run at position offset 0 (weight 1/2) and at the 7 offsets j L' / 8, j = 1..7
    (weight 1/14 each; L' = min(L, max_position_embeddings), offsets capped at L' - calib_len), so the post-RoPE
    statistics (WUSH M_K / H_K, C_q, centering means, KIVI-C sampled queries) cover the RoPE phases of positions up to
    L' instead of 0..calib_len-1.  Implemented by repeating the offset-0 pass 7 times (same statistics as the offline
    mixture 0.5 S + 0.5 mean_o R(o) S R(o)^T up to BF16 arithmetic); only the calibration changes.
  kv_rotatekv: "" | gh<G>[+ro][+ma<n>][+sel]  RotateKV: keys quantized PRE-RoPE after a grouped-head
    Walsh-Hadamard over G KV heads (G reduced to a power of two dividing Hkv) and, with ro, one calibrated permutation of
    all Hkv * D channels per layer (argsort of the rotated calibration keys' channel sums, calib_windows TRAIN windows);
    values after the per-head H_D; ma<n> keeps the tokens of the top-n |x| entries of each layer's input hidden states
    in full precision (attention-sink-aware quantization); sel = + FlagRot's per-token key selection (cq+f+konly) on the
    stored pre-RoPE basis, weighted by the calibrated C_q mapped through RoPE (qattention._rk_sel_calib).  Quantizer = kv_k / kv_v (RotateKV's: int<b>ac<clip>:<g>,
    e.g. int2ac80:128).  Combines with kv_sink / kv_resid; not with kv_flags / kv_xform / kv_kvq / kv_center.
  eval_data: wikitext | pg19 | code   eval_len: tokens per evaluation window (2048; > 2048 uses chunked logits).
    owm | wiki_zh | wiki_de = documents of the first parquet shard in file order (OpenWebMath; Wikipedia
    zh / de), joined with blank lines up to TEXT_EVAL characters and cut into eval_len windows (as WikiText-2)
    pg19 = the first eval_len tokens of each PG-19 test book with >= 32768 tokens (file order, one window per book);
    code = codeparrot-clean-valid Python files joined with blank lines, cut into eval_len windows (as WikiText-2)
  eval: kl | lm      lm = zero-shot lm-eval tasks (lm_tasks, first lm_limit docs each), per-sample correctness stored
Metric (eval = kl): KL(BF16 || quantized) of the next-token distributions, averaged over tokens of the first `windows`
TEST windows; every window is run twice in the same process (reference then quantized), and per-window KL / NLL /
top-1 agreement are stored so that configs can be compared with paired statistics.
usage: python -m blockflag.run --config '{"model": "qwen8b", "w": "nvfp4", "a": "fp8:128", "rb": 16, "flags": "greedy"}'
       [--out results/runs/<id>.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import __version__, paths
from . import qattention as QA
from .modeling import attn_layers, decoder_linears, head_dim, load_model, text_config, token_stream, windows
from . import formats as FM
from .qlinear import HESS, STATE, SelLinear, greedy_masks, layer_masks, protected_vectors, rule_mask, share_key

torch.backends.cuda.matmul.allow_tf32 = False
DEFAULTS = dict(w="bf16", a="bf16", rb=64, flags="none", kv_k="bf16", kv_v="bf16", kv_rb=None, kv_flags="none",
                calib_windows=4, windows=40, wq="rtn", gptq_windows=16, mask_file="", protect=0, proxy="rtn",
                joint_calib=False, kv_k_axis="head", kv_resid=0, kv_sink=0, eval="kl", eval_split="test", lm_tasks="arc_easy,arc_challenge,piqa,winogrande,hellaswag,lambada_openai",
                lm_limit=0, lm_max_len=0, kv_cands="", kv_gran="head", kv_proxy="score", kv_dyn="",
                kv_xform="", kv_skip=0, kv_flush=0, calib_data="wikitext", calib_len=2048,
                kv_resid_mode="sliding", kv_static_only="", kv_kvq="", eval_data="wikitext", eval_len=2048,
                kv_center="", calib_rot="", kv_rotatekv="", kc_windows=0, kv_dyn_off="",
                kv_att_nq=0, kv_att_w="")


def calibrated(f: str) -> bool:
    """flag modes that need calibration statistics: greedy[:m] (per vector), layer (linear), head (KV)."""
    return f.startswith("greedy") or f in ("layer", "head", "headsel", "sel", "dyn")


def canonical(cfg: dict) -> dict:
    """fill defaults and blank out fields that cannot influence the result, so equal experiments share one id."""
    c = {**DEFAULTS, **cfg}
    if c["kv_rb"] is None:
        c["kv_rb"] = c["rb"]
    if c["w"] == "bf16" and c["a"] == "bf16":
        c["flags"] = "none"
    if c["flags"] == "none":
        c["rb"] = 0
    if c["kv_k"] == "bf16" and c["kv_v"] == "bf16":
        c["kv_flags"] = "none"
    if c["kv_flags"] == "none":
        c["kv_rb"] = 0
    if c["kv_k"] == "bf16" and c["kv_v"] == "bf16":
        c.update(kv_xform="", kv_skip=0, kv_rotatekv="")
    if c["kv_xform"] in ("smooth", "smoothpost"):
        c["kv_xform"] += ":0.5"
    if c["flags"] in ("none", "file"):
        c["protect"] = 0
    if not (calibrated(c["flags"]) or calibrated(c["kv_flags"]) or c["protect"] or c["flags"] == "rule"
            or (c["kv_xform"] and not c["kv_xform"].startswith("file:")) or c["kv_kvq"] or c["kv_center"]
            or bool({"ro", "sel"} & set(c["kv_rotatekv"].split("+")))):
        c["calib_windows"] = 0
    if c["w"] == "bf16" or FM.group_of(c["w"], 1 << 20) > 128:        # GPTQ only for group-scaled weight formats
        c["wq"] = "rtn"
    if not calibrated(c["flags"]) or c["w"] == "bf16" or FM.group_of(c["w"], 1 << 20) > 128:
        c["proxy"] = "rtn"
    if c["wq"] == "rtn" and c["proxy"] == "rtn":
        c["gptq_windows"] = 0
    if c["kv_flags"] != "sel":                     # design-space keys only matter for kv_flags='sel'
        # opt-in (mechanism study): kv_flags='headsel' with kv_proxy='mse' ranks the static choices by plain MSE
        # (C_q replaced by the identity); every other non-sel configuration keeps the blanked 'score'
        keep_mse = c["kv_flags"] == "headsel" and c["kv_proxy"] == "mse"
        c.update(kv_cands="", kv_gran="head", kv_proxy="mse" if keep_mse else "score")
    if c["kv_flags"] != "dyn":
        c["kv_dyn"] = ""
    elif not c["kv_dyn"]:
        c["kv_dyn"] = "cq"
    if not calibrated(c["kv_flags"]) or (c["w"] == "bf16" and c["a"] == "bf16"):
        c["joint_calib"] = False
    if c["kv_flags"] not in ("headsel", "head", "all") and not c["kv_flags"].startswith("greedy"):
        c["kv_static_only"] = ""
    if c["kv_k"] == "bf16" and c["kv_v"] == "bf16":
        c.update(kv_k_axis="head", kv_resid=0, kv_sink=0, kv_flush=0, kv_kvq="", kv_center="")
    if c["kv_k"] == "bf16":
        c["kv_center"] = ""
    if not c["kv_resid"] or c["kv_flush"]:
        c["kv_resid_mode"] = "sliding"
    if not c["calib_windows"]:
        c["calib_rot"] = ""
    if not c["kv_center"]:
        c["kc_windows"] = 0                 # opt-in: centering statistics on the first kc_windows calibration windows
    c["kc_windows"] = int(c["kc_windows"] or 0)
    if c["kv_flags"] != "dyn":
        c["kv_dyn_off"] = ""                # opt-in: KV layers (comma list) that keep the plain cache under kv_flags dyn
    c["kv_dyn_off"] = str(c["kv_dyn_off"] or "")
    # opt-in (calibration sensitivity): att rule only -- kv_att_nq = number of kept calibration queries per
    # KV head, sampled evenly over the calibration windows (0: the default 64 positions per window, last 512 kept);
    # kv_att_w = '' (w = min(1, exp(s q.x - logZ_q)), default) | nocap (no cap at 1) | renorm (per calibration query a
    # softmax over the keys of the window being encoded)
    if not (c["kv_flags"] == "dyn" and c["kv_dyn"].split("+")[0] == "att"):
        c["kv_att_nq"], c["kv_att_w"] = 0, ""
    c["kv_att_nq"], c["kv_att_w"] = int(c["kv_att_nq"] or 0), str(c["kv_att_w"] or "")
    if c["kv_att_w"] not in ("", "nocap", "renorm"):
        raise ValueError(f"kv_att_w {c['kv_att_w']!r}: '' | nocap | renorm")
    c["calib_rot"] = str(c["calib_rot"] or "")
    if c["eval"] == "kl":
        c["lm_tasks"], c["lm_limit"], c["lm_max_len"] = "", 0, 0
    else:
        c["windows"] = 0
        c["eval_data"], c["eval_len"] = "wikitext", 2048
    return {k: c[k] for k in sorted(c)}


LATER_KEYS = {"wq": "rtn", "gptq_windows": 0, "mask_file": "", "protect": 0, "proxy": "rtn", "joint_calib": False,
              "kv_k_axis": "head", "kv_resid": 0, "kv_sink": 0, "eval_split": "test",
              "eval": "kl", "lm_tasks": "", "lm_limit": 0, "lm_max_len": 0,
              "kv_cands": "", "kv_gran": "head", "kv_proxy": "score", "kv_dyn": "",
              "kv_xform": "", "kv_skip": 0, "kv_flush": 0, "calib_data": "wikitext", "calib_len": 2048,
              "kv_resid_mode": "sliding", "kv_static_only": "", "kv_kvq": "", "eval_data": "wikitext", "eval_len": 2048,
              "kv_center": "", "calib_rot": "", "kv_rotatekv": "", "kc_windows": 0, "kv_dyn_off": "",
              "kv_att_nq": 0, "kv_att_w": ""}
# keys added after the first runs: omitted at their default


def config_id(cfg: dict) -> str:
    """stable id: keys introduced later are dropped when at their default, so earlier results keep their ids."""
    c = canonical(cfg)
    c = {k: v for k, v in c.items() if not (k in LATER_KEYS and v == LATER_KEYS[k])}
    return hashlib.sha1(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]


@torch.no_grad()
def _e2e_refine(model, tok, c):
    """kv_gran='e2e' (upper bound on what a proxy can reach at layer granularity): start from the per-head proxy
    choice, then for every layer and for K and V separately try each candidate transform for the whole layer and keep
    the one with the lowest true KL(BF16 || Q) on the calibration windows (layers in order, one sweep)."""
    wins = windows(token_stream(c["model"], tok, "train"), c["calib_windows"]).cuda()
    STATE["quant"] = False
    ref = [F.log_softmax(model(w[None]).logits[0].float(), -1) for w in wins]
    STATE["quant"] = True

    def kl():
        tot = 0.0
        for w, lr in zip(wins, ref):
            lq = F.log_softmax(model(w[None]).logits[0].float(), -1)
            tot += float((lr.exp() * (lr - lq)).sum(-1).mean())
        return tot
    best = kl()
    for li in sorted(QA.KV["measure"]["k"]):
        for kind in ("k", "v"):
            if kind not in QA.KV["measure"]:
                continue
            keep = QA.KV[f"{kind}x"][li]
            for name, md in QA.KV["measure"][kind][li].items():
                QA.KV[f"{kind}x"][li] = md
                e = kl()
                if e < best - 1e-9:
                    best, keep = e, md
            QA.KV[f"{kind}x"][li] = keep
    STATE["quant"] = False


@torch.no_grad()
def _center_auto(model, tok, c):
    """kv_center='auto': one decision per run (model, cache method, protection) between the selective centering of
    kv_center='sel' and no centering, by KL(BF16 || Q) of the complete quantized cache on the calibration windows.
    (Static flags of kv_flags headsel / greedy were calibrated on the centered keys and are kept for both candidates.)"""
    kc = QA.KV["kc"]
    kl = {"none": 0.0, "sel": 0.0}
    for w in windows(token_stream(c["model"], tok, "train"), c["calib_windows"]):
        b = w[None].cuda()
        STATE["quant"] = False
        lr = F.log_softmax(model(b).logits[0].float(), -1)
        STATE["quant"] = True
        for name in kl:
            QA.KV["kc"] = kc if name == "sel" else None
            lq = F.log_softmax(model(b).logits[0].float(), -1)
            kl[name] += float((lr.exp() * (lr - lq)).sum(-1).mean()) / c["calib_windows"]
            del lq
        del lr
    STATE["quant"] = False
    choice = "sel" if kl["sel"] < kl["none"] else "none"
    QA.KV["kc"] = kc
    QA.KV["kc_auto"] = dict(calib_kl_none=kl["none"], calib_kl_sel=kl["sel"], choice=choice, sel=QA.kc_summary())
    QA.KV["kc"] = kc if choice == "sel" else None
    QA.KV["dynstat"], QA.KV["kvqstat"] = {}, {}


CALIB = {"data": "wikitext", "len": 2048, "offs": None}   # calibration source of the current run (set in build)
# offs: calib_rot position offsets, one forward per (window, offset) of the train split (offset 0 repeated: weight 1/2)


def _forward_windows(model, tok, key, split, n):
    """calibration / Hessian passes run on 'train': WikiText-2 train windows, or FineWeb-Edu windows of CALIB['len']
    tokens (calib_data = fineweb, as WUSH-KV calibrates)."""
    if split == "train" and CALIB["data"] == "fineweb":
        split, seqlen = "fineweb", CALIB["len"]
    elif split == "train" and CALIB["data"] == "wikitext" and CALIB["len"] != 2048:
        seqlen = CALIB["len"]               # opt-in longer WikiText-2 calibration windows
    else:
        seqlen = 2048
    offs = CALIB.get("offs") if split in ("train", "fineweb") else None
    with torch.no_grad():
        for win in windows(token_stream(key, tok, split), n, seqlen):
            if not offs:
                model(win[None].cuda())
                continue
            for o in offs:
                model(win[None].cuda(), position_ids=torch.arange(o, o + seqlen, device="cuda")[None])


def _rot_offsets(model, L: int, seqlen: int) -> list[int]:
    """calib_rot offsets: 0 (x 7, weight 1/2) and j L' / 8, j = 1..7, capped at L' - seqlen (L' = min(L, max pos))."""
    tc = getattr(model.config, "text_config", None) or model.config
    Lc = min(int(L), int(getattr(tc, "max_position_embeddings", L) or L))
    # interleaved (0, o1, 0, o2, ...): sums are order-free; the KIVI-C sampled-query buffer keeps the latest passes
    return [x for j in range(1, 8) for x in (0, min(j * Lc // 8, max(0, Lc - seqlen)))]


KV_DYN_TOKENS = {"S", "P", "F", "FS", "KONLY", "VONLY", "STATIC", "SW", "J"}
# SW (opt-in, Gemma-3 diagnostic): the att rule's log Z_q of sliding-window layers is taken over the window
# only; J (opt-in): per token group the best of the 2^(D/r_b) joint {I, H_rb} slot combinations under the full criterion   # kv_dyn candidate / option tokens (case-free)


def build(c: dict):
    """load the model, calibrate the flags and quantize the weights; returns (model, tok, mods, timings)."""
    t0 = time.time()
    model, tok = load_model(c["model"], attn_implementation="blockflag")
    # calib_len = "auto": WUSH-KV's 32k-token calibration sequences, capped at the model's context length
    clen = c["calib_len"]
    if clen == "auto":
        tc = getattr(model.config, "text_config", None) or model.config
        clen = min(32768, int(getattr(tc, "max_position_embeddings", 32768)))
    CALIB.update(data=c["calib_data"], len=int(clen),
                 offs=_rot_offsets(model, int(c["calib_rot"]), int(clen)) if c.get("calib_rot") else None)
    mcfg = text_config(model)
    D = head_dim(mcfg)
    # KV-cache layers only (hybrid models: the full-attention layers), by layer_idx; kv_skip counts these layers
    ATT = attn_layers(model)
    Hkv, nl, kvl = mcfg.num_key_value_heads, len(ATT), sorted(ATT)
    kv_on = c["kv_k"] != "bf16" or c["kv_v"] != "bf16"
    kv_rb = D if (c["kv_rb"] or 0) < 0 else (c["kv_rb"] or 64)      # -1 = whole head_dim (QuaRot-style baseline)
    kv_greedy = kv_on and calibrated(c["kv_flags"])
    QA.KV.update(k=c["kv_k"], v=c["kv_v"], rb=kv_rb, calib_on=kv_greedy and not c["joint_calib"],
                 acc={}, kmask=None, vmask=None, k_axis=c["kv_k_axis"], resid=c["kv_resid"], sink=c["kv_sink"], flush=c["kv_flush"], resid_mode=c["kv_resid_mode"],
                 accfull={}, kfull=None, vfull=None, accx={}, measure=None, meas={}, kx=None, vx=None,
                 sel=None, dyn=None, cq={}, dynstat={}, qs={}, lz={}, xf=None, xf_calib=None, xs={},
                 skip=c["kv_skip"], nl=nl,
                 bf16_layers=set(kvl[:c["kv_skip"]] + kvl[len(kvl) - c["kv_skip"]:]),
                 kvq=(c["kv_kvq"] or None) if kv_on else None, kvq_calib=False, kvq_cb={}, kvq_thr={}, kvq_acc={},
                 kvqstat={}, rope=getattr(model.model, "rotary_emb", None), kc=None, kc_calib=False, kc_acc={}, kc_meas=False, kc_auto=None,
                 kc_err={}, kc_frac={}, rk=None, rk_calib=False, rk_acc={}, rk_perm={}, rk_k={}, rk_piv={},
                 rk_cur=None, rk_stat={})
    if kv_on and c["kv_center"]:
        if c["kv_center"] not in ("post", "sel", "auto") or c["kv_kvq"]:
            raise ValueError("kv_center: post | sel | auto, not combined with kv_kvq")
        # static key centering: one pass over the calibration windows (BF16 cache) for the post-RoPE key means, before
        # every other calibration (flags / transforms are then calibrated on the centered keys); linears are still BF16
        con = QA.KV["calib_on"]
        STATE["calib"], QA.KV["kc_calib"], QA.KV["calib_on"] = True, True, False
        _forward_windows(model, tok, c["model"], "train", c["kc_windows"] or c["calib_windows"])
        STATE["calib"], QA.KV["kc_calib"], QA.KV["calib_on"] = False, False, con
        QA.KV["kc"] = {li: (s_ / n_).float() for li, (s_, n_) in QA.KV["kc_acc"].items()}
        QA.KV["kc_acc"] = {}
        if c["kv_center"] in ("sel", "auto"):
            # calibrated selection: per (layer, KV head, scale group of the key format) keep the mean or 0, whichever
            # gives the lower attention-weighted logit-error variance on the same calibration windows (any constant
            # vector keeps attention exact, so a partial mean is as exact as the full one)
            STATE["calib"], QA.KV["kc_meas"], QA.KV["calib_on"] = True, True, False
            _forward_windows(model, tok, c["model"], "train", c["kc_windows"] or c["calib_windows"])
            STATE["calib"], QA.KV["kc_meas"], QA.KV["calib_on"] = False, False, con
            QA.kc_select()
    QA.KV["att_nq"], QA.KV["att_nw"], QA.KV["att_w"] = c["kv_att_nq"], c["calib_windows"], c["kv_att_w"]   # opt-in
    if c["kv_dyn_off"]:                     # opt-in per-layer guard: dynamic selection off in these KV layers
        off = {int(x) for x in c["kv_dyn_off"].split(",") if x.strip()}
        QA.KV["dyn_layers"] = set(attn_layers(model)) - off
    else:
        QA.KV["dyn_layers"] = None
    rk_hooks = []
    if kv_on and c["kv_rotatekv"]:
        parts = c["kv_rotatekv"].split("+")
        if not re.fullmatch(r"gh\d+", parts[0]) or any(not (p in ("ro", "sel") or re.fullmatch(r"ma\d+", p)) for p in parts[1:]):
            raise ValueError(f"kv_rotatekv: gh<G>[+ro][+ma<n>][+sel], got {c['kv_rotatekv']}")
        if (c["kv_flags"] != "none" or c["kv_xform"] or c["kv_kvq"] or c["kv_center"] or c["kv_k_axis"] != "head"
                or c["kv_flush"] or c["kv_resid_mode"] != "sliding"):
            raise ValueError("kv_rotatekv is a complete cache recipe: only kv_k / kv_v / kv_sink / kv_resid / kv_skip")
        ma = [int(p[2:]) for p in parts[1:] if p.startswith("ma")]
        QA.KV.update(rk=dict(G=QA.rk_group(Hkv, int(parts[0][2:])), ro="ro" in parts, ma=ma[0] if ma else 0,
                             sel="sel" in parts), rk_calib="ro" in parts or "sel" in parts, rk_W={}, rk_M={})
        rk_hooks = QA.rk_install(model)
    if QA.KV["kvq"] is not None:
        if c["kv_k_axis"] != "head" or c["kv_flags"] != "none" or c["kv_xform"]:
            raise ValueError("kv_kvq is a complete cache recipe: no kv_k_axis / kv_flags / kv_xform on top")
    xform = c["kv_xform"] if kv_on else ""
    xfile = xform[5:] if xform.startswith("file:") else ""     # fixed transform from a file: no calibration
    if xfile:
        xform = ""
    if (xform or xfile) and c["kv_flags"] not in ("none", "all", "dyn"):
        raise ValueError("kv_xform combines only with kv_flags none | all | dyn")
    if (xform or xfile) and c["kv_flags"] == "dyn" and not c["kv_dyn"].startswith(("cq", "att")):
        raise ValueError("kv_xform with dynamic bits needs the cq or att rule")   # att: queries mapped by X_q (kivi_c)
    if c["kv_flags"] == "dyn":                             # kv_dyn = <rule>[+f]: rule mse | cq, +f whole-head option
        # kv_dyn = <rule>[+<cand>]...: rule mse | cq | att; per-vector candidates S (sign-flipped H_rb, besides H_rb);
        # per-head candidates P (RoPE-pair-interleaved blocks), f / F (whole-head H), FS (whole-head H after signs)
        parts = c["kv_dyn"].split("+")
        cand = {x.upper() for x in parts[1:]}
        # fail loudly on a token the parser does not know (an unknown token used to be ignored silently, e.g. 'static'
        # on a tree without the static-selection option would have run the dynamic rule); validation only, so no
        # config id changes
        bad = [x for x in parts[1:] if not (x.upper() in KV_DYN_TOKENS or re.fullmatch(r"[QRG]\d+", x.upper()))]
        if parts[0] not in ("mse", "cq", "att") or bad:
            raise ValueError(f"kv_dyn {c['kv_dyn']!r}: unknown rule / token(s) {bad or parts[0]!r} (rule mse | cq | att; "
                             f"tokens {sorted(KV_DYN_TOKENS)} | Q<k> | R<k> | G<k>)")
        # codebook-size scan: Q<k> adds k sign-flipped H_rb per sub-vector, R<k> adds k sign-flipped whole-head transforms
        qk = max([int(c[1:]) for c in cand if c[:1] == "Q" and c[1:].isdigit()] or [0])
        rk = max([int(c[1:]) for c in cand if c[:1] == "R" and c[1:].isdigit()] or [0])
        local = ["H"] + (["HS"] if "S" in cand else []) + [f"HS{i}" for i in range(2, qk + 2)]
        head = [h for h in ("P", "F", "FS") if h in cand] + [f"FS{i}" for i in range(2, rk + 2)]
        grp = max([int(c[1:]) for c in cand if c[:1] == "G" and c[1:].isdigit()] or [1])   # G<g>: shared per g tokens
        QA.KV["dyn"] = dict(rule=parts[0], local=local, head=head, group=grp,
                            only="k" if "KONLY" in cand else ("v" if "VONLY" in cand else None),
                            static="STATIC" in cand, sw="SW" in cand, joint="J" in cand)
        # kv_dyn = <rule>+...+static (opt-in, kivi2s control): the same candidates, two-level comparison and criterion,
        # but one frozen choice per (layer, KV head, sub-vector slot) for K and V, made on the calibration windows by
        # the summed criterion (qattention.static_select); evaluation then runs the static masks (kmask / kfull ...)
        if QA.KV["dyn"]["static"] and (c["kv_xform"] or QA.KV["dyn"]["local"] != ["H"]
                                       or QA.KV["dyn"]["head"] not in ([], ["F"]) or QA.KV["dyn"]["only"]):
            raise ValueError("kv_dyn ...+static: local H (+ F) on K and V, no kv_xform")
    cands = [x for x in c["kv_cands"].split(",") if x]
    if c["kv_flags"] == "sel":
        # 'f' and block sizes >= D are the whole head; other block sizes need their own pattern statistics
        rbs = sorted({int(x[1:]) for x in cands if x[0] in "ab" and int(x[1:]) < D} - {kv_rb})
        cands = [("f" if x[0] in "ab" and int(x[1:]) >= D else x) for x in cands]
        cands = list(dict.fromkeys(cands))
        QA.KV["sel"] = dict(rbs=rbs, proxy=c["kv_proxy"])
    QA.KV["cq_eye"] = c["kv_flags"] == "headsel" and c["kv_proxy"] == "mse"     # opt-in, see canonical
    QA.KV["co"] = {li: at.o_proj.weight.float().square().sum(0).view(Hkv, -1, D).sum(1) for li, at in ATT.items()}
    lin_on = c["w"] != "bf16" or c["a"] != "bf16"
    greedy = lin_on and calibrated(c["flags"])
    mods: dict[str, SelLinear] = {}
    if lin_on:
        for name, parent, pn in decoder_linears(model):
            mods[name] = SelLinear(getattr(parent, pn), c["a"], c["w"], c["rb"] or 64)
            mods[name].calib_on = greedy and c["proxy"] == "rtn"
            mods[name].track_amax = c["protect"] > 0 or c["flags"] == "rule"
            setattr(parent, pn, mods[name])
    need_h = lin_on and (c["wq"] == "gptq" or (greedy and c["proxy"] == "gptq"))
    if need_h:                                                       # one Hessian per shared input
        seen = set()
        for name, m in mods.items():
            if share_key(name) not in seen:
                m.hess_key = share_key(name)
                seen.add(m.hess_key)
    # phase A: RTN-proxy flag calibration, activation maxima for protection, KV calibration (unless joint)
    hooks = []
    if xform:                                   # static KV transform statistics (WUSH-KV / SmoothAttention)
        QA.KV["xf_calib"] = xform.split(":")[0]
        hooks = QA.smooth_hooks(model) if xform.split(":")[0] == "smooth" else []
    # FineWeb-Edu calibration (WUSH-KV's protocol, 128 x 32k tokens) costs ~2.7 h per model on an H20, so its KV
    # statistics are cached per (model, data, length, windows): the WUSH sums M_K / H_K / M_V (always collected in such
    # a pass, they are cheap) and C_q of the dynamic rule.  Only configurations whose calibration needs nothing else use
    # the cache (no linear quantization, static transform none | wush, dynamic rule cq or no KV calibration).
    need = ({"xs"} if xform.split(":")[0] == "wush" else set()) | ({"cq"} if QA.KV["calib_on"] and QA.KV["dyn"] is not None else set())
    cacheable = (c["calib_data"] == "fineweb" and not lin_on and xform.split(":")[0] in ("", "wush")
                 and not c.get("calib_rot")                                     # the cache key has no offsets
                 and (not QA.KV["calib_on"] or (QA.KV["dyn"] is not None and QA.KV["dyn"]["rule"] == "cq")))
    cfile = paths.CACHE / f"calibstats_{c['model']}_{c['calib_data']}{CALIB['len']}_{c['calib_windows']}.pt"
    cached = torch.load(cfile) if cacheable and cfile.exists() else {}
    if cacheable and need and need <= set(cached):
        if "xs" in need:
            QA.KV["xs"] = {li: {k: v.cuda() for k, v in st.items()} for li, st in cached["xs"].items()}
        if "cq" in need:
            QA.KV["cq"] = {li: v.cuda() for li, v in cached["cq"].items()}
    elif (greedy and c["proxy"] == "rtn") or c["protect"] > 0 or c["flags"] == "rule" or QA.KV["calib_on"] or xform \
            or QA.KV["kvq"] is not None or QA.KV["rk_calib"]:
        if cacheable:
            QA.KV["xf_calib"] = "wush"
        STATE["calib"] = True
        QA.KV["kvq_calib"] = QA.KV["kvq"] is not None
        _forward_windows(model, tok, c["model"], "train", c["calib_windows"])
        STATE["calib"] = False
        QA.KV["kvq_calib"] = False
        if cacheable:
            new = {"xs": {li: {k: v.cpu() for k, v in st.items()} for li, st in QA.KV["xs"].items()}}
            if QA.KV["dyn"] is not None and QA.KV["calib_on"]:
                new["cq"] = {li: v.cpu() for li, v in QA.KV["cq"].items()}
            paths.CACHE.mkdir(parents=True, exist_ok=True)
            torch.save({**cached, **new}, str(cfile) + ".tmp")
            Path(str(cfile) + ".tmp").replace(cfile)
            if not xform:
                QA.KV["xf_calib"], QA.KV["xs"] = None, {}
    for h in hooks:
        h.remove()
    if QA.KV["dyn"] is not None and QA.KV["dyn"].get("static"):
        # static selection (kivi2s): two more passes over the calibration windows (BF16 cache, linears BF16 as in
        # every KV calibration): per-slot candidate errors, then the head-level option given the frozen slot choice
        for stage in ("local", "head"):
            STATE["calib"], QA.KV["sacc"] = True, stage
            _forward_windows(model, tok, c["model"], "train", c["calib_windows"])
            STATE["calib"], QA.KV["sacc"] = False, None
            QA.static_finalize(stage)
    if QA.KV["kvq"] is not None:                # KVQuant-style: thresholds and codebooks from the calibration keys / values
        QA.kvq_finalize()
    if QA.KV["rk_calib"]:                       # RotateKV reordering indices (Algorithm 1)
        QA.rk_finalize()
        QA.KV["rk_calib"] = False
    if xform:
        QA.KV["xf"] = QA.xf_finalize(model, xform, c["kv_k"], c["kv_v"])
        QA.KV["xf_calib"], QA.KV["xs"] = None, {}
    if xfile:
        QA.KV["xf"] = {int(li): {k: v.float().cuda().contiguous() for k, v in X.items()}
                       for li, X in torch.load(xfile).items()}
    # phase B: Hessians (GPTQ quantization and / or GPTQ-aware proxy)
    HESS.clear()
    if need_h:
        STATE["hess"] = True
        _forward_windows(model, tok, c["model"], "train", c["gptq_windows"])
        STATE["hess"] = False
    if greedy and c["proxy"] == "gptq":
        for n, m in mods.items():
            m.gptq_pattern_errors(HESS[share_key(n)])
    t_cal = time.time() - t0
    margin = lambda f: float(f.split(":")[1]) if ":" in f else 0.0  # noqa: E731  greedy:<m> = conservative flags
    if greedy:
        gm = layer_masks(mods) if c["flags"] == "layer" else greedy_masks(mods, margin(c["flags"]))
    else:
        gm = {}
    fm = torch.load(c["mask_file"]) if c["flags"] == "file" else {}
    for n, m in mods.items():
        if c["flags"] == "none":
            mask = None
        elif c["flags"] == "all":
            mask = torch.ones(m.nvec, dtype=torch.bool, device="cuda")
        elif c["flags"].startswith("proj:"):
            mask = torch.full((m.nvec,), n.split(".")[-1] in c["flags"][5:].split("+"), dtype=torch.bool, device="cuda")
        elif calibrated(c["flags"]):
            mask = gm[n]
        elif c["flags"] == "rule":                          # closed-form ablation, shared per input like the others
            src = next(mm for nn, mm in mods.items() if share_key(nn) == share_key(n))
            mask = rule_mask(src).to("cuda")
        elif c["flags"] == "file":
            mask = fm.get(n)
            mask = None if mask is None else mask.to("cuda").bool()
        else:
            raise ValueError(c["flags"])
        if c["protect"] > 0 and mask is not None:            # protection is per shared input (same activation)
            src = next(mm for nn, mm in mods.items() if share_key(nn) == share_key(n))
            mask = mask & ~protected_vectors(src, c["protect"]).to(mask.device)
        m.finalize(mask, HESS.get(share_key(n)) if c["wq"] == "gptq" else None)
        m.acc = None
    HESS.clear()
    # phase C: joint KV calibration with the quantized linears in place (the cache itself stays BF16 meanwhile)
    if kv_greedy and c["joint_calib"]:
        for m in mods.values():
            m.calib_on = m.track_amax = False
        QA.KV["calib_on"] = True
        STATE["quant"] = STATE["calib"] = True
        _forward_windows(model, tok, c["model"], "train", c["calib_windows"])
        STATE["quant"] = STATE["calib"] = False
        QA.KV["calib_on"] = False
    if kv_greedy and c["kv_flags"] == "sel":            # pass 2: head-level error of every candidate transform
        QA.KV["measure"] = {kind: QA.sel_candidates(kind, cands, Hkv, D, "cuda") for kind in ("k", "v")
                            if c[f"kv_{kind}"] != "bf16"}
        QA.KV["calib_on"] = False
        STATE["calib"] = True
        _forward_windows(model, tok, c["model"], "train", c["calib_windows"])
        STATE["calib"] = False
    for kind in ("k", "v"):
        if c[f"kv_{kind}"] == "bf16" or c["kv_flags"] == "none":
            continue
        if c["kv_static_only"] and kind != c["kv_static_only"]:     # static flags on the other tensor only
            continue
        if c["kv_flags"] == "all":
            QA.KV[f"{kind}mask"] = {li: torch.ones(Hkv, D // QA.KV["rb"], dtype=torch.bool, device="cuda") for li in kvl}
        elif c["kv_flags"] == "head":
            QA.KV[f"{kind}mask"] = QA.head_kv_masks(kind)
        elif c["kv_flags"] == "headsel":
            QA.KV[f"{kind}mask"], QA.KV[f"{kind}full"] = QA.headsel_kv_masks(kind)
        elif c["kv_flags"] == "sel":
            QA.KV[f"{kind}x"] = QA.sel_choose(kind, "head" if c["kv_gran"] == "e2e" else c["kv_gran"])
        elif c["kv_flags"] == "dyn":
            QA.KV["dynstat"] = {}                         # statistics are collected on the evaluation windows
        elif c["kv_flags"].startswith("greedy"):
            QA.KV[f"{kind}mask"] = QA.greedy_kv_masks(kind, margin(c["kv_flags"]))
        else:
            raise ValueError(c["kv_flags"])
    if c["kv_flags"] == "sel" and c["kv_gran"] == "e2e":
        _e2e_refine(model, tok, c)
    if kv_on and c["kv_center"] == "auto":
        _center_auto(model, tok, c)
    QA.KV["measure"] = None
    torch.cuda.empty_cache()
    return model, tok, mods, dict(build=time.time() - t0, calibration=t_cal)


def eval_windows(model_key: str, tok, c: dict) -> torch.Tensor:
    """[n, eval_len] evaluation token windows: WikiText-2 (the paper's protocol), PG-19 books or code (eval_data)."""
    L, n = int(c["eval_len"]), c["windows"]
    if c["eval_data"] == "wikitext":
        return windows(token_stream(model_key, tok, c["eval_split"]), n, L)
    paths.CACHE.mkdir(parents=True, exist_ok=True)
    if c["eval_data"] == "pg19":
        # one window per book: the first L tokens of each test book that has >= 32768 tokens (the same books at every
        # L <= 32768, so that lengths are compared on the same documents), in file order
        cp = paths.CACHE / f"{model_key}_pg19_books32k.pt"
        if cp.exists():
            books = torch.load(cp)
        else:
            import pandas as pd
            books = []
            for t in pd.read_parquet(paths.DATA["pg19"], columns=["text"])["text"]:
                ids = tok(t[:400_000], return_tensors="pt").input_ids[0]
                if ids.numel() >= 32768:
                    books.append(ids[:32768].clone())
            books = torch.stack(books)
            torch.save(books, cp)
        assert L <= books.shape[1]
        return books[: (n or books.shape[0]), :L]
    if c["eval_data"] == "code":
        cp = paths.CACHE / f"{model_key}_code_ids.pt"
        if cp.exists():
            ids = torch.load(cp)
        else:
            import gzip
            docs, nch = [], 0
            with gzip.open(paths.DATA["code"], "rt") as fh:
                for line in fh:                          # files in file order, ~2.5M characters (> 400 x 2048 tokens)
                    docs.append(json.loads(line)["content"])
                    nch += len(docs[-1])
                    if nch > 2_500_000:
                        break
            ids = tok("\n\n".join(docs), return_tensors="pt").input_ids[0]
            torch.save(ids, cp)
        return windows(ids, n, L)
    if c["eval_data"] in TEXT_EVAL:
        cp = paths.CACHE / f"{model_key}_{c['eval_data']}_ids.pt"
        if cp.exists():
            ids = torch.load(cp)
        else:
            import pandas as pd
            docs, nch = [], 0
            for t in pd.read_parquet(paths.DATA[c["eval_data"]], columns=["text"])["text"]:
                docs.append(t)
                nch += len(t)
                if nch > TEXT_EVAL[c["eval_data"]]:
                    break
            ids = tok("\n\n".join(docs), return_tensors="pt").input_ids[0]
            torch.save(ids, cp)
        return windows(ids, n, L)
    raise ValueError(c["eval_data"])


TEXT_EVAL = {"owm": 2_500_000, "wiki_de": 2_500_000, "wiki_zh": 1_200_000}   # characters read (> 150 x 2048 tokens)


BUCKETS = (0, 2048, 8192, 32768)                       # position buckets of the per-window KL (long windows)


def eval_kl(model, tok, c: dict) -> dict:
    per_win = []
    long_ = int(c["eval_len"]) > 2048
    with torch.no_grad():
        for win in eval_windows(c["model"], tok, c):
            b = win[None].cuda()
            tgt = b[0, 1:]
            if not long_:
                STATE["quant"] = False
                lr = F.log_softmax(model(b).logits[0, :-1].float(), -1)
                STATE["quant"] = True
                lq = F.log_softmax(model(b).logits[0, :-1].float(), -1)
                per_win.append(dict(kl=float((lr.exp() * (lr - lq)).sum(-1).mean()),
                                    nll_ref=float(-lr.gather(1, tgt[:, None]).mean()),
                                    nll_q=float(-lq.gather(1, tgt[:, None]).mean()),
                                    agree=float((lr.argmax(-1) == lq.argmax(-1)).float().mean()), tokens=int(tgt.numel())))
                del lr, lq
                continue
            # long windows: final hidden states of both passes, logits in chunks of 2048 positions (a 32k x 152k
            # FP32 logit matrix would not fit); identical metrics, plus the KL per position bucket
            STATE["quant"] = False
            hr = model.model(b).last_hidden_state[0, :-1]
            STATE["quant"] = True
            hq = model.model(b).last_hidden_state[0, :-1]
            head = model.get_output_embeddings()
            kl = torch.zeros(tgt.numel(), dtype=torch.float64, device=b.device)
            s_nr = s_nq = s_ag = 0.0
            for c0 in range(0, tgt.numel(), 2048):
                c1 = min(tgt.numel(), c0 + 2048)
                lr = F.log_softmax(head(hr[c0:c1]).float(), -1)
                lq = F.log_softmax(head(hq[c0:c1]).float(), -1)
                kl[c0:c1] = (lr.exp() * (lr - lq)).sum(-1).double()
                tg = tgt[c0:c1, None]
                s_nr += float(-lr.gather(1, tg).double().sum())
                s_nq += float(-lq.gather(1, tg).double().sum())
                s_ag += float((lr.argmax(-1) == lq.argmax(-1)).double().sum())
                del lr, lq
            N = tgt.numel()
            bk = {f"{a}-{b_}": float(kl[a:min(b_, N)].mean()) for a, b_ in zip(BUCKETS[:-1], BUCKETS[1:]) if a < N}
            per_win.append(dict(kl=float(kl.mean()), nll_ref=s_nr / N, nll_q=s_nq / N, agree=s_ag / N, tokens=int(N),
                                kl_bucket=bk))
            del hr, hq, kl
    bad = [i for i, w in enumerate(per_win) if not all(math.isfinite(v) for v in w.values() if not isinstance(v, dict))]
    if bad:                                    # never store a non-finite result: fail so the scheduler records it
        raise FloatingPointError(f"non-finite metrics in test windows {bad[:10]}")
    N = sum(w["tokens"] for w in per_win)
    mean = lambda k: sum(w[k] * w["tokens"] for w in per_win) / N  # noqa: E731
    return dict(kl=mean("kl"), ppl_ref=math.exp(mean("nll_ref")), ppl_q=math.exp(mean("nll_q")),
                top1_agree=mean("agree"), tokens=N, per_window=per_win)


def _dyn_summary() -> dict:
    """kv_flags='dyn': rotated fraction of (token, vector) pairs, whole-head fraction, and the consistency of the bit
    across tokens: mean over (layer, head, vector) of max(p, 1 - p), p = fraction of tokens with the bit set
    (1.0 = a static bit per vector reproduces the per-token oracle exactly)."""
    out = {}
    for kind, st in QA.KV["dynstat"].items():
        if kind == "k_mode":                                       # fraction of (group, head) choosing per-channel keys
            out[kind] = sum(a / n for a, n in st.values()) / max(len(st), 1)
            continue
        cnt = torch.cat([v[0].flatten() for v in st.values()])
        ntok = sum(v[1] for v in st.values()) / len(st)
        hc = torch.cat([v[2] for v in st.values()], 1)                 # [n head candidates, layers * heads]
        p = cnt / ntok
        out[kind] = dict(rot=float(p.mean()), consistency=float(torch.maximum(p, 1 - p).mean()),
                         frac_consistent_90=float((torch.maximum(p, 1 - p) >= 0.9).double().mean()),
                         head=dict(zip(QA.KV["dynnames"], (hc / ntok).mean(1).tolist())))
    return out


def _sel_summary(d: dict) -> dict:
    """kv_flags='sel': fraction of channels rotated per block size (key 'D' = whole head) over all layers / heads."""
    tot, per = 0, {}
    for md in d.values():
        for r, m in md.items():
            per[r] = per.get(r, 0) + int(m.sum()) * r
    for li in d:
        tot += 1
    if not d:
        return {}
    Hd = next((m.shape[0] * m.shape[1] * r for md in d.values() for r, m in md.items()), 1)
    return {str(r): v / (tot * Hd) for r, v in sorted(per.items())}


def kv_bits(c: dict, D: int, nl: int) -> dict:
    """storage bits per cached K / V element (formats.bits: scales, zero points, norms) plus the per-token side bits of
    kv_flags='dyn' (fixed-length code of the per-sub-vector choice and of the per-head choice), averaged over layers
    with kv_skip boundary layers at 16 bits (nl = number of KV-cache layers)."""
    c = canonical(c)
    side = 0.0
    if c["kv_flags"] == "dyn":
        cand = {x.upper() for x in c["kv_dyn"].split("+")[1:]}
        rb = D if (c["kv_rb"] or 0) < 0 else (c["kv_rb"] or 64)
        n_local = 2 + ("S" in cand)
        n_head = 1 + sum(h in cand for h in ("P", "F", "FS"))
        side = math.ceil(math.log2(n_local)) / rb + (math.ceil(math.log2(n_head)) / D if n_head > 1 else 0.0)
        if "STATIC" in cand:          # static selection: frozen per (layer, head, slot), no per-token side bits
            side = 0.0
    q = nl - 2 * min(c["kv_skip"], nl // 2)
    out = {}
    for kind in ("k", "v"):
        b = FM.bits(c[f"kv_{kind}"], D) + (side if c[f"kv_{kind}"] != "bf16" else 0.0)
        out[kind] = (b * q + 16.0 * (nl - q)) / nl
    out["mean"] = (out["k"] + out["v"]) / 2
    out["side"] = side
    return out


def run(cfg: dict) -> tuple[dict, dict]:
    c = canonical(cfg)
    model, tok, mods, t = build(c)
    bits = kv_bits(c, head_dim(text_config(model)), len(attn_layers(model)))
    t1 = time.time()
    if c["eval"] == "kl":
        res = eval_kl(model, tok, c)
    else:
        from .downstream import eval_lm
        STATE["quant"] = True
        res = eval_lm(model, tok, c["lm_tasks"].split(","), c["lm_limit"] or None, max_len=c["lm_max_len"],
                      pretrained=str(paths.MODELS[c["model"]]))
    masks = {"linear": {n: m.mask.cpu() for n, m in mods.items() if m.mask is not None},
             "k": {li: m.cpu() for li, m in (QA.KV["kmask"] or {}).items()},
             "v": {li: m.cpu() for li, m in (QA.KV["vmask"] or {}).items()},
             "k_full": {li: m.cpu() for li, m in (QA.KV["kfull"] or {}).items()},
             "v_full": {li: m.cpu() for li, m in (QA.KV["vfull"] or {}).items()},
             "kx": {li: {r: m.cpu() for r, m in md.items()} for li, md in (QA.KV["kx"] or {}).items()},
             "vx": {li: {r: m.cpu() for r, m in md.items()} for li, md in (QA.KV["vx"] or {}).items()}}
    frac = lambda d: (sum(int(m.sum()) for m in d.values()) / max(sum(m.numel() for m in d.values()), 1)) if d else 0.0  # noqa: E731
    nvec_lin = sum(m.nvec for m in mods.values())
    lin_frac = sum(int(m.mask.sum()) for m in mods.values() if m.mask is not None) / max(nvec_lin, 1)
    return dict(id=config_id(c), config=c, **res, rot_frac=dict(linear=lin_frac, k=frac(masks["k"]), v=frac(masks["v"]),
                                                               k_full=frac(masks["k_full"]), v_full=frac(masks["v_full"]),
                                                               **{f"{kd}_sel": _sel_summary(masks[f"{kd}x"]) for kd in "kv" if masks[f"{kd}x"]},
                                                               **({"dyn": _dyn_summary()} if QA.KV["dyn"] is not None else {})),
                kv_bits=bits, **({"kvq": QA.kvq_summary()} if QA.KV.get("kvq") else {}),
                **({"kc": QA.kc_summary()} if QA.KV.get("kc") is not None else {}),
                **({"kc_auto": QA.KV["kc_auto"]} if QA.KV.get("kc_auto") else {}),
                **({"joint_stat": QA.KV["joint_stat"]} if QA.KV.get("joint_stat") else {}),
                **({"rotatekv": QA.rk_summary()} if QA.KV.get("rk") else {}), seconds=dict(total=t["build"] + time.time() - t1, calibration=t["calibration"]),
                env=dict(blockflag=__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                         host=platform.node())), masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON object")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = json.loads(a.config)
    res, masks = run(cfg)
    out = Path(a.out) if a.out else paths.RESULTS / "runs" / f"{res['id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    (paths.RESULTS / "masks").mkdir(parents=True, exist_ok=True)
    torch.save(masks, paths.RESULTS / "masks" / f"{res['id']}.pt")
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1))
    tmp.rename(out)                                            # atomic: a result file is always complete
    print(json.dumps({k: res.get(k) for k in ("id", "kl", "ppl_q", "top1_agree", "lm", "rot_frac")}, default=str)[:600])


if __name__ == "__main__":
    main()
