"""Export the FlagRot KV-cache calibration statistics for the vLLM plugin (runs in the RESEARCH env, torch 2.6).

The statistics are produced by exactly the code path of the research simulation: blockflag.run.build() with the
reference config (kv_k = kv_v = int4:32, kv_rb = 32, kv_flags = dyn, kv_dyn = cq+f+S+Q5+FS+R5 by default) runs the calibration
windows through the HF model with attn_implementation='blockflag', which accumulates
  C_q[l, h] = sum over the 4 calibration WikiText-2 TRAIN windows and the GQA query heads of head h of q q^T
              (post-RoPE queries, qattention.kv_calibrate / _cq)
  c_o[l, h] = column energy of o_proj restricted to the query heads of KV head h (run.build)
and we save them with the fixed transform constants (sign vector of HS / FS, RoPE-pair permutation).

usage (repo root, research env):
  CUDA_VISIBLE_DEVICES=6 python3 integrations/vllm_flagrot/scripts/export_calib.py --model llama8b \
      --out calib/flagrot_calib_llama8b.pt

Static key centering (int2c, opt-in plugin option "center"):
  --center         flagrot_center_<model>.pt: mu [La, Hkv, D] fp32 = QA.KV['kc'] of blockflag.run.build with
                   kv_center = post (per (layer, KV head, channel) mean of the post-RoPE keys of the 4 calibration
                   windows, summed in FP64 over all positions, divided, cast to fp32: qattention.blockflag_attention),
                   i.e. the simulation's own vector, plus its per-layer norms (= the run files' kc.mu_norm)
  --kv_center post with --static / --wush: the key-dependent calibrations made on the centered keys, as the simulation
                   does with kv_center (headsel -> flagrot_static_a2c_<model>.pt; WUSH base -> flagrot_calib_cc_<model>.pt)

Hybrid models (Qwen3.5 / 3.8: Gated DeltaNet + every 4th layer full attention): only the KV-cache layers have
statistics; the file then stores them for those layers only, in order, and meta["layers"] lists their model layer
indices (the plugin maps vLLM's layer index through it).  Head dim 256 and any GQA group are recorded in meta.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from blockflag import paths  # noqa: E402
from blockflag import qattention as QA  # noqa: E402
from blockflag import run as R  # noqa: E402

REF_CFG = dict(kv_k="int4:32", kv_v="int4:32", kv_rb=32, kv_flags="dyn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="blockflag.paths.MODELS key, e.g. llama8b, qwen8b")
    ap.add_argument("--calib_windows", type=int, default=4)
    ap.add_argument("--kv_dyn", default="cq+f+S+Q5+FS+R5",
                    help="research codebook string (locked method; 'cq+f+S+FS+P' = the earlier codebook). C_q / c_o "
                         "do not depend on it; it is recorded so the plugin picks the matching codebook")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fmt", default="int4:32", choices=["int4:32", "int4a:32", "int4ah:32", "int2a:32"],
                    help="--static: KV quantizer the headsel choice is calibrated for (int4a:32 = asymmetric INT4, "
                         "plugin quant=int4a; default output flagrot_static_a_<model>.pt)")
    ap.add_argument("--static", action="store_true",
                    help="export the STATIC FlagRot transforms instead (kv_flags=headsel, kv_rb=32: per (layer, KV "
                         "head, K/V) none | rb-32 block bits | whole-head H128) for flagrot_int4_static")
    ap.add_argument("--wush", action="store_true",
                    help="FlagRot-C (research method wush_k2): the calibrated block-diagonal WUSH base transform "
                         "(kv_xform=wush, 32-channel blocks = the int4a:32 scale groups) + the key-only 2x2 codebook "
                         "cq+f+konly on top; C_q is exported in the base's basis.  Default output "
                         "flagrot_calib_c_<model>.pt")
    ap.add_argument("--center", nargs="?", const="post", default="", choices=["post", "sel", "auto"],
                    help="export the static key-centering vector mu: post (default) -> flagrot_center_<model>.pt; "
                         "sel -> flagrot_center_sel_<prot>_<model>.pt (per (layer, head, 32-group) mean or 0, chosen "
                         "with the INT2 RTN proxy and the protection --prot); auto -> flagrot_center_auto_<method>_"
                         "<prot>_<model>.pt (sel or nothing for the whole run, by calibration KL of the --method cache; "
                         "'nothing' is stored as mu = 0, which the plugin treats as centering off)")
    ap.add_argument("--method", default="flagrot", choices=["flagrot", "rtn"], help="--center auto: cache method")
    ap.add_argument("--prot", default="s4w128", choices=["s4w128", "w0"], help="--center sel / auto: protection")
    ap.add_argument("--kv_center", default="", choices=["", "post"],
                    help="--static / --wush: calibrate on centered keys (research kv_center); default output names "
                         "flagrot_static_a2c_ / flagrot_calib_cc_<model>.pt")
    ap.add_argument("--kivi", default="", choices=["", "kivi", "kivi_c", "kivi_s", "kivi_sp8"],
                    help="kivi2: FlagRot-KIVI (simulator kivi_sel: kivi2:32 K per channel over 32-token groups, V per "
                         "token, kv_dyn att+f, S4W128) -> flagrot_kivi_<model>.pt; kivi_c: + the calibrated WUSH base "
                         "(kv_xform wush) and --calib_rot (default 32768: the phase-covered calibration) -> "
                         "flagrot_kivi_c_<model>.pt")
    ap.add_argument("--calib_rot", default="32768", help="--kivi kivi_c: research calib_rot ('' = off)")
    a = ap.parse_args()
    if a.kivi:
        return export_kivi(a)
    if a.center:
        return export_center(a)
    if a.static:
        return export_static(a)
    if a.wush:
        return export_wush(a)
    cfg = R.canonical(dict(model=a.model, windows=0, calib_windows=a.calib_windows, kv_dyn=a.kv_dyn, **REF_CFG))
    t0 = time.time()
    model, tok, _, t = R.build(cfg)
    mc = model.config.get_text_config()
    D = getattr(mc, "head_dim", None) or mc.hidden_size // mc.num_attention_heads
    L, Hkv = mc.num_hidden_layers, mc.num_key_value_heads
    layers = sorted(QA.KV["cq"])                                                  # KV-cache layers (hybrid: a subset)
    cq = torch.stack([QA.KV["cq"][li].float().cpu() for li in layers])           # [La, Hkv, D, D]
    co = torch.stack([QA.KV["co"][li].float().cpu() for li in layers])           # [La, Hkv, D]
    assert cq.shape == (len(layers), Hkv, D, D) and co.shape == (len(layers), Hkv, D), (cq.shape, co.shape)
    meta = dict(model=a.model, model_path=str(paths.MODELS[a.model]), num_layers=L, num_kv_heads=Hkv,
                layers=None if layers == list(range(L)) else layers,
                num_heads=mc.num_attention_heads, head_dim=D, rb=32, fmt="int4:32",
                kv_dyn=a.kv_dyn, local=QA.KV["dyn"]["local"], head=QA.KV["dyn"]["head"],
                calib_windows=a.calib_windows, calib_split="train",
                created=time.strftime("%Y-%m-%d %H:%M:%S"), torch=str(torch.__version__))
    out = Path(a.out) if a.out else paths.CALIB / f"flagrot_calib_{a.model}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(cq=cq.contiguous(), co=co.contiguous(),
                    signs=torch.stack([QA._signs(D, "cpu", s).float() for s in range(8)]),  # seeds 0..7 (check)
                    perm=QA._perm(D, "cpu").long(),                              # 'P': RoPE-pair interleaving
                    meta=meta), out)
    print(json.dumps(dict(out=str(out), seconds=round(time.time() - t0, 1), **meta)))


def export_static(a):
    cfg = R.canonical(dict(model=a.model, windows=0, calib_windows=a.calib_windows, kv_k=a.fmt, kv_v=a.fmt,
                           kv_rb=32, kv_flags="headsel", **({"kv_center": a.kv_center} if a.kv_center else {})))
    t0 = time.time()
    model, tok, _, t = R.build(cfg)
    mc = model.config.get_text_config()
    D = getattr(mc, "head_dim", None) or mc.hidden_size // mc.num_attention_heads
    L, Hkv = mc.num_hidden_layers, mc.num_key_value_heads
    out = {}
    layers = None
    for kind in ("k", "v"):
        m, f = QA.KV[f"{kind}mask"], QA.KV[f"{kind}full"]
        lk = sorted(m) if isinstance(m, dict) else list(range(L))                 # KV-cache layers (hybrid: a subset)
        layers = lk if layers is None else layers
        assert lk == layers, (lk, layers)
        out[f"{kind}mask"] = torch.stack([m[li].bool().cpu() for li in layers])             # [La, Hkv, D/32]
        out[f"{kind}full"] = torch.stack([f[li].bool().cpu() for li in layers])             # [La, Hkv]
        assert not (out[f"{kind}mask"].any(-1) & out[f"{kind}full"]).any(), "headsel: blocks and whole head exclusive"
    meta = dict(model=a.model, model_path=str(paths.MODELS[a.model]), num_layers=L, num_kv_heads=Hkv,
                layers=None if layers == list(range(L)) else layers,
                num_heads=mc.num_attention_heads, head_dim=D, rb=32, fmt=a.fmt, kv_flags="headsel",
                kv_center=a.kv_center, calib_windows=a.calib_windows, calib_split="train", config_id=R.config_id(cfg),
                created=time.strftime("%Y-%m-%d %H:%M:%S"), torch=str(torch.__version__))
    name = {"int4a:32": "flagrot_static_a", "int2a:32": "flagrot_static_a2"}.get(a.fmt, "flagrot_static")
    name += "c" if a.kv_center else ""
    path = Path(a.out) if a.out else paths.CALIB / f"{name}_{a.model}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(**out, meta=meta), path)
    stats = {k: round(float(v.float().mean()), 4) for k, v in out.items()}
    print(json.dumps(dict(out=str(path), seconds=round(time.time() - t0, 1), frac=stats, **meta)))


WUSH_CFG = dict(kv_k="int4a:32", kv_v="int4a:32", kv_xform="wush", kv_rb=32, kv_flags="dyn", kv_dyn="cq+f+konly")


def export_wush(a):
    """FlagRot-C = configs/intent.yaml method wush_k2.  blockflag.run.build runs the 4 calibration windows once and
    collects, in the same pass, C_q (qattention.kv_calibrate, original basis) and the WUSH sums M_K / H_K / M_V
    (xf_calibrate); xf_finalize turns them into per-(layer, KV head) block-diagonal row-vector matrices
      xk = T_K^T (k' = k xk), xq = T_K^-1 (q' = q xq, q'.k' = q.k), xv = T_V^T (v' = v xv), xvinv = T_V^-T (o = o' xvinv).
    The key-only 2x2 choice of the simulation weights the key error with xq^T C_q xq (qattention._weights, same
    einsum, same device / dtype), which is what we export as `cq`; `cq_orig` = C_q in the original basis."""
    cfg = R.canonical(dict(model=a.model, windows=0, calib_windows=a.calib_windows, **WUSH_CFG,
                           **({"kv_center": a.kv_center} if a.kv_center else {})))
    t0 = time.time()
    model, tok, _, t = R.build(cfg)
    mc = model.config.get_text_config()
    D = getattr(mc, "head_dim", None) or mc.hidden_size // mc.num_attention_heads
    L, Hkv = mc.num_hidden_layers, mc.num_key_value_heads
    xf = QA.KV["xf"]
    layers = sorted(QA.KV["cq"])                                                  # KV-cache layers (hybrid: a subset)
    assert xf is not None and sorted(xf) == layers, "kv_xform=wush did not produce a transform per KV-cache layer"
    cqx, X = [], {n: [] for n in ("k", "q", "v", "vinv")}
    for li in layers:
        C = QA.KV["cq"][li]
        cqx.append(torch.einsum("hdi,hde,hej->hij", xf[li]["q"], C, xf[li]["q"]).float().cpu())   # = _weights("k")
        for n in X:
            X[n].append(xf[li][n].float().cpu())
    cq = torch.stack(cqx)
    cq0 = torch.stack([QA.KV["cq"][li].float().cpu() for li in layers])
    co = torch.stack([QA.KV["co"][li].float().cpu() for li in layers])
    xs = {n: torch.stack(v).contiguous() for n, v in X.items()}                 # [L, Hkv, D, D] each
    # block-diagonal with 32-wide blocks (the int4a:32 scale groups): the plugin applies it per group
    blk = torch.block_diag(*[torch.ones(32, 32)] * (D // 32)).bool()
    for n, v in xs.items():
        assert float(v[..., ~blk].abs().max()) == 0.0, f"x{n} is not block-diagonal"
    meta = dict(model=a.model, model_path=str(paths.MODELS[a.model]), num_layers=L, num_kv_heads=Hkv,
                layers=None if layers == list(range(L)) else layers,
                num_heads=mc.num_attention_heads, head_dim=D, rb=32, fmt="int4a:32", kv_dyn=WUSH_CFG["kv_dyn"],
                kv_xform="wush", xform_block=32, local=QA.KV["dyn"]["local"], head=QA.KV["dyn"]["head"],
                kv_center=a.kv_center,
                calib_windows=a.calib_windows, calib_split="train", config_id=R.config_id(cfg), method="wush_k2",
                created=time.strftime("%Y-%m-%d %H:%M:%S"), torch=str(torch.__version__))
    out = Path(a.out) if a.out else paths.CALIB / \
        f"flagrot_calib_{'cc' if a.kv_center else 'c'}_{a.model}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(cq=cq.contiguous(), cq_orig=cq0.contiguous(), co=co.contiguous(),
                    xk=xs["k"], xq=xs["q"], xv=xs["v"], xvinv=xs["vinv"],
                    signs=torch.stack([QA._signs(D, "cpu", s).float() for s in range(8)]),
                    perm=QA._perm(D, "cpu").long(), meta=meta), out)
    print(json.dumps(dict(out=str(out), seconds=round(time.time() - t0, 1), **meta)))


KIVI_CFG = dict(calib_data="wikitext", calib_len=2048, eval="kl", eval_data="wikitext", eval_len=2048,
                eval_split="test", kv_dyn="att+f", kv_flags="dyn", kv_k="kivi2:32", kv_k_axis="token", kv_v="kivi2:32",
                kv_rb=32, kv_resid=128, kv_resid_mode="sliding", kv_sink=4, windows=1000)


def export_kivi(a):
    """kivi2 (FlagRot-KIVI / FlagRot-KIVI-C): the exact simulator KL configuration (config id checked against the
    run files) is built by blockflag.run.build, which runs the calibration windows and keeps, per KV-cache layer,
    the attention-weighted ('att') rule's state: QA.KV['qs'] [Hkv, nq <= 512, D] (sampled post-RoPE calibration
    queries of all query heads of the KV head), QA.KV['lz'] [Hkv, nq] (their logsumexp over the causal context),
    QA.KV['qscale'] (the attention scaling), QA.KV['co'] [Hkv, D] (o_proj column energy).  The writer's choice per
    (32-token key group | value token) uses exactly qattention._weights:
      keys    w[t, q] = min(1, exp(qscale x_t . qs_q - lz_q)), error sum_q w (qs_q . e_t)^2 (per 32-dim sub-vector /
              whole head), pooled over the group's tokens;
      values  per-token scalar sum_q w times sum_i co_i e_i^2 (KIVI) or e^T M e, M = xvinv diag(co) xvinv^T (KIVI-C):
              the per-token scalar does not change a per-token argmin, so only co / M are exported.
    kivi_c: queries are stored mapped by the base (qs' = qs xq, the same einsum as qattention._weights), the keys the
    writer sees are k' = k xk; xk / xq / xv / xvinv as in export_wush."""
    cfg = dict(KIVI_CFG, model=a.model, calib_windows=a.calib_windows)
    if a.kivi == "kivi_c":
        cfg.update(kv_xform="wush", calib_rot=a.calib_rot)
    if a.kivi in ("kivi_s", "kivi_sp8"):   # kivi2s: static selection (kv_dyn att+f+static, qattention.static_select)
        cfg.update(kv_dyn="att+f+static")
    if a.kivi == "kivi_sp8":          # kivi2p8ffs: the static choice frozen under the p8 format (kv_k = kv_v = kivi2f8:32)
        cfg.update(kv_k="kivi2f8:32", kv_v="kivi2f8:32")
    cfg = R.canonical(cfg)
    cid = R.config_id(cfg)
    t0 = time.time()
    model, tok, _, t = R.build(cfg)
    mc = model.config.get_text_config()
    D = getattr(mc, "head_dim", None) or mc.hidden_size // mc.num_attention_heads
    L, Hkv = mc.num_hidden_layers, mc.num_key_value_heads
    layers = sorted(QA.KV["qs"])
    xf = QA.KV["xf"]
    static = {}
    if a.kivi in ("kivi_s", "kivi_sp8"):
        for n in ("kmask", "kfull", "vmask", "vfull"):
            static[n] = torch.stack([QA.KV[n][li].bool().cpu() for li in layers]).contiguous()
    qs, lz, co, X, vM = [], [], [], {n: [] for n in ("k", "q", "v", "vinv")}, []
    for li in layers:
        q = QA.KV["qs"][li]
        if xf is not None:
            Xl = xf[li]
            q = torch.einsum("hnd,hde->hne", q, Xl["q"])                    # = qattention._weights (kivi_c)
            vM.append(torch.einsum("hid,hd,hjd->hij", Xl["vinv"], QA.KV["co"][li], Xl["vinv"]).float().cpu())
            for n in X:
                X[n].append(Xl[n].float().cpu())
        qs.append(q.float().cpu())
        lz.append(QA.KV["lz"][li].float().cpu())
        co.append(QA.KV["co"][li].float().cpu())
    nq = {int(x.shape[1]) for x in qs}
    assert len(nq) == 1, nq
    out = dict(qs=torch.stack(qs).contiguous(), lz=torch.stack(lz).contiguous(), co=torch.stack(co).contiguous(),
               qscale=torch.tensor(float(QA.KV["qscale"])))
    out.update(static)
    if xf is not None:
        out.update(xk=torch.stack(X["k"]).contiguous(), xq=torch.stack(X["q"]).contiguous(),
                   xv=torch.stack(X["v"]).contiguous(), xvinv=torch.stack(X["vinv"]).contiguous(),
                   vM=torch.stack(vM).contiguous())
    meta = dict(model=a.model, model_path=str(paths.MODELS[a.model]), num_layers=L, num_kv_heads=Hkv,
                layers=None if layers == list(range(L)) else layers, num_heads=mc.num_attention_heads, head_dim=D,
                rb=32, fmt="kivi2:32", kivi_fmt=cfg["kv_k"], kv_dyn="att+f", kivi=a.kivi, kv_xform=cfg["kv_xform"],
                calib_rot=cfg.get("calib_rot", ""), nq=nq.pop(), qscale=float(QA.KV["qscale"]), dyn=QA.KV["dyn"],
                calib_windows=a.calib_windows, calib_split="train", config_id=cid, kv_center="",
                created=time.strftime("%Y-%m-%d %H:%M:%S"), torch=str(torch.__version__),
                device=torch.cuda.get_device_name())
    name = {"kivi": "flagrot_kivi", "kivi_c": "flagrot_kivi_c", "kivi_s": "flagrot_kivi_s",
            "kivi_sp8": "flagrot_kivi_sp8"}[a.kivi]
    if static:
        meta["static_summary"] = QA.KV.get("static_summary")
    path = Path(a.out) if a.out else paths.CALIB / f"{name}_{a.model}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(**out, meta=meta), path)
    print(json.dumps(dict(out=str(path), seconds=round(time.time() - t0, 1),
                          **{k: v for k, v in meta.items() if k != "dyn"})))


CENTER_CFG = dict(kv_k="int2a:32", kv_v="int2a:32", kv_center="post")
PROT = {"s4w128": dict(kv_sink=4, kv_resid=128), "w0": {}}                    # = scripts/sim/sweep.py
METHOD = {"rtn": {}, "flagrot": dict(kv_rb=32, kv_flags="dyn", kv_dyn="cq+f+konly")}


def export_center(a):
    """static key centering (research kv_center = post): blockflag.run.build computes mu in one pass over the 4
    calibration windows with the BF16 cache, before any other calibration, so mu does not depend on the cache method
    (RTN config here; the run files of every method carry the same kc.mu_norm).  sel / auto: the same build with the
    simulator configuration (protection; auto: + method), QA.KV['kc'] after kc_select / _center_auto."""
    extra = {}
    if a.center == "sel":                      # sel: the choice uses the format's RTN error and the run's protection
        extra = dict(PROT[a.prot], kv_center="sel")
    elif a.center == "auto":                   # auto: + the complete cache of the method (simulator config)
        extra = dict(PROT[a.prot], **METHOD[a.method], kv_center="auto")
    cfg = R.canonical(dict(model=a.model, windows=0, calib_windows=a.calib_windows, **dict(CENTER_CFG, **extra)))
    t0 = time.time()
    model, tok, _, t = R.build(cfg)
    mc = model.config.get_text_config()
    D = getattr(mc, "head_dim", None) or mc.hidden_size // mc.num_attention_heads
    L, Hkv = mc.num_hidden_layers, mc.num_key_value_heads
    kc = QA.KV["kc"]
    auto = QA.KV.get("kc_auto")
    if kc is None:                             # auto chose no centering: store mu = 0 (the plugin turns centering off)
        assert a.center == "auto" and auto and auto["choice"] == "none", auto
        mu0 = auto["sel"]["mu_norm"]
        layers = sorted(int(k) for k in mu0)
        mu = torch.zeros(len(layers), Hkv, D)
        summ = dict(mu_norm={str(li): 0.0 for li in layers})
    else:
        layers = sorted(kc)                                                       # KV-cache layers (hybrid: a subset)
        mu = torch.stack([kc[li].float().cpu() for li in layers])                 # [La, Hkv, D] fp32
        summ = QA.kc_summary()                                                    # = the run files' "kc"
    assert mu.shape == (len(layers), Hkv, D), mu.shape
    meta = dict(model=a.model, model_path=str(paths.MODELS[a.model]), num_layers=L, num_kv_heads=Hkv,
                layers=None if layers == list(range(L)) else layers, num_heads=mc.num_attention_heads, head_dim=D,
                kv_center=a.center, calib_windows=a.calib_windows, calib_split="train", config_id=R.config_id(cfg),
                mu_norm=summ["mu_norm"], frac_centered=summ.get("frac_centered_mean"),
                prot=a.prot if a.center != "post" else None, method=a.method if a.center == "auto" else None,
                kc_auto=auto, created=time.strftime("%Y-%m-%d %H:%M:%S"), torch=str(torch.__version__),
                device=torch.cuda.get_device_name())
    name = {"post": "flagrot_center", "sel": f"flagrot_center_sel_{a.prot}",
            "auto": f"flagrot_center_auto_{a.method}_{a.prot}"}[a.center]
    out = Path(a.out) if a.out else paths.CALIB / f"{name}_{a.model}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(mu=mu.contiguous(), meta=meta), out)
    print(json.dumps(dict(out=str(out), seconds=round(time.time() - t0, 1),
                          **{k: v for k, v in meta.items() if k not in ("mu_norm", "kc_auto")},
                          choice=(auto or {}).get("choice"))))


if __name__ == "__main__":
    main()
