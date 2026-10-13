"""KV-cache baselines: agreement with vLLM's own kernels, exactness of the static
transforms, config-id stability.  The vLLM reference (tests/data/vllm_kv_reference.pt) was produced
with vLLM's Triton / CUDA kernels on an H20 (generator script not included)."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from blockflag import formats as FM
from blockflag import qattention as QA
from blockflag.qlinear import STATE
from blockflag.run import canonical, config_id, kv_bits

DATA = Path(__file__).parent / "data"
REF = torch.load(DATA / "vllm_kv_reference.pt")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
TQ = {"turboquant_k8v4": ("tqk8", "tqv4"), "turboquant_4bit_nc": ("tqk4n", "tqv4"),
      "turboquant_k3v4_nc": ("tqk3n", "tqv4"), "turboquant_3bit_nc": ("tqk3n", "tqv3")}


# ------------------------------------------------------------------ config ids
LEGACY = [  # ids computed with the code before kv_xform / kv_skip existed
    ({"model": "llama8b", "kv_k": "int4:32", "kv_v": "int4:32", "windows": 8}, "7466b68c8056"),
    ({"model": "llama8b", "kv_k": "int4:32", "kv_v": "int4:32", "kv_rb": 32, "kv_flags": "dyn",
      "kv_dyn": "cq+f+S+FS+P", "windows": 8, "calib_windows": 4}, "2026abeadd18"),
    ({"model": "qwen8b", "kv_k": "int4:128", "kv_v": "int4:128", "kv_rb": -1, "kv_flags": "all"}, "e7a67b2fd54c"),
    ({"model": "qwen8b", "kv_k": "nvfp4", "kv_v": "nvfp4", "kv_rb": 16, "kv_flags": "greedy"}, "2c3446b75713"),
    ({"model": "llama8b", "kv_k": "int2a:32", "kv_v": "int2a:32", "kv_k_axis": "token", "kv_resid": 128}, "f3877c032657"),
    ({"model": "llama8b", "w": "int4:128", "a": "fp8:128", "rb": 32, "flags": "greedy", "wq": "gptq"}, "e219f148f808"),
    ({"model": "qwen8b", "kv_k": "fp8t", "kv_v": "fp8t", "eval": "lm", "lm_limit": 1000}, "279a0d5b25ae"),
]


@pytest.mark.parametrize("cfg,cid", LEGACY)
def test_legacy_config_ids_unchanged(cfg, cid):
    assert config_id(cfg) == cid


def test_stored_result_configs_keep_their_ids():
    for r in json.loads((DATA / "legacy_config_ids.json").read_text()):   # canonical configs of stored results
        assert config_id(r["config"]) == r["id"]


def test_new_keys_canonicalization():
    base = {"model": "m", "kv_k": "int4:32", "kv_v": "int4:32"}
    assert config_id({**base, "kv_xform": "", "kv_skip": 0}) == config_id(base)
    assert config_id({"model": "m", "kv_xform": "wush", "kv_skip": 2}) == config_id({"model": "m"})   # BF16 cache
    assert config_id({**base, "kv_xform": "smooth"}) == config_id({**base, "kv_xform": "smooth:0.5"})
    assert config_id({**base, "kv_xform": "wush"}) != config_id(base)
    assert config_id({**base, "kv_skip": 2}) != config_id(base)
    assert canonical({**base, "kv_xform": "wush"})["calib_windows"] == 4
    assert canonical(base)["calib_windows"] == 0


def test_bits_accounting():
    assert kv_bits({"model": "m", "kv_k": "int4:32", "kv_v": "int4:32"}, 128, 32)["mean"] == 4.5
    assert kv_bits({"model": "m", "kv_k": "int4ph", "kv_v": "int4ph"}, 128, 32)["mean"] == 4.25
    b = kv_bits({"model": "m", "kv_k": "tqk4n", "kv_v": "tqv4"}, 128, 32)
    assert (b["k"], b["v"]) == (4.125, 4.25)
    assert kv_bits({"model": "m", "kv_k": "tqk4n", "kv_v": "tqv4", "kv_skip": 2}, 128, 32)["k"] == (4.125 * 28 + 64) / 32
    d = kv_bits({"model": "m", "kv_k": "int4:32", "kv_v": "int4:32", "kv_rb": 32, "kv_flags": "dyn",
                 "kv_dyn": "cq+f+S+FS+P"}, 128, 32)
    assert d["side"] == 2 / 32 + 2 / 128
    assert FM.bits("nvfp4_46", 128) == FM.bits("nvfp4", 128) == 4.5


# ------------------------------------------------------------------ agreement with vLLM
@pytest.mark.parametrize("D", [64, 128])
def test_vllm_rht_signs_and_hadacore_bit_exact(D):
    assert torch.equal(FM.vllm_rht_signs(D), REF[f"rht_signs_{D}"])
    assert torch.equal(FM.hadacore(REF[f"hadacore_in_{D}"].float()), REF[f"hadacore_out_{D}"].float())


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("dev", DEVICES)
def test_int4_per_token_head_matches_vllm(D, dev):
    for kind in ("k", "v"):
        x = REF[f"in_{kind}_{D}"].to(dev)
        _, yh = FM.int4ph_rotated(x)                                  # dequantized cache entry, rotated basis
        ref = REF[f"int4ph_{kind}rot_{D}"].to(dev)
        same = (yh == ref).all(-1)
        if dev == "cuda":                                             # Triton port of the store kernel: bit-exact
            assert bool(same.all())
        else:                                                         # IEEE division: rare flips at rounding ties
            assert same.float().mean() > 0.9
        rec, rref = FM.fake(x, "int4ph"), REF[f"int4ph_{kind}_{D}"].to(dev)
        # vLLM's inverse RHT runs in BF16 (hadacore): agreement to BF16 rounding of the reconstruction
        tol = 1e-2 if dev == "cuda" else 3e-2
        assert ((rec - rref).abs().amax(-1) / rref.abs().amax(-1)).max() < tol


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("preset", list(TQ))
@pytest.mark.parametrize("dev", DEVICES)
def test_turboquant_matches_vllm(D, preset, dev):
    ks, vs = TQ[preset]
    k, v = REF[f"in_k_{D}"].to(dev), REF[f"in_v_{D}"].to(dev)
    if ks == "tqk8":
        assert torch.equal(FM.fake(k, ks), REF[f"{preset}_k_{D}"].to(dev))
    else:
        nb = int(ks[3])
        assert torch.equal(FM.tq_tables(D, nb)[0], REF[f"{preset}_centroids_{D}"].sort().values)
        # vLLM's dequant kernel writes FP16: norm * centroid vector in the rotated basis, compared after FP16 rounding
        assert torch.equal(FM.tq_key_rotated(k, nb, ks.endswith("n")).half(), REF[f"{preset}_krot_{D}"].to(dev).half())
        rk = REF[f"{preset}_k_{D}"].to(dev)
        assert ((FM.fake(k, ks) - rk).abs().max() / rk.abs().max()) < 1e-3
    ov, rv = FM.fake(v, vs).half(), REF[f"{preset}_v_{D}"].to(dev).half()
    if dev == "cuda":
        assert torch.equal(ov, rv)
    else:
        assert (ov == rv).float().mean() > 0.995


def test_nvfp4_4over6_matches_vllm_kernel_transcription():
    x, ks = REF["nvfp4_in"], REF["nvfp4_k_scale"]
    assert torch.allclose(FM.fake_nvfp4_46(x, G=ks), REF["nvfp4_46_out"], rtol=1e-6, atol=0)
    # vLLM's test_nvfp4_4over6_selects_lower_error_scale: [6, 4.5 x 15] with scale 1
    src = torch.tensor([[6.0] + [4.5] * 15])
    e46 = (FM.fake_nvfp4_46(src, G=1.0) - src).square().mean()
    G6 = torch.tensor(1.0)
    b = src.reshape(1, 1, 16)
    sf = (b.abs().amax(-1, keepdim=True) / 6 / G6).to(torch.float8_e4m3fn).float()
    e6 = (FM.e2m1(b / sf) * sf - b).square().mean()
    assert e46 < e6


def test_nvfp4_46_never_worse_than_its_scale6_candidate():
    x = torch.randn(64, 256) * 3
    G = x.abs().amax() / (256 * 6)
    r = FM.fake_nvfp4_46(x, G=G)
    b = x.reshape(64, 16, 16)
    sf = (b.abs().amax(-1, keepdim=True) / 6 / G).clamp_max(448).to(torch.float8_e4m3fn).float()
    r6 = (FM.e2m1(b / (sf * G)) * sf * G).reshape(x.shape)
    assert ((r - x).reshape(64, 16, 16).square().sum(-1) <= (r6 - x).reshape(64, 16, 16).square().sum(-1) + 1e-9).all()


# ------------------------------------------------------------------ exactness of the transforms
def test_internal_rotations_are_exact():
    x = torch.randn(7, 128, dtype=torch.float64)
    s = FM.vllm_rht_signs(128).double()
    assert torch.allclose(FM.wht(FM.wht(x * s)) * s / 128, x, atol=1e-12)          # int4ph: s * H(H(s x)) / D
    H = FM._tq_hadamard(128).double()
    assert torch.allclose(x @ H @ H, x, atol=1e-6)                                 # TurboQuant: H orthonormal
    assert abs(FM.hadacore_gain(64) - 1) < 1e-7 and abs(FM.hadacore_gain(128) - 0.99989) < 1e-5


def test_wush_transform():
    g = torch.Generator().manual_seed(0)
    A = torch.randn(3, 32, 32, generator=g, dtype=torch.float64)
    B = torch.randn(3, 32, 32, generator=g, dtype=torch.float64)
    M, Hs = A @ A.transpose(-1, -2), B @ B.transpose(-1, -2)
    T = QA.wush(M, Hs)
    assert torch.linalg.matrix_rank(T).min() == 32
    # with isotropic statistics WUSH is a scaled orthogonal (Hadamard-like) transform
    Ti = QA.wush(torch.eye(16, dtype=torch.float64), torch.eye(16, dtype=torch.float64))
    G_ = Ti @ Ti.T
    assert torch.allclose(G_, G_[0, 0] * torch.eye(16, dtype=torch.float64), atol=1e-10)


def _attn(q_, k_, v_, n_rep, **kv):
    mod = SimpleNamespace(layer_idx=0, num_key_value_groups=n_rep)
    old = {n: QA.KV[n] for n in kv}
    QA.KV.update(**kv)
    STATE["quant"] = True
    try:
        return QA.blockflag_attention(mod, q_, k_, v_, None, q_.shape[-1] ** -0.5)[0]
    finally:
        STATE["quant"] = False
        QA.KV.update(**old)


@pytest.mark.parametrize("mode", ["wush", "smooth"])
@pytest.mark.parametrize("flags", ["none", "all"])
def test_static_transforms_are_exact_without_quantization(mode, flags):
    torch.manual_seed(0)
    B, H, Hkv, S, D = 1, 4, 2, 23, 64
    q, k, v = torch.randn(B, H, S, D), torch.randn(B, Hkv, S, D), torch.randn(B, Hkv, S, D)
    k[..., 5] *= 20
    ref = _attn(q, k, v, H // Hkv, k="bf16", v="bf16")
    if mode == "wush":
        Mk = torch.einsum("bhsd,bhse->hde", k.double(), k.double())
        Hq = 2 * torch.einsum("bhrsd,bhrse->hde", *(2 * [q.double().reshape(B, Hkv, 2, S, D)]))
        Mv = torch.einsum("bhsd,bhse->hde", v.double(), v.double())
        Tk, Tv = QA._blockdiag_wush(Mk, Hq, 32), QA._blockdiag_wush(Mv, torch.eye(D, dtype=torch.float64).expand(Hkv, D, D), D)
        X = {"k": Tk.transpose(-1, -2), "q": torch.linalg.inv(Tk), "v": Tv.transpose(-1, -2),
             "vinv": torch.linalg.inv(Tv).transpose(-1, -2)}
    else:
        lam = k.abs().amax((0, 2)).double().pow(0.5)
        X = {"k": torch.diag_embed(1 / lam), "q": torch.diag_embed(lam)}
    X = {n: t.float() for n, t in X.items()}
    mask = {0: torch.ones(Hkv, D // 16, dtype=torch.bool)} if flags == "all" else None
    out = _attn(q, k, v, H // Hkv, k="id", v="id", rb=16, xf={0: X}, kmask=mask, vmask=mask)
    assert torch.allclose(out, ref, atol=2e-5)
    q4 = _attn(q, k, v, H // Hkv, k="int4:32", v="int4:32", rb=16, xf={0: X}, kmask=mask, vmask=mask)
    assert not torch.allclose(q4, ref, atol=1e-4)                     # the transform does reach the quantizer


def test_kv_skip_keeps_boundary_layers_bf16():
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 2, 9, 32), torch.randn(1, 2, 9, 32), torch.randn(1, 2, 9, 32)
    ref = _attn(q, k, v, 1, k="bf16", v="bf16")
    assert torch.allclose(_attn(q, k, v, 1, k="int4:32", v="int4:32", skip=1, nl=4), ref)   # layer 0 skipped
    assert not torch.allclose(_attn(q, k, v, 1, k="int4:32", v="int4:32", skip=0, nl=4), ref)


@pytest.mark.parametrize("spec", ["int4ph", "tqk4n", "tqk3n", "tqv4", "tqv3", "nvfp4_46", "int4p96", "int4p92:32"])
def test_baseline_specs_quantize_reasonably(spec):
    x = torch.randn(2, 3, 50, 128) * 2
    r = FM.fake(x, spec)
    rel = ((r - x).square().sum() / x.square().sum()).item()
    assert r.shape == x.shape and 1e-5 < rel < (0.06 if "3" in spec else 0.02)
