"""Unit tests (CPU unless noted): python -m pytest -q tests"""
import torch
import pytest

from blockflag import formats as FM
from blockflag.hadamard import fwht, pattern_mask, sel_fwht, sylvester
from blockflag.qlinear import STATE, SelLinear, greedy_masks, share_key
from blockflag.run import canonical, config_id

torch.manual_seed(0)


@pytest.mark.parametrize("rb", [2, 8, 16, 32, 64, 128])
def test_fwht_matches_matrix_and_is_involutory(rb):
    x = torch.randn(5, 4 * rb, dtype=torch.float64)
    H = sylvester(rb, dtype=torch.float64)
    ref = (x.view(5, 4, rb) @ H).view(5, -1)
    assert torch.allclose(fwht(x, rb), ref, atol=1e-12)
    assert torch.allclose(fwht(fwht(x, rb), rb), x, atol=1e-12)


def test_selective_rotation_preserves_gemm():
    x, W = torch.randn(7, 256, dtype=torch.float64), torch.randn(9, 256, dtype=torch.float64)
    m = torch.rand(256 // 16) < 0.5
    assert torch.allclose(sel_fwht(x, m, 16) @ sel_fwht(W, m, 16).t(), x @ W.t(), atol=1e-10)
    assert torch.equal(sel_fwht(x, torch.zeros_like(m), 16), x)


def test_pattern_mask():
    assert pattern_mask(8, 4, 1).tolist() == [False, True, False, False] * 2


def test_e2m1_grid_and_ties():
    t = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, -0.25, -2.5, 0.3])
    assert FM.e2m1(t).tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, -0.0, -2.0, 0.5]


@pytest.mark.parametrize("spec,bound", [("fp8:128", 2e-3), ("fp8:row", 2e-3), ("fp8t", 2e-3), ("mxfp4", 3e-2),
                                        ("nvfp4", 2e-2), ("int4:32", 2e-2), ("int8:128", 1e-4)])
def test_formats_error_and_codes(spec, bound):
    x = torch.randn(64, 256) * 3
    c, s = FM.quantize(x, spec)
    q = FM.dequant(c, s)
    assert torch.equal(c.to(torch.bfloat16).float(), c)          # codes are exact in BF16
    assert ((q - x).square().sum() / x.square().sum()).item() < bound
    assert torch.allclose(q, FM.fake(x, spec))


def test_fp8_block_weight_scale_layout():
    W = torch.randn(300, 256)
    c, s = FM.quantize_weight(W, "fp8:128")
    assert s.shape == (300, 2, 1)
    assert torch.allclose(s[0], s[127]) and not torch.allclose(s[0], s[128])
    assert ((FM.dequant(c, s) - W).abs() <= W.abs().amax() / 448 * 32).all()


def test_mxfp4_scale_is_power_of_two():
    _, s = FM.quantize(torch.randn(4, 64), "mxfp4")
    assert torch.equal(torch.log2(s), torch.round(torch.log2(s)))


def test_sellinear_bf16_is_near_exact_with_any_mask():
    lin = torch.nn.Linear(256, 32, bias=False)
    m = SelLinear(lin, "bf16", "bf16", 32)
    m.finalize(torch.rand(8) < 0.5)                  # rotated weight is stored in BF16 (as a deployment would)
    x = torch.randn(3, 5, 256)
    STATE["quant"] = True
    try:
        y, ref = m(x), lin(x)
        assert ((y - ref).norm() / ref.norm()).item() < 1e-2
    finally:
        STATE["quant"] = False


def test_greedy_masks_share_qkv():
    mods = {}
    for pn in ("q_proj", "k_proj", "v_proj"):
        lin = torch.nn.Linear(256, 16, bias=False)
        mods[f"layers.0.self_attn.{pn}"] = m = SelLinear(lin, "fp8:128", "fp8:128", 64)
        m.acc = torch.tensor([[1.0, 1.0], [0.5, 2.0], [2.0, 0.1], [1.5, 1.5]], dtype=torch.float64)   # [per_unit + 2, J]
    out = greedy_masks(mods)
    assert share_key("layers.0.self_attn.k_proj") == "layers.0.qkv"
    assert out["layers.0.self_attn.q_proj"].tolist() == [True, False, False, True]


def test_canonical_dedupes_irrelevant_fields():
    a = config_id({"model": "m", "w": "fp8:128", "a": "fp8:128", "rb": 64, "flags": "none", "calib_windows": 8})
    b = config_id({"model": "m", "w": "fp8:128", "a": "fp8:128", "rb": 16, "flags": "none"})
    assert a == b
    c = canonical({"model": "m", "kv_k": "nvfp4", "kv_v": "nvfp4", "rb": 16, "kv_flags": "all"})
    assert c["kv_rb"] == 16 and c["flags"] == "none" and c["calib_windows"] == 0


def test_greedy_margin_is_conservative():
    lin = torch.nn.Linear(256, 16, bias=False)
    m = SelLinear(lin, "fp8:128", "fp8:128", 64)
    m.acc = torch.tensor([[1.0, 1.0], [0.9, 0.5], [0.97, 2.0], [1.0, 1.0]], dtype=torch.float64)   # gains: 10 %, 3 % | 50 %, -
    name = "layers.0.mlp.down_proj"
    assert greedy_masks({name: m})[name].tolist() == [True, True, True, False]
    assert greedy_masks({name: m}, margin=0.05)[name].tolist() == [True, False, True, False]


def test_config_id_is_stable_for_legacy_configs():
    legacy = {"model": "m", "w": "int4:128", "a": "fp8:128", "rb": 32, "flags": "greedy"}
    assert config_id(legacy) == config_id({**legacy, "wq": "rtn", "gptq_windows": 16})
    assert config_id(legacy) != config_id({**legacy, "wq": "gptq"})


@pytest.mark.parametrize("spec", ["nvfp4", "fp8:128", "int4:128", "mxfp4"])
def test_gptq_is_finite_when_updates_exceed_initial_amax(spec):
    from blockflag.gptq import gptq_quantize
    g = torch.Generator().manual_seed(1)
    X = torch.randn(512, 256, generator=g)
    X[:, :8] *= 30                                  # strongly correlated / outlier channels -> large GPTQ updates
    W = torch.randn(128, 256, generator=g)
    c, s = gptq_quantize(W, X.t() @ X, spec)
    assert torch.isfinite(c).all() and torch.isfinite(s).all()


def test_protected_vectors_flags_massive_channels():
    from blockflag.qlinear import protected_vectors
    m = SelLinear(torch.nn.Linear(256, 8, bias=False), "int4:128", "int4:128", 16)
    m.amax = torch.ones(256)
    m.amax[37] = 5000.0                                  # a massive-activation channel in vector 2
    assert protected_vectors(m, 500).nonzero().flatten().tolist() == [2]
    assert config_id({"model": "m", "w": "int4:128", "a": "bf16"}) == config_id({"model": "m", "w": "int4:128", "a": "bf16", "protect": 500})


def test_paired_acc_and_new_keys_are_id_stable():
    from blockflag.stats import paired_acc
    a = {"lm": {"t1": {"per_doc": [1, 1, 1, 0] * 50}, "t2": {"per_doc": [1, 0] * 100}, "mean": {}}}
    b = {"lm": {"t1": {"per_doc": [1, 0, 1, 0] * 50}, "t2": {"per_doc": [1, 0] * 100}, "mean": {}}}
    c = paired_acc(a, b, n_boot=500)
    assert abs(c["delta"] - 0.125) < 1e-9 and c["lo"] > 0
    base = {"model": "m", "w": "int4:128", "a": "fp8:128", "rb": 32, "flags": "greedy"}
    assert config_id(base) == config_id({**base, "proxy": "rtn", "joint_calib": True, "eval": "kl"})
    assert config_id(base) != config_id({**base, "eval": "lm"})


def test_asym_int_and_token_axis_kv_quant():
    from blockflag.qattention import kv_quant
    x = torch.rand(4, 64) * 3 + 1                    # all positive: asymmetric uses the whole range
    e_sym = (FM.fake(x, "int4:32") - x).square().mean()
    e_asym = (FM.fake(x, "int4a:32") - x).square().mean()
    assert e_asym < e_sym
    t = torch.randn(1, 2, 70, 16)
    t[..., 3] *= 50                                   # an outlier channel: per-channel grouping isolates it
    e_head = (kv_quant(t, "int4a:16", None, 16, "head") - t)[..., [0, 1, 2]].square().mean()
    e_tok = (kv_quant(t, "int4a:32", None, 16, "token") - t)[..., [0, 1, 2]].square().mean()
    assert e_tok < e_head and kv_quant(t, "int4a:32", None, 16, "token").shape == t.shape


def test_layer_masks_follow_the_all_pattern():
    from blockflag.qlinear import layer_masks
    lin = torch.nn.Linear(256, 16, bias=False)
    m = SelLinear(lin, "fp8:128", "fp8:128", 64)
    m.acc = torch.tensor([[1.0, 1.0], [0.5, 2.0], [2.0, 0.1], [0.8, 0.9]], dtype=torch.float64)   # all < none
    assert layer_masks({"layers.0.mlp.down_proj": m})["layers.0.mlp.down_proj"].all()
    m.acc[-1] = torch.tensor([1.5, 1.5], dtype=torch.float64)
    assert not layer_masks({"layers.0.mlp.down_proj": m})["layers.0.mlp.down_proj"].any()


def test_chunked_attention_matches_unchunked():
    from types import SimpleNamespace
    from blockflag import qattention as QA
    torch.manual_seed(0)
    B, H, Hkv, S, D = 1, 4, 2, 37, 32
    q, k, v = torch.randn(B, H, S, D), torch.randn(B, Hkv, S, D), torch.randn(B, Hkv, S, D)
    mod = SimpleNamespace(layer_idx=0, num_key_value_groups=H // Hkv)
    QA.KV.update(k="int4a:16", v="int4a:16", rb=16, kmask={0: torch.tensor([[True, False], [False, True]])},
                 vmask={0: torch.tensor([[True, True], [False, False]])}, resid=5, sink=2, k_axis="token", calib_on=False)
    STATE["quant"] = True
    try:
        ref, _ = QA.blockflag_attention(mod, q, k, v, None, D ** -0.5)
        old = dict(QA.CHUNK)
        QA.CHUNK.update(limit=1, elems=H * S * 4)          # 4 query rows per block
        out, _ = QA.blockflag_attention(mod, q, k, v, None, D ** -0.5)
        QA.CHUNK.update(old)
    finally:
        STATE["quant"] = False
        QA.KV.update(kmask=None, vmask=None, resid=0, sink=0, k_axis="head", k="bf16", v="bf16")
    assert torch.allclose(out, ref, atol=1e-5)
