"""Codec parity: vllm_flagrot.codec vs the research encoder blockflag.qattention.dyn_quant (same env, same inputs),
for the locked codebook "cq+f+S+Q5+FS+R5" and the earlier "cq+f+S+FS+P".

run (vLLM env, repo root; the vLLM env has no pytest, the file also runs as a script):
  CUDA_VISIBLE_DEVICES=6 python integrations/vllm_flagrot/tests/test_codec.py
Uses the exported calibration file when present (calib/flagrot_calib_llama8b.pt), random weights otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

try:
    import pytest
except ImportError:
    pytest = None

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integrations/vllm_flagrot"))

from blockflag import qattention as QA  # noqa: E402
from vllm_flagrot import codec as C  # noqa: E402

DEV = "cuda"
CALIB = ROOT / "calib/flagrot_calib_llama8b.pt"
CODEBOOKS = ["cq+f+S+Q5+FS+R5", "cq+f+S+FS+P"]


def _inputs(T=512, H=8, D=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, H, D, generator=g)
    x = x * (1 + 6 * (torch.rand(1, H, D, generator=g) > 0.95))           # outlier channels
    x[:, :, :8] += 4 * torch.randn(1, H, 8, generator=g)                    # RoPE-like low-frequency offsets
    return x.to(torch.bfloat16).to(DEV)


def _weights(H=8, D=128, li=3):
    if CALIB.exists():
        cal = torch.load(CALIB, weights_only=True)
        return cal["cq"][li].to(DEV), cal["co"][li].to(DEV)
    g = torch.Generator().manual_seed(1)
    a = torch.randn(H, D, D, generator=g)
    return (a @ a.transpose(1, 2)).to(DEV), torch.rand(H, D, generator=g).to(DEV)


def _param(f):
    if pytest is None:
        return f
    return pytest.mark.parametrize("kind", ["k", "v"])(pytest.mark.parametrize("kv_dyn", CODEBOOKS)(f))


@_param
def test_matches_dyn_quant(kv_dyn, kind):
    x = _inputs()
    T, H, D = x.shape
    cq, co = _weights(H, D)
    cb = C.Codebook.parse(kv_dyn)
    for s in range(8):
        assert torch.equal(C.research_signs(D, s), QA._signs(D, "cpu", s).float())
    QA.KV.update(dyn=dict(rule="cq", local=list(cb.local), head=list(cb.head), only=None), cq={0: cq}, co={0: co},
                 xf=None, dynstat={})
    ref = QA.dyn_quant(x.permute(1, 0, 2)[None], "int4:32", 32, kind, 0)[0].permute(1, 0, 2)   # [T, H, D] fp32
    cod = C.FlagRotCodec(cb, D, DEV)
    assert torch.equal(cod.perm.cpu(), QA._perm(D, "cpu"))
    W = C.make_weights("mat", cq, cod.perm) if kind == "k" else C.make_weights("diag", co, cod.perm)
    codes, scales, side, rec = cod.encode(x, W, return_recon=True)
    same = (rec == ref).all(-1).float().mean().item()
    assert same > 0.999, f"fp32 reconstruction identical for only {same:.4%} of (token, head)"
    assert int(side.max()) < (1 << cb.side_bits(D))
    # cache round trip: fp16 scales -> relative deviation bounded by fp16 rounding of the scale
    cell = cod.pack(codes, scales, side)
    nc, ns = D // 2, 2 * (D // 32)
    dec = cod.decode_kind(cell[..., :nc], cell[..., nc:nc + ns], cell[..., nc + ns:])
    rel = ((dec - rec).norm() / rec.norm()).item()
    assert rel < 1e-3, rel
    modes = torch.bincount((side >> cod.hshift).flatten(), minlength=cb.n_head).float() / side.numel()
    loc = torch.bincount(((side[..., None] >> cod.shifts) & cod.lmask)[(side >> cod.hshift) == 0].flatten(),
                         minlength=cb.n_local).float()
    print(f"{kv_dyn} {kind}: identical {same:.4f}, round-trip rel {rel:.2e}, side bits {cb.side_bits(D)}, "
          f"head {[round(v, 3) for v in modes.tolist()]}, per-vector {[round(v, 3) for v in (loc / loc.sum()).tolist()]}")


def test_cells_roundtrip():
    x, y = _inputs(seed=2), _inputs(seed=3)
    cq, co = _weights()
    cod = C.FlagRotCodec(C.Codebook.parse(C.DEFAULT_KV_DYN), 128, DEV)
    Wk, Wv = C.make_weights("mat", cq, cod.perm), C.make_weights("diag", co, cod.perm)
    cells = cod.encode_cells(x, y, Wk, Wv)
    assert cells.shape[-1] == C.cell_bytes(128) == 148 and cells.dtype == torch.uint8
    k, v = cod.decode_cells(cells, torch.float32)
    rk = cod.encode(x, Wk, True)[3]
    rv = cod.encode(y, Wv, True)[3]
    assert ((k - rk).norm() / rk.norm()) < 1e-3 and ((v - rv).norm() / rv.norm()) < 1e-3


if __name__ == "__main__":
    for cbk in CODEBOOKS:
        for kd in ("k", "v"):
            test_matches_dyn_quant(cbk, kd)
    test_cells_roundtrip()
    print("OK")
