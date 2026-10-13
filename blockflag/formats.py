"""Quantization formats along the last dim (the GEMM contraction dim).

Every quantizer returns (codes, scales): codes have the input's shape and hold the element values of the format
(exactly representable in BF16: E4M3, E2M1 and small integers), scales hold one FP32 factor per group
([..., K/g, 1]).  dequant(codes, scales) = codes * scales.

spec          elements  group along K            scale
fp8:<g>       E4M3      g (default 128)          absmax / 448, FP32                (DeepGEMM 1 x 128)
fp8e:<g>      E4M3      g (default 128)          power of two >= absmax / 448 (UE8M0; vLLM's DeepGEMM scales on Hopper)
fp8:row       E4M3      whole row                absmax / 448, FP32                (torch._scaled_mm rowwise)
fp8t          E4M3      whole tensor             absmax / 448, FP32                (vLLM default FP8 KV cache)
mxfp4[:g]     E2M1      32                       E8M0: 2^(floor(log2 absmax) - 2)  (OCP MX v1.0; clamp to +-6)
nvfp4[:g]     E2M1      16                       E4M3(absmax / 6 / G) * G, G = absmax_tensor / (448 * 6)
e2m1f[:g]     E2M1      16                       absmax / 6, FP32 (NVFP4 without scale quantization; analysis only)
nvint4[:g]    int, 15 levels k * 6/7, |k| <= 7   NVFP4's scale (E4M3 * G) with a uniform grid of the same range [-6, 6]
                                                 instead of E2M1 (analysis only; with an FP32 scale use int4:16)
int<b>:<g>    int b     g (default 128)          absmax / (2^(b-1) - 1)
int<b>a:<g>   uint b    g (default 32)           asymmetric min/max with zero point (KIVI-style; fake-quant only)
int4az:<g>    uint 4    g (default 32)           int4a with the range widened to include 0 (zero point in [0, 15]);
                                                 identical to int4a for every group with min <= 0 <= max
kivi<b>f8:<g> uint b    g (default 32)           KIVI min/max with FP8 E4M3 scale + FP8 minimum (b + 16/g bits; opt-in)
int<b>ah:<g>  uint b    g (default 32)           b = 2, 3 (opt-in): int4ah with 2^b levels, b + 16/g bits per value
int4ah:<g>    uint 4    g (default 32)           the H20 kernels' asymmetric cache): int4az codes and
                                                 zero point; one FP16 scale word whose 4 low bits are the zero point
                                                 (the nearest such FP16 to the scale: 6 free mantissa bits)
bf16          -         -                        no quantization
Weights: 'fp8:<g>' is a g x g block (DeepGEMM 128 x 128); all other weight specs group along K per output row.

KV-cache baselines (fake quantization of whole head vectors [..., D], reconstruction in the original basis).  They are selected as kv_k / kv_v specs:
nvfp4_46      E2M1      16                       Four Over Six (arXiv 2512.02010; vLLM 'nvfp4_4over6'): per block the
                                                 E4M3 scale mapping the block max to 6 or to 4, whichever has the lower
                                                 squared error; tensor scale absmax / (256 * 6) (paper)
int4ph        uint4     whole vector             vLLM 'int4_per_token_head': randomized Hadamard H diag(s) (vLLM's
                                                 fixed signs, unnormalized, rounded to BF16), asymmetric min/max with
                                                 the 4-bit zero point stored in the low mantissa bits of the FP32 scale
tqk<b>[n]     b-bit idx whole vector             vLLM TurboQuant keys (b = 3, 4): x / ||x||, orthonormal Hadamard,
                                                 Lloyd-Max centroids of N(0, 1/D), FP16 norm; n = norm correction
tqk8          E4M3      -                        vLLM TurboQuant 'k8' keys: plain cast to E4M3, no scale
tqv<b>        uint b    whole vector             vLLM TurboQuant values (b = 3, 4): min/max, FP16 scale and min
int<b>p<k>[:g] uint b   g (default whole vector) OSCAR-style percentile-clipped affine quantizer (WUSH-KV eq. 7):
                                                 range clipped at the k/100 quantile of |x| (e.g. int4p96)
int<b>a[z]c<r>:<g> uint b g (default 32)         QuaRot / SpinQuant ActQuantizer (fake_quant/quant_utils.py), asymmetric
                                                 min/max scaled by the clip ratio r/100 (e.g. int4ac95:128 = QuaRot's
                                                 V cache, clip 0.95); 'z' widens the range to include 0 first (the
                                                 K-cache path of QKRotationWrapper: find_params with groupsize -1)
kivi<b>:<g>   uint b    g (default 32)           KIVI's quantizer (jy-yuan/KIVI quant/new_pack.py): s = (max - min) / (2^b - 1),
                                                 code = round(clamp((x - min) / s, 0, 2^b - 1)), x_hat = code s + min
                                                 (the minimum itself is stored, no integer zero point; FP32 here)
id            -         -                        identity, but counts as a quantized cache (tests of exactness)
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

import torch

E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_MID = (E2M1_GRID[1:] + E2M1_GRID[:-1]) / 2
DEFAULT_GROUP = {"fp8e": 128, "fp8": 128, "mxfp4": 32, "nvfp4": 16, "int4": 128, "int8": 128, "e2m1f": 16, "nvint4": 16}


def parse(spec: str) -> tuple[str, str]:
    name, _, arg = spec.partition(":")
    return name, arg


def group_of(spec: str, K: int) -> int:
    """scale-group length along K (K for row / tensor scales and for bf16)."""
    name, arg = parse(spec)
    if name in ("bf16", "fp8t", "id") or arg == "row" or (_whole(name) and not arg):
        return K
    if name == "nvfp4_46":
        return 16
    if _PCLIP.fullmatch(name) or _QUEST.fullmatch(name):
        return int(arg)
    if _ACLIP.fullmatch(name) or _KIVI.fullmatch(name):
        return int(arg) if arg else 32
    if name.startswith("int") and name.endswith("a") or name in _ASYM_DEP or _AH.fullmatch(name) or _KF8.fullmatch(name):
        return int(arg) if arg else 32
    return int(arg) if arg else DEFAULT_GROUP[name]


def fake_asym(v: torch.Tensor, bits: int, g: int) -> torch.Tensor:
    """asymmetric uniform quantization per group: q = clamp(round(x/s) + z, 0, 2^b - 1), s = (max - min)/(2^b - 1)."""
    b = _grp(v.float(), g)
    mn, mx = b.amin(-1, keepdim=True), b.amax(-1, keepdim=True)
    lv = 2 ** bits - 1
    s = ((mx - mn) / lv).clamp_min(1e-12)
    z = torch.round(-mn / s)
    q = (torch.round(b / s) + z).clamp(0, lv)
    return ((q - z) * s).reshape(v.shape)


_ASYM_DEP = ("int4az", "int4ah")
_AH = re.compile(r"int([23])ah")
_KF8 = re.compile(r"kivi(\d)f8")   # opt-in KIVI with FP8 (e4m3) scale + FP8 minimum (packed KIVI family)
_ACLIP = re.compile(r"int(\d)a(z?)c(\d+)")
_KIVI = re.compile(r"kivi(\d)")


def fake_kivi(v: torch.Tensor, bits: int, g: int) -> torch.Tensor:
    """KIVI quant_and_pack_{k,v}cache / triton_quantize_and_pack_along_last_dim + the dequantization of its kernels
    (code * scale + mn), groups of g along the last dim."""
    b = _grp(v.float(), g)
    mn, mx = b.amin(-1, keepdim=True), b.amax(-1, keepdim=True)
    lv = 2 ** bits - 1
    s = (mx - mn) / lv
    # a constant group (s = 0) is 0/0 in the official code too; it occurs on Qwen2.5-7B, whose key bias leaves some
    # channels constant over a 32-token group: such a group is reproduced exactly (q = 0, x_hat = mn)
    q = ((b - mn) / torch.where(s == 0, torch.ones_like(s), s)).clamp(0, lv).round()
    return (q * s + mn).reshape(v.shape)


def fake_asym_clip(v: torch.Tensor, bits: int, g: int, clip: float, with_zero: bool) -> torch.Tensor:
    """QuaRot / SpinQuant ActQuantizer.find_params + AsymSTEQuantize (FP32 here; their code runs on BF16 inputs):
    xmin = min(x) * clip, xmax = max(x) * clip (with_zero: min(min x, 0) and max(max x, 0) before the clip, the K-cache
    path), an all-zero group gets [-1, 1], scale = (xmax - xmin) / (2^b - 1), zero = round(-xmin / scale),
    q = clamp(round(x / scale) + zero, 0, 2^b - 1), x_hat = scale * (q - zero)."""
    b = _grp(v.float(), g)
    mn, mx = b.amin(-1, keepdim=True), b.amax(-1, keepdim=True)
    if with_zero:
        mn, mx = mn.clamp_max(0.0), mx.clamp_min(0.0)
    mn, mx = mn * clip, mx * clip
    zz = (mn == 0) & (mx == 0)
    mn, mx = torch.where(zz, -torch.ones_like(mn), mn), torch.where(zz, torch.ones_like(mx), mx)
    lv = 2 ** bits - 1
    s = (mx - mn) / torch.full_like(mx, lv)             # IEEE division as in the official code (maxq is a tensor;
    z = torch.round(-mn / s)                            # torch divides by a Python scalar via its reciprocal)
    q = (torch.round(b / s) + z).clamp(0, lv)
    return (s * (q - z)).reshape(v.shape)


def scale_word(s: torch.Tensor, z: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """FP32 scale s and zero point z in [0, 15] -> the 16-bit scale word of the H20 cache (int32 holding the bits): the
    FP16 bit pattern nearest to RN_fp16(s) (integer distance on the bit pattern, ties upwards) whose 4 low bits equal
    z, so the word read as FP16 is the scale (|word - fp16(s)| <= 8 units of the last place, relative error <= 2^-7 +
    2^-11 in the normal range) and word & 15 is the zero point -- the kernels use the word as the scale unmasked.
    bits != 4: scale_word_b (b-bit zero point, INT2 / INT3 pages)."""
    if bits != 4:
        return scale_word_b(s, z, bits)
    h = s.half().view(torch.int16).to(torch.int32)
    zi = z.to(torch.int32)
    x = ((h - zi + 8).clamp_min(0) & ~15).clamp_max(0x7BF0)
    return x + zi


def word_value(wd: torch.Tensor) -> torch.Tensor:
    return wd.to(torch.int16).view(torch.float16).float()


def quant_asym_dep(v: torch.Tensor, g: int, half_scale: bool, bits: int = 4):
    """(codes q in [0, 15], zero point z in [0, 15], scale s (FP32), stored scale) of int4az / int4ah, groups of g along
    the last dim ([..., K/g, g] / [..., K/g, 1]).  Same operations as fake_asym (CUDA: (max - min) / 15 is a multiply by
    fp32(1/15), the other two divisions are IEEE), with min clamped to <= 0 and max to >= 0.  bits = 2 / 3: the same
    with 2^b levels and the b-bit zero point in the scale word (the INT2 pages of the vLLM plugin; = int<b>ah)."""
    lv = 2 ** bits - 1
    b = _grp(v.float(), g)
    mn, mx = b.amin(-1, keepdim=True).clamp_max(0.0), b.amax(-1, keepdim=True).clamp_min(0.0)
    s = ((mx - mn) / lv).clamp_min(1e-12)
    z = torch.round(-mn / s)
    q = (torch.round(b / s) + z).clamp(0, lv)
    word = scale_word(s, z, bits)
    return q, z, s, (word_value(word) if half_scale else s)


def scale_word_b(s: torch.Tensor, z: torch.Tensor, bits: int) -> torch.Tensor:
    """scale_word for a b-bit zero point: the FP16 bit pattern nearest to RN_fp16(s) whose b low bits equal z."""
    h = s.half().view(torch.int16).to(torch.int32)
    zi = z.to(torch.int32)
    m = 2 ** bits
    x = ((h - zi + m // 2).clamp_min(0) & ~(m - 1)).clamp_max(0x7BFF & ~(m - 1))
    return x + zi


def fake_asym_packed(v: torch.Tensor, bits: int, g: int) -> torch.Tensor:
    """int<b>ah:<g> (b = 2, 3; opt-in): int4ah with 2^b levels -- range widened to include 0 (min <= 0 <= max), integer
    zero point z in [0, 2^b - 1] stored in the b low mantissa bits of the FP16 scale word (16 bits of metadata per
    group: b + 16/g bits per value); dequantization (q - z) * word."""
    b = _grp(v.float(), g)
    lv = 2 ** bits - 1
    mn, mx = b.amin(-1, keepdim=True).clamp_max(0.0), b.amax(-1, keepdim=True).clamp_min(0.0)
    s = ((mx - mn) / lv).clamp_min(1e-12)
    z = torch.round(-mn / s).clamp(0, lv)
    q = (torch.round(b / s) + z).clamp(0, lv)
    return ((q - z) * word_value(scale_word_b(s, z, bits))).reshape(v.shape)


def _e4m3(t: torch.Tensor) -> torch.Tensor:
    """round to nearest E4M3 (saturating at +-448)."""
    return t.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()


def fake_kivi_f8(v: torch.Tensor, bits: int, g: int) -> torch.Tensor:
    """kivi<b>f8:<g> (opt-in): KIVI's asymmetric min/max quantizer with the scale and the minimum both stored in FP8
    E4M3 (round to nearest; 16 bits of metadata per group, b + 16/g bits per value); codes from the stored values,
    q = clamp(round((x - m8) / s8), 0, 2^b - 1), dequantization q * s8 + m8."""
    b = _grp(v.float(), g)
    lv = 2 ** bits - 1
    mn, mx = b.amin(-1, keepdim=True), b.amax(-1, keepdim=True)
    s8 = _e4m3((mx - mn) / lv)
    m8 = _e4m3(mn)
    q = ((b - m8) / torch.where(s8 == 0, torch.ones_like(s8), s8)).round().clamp(0, lv)
    return (q * s8 + m8).reshape(v.shape)


def fake_asym_dep(v: torch.Tensor, g: int, half_scale: bool, bits: int = 4) -> torch.Tensor:
    q, z, _, st = quant_asym_dep(v, g, half_scale, bits)
    return ((q - z) * st).reshape(v.shape)


def _grp(v: torch.Tensor, g: int) -> torch.Tensor:
    K = v.shape[-1]
    assert K % g == 0, (K, g)
    return v.reshape(*v.shape[:-1], K // g, g)


def e2m1(t: torch.Tensor) -> torch.Tensor:
    """round to the E2M1 grid, saturating at 6, ties to the even code (as cvt.rn.satfinite.e2m1x2)."""
    a = t.abs().clamp_max(6.0)
    mid = _E2M1_MID.to(t.device)
    idx = torch.bucketize(a, mid)                          # a == mid[i] lands on i (the lower grid point)
    tie = (idx < mid.numel()) & (a == mid[idx.clamp_max(mid.numel() - 1)])
    idx = torch.where(tie & (idx % 2 == 1), idx + 1, idx)  # odd lower code -> round up to the even code
    return E2M1_GRID.to(t.device)[idx] * t.sign()


def quantize(v: torch.Tensor, spec: str) -> tuple[torch.Tensor, torch.Tensor]:
    v = v.float()
    K = v.shape[-1]
    name, arg = parse(spec)
    if name == "fp8t":
        s = (v.abs().amax().clamp_min(1e-12) / 448.0).reshape(*([1] * (v.dim() - 1)), 1, 1)
        b = _grp(v, K)
        return (b / s).to(torch.float8_e4m3fn).float().reshape(v.shape), s.expand(*b.shape[:-1], 1)
    b = _grp(v, group_of(spec, K))
    amax = b.abs().amax(-1, keepdim=True)
    if name == "fp8":
        s = amax.clamp_min(1e-12) / 448.0
        c = (b / s).to(torch.float8_e4m3fn).float()
    elif name == "fp8e":                                   # UE8M0 scale: round the FP32 scale up to a power of two
        s = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-12) / 448.0)))
        c = (b / s).to(torch.float8_e4m3fn).float()
    elif name == "mxfp4":
        s = torch.exp2(torch.floor(torch.log2(amax.clamp_min(1e-30))) - 2)
        c = e2m1(b / s)
    elif name == "nvfp4":
        G = v.abs().amax().clamp_min(1e-12) / (448.0 * 6.0)
        s = (amax / 6.0 / G).clamp_max(448.0).to(torch.float8_e4m3fn).float().clamp_min(2.0 ** -9) * G
        c = e2m1(b / s)
    elif name == "e2m1f":                                  # NVFP4 grid, exact FP32 scale
        s = amax.clamp_min(1e-12) / 6.0
        c = e2m1(b / s)
    elif name == "nvint4":                                 # NVFP4 scale, uniform 15-level grid on [-6, 6]
        G = v.abs().amax().clamp_min(1e-12) / (448.0 * 6.0)
        s = (amax / 6.0 / G).clamp_max(448.0).to(torch.float8_e4m3fn).float().clamp_min(2.0 ** -9) * G
        c = torch.round(b / s * (7.0 / 6.0)).clamp(-7, 7) * (6.0 / 7.0)
    elif name.startswith("int"):
        lim = 2 ** (int(name[3:]) - 1) - 1
        s = amax.clamp_min(1e-12) / lim
        c = torch.round(b / s).clamp(-lim, lim)
    else:
        raise ValueError(spec)
    return c.reshape(v.shape), s


def dequant(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    g = codes.shape[-1] // scales.shape[-2]
    return (_grp(codes.float(), g) * scales).reshape(codes.shape)


def fake(v: torch.Tensor, spec: str) -> torch.Tensor:
    if spec in ("bf16", "id"):
        return v.float()
    name, _ = parse(spec)
    if name in _BASELINES or _PCLIP.fullmatch(name) or _TQ.fullmatch(name) or _QUEST.fullmatch(name) or _PH.fullmatch(name):
        return fake_baseline(v, spec)
    m = _KIVI.fullmatch(name)
    if m:
        return fake_kivi(v, int(m.group(1)), group_of(spec, v.shape[-1]))
    m = _ACLIP.fullmatch(name)
    if m:
        return fake_asym_clip(v, int(m.group(1)), group_of(spec, v.shape[-1]), int(m.group(3)) / 100.0, m.group(2) == "z")
    if name.startswith("int") and name.endswith("a"):
        return fake_asym(v, int(name[3:-1]), group_of(spec, v.shape[-1]))
    if name in _ASYM_DEP:
        return fake_asym_dep(v, group_of(spec, v.shape[-1]), name == "int4ah")
    m = _AH.fullmatch(name)
    if m:
        return fake_asym_packed(v, int(m.group(1)), group_of(spec, v.shape[-1]))
    m = _KF8.fullmatch(name)
    if m:
        return fake_kivi_f8(v, int(m.group(1)), group_of(spec, v.shape[-1]))
    return dequant(*quantize(v, spec))


def quantize_weight(W: torch.Tensor, spec: str) -> tuple[torch.Tensor, torch.Tensor]:
    """codes [N, K] and scales [N, K/g, 1]; 'fp8:<g>' uses one scale per g x g block (expanded to rows)."""
    name, arg = parse(spec)
    if name in ("fp8", "fp8e") and arg != "row":
        g = int(arg or 128)
        R, K = W.shape
        Rp = (R + g - 1) // g * g
        Wp = torch.nn.functional.pad(W.float(), (0, 0, 0, Rp - R)).reshape(Rp // g, g, K // g, g)
        s = Wp.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12) / 448.0
        if name == "fp8e":
            s = torch.exp2(torch.ceil(torch.log2(s)))
        c = (Wp / s).to(torch.float8_e4m3fn).float().reshape(Rp, K)[:R]
        s = s.expand(Rp // g, g, K // g, 1).reshape(Rp, K // g, 1)[:R]
        return c, s
    if spec == "bf16":
        return W.float(), torch.ones(*W.shape[:-1], 1, 1, device=W.device)
    return quantize(W, spec)


def fake_weight(W: torch.Tensor, spec: str) -> torch.Tensor:
    return dequant(*quantize_weight(W, spec))


# ---------------------------------------------------------------------------------------------------------------------
# KV-cache baselines (vLLM int4_per_token_head / TurboQuant / nvfp4_4over6, WUSH-KV's clipped quantizer)
# ---------------------------------------------------------------------------------------------------------------------
_BASELINES = {"nvfp4_46", "int4ph"}
_TQ = re.compile(r"tq([kv])(\d)(n?)")
_PCLIP = re.compile(r"int(\d)p(\d+)")
_QUEST = re.compile(r"quest(\d)")
_PH = re.compile(r"int([23])ph")      # opt-in 2 / 3-bit analogue of int4ph (no vLLM preset)


def _whole(name: str) -> bool:
    """baselines whose quantization group is the whole head vector."""
    return name == "int4ph" or bool(_PH.fullmatch(name)) or bool(_TQ.fullmatch(name)) or bool(_PCLIP.fullmatch(name)) or bool(_QUEST.fullmatch(name))


def _round_half_away(x: torch.Tensor) -> torch.Tensor:
    """Triton's (x +- 0.5).to(int32): round half away from zero."""
    return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))


def wht(x: torch.Tensor) -> torch.Tensor:
    """unnormalized Walsh-Hadamard transform along the last dim (vLLM fast_hadamard_transform, butterfly tier)."""
    d = x.shape[-1]
    h = 1
    while h < d:
        xv = x.reshape(*x.shape[:-1], d // (2 * h), 2, h)
        x = torch.stack([xv[..., 0, :] + xv[..., 1, :], xv[..., 0, :] - xv[..., 1, :]], dim=-2).reshape(*x.shape[:-1], d)
        h <<= 1
    return x


@lru_cache(maxsize=None)
def vllm_rht_signs(d: int) -> torch.Tensor:
    """vLLM int4_per_token_head._get_rht_signs(d, 0): fixed Bernoulli signs from a CPU generator."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0x9E3779B9)
    return 2.0 * torch.bernoulli(torch.full((d,), 0.5), generator=gen) - 1.0


_HADACORE_FAC = {1: 0.70703125, 2: 0.5, 3: 0.353515625, 4: 0.25}     # BF16 1/sqrt(2^c) constants of hadacore


def _hadacore_stages(D: int) -> list[int]:
    L = D.bit_length() - 1
    assert 1 << L == D and L <= 8, D
    return [L] if L <= 4 else [4, L - 4]


def hadacore(x: torch.Tensor) -> torch.Tensor:
    """vLLM's CUDA hadacore_transform on BF16 input (D <= 256): a 16-point stage on the low 4 index bits, then the
    remaining bits, each scaled by the BF16 constant 1/sqrt(2^c), FP32 accumulation, rounded to BF16 after every
    stage (bit-exact with the kernel on random inputs, see tests).  Output normalized ~ H x / sqrt(D)."""
    x = x.to(torch.bfloat16).float()
    D = x.shape[-1]
    lo = 1
    for c in _hadacore_stages(D):
        n = 1 << c
        xv = x.reshape(*x.shape[:-1], D // (n * lo), n, lo)
        y = wht(xv.transpose(-1, -2)).transpose(-1, -2) * _HADACORE_FAC[c]
        x = y.to(torch.bfloat16).float().reshape(x.shape)
        lo *= n
    return x


def hadacore_gain(D: int) -> float:
    """effective gain of vLLM's fast_hadamard_transform relative to the exact unnormalized H (BF16 constants times
    the FP32 rescale sqrt(D)): 1 for D = 64 / 256, 0.99989 for D = 128."""
    g = torch.tensor(D ** 0.5, dtype=torch.float32)
    for c in _hadacore_stages(D):
        g = g * _HADACORE_FAC[c]
    return float(g)


def int4ph_rotated(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(BF16 rotated input y, dequantized y_hat) of vLLM's int4_per_token_head store kernel."""
    x = v.float()
    D = x.shape[-1]
    s = vllm_rht_signs(D).to(x.device)
    y = (hadacore(x * s) * torch.tensor(D ** 0.5, dtype=torch.float32)).to(torch.bfloat16).float()
    from . import vllm_kernels as VK
    if VK.available(y):                                # Triton port: vLLM's division / FMA rounding, bit-exact
        return y, VK.int4ph_quant(y)
    mn, mx = y.amin(-1, keepdim=True), y.amax(-1, keepdim=True)
    scale = ((mx - mn) / 15.0).clamp_min(1e-6)
    zp = _round_half_away(-mn / scale).clamp(0, 15)
    q = _round_half_away(y * (1.0 / scale) + zp).clamp(0, 15)
    st = (scale.contiguous().view(torch.int32) & -16).view(torch.float32)
    return y, (q - zp) * st


def fake_intph(v: torch.Tensor, nb: int) -> torch.Tensor:
    """int<b>ph (b = 2, 3; opt-in): int4ph with 2^b levels -- the same randomized Hadamard
    (vLLM's signs and BF16 hadacore), per (token, head) scale = max((max - min) / (2^b - 1), 1e-6), zero point and codes
    rounded half away from zero and clamped to [0, 2^b - 1], the zero point stored in the b low mantissa bits of the
    FP32 scale (scale_bits & -2^b).  vLLM has no 2 / 3-bit per-token-head cache: IEEE torch arithmetic (the int4ph torch
    fallback with 2^b - 1 levels), not a bit-exact port of a shipped kernel."""
    x = v.float()
    D = x.shape[-1]
    s = vllm_rht_signs(D).to(x.device)
    y = (hadacore(x * s) * torch.tensor(D ** 0.5, dtype=torch.float32)).to(torch.bfloat16).float()
    L = float(2 ** nb - 1)
    mn, mx = y.amin(-1, keepdim=True), y.amax(-1, keepdim=True)
    scale = ((mx - mn) / L).clamp_min(1e-6)
    zp = _round_half_away(-mn / scale).clamp(0, L)
    q = _round_half_away(y * (1.0 / scale) + zp).clamp(0, L)
    st = (scale.contiguous().view(torch.int32) & -(2 ** nb)).view(torch.float32)
    yh = (q - zp) * st
    return wht(yh) * s * (hadacore_gain(D) / D)


def fake_int4ph(v: torch.Tensor) -> torch.Tensor:
    """vLLM kv_cache_dtype='int4_per_token_head' (vllm/v1/attention/ops/int4_per_token_head.py) on CUDA (hadacore):
    y = fast_hadamard(s * x) in BF16 (unnormalized, gain g = hadacore_gain(D)), per (token, head)
    scale = max((max - min) / 15, 1e-6), zero point and codes rounded half away from zero and clamped to [0, 15]; the
    stored scale loses its 4 low mantissa bits to the zero point (scale_bits & -16), dequant (q - zp) * scale.
    Attention uses the same transform on q (softmax scale / D) and s * fast_hadamard(o) / D on the output, so the
    equivalent reconstruction is x_hat = g * s * H(y_hat) / D (exact in FP32; vLLM's BF16 rounding of q and of the
    output is not part of the cache error and is not simulated)."""
    D = v.shape[-1]
    _, yh = int4ph_rotated(v)
    return wht(yh) * vllm_rht_signs(D).to(yh.device) * (hadacore_gain(D) / D)


def _lloyd_max(d: int, bits: int, max_iter: int = 200, tol: float = 1e-10) -> list[float]:
    """vLLM turboquant/centroids.py solve_lloyd_max (trapezoidal integration, same iteration) -> centroids."""
    n = 2 ** bits
    sigma2 = 1.0 / d
    sigma = math.sqrt(sigma2)

    def pdf(x):
        return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-x * x / (2 * sigma2))

    def trapz(f, a, b, m=200):
        h = (b - a) / m
        r = 0.5 * (f(a) + f(b))
        for i in range(1, m):
            r += f(a + i * h)
        return r * h
    lo, hi = -3.5 * sigma, 3.5 * sigma
    c = [lo + (hi - lo) * (i + 0.5) / n for i in range(n)]
    for _ in range(max_iter):
        bnd = [(c[i] + c[i + 1]) / 2.0 for i in range(n - 1)]
        edges = [lo * 3] + bnd + [hi * 3]
        new = []
        for i in range(n):
            a, b = edges[i], edges[i + 1]
            num, den = trapz(lambda x: x * pdf(x), a, b), trapz(pdf, a, b)
            new.append(num / den if den > 1e-15 else c[i])
        if max(abs(new[i] - c[i]) for i in range(n)) < tol:
            break
        c = new
    return c


@lru_cache(maxsize=None)
def tq_tables(d: int, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(sorted centroids, midpoints) in FP32 as vLLM's TurboQuant backend builds them."""
    c = torch.tensor(_lloyd_max(d, bits), dtype=torch.float32).sort().values
    return c, (c[:-1] + c[1:]) / 2


@lru_cache(maxsize=None)
def _tq_hadamard(d: int) -> torch.Tensor:
    """turboquant_attn._build_hadamard: Sylvester matrix in FP32 divided by sqrt(d)."""
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(d)


def fake_tq_key(v: torch.Tensor, bits: int, norm_correction: bool) -> torch.Tensor:
    """vLLM TurboQuant MSE keys (triton_turboquant_store.py + decode): y = (x / (||x|| + 1e-8)) H, index = number of
    Lloyd-Max midpoints <= y, FP16 norm; decode uses c (renormalized to unit norm if norm_correction) times the norm,
    against the rotated query, i.e. x_hat = ||x||_fp16 * c_hat H (H symmetric orthonormal)."""
    if bits == 8:                                      # turboquant_k8*: keys cast to E4M3 without a scale
        return v.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return tq_key_rotated(v, bits, norm_correction) @ _tq_hadamard(v.shape[-1]).to(v.device)


def tq_key_rotated(v: torch.Tensor, bits: int, norm_correction: bool) -> torch.Tensor:
    """dequantized TurboQuant key in the rotated basis: ||x||_fp16 * c_hat."""
    x = v.float()
    D = x.shape[-1]
    H = _tq_hadamard(D).to(x.device)
    c, mids = (t.to(x.device) for t in tq_tables(D, bits))
    norms = x.norm(dim=-1, keepdim=True)
    y = (x / (norms + 1e-8)) @ H
    cv = c[torch.bucketize(y, mids, right=True)]
    if norm_correction:
        cv = cv * (1.0 / torch.sqrt(cv.square().sum(-1, keepdim=True) + 1e-16))
    return norms.half().float() * cv


def fake_tq_value(v: torch.Tensor, bits: int) -> torch.Tensor:
    """vLLM TurboQuant values (_store_quantized_value): per (token, head) min/max, q = clamp(int((x - min) / s + 0.5)),
    s = max((max - min) / (2^b - 1), 1e-8); scale and min stored in FP16, dequant q * s16 + min16."""
    x = v.float()
    from . import vllm_kernels as VK
    if VK.available(x):
        return VK.tq_value_quant(x, bits)
    L = 2 ** bits - 1
    mn, mx = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    s = (mx - mn) / L
    s = torch.where(s > 1e-8, s, torch.full_like(s, 1e-8))
    q = torch.trunc((x - mn) / s + 0.5).clamp(0, L)
    return q * s.half().float() + mn.half().float()


def fake_nvfp4_46(v: torch.Tensor, G=None, tensor_den: float = 256.0 * 6.0) -> torch.Tensor:
    """Four Over Six: per 16-element block quantize with the E4M3 block scale for max -> 6 and for max -> 4 and keep
    the one with the lower squared error (ties keep 6), as vLLM's cvt_warp_fp16_to_fp4_4over6.  G = dequant tensor
    scale (vLLM: k_scale / v_scale); default absmax / (256 * 6), the paper's tensor scale (256 * 6/4 = 384 stays exact
    in E4M3).  A zero block scale reconstructs zeros (vLLM)."""
    x = v.float()
    b = _grp(x, 16)
    if G is None:
        G = x.abs().amax().clamp_min(1e-12) / tensor_den
    amax = b.abs().amax(-1, keepdim=True)
    recs, errs = [], []
    for den in (6.0, 4.0):
        sf = ((1.0 / G) * (amax * (1.0 / den))).clamp_max(448.0).to(torch.float8_e4m3fn).float()   # satfinite E4M3
        nz = sf > 0
        osc = torch.where(nz, 1.0 / (sf * G), torch.zeros_like(sf))          # kernel: codes = e2m1(x * outputScale)
        c = e2m1(b * osc)
        dq = torch.where(nz, 1.0 / osc, torch.zeros_like(osc))
        errs.append((c * dq - b).square().sum(-1, keepdim=True))
        recs.append(c * sf * G)                                               # read side: fp4 * sf * k_scale
    return torch.where(errs[1] < errs[0], recs[1], recs[0]).reshape(x.shape)


def _quantile_lastdim(a: torch.Tensor, k: float) -> torch.Tensor:
    """torch.quantile(a, k, dim=-1, keepdim=True) ('linear'), without torch.quantile's input-size limit."""
    srt = a.sort(-1).values
    pos = k * (a.shape[-1] - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, a.shape[-1] - 1)
    f = pos - lo
    return srt[..., lo:lo + 1] * (1 - f) + srt[..., hi:hi + 1] * f


def fake_pclip(v: torch.Tensor, bits: int, kappa: float, g: int) -> torch.Tensor:
    """OSCAR-style percentile-clipped affine quantizer (WUSH-KV eq. 7): x_min = max(min x, -quantile_k |x|),
    x_max = min(max x, quantile_k |x|), D = (x_max - x_min) / (2^b - 1), z = -x_min / D,
    Q(x) = D (clamp(round(x / D + z), 0, 2^b - 1) - z)."""
    b = _grp(v.float(), g)
    L = 2 ** bits - 1
    qk = _quantile_lastdim(b.abs(), kappa)
    mn = torch.maximum(b.amin(-1, keepdim=True), -qk)
    mx = torch.minimum(b.amax(-1, keepdim=True), qk)
    d = ((mx - mn) / L).clamp_min(1e-12)
    z = -mn / d
    return (d * (torch.round(b / d + z).clamp(0, L) - z)).reshape(v.shape)


@lru_cache(maxsize=None)
def quest_alpha(bits: int) -> float:
    """alpha_b of the QuEST quantizer (WUSH-KV eq. 5): the clipping multiplier whose fixed-scale midrise grid
    (2^b levels, step 2 alpha / (2^b - 1)) has the lowest mean squared error on a standard Gaussian (the paper defines
    alpha_b this way without giving its value).  Exact MSE by quadrature, golden-section search over alpha."""
    x = torch.linspace(-12, 12, 2_400_001, dtype=torch.float64)
    w = torch.exp(-x * x / 2) / math.sqrt(2 * math.pi) * (x[1] - x[0])
    lo_, hi_ = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1

    def mse(a):
        d = 2 * a / (2 ** bits - 1)
        q = d * (torch.floor(x / d).clamp(lo_, hi_) + 0.5)
        return float(((x - q) ** 2 * w).sum())
    a, b = 0.1, 8.0
    g = (math.sqrt(5) - 1) / 2
    for _ in range(200):
        c, d = b - g * (b - a), a + g * (b - a)
        if mse(c) < mse(d):
            b = d
        else:
            a = c
    return (a + b) / 2


def fake_quest(v: torch.Tensor, bits: int, g: int) -> torch.Tensor:
    """QuEST quantizer (WUSH-KV eq. 5), groups of g along the last dim (WUSH-KV: g = head dim, one scale per token and
    head): D = 2 alpha_b (2^b - 1)^-1 g^-1/2 ||x||, Q(x) = D (clamp(floor(x / D), -2^(b-1), 2^(b-1) - 1) + 1/2),
    Q(0) = 0."""
    b = _grp(v.float(), g)
    d = 2 * quest_alpha(bits) / (2 ** bits - 1) * b.norm(dim=-1, keepdim=True) / math.sqrt(g)
    q = torch.floor(b / d.clamp_min(1e-30)).clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    return torch.where(d > 0, d * (q + 0.5), torch.zeros_like(b)).reshape(v.shape)


def fake_baseline(v: torch.Tensor, spec: str) -> torch.Tensor:
    name, arg = parse(spec)
    if name == "nvfp4_46":
        return fake_nvfp4_46(v)
    if name == "int4ph":
        return fake_int4ph(v)
    m = _PH.fullmatch(name)
    if m:
        return fake_intph(v, int(m.group(1)))
    m = _TQ.fullmatch(name)
    if m:
        kind, nb, nc = m.group(1), int(m.group(2)), m.group(3) == "n"
        return fake_tq_key(v, nb, nc) if kind == "k" else fake_tq_value(v, nb)
    m = _PCLIP.fullmatch(name)
    if m:
        return fake_pclip(v, int(m.group(1)), int(m.group(2)) / 100.0, group_of(spec, v.shape[-1]))
    m = _QUEST.fullmatch(name)
    if m:
        return fake_quest(v, int(m.group(1)), group_of(spec, v.shape[-1]))
    raise ValueError(spec)


def bits(spec: str, D: int) -> float:
    """storage bits per cached element including scales / zero points / norms, at deployment precision: FP16 scale per
    symmetric INT group, FP16 scale + FP16 zero per asymmetric group, E4M3 / E8M0 block scales for NVFP4 / MXFP4,
    FP32 scales for FP8 groups (per-tensor scales are ignored)."""
    name, arg = parse(spec)
    if spec in ("bf16", "id"):
        return 16.0
    if name == "fp8t":
        return 8.0
    if name in ("fp8", "fp8e"):
        g = D if arg == "row" else (int(arg) if arg else 128)
        return 8.0 + (32.0 if name == "fp8" else 8.0) / g
    if name in ("nvfp4", "nvfp4_46", "nvint4"):
        return 4.0 + 8.0 / 16
    if name == "e2m1f":
        return 4.0 + 32.0 / (int(arg) if arg else 16)
    if name == "mxfp4":
        return 4.0 + 8.0 / (int(arg) if arg else 32)
    if name == "int4ph":
        return 4.0 + 32.0 / D                          # FP32 scale carrying the 4-bit zero point
    m = _PH.fullmatch(name)
    if m:
        return int(m.group(1)) + 32.0 / D              # FP32 scale carrying the b-bit zero point
    m = _KIVI.fullmatch(name) or _ACLIP.fullmatch(name)
    if m:                                              # FP16 scale + FP16 zero point (KIVI: FP16 minimum) per group
        return int(m.group(1)) + 32.0 / min(group_of(spec, D), D)
    m = _KF8.fullmatch(name)
    if m:                                              # FP8 scale + FP8 minimum per group
        return int(m.group(1)) + 16.0 / min(group_of(spec, D), D)
    m = _AH.fullmatch(name)
    if m:                                              # one FP16 word per group: scale with the b-bit zero point
        return int(m.group(1)) + 16.0 / min(group_of(spec, D), D)
    if name in _ASYM_DEP:                              # FP16 scale (+ FP16 zero for int4az; int4ah: zero in the scale)
        return 4.0 + (16.0 if name == "int4ah" else 32.0) / min(group_of(spec, D), D)
    m = _TQ.fullmatch(name)
    if m:
        nb = int(m.group(2))
        if m.group(1) == "k":
            return 8.0 if nb == 8 else nb + 16.0 / D   # FP16 norm
        return nb + 32.0 / D                           # FP16 scale + FP16 min
    m = _PCLIP.fullmatch(name)
    if m:
        return int(m.group(1)) + 32.0 / min(group_of(spec, D), D)
    m = _QUEST.fullmatch(name)
    if m:                                              # one FP16 scale (the norm) per group
        return int(m.group(1)) + 16.0 / min(group_of(spec, D), D)
    if name.startswith("int"):
        nb = int(name[3:-1] if name.endswith("a") else name[3:])
        g = min(group_of(spec, D), D)
        return nb + (32.0 if name.endswith("a") else 16.0) / g
    raise ValueError(spec)
