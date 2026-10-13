"""Kernel-level tests of libflagrot_kernels.so in the vLLM env (torch 2.13): paged writers + decode attention (static
FlagRot INT4 v2, symmetric and asymmetric int4a; per-token codebook incl. key-only 2x2 konly / konlya), BF16 staging
window with row indirection, a repeated block-table row (+ row stride 0 for the codebook kernels), empty rows.
Reference = the PyTorch layout parser (kernels.dequant_*) + float64 attention; criterion as kernels/test_attn_v2.py:
max |err| / max |ref| <= max(2 x BF16-SDPA error, 4e-3).  The asymmetric writer + parser are also checked against
the research quantizer blockflag.formats int4ah:32 (test_int4a_writer); the fused writers v3 against v1, byte for
byte on the whole pool (test_writer_v3; --v3-only: the writer tests alone).
Phase B): the same cases for GQA groups 5 / 6 / 8 / 12 and head dim 256 (static int4 / int4a and
the key-only 2x2 konly / konlya caches on the generalized kernel; fr_head_fwht at D = 256).

  CUDA_VISIBLE_DEVICES=5 python integrations/vllm_flagrot/tests/test_kernels.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vllm_flagrot import kernels as K  # noqa: E402

DEV = "cuda"
HKV, HQ, D, P = 8, 32, 128, 16
LOG2E = 1.4426950408889634


def sel_rot(x, mask, rb):
    """x [..., H, Dh] fp; mask [H, Dh/rb] bool: per-(head, block) orthonormal Hadamard."""
    H = K.hadamard(rb, x.device, x.dtype)
    xb = x.reshape(*x.shape[:-1], x.shape[-1] // rb, rb)
    y = torch.where(mask[..., None].bool(), xb @ H, xb)
    return y.reshape(x.shape)


def ref_attn(q, Kc, Vc, lens, scale, tails=None, kv_start=0):
    """q [B, Hq, Dh]; Kc/Vc [B, Lmax, Hkv, Dh] (valid range [kv_start, lens[b])); tails: list of (k, v) [n, Hkv, Dh]
    or None."""
    B, Hq, Dh = q.shape
    G = Hq // Kc.shape[2]
    out = torch.zeros(B, Hq, Dh, dtype=torch.float64, device=q.device)
    for b in range(B):
        a = min(kv_start, lens[b])
        ks, vs = [Kc[b, a:lens[b]]], [Vc[b, a:lens[b]]]
        if tails is not None and tails[b] is not None:
            ks.append(tails[b][0]); vs.append(tails[b][1])
        k = torch.cat(ks).double().repeat_interleave(G, 1)                   # [L, Hq, Dh]
        v = torch.cat(vs).double().repeat_interleave(G, 1)
        if k.shape[0] == 0:
            continue
        s = torch.einsum("hd,lhd->hl", q[b].double(), k) * scale
        p = torch.exp(s - s.amax(-1, keepdim=True))
        p = p / p.sum(-1, keepdim=True)
        out[b] = torch.einsum("hl,lhd->hd", p, v)
    return out


def sdpa_err(q, Kc, Vc, lens, scale, ref, tails=None, kv_start=0):
    o = ref_attn(q.bfloat16().float(), Kc.bfloat16().float(), Vc.bfloat16().float(), lens, scale,
                 None if tails is None else [None if t is None else (t[0].bfloat16(), t[1].bfloat16()) for t in tails],
                 kv_start)
    return (o.float() - ref.float()).abs().max().item()


def alloc(lens, num_pages, g):
    need = [(n + P - 1) // P for n in lens]
    ids = torch.randperm(num_pages, generator=g)[:sum(need)]
    bt = torch.zeros(len(lens), max(max(need), 1), dtype=torch.int32)
    o = 0
    for b, n in enumerate(need):
        bt[b, :n] = ids[o:o + n].int()
        o += n
    return bt.to(DEV)


def slots_of(bt, b, n):
    pos = torch.arange(n, device=DEV)
    return bt[b].long()[pos // P] * P + pos % P


def run_case(mode, lens, win=None, bt_shared=False, seed=0, kind="final", fmt="int4", G=4, Dh=128, Hkv=HKV, kv_start=0):
    HKV, D, HQ = Hkv, Dh, G * Hkv  # noqa: N806 (this case's shapes)
    g = torch.Generator().manual_seed(seed)
    B = len(lens)
    num_pages = sum((n + P - 1) // P for n in lens) + 7
    chunk = K.v2_chunk_bytes(P, D, fmt) if mode == "static" else K.cb_chunk_bytes(kind, P, D)
    pool = torch.randint(0, 256, (num_pages, HKV, chunk), dtype=torch.uint8, generator=g).to(DEV)   # garbage pool
    bt = alloc(lens, num_pages, g)
    if bt_shared:  # all rows read row 0 (prefill through the decode kernel): same pages, different lengths
        # "copy": row 0 repeated (what the plugin passes); "stride0": a broadcast row (row stride 0), which the
        # codebook kernels support; the static v2 kernel does not (its early page reads are bounded
        # by the row stride)
        bt = bt[:1].expand(B, -1)
        if bt_shared != "stride0":
            bt = bt.contiguous()
    kv_all = []
    rb = 32
    mask = (torch.rand(2 * HKV, D // rb, generator=g) < 0.5).to(torch.uint8).to(DEV)
    Cq = None
    if mode == "codebook":
        qs = torch.randn(HKV, 256, D, generator=g)  # (256 queries)
        qs[:, :, :8] *= 4
        Cq = torch.einsum("hnd,hne->hde", qs, qs).to(DEV)
        CT = Cq.transpose(1, 2).contiguous()
        co = (torch.rand(HKV, D, generator=g) + 0.1).to(DEV)
    for b in range(B if not bt_shared else 1):
        n = max(lens) if bt_shared else lens[b]
        kv = torch.randn(n, 2 * HKV, D, generator=g)
        kv[:, :HKV, :4] *= 6
        kv = kv.bfloat16().to(DEV)
        kv_all.append(kv)
        sl = slots_of(bt, b, n)
        if mode == "static":
            K.v2_write(kv, sl, mask, rb, pool, P, fmt=fmt)
        else:
            K.cb_write(kv, sl, CT, co, kind, pool, P)
    # reference K / V from the cache bytes
    if mode == "static":
        Kp, Vp = K.dequant_v2_pages(pool, P, torch.float64, fmt=fmt, Dh=D)
    else:
        Kp, Vp = K.dequant_cb_pages(pool, P, kind, torch.float64, Dh=D)
    Lmax = max(max(lens), 1)
    npg = (Lmax + P - 1) // P
    Kc = Kp[bt[:, :npg].long()].reshape(B, npg * P, HKV, D)
    Vc = Vp[bt[:, :npg].long()].reshape(B, npg * P, HKV, D)
    q = torch.randn(B, HQ, D, generator=g).bfloat16().to(DEV)
    scale = 1 / math.sqrt(D)
    tails = None
    tail = tail_len = tail_idx = None
    if win is not None:
        W, R = win
        ring = torch.randn(R, W, 2 * HKV, D, generator=g).bfloat16().to(DEV)
        tail_idx = torch.randperm(R, generator=g)[:B].int().to(DEV)
        tl = torch.randint(0, W + 1, (B,), generator=g).int()
        tail_len = tl.to(DEV)
        tails = [(ring[int(tail_idx[b]), :int(tl[b]), :HKV], ring[int(tail_idx[b]), :int(tl[b]), HKV:]) for b in range(B)]
        tail = ring
    if mode == "static":
        km, vm = mask[:HKV].bool(), mask[HKV:].bool()
        # stored K = rot(k); q . k = rot(q) . rot(k); output (rotated basis) -> back-rotated by the kernel (vrot)
        qr = sel_rot(q.double().view(B, HKV, G, D).transpose(1, 2), km, rb).transpose(1, 2).reshape(B, HQ, D)
        tails_r = None if tails is None else [(sel_rot(t[0].double(), km, rb), sel_rot(t[1].double(), vm, rb)) for t in tails]
        o = ref_attn(qr, Kc, Vc, lens, scale, tails_r, kv_start)
        ref = sel_rot(o.view(B, HKV, G, D).transpose(1, 2), vm, rb).transpose(1, 2).reshape(B, HQ, D)
        # (orthonormal symmetric blocks: the inverse is the same rotation)
        sd = sdpa_err(qr, Kc, Vc, lens, scale, o, tails_r, kv_start)
    else:
        ref = ref_attn(q.double(), Kc, Vc, lens, scale, tails, kv_start)
        sd = sdpa_err(q.double(), Kc, Vc, lens, scale, ref, tails, kv_start)
    seq = torch.tensor(lens, dtype=torch.int32, device=DEV)
    out = torch.full((B, HQ * D), float("nan"), dtype=torch.bfloat16, device=DEV)
    ns, ch = (K.plan_v2(B, Lmax, HKV, P, fmt=fmt, G=G, Dh=D) if mode == "static"
              else K.plan_cb(B, Lmax, HKV, P, kind, G=G, Dh=D))
    gfmt = fmt if mode == "static" else kind
    if mode == "static" or (kind in K.K2 and (G != 4 or D != 128)):
        HV, GT = K.vheads(HKV, G, D, gfmt), K.tile_of(K.v2_variant(gfmt, G, D))
    else:
        HV, GT = HKV, 4
    ws = K.Workspace(B, HV, B * HV * (ns + 1), DEV, GT, D)
    tkw = {} if tail is None else dict(tail=tail, tail_len=tail_len, tail_idx=tail_idx, tail_bs=tail.stride(0),
                                       tail_ts=tail.stride(1))
    if mode == "static":
        K.v2_attn(q, pool, bt, seq, out, ns, ch, ws, scale * LOG2E, kmask=mask[:HKV], vmask=mask[HKV:], rb=rb,
                  fmt=fmt, kv_start=kv_start, **tkw)
    else:
        K.cb_attn(q, pool, bt, seq, out, ns, ch, ws, scale * LOG2E, kind, kv_start=kv_start, **tkw)
    torch.cuda.synchronize()
    err = (out.view(B, HQ, D).double() - ref).abs()
    valid = torch.tensor([lens[b] > 0 or (tails is not None and tails[b][0].shape[0] > 0) for b in range(B)], device=DEV)
    e = err[valid].max().item() if valid.any() else 0.0
    m = ref[valid].abs().max().item() if valid.any() else 1.0
    lim = max(2 * sd / m, 4e-3)
    ok = e / m <= lim and not torch.isnan(out.view(B, HQ, D)[valid]).any()
    return ok, e / m, lim, ns, ch


def test_head_fwht():
    """fr_head_fwht == float64 matmul with the +-1 Sylvester matrix (scaled, rounded once to bf16: within one bf16
    rounding step); strided input, in-place output; unflagged heads copied bit-exactly.  D = 128 and 256."""
    g = torch.Generator().manual_seed(7)
    ok = True
    for T, H, D in ((1, 32, 128), (5, 16, 128), (300, 8, 128), (1, 24, 256), (300, 4, 256)):
        base = torch.randn(T, H + 3, D, generator=g).bfloat16().to(DEV)
        x = base[:, 1:H + 1]                                                    # token stride (H + 3) * D
        flags = (torch.rand(H, generator=g) < 0.5).to(torch.uint8).to(DEV)
        Hm = K.hadamard(D, DEV, torch.float32) * math.sqrt(D)
        ref = torch.where(flags.bool()[None, :, None], (x.double() @ Hm.double()) / 16, x.double())
        tol = ref.abs() * 2.0 ** -8 + 1e-30                                     # one bf16 rounding step
        y = K.head_fwht(x, flags, 1 / 16)
        ok &= bool(((y.double() - ref).abs() <= tol).all())
        ok &= torch.equal(y[:, ~flags.bool()], x[:, ~flags.bool()])
        xc = x.contiguous()
        K.head_fwht(xc, flags, 1 / 16, out=xc)
        ok &= torch.equal(xc, y)
    print(f"{'ok ' if ok else 'FAIL'} head_fwht (within one bf16 rounding of float64 H x / 16)")
    return ok


def test_int4a_writer():
    """static int4a writer + plugin parser == blockflag int4ah:32 (fake_asym_dep, FP16 scale word with the zero point
    in its 4 low bits) on the same (FlagRot-rotated) vectors: bit-exact without rotation (incl. groups whose range
    excludes 0, all-zero groups, tiny / large groups); with block rotations the writer's fp32 FWHT and the float32
    matmul of the reference may differ by an ulp, so there a few code flips are tolerated (counted)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from blockflag import formats as FM
    g = torch.Generator().manual_seed(11)
    ok = True
    for rot in (False, True):
        T = 333
        kv = torch.randn(T, 2 * HKV, D, generator=g)
        kv[:, :HKV, :4] *= 6
        grp = kv.view(T, 2 * HKV, 4, 32)
        sel = torch.rand(T, 2 * HKV, 4, generator=g)
        grp[sel < 0.05] = grp[sel < 0.05].abs() + 0.5                      # all positive (range excludes 0)
        grp[(sel >= 0.05) & (sel < 0.1)] = -grp[(sel >= 0.05) & (sel < 0.1)].abs() - 0.5   # all negative
        grp[(sel >= 0.1) & (sel < 0.12)] = 0.0                              # all zero
        grp[(sel >= 0.12) & (sel < 0.15)] *= 1e-5                           # FP16-subnormal scales
        grp[(sel >= 0.15) & (sel < 0.17)] *= 300
        kv = kv.bfloat16().to(DEV)
        mask = ((torch.rand(2 * HKV, D // 32, generator=g) < 0.5) & rot).to(torch.uint8).to(DEV)
        npg = (T + P - 1) // P + 3
        pool = torch.randint(0, 256, (npg, HKV, K.v2_chunk_bytes(P)), dtype=torch.uint8, generator=g).to(DEV)
        slots = torch.randperm(npg * P, generator=g)[:T].to(DEV)
        K.v2_write(kv, slots, mask, 32, pool, P, fmt="int4a")
        Kp, Vp = K.dequant_v2_pages(pool, P, torch.float32, fmt="int4a")          # [Np, P, H, 128]
        got = torch.cat([Kp.reshape(-1, HKV, D), Vp.reshape(-1, HKV, D)], 1)[slots]  # [T, 2H, 128]
        x = sel_rot(kv.float(), mask.bool(), 32)
        ref = FM.fake_asym_dep(x, 32, True)
        diff = (got - ref).abs()
        nbad = int((diff > 1e-6 * ref.abs().amax()).sum())
        this = nbad == 0 if not rot else nbad <= 1e-3 * ref.numel()
        ok &= this
        print(f"{'ok ' if this else 'FAIL'} int4a writer vs blockflag int4ah:32 (rotation {rot}): {nbad} of {ref.numel()} "
              f"values differ, max |diff| {diff.max().item():.3g}")
    return ok


def _edge_rows(T, g, scale_k=6.0):
    """bf16 [T, 2 HKV, 128]: Gaussian rows with outlier key channels and boundary groups (all positive / negative /
    zero, FP16-subnormal scales, x300, constant, a single non-zero element)."""
    kv = torch.randn(T, 2 * HKV, D, generator=g)
    kv[:, :HKV, :4] *= scale_k
    grp = kv.view(T, 2 * HKV, 4, 32)
    sel = torch.rand(T, 2 * HKV, 4, generator=g)
    grp[sel < 0.04] = grp[sel < 0.04].abs() + 0.5
    m = (sel >= 0.04) & (sel < 0.08)
    grp[m] = -grp[m].abs() - 0.5
    grp[(sel >= 0.08) & (sel < 0.10)] = 0.0
    grp[(sel >= 0.10) & (sel < 0.12)] *= 1e-5
    grp[(sel >= 0.12) & (sel < 0.14)] *= 300
    grp[(sel >= 0.14) & (sel < 0.15)] = 1.25
    one = (sel >= 0.15) & (sel < 0.16)
    grp[one] = 0.0
    grp[..., 5][one] = 3.0
    return kv.bfloat16().to(DEV)


def _calib_cq():
    """C_q of real layers (Llama-3.1-8B-Instruct codebook calibration) when available, else a random PSD matrix."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    try:
        from common import calib_path
        cal = torch.load(calib_path("llama8b_it", "konlya"), map_location="cpu", weights_only=True)
        return [cal["cq"][li].float() for li in (0, 16, 31)]
    except Exception:  # noqa: BLE001
        g = torch.Generator().manual_seed(3)
        A = torch.randn(HKV, D, D, generator=g)
        return [A @ A.transpose(1, 2) / D]


def test_writer_v3():
    """writer v3 == writer v1, byte for byte, on the whole pool (random garbage pages, random / per-page-contiguous /
    holed slots, strided kv rows):
      codebook: fr_cb_write3 (one stream, and K / V on two streams) vs fr_cb_write, kinds konlya / konly / final;
      static:   fr_w3_static_write(kv, mask, hfull, path 0 / 1 / 2) vs fr_head_fwht(H x / 16) + fr_v2_write,
                int4 / int4a x no / random whole-head flags x zero / random block flags."""
    g = torch.Generator().manual_seed(23)
    ok, n = True, 0
    cqs = _calib_cq()
    for T in (1, 7, 64, 129, 256, 257, 1000, 4097):
        kvb = _edge_rows(T, g)
        base = torch.zeros(T, 2 * HKV + 8, D, dtype=torch.bfloat16, device=DEV)   # qkv-like token stride
        base[:, 4:4 + 2 * HKV] = kvb
        kv = base[:, 4:4 + 2 * HKV]
        npg = (T + P - 1) // P + 5
        for smode in ("rand", "contig", "holes"):
            if smode == "rand":
                slots = torch.randperm(npg * P, generator=g)[:T]
            elif smode == "contig":
                slots = torch.arange(T) + 3 * P
            else:
                slots = torch.arange(T) + 3
                slots[torch.rand(T, generator=g) < 0.1] = -1
            slots = slots.to(torch.int64).to(DEV)
            # codebook kinds
            for kind in ("konlya", "konly", "final"):
                cq = cqs[n % len(cqs)].to(DEV)
                CT = cq.transpose(1, 2).contiguous()
                co = (torch.rand(HKV, D, generator=g) + 0.1).to(DEV)
                C1, beta = K.w3_calib(cq.cpu())
                C1, beta = C1.to(DEV), beta.to(DEV)
                pool0 = torch.randint(0, 256, (npg, HKV, K.cb_chunk_bytes(kind, P)), dtype=torch.uint8,
                                      generator=g).to(DEV)
                p1, p3, p3s = pool0.clone(), pool0.clone(), pool0.clone()
                K.cb_write(kv, slots, CT, co, kind, p1, P)
                K.cb_write3(kv, slots, CT, co, C1, beta, kind, p3, P, split=False)
                K.cb_write3(kv, slots, CT, co, C1, beta, kind, p3s, P, split=True)
                torch.cuda.synchronize()
                this = torch.equal(p1, p3) and torch.equal(p1, p3s)
                ok &= this
                n += 1
                if not this:
                    print(f"FAIL writer v3 {kind} T={T} slots={smode}: {int((p1 != p3).sum())} / "
                          f"{int((p1 != p3s).sum())} bytes differ (one stream / two streams)")
            # static
            for fmt in ("int4", "int4a"):
                for hmode in ("none", "rand"):
                    for bmode in ("zero", "rand"):
                        hfull = None
                        if hmode == "rand":
                            hfull = (torch.rand(2 * HKV, generator=g) < 0.5).to(torch.uint8).to(DEV)
                        mask = ((torch.rand(2 * HKV, D // 32, generator=g) < 0.5) & (bmode == "rand"))
                        mask = mask.to(torch.uint8).to(DEV)
                        pool0 = torch.randint(0, 256, (npg, HKV, K.v2_chunk_bytes(P)), dtype=torch.uint8,
                                              generator=g).to(DEV)
                        p1 = pool0.clone()
                        x = K.head_fwht(kv, hfull, 1 / 16) if hfull is not None else kv
                        K.v2_write(x, slots, mask, 32, p1, P, fmt=fmt)
                        for path in (0, 1, 2):
                            p3 = pool0.clone()
                            K.w3_static_write(kv, slots, mask, hfull, p3, P, fmt=fmt, path=path)
                            torch.cuda.synchronize()
                            this = torch.equal(p1, p3)
                            ok &= this
                            n += 1
                            if not this:
                                print(f"FAIL writer v3 static {fmt} T={T} slots={smode} hfull={hmode} "
                                      f"blocks={bmode} path={path}: {int((p1 != p3).sum())} bytes differ")
    print(f"{'ok ' if ok else 'FAIL'} writer v3 == v1 byte for byte ({n} pool comparisons)")
    return ok


def main():
    hf_ok = wr_ok = True
    if "--kv-start-only" not in sys.argv:
        hf_ok = test_head_fwht()
        wr_ok = test_int4a_writer()
        wr_ok &= test_writer_v3()
    if "--v3-only" in sys.argv:
        sys.exit(0 if wr_ok else 1)
    cases = []
    for mode, fmt in (("static", "int4"), ("static", "int4a"), ("codebook", "int4")):
        for lens in ([1], [77], [300, 4101, 17], [1000] * 5, [33000], [0, 513, 64]):
            for win in (None, (32, 9), (256, 7)):
                cases.append(dict(mode=mode, lens=lens, win=win, fmt=fmt))
        cases.append(dict(mode=mode, lens=list(range(1, 200, 7)), bt_shared="copy", fmt=fmt))
        if mode == "codebook":
            cases.append(dict(mode=mode, lens=list(range(1, 200, 7)), bt_shared="stride0", fmt=fmt))
            for kind in ("konly", "konlya"):
                for lens in ([1], [77], [300, 4101, 17], [0, 513, 64], [33000]):
                    for win in (None, (64, 9), (256, 7)):
                        cases.append(dict(mode=mode, lens=lens, win=win, kind=kind))
                cases.append(dict(mode=mode, lens=list(range(1, 200, 7)), bt_shared="copy", win=(64, 40), kind=kind))
        cases.append(dict(mode=mode, lens=list(range(1, 200, 7)), bt_shared="copy", win=(64, 40), fmt=fmt))
    # Phase B: GQA groups 5 / 6 / 8 / 12, head dim 256 (Qwen3-14B, Qwen3.8-27B, Qwen3-32B, GLM-4.5-Air shapes)
    for (G, Dh, Hkv) in ((5, 128, 8), (8, 128, 8), (12, 128, 4), (6, 256, 4), (6, 128, 2)):
        for mode, fmt, kind in (("static", "int4", None), ("static", "int4a", None), ("codebook", None, "konly"),
                                ("codebook", None, "konlya")):
            kw = dict(mode=mode, G=G, Dh=Dh, Hkv=Hkv, **({"fmt": fmt} if fmt else {"kind": kind}))
            for lens in ([1], [300, 4101, 17], [0, 513, 64], [33000]):
                for win in (None, (64, 9), (256, 7)):
                    cases.append(dict(lens=lens, win=win, **kw))
            cases.append(dict(lens=list(range(1, 200, 7)), bt_shared="copy", win=(64, 40), **kw))
    # Phase 2k: kv_start (the sink tokens stored at their real positions, masked in the paged body; the plugin passes
    # lengths 0 or > kv_start), every format the plugin runs with sinks, head 128 / GQA 4 and the generalized kernel
    for (G, Dh, Hkv) in ((4, 128, 8), (6, 256, 4), (8, 128, 8)):
        for mode, fmt, kind in (("static", "int4", None), ("static", "int4a", None), ("codebook", None, "konlya"),
                                ("codebook", None, "konly")):
            kw = dict(mode=mode, G=G, Dh=Dh, Hkv=Hkv, **({"fmt": fmt} if fmt else {"kind": kind}))
            for ks in (4, 16, 37):
                for lens in ([ks + 1], [77], [300, 4101, 17 + ks], [0, 513, 64 + ks], [33000]):
                    for win in (None, (132, 9)):
                        cases.append(dict(lens=lens, win=win, kv_start=ks, **kw))
    if "--kv-start-only" in sys.argv:
        cases = [c for c in cases if c.get("kv_start")]
    fails = 0
    for i, c in enumerate(cases):
        ok, r, lim, ns, ch = run_case(seed=i, **c)
        fails += not ok
        name = c["mode"] + ("/" + c["kind"] if "kind" in c else "/" + c.get("fmt", "int4") if c["mode"] == "static" else "")
        if "G" in c:
            name += f" G{c['G']} D{c['Dh']} Hkv{c['Hkv']}"
        if c.get("kv_start"):
            name += f" kv_start {c['kv_start']}"
        print(f"{'ok ' if ok else 'FAIL'} {name:15s} lens={str(c['lens'])[:40]:40s} win={c.get('win')} "
              f"shared={c.get('bt_shared', False)} ns={ns} chunk={ch}  err/max={r:.2e} (lim {lim:.2e})")
    print(f"{len(cases) - fails}/{len(cases)} passed")
    sys.exit(1 if fails or not hf_ok or not wr_ok else 0)


if __name__ == "__main__":
    main()
