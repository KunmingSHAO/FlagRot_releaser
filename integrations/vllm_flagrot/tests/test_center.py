"""int2c: static post-RoPE key centering in the cache writers (kernel-level, vLLM env).

  CUDA_VISIBLE_DEVICES=2 python integrations/vllm_flagrot/tests/test_center.py [--quick]

The writers take the raw bf16 K | V rows and kmu [Hkv, Dh] fp32 and encode x - kmu (fp32, round-to-nearest) for the K
rows, exactly the simulation's key.float() - mu (blockflag.qattention.blockflag_attention, kv_center = post).

  1. exact-grid identity, every writer and format: K rows and mu on a 2^-6 grid with |k|, |mu| <= 2, so k - mu is
     exactly representable in bf16; writer(k, kmu = mu) must write the same bytes as writer(bf16(k - mu), no kmu)
     (static v1 / v3 paths 0 1 2 with and without whole-head / block flags: int4, int4a, int2a; codebook v1 / v3:
     konlya2, konlya, konly, final; head dim 256: static v1 int2a / int4a, konlya2 / konlya v1).  V rows untouched.
  2. vs the simulation with a general fp32 mu (the exported mu of real models, keys = mu + noise): static int2a writer
     == blockflag int2ah:32 (fake_asym_dep) of sel_rot(k.float() - mu); konlya2 writer v1 == codebook_ref.encode(
     k.float() - mu, asym2) codes / scale words / flags (head dim 128 and 256); writer v3 == v1 byte for byte;
     codebook_ref asym2 == blockflag dyn_quant(int2a:32, cq+f+konly) on the centered fp32 keys; rtn (zero masks) too.
  3. decode attention over centered pages + centered BF16 window rows vs float64 (criterion of test_kernels.py), and
     shift invariance: float64 attention over the uncentered reconstruction (dequantized + mu) == over the centered
     cache (rel. difference ~1e-15).
  4. identity (no quantization): BF16-window-only decode (every key in the window) with tail rows k vs k - mu on the
     exact grid: the outputs agree to fp32 softmax rounding (centering alone changes nothing).
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.append(str(HERE.parents[2] / "kernels"))
import test_kernels as TK  # noqa: E402
from test_kernels import DEV, HKV, P, K, sel_rot, ref_attn, sdpa_err, alloc, slots_of, LOG2E  # noqa: E402

D = 128
CAL = Path(os.environ.get("FLAGROT_CALIB_DIR", str(HERE.parents[2] / "calib")))


def grid_rows(T, H2, Dh, g, lim=2.0):
    """bf16 rows on the 2^-6 grid, |x| <= lim (exactly representable; differences of two such values too)."""
    x = (torch.randn(T, H2, Dh, generator=g) * 0.7 * 64).round().clamp(-lim * 64, lim * 64) / 64
    return x


def grid_mu(Hkv, Dh, g, lim=2.0):
    return ((torch.randn(Hkv, Dh, generator=g) * 64).round().clamp(-lim * 64, lim * 64) / 64).float()


def _cq(Hkv, Dh, g):
    qs = torch.randn(Hkv, 256, Dh, generator=g)
    qs[:, :, :8] *= 4
    return torch.einsum("hnd,hne->hde", qs, qs) / 256


def test_exact_grid():
    g = torch.Generator().manual_seed(51)
    ok, n = True, 0
    for Dh in (128, 256):
        Hkv = HKV if Dh == 128 else 4
        for T in ((1, 7, 64, 257, 1000) if Dh == 128 else (1, 77, 300)):
            x = grid_rows(T, 2 * Hkv, Dh, g)
            mu = grid_mu(Hkv, Dh, g)
            kv = x.bfloat16().to(DEV)
            xc = x.clone()
            xc[:, :Hkv] -= mu
            assert torch.equal(xc.bfloat16().float(), xc), "grid construction not exact"
            kvc = xc.bfloat16().to(DEV)
            kmu = mu.to(DEV).contiguous()
            npg = (T + P - 1) // P + 4
            slots = torch.randperm(npg * P, generator=g)[:T].to(torch.int64).to(DEV)
            # static
            for fmt in ("int4", "int4a", "int2a"):
                if Dh == 256 and fmt == "int4":
                    continue
                for bmode in ("zero", "rand"):
                    mask = ((torch.rand(2 * Hkv, Dh // 32, generator=g) < 0.5) & (bmode == "rand")).to(torch.uint8).to(DEV)
                    pool0 = torch.randint(0, 256, (npg, Hkv, K.v2_chunk_bytes(P, Dh, fmt)), dtype=torch.uint8,
                                          generator=g).to(DEV)
                    pa, pb = pool0.clone(), pool0.clone()
                    K.v2_write(kv, slots, mask, 32, pa, P, fmt=fmt, kmu=kmu)
                    K.v2_write(kvc, slots, mask, 32, pb, P, fmt=fmt)
                    this = torch.equal(pa, pb)
                    n += 1
                    if Dh == 128:   # v3 static, with whole-head flags: the writer centers in the stored basis
                        for hmode in ("none", "rand"):
                            hfull = (torch.rand(2 * Hkv, generator=g) < 0.5).to(torch.uint8).to(DEV) if hmode == "rand" else None
                            kmu_s = kmu
                            if hfull is not None:   # stored basis of whole-head K heads: H mu / 16 (exact on the grid)
                                hm = (K.hadamard(Dh, DEV, torch.float64) * math.sqrt(Dh))
                                kmu_s = torch.where(hfull[:Hkv].bool()[:, None], (kmu.double() @ hm) / 16, kmu.double()).float().contiguous()
                            xs = K.head_fwht(kv, hfull, 1 / 16) if hfull is not None else kv
                            p1 = pool0.clone()
                            K.v2_write(xs, slots, mask, 32, p1, P, fmt=fmt, kmu=kmu_s)
                            for path in (0, 1, 2):
                                p3 = pool0.clone()
                                K.w3_static_write(kv, slots, mask, hfull, p3, P, fmt=fmt, path=path, kmu=kmu_s)
                                torch.cuda.synchronize()
                                e = torch.equal(p1, p3)
                                this &= e
                                n += 1
                                if not e:
                                    print(f"FAIL v3 static {fmt} T={T} hfull={hmode} blocks={bmode} path={path}: "
                                          f"{int((p1 != p3).sum())} bytes differ from v1")
                            if hfull is None:
                                this &= torch.equal(p1, pa)
                    ok &= this
                    if not this:
                        print(f"FAIL exact-grid static {fmt} D{Dh} T={T} blocks={bmode}: {int((pa != pb).sum())} bytes")
            # codebook
            kinds = ("konlya2", "konlya", "konly", "final") if Dh == 128 else ("konlya2", "konlya")
            for kind in kinds:
                Cq = _cq(Hkv, Dh, g).to(DEV)
                CT = Cq.transpose(1, 2).contiguous()
                co = (torch.rand(Hkv, Dh, generator=g) + 0.1).to(DEV)
                pool0 = torch.randint(0, 256, (npg, Hkv, K.cb_chunk_bytes(kind, P, Dh)), dtype=torch.uint8,
                                      generator=g).to(DEV)
                pa, pb = pool0.clone(), pool0.clone()
                K.cb_write(kv, slots, CT, co, kind, pa, P, kmu=kmu)
                K.cb_write(kvc, slots, CT, co, kind, pb, P)
                torch.cuda.synchronize()
                this = torch.equal(pa, pb)
                n += 1
                if Dh == 128 and kind in K.W3_KINDS:
                    C1, beta = K.w3_calib(Cq.cpu())
                    C1, beta = C1.to(DEV), beta.to(DEV)
                    for split in (False, True):
                        p3 = pool0.clone()
                        K.cb_write3(kv, slots, CT, co, C1, beta, kind, p3, P, split=split, kmu=kmu)
                        torch.cuda.synchronize()
                        e = torch.equal(p3, pa)
                        this &= e
                        n += 1
                        if not e:
                            print(f"FAIL v3 {kind} T={T} split={split}: {int((p3 != pa).sum())} bytes differ from v1")
                ok &= this
                if not this:
                    print(f"FAIL exact-grid {kind} D{Dh} T={T}: {int((pa != pb).sum())} bytes")
    print(f"{'ok ' if ok else 'FAIL'} exact grid: writer(k, kmu) == writer(bf16(k - mu)) byte for byte, every writer "
          f"({n} pool comparisons)")
    return ok


def real_mu(model, Dh):
    """the exported mu of a real model (layer 0, 2 and the last), [Hkv, Dh] fp32 each; random when not available."""
    f = CAL / f"flagrot_center_{model}.pt"
    if not f.exists():
        return None
    mu = torch.load(f, map_location="cpu", weights_only=True)["mu"]
    if mu.shape[-1] != Dh:
        return None
    return [mu[i].float() for i in (0, 2, mu.shape[0] - 1)]


def centered_keys(mu, T, g, noise=1.0):
    """keys = mu + noise (the real offset structure), bf16; the K rows of [T, 2 Hkv, Dh] with Gaussian values."""
    Hkv, Dh = mu.shape
    kv = torch.randn(T, 2 * Hkv, Dh, generator=g)
    kv[:, :Hkv] = kv[:, :Hkv] * noise * (1 + mu.abs().mean()) * 0.3 + mu
    kv[:, :Hkv, :4] *= 2
    return kv.bfloat16()


def test_vs_sim():
    from blockflag import formats as FM
    from blockflag import qattention as QA
    import codebook_ref as CR
    g = torch.Generator().manual_seed(61)
    ok = True
    for model, Dh in (("qwen8b", 128), ("llama8b", 128), ("qwen38_27b", 256)):
        mus = real_mu(model, Dh)
        if mus is None:
            print(f"  (no center file for {model}: skipped)")
            continue
        for li, mu in enumerate(mus):
            Hkv = mu.shape[0]
            T = 300
            kv = centered_keys(mu, T, g).to(DEV)
            kmu = mu.to(DEV).contiguous()
            xc = kv.float().clone()
            xc[:, :Hkv] = xc[:, :Hkv] - kmu                     # the simulation: key.float() - mu (fp32)
            npg = (T + P - 1) // P + 3
            slots = torch.randperm(npg * P, generator=g)[:T].to(DEV)
            # static int2a: rtn (zero masks) and random block flags
            for rot in (False, True):
                mask = ((torch.rand(2 * Hkv, Dh // 32, generator=g) < 0.5) & rot).to(torch.uint8).to(DEV)
                pool = torch.randint(0, 256, (npg, Hkv, K.v2_chunk_bytes(P, Dh, "int2a")), dtype=torch.uint8,
                                     generator=g).to(DEV)
                K.v2_write(kv, slots, mask, 32, pool, P, fmt="int2a", kmu=kmu)
                Kp, Vp = K.dequant_v2_pages(pool, P, torch.float32, fmt="int2a", Dh=Dh)
                got = torch.cat([Kp.reshape(-1, Hkv, Dh), Vp.reshape(-1, Hkv, Dh)], 1)[slots]
                ref = FM.fake_asym_dep(sel_rot(xc, mask.bool(), 32), 32, True, 2)
                diff = (got - ref).abs()
                nbad = int((diff > 1e-6 * ref.abs().amax()).sum())
                this = nbad == 0 if not rot else nbad <= 1e-3 * ref.numel()
                ok &= this
                print(f"{'ok ' if this else 'FAIL'} static int2a writer (kmu) vs blockflag int2ah:32 of sel_rot(k - mu) "
                      f"[{model} layer #{li}, D {Dh}, rotation {rot}]: {nbad} of {ref.numel()} differ")
            # konlya2 v1 vs codebook_ref (asym2) on the centered fp32 keys
            Cq = _cq(Hkv, Dh, g).to(DEV)
            CT = Cq.transpose(1, 2).contiguous()
            co = (torch.rand(Hkv, Dh, generator=g) + 0.1).to(DEV)
            pool = torch.randint(0, 256, (npg, Hkv, K.cb_chunk_bytes("konlya2", P, Dh)), dtype=torch.uint8, generator=g).to(DEV)
            K.cb_write(kv, slots, CT, co, "konlya2", pool, P, kmu=kmu)
            torch.cuda.synchronize()
            codes, mag, sbits = K.parse_int2_pages(pool, P, Dh)
            codes = codes.transpose(1, 2).reshape(-1, Hkv, 2, Dh)[slots]
            sbits = sbits.transpose(1, 2).reshape(-1, Hkv, 2, Dh // 32)[slots].to(torch.int32) & 0xFFFF
            ek = CR.encode(xc[:, :Hkv], Cq, co, "k", cfg="konlya2", qk="asym2")
            ev = CR.encode(xc[:, Hkv:], Cq, co, "v", cfg="konlya2", qk="asym2")
            bad = 0
            for kind, e in ((0, ek), (1, ev)):
                word = e["s"].half().view(torch.int16).to(torch.int32) & 0xFFFF
                if kind == 0:
                    word = word | (((e["mode"] == 0)[..., None] & (e["local"] == 1)).to(torch.int32) << 15)
                else:
                    word[..., 0] |= (ek["mode"] == 2).to(torch.int32) << 15
                z = word & 3
                cm = e["q"].to(torch.int32) - z.repeat_interleave(32, -1)
                bad += int((codes[:, :, kind].to(torch.int32) != cm).sum()) + int((sbits[:, :, kind] != word).sum())
            this = bad == 0
            ok &= this
            print(f"{'ok ' if this else 'FAIL'} konlya2 writer v1 (kmu) vs codebook_ref asym2 of k - mu [{model} layer "
                  f"#{li}, D {Dh}]: {bad} mismatching codes / scale words")
            if Dh == 128:
                C1, beta = K.w3_calib(Cq.cpu())
                for split in (False, True):
                    p3 = torch.randint(0, 256, pool.shape, dtype=torch.uint8, generator=g).to(DEV)
                    p1 = p3.clone()
                    K.cb_write(kv, slots, CT, co, "konlya2", p1, P, kmu=kmu)
                    K.cb_write3(kv, slots, CT, co, C1.to(DEV), beta.to(DEV), "konlya2", p3, P, split=split, kmu=kmu)
                    torch.cuda.synchronize()
                    this = torch.equal(p1, p3)
                    ok &= this
                    print(f"{'ok ' if this else 'FAIL'} konlya2 writer v3 == v1 with kmu (split {split}): "
                          f"{int((p1 != p3).sum())} bytes differ")
                # codebook_ref asym2 == the simulation's dyn_quant on the centered fp32 keys
                QA.KV["dyn"] = dict(rule="cq", local=["H"], head=["F"], name="cq+f+konly")
                QA.KV["dynstat"] = {}
                QA.KV["cq"] = {0: Cq}
                QA.KV["co"] = {0: co}
                sim = QA.dyn_quant(xc[:, :Hkv].permute(1, 0, 2)[None], "int2a:32", 32, "k", 0)[0].permute(1, 0, 2)
                nb = int((sim - ek["rec"]).abs().gt(0).sum())
                ok &= nb == 0
                print(f"{'ok ' if nb == 0 else 'FAIL'} codebook_ref asym2 == dyn_quant int2a:32 on k - mu [{model} "
                      f"layer #{li}]: {nb} of {sim.numel()} reconstruction elements differ")
    return ok


def run_case_c(mode, lens, win=None, seed=0, kind="konlya2", fmt="int2a", G=4, Dh=128, Hkv=HKV, kv_start=0):
    """test_kernels.run_case with key centering: pages written with kmu, BF16 window rows bf16(k - mu)."""
    D_, HQ = Dh, G * Hkv
    g = torch.Generator().manual_seed(seed)
    B = len(lens)
    num_pages = sum((n + P - 1) // P for n in lens) + 7
    chunk = K.v2_chunk_bytes(P, D_, fmt) if mode == "static" else K.cb_chunk_bytes(kind, P, D_)
    pool = torch.randint(0, 256, (num_pages, Hkv, chunk), dtype=torch.uint8, generator=g).to(DEV)
    bt = alloc(lens, num_pages, g)
    mu = (torch.randn(Hkv, D_, generator=g) * 3).float()
    mu[:, :8] *= 5                                                     # a few large offset channels
    kmu = mu.to(DEV).contiguous()
    mask = (torch.rand(2 * Hkv, D_ // 32, generator=g) < 0.5).to(torch.uint8).to(DEV)
    if mode == "codebook":
        qs = torch.randn(Hkv, 256, D_, generator=g)
        qs[:, :, :8] *= 4
        Cq = torch.einsum("hnd,hne->hde", qs, qs).to(DEV)
        CT = Cq.transpose(1, 2).contiguous()
        co = (torch.rand(Hkv, D_, generator=g) + 0.1).to(DEV)
    for b in range(B):
        n = lens[b]
        kv = torch.randn(n, 2 * Hkv, D_, generator=g)
        kv[:, :Hkv, :4] *= 6                                           # outlier channels around the mean (as run_case)
        kv[:, :Hkv] += mu
        kv = kv.bfloat16().to(DEV)
        sl = slots_of(bt, b, n)
        if mode == "static":
            K.v2_write(kv, sl, mask, 32, pool, P, fmt=fmt, kmu=kmu)
        else:
            K.cb_write(kv, sl, CT, co, kind, pool, P, kmu=kmu)
    if mode == "static":
        Kp, Vp = K.dequant_v2_pages(pool, P, torch.float64, fmt=fmt, Dh=D_)
    else:
        Kp, Vp = K.dequant_cb_pages(pool, P, kind, torch.float64, Dh=D_)
    Lmax = max(max(lens), 1)
    npg = (Lmax + P - 1) // P
    Kc = Kp[bt[:, :npg].long()].reshape(B, npg * P, Hkv, D_)
    Vc = Vp[bt[:, :npg].long()].reshape(B, npg * P, Hkv, D_)
    q = torch.randn(B, HQ, D_, generator=g).bfloat16().to(DEV)
    scale = 1 / math.sqrt(D_)
    tails = tail = tail_len = tail_idx = None
    tails_u = None
    if win is not None:
        W, R = win
        raw = torch.randn(R, W, 2 * Hkv, D_, generator=g)
        raw[:, :, :Hkv] += mu
        raw = raw.bfloat16()
        ring = raw.float()
        ring[:, :, :Hkv] -= mu
        ring = ring.bfloat16().to(DEV)                                 # BF16 slots: bf16(k - mu)
        tail_idx = torch.randperm(R, generator=g)[:B].int().to(DEV)
        tl = torch.randint(0, W + 1, (B,), generator=g).int()
        tail_len = tl.to(DEV)
        tails = [(ring[int(tail_idx[b]), :int(tl[b]), :Hkv], ring[int(tail_idx[b]), :int(tl[b]), Hkv:]) for b in range(B)]
        tail = ring
    if mode == "static":
        km, vm = mask[:Hkv].bool(), mask[Hkv:].bool()
        qr = sel_rot(q.double().view(B, Hkv, G, D_).transpose(1, 2), km, 32).transpose(1, 2).reshape(B, HQ, D_)
        tails_r = None if tails is None else [(sel_rot(t[0].double(), km, 32), sel_rot(t[1].double(), vm, 32)) for t in tails]
        o = ref_attn(qr, Kc, Vc, lens, scale, tails_r, kv_start)
        ref = sel_rot(o.view(B, Hkv, G, D_).transpose(1, 2), vm, 32).transpose(1, 2).reshape(B, HQ, D_)
        sd = sdpa_err(qr, Kc, Vc, lens, scale, o, tails_r, kv_start)
        # shift invariance in float64: original-basis keys of the cache (stored -> original) + mu
        Ko = sel_rot(Kc, km, 32) + mu.double().to(DEV)
        tails_u = None if tails is None else [(t[0].double() + mu.double().to(DEV), t[1].double()) for t in tails]
        ou = ref_attn(q.double(), Ko, sel_rot(Vc, vm, 32), lens, scale, tails_u, kv_start)
    else:
        ref = ref_attn(q.double(), Kc, Vc, lens, scale, tails, kv_start)
        sd = sdpa_err(q.double(), Kc, Vc, lens, scale, ref, tails, kv_start)
        tails_u = None if tails is None else [(t[0].double() + mu.double().to(DEV), t[1].double()) for t in tails]
        ou = ref_attn(q.double(), Kc + mu.double().to(DEV), Vc, lens, scale, tails_u, kv_start)
    seq = torch.tensor(lens, dtype=torch.int32, device=DEV)
    out = torch.full((B, HQ * D_), float("nan"), dtype=torch.bfloat16, device=DEV)
    ns, ch = (K.plan_v2(B, Lmax, Hkv, P, fmt=fmt, G=G, Dh=D_) if mode == "static"
              else K.plan_cb(B, Lmax, Hkv, P, kind, G=G, Dh=D_))
    gfmt = fmt if mode == "static" else kind
    if mode == "static" or (kind in K.K2 and (G != 4 or D_ != 128)):
        HV, GT = K.vheads(Hkv, G, D_, gfmt), K.tile_of(K.v2_variant(gfmt, G, D_))
    else:
        HV, GT = Hkv, 4
    ws = K.Workspace(B, HV, B * HV * (ns + 1), DEV, GT, D_)
    tkw = {} if tail is None else dict(tail=tail, tail_len=tail_len, tail_idx=tail_idx, tail_bs=tail.stride(0),
                                       tail_ts=tail.stride(1))
    if mode == "static":
        K.v2_attn(q, pool, bt, seq, out, ns, ch, ws, scale * LOG2E, kmask=mask[:Hkv], vmask=mask[Hkv:], rb=32,
                  fmt=fmt, kv_start=kv_start, **tkw)
    else:
        K.cb_attn(q, pool, bt, seq, out, ns, ch, ws, scale * LOG2E, kind, kv_start=kv_start, **tkw)
    torch.cuda.synchronize()
    valid = torch.tensor([lens[b] > 0 or (tails is not None and tails[b][0].shape[0] > 0) for b in range(B)], device=DEV)
    err = (out.view(B, HQ, D_).double() - ref).abs()
    e = err[valid].max().item() if valid.any() else 0.0
    m = ref[valid].abs().max().item() if valid.any() else 1.0
    lim = max(2 * sd / m, 4e-3)
    good = e / m <= lim and not torch.isnan(out.view(B, HQ, D_)[valid]).any()
    shift = ((ou - ref).abs()[valid].max().item() / m) if valid.any() else 0.0
    return good and shift < 1e-9, e / m, lim, shift


def test_attention(quick=False):
    cases = []
    lens_set = ([1], [300, 4101, 17], [0, 513, 64]) if quick else ([1], [77], [300, 4101, 17], [1000] * 5, [0, 513, 64], [33000])
    for mode, kw in (("static", dict(fmt="int2a")), ("codebook", dict(kind="konlya2")),
                     ("static", dict(fmt="int4a")), ("codebook", dict(kind="konlya"))):
        for lens in lens_set:
            for win in (None, (128, 9)):
                cases.append(dict(mode=mode, lens=lens, win=win, **kw))
        for (G, Dh, Hkv) in ((6, 256, 4), (8, 128, 8)):
            for lens in ([300, 4101, 17],):
                for win in (None, (132, 9)):
                    cases.append(dict(mode=mode, lens=lens, win=win, G=G, Dh=Dh, Hkv=Hkv, kv_start=4, **kw))
    fails = 0
    worst_shift = 0.0
    for i, c in enumerate(cases):
        good, e, lim, shift = run_case_c(seed=2000 + i, **c)
        worst_shift = max(worst_shift, shift)
        fails += not good
        name = c["mode"] + "/" + (c.get("fmt") or c.get("kind")) + (f" G{c['G']} D{c['Dh']}" if "G" in c else "")
        print(f"{'ok ' if good else 'FAIL'} centered decode {name:22s} lens={str(c['lens'])[:24]:24s} win={c.get('win')} "
              f"err/max={e:.2e} (lim {lim:.2e}) shift-invariance rel {shift:.1e}", flush=True)
    print(f"centered decode attention vs float64: {len(cases) - fails}/{len(cases)} passed; worst float64 shift-invariance "
          f"difference {worst_shift:.1e}")
    return fails == 0


def test_identity_window():
    """no quantization: every key in the BF16 window (empty paged body); tail K rows k vs k - mu (exact grid)."""
    g = torch.Generator().manual_seed(71)
    worst, n = 0.0, 0
    for (G, Dh, Hkv, kind) in ((4, 128, 8, "konlya2"), (6, 256, 4, "konlya2"), (4, 128, 8, "int2a")):
        B, W = 5, 128
        num_pages = 8
        chunk = K.cb_chunk_bytes(kind, P, Dh) if kind != "int2a" else K.v2_chunk_bytes(P, Dh, "int2a")
        pool = torch.zeros(num_pages, Hkv, chunk, dtype=torch.uint8, device=DEV)
        bt = torch.zeros(B, 1, dtype=torch.int32, device=DEV)
        seq = torch.zeros(B, dtype=torch.int32, device=DEV)
        x = grid_rows(B * W, 2 * Hkv, Dh, g).view(B, W, 2 * Hkv, Dh)
        mu = grid_mu(Hkv, Dh, g)
        xc = x.clone()
        xc[:, :, :Hkv] -= mu
        q = torch.randn(B, G * Hkv, Dh, generator=g).bfloat16().to(DEV)
        outs = []
        for ring in (x, xc):
            ring = ring.bfloat16().to(DEV).contiguous()
            tl = torch.full((B,), W, dtype=torch.int32, device=DEV)
            ti = torch.arange(B, dtype=torch.int32, device=DEV)
            out = torch.empty(B, G * Hkv * Dh, dtype=torch.bfloat16, device=DEV)
            if kind == "int2a":
                ns, ch = K.plan_v2(B, W, Hkv, P, fmt="int2a", G=G, Dh=Dh)
                ws = K.Workspace(B, K.vheads(Hkv, G, Dh, "int2a"), B * K.vheads(Hkv, G, Dh, "int2a") * (ns + 1), DEV,
                                 K.tile_of(K.v2_variant("int2a", G, Dh)), Dh)
                K.v2_attn(q, pool, bt, seq, out, ns, ch, ws, LOG2E / math.sqrt(Dh), fmt="int2a", tail=ring,
                          tail_len=tl, tail_idx=ti, tail_bs=ring.stride(0), tail_ts=ring.stride(1))
            else:
                ns, ch = K.plan_cb(B, W, Hkv, P, kind, G=G, Dh=Dh)
                HV, GT = (K.vheads(Hkv, G, Dh, kind), K.tile_of(K.v2_variant(kind, G, Dh))) if (G != 4 or Dh != 128) else (Hkv, 4)
                ws = K.Workspace(B, HV, B * HV * (ns + 1), DEV, GT, Dh)
                K.cb_attn(q, pool, bt, seq, out, ns, ch, ws, LOG2E / math.sqrt(Dh), kind, tail=ring, tail_len=tl,
                          tail_idx=ti, tail_bs=ring.stride(0), tail_ts=ring.stride(1))
            torch.cuda.synchronize()
            outs.append(out.float())
        d = ((outs[0] - outs[1]).abs().max() / outs[0].abs().max()).item()
        worst = max(worst, d)
        n += 1
        print(f"  identity ({kind}, G{G}, D{Dh}): max |o(k) - o(k - mu)| / max |o| = {d:.2e}")
    ok = worst <= 2 ** -7          # bf16 output rounding of the same fp32 attention (logits shifted by q.mu)
    print(f"{'ok ' if ok else 'FAIL'} identity (BF16 window only, no quantization): centering changes the decode output "
          f"by at most {worst:.2e} (bf16 output rounding)")
    return ok


def main():
    quick = "--quick" in sys.argv
    ok = True
    ok &= test_exact_grid()
    ok &= test_vs_sim()
    ok &= test_attention(quick)
    ok &= test_identity_window()
    print("ALL OK" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
