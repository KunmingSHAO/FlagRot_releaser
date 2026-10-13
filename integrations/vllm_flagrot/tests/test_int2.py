"""Phase 3a: INT2 KV pages (int2a:32 stored as blockflag int2ah:32; static "int2a" and key-only 2x2 "konlya2").

  CUDA_VISIBLE_DEVICES=7 python integrations/vllm_flagrot/tests/test_int2.py [--quick]

Checks (vLLM env):
  1. static writer v1 (fr_v2_write, int2a) + the plugin's INT2 page parser == blockflag int2ah:32 (fake_asym_dep, bits 2)
     on the same (FlagRot-rotated) vectors, incl. boundary groups (range excluding 0, all zero, FP16-subnormal scales,
     x300, constant, single non-zero element); head dim 128 and 256;
  2. static writer v3 (fr_w3_static_write, asym 2, paths 0 / 1 / 2) == fr_head_fwht + fr_v2_write, byte for byte;
  3. key-only 2x2 writer v1 (fr_cb_write, konlya2, head dim 128 and 256) == kernels/codebook_ref.encode(cfg konlya2,
     qk asym2): codes, scale words (zero points) and the H32 / whole-head flags, keys and values;
     codebook_ref (asym2) reconstruction == blockflag dyn_quant(int2a:32, cq+f+konly) (the simulation);
     writer v3 (fr_cb_write3) == v1, byte for byte (one stream and K / V on two streams);
  4. decode attention (static v2 variant 264, the generalized GQA / head-256 kernel, k_cbk) vs float64 attention over
     the parsed pages, with BF16 windows, shared block-table rows and kv_start (sinks), criterion of test_kernels.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.append(str(HERE.parents[2] / "kernels"))
import test_kernels as TK  # noqa: E402
from test_kernels import DEV, HKV, P, K, sel_rot, _edge_rows  # noqa: E402

D = 128


def edge_rows(T, g, Dh=128, Hkv=HKV):
    if Dh == 128 and Hkv == HKV:
        return _edge_rows(T, g)
    a = torch.cat([_edge_rows(T, g).float() for _ in range(Dh // 128)], -1)[:, :2 * Hkv] if Hkv <= HKV else None
    return a.bfloat16().to(DEV)


def test_int2a_writer():
    from blockflag import formats as FM
    g = torch.Generator().manual_seed(31)
    ok = True
    for Dh in (128, 256):
        for rot in (False, True):
            T = 333
            kv = edge_rows(T, g, Dh)
            mask = ((torch.rand(2 * HKV, Dh // 32, generator=g) < 0.5) & rot).to(torch.uint8).to(DEV)
            npg = (T + P - 1) // P + 3
            pool = torch.randint(0, 256, (npg, HKV, K.v2_chunk_bytes(P, Dh, "int2a")), dtype=torch.uint8,
                                 generator=g).to(DEV)
            slots = torch.randperm(npg * P, generator=g)[:T].to(DEV)
            K.v2_write(kv, slots, mask, 32, pool, P, fmt="int2a")
            Kp, Vp = K.dequant_v2_pages(pool, P, torch.float32, fmt="int2a", Dh=Dh)
            got = torch.cat([Kp.reshape(-1, HKV, Dh), Vp.reshape(-1, HKV, Dh)], 1)[slots]
            x = sel_rot(kv.float(), mask.bool(), 32)
            ref = FM.fake_asym_dep(x, 32, True, 2)
            diff = (got - ref).abs()
            nbad = int((diff > 1e-6 * ref.abs().amax()).sum())
            this = nbad == 0 if not rot else nbad <= 1e-3 * ref.numel()
            ok &= this
            print(f"{'ok ' if this else 'FAIL'} int2a writer vs blockflag int2ah:32 (D {Dh}, rotation {rot}): {nbad} of "
                  f"{ref.numel()} values differ, max |diff| {diff.max().item():.3g}")
    return ok


def test_v3_static():
    g = torch.Generator().manual_seed(37)
    ok, n = True, 0
    for T in (1, 7, 64, 129, 257, 1000):
        kvb = _edge_rows(T, g)
        base = torch.zeros(T, 2 * HKV + 8, D, dtype=torch.bfloat16, device=DEV)
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
            for hmode in ("none", "rand"):
                for bmode in ("zero", "rand"):
                    hfull = (torch.rand(2 * HKV, generator=g) < 0.5).to(torch.uint8).to(DEV) if hmode == "rand" else None
                    mask = ((torch.rand(2 * HKV, D // 32, generator=g) < 0.5) & (bmode == "rand")).to(torch.uint8).to(DEV)
                    pool0 = torch.randint(0, 256, (npg, HKV, K.v2_chunk_bytes(P, D, "int2a")), dtype=torch.uint8,
                                          generator=g).to(DEV)
                    p1 = pool0.clone()
                    x = K.head_fwht(kv, hfull, 1 / 16) if hfull is not None else kv
                    K.v2_write(x, slots, mask, 32, p1, P, fmt="int2a")
                    for path in (0, 1, 2):
                        p3 = pool0.clone()
                        K.w3_static_write(kv, slots, mask, hfull, p3, P, fmt="int2a", path=path)
                        torch.cuda.synchronize()
                        this = torch.equal(p1, p3)
                        ok &= this
                        n += 1
                        if not this:
                            print(f"FAIL v3 static int2a T={T} slots={smode} hfull={hmode} blocks={bmode} path={path}: "
                                  f"{int((p1 != p3).sum())} bytes differ")
    print(f"{'ok ' if ok else 'FAIL'} static writer v3 == v1, int2a ({n} pool comparisons)")
    return ok


def _cq(Hkv, Dh, g):
    qs = torch.randn(Hkv, 256, Dh, generator=g)
    qs[:, :, :8] *= 4
    return torch.einsum("hnd,hne->hde", qs, qs) / 256


def test_konlya2_writer():
    import codebook_ref as CR
    from blockflag import formats as FM
    g = torch.Generator().manual_seed(41)
    ok = True
    for Dh in (128, 256):
        Hkv = HKV if Dh == 128 else 4
        T = 300
        kv = torch.randn(T, 2 * Hkv, Dh, generator=g)
        kv[:, :Hkv, :4] *= 6
        grp = kv.view(T, 2 * Hkv, Dh // 32, 32)
        sel = torch.rand(T, 2 * Hkv, Dh // 32, generator=g)
        grp[sel < 0.03] = grp[sel < 0.03].abs() + 0.5
        grp[(sel >= 0.03) & (sel < 0.05)] = 0.0
        grp[(sel >= 0.05) & (sel < 0.07)] *= 1e-5
        kv = kv.bfloat16().to(DEV)
        Cq = _cq(Hkv, Dh, g).to(DEV)
        co = (torch.rand(Hkv, Dh, generator=g) + 0.1).to(DEV)
        npg = (T + P - 1) // P + 3
        pool = torch.randint(0, 256, (npg, Hkv, K.cb_chunk_bytes("konlya2", P, Dh)), dtype=torch.uint8, generator=g).to(DEV)
        slots = torch.randperm(npg * P, generator=g)[:T].to(DEV)
        K.cb_write(kv, slots, Cq.transpose(1, 2).contiguous(), co, "konlya2", pool, P)
        torch.cuda.synchronize()
        codes, mag, sbits = K.parse_int2_pages(pool, P, Dh)                            # [Np, H, P, 2, Dh]
        codes = codes.transpose(1, 2).reshape(-1, Hkv, 2, Dh)[slots]                # [T, H, 2, Dh]
        sbits = sbits.transpose(1, 2).reshape(-1, Hkv, 2, Dh // 32)[slots].to(torch.int32) & 0xFFFF
        ek = CR.encode(kv[:, :Hkv].float(), Cq, co, "k", cfg="konlya2", qk="asym2")
        ev = CR.encode(kv[:, Hkv:].float(), Cq, co, "v", cfg="konlya2", qk="asym2")
        bad = 0
        for kind, e in ((0, ek), (1, ev)):
            word = e["s"].half().view(torch.int16).to(torch.int32) & 0xFFFF                 # [T, H, ng]
            if kind == 0:
                word = word | (((e["mode"] == 0)[..., None] & (e["local"] == 1)).to(torch.int32) << 15)
            else:
                word[..., 0] |= (ek["mode"] == 2).to(torch.int32) << 15
            z = word & 3
            cm = e["q"].to(torch.int32) - z.repeat_interleave(32, -1)
            bad += int((codes[:, :, kind].to(torch.int32) != cm).sum()) + int((sbits[:, :, kind] != word).sum())
        this = bad == 0
        ok &= this
        print(f"{'ok ' if this else 'FAIL'} konlya2 writer v1 vs codebook_ref asym2 (D {Dh}): {bad} mismatching codes / "
              f"scale words")
        if Dh == 128:   # writer v3 == v1
            C1, beta = K.w3_calib(Cq.cpu())
            C1, beta = C1.to(DEV), beta.to(DEV)
            for split in (False, True):
                p3 = torch.randint(0, 256, pool.shape, dtype=torch.uint8, generator=g).to(DEV)
                p1 = p3.clone()
                K.cb_write(kv, slots, Cq.transpose(1, 2).contiguous(), co, "konlya2", p1, P)
                K.cb_write3(kv, slots, Cq.transpose(1, 2).contiguous(), co, C1, beta, "konlya2", p3, P, split=split)
                torch.cuda.synchronize()
                this = torch.equal(p1, p3)
                ok &= this
                print(f"{'ok ' if this else 'FAIL'} konlya2 writer v3 == v1 (split {split}): "
                      f"{int((p1 != p3).sum())} bytes differ")
            # contiguous slots (whole V^T words)
            sl2 = (torch.arange(T) + 2 * P).to(torch.int64).to(DEV)
            npg2 = (T + 3 * P) // P + 1
            p3 = torch.randint(0, 256, (npg2, Hkv, pool.shape[2]), dtype=torch.uint8, generator=g).to(DEV)
            p1 = p3.clone()
            K.cb_write(kv, sl2, Cq.transpose(1, 2).contiguous(), co, "konlya2", p1, P)
            K.cb_write3(kv, sl2, Cq.transpose(1, 2).contiguous(), co, C1, beta, "konlya2", p3, P, split=False)
            torch.cuda.synchronize()
            this = torch.equal(p1, p3)
            ok &= this
            print(f"{'ok ' if this else 'FAIL'} konlya2 writer v3 == v1 (contiguous slots): {int((p1 != p3).sum())} bytes differ")
    return ok


def test_ref_vs_sim():
    """codebook_ref (asym2) == blockflag dyn_quant(int2a:32, cq+f+konly): the reconstruction under the choice error form."""
    import codebook_ref as CR
    from blockflag import qattention as QA
    g = torch.Generator().manual_seed(43)
    T, Hkv = 2000, 8
    x = torch.randn(T, Hkv, D, generator=g)
    x[..., :4] *= 6
    x = x.bfloat16().float().to(DEV)
    Cq = _cq(Hkv, D, g).to(DEV)
    co = (torch.rand(Hkv, D, generator=g) + 0.1).to(DEV)
    QA.KV["dyn"] = dict(rule="cq", local=["H"], head=["F"], name="cq+f+konly")
    QA.KV["dynstat"] = {}
    ok = True
    for kind in ("k", "v"):
        try:
            QA.KV["cq"] = {0: Cq}
            QA.KV["co"] = {0: co}
            sim = QA.dyn_quant(x.permute(1, 0, 2)[None], "int2a:32", 32, kind, 0)[0].permute(1, 0, 2)
        except Exception as ex:  # noqa: BLE001
            print("  dyn_quant call failed:", repr(ex))
            return True
        if kind == "v":   # FlagRot (cq+f+konly): values plain int2a:32 in the simulation (kv_dyn 'only' k)
            from blockflag import formats as FM
            sim = FM.fake(x, "int2a:32")
        e = CR.encode(x, Cq, co, kind, cfg="konlya2", qk="asym2")
        nbad = int((sim - e["rec"]).abs().gt(0).sum())
        print(f"  ref vs sim ({kind}): {nbad} of {sim.numel()} reconstruction elements differ")
        ok &= nbad == 0
    print(f"{'ok ' if ok else 'FAIL'} codebook_ref asym2 == dyn_quant int2a:32")
    return ok


def attention_cases(quick=False):
    cases = []
    lens_set = ([1], [77], [300, 4101, 17], [1000] * 5, [33000], [0, 513, 64])
    if quick:
        lens_set = ([77], [300, 4101, 17], [0, 513, 64])
    for mode, fmt, kind in (("static", "int2a", None), ("codebook", None, "konlya2")):
        kw = dict(mode=mode, **({"fmt": fmt} if fmt else {"kind": kind}))
        for lens in lens_set:
            for win in (None, (32, 9), (256, 7)):
                cases.append(dict(lens=lens, win=win, **kw))
        cases.append(dict(lens=list(range(1, 200, 7)), bt_shared="copy", win=(64, 40), **kw))
        for (G, Dh, Hkv) in ((5, 128, 8), (8, 128, 8), (6, 256, 4), (12, 128, 4)):
            for lens in ([1], [300, 4101, 17], [0, 513, 64], [33000]):
                for win in (None, (64, 9)):
                    cases.append(dict(lens=lens, win=win, G=G, Dh=Dh, Hkv=Hkv, **kw))
        for (G, Dh, Hkv) in ((4, 128, 8), (6, 256, 4), (8, 128, 8)):
            for ks in (4, 37):
                for lens in ([ks + 1], [300, 4101, 17 + ks], [33000]):
                    for win in (None, (132, 9)):
                        cases.append(dict(lens=lens, win=win, kv_start=ks, G=G, Dh=Dh, Hkv=Hkv, **kw))
    return cases


def main():
    quick = "--quick" in sys.argv
    ok = True
    if "--attn-only" not in sys.argv:
        ok &= test_int2a_writer()
        ok &= test_v3_static()
        ok &= test_konlya2_writer()
        ok &= test_ref_vs_sim()
    fails = 0
    cases = attention_cases(quick)
    for i, c in enumerate(cases):
        r = TK.run_case(seed=1000 + i, **c)
        good, e, lim, ns, ch = r
        fails += not good
        name = c["mode"] + "/" + (c.get("fmt") or c.get("kind"))
        if "G" in c:
            name += f" G{c['G']} D{c['Dh']} Hkv{c['Hkv']}"
        if c.get("kv_start"):
            name += f" kv_start {c['kv_start']}"
        print(f"{'ok ' if good else 'FAIL'} {name:28s} lens={str(c['lens'])[:30]:30s} win={c.get('win')} "
              f"shared={c.get('bt_shared', False)} ns={ns} chunk={ch} err/max={e:.2e} (lim {lim:.2e})", flush=True)
    print(f"attention: {len(cases) - fails}/{len(cases)} passed")
    sys.exit(0 if ok and not fails else 1)


if __name__ == "__main__":
    main()
