// shared by the v2 / v3 consumers (included inside the block loop): mask, online softmax, P -> E4M3, FP8 PV,
// promotion.  In: sacc[32] raw scores, fac0 / fac1, n, kfull, pos0 / pos1, sv, tb; updates m*, l*, o.
      const int n0 = n * BN;
      if (n0 + BN > kfull) {
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int key = n0 + 8 * j + 2 * tq4 + e;
            if (key > pos0 - p.shift || key >= p.nk) sacc[4 * j + e] = NEG_INF;
            if (key > pos1 - p.shift || key >= p.nk) sacc[4 * j + 2 + e] = NEG_INF;
          }
      }
      // softmax on raw scores: x = S fac (fac > 0), p = 2^(S fac - m + 8)
      float mx0 = NEG_INF, mx1 = NEG_INF;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        mx0 = fmaxf(mx0, fmaxf(sacc[4 * j + 0], sacc[4 * j + 1]));
        mx1 = fmaxf(mx1, fmaxf(sacc[4 * j + 2], sacc[4 * j + 3]));
      }
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
      const float mn0 = fmaxf(m0, mx0 * fac0), mn1 = fmaxf(m1, mx1 * fac1);
      const float mu0 = mn0 == NEG_INF ? 0.f : mn0, mu1 = mn1 == NEG_INF ? 0.f : mn1;
      const float al0 = ex2(m0 - mu0), al1 = ex2(m1 - mu1);
      m0 = mn0;
      m1 = mn1;
      const float sh0 = POFF - mu0, sh1 = POFF - mu1;
      float ps0 = 0.f, ps1 = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float pa = ex2(fmaf(sacc[4 * j + e], fac0, sh0)), pb = ex2(fmaf(sacc[4 * j + 2 + e], fac1, sh1));
          sacc[4 * j + e] = pa;
          sacc[4 * j + 2 + e] = pb;
          ps0 += pa;
          ps1 += pb;
        }
      }
      l0 = l0 * al0 + ps0;
      l1 = l1 * al1 + ps1;
      uint32_t pa[2][4];
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const int j = 4 * kk;
        pa[kk][0] = pack4(sacc[4 * j + 0], sacc[4 * j + 1], sacc[4 * (j + 1) + 0], sacc[4 * (j + 1) + 1]);
        pa[kk][1] = pack4(sacc[4 * j + 2], sacc[4 * j + 3], sacc[4 * (j + 1) + 2], sacc[4 * (j + 1) + 3]);
        pa[kk][2] = pack4(sacc[4 * (j + 2) + 0], sacc[4 * (j + 2) + 1], sacc[4 * (j + 3) + 0], sacc[4 * (j + 3) + 1]);
        pa[kk][3] = pack4(sacc[4 * (j + 2) + 2], sacc[4 * (j + 2) + 3], sacc[4 * (j + 3) + 2], sacc[4 * (j + 3) + 3]);
      }
      float ot[64];
      PP_TURN();
      wg_fence();
      wg_rs_n128_0(ot, pa[0], sdesc(sv));
      wg_rs_n128_1(ot, pa[1], sdesc(sv + 4096));
      wg_commit();
      PP_PASS();
      wg_wait_all();
      acc_sync(ot);
      if (lane == 0) mbar_arrive(tile_empty + 8 * tb);
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        o[4 * j + 0] = fmaf(o[4 * j + 0], al0, ot[4 * j + 0]);
        o[4 * j + 1] = fmaf(o[4 * j + 1], al0, ot[4 * j + 1]);
        o[4 * j + 2] = fmaf(o[4 * j + 2], al1, ot[4 * j + 2]);
        o[4 * j + 3] = fmaf(o[4 * j + 3], al1, ot[4 * j + 3]);
      }
