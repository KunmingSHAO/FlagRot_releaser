// shared epilogue of the v2 / v3 consumers: normalize, store o (bf16) and lse
  l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
  l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
  constexpr float LN2 = 0.6931471805599453f;
  if (v0) {
    __nv_bfloat16* dst = p.out + tq0 * p.o_ts + hq0 * D;
#pragma unroll
    for (int j = 0; j < 16; ++j)
      *reinterpret_cast<__nv_bfloat162*>(dst + 8 * j + 2 * tq4) = __floats2bfloat162_rn(o[4 * j] * i0, o[4 * j + 1] * i0);
    if (tq4 == 0) p.lse[static_cast<long long>(hq0) * p.nq + tq0] = l0 > 0.f ? (m0 + __log2f(l0) - POFF) * LN2 : NEG_INF;
  }
  if (v1) {
    __nv_bfloat16* dst = p.out + tq1 * p.o_ts + hq1 * D;
#pragma unroll
    for (int j = 0; j < 16; ++j)
      *reinterpret_cast<__nv_bfloat162*>(dst + 8 * j + 2 * tq4) =
          __floats2bfloat162_rn(o[4 * j + 2] * i1, o[4 * j + 3] * i1);
    if (tq4 == 0) p.lse[static_cast<long long>(hq1) * p.nq + tq1] = l1 > 0.f ? (m1 + __log2f(l1) - POFF) * LN2 : NEG_INF;
  }
