// Split-K merge shared by the static v2 kernel (attn_decode_v2.cu) and the codebook kernels (attn_codebook_attn.cuh),
//   Included inside their anonymous namespaces after DEVI, FULL, NEG_INF, ex2, G (q heads per kv head) and
// D (head dim) are defined.
#pragma once

// Where the split partials of one (sequence, kv head) live: part s2 < nbody (body CTAs) in slot b0 + stride s2 (+ adj
// for s2 = 0), part s2 >= nbody (BF16 staging window) in slot t0 + s2 - nbody; slot = one [G][D] block of opart and
// one [G] block of mlpart.  Split mode: b0 = (b Hkv + kvh) nsplit, stride 1 (window parts follow the body parts);
// stream-K mode: see sk_pair.
struct PartMap {
  int b0, stride, adj, nbody, t0;
  DEVI int slot(int s2) const { return s2 < nbody ? b0 + stride * s2 + (s2 == 0 ? adj : 0) : t0 + (s2 - nbody); }
};

// merge of the nparts partials of a (sequence, kv head) (unnormalized o [G][D], (max, sum) [G]) into x: warp h = q
// head h, lane: d = 4 lane .. 4 lane + 3.  Latency-bound (one CTA runs it while the others have finished): every warp
// computes the weights of its own head (the (max, sum) pairs read once, kept in registers for nparts <= 128, shuffled
// to the lanes), and the float4 partials are read with 16 loads in flight per thread, the first 16 together with the
// (max, sum) pairs.  No shared memory,
// no barrier.
template <int NB, bool EARLY, class Params>  // NB partial rows in flight per thread; EARLY: first rows loaded with the
                                              // (max, sum) pairs; Params: any struct with float* opart / mlpart
DEVI void merge_warp(const Params& p, const PartMap& pm, int nparts, float (&x)[4]) {
  const int lane = threadIdx.x & 31, h = threadIdx.x >> 5;
  const float2* mlp = reinterpret_cast<const float2*>(p.mlpart);
  const float4* op = reinterpret_cast<const float4*>(p.opart) + h * (D / 4) + lane;
  constexpr int R = 4;   // (max, sum) pairs per lane kept in registers (nparts <= 128; more: re-read)
  // the first NB partial rows are loaded together with the (max, sum) pairs: they do not depend on the weights
  float4 v[NB];
  if constexpr (EARLY) {
#pragma unroll
    for (int k = 0; k < NB; ++k)
      v[k] = k < nparts ? __ldcg(op + static_cast<size_t>(pm.slot(k)) * (G * D / 4)) : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  float2 ml[R];
#pragma unroll
  for (int k = 0; k < R; ++k) {
    const int s2 = lane + 32 * k;
    ml[k] = s2 < nparts ? __ldcg(mlp + pm.slot(s2) * G + h) : make_float2(NEG_INF, 0.f);
  }
  float M = NEG_INF;
#pragma unroll
  for (int k = 0; k < R; ++k) M = fmaxf(M, ml[k].x);
  for (int s2 = lane + 32 * R; s2 < nparts; s2 += 32) M = fmaxf(M, __ldcg(mlp + pm.slot(s2) * G + h).x);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(FULL, M, o));
  const float Ms = M == NEG_INF ? 0.f : M;
  float wr[R], l = 0.f;
#pragma unroll
  for (int k = 0; k < R; ++k) {
    wr[k] = ml[k].x == NEG_INF ? 0.f : ex2(ml[k].x - Ms);
    l += wr[k] * ml[k].y;
  }
  for (int s2 = lane + 32 * R; s2 < nparts; s2 += 32) {
    const float2 m2 = __ldcg(mlp + pm.slot(s2) * G + h);
    l += (m2.x == NEG_INF ? 0.f : ex2(m2.x - Ms)) * m2.y;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(FULL, l, o);
  const float rl = l > 0.f ? 1.f / l : 0.f;
  auto weight = [&](int s2) {  // weight of part s2 (s2 warp-uniform)
    if (s2 < 32 * R) {
      const int kk = s2 >> 5;
      return __shfl_sync(FULL, kk == 0 ? wr[0] : kk == 1 ? wr[1] : kk == 2 ? wr[2] : wr[3], s2 & 31);
    }
    const float2 m2 = s2 < nparts ? __ldcg(mlp + pm.slot(s2) * G + h) : make_float2(NEG_INF, 0.f);
    return m2.x == NEG_INF ? 0.f : ex2(m2.x - Ms);
  };
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int s0 = 0; s0 < nparts; s0 += NB) {  // (s0 warp-uniform)
    if (s0 > 0 || !EARLY) {
#pragma unroll
      for (int k = 0; k < NB; ++k)
        v[k] = s0 + k < nparts ? __ldcg(op + static_cast<size_t>(pm.slot(s0 + k)) * (G * D / 4)) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int k = 0; k < NB; ++k) {
      const float wt = weight(s0 + k);
      acc.x = fmaf(wt, v[k].x, acc.x); acc.y = fmaf(wt, v[k].y, acc.y);
      acc.z = fmaf(wt, v[k].z, acc.z); acc.w = fmaf(wt, v[k].w, acc.w);
    }
  }
  x[0] = acc.x * rl; x[1] = acc.y * rl; x[2] = acc.z * rl; x[3] = acc.w * rl;
}

// merge_warp for the generalized kernels (attn_decode_v2g.cuh, k_cbkg of attn_codebook_attn.cuh): a slot
// holds GT q heads x DD = 128 NH2 dims ([GT][DD] floats of opart, [GT] (max, sum) pairs of mlpart); the warp merges q
// head h of the slot; x[s] = this lane's d = 128 s + 4 lane .. + 3.  Same arithmetic per element as merge_warp.
template <int NB, int GT, int NH2, class Params>
DEVI void merge_warp_g(const Params& p, const PartMap& pm, int nparts, int h, float (&x)[NH2][4]) {
  constexpr int DD = 128 * NH2;
  const int lane = threadIdx.x & 31;
  const float2* mlp = reinterpret_cast<const float2*>(p.mlpart);
  const float4* op = reinterpret_cast<const float4*>(p.opart) + h * (DD / 4) + lane;
  constexpr int R = 4;
  float2 ml[R];
#pragma unroll
  for (int k = 0; k < R; ++k) {
    const int s2 = lane + 32 * k;
    ml[k] = s2 < nparts ? __ldcg(mlp + pm.slot(s2) * GT + h) : make_float2(NEG_INF, 0.f);
  }
  float M = NEG_INF;
#pragma unroll
  for (int k = 0; k < R; ++k) M = fmaxf(M, ml[k].x);
  for (int s2 = lane + 32 * R; s2 < nparts; s2 += 32) M = fmaxf(M, __ldcg(mlp + pm.slot(s2) * GT + h).x);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(FULL, M, o));
  const float Ms = M == NEG_INF ? 0.f : M;
  float wr[R], l = 0.f;
#pragma unroll
  for (int k = 0; k < R; ++k) {
    wr[k] = ml[k].x == NEG_INF ? 0.f : ex2(ml[k].x - Ms);
    l += wr[k] * ml[k].y;
  }
  for (int s2 = lane + 32 * R; s2 < nparts; s2 += 32) {
    const float2 m2 = __ldcg(mlp + pm.slot(s2) * GT + h);
    l += (m2.x == NEG_INF ? 0.f : ex2(m2.x - Ms)) * m2.y;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(FULL, l, o);
  const float rl = l > 0.f ? 1.f / l : 0.f;
  auto weight = [&](int s2) {
    if (s2 < 32 * R) {
      const int kk = s2 >> 5;
      return __shfl_sync(FULL, kk == 0 ? wr[0] : kk == 1 ? wr[1] : kk == 2 ? wr[2] : wr[3], s2 & 31);
    }
    const float2 m2 = s2 < nparts ? __ldcg(mlp + pm.slot(s2) * GT + h) : make_float2(NEG_INF, 0.f);
    return m2.x == NEG_INF ? 0.f : ex2(m2.x - Ms);
  };
  float4 acc[NH2];
#pragma unroll
  for (int s = 0; s < NH2; ++s) acc[s] = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int s0 = 0; s0 < nparts; s0 += NB) {  // (s0 warp-uniform)
    float4 v[NB][NH2];
#pragma unroll
    for (int k = 0; k < NB; ++k)
#pragma unroll
      for (int s = 0; s < NH2; ++s)
        v[k][s] = s0 + k < nparts ? __ldcg(op + static_cast<size_t>(pm.slot(s0 + k)) * (GT * DD / 4) + 32 * s)
                                  : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int k = 0; k < NB; ++k) {
      const float wt = weight(s0 + k);
#pragma unroll
      for (int s = 0; s < NH2; ++s) {
        acc[s].x = fmaf(wt, v[k][s].x, acc[s].x); acc[s].y = fmaf(wt, v[k][s].y, acc[s].y);
        acc[s].z = fmaf(wt, v[k][s].z, acc[s].z); acc[s].w = fmaf(wt, v[k][s].w, acc[s].w);
      }
    }
  }
#pragma unroll
  for (int s = 0; s < NH2; ++s) {
    x[s][0] = acc[s].x * rl; x[s][1] = acc[s].y * rl; x[s][2] = acc[s].z * rl; x[s][3] = acc[s].w * rl;
  }
}

