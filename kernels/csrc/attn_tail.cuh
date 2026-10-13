// BF16 staging window ("tail") of the paged decode-attention kernels: the newest tl <= TAIL_MAX tokens of a sequence
// are not encoded yet and are kept in BF16.  A small kernel on a side stream, concurrent with the attention kernel,
// computes per (sequence, kv head, 32-token slice) the partial softmax of the 4 q heads on CUDA cores and stores it as
// one more split partial (slots ns_b .. ns_b + nslot - 1, after the body splits; empty slices store an empty partial)
// with a merge ticket: whichever CTA arrives last merges.  (Inside the attention kernel as extra CTAs, the tail code
// cost the main loop 1-2 % through register allocation / scheduling, measured; one CTA per sequence and kv head for
// the whole window took 13 us at B = 8, W = 256: latency-bound.)  Included by attn_decode_v2.cu (static INT4 v2) and attn_codebook_attn.cuh (codebook kernels); the
// including file provides DEVI, ex2, FULL, NEG_INF and includes this header inside its anonymous namespace (after
// cuda_bf16.h).
#pragma once

constexpr int TAIL_MAX = 256;            // largest staging window
constexpr int TAIL_TOK = 32;             // window tokens per tail CTA (one merge slot each)
constexpr int TAIL_SMEM_FLOATS = 4 * 128 + TAIL_TOK * 4 + 8 * 4 * 128;  // q | p [t][4] | PV slice partials

// bf16x8 (uint4) -> 8 floats
DEVI void bf8(const uint4 u, float (&f)[8]) {
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}

// partial softmax of the 4 q heads over tn <= TAIL_TOK window tokens.  qh: the 4 q rows (bf16 [4][128]) of the kv
// head; kt / vt: K / V row of the first token (token stride ts elements); scores in the log2 domain (qscale = softmax
// scale * log2 e).  Results in shared memory: o [4][128] (unnormalized, relative to m), ml [4][2] = (m, l); tn <= 0:
// an empty partial (m = -inf).  sm: TAIL_SMEM_FLOATS floats.  All 128 threads.
DEVI void tail_partial(const __nv_bfloat16* __restrict__ qh, float qscale, const __nv_bfloat16* __restrict__ kt,
                       const __nv_bfloat16* __restrict__ vt, long long ts, int tn, float* sm, float* o, float* ml) {
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  float* qs = sm;                     // [4][128]
  float* pp = sm + 4 * 128;           // [TAIL_TOK][4]: scores, then p
  float* part = pp + TAIL_TOK * 4;    // [8 slices][4][128]
  if (tn <= 0) {
    for (int i = tid; i < 4 * 128; i += 128) o[i] = 0.f;
    if (tid < 4) { ml[2 * tid] = NEG_INF; ml[2 * tid + 1] = 0.f; }
    __syncthreads();
    return;
  }
  // scores: 4 threads per token (32 dims each, 4 x 16-B loads issued before q is staged), reduced by shuffles
  const int tt = tid >> 2, qd = tid & 3;
  uint4 kv[4];
  if (tt < tn) {
    const uint4* kr = reinterpret_cast<const uint4*>(kt + tt * ts + 32 * qd);
#pragma unroll
    for (int c = 0; c < 4; ++c) kv[c] = __ldg(kr + c);
  }
  // V pieces of the PV phase: thread (8 dims d8, slice sl) takes tokens sl, sl + 8, sl + 16, sl + 24
  const int d8 = 8 * (tid & 15), sl = tid >> 4;
  uint4 vv[TAIL_TOK / 8];
#pragma unroll
  for (int i = 0; i < TAIL_TOK / 8; ++i) {
    const int t = sl + 8 * i;
    vv[i] = t < tn ? __ldg(reinterpret_cast<const uint4*>(vt + t * ts + d8)) : make_uint4(0, 0, 0, 0);
  }
  for (int i = tid; i < 4 * 128; i += 128) qs[i] = __bfloat162float(qh[i]) * qscale;
  __syncthreads();
  {
    float s[4] = {0.f, 0.f, 0.f, 0.f};
    if (tt < tn) {
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        float k[8];
        bf8(kv[c], k);
#pragma unroll
        for (int h = 0; h < 4; ++h) {
          const float4 qa = *reinterpret_cast<const float4*>(qs + h * 128 + 32 * qd + 8 * c);
          const float4 qb = *reinterpret_cast<const float4*>(qs + h * 128 + 32 * qd + 8 * c + 4);
          s[h] = fmaf(qa.x, k[0], fmaf(qa.y, k[1], fmaf(qa.z, k[2], fmaf(qa.w, k[3], s[h]))));
          s[h] = fmaf(qb.x, k[4], fmaf(qb.y, k[5], fmaf(qb.z, k[6], fmaf(qb.w, k[7], s[h]))));
        }
      }
    }
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      s[h] += __shfl_xor_sync(FULL, s[h], 1);
      s[h] += __shfl_xor_sync(FULL, s[h], 2);
    }
    if (qd == 0 && tt < TAIL_TOK) *reinterpret_cast<float4*>(pp + 4 * tt) = make_float4(s[0], s[1], s[2], s[3]);
  }
  __syncthreads();
  {  // softmax of head w over the window tokens (lane = token)
    const float x = lane < tn ? pp[4 * lane + w] : NEG_INF;
    float m = x;
#pragma unroll
    for (int k = 16; k > 0; k >>= 1) m = fmaxf(m, __shfl_xor_sync(FULL, m, k));
    const float e = lane < tn ? ex2(x - m) : 0.f;
    float l = e;
#pragma unroll
    for (int k = 16; k > 0; k >>= 1) l += __shfl_xor_sync(FULL, l, k);
    pp[4 * lane + w] = e;
    if (lane == 0) { ml[2 * w] = m; ml[2 * w + 1] = l; }
  }
  __syncthreads();
  {
    float acc[4][8];
#pragma unroll
    for (int h = 0; h < 4; ++h)
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[h][i] = 0.f;
#pragma unroll
    for (int j = 0; j < TAIL_TOK / 8; ++j) {
      const int t = sl + 8 * j;
      float v[8];
      bf8(vv[j], v);
      const float4 pw = *reinterpret_cast<const float4*>(pp + 4 * t);  // (0 for t >= tn)
      const float pr[4] = {pw.x, pw.y, pw.z, pw.w};
#pragma unroll
      for (int h = 0; h < 4; ++h)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[h][i] = fmaf(pr[h], v[i], acc[h][i]);
    }
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      *reinterpret_cast<float4*>(part + (sl * 4 + h) * 128 + d8) = make_float4(acc[h][0], acc[h][1], acc[h][2], acc[h][3]);
      *reinterpret_cast<float4*>(part + (sl * 4 + h) * 128 + d8 + 4) = make_float4(acc[h][4], acc[h][5], acc[h][6], acc[h][7]);
    }
  }
  __syncthreads();
#pragma unroll
  for (int h = 0; h < 4; ++h) {
    float x = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) x += part[(k * 4 + h) * 128 + tid];
    o[h * 128 + tid] = x;
  }
  __syncthreads();
}

// fork / join of the staging-window kernel on a side stream (runs concurrently with the attention kernel; works under
// CUDA graph capture: the side stream joins through the recorded events)
inline void tail_fork(cudaStream_t st, cudaStream_t& side, cudaEvent_t& e0, cudaEvent_t& e1) {
#ifdef ATTN_TAIL_SERIAL  // experiment: the window kernel in the same stream, before the attention kernel
  side = st;
  return;
#endif
  if (side == nullptr) {
    cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&e0, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&e1, cudaEventDisableTiming);
  }
  cudaEventRecord(e0, st);
  cudaStreamWaitEvent(side, e0, 0);
}
inline void tail_join(cudaStream_t st, cudaStream_t side, cudaEvent_t e1) {
#ifdef ATTN_TAIL_SERIAL
  return;
#endif
  cudaEventRecord(e1, side);
  cudaStreamWaitEvent(st, e1, 0);
}

// the whole window of one (sequence, kv head) in one CTA: 32-token slices, combined with an online softmax (the CTA
// runs concurrently with the attention kernel, so its latency is off the critical path); same outputs as tail_partial
DEVI void tail_window(const __nv_bfloat16* __restrict__ qh, float qscale, const __nv_bfloat16* __restrict__ kt,
                      const __nv_bfloat16* __restrict__ vt, long long ts, int tl, float* sm, float* o, float* ml) {
  __shared__ float sml[8];
  const int tid = threadIdx.x;
  float acc[4] = {0.f, 0.f, 0.f, 0.f}, M[4], Lh[4];
#pragma unroll
  for (int h = 0; h < 4; ++h) { M[h] = NEG_INF; Lh[h] = 0.f; }
  for (int t0 = 0; t0 < max(tl, 1); t0 += TAIL_TOK) {
    tail_partial(qh, qscale, kt + t0 * ts, vt + t0 * ts, ts, min(tl - t0, TAIL_TOK), sm, o, sml);
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      const float m = sml[2 * h], l = sml[2 * h + 1], Mn = fmaxf(M[h], m);
      const float a = M[h] == NEG_INF ? 0.f : ex2(M[h] - Mn), c = m == NEG_INF ? 0.f : ex2(m - Mn);
      acc[h] = acc[h] * a + o[h * 128 + tid] * c;
      Lh[h] = Lh[h] * a + l * c;
      M[h] = Mn;
    }
    __syncthreads();
  }
#pragma unroll
  for (int h = 0; h < 4; ++h) o[h * 128 + tid] = acc[h];
  if (tid < 4) { ml[2 * tid] = M[tid]; ml[2 * tid + 1] = Lh[tid]; }
  __syncthreads();
}

// ------------------------------------------------------------------ generalized window
// GT q heads (rows qh + h DD, the first nq valid, the others read as 0) x head dim DD = 128 NH2.  Same algorithm as
// tail_partial / tail_window (4 threads per token for the scores, 16 x 8 dims x 8 token slices for PV).
template <int GT, int NH2>
constexpr int tail_g_smem_floats() { return GT * 128 * NH2 + TAIL_TOK * GT + 8 * GT * 128 * NH2; }

template <int GT, int NH2>
DEVI void tail_partial_g(const __nv_bfloat16* __restrict__ qh, int nq, float qscale, const __nv_bfloat16* __restrict__ kt,
                         const __nv_bfloat16* __restrict__ vt, long long ts, int tn, float* sm, float* o, float* ml) {
  constexpr int DD = 128 * NH2;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  float* qs = sm;                      // [GT][DD]
  float* pp = sm + GT * DD;            // [TAIL_TOK][GT]
  float* part = pp + TAIL_TOK * GT;    // [8 slices][GT][DD]
  if (tn <= 0) {
    for (int i = tid; i < GT * DD; i += 128) o[i] = 0.f;
    if (tid < GT) { ml[2 * tid] = NEG_INF; ml[2 * tid + 1] = 0.f; }
    __syncthreads();
    return;
  }
  const int tt = tid >> 2, qd = tid & 3;  // scores: token tt, dims [32 NH2 qd, 32 NH2 (qd + 1))
  uint4 kv[4 * NH2];
  if (tt < tn) {
    const uint4* kr = reinterpret_cast<const uint4*>(kt + tt * ts + 32 * NH2 * qd);
#pragma unroll
    for (int c = 0; c < 4 * NH2; ++c) kv[c] = __ldg(kr + c);
  }
  for (int i = tid; i < GT * DD; i += 128) qs[i] = i / DD < nq ? __bfloat162float(qh[i]) * qscale : 0.f;
  __syncthreads();
  {
    float s[GT];
#pragma unroll
    for (int h = 0; h < GT; ++h) s[h] = 0.f;
    if (tt < tn) {
#pragma unroll
      for (int c = 0; c < 4 * NH2; ++c) {
        float k[8];
        bf8(kv[c], k);
#pragma unroll
        for (int h = 0; h < GT; ++h) {
          const float4 qa = *reinterpret_cast<const float4*>(qs + h * DD + 32 * NH2 * qd + 8 * c);
          const float4 qb = *reinterpret_cast<const float4*>(qs + h * DD + 32 * NH2 * qd + 8 * c + 4);
          s[h] = fmaf(qa.x, k[0], fmaf(qa.y, k[1], fmaf(qa.z, k[2], fmaf(qa.w, k[3], s[h]))));
          s[h] = fmaf(qb.x, k[4], fmaf(qb.y, k[5], fmaf(qb.z, k[6], fmaf(qb.w, k[7], s[h]))));
        }
      }
    }
#pragma unroll
    for (int h = 0; h < GT; ++h) {
      s[h] += __shfl_xor_sync(FULL, s[h], 1);
      s[h] += __shfl_xor_sync(FULL, s[h], 2);
    }
    if (qd == 0 && tt < TAIL_TOK) {
#pragma unroll
      for (int h = 0; h < GT; ++h) pp[GT * tt + h] = s[h];
    }
  }
  __syncthreads();
#pragma unroll
  for (int hh = 0; hh < GT / 4; ++hh) {  // softmax of head w + 4 hh over the window tokens (lane = token)
    const int h = w + 4 * hh;
    const float x = lane < tn ? pp[GT * lane + h] : NEG_INF;
    float m = x;
#pragma unroll
    for (int k = 16; k > 0; k >>= 1) m = fmaxf(m, __shfl_xor_sync(FULL, m, k));
    const float e = lane < tn ? ex2(x - m) : 0.f;
    float l = e;
#pragma unroll
    for (int k = 16; k > 0; k >>= 1) l += __shfl_xor_sync(FULL, l, k);
    pp[GT * lane + h] = e;
    if (lane == 0) { ml[2 * h] = m; ml[2 * h + 1] = l; }
  }
  __syncthreads();
  const int d8 = 8 * (tid & 15), sl = tid >> 4;
#pragma unroll
  for (int sh = 0; sh < NH2; ++sh) {
    float acc[GT][8];
#pragma unroll
    for (int h = 0; h < GT; ++h)
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[h][i] = 0.f;
#pragma unroll
    for (int j = 0; j < TAIL_TOK / 8; ++j) {
      const int t = sl + 8 * j;
      if (t < tn) {
        float v[8];
        bf8(__ldg(reinterpret_cast<const uint4*>(vt + t * ts + 128 * sh + d8)), v);
#pragma unroll
        for (int h = 0; h < GT; ++h) {
          const float pr = pp[GT * t + h];
#pragma unroll
          for (int i = 0; i < 8; ++i) acc[h][i] = fmaf(pr, v[i], acc[h][i]);
        }
      }
    }
#pragma unroll
    for (int h = 0; h < GT; ++h) {
      float* dst = part + (sl * GT + h) * DD + 128 * sh + d8;
      *reinterpret_cast<float4*>(dst) = make_float4(acc[h][0], acc[h][1], acc[h][2], acc[h][3]);
      *reinterpret_cast<float4*>(dst + 4) = make_float4(acc[h][4], acc[h][5], acc[h][6], acc[h][7]);
    }
  }
  __syncthreads();
#pragma unroll
  for (int h = 0; h < GT; ++h)
#pragma unroll
    for (int sh = 0; sh < NH2; ++sh) {
      float x = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k) x += part[(k * GT + h) * DD + 128 * sh + tid];
      o[h * DD + 128 * sh + tid] = x;
    }
  __syncthreads();
}

// the whole window of one (sequence, kv head, q tile) in one CTA (as tail_window): o [GT][DD], ml [GT][2]
template <int GT, int NH2>
DEVI void tail_window_g(const __nv_bfloat16* __restrict__ qh, int nq, float qscale, const __nv_bfloat16* __restrict__ kt,
                        const __nv_bfloat16* __restrict__ vt, long long ts, int tl, float* sm, float* o, float* ml) {
  constexpr int DD = 128 * NH2;
  __shared__ float sml[2 * GT];
  const int tid = threadIdx.x;
  float acc[GT][NH2], M[GT], Lh[GT];
#pragma unroll
  for (int h = 0; h < GT; ++h) {
    M[h] = NEG_INF; Lh[h] = 0.f;
#pragma unroll
    for (int s = 0; s < NH2; ++s) acc[h][s] = 0.f;
  }
  for (int t0 = 0; t0 < max(tl, 1); t0 += TAIL_TOK) {
    tail_partial_g<GT, NH2>(qh, nq, qscale, kt + t0 * ts, vt + t0 * ts, ts, min(tl - t0, TAIL_TOK), sm, o, sml);
#pragma unroll
    for (int h = 0; h < GT; ++h) {
      const float m = sml[2 * h], l = sml[2 * h + 1], Mn = fmaxf(M[h], m);
      const float a = M[h] == NEG_INF ? 0.f : ex2(M[h] - Mn), c = m == NEG_INF ? 0.f : ex2(m - Mn);
#pragma unroll
      for (int s = 0; s < NH2; ++s) acc[h][s] = acc[h][s] * a + o[h * DD + 128 * s + tid] * c;
      Lh[h] = Lh[h] * a + l * c;
      M[h] = Mn;
    }
    __syncthreads();
  }
#pragma unroll
  for (int h = 0; h < GT; ++h)
#pragma unroll
    for (int s = 0; s < NH2; ++s) o[h * DD + 128 * s + tid] = acc[h][s];
  if (tid < GT) {
#pragma unroll
    for (int h = 0; h < GT; ++h)
      if (h == tid) { ml[2 * h] = M[h]; ml[2 * h + 1] = Lh[h]; }
  }
  __syncthreads();
}
