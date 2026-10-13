// kivi2: FlagRot-KIVI INT2 KV pages for vLLM (simulator methods kivi_sel / kivi_c: kv_k = kv_v = kivi2:32,
// kv_k_axis = token, kv_flags = dyn, kv_dyn = att+f, kv_rb = 32).  Plain C ABI (ctypes), CUDA runtime linked
// statically (csrc/build_kivi.sh), raw device pointers + the caller's stream.
//
// Quantizer (blockflag.formats.fake_kivi, jy-yuan/KIVI): per group of 32 values s = (max - min) / 3,
// code = round(clamp((x - min) / s, 0, 3)) (s = 0: code 0), x_hat = code * s + min; stored as FP16 s and FP16 min.
//   keys   : per channel over a 32-token group (KIVI's per-channel key quantization); group g = absolute positions
//            [32 g, 32 g + 32), the sink positions [0, S) excluded (vLLM layout option A: the first group is [S, 32)).
//            Per (group, KV head) transform: per 32-dim sub-vector j identity or H32 (normalized Sylvester), or the
//            whole-head H_D, chosen by the attention-weighted rule 'att' with the frozen calibration queries
//            (qattention._weights / _vec_err / _head_err / _local_best / dyn_quant, errors pooled over the group).
//   values : per token, groups of 32 along head_dim; per token the same candidates, error weighted by the o_proj
//            column energy (diag co) or, with a base transform (KIVI-C), by M = xvinv diag(co) xvinv^T.
//
// Page layout: one chunk per (16-token page, KV head), D = head size (128 | 256), 12 D + 16 bytes (97 / 193 B per
// token and KV head):
//   KC [16][D/4]       key codes of the token: word c/16 (16 channels), channel c at bit kbit(c % 16) =
//                      2 ((c % 16) >> 1) + 16 ((c % 16) & 1) (even channels in the low half, odd in the high half:
//                      the mma A fragment pairs (2 i, 2 i + 1) come out of one shift + LOP3)
//   KM [16][D/32] half2  key (scale, min) of the token's GROUP, channels [i D/32, (i + 1) D/32), i = position % 32
//                      (the 32 cells of a group hold its D (scale, min) pairs); flags in scale sign bits:
//                      channel j < D/32 (cell i = 0): sub-vector j takes H32; channel D/32 (cell i = 1): whole-head H_D
//   VC [16][D/4]       value codes
//   VM [16][D/32] half2  value (scale, min) per 32-dim group; scale sign bit = sub-vector takes H32
//   VF [16] u8         1 = the token's value takes the whole-head H_D
// A group spans two pages (block-table entries 2 g and 2 g + 1).
//
// kivi2p8 (f8 = 1; research kivi2f8:32 = formats.fake_kivi_f8): s8 = E4M3_RN((max - min) / 3), m8 = E4M3_RN(min) (both
// saturating at +-448), codes from the stored values q = clamp(round((x - m8) / s8), 0, 3), x_hat = q s8 + m8;
// chunk 10 D bytes (80 / 160 B per token and KV head at head 128 / 256):
//   KC [16][D/4] (as above) | KM [16][D/32][2] u8 (s8 | flag << 7, m8) of the GROUP | VC [16][D/4] |
//   VM [16][D/32][2] u8 (s8 | H32 << 7, m8) per 32-dim group
// flags in the sign bits of the non-negative FP8 scales: key sub-vector j -> K scale of cell j slot 0, key whole-head
// H_D -> cell D/32 slot 0; the VALUE whole-head flag of the token in cell i -> K scale of cell i slot D/32 - 1 (written
// by the value writer; the group encoder keeps that bit when it writes the cell's key scales later).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <type_traits>

// kivi2p8f: resident CTAs per SM the fast k_attn3 is compiled for (register cap; tuned on H20 without spills)
#ifndef KA3F_MINB_128S
#define KA3F_MINB_128S 7
#endif
#ifndef KA3F_MINB_128L
#define KA3F_MINB_128L 6
#endif
#ifndef KA3F_MINB_256
#define KA3F_MINB_256 3
#endif
#define KIVI_ABI 9
#define PAGE 16
#define GRP 32

static char g_err[512];
#ifdef KIVI_DEBUG
__device__ float g_dbg[4096 * 16];
extern "C" int kv_dbg(float* host) { return (int)cudaMemcpyFromSymbol(host, g_dbg, sizeof(float) * 4096 * 16); }
#endif
extern "C" const char* kv_last_error() { return g_err; }
extern "C" int kv_abi_version() { return KIVI_ABI; }

static int check_launch(const char* what) {
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    snprintf(g_err, sizeof(g_err), "%s: %s", what, cudaGetErrorString(e));
    return 1;
  }
  return 0;
}

__host__ __device__ constexpr int chunk_bytes(int D) { return 12 * D + 16; }
__host__ __device__ constexpr int off_km(int D) { return 4 * D; }
__host__ __device__ constexpr int off_vc(int D) { return 6 * D; }
__host__ __device__ constexpr int off_vm(int D) { return 10 * D; }
__host__ __device__ constexpr int off_vf(int D) { return 12 * D; }

// kivi2p8 layout (f8)
__host__ __device__ constexpr int chunk_bytes8(int D) { return 10 * D; }
__host__ __device__ constexpr int off_km8(int D) { return 4 * D; }
__host__ __device__ constexpr int off_vc8(int D) { return 5 * D; }
__host__ __device__ constexpr int off_vm8(int D) { return 9 * D; }
__host__ __device__ constexpr int chunk_bytes_f(int D, int f8) { return f8 ? chunk_bytes8(D) : chunk_bytes(D); }

extern "C" int kv_chunk_bytes(int D) { return chunk_bytes(D); }
extern "C" int kv_chunk_bytes8(int D) { return chunk_bytes8(D); }
__host__ __device__ constexpr int kbit(int i) { return 2 * (i >> 1) + 16 * (i & 1); }

// exact fp32 ops (no FMA contraction: the reference computes q * s then + min, (x - min) / s, ...)
__device__ __forceinline__ float fmul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float fadd(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float fsub(float a, float b) { return __fsub_rn(a, b); }

// KIVI quantization of one value given the group's min / max: returns the code; rec = code * s + mn (fp32)
// s = (max - min) / 3 as torch computes it on the GPU: a tensor divided by a Python scalar is multiplied by the
// fp32 reciprocal (div_true_kernel_cuda), i.e. (max - min) * fp32(1 / 3)
__device__ __forceinline__ float kivi_scale(float mn, float mx) { return fmul(fsub(mx, mn), 1.0f / 3.0f); }
__device__ __forceinline__ int kivi_code(float x, float mn, float s) {
  float d = (s == 0.f) ? 1.f : s;
  float q = __fdiv_rn(fsub(x, mn), d);
  q = fminf(fmaxf(q, 0.f), 3.f);
  return (int)rintf(q);
}
__device__ __forceinline__ float kivi_rec(int code, float mn, float s) { return fadd(fmul((float)code, s), mn); }
// E4M3 round to nearest (even), saturating at +-448 (= formats._e4m3: clamp, then .to(float8_e4m3fn))
__device__ __forceinline__ uint8_t e4m3_bits(float x) {
  __nv_fp8_e4m3 v(fminf(fmaxf(x, -448.f), 448.f));
  return (uint8_t)v.__x;
}
__device__ __forceinline__ float e4m3_val(uint8_t b) {
  __nv_fp8_e4m3 v;
  v.__x = (__nv_fp8_storage_t)b;
  return float(v);
}
// stored (scale, min) of a group with min / max: KIVI (FP32 values, FP16 storage) or kivi2f8 (FP8 values)
__device__ __forceinline__ void qparams(float mn, float mx, int f8, float& s, float& m) {
  s = kivi_scale(mn, mx);
  m = mn;
  if (f8) { s = e4m3_val(e4m3_bits(s)); m = e4m3_val(e4m3_bits(mn)); }
}

__device__ __forceinline__ float rsqrt_rb(int rb) { return rb == 32 ? 0.17677669529663687f : rb == 128 ? 0.08838834764831845f
                                                       : rb == 256 ? 0.0625f : rb == 64 ? 0.125f : 1.f; }

// ------------------------------------------------------------------------------------------------ register FWHT
// v[E] = E consecutive elements (index lane * E + i) of a vector distributed over the 32 lanes of a warp; staged
// butterflies of blockflag.hadamard.fwht over blocks of rb (rb <= 32 E), lower index a + b, upper a - b, then the
// normalization 1 / sqrt(rb) (one fp32 multiply).  All lanes must participate (shuffles).
template <int E>
__device__ __forceinline__ void reg_fwht(float (&v)[E], int rb, int lane) {
  for (int h = 1; h < rb; h <<= 1) {
    if (h < E) {
#pragma unroll
      for (int i = 0; i < E; i++)
        if ((i & h) == 0) {
          float a = v[i], b = v[i + h];
          v[i] = fadd(a, b);
          v[i + h] = fsub(a, b);
        }
    } else {
      int m = h / E;
      bool upper = (lane & m) != 0;
#pragma unroll
      for (int i = 0; i < E; i++) {
        float o = __shfl_xor_sync(0xffffffffu, v[i], m);
        v[i] = upper ? fsub(o, v[i]) : fadd(v[i], o);
      }
    }
  }
  float c = rsqrt_rb(rb);
#pragma unroll
  for (int i = 0; i < E; i++) v[i] = fmul(v[i], c);
}

// ================================================================================================ value writer
// one warp per (token, KV head); lane holds E = D / 32 consecutive channels.  kv: [n, 2 Hkv, D] bf16 rows (row
// stride kv_rs elements; V rows Hkv..2 Hkv - 1), slots [n] int64 (< 0: skip).  co [Hkv, D] (diag) or M [Hkv, D, D].
template <int D>
__global__ void __launch_bounds__(128) k_v_write(const __nv_bfloat16* __restrict__ kv, long long kv_rs, int n, int Hkv,
                                                 const long long* __restrict__ slots, const float* __restrict__ co,
                                                 const float* __restrict__ M, uint8_t* __restrict__ pool, int ident,
                                                 const int* __restrict__ vfix, int f8) {
  constexpr int E = D / 32;              // channels per lane
  constexpr int LS = 32 / E;             // lanes per 32-dim group (sub-vector)
  constexpr int NV = D / 32;
  __shared__ float esh[4][3][D];        // wmat: error vectors of the warp (I, H, F) for the matrix products
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  long long item = (long long)blockIdx.x * 4 + warp;
  if (item >= (long long)n * Hkv) return;
  int t = (int)(item / Hkv), h = (int)(item % Hkv);
  long long slot = slots[t];
  if (slot < 0) return;
  const __nv_bfloat16* src = kv + (long long)t * kv_rs + (long long)(Hkv + h) * D + lane * E;
  float x[E];
#pragma unroll
  for (int i = 0; i < E; i++) x[i] = __bfloat162float(src[i]);
  const float* cw = co + (long long)h * D + lane * E;

  // candidate reconstructions (in the original basis) and their quantization params
  float rec[3][E];
  float sc[3], mnv[3];          // this lane's group's (scale, min) per candidate
  int code[3][E];
#pragma unroll
  for (int cand = 0; cand < 3; cand++) {
    if (cand > 0 && ident) break;        // kivi2id: identity only (plain KIVI), no selection
    float y[E];
#pragma unroll
    for (int i = 0; i < E; i++) y[i] = x[i];
    int rb = cand == 1 ? 32 : D;
    if (cand > 0) reg_fwht<E>(y, rb, lane);
    float mn = y[0], mx = y[0];
#pragma unroll
    for (int i = 1; i < E; i++) { mn = fminf(mn, y[i]); mx = fmaxf(mx, y[i]); }
#pragma unroll
    for (int m = 1; m < LS; m <<= 1) {
      mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, m));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, m));
    }
    float s, m0;
    qparams(mn, mx, f8, s, m0);
    sc[cand] = s; mnv[cand] = m0;
#pragma unroll
    for (int i = 0; i < E; i++) { code[cand][i] = kivi_code(y[i], m0, s); y[i] = kivi_rec(code[cand][i], m0, s); }
    if (cand > 0) reg_fwht<E>(y, rb, lane);
#pragma unroll
    for (int i = 0; i < E; i++) rec[cand][i] = y[i];
  }
  // errors
  float eI = 0.f, eH = 0.f, eF = 0.f;          // per lane partial (sub-vector for I / H, head for F)
  bool wmat = M != nullptr && !ident && !vfix;
  int cand = 0;
  bool useF = false;
  if (vfix && !ident) {          // kivi2s: frozen choice of this (layer, KV head) (bits as kfix)
    useF = (vfix[h] >> 16) & 1;
    cand = useF ? 2 : ((vfix[h] >> (lane / LS)) & 1);
  }
  if (!ident && !vfix) {
  if (!wmat) {
#pragma unroll
    for (int i = 0; i < E; i++) {
      float d0 = fsub(rec[0][i], x[i]), d1 = fsub(rec[1][i], x[i]), d2 = fsub(rec[2][i], x[i]);
      eI += d0 * d0 * cw[i];
      eH += d1 * d1 * cw[i];
      eF += d2 * d2 * cw[i];
    }
  } else {
#pragma unroll
    for (int i = 0; i < E; i++) {
      esh[warp][0][lane * E + i] = fsub(rec[0][i], x[i]);
      esh[warp][1][lane * E + i] = fsub(rec[1][i], x[i]);
      esh[warp][2][lane * E + i] = fsub(rec[2][i], x[i]);
    }
    __syncwarp();
    const float* Mh = M + (long long)h * D * D;
    int j = lane / LS;
#pragma unroll
    for (int i = 0; i < E; i++) {
      int r = lane * E + i;
      float a0 = 0.f, a1 = 0.f, a2 = 0.f;
      for (int k = j * 32; k < j * 32 + 32; k++) {           // block-diagonal part (sub-vector j)
        float m = Mh[(long long)r * D + k];
        a0 += m * esh[warp][0][k];
        a1 += m * esh[warp][1][k];
      }
      for (int k = 0; k < D; k++) a2 += Mh[(long long)r * D + k] * esh[warp][2][k];
      eI += esh[warp][0][r] * a0;
      eH += esh[warp][1][r] * a1;
      eF += esh[warp][2][r] * a2;
    }
  }
  // per sub-vector sums (lanes of the group), head sum of F
#pragma unroll
  for (int m = 1; m < LS; m <<= 1) {
    eI += __shfl_xor_sync(0xffffffffu, eI, m);
    eH += __shfl_xor_sync(0xffffffffu, eH, m);
  }
#pragma unroll
  for (int m = 1; m < 32; m <<= 1) eF += __shfl_xor_sync(0xffffffffu, eF, m);
  bool useH = eH < eI;                        // argmin over (I, H): identity on ties
  // base (combination) head error
  float eB;
  if (!wmat) {
    float v = useH ? eH : eI;                 // sub-vector sums; add each group once (lane % LS == 0)
    eB = (lane % LS == 0) ? v : 0.f;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) eB += __shfl_xor_sync(0xffffffffu, eB, m);
  } else {
    __syncwarp();
#pragma unroll
    for (int i = 0; i < E; i++) esh[warp][0][lane * E + i] = useH ? esh[warp][1][lane * E + i] : esh[warp][0][lane * E + i];
    __syncwarp();
    const float* Mh = M + (long long)h * D * D;
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < E; i++) {
      int r = lane * E + i;
      float s_ = 0.f;
      for (int k = 0; k < D; k++) s_ += Mh[(long long)r * D + k] * esh[warp][0][k];
      a += esh[warp][0][r] * s_;
    }
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) a += __shfl_xor_sync(0xffffffffu, a, m);
    eB = a;
  }
  useF = eF < eB;                              // argmin over (base, F): base on ties
  cand = useF ? 2 : (useH ? 1 : 0);
  }
  // pack and store
  long long blk = slot / PAGE;
  int off = (int)(slot % PAGE);
  uint8_t* chunk = pool + (blk * Hkv + h) * (long long)chunk_bytes_f(D, f8);
  // codes: E channels per lane -> E / 4 bytes
#pragma unroll
  for (int b = 0; b < E / 4; b++) {
    int byte = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) byte |= (code[cand][b * 4 + i] & 3) << (2 * i);
    chunk[(f8 ? off_vc8(D) : off_vc(D)) + off * (D / 4) + lane * (E / 4) + b] = (uint8_t)byte;
  }
  if (f8) {
    if (lane % LS == 0) {
      int j = lane / LS;
      uint8_t* vm = chunk + off_vm8(D) + off * (2 * NV) + 2 * j;
      vm[0] = e4m3_bits(sc[cand]) | (cand == 1 ? 0x80 : 0);    // sign bit = H32 sub-vector
      vm[1] = e4m3_bits(mnv[cand]);
    }
    if (lane == 0) {          // value whole-head flag: K scale sign bit of this token's cell, slot NV - 1
      uint8_t* kb = chunk + off_km8(D) + off * (2 * NV) + 2 * (NV - 1);
      *kb = (uint8_t)((*kb & 0x7F) | (useF ? 0x80 : 0));
    }
    return;
  }
  if (lane % LS == 0) {
    int j = lane / LS;
    __half sh = __float2half_rn(sc[cand]);
    if (cand == 1) sh = __hneg(__habs(sh));          // sign bit = H32 sub-vector
    __half mh = __float2half_rn(mnv[cand]);
    __half2* vm = reinterpret_cast<__half2*>(chunk + off_vm(D) + off * (D / 8));
    vm[j] = __halves2half2(sh, mh);
  }
  if (lane == 0) chunk[off_vf(D) + off] = useF ? 1 : 0;
  (void)NV;
}

// ================================================================================================ key group encoder
// one CTA (256 threads) per (group item, KV head).  src / src2: bf16 rows (row stride src_rs / src2_rs elements, K
// row of head h at + h D); sidx [N, 32] int64 source of position 32 g + i: r >= 0 -> src row r, r <= -2 -> src2 row
// -r - 2 (prefill: the BF16 window rows of positions before the chunk), -1 -> not part of the group (sinks);
// dslot [N, 32] int64 page slot of position 32 g + i (dslot[n][0] < 0: inactive item, nothing written).
// qs [Hkv, NQ, D] fp32, lz [Hkv, NQ] fp32, qsc = the attention scaling of the research rule.
template <int D>
struct KEncSmem {
  static constexpr int LD = D + 4;          // padded rows (16-byte aligned for float4)
  static constexpr int QT = D == 128 ? 64 : 32;   // queries per tile: 16 thread columns x QPT
  static constexpr int QPT = QT / 16;
};

template <int D>
__global__ void __launch_bounds__(256) k_k_encode(const __nv_bfloat16* __restrict__ src, long long src_rs,
                                                  const __nv_bfloat16* __restrict__ src2, long long src2_rs,
                                                  const long long* __restrict__ sidx, const long long* __restrict__ dslot,
                                                  int Hkv, const float* __restrict__ qs, const float* __restrict__ lz,
                                                  int NQ, float qsc, uint8_t* __restrict__ pool, int ident,
                                                  const int* __restrict__ kfix, int f8) {
  constexpr int LD = KEncSmem<D>::LD, QT = KEncSmem<D>::QT, NV = D / 32;
  extern __shared__ float sm[];
  float* xs = sm;                            // [32][LD] keys (fp32)
  float* es = xs + 32 * LD;                  // [3][32][LD] errors of I / H / F (pass 2: es[0] = base error)
  float* qt = es + 3 * 32 * LD;              // [QT][LD] query tile
  float* lzt = qt + QT * LD;                 // [QT] its logsumexp
  float* red = lzt + QT;                     // [8 warps][2 NV + 2] partial error sums
  __shared__ int s_valid[32];
  __shared__ float s_par[3][2][D];           // per candidate (min, scale) per channel
  __shared__ int s_choice[NV + 1];
  __shared__ float s_ef;
  int item = blockIdx.x, h = blockIdx.y;
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const long long* si = sidx + (long long)item * 32;
  const long long* ds = dslot + (long long)item * 32;
  if (ds[0] < 0) return;
  if (tid < 32) s_valid[tid] = si[tid] != -1;
  __syncthreads();
  // load keys
  for (int e = tid; e < 32 * D; e += 256) {
    int t = e / D, c = e % D;
    long long r = si[t];
    xs[t * LD + c] = r >= 0 ? __bfloat162float(src[r * src_rs + (long long)h * D + c])
                   : r <= -2 ? __bfloat162float(src2[(-r - 2) * src2_rs + (long long)h * D + c]) : 0.f;
  }
  __syncthreads();
  if (ident) {
    // kivi2id (plain KIVI): per channel min / max of the keys themselves, identity everywhere, no selection
    for (int c = tid; c < D; c += 256) {
      float mn = INFINITY, mx = -INFINITY;
      for (int t = 0; t < 32; t++)
        if (s_valid[t]) { float v = xs[t * LD + c]; mn = fminf(mn, v); mx = fmaxf(mx, v); }
      qparams(mn, mx, f8, s_par[0][1][c], s_par[0][0][c]);
    }
    if (tid <= NV) s_choice[tid] = 0;
    __syncthreads();
  } else {
  // candidates: warp w handles tokens w, w + 8, ...: per token the transformed vectors (y) in es[cand]
  {
    constexpr int E = D / 32;
    for (int t = warp; t < 32; t += 8) {
      float x[E];
#pragma unroll
      for (int i = 0; i < E; i++) x[i] = xs[t * LD + lane * E + i];
#pragma unroll
      for (int cand = 0; cand < 3; cand++) {
        float y[E];
#pragma unroll
        for (int i = 0; i < E; i++) y[i] = x[i];
        if (cand > 0) reg_fwht<E>(y, cand == 1 ? 32 : D, lane);
#pragma unroll
        for (int i = 0; i < E; i++) es[(cand * 32 + t) * LD + lane * E + i] = y[i];
      }
    }
  }
  __syncthreads();
  // per channel min / max over the valid tokens, scale, reconstruction in the transformed basis (in place)
  for (int e = tid; e < 3 * D; e += 256) {
    int cand = e / D, c = e % D;
    float* col = es + cand * 32 * LD + c;
    float mn = INFINITY, mx = -INFINITY;
    for (int t = 0; t < 32; t++)
      if (s_valid[t]) { float v = col[t * LD]; mn = fminf(mn, v); mx = fmaxf(mx, v); }
    float s, m0;
    qparams(mn, mx, f8, s, m0);
    s_par[cand][0][c] = m0;
    s_par[cand][1][c] = s;
    for (int t = 0; t < 32; t++)
      if (s_valid[t]) col[t * LD] = kivi_rec(kivi_code(col[t * LD], m0, s), m0, s);
  }
  __syncthreads();
  if (kfix) {
    // kivi2s (static selection): the frozen choice of this (layer, KV head): bits 0..NV-1 = H32 per sub-vector,
    // bit 16 = whole-head H_D; the codes come from the chosen candidates' own quantization above
    if (tid <= NV) s_choice[tid] = tid < NV ? (kfix[h] >> tid) & 1 : (kfix[h] >> 16) & 1;
    __syncthreads();
  } else {
  // back to the original basis and errors e = rec - x (invalid tokens: 0)
  {
    constexpr int E = D / 32;
    for (int t = warp; t < 32; t += 8) {
#pragma unroll
      for (int cand = 0; cand < 3; cand++) {
        float y[E];
#pragma unroll
        for (int i = 0; i < E; i++) y[i] = es[(cand * 32 + t) * LD + lane * E + i];
        if (cand > 0) reg_fwht<E>(y, cand == 1 ? 32 : D, lane);
#pragma unroll
        for (int i = 0; i < E; i++) {
          float xv = xs[t * LD + lane * E + i];
          es[(cand * 32 + t) * LD + lane * E + i] = s_valid[t] ? fsub(y[i], xv) : 0.f;
        }
      }
    }
  }
  __syncthreads();
#ifdef KIVI_DEBUG
  if (item == 0 && h == 3) for (int e = tid; e < 32 * D; e += 256) g_dbg[32768 + e] = es[(e / D) * LD + e % D];
#endif
  // pass 1: weights w[t][q] = min(1, exp(qsc x_t . q - lz_q)) and the pooled errors E_I[j], E_H[j], E_F, as a
  // register-tiled product: thread (tt = tid / 16, tq = tid % 16) <-> tokens 2 tt, 2 tt + 1 x queries tq QPT .. + QPT
  constexpr int QPT = KEncSmem<D>::QPT;
  const int tt = tid >> 4, tq = tid & 15;
  const int ta = 2 * tt, tb = 2 * tt + 1;
  const bool va = s_valid[ta], vb = s_valid[tb];
  float accI[NV], accH[NV], accF = 0.f, accB = 0.f;
#pragma unroll
  for (int j = 0; j < NV; j++) { accI[j] = 0.f; accH[j] = 0.f; }
  const float* qh = qs + (long long)h * NQ * D;
  const float* lzh = lz + (long long)h * NQ;
  auto load_tile = [&](int q0) {
    __syncthreads();
    for (int e = tid; e < QT * (D / 4); e += 256) {
      int qq = e / (D / 4), c4 = e % (D / 4);
      float4 v = (q0 + qq < NQ) ? reinterpret_cast<const float4*>(qh + (long long)(q0 + qq) * D)[c4]
                                : make_float4(0.f, 0.f, 0.f, 0.f);
      *reinterpret_cast<float4*>(qt + qq * LD + 4 * c4) = v;
    }
    for (int e = tid; e < QT; e += 256) lzt[e] = (q0 + e < NQ) ? lzh[q0 + e] : 0.f;
    __syncthreads();
  };
  auto dot4 = [](float acc, float4 a, float4 b) {
    acc = fmaf(a.x, b.x, acc); acc = fmaf(a.y, b.y, acc); acc = fmaf(a.z, b.z, acc); return fmaf(a.w, b.w, acc);
  };
  // weights of this thread's (token, query) pairs for the current tile
  auto weights = [&](int q0, float (&w)[2][QPT]) {
    float lg[2][QPT];
#pragma unroll
    for (int i = 0; i < QPT; i++) { lg[0][i] = 0.f; lg[1][i] = 0.f; }
    const float4* xa = reinterpret_cast<const float4*>(xs + ta * LD);
    const float4* xb = reinterpret_cast<const float4*>(xs + tb * LD);
#pragma unroll 4
    for (int c4 = 0; c4 < D / 4; c4++) {
      float4 a = xa[c4], b = xb[c4];
#pragma unroll
      for (int i = 0; i < QPT; i++) {
        float4 qv = reinterpret_cast<const float4*>(qt + (tq + 16 * i) * LD)[c4];
        lg[0][i] = dot4(lg[0][i], a, qv);
        lg[1][i] = dot4(lg[1][i], b, qv);
      }
    }
#pragma unroll
    for (int i = 0; i < QPT; i++) {
      int qq = tq + 16 * i;
      bool qv = q0 + qq < NQ;
      float l = lzt[qq];
      w[0][i] = (qv && va) ? fminf(expf(fsub(fmul(qsc, lg[0][i]), l)), 1.f) : 0.f;
      w[1][i] = (qv && vb) ? fminf(expf(fsub(fmul(qsc, lg[1][i]), l)), 1.f) : 0.f;
    }
  };
  for (int q0 = 0; q0 < NQ; q0 += QT) {
    load_tile(q0);
    float w[2][QPT];
    weights(q0, w);
    float pf[2][QPT];
#pragma unroll
    for (int i = 0; i < QPT; i++) { pf[0][i] = 0.f; pf[1][i] = 0.f; }
#pragma unroll
    for (int j = 0; j < NV; j++) {
      float p0[2][QPT], p1[2][QPT];
#pragma unroll
      for (int i = 0; i < QPT; i++) { p0[0][i] = p0[1][i] = p1[0][i] = p1[1][i] = 0.f; }
#pragma unroll 2
      for (int c4 = 8 * j; c4 < 8 * j + 8; c4++) {
        float4 e0a = reinterpret_cast<const float4*>(es + (0 * 32 + ta) * LD)[c4];
        float4 e0b = reinterpret_cast<const float4*>(es + (0 * 32 + tb) * LD)[c4];
        float4 e1a = reinterpret_cast<const float4*>(es + (1 * 32 + ta) * LD)[c4];
        float4 e1b = reinterpret_cast<const float4*>(es + (1 * 32 + tb) * LD)[c4];
        float4 e2a = reinterpret_cast<const float4*>(es + (2 * 32 + ta) * LD)[c4];
        float4 e2b = reinterpret_cast<const float4*>(es + (2 * 32 + tb) * LD)[c4];
#pragma unroll
        for (int i = 0; i < QPT; i++) {
          float4 qv = reinterpret_cast<const float4*>(qt + (tq + 16 * i) * LD)[c4];
          p0[0][i] = dot4(p0[0][i], e0a, qv); p0[1][i] = dot4(p0[1][i], e0b, qv);
          p1[0][i] = dot4(p1[0][i], e1a, qv); p1[1][i] = dot4(p1[1][i], e1b, qv);
          pf[0][i] = dot4(pf[0][i], e2a, qv); pf[1][i] = dot4(pf[1][i], e2b, qv);
        }
      }
#pragma unroll
      for (int i = 0; i < QPT; i++) {
        accI[j] += w[0][i] * p0[0][i] * p0[0][i] + w[1][i] * p0[1][i] * p0[1][i];
        accH[j] += w[0][i] * p1[0][i] * p1[0][i] + w[1][i] * p1[1][i] * p1[1][i];
      }
    }
#pragma unroll
    for (int i = 0; i < QPT; i++) accF += w[0][i] * pf[0][i] * pf[0][i] + w[1][i] * pf[1][i] * pf[1][i];
  }
  // deterministic block reduction: warp shuffle (fixed order), then warps in order
#pragma unroll
  for (int j = 0; j < NV; j++)
    for (int m = 16; m >= 1; m >>= 1) {
      accI[j] += __shfl_xor_sync(0xffffffffu, accI[j], m);
      accH[j] += __shfl_xor_sync(0xffffffffu, accH[j], m);
    }
  for (int m = 16; m >= 1; m >>= 1) accF += __shfl_xor_sync(0xffffffffu, accF, m);
  __syncthreads();
  if (lane == 0) {
#pragma unroll
    for (int j = 0; j < NV; j++) { red[warp * (2 * NV + 2) + j] = accI[j]; red[warp * (2 * NV + 2) + NV + j] = accH[j]; }
    red[warp * (2 * NV + 2) + 2 * NV] = accF;
  }
  __syncthreads();
  if (tid < NV) {
    float a = 0.f, b = 0.f;
    for (int w_ = 0; w_ < 8; w_++) { a += red[w_ * (2 * NV + 2) + tid]; b += red[w_ * (2 * NV + 2) + NV + tid]; }
    s_choice[tid] = b < a ? 1 : 0;           // argmin (I, H): identity on ties
  }
  if (tid == 32) {
    float f = 0.f;
    for (int w_ = 0; w_ < 8; w_++) f += red[w_ * (2 * NV + 2) + 2 * NV];
    s_ef = f;
  }
  __syncthreads();
  // pass 2: base error with the chosen sub-vector candidates (errors of the choice into es[0]); weights recomputed
  for (int e = tid; e < 32 * D; e += 256) {
    int t = e / D, c = e % D;
    if (s_choice[c / 32]) es[t * LD + c] = es[(32 + t) * LD + c];
  }
  for (int q0 = 0; q0 < NQ; q0 += QT) {
    load_tile(q0);
    float w[2][QPT];
    weights(q0, w);
    float pb[2][QPT];
#pragma unroll
    for (int i = 0; i < QPT; i++) { pb[0][i] = 0.f; pb[1][i] = 0.f; }
#pragma unroll 4
    for (int c4 = 0; c4 < D / 4; c4++) {
      float4 ea = reinterpret_cast<const float4*>(es + ta * LD)[c4];
      float4 eb = reinterpret_cast<const float4*>(es + tb * LD)[c4];
#pragma unroll
      for (int i = 0; i < QPT; i++) {
        float4 qv = reinterpret_cast<const float4*>(qt + (tq + 16 * i) * LD)[c4];
        pb[0][i] = dot4(pb[0][i], ea, qv);
        pb[1][i] = dot4(pb[1][i], eb, qv);
      }
    }
#pragma unroll
    for (int i = 0; i < QPT; i++) accB += w[0][i] * pb[0][i] * pb[0][i] + w[1][i] * pb[1][i] * pb[1][i];
  }
  for (int m = 16; m >= 1; m >>= 1) accB += __shfl_xor_sync(0xffffffffu, accB, m);
  __syncthreads();
  if (lane == 0) red[warp] = accB;
  __syncthreads();
  if (tid == 0) {
    float b = 0.f;
    for (int w_ = 0; w_ < 8; w_++) b += red[w_];
    s_choice[NV] = s_ef < b ? 1 : 0;          // argmin (base, F): base on ties
  }
  __syncthreads();
  }   // !kfix
  }   // !ident
  // write: codes of the chosen candidate (per sub-vector, or whole-head F) of the valid tokens, and the meta slices
  // (scale | flag sign, min) of all 32 cells of the group
  const bool useF = s_choice[NV] != 0;
  {
    constexpr int E = D / 32;
    constexpr int LS = 32 / E;
    for (int t = warp; t < 32; t += 8) {
      if (!s_valid[t]) continue;
      float x[E], yh[E], yf[E];
#pragma unroll
      for (int i = 0; i < E; i++) { x[i] = xs[t * LD + lane * E + i]; yh[i] = x[i]; yf[i] = x[i]; }
      reg_fwht<E>(yh, 32, lane);
      reg_fwht<E>(yf, D, lane);
      int j = lane / LS;
      int cand = useF ? 2 : s_choice[j];
      long long slot = ds[t];
      long long blk = slot / PAGE;
      int off = (int)(slot % PAGE);
      uint8_t* chunk = pool + (blk * Hkv + h) * (long long)chunk_bytes_f(D, f8);
      uint32_t word = 0;                         // this lane's channels' bits of key-code word (lane E) / 16
#pragma unroll
      for (int i = 0; i < E; i++) {
        int c = lane * E + i;
        float y = cand == 0 ? x[i] : cand == 1 ? yh[i] : yf[i];
        word |= (uint32_t)(kivi_code(y, s_par[cand][0][c], s_par[cand][1][c]) & 3) << kbit(c % 16);
      }
#pragma unroll
      for (int m = 1; m < 16 / E; m <<= 1) word |= __shfl_xor_sync(0xffffffffu, word, m);
      if ((lane * E) % 16 == 0)
        reinterpret_cast<uint32_t*>(chunk + off * (D / 4))[(lane * E) / 16] = word;
    }
  }
  if (f8) {
    for (int c = tid; c < D; c += 256) {
      int cand = useF ? 2 : s_choice[c / 32];
      int cell = c / NV, k = c % NV;
      bool neg = (k == 0 && cell < NV) ? (!useF && s_choice[cell] != 0) : (k == 0 && cell == NV ? useF : false);
      long long slot = ds[cell];
      long long blk = slot / PAGE;
      int off = (int)(slot % PAGE);
      uint8_t* km = pool + (blk * Hkv + h) * (long long)chunk_bytes8(D) + off_km8(D) + off * (2 * NV) + 2 * k;
      uint8_t sb = e4m3_bits(s_par[cand][1][c]) | (neg ? 0x80 : 0);
      if (k == NV - 1) sb = (uint8_t)((sb & 0x7F) | (km[0] & 0x80));   // keep the value whole-head flag of the cell
      km[0] = sb;
      km[1] = e4m3_bits(s_par[cand][0][c]);
    }
    return;
  }
  for (int c = tid; c < D; c += 256) {
    int j = c / 32;
    int cand = useF ? 2 : s_choice[j];
    __half sh = __float2half_rn(s_par[cand][1][c]);
    bool neg = (c < NV) ? (!useF && s_choice[c] != 0) : (c == NV ? useF : false);
    sh = neg ? __hneg(__habs(sh)) : __habs(sh);
    __half mh = __float2half_rn(s_par[cand][0][c]);
    int cell = c / NV, k = c % NV;              // cell i of the group holds channels [i NV, (i + 1) NV)
    long long slot = ds[cell];
    long long blk = slot / PAGE;
    int off = (int)(slot % PAGE);
    uint8_t* chunk = pool + (blk * Hkv + h) * (long long)chunk_bytes(D);
    reinterpret_cast<__half2*>(chunk + off_km(D) + off * (D / 8))[k] = __halves2half2(sh, mh);
  }
}

template <int D>
static size_t kenc_smem(int NQ) {
  constexpr int LD = KEncSmem<D>::LD, QT = KEncSmem<D>::QT, NV = D / 32;
  return sizeof(float) * (size_t)(32 * LD + 3 * 32 * LD + QT * LD + QT + 8 * (2 * NV + 2));
}

// ================================================================================================ decode attention
// grid (B, Hkv, NS + 1), 128 threads; CTA (b, h, s < NS) attends to the page tokens [s chunk, (s + 1) chunk) clipped
// to [kv_start, cached[b]) (chunk a multiple of 32: whole groups), CTA s = NS to the BF16 slots [0, tail_len[b]) of
// ring row tail_idx[b] (sinks + window, research kv_sink / kv_resid).  Partials: o (unnormalized, original basis),
// m (log2 domain), l per q head -> merge kernel.
// q [B, Hkv G, D] bf16 (row stride q_rs), bt [B, bt_rs] int32 block table, ring [(R + 1) T, 2 Hkv, D] bf16.
template <int D>
struct AttnSmem {
  float q[3][8][D];          // q, H32 q (per sub-vector), H_D q   (G <= 8)
  float qs[8][D];            // per-group q' * scale
  float part[4][8][32];      // partial scores [channel quarter][head][token]
  float bias[4][8];          // partial q' . min [warp][head]
  float p[8][32];            // probabilities [head][token]
  float alpha[8];
  float mfin[8], lfin[8];
  uint32_t km[32][D / 32];   // the group's key meta (half2 bits) [cell][k]
  int flags[2];              // local-bit mask, F
};

// in-smem normalized FWHT of x[G][D] over blocks of rb (all 128 threads; ends with __syncthreads)
template <int D>
__device__ void smem_fwht(float (*x)[D], int G, int rb) {
  int tid = threadIdx.x;
  for (int h = 1; h < rb; h <<= 1) {
    for (int e = tid; e < G * D / 2; e += 128) {
      int g = e / (D / 2), k = e % (D / 2);
      int lo = (k / h) * 2 * h + (k % h), hi = lo + h;
      float a = x[g][lo], b = x[g][hi];
      x[g][lo] = fadd(a, b);
      x[g][hi] = fsub(a, b);
    }
    __syncthreads();
  }
  float c = rsqrt_rb(rb);
  for (int e = tid; e < G * D; e += 128) x[e / D][e % D] = fmul(x[e / D][e % D], c);
  __syncthreads();
}
// kivi2p8f: smem_fwht with shift / mask indexing (h is a power of two; same butterflies and order)
template <int D>
__device__ void smem_fwht_f(float (*x)[D], int G, int rb) {
  int tid = threadIdx.x;
  for (int h = 1; h < rb; h <<= 1) {
    for (int e = tid; e < G * D / 2; e += 128) {
      int g = e / (D / 2), k = e % (D / 2);
      int lo = ((k & ~(h - 1)) << 1) | (k & (h - 1)), hi = lo + h;
      float a = x[g][lo], b = x[g][hi];
      x[g][lo] = fadd(a, b);
      x[g][hi] = fsub(a, b);
    }
    __syncthreads();
  }
  float c = rsqrt_rb(rb);
  for (int e = tid; e < G * D; e += 128) x[e / D][e % D] = fmul(x[e / D][e % D], c);
  __syncthreads();
}


template <int D>
__global__ void __launch_bounds__(128) k_attn(const __nv_bfloat16* __restrict__ q, long long q_rs, const uint8_t* __restrict__ pool,
                                              const int* __restrict__ bt, long long bt_rs, const int* __restrict__ cached,
                                              int kv_start, int Hkv, int G, int NS, int chunk, float qscale,
                                              const __nv_bfloat16* __restrict__ ring, const int* __restrict__ tail_len,
                                              const int* __restrict__ tail_idx, int T,
                                              float* __restrict__ opart, float* __restrict__ ml) {
  constexpr int CPT = D / 128;     // channels per thread in the PV / qs phases (c = tid + 128 k)
  constexpr int NV = D / 32;
  __shared__ AttnSmem<D> S;
  const int b = blockIdx.x, h = blockIdx.y, sp = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const long long pidx = ((long long)(b * Hkv + h) * (NS + 1) + sp);
  float* op = opart + pidx * G * D;
  float* mlp = ml + pidx * G * 2;
  const bool tail = sp == NS;
  int t0, t1;
  if (tail) {
    t0 = 0;
    t1 = tail_len ? tail_len[b] : 0;
  } else {
    int L = cached[b];
    t0 = max(sp * chunk, kv_start);
    t1 = min((sp + 1) * chunk, L);
  }
  if (t1 <= t0) {
    for (int e = tid; e < G; e += 128) { mlp[e * 2] = -INFINITY; mlp[e * 2 + 1] = 0.f; }
    return;
  }
  // queries (fp32), and for page CTAs their H32 / H_D transforms
  const __nv_bfloat16* qb = q + (long long)b * q_rs + (long long)h * G * D;
  for (int e = tid; e < G * D; e += 128) {
    float v = __bfloat162float(qb[e]);
    S.q[0][e / D][e % D] = v;
    S.q[1][e / D][e % D] = v;
    S.q[2][e / D][e % D] = v;
  }
  __syncthreads();
  if (!tail) {
    smem_fwht<D>(S.q[1], G, 32);
    smem_fwht<D>(S.q[2], G, D);
  }
  float accI[8][CPT], accH[8][CPT], accF[8][CPT];
#pragma unroll
  for (int g = 0; g < 8; g++)
#pragma unroll
    for (int k = 0; k < CPT; k++) { accI[g][k] = 0.f; accH[g][k] = 0.f; accF[g][k] = 0.f; }
  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};   // heads warp, warp + 4 (lanes: replicated)
  const int* btb = bt + (long long)b * bt_rs;
  const long long CB = chunk_bytes(D);

  int ring_row = tail ? (tail_idx ? tail_idx[b] : 0) : 0;
  const __nv_bfloat16* rbase = ring + (long long)ring_row * T * 2 * Hkv * D;
  for (int base = tail ? 0 : (t0 / GRP) * GRP; base < t1; base += GRP) {
    const uint8_t* chA = nullptr;
    const uint8_t* chB = nullptr;
    if (!tail) {
      int g = base / GRP;
      chA = pool + ((long long)btb[2 * g] * Hkv + h) * CB;
      chB = pool + ((long long)btb[2 * g + 1] * Hkv + h) * CB;
      // stage the group's key meta and flags
      for (int e = tid; e < 32 * NV; e += 128) {
        int cell = e / NV, k = e % NV;
        const uint8_t* ch = cell < 16 ? chA : chB;
        S.km[cell][k] = reinterpret_cast<const uint32_t*>(ch + off_km(D) + (cell % 16) * (D / 8))[k];
      }
      __syncthreads();
      if (tid == 0) {
        int lm = 0;
        for (int j = 0; j < NV; j++) lm |= ((S.km[0][j] >> 15) & 1) << j;   // sign of the scale (low half)
        S.flags[0] = lm;
        S.flags[1] = (S.km[1][0] >> 15) & 1;
      }
      __syncthreads();
      // q' * scale and q' . min
      const int lmask = S.flags[0], useF = S.flags[1];
      float bp[8];
#pragma unroll
      for (int g = 0; g < 8; g++) bp[g] = 0.f;
#pragma unroll
      for (int k = 0; k < CPT; k++) {
        int c = tid + 128 * k;
        uint32_t w = S.km[c / NV][c % NV];
        __half2 hv = *reinterpret_cast<__half2*>(&w);
        float s = fabsf(__low2float(hv)), mn = __high2float(hv);
        int which = useF ? 2 : ((lmask >> (c / 32)) & 1);
        for (int g = 0; g < G; g++) {
          float qp = S.q[which][g][c];
          S.qs[g][c] = qp * s;
          bp[g] += qp * mn;
        }
      }
      for (int g = 0; g < G; g++) {
        float v = bp[g];
        for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
        if (lane == 0) S.bias[warp][g] = v;
      }
      __syncthreads();
      // partial scores: thread (warp = channel quarter, lane = token)
      {
        const uint8_t* ch = lane < 16 ? chA : chB;
        const uint8_t* kc = ch + (lane % 16) * (D / 4) + warp * (D / 16);
        float acc[8];
#pragma unroll
        for (int g = 0; g < 8; g++) acc[g] = 0.f;
#pragma unroll
        for (int wd = 0; wd < D / 64; wd++) {                  // D/16 bytes = D/64 words
          uint32_t word = reinterpret_cast<const uint32_t*>(kc)[wd];
#pragma unroll
          for (int i = 0; i < 16; i++) {
            int c = warp * (D / 4) + wd * 16 + i;
            float cv = (float)((word >> kbit(i)) & 3);
#pragma unroll
            for (int g = 0; g < 8; g++)
              if (g < G) acc[g] = fmaf(S.qs[g][c], cv, acc[g]);
          }
        }
        for (int g = 0; g < G; g++) S.part[warp][g][lane] = acc[g];
      }
      __syncthreads();
    } else {
      // BF16 slots: plain q . k
      int j = base + lane;
      float acc[8];
#pragma unroll
      for (int g = 0; g < 8; g++) acc[g] = 0.f;
      if (j < t1) {
        const __nv_bfloat16* kr = rbase + (long long)j * 2 * Hkv * D + (long long)h * D + warp * (D / 4);
        for (int i = 0; i < D / 4; i++) {
          float kv = __bfloat162float(kr[i]);
          int c = warp * (D / 4) + i;
#pragma unroll
          for (int g = 0; g < 8; g++)
            if (g < G) acc[g] = fmaf(S.q[0][g][c], kv, acc[g]);
        }
      }
      for (int g = 0; g < G; g++) S.part[warp][g][lane] = acc[g];
      if (lane < G) S.bias[warp][lane] = 0.f;
      __syncthreads();
    }
    // online softmax: warp w handles heads w and w + 4
#pragma unroll
    for (int r = 0; r < 2; r++) {
      int g = warp + 4 * r;
      if (g >= G) continue;
      int pos = base + lane;
      bool valid = tail ? (pos < t1) : (pos >= t0 && pos < t1);
      float sc = S.part[0][g][lane] + S.part[1][g][lane] + S.part[2][g][lane] + S.part[3][g][lane]
                 + S.bias[0][g] + S.bias[1][g] + S.bias[2][g] + S.bias[3][g];
      sc = valid ? sc * qscale : -INFINITY;
      float mx = sc;
      for (int m = 16; m >= 1; m >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, m));
      float mnew = fmaxf(m_run[r], mx);
      float p = (mnew == -INFINITY || !valid) ? 0.f : exp2f(sc - mnew);
      float al = (m_run[r] == -INFINITY) ? 0.f : exp2f(m_run[r] - mnew);
      float ps = p;
      for (int m = 16; m >= 1; m >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, m);
      l_run[r] = l_run[r] * al + ps;
      m_run[r] = mnew;
      S.p[g][lane] = p;
      if (lane == 0) S.alpha[g] = al;
    }
    __syncthreads();
    // PV: thread <-> channel(s) c = tid + 128 k; the class (I / H32 / H_D) is uniform over a warp
#pragma unroll
    for (int g = 0; g < 8; g++) {
      if (g >= G) break;
      float al = S.alpha[g];
#pragma unroll
      for (int k = 0; k < CPT; k++) { accI[g][k] *= al; accH[g][k] *= al; accF[g][k] *= al; }
    }
    int nt = min(GRP, t1 - base);
    for (int i = max(0, t0 - base); i < nt; i++) {
      if (!tail) {
        const uint8_t* ch = i < 16 ? chA : chB;
        int cell = i % 16;
        int vf = ch[off_vf(D) + cell];
#pragma unroll
        for (int k = 0; k < CPT; k++) {
          int c = tid + 128 * k;
          uint8_t byte = ch[off_vc(D) + cell * (D / 4) + c / 4];
          float cv = (float)((byte >> (2 * (c % 4))) & 3);
          uint32_t w = reinterpret_cast<const uint32_t*>(ch + off_vm(D) + cell * (D / 8))[c / 32];
          __half2 hv = *reinterpret_cast<__half2*>(&w);
          float s = __low2float(hv);
          float d = fadd(fmul(cv, fabsf(s)), __high2float(hv));
          if (vf) {
            for (int g = 0; g < G; g++) accF[g][k] = fmaf(S.p[g][i], d, accF[g][k]);
          } else if (signbit(s)) {
            for (int g = 0; g < G; g++) accH[g][k] = fmaf(S.p[g][i], d, accH[g][k]);
          } else {
            for (int g = 0; g < G; g++) accI[g][k] = fmaf(S.p[g][i], d, accI[g][k]);
          }
        }
      } else {
        const __nv_bfloat16* vr = rbase + (long long)(base + i) * 2 * Hkv * D + (long long)(Hkv + h) * D;
#pragma unroll
        for (int k = 0; k < CPT; k++) {
          float d = __bfloat162float(vr[tid + 128 * k]);
          for (int g = 0; g < G; g++) accI[g][k] = fmaf(S.p[g][i], d, accI[g][k]);
        }
      }
    }
    __syncthreads();
  }
  // epilogue: H32 of accH across the lanes of a warp (channel c = tid + 128 k: lane = c % 32), H_D of accF via smem
  if (!tail) {
#pragma unroll
    for (int g = 0; g < 8; g++) {
      if (g >= G) break;
#pragma unroll
      for (int k = 0; k < CPT; k++) {
        float v[1] = {accH[g][k]};
        reg_fwht<1>(v, 32, lane);
        accI[g][k] = fadd(accI[g][k], v[0]);
      }
    }
    // accF -> smem (reuse q[2]), H_D, add
    __syncthreads();
    for (int g = 0; g < G; g++)
      for (int k = 0; k < CPT; k++) S.q[2][g][tid + 128 * k] = accF[g][k];
    __syncthreads();
    smem_fwht<D>(S.q[2], G, D);
    for (int g = 0; g < G; g++)
      for (int k = 0; k < CPT; k++) accI[g][k] = fadd(accI[g][k], S.q[2][g][tid + 128 * k]);
  }
  for (int g = 0; g < G; g++)
    for (int k = 0; k < CPT; k++) op[g * D + tid + 128 * k] = accI[g][k];
#pragma unroll
  for (int r = 0; r < 2; r++) {
    int g = warp + 4 * r;
    if (g < G && lane == 0) { mlp[g * 2] = m_run[r]; mlp[g * 2 + 1] = l_run[r]; }
  }
}

// ------------------------------------------------------------------------------------------------ decode attention v2
// Same contract as k_attn; G templated (registers, no local memory), the group's page data staged in shared memory
// (register prefetch of the next group during the current one), key codes stored transposed [word][token]
// (conflict-free), scores as sum_c (4 q'_c s_c) f_c with f = 1 + code / 4 built by bit operations (no int->float
// conversion) and the constant folded into the per-group bias, PV with the class (I / H32 / H_D) uniform per warp.
template <int D, int G>
struct A2Smem {
  static constexpr int NV = D / 32, PG = (G <= 4 ? 4 : 8);
  float q[3][G][D];
  __align__(16) float qs[G][D];
  __align__(16) uint32_t kcT[2][D / 16][32];
  __align__(16) uint32_t km[2][32 * NV];
  __align__(16) uint8_t vc[2][32 * D / 4];
  __align__(16) uint32_t vm[2][32 * NV];
  __align__(16) uint8_t vf[2][32];
  float part[4][G][33];
  float bias[4][G];
  __align__(16) float p[32][PG];
  float alpha[G];
};

template <int D, int G>
__global__ void __launch_bounds__(128) k_attn2(const __nv_bfloat16* __restrict__ q, long long q_rs, const uint8_t* __restrict__ pool,
                                               const int* __restrict__ bt, long long bt_rs, const int* __restrict__ cached,
                                               int kv_start, int Hkv, int NS, int chunk, float qscale,
                                               const __nv_bfloat16* __restrict__ ring, const int* __restrict__ tail_len,
                                               const int* __restrict__ tail_idx, int T,
                                               float* __restrict__ opart, float* __restrict__ ml) {
  constexpr int CPT = D / 128, NV = D / 32, PG = A2Smem<D, G>::PG;
  constexpr int NCK = 3 * D / 4 + 1;                 // uint4 chunks per page chunk
  constexpr int NLD = (2 * NCK + 127) / 128;         // prefetch registers per thread
  constexpr int R2 = (G + 3) / 4;                    // heads per warp in the softmax phase
  __shared__ A2Smem<D, G> S;
  const int b = blockIdx.x, h = blockIdx.y, sp = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const long long pidx = ((long long)(b * Hkv + h) * (NS + 1) + sp);
  float* op = opart + pidx * G * D;
  float* mlp = ml + pidx * G * 2;
  const bool tail = sp == NS;
  int t0, t1;
  if (tail) { t0 = 0; t1 = tail_len ? tail_len[b] : 0; }
  else { t0 = max(sp * chunk, kv_start); t1 = min((sp + 1) * chunk, cached[b]); }
  if (t1 <= t0) {
    if (tid < G) { mlp[tid * 2] = -INFINITY; mlp[tid * 2 + 1] = 0.f; }
    return;
  }
  const __nv_bfloat16* qb = q + (long long)b * q_rs + (long long)h * G * D;
  for (int e = tid; e < G * D; e += 128) {
    float v = __bfloat162float(qb[e]);
    S.q[0][e / D][e % D] = v; S.q[1][e / D][e % D] = v; S.q[2][e / D][e % D] = v;
  }
  __syncthreads();
  if (!tail) { smem_fwht<D>(S.q[1], G, 32); smem_fwht<D>(S.q[2], G, D); }
  float accI[G][CPT], accH[G][CPT], accF[G][CPT];
#pragma unroll
  for (int g = 0; g < G; g++)
#pragma unroll
    for (int k = 0; k < CPT; k++) { accI[g][k] = 0.f; accH[g][k] = 0.f; accF[g][k] = 0.f; }
  float m_run[R2], l_run[R2];
#pragma unroll
  for (int r = 0; r < R2; r++) { m_run[r] = -INFINITY; l_run[r] = 0.f; }

  // ---------------------------------------------------------------- shared softmax step (scores in S.part / S.bias)
  auto softmax_step = [&](int base, int lo, int hi) {
#pragma unroll
    for (int r = 0; r < R2; r++) {
      int g = warp + 4 * r;
      if (g < G) {
        int pos = base + lane;
        bool valid = pos >= lo && pos < hi;
        float sc = S.part[0][g][lane] + S.part[1][g][lane] + S.part[2][g][lane] + S.part[3][g][lane]
                   + (S.bias[0][g] + S.bias[1][g] + S.bias[2][g] + S.bias[3][g]);
        sc = valid ? sc * qscale : -INFINITY;
        float mx = sc;
#pragma unroll
        for (int m = 16; m >= 1; m >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, m));
        float mnew = fmaxf(m_run[r], mx);
        float pp = valid ? exp2f(sc - mnew) : 0.f;
        float al = (m_run[r] == -INFINITY) ? 0.f : exp2f(m_run[r] - mnew);
        float ps = pp;
#pragma unroll
        for (int m = 16; m >= 1; m >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, m);
        l_run[r] = l_run[r] * al + ps;
        m_run[r] = mnew;
        S.p[lane][g] = pp;
        if (lane == 0) S.alpha[g] = al;
      }
    }
  };

  if (tail) {
    const __nv_bfloat16* rbase = ring + (long long)(tail_idx ? tail_idx[b] : 0) * T * 2 * Hkv * D;
    for (int base = 0; base < t1; base += GRP) {
      float acc[G];
#pragma unroll
      for (int g = 0; g < G; g++) acc[g] = 0.f;
      int j = base + lane;
      if (j < t1) {
        const uint4* kr = reinterpret_cast<const uint4*>(rbase + (long long)j * 2 * Hkv * D + (long long)h * D + warp * (D / 4));
#pragma unroll
        for (int u = 0; u < D / 32; u++) {
          uint4 w = kr[u];
          const __nv_bfloat162* hv = reinterpret_cast<const __nv_bfloat162*>(&w);
#pragma unroll
          for (int e = 0; e < 4; e++) {
            float2 kv = __bfloat1622float2(hv[e]);
            int c = warp * (D / 4) + u * 8 + e * 2;
#pragma unroll
            for (int g = 0; g < G; g++) acc[g] = fmaf(S.q[0][g][c + 1], kv.y, fmaf(S.q[0][g][c], kv.x, acc[g]));
          }
        }
      }
#pragma unroll
      for (int g = 0; g < G; g++) S.part[warp][g][lane] = acc[g];
      if (lane < G) S.bias[warp][lane] = 0.f;
      __syncthreads();
      softmax_step(base, 0, t1);
      __syncthreads();
#pragma unroll
      for (int g = 0; g < G; g++) {
        float al = S.alpha[g];
#pragma unroll
        for (int k = 0; k < CPT; k++) accI[g][k] *= al;
      }
      int nt = min(GRP, t1 - base);
      for (int i = 0; i < nt; i++) {
        const __nv_bfloat16* vr = rbase + (long long)(base + i) * 2 * Hkv * D + (long long)(Hkv + h) * D;
#pragma unroll
        for (int k = 0; k < CPT; k++) {
          float d = __bfloat162float(vr[tid + 128 * k]);
#pragma unroll
          for (int g = 0; g < G; g++) accI[g][k] = fmaf(S.p[i][g], d, accI[g][k]);
        }
      }
      __syncthreads();
    }
  } else {
    const int* btb = bt + (long long)b * bt_rs;
    const long long CB = chunk_bytes(D);
    const int gfirst = t0 / GRP, glast = (t1 - 1) / GRP;
    uint4 pre[NLD];
    auto load = [&](int g) {
      const uint4* pa = reinterpret_cast<const uint4*>(pool + ((long long)btb[2 * g] * Hkv + h) * CB);
      const uint4* pb = reinterpret_cast<const uint4*>(pool + ((long long)btb[2 * g + 1] * Hkv + h) * CB);
#pragma unroll
      for (int r = 0; r < NLD; r++) {
        int ci = tid + 128 * r;
        if (ci < 2 * NCK) pre[r] = ci < NCK ? pa[ci] : pb[ci - NCK];
      }
    };
    auto store = [&](int buf) {
#pragma unroll
      for (int r = 0; r < NLD; r++) {
        int ci = tid + 128 * r;
        if (ci >= 2 * NCK) continue;
        int pg = ci >= NCK, k = ci - pg * NCK;
        uint4 u = pre[r];
        if (k < D / 4) {                                     // key codes -> transposed [word][token]
          int cell = (16 * k) / (D / 4), w0 = ((16 * k) % (D / 4)) / 4, tok = pg * 16 + cell;
          S.kcT[buf][w0][tok] = u.x; S.kcT[buf][w0 + 1][tok] = u.y; S.kcT[buf][w0 + 2][tok] = u.z; S.kcT[buf][w0 + 3][tok] = u.w;
        } else if (k < 3 * D / 8) {
          reinterpret_cast<uint4*>(S.km[buf])[pg * (D / 8) + (k - D / 4)] = u;
        } else if (k < 5 * D / 8) {
          reinterpret_cast<uint4*>(S.vc[buf])[pg * (D / 4) + (k - 3 * D / 8)] = u;
        } else if (k < 3 * D / 4) {
          reinterpret_cast<uint4*>(S.vm[buf])[pg * (D / 8) + (k - 5 * D / 8)] = u;
        } else {
          reinterpret_cast<uint4*>(S.vf[buf])[pg] = u;
        }
      }
    };
    load(gfirst);
    store(0);
    __syncthreads();
    for (int g = gfirst, it = 0; g <= glast; g++, it++) {
      const int cur = it & 1;
      if (g < glast) load(g + 1);
      const int base = g * GRP;
      // (1) q' * 4 s and the bias sum_c q' (min - 4 s)
      {
        const uint32_t* km = S.km[cur];
        const int useF = (km[NV] >> 15) & 1;                  // cell 1, channel NV
        float bp[G];
#pragma unroll
        for (int gg = 0; gg < G; gg++) bp[gg] = 0.f;
#pragma unroll
        for (int k = 0; k < CPT; k++) {
          int c = tid + 128 * k;
          int which = useF ? 2 : (int)((km[c / 32] >> 15) & 1);
          uint32_t w = km[c];                                 // cell c / NV, entry c % NV = flat index c
          __half2 hv = *reinterpret_cast<__half2*>(&w);
          float s4 = 4.f * fabsf(__low2float(hv)), mn = __high2float(hv);
          float cm = mn - s4;
#pragma unroll
          for (int gg = 0; gg < G; gg++) {
            float qp = S.q[which][gg][c];
            S.qs[gg][c] = qp * s4;
            bp[gg] = fmaf(qp, cm, bp[gg]);
          }
        }
#pragma unroll
        for (int gg = 0; gg < G; gg++) {
          float v = bp[gg];
#pragma unroll
          for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
          if (lane == 0) S.bias[warp][gg] = v;
        }
      }
      __syncthreads();
      // (2) partial scores: warp = channel quarter, lane = token
      {
        float acc[G];
#pragma unroll
        for (int gg = 0; gg < G; gg++) acc[gg] = 0.f;
#pragma unroll
        for (int wd = 0; wd < D / 64; wd++) {
          int word = warp * (D / 64) + wd;
          uint32_t u = S.kcT[cur][word][lane];
#pragma unroll
          for (int i4 = 0; i4 < 4; i4++) {
            float f[4];
#pragma unroll
            for (int e = 0; e < 4; e++) {
              int sh = kbit(i4 * 4 + e);
              f[e] = __uint_as_float(0x3F800000u | (((u >> sh) & 3u) << 21));
            }
            int c = word * 16 + i4 * 4;
#pragma unroll
            for (int gg = 0; gg < G; gg++) {
              float4 qv = *reinterpret_cast<const float4*>(&S.qs[gg][c]);
              acc[gg] = fmaf(qv.x, f[0], acc[gg]);
              acc[gg] = fmaf(qv.y, f[1], acc[gg]);
              acc[gg] = fmaf(qv.z, f[2], acc[gg]);
              acc[gg] = fmaf(qv.w, f[3], acc[gg]);
            }
          }
        }
#pragma unroll
        for (int gg = 0; gg < G; gg++) S.part[warp][gg][lane] = acc[gg];
      }
      __syncthreads();
      softmax_step(base, t0, t1);
      __syncthreads();
      // (3) PV
#pragma unroll
      for (int gg = 0; gg < G; gg++) {
        float al = S.alpha[gg];
#pragma unroll
        for (int k = 0; k < CPT; k++) { accI[gg][k] *= al; accH[gg][k] *= al; accF[gg][k] *= al; }
      }
      const int i0 = max(0, t0 - base), nt = min(GRP, t1 - base);
      for (int i = i0; i < nt; i++) {
        float pv[PG];
#pragma unroll
        for (int q4 = 0; q4 < PG; q4 += 4) {
          float4 t4 = *reinterpret_cast<const float4*>(&S.p[i][q4]);
          pv[q4] = t4.x; pv[q4 + 1] = t4.y; pv[q4 + 2] = t4.z; pv[q4 + 3] = t4.w;
        }
        const int vfl = S.vf[cur][i];
#pragma unroll
        for (int k = 0; k < CPT; k++) {
          int c = tid + 128 * k;
          uint32_t code = (S.vc[cur][i * (D / 4) + c / 4] >> (2 * (c % 4))) & 3u;
          float cf = __uint_as_float(0x4B000000u | code) - 8388608.f;
          uint32_t w = S.vm[cur][i * NV + c / 32];
          __half2 hv = *reinterpret_cast<__half2*>(&w);
          float sv = __low2float(hv);
          float d = fmaf(cf, fabsf(sv), __high2float(hv));
          if (vfl) {
#pragma unroll
            for (int gg = 0; gg < G; gg++) accF[gg][k] = fmaf(pv[gg], d, accF[gg][k]);
          } else if (w & 0x8000u) {
#pragma unroll
            for (int gg = 0; gg < G; gg++) accH[gg][k] = fmaf(pv[gg], d, accH[gg][k]);
          } else {
#pragma unroll
            for (int gg = 0; gg < G; gg++) accI[gg][k] = fmaf(pv[gg], d, accI[gg][k]);
          }
        }
      }
      if (g < glast) store(cur ^ 1);
      __syncthreads();
    }
    // epilogue: inverse H32 (lanes) and H_D (shared memory) of the class accumulators
#pragma unroll
    for (int gg = 0; gg < G; gg++)
#pragma unroll
      for (int k = 0; k < CPT; k++) {
        float v[1] = {accH[gg][k]};
        reg_fwht<1>(v, 32, lane);
        accI[gg][k] = fadd(accI[gg][k], v[0]);
      }
#pragma unroll
    for (int gg = 0; gg < G; gg++)
#pragma unroll
      for (int k = 0; k < CPT; k++) S.q[2][gg][tid + 128 * k] = accF[gg][k];
    __syncthreads();
    smem_fwht<D>(S.q[2], G, D);
#pragma unroll
    for (int gg = 0; gg < G; gg++)
#pragma unroll
      for (int k = 0; k < CPT; k++) accI[gg][k] = fadd(accI[gg][k], S.q[2][gg][tid + 128 * k]);
  }
#pragma unroll
  for (int gg = 0; gg < G; gg++)
#pragma unroll
    for (int k = 0; k < CPT; k++) op[gg * D + tid + 128 * k] = accI[gg][k];
#pragma unroll
  for (int r = 0; r < R2; r++) {
    int g = warp + 4 * r;
    if (g < G && lane == 0) { mlp[g * 2] = m_run[r]; mlp[g * 2 + 1] = l_run[r]; }
  }
}

// ------------------------------------------------------------------------------------------------ decode attention v3
// k_attn2 with the two products on the tensor cores (mma.sync m16n8k16, FP16 operands, FP32 accumulation):
//   scores  S[token, head] = sum_c code[token, c] (q'_c s_c)  (A = exact 2-bit codes, B = q' * s in FP16) + bias
//   PV      O[c, head]     = sum_t code[t, c] (p_t s_t) + sum_t p_t min_t  per value class (I / H32 / H_D) and 32-dim
//           sub-vector (B = p * s or p * min masked by the token's class, FP16; the min term uses an all-ones A)
// the class sums are mapped back (H32 / H_D) once per CTA in shared memory.  Heads padded to n = 8.
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// the same without volatile (kivi2p8f: lets ptxas interleave the MMAs with the operand construction; same results)
__device__ __forceinline__ void mma16816f(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <bool FAST>
__device__ __forceinline__ void mma_k3(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  if constexpr (FAST) mma16816f(c, a, b0, b1); else mma16816(c, a, b0, b1);
}
// two 2-bit codes (bits 1:0 and 3:2 of x) -> half2 (exact integers)
__device__ __forceinline__ uint32_t codes2(uint32_t x) {
  uint32_t v = (x & 3u) | ((x & 0xCu) << 14) | 0x64006400u;
  __half2 h = __hsub2(*reinterpret_cast<__half2*>(&v), __halves2half2(__float2half(1024.f), __float2half(1024.f)));
  return *reinterpret_cast<uint32_t*>(&h);
}
// kivi2p8f: codesw with one sub.f16x2 against the magic constant (the same exact integers)
__device__ __forceinline__ uint32_t codesw_f(uint32_t x) {
  uint32_t v = (x & 0x00030003u) | 0x64006400u, r;
  asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(v), "r"(0x64006400u));
  return r;
}
// codes at bits 1:0 (half 0) and 17:16 (half 1) of x -> half2 (exact integers)
__device__ __forceinline__ uint32_t codesw(uint32_t x) {
  uint32_t v = (x & 0x00030003u) | 0x64006400u;
  __half2 h = __hsub2(*reinterpret_cast<__half2*>(&v), __halves2half2(__float2half(1024.f), __float2half(1024.f)));
  return *reinterpret_cast<uint32_t*>(&h);
}
template <bool FAST>
__device__ __forceinline__ uint32_t codesw_k(uint32_t x) {
  if constexpr (FAST) return codesw_f(x); else return codesw(x);
}
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

template <int D, int G>
struct A3Smem {
  static constexpr int NV = D / 32;
  float qx[3][8][D];                        // q / H32 q / H_D q (loop), class outputs X[cls][head][c] (epilogue)
  __align__(16) __half qsh[8][D + 8];
  __align__(16) uint32_t kcT[2][D / 16][32];
  __align__(16) uint32_t km[2][32 * NV];
  __align__(16) uint8_t vc[2][32 * D / 4];
  __align__(16) uint32_t vm[2][32 * NV];
  __align__(16) uint8_t vf[2][32];
  float part[2][32][9];
  float bias[4][8];
  float p[32][9];
  float alpha[8];
};

template <int D, int G, bool F8 = false, bool FAST = false>
__global__ void __launch_bounds__(128, !FAST ? 1 : D == 256 ? KA3F_MINB_256 : G <= 4 ? KA3F_MINB_128S : KA3F_MINB_128L) k_attn3(const __nv_bfloat16* __restrict__ q, long long q_rs, const uint8_t* __restrict__ pool,
                                               const int* __restrict__ bt, long long bt_rs, const int* __restrict__ cached,
                                               int kv_start, int Hkv, int NS, int chunk, float qscale,
                                               const __nv_bfloat16* __restrict__ ring, const int* __restrict__ tail_len,
                                               const int* __restrict__ tail_idx, int T,
                                               float* __restrict__ opart, float* __restrict__ ml) {
  constexpr int CPT = D / 128, NV = D / 32;
  constexpr int NCK = F8 ? 10 * D / 16 : 3 * D / 4 + 1;   // uint4 chunks per page chunk
  constexpr int NLD = (2 * NCK + 127) / 128;
  constexpr int CHL = F8 ? NV : 1;                          // channel of key flag j: j CHL (f8: cell j, slot 0)
  constexpr int R2 = (G + 3) / 4;
  constexpr bool RAW = FAST && F8;      // FP8 meta kept raw (u16 pairs in the km / vm buffers), converted at use
  __shared__ A3Smem<D, G> S;
  // meta words: half2 (scale with its flag sign | min << 16) of key channel / value (token, sub-vector) idx
  auto kmw = [&](int buf, int idx) -> uint32_t {
    if constexpr (RAW) { __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(reinterpret_cast<const uint16_t*>(S.km[buf])[idx], __NV_E4M3); return (uint32_t)r.x | ((uint32_t)r.y << 16); }
    else return S.km[buf][idx];
  };
  // kivi2p8f (RAW): phase (1) converts the group's value meta into the other vm buffer (free until the end-of-
  // iteration store) and keeps the key minima as halves in the unused upper half of its km buffer
  auto vmw = [&](int buf, int idx) -> uint32_t {
    if constexpr (RAW) return S.vm[buf ^ 1][idx];
    else return S.vm[buf][idx];
  };
  auto kmin = [&](int buf, int c) -> float {
    if constexpr (RAW) return __half2float(reinterpret_cast<const __half*>(S.km[buf])[32 * NV + c]);
    else { uint32_t w = S.km[buf][c]; return __high2float(*reinterpret_cast<__half2*>(&w)); }
  };
  auto kflag = [&](int buf, int idx) -> int {
    if constexpr (RAW) return (reinterpret_cast<const uint16_t*>(S.km[buf])[idx] >> 7) & 1; else return (S.km[buf][idx] >> 15) & 1;
  };
  auto vfl = [&](int buf, int tok) -> int {
    if constexpr (RAW) return (reinterpret_cast<const uint16_t*>(S.km[buf])[tok * NV + NV - 1] >> 7) & 1; else return S.vf[buf][tok];
  };
  const int b = blockIdx.x, h = blockIdx.y, sp = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, t4 = lane & 3;
  const long long pidx = ((long long)(b * Hkv + h) * (NS + 1) + sp);
  float* op = opart + pidx * G * D;
  float* mlp = ml + pidx * G * 2;
  const bool tail = sp == NS;
  int t0, t1;
  if (tail) { t0 = 0; t1 = tail_len ? tail_len[b] : 0; }
  else { t0 = max(sp * chunk, kv_start); t1 = min((sp + 1) * chunk, cached[b]); }
  if (t1 <= t0) {
    if (tid < G) { mlp[tid * 2] = -INFINITY; mlp[tid * 2 + 1] = 0.f; }
    return;
  }
  const __nv_bfloat16* qb = q + (long long)b * q_rs + (long long)h * G * D;
  for (int e = tid; e < 8 * D; e += 128) {
    int g = e / D, c = e % D;
    float v = g < G ? __bfloat162float(qb[g * D + c]) : 0.f;
    S.qx[0][g][c] = v; S.qx[1][g][c] = v; S.qx[2][g][c] = v;
  }
  if constexpr (FAST) {                     // padded heads: p stays 0 (their PV columns stay 0); one bias row
    for (int e = tid; e < 32 * 8; e += 128)
      if (e % 8 >= G) S.p[e / 8][e % 8] = 0.f;
    if (tid < 32) S.bias[tid / 8][tid % 8] = 0.f;
  }
  __syncthreads();
  float m_run[R2], l_run[R2];
#pragma unroll
  for (int r = 0; r < R2; r++) { m_run[r] = -INFINITY; l_run[r] = 0.f; }
  auto softmax_step = [&](int base, int lo, int hi, bool two) {
#pragma unroll
    for (int r = 0; r < R2; r++) {
      int g = warp + 4 * r;
      if (g < G) {
        int pos = base + lane;
        bool valid = pos >= lo && pos < hi;
        float sc = S.part[0][lane][g] + (two ? S.part[1][lane][g] : 0.f)
                   + (S.bias[0][g] + S.bias[1][g] + S.bias[2][g] + S.bias[3][g]);
        sc = valid ? sc * qscale : -INFINITY;
        float mx = sc;
#pragma unroll
        for (int m = 16; m >= 1; m >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, m));
        float mnew = fmaxf(m_run[r], mx);
        float pp = valid ? exp2f(sc - mnew) : 0.f;
        float al = (m_run[r] == -INFINITY) ? 0.f : exp2f(m_run[r] - mnew);
        float ps = pp;
#pragma unroll
        for (int m = 16; m >= 1; m >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, m);
        l_run[r] = l_run[r] * al + ps;
        m_run[r] = mnew;
        S.p[lane][g] = pp;
        if (lane == 0) S.alpha[g] = al;
      }
    }
  };

  if (tail) {
    float accI[G][CPT];
#pragma unroll
    for (int g = 0; g < G; g++)
#pragma unroll
      for (int k = 0; k < CPT; k++) accI[g][k] = 0.f;
    const __nv_bfloat16* rbase = ring + (long long)(tail_idx ? tail_idx[b] : 0) * T * 2 * Hkv * D;
    for (int base = 0; base < t1; base += GRP) {
      // scores: warp = channel quarter, lane = token; quarters summed through part[0] / part[1] and bias rows
      float acc[G];
#pragma unroll
      for (int g = 0; g < G; g++) acc[g] = 0.f;
      int j = base + lane;
      if (j < t1) {
        const uint4* kr = reinterpret_cast<const uint4*>(rbase + (long long)j * 2 * Hkv * D + (long long)h * D + warp * (D / 4));
#pragma unroll
        for (int u = 0; u < D / 32; u++) {
          uint4 w = kr[u];
          const __nv_bfloat162* hv = reinterpret_cast<const __nv_bfloat162*>(&w);
#pragma unroll
          for (int e = 0; e < 4; e++) {
            float2 kv = __bfloat1622float2(hv[e]);
            int c = warp * (D / 4) + u * 8 + e * 2;
#pragma unroll
            for (int g = 0; g < G; g++) acc[g] = fmaf(S.qx[0][g][c + 1], kv.y, fmaf(S.qx[0][g][c], kv.x, acc[g]));
          }
        }
      }
      __shared__ float tpart[4][G][32];
#pragma unroll
      for (int g = 0; g < G; g++) tpart[warp][g][lane] = acc[g];
      __syncthreads();
      for (int e = tid; e < 32 * G; e += 128) {
        int g = e / 32, l = e % 32;
        S.part[0][l][g] = tpart[0][g][l] + tpart[1][g][l] + tpart[2][g][l] + tpart[3][g][l];
      }
      if (tid < 32) S.bias[tid / 8][tid % 8] = 0.f;
      __syncthreads();
      softmax_step(base, 0, t1, false);
      __syncthreads();
#pragma unroll
      for (int g = 0; g < G; g++) {
        float al = S.alpha[g];
#pragma unroll
        for (int k = 0; k < CPT; k++) accI[g][k] *= al;
      }
      int nt = min(GRP, t1 - base);
      for (int i = 0; i < nt; i++) {
        const __nv_bfloat16* vr = rbase + (long long)(base + i) * 2 * Hkv * D + (long long)(Hkv + h) * D;
#pragma unroll
        for (int k = 0; k < CPT; k++) {
          float d = __bfloat162float(vr[tid + 128 * k]);
#pragma unroll
          for (int g = 0; g < G; g++) accI[g][k] = fmaf(S.p[i][g], d, accI[g][k]);
        }
      }
      __syncthreads();
    }
#pragma unroll
    for (int g = 0; g < G; g++)
#pragma unroll
      for (int k = 0; k < CPT; k++) op[g * D + tid + 128 * k] = accI[g][k];
  } else {
    if constexpr (FAST) { smem_fwht_f<D>(S.qx[1], G, 32); smem_fwht_f<D>(S.qx[2], G, D); }   // rows >= G are 0
    else { smem_fwht<D>(S.qx[1], 8, 32); smem_fwht<D>(S.qx[2], 8, D); }
    for (int e = tid; e < 8 * (D + 8); e += 128) (&S.qsh[0][0])[e] = __float2half(0.f);   // padded heads stay 0
    const int* btb = bt + (long long)b * bt_rs;
    const long long CB = F8 ? chunk_bytes8(D) : chunk_bytes(D);
    const int gfirst = t0 / GRP, glast = (t1 - 1) / GRP;
    uint4 pre[NLD];
    auto load_pg = [&](int p0, int p1) {
      const uint4* pa = reinterpret_cast<const uint4*>(pool + ((long long)p0 * Hkv + h) * CB);
      const uint4* pb = reinterpret_cast<const uint4*>(pool + ((long long)p1 * Hkv + h) * CB);
#pragma unroll
      for (int r = 0; r < NLD; r++) {
        int ci = tid + 128 * r;
        if (ci < 2 * NCK) pre[r] = ci < NCK ? pa[ci] : pb[ci - NCK];
      }
    };
    auto load = [&](int g) { load_pg(btb[2 * g], btb[2 * g + 1]); };

    auto store = [&](int buf) {
#pragma unroll
      for (int r = 0; r < NLD; r++) {
        int ci = tid + 128 * r;
        if (ci >= 2 * NCK) continue;
        int pg = ci >= NCK, k = ci - pg * NCK;
        uint4 u = pre[r];
        if constexpr (RAW) {
          if (k < D / 4) {
            int cell = (16 * k) / (D / 4), w0 = ((16 * k) % (D / 4)) / 4, tok = pg * 16 + cell;
            S.kcT[buf][w0][tok] = u.x; S.kcT[buf][w0 + 1][tok] = u.y; S.kcT[buf][w0 + 2][tok] = u.z; S.kcT[buf][w0 + 3][tok] = u.w;
          } else if (k < D / 4 + D / 16) {
            reinterpret_cast<uint4*>(S.km[buf])[pg * (D / 16) + (k - D / 4)] = u;
          } else if (k >= 9 * D / 16) {
            reinterpret_cast<uint4*>(S.vm[buf])[pg * (D / 16) + (k - 9 * D / 16)] = u;
          } else {
            reinterpret_cast<uint4*>(S.vc[buf])[pg * (D / 4) + (k - D / 4 - D / 16)] = u;
          }
          continue;
        }
        if (F8) {
          // kivi2p8: FP8 (s8 | flag, m8) byte pairs -> the half2 (scale with its sign, min) words of the FP16 layout;
          // the value whole-head flag of token cell i = sign of the key scale of cell i slot NV - 1
          if (k < D / 4) {
            int cell = (16 * k) / (D / 4), w0 = ((16 * k) % (D / 4)) / 4, tok = pg * 16 + cell;
            S.kcT[buf][w0][tok] = u.x; S.kcT[buf][w0 + 1][tok] = u.y; S.kcT[buf][w0 + 2][tok] = u.z; S.kcT[buf][w0 + 3][tok] = u.w;
          } else if (k < D / 4 + D / 16 || (k >= 9 * D / 16 && k < 10 * D / 16)) {
            const bool isk = k < D / 4 + D / 16;
            const int pbase = 8 * (isk ? k - D / 4 : k - 9 * D / 16);     // byte-pair index within the page region
            const uint8_t* bb = reinterpret_cast<const uint8_t*>(&u);
#pragma unroll
            for (int e = 0; e < 8; e++) {
              int pi = pbase + e, cell = pi / NV, slot = pi % NV;
              __half_raw hs = __nv_cvt_fp8_to_halfraw(bb[2 * e], __NV_E4M3);
              __half_raw hm = __nv_cvt_fp8_to_halfraw(bb[2 * e + 1], __NV_E4M3);
              uint32_t w = (uint32_t)hs.x | ((uint32_t)hm.x << 16);
              if (isk) {
                S.km[buf][(pg * 16 + cell) * NV + slot] = w;
                if (slot == NV - 1) S.vf[buf][pg * 16 + cell] = bb[2 * e] >> 7;
              } else {
                S.vm[buf][(pg * 16 + cell) * NV + slot] = w;
              }
            }
          } else {
            reinterpret_cast<uint4*>(S.vc[buf])[pg * (D / 4) + (k - D / 4 - D / 16)] = u;
          }
          continue;
        }
        if (k < D / 4) {
          int cell = (16 * k) / (D / 4), w0 = ((16 * k) % (D / 4)) / 4, tok = pg * 16 + cell;
          S.kcT[buf][w0][tok] = u.x; S.kcT[buf][w0 + 1][tok] = u.y; S.kcT[buf][w0 + 2][tok] = u.z; S.kcT[buf][w0 + 3][tok] = u.w;
        } else if (k < 3 * D / 8) {
          reinterpret_cast<uint4*>(S.km[buf])[pg * (D / 8) + (k - D / 4)] = u;
        } else if (k < 5 * D / 8) {
          reinterpret_cast<uint4*>(S.vc[buf])[pg * (D / 4) + (k - 3 * D / 8)] = u;
        } else if (k < 3 * D / 4) {
          reinterpret_cast<uint4*>(S.vm[buf])[pg * (D / 8) + (k - 5 * D / 8)] = u;
        } else {
          reinterpret_cast<uint4*>(S.vf[buf])[pg] = u;
        }
      }
    };
    // PV accumulators: per owned sub-vector (warp + 4 k), class, m-tile (16 channels): 16 x 8 fragment; min terms
    float Cc[CPT][3][2][4], Cm[CPT][3][4];
#pragma unroll
    for (int k = 0; k < CPT; k++)
#pragma unroll
      for (int c = 0; c < 3; c++) {
#pragma unroll
        for (int e = 0; e < 4; e++) { Cc[k][c][0][e] = 0.f; Cc[k][c][1][e] = 0.f; Cm[k][c][e] = 0.f; }
      }
    const uint32_t ONES = 0x3C003C00u;
    const uint32_t onesA[4] = {ONES, ONES, ONES, ONES};
    load(gfirst);
    store(0);
    __syncthreads();
    for (int g = gfirst, it = 0; g <= glast; g++, it++) {
      const int cur = it & 1;
      if (g < glast) load(g + 1);
      const int base = g * GRP;
      // (1) qs = q' s (FP16, heads padded with 0) and bias = sum_c q' min (FP32)
      if constexpr (FAST) {
        // q' s per channel for every head (thread = channel), the channel minima (bias: per warp in the softmax
        // step) and the group's value meta words
        const int useF = kflag(cur, NV * CHL);
#pragma unroll
        for (int k = 0; k < CPT; k++) {
          int c = tid + 128 * k;
          int which = useF ? 2 : kflag(cur, (c / 32) * CHL);
          uint32_t w = kmw(cur, c);
          __half2 hv = *reinterpret_cast<__half2*>(&w);
          float sv = fabsf(__low2float(hv));
          if constexpr (RAW) reinterpret_cast<__half*>(S.km[cur])[32 * NV + c] = __high2half(hv);
#pragma unroll
          for (int gg = 0; gg < G; gg++) S.qsh[gg][c] = __float2half_rn(S.qx[which][gg][c] * sv);
        }
        if constexpr (RAW) {
#pragma unroll
          for (int k = 0; k < (32 * NV + 127) / 128; k++) {
            int idx = tid + 128 * k;
            if (idx < 32 * NV) {
              __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(reinterpret_cast<const uint16_t*>(S.vm[cur])[idx], __NV_E4M3);
              S.vm[cur ^ 1][idx] = (uint32_t)r.x | ((uint32_t)r.y << 16);
            }
          }
        }
      } else {
        const uint32_t* km = S.km[cur];
        const int useF = (km[NV * CHL] >> 15) & 1;
        float bp[8];
#pragma unroll
        for (int gg = 0; gg < 8; gg++) bp[gg] = 0.f;
#pragma unroll
        for (int k = 0; k < CPT; k++) {
          int c = tid + 128 * k;
          int which = useF ? 2 : (int)((km[(c / 32) * CHL] >> 15) & 1);
          uint32_t w = km[c];
          __half2 hv = *reinterpret_cast<__half2*>(&w);
          float sv = fabsf(__low2float(hv)), mn = __high2float(hv);
#pragma unroll
          for (int gg = 0; gg < G; gg++) {
            float qp = S.qx[which][gg][c];
            S.qsh[gg][c] = __float2half_rn(qp * sv);
            bp[gg] = fmaf(qp, mn, bp[gg]);
          }
        }
#pragma unroll
        for (int gg = 0; gg < G; gg++) {
          float v = bp[gg];
#pragma unroll
          for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
          if (lane == 0) S.bias[warp][gg] = v;
        }
      }
      __syncthreads();
      // (2) scores: warp w -> m-tile (tokens) w & 1, k-steps (channels) half w >> 1
#ifndef KIVI_SKIP_SC
      {
        const int mt = warp & 1, kh = warp >> 1;
        float c4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kk = 0; kk < D / 32; kk++) {
          int ks = kh * (D / 32) + kk;                         // 16-channel step = key-code word ks
          uint32_t u0 = S.kcT[cur][ks][mt * 16 + gid], u1 = S.kcT[cur][ks][mt * 16 + gid + 8];
          uint32_t a[4] = {codesw_k<FAST>(u0 >> (2 * t4)), codesw_k<FAST>(u1 >> (2 * t4)), codesw_k<FAST>(u0 >> (2 * t4 + 8)),
                           codesw_k<FAST>(u1 >> (2 * t4 + 8))};
          uint32_t b0 = *reinterpret_cast<const uint32_t*>(&S.qsh[gid][ks * 16 + 2 * t4]);
          uint32_t b1 = *reinterpret_cast<const uint32_t*>(&S.qsh[gid][ks * 16 + 2 * t4 + 8]);
          mma_k3<FAST>(c4, a, b0, b1);
        }
        S.part[kh][mt * 16 + gid][2 * t4] = c4[0];
        S.part[kh][mt * 16 + gid][2 * t4 + 1] = c4[1];
        S.part[kh][mt * 16 + gid + 8][2 * t4] = c4[2];
        S.part[kh][mt * 16 + gid + 8][2 * t4 + 1] = c4[3];
      }
#endif
      __syncthreads();
      if constexpr (FAST) {
        const int useF = kflag(cur, NV * CHL);
#pragma unroll
        for (int r = 0; r < R2; r++) {
          const int hd = warp + 4 * r;
          if (hd < G) {
            float v = 0.f;
#pragma unroll
            for (int i = 0; i < D / 32; i++) {
              const int wh = useF ? 2 : kflag(cur, i * CHL);
              v = fmaf(S.qx[wh][hd][lane + 32 * i], kmin(cur, lane + 32 * i), v);
            }
#pragma unroll
            for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
            int pos = base + lane;
            bool valid = pos >= t0 && pos < t1;
            float sc = S.part[0][lane][hd] + S.part[1][lane][hd] + v;
            sc = valid ? sc * qscale : -INFINITY;
            // the running max moves only if some score exceeds it (one vote; else max(m_run, mx) == m_run exactly)
            float mnew = m_run[r], al = 1.f;
            if (__any_sync(0xffffffffu, sc > m_run[r])) {
              float mx = sc;
#pragma unroll
              for (int m = 16; m >= 1; m >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, m));
              mnew = fmaxf(m_run[r], mx);
              al = (m_run[r] == -INFINITY) ? 0.f : exp2f(m_run[r] - mnew);
            }
            float pp = valid ? exp2f(sc - mnew) : 0.f;
            l_run[r] = l_run[r] * al + pp;          // per-lane partial denominators, summed over lanes at the end
            m_run[r] = mnew;
            S.p[lane][hd] = pp;
            if (lane == 0) S.alpha[hd] = al;
          }
        }
      } else {
        softmax_step(base, t0, t1, true);
      }
      __syncthreads();
      // (3) PV on the tensor cores
#ifndef KIVI_SKIP_PV
      if constexpr (FAST && G <= 4) {
        const uint32_t vfm = __ballot_sync(0xffffffffu, vfl(cur, lane));   // value whole-head flags of the 32 tokens
        // kivi2p8f, G <= 4: two value classes share one MMA (n = 8 columns: column n < 4 = class I of head n,
        // n >= 4 = class H32 of head n - 4; the second MMA: class H_D of head n < 4, zero columns n >= 4); the
        // per-column products / sums are those of the three-MMA path
        const int hh = gid & 3;
        const bool hi = gid >= 4;
        const int h0 = (2 * t4) & 3, h1 = h0 + 1;
        float a0 = h0 < G ? S.alpha[h0] : 0.f, a1 = h1 < G ? S.alpha[h1] : 0.f;
        // rescale only when a valid head's running max moved (alpha == 1 exactly otherwise)
        if (__any_sync(0xffffffffu, (h0 < G && a0 != 1.f) || (h1 < G && a1 != 1.f))) {
#pragma unroll
          for (int k = 0; k < CPT; k++)
#pragma unroll
            for (int c = 0; c < 2; c++) {
#pragma unroll
              for (int m = 0; m < 2; m++) {
                Cc[k][c][m][0] *= a0; Cc[k][c][m][1] *= a1; Cc[k][c][m][2] *= a0; Cc[k][c][m][3] *= a1;
              }
              Cm[k][c][0] *= a0; Cm[k][c][1] *= a1; Cm[k][c][2] *= a0; Cm[k][c][3] *= a1;
            }
        }
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
          const int ta = ks * 16 + 2 * t4;
          const uint32_t p01 = pack2(S.p[ta][hh], S.p[ta + 1][hh]), p89 = pack2(S.p[ta + 8][hh], S.p[ta + 9][hh]);
          const uint32_t b01 = vfm >> ta, b89 = vfm >> (ta + 8);
          const uint32_t f01 = ((b01 & 1u) ? 0xFFFFu : 0u) | ((b01 & 2u) ? 0xFFFF0000u : 0u);
          const uint32_t f89 = ((b89 & 1u) ? 0xFFFFu : 0u) | ((b89 & 2u) ? 0xFFFF0000u : 0u);
#pragma unroll
          for (int k = 0; k < CPT; k++) {
            const int j = warp + 4 * k;
            uint32_t B0[2], B1[2], M0[2], M1[2];
#pragma unroll
            for (int pr = 0; pr < 2; pr++) {
              const int t = ta + 8 * pr;
              const uint32_t wA = vmw(cur, t * NV + j), wB = vmw(cur, (t + 1) * NV + j);
              uint32_t s2 = __byte_perm(wA, wB, 0x5410), m2 = __byte_perm(wA, wB, 0x7632);
              uint32_t mF = pr ? f89 : f01;
              uint32_t mH = ((s2 >> 15) & 0x00010001u) * 0xFFFFu & ~mF;
              uint32_t mI = ~(mH | mF);
              s2 &= 0x7FFF7FFFu;
              uint32_t pp = pr ? p89 : p01;
              __half2 ps = __hmul2(*reinterpret_cast<__half2*>(&pp), *reinterpret_cast<__half2*>(&s2));
              __half2 pm = __hmul2(*reinterpret_cast<__half2*>(&pp), *reinterpret_cast<__half2*>(&m2));
              uint32_t psu = *reinterpret_cast<uint32_t*>(&ps), pmu = *reinterpret_cast<uint32_t*>(&pm);
              const uint32_t m0 = hi ? mH : mI, m1 = hi ? 0u : mF;
              uint32_t* Bc = pr ? B1 : B0;
              uint32_t* Bm = pr ? M1 : M0;
              Bc[0] = psu & m0; Bc[1] = psu & m1;
              Bm[0] = pmu & m0; Bm[1] = pmu & m1;
            }
#pragma unroll
            for (int m = 0; m < 2; m++) {
              const int mb = 32 * j + 16 * m;
              const uint32_t wa = *reinterpret_cast<const uint32_t*>(&S.vc[cur][ta * (D / 4) + mb / 4]);
              const uint32_t wb = *reinterpret_cast<const uint32_t*>(&S.vc[cur][(ta + 1) * (D / 4) + mb / 4]);
              const uint32_t wc = *reinterpret_cast<const uint32_t*>(&S.vc[cur][(ta + 8) * (D / 4) + mb / 4]);
              const uint32_t wd = *reinterpret_cast<const uint32_t*>(&S.vc[cur][(ta + 9) * (D / 4) + mb / 4]);
              const int sh = 2 * gid;
              uint32_t a[4] = {codesw_k<FAST>(__byte_perm(wa, wb, 0x5410) >> sh), codesw_k<FAST>(__byte_perm(wa, wb, 0x7632) >> sh),
                               codesw_k<FAST>(__byte_perm(wc, wd, 0x5410) >> sh), codesw_k<FAST>(__byte_perm(wc, wd, 0x7632) >> sh)};
              mma16816f(Cc[k][0][m], a, B0[0], B1[0]);
              mma16816f(Cc[k][1][m], a, B0[1], B1[1]);
            }
            mma16816f(Cm[k][0], onesA, M0[0], M1[0]);
            mma16816f(Cm[k][1], onesA, M0[1], M1[1]);
          }
        }
      } else
      {
        float a0 = S.alpha[min(2 * t4, 7)], a1 = S.alpha[min(2 * t4 + 1, 7)];
        if (2 * t4 >= G) a0 = 0.f;
        if (2 * t4 + 1 >= G) a1 = 0.f;
        const bool resc = !FAST || __any_sync(0xffffffffu, (2 * t4 < G && a0 != 1.f) || (2 * t4 + 1 < G && a1 != 1.f));
        if (resc) {
#pragma unroll
        for (int k = 0; k < CPT; k++)
#pragma unroll
          for (int c = 0; c < 3; c++) {
#pragma unroll
            for (int m = 0; m < 2; m++) {
              Cc[k][c][m][0] *= a0; Cc[k][c][m][1] *= a1; Cc[k][c][m][2] *= a0; Cc[k][c][m][3] *= a1;
            }
            Cm[k][c][0] *= a0; Cm[k][c][1] *= a1; Cm[k][c][2] *= a0; Cm[k][c][3] *= a1;
          }
        }
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {                      // 16 tokens per k-step
          const int ta = ks * 16 + 2 * t4;                    // B rows / A columns: tokens ta, ta + 1, ta + 8, ta + 9
          // p of head gid for the 4 tokens as half2 pairs
          const uint32_t p01 = pack2(S.p[ta][gid], S.p[ta + 1][gid]), p89 = pack2(S.p[ta + 8][gid], S.p[ta + 9][gid]);
          const uint32_t f01 = (vfl(cur, ta) ? 0xFFFFu : 0u) | (vfl(cur, ta + 1) ? 0xFFFF0000u : 0u);
          const uint32_t f89 = (vfl(cur, ta + 8) ? 0xFFFFu : 0u) | (vfl(cur, ta + 9) ? 0xFFFF0000u : 0u);
#pragma unroll
          for (int k = 0; k < CPT; k++) {
            const int j = warp + 4 * k;                      // sub-vector = channels [32 j, 32 j + 32)
            uint32_t B0[3], B1[3], M0[3], M1[3];
#pragma unroll
            for (int pr = 0; pr < 2; pr++) {
              const int t = ta + 8 * pr;
              const uint32_t wA = vmw(cur, t * NV + j), wB = vmw(cur, (t + 1) * NV + j);
              uint32_t s2 = __byte_perm(wA, wB, 0x5410), m2 = __byte_perm(wA, wB, 0x7632);
              uint32_t mF = pr ? f89 : f01;
              uint32_t mH = ((s2 >> 15) & 0x00010001u) * 0xFFFFu & ~mF;
              uint32_t mI = ~(mH | mF);
              s2 &= 0x7FFF7FFFu;
              uint32_t pp = pr ? p89 : p01;
              __half2 ps = __hmul2(*reinterpret_cast<__half2*>(&pp), *reinterpret_cast<__half2*>(&s2));
              __half2 pm = __hmul2(*reinterpret_cast<__half2*>(&pp), *reinterpret_cast<__half2*>(&m2));
              uint32_t psu = *reinterpret_cast<uint32_t*>(&ps), pmu = *reinterpret_cast<uint32_t*>(&pm);
              uint32_t* Bc = pr ? B1 : B0;
              uint32_t* Bm = pr ? M1 : M0;
              Bc[0] = psu & mI; Bc[1] = psu & mH; Bc[2] = psu & mF;
              Bm[0] = pmu & mI; Bm[1] = pmu & mH; Bm[2] = pmu & mF;
            }
#pragma unroll
            for (int m = 0; m < 2; m++) {
              const int mb = 32 * j + 16 * m;                // A rows: channels mb + gid, mb + gid + 8
              const uint32_t wa = *reinterpret_cast<const uint32_t*>(&S.vc[cur][ta * (D / 4) + mb / 4]);
              const uint32_t wb = *reinterpret_cast<const uint32_t*>(&S.vc[cur][(ta + 1) * (D / 4) + mb / 4]);
              const uint32_t wc = *reinterpret_cast<const uint32_t*>(&S.vc[cur][(ta + 8) * (D / 4) + mb / 4]);
              const uint32_t wd = *reinterpret_cast<const uint32_t*>(&S.vc[cur][(ta + 9) * (D / 4) + mb / 4]);
              const int sh = 2 * gid;
              uint32_t a[4] = {codesw_k<FAST>(__byte_perm(wa, wb, 0x5410) >> sh), codesw_k<FAST>(__byte_perm(wa, wb, 0x7632) >> sh),
                               codesw_k<FAST>(__byte_perm(wc, wd, 0x5410) >> sh), codesw_k<FAST>(__byte_perm(wc, wd, 0x7632) >> sh)};
#pragma unroll
              for (int c = 0; c < 3; c++) mma_k3<FAST>(Cc[k][c][m], a, B0[c], B1[c]);
            }
#pragma unroll
            for (int c = 0; c < 3; c++) mma_k3<FAST>(Cm[k][c], onesA, M0[c], M1[c]);
          }
        }
      }
#endif
      if (g < glast) store(cur ^ 1);
      __syncthreads();
    }
    // epilogue: X[cls][head][c] = code part + min part; H32 (class 1) / H_D (class 2); sum
    if constexpr (FAST && G <= 4) {
      const int c0 = t4 >> 1, h0 = (2 * t4) & 3;
#pragma unroll
      for (int k = 0; k < CPT; k++) {
        const int j = warp + 4 * k;
#pragma unroll
        for (int m = 0; m < 2; m++) {
          int ch = 32 * j + 16 * m + gid;
          S.qx[c0][h0][ch] = Cc[k][0][m][0] + Cm[k][0][0];
          S.qx[c0][h0 + 1][ch] = Cc[k][0][m][1] + Cm[k][0][1];
          S.qx[c0][h0][ch + 8] = Cc[k][0][m][2] + Cm[k][0][2];
          S.qx[c0][h0 + 1][ch + 8] = Cc[k][0][m][3] + Cm[k][0][3];
          if (t4 < 2) {
            S.qx[2][h0][ch] = Cc[k][1][m][0] + Cm[k][1][0];
            S.qx[2][h0 + 1][ch] = Cc[k][1][m][1] + Cm[k][1][1];
            S.qx[2][h0][ch + 8] = Cc[k][1][m][2] + Cm[k][1][2];
            S.qx[2][h0 + 1][ch + 8] = Cc[k][1][m][3] + Cm[k][1][3];
          }
        }
      }
    } else
#pragma unroll
    for (int k = 0; k < CPT; k++) {
      const int j = warp + 4 * k;
#pragma unroll
      for (int c = 0; c < 3; c++)
#pragma unroll
        for (int m = 0; m < 2; m++) {
          int ch = 32 * j + 16 * m + gid;
          S.qx[c][2 * t4][ch] = Cc[k][c][m][0] + Cm[k][c][0];
          S.qx[c][2 * t4 + 1][ch] = Cc[k][c][m][1] + Cm[k][c][1];
          S.qx[c][2 * t4][ch + 8] = Cc[k][c][m][2] + Cm[k][c][2];
          S.qx[c][2 * t4 + 1][ch + 8] = Cc[k][c][m][3] + Cm[k][c][3];
        }
    }
    __syncthreads();
    if constexpr (FAST) { smem_fwht_f<D>(S.qx[1], G, 32); smem_fwht_f<D>(S.qx[2], G, D); }
    else { smem_fwht<D>(S.qx[1], G, 32); smem_fwht<D>(S.qx[2], G, D); }
    for (int e = tid; e < G * D; e += 128) {
      int gg = e / D, c = e % D;
      op[gg * D + c] = fadd(fadd(S.qx[0][gg][c], S.qx[1][gg][c]), S.qx[2][gg][c]);
    }
  }
  if constexpr (FAST) {
    if (!tail) {
#pragma unroll
      for (int r = 0; r < R2; r++)
#pragma unroll
        for (int m = 16; m >= 1; m >>= 1) l_run[r] += __shfl_xor_sync(0xffffffffu, l_run[r], m);
    }
  }
#pragma unroll
  for (int r = 0; r < R2; r++) {
    int g = warp + 4 * r;
    if (g < G && lane == 0) { mlp[g * 2] = m_run[r]; mlp[g * 2 + 1] = l_run[r]; }
  }
}

// merge: grid (B, Hkv G), D threads... (128 threads, CPT channels each)
template <int D>
__global__ void __launch_bounds__(128) k_merge(const float* __restrict__ opart, const float* __restrict__ ml, int Hkv, int G,
                                               int NS, __nv_bfloat16* __restrict__ out, long long out_rs) {
  constexpr int CPT = D / 128;
  int b = blockIdx.x, hq = blockIdx.y;
  int h = hq / G, g = hq % G;
  long long p0 = (long long)(b * Hkv + h) * (NS + 1);
  float M = -INFINITY;
  for (int s = 0; s <= NS; s++) M = fmaxf(M, ml[((p0 + s) * G + g) * 2]);
  float o[CPT], L = 0.f;
  for (int k = 0; k < CPT; k++) o[k] = 0.f;
  if (M != -INFINITY) {
    for (int s = 0; s <= NS; s++) {
      float m = ml[((p0 + s) * G + g) * 2];
      if (m == -INFINITY) continue;
      float f = exp2f(m - M);
      L += f * ml[((p0 + s) * G + g) * 2 + 1];
      const float* op = opart + ((p0 + s) * G + g) * D;
      for (int k = 0; k < CPT; k++) o[k] = fmaf(f, op[threadIdx.x + 128 * k], o[k]);
    }
  }
  float inv = L > 0.f ? 1.f / L : 0.f;
  for (int k = 0; k < CPT; k++)
    out[(long long)b * out_rs + (long long)hq * D + threadIdx.x + 128 * k] = __float2bfloat16(o[k] * inv);
}

// ================================================================================================ C ABI
extern "C" int kv_v_write(const void* kv, long long kv_rs, int n, int Hkv, int D, const void* slots, const void* co,
                          const void* M, void* pool, int ident, const void* vfix, int f8, void* stream) {
  if (n <= 0) return 0;
  cudaStream_t st = (cudaStream_t)stream;
  dim3 grid((unsigned)(((long long)n * Hkv + 3) / 4));
  if (D == 128)
    k_v_write<128><<<grid, 128, 0, st>>>((const __nv_bfloat16*)kv, kv_rs, n, Hkv, (const long long*)slots,
                                         (const float*)co, (const float*)M, (uint8_t*)pool, ident, (const int*)vfix, f8);
  else if (D == 256)
    k_v_write<256><<<grid, 128, 0, st>>>((const __nv_bfloat16*)kv, kv_rs, n, Hkv, (const long long*)slots,
                                         (const float*)co, (const float*)M, (uint8_t*)pool, ident, (const int*)vfix, f8);
  else { snprintf(g_err, sizeof(g_err), "kv_v_write: D %d", D); return 2; }
  return check_launch("kv_v_write");
}

extern "C" int kv_k_encode(const void* src, long long src_rs, const void* src2, long long src2_rs, const void* sidx, const void* dslot, int N, int Hkv, int D,
                           const void* qs, const void* lz, int NQ, float qsc, void* pool, int ident, const void* kfix,
                           int f8, void* stream) {
  if (N <= 0) return 0;
  cudaStream_t st = (cudaStream_t)stream;
  dim3 grid(N, Hkv);
  if (D == 128) {
    size_t sm = kenc_smem<128>(NQ);
    static size_t set128 = 0;
    if (sm > set128) { cudaFuncSetAttribute(k_k_encode<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); set128 = sm; }
    k_k_encode<128><<<grid, 256, sm, st>>>((const __nv_bfloat16*)src, src_rs, (const __nv_bfloat16*)src2, src2_rs,
                                           (const long long*)sidx,
                                           (const long long*)dslot, Hkv, (const float*)qs, (const float*)lz, NQ, qsc,
                                           (uint8_t*)pool, ident, (const int*)kfix, f8);
  } else if (D == 256) {
    size_t sm = kenc_smem<256>(NQ);
    static size_t set256 = 0;
    if (sm > set256) { cudaFuncSetAttribute(k_k_encode<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); set256 = sm; }
    k_k_encode<256><<<grid, 256, sm, st>>>((const __nv_bfloat16*)src, src_rs, (const __nv_bfloat16*)src2, src2_rs,
                                           (const long long*)sidx,
                                           (const long long*)dslot, Hkv, (const float*)qs, (const float*)lz, NQ, qsc,
                                           (uint8_t*)pool, ident, (const int*)kfix, f8);
  } else { snprintf(g_err, sizeof(g_err), "kv_k_encode: D %d", D); return 2; }
  return check_launch("kv_k_encode");
}

// shared-memory carveout: the decode kernels are register / smem bound at a few CTAs per SM; max carveout lets
// more of them stay resident (no effect on results)
template <typename KF>
static void carve_once(KF k) {
  static bool done = false;
  if (!done) { cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared); done = true; }
}

extern "C" int kv_attn(const void* q, long long q_rs, const void* pool, const void* bt, long long bt_rs, const void* cached,
                       int kv_start, int B, int Hkv, int G, int D, int NS, int chunk, float qscale, const void* ring,
                       const void* tail_len, const void* tail_idx, int T, void* opart, void* ml, void* out,
                       long long out_rs, int f8, void* stream) {
  if (B <= 0) return 0;
  // f8: bit 0 = kivi2p8 pages (FP8 scale / min), bit 1 = the kivi2p8f fast decode variant of k_attn3 (same math)
  const bool fast = (f8 & 2) != 0;
  f8 &= 1;
  if (G > 8 || chunk % GRP) { snprintf(g_err, sizeof(g_err), "kv_attn: G %d chunk %d", G, chunk); return 2; }
  cudaStream_t st = (cudaStream_t)stream;
  dim3 grid(B, Hkv, NS + 1), mg(B, Hkv * G);
  static int ver = -1;
  if (ver < 0) { const char* e = getenv("FLAGROT_KIVI_ATTN"); ver = (e && e[0] == '1') ? 1 : (e && e[0] == '2') ? 2 : 3; }
  bool done = false;
#define KA2(DD, GG) if (D == DD && G == GG) { k_attn2<DD, GG><<<grid, 128, 0, st>>>((const __nv_bfloat16*)q, q_rs, \
      (const uint8_t*)pool, (const int*)bt, bt_rs, (const int*)cached, kv_start, Hkv, NS, chunk, qscale, \
      (const __nv_bfloat16*)ring, (const int*)tail_len, (const int*)tail_idx, T, (float*)opart, (float*)ml); done = true; }
  if (ver == 2 && !f8) {
    KA2(128, 1) KA2(128, 2) KA2(128, 4) KA2(128, 5) KA2(128, 6) KA2(128, 8)
    KA2(256, 1) KA2(256, 2) KA2(256, 4) KA2(256, 6)
  }
#undef KA2
#define KA3(DD, GG) if (D == DD && G == GG) { if (fast) { carve_once(k_attn3<DD, GG, true, true>); carve_once(k_attn3<DD, GG, false, true>); \
      if (f8) k_attn3<DD, GG, true, true><<<grid, 128, 0, st>>>((const __nv_bfloat16*)q, q_rs, \
      (const uint8_t*)pool, (const int*)bt, bt_rs, (const int*)cached, kv_start, Hkv, NS, chunk, qscale, \
      (const __nv_bfloat16*)ring, (const int*)tail_len, (const int*)tail_idx, T, (float*)opart, (float*)ml); else k_attn3<DD, GG, false, true><<<grid, 128, 0, st>>>((const __nv_bfloat16*)q, q_rs, \
      (const uint8_t*)pool, (const int*)bt, bt_rs, (const int*)cached, kv_start, Hkv, NS, chunk, qscale, \
      (const __nv_bfloat16*)ring, (const int*)tail_len, (const int*)tail_idx, T, (float*)opart, (float*)ml); done = true; } else \
      if (f8) k_attn3<DD, GG, true><<<grid, 128, 0, st>>>((const __nv_bfloat16*)q, q_rs, \
      (const uint8_t*)pool, (const int*)bt, bt_rs, (const int*)cached, kv_start, Hkv, NS, chunk, qscale, \
      (const __nv_bfloat16*)ring, (const int*)tail_len, (const int*)tail_idx, T, (float*)opart, (float*)ml); else k_attn3<DD, GG><<<grid, 128, 0, st>>>((const __nv_bfloat16*)q, q_rs, \
      (const uint8_t*)pool, (const int*)bt, bt_rs, (const int*)cached, kv_start, Hkv, NS, chunk, qscale, \
      (const __nv_bfloat16*)ring, (const int*)tail_len, (const int*)tail_idx, T, (float*)opart, (float*)ml); done = true; }
  if (ver == 3 || f8 || fast) {
    KA3(128, 1) KA3(128, 2) KA3(128, 4) KA3(128, 5) KA3(128, 6) KA3(128, 8)
    KA3(256, 1) KA3(256, 2) KA3(256, 4) KA3(256, 6)
  }
#undef KA3
  if ((f8 || fast) && !done) { snprintf(g_err, sizeof(g_err), "kv_attn: kivi2p8 / fast has no kernel for D %d G %d", D, G); return 2; }
  if (done) {
    if (D == 128) k_merge<128><<<mg, 128, 0, st>>>((const float*)opart, (const float*)ml, Hkv, G, NS, (__nv_bfloat16*)out, out_rs);
    else k_merge<256><<<mg, 128, 0, st>>>((const float*)opart, (const float*)ml, Hkv, G, NS, (__nv_bfloat16*)out, out_rs);
  } else if (D == 128) {
    k_attn<128><<<grid, 128, 0, st>>>((const __nv_bfloat16*)q, q_rs, (const uint8_t*)pool, (const int*)bt, bt_rs,
                                      (const int*)cached, kv_start, Hkv, G, NS, chunk, qscale,
                                      (const __nv_bfloat16*)ring, (const int*)tail_len, (const int*)tail_idx, T,
                                      (float*)opart, (float*)ml);
    k_merge<128><<<mg, 128, 0, st>>>((const float*)opart, (const float*)ml, Hkv, G, NS, (__nv_bfloat16*)out, out_rs);
  } else if (D == 256) {
    k_attn<256><<<grid, 128, 0, st>>>((const __nv_bfloat16*)q, q_rs, (const uint8_t*)pool, (const int*)bt, bt_rs,
                                      (const int*)cached, kv_start, Hkv, G, NS, chunk, qscale,
                                      (const __nv_bfloat16*)ring, (const int*)tail_len, (const int*)tail_idx, T,
                                      (float*)opart, (float*)ml);
    k_merge<256><<<mg, 128, 0, st>>>((const float*)opart, (const float*)ml, Hkv, G, NS, (__nv_bfloat16*)out, out_rs);
  } else { snprintf(g_err, sizeof(g_err), "kv_attn: D %d", D); return 2; }
  return check_launch("kv_attn");
}

// ================================================================================================ page dequantization
// prefill "fa" path: positions [32 g0, 32 g1) of one request (block-table row bt) -> K, V [(g1 - g0) 32, Hkv, D] in the
// basis the writers saw (inverse H32 / H_D applied: original basis, or the primed basis of KIVI-C), bf16 or fp32.
// grid (g1 - g0, Hkv), 128 threads.
template <int D>
__device__ void smem_fwht_masked(float (*x)[D + 1], int R, int rb, const uint32_t* en /* [R] block bitmask */) {
  int tid = threadIdx.x;
  for (int h = 1; h < rb; h <<= 1) {
    for (int e = tid; e < R * D / 2; e += 128) {
      int r = e / (D / 2), k = e % (D / 2);
      int lo = (k / h) * 2 * h + (k % h), hi = lo + h;
      if (!((en[r] >> (lo / rb)) & 1)) continue;
      float a = x[r][lo], b = x[r][hi];
      x[r][lo] = fadd(a, b);
      x[r][hi] = fsub(a, b);
    }
    __syncthreads();
  }
  float c = rsqrt_rb(rb);
  for (int e = tid; e < R * D; e += 128) {
    int r = e / D, k = e % D;
    if ((en[r] >> (k / rb)) & 1) x[r][k] = fmul(x[r][k], c);
  }
  __syncthreads();
}

template <int D, typename OT, bool F8 = false>
__global__ void __launch_bounds__(128) k_dequant(const uint8_t* __restrict__ pool, const int* __restrict__ bt, int g0,
                                                 int Hkv, int nvalid, OT* __restrict__ ok, OT* __restrict__ ov) {
  constexpr int NV = D / 32;
  __shared__ float x[32][D + 1];
  __shared__ uint32_t en32[32], enD[32];
  const int gi = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  const int g = g0 + gi;
  const long long CB = F8 ? chunk_bytes8(D) : chunk_bytes(D);
  const uint8_t* chA = pool + ((long long)bt[2 * g] * Hkv + h) * CB;
  const uint8_t* chB = pool + ((long long)bt[2 * g + 1] * Hkv + h) * CB;
  // keys: group meta (flags in the scale sign bits of channels 0..NV)
  uint32_t lm = 0, fF = 0;
  if (F8) {
    for (int j = 0; j < NV; j++) lm |= ((uint32_t)(chA[off_km8(D) + j * 2 * NV] >> 7) & 1u) << j;
    const uint8_t* cf = NV < 16 ? chA : chB;
    fF = (cf[off_km8(D) + (NV % 16) * 2 * NV] >> 7) & 1u;
  } else {
    const uint32_t* m0 = reinterpret_cast<const uint32_t*>(chA + off_km(D));          // cell 0
    const uint32_t* m1 = reinterpret_cast<const uint32_t*>(chA + off_km(D) + D / 8);  // cell 1
    for (int j = 0; j < NV; j++) lm |= ((m0[j] >> 15) & 1u) << j;
    fF = (m1[0] >> 15) & 1u;
  }
  for (int e = tid; e < 32 * D; e += 128) {
    int t = e / D, c = e % D;
    const uint8_t* ch = t < 16 ? chA : chB;
    int code = (reinterpret_cast<const uint32_t*>(ch + (t % 16) * (D / 4))[c / 16] >> kbit(c % 16)) & 3;
    int cell = c / NV;
    const uint8_t* cm = cell < 16 ? chA : chB;
    if (F8) {
      const uint8_t* kb = cm + off_km8(D) + (cell % 16) * 2 * NV + 2 * (c % NV);
      x[t][c] = fadd(fmul((float)code, e4m3_val(kb[0] & 0x7F)), e4m3_val(kb[1]));
    } else {
      uint32_t w = reinterpret_cast<const uint32_t*>(cm + off_km(D) + (cell % 16) * (D / 8))[c % NV];
      __half2 hv = *reinterpret_cast<__half2*>(&w);
      x[t][c] = fadd(fmul((float)code, fabsf(__low2float(hv))), __high2float(hv));
    }
  }
  if (tid < 32) { en32[tid] = fF ? 0u : lm; enD[tid] = fF ? 1u : 0u; }
  __syncthreads();
  if (lm && !fF) smem_fwht_masked<D>(x, 32, 32, en32);
  if (fF) smem_fwht_masked<D>(x, 32, D, enD);
  for (int e = tid; e < 32 * D; e += 128) {
    int t = e / D, c = e % D;
    long long o = ((long long)gi * 32 + t);
    if (o < nvalid) ok[(o * Hkv + h) * D + c] = (OT)x[t][c];
  }
  __syncthreads();
  // values: per token flags
  if (tid < 32) {
    const uint8_t* ch = tid < 16 ? chA : chB;
    int cell = tid % 16;
    uint32_t l = 0;
    bool f;
    if (F8) {
      for (int j = 0; j < NV; j++) l |= ((uint32_t)(ch[off_vm8(D) + cell * 2 * NV + 2 * j] >> 7) & 1u) << j;
      f = (ch[off_km8(D) + cell * 2 * NV + 2 * (NV - 1)] >> 7) & 1;
    } else {
      const uint32_t* vm = reinterpret_cast<const uint32_t*>(ch + off_vm(D) + cell * (D / 8));
      for (int j = 0; j < NV; j++) l |= ((vm[j] >> 15) & 1u) << j;
      f = ch[off_vf(D) + cell] != 0;
    }
    en32[tid] = f ? 0u : l;
    enD[tid] = f ? 1u : 0u;
  }
  for (int e = tid; e < 32 * D; e += 128) {
    int t = e / D, c = e % D;
    const uint8_t* ch = t < 16 ? chA : chB;
    int cell = t % 16;
    int code = (ch[(F8 ? off_vc8(D) : off_vc(D)) + cell * (D / 4) + c / 4] >> (2 * (c % 4))) & 3;
    if (F8) {
      const uint8_t* vb = ch + off_vm8(D) + cell * 2 * NV + 2 * (c / 32);
      x[t][c] = fadd(fmul((float)code, e4m3_val(vb[0] & 0x7F)), e4m3_val(vb[1]));
    } else {
      uint32_t w = reinterpret_cast<const uint32_t*>(ch + off_vm(D) + cell * (D / 8))[c / 32];
      __half2 hv = *reinterpret_cast<__half2*>(&w);
      x[t][c] = fadd(fmul((float)code, fabsf(__low2float(hv))), __high2float(hv));
    }
  }
  __syncthreads();
  smem_fwht_masked<D>(x, 32, 32, en32);
  smem_fwht_masked<D>(x, 32, D, enD);
  for (int e = tid; e < 32 * D; e += 128) {
    int t = e / D, c = e % D;
    long long o = ((long long)gi * 32 + t);
    if (o < nvalid) ov[(o * Hkv + h) * D + c] = (OT)x[t][c];
  }
}

// kivi2p8ff: the same dequantization (bit-identical values) with a warp per token: lane = E = D / 32 consecutive
// channels (one key cell's meta), the inverse H32 / H_D as in-register butterflies (reg_fwht: the same pairs, order
// and scaling as smem_fwht_masked), vectorized loads / stores.  grid (g1 - g0, Hkv), 128 threads.
// kivi2p8ff: E consecutive outputs of one lane (vector store for bf16)
template <int E, typename OT>
__device__ __forceinline__ void store_row(OT* dst, const float (&x)[E]) {
  if constexpr (std::is_same<OT, __nv_bfloat16>::value && E % 4 == 0) {
#pragma unroll
    for (int i = 0; i < E; i += 4) {
      __nv_bfloat162 a = __floats2bfloat162_rn(x[i], x[i + 1]), b = __floats2bfloat162_rn(x[i + 2], x[i + 3]);
      uint2 u;
      u.x = *reinterpret_cast<uint32_t*>(&a);
      u.y = *reinterpret_cast<uint32_t*>(&b);
      *reinterpret_cast<uint2*>(dst + i) = u;
    }
  } else {
#pragma unroll
    for (int i = 0; i < E; i++) dst[i] = (OT)x[i];
  }
}

template <int D, typename OT, bool F8 = false>
__global__ void __launch_bounds__(128) k_dequant2(const uint8_t* __restrict__ pool, const int* __restrict__ bt, int g0,
                                                  int Hkv, int nvalid, OT* __restrict__ ok, OT* __restrict__ ov) {
  constexpr int NV = D / 32, E = D / 32;
  const int gi = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = g0 + gi;
  const long long CB = F8 ? chunk_bytes8(D) : chunk_bytes(D);
  const uint8_t* chA = pool + ((long long)bt[2 * g] * Hkv + h) * CB;
  const uint8_t* chB = pool + ((long long)bt[2 * g + 1] * Hkv + h) * CB;
  // keys: group flags, this lane's channels [lane E, lane E + E) = cell `lane`, slots 0 .. NV - 1
  uint32_t lm = 0, fF = 0;
  if (F8) {
    for (int j = 0; j < NV; j++) lm |= ((uint32_t)(chA[off_km8(D) + j * 2 * NV] >> 7) & 1u) << j;
    const uint8_t* cf = NV < 16 ? chA : chB;
    fF = (cf[off_km8(D) + (NV % 16) * 2 * NV] >> 7) & 1u;
  } else {
    const uint32_t* m0 = reinterpret_cast<const uint32_t*>(chA + off_km(D));
    const uint32_t* m1 = reinterpret_cast<const uint32_t*>(chA + off_km(D) + D / 8);
    for (int j = 0; j < NV; j++) lm |= ((m0[j] >> 15) & 1u) << j;
    fF = (m1[0] >> 15) & 1u;
  }
  const uint8_t* cm = lane < 16 ? chA : chB;
  float ks[E], kmn[E];
#pragma unroll
  for (int i = 0; i < E; i++) {
    if (F8) {
      const uint8_t* kb = cm + off_km8(D) + (lane % 16) * 2 * NV + 2 * i;
      ks[i] = e4m3_val(kb[0] & 0x7F);
      kmn[i] = e4m3_val(kb[1]);
    } else {
      uint32_t w = reinterpret_cast<const uint32_t*>(cm + off_km(D) + (lane % 16) * (D / 8))[i];
      __half2 hv = *reinterpret_cast<__half2*>(&w);
      ks[i] = fabsf(__low2float(hv));
      kmn[i] = __high2float(hv);
    }
  }
  const int jl = (lane * E) / 32;                   // this lane's sub-vector
  const bool kH = !fF && ((lm >> jl) & 1u);
  for (int t = warp; t < 32; t += 4) {
    const uint8_t* ch = t < 16 ? chA : chB;
    float x[E];
#pragma unroll
    for (int i = 0; i < E; i++) {
      const int c = lane * E + i;
      int code = (reinterpret_cast<const uint32_t*>(ch + (t % 16) * (D / 4))[c / 16] >> kbit(c % 16)) & 3;
      x[i] = fadd(fmul((float)code, ks[i]), kmn[i]);
    }
    if (fF) {
      reg_fwht<E>(x, D, lane);
    } else if (lm) {
      float y[E];
#pragma unroll
      for (int i = 0; i < E; i++) y[i] = x[i];
      reg_fwht<E>(y, 32, lane);
      if (kH) {
#pragma unroll
        for (int i = 0; i < E; i++) x[i] = y[i];
      }
    }
    long long o = (long long)gi * 32 + t;
    if (o < nvalid) {
      store_row<E, OT>(ok + (o * Hkv + h) * D + lane * E, x);
    }
  }
  // values: per token flags; lane = sub-vector jl's channels
  for (int t = warp; t < 32; t += 4) {
    const uint8_t* ch = t < 16 ? chA : chB;
    const int cell = t % 16;
    uint32_t l = 0;
    bool f;
    if (F8) {
      for (int j = 0; j < NV; j++) l |= ((uint32_t)(ch[off_vm8(D) + cell * 2 * NV + 2 * j] >> 7) & 1u) << j;
      f = (ch[off_km8(D) + cell * 2 * NV + 2 * (NV - 1)] >> 7) & 1;
    } else {
      const uint32_t* vm = reinterpret_cast<const uint32_t*>(ch + off_vm(D) + cell * (D / 8));
      for (int j = 0; j < NV; j++) l |= ((vm[j] >> 15) & 1u) << j;
      f = ch[off_vf(D) + cell] != 0;
    }
    float vs, vmn;
    if (F8) {
      const uint8_t* vb = ch + off_vm8(D) + cell * 2 * NV + 2 * jl;
      vs = e4m3_val(vb[0] & 0x7F);
      vmn = e4m3_val(vb[1]);
    } else {
      uint32_t w = reinterpret_cast<const uint32_t*>(ch + off_vm(D) + cell * (D / 8))[jl];
      __half2 hv = *reinterpret_cast<__half2*>(&w);
      vs = fabsf(__low2float(hv));
      vmn = __high2float(hv);
    }
    float x[E];
#pragma unroll
    for (int i = 0; i < E; i++) {
      const int c = lane * E + i;
      int code = (ch[(F8 ? off_vc8(D) : off_vc(D)) + cell * (D / 4) + c / 4] >> (2 * (c % 4))) & 3;
      x[i] = fadd(fmul((float)code, vs), vmn);
    }
    if (f) {
      reg_fwht<E>(x, D, lane);
    } else if (l) {
      float y[E];
#pragma unroll
      for (int i = 0; i < E; i++) y[i] = x[i];
      reg_fwht<E>(y, 32, lane);
      if ((l >> jl) & 1u) {
#pragma unroll
        for (int i = 0; i < E; i++) x[i] = y[i];
      }
    }
    long long o = (long long)gi * 32 + t;
    if (o < nvalid) {
      store_row<E, OT>(ov + (o * Hkv + h) * D + lane * E, x);
    }
  }
}

// kivi2p8ff: the LSE merge of the "fa" prefill's two partial attentions in one kernel, bit-identical to the PyTorch
// expression of backend._merge: l -> (+inf -> -inf); m = max(la, lb); wa = exp(la - m), wb = exp(lb - m);
// o = bf16((oa * wa + ob * wb) / (wa + wb)) with every op rounded separately (no FMA contraction), expf.
__global__ void k_lse_merge(const __nv_bfloat16* __restrict__ oa, const float* __restrict__ la, long long la_hs,
                            const __nv_bfloat16* __restrict__ ob, const float* __restrict__ lb, long long lb_hs,
                            __nv_bfloat16* __restrict__ out, long long out_rs, int na, int H, int D) {
  long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long tot = (long long)na * H * D;
  if (e >= tot) return;
  int c = (int)(e % D);
  long long rh = e / D;
  int hh = (int)(rh % H);
  long long r = rh / H;
  float a = la[hh * la_hs + r], b = lb[hh * lb_hs + r];
  if (a == INFINITY) a = -INFINITY;
  if (b == INFINITY) b = -INFINITY;
  float m = (a != a || b != b) ? NAN : fmaxf(a, b);
  float wa = expf(__fsub_rn(a, m)), wb = expf(__fsub_rn(b, m));
  float va = __bfloat162float(oa[(r * H + hh) * D + c]), vb = __bfloat162float(ob[(r * H + hh) * D + c]);
  float num = __fadd_rn(__fmul_rn(va, wa), __fmul_rn(vb, wb));
  out[r * out_rs + (long long)hh * D + c] = __float2bfloat16_rn(__fdiv_rn(num, __fadd_rn(wa, wb)));
}

extern "C" int kv_lse_merge(const void* oa, const void* la, long long la_hs, const void* ob, const void* lb,
                            long long lb_hs, void* out, long long out_rs, int na, int H, int D, void* stream) {
  if (na <= 0) return 0;
  long long tot = (long long)na * H * D;
  k_lse_merge<<<(unsigned)((tot + 255) / 256), 256, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16*)oa, (const float*)la, la_hs, (const __nv_bfloat16*)ob, (const float*)lb, lb_hs,
      (__nv_bfloat16*)out, out_rs, na, H, D);
  return check_launch("kv_lse_merge");
}

extern "C" int kv_dequant(const void* pool, const void* bt, int g0, int g1, int nvalid, int Hkv, int D, int fp32,
                          void* ok, void* ov, int f8, void* stream) {
  if (g1 <= g0) return 0;
  cudaStream_t st = (cudaStream_t)stream;
  dim3 grid(g1 - g0, Hkv);
  // f8: bit 0 = kivi2p8 pages, bit 1 = the kivi2p8ff dequant kernel (k_dequant2, bit-identical values)
  const bool v2 = (f8 & 2) != 0;
  f8 &= 1;
#define KDQ(DD, T) do { if (v2) { if (f8) k_dequant2<DD, T, true><<<grid, 128, 0, st>>>((const uint8_t*)pool, (const int*)bt, g0, Hkv, nvalid, (T*)ok, (T*)ov); \
                        else k_dequant2<DD, T, false><<<grid, 128, 0, st>>>((const uint8_t*)pool, (const int*)bt, g0, Hkv, nvalid, (T*)ok, (T*)ov); } \
                        else if (f8) k_dequant<DD, T, true><<<grid, 128, 0, st>>>((const uint8_t*)pool, (const int*)bt, g0, Hkv, nvalid, (T*)ok, (T*)ov); \
                        else k_dequant<DD, T, false><<<grid, 128, 0, st>>>((const uint8_t*)pool, (const int*)bt, g0, Hkv, nvalid, (T*)ok, (T*)ov); } while (0)
  if (D == 128) { if (fp32) KDQ(128, float); else KDQ(128, __nv_bfloat16); }
  else if (D == 256) { if (fp32) KDQ(256, float); else KDQ(256, __nv_bfloat16); }
  else { snprintf(g_err, sizeof(g_err), "kv_dequant: D %d", D); return 2; }
#undef KDQ
  return check_launch("kv_dequant");
}
