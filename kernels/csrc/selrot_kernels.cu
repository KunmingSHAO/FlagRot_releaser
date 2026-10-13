// selective-FWHT + quantization kernels for H20 (sm_90a).
//
// Thread mapping: every thread owns one 64-element "slot" of a row (8 x 16-byte chunks).  A warp owns a 2048-element
// slab (32 slots).  The slab is brought into shared memory with coalesced 16-byte cp.async (chunk c of slot r is
// stored at r*128 + ((c ^ (r & 7)) << 4): a 128B-swizzle), after which lane L reads slot L back with bank-conflict-
// free 16-byte LDS.  The whole rb-point FWHT (rb <= 64) then runs inside one thread's registers: log2(rb) FADD per
// element, zero shuffles (the shuffle pipe on sm_90 is only 32 lanes/clk/SM and would be the bottleneck).
// The fp8/fp4 codes are staged through shared memory again so the global stores are coalesced 16-byte STG.
//
// Numerics are chosen to be bit-identical to the PyTorch reference (ref.py):
//   * butterfly stage order h = 1, 2, 4, ... (a+b, a-b), then * fp32(1/sqrt(rb)) (rb = 8, 32);
//     for rb = 16, 64 the power-of-two factor is folded into the scale (exact);
//   * scale = max(amax, 1e-12) * fp32(1/448)  (torch evaluates tensor / python-scalar as a multiply by the reciprocal);
//   * q = RN(y / scale) (correctly rounded division via reciprocal + one FMA correction), cvt.rn.satfinite to e4m3.
#include "selrot.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include <algorithm>

#define DEVI __device__ __forceinline__

namespace {

constexpr int VE = 64;          // elements per thread slot
constexpr int NWARP = 4;        // warps per CTA (flat / swiglu kernels)
#ifndef SELROT_RMS_R
#define SELROT_RMS_R(SPR) 1
#endif
#ifndef SELROT_MINB
#define SELROT_MINB 4
#endif
constexpr unsigned FULL = 0xffffffffu;

DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

DEVI void cp_async16(uint32_t saddr, const void* g, bool pred) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(saddr), "l"(g), "r"(pred ? 16 : 0)
               : "memory");
}
DEVI void cp_async16_nopred(uint32_t saddr, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(saddr), "l"(g) : "memory");
}
DEVI uint4 lds128(uint32_t a) {
  uint4 u;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(u.x), "=r"(u.y), "=r"(u.z), "=r"(u.w) : "r"(a));
  return u;
}
DEVI void sts128(uint32_t a, uint4 u) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(a), "r"(u.x), "r"(u.y), "r"(u.z), "r"(u.w)
               : "memory");
}
DEVI void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
DEVI void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

DEVI void unpack8(const uint4 u, float* f) {
  f[0] = __uint_as_float(u.x << 16); f[1] = __uint_as_float(u.x & 0xffff0000u);
  f[2] = __uint_as_float(u.y << 16); f[3] = __uint_as_float(u.y & 0xffff0000u);
  f[4] = __uint_as_float(u.z << 16); f[5] = __uint_as_float(u.z & 0xffff0000u);
  f[6] = __uint_as_float(u.w << 16); f[7] = __uint_as_float(u.w & 0xffff0000u);
}
DEVI float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
// torch's silu for fp32 opmath: x / (1 + expf(-x)) with IEEE division.  d = 1 + exp(-x) >= 1, so the division fast
// path (rcp.approx + one Newton step + one FMA correction == what div.rn.f32 does when FCHK passes) is correctly
// rounded as long as the quotient is normal: true for x >= -80.  Callers must redo x < -80 with silu_ref().
DEVI float silu_fast(float x) {
  const float d = 1.0f + expf(-x);
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(d));
  r = fmaf(r, fmaf(-d, r, 1.0f), r);
  const float q = __fmul_rn(x, r);
  return fmaf(-fmaf(d, q, -x), r, q);  // residual sign arranged so that silu(-0) = -0 (see div_rn)
}
DEVI float silu_ref(float x) { return __fdiv_rn(x, 1.0f + expf(-x)); }

// HF-style "bf16(a) * b" on pairs: pack two fp32 to bf16x2 (one F2FP, RNE), multiply by a raw bf16x2 word with one
// HMUL2.BF16 (exact product, single RNE rounding == bf16(float(a) * float(b)) as torch computes it), unpack to fp32.
DEVI void bf16_round_mul2(float a0, float a1, uint32_t bw, float& o0, float& o1) {
  const __nv_bfloat162 a = __floats2bfloat162_rn(a0, a1);
  __nv_bfloat162 b;
  *reinterpret_cast<uint32_t*>(&b) = bw;
  const __nv_bfloat162 y = __hmul2(a, b);
  const uint32_t yw = *reinterpret_cast<const uint32_t*>(&y);
  o0 = __uint_as_float(yw << 16);
  o1 = __uint_as_float(yw & 0xffff0000u);
}

// correctly rounded y / d given inv = RN(1/d) (d > 0).  The residual is formed as n = q*d - y and applied as
// q + (-n)*inv so that y = -0 gives -0 (fma(+0*inv, ...) would turn -0 into +0), matching torch's IEEE division.
DEVI float div_rn(float y, float d, float inv) {
#ifdef SELROT_FAST_DIV
  return __fmul_rn(y, inv);  // optional: y * RN(1/d), not bit-exact vs torch's y / d (differs in rare ties)
#endif
  const float q = __fmul_rn(y, inv);
  const float n = fmaf(q, d, -y);
  return fmaf(-n, inv, q);
}

template <int RB> struct RotC;
template <> struct RotC<8>  { static constexpr bool FOLD = false; static constexpr float RS = (float)0.35355339059327373; };
template <> struct RotC<16> { static constexpr bool FOLD = true;  static constexpr float RS = 0.25f; };
template <> struct RotC<32> { static constexpr bool FOLD = false; static constexpr float RS = (float)0.17677669529663687; };
template <> struct RotC<64> { static constexpr bool FOLD = true;  static constexpr float RS = 0.125f; };

// gather 4 byte-flags (values 0/1) of a word into bits 0..3
DEVI uint32_t bytebits4(uint32_t w) { return (((w & 0x01010101u) * 0x01020408u) >> 24) & 0xFu; }

// flag bits of the NSUB = 64/RB sub-vectors of slot j (mask values must be 0/1; one vector load)
template <int RB>
DEVI uint32_t load_flags(const uint8_t* __restrict__ mask, uint32_t j) {
  if (RB == 8) {
    const uint2 w = __ldg(reinterpret_cast<const uint2*>(mask) + j);
    return bytebits4(w.x) | (bytebits4(w.y) << 4);
  } else if (RB == 16) {
    return bytebits4(__ldg(reinterpret_cast<const uint32_t*>(mask) + j));
  } else if (RB == 32) {
    return bytebits4(__ldg(reinterpret_cast<const uint16_t*>(mask) + j));
  } else {
    return __ldg(mask + j) & 1u;
  }
}

// n / d for n * d < 2^32 via one IMAD.HI (host checks the bound, else falls back to '/')
struct FastDiv {
  uint32_t d, m;
  DEVI uint32_t div(uint32_t n) const { return m ? __umulhi(n, m) : n / d; }
};
FastDiv make_fastdiv(uint32_t d, uint64_t nmax) {
  FastDiv f{d, 0u};
  if (d > 1 && nmax * d < (1ull << 32)) f.m = static_cast<uint32_t>(((1ull << 32) + d - 1) / d);
  return f;
}

// selective block FWHT on the 64 register values (sub-vector m of size RB rotated iff bit m of fb)
template <int RB>
DEVI void apply_selrot(float (&v)[VE], uint32_t fb, bool any) {
  constexpr int NSUB = VE / RB;
  if (!any) return;  // warp-uniform
#pragma unroll
  for (int m = 0; m < NSUB; ++m) {
    if ((fb >> m) & 1u) {
#pragma unroll
      for (int h = 1; h < RB; h <<= 1) {
#pragma unroll
        for (int i = 0; i < RB; ++i) {
          if ((i & h) == 0) {
            const float a = v[m * RB + i], b = v[m * RB + i + h];
            v[m * RB + i] = a + b;
            v[m * RB + i + h] = a - b;
          }
        }
      }
      if (!RotC<RB>::FOLD) {
#pragma unroll
        for (int i = 0; i < RB; ++i) v[m * RB + i] = __fmul_rn(v[m * RB + i], RotC<RB>::RS);
      }
    }
  }
}

// per-8-chunk |max| of the TRUE rotated values (fold factor applied), plus per-chunk "is folded" flags
template <int RB>
DEVI void chunk_amax(const float (&v)[VE], uint32_t fb, float (&am)[8], bool (&fold)[8]) {
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    float a = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) a = fmaxf(a, fabsf(v[c * 8 + e]));
    fold[c] = RotC<RB>::FOLD && ((fb >> ((c * 8) / RB)) & 1u);
    am[c] = fold[c] ? a * RotC<RB>::RS : a;
  }
}

// ---------------------------------------------------------------- FP8
template <int RB, int G>
DEVI float quant_fp8(const float (&v)[VE], uint32_t fb, uint32_t (&w)[16]) {
  float am[8]; bool fold[8];
  chunk_amax<RB>(v, fb, am, fold);
  float a = am[0];
#pragma unroll
  for (int c = 1; c < 8; ++c) a = fmaxf(a, am[c]);
  if (G == 128) a = fmaxf(a, __shfl_xor_sync(FULL, a, 1));
  const float sc = __fmul_rn(fmaxf(a, 1e-12f), 1.0f / 448.0f);
  const float inv = __frcp_rn(sc);
  const float scF = sc * (1.0f / RotC<RB>::RS), invF = inv * RotC<RB>::RS;  // exact (powers of two) when FOLD
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    const float d = fold[c] ? scF : sc, iv = fold[c] ? invF : inv;
#pragma unroll
    for (int e = 0; e < 8; e += 4) {
      float2 p0 = make_float2(div_rn(v[c * 8 + e], d, iv), div_rn(v[c * 8 + e + 1], d, iv));
      float2 p1 = make_float2(div_rn(v[c * 8 + e + 2], d, iv), div_rn(v[c * 8 + e + 3], d, iv));
      uint32_t lo = __nv_cvt_float2_to_fp8x2(p0, __NV_SATFINITE, __NV_E4M3);
      uint32_t hi = __nv_cvt_float2_to_fp8x2(p1, __NV_SATFINITE, __NV_E4M3);
      w[c * 2 + e / 4] = lo | (hi << 16);
    }
  }
  return sc;
}

// ---------------------------------------------------------------- FP4 (E2M1)
// RNE to the e2m1 grid {0, .5, 1, 1.5, 2, 3, 4, 6} with saturation; returns the 4-bit code (sign in bit 3).
// Branch-free, no XU ops: with E = the binade of |x| clamped to >= 1.0, C = 1.5 * 2^(e + 22) makes the fp32 ulp of
// a + C equal the e2m1 step of that binade (0.5 below 2, 1 in [2,4), 2 in [4,6]), so one RNE FADD performs exactly the
// e2m1 rounding (ties to the even step == even code); the code is steps + 2e, read from the bits.
DEVI uint32_t e2m1_code(float x) {
  const float a = fminf(fabsf(x), 6.0f);
  const uint32_t E = max(__float_as_uint(a) & 0x7F800000u, 0x3F800000u);
  const uint32_t sb = __float_as_uint(__fadd_rn(a, __uint_as_float(E + 0x0B400000u)));
  const uint32_t code = sb - E + (E >> 22) - (0x0B400000u + 254u);
  return code | ((__float_as_uint(x) >> 28) & 8u);
}

template <int RB>
DEVI uint32_t quant_nvfp4(const float (&v)[VE], uint32_t fb, float gs, uint32_t (&w)[8]) {
  float am[8]; bool fold[8];
  chunk_amax<RB>(v, fb, am, fold);
  uint32_t sfw = 0;
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const float a = fmaxf(am[2 * b], am[2 * b + 1]);
    const float sfv = __fmul_rn(gs, __fmul_rn(a, 1.0f / 6.0f));
    const __nv_fp8_storage_t sf = __nv_cvt_float_to_fp8(sfv, __NV_SATFINITE, __NV_E4M3);
    sfw |= static_cast<uint32_t>(sf) << (8 * b);
    const float sff = __half2float(__half(__nv_cvt_fp8_to_halfraw(sf, __NV_E4M3)));
    const float bs = __fdiv_rn(sff, gs);
    const float inv = bs != 0.f ? __frcp_rn(bs) : 0.f;
#pragma unroll
    for (int cc = 0; cc < 2; ++cc) {
      const int c = 2 * b + cc;
      const float d = fold[c] ? bs * (1.0f / RotC<RB>::RS) : bs, iv = fold[c] ? inv * RotC<RB>::RS : inv;
      uint32_t word = 0;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float y = bs != 0.f ? div_rn(v[c * 8 + e], d, iv) : 0.f;
        word |= e2m1_code(y) << (4 * e);
      }
      w[c] = word;
    }
  }
  return sfw;
}

template <int RB>
DEVI uint32_t quant_mxfp4(const float (&v)[VE], uint32_t fb, uint32_t (&w)[8]) {
  float am[8]; bool fold[8];
  chunk_amax<RB>(v, fb, am, fold);
  uint32_t sfw = 0;
#pragma unroll
  for (int b = 0; b < 2; ++b) {
    const float a = fmaxf(fmaxf(am[4 * b], am[4 * b + 1]), fmaxf(am[4 * b + 2], am[4 * b + 3]));
    int e = -127;
    if (a > 0.f) { int ex; frexpf(a, &ex); e = ex - 1 - 2; }   // floor(log2(amax)) - emax(e2m1)
    e = max(-127, min(127, e));
    sfw |= static_cast<uint32_t>(e + 127) << (8 * b);
    const float mul = ldexpf(1.0f, -e);
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) {
      const int c = 4 * b + cc;
      const float m = fold[c] ? mul * RotC<RB>::RS : mul;   // power of two times power of two: exact
      uint32_t word = 0;
#pragma unroll
      for (int e2 = 0; e2 < 8; ++e2) word |= e2m1_code(__fmul_rn(v[c * 8 + e2], m)) << (4 * e2);
      w[c] = word;
    }
  }
  return sfw;
}

// symmetric INT4 per 32 elements (blockflag 'int4:32' with the scale stored in fp16): s = fp16(amax * fp32(1/7)),
// q = clamp(RN(y / float(s)), -7, 7) (correctly rounded division, RNE), code = q + 8 (offset binary, 1..15);
// s == 0 (all-zero block) -> code 8.  Returns the two fp16 scales of the slot (block 0 in the low half).
template <int RB>
DEVI uint32_t quant_int4(const float (&v)[VE], uint32_t fb, uint32_t (&w)[8]) {
  float am[8]; bool fold[8];
  chunk_amax<RB>(v, fb, am, fold);
  uint32_t sfw = 0;
#pragma unroll
  for (int b = 0; b < 2; ++b) {
    const float a = fmaxf(fmaxf(am[4 * b], am[4 * b + 1]), fmaxf(am[4 * b + 2], am[4 * b + 3]));
    const __half sh = __float2half_rn(__fmul_rn(a, 1.0f / 7.0f));
    sfw |= static_cast<uint32_t>(__half_as_ushort(sh)) << (16 * b);
    const float sf = __half2float(sh);
    const float inv = sf != 0.f ? __frcp_rn(sf) : 0.f;
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) {
      const int c = 4 * b + cc;
      uint32_t word = 0;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float y = fold[c] ? __fmul_rn(v[c * 8 + e], RotC<RB>::RS) : v[c * 8 + e];  // exact (power of two)
        float q = sf != 0.f ? rintf(div_rn(y, sf, inv)) : 0.f;
        q = fminf(fmaxf(q, -7.f), 7.f);
        word |= static_cast<uint32_t>(static_cast<int>(q) + 8) << (4 * e);
      }
      w[c] = word;
    }
  }
  return sfw;
}

// ---------------------------------------------------------------- shared tail: quantize + staged coalesced store
// buf: shared address (128-aligned) of this warp's staging buffer (already consumed).  If `full`, the 32 slots of the
// slab are contiguous in the output starting at qbase (the store then is 2-4 plain 16-byte STG per lane); otherwise
// rowaddr(r, bytes) gives the global address of slot r (nullptr if invalid).  Also writes the scales.
// smem layout: fp8 slot rows of 64 B, chunk c at c ^ ((r >> 1) & 3); fp4 rows of 32 B, chunk c at c ^ ((r >> 2) & 1).
template <int RB, int G, int OUT, typename RowAddr>
DEVI void quant_store(const float (&v)[VE], uint32_t fb, bool valid, uint32_t t, uint32_t j, uint32_t vflat,
                      const SelrotParams& p, uint32_t buf, int lane, bool full, uint8_t* qbase, RowAddr rowaddr) {
  if (OUT == OUT_FP8_MN || OUT == OUT_FP8_ROW) {
    uint32_t w[16];
    const float sc = quant_fp8<RB, G>(v, fb, w);
    if (valid && (G == 64 || (j & 1) == 0)) {
      float* S = static_cast<float*>(p.s);
      if (OUT == OUT_FP8_MN) S[static_cast<size_t>(j * VE / G) * p.sld + t] = sc;
      else S[p.lds ? static_cast<size_t>(t) * p.lds + j * VE / G : static_cast<size_t>(vflat) * VE / G] = sc;
    }
    const uint32_t wb = buf + lane * 64 + (((lane >> 1) & 3) << 4);
#pragma unroll
    for (int c = 0; c < 4; ++c) sts128(wb ^ (c << 4), make_uint4(w[4 * c], w[4 * c + 1], w[4 * c + 2], w[4 * c + 3]));
    __syncwarp();
    const uint32_t rb_ = buf + (lane >> 2) * 64 + (((lane & 3) ^ (lane >> 3)) << 4);
    if (full) {
#pragma unroll
      for (int k = 0; k < 4; ++k) *reinterpret_cast<uint4*>(qbase + k * 512 + lane * 16) = lds128(rb_ + k * 512);
    } else {
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        uint8_t* dst = rowaddr(k * 8 + (lane >> 2), 64);
        if (dst) *reinterpret_cast<uint4*>(dst + (lane & 3) * 16) = lds128(rb_ + k * 512);
      }
    }
  } else {
    uint32_t w[8];
    uint32_t sfw;
    const size_t si = p.lds ? static_cast<size_t>(t) * p.lds + j : static_cast<size_t>(vflat);
    if (OUT == OUT_NVFP4) {
      sfw = quant_nvfp4<RB>(v, fb, __ldg(p.gscale), w);
      if (valid) static_cast<uint32_t*>(p.s)[si] = sfw;
    } else if (OUT == OUT_INT4) {
      sfw = quant_int4<RB>(v, fb, w);
      if (valid) static_cast<uint32_t*>(p.s)[si] = sfw;
    } else {
      sfw = quant_mxfp4<RB>(v, fb, w);
      if (valid) static_cast<uint16_t*>(p.s)[si] = static_cast<uint16_t>(sfw);
    }
    const uint32_t wb = buf + lane * 32 + (((lane >> 2) & 1) << 4);
#pragma unroll
    for (int c = 0; c < 2; ++c) sts128(wb ^ (c << 4), make_uint4(w[4 * c], w[4 * c + 1], w[4 * c + 2], w[4 * c + 3]));
    __syncwarp();
    const uint32_t rb_ = buf + (lane >> 1) * 32 + (((lane & 1) ^ ((lane >> 3) & 1)) << 4);
    if (full) {
#pragma unroll
      for (int k = 0; k < 2; ++k) *reinterpret_cast<uint4*>(qbase + k * 512 + lane * 16) = lds128(rb_ + k * 512);
    } else {
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        uint8_t* dst = rowaddr(k * 16 + (lane >> 1), 32);
        if (dst) *reinterpret_cast<uint4*>(dst + (lane & 1) * 16) = lds128(rb_ + k * 512);
      }
    }
  }
}

// slot `lane` of a swizzled 4 KB input buffer at shared address buf (128-aligned): chunk c at c ^ (lane & 7)
DEVI void read_slot(uint32_t buf, int lane, float (&v)[VE]) {
  const uint32_t b = buf + lane * 128 + ((lane & 7) << 4);
#pragma unroll
  for (int c = 0; c < 8; ++c) unpack8(lds128(b ^ (c << 4)), v + c * 8);
}

// ---------------------------------------------------------------- flat (contiguous x) and SwiGLU kernels
// Persistent: each warp walks slabs sl = gw, gw + stride, ...; STAGES-1 slabs are prefetched with cp.async while the
// current one is transformed and stored (STAGES x NB x 4 KB of smem per warp).
// PRO_NONE: slab sl = slots [32 sl, 32 sl + 32) of the flattened [T*K/64] slot array (x contiguous), dv = VPR.
// PRO_SWIGLU: slab sl = (row t = sl / SPR, slab s = sl % SPR) of the [T, I] output, dv = SPR.
template <int PRO> struct Stages { static constexpr int N = PRO == PRO_SWIGLU ? 2 : 3; };

template <int RB, int G, int PRO, int OUT, bool HFR>
__global__ void __launch_bounds__(NWARP * 32, SELROT_MINB) k_selrot(const SelrotParams p, const uint32_t nslabs, const FastDiv dv) {
  constexpr int NB = PRO == PRO_SWIGLU ? 2 : 1;
  constexpr int ST = Stages<PRO>::N;
  constexpr int BPE2 = (OUT == OUT_NVFP4 || OUT == OUT_MXFP4 || OUT == OUT_INT4) ? 1 : 2;  // 2 * bytes per output element
  extern __shared__ __align__(1024) uint8_t dsm[];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const uint32_t wbuf = smem_u32(dsm) + wid * (ST * NB * 4096);
  const uint32_t VPR = static_cast<uint32_t>(p.K) / VE;
  const uint32_t NV = static_cast<uint32_t>(p.T) * VPR;
  const uint32_t SPR = (VPR + 31) / 32;
  const uint32_t stride = gridDim.x * NWARP;
  uint32_t sl = blockIdx.x * NWARP + wid;
  if (sl >= nslabs) return;
  const __nv_bfloat16* X = static_cast<const __nv_bfloat16*>(p.x);
  const __nv_bfloat16* X2 = static_cast<const __nv_bfloat16*>(p.x2);
  uint8_t* Q = static_cast<uint8_t*>(p.q);
  const int cc = lane & 7;
  // smem destination of chunk k copied by this lane: k*512 + (k odd ? sb ^ 64 : sb)  (128B swizzle, see header)
  const uint32_t sb = (lane >> 3) * 128 + ((cc ^ (lane >> 3)) << 4);

  auto issue = [&](uint32_t s_, uint32_t buf) {
    if (PRO == PRO_SWIGLU) {
      const uint32_t t = dv.div(s_), s = s_ - t * SPR;
      const __nv_bfloat16* g = X + static_cast<size_t>(t) * p.ld1 + (s * 32) * VE + (lane >> 3) * VE + cc * 8;
      const __nv_bfloat16* u = X2 + static_cast<size_t>(t) * p.ld2 + (s * 32) * VE + (lane >> 3) * VE + cc * 8;
      if (s * 32 + 32 <= VPR) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const uint32_t d = buf + k * 512 + ((k & 1) ? (sb ^ 64) : sb);
          cp_async16_nopred(d, g + k * 4 * VE);
          cp_async16_nopred(d + 4096, u + k * 4 * VE);
        }
      } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const uint32_t d = buf + k * 512 + ((k & 1) ? (sb ^ 64) : sb);
          const bool ok = s * 32 + k * 4 + (lane >> 3) < VPR;
          cp_async16(d, ok ? g + k * 4 * VE : X, ok);
          cp_async16(d + 4096, ok ? u + k * 4 * VE : X2, ok);
        }
      }
    } else {
      const __nv_bfloat16* src;
      if (p.ld1) {  // strided rows (K % 2048 == 0: the slab lies in one row)
        const uint32_t ts = dv.div(s_ * 32);
        src = X + static_cast<size_t>(ts) * p.ld1 + (s_ * 32 - ts * VPR) * VE + lane * 8;
      } else {
        src = X + static_cast<size_t>(s_) * (32 * VE) + lane * 8;
      }
      if ((s_ + 1) * 32 <= NV) {
#pragma unroll
        for (int k = 0; k < 8; ++k) cp_async16_nopred(buf + k * 512 + ((k & 1) ? (sb ^ 64) : sb), src + k * 256);
      } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const bool ok = s_ * 32 + k * 4 + (lane >> 3) < NV;
          cp_async16(buf + k * 512 + ((k & 1) ? (sb ^ 64) : sb), ok ? src + k * 256 : X, ok);
        }
      }
    }
  };

#pragma unroll
  for (int i = 0; i < ST - 1; ++i) {
    if (sl + i * stride < nslabs) issue(sl + i * stride, wbuf + i * (NB * 4096));
    cp_async_commit();
  }
  for (int it = 0; sl < nslabs; sl += stride, ++it) {
    const uint32_t cur = wbuf + (it % ST) * (NB * 4096);
    {
      const uint32_t ns = sl + (ST - 1) * stride;
      if (ns < nslabs) issue(ns, wbuf + ((it + ST - 1) % ST) * (NB * 4096));
      cp_async_commit();
    }
    cp_async_wait<ST - 1>();
    __syncwarp();

    uint32_t t, j, vflat, s = 0;
    bool valid, full;
    if (PRO == PRO_SWIGLU) {
      t = dv.div(sl); s = sl - t * SPR; j = s * 32 + lane; valid = j < VPR; vflat = t * VPR + j;
      full = s * 32 + 32 <= VPR;
    } else {
      vflat = sl * 32 + lane; valid = vflat < NV; t = dv.div(vflat); j = vflat - t * VPR;
      full = (sl + 1) * 32 <= NV;
    }
    float v[VE];
    if (PRO == PRO_SWIGLU) {
      const uint32_t b = cur + lane * 128 + ((lane & 7) << 4);
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        float g8[8], sg[8];
        unpack8(lds128(b ^ (c << 4)), g8);
        const uint4 uw = lds128((b + 4096) ^ (c << 4));
#pragma unroll
        for (int e = 0; e < 8; ++e) sg[e] = silu_fast(g8[e]);  // torch silu: x / (1 + exp(-x))
        if (fminf(fminf(fminf(g8[0], g8[1]), fminf(g8[2], g8[3])), fminf(fminf(g8[4], g8[5]), fminf(g8[6], g8[7]))) <
            -80.f) {  // rare: denormal quotients -> IEEE slow path
#pragma unroll
          for (int e = 0; e < 8; ++e) sg[e] = silu_ref(g8[e]);
        }
        if (HFR) {
          bf16_round_mul2(sg[0], sg[1], uw.x, v[c * 8 + 0], v[c * 8 + 1]);
          bf16_round_mul2(sg[2], sg[3], uw.y, v[c * 8 + 2], v[c * 8 + 3]);
          bf16_round_mul2(sg[4], sg[5], uw.z, v[c * 8 + 4], v[c * 8 + 5]);
          bf16_round_mul2(sg[6], sg[7], uw.w, v[c * 8 + 6], v[c * 8 + 7]);
        } else {
          float u8[8];
          unpack8(uw, u8);
#pragma unroll
          for (int e = 0; e < 8; ++e) v[c * 8 + e] = __fmul_rn(sg[e], u8[e]);
        }
      }
    } else {
      read_slot(cur, lane, v);
    }
    const uint32_t fb = valid ? load_flags<RB>(p.mask, j) : 0u;
    const bool any = __any_sync(FULL, fb != 0u);
    apply_selrot<RB>(v, fb, any);
    __syncwarp();  // every lane finished reading its slot: buffer reusable for the output
    if (PRO == PRO_SWIGLU) {
      uint8_t* qrow = Q + static_cast<size_t>(t) * (static_cast<size_t>(p.K) * BPE2 / 2) + (s * 32) * (VE * BPE2 / 2);
      quant_store<RB, G, OUT>(v, fb, valid, t, j, vflat, p, cur, lane, full, qrow,
                              [&](int r, int bytes) -> uint8_t* { return (s * 32 + r < VPR) ? qrow + r * bytes : nullptr; });
    } else {
      uint8_t* qs;
      if (p.ldq) {
        const uint32_t ts = dv.div(sl * 32);
        qs = Q + static_cast<size_t>(ts) * p.ldq + (sl * 32 - ts * VPR) * (VE * BPE2 / 2);
      } else {
        qs = Q + static_cast<size_t>(sl) * (32 * VE * BPE2 / 2);
      }
      quant_store<RB, G, OUT>(v, fb, valid, t, j, vflat, p, cur, lane, full, qs,
                              [&](int r, int bytes) -> uint8_t* { return (sl * 32 + r < NV) ? qs + r * bytes : nullptr; });
    }
    __syncwarp();  // staging reads done before this buffer is refilled by a later prefetch
  }
  cp_async_wait<0>();
}

// ---------------------------------------------------------------- RMSNorm kernel (persistent, R rows per CTA)
// blockDim = SPR * R warps; warp (rr, s) handles slab s of row base + rr.  The RMS weight is staged once per CTA.
// The next row's slab is prefetched with cp.async while the current one is processed (double buffer).
template <int RB, int G, bool HFR, bool PERM>
__global__ void __launch_bounds__(256, 2) k_rms_selrot(const SelrotParams p, int R) {
  extern __shared__ __align__(128) uint8_t dsm[];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
  const int VPR = p.K / VE, SPR = (VPR + 31) / 32;
  const int s = wid % SPR, rr = wid / SPR;
  uint8_t* wsm = dsm;                                   // SPR * 4 KB
  uint8_t* xsm = dsm + SPR * 4096;                      // 3 stages * nw * 4 KB
  float* red = reinterpret_cast<float*>(xsm + 3 * nw * 4096);
  const __nv_bfloat16* X = static_cast<const __nv_bfloat16*>(p.x);
  const __nv_bfloat16* W = static_cast<const __nv_bfloat16*>(p.x2);

  const int cc = lane & 7;
  const uint32_t sb = (lane >> 3) * 128 + ((cc ^ (lane >> 3)) << 4);  // see k_selrot
  const bool slab_full = s * 32 + 32 <= VPR;
  // PERM: K permutation applied as a gather on load (p.perm: source 8-element chunk of every output chunk; the
  // cp.async of output chunk k simply reads its source chunk).  The warp's slab s is fixed for the whole kernel, so
  // the 8 offsets per lane are read once.  Separate instantiation: the PERM=false kernel is the original one (it sits
  // close to the 128-register limit; carrying the offsets there cost ~4 %).  (A variant that gathers whole 256-byte
  // blocks and applies the in-block order on the shared-memory read spilled registers and was slower.)
  int off[8];
  if constexpr (PERM) {
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int slot = s * 32 + k * 4 + (lane >> 3);
      off[k] = slot < VPR ? __ldg(p.perm + slot * (VE / 8) + cc) * 8 : 0;
    }
  }
  auto src_off = [&](int k) -> int { return off[k]; };
  auto issue = [&](long long t, uint8_t* bufp) {
    if (t >= p.T) return;  // rows past the end: nothing loaded, results never stored
    const uint32_t buf = smem_u32(bufp);
    const __nv_bfloat16* src = PERM ? X + t * p.ld1 : X + t * p.ld1 + (s * 32 + (lane >> 3)) * VE + cc * 8;
    if (slab_full) {
#pragma unroll
      for (int k = 0; k < 8; ++k)
        cp_async16_nopred(buf + k * 512 + ((k & 1) ? (sb ^ 64) : sb), PERM ? src + src_off(k) : src + k * 4 * VE);
    } else {
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const bool ok = s * 32 + k * 4 + (lane >> 3) < VPR;
        cp_async16(buf + k * 512 + ((k & 1) ? (sb ^ 64) : sb), ok ? (PERM ? src + src_off(k) : src + k * 4 * VE) : X, ok);
      }
    }
  };
  if (rr == 0) {
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int r = k * 4 + (lane >> 3), c = lane & 7;
      const bool ok = s * 32 + r < VPR;
      cp_async16(smem_u32(wsm + s * 4096 + r * 128 + ((c ^ (r & 7)) << 4)),
                 ok ? W + static_cast<long long>(s * 32 + r) * VE + c * 8 : W, ok);
    }
  }
  constexpr int ST = 3;
  const long long stride = static_cast<long long>(gridDim.x) * R;
  long long base = static_cast<long long>(blockIdx.x) * R;
#pragma unroll
  for (int i = 0; i < ST - 1; ++i) {
    if (base + i * stride < p.T) issue(base + i * stride + rr, xsm + (i * nw + wid) * 4096);
    cp_async_commit();  // group 0 also carries the weights
  }
  const float invK = 1.0f / static_cast<float>(p.K);
  uint8_t* Q = static_cast<uint8_t*>(p.q);
  for (int it = 0; base < p.T; base += stride, ++it) {
    uint8_t* cur = xsm + ((it % ST) * nw + wid) * 4096;
    if (base + (ST - 1) * stride < p.T) issue(base + (ST - 1) * stride + rr, xsm + (((it + ST - 1) % ST) * nw + wid) * 4096);
    cp_async_commit();
    cp_async_wait<ST - 1>();
    __syncthreads();  // current x slab (and, first time, the weights) visible
    const long long t = base + rr;
    const long long j = s * 32 + lane;
    const bool valid = t < p.T && j < VPR;
    float v[VE];
    read_slot(smem_u32(cur), lane, v);
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < VE; ++i) ss = fmaf(v[i], v[i], ss);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
    if (lane == 0) red[wid] = ss;
    __syncthreads();
    float tot = 0.f;
    for (int k = 0; k < SPR; ++k) tot += red[rr * SPR + k];
    const float rs = rsqrtf(__fadd_rn(__fmul_rn(tot, invK), p.eps));
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const uint4 ww = lds128((smem_u32(wsm) + s * 4096 + lane * 128 + ((lane & 7) << 4)) ^ (c << 4));
      float* vc = v + c * 8;
      if (HFR) {  // HF: weight * (x * rsqrt).to(bf16), product in bf16
        bf16_round_mul2(__fmul_rn(vc[0], rs), __fmul_rn(vc[1], rs), ww.x, vc[0], vc[1]);
        bf16_round_mul2(__fmul_rn(vc[2], rs), __fmul_rn(vc[3], rs), ww.y, vc[2], vc[3]);
        bf16_round_mul2(__fmul_rn(vc[4], rs), __fmul_rn(vc[5], rs), ww.z, vc[4], vc[5]);
        bf16_round_mul2(__fmul_rn(vc[6], rs), __fmul_rn(vc[7], rs), ww.w, vc[6], vc[7]);
      } else {
        float w8[8];
        unpack8(ww, w8);
#pragma unroll
        for (int e = 0; e < 8; ++e) vc[e] = __fmul_rn(__fmul_rn(vc[e], rs), w8[e]);
      }
    }
    const uint32_t fb = valid ? load_flags<RB>(p.mask, j) : 0u;
    const bool any = __any_sync(FULL, fb != 0u);
    apply_selrot<RB>(v, fb, any);
    __syncwarp();
    uint8_t* qrow = Q + t * p.K + static_cast<long long>(s * 32) * VE;
    const bool full = t < p.T && s * 32 + 32 <= VPR;
    quant_store<RB, G, OUT_FP8_MN>(v, fb, valid, static_cast<uint32_t>(t), static_cast<uint32_t>(j),
                                   static_cast<uint32_t>(t * VPR + j), p, smem_u32(cur), lane, full, qrow,
                                   [&](int r, int bytes) -> uint8_t* {
                                     return (t < p.T && s * 32 + r < VPR) ? qrow + r * bytes : nullptr;
                                   });
    __syncthreads();  // red[] reuse + all warps done with this iteration
  }
  cp_async_wait<0>();
}

// ---------------------------------------------------------------- standalone bf16 helpers (unfused baselines)
__global__ void k_rmsnorm_bf16(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                               __nv_bfloat16* __restrict__ y, int K, float eps) {
  // one CTA per row, 8 elements per thread per iteration (HF LlamaRMSNorm rounding)
  const long long row = blockIdx.x;
  const uint4* xr = reinterpret_cast<const uint4*>(x + row * K);
  const uint4* wr = reinterpret_cast<const uint4*>(w);
  uint4* yr = reinterpret_cast<uint4*>(y + row * K);
  const int nvec = K / 8;
  float ss = 0.f;
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    float f[8]; unpack8(xr[i], f);
#pragma unroll
    for (int e = 0; e < 8; ++e) ss = fmaf(f[e], f[e], ss);
  }
  __shared__ float red[32];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(FULL, t, o);
    if (threadIdx.x == 0) red[0] = t;
  }
  __syncthreads();
  const float rs = rsqrtf(red[0] * (1.0f / static_cast<float>(K)) + eps);
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    float f[8], g[8]; unpack8(xr[i], f); unpack8(__ldg(wr + i), g);
    __align__(16) __nv_bfloat16 o[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) o[e] = __float2bfloat16_rn(g[e] * bf16r(f[e] * rs));
    yr[i] = *reinterpret_cast<uint4*>(o);
  }
}

__global__ void k_silu_mul_bf16(const __nv_bfloat16* __restrict__ g, const __nv_bfloat16* __restrict__ u,
                                __nv_bfloat16* __restrict__ y, long long T, int I, long long ldg, long long ldu) {
  const long long nvr = I / 8, n = T * nvr;
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<long long>(gridDim.x) * blockDim.x) {
    const long long t = i / nvr, c = (i - t * nvr) * 8;
    float a[8], b[8];
    unpack8(*reinterpret_cast<const uint4*>(g + t * ldg + c), a);
    unpack8(*reinterpret_cast<const uint4*>(u + t * ldu + c), b);
    __align__(16) __nv_bfloat16 o[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) o[e] = __float2bfloat16_rn(bf16r(a[e] / (1.0f + expf(-a[e]))) * b[e]);
    *reinterpret_cast<uint4*>(y + t * I + c) = *reinterpret_cast<uint4*>(o);
  }
}

// plain bf16 -> fp8 e4m3 cast (no scaling): the bandwidth ceiling for the 2 B read + 1 B write pattern.
// Each thread moves UNR x 16 elements per iteration (2 x 16 B loads -> 1 x 16 B store each).
template <int UNR>
__global__ void __launch_bounds__(256) k_cast_bf16_fp8(const uint4* __restrict__ x, uint4* __restrict__ q, long long n16) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x * UNR;
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x * UNR + threadIdx.x; i < n16; i += stride) {
    uint4 a[UNR], b[UNR];
#pragma unroll
    for (int u = 0; u < UNR; ++u) {
      const long long k = i + u * blockDim.x;
      if (k < n16) { a[u] = __ldcs(x + 2 * k); b[u] = __ldcs(x + 2 * k + 1); }
    }
#pragma unroll
    for (int u = 0; u < UNR; ++u) {
      const long long k = i + u * blockDim.x;
      if (k < n16) {
        float f[16];
        unpack8(a[u], f); unpack8(b[u], f + 8);
        uint32_t w[4];
#pragma unroll
        for (int e = 0; e < 4; ++e)
          w[e] = __nv_cvt_float2_to_fp8x2(make_float2(f[4 * e], f[4 * e + 1]), __NV_SATFINITE, __NV_E4M3) |
                 (static_cast<uint32_t>(__nv_cvt_float2_to_fp8x2(make_float2(f[4 * e + 2], f[4 * e + 3]),
                                                                __NV_SATFINITE, __NV_E4M3)) << 16);
        __stcs(q + k, make_uint4(w[0], w[1], w[2], w[3]));
      }
    }
  }
}

// ---------------------------------------------------------------- dispatch
template <int RB, int G, int PRO, int OUT, bool HFR>
int launch_t(const SelrotParams& p, int num_sms, cudaStream_t st) {
  const int VPR = p.K / VE;
  if (p.K % VE) return -5;
  if (static_cast<unsigned long long>(p.T) * p.K >= (1ull << 31)) return -6;  // 32-bit slot indexing
  if (PRO == PRO_NONE && (p.ld1 || p.ldq || p.lds) && (VPR % 32)) return -7;  // strided KV mode: K % 2048 == 0
  if (PRO == PRO_RMS) {
    const int SPR = (VPR + 31) / 32;
    if (SPR > 8) return -3;
    const int R = SELROT_RMS_R(SPR), nw = SPR * R;
    const size_t smem = SPR * 4096 + 3 * nw * 4096 + nw * sizeof(float);
    const bool perm = p.perm != nullptr;
    auto kern = perm ? k_rms_selrot<RB, G, HFR, true> : k_rms_selrot<RB, G, HFR, false>;
    static bool attr[2] = {false, false};
    if (!attr[perm]) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024); attr[perm] = true; }
    static int per_sm_[2] = {-1, -1};
    int& per_sm = per_sm_[perm];
    if (per_sm < 0) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, nw * 32, smem);
    long long want = (p.T + R - 1) / R;
    long long grid = std::min<long long>(want, static_cast<long long>(std::max(per_sm, 1)) * num_sms);
    kern<<<static_cast<unsigned>(grid), nw * 32, smem, st>>>(p, R);
  } else {
    constexpr int NB = PRO == PRO_SWIGLU ? 2 : 1;
    const size_t smem = NWARP * Stages<PRO>::N * NB * 4096;
    auto kern = k_selrot<RB, G, PRO, OUT, HFR>;
    static int per_sm = -1;
    if (per_sm < 0) {
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, NWARP * 32, smem);
      per_sm = std::max(per_sm, 1);
    }
    uint32_t nslabs;
    FastDiv dv;
    if (PRO == PRO_SWIGLU) {
      const uint32_t SPR = (VPR + 31) / 32;
      nslabs = static_cast<uint32_t>(p.T) * SPR;
      dv = make_fastdiv(SPR, nslabs);
    } else {
      const uint64_t NV = static_cast<uint64_t>(p.T) * VPR;
      nslabs = static_cast<uint32_t>((NV + 31) / 32);
      dv = make_fastdiv(VPR, NV);
    }
    const long long grid = std::min<long long>((nslabs + NWARP - 1) / NWARP, static_cast<long long>(per_sm) * num_sms);
    kern<<<static_cast<unsigned>(grid), NWARP * 32, smem, st>>>(p, nslabs, dv);
  }
  return 0;
}

template <int RB, int PRO, int OUT>
int launch_g(int g, const SelrotParams& p, int n, cudaStream_t st) {
  if (PRO != PRO_NONE && !p.hf_round) {
    if (g == 64) return launch_t<RB, 64, PRO, OUT, false>(p, n, st);
    if (g == 128) return launch_t<RB, 128, PRO, OUT, false>(p, n, st);
  } else {
    if (g == 64) return launch_t<RB, 64, PRO, OUT, true>(p, n, st);
    if (g == 128) return launch_t<RB, 128, PRO, OUT, true>(p, n, st);
  }
  return -2;
}

template <int PRO, int OUT>
int launch_rb(int rb, int g, const SelrotParams& p, int n, cudaStream_t st) {
  switch (rb) {
    case 8: return launch_g<8, PRO, OUT>(g, p, n, st);
    case 16: return launch_g<16, PRO, OUT>(g, p, n, st);
    case 32: return launch_g<32, PRO, OUT>(g, p, n, st);
    case 64: return launch_g<64, PRO, OUT>(g, p, n, st);
  }
  return -1;
}

}  // namespace

int selrot_launch(int pro, int out, int rb, int g, const SelrotParams& p, int num_sms, cudaStream_t st) {
  if (p.T == 0) return 0;
  if (pro == PRO_NONE) {
    switch (out) {
      case OUT_FP8_MN: return launch_rb<PRO_NONE, OUT_FP8_MN>(rb, g, p, num_sms, st);
      case OUT_FP8_ROW: return launch_rb<PRO_NONE, OUT_FP8_ROW>(rb, g, p, num_sms, st);
      case OUT_NVFP4: return launch_rb<PRO_NONE, OUT_NVFP4>(rb, 128, p, num_sms, st);
      case OUT_MXFP4: return launch_rb<PRO_NONE, OUT_MXFP4>(rb, 128, p, num_sms, st);
      case OUT_INT4: return launch_rb<PRO_NONE, OUT_INT4>(rb, 128, p, num_sms, st);
    }
    return -4;
  }
  if (out != OUT_FP8_MN) return -4;
  if (pro == PRO_SWIGLU) return launch_rb<PRO_SWIGLU, OUT_FP8_MN>(rb, g, p, num_sms, st);
  if (pro == PRO_RMS) return launch_rb<PRO_RMS, OUT_FP8_MN>(rb, g, p, num_sms, st);
  return -4;
}

void rmsnorm_bf16_launch(const void* x, const void* w, void* y, long long T, int K, float eps, cudaStream_t st) {
  if (T == 0) return;
  k_rmsnorm_bf16<<<static_cast<unsigned>(T), 256, 0, st>>>(static_cast<const __nv_bfloat16*>(x),
                                                          static_cast<const __nv_bfloat16*>(w),
                                                          static_cast<__nv_bfloat16*>(y), K, eps);
}

void silu_mul_bf16_launch(const void* g, const void* u, void* y, long long T, int I, long long ldg, long long ldu,
                          cudaStream_t st) {
  if (T == 0) return;
  const long long n = T * (I / 8);
  const long long grid = std::min<long long>((n + 255) / 256, 78LL * 16);
  k_silu_mul_bf16<<<static_cast<unsigned>(grid), 256, 0, st>>>(static_cast<const __nv_bfloat16*>(g),
                                                               static_cast<const __nv_bfloat16*>(u),
                                                               static_cast<__nv_bfloat16*>(y), T, I, ldg, ldu);
}

void cast_bf16_fp8_launch(const void* x, void* q, long long n, int num_sms, cudaStream_t st) {
  const long long n16 = n / 16;
  if (n16 == 0) return;
  const long long grid = std::min<long long>((n16 + 256 * 4 - 1) / (256 * 4), static_cast<long long>(num_sms) * 8);
  k_cast_bf16_fp8<4><<<static_cast<unsigned>(grid), 256, 0, st>>>(static_cast<const uint4*>(x), static_cast<uint4*>(q), n16);
}
