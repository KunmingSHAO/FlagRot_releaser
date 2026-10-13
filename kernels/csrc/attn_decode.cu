// Split-K GQA decode attention over a quantized KV cache (BF16 / FP8 / NVFP4 / MXFP4 / INT4), H20 (sm_90a).
//
// Grid (kv head, split, batch), 4 warps / CTA; the 4 query heads of a kv head (GQA 4) share every K/V byte.
// Each warp streams its own 16*NT-token groups of the CTA's token range through a private ST-stage cp.async ring
// (16-byte copies of the K/V code rows, 4/8-byte copies of the scale rows; smem layouts swizzled so that every
// fragment read below is bank-conflict free), so no __syncthreads in the main loop.
//
// Tensor cores, "swap AB" on both products (tokens / d on the MMA M side, the 4 heads (+4 zero heads) on N = 8):
//   S^T[16 tok, 8 h] = K[16 tok, 128 d] . Q^T          8 x mma.m16n8k16 per 16 tokens
//   O^T[128 d, 8 h] += V^T[128 d, 16 tok] . P^T         8 x mma.m16n8k16 per 16 tokens
// P^T is built from the S^T accumulators with two movmatrix.trans (no shared memory).
// Thread (gid = lane/4, tig = lane%4) owns K rows gid, gid+8 at d in [32 tig, 32 tig + 32) and V tokens
// {2tig, 2tig+1, 2tig+8, 2tig+9} at d in [16 gid, 16 gid + 16): every fragment comes from ONE 16-B (fp4), 32-B (fp8)
// or 64-B (bf16) contiguous piece of a row.  The MMA reduction order over d is permuted to whatever order the
// dequantizer produces pairs in; Q (once per CTA) is permuted identically, the output is written back in order.
// Dequant (per element, fp16 MMA for all quantized formats):
//   fp8   : cvt.rn.f16x2.e4m3x2 (1 op / 2 el); the per-(token, head) scale is applied to S (K) and to P (V)
//   e2m1  : 4 nibbles -> 4 e4m3 bytes with 2-3 LOP/SHF per 4 el (sign -> bit 7, |code| -> bits 4..2 gives exactly
//           e2m1 * 2^-6, incl. the subnormal 0.5), cvt e4m3x2 -> f16x2, HMUL2 by (block scale * 64)
//   int4  : (nibbles & 0x000F000F) | 0x6400 6400 = 1024 + code, HSUB2 1032, HMUL2 scale
//   V pairs need two tokens per register: one PRMT per 4 bytes interleaves the two token rows.
// Online softmax in the log2 domain (q pre-scaled by softmax_scale * log2 e); split partials are merged by the last
// CTA of each (batch, kv head) (atomic ticket), which also applies the optional FlagRot back-rotation of V.
// FlagRot on q (QROT): keys were rotated per (kv head, rb sub-vector) at write time, so q is rotated with the same
// flags (fp32 butterfly, warp shuffles) in the prologue; values: either folded into W_o (default) or vrot.
#include "attn_decode.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>

#define DEVI __device__ __forceinline__

namespace {

constexpr int G = 4;    // q heads per kv head
constexpr int NW = 4;   // warps per CTA
constexpr int D = 128;  // head dim
constexpr unsigned FULL = 0xffffffffu;
constexpr float NEG_INF = -__builtin_huge_valf();

template <int F> struct Fmt;
template <> struct Fmt<AF_BF16>  { static constexpr int RB = 256, SB = 0; };
template <> struct Fmt<AF_FP8>   { static constexpr int RB = 128, SB = 4; };
template <> struct Fmt<AF_NVFP4> { static constexpr int RB = 64,  SB = 8; };
template <> struct Fmt<AF_MXFP4> { static constexpr int RB = 64,  SB = 4; };
template <> struct Fmt<AF_INT4>  { static constexpr int RB = 64,  SB = 8; };

DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
DEVI void cp16(uint32_t s, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(g) : "memory");
}
template <int N>
DEVI void cpsmall(uint32_t s, const void* g) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(s), "l"(g), "n"(N) : "memory");
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
DEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
DEVI uint4 lds128(uint32_t a) {
  uint4 u;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(u.x), "=r"(u.y), "=r"(u.z), "=r"(u.w) : "r"(a));
  return u;
}
DEVI uint2 lds64(uint32_t a) {
  uint2 u;
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];\n" : "=r"(u.x), "=r"(u.y) : "r"(a));
  return u;
}
DEVI uint32_t lds32(uint32_t a) {
  uint32_t u;
  asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(u) : "r"(a));
  return u;
}
DEVI uint32_t lds16(uint32_t a) {
  unsigned short u;
  asm volatile("ld.shared.u16 %0, [%1];\n" : "=h"(u) : "r"(a));
  return u;
}
DEVI uint32_t lds8(uint32_t a) {
  unsigned short u;
  asm volatile("ld.shared.u8 %0, [%1];\n" : "=h"(u) : "r"(a));
  return u;
}

template <bool BF>
DEVI void mma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#ifdef ATTN_NOMMA  // experiment only: replace the MMA by 2 FMAs (keeps the operands live)
  c[0] = fmaf(__uint_as_float(a0 ^ a1), __uint_as_float(b0), c[0]); c[1] = fmaf(__uint_as_float(a2 ^ a3), __uint_as_float(b1), c[1]);
  return;
#endif
  if constexpr (BF)
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  else
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
DEVI uint32_t movt(uint32_t x) {
  uint32_t y;
  asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(y) : "r"(x));
  return y;
}
DEVI uint32_t e4m3lo(uint32_t v) {  // bytes 0,1 -> f16x2 (byte 0 -> low half)
  uint32_t r;
  asm("{ .reg .b16 l, h; mov.b32 {l, h}, %1; cvt.rn.f16x2.e4m3x2 %0, l; }" : "=r"(r) : "r"(v));
  return r;
}
DEVI uint32_t e4m3hi(uint32_t v) {  // bytes 2,3 (F2FP reads the .H1 half directly: no shift)
  uint32_t r;
  asm("{ .reg .b16 l, h; mov.b32 {l, h}, %1; cvt.rn.f16x2.e4m3x2 %0, h; }" : "=r"(r) : "r"(v));
  return r;
}
DEVI __half2 u2h(uint32_t u) { return *reinterpret_cast<__half2*>(&u); }
DEVI uint32_t h2u(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
DEVI uint32_t hmul2(uint32_t a, __half2 b) { return h2u(__hmul2(u2h(a), b)); }
DEVI float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
template <bool BF>
DEVI uint32_t pack2(float a, float b) {  // a -> low half
  if constexpr (BF) {
    __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
  } else {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
  }
}
// e8m0 exponent e -> fp16 bits of 2^(e-127) * 2^7 (see fp4x8); blocks with 2^(e-120) outside the fp16 normal range
// are flushed to 0 (e < 106: |block| < 2^-19) or clamped (e > 135: |block| > 2^10)
DEVI uint32_t mx64h(uint32_t e) { return static_cast<uint32_t>(max(min(static_cast<int>(e), 135), 105) - 105) << 10; }
DEVI void fp4x8(uint32_t w, uint32_t& lo, uint32_t& hi) {
  lo = ((w << 4) & 0x80808080u) | ((w << 1) & 0x0E0E0E0Eu);
  hi = (w & 0x80808080u) | (__umulhi(w, 0x20000000u) & 0x0E0E0E0Eu);
}
DEVI uint32_t b2h_lo(uint32_t x) { return __byte_perm(x, 0u, 0x1404); }  // bytes 0,1 -> high bytes of the 2 halves
DEVI uint32_t b2h_hi(uint32_t x) { return __byte_perm(x, 0u, 0x3424); }  // bytes 2,3
DEVI uint32_t i4pair(uint32_t x, int sh, __half2 s2) {  // nibbles at bit sh (low half) and 16+sh (high half)
  const uint32_t h = ((x >> sh) & 0x000F000Fu) | 0x64006400u;   // 1024 + code
  return h2u(__hmul2(__hsub2(u2h(h), u2h(0x64086408u)), s2));  // (code - 8) * s
}

// FlagRot (Sylvester FWHT / sqrt(rb) on every flagged rb-sub-vector) of a 128-vector held by a warp, lane l holding
// elements 4l..4l+3; same stage order / operand order as ref.fwht (bit-identical fp32 arithmetic).
DEVI float rb_rsqrt(int rb) {
  switch (rb) {
    case 2: return 0.70710678118654752f;
    case 4: return 0.5f;
    case 8: return 0.35355339059327373f;
    case 16: return 0.25f;
    case 32: return 0.17677669529663687f;
    case 64: return 0.125f;
    default: return 0.08838834764831845f;  // 128
  }
}
DEVI void warp_selrot(float (&x)[4], const uint8_t* __restrict__ flags, int rb, int lane) {
  const bool f = __ldg(flags + (lane * 4) / rb) != 0;
  if (!__any_sync(FULL, f)) return;
  for (int h = 1; h < rb; h <<= 1) {
    if (h < 4) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (!(i & h)) {
          const float a = x[i], c = x[i + h];
          if (f) { x[i] = a + c; x[i + h] = a - c; }
        }
      }
    } else {
      const int lm = h >> 2;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float o = __shfl_xor_sync(FULL, x[i], lm);
        if (f) x[i] = (lane & lm) ? (o - x[i]) : (x[i] + o);
      }
    }
  }
  if (f) {
    const float rs = rb_rsqrt(rb);
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = x[i] * rs;
  }
}

// d offsets (within the thread's 32-d K range) of the two elements of K pair p (p = 0..15, k-step p/2)
template <int F>
DEVI constexpr int pd0(int p) {
  if (F == AF_BF16 || F == AF_FP8) return 2 * p;
  if (F == AF_INT4) return 8 * (p / 4) + (p % 4);
  return 8 * (p / 4) + ((p & 1) ? 4 : 0) + ((p >> 1) & 1);  // e2m1: (0,2) (4,6) (1,3) (5,7)
}
template <int F>
DEVI constexpr int pd1(int p) {
  if (F == AF_BF16 || F == AF_FP8) return 2 * p + 1;
  if (F == AF_INT4) return pd0<F>(p) + 4;
  return pd0<F>(p) + 2;
}

// smem layout (per warp, per stage): K rows [TOK][RB], V rows [TOK][RB], K scales [TOK][SB], V scales [TOK][SB]
template <int F>
DEVI int kswz(int r, int u) {  // physical 16-B unit of K unit u in row r
  if (F == AF_BF16) return u ^ (r & 1) ^ (((u >> 3) & 1) << 1);
  if (F == AF_FP8) return u ^ (r & 1);
  return u;
}
template <int F>
DEVI int vrow(int r) {  // physical row slot of V row r
  if (Fmt<F>::RB == 64) return (r & ~7) | ((r & 7) ^ (((r & 7) >> 1) & 1));
  return r;
}
template <int F>
DEVI int vswz(int r, int u) {  // physical 16-B unit of V unit u in row r
  if (F == AF_BF16) return u ^ (((r >> 1) & 1) | (((r >> 2) & 1) << 2));
  if (F == AF_FP8) return u ^ (((r >> 1) & 3) << 1);
  return u ^ (((r >> 2) & 1) << 1);
}

// K row r (of this stage) -> 16 packed pairs (bf16x2 / f16x2) of d [32 tig, +32) in pd order (scales applied)
template <int F>
DEVI void load_k(uint32_t kb, uint32_t ksb, int r, int tig, __half2 kfac, uint32_t (&hk)[16]) {
  constexpr int RB = Fmt<F>::RB;
  const uint32_t base = kb + r * RB;
  if constexpr (F == AF_BF16) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint4 x = lds128(base + kswz<F>(r, 4 * tig + j) * 16);
      hk[4 * j] = x.x; hk[4 * j + 1] = x.y; hk[4 * j + 2] = x.z; hk[4 * j + 3] = x.w;
    }
  } else if constexpr (F == AF_FP8) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const uint4 x = lds128(base + kswz<F>(r, 2 * tig + j) * 16);
      const uint32_t wv[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        hk[8 * j + 2 * k] = e4m3lo(wv[k]);
        hk[8 * j + 2 * k + 1] = e4m3hi(wv[k]);
      }
    }
  } else {
    const uint4 x = lds128(base + tig * 16);
    const uint32_t wv[4] = {x.x, x.y, x.z, x.w};
    __half2 s2[4];
    if constexpr (F == AF_NVFP4) {  // the two e4m3 block scales of this thread -> f16x2, x (64 / global scale)
      const __half2 sc = u2h(e4m3lo(lds16(ksb + r * 8 + 2 * tig)));  // 1 / global scale is folded into q
      s2[0] = s2[1] = __low2half2(sc);
      s2[2] = s2[3] = __high2half2(sc);
    } else if constexpr (F == AF_MXFP4) {
      const uint32_t h = mx64h(lds8(ksb + r * 4 + tig));
      s2[0] = s2[1] = s2[2] = s2[3] = u2h(h | (h << 16));
    } else {
      const uint32_t sv = lds16(ksb + r * 8 + 2 * tig);
      s2[0] = s2[1] = s2[2] = s2[3] = u2h(sv | (sv << 16));
    }
#ifdef ATTN_NODEQ  // experiment: skip the dequant (garbage numerics) to measure the memory / MMA floor
#pragma unroll
    for (int k = 0; k < 4; ++k) { hk[4 * k] = wv[k]; hk[4 * k + 1] = wv[k] ^ h2u(s2[k]); hk[4 * k + 2] = wv[k] >> 1; hk[4 * k + 3] = wv[k] >> 2; }
    return;
#endif
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      if constexpr (F == AF_INT4) {
#pragma unroll
        for (int i = 0; i < 4; ++i) hk[4 * k + i] = i4pair(wv[k], 4 * i, s2[k]);
      } else {
        uint32_t lo, hi;
        fp4x8(wv[k], lo, hi);
        hk[4 * k + 0] = hmul2(b2h_lo(lo), s2[k]);
        hk[4 * k + 1] = hmul2(b2h_hi(lo), s2[k]);
        hk[4 * k + 2] = hmul2(b2h_lo(hi), s2[k]);
        hk[4 * k + 3] = hmul2(b2h_hi(hi), s2[k]);
      }
    }
  }
}

// V rows ra, rb (a token pair) -> 16 packed pairs (token ra in the low half) for d = 16 gid + e, e = 0..15
template <int F>
DEVI void load_v(uint32_t vb, uint32_t vsb, int ra, int rb, int gid, __half2 vfac, uint32_t (&pv)[16]) {
  constexpr int RB = Fmt<F>::RB;
  if constexpr (F == AF_BF16) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const uint4 xa = lds128(vb + vrow<F>(ra) * RB + vswz<F>(ra, 2 * gid + j) * 16);
      const uint4 xb = lds128(vb + vrow<F>(rb) * RB + vswz<F>(rb, 2 * gid + j) * 16);
      const uint32_t wa[4] = {xa.x, xa.y, xa.z, xa.w}, wb[4] = {xb.x, xb.y, xb.z, xb.w};
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        pv[8 * j + 2 * k] = __byte_perm(wa[k], wb[k], 0x5410);
        pv[8 * j + 2 * k + 1] = __byte_perm(wa[k], wb[k], 0x7632);
      }
    }
  } else if constexpr (F == AF_FP8) {
    const uint4 xa = lds128(vb + vrow<F>(ra) * RB + vswz<F>(ra, gid) * 16);
    const uint4 xb = lds128(vb + vrow<F>(rb) * RB + vswz<F>(rb, gid) * 16);
    const uint32_t wa[4] = {xa.x, xa.y, xa.z, xa.w}, wb[4] = {xb.x, xb.y, xb.z, xb.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const uint32_t x = __byte_perm(wa[k], wb[k], 0x5140), y = __byte_perm(wa[k], wb[k], 0x7362);
      pv[4 * k] = e4m3lo(x); pv[4 * k + 1] = e4m3hi(x);
      pv[4 * k + 2] = e4m3lo(y); pv[4 * k + 3] = e4m3hi(y);
    }
  } else {
    const uint2 xa = lds64(vb + vrow<F>(ra) * 64 + ((gid * 8) ^ (((ra >> 2) & 1) << 5)));
    const uint2 xb = lds64(vb + vrow<F>(rb) * 64 + ((gid * 8) ^ (((rb >> 2) & 1) << 5)));
    const uint32_t wa[2] = {xa.x, xa.y}, wb[2] = {xb.x, xb.y};
    __half2 s2;
    if constexpr (F == AF_NVFP4) {
      s2 = u2h(e4m3lo(lds8(vsb + ra * 8 + gid) | (lds8(vsb + rb * 8 + gid) << 8)));  // 1 / gs folded into P
    } else if constexpr (F == AF_MXFP4) {
      s2 = u2h(mx64h(lds8(vsb + ra * 4 + (gid >> 1))) | (mx64h(lds8(vsb + rb * 4 + (gid >> 1))) << 16));
    } else {
      s2 = u2h(lds16(vsb + ra * 8 + 2 * (gid >> 1)) | (lds16(vsb + rb * 8 + 2 * (gid >> 1)) << 16));
    }
#ifdef ATTN_NODEQ
#pragma unroll
    for (int k = 0; k < 2; ++k) for (int i = 0; i < 8; ++i) pv[8 * k + i] = (wa[k] >> i) ^ (wb[k] << i) ^ h2u(s2);
    return;
#endif
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      if constexpr (F == AF_INT4) {
        const uint32_t x = __byte_perm(wa[k], wb[k], 0x5410), y = __byte_perm(wa[k], wb[k], 0x7632);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          pv[8 * k + i] = i4pair(x, 4 * i, s2);
          pv[8 * k + 4 + i] = i4pair(y, 4 * i, s2);
        }
      } else {
        // interleave the token bytes first: x = [a.b0, b.b0, a.b1, b.b1] holds nibbles a0 a1 | b0 b1 | a2 a3 | b2 b3,
        // so expand(x) gives lo = [a0, b0, a2, b2], hi = [a1, b1, a3, b3] (token pairs per byte pair)
        const uint32_t x = __byte_perm(wa[k], wb[k], 0x5140), y = __byte_perm(wa[k], wb[k], 0x7362);
        uint32_t lx, hx, ly, hy;
        fp4x8(x, lx, hx);
        fp4x8(y, ly, hy);
        pv[8 * k + 0] = hmul2(b2h_lo(lx), s2); pv[8 * k + 2] = hmul2(b2h_hi(lx), s2);
        pv[8 * k + 1] = hmul2(b2h_lo(hx), s2); pv[8 * k + 3] = hmul2(b2h_hi(hx), s2);
        pv[8 * k + 4] = hmul2(b2h_lo(ly), s2); pv[8 * k + 6] = hmul2(b2h_hi(ly), s2);
        pv[8 * k + 5] = hmul2(b2h_lo(hy), s2); pv[8 * k + 7] = hmul2(b2h_hi(hy), s2);
      }
    }
  }
}

// e2m1 formats: K / V operands come out as value * 2^-7 (MXFP4: e2m1 * 2^-14 * (2^(e-127) * 2^7)) or as
// value * gs * 2^-14 (NVFP4: e2m1 * 2^-14 * e4m3 scale, the 1/gs not applied); q and P are pre-multiplied by the
// inverse factor pmul (exact powers of two except the NVFP4 1/gs).
template <int F>
DEVI float operand_pmul(const AttnParams& p) {
  if constexpr (F == AF_NVFP4) return 16384.0f / __ldg(p.gscale);
  if constexpr (F == AF_MXFP4) return 128.0f;
  return 1.0f;
}

template <int F, int NT, int ST>
struct Cfg {
  static constexpr int RB = Fmt<F>::RB, SB = Fmt<F>::SB, TOK = 16 * NT;
  static constexpr int KB = TOK * RB;
  static constexpr int STG = 2 * KB + 2 * TOK * SB;  // bytes per warp per stage
  // ring; reused by the prologue (q, 2 KB) and the epilogue (warp partials 8 KB, output 2 KB, merge weights <= 8 KB)
  static constexpr int SMEM = NW * ST * STG > 18 * 1024 ? NW * ST * STG : 18 * 1024;
};

// split-K partial write / last-CTA merge / FlagRot back-rotation / bf16 store.  Called by all 128 threads with the
// CTA's unnormalized output oc[h] for d = threadIdx.x, its running max Mc[h] (log2 domain) and sum Lc[h].
// smem: final output staged at +8 KB ([G][D] fp32), merge weights at +10 KB (<= 512 splits).
DEVI void attn_epilogue(const AttnParams& p, uint8_t* smem, int kvh, int split, int b, const float (&oc)[G],
                        const float (&Mc)[G], const float (&Lc)[G]) {
  __shared__ int last_flag;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int d = threadIdx.x;
  float (*fin)[D] = reinterpret_cast<float (*)[D]>(smem + NW * G * D * 4);
  if (p.nsplit == 1) {
#pragma unroll
    for (int h = 0; h < G; ++h) fin[h][d] = oc[h] / Lc[h];
  } else {
    const size_t base = (static_cast<size_t>(b) * p.Hkv + kvh) * p.nsplit;
#pragma unroll
    for (int h = 0; h < G; ++h) p.opart[((base + split) * G + h) * D + d] = oc[h];
    if (d < G) {
      float2 ml;
      ml.x = d == 0 ? Mc[0] : d == 1 ? Mc[1] : d == 2 ? Mc[2] : Mc[3];
      ml.y = d == 0 ? Lc[0] : d == 1 ? Lc[1] : d == 2 ? Lc[2] : Lc[3];
      reinterpret_cast<float2*>(p.mlpart)[(base + split) * G + d] = ml;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      const int prev = atomicAdd(p.counter + b * p.Hkv + kvh, 1);
      last_flag = prev == p.nsplit - 1;
    }
    __syncthreads();
    if (!last_flag) return;
    __threadfence();
    const float2* mlp = reinterpret_cast<const float2*>(p.mlpart) + base * G;
    const float* op = p.opart + base * G * D;
    float* wgt = reinterpret_cast<float*>(smem + NW * G * D * 4 + G * D * 4);  // [nsplit][G] merge weights
    __shared__ float Lh[G];
    if (threadIdx.x < 32) {  // warp 0: per-head max and total weight over the splits
#pragma unroll
      for (int h = 0; h < G; ++h) {
        float M = NEG_INF;
        for (int s = lane; s < p.nsplit; s += 32) M = fmaxf(M, __ldcg(mlp + s * G + h).x);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(FULL, M, o));
        const float Ms = M == NEG_INF ? 0.f : M;
        float l = 0.f;
        for (int s = lane; s < p.nsplit; s += 32) {
          const float2 ml = __ldcg(mlp + s * G + h);
          const float sc = ex2(ml.x - Ms);
          wgt[s * G + h] = sc;
          l += sc * ml.y;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(FULL, l, o);
        if (lane == 0) Lh[h] = 1.f / l;
      }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < G; ++h) {
      float o = 0.f;
#pragma unroll 4
      for (int s = 0; s < p.nsplit; ++s) o += wgt[s * G + h] * __ldcg(op + (s * G + h) * D + d);
      fin[h][d] = o * Lh[h];
    }
    if (threadIdx.x == 0) p.counter[b * p.Hkv + kvh] = 0;  // ready for the next launch / graph replay
  }
  __syncthreads();
  // ---------------------------------------------------------------- finalize: warp w = q head w
  float x[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] = fin[w][lane * 4 + i];
  if (p.vrot) warp_selrot(x, p.vmask + kvh * (D / p.rb), p.rb, lane);
  __nv_bfloat162 o01 = __floats2bfloat162_rn(x[0], x[1]), o23 = __floats2bfloat162_rn(x[2], x[3]);
  uint2 ov;
  ov.x = *reinterpret_cast<uint32_t*>(&o01);
  ov.y = *reinterpret_cast<uint32_t*>(&o23);
  *reinterpret_cast<uint2*>(static_cast<__nv_bfloat16*>(p.out) + b * p.o_bs + (kvh * G + w) * D + lane * 4) = ov;
}

#ifndef ATTN_MINB
#define ATTN_MINB(F, NT, ST) ((Fmt<F>::RB == 64 && ST == 2) ? 5 : 1)
#endif
template <int F, int NT, int ST, bool QROT>
__global__ void __launch_bounds__(NW * 32, ATTN_MINB(F, NT, ST)) k_attn(const AttnParams p) {
  using C = Cfg<F, NT, ST>;
  constexpr int RB = C::RB, SB = C::SB, TOK = C::TOK, KB = C::KB, STG = C::STG;
  constexpr bool BF = F == AF_BF16;
  extern __shared__ __align__(128) uint8_t smem[];
  float (*qs)[D] = reinterpret_cast<float (*)[D]>(smem);  // prologue: q [G][D] (ring not yet in use)
  __shared__ float red_m[NW][G], red_l[NW][G];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, gid = lane >> 2, tig = lane & 3;
  const int kvh = blockIdx.x, split = blockIdx.y, b = blockIdx.z;
  const int c0 = split * p.chunk, c1 = min(p.L, c0 + p.chunk);

  // ---------------------------------------------------------------- prologue: q (warp w = q head w of the group)
  {
    const __nv_bfloat16* qp = static_cast<const __nv_bfloat16*>(p.q) + b * p.q_bs + (kvh * G + w) * D + lane * 4;
    const uint2 raw = *reinterpret_cast<const uint2*>(qp);
    float x[4] = {__uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xffff0000u),
                  __uint_as_float(raw.y << 16), __uint_as_float(raw.y & 0xffff0000u)};
    if constexpr (QROT) warp_selrot(x, p.kmask + kvh * (D / p.rb), p.rb, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) qs[w][lane * 4 + i] = (BF ? x[i] : x[i] * (p.qscale * operand_pmul<F>(p)));
  }
  __half2 kfac = __float2half2_rn(0.f);
  if constexpr (F == AF_NVFP4) kfac = __float2half2_rn(128.0f / __ldg(p.gscale));
  __syncthreads();
  uint32_t bq[8][2];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int pi = 2 * j + hh;
      const float a = gid < G ? qs[gid & 3][32 * tig + pd0<F>(pi)] : 0.f;
      const float c = gid < G ? qs[gid & 3][32 * tig + pd1<F>(pi)] : 0.f;
      bq[j][hh] = pack2<BF>(a, c);
    }
  }
  __syncthreads();  // q staging area is part of the cp.async ring

  // ---------------------------------------------------------------- main loop
  float acc[8][4];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
  float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.f, l1 = 0.f;  // heads 2 tig, 2 tig + 1 (>= 4: padding)

  const int ntok = max(c1 - c0, 0);
  const int ng = (ntok + TOK - 1) / TOK;
  const int nwg = ng > w ? (ng - w + NW - 1) / NW : 0;
  const uint8_t* kg = p.k + b * p.bat_stride + kvh * RB;
  const uint8_t* vg = p.v + b * p.bat_stride + kvh * RB;
  const uint8_t* ksg = SB ? p.ks + b * p.sbat_stride + kvh * SB : nullptr;
  const uint8_t* vsg = SB ? p.vs + b * p.sbat_stride + kvh * SB : nullptr;
  const uint32_t wsm = smem_u32(smem) + w * ST * STG;
  const size_t tstr = static_cast<size_t>(p.tok_stride), sstr = static_cast<size_t>(p.stok_stride);
  const int Lm1 = p.L - 1;

  // cp.async issue.  Copy i of a lane: 16-B unit u = lane % UPR of row r = i * RPI + lane / UPR (RPI = 32 / UPR), so
  // every global / shared offset is a lane constant plus a compile-time (or uniform) multiple of i.
  constexpr int UPR = RB / 16, RPI = 32 / UPR, NCP = TOK * UPR / 32;
  const int r0 = lane / UPR, u0 = lane % UPR;
  const size_t goff0 = static_cast<size_t>(r0) * tstr + u0 * 16;
  const size_t gstep = static_cast<size_t>(RPI) * tstr;
  const uint32_t kdst0 = r0 * RB + kswz<F>(r0, u0) * 16;  // kswz depends on r only through r & 1 = r0 & 1
  const long long vgap = vg - kg;
  const int sr0 = lane;  // scale copies: c = i * 32 + lane (K rows first, then V rows)
  auto issue_tail = [&](int t0, uint32_t sb) {
#pragma unroll
    for (int i = 0; i < NCP; ++i) {
      const int r = i * RPI + r0;
      const size_t off = static_cast<size_t>(min(t0 + r, Lm1)) * tstr + u0 * 16;
      cp16(sb + r * RB + kswz<F>(r, u0) * 16, kg + off);
      cp16(sb + KB + vrow<F>(r) * RB + vswz<F>(r, u0) * 16, vg + off);
    }
    if constexpr (SB > 0) {
#pragma unroll
      for (int i = 0; i < 2 * TOK / 32; ++i) {
        const int c = i * 32 + sr0;
        const bool isv = c >= TOK;
        const int r = isv ? c - TOK : c;
        cpsmall<SB>(sb + 2 * KB + c * SB, (isv ? vsg : ksg) + static_cast<size_t>(min(t0 + r, Lm1)) * sstr);
      }
    }
  };
  auto issue = [&](int g, int stg) {
    const uint32_t sb = wsm + stg * STG;
    const int t0 = c0 + g * TOK;
    if (t0 + TOK > p.L) { issue_tail(t0, sb); return; }
    const uint8_t* src = kg + static_cast<size_t>(t0) * tstr + goff0;
#pragma unroll
    for (int i = 0; i < NCP; ++i) {
      const int r = i * RPI + r0;
      cp16(sb + i * RPI * RB + kdst0, src);
      cp16(sb + KB + vrow<F>(r) * RB + vswz<F>(r, u0) * 16, src + vgap);
      src += gstep;
    }
    if constexpr (SB > 0) {
#pragma unroll
      for (int i = 0; i < 2 * TOK / 32; ++i) {
        const int c = i * 32 + sr0;
        const bool isv = c >= TOK;
        const int r = isv ? c - TOK : c;
        cpsmall<SB>(sb + 2 * KB + c * SB, (isv ? vsg : ksg) + static_cast<size_t>(t0 + r) * sstr);
      }
    }
  };

#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < nwg) issue(w + NW * s, s);
    cp_commit();
  }
  const __half2 vfac = kfac;
  const float pshift = log2f(operand_pmul<F>(p));  // exact for the powers of two (MXFP4: 7; others 0)
  for (int it = 0; it < nwg; ++it) {
    const int stg = it % ST;
    {
      const int nx = it + ST - 1;
      if (nx < nwg) issue(w + NW * nx, nx % ST);
      cp_commit();
    }
    cp_wait<ST - 1>();
    __syncwarp();
    const uint32_t sb = wsm + stg * STG;
    const uint32_t kb = sb, vb = sb + KB, ksb = sb + 2 * KB, vsb = sb + 2 * KB + TOK * SB;
    const int t0 = c0 + (w + NW * it) * TOK;

    float sacc[NT][4];
#pragma unroll
    for (int tt = 0; tt < NT; ++tt) {
      float s2acc[4] = {0.f, 0.f, 0.f, 0.f};
      sacc[tt][0] = sacc[tt][1] = sacc[tt][2] = sacc[tt][3] = 0.f;
      uint32_t h0[16], h1[16];
      load_k<F>(kb, ksb, tt * 16 + gid, tig, kfac, h0);
      load_k<F>(kb, ksb, tt * 16 + gid + 8, tig, kfac, h1);
#pragma unroll
      for (int j = 0; j < 8; j += 2) {  // two independent accumulation chains (halves the dependent-MMA latency)
        mma<BF>(sacc[tt], h0[2 * j], h1[2 * j], h0[2 * j + 1], h1[2 * j + 1], bq[j][0], bq[j][1]);
        mma<BF>(s2acc, h0[2 * j + 2], h1[2 * j + 2], h0[2 * j + 3], h1[2 * j + 3], bq[j + 1][0], bq[j + 1][1]);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) sacc[tt][i] = BF ? (sacc[tt][i] + s2acc[i]) * p.qscale : sacc[tt][i] + s2acc[i];
      if constexpr (F == AF_FP8) {
        const float s0 = __uint_as_float(lds32(ksb + (tt * 16 + gid) * 4));
        const float s1 = __uint_as_float(lds32(ksb + (tt * 16 + gid + 8) * 4));
        sacc[tt][0] *= s0; sacc[tt][1] *= s0; sacc[tt][2] *= s1; sacc[tt][3] *= s1;
      }
    }
    {  // tail mask, branch-free (predicated selects)
#pragma unroll
      for (int tt = 0; tt < NT; ++tt) {
        const int r0 = t0 + tt * 16 + gid;
        sacc[tt][0] = r0 < c1 ? sacc[tt][0] : NEG_INF;
        sacc[tt][1] = r0 < c1 ? sacc[tt][1] : NEG_INF;
        sacc[tt][2] = r0 + 8 < c1 ? sacc[tt][2] : NEG_INF;
        sacc[tt][3] = r0 + 8 < c1 ? sacc[tt][3] : NEG_INF;
      }
    }
    float x0 = NEG_INF, x1 = NEG_INF;
#pragma unroll
    for (int tt = 0; tt < NT; ++tt) {
      x0 = fmaxf(x0, fmaxf(sacc[tt][0], sacc[tt][2]));
      x1 = fmaxf(x1, fmaxf(sacc[tt][1], sacc[tt][3]));
    }
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) {
      x0 = fmaxf(x0, __shfl_xor_sync(FULL, x0, o));
      x1 = fmaxf(x1, __shfl_xor_sync(FULL, x1, o));
    }
    const float mn0 = fmaxf(m0, x0), mn1 = fmaxf(m1, x1);
    const float ms0 = mn0 == NEG_INF ? 0.f : mn0, ms1 = mn1 == NEG_INF ? 0.f : mn1;
    const float al0 = ex2(m0 - ms0), al1 = ex2(m1 - ms1);
    m0 = mn0; m1 = mn1;
    // P is produced directly as p * 2^pshift (the operand scale of the e2m1 formats; l is rescaled at the end)
    const float me0 = ms0 - pshift, me1 = ms1 - pshift;
    float ps[NT][4];
    float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
    for (int tt = 0; tt < NT; ++tt) {
      ps[tt][0] = ex2(sacc[tt][0] - me0); ps[tt][1] = ex2(sacc[tt][1] - me1);
      ps[tt][2] = ex2(sacc[tt][2] - me0); ps[tt][3] = ex2(sacc[tt][3] - me1);
      sum0 += ps[tt][0] + ps[tt][2];
      sum1 += ps[tt][1] + ps[tt][3];
    }
    l0 = l0 * al0 + sum0;
    l1 = l1 * al1 + sum1;
#ifndef ATTN_EAGER_RESCALE
    if (!__all_sync(FULL, al0 == 1.f && al1 == 1.f))  // skip the 32 FMUL when no head's running max moved
#endif
    {
#pragma unroll
      for (int mt = 0; mt < 8; ++mt) {
        acc[mt][0] *= al0; acc[mt][2] *= al0;
        acc[mt][1] *= al1; acc[mt][3] *= al1;
      }
    }
#pragma unroll
    for (int tt = 0; tt < NT; ++tt) {
      float q0 = ps[tt][0], q1 = ps[tt][1], q2 = ps[tt][2], q3 = ps[tt][3];
      if constexpr (F == AF_FP8) {  // V scale per token folded into P (the MMA operand only, not into l)
        const float s0 = __uint_as_float(lds32(vsb + (tt * 16 + gid) * 4));
        const float s1 = __uint_as_float(lds32(vsb + (tt * 16 + gid + 8) * 4));
        q0 *= s0; q1 *= s0; q2 *= s1; q3 *= s1;
      }
      const uint32_t b0 = movt(pack2<BF>(q0, q1)), b1 = movt(pack2<BF>(q2, q3));
      uint32_t pa[16], pb[16];
      load_v<F>(vb, vsb, tt * 16 + 2 * tig, tt * 16 + 2 * tig + 1, gid, vfac, pa);
      load_v<F>(vb, vsb, tt * 16 + 2 * tig + 8, tt * 16 + 2 * tig + 9, gid, vfac, pb);
#pragma unroll
      for (int mt = 0; mt < 8; ++mt) mma<BF>(acc[mt], pa[2 * mt], pa[2 * mt + 1], pb[2 * mt], pb[2 * mt + 1], b0, b1);
    }
    __syncwarp();
  }
  cp_wait<0>();

  // ---------------------------------------------------------------- CTA merge of the 4 warps
  l0 *= ex2(-pshift);
  l1 *= ex2(-pshift);
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) {
    l0 += __shfl_xor_sync(FULL, l0, o);
    l1 += __shfl_xor_sync(FULL, l1, o);
  }
  __syncthreads();  // all warps done with their rings -> reuse
  float* ow = reinterpret_cast<float*>(smem);  // [NW][G][D]
  if (tig < 2) {
    const int h0 = 2 * tig, h1 = 2 * tig + 1;
#pragma unroll
    for (int mt = 0; mt < 8; ++mt) {
      const int d = 16 * gid + 2 * mt;
      ow[(w * G + h0) * D + d] = acc[mt][0];
      ow[(w * G + h0) * D + d + 1] = acc[mt][2];
      ow[(w * G + h1) * D + d] = acc[mt][1];
      ow[(w * G + h1) * D + d + 1] = acc[mt][3];
    }
    if (gid == 0) {
      red_m[w][h0] = m0; red_m[w][h1] = m1;
      red_l[w][h0] = l0; red_l[w][h1] = l1;
    }
  }
  __syncthreads();
  const int d = threadIdx.x;
  float oc[G], Mc[G], Lc[G];
#pragma unroll
  for (int h = 0; h < G; ++h) {
    float M = NEG_INF;
#pragma unroll
    for (int ww = 0; ww < NW; ++ww) M = fmaxf(M, red_m[ww][h]);
    const float Ms = M == NEG_INF ? 0.f : M;
    float o = 0.f, l = 0.f;
#pragma unroll
    for (int ww = 0; ww < NW; ++ww) {
      const float sc = ex2(red_m[ww][h] - Ms);
      o += sc * ow[(ww * G + h) * D + d];
      l += sc * red_l[ww][h];
    }
    oc[h] = o; Mc[h] = M; Lc[h] = l;
  }
  attn_epilogue(p, smem, kvh, split, b, oc, Mc, Lc);
}

// =====================================================================================================================
// Warpgroup variant (wgmma).  On H20, mma.sync m16n8k16 sustains ~96 TFLOPS but wgmma m64n8k16 (A from registers)
// ~141 TFLOPS (microbenchmarked); with GQA 4 half of N = 8 is padding, so for 4-bit caches the mma.sync kernel above
// is tensor-pipe bound.  Here the 4 warps form one warpgroup and process 64-token blocks together:
//   S^T[64 tok, 8 h]  = K[64, 128] . Q^T          8 x wgmma.m64n8k16, A = K fragments (registers, warp w = tokens
//                                                  16w..16w+15, same thread layout as mma.sync), B = Q^T (smem)
//   O^T[128 d, 8 h] += V^T[128, 64] . P^T          2 (d tiles) x 4 (k steps) wgmma.m64n8k16, A = V^T fragments
//                                                  (warp w = d rows 32w..32w+31), B = P^T (smem, written per block)
// The QK wgmmas run asynchronously while the warp dequantizes its V fragments.  Per 64-token block: one cp.async
// stage barrier, one barrier for the block max (all 64 tokens share one softmax reference), one for P^T.
// V fragment: thread (gid, tig) of warp w holds tokens {2tig, 2tig+1, 2tig+8, 2tig+9} (+16 j) at d = 32w + 4gid + e,
// e = 0..3 (tile mt row gid <-> e = 2 mt, row gid + 8 <-> e = 2 mt + 1); V rows are stored with the 16-B units XORed
// by (VSWZ * ((token >> 1) & 3)) so these 2/4/8-byte reads are bank-conflict free.
template <int F>
struct WgC {
  static constexpr int RB = Fmt<F>::RB, SB = Fmt<F>::SB, TB = 64;
  static constexpr int KB = TB * RB;
  static constexpr int STG = 2 * KB + 2 * TB * SB;
  static constexpr int VSWZ = RB == 64 ? 1 : 2;
  static constexpr int HDR = 3072;  // Q^T operand (2 KB) + P^T operand (1 KB)
};
template <int F, int ST>
struct WgSmem {
  static constexpr int SMEM = WgC<F>::HDR + ST * WgC<F>::STG > 18 * 1024 ? WgC<F>::HDR + ST * WgC<F>::STG : 18 * 1024;
};

DEVI uint64_t sdesc(uint32_t saddr) {  // no swizzle, K-major core matrices: LBO (K) = 128 B, SBO (N) = 256 B
  return static_cast<uint64_t>((saddr & 0x3FFFF) >> 4) | (static_cast<uint64_t>(128 >> 4) << 16) |
         (static_cast<uint64_t>(256 >> 4) << 32);
}
DEVI void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
DEVI void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
DEVI void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
template <bool BF>
DEVI void wgmma_rs(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
  if constexpr (BF)
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  else
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
}
// wait for all committed wgmma groups; the empty asm statements keep the A-operand registers alive (unmodified) up to
// this point (the compiler cannot see that wgmma reads them asynchronously)
DEVI void wg_wait0(float (&d)[4]) {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) :: "memory");
}
DEVI void pin16(uint32_t (&r)[16]) {
  asm volatile("" : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]), "+r"(r[3]), "+r"(r[4]), "+r"(r[5]), "+r"(r[6]), "+r"(r[7]),
               "+r"(r[8]), "+r"(r[9]), "+r"(r[10]), "+r"(r[11]), "+r"(r[12]), "+r"(r[13]), "+r"(r[14]), "+r"(r[15]));
}

// V fragments of the WG kernel: tokens ta, tb (a pair) at d = 32 w + 4 gid + e -> pv[e] (token ta in the low half)
template <int F>
DEVI void load_v_wg(uint32_t vb, uint32_t vsb, int ta, int tb, int w, int gid, __half2 vfac, uint32_t (&pv)[4]) {
  constexpr int RB = Fmt<F>::RB, VS = WgC<F>::VSWZ;
  auto va = [&](int t, int boff) -> uint32_t {
    return vb + t * RB + ((((boff >> 4) ^ (VS * ((t >> 1) & 3)))) << 4) + (boff & 15);
  };
  if constexpr (F == AF_BF16) {
    const uint2 a = lds64(va(ta, 64 * w + 8 * gid)), c = lds64(va(tb, 64 * w + 8 * gid));
    pv[0] = __byte_perm(a.x, c.x, 0x5410); pv[1] = __byte_perm(a.x, c.x, 0x7632);
    pv[2] = __byte_perm(a.y, c.y, 0x5410); pv[3] = __byte_perm(a.y, c.y, 0x7632);
  } else if constexpr (F == AF_FP8) {
    const uint32_t a = lds32(va(ta, 32 * w + 4 * gid)), c = lds32(va(tb, 32 * w + 4 * gid));
    const uint32_t x = __byte_perm(a, c, 0x5140), y = __byte_perm(a, c, 0x7362);
    pv[0] = e4m3lo(x); pv[1] = e4m3hi(x); pv[2] = e4m3lo(y); pv[3] = e4m3hi(y);
  } else {
    const uint32_t x = lds16(va(ta, 16 * w + 2 * gid)) | (lds16(va(tb, 16 * w + 2 * gid)) << 16);  // a d0..3 | b d0..3
    __half2 s2;
    if constexpr (F == AF_NVFP4) {
      s2 = u2h(e4m3lo(lds8(vsb + ta * 8 + 2 * w + (gid >> 2)) | (lds8(vsb + tb * 8 + 2 * w + (gid >> 2)) << 8)));
    } else if constexpr (F == AF_MXFP4) {
      s2 = u2h(mx64h(lds8(vsb + ta * 4 + w)) | (mx64h(lds8(vsb + tb * 4 + w)) << 16));
    } else {
      s2 = u2h(lds16(vsb + ta * 8 + 2 * w) | (lds16(vsb + tb * 8 + 2 * w) << 16));
    }
    if constexpr (F == AF_INT4) {
#pragma unroll
      for (int i = 0; i < 4; ++i) pv[i] = i4pair(x, 4 * i, s2);
    } else {
      uint32_t lo, hi;  // lo = [a.d0, a.d2, b.d0, b.d2], hi = [a.d1, a.d3, b.d1, b.d3]
      fp4x8(x, lo, hi);
      pv[0] = hmul2(__byte_perm(lo, 0u, 0x2404), s2); pv[2] = hmul2(__byte_perm(lo, 0u, 0x3414), s2);
      pv[1] = hmul2(__byte_perm(hi, 0u, 0x2404), s2); pv[3] = hmul2(__byte_perm(hi, 0u, 0x3414), s2);
    }
  }
}

template <int F, int ST, bool QROT>
__global__ void __launch_bounds__(NW * 32) k_attn_wg(const AttnParams p) {
  using C = WgC<F>;
  constexpr int RB = C::RB, SB = C::SB, TB = C::TB, KB = C::KB, STG = C::STG, VS = C::VSWZ;
  constexpr bool BF = F == AF_BF16;
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ float red[NW][8], lred[NW][8], mfin[8];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, gid = lane >> 2, tig = lane & 3;
  const int kvh = blockIdx.x, split = blockIdx.y, b = blockIdx.z;
  const int c0 = split * p.chunk, c1 = min(p.L, c0 + p.chunk);
  uint8_t* qb = smem;            // Q^T  [8 k-steps][2 core][8 heads][16 B]
  uint8_t* pt = smem + 2048;     // P^T  [8 core (8 tokens each)][8 heads][16 B]
  uint8_t* ring = smem + C::HDR;
  const uint32_t qb_u = smem_u32(qb), pt_u = smem_u32(pt), ring_u = smem_u32(ring);

  // ---------------------------------------------------------------- prologue: q -> Q^T operand (fp16 / bf16)
  {
    float (*qs)[D] = reinterpret_cast<float (*)[D]>(ring);
    const __nv_bfloat16* qp = static_cast<const __nv_bfloat16*>(p.q) + b * p.q_bs + (kvh * G + w) * D + lane * 4;
    const uint2 raw = *reinterpret_cast<const uint2*>(qp);
    float x[4] = {__uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xffff0000u),
                  __uint_as_float(raw.y << 16), __uint_as_float(raw.y & 0xffff0000u)};
    if constexpr (QROT) warp_selrot(x, p.kmask + kvh * (D / p.rb), p.rb, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) qs[w][lane * 4 + i] = (BF ? x[i] : x[i] * (p.qscale * operand_pmul<F>(p)));
    __syncthreads();
    for (int i = tid; i < 8 * 16 * 8; i += NW * 32) {
      const int j = i >> 7, kl = (i >> 3) & 15, n = i & 7;      // k step, local k, head
      const int tg = (kl & 7) >> 1, e = kl & 1, pi = 2 * j + (kl >> 3);
      const int dd = 32 * tg + (e ? pd1<F>(pi) : pd0<F>(pi));
      const float v = n < G ? qs[n][dd] : 0.f;
      uint16_t hv;
      if constexpr (BF) hv = __bfloat16_as_ushort(__float2bfloat16_rn(v));
      else hv = __half_as_ushort(__float2half_rn(v));
      *reinterpret_cast<uint16_t*>(qb + j * 256 + (kl >> 3) * 128 + n * 16 + (kl & 7) * 2) = hv;
    }
    for (int i = tid; i < 256; i += NW * 32) reinterpret_cast<uint32_t*>(pt)[i] = 0u;  // padding heads 4..7 stay 0
    fence_async_smem();
    __syncthreads();
  }
  __half2 kfac = __float2half2_rn(0.f);
  if constexpr (F == AF_NVFP4) kfac = __float2half2_rn(128.0f / __ldg(p.gscale));
  const __half2 vfac = kfac;
  const uint64_t qdesc = sdesc(qb_u), pdesc = sdesc(pt_u);

  // ---------------------------------------------------------------- cp.async ring (CTA-wide, 64-token blocks)
  const int ntok = max(c1 - c0, 0), nblk = (ntok + TB - 1) / TB;
  const uint8_t* kg = p.k + b * p.bat_stride + kvh * RB;
  const uint8_t* vg = p.v + b * p.bat_stride + kvh * RB;
  const uint8_t* ksg = SB ? p.ks + b * p.sbat_stride + kvh * SB : nullptr;
  const uint8_t* vsg = SB ? p.vs + b * p.sbat_stride + kvh * SB : nullptr;
  const size_t tstr = static_cast<size_t>(p.tok_stride), sstr = static_cast<size_t>(p.stok_stride);
  const int Lm1 = p.L - 1;
  constexpr int UPR = RB / 16, RPI = 128 / UPR, NCP = TB * UPR / 128;
  const int r0 = tid / UPR, u0 = tid % UPR;
  const uint32_t kdst0 = r0 * RB + kswz<F>(r0, u0) * 16;
  const uint32_t vdst0 = KB + r0 * RB + ((u0 ^ (VS * ((r0 >> 1) & 3))) << 4);  // (r >> 1) & 3 = (r0 >> 1) & 3
  const long long vgap = vg - kg;
  auto issue = [&](int blk, int stg) {
    const uint32_t sb = ring_u + stg * STG;
    const int t0 = c0 + blk * TB;
    if (t0 + TB <= p.L) {
      const uint8_t* src = kg + static_cast<size_t>(t0 + r0) * tstr + u0 * 16;
#pragma unroll
      for (int i = 0; i < NCP; ++i) {
        cp16(sb + kdst0 + i * RPI * RB, src);
        cp16(sb + vdst0 + i * RPI * RB, src + vgap);
        src += static_cast<size_t>(RPI) * tstr;
      }
      if constexpr (SB > 0)
        cpsmall<SB>(sb + 2 * KB + tid * SB, (tid < TB ? ksg : vsg) + static_cast<size_t>(t0 + (tid & (TB - 1))) * sstr);
    } else {
#pragma unroll
      for (int i = 0; i < NCP; ++i) {
        const uint8_t* src = kg + static_cast<size_t>(min(t0 + i * RPI + r0, Lm1)) * tstr + u0 * 16;
        cp16(sb + kdst0 + i * RPI * RB, src);
        cp16(sb + vdst0 + i * RPI * RB, src + vgap);
      }
      if constexpr (SB > 0)
        cpsmall<SB>(sb + 2 * KB + tid * SB, (tid < TB ? ksg : vsg) + static_cast<size_t>(min(t0 + (tid & (TB - 1)), Lm1)) * sstr);
    }
  };
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < nblk) issue(s, s);
    cp_commit();
  }

  float oacc[2][4];
#pragma unroll
  for (int i = 0; i < 2; ++i) oacc[i][0] = oacc[i][1] = oacc[i][2] = oacc[i][3] = 0.f;
  float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.f, l1 = 0.f;  // heads 2 tig, 2 tig + 1
  for (int blk = 0; blk < nblk; ++blk) {
    cp_wait<ST - 2>();
    __syncthreads();  // (A) block blk landed for all threads; everybody is done with block blk - 1
    {
      const int nx = blk + ST - 1;
      if (nx < nblk) issue(nx, nx % ST);
      cp_commit();
    }
    const uint32_t sb = ring_u + (blk % ST) * STG;
    const uint32_t kb = sb, vb = sb + KB, ksb = sb + 2 * KB, vsb = sb + 2 * KB + TB * SB;
    const int t0 = c0 + blk * TB;

    uint32_t h0[16], h1[16];
    load_k<F>(kb, ksb, 16 * w + gid, tig, kfac, h0);
    load_k<F>(kb, ksb, 16 * w + gid + 8, tig, kfac, h1);
    float sacc[4] = {0.f, 0.f, 0.f, 0.f};
    wg_fence();
#pragma unroll
    for (int j = 0; j < 8; ++j) wgmma_rs<BF>(sacc, h0[2 * j], h1[2 * j], h0[2 * j + 1], h1[2 * j + 1], qdesc + j * 16);
    wg_commit();
    uint32_t vab[4][4], vcd[4][4];  // [k step][e]: tokens (2tig, 2tig+1) and (2tig+8, 2tig+9) of k step j
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      load_v_wg<F>(vb, vsb, 16 * j + 2 * tig, 16 * j + 2 * tig + 1, w, gid, vfac, vab[j]);
      load_v_wg<F>(vb, vsb, 16 * j + 2 * tig + 8, 16 * j + 2 * tig + 9, w, gid, vfac, vcd[j]);
    }
    wg_wait0(sacc);
    pin16(h0); pin16(h1);
    if constexpr (BF) {  // bf16: q is used unscaled (no extra rounding); softmax scale applied in fp32
      sacc[0] *= p.qscale; sacc[1] *= p.qscale; sacc[2] *= p.qscale; sacc[3] *= p.qscale;
    }
    if constexpr (F == AF_FP8) {
      const float s0 = __uint_as_float(lds32(ksb + (16 * w + gid) * 4));
      const float s1 = __uint_as_float(lds32(ksb + (16 * w + gid + 8) * 4));
      sacc[0] *= s0; sacc[1] *= s0; sacc[2] *= s1; sacc[3] *= s1;
    }
    {
      const int rr = t0 + 16 * w + gid;
      sacc[0] = rr < c1 ? sacc[0] : NEG_INF;
      sacc[1] = rr < c1 ? sacc[1] : NEG_INF;
      sacc[2] = rr + 8 < c1 ? sacc[2] : NEG_INF;
      sacc[3] = rr + 8 < c1 ? sacc[3] : NEG_INF;
    }
    float x0 = fmaxf(sacc[0], sacc[2]), x1 = fmaxf(sacc[1], sacc[3]);
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) {
      x0 = fmaxf(x0, __shfl_xor_sync(FULL, x0, o));
      x1 = fmaxf(x1, __shfl_xor_sync(FULL, x1, o));
    }
    if (gid == 0) { red[w][2 * tig] = x0; red[w][2 * tig + 1] = x1; }
    __syncthreads();  // (B) block max over the 4 warps' tokens
    x0 = fmaxf(fmaxf(red[0][2 * tig], red[1][2 * tig]), fmaxf(red[2][2 * tig], red[3][2 * tig]));
    x1 = fmaxf(fmaxf(red[0][2 * tig + 1], red[1][2 * tig + 1]), fmaxf(red[2][2 * tig + 1], red[3][2 * tig + 1]));
    const float mn0 = fmaxf(m0, x0), mn1 = fmaxf(m1, x1);
    const float ms0 = mn0 == NEG_INF ? 0.f : mn0, ms1 = mn1 == NEG_INF ? 0.f : mn1;
    const float al0 = ex2(m0 - ms0), al1 = ex2(m1 - ms1);
    m0 = mn0; m1 = mn1;
    const float p0 = ex2(sacc[0] - ms0), p1 = ex2(sacc[1] - ms1), p2 = ex2(sacc[2] - ms0), p3 = ex2(sacc[3] - ms1);
    l0 = l0 * al0 + p0 + p2;
    l1 = l1 * al1 + p1 + p3;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      oacc[mt][0] *= al0; oacc[mt][2] *= al0;
      oacc[mt][1] *= al1; oacc[mt][3] *= al1;
    }
    if (tig < 2) {  // P^T operand (padding heads stay 0)
      const float pm = operand_pmul<F>(p);
      float q0 = p0 * pm, q1 = p1 * pm, q2 = p2 * pm, q3 = p3 * pm;
      if constexpr (F == AF_FP8) {  // V scale per token folded into P (operand only)
        const float s0 = __uint_as_float(lds32(vsb + (16 * w + gid) * 4));
        const float s1 = __uint_as_float(lds32(vsb + (16 * w + gid + 8) * 4));
        q0 *= s0; q1 *= s0; q2 *= s1; q3 *= s1;
      }
      const uint32_t pw = pack2<BF>(q0, q1), pz = pack2<BF>(q2, q3);
      uint16_t* P = reinterpret_cast<uint16_t*>(pt);
      P[((2 * w) * 128 + (2 * tig) * 16) / 2 + gid] = static_cast<uint16_t>(pw);
      P[((2 * w) * 128 + (2 * tig + 1) * 16) / 2 + gid] = static_cast<uint16_t>(pw >> 16);
      P[((2 * w + 1) * 128 + (2 * tig) * 16) / 2 + gid] = static_cast<uint16_t>(pz);
      P[((2 * w + 1) * 128 + (2 * tig + 1) * 16) / 2 + gid] = static_cast<uint16_t>(pz >> 16);
    }
    fence_async_smem();
    __syncthreads();  // (C) P^T complete
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
        wgmma_rs<BF>(oacc[mt], vab[j][2 * mt], vab[j][2 * mt + 1], vcd[j][2 * mt], vcd[j][2 * mt + 1], pdesc + j * 16);
    }
    wg_commit();
    wg_wait0(oacc[0]);
    wg_wait0(oacc[1]);
    {
      uint32_t (&va16)[16] = *reinterpret_cast<uint32_t (*)[16]>(&vab[0][0]);
      uint32_t (&vc16)[16] = *reinterpret_cast<uint32_t (*)[16]>(&vcd[0][0]);
      pin16(va16); pin16(vc16);
    }
  }
  cp_wait<0>();

  // ---------------------------------------------------------------- CTA result: d = 32 w + 4 gid + e
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) {
    l0 += __shfl_xor_sync(FULL, l0, o);
    l1 += __shfl_xor_sync(FULL, l1, o);
  }
  __syncthreads();
  float (*stg)[D] = reinterpret_cast<float (*)[D]>(smem);  // [G][D] (Q^T / P^T no longer needed)
  if (tig < 2) {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const int d = 32 * w + 4 * gid + 2 * mt;
      stg[2 * tig][d] = oacc[mt][0];
      stg[2 * tig + 1][d] = oacc[mt][1];
      stg[2 * tig][d + 1] = oacc[mt][2];
      stg[2 * tig + 1][d + 1] = oacc[mt][3];
    }
    if (gid == 0) {
      lred[w][2 * tig] = l0; lred[w][2 * tig + 1] = l1;
      if (w == 0) { mfin[2 * tig] = m0; mfin[2 * tig + 1] = m1; }
    }
  }
  __syncthreads();
  float oc[G], Mc[G], Lc[G];
#pragma unroll
  for (int h = 0; h < G; ++h) {
    oc[h] = stg[h][tid];
    Mc[h] = mfin[h];
    Lc[h] = lred[0][h] + lred[1][h] + lred[2][h] + lred[3][h];
  }
  __syncthreads();  // stg is read before the epilogue reuses smem
  attn_epilogue(p, smem, kvh, split, b, oc, Mc, Lc);
}

template <int F, int ST, bool QR>
int launch_wg(const AttnParams* p, cudaStream_t st, int* occ) {
  constexpr int SM = WgSmem<F, ST>::SMEM;
  if (SM > 200 * 1024) return -2;
  auto k = k_attn_wg<F, ST, QR>;
  static int per_sm = -1;
  if (per_sm < 0) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k, NW * 32, SM);
  }
  if (occ) { *occ = per_sm; return 0; }
  dim3 grid(p->Hkv, p->nsplit, p->B);
  k<<<grid, NW * 32, SM, st>>>(*p);
  return 0;
}

// =====================================================================================================================
// Hybrid variant: QK^T with warpgroup wgmma (64-token blocks, Q^T operand in smem, K fragments from registers),
// P.V with per-warp mma.sync exactly as in k_attn (each warp owns 16 tokens of the block: own online softmax, V
// fragments with 16 d per token per thread).  One barrier per block (cp.async stage landed / previous block done).
template <int F, int ST>
struct HySmem {
  static constexpr int STG = Cfg<F, 4, 1>::STG;  // 64 rows of K, V and scales (same layout as k_attn with NT = 4)
  static constexpr int SMEM = 2048 + ST * STG > 18 * 1024 ? 2048 + ST * STG : 18 * 1024;
};

template <int F, int ST, bool QROT>
__global__ void __launch_bounds__(NW * 32) k_attn_hy(const AttnParams p) {
  constexpr int RB = Fmt<F>::RB, SB = Fmt<F>::SB, TB = 64, KB = TB * RB, STG = HySmem<F, ST>::STG;
  constexpr bool BF = F == AF_BF16;
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ float red_m[NW][G], red_l[NW][G];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, gid = lane >> 2, tig = lane & 3;
  const int kvh = blockIdx.x, split = blockIdx.y, b = blockIdx.z;
  const int c0 = split * p.chunk, c1 = min(p.L, c0 + p.chunk);
  uint8_t* qb = smem;
  uint8_t* ring = smem + 2048;
  const uint32_t qb_u = smem_u32(qb), ring_u = smem_u32(ring);
  {
    float (*qs)[D] = reinterpret_cast<float (*)[D]>(ring);
    const __nv_bfloat16* qp = static_cast<const __nv_bfloat16*>(p.q) + b * p.q_bs + (kvh * G + w) * D + lane * 4;
    const uint2 raw = *reinterpret_cast<const uint2*>(qp);
    float x[4] = {__uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xffff0000u),
                  __uint_as_float(raw.y << 16), __uint_as_float(raw.y & 0xffff0000u)};
    if constexpr (QROT) warp_selrot(x, p.kmask + kvh * (D / p.rb), p.rb, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) qs[w][lane * 4 + i] = (BF ? x[i] : x[i] * (p.qscale * operand_pmul<F>(p)));
    __syncthreads();
    for (int i = tid; i < 8 * 16 * 8; i += NW * 32) {
      const int j = i >> 7, kl = (i >> 3) & 15, n = i & 7;
      const int tg = (kl & 7) >> 1, e = kl & 1, pi = 2 * j + (kl >> 3);
      const int dd = 32 * tg + (e ? pd1<F>(pi) : pd0<F>(pi));
      const float v = n < G ? qs[n][dd] : 0.f;
      uint16_t hv;
      if constexpr (BF) hv = __bfloat16_as_ushort(__float2bfloat16_rn(v));
      else hv = __half_as_ushort(__float2half_rn(v));
      *reinterpret_cast<uint16_t*>(qb + j * 256 + (kl >> 3) * 128 + n * 16 + (kl & 7) * 2) = hv;
    }
    fence_async_smem();
    __syncthreads();
  }
  __half2 kfac = __float2half2_rn(0.f);
  if constexpr (F == AF_NVFP4) kfac = __float2half2_rn(128.0f / __ldg(p.gscale));
  const __half2 vfac = kfac;
  const uint64_t qdesc = sdesc(qb_u);

  const int ntok = max(c1 - c0, 0), nblk = (ntok + TB - 1) / TB;
  const uint8_t* kg = p.k + b * p.bat_stride + kvh * RB;
  const uint8_t* vg = p.v + b * p.bat_stride + kvh * RB;
  const uint8_t* ksg = SB ? p.ks + b * p.sbat_stride + kvh * SB : nullptr;
  const uint8_t* vsg = SB ? p.vs + b * p.sbat_stride + kvh * SB : nullptr;
  const size_t tstr = static_cast<size_t>(p.tok_stride), sstr = static_cast<size_t>(p.stok_stride);
  const int Lm1 = p.L - 1;
  constexpr int UPR = RB / 16, RPI = 128 / UPR, NCP = TB * UPR / 128;
  const int r0 = tid / UPR, u0 = tid % UPR;
  const uint32_t kdst0 = r0 * RB + kswz<F>(r0, u0) * 16;
  const uint32_t vdst0 = KB + vrow<F>(r0) * RB + vswz<F>(r0, u0) * 16;  // depend on r & 7 = r0 & 7 only
  const long long vgap = vg - kg;
  auto issue = [&](int blk, int stg) {
    const uint32_t sb = ring_u + stg * STG;
    const int t0 = c0 + blk * TB;
    const bool full = t0 + TB <= p.L;
#pragma unroll
    for (int i = 0; i < NCP; ++i) {
      const int t = full ? t0 + i * RPI + r0 : min(t0 + i * RPI + r0, Lm1);
      const uint8_t* src = kg + static_cast<size_t>(t) * tstr + u0 * 16;
      cp16(sb + kdst0 + i * RPI * RB, src);
      cp16(sb + vdst0 + i * RPI * RB, src + vgap);
    }
    if constexpr (SB > 0) {
      const int t = full ? t0 + (tid & (TB - 1)) : min(t0 + (tid & (TB - 1)), Lm1);
      cpsmall<SB>(sb + 2 * KB + tid * SB, (tid < TB ? ksg : vsg) + static_cast<size_t>(t) * sstr);
    }
  };
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < nblk) issue(s, s);
    cp_commit();
  }
  float acc[8][4];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
  float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.f, l1 = 0.f;
  for (int blk = 0; blk < nblk; ++blk) {
    cp_wait<ST - 2>();
    __syncthreads();
    {
      const int nx = blk + ST - 1;
      if (nx < nblk) issue(nx, nx % ST);
      cp_commit();
    }
    const uint32_t sb = ring_u + (blk % ST) * STG;
    const uint32_t kb = sb, vb = sb + KB, ksb = sb + 2 * KB, vsb = sb + 2 * KB + TB * SB;
    const int t0 = c0 + blk * TB;
    const int rw = 16 * w;  // this warp's 16 tokens of the block
    uint32_t h0[16], h1[16];
    load_k<F>(kb, ksb, rw + gid, tig, kfac, h0);
    load_k<F>(kb, ksb, rw + gid + 8, tig, kfac, h1);
    float sacc[4] = {0.f, 0.f, 0.f, 0.f};
    wg_fence();
#pragma unroll
    for (int j = 0; j < 8; ++j) wgmma_rs<BF>(sacc, h0[2 * j], h1[2 * j], h0[2 * j + 1], h1[2 * j + 1], qdesc + j * 16);
    wg_commit();
    uint32_t pa[16], pb[16];  // V fragments (independent of S): overlap with the QK wgmma
    load_v<F>(vb, vsb, rw + 2 * tig, rw + 2 * tig + 1, gid, vfac, pa);
    load_v<F>(vb, vsb, rw + 2 * tig + 8, rw + 2 * tig + 9, gid, vfac, pb);
    wg_wait0(sacc);
    pin16(h0); pin16(h1);
    if constexpr (BF) {  // bf16: q is used unscaled (no extra rounding); softmax scale applied in fp32
      sacc[0] *= p.qscale; sacc[1] *= p.qscale; sacc[2] *= p.qscale; sacc[3] *= p.qscale;
    }
    if constexpr (F == AF_FP8) {
      const float s0 = __uint_as_float(lds32(ksb + (rw + gid) * 4));
      const float s1 = __uint_as_float(lds32(ksb + (rw + gid + 8) * 4));
      sacc[0] *= s0; sacc[1] *= s0; sacc[2] *= s1; sacc[3] *= s1;
    }
    {
      const int rr = t0 + rw + gid;
      sacc[0] = rr < c1 ? sacc[0] : NEG_INF;
      sacc[1] = rr < c1 ? sacc[1] : NEG_INF;
      sacc[2] = rr + 8 < c1 ? sacc[2] : NEG_INF;
      sacc[3] = rr + 8 < c1 ? sacc[3] : NEG_INF;
    }
    float x0 = fmaxf(sacc[0], sacc[2]), x1 = fmaxf(sacc[1], sacc[3]);
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) {
      x0 = fmaxf(x0, __shfl_xor_sync(FULL, x0, o));
      x1 = fmaxf(x1, __shfl_xor_sync(FULL, x1, o));
    }
    const float mn0 = fmaxf(m0, x0), mn1 = fmaxf(m1, x1);
    const float ms0 = mn0 == NEG_INF ? 0.f : mn0, ms1 = mn1 == NEG_INF ? 0.f : mn1;
    const float al0 = ex2(m0 - ms0), al1 = ex2(m1 - ms1);
    m0 = mn0; m1 = mn1;
    float q0 = ex2(sacc[0] - ms0), q1 = ex2(sacc[1] - ms1), q2 = ex2(sacc[2] - ms0), q3 = ex2(sacc[3] - ms1);
    l0 = l0 * al0 + q0 + q2;
    l1 = l1 * al1 + q1 + q3;
#pragma unroll
    for (int mt = 0; mt < 8; ++mt) {
      acc[mt][0] *= al0; acc[mt][2] *= al0;
      acc[mt][1] *= al1; acc[mt][3] *= al1;
    }
    if constexpr (F == AF_FP8) {
      const float s0 = __uint_as_float(lds32(vsb + (rw + gid) * 4));
      const float s1 = __uint_as_float(lds32(vsb + (rw + gid + 8) * 4));
      q0 *= s0; q1 *= s0; q2 *= s1; q3 *= s1;
    }
    if constexpr (F == AF_NVFP4 || F == AF_MXFP4) {
      const float pm = operand_pmul<F>(p);
      q0 *= pm; q1 *= pm; q2 *= pm; q3 *= pm;
    }
    const uint32_t b0 = movt(pack2<BF>(q0, q1)), b1 = movt(pack2<BF>(q2, q3));
#pragma unroll
    for (int mt = 0; mt < 8; ++mt) mma<BF>(acc[mt], pa[2 * mt], pa[2 * mt + 1], pb[2 * mt], pb[2 * mt + 1], b0, b1);
  }
  cp_wait<0>();
  // CTA merge of the 4 warps (as in k_attn)
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) {
    l0 += __shfl_xor_sync(FULL, l0, o);
    l1 += __shfl_xor_sync(FULL, l1, o);
  }
  __syncthreads();
  float* ow = reinterpret_cast<float*>(smem);
  if (tig < 2) {
    const int h0i = 2 * tig, h1i = 2 * tig + 1;
#pragma unroll
    for (int mt = 0; mt < 8; ++mt) {
      const int d = 16 * gid + 2 * mt;
      ow[(w * G + h0i) * D + d] = acc[mt][0];
      ow[(w * G + h0i) * D + d + 1] = acc[mt][2];
      ow[(w * G + h1i) * D + d] = acc[mt][1];
      ow[(w * G + h1i) * D + d + 1] = acc[mt][3];
    }
    if (gid == 0) {
      red_m[w][h0i] = m0; red_m[w][h1i] = m1;
      red_l[w][h0i] = l0; red_l[w][h1i] = l1;
    }
  }
  __syncthreads();
  float oc[G], Mc[G], Lc[G];
#pragma unroll
  for (int h = 0; h < G; ++h) {
    float M = NEG_INF;
#pragma unroll
    for (int ww = 0; ww < NW; ++ww) M = fmaxf(M, red_m[ww][h]);
    const float Ms = M == NEG_INF ? 0.f : M;
    float o = 0.f, l = 0.f;
#pragma unroll
    for (int ww = 0; ww < NW; ++ww) {
      const float sc = ex2(red_m[ww][h] - Ms);
      o += sc * ow[(ww * G + h) * D + tid];
      l += sc * red_l[ww][h];
    }
    oc[h] = o; Mc[h] = M; Lc[h] = l;
  }
  attn_epilogue(p, smem, kvh, split, b, oc, Mc, Lc);
}

template <int F, int ST, bool QR>
int launch_hy(const AttnParams* p, cudaStream_t st, int* occ) {
  constexpr int SM = HySmem<F, ST>::SMEM;
  if (SM > 200 * 1024) return -2;
  auto k = k_attn_hy<F, ST, QR>;
  static int per_sm = -1;
  if (per_sm < 0) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k, NW * 32, SM);
  }
  if (occ) { *occ = per_sm; return 0; }
  dim3 grid(p->Hkv, p->nsplit, p->B);
  k<<<grid, NW * 32, SM, st>>>(*p);
  return 0;
}

__global__ void __launch_bounds__(128) k_qrot(const __nv_bfloat16* __restrict__ q, long long q_bs,
                                               __nv_bfloat16* __restrict__ out, int Hkv,
                                               const uint8_t* __restrict__ kmask, int rb) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int b = blockIdx.y, kvh = blockIdx.x;
  const uint2 raw = *reinterpret_cast<const uint2*>(q + b * q_bs + (kvh * G + w) * D + lane * 4);
  float x[4] = {__uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xffff0000u),
                __uint_as_float(raw.y << 16), __uint_as_float(raw.y & 0xffff0000u)};
  warp_selrot(x, kmask + kvh * (D / rb), rb, lane);
  __nv_bfloat162 o01 = __floats2bfloat162_rn(x[0], x[1]), o23 = __floats2bfloat162_rn(x[2], x[3]);
  uint2 ov;
  ov.x = *reinterpret_cast<uint32_t*>(&o01);
  ov.y = *reinterpret_cast<uint32_t*>(&o23);
  *reinterpret_cast<uint2*>(out + (static_cast<long long>(b) * Hkv * G + kvh * G + w) * D + lane * 4) = ov;
}

template <int F, int NT, int ST, bool QR>
int launch_t(const AttnParams* p, cudaStream_t st, int* occ) {
  using C = Cfg<F, NT, ST>;
  if (C::SMEM > 200 * 1024) return -2;
  auto k = k_attn<F, NT, ST, QR>;
  static int per_sm = -1;
  if (per_sm < 0) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k, NW * 32, C::SMEM);
  }
  if (occ) { *occ = per_sm; return 0; }
  dim3 grid(p->Hkv, p->nsplit, p->B);
  k<<<grid, NW * 32, C::SMEM, st>>>(*p);
  return 0;
}

template <int F, bool QR>
int launch_v(int variant, const AttnParams* p, cudaStream_t st, int* occ) {
  switch (variant) {
    case 0: return launch_t<F, 1, 3, QR>(p, st, occ);
    case 1: return launch_t<F, 1, 4, QR>(p, st, occ);
    case 2: return launch_t<F, 2, 3, QR>(p, st, occ);
    case 3: return launch_t<F, 2, 4, QR>(p, st, occ);
    case 4: return launch_t<F, 1, 6, QR>(p, st, occ);
    case 5: return launch_t<F, 2, 6, QR>(p, st, occ);
    case 6: return launch_t<F, 2, 2, QR>(p, st, occ);
    case 7: return launch_t<F, 4, 2, QR>(p, st, occ);
    case 8: return launch_wg<F, 2, QR>(p, st, occ);
    case 9: return launch_wg<F, 3, QR>(p, st, occ);
    case 10: return launch_wg<F, 4, QR>(p, st, occ);
    case 11: return launch_wg<F, 6, QR>(p, st, occ);
    case 12: return launch_hy<F, 2, QR>(p, st, occ);
    case 13: return launch_hy<F, 3, QR>(p, st, occ);
    case 14: return launch_hy<F, 4, QR>(p, st, occ);
    case 15: return launch_hy<F, 6, QR>(p, st, occ);
  }
  return -3;
}

template <bool QR>
int launch_f(int fmt, int variant, const AttnParams* p, cudaStream_t st, int* occ) {
  switch (fmt) {
    case AF_BF16: return launch_v<AF_BF16, QR>(variant, p, st, occ);
    case AF_FP8: return launch_v<AF_FP8, QR>(variant, p, st, occ);
    case AF_NVFP4: return launch_v<AF_NVFP4, QR>(variant, p, st, occ);
    case AF_MXFP4: return launch_v<AF_MXFP4, QR>(variant, p, st, occ);
    case AF_INT4: return launch_v<AF_INT4, QR>(variant, p, st, occ);
  }
  return -4;
}

}  // namespace

int attn_decode_launch(int fmt, bool qrot, int variant, const AttnParams& p, cudaStream_t st) {
  if (p.L <= 0 || p.B <= 0) return 0;
  return qrot ? launch_f<true>(fmt, variant, &p, st, nullptr) : launch_f<false>(fmt, variant, &p, st, nullptr);
}

int attn_occupancy(int fmt, bool qrot, int variant) {
  int occ = 0;
  const int rc = qrot ? launch_f<true>(fmt, variant, nullptr, nullptr, &occ) : launch_f<false>(fmt, variant, nullptr, nullptr, &occ);
  return rc == 0 ? occ : 0;
}

void qrot_launch(const void* q, long long q_bs, void* out, int B, int Hkv, const uint8_t* kmask, int rb,
                 cudaStream_t st) {
  k_qrot<<<dim3(Hkv, B), 128, 0, st>>>(static_cast<const __nv_bfloat16*>(q), q_bs, static_cast<__nv_bfloat16*>(out),
                                       Hkv, kmask, rb);
}
