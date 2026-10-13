// Phase 2f: paged INT4 prefill attention that reads the FlagRot INT4 pages directly (sm_90a, wgmma e4m3).
//
// Design A+C: per 32-channel group the asymmetric codes c - z (|c - z| <= 15) are exact in E4M3
// (byte v = v * 2^-9), q is an E4M3 hi + lo pair in the stored basis (row scale sigma = power of two, folded into the
// softmax factor), one fresh fp32 accumulator per group (QK = q_hi . k + q_lo . k, two k32 wgmmas, the lo one only for
// the groups in lomask), score = sum_g s_g 2^9 acc_g in fp32.  P = 2^8 exp2(x - m) in E4M3 (FA3's Max_offset 8); V =
// s (c - z) rounded to E4M3; PV = wgmma m64n128k32 with P from registers.
//
// CTA = one warpgroup, 64 rows = 16 query tokens x the 4 q heads of one KV head (row r: token r / 4, head r % 4), keys
// in blocks of 64 (4 pages).  Per block: the 4 page chunks (2304 B each, one cp.async.bulk per page, ST-stage ring)
// are converted by the whole warpgroup into E4M3 wgmma tiles (K: [64 keys x 128 ch], V^T: [128 dims x 64 keys], no
// swizzle, K-major core matrices; double buffered) + fp32 K scales; then QK, online softmax, PV.  No gather / dequant
// workspace in global memory.
//
// Page chunk (kernels/csrc v2 INT4 layout, int4ah:32, P = 16): K codes [16 tok][64 B] (byte 16 tg + 4 g + i: low
// nibble = channel 32 g + 4 tg + i, high = + 16) | V^T codes 1024 B ((w 4, gid 8, c8 2, sub 4, i 4): low / high nibble
// = token 8 c8 + 2 i (+1), dim 64 (sub >> 1) + 16 w + gid + 8 (sub & 1)) | K scale words [16 tok][4 g] | V scale
// words [4 g][16 slots] (slot q holds token 8 (q / 8) + perm8[q % 8], perm8 = 0 2 4 6 1 3 5 7).  Scale word: FP16
// with |word| = scale, zero point = word & 15 (konlya: the sign bits carry transform flags, masked here).
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#define DEVI __device__ __forceinline__

namespace {

constexpr int D = 128, PAGE = 16, CHUNK = 144 * PAGE, BN = 64, PPB = BN / PAGE, TOK = 16, M = 64;
constexpr float NEG_INF = -__builtin_huge_valf();
constexpr float POFF = 8.f;  // P = 2^(x - m + 8)

struct Params {
  const uint8_t* qh;   // [nq, Hq, 128] e4m3 (stored basis)
  const uint8_t* ql;
  const float* qfac;   // [nq, Hq]: softmax scale * log2(e) / sigma
  long long q_ts;      // token stride of qh / ql (bytes) and qfac (elements: q_ts / 128)
  const uint8_t* pool;
  long long page_stride;  // bytes
  const int* bt;
  int nq, nk, p0, shift, Hkv;
  const uint8_t* lomask;  // [Hkv] bit g: group g gets the lo wgmma (null: all)
  __nv_bfloat16* out;     // [nq, Hq, 128]
  long long o_ts;         // elements
  float* lse;             // [Hq, nq] natural log
};

DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
DEVI uint32_t lds32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(a));
  return v;
}
DEVI uint32_t lds16(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u16 %0, [%1];\n" : "=r"(v) : "r"(a));
  return v;
}
DEVI uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
DEVI float2 lds64f(uint32_t a) {
  float2 v;
  asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];\n" : "=f"(v.x), "=f"(v.y) : "r"(a));
  return v;
}
DEVI void sts128(uint32_t a, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(a), "r"(x), "r"(y), "r"(z), "r"(w));
}
DEVI void stsf(uint32_t a, float v) { asm volatile("st.shared.f32 [%0], %1;\n" ::"r"(a), "f"(v)); }
DEVI void mbar_init(uint32_t a, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(a), "r"(n)); }
DEVI void mbar_expect_tx(uint32_t a, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(a), "r"(bytes) : "memory");
}
DEVI void mbar_wait(uint32_t a, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred P1;\n WAIT_%=:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      " @!P1 bra WAIT_%=;\n}\n" ::"r"(a), "r"(parity) : "memory");
}
DEVI void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t mbar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(mbar) : "memory");
}
DEVI void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}
DEVI void cp_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }
DEVI void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
DEVI void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
DEVI void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
DEVI void wg_wait_all() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
template <int N>
DEVI void acc_sync(float (&d)[N]) {  // the accumulators are written by the wgmma just waited for
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(d[i])::"memory");
}
DEVI uint64_t sdesc(uint32_t saddr) {  // no swizzle, K-major core matrices: LBO (K) = 128 B, SBO (M / N) = 256 B
  return static_cast<uint64_t>((saddr & 0x3FFFF) >> 4) | (static_cast<uint64_t>(128 >> 4) << 16) |
         (static_cast<uint64_t>(256 >> 4) << 32);
}
DEVI float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
DEVI uint32_t cvt_e4m3x2(float lo, float hi) {  // byte 0 = lo, byte 1 = hi (32-bit output: ptxas 12.9 "=h" bug)
  uint32_t r;
  asm("{\n .reg .b16 t;\n cvt.rn.satfinite.e4m3x2.f32 t, %1, %2;\n cvt.u32.u16 %0, t;\n}\n" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
DEVI uint32_t hsub2_u(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("sub.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
DEVI uint32_t hmul2_u(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
DEVI uint32_t cvt_e4m3x2_h2(uint32_t a) {  // f16x2 -> E4M3x2 (low half -> low byte), 32-bit result
  uint32_t r;
  asm("{\n .reg .b16 t;\n cvt.rn.satfinite.e4m3x2.f16x2 t, %1;\n cvt.u32.u16 %0, t;\n}\n" : "=r"(r) : "r"(a));
  return r;
}
DEVI uint32_t pack4(float a, float b, float c, float d) { return cvt_e4m3x2(a, b) | (cvt_e4m3x2(c, d) << 16); }
// E4M3 bytes of (x - z) for the 4 unsigned 4-bit codes in the bytes of x (|x - z| <= 15: byte = sign | magnitude)
DEVI uint32_t enc_diff(uint32_t x, uint32_t zz) {
  return __vabsdiffu4(x, zz) | (__vcmpltu4(x, zz) & 0x80808080u);
}
DEVI float half_bits(uint32_t w) { return __half2float(__ushort_as_half(static_cast<unsigned short>(w & 0x7FFF))); }

#include "int4_prefill_wgmma.h"

constexpr int SQ = 0;                 // q hi [4 g][64 rows][32 B] then q lo: 8192 B each
constexpr int SK = 16384;             // K tiles x2: [4 g][64 keys][32 B]
constexpr int SV = SK + 2 * 8192;     // V^T tiles x2: [2 kk][128 dims][32 B]
constexpr int SS = SV + 2 * 8192;     // K scales x2: [4 g][64 keys] fp32 (2^9 s, 0 for invalid keys)
constexpr int SR = SS + 2 * 1024;     // raw page ring: ST x 4 x 2304 B
template <int ST>
constexpr int smem_bytes() { return SR + ST * PPB * CHUNK + ST * 8; }

// q tile address of (row, byte k in 0..127) in the [g][rows/8][kchunk][8][16] layout
DEVI uint32_t tile_off(int row, int g, int kc) { return g * 2048 + (row >> 3) * 256 + kc * 128 + (row & 7) * 16; }

template <int ST, bool DOK = true>
DEVI void convert_block(uint32_t raw, uint32_t sk, uint32_t sv, uint32_t ss, int nvalid, int tid) {
  // ---- K: 2 (key, group) items per thread; tile byte order = natural channel order of the group
#pragma unroll
  for (int it = 0; it < (DOK ? 2 : 0); ++it) {
    const int item = tid + 128 * it;
    const int key = item >> 2, g = item & 3;
    const uint32_t tb = raw + (key >> 4) * CHUNK + (key & 15) * 64 + 4 * g;
    const uint32_t sw = lds16(raw + (key >> 4) * CHUNK + 2048 + (key & 15) * 8 + 2 * g);
    const uint32_t zz = (sw & 15u) * 0x01010101u;
    uint32_t w[4];
#pragma unroll
    for (int tg = 0; tg < 4; ++tg) w[tg] = lds32(tb + 16 * tg);
    uint32_t lo[4], hi[4];
#pragma unroll
    for (int tg = 0; tg < 4; ++tg) {
      lo[tg] = enc_diff(w[tg] & 0x0F0F0F0Fu, zz);
      hi[tg] = enc_diff((w[tg] >> 4) & 0x0F0F0F0Fu, zz);
    }
    const uint32_t dst = sk + tile_off(key, g, 0);
    sts128(dst, lo[0], lo[1], lo[2], lo[3]);
    sts128(dst + 128, hi[0], hi[1], hi[2], hi[3]);
    stsf(ss + (g * 64 + key) * 4, key < nvalid ? half_bits(sw) * 512.f : 0.f);
  }
  // ---- V^T: thread -> (page p, group g) = tid / 8, 4 dims 32 g + 4 u + e
  {
    const int pg = tid >> 3, u = tid & 7;
    const int p = pg >> 2, g = pg & 3;
    const uint32_t pr = raw + p * CHUNK;
    // scale words of the 16 slots of group g (slot q holds token 8 (q / 8) + perm8[q % 8]); packed FP16 math:
    // pair A_t = tokens (2t, 2t+1) = slots (t, t+4), pair B_t = tokens (8+2t, 9+2t) = slots (t+8, t+12);
    // half(0x6400 | c) = 1024 + c, minus half(0x6400 | z) = c - z exactly, times the FP16 scale, -> E4M3x2
    const uint4 w0 = lds128(pr + 2176 + g * 32), w1 = lds128(pr + 2176 + g * 32 + 16);
    const uint32_t ws[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
    uint32_t s2[8], z2[8];
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const uint32_t sel = (t & 1) ? 0x7632u : 0x5410u;
      const uint32_t pa = __byte_perm(ws[t >> 1], ws[(t >> 1) + 2], sel);
      const uint32_t pb = __byte_perm(ws[(t >> 1) + 4], ws[(t >> 1) + 6], sel);
      s2[t] = pa & 0x7FFF7FFFu;
      s2[4 + t] = pb & 0x7FFF7FFFu;
      z2[t] = 0x64006400u | (pa & 0x000F000Fu);
      z2[4 + t] = 0x64006400u | (pb & 0x000F000Fu);
    }
    if (16 * p + 16 > nvalid) {        // tokens past the valid keys: scale 0 (their pages may hold garbage)
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int base = 16 * p;
        s2[t] &= ((base + 2 * t) < nvalid ? 0xFFFFu : 0u) | ((base + 2 * t + 1) < nvalid ? 0xFFFF0000u : 0u);
        s2[4 + t] &= ((base + 8 + 2 * t) < nvalid ? 0xFFFFu : 0u) | ((base + 9 + 2 * t) < nvalid ? 0xFFFF0000u : 0u);
      }
    }
    const int kk = p >> 1, h = p & 1;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int d = 32 * g + 4 * u + e;
      const int wv = (d & 63) >> 4, gid = d & 7, sub = 2 * (d >> 6) + ((d & 15) >> 3);
      const uint32_t b0 = lds32(pr + 1024 + wv * 256 + gid * 32 + sub * 4);       // tokens 0..7
      const uint32_t b1 = lds32(pr + 1024 + wv * 256 + gid * 32 + 16 + sub * 4);  // tokens 8..15
      const uint32_t l0 = b0 & 0x0F0F0F0Fu, h0 = (b0 >> 4) & 0x0F0F0F0Fu;
      const uint32_t l1 = b1 & 0x0F0F0F0Fu, h1 = (b1 >> 4) & 0x0F0F0F0Fu;
      uint32_t o[4];
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        // kslots 4t .. 4t+3 = tokens 2t, 2t+1 (byte t of b0: low, high nibble), 8+2t, 9+2t (byte t of b1)
        const uint32_t sel = static_cast<uint32_t>(t | (t << 4) | ((4 + t) << 8) | ((4 + t) << 12));
        const uint32_t xa = (__byte_perm(l0, h0, sel) & 0x000F000Fu) | 0x64006400u;
        const uint32_t xb = (__byte_perm(l1, h1, sel) & 0x000F000Fu) | 0x64006400u;
        o[t] = cvt_e4m3x2_h2(hmul2_u(hsub2_u(xa, z2[t]), s2[t])) |
               (cvt_e4m3x2_h2(hmul2_u(hsub2_u(xb, z2[4 + t]), s2[4 + t])) << 16);
      }
      sts128(sv + kk * 4096 + (d >> 3) * 256 + h * 128 + (d & 7) * 16, o[0], o[1], o[2], o[3]);
    }
  }
}

template <int ST>
__global__ void __launch_bounds__(128) k_int4_prefill(const Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sb = smem_u32(smem);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ntile = (p.nq + TOK - 1) / TOK;
  const int tile = ntile - 1 - static_cast<int>(blockIdx.x);   // heaviest (latest) tiles first
  const int h = blockIdx.y;
  const int Hq = p.Hkv * 4;
  const int t0 = tile * TOK;
  const uint32_t mbar = sb + SR + ST * PPB * CHUNK;

  // rows held by this thread (wgmma accumulator layout): r0 = 16 warp + lane / 4, r1 = r0 + 8
  const int r0 = 16 * warp + (lane >> 2), r1 = r0 + 8;
  const int tq0 = t0 + (r0 >> 2), tq1 = t0 + (r1 >> 2);
  const int hq0 = 4 * h + (r0 & 3), hq1 = 4 * h + (r1 & 3);
  const bool v0 = tq0 < p.nq, v1 = tq1 < p.nq;
  const int pos0 = v0 ? p.p0 + tq0 : -(1 << 30), pos1 = v1 ? p.p0 + tq1 : -(1 << 30);
  const long long fts = p.q_ts / D;
  const float fac0 = v0 ? p.qfac[tq0 * fts + hq0] : 0.f, fac1 = v1 ? p.qfac[tq1 * fts + hq1] : 0.f;

  const int tlast = min(p.nq, t0 + TOK) - 1;
  const int kend = min(p.nk, p.p0 + tlast - p.shift + 1);
  const int kfull = max(0, min(p.nk, p.p0 + t0 - p.shift + 1));
  const int nblk = kend > 0 ? (kend + BN - 1) / BN : 0;
  const int npages = kend > 0 ? (kend + PAGE - 1) / PAGE : 0;
  const unsigned lom = p.lomask ? p.lomask[h] : 15u;

  if (tid == 0) {
    for (int s = 0; s < ST; ++s) mbar_init(mbar + 8 * s, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  auto issue = [&](int n) {  // raw pages of block n -> ring slot n % ST (thread 0)
    const int s = n % ST;
    const int pa = n * PPB, pb = min(npages, pa + PPB);
    const uint32_t dst = sb + SR + s * PPB * CHUNK;
    mbar_expect_tx(mbar + 8 * s, (pb - pa) * CHUNK);
    for (int pg = pa; pg < pb; ++pg) {
      const long long page = p.bt[pg];
      bulk_g2s(dst + (pg - pa) * CHUNK, p.pool + page * p.page_stride + static_cast<long long>(h) * CHUNK, CHUNK,
               mbar + 8 * s);
    }
  };
  if (tid == 0)
    for (int n = 0; n < min(ST, nblk); ++n) issue(n);
  // q tiles (hi, lo): row r = (token r / 4, head r % 4): 8 chunks of 16 B per row and part
#pragma unroll
  for (int it = 0; it < 8; ++it) {
    const int idx = tid + 128 * it;           // 0 .. 1023
    const int part = idx >> 9, row = (idx >> 3) & 63, c = idx & 7;
    const int tq = t0 + (row >> 2);
    const uint32_t dst = sb + SQ + part * 8192 + tile_off(row, c >> 1, c & 1);
    if (tq < p.nq) {
      const uint8_t* src = (part ? p.ql : p.qh) + tq * p.q_ts + (4 * h + (row & 3)) * D + 16 * c;
      cp16(dst, src);
    } else {
      sts128(dst, 0, 0, 0, 0);
    }
  }
  cp_wait_all();

  float o[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) o[i] = 0.f;
  float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.f, l1 = 0.f;
  const int tq4 = lane & 3;

  for (int n = 0; n < nblk; ++n) {
    const int s = n % ST, b = n & 1;
    const uint32_t raw = sb + SR + s * PPB * CHUNK;
    const uint32_t sk = sb + SK + b * 8192, sv = sb + SV + b * 8192, ss = sb + SS + b * 1024;
    mbar_wait(mbar + 8 * s, (n / ST) & 1);
    convert_block<ST>(raw, sk, sv, ss, kend - n * BN, tid);
    fence_async_smem();
    __syncthreads();
    if (tid == 0 && n + ST < nblk) {
      fence_async_smem();
      issue(n + ST);
    }
    // ---- QK: two groups per round, fresh accumulators, fp32 scale-and-sum
    float sacc[32];
#pragma unroll
    for (int rnd = 0; rnd < 2; ++rnd) {
      float a0[32], a1[32];
      const int g0 = 2 * rnd, g1 = 2 * rnd + 1;
      wg_fence();
      wg_ss_n64_0(a0, sdesc(sb + SQ + g0 * 2048), sdesc(sk + g0 * 2048));
      if (lom & (1u << g0)) wg_ss_n64_1(a0, sdesc(sb + SQ + 8192 + g0 * 2048), sdesc(sk + g0 * 2048));
      wg_ss_n64_0(a1, sdesc(sb + SQ + g1 * 2048), sdesc(sk + g1 * 2048));
      if (lom & (1u << g1)) wg_ss_n64_1(a1, sdesc(sb + SQ + 8192 + g1 * 2048), sdesc(sk + g1 * 2048));
      wg_commit();
      wg_wait_all();
      acc_sync(a0);
      acc_sync(a1);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float2 s0 = lds64f(ss + (g0 * 64 + 8 * j + 2 * tq4) * 4);
        const float2 s1 = lds64f(ss + (g1 * 64 + 8 * j + 2 * tq4) * 4);
        if (rnd == 0) {
          sacc[4 * j + 0] = a0[4 * j + 0] * s0.x + a1[4 * j + 0] * s1.x;
          sacc[4 * j + 1] = a0[4 * j + 1] * s0.y + a1[4 * j + 1] * s1.y;
          sacc[4 * j + 2] = a0[4 * j + 2] * s0.x + a1[4 * j + 2] * s1.x;
          sacc[4 * j + 3] = a0[4 * j + 3] * s0.y + a1[4 * j + 3] * s1.y;
        } else {
          sacc[4 * j + 0] += a0[4 * j + 0] * s0.x + a1[4 * j + 0] * s1.x;
          sacc[4 * j + 1] += a0[4 * j + 1] * s0.y + a1[4 * j + 1] * s1.y;
          sacc[4 * j + 2] += a0[4 * j + 2] * s0.x + a1[4 * j + 2] * s1.x;
          sacc[4 * j + 3] += a0[4 * j + 3] * s0.y + a1[4 * j + 3] * s1.y;
        }
      }
    }
    // ---- online softmax (log2 units), causal / length mask on the boundary blocks
    const int n0 = n * BN;
    const bool need_mask = n0 + BN > kfull;
    float mx0 = NEG_INF, mx1 = NEG_INF;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int key = n0 + 8 * j + 2 * tq4 + e;
        float x0 = sacc[4 * j + e] * fac0, x1 = sacc[4 * j + 2 + e] * fac1;
        if (need_mask) {
          if (key > pos0 - p.shift || key >= p.nk) x0 = NEG_INF;
          if (key > pos1 - p.shift || key >= p.nk) x1 = NEG_INF;
        }
        sacc[4 * j + e] = x0;
        sacc[4 * j + 2 + e] = x1;
        mx0 = fmaxf(mx0, x0);
        mx1 = fmaxf(mx1, x1);
      }
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
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
        const float pa = ex2(sacc[4 * j + e] + sh0), pb = ex2(sacc[4 * j + 2 + e] + sh1);
        sacc[4 * j + e] = pa;
        sacc[4 * j + 2 + e] = pb;
        ps0 += pa;
        ps1 += pb;
      }
    }
    l0 = l0 * al0 + ps0;
    l1 = l1 * al1 + ps1;
    // ---- P -> E4M3 A fragments (k slot 4t+i <- key 2t+i | 8+2t+i-2, +16; the V^T tile uses the same order)
    uint32_t pa[2][4];
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      const int j = 4 * kk;
      pa[kk][0] = pack4(sacc[4 * j + 0], sacc[4 * j + 1], sacc[4 * (j + 1) + 0], sacc[4 * (j + 1) + 1]);
      pa[kk][1] = pack4(sacc[4 * j + 2], sacc[4 * j + 3], sacc[4 * (j + 1) + 2], sacc[4 * (j + 1) + 3]);
      pa[kk][2] = pack4(sacc[4 * (j + 2) + 0], sacc[4 * (j + 2) + 1], sacc[4 * (j + 3) + 0], sacc[4 * (j + 3) + 1]);
      pa[kk][3] = pack4(sacc[4 * (j + 2) + 2], sacc[4 * (j + 2) + 3], sacc[4 * (j + 3) + 2], sacc[4 * (j + 3) + 3]);
    }
    // PV into a fresh accumulator, promoted to the fp32 output in registers (FP8 wgmma accumulation keeps only ~14
    // bits: chaining it over a long context grows the error,
    float ot[64];
    wg_fence();
    wg_rs_n128_0(ot, pa[0], sdesc(sv));
    wg_rs_n128_1(ot, pa[1], sdesc(sv + 4096));
    wg_commit();
    wg_wait_all();
    acc_sync(ot);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      o[4 * j + 0] = fmaf(o[4 * j + 0], al0, ot[4 * j + 0]);
      o[4 * j + 1] = fmaf(o[4 * j + 1], al0, ot[4 * j + 1]);
      o[4 * j + 2] = fmaf(o[4 * j + 2], al1, ot[4 * j + 2]);
      o[4 * j + 3] = fmaf(o[4 * j + 3], al1, ot[4 * j + 3]);
    }
  }
  // ---- epilogue
  l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
  l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
  if (v0) {
    __nv_bfloat16* dst = p.out + tq0 * p.o_ts + hq0 * D;
#pragma unroll
    for (int j = 0; j < 16; ++j)
      *reinterpret_cast<__nv_bfloat162*>(dst + 8 * j + 2 * tq4) = __floats2bfloat162_rn(o[4 * j] * i0, o[4 * j + 1] * i0);
    if (tq4 == 0) p.lse[static_cast<long long>(hq0) * p.nq + tq0] = l0 > 0.f ? (m0 + __log2f(l0) - POFF) * 0.6931471805599453f : NEG_INF;
  }
  if (v1) {
    __nv_bfloat16* dst = p.out + tq1 * p.o_ts + hq1 * D;
#pragma unroll
    for (int j = 0; j < 16; ++j)
      *reinterpret_cast<__nv_bfloat162*>(dst + 8 * j + 2 * tq4) =
          __floats2bfloat162_rn(o[4 * j + 2] * i1, o[4 * j + 3] * i1);
    if (tq4 == 0) p.lse[static_cast<long long>(hq1) * p.nq + tq1] = l1 > 0.f ? (m1 + __log2f(l1) - POFF) * 0.6931471805599453f : NEG_INF;
  }
}

// ---------------------------------------------------------------------------------------------- v2: warp-specialized
// CTA = 3 warpgroups (384 threads): WG0 = producer (bulk copies of the page chunks + conversion into E4M3 tiles),
// WG1 / WG2 = consumers, 64 rows each (16 query tokens x 4 heads; the CTA covers 32 tokens).  Tiles are handed over
// through mbarriers: raw_full (bulk copy -> producer), tile_full (producer -> consumers), tile_empty (8 consumer warps ->
// producer).  NB converted-tile stages, SRW raw stages.
constexpr int W2_TOK = 32;
constexpr int W2_SQ = 0;                       // [2 wg][hi | lo][8192]
constexpr int W2_TILE = 8192 + 8192 + 1024;    // K tile | V^T tile | K scales
constexpr int W2_ST = 32768;
template <int NB, int SRW>
struct W2 {
  static constexpr int RAW = W2_ST + NB * W2_TILE;
  static constexpr int BAR = RAW + SRW * PPB * CHUNK;     // raw_full[SRW], tile_full[NB], tile_empty[NB]
  static constexpr int BYTES = BAR + 8 * (SRW + 2 * NB);
};
DEVI void mbar_arrive(uint32_t a) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(a) : "memory"); }
DEVI void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
DEVI void named_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
// consumer ping-pong (FA3 style): the two consumer warpgroups take turns issuing their wgmma (QK, then PV), so that one
// warpgroup's softmax / promotion runs while the other's GEMMs occupy the tensor cores.  Barrier 5 + c = "c's turn".
#ifndef FRP_PP
#define FRP_PP 0
#endif
#define PP_TURN()  do { if (FRP_PP) named_bar(5 + c, 256); } while (0)
#define PP_PASS()  do { if (FRP_PP) named_arrive(6 - c, 256); } while (0)

template <int NB, int SRW, int QG, int PREG, int CREG>
__global__ void __launch_bounds__(384, 1) k_int4_prefill_ws(const Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  using L = W2<NB, SRW>;
  const uint32_t sb = smem_u32(smem);
  const int tid = threadIdx.x, wg = tid >> 7, wt = tid & 127;
  const int ntile = (p.nq + W2_TOK - 1) / W2_TOK;
  const int tile = ntile - 1 - static_cast<int>(blockIdx.x);
  const int h = blockIdx.y;
  const int t0 = tile * W2_TOK;
  const uint32_t raw_full = sb + L::BAR, tile_full = raw_full + 8 * SRW, tile_empty = tile_full + 8 * NB;
  // CTA key range (the later consumer's tokens)
  const int tlast = min(p.nq, t0 + W2_TOK) - 1;
  const int kend = min(p.nk, p.p0 + tlast - p.shift + 1);
  const int nblk = kend > 0 ? (kend + BN - 1) / BN : 0;
  const int npages = kend > 0 ? (kend + PAGE - 1) / PAGE : 0;
  if (tid == 0) {
    for (int s = 0; s < SRW; ++s) mbar_init(raw_full + 8 * s, 1);
    for (int s = 0; s < NB; ++s) {
      mbar_init(tile_full + 8 * s, 1);
      mbar_init(tile_empty + 8 * s, 8);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (wg == 0) {
    // register reallocation (PREG > 0): the producer needs few, the consumers hold the accumulators
    if constexpr (PREG > 0) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(PREG));
    // ================================================================ producer
    auto issue = [&](int n) {
      const int s = n % SRW;
      const int pa = n * PPB, pb = min(npages, pa + PPB);
      const uint32_t dst = sb + L::RAW + s * PPB * CHUNK;
      mbar_expect_tx(raw_full + 8 * s, (pb - pa) * CHUNK);
      for (int pg = pa; pg < pb; ++pg) {
        const long long page = p.bt[pg];
        bulk_g2s(dst + (pg - pa) * CHUNK, p.pool + page * p.page_stride + static_cast<long long>(h) * CHUNK, CHUNK,
                 raw_full + 8 * s);
      }
    };
    if (wt == 0)
      for (int n = 0; n < min(SRW, nblk); ++n) issue(n);
    for (int n = 0; n < nblk; ++n) {
      const int rs = n % SRW, tb = n % NB;
      mbar_wait(raw_full + 8 * rs, (n / SRW) & 1);
      if (n >= NB) mbar_wait(tile_empty + 8 * tb, ((n / NB) - 1) & 1);
      const uint32_t tbase = sb + W2_ST + tb * W2_TILE;
      convert_block<2>(sb + L::RAW + rs * PPB * CHUNK, tbase, tbase + 8192, tbase + 16384, kend - n * BN, wt);
      fence_async_smem();
      named_bar(1, 128);
      if (wt == 0) {
        mbar_arrive(tile_full + 8 * tb);
        if (n + SRW < nblk) {
          fence_async_smem();
          issue(n + SRW);
        }
      }
    }
    return;
  }
  // ================================================================ consumers
  if constexpr (PREG > 0) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(CREG));
  const int c = wg - 1;
  const int warp = wt >> 5, lane = wt & 31;
  const int ct0 = t0 + 16 * c;                     // first token of this consumer
  const int r0 = 16 * warp + (lane >> 2), r1 = r0 + 8;
  const int tq0 = ct0 + (r0 >> 2), tq1 = ct0 + (r1 >> 2);
  const int hq0 = 4 * h + (r0 & 3), hq1 = 4 * h + (r1 & 3);
  const bool v0 = tq0 < p.nq, v1 = tq1 < p.nq;
  const int pos0 = v0 ? p.p0 + tq0 : -(1 << 30), pos1 = v1 ? p.p0 + tq1 : -(1 << 30);
  const long long fts = p.q_ts / D;
  const float fac0 = v0 ? p.qfac[tq0 * fts + hq0] : 0.f, fac1 = v1 ? p.qfac[tq1 * fts + hq1] : 0.f;
  const int clast = min(p.nq, ct0 + 16) - 1;
  const int ckend = ct0 < p.nq ? min(p.nk, p.p0 + clast - p.shift + 1) : 0;
  const int cnblk = ckend > 0 ? (ckend + BN - 1) / BN : 0;
  const int kfull = max(0, min(p.nk, p.p0 + ct0 - p.shift + 1));
  const unsigned lom = p.lomask ? p.lomask[h] : 15u;
  const uint32_t sq = sb + W2_SQ + c * 16384;
#pragma unroll
  for (int it = 0; it < 8; ++it) {
    const int idx = wt + 128 * it;
    const int part = idx >> 9, row = (idx >> 3) & 63, cc = idx & 7;
    const int tq = ct0 + (row >> 2);
    const uint32_t dst = sq + part * 8192 + tile_off(row, cc >> 1, cc & 1);
    if (tq < p.nq) {
      const uint8_t* src = (part ? p.ql : p.qh) + tq * p.q_ts + (4 * h + (row & 3)) * D + 16 * cc;
      cp16(dst, src);
    } else {
      sts128(dst, 0, 0, 0, 0);
    }
  }
  cp_wait_all();
  fence_async_smem();
  named_bar(2 + c, 128);
  if (FRP_PP && c == 1) named_arrive(5, 256);  // consumer 0 issues first

  float o[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) o[i] = 0.f;
  float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.f, l1 = 0.f;
  const int tq4 = lane & 3;
  for (int n = 0; n < nblk; ++n) {
    const int tb = n % NB;
    mbar_wait(tile_full + 8 * tb, (n / NB) & 1);
    if (n < cnblk) {
      const uint32_t sk = sb + W2_ST + tb * W2_TILE, sv = sk + 8192, ss = sk + 16384;
      float sacc[32];
#pragma unroll
      for (int rnd = 0; rnd < 4 / QG; ++rnd) {
        float a[QG][32];
        if (rnd == 0) PP_TURN();
        wg_fence();
#pragma unroll
        for (int u = 0; u < QG; ++u) {
          const int g = rnd * QG + u;
          wg_ss_n64_0(a[u], sdesc(sq + g * 2048), sdesc(sk + g * 2048));
          if (lom & (1u << g)) wg_ss_n64_1(a[u], sdesc(sq + 8192 + g * 2048), sdesc(sk + g * 2048));
        }
        wg_commit();
        if (rnd == 4 / QG - 1) PP_PASS();
        wg_wait_all();
#pragma unroll
        for (int u = 0; u < QG; ++u) acc_sync(a[u]);
#pragma unroll
        for (int u = 0; u < QG; ++u) {
          const int g = rnd * QG + u;
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const float2 s2 = lds64f(ss + (g * 64 + 8 * j + 2 * tq4) * 4);
            if (g == 0) {
              sacc[4 * j + 0] = a[u][4 * j + 0] * s2.x;
              sacc[4 * j + 1] = a[u][4 * j + 1] * s2.y;
              sacc[4 * j + 2] = a[u][4 * j + 2] * s2.x;
              sacc[4 * j + 3] = a[u][4 * j + 3] * s2.y;
            } else {
              sacc[4 * j + 0] = fmaf(a[u][4 * j + 0], s2.x, sacc[4 * j + 0]);
              sacc[4 * j + 1] = fmaf(a[u][4 * j + 1], s2.y, sacc[4 * j + 1]);
              sacc[4 * j + 2] = fmaf(a[u][4 * j + 2], s2.x, sacc[4 * j + 2]);
              sacc[4 * j + 3] = fmaf(a[u][4 * j + 3], s2.y, sacc[4 * j + 3]);
            }
          }
        }
      }
#include "int4_prefill_softmax_pv.inc"
    } else {
      PP_TURN();
      PP_PASS();
      PP_TURN();
      PP_PASS();
      if (lane == 0) mbar_arrive(tile_empty + 8 * tb);
    }
  }
  if (FRP_PP && c == 0) named_bar(5, 256);   // consume consumer 1's last hand-over
#include "int4_prefill_epilogue.inc"
}

// ---------------------------------------------------------------------------------------------- v3: FP16 QK
// The "inverse-transform the K tile, then round" design (needed by FlagRot's per-token key transforms, usable for the
// static modes too): the producer dequantizes K in fp32 ((c - z) s), applies the inverse FlagRot transform of every key
// (static: per-head block H32 flags, then whole-head H128 / 8; konlya: per key either H128 (orthonormal, flag F = sign
// of the token's group-0 V scale word) or per 32-group H32 (flag = sign of the group's K scale word)) in packed FP16
// (exact on the integer codes; one rounding at the scale), into a [64 keys x 128] FP16 tile (11-bit mantissa: finer
// than the BF16 path's K); the consumers run QK as 8 wgmma m64n64k16 f16 with the original q converted to FP16 (exact
// for bf16 values in the fp16 range; single accumulator); softmax / PV / V exactly as v2 (FP8 PV, V rounded to E4M3).
struct KParams {
  const __nv_bfloat16* q;     // [nq, Hq, 128] bf16 original basis
  long long q_ts;             // elements
  float qscale;               // softmax scale * log2(e)
  const uint8_t* kmask;       // static: [Hkv, 4] block flags (null: none)
  const uint8_t* kfull;       // static: [Hkv] whole-head flags (null: none)
  int konly;                  // 1: per-token flags in the scale sign bits (key-only 2x2 codebook)
  int qf16;                   // 1: q is already FP16 in the keys' stored basis (prep_q f16 mode): no conversion
};
constexpr int W3_TILE = 16384 + 8192;           // K bf16 tile | V^T e4m3 tile
template <int NB, int SRW>
struct W3 {
  static constexpr int ST = 32768;
  static constexpr int RAW = ST + NB * W3_TILE;
  static constexpr int BAR = RAW + SRW * PPB * CHUNK;
  static constexpr int BYTES = BAR + 8 * (SRW + 2 * NB);
};
DEVI uint32_t pack_bf16(float a, float b) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&v);
}
template <int N>
DEVI void fwht_reg(float* x) {  // unnormalized in-register FWHT of N values
#pragma unroll
  for (int m = 1; m < N; m <<= 1)
#pragma unroll
    for (int i = 0; i < N; ++i)
      if ((i & m) == 0) {
        const float a = x[i], b = x[i + m];
        x[i] = a + b;
        x[i + m] = a - b;
      }
}
DEVI uint32_t hadd2_u(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("add.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
DEVI uint32_t h2_dup(float x) {  // fp32 -> fp16 in both halves
  const unsigned short u = __half_as_ushort(__float2half_rn(x));
  return static_cast<uint32_t>(u) * 0x10001u;
}
// in-register FWHT of 32 channels held as 16 f16x2 registers r[j] = (channel 2j, 2j+1), unnormalized
DEVI void fwht32_h2(uint32_t (&r)[16]) {
#pragma unroll
  for (int j = 0; j < 16; ++j) {          // channel offset 1: inside the register
    const uint32_t sw = __byte_perm(r[j], r[j], 0x1032);
    r[j] = __byte_perm(hadd2_u(r[j], sw), hsub2_u(r[j], sw), 0x5410);
  }
#pragma unroll
  for (int m = 1; m < 16; m <<= 1)       // channel offsets 2, 4, 8, 16
#pragma unroll
    for (int j = 0; j < 16; ++j)
      if ((j & m) == 0) {
        const uint32_t a = r[j], b = r[j + m];
        r[j] = hadd2_u(a, b);
        r[j + m] = hsub2_u(a, b);
      }
}
// K of one key, half of its channels (groups 2 half, 2 half + 1) -> FP16 tile in the ORIGINAL basis.
// Packed FP16: y = c - z exact (half(0x6400 | c) - half(0x6400 | z)), H32 of y exact (|.| <= 480), one rounding when
// the FP16 scale is applied (and the few roundings of the 2-stage H4 for whole-head keys).
//   block (H32) key group:  s H32(y) / sqrt(32)
//   whole-head key:         H4 over the groups of u_g, u_g = s H32(y) (konlya; static non-block group) or s sqrt(32) y
//                           (static block group: H32 (H32 y / sqrt 32) = sqrt(32) y), times 1 / sqrt(128) (konlya,
//                           orthonormal H128) or 1 / 8 (static whole-head heads: k = H B k_s / 8)
DEVI void convert_k_f16(uint32_t raw, uint32_t sk, int nvalid, int wt, int h, const KParams& kp) {
  const int key = wt >> 1, half = wt & 1;
  const uint32_t pr = raw + (key >> 4) * CHUNK;
  const int tok = key & 15;
  uint32_t y[2][16];
  uint32_t s2[2];
  float sf[2];
  bool lk[2];
#pragma unroll
  for (int gg = 0; gg < 2; ++gg) {
    const int g = 2 * half + gg;
    const uint32_t sw = lds16(pr + 2048 + tok * 8 + 2 * g);
    const uint32_t sbits = key < nvalid ? (sw & 0x7FFFu) : 0u;
    s2[gg] = sbits * 0x10001u;
    sf[gg] = __half2float(__ushort_as_half(static_cast<unsigned short>(sbits)));
    const uint32_t zz = 0x64006400u | ((sw & 15u) * 0x10001u);
    lk[gg] = kp.konly ? (sw >> 15) != 0 : (kp.kmask != nullptr && kp.kmask[h * 4 + g] != 0);
#pragma unroll
    for (int tg = 0; tg < 4; ++tg) {
      const uint32_t w = lds32(pr + tok * 64 + 16 * tg + 4 * g);
      const uint32_t l = w & 0x0F0F0F0Fu, hh = (w >> 4) & 0x0F0F0F0Fu;
      // channels 4tg .. 4tg+3 = bytes 0..3 of l; channels 16 + 4tg .. = bytes 0..3 of hh
      y[gg][2 * tg] = hsub2_u(__byte_perm(l, 0, 0x4140) | 0x64006400u, zz);
      y[gg][2 * tg + 1] = hsub2_u(__byte_perm(l, 0, 0x4342) | 0x64006400u, zz);
      y[gg][8 + 2 * tg] = hsub2_u(__byte_perm(hh, 0, 0x4140) | 0x64006400u, zz);
      y[gg][8 + 2 * tg + 1] = hsub2_u(__byte_perm(hh, 0, 0x4342) | 0x64006400u, zz);
    }
  }
  bool full;
  float fnorm;
  if (kp.konly) {
    const int q = 8 * (tok >> 3) + ((tok & 7) >> 1) + 4 * (tok & 1);   // V scale slot of the token, group 0
    full = (lds16(pr + 2176 + 2 * q) >> 15) != 0;
    fnorm = 0.08838834764831845f;
    if (full) lk[0] = lk[1] = false;
  } else {
    full = kp.kfull != nullptr && kp.kfull[h] != 0;
    fnorm = 0.125f;
  }
  constexpr float R32 = 0.17677669529663687f, S32 = 5.656854249492381f;
  uint32_t o[2][16];
  // warp-uniform: the two lanes of a key hold different groups (static block flags differ per group)
  const bool any_t = __any_sync(0xffffffffu, kp.konly || full || lk[0] || lk[1]);
  if (!any_t) {
#pragma unroll
    for (int gg = 0; gg < 2; ++gg)
#pragma unroll
      for (int j = 0; j < 16; ++j) o[gg][j] = hmul2_u(y[gg][j], s2[gg]);
  } else {
    uint32_t z[2][16];
#pragma unroll
    for (int gg = 0; gg < 2; ++gg) {
#pragma unroll
      for (int j = 0; j < 16; ++j) z[gg][j] = y[gg][j];
      fwht32_h2(z[gg]);
      const uint32_t sb = h2_dup(sf[gg] * R32);
#pragma unroll
      for (int j = 0; j < 16; ++j) o[gg][j] = lk[gg] ? hmul2_u(z[gg][j], sb) : hmul2_u(y[gg][j], s2[gg]);
    }
    if (__any_sync(0xffffffffu, full)) {    // both lanes of a key take the same value of `full`
      uint32_t u[2][16];
#pragma unroll
      for (int gg = 0; gg < 2; ++gg) {
        const uint32_t su = h2_dup(sf[gg] * S32);
#pragma unroll
        for (int j = 0; j < 16; ++j) u[gg][j] = lk[gg] ? hmul2_u(y[gg][j], su) : hmul2_u(z[gg][j], s2[gg]);
      }
      const uint32_t fn = h2_dup(fnorm);
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const uint32_t a = hadd2_u(u[0][j], u[1][j]), b = hsub2_u(u[0][j], u[1][j]);   // H2 over the thread's groups
        const uint32_t oa = __shfl_xor_sync(0xffffffffu, a, 1), ob = __shfl_xor_sync(0xffffffffu, b, 1);
        const uint32_t ra = half ? hsub2_u(oa, a) : hadd2_u(a, oa);                       // H2 over the lanes
        const uint32_t rb = half ? hsub2_u(ob, b) : hadd2_u(b, ob);
        if (full) {
          o[0][j] = hmul2_u(ra, fn);
          o[1][j] = hmul2_u(rb, fn);
        }
      }
    }
  }
  // FP16 tile [ks (16 ch)][key / 8][kc][key % 8][16 B]
#pragma unroll
  for (int gg = 0; gg < 2; ++gg)
#pragma unroll
    for (int c8 = 0; c8 < 4; ++c8) {
      const int ch = 64 * half + 32 * gg + 8 * c8;
      sts128(sk + (ch >> 4) * 2048 + (key >> 3) * 256 + ((ch >> 3) & 1) * 128 + (key & 7) * 16, o[gg][4 * c8],
             o[gg][4 * c8 + 1], o[gg][4 * c8 + 2], o[gg][4 * c8 + 3]);
    }
}

template <int NB, int SRW>
__global__ void __launch_bounds__(384, 1) k_int4_prefill_kbf(const Params p, const KParams kp) {
  extern __shared__ __align__(128) uint8_t smem[];
  using L = W3<NB, SRW>;
  const uint32_t sb = smem_u32(smem);
  const int tid = threadIdx.x, wg = tid >> 7, wt = tid & 127;
  const int ntile = (p.nq + W2_TOK - 1) / W2_TOK;
  const int tile = ntile - 1 - static_cast<int>(blockIdx.x);
  const int h = blockIdx.y;
  const int t0 = tile * W2_TOK;
  const uint32_t raw_full = sb + L::BAR, tile_full = raw_full + 8 * SRW, tile_empty = tile_full + 8 * NB;
  const int tlast = min(p.nq, t0 + W2_TOK) - 1;
  const int kend = min(p.nk, p.p0 + tlast - p.shift + 1);
  const int nblk = kend > 0 ? (kend + BN - 1) / BN : 0;
  const int npages = kend > 0 ? (kend + PAGE - 1) / PAGE : 0;
  if (tid == 0) {
    for (int s = 0; s < SRW; ++s) mbar_init(raw_full + 8 * s, 1);
    for (int s = 0; s < NB; ++s) {
      mbar_init(tile_full + 8 * s, 1);
      mbar_init(tile_empty + 8 * s, 8);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (wg == 0) {
    auto issue = [&](int n) {
      const int s = n % SRW;
      const int pa = n * PPB, pb = min(npages, pa + PPB);
      const uint32_t dst = sb + L::RAW + s * PPB * CHUNK;
      mbar_expect_tx(raw_full + 8 * s, (pb - pa) * CHUNK);
      for (int pg = pa; pg < pb; ++pg) {
        const long long page = p.bt[pg];
        bulk_g2s(dst + (pg - pa) * CHUNK, p.pool + page * p.page_stride + static_cast<long long>(h) * CHUNK, CHUNK,
                 raw_full + 8 * s);
      }
    };
    if (wt == 0)
      for (int n = 0; n < min(SRW, nblk); ++n) issue(n);
    for (int n = 0; n < nblk; ++n) {
      const int rs = n % SRW, tb = n % NB;
      mbar_wait(raw_full + 8 * rs, (n / SRW) & 1);
      if (n >= NB) mbar_wait(tile_empty + 8 * tb, ((n / NB) - 1) & 1);
      const uint32_t tbase = sb + L::ST + tb * W3_TILE;
      const uint32_t raw = sb + L::RAW + rs * PPB * CHUNK;
#ifndef FRP_DBG_NOCONV
      convert_k_f16(raw, tbase, kend - n * BN, wt, h, kp);
      convert_block<2, false>(raw, 0, tbase + 16384, 0, kend - n * BN, wt);
#endif
      fence_async_smem();
      named_bar(1, 128);
      if (wt == 0) {
        mbar_arrive(tile_full + 8 * tb);
        if (n + SRW < nblk) {
          fence_async_smem();
          issue(n + SRW);
        }
      }
    }
    return;
  }
  const int c = wg - 1;
  const int warp = wt >> 5, lane = wt & 31;
  const int ct0 = t0 + 16 * c;
  const int r0 = 16 * warp + (lane >> 2), r1 = r0 + 8;
  const int tq0 = ct0 + (r0 >> 2), tq1 = ct0 + (r1 >> 2);
  const int hq0 = 4 * h + (r0 & 3), hq1 = 4 * h + (r1 & 3);
  const bool v0 = tq0 < p.nq, v1 = tq1 < p.nq;
  const int pos0 = v0 ? p.p0 + tq0 : -(1 << 30), pos1 = v1 ? p.p0 + tq1 : -(1 << 30);
  const float fac0 = kp.qscale, fac1 = kp.qscale;
  const int clast = min(p.nq, ct0 + 16) - 1;
  const int ckend = ct0 < p.nq ? min(p.nk, p.p0 + clast - p.shift + 1) : 0;
  const int cnblk = ckend > 0 ? (ckend + BN - 1) / BN : 0;
  const int kfull = max(0, min(p.nk, p.p0 + ct0 - p.shift + 1));
  const uint32_t sq = sb + c * 16384;
  // q tile in FP16 (bf16 -> fp16 is exact in the fp16 range) [ks][row / 8][kc][row % 8][16 B]: 64 rows x 16 chunks
#pragma unroll
  for (int it = 0; it < 8; ++it) {
    const int idx = wt + 128 * it;
    const int row = idx >> 4, cc = idx & 15;          // chunk cc = channels 8 cc .. 8 cc + 7
    const int tq = ct0 + (row >> 2);
    const uint32_t dst = sq + (cc >> 1) * 2048 + (row >> 3) * 256 + (cc & 1) * 128 + (row & 7) * 16;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (tq < p.nq) v = *reinterpret_cast<const uint4*>(kp.q + tq * kp.q_ts + (4 * h + (row & 3)) * D + 8 * cc);
    uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < (kp.qf16 ? 0 : 4); ++i) {
      const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(&w[i]);
      const __half2 f2 = __floats2half2_rn(__low2float(b2), __high2float(b2));
      w[i] = *reinterpret_cast<const uint32_t*>(&f2);
    }
    sts128(dst, w[0], w[1], w[2], w[3]);
  }
  fence_async_smem();
  named_bar(2 + c, 128);
  if (FRP_PP && c == 1) named_arrive(5, 256);  // consumer 0 issues first

  float o[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) o[i] = 0.f;
  float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.f, l1 = 0.f;
  const int tq4 = lane & 3;
  for (int n = 0; n < nblk; ++n) {
    const int tb = n % NB;
    mbar_wait(tile_full + 8 * tb, (n / NB) & 1);
#ifdef FRP_DBG_NOCONS
    if (false) {
#else
    if (n < cnblk) {
#endif
      const uint32_t sk = sb + L::ST + tb * W3_TILE, sv = sk + 16384;
      float sacc[32];
      PP_TURN();
      wg_fence();
      wg_f16_n64_0(sacc, sdesc(sq), sdesc(sk));
#pragma unroll
      for (int ks = 1; ks < 8; ++ks) wg_f16_n64_1(sacc, sdesc(sq + ks * 2048), sdesc(sk + ks * 2048));
      wg_commit();
      PP_PASS();
      wg_wait_all();
      acc_sync(sacc);
#include "int4_prefill_softmax_pv.inc"
    } else {
      PP_TURN();
      PP_PASS();
      PP_TURN();
      PP_PASS();
      if (lane == 0) mbar_arrive(tile_empty + 8 * tb);
    }
  }
  if (FRP_PP && c == 0) named_bar(5, 256);   // consume consumer 1's last hand-over
#include "int4_prefill_epilogue.inc"
}

// ---------------------------------------------------------------------------------------------- q preparation
// one warp per (token, q head): q bf16 -> stored basis (static FlagRot: whole-head H128 / 8 if kfull[kv head], then
// orthonormal H32 on the block-flagged groups) -> sigma = 2^floor(log2(448 / max|q|)), hi = e4m3(sigma q), lo =
// e4m3(sigma q - hi); qfac = scale * log2(e) / sigma.  Lane l holds channels 4l .. 4l+3.
DEVI void fwht_lanes(float (&x)[4], int lane, int span_lanes) {
  // in-thread stages (channel offsets 1, 2)
  float a = x[0] + x[1], b = x[0] - x[1], c = x[2] + x[3], d = x[2] - x[3];
  x[0] = a + c; x[1] = b + d; x[2] = a - c; x[3] = b - d;
  for (int m = 1; m < span_lanes; m <<= 1) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float y = __shfl_xor_sync(0xffffffffu, x[i], m);
      x[i] = (lane & m) ? y - x[i] : x[i] + y;
    }
  }
}

__global__ void k_prep_q(const __nv_bfloat16* q, long long q_ts, int nq, int Hq, const uint8_t* kfull,
                         const uint8_t* kmask, float scale_log2e, uint8_t* qh, uint8_t* ql, float* qfac) {
  const long long w = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (w >= static_cast<long long>(nq) * Hq) return;
  const int t = static_cast<int>(w / Hq), hq = static_cast<int>(w % Hq), hk = hq >> 2;
  const __nv_bfloat16* src = q + t * q_ts + hq * D + 4 * lane;
  const uint2 raw = *reinterpret_cast<const uint2*>(src);
  float x[4];
  {
    const __nv_bfloat162 p0 = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
    const __nv_bfloat162 p1 = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
    x[0] = __low2float(p0); x[1] = __high2float(p0); x[2] = __low2float(p1); x[3] = __high2float(p1);
  }
  if (kfull && kfull[hk]) {
    fwht_lanes(x, lane, 32);
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] *= 0.125f;
  }
  if (kmask) {                      // all lanes transform (shuffles stay inside each 8-lane group), then select
    const bool f = kmask[hk * 4 + (lane >> 3)] != 0;
    float y[4] = {x[0], x[1], x[2], x[3]};
    fwht_lanes(y, lane, 8);
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = f ? y[i] * 0.17677669529663687f : x[i];
  }
  if (ql == nullptr) {               // FP16 output (q in the stored basis for the FP16-QK kernel), no scale
    const __half2 a = __floats2half2_rn(x[0], x[1]), b = __floats2half2_rn(x[2], x[3]);
    uint2 v;
    v.x = *reinterpret_cast<const uint32_t*>(&a);
    v.y = *reinterpret_cast<const uint32_t*>(&b);
    *reinterpret_cast<uint2*>(qh + ((static_cast<long long>(t) * Hq + hq) * D + 4 * lane) * 2) = v;
    return;
  }
  float am = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
#pragma unroll
  for (int m = 16; m >= 1; m >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, m));
  am = fmaxf(am, 1e-30f);
  const float sig = exp2f(floorf(log2f(448.f / am)));
  float y[4], hf[4];
  uint32_t hb = 0, lb = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) y[i] = x[i] * sig;
  hb = pack4(y[0], y[1], y[2], y[3]);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __nv_fp8_e4m3 e = *reinterpret_cast<const __nv_fp8_e4m3*>(reinterpret_cast<const uint8_t*>(&hb) + i);
    hf[i] = static_cast<float>(e);
  }
  lb = pack4(y[0] - hf[0], y[1] - hf[1], y[2] - hf[2], y[3] - hf[3]);
  const long long o = (static_cast<long long>(t) * Hq + hq) * D + 4 * lane;
  *reinterpret_cast<uint32_t*>(qh + o) = hb;
  *reinterpret_cast<uint32_t*>(ql + o) = lb;
  if (lane == 0) qfac[static_cast<long long>(t) * Hq + hq] = scale_log2e / sig;
}

}  // namespace

extern "C" {

int frp_abi_version() { return 1; }

int frp_prep_q(const void* q, long long q_ts, int nq, int Hq, const uint8_t* kfull, const uint8_t* kmask,
               float scale_log2e, uint8_t* qh, uint8_t* ql, float* qfac, void* stream) {
  if (nq <= 0) return 0;
  const long long warps = static_cast<long long>(nq) * Hq;
  const unsigned blocks = static_cast<unsigned>((warps * 32 + 255) / 256);
  k_prep_q<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<const __nv_bfloat16*>(q), q_ts, nq, Hq,
                                                                     kfull, kmask, scale_log2e, qh, ql, qfac);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

int frp_int4_prefill(const uint8_t* qh, const uint8_t* ql, const float* qfac, long long q_ts, const uint8_t* pool,
                     long long page_stride, const int* bt, int nq, int nk, int p0, int shift, int Hkv,
                     const uint8_t* lomask, void* out, long long o_ts, float* lse, int stages, void* stream) {
  if (nq <= 0) return 0;
  Params p{};
  p.qh = qh; p.ql = ql; p.qfac = qfac; p.q_ts = q_ts; p.pool = pool; p.page_stride = page_stride; p.bt = bt;
  p.nq = nq; p.nk = nk; p.p0 = p0; p.shift = shift; p.Hkv = Hkv; p.lomask = lomask;
  p.out = static_cast<__nv_bfloat16*>(out); p.o_ts = o_ts; p.lse = lse;
  const dim3 grid((nq + TOK - 1) / TOK, Hkv);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (stages >= 100) {   // v2 (warp-specialized): stages = 10000 reg + 1000 QG + 100 + 10 NB + SRW (QG 0 -> 1)
    const dim3 g2((nq + W2_TOK - 1) / W2_TOK, Hkv);
    const int nb = (stages / 10) % 10, srw = stages % 10, qg = max(1, (stages / 1000) % 10), reg = stages / 10000;
#define FRP_W2(NB_, SR_, QG_, PR_, CR_)                                                                       \
    if (nb == NB_ && srw == SR_ && qg == QG_ && reg == (PR_ > 0)) {                                         \
      constexpr int sm = W2<NB_, SR_>::BYTES;                                                               \
      auto kern = k_int4_prefill_ws<NB_, SR_, QG_, PR_, CR_>;                                               \
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);                          \
      kern<<<g2, 384, sm, s>>>(p);                                                                          \
      return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;                                                 \
    }
    FRP_W2(2, 3, 1, 0, 0) FRP_W2(3, 3, 1, 0, 0) FRP_W2(3, 3, 2, 0, 0) FRP_W2(3, 3, 2, 64, 216)
    FRP_W2(3, 3, 4, 64, 216) FRP_W2(2, 3, 2, 64, 216) FRP_W2(4, 4, 2, 64, 216)
#undef FRP_W2
    return -2;
  }
  if (stages == 3) {
    constexpr int sm = smem_bytes<3>();
    cudaFuncSetAttribute(k_int4_prefill<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    k_int4_prefill<3><<<grid, 128, sm, s>>>(p);
  } else {
    constexpr int sm = smem_bytes<2>();
    cudaFuncSetAttribute(k_int4_prefill<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    k_int4_prefill<2><<<grid, 128, sm, s>>>(p);
  }
  return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

int frp_int4_prefill_kbf(const void* q, long long q_ts, float qscale, const uint8_t* kmask, const uint8_t* kfull,
                         int konly, int qf16, const uint8_t* pool, long long page_stride, const int* bt, int nq, int nk, int p0,
                         int shift, int Hkv, void* out, long long o_ts, float* lse, int stages, void* stream) {
  if (nq <= 0) return 0;
  Params p{};
  p.pool = pool; p.page_stride = page_stride; p.bt = bt; p.nq = nq; p.nk = nk; p.p0 = p0; p.shift = shift;
  p.Hkv = Hkv; p.out = static_cast<__nv_bfloat16*>(out); p.o_ts = o_ts; p.lse = lse;
  KParams kp{};
  kp.q = static_cast<const __nv_bfloat16*>(q); kp.q_ts = q_ts; kp.qscale = qscale; kp.kmask = kmask; kp.kfull = kfull;
  kp.konly = konly;
  kp.qf16 = qf16;
  const dim3 g2((nq + W2_TOK - 1) / W2_TOK, Hkv);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  const int nb = (stages / 10) % 10, srw = stages % 10;
#define FRP_W3(NB_, SR_)                                                                                  \
  if (nb == NB_ && srw == SR_) {                                                                         \
    constexpr int sm = W3<NB_, SR_>::BYTES;                                                              \
    auto kern = k_int4_prefill_kbf<NB_, SR_>;                                                            \
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);                         \
    kern<<<g2, 384, sm, s>>>(p, kp);                                                                     \
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;                                                \
  }
  FRP_W3(2, 3) FRP_W3(3, 3) FRP_W3(3, 4)
#undef FRP_W3
  return -2;
}

const char* frp_last_error() { return cudaGetErrorString(cudaGetLastError()); }
}
