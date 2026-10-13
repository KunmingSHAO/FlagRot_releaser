// Phase 2i: prefix-cached prefill attention straight from the FlagRot INT4 pages (sm_90, mma.sync f16).
//
// Multi-turn / agentic serving prefills a few new tokens on top of a long cached prefix (tau2 retail: ~145 new tokens
// over ~10k cached, prefix-hit rate 98 %).  The default "fa" prefill path gathers + dequantizes the whole prefix in
// PyTorch for every chunk and runs BF16 FlashAttention; this kernel attends over the paged INT4 cache directly:
//
//   CTA = (work item, KV head), 4 warps, 64 rows = TQ = 64 / G query tokens x the G q heads of the KV head
//         (row r: token r / G, q head h G + r % G); a work item = one q tile x one key range (split-K over the
//         prefix: many CTAs even for a few query tokens; partials merged by k_mq_merge).
//   keys  in blocks of 64 (4 pages): the 4 page chunks (2304 B per (page, KV head), int4ah:32 v2 / key-only 2x2
//         layout, arrive by cp.async (double buffered); the CTA converts them into FP16 tiles
//         K [64 keys x 128] and V^T [128 x 64 keys]: (c - z) exact in f16x2, one rounding at the FP16 scale (11-bit
//         mantissa, finer than the BF16 path's dequantized K / V); key-only 2x2 (konlya): the per-token inverse
//         transform (per-group H32 or whole-head H128, flags in the scale sign bits) in packed FP16 (= Phase 2f).
//   QK    q pre-rotated into the keys' stored basis for the static formats (whole-head H128 / 8, block H32: k_mq_prep)
//         and pre-scaled by softmax_scale * log2(e), FP16; mma.sync m16n8k16 f16 -> fp32; online softmax in fp32
//         (exp2), P rounded to FP16, PV mma.sync f16 -> fp32.  Causal: key j visible to the query at
//         position p iff kmin <= j <= p - shift (shift = BF16 window W, kmin = sink tokens S: the window / sink keys
//         are attended outside; Phase 2k: pages hold every token at its real position).
//   out   per item: unnormalized O (fp32), row max m (log2 units) and sum l; k_mq_merge combines the items of a tile
//         -> bf16 rows of the layer output (or fp32 for a later basis change) + optional natural-log LSE.
//
// Numerics: every operand is at least as precise as in the BF16 "fa" path (FP16 K / V / q / P vs BF16), fp32
// accumulation and softmax: prompt KL equals the default path within noise.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

#define DEVI __device__ __forceinline__

namespace {

constexpr int D = 128, PAGE = 16, CHUNK = 144 * PAGE, BN = 64, PPB = BN / PAGE, ROWS = 64;
// raw page chunk in shared memory, padded against bank conflicts of the converters (cp.async places the 16-B segments):
// K codes 16 token rows of 80 B (64 + 16), V^T codes 4 w-slabs of 288 B (256 + 32), K scales, V scales
constexpr int R_KC = 0, R_KROW = 80, R_VC = 16 * R_KROW, R_VW = 288, R_KS = R_VC + 4 * R_VW, R_VS = R_KS + 128,
              RCH = R_VS + 128;   // 2688 B
DEVI int raw_dst(int off) {       // source byte offset (16-B aligned) inside the 2304-B chunk -> padded offset
  return off < 1024 ? (off >> 6) * R_KROW + (off & 63)
                    : (off < 2048 ? R_VC + ((off - 1024) >> 8) * R_VW + ((off - 1024) & 255) : R_KS + (off - 2048));
}
// Phase 3a: INT2 pages (kernels/csrc/attn_decode_v2.cuh INT2 layout: 1280-B chunk = K codes 16 x 32 B (rows swizzled:
// token t at row t ^ ((t >> 1) & 2)) | V^T codes 64 entries x 8 B | K scales | V scales) in the same raw slot (RCH):
// K rows padded to 48 B, V^T w-slabs (16 entries) to 144 B; the scale words at the INT4 offsets R_KS / R_VS
DEVI uint32_t lds32(uint32_t a);
constexpr int CHUNK2 = 80 * PAGE, R2_KROW = 48, R2_VC = 16 * R2_KROW, R2_VW = 144;
static_assert(R2_VC + 4 * R2_VW <= R_KS, "INT2 raw layout");
DEVI int raw_dst2(int off) {
  return off < 512 ? (off >> 5) * R2_KROW + (off & 31)
                   : (off < 1024 ? R2_VC + ((off - 512) >> 7) * R2_VW + ((off - 512) & 127)
                                 : (off < 1152 ? R_KS + (off - 1024) : R_VS + (off - 1152)));
}
template <bool B2> DEVI int rdst(int off) { return B2 ? raw_dst2(off) : raw_dst(off); }
template <bool B2> constexpr int chunk_of() { return B2 ? CHUNK2 : CHUNK; }
// K code word of (token tok, tg, group g) -> A / B code bytes (unsigned, in byte lanes) of that group
template <bool B2>
DEVI void k_codes(uint32_t pr, int tok, int tg, int g, uint32_t& l, uint32_t& hh) {
  if constexpr (B2) {
    const uint32_t w = lds32(pr + (tok ^ ((tok >> 1) & 2)) * R2_KROW + 8 * tg + 4 * (g >> 1));
    const int sh = 4 * (g & 1);
    l = (w >> sh) & 0x03030303u;
    hh = (w >> (sh + 2)) & 0x03030303u;
  } else {
    const uint32_t w = lds32(pr + tok * R_KROW + 16 * tg + 4 * g);
    l = w & 0x0F0F0F0Fu;
    hh = (w >> 4) & 0x0F0F0F0Fu;
  }
}
// V^T code word (w, gid, c8, sub) (INT4 numbering) -> low / high element bytes (tokens 8 c8 + 2 i, + 1)
template <bool B2>
DEVI void v_codes(uint32_t pr, int w, int gid, int c8, int sub, uint32_t& l, uint32_t& hh) {
  if constexpr (B2) {
    const int e = (w * 8 + gid) * 2 + c8;                    // 8-byte entry, word sub >> 1, rows r = sub & 1
    const uint32_t cw = lds32(pr + R2_VC + (e >> 4) * R2_VW + 8 * (e & 15) + 4 * (sub >> 1));
    const int sh = 4 * (sub & 1);
    l = (cw >> sh) & 0x03030303u;
    hh = (cw >> (sh + 2)) & 0x03030303u;
  } else {
    const int wd = ((w * 8 + gid) * 2 + c8) * 4 + sub;
    const uint32_t cw = lds32(pr + R_VC + (wd >> 6) * R_VW + 4 * (wd & 63));
    l = cw & 0x0F0F0F0Fu;
    hh = (cw >> 4) & 0x0F0F0F0Fu;
  }
}
template <bool B2> constexpr uint32_t zmask() { return B2 ? 3u : 15u; }
constexpr int KSTR = 136;   // halves per K tile row (128 + 8 pad: conflict-free ldmatrix)
constexpr int VSTR = 72;    // halves per V^T tile row (64 + 8 pad)
constexpr int SM_RAW = 0, SM_KS = 2 * PPB * RCH, SM_VT = SM_KS + BN * KSTR * 2, SM_QS = SM_VT + D * VSTR * 2,
              SM_BYTES = SM_QS + ROWS * KSTR * 2;   // raw ring | K tile | V^T tile | q tile (70 KB: 3 CTAs / SM)
constexpr float NEG_INF = -__builtin_huge_valf();
constexpr int TILE_INTS = 9, ITEM_INTS = 4;

struct Params {
  const __half* q;        // [m, Hq, 128] fp16, stored basis, pre-scaled by softmax scale * log2(e)
  const uint8_t* pool;    // [pages, Hkv, 2304]
  long long page_stride;  // bytes
  const int* bt;          // block table [rows, bt_stride] int32
  long long bt_stride;
  const int* tiles;       // [ntiles, 9]: bt row, crow0, orow0, ntok, pos0, nk, item0, nitem, ext (every row also
                          // sees the page keys <= ext: a request's window positions before its first computed token
                          // (reused prefix) are read from the pages; -2^30 = none)
  const int* items;       // [nitems, 4]: tile, kb0, kb1, -
  int Hkv, G, shift;
  int kmin;               // keys [0, kmin) masked (Phase 2k: sink tokens at their real positions, attended in BF16)
  float* opart;           // [nitems, Hkv, 64, 128]
  float* ml;              // [nitems, Hkv, 64, 2]
  int dbg;                // profiling (env FLAGROT_MQ_DBG): 1 = producer skips the conversion, 2 = consumers skip the MMAs
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
DEVI void sts128(uint32_t a, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(a), "r"(x), "r"(y), "r"(z), "r"(w));
}
DEVI void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
DEVI void cp_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }
DEVI void ldsm_x4(uint32_t a, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
DEVI void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
DEVI uint32_t hsub2_u(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("sub.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
DEVI uint32_t hadd2_u(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("add.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
DEVI uint32_t hmul2_u(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
DEVI uint32_t h2_dup(float x) {  // fp32 -> fp16 in both halves
  const unsigned short u = __half_as_ushort(__float2half_rn(x));
  return static_cast<uint32_t>(u) * 0x10001u;
}
DEVI uint32_t pack_h2(float lo, float hi) {
  const __half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}
// in-register FWHT of 32 channels held as 16 f16x2 registers r[j] = (channel 2j, 2j+1), unnormalized (Phase 2f)
DEVI void fwht32_h2(uint32_t (&r)[16]) {
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const uint32_t sw = __byte_perm(r[j], r[j], 0x1032);
    r[j] = __byte_perm(hadd2_u(r[j], sw), hsub2_u(r[j], sw), 0x5410);
  }
#pragma unroll
  for (int m = 1; m < 16; m <<= 1)
#pragma unroll
    for (int j = 0; j < 16; ++j)
      if ((j & m) == 0) {
        const uint32_t a = r[j], b = r[j + m];
        r[j] = hadd2_u(a, b);
        r[j + m] = hsub2_u(a, b);
      }
}

// head 256 (k_mq3): fwht32_h2 with its first stage (channels 2j, 2j+1 of one register) as one packed-FP16
// fma(r, (1, -1), swap(r)) instead of add + sub + two permutes; bit-identical (x +-1 is exact, one rounding).
DEVI uint32_t hfma2_u(uint32_t a, uint32_t b, uint32_t c) {
  const __half2 r = __hfma2(*reinterpret_cast<const __half2*>(&a), *reinterpret_cast<const __half2*>(&b),
                            *reinterpret_cast<const __half2*>(&c));
  return *reinterpret_cast<const uint32_t*>(&r);
}
constexpr uint32_t H2_PM = 0xBC003C00u;   // (1, -1)
DEVI void fwht32_h2f(uint32_t (&r)[16]) {
#pragma unroll
  for (int j = 0; j < 16; ++j) r[j] = hfma2_u(r[j], H2_PM, __byte_perm(r[j], r[j], 0x1032));
#pragma unroll
  for (int m = 1; m < 16; m <<= 1)
#pragma unroll
    for (int j = 0; j < 16; ++j)
      if ((j & m) == 0) {
        const uint32_t a = r[j], b = r[j + m];
        r[j] = hadd2_u(a, b);
        r[j + m] = hsub2_u(a, b);
      }
}

// K of one key (thread pair: half = channels 64 half .. +63, i.e. groups 2 half, 2 half + 1) -> FP16 K tile row.
// raw = the 4 page chunks of the block; nvalid = valid keys in the block (others -> 0).  KONLY: key-only 2x2
// inverse transform to the original basis (per group H32 / sqrt 32 if the K scale sign bit is set, or the whole-head
// orthonormal H128 if the sign bit of the token's group-0 V scale word is set); else stored basis (static formats).
// LAY 0: K tile [key][KSTR halves] (ldmatrix, k_mq); LAY 1: wgmma no-swizzle K-major core matrices (k_mq2)
// [ch / 16][key / 8][(ch / 8) % 2][key % 8][16 B]
template <bool KONLY, int LAY, bool B2 = false>
DEVI void convert_k(uint32_t raw, uint32_t ks, int nvalid, int wt) {
  const int key = wt >> 1, half = wt & 1;
  const uint32_t pr = raw + (key >> 4) * RCH;
  const int tok = key & 15;
  bool full = false;
  if (KONLY) {
    const int q = 8 * (tok >> 3) + ((tok & 7) >> 1) + 4 * (tok & 1);   // V scale slot of the token, group 0
    full = (lds16(pr + R_VS + 2 * q) >> 15) != 0;
  }
  constexpr float R32 = 0.17677669529663687f, FN = 0.08838834764831845f;
  uint32_t o[2][16];
#pragma unroll
  for (int gg = 0; gg < 2; ++gg) {
    const int g = 2 * half + gg;
    const uint32_t sw = lds16(pr + R_KS + tok * 8 + 2 * g);
    const uint32_t sbits = key < nvalid ? (sw & 0x7FFFu) : 0u;
    const uint32_t s2 = sbits * 0x10001u;
    const uint32_t zz = 0x64006400u | ((sw & zmask<B2>()) * 0x10001u);
    uint32_t y[16];
#pragma unroll
    for (int tg = 0; tg < 4; ++tg) {
      uint32_t l, hh;
      k_codes<B2>(pr, tok, tg, g, l, hh);
      y[2 * tg] = hsub2_u(__byte_perm(l, 0, 0x4140) | 0x64006400u, zz);
      y[2 * tg + 1] = hsub2_u(__byte_perm(l, 0, 0x4342) | 0x64006400u, zz);
      y[8 + 2 * tg] = hsub2_u(__byte_perm(hh, 0, 0x4140) | 0x64006400u, zz);
      y[8 + 2 * tg + 1] = hsub2_u(__byte_perm(hh, 0, 0x4342) | 0x64006400u, zz);
    }
    if (!KONLY) {
#pragma unroll
      for (int j = 0; j < 16; ++j) o[gg][j] = hmul2_u(y[j], s2);
    } else {
      // per group: H32 / sqrt 32 (sign bit of the K scale word), or (whole-head key) s H32(y) as the first stage of
      // the orthonormal H128 = H4 x H32 (H4 over the 4 groups below), or none
      const bool lk = ((sw >> 15) & 1u) != 0 && !full;
      const uint32_t sb = h2_dup(__half2float(__ushort_as_half(static_cast<unsigned short>(sbits))) * R32);
      const uint32_t sz = full ? s2 : sb;
      uint32_t z[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) z[j] = y[j];
      fwht32_h2(z);                              // exact: integers |.| <= 480
#pragma unroll
      for (int j = 0; j < 16; ++j) o[gg][j] = (full || lk) ? hmul2_u(z[j], sz) : hmul2_u(y[j], s2);
    }
  }
  if (KONLY && __any_sync(0xffffffffu, full)) {  // both lanes of a key share `full`
    const uint32_t fn = h2_dup(FN);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const uint32_t a = hadd2_u(o[0][j], o[1][j]), b = hsub2_u(o[0][j], o[1][j]);   // H2 over the thread's groups
      const uint32_t oa = __shfl_xor_sync(0xffffffffu, a, 1), ob = __shfl_xor_sync(0xffffffffu, b, 1);
      const uint32_t ra = half ? hsub2_u(oa, a) : hadd2_u(a, oa);                       // H2 over the lane pair
      const uint32_t rb = half ? hsub2_u(ob, b) : hadd2_u(b, ob);
      if (full) {
        o[0][j] = hmul2_u(ra, fn);
        o[1][j] = hmul2_u(rb, fn);
      }
    }
  }
  const uint32_t row = LAY == 0 ? ks + key * (KSTR * 2) + half * 128 : ks + (key >> 3) * 256 + (key & 7) * 16;
#pragma unroll
  for (int gg = 0; gg < 2; ++gg)
#pragma unroll
    for (int c8 = 0; c8 < 4; ++c8) {
      const int ch8 = 8 * half + 4 * gg + c8;     // 8-channel chunk
      const uint32_t a = LAY == 0 ? row + gg * 64 + c8 * 16 : row + (ch8 >> 1) * 2048 + (ch8 & 1) * 128;
      sts128(a, o[gg][4 * c8], o[gg][4 * c8 + 1], o[gg][4 * c8 + 2], o[gg][4 * c8 + 3]);
    }
}

// V^T: 1024 code words per block (4 pages x 128 dims x 2 token octets); word (w, gid, c8, sub) of a page = dim
// 64 (sub >> 1) + 16 w + gid + 8 (sub & 1), tokens 8 c8 + 2 i (low nibble of byte i), + 1 (high nibble).  The 8 V
// scale words of the octet (group g = dim / 32) are slots 8 c8 .. 8 c8 + 7 = tokens 0 2 4 6 1 3 5 7.
// LAY 0: V^T tile [dim][VSTR halves] (k_mq); LAY 1: wgmma core matrices [key / 16][dim / 8][(key / 8) % 2][dim % 8][16 B]
// (k_mq2).  nthr threads, thread index tid.
template <int LAY, bool B2 = false>
DEVI void convert_v(uint32_t raw, uint32_t vt, int nvalid, int tid) {
  // thread -> (page pg, token octet c8, value group g) = tid / 4 and 8 of the group's 32 dims (r = 8 u .. 8 u + 7):
  // the octet's 8 scale words are read and unpacked once; dim 32 g + r = word (w, gid, c8, sub) with sub = 2 (g / 2)
  // + (r / 8) % 2, w = 2 (g % 2) + r / 16, gid = r % 8
  const int cmb = tid >> 2, u = tid & 3;
  const int pg = cmb >> 3, c8 = (cmb >> 2) & 1, g = cmb & 3;
  const uint32_t pr = raw + pg * RCH;
  const uint4 sv = lds128(pr + R_VS + g * 32 + 16 * c8);
  const int nv = min(max(nvalid - (16 * pg + 8 * c8), 0), 8);
  uint32_t mag[4], zz[4];
  {
    uint32_t sp[4];
    sp[0] = __byte_perm(sv.x, sv.z, 0x5410);    // tokens (0, 1)
    sp[1] = __byte_perm(sv.x, sv.z, 0x7632);    // (2, 3)
    sp[2] = __byte_perm(sv.y, sv.w, 0x5410);    // (4, 5)
    sp[3] = __byte_perm(sv.y, sv.w, 0x7632);    // (6, 7)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      zz[j] = 0x64006400u | (sp[j] & (zmask<B2>() * 0x10001u));
      mag[j] = sp[j] & 0x7FFF7FFFu;
      if (2 * j >= nv) mag[j] &= 0xFFFF0000u;
      if (2 * j + 1 >= nv) mag[j] &= 0x0000FFFFu;
    }
  }
  const int sub = 2 * (g >> 1) + (u & 1), w = 2 * (g & 1) + (u >> 1);
  const int dim0 = 32 * g + 8 * u;
#pragma unroll
  for (int gid = 0; gid < 8; ++gid) {
    uint32_t l, hh;
    v_codes<B2>(pr, w, gid, c8, sub, l, hh);
    uint32_t o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // half2 (token 2 j, 2 j + 1) = (low, high nibble of byte j)
      const uint32_t c = (__byte_perm(l, hh, static_cast<uint32_t>(j | ((4 + j) << 8))) & 0x000F000Fu) | 0x64006400u;
      o[j] = hmul2_u(hsub2_u(c, zz[j]), mag[j]);
    }
    const int dim = dim0 + gid;
    const uint32_t a = LAY == 0 ? vt + dim * (VSTR * 2) + (16 * pg + 8 * c8) * 2
                                : vt + pg * 4096 + (dim >> 3) * 256 + c8 * 128 + (dim & 7) * 16;
    sts128(a, o[0], o[1], o[2], o[3]);
  }
}

template <bool KONLY>
__global__ void __launch_bounds__(128, 3) k_mq(const Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sb = smem_u32(smem);
  const uint32_t ks = sb + SM_KS, vt = sb + SM_VT, qs = sb + SM_QS;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, c4 = lane & 3;
  const int item = blockIdx.x, h = blockIdx.y;
  const int* it = p.items + ITEM_INTS * item;
  const int* tl = p.tiles + TILE_INTS * it[0];
  const int req = tl[0], crow0 = tl[1], ntok = tl[3], pos0 = tl[4], nk = tl[5], ext = tl[8];
  const int G = p.G, Hq = p.Hkv * G, TQ = ROWS / G;
  const int* bt = p.bt + static_cast<long long>(req) * p.bt_stride;
  // keys of this item: blocks [kb0, kb1), clipped to the pages and to the last token's causal limit
  const int kend = min(min(nk, it[2] * BN), max(pos0 + ntok - p.shift, ext + 1));
  const int kb0 = it[1];
  const int kb1 = kend > kb0 * BN ? (kend + BN - 1) / BN : kb0;
  // this thread's two rows (g8, g8 + 8 of the warp's 16)
  int rpos[2];
  bool rval[2];
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    const int r = warp * 16 + g8 + 8 * rh;
    const int t = r / G;
    rval[rh] = r < TQ * G && t < ntok;
    rpos[rh] = max(pos0 + t - p.shift, ext);    // last visible key of the row
  }
  // q tile -> shared memory (FP16 rows, zero for padding rows); A fragments by ldmatrix in the QK loop
  for (int c = tid; c < ROWS * (D / 8); c += 128) {
    const int r = c >> 4, c16 = c & 15, t = r / G;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (r < TQ * G && t < ntok)
      v = *reinterpret_cast<const uint4*>(p.q + (static_cast<long long>(crow0 + t) * Hq + h * G + r % G) * D + 8 * c16);
    sts128(qs + r * (KSTR * 2) + c16 * 16, v.x, v.y, v.z, v.w);
  }
  float o[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float mrow[2] = {NEG_INF, NEG_INF}, lrow[2] = {0.f, 0.f};

  const uint8_t* poolh = p.pool + static_cast<long long>(h) * CHUNK;
  auto load_raw = [&](int kb, int buf) {
    const int npg = min(PPB, (kend - kb * BN + PAGE - 1) / PAGE);
    const uint32_t dst = sb + SM_RAW + buf * (PPB * RCH);
    for (int s = tid; s < npg * (CHUNK / 16); s += 128) {
      const int pg = s / (CHUNK / 16), off = (s % (CHUNK / 16)) * 16;
      const long long page = bt[kb * PPB + pg];
      cp16(dst + pg * RCH + raw_dst(off), poolh + page * p.page_stride + off);
    }
    cp_commit();
  };
  if (kb0 < kb1) load_raw(kb0, 0);
  for (int kb = kb0; kb < kb1; ++kb) {
    const int buf = (kb - kb0) & 1;
    cp_wait_all();
    __syncthreads();
    if (kb + 1 < kb1) load_raw(kb + 1, buf ^ 1);
    const uint32_t raw = sb + SM_RAW + buf * (PPB * RCH);
    const int nvalid = kend - kb * BN;
    convert_k<KONLY, 0>(raw, ks, nvalid, tid);
    convert_v<0>(raw, vt, nvalid, tid);
    __syncthreads();
    // S = q K^T (16 rows x 64 keys per warp), log2 units
    float s[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      uint32_t qa[4];
      ldsm_x4(qs + (warp * 16 + (lane & 15)) * (KSTR * 2) + (16 * kk + 8 * (lane >> 4)) * 2, qa[0], qa[1], qa[2], qa[3]);
#pragma unroll
      for (int nt = 0; nt < 8; nt += 2) {
        const int mi = lane >> 3, ri = lane & 7;
        const uint32_t a = ks + (8 * (nt + (mi >> 1)) + ri) * (KSTR * 2) + (16 * kk + 8 * (mi & 1)) * 2;
        uint32_t b0, b1, b2, b3;
        ldsm_x4(a, b0, b1, b2, b3);
        mma16816(s[nt], qa, b0, b1);
        mma16816(s[nt + 1], qa, b2, b3);
      }
    }
    // mask + online softmax
    const int j0 = kb * BN + 2 * c4;
    float alpha[2];
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const int lim = min(kend - 1, rpos[rh]);
      float mx = NEG_INF;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int j = j0 + 8 * nt + e;
          float& x = s[nt][2 * rh + e];
          x = (rval[rh] && j <= lim && j >= p.kmin) ? x : NEG_INF;
          mx = fmaxf(mx, x);
        }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      const float mn = fmaxf(mrow[rh], mx);
      const float base = mn == NEG_INF ? 0.f : mn;
      alpha[rh] = ex2(mrow[rh] - base);
      mrow[rh] = mn;
      float sum = 0.f;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          float& x = s[nt][2 * rh + e];
          x = ex2(x - base);
          sum += x;
        }
      lrow[rh] = lrow[rh] * alpha[rh] + sum;
    }
#pragma unroll
    for (int dn = 0; dn < 16; ++dn) {
      o[dn][0] *= alpha[0];
      o[dn][1] *= alpha[0];
      o[dn][2] *= alpha[1];
      o[dn][3] *= alpha[1];
    }
    // O += P V (P from the S accumulators, FP16)
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t pa[4];
      pa[0] = pack_h2(s[2 * kk][0], s[2 * kk][1]);
      pa[1] = pack_h2(s[2 * kk][2], s[2 * kk][3]);
      pa[2] = pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
      pa[3] = pack_h2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
      for (int dn = 0; dn < 16; dn += 2) {
        const int mi = lane >> 3, ri = lane & 7;
        const uint32_t a = vt + (8 * (dn + (mi >> 1)) + ri) * (VSTR * 2) + (16 * kk + 8 * (mi & 1)) * 2;
        uint32_t b0, b1, b2, b3;
        ldsm_x4(a, b0, b1, b2, b3);
        mma16816(o[dn], pa, b0, b1);
        mma16816(o[dn + 1], pa, b2, b3);
      }
    }
    __syncthreads();
  }
  // partials: unnormalized O, m (log2), l
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    float l = lrow[rh];
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    if (!rval[rh]) continue;
    const int r = warp * 16 + g8 + 8 * rh;
    const long long base = (static_cast<long long>(item) * p.Hkv + h) * ROWS + r;
    float* op = p.opart + base * D + 2 * c4;
#pragma unroll
    for (int dn = 0; dn < 16; ++dn)
      *reinterpret_cast<float2*>(op + 8 * dn) = make_float2(o[dn][2 * rh], o[dn][2 * rh + 1]);
    if (c4 == 0) *reinterpret_cast<float2*>(p.ml + base * 2) = make_float2(mrow[rh], l);
  }
}

// ---------------------------------------------------------------------------------------------- k_mq2 (wgmma)
// CTA = 3 warpgroups: WG0 producer (cp.async page loads into a raw ring of NR2 blocks, conversion into FP16 wgmma
// tiles, NS2-stage ring), WG1 / WG2 consumers (64 rows each: 128 rows = 128 / G tokens x G q heads).  Per 64-key
// block a consumer runs QK as 8 wgmma m64n64k16 (q and K from shared memory) and PV as 4 wgmma m64n128k16 (P from
// registers, V^T from shared memory); online softmax in fp32 between them (masking only on boundary blocks).
// Producer / consumer hand-over by named barriers: FULL[s] (producer arrives, consumers sync), EMPTY[s] (consumers
// arrive, producer syncs).  Same numerics as k_mq (FP16 operands, fp32 accumulation).
constexpr int ROWS2 = 128, NS2 = 2, NR2 = 3;
constexpr int T2_Q = 0, T2_K = T2_Q + 2 * 16384, T2_V = T2_K + NS2 * 16384, T2_RAW = T2_V + NS2 * 16384;
template <int NP>
constexpr int t2_bytes() { return T2_RAW + NP * NR2 * PPB * RCH; }   // 96 KB + 31.5 KB per producer warpgroup
constexpr int BAR_PROD = 1, BAR_WG = 3, BAR_FULL = 5, BAR_EMPTY = 5 + NS2;   // named barriers (0 = __syncthreads)
DEVI void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
DEVI void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
DEVI void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
DEVI void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
DEVI void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
DEVI void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
template <int N>
DEVI void acc_sync(float (&d)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(d[i])::"memory");
}
DEVI uint64_t sdesc(uint32_t saddr) {  // no swizzle, K-major core matrices: LBO (K) = 128 B, SBO (M / N) = 256 B
  return static_cast<uint64_t>((saddr & 0x3FFFF) >> 4) | (static_cast<uint64_t>(128 >> 4) << 16) |
         (static_cast<uint64_t>(256 >> 4) << 32);
}
#include "mq_wgmma.h"

// NP producer warpgroups (1 or 2): producer w converts the blocks b = w, w + NP, ... (more warps hide the converters'
// shared-memory latency); with NP = 2 and NS2 = 2 stage s is always filled by producer s.
template <bool KONLY, int NP, bool B2 = false>
__global__ void __launch_bounds__(128 * (NP + 2), 1) k_mq2(const Params p) {
  constexpr int CH = chunk_of<B2>();
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sb = smem_u32(smem);
  const int tid = threadIdx.x, wg = tid >> 7, wt = tid & 127;
  const int item = blockIdx.x, h = blockIdx.y;
  const int* it = p.items + ITEM_INTS * item;
  const int* tl = p.tiles + TILE_INTS * it[0];
  const int req = tl[0], crow0 = tl[1], ntok = tl[3], pos0 = tl[4], nk = tl[5], ext = tl[8];
  const int G = p.G, Hq = p.Hkv * G, TQ = ROWS2 / G;
  const int kend = min(min(nk, it[2] * BN), max(pos0 + ntok - p.shift, ext + 1));
  const int kb0 = it[1];
  const int kb1 = kend > kb0 * BN ? (kend + BN - 1) / BN : kb0;
  const int nblk = kb1 - kb0;
  if (wg < NP) {
    // ------------------------------------------------------------------ producer
    const int pw = wg;
    const int* bt = p.bt + static_cast<long long>(req) * p.bt_stride;
    const uint8_t* poolh = p.pool + static_cast<long long>(h) * CH;
    const uint32_t rawb = sb + T2_RAW + pw * (NR2 * PPB * RCH);
    auto load_raw = [&](int i) {                     // i-th block of this producer: b = pw + NP i
      const int b = pw + NP * i;
      if (b < nblk) {
        const int kb = kb0 + b;
        const int npg = min(PPB, (kend - kb * BN + PAGE - 1) / PAGE);
        const uint32_t dst = rawb + (i % NR2) * (PPB * RCH);
        for (int s = wt; s < npg * (CH / 16); s += 128) {
          const int pg = s / (CH / 16), off = (s % (CH / 16)) * 16;
          const long long page = bt[kb * PPB + pg];
          cp16(dst + pg * RCH + rdst<B2>(off), poolh + page * p.page_stride + off);
        }
      }
      cp_commit();                                  // (possibly empty group: keeps the wait_group count uniform)
    };
#pragma unroll 1
    for (int i = 0; i < NR2 - 1; ++i) load_raw(i);
#pragma unroll 1
    for (int i = 0; pw + NP * i < nblk; ++i) {
      const int b = pw + NP * i;
      load_raw(i + NR2 - 1);                         // prefetch NR2 - 1 blocks ahead
      asm volatile("cp.async.wait_group %0;\n" ::"n"(NR2 - 1) : "memory");
      bar_sync(BAR_PROD + pw, 128);                  // block b's raw pages visible to the whole producer WG
      const int s = b % NS2;
      if (b >= NS2) bar_sync(BAR_EMPTY + s, 384);    // consumers done with stage s (block b - NS2)
      const uint32_t raw = rawb + (i % NR2) * (PPB * RCH);
      const int nvalid = kend - (kb0 + b) * BN;
      if (p.dbg != 1) {
        convert_k<KONLY, 1, B2>(raw, sb + T2_K + s * 16384, nvalid, wt);
        convert_v<1, B2>(raw, sb + T2_V + s * 16384, nvalid, wt);
      }
      fence_async_smem();                            // generic-proxy tile stores -> wgmma (async proxy) reads
      bar_arrive(BAR_FULL + s, 384);
      bar_sync(BAR_PROD + pw, 128);                  // raw slot i % NR2 is refilled next iteration
    }
    cp_wait_all();
    return;
  }
  // -------------------------------------------------------------------- consumers
  const int c = wg - NP;                             // rows 64 c .. 64 c + 63
  const int warp = wt >> 5, lane = wt & 31, g8 = lane >> 2, c4 = lane & 3;
  const uint32_t qs = sb + T2_Q + c * 16384;
  for (int i = wt; i < 64 * (D / 8); i += 128) {
    const int rl = i >> 4, c16 = i & 15, r = 64 * c + rl, t = r / G;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (r < TQ * G && t < ntok)
      v = *reinterpret_cast<const uint4*>(p.q + (static_cast<long long>(crow0 + t) * Hq + h * G + r % G) * D + 8 * c16);
    sts128(qs + (c16 >> 1) * 2048 + (rl >> 3) * 256 + (c16 & 1) * 128 + (rl & 7) * 16, v.x, v.y, v.z, v.w);
  }
  fence_async_smem();
  bar_sync(BAR_WG + c, 128);
  int rpos[2];
  bool rval[2];
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    const int r = 64 * c + warp * 16 + g8 + 8 * rh;
    const int t = r / G;
    rval[rh] = r < TQ * G && t < ntok;
    rpos[rh] = max(pos0 + t - p.shift, ext);
  }
  // no masking needed on a block whose last key every valid row sees (first token = smallest limit)
  const int lim_all = min(kend - 1, max(pos0 - p.shift, ext));
  float o[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) o[i] = 0.f;
  float mrow[2] = {NEG_INF, NEG_INF}, lrow[2] = {0.f, 0.f};
#pragma unroll 1
  for (int b = 0; b < nblk; ++b) {
    const int s = b % NS2, kb = kb0 + b;
    bar_sync(BAR_FULL + s, 384);
    float sc[32];
    const uint32_t kt = sb + T2_K + s * 16384, vtl = sb + T2_V + s * 16384;
    if (p.dbg != 2) {
      wg_fence();
      wg_ss_f16_n64_0(sc, sdesc(qs), sdesc(kt));
#pragma unroll
      for (int ks = 1; ks < 8; ++ks) wg_ss_f16_n64_1(sc, sdesc(qs + ks * 2048), sdesc(kt + ks * 2048));
      wg_commit();
      wg_wait0();
      acc_sync(sc);
    } else {
#pragma unroll
      for (int i = 0; i < 32; ++i) sc[i] = 0.f;
    }
    const bool edge = kb * BN + BN - 1 > lim_all || kb * BN < p.kmin;
    const int j0 = kb * BN + 2 * c4;
    float alpha[2];
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      float mx = NEG_INF;
      if (edge) {
        const int lim = min(kend - 1, rpos[rh]);
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            float& x = sc[4 * nt + 2 * rh + e];
            x = (rval[rh] && j0 + 8 * nt + e <= lim && j0 + 8 * nt + e >= p.kmin) ? x : NEG_INF;
            mx = fmaxf(mx, x);
          }
      } else {
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) mx = fmaxf(mx, fmaxf(sc[4 * nt + 2 * rh], sc[4 * nt + 2 * rh + 1]));
      }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      const float mn = fmaxf(mrow[rh], mx);
      const float base = mn == NEG_INF ? 0.f : mn;
      alpha[rh] = ex2(mrow[rh] - base);
      mrow[rh] = mn;
      float sum = 0.f;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          float& x = sc[4 * nt + 2 * rh + e];
          x = ex2(x - base);
          sum += x;
        }
      lrow[rh] = lrow[rh] * alpha[rh] + sum;
    }
#pragma unroll
    for (int dn = 0; dn < 16; ++dn) {
      o[4 * dn] *= alpha[0];
      o[4 * dn + 1] *= alpha[0];
      o[4 * dn + 2] *= alpha[1];
      o[4 * dn + 3] *= alpha[1];
    }
    uint32_t pa[4][4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      pa[kk][0] = pack_h2(sc[8 * kk], sc[8 * kk + 1]);
      pa[kk][1] = pack_h2(sc[8 * kk + 2], sc[8 * kk + 3]);
      pa[kk][2] = pack_h2(sc[8 * kk + 4], sc[8 * kk + 5]);
      pa[kk][3] = pack_h2(sc[8 * kk + 6], sc[8 * kk + 7]);
    }
    if (p.dbg != 2) {
      wg_fence();
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) wg_rs_f16_n128(o, pa[kk], sdesc(vtl + kk * 4096));
      wg_commit();
      wg_wait0();
      acc_sync(o);
    } else {
      o[0] += __uint_as_float(pa[0][0] & pa[1][1] & pa[2][2] & pa[3][3]) * 1e-30f;
    }
    if (b + NS2 < nblk) bar_arrive(BAR_EMPTY + s, 384);   // the producer waits for it before refilling stage s
  }
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    float l = lrow[rh];
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    if (!rval[rh]) continue;
    const int r = 64 * c + warp * 16 + g8 + 8 * rh;
    const long long base = (static_cast<long long>(item) * p.Hkv + h) * ROWS2 + r;
    float* op = p.opart + base * D + 2 * c4;
#pragma unroll
    for (int dn = 0; dn < 16; ++dn)
      *reinterpret_cast<float2*>(op + 8 * dn) = make_float2(o[4 * dn + 2 * rh], o[4 * dn + 2 * rh + 1]);
    if (c4 == 0) *reinterpret_cast<float2*>(p.ml + base * 2) = make_float2(mrow[rh], l);
  }
}

// one warp per (tile, KV head, row): (M, L) with the lanes over the items, then O (lane = 4 dims)
__global__ void __launch_bounds__(128) k_mq_merge(const int* tiles, const float* opart, const float* ml, int Hkv,
                                                  int G, int rows, void* out, long long o_ts, int out_fp32, float* lse,
                                                  long long lse_ts) {
  const int* tl = tiles + TILE_INTS * blockIdx.x;
  const int h = blockIdx.y;
  const int crow0 = tl[1], orow0 = tl[2], ntok = tl[3], item0 = tl[6], nitem = tl[7];
  const int lane = threadIdx.x & 31;
  const int r = blockIdx.z * 4 + (threadIdx.x >> 5);
  const int TQ = rows / G, Hq = Hkv * G;
  const int t = r / G;
  if (r >= TQ * G || t >= ntok) return;
  const int hq = h * G + r % G;
  const long long b0 = (static_cast<long long>(item0) * Hkv + h) * rows + r;    // item s: b0 + s * Hkv * rows
  const long long bs = static_cast<long long>(Hkv) * rows;
  float M = NEG_INF;
  for (int s = lane; s < nitem; s += 32) M = fmaxf(M, ml[2 * (b0 + s * bs)]);
#pragma unroll
  for (int m = 16; m >= 1; m >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, m));
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  float L = 0.f;
  if (M != NEG_INF) {
    for (int s = lane; s < nitem; s += 32) {
      const float2 v = *reinterpret_cast<const float2*>(ml + 2 * (b0 + s * bs));
      L += ex2(v.x - M) * v.y;
    }
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) L += __shfl_xor_sync(0xffffffffu, L, m);
#pragma unroll 4
    for (int s = 0; s < nitem; ++s) {
      const long long b = b0 + s * bs;
      const float w = ex2(ml[2 * b] - M);
      const float4 x = *reinterpret_cast<const float4*>(opart + b * D + 4 * lane);
      acc.x += w * x.x;
      acc.y += w * x.y;
      acc.z += w * x.z;
      acc.w += w * x.w;
    }
    const float inv = 1.f / L;
    acc.x *= inv;
    acc.y *= inv;
    acc.z *= inv;
    acc.w *= inv;
  }
  if (out_fp32) {
    float* dst = static_cast<float*>(out) + (static_cast<long long>(crow0 + t) * Hq + hq) * D + 4 * lane;
    *reinterpret_cast<float4*>(dst) = acc;
  } else {
    __nv_bfloat16* dst = static_cast<__nv_bfloat16*>(out) + static_cast<long long>(orow0 + t) * o_ts + hq * D + 4 * lane;
    const __nv_bfloat162 a = __floats2bfloat162_rn(acc.x, acc.y), b = __floats2bfloat162_rn(acc.z, acc.w);
    uint2 v;
    v.x = *reinterpret_cast<const uint32_t*>(&a);
    v.y = *reinterpret_cast<const uint32_t*>(&b);
    *reinterpret_cast<uint2*>(dst) = v;
  }
  if (lse != nullptr && lane == 0)
    lse[hq * lse_ts + crow0 + t] = M == NEG_INF ? NEG_INF : (M + log2f(L)) * 0.6931471805599453f;
}

// q -> FP16 in the keys' stored basis, times softmax scale * log2(e).  One warp per (row, q head), lane = channels
// 4 l .. 4 l + 3.  Static FlagRot: whole-head H128 / 8 if kfull[kv head], then orthonormal H32 on the block-flagged
// groups (kmask[kv head][g]) = the writer's order (k -> H k / 16 outside / fused, then block flags).
DEVI void fwht_lanes(float (&x)[4], int lane, int span_lanes) {
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

__global__ void k_mq_prep(const void* q, int q_fp32, long long q_ts, const int* idx, int m, int Hq, int G,
                          const uint8_t* kfull, const uint8_t* kmask, float qscale, __half* q16) {
  const long long w = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (w >= static_cast<long long>(m) * Hq) return;
  const int t = static_cast<int>(w / Hq), hq = static_cast<int>(w % Hq), hk = hq / G;
  const long long row = idx != nullptr ? idx[t] : t;
  float x[4];
  if (q_fp32) {
    const float4 v = *reinterpret_cast<const float4*>(static_cast<const float*>(q) + row * q_ts + hq * D + 4 * lane);
    x[0] = v.x; x[1] = v.y; x[2] = v.z; x[3] = v.w;
  } else {
    const uint2 raw = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(q) + row * q_ts + hq * D +
                                                      4 * lane);
    const __nv_bfloat162 p0 = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
    const __nv_bfloat162 p1 = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
    x[0] = __low2float(p0); x[1] = __high2float(p0); x[2] = __low2float(p1); x[3] = __high2float(p1);
  }
  if (kfull && kfull[hk]) {
    fwht_lanes(x, lane, 32);
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] *= 0.125f;
  }
  if (kmask) {                       // all lanes transform (shuffles stay inside each 8-lane group), then select
    const bool f = kmask[hk * 4 + (lane >> 3)] != 0;
    float y[4] = {x[0], x[1], x[2], x[3]};
    fwht_lanes(y, lane, 8);
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = f ? y[i] * 0.17677669529663687f : x[i];
  }
  uint2 v;
  v.x = pack_h2(x[0] * qscale, x[1] * qscale);
  v.y = pack_h2(x[2] * qscale, x[3] * qscale);
  *reinterpret_cast<uint2*>(q16 + (static_cast<long long>(t) * Hq + hq) * D + 4 * lane) = v;
}

// ---------------------------------------------------------------------------------------------- k_mq3 (head 256)
// Phase 2j: head size 256 (Qwen3.5 / 3.8 full-attention layers).  A (page, KV head) chunk = two 128-d half chunks of
// the head-128 layout (4608 B): dims 0-127 in half 0, 128-255 in half 1; key-only 2x2: the H32
// bit of group g (0-7) = sign of its K scale word (half g / 4, group g % 4), the whole-head H256 bit = sign of the
// token's group-0 V scale word in half 0.  Same algorithm and numerics as k_mq2; register / shared-memory budget:
//   CTA = 1 producer warpgroup + 2 consumer warpgroups (384 threads, 1 CTA / SM).  A consumer holds 64 rows (128
//         rows per CTA = 128 / G tokens x G q heads, G = 6: 21 tokens) and O 64 x 256 fp32 = 128 registers per
//         thread + S 64 x 64 (32), so the warpgroups rebalance their registers with setmaxnreg: launch 168 per thread,
//         producer -> 120, consumers -> 192 (ptxas: no spills).  PV as two m64n128k16 per k-step (an m64n256
//         instruction needs a kernel register target >= 158).
//   smem  q 2 x 32 KB | K tiles NS3 x 32 KB | V^T tiles NS3 x 32 KB | raw ring NR3 half blocks (2 pages x 2 half
//         chunks, padded: 10.5 KB) = 224.3 KB.  The producer loads and converts half blocks of 32 keys: K with 4
//         threads per key (two 32-d groups each; lanes key-major, so a 16-B store phase covers 8 keys = 8 bank
//         quads), key-only 2x2 whole-head keys: H8 over the 8 groups = an in-thread H2 + two lane butterflies (16
//         shuffles issued per round); V^T with 8 dims per thread, dim loop rotated by lane (conflict-free stores).
//         The consumers run QK 16 x wgmma m64n64k16 and 8 x m64n128k16 PV per 64 keys, the O rescale skipped when
//         no row max of the warp changed.
// Tuning (H20, 1024 new tokens over 32k, Qwen3.8 shapes): RTN is consumer-bound (7.5 ms = FA3
// BF16 on dequantized K / V, 7.2 ms); the key-only 2x2 converter went 9.6 -> 6.3 ms alone (kernel 11.4 -> 8.1 ms)
// with the conflict-free mappings, the batched shuffles and the FMA first FWHT stage.  Two producer warpgroups
// (64 registers each) spill and were slower.
constexpr int D3 = 256, CHUNK3 = 2 * CHUNK, NS3 = 2, NR3 = 3, HB_RAW = 2 * 2 * RCH;
constexpr int T3_Q = 0, T3_K = T3_Q + 2 * 32768, T3_V = T3_K + NS3 * 32768, T3_RAW = T3_V + NS3 * 32768,
              T3_BYTES = T3_RAW + NR3 * HB_RAW;   // 229,632 B
static_assert(T3_BYTES <= 232448, "k_mq3 shared memory");
template <int NP>
struct Mq3Regs;   // setmaxnreg targets: NP * 128 * PREG + 256 * CREG <= registers at launch (65536 / 128 (NP + 2))
template <>
struct Mq3Regs<1> { static constexpr int PREG = 120, CREG = 192; };   // ptxas: no spills (168 at launch)
static_assert(128 * Mq3Regs<1>::PREG + 256 * Mq3Regs<1>::CREG <= 168 * 384, "k_mq3 register split");
constexpr int B3_PROD = 1, B3_WG = 2, B3_FULL = 4, B3_EMPTY = 4 + NS3;   // named barriers (0 = __syncthreads)

// K of 32 keys of a half block -> FP16 K tile rows kb0 + key, wgmma no-swizzle K-major core matrices
// [ch / 16][key / 8][(ch / 8) % 2][key % 8][16 B] (16 slabs of 2 KB).  NGT = 32-d groups per thread (2: 4 threads
// per key, 1: 8 threads per key); thread q of a key holds groups NGT q .. NGT q + NGT - 1 of 8.  Lanes are key-major
// (lane = key + KPW q): the 8 lanes of a 16-B store phase write 8 different keys, i.e. 8 different bank quads.
template <bool KONLY, int NGT, bool B2 = false>
DEVI void convert_k256(uint32_t raw, uint32_t ks, int kb0, int nvalid, int t) {
  constexpr int TPK = 8 / NGT, KPW = 32 / TPK;        // threads per key, keys per warp
  const int lane = t & 31;
  const int key = KPW * (t >> 5) + lane % KPW, q = lane / KPW, grp0 = NGT * q, hh = grp0 >> 2;
  const uint32_t pg = raw + (key >> 4) * (2 * RCH);   // the key's page, half chunk 0
  const uint32_t pr = pg + hh * RCH;                   // this thread's half chunk
  const int tok = key & 15;
  bool full = false;
  if (KONLY) {
    const int qs = 8 * (tok >> 3) + ((tok & 7) >> 1) + 4 * (tok & 1);   // V scale slot of the token, group 0
    full = (lds16(pg + R_VS + 2 * qs) >> 15) != 0;
  }
  constexpr float R32 = 0.17677669529663687f, FN = 0.0625f;   // 1 / sqrt 32, 1 / sqrt 256
  uint32_t o[NGT][16];
#pragma unroll
  for (int gg = 0; gg < NGT; ++gg) {
    const int g = (grp0 + gg) & 3;                     // group inside the half chunk
    const uint32_t sw = lds16(pr + R_KS + tok * 8 + 2 * g);
    const uint32_t sbits = key < nvalid ? (sw & 0x7FFFu) : 0u;
    const uint32_t s2 = sbits * 0x10001u;
    const uint32_t zz = 0x64006400u | ((sw & zmask<B2>()) * 0x10001u);
    uint32_t y[16];
#pragma unroll
    for (int tg = 0; tg < 4; ++tg) {
      uint32_t l, hb;
      k_codes<B2>(pr, tok, tg, g, l, hb);
      y[2 * tg] = hsub2_u(__byte_perm(l, 0, 0x4140) | 0x64006400u, zz);
      y[2 * tg + 1] = hsub2_u(__byte_perm(l, 0, 0x4342) | 0x64006400u, zz);
      y[8 + 2 * tg] = hsub2_u(__byte_perm(hb, 0, 0x4140) | 0x64006400u, zz);
      y[8 + 2 * tg + 1] = hsub2_u(__byte_perm(hb, 0, 0x4342) | 0x64006400u, zz);
    }
    if (!KONLY) {
#pragma unroll
      for (int j = 0; j < 16; ++j) o[gg][j] = hmul2_u(y[j], s2);
    } else {
      // per group: H32 / sqrt 32 (sign bit of the K scale word), or (whole-head key) s H32(y) as the first stage of
      // the orthonormal H256 = H8 x H32 (H8 over the 8 groups below), or none
      const bool lk = ((sw >> 15) & 1u) != 0 && !full;
      const uint32_t sb = h2_dup(__half2float(__ushort_as_half(static_cast<unsigned short>(sbits))) * R32);
      const uint32_t sz = full ? s2 : sb;
      if (full || lk) {
        fwht32_h2f(y);                                 // exact: integers |.| <= 480
#pragma unroll
        for (int j = 0; j < 16; ++j) o[gg][j] = hmul2_u(y[j], sz);
      } else {
#pragma unroll
        for (int j = 0; j < 16; ++j) o[gg][j] = hmul2_u(y[j], s2);
      }
    }
  }
  if (KONLY && __any_sync(0xffffffffu, full)) {      // the TPK lanes of a key share `full`
    // H8 over the 8 groups: (NGT = 2) in-thread H2 over the thread's groups (group bit 0), then one butterfly per
    // remaining group bit over the key's lanes; each round issues its 16 x NGT shuffles before the adds
    const uint32_t fn = h2_dup(FN);
    uint32_t a[NGT][16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      if (NGT == 2) {
        a[0][j] = hadd2_u(o[0][j], o[NGT - 1][j]);
        a[NGT - 1][j] = hsub2_u(o[0][j], o[NGT - 1][j]);
      } else {
        a[0][j] = o[0][j];
      }
    }
#pragma unroll
    for (int m = 1; m < TPK; m <<= 1) {
      const bool hi = (q & m) != 0;                    // hi lane: t - a, lo lane: a + t
#pragma unroll
      for (int gg = 0; gg < NGT; ++gg) {
        uint32_t t[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) t[j] = __shfl_xor_sync(0xffffffffu, a[gg][j], m * KPW);
#pragma unroll
        for (int j = 0; j < 16; ++j)
          a[gg][j] = hi ? hsub2_u(t[j], a[gg][j]) : hadd2_u(a[gg][j], t[j]);
      }
    }
    if (full) {
#pragma unroll
      for (int gg = 0; gg < NGT; ++gg)
#pragma unroll
        for (int j = 0; j < 16; ++j) o[gg][j] = hmul2_u(a[gg][j], fn);
    }
  }
  const int kr = kb0 + key;
  const uint32_t row = ks + (kr >> 3) * 256 + (kr & 7) * 16;
#pragma unroll
  for (int gg = 0; gg < NGT; ++gg)
#pragma unroll
    for (int c8 = 0; c8 < 4; ++c8) {
      const int ch8 = 4 * (grp0 + gg) + c8;            // 8-channel chunk (0..31)
      sts128(row + (ch8 >> 1) * 2048 + (ch8 & 1) * 128, o[gg][4 * c8], o[gg][4 * c8 + 1], o[gg][4 * c8 + 2],
             o[gg][4 * c8 + 3]);
    }
}

// V^T of a half block (2 pages x 2 half chunks): thread t % 128 -> (page pp, half hh, token octet c8, value group g)
// and 8 dims of the group as convert_v (NP producer warpgroups: warpgroup t / 128 takes 8 / NP of them).  Tile
// layout [key / 16][dim / 8][(key / 8) % 2][dim % 8][16 B] (4 slabs of 8 KB); the dim loop starts at lane % 8, so the
// 8 lanes of a 16-B store phase write 8 different dims mod 8 (bank quads).
template <int NP, bool B2 = false>
DEVI void convert_v256(uint32_t raw, uint32_t vt, int kb0, int nvalid, int t) {
  const int tid = t & 127, wsel = t >> 7;
  const int cmb = tid >> 2, u = tid & 3;
  const int x = cmb >> 3, c8 = (cmb >> 2) & 1, g = cmb & 3;
  const int pp = x >> 1, hh = x & 1;
  const uint32_t pr = raw + (2 * pp + hh) * RCH;
  const uint4 sv = lds128(pr + R_VS + g * 32 + 16 * c8);
  const int nv = min(max(nvalid - (16 * pp + 8 * c8), 0), 8);
  uint32_t mag[4], zz[4];
  {
    uint32_t sp[4];
    sp[0] = __byte_perm(sv.x, sv.z, 0x5410);
    sp[1] = __byte_perm(sv.x, sv.z, 0x7632);
    sp[2] = __byte_perm(sv.y, sv.w, 0x5410);
    sp[3] = __byte_perm(sv.y, sv.w, 0x7632);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      zz[j] = 0x64006400u | (sp[j] & (zmask<B2>() * 0x10001u));
      mag[j] = sp[j] & 0x7FFF7FFFu;
      if (2 * j >= nv) mag[j] &= 0xFFFF0000u;
      if (2 * j + 1 >= nv) mag[j] &= 0x0000FFFFu;
    }
  }
  const int sub = 2 * (g >> 1) + (u & 1), w = 2 * (g & 1) + (u >> 1);
  const int dim0 = 128 * hh + 32 * g + 8 * u;
  const int kr = kb0 + 16 * pp;                         // first key of the page in the block
  constexpr int NG = 8 / NP;
#pragma unroll
  for (int gi = 0; gi < NG; ++gi) {
    const int gid = NG * wsel + ((gi + t) & (NG - 1));
    uint32_t l, hb;
    v_codes<B2>(pr, w, gid, c8, sub, l, hb);
    uint32_t o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t c = (__byte_perm(l, hb, static_cast<uint32_t>(j | ((4 + j) << 8))) & 0x000F000Fu) | 0x64006400u;
      o[j] = hmul2_u(hsub2_u(c, zz[j]), mag[j]);
    }
    const int dim = dim0 + gid;
    sts128(vt + (kr >> 4) * 8192 + (dim >> 3) * 256 + c8 * 128 + (dim & 7) * 16, o[0], o[1], o[2], o[3]);
  }
}

template <bool KONLY, int NP, bool B2 = false>
__global__ void __launch_bounds__(128 * (NP + 2), 1) k_mq3(const Params p) {
  constexpr int CH = chunk_of<B2>();
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int NPT = 128 * NP, NBAR = NPT + 256;
  const uint32_t sb = smem_u32(smem);
  const int tid = threadIdx.x, wg = tid >> 7, wt = tid & 127;
  const int item = blockIdx.x, h = blockIdx.y;
  const int* it = p.items + ITEM_INTS * item;
  const int* tl = p.tiles + TILE_INTS * it[0];
  const int req = tl[0], crow0 = tl[1], ntok = tl[3], pos0 = tl[4], nk = tl[5], ext = tl[8];
  const int G = p.G, Hq = p.Hkv * G, TQ = ROWS2 / G;
  const int kend = min(min(nk, it[2] * BN), max(pos0 + ntok - p.shift, ext + 1));
  const int kb0 = it[1];
  const int kb1 = kend > kb0 * BN ? (kend + BN - 1) / BN : kb0;
  const int nblk = kb1 - kb0;
  if (wg < NP) {
    // ------------------------------------------------------------------ producers (half blocks of 32 keys)
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(Mq3Regs<NP>::PREG));
    const int* bt = p.bt + static_cast<long long>(req) * p.bt_stride;
    const uint8_t* poolh = p.pool + static_cast<long long>(h) * (2 * CH);
    const uint32_t rawb = sb + T3_RAW;
    const int nhb = 2 * nblk;
    auto load_raw = [&](int i) {                      // half block i: keys kb0 * 64 + 32 i ..
      if (i < nhb) {
        const int key0 = kb0 * BN + 32 * i;
        const int npg = min(2, (kend - key0 + PAGE - 1) / PAGE);
        const uint32_t dst = rawb + (i % NR3) * HB_RAW;
        for (int s = tid; s < npg * (2 * CH / 16); s += NPT) {
          const int pg = s / (2 * CH / 16), off = (s % (2 * CH / 16)) * 16;
          const int hh = off >= CH;
          const long long page = bt[key0 / PAGE + pg];
          cp16(dst + (2 * pg + hh) * RCH + rdst<B2>(off - hh * CH), poolh + page * p.page_stride + off);
        }
      }
      cp_commit();                                    // (possibly empty group: keeps the wait_group count uniform)
    };
#pragma unroll 1
    for (int i = 0; i < NR3 - 1; ++i) load_raw(i);
#pragma unroll 1
    for (int b = 0; b < nblk; ++b) {
      const int s = b % NS3;
#pragma unroll 1
      for (int hb = 0; hb < 2; ++hb) {
        const int i = 2 * b + hb;
        load_raw(i + NR3 - 1);                         // prefetch NR3 - 1 half blocks ahead
        asm volatile("cp.async.wait_group %0;\n" ::"n"(NR3 - 1) : "memory");
        bar_sync(B3_PROD, NPT);                        // half block i's raw pages visible to all producer threads
        if (hb == 0 && b >= NS3) bar_sync(B3_EMPTY + s, NBAR);   // consumers done with stage s (block b - NS3)
        const uint32_t raw = rawb + (i % NR3) * HB_RAW;
        const int nvalid = kend - ((kb0 + b) * BN + 32 * hb);
        if (p.dbg != 1) {
          convert_k256<KONLY, 2 / NP, B2>(raw, sb + T3_K + s * 32768, 32 * hb, nvalid, tid);
          convert_v256<NP, B2>(raw, sb + T3_V + s * 32768, 32 * hb, nvalid, tid);
        }
        bar_sync(B3_PROD, NPT);                        // raw slot i % NR3 is refilled next iteration
      }
      fence_async_smem();                              // generic-proxy tile stores -> wgmma (async proxy) reads
      bar_arrive(B3_FULL + s, NBAR);
    }
    cp_wait_all();
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(Mq3Regs<NP>::CREG));
  // -------------------------------------------------------------------- consumers
  const int c = wg - NP;                             // rows 64 c .. 64 c + 63
  const int warp = wt >> 5, lane = wt & 31, g8 = lane >> 2, c4 = lane & 3;
  const uint32_t qs = sb + T3_Q + c * 32768;
  for (int i = wt; i < 64 * (D3 / 8); i += 128) {
    const int rl = i >> 5, c16 = i & 31, r = 64 * c + rl, t = r / G;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (r < TQ * G && t < ntok)
      v = *reinterpret_cast<const uint4*>(p.q + (static_cast<long long>(crow0 + t) * Hq + h * G + r % G) * D3 + 8 * c16);
    sts128(qs + (c16 >> 1) * 2048 + (rl >> 3) * 256 + (c16 & 1) * 128 + (rl & 7) * 16, v.x, v.y, v.z, v.w);
  }
  fence_async_smem();
  bar_sync(B3_WG + c, 128);
  int rpos[2];
  bool rval[2];
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    const int r = 64 * c + warp * 16 + g8 + 8 * rh;
    const int t = r / G;
    rval[rh] = r < TQ * G && t < ntok;
    rpos[rh] = max(pos0 + t - p.shift, ext);
  }
  const int lim_all = min(kend - 1, max(pos0 - p.shift, ext));
  float o[2][64];                                    // dims 0-127 | 128-255 (two m64n128 accumulators)
#pragma unroll
  for (int i = 0; i < 64; ++i) o[0][i] = o[1][i] = 0.f;
  float mrow[2] = {NEG_INF, NEG_INF}, lrow[2] = {0.f, 0.f};
#pragma unroll 1
  for (int b = 0; b < nblk; ++b) {
    const int s = b % NS3, kb = kb0 + b;
    bar_sync(B3_FULL + s, NBAR);
    float sc[32];
    const uint32_t kt = sb + T3_K + s * 32768, vtl = sb + T3_V + s * 32768;
    if (p.dbg != 2) {
      wg_fence();
      wg_ss_f16_n64_0(sc, sdesc(qs), sdesc(kt));
#pragma unroll
      for (int ks = 1; ks < 16; ++ks) wg_ss_f16_n64_1(sc, sdesc(qs + ks * 2048), sdesc(kt + ks * 2048));
      wg_commit();
      wg_wait0();
      acc_sync(sc);
    } else {
#pragma unroll
      for (int i = 0; i < 32; ++i) sc[i] = 0.f;
    }
    const bool edge = kb * BN + BN - 1 > lim_all || kb * BN < p.kmin;
    const int j0 = kb * BN + 2 * c4;
    float alpha[2];
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      float mx = NEG_INF;
      if (edge) {
        const int lim = min(kend - 1, rpos[rh]);
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            float& x = sc[4 * nt + 2 * rh + e];
            x = (rval[rh] && j0 + 8 * nt + e <= lim && j0 + 8 * nt + e >= p.kmin) ? x : NEG_INF;
            mx = fmaxf(mx, x);
          }
      } else {
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) mx = fmaxf(mx, fmaxf(sc[4 * nt + 2 * rh], sc[4 * nt + 2 * rh + 1]));
      }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      const float mn = fmaxf(mrow[rh], mx);
      const float base = mn == NEG_INF ? 0.f : mn;
      alpha[rh] = ex2(mrow[rh] - base);
      mrow[rh] = mn;
      float sum = 0.f;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          float& x = sc[4 * nt + 2 * rh + e];
          x = ex2(x - base);
          sum += x;
        }
      lrow[rh] = lrow[rh] * alpha[rh] + sum;
    }
    if (__any_sync(0xffffffffu, alpha[0] != 1.f || alpha[1] != 1.f)) {   // skip the rescale when no row max changed
#pragma unroll
      for (int hf = 0; hf < 2; ++hf)
#pragma unroll
        for (int dn = 0; dn < 16; ++dn) {
          o[hf][4 * dn] *= alpha[0];
          o[hf][4 * dn + 1] *= alpha[0];
          o[hf][4 * dn + 2] *= alpha[1];
          o[hf][4 * dn + 3] *= alpha[1];
        }
    }
    uint32_t pa[4][4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      pa[kk][0] = pack_h2(sc[8 * kk], sc[8 * kk + 1]);
      pa[kk][1] = pack_h2(sc[8 * kk + 2], sc[8 * kk + 3]);
      pa[kk][2] = pack_h2(sc[8 * kk + 4], sc[8 * kk + 5]);
      pa[kk][3] = pack_h2(sc[8 * kk + 6], sc[8 * kk + 7]);
    }
    if (p.dbg != 2) {
      wg_fence();
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {    // m64n256 needs a 158-register kernel target: two m64n128 per k-step
        wg_rs_f16_n128(o[0], pa[kk], sdesc(vtl + kk * 8192));
        wg_rs_f16_n128(o[1], pa[kk], sdesc(vtl + kk * 8192 + 4096));
      }
      wg_commit();
      wg_wait0();
      acc_sync(o[0]);
      acc_sync(o[1]);
    } else {
      o[0][0] += __uint_as_float(pa[0][0] & pa[1][1] & pa[2][2] & pa[3][3]) * 1e-30f;
    }
    if (b + NS3 < nblk) bar_arrive(B3_EMPTY + s, NBAR);   // the producers wait for it before refilling stage s
  }
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    float l = lrow[rh];
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    if (!rval[rh]) continue;
    const int r = 64 * c + warp * 16 + g8 + 8 * rh;
    const long long base = (static_cast<long long>(item) * p.Hkv + h) * ROWS2 + r;
    float* op = p.opart + base * D3 + 2 * c4;
#pragma unroll
    for (int hf = 0; hf < 2; ++hf)
#pragma unroll
      for (int dn = 0; dn < 16; ++dn)
        *reinterpret_cast<float2*>(op + 128 * hf + 8 * dn) = make_float2(o[hf][4 * dn + 2 * rh], o[hf][4 * dn + 2 * rh + 1]);
    if (c4 == 0) *reinterpret_cast<float2*>(p.ml + base * 2) = make_float2(mrow[rh], l);
  }
}

// head 256: merge (lane = dims 4 l .. 4 l + 3 and 128 + 4 l ..), the same arithmetic per dim as k_mq_merge
__global__ void __launch_bounds__(128) k_mq_merge256(const int* tiles, const float* opart, const float* ml, int Hkv,
                                                     int G, int rows, void* out, long long o_ts, int out_fp32,
                                                     float* lse, long long lse_ts) {
  const int* tl = tiles + TILE_INTS * blockIdx.x;
  const int h = blockIdx.y;
  const int crow0 = tl[1], orow0 = tl[2], ntok = tl[3], item0 = tl[6], nitem = tl[7];
  const int lane = threadIdx.x & 31;
  const int r = blockIdx.z * 4 + (threadIdx.x >> 5);
  const int TQ = rows / G, Hq = Hkv * G;
  const int t = r / G;
  if (r >= TQ * G || t >= ntok) return;
  const int hq = h * G + r % G;
  const long long b0 = (static_cast<long long>(item0) * Hkv + h) * rows + r;
  const long long bs = static_cast<long long>(Hkv) * rows;
  float M = NEG_INF;
  for (int s = lane; s < nitem; s += 32) M = fmaxf(M, ml[2 * (b0 + s * bs)]);
#pragma unroll
  for (int m = 16; m >= 1; m >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, m));
  float4 acc[2] = {make_float4(0.f, 0.f, 0.f, 0.f), make_float4(0.f, 0.f, 0.f, 0.f)};
  float L = 0.f;
  if (M != NEG_INF) {
    for (int s = lane; s < nitem; s += 32) {
      const float2 v = *reinterpret_cast<const float2*>(ml + 2 * (b0 + s * bs));
      L += ex2(v.x - M) * v.y;
    }
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) L += __shfl_xor_sync(0xffffffffu, L, m);
#pragma unroll 4
    for (int s = 0; s < nitem; ++s) {
      const long long b = b0 + s * bs;
      const float w = ex2(ml[2 * b] - M);
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const float4 x = *reinterpret_cast<const float4*>(opart + b * D3 + 128 * hf + 4 * lane);
        acc[hf].x += w * x.x;
        acc[hf].y += w * x.y;
        acc[hf].z += w * x.z;
        acc[hf].w += w * x.w;
      }
    }
    const float inv = 1.f / L;
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      acc[hf].x *= inv;
      acc[hf].y *= inv;
      acc[hf].z *= inv;
      acc[hf].w *= inv;
    }
  }
#pragma unroll
  for (int hf = 0; hf < 2; ++hf) {
    if (out_fp32) {
      float* dst = static_cast<float*>(out) + (static_cast<long long>(crow0 + t) * Hq + hq) * D3 + 128 * hf + 4 * lane;
      *reinterpret_cast<float4*>(dst) = acc[hf];
    } else {
      __nv_bfloat16* dst =
          static_cast<__nv_bfloat16*>(out) + static_cast<long long>(orow0 + t) * o_ts + hq * D3 + 128 * hf + 4 * lane;
      const __nv_bfloat162 a = __floats2bfloat162_rn(acc[hf].x, acc[hf].y);
      const __nv_bfloat162 b = __floats2bfloat162_rn(acc[hf].z, acc[hf].w);
      uint2 v;
      v.x = *reinterpret_cast<const uint32_t*>(&a);
      v.y = *reinterpret_cast<const uint32_t*>(&b);
      *reinterpret_cast<uint2*>(dst) = v;
    }
  }
  if (lse != nullptr && lane == 0)
    lse[hq * lse_ts + crow0 + t] = M == NEG_INF ? NEG_INF : (M + log2f(L)) * 0.6931471805599453f;
}

// head 256 q prep: lane = channels 8 l .. 8 l + 7.  Whole-head H256 / 16 = in-thread H8 x H32 over the lanes; block
// H32 on group g = lanes 4 g .. 4 g + 3 = in-thread H8 x H4 over the lanes, / sqrt 32 (writer order as at 128).
DEVI void fwht8_lanes(float (&x)[8], int lane, int span_lanes) {
#pragma unroll
  for (int m = 1; m < 8; m <<= 1)
#pragma unroll
    for (int i = 0; i < 8; ++i)
      if ((i & m) == 0) {
        const float a = x[i], b = x[i + m];
        x[i] = a + b;
        x[i + m] = a - b;
      }
  for (int m = 1; m < span_lanes; m <<= 1) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float y = __shfl_xor_sync(0xffffffffu, x[i], m);
      x[i] = (lane & m) ? y - x[i] : x[i] + y;
    }
  }
}

__global__ void k_mq_prep256(const void* q, int q_fp32, long long q_ts, const int* idx, int m, int Hq, int G,
                             const uint8_t* kfull, const uint8_t* kmask, float qscale, __half* q16) {
  const long long w = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (w >= static_cast<long long>(m) * Hq) return;
  const int t = static_cast<int>(w / Hq), hq = static_cast<int>(w % Hq), hk = hq / G;
  const long long row = idx != nullptr ? idx[t] : t;
  float x[8];
  if (q_fp32) {
    const float* src = static_cast<const float*>(q) + row * q_ts + hq * D3 + 8 * lane;
    const float4 v0 = *reinterpret_cast<const float4*>(src), v1 = *reinterpret_cast<const float4*>(src + 4);
    x[0] = v0.x; x[1] = v0.y; x[2] = v0.z; x[3] = v0.w; x[4] = v1.x; x[5] = v1.y; x[6] = v1.z; x[7] = v1.w;
  } else {
    const uint4 raw = *reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(q) + row * q_ts + hq * D3 +
                                                      8 * lane);
    const uint32_t wd[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const __nv_bfloat162 pr = *reinterpret_cast<const __nv_bfloat162*>(&wd[i]);
      x[2 * i] = __low2float(pr);
      x[2 * i + 1] = __high2float(pr);
    }
  }
  if (kfull && kfull[hk]) {
    fwht8_lanes(x, lane, 32);
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i] *= 0.0625f;
  }
  if (kmask) {                       // all lanes transform (shuffles stay inside each 4-lane group), then select
    const bool f = kmask[hk * 8 + (lane >> 2)] != 0;
    float y[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) y[i] = x[i];
    fwht8_lanes(y, lane, 4);
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i] = f ? y[i] * 0.17677669529663687f : x[i];
  }
  uint4 v;
  v.x = pack_h2(x[0] * qscale, x[1] * qscale);
  v.y = pack_h2(x[2] * qscale, x[3] * qscale);
  v.z = pack_h2(x[4] * qscale, x[5] * qscale);
  v.w = pack_h2(x[6] * qscale, x[7] * qscale);
  *reinterpret_cast<uint4*>(q16 + (static_cast<long long>(t) * Hq + hq) * D3 + 8 * lane) = v;
}

// ---------------------------------------------------------------------------------------------- Phase 2k: BF16 parts
// The BF16 sink / window parts of the INT4-direct prefill (sinks + window modes), batched over every request of a step:
//   k_win_gather  K / V rows of the window FlashAttention call, one destination row per CTA, from three sources
//                 (map = (kind << 32) | row): 0 = sink rows [*, 2 Hkv, DH] (original basis), 1 = the gathered window
//                 rows of the cache update [*, 2 Hkv, DH] (original basis), 2 = the batch's own K / V rows (strided)
//   k_bf_merge    per (compact row, q head): the sink part (keys j < min(S, len), j <= pos - W, fp32 softmax) merged
//                 by LSE with the page part (fp32 o, natural-log lse; -inf = empty) and the window part (FA output
//                 bf16 + lse) -> the bf16 output row idx[c]
template <int DH>
__global__ void __launch_bounds__(128) k_win_gather(const long long* map, int nk, const __nv_bfloat16* sinkb,
                                                    const __nv_bfloat16* preb, const __nv_bfloat16* kch, long long k_ts,
                                                    const __nv_bfloat16* vch, long long v_ts, int Hkv,
                                                    __nv_bfloat16* kw, __nv_bfloat16* vw) {
  const int d = blockIdx.x;
  if (d >= nk) return;
  const long long code = map[d];
  const int kind = static_cast<int>(code >> 32);
  const long long row = code & 0xffffffffLL;
  const int nvec = Hkv * DH / 8;                                    // 16-B vectors per K (or V) row
  const uint4 *ks, *vs;
  if (kind == 2) {
    ks = reinterpret_cast<const uint4*>(kch + row * k_ts);
    vs = reinterpret_cast<const uint4*>(vch + row * v_ts);
  } else {
    const __nv_bfloat16* b = (kind == 0 ? sinkb : preb) + row * 2LL * Hkv * DH;
    ks = reinterpret_cast<const uint4*>(b);
    vs = reinterpret_cast<const uint4*>(b + Hkv * DH);
  }
  uint4* kd = reinterpret_cast<uint4*>(kw + static_cast<long long>(d) * Hkv * DH);
  uint4* vd = reinterpret_cast<uint4*>(vw + static_cast<long long>(d) * Hkv * DH);
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    kd[i] = ks[i];
    vd[i] = vs[i];
  }
}

template <int DH>
__global__ void __launch_bounds__(128) k_bf_merge(int m, int Hq, int Hkv, const __nv_bfloat16* q, long long q_ts,
                                                  const long long* idx, const float* o_p, const float* lse_p,
                                                  const __nv_bfloat16* o_w, const float* lse_w,
                                                  const __nv_bfloat16* sk, const long long* row_req,
                                                  const long long* row_pos, const long long* sk_len, int S, int W,
                                                  float scale, __nv_bfloat16* out, long long o_ts) {
  constexpr int E = DH / 32;                                        // elements per lane (4 | 8)
  const int c = blockIdx.x;
  const int hq = blockIdx.y * 4 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (c >= m || hq >= Hq) return;
  const int G = Hq / Hkv, kvh = hq / G;
  const long long brow = idx[c];
  float acc[E];
  float M = NEG_INF, L = 0.f;                                      // running merge (natural log units)
  // page part
  {
    const float l = lse_p[static_cast<long long>(hq) * m + c];
    const float* op = o_p + (static_cast<long long>(c) * Hq + hq) * DH + lane * E;
#pragma unroll
    for (int e = 0; e < E; ++e) acc[e] = 0.f;
    if (l != NEG_INF && !isinf(l)) {
      M = l;
      L = 1.f;
#pragma unroll
      for (int e = 0; e < E; ++e) acc[e] = op[e];
    }
  }
  auto merge = [&](float l, const float* x) {                       // acc / L at log-scale M  (+)  x at log-scale l
    if (l == NEG_INF || isinf(l)) return;
    const float nm = fmaxf(M, l);
    const float a = M == NEG_INF ? 0.f : expf(M - nm), b = expf(l - nm);
#pragma unroll
    for (int e = 0; e < E; ++e) acc[e] = acc[e] * a + x[e] * b;
    L = L * a + b;
    M = nm;
  };
  if (o_w != nullptr) {                                             // window part (FlashAttention output, bf16)
    const float l = lse_w[static_cast<long long>(hq) * m + c];
    const __nv_bfloat16* ow = o_w + (static_cast<long long>(c) * Hq + hq) * DH + lane * E;
    float x[E];
#pragma unroll
    for (int e = 0; e < E; ++e) x[e] = __bfloat162float(ow[e]);
    merge(l, x);
  }
  if (sk != nullptr) {                                              // sink part: fp32 softmax over <= S keys
    const long long rq = row_req[c];
    const long long lim = row_pos[c] - W;
    const long long nsk = sk_len[rq];
    const __nv_bfloat16* qp = q + brow * q_ts + static_cast<long long>(hq) * DH + lane * E;
    float qv[E];
#pragma unroll
    for (int e = 0; e < E; ++e) qv[e] = __bfloat162float(qp[e]);
    float sm = NEG_INF, sl = 0.f, so[E];
#pragma unroll
    for (int e = 0; e < E; ++e) so[e] = 0.f;
    for (int j = 0; j < S; ++j) {
      if (j >= nsk || j > lim) break;
      const __nv_bfloat16* kr = sk + ((rq * S + j) * 2LL * Hkv + kvh) * DH + lane * E;
      const __nv_bfloat16* vr = sk + ((rq * S + j) * 2LL * Hkv + Hkv + kvh) * DH + lane * E;
      float dot = 0.f;
#pragma unroll
      for (int e = 0; e < E; ++e) dot += qv[e] * __bfloat162float(kr[e]);
#pragma unroll
      for (int o = 16; o >= 1; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
      const float s = dot * scale;
      const float nm = fmaxf(sm, s);
      const float a = sm == NEG_INF ? 0.f : expf(sm - nm), b = expf(s - nm);
#pragma unroll
      for (int e = 0; e < E; ++e) so[e] = so[e] * a + __bfloat162float(vr[e]) * b;
      sl = sl * a + b;
      sm = nm;
    }
    if (sm != NEG_INF) {
      const float inv = 1.f / sl;
#pragma unroll
      for (int e = 0; e < E; ++e) so[e] *= inv;
      merge(sm + logf(sl), so);
    }
  }
  const float inv = L > 0.f ? 1.f / L : 0.f;
  __nv_bfloat16* dst = out + brow * o_ts + static_cast<long long>(hq) * DH + lane * E;
#pragma unroll
  for (int e = 0; e < E; e += 2) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(acc[e] * inv, acc[e + 1] * inv);
    *reinterpret_cast<__nv_bfloat162*>(dst + e) = v;
  }
}

}  // namespace

extern "C" {

int frmq_abi_version() { return 7; }   // 7 (Phase 3a): konly bit 1 = INT2 pages (k_mq2 / k_mq3 only)   // 2: kernel / rows arguments (k_mq2); 3: head size argument (k_mq3); 4: kmin (Phase 2k);
                                       // 5: frmq_win_gather / frmq_bf_merge (Phase 2k BF16 parts); 6: tile ext

int frmq_prep_q(const void* q, int q_fp32, long long q_ts, const int* idx, int m, int Hq, int G,
                const uint8_t* kfull, const uint8_t* kmask, float qscale, void* q16, int Dh, void* stream) {
  if (m <= 0) return 0;
  if (Dh != D && Dh != D3) return -4;
  const long long warps = static_cast<long long>(m) * Hq;
  const unsigned blocks = static_cast<unsigned>((warps * 32 + 255) / 256);
  if (Dh == D3) {
    k_mq_prep256<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(q, q_fp32, q_ts, idx, m, Hq, G, kfull, kmask,
                                                                          qscale, static_cast<__half*>(q16));
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
  }
  k_mq_prep<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(q, q_fp32, q_ts, idx, m, Hq, G, kfull, kmask,
                                                                     qscale, static_cast<__half*>(q16));
  return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

int frmq_attn(const void* q16, const uint8_t* pool, long long page_stride, const int* bt, long long bt_stride,
              const int* tiles, const int* items, int nitems, int Hkv, int G, int shift, int konly, float* opart,
              float* ml, int kernel, int Dh, void* stream, int kmin) {
  if (nitems <= 0) return 0;
  if (G < 1 || G > ROWS) return -3;
  if (Dh != D && Dh != D3) return -4;
  Params p{};
  p.q = static_cast<const __half*>(q16); p.pool = pool; p.page_stride = page_stride; p.bt = bt;
  p.bt_stride = bt_stride; p.tiles = tiles; p.items = items; p.Hkv = Hkv; p.G = G; p.shift = shift;
  p.kmin = kmin;
  p.opart = opart; p.ml = ml;
  {
    const char* e = getenv("FLAGROT_MQ_DBG");
    p.dbg = e != nullptr ? atoi(e) : 0;
  }
  const dim3 grid(nitems, Hkv);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (Dh == D3) {                        // head 256: k_mq3 (128 rows per CTA, NP producer + 2 consumer WGs) only
#define FRMQ_K3(KO, NP, B2)                                                                 \
    {                                                                                       \
      cudaFuncSetAttribute(k_mq3<KO, NP, B2>, cudaFuncAttributeMaxDynamicSharedMemorySize, T3_BYTES); \
      k_mq3<KO, NP, B2><<<grid, 128 * (NP + 2), T3_BYTES, s>>>(p);                          \
    }
    if (konly == 3) FRMQ_K3(true, 1, true)
    else if (konly == 2) FRMQ_K3(false, 1, true)
    else if (konly) FRMQ_K3(true, 1, false)
    else FRMQ_K3(false, 1, false)
#undef FRMQ_K3
  } else if (kernel == 2 || kernel == 3) {      // wgmma, 128 rows per CTA, 1 (kernel 2) or 2 (kernel 3) producer WGs
#define FRMQ_K2(KO, NP, B2)                                                                             \
    {                                                                                                   \
      cudaFuncSetAttribute(k_mq2<KO, NP, B2>, cudaFuncAttributeMaxDynamicSharedMemorySize, t2_bytes<NP>()); \
      k_mq2<KO, NP, B2><<<grid, 128 * (NP + 2), t2_bytes<NP>(), s>>>(p);                                \
    }
    const bool b2 = (konly & 2) != 0, ko = (konly & 1) != 0;
    if (b2) {   // INT2 pages: 2 producer warpgroups (kernel 3) only
      if (ko) FRMQ_K2(true, 2, true)
      else FRMQ_K2(false, 2, true)
    }
    else if (ko && kernel == 2) FRMQ_K2(true, 1, false)
    else if (ko) FRMQ_K2(true, 2, false)
    else if (kernel == 2) FRMQ_K2(false, 1, false)
    else FRMQ_K2(false, 2, false)
#undef FRMQ_K2
  } else if (konly & 2) {
    return -5;                           // (kernel 1: INT4 pages only)
  } else if (konly) {
    cudaFuncSetAttribute(k_mq<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_BYTES);
    k_mq<true><<<grid, 128, SM_BYTES, s>>>(p);
  } else {
    cudaFuncSetAttribute(k_mq<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_BYTES);
    k_mq<false><<<grid, 128, SM_BYTES, s>>>(p);
  }
  return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

int frmq_merge(const int* tiles, int ntiles, const float* opart, const float* ml, int Hkv, int G, int rows,
               void* out, long long o_ts, int out_fp32, float* lse, long long lse_ts, int Dh, void* stream) {
  if (ntiles <= 0) return 0;
  if (Dh != D && Dh != D3) return -4;
  if (Dh == D3) {
    k_mq_merge256<<<dim3(ntiles, Hkv, rows / 4), 128, 0, static_cast<cudaStream_t>(stream)>>>(
        tiles, opart, ml, Hkv, G, rows, out, o_ts, out_fp32, lse, lse_ts);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
  }
  k_mq_merge<<<dim3(ntiles, Hkv, rows / 4), 128, 0, static_cast<cudaStream_t>(stream)>>>(tiles, opart, ml, Hkv, G, rows,
                                                                                   out, o_ts, out_fp32, lse, lse_ts);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

// Phase 2k: BF16 window / sink parts of the INT4-direct prefill (k_win_gather, k_bf_merge above); Dh = 128 | 256
int frmq_win_gather(const long long* map, int nk, const void* sinkb, const void* preb, const void* kch, long long k_ts,
                    const void* vch, long long v_ts, int Hkv, void* kw, void* vw, int Dh, void* stream) {
  if (nk <= 0) return 0;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  using B = __nv_bfloat16;
  if (Dh == 128)
    k_win_gather<128><<<nk, 128, 0, s>>>(map, nk, static_cast<const B*>(sinkb), static_cast<const B*>(preb),
                                         static_cast<const B*>(kch), k_ts, static_cast<const B*>(vch), v_ts, Hkv,
                                         static_cast<B*>(kw), static_cast<B*>(vw));
  else if (Dh == 256)
    k_win_gather<256><<<nk, 128, 0, s>>>(map, nk, static_cast<const B*>(sinkb), static_cast<const B*>(preb),
                                         static_cast<const B*>(kch), k_ts, static_cast<const B*>(vch), v_ts, Hkv,
                                         static_cast<B*>(kw), static_cast<B*>(vw));
  else
    return -4;
  return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

int frmq_bf_merge(int m, int Hq, int Hkv, const void* q, long long q_ts, const long long* idx, const float* o_p,
                  const float* lse_p, const void* o_w, const float* lse_w, const void* sk, const long long* row_req,
                  const long long* row_pos, const long long* sk_len, int S, int W, float scale, void* out,
                  long long o_ts, int Dh, void* stream) {
  if (m <= 0) return 0;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  using B = __nv_bfloat16;
  const dim3 grid(m, (Hq + 3) / 4);
#define FRMQ_BFM(DH)                                                                                              \
  k_bf_merge<DH><<<grid, 128, 0, s>>>(m, Hq, Hkv, static_cast<const B*>(q), q_ts, idx, o_p, lse_p,             \
                                      static_cast<const B*>(o_w), lse_w, static_cast<const B*>(sk), row_req,       \
                                      row_pos, sk_len, S, W, scale, static_cast<B*>(out), o_ts)
  if (Dh == 128) FRMQ_BFM(128);
  else if (Dh == 256) FRMQ_BFM(256);
  else return -4;
#undef FRMQ_BFM
  return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

int frmq_smem_bytes() { return SM_BYTES; }

const char* frmq_last_error() { return cudaGetErrorString(cudaGetLastError()); }
}
