// v2: paged split-K GQA decode attention over 4-bit KV caches (NVFP4 / MXFP4 / INT4) on H20 (sm_90a).
//
// Why a new kernel : on sm_90 mma.sync with e4m3 operands is emulated
// (F2FP unpack + 2 x HMMA.16816), mma.sync f16 sustains only ~95 TFLOPS, while warpgroup wgmma reaches ~275 TFLOPS
// (e4m3, k32) / ~137 TFLOPS (f16).  v1 (attn_decode.cu) spends 2.0-2.5 instructions per KV element on dequant
// (half on ALU pipes) and half of every MMA on padding heads.  v2:
//   * one warpgroup (128 threads) per CTA, 64-token blocks; whole pages (K codes, V^T codes, scales: one contiguous
//     chunk per (page, head)) are fetched into an ST-stage ring guarded by mbarriers, either with cp.async.bulk (one
//     page per warp and block: a single issuing warp would delay the whole warpgroup at the next wgmma) or with
//     16-B cp.async spread over all threads (MODE bit 1); the CTA's page ids are staged in shared memory first;
//   * QK:  S^T[64 tok, N] = K[64, 32 d] . [q_hi | 16 q_lo]^T  with wgmma m64nNk32 e4m3 (A = K from registers, B = Q^T
//          in smem), one fresh accumulator per k-step = per scale group (MXFP4/INT4: 32 d; NVFP4: N = 16, the two
//          16-d groups of a k-step in separate column halves), scales applied to the fp32 partials (no per-element
//          scale), S = sum_g s_g (acc_hi + acc_lo / 16);  q is rotated (FlagRot), scaled and split in the prologue;
//   * dequant of a 32-bit code word (8 elements) to e4m3 bytes: 3 (INT4) / 5 (e2m1) instructions (custom bit layout,
//          see attn_decode_v2.cuh), no per-element scaling;
//   * PV, MODE bit 0 = 0 (MXFP4 / INT4): O^T[64 d, 16] += V^T[64 d, 32 tok] . P'^T with wgmma m64n16k32 e4m3, the V
//          scale of (token, 32-d group) folded into P' = p * s_v / 2^Eref[g] (f16x2 math, split hi = RN_e4m3(P'),
//          lo = RN_e4m3(16 (P' - hi)); columns = 2 groups x 4 heads x hi/lo; the 64 d rows of a tile = 2 groups);
//          MODE bit 0 = 1 (all formats, the only one for NVFP4): V^T decoded to f16 with the per-(token, group)
//          scale (HMUL2), wgmma m64n8k16 f16, P in f16;
//   * FP8 wgmma accumulates with ~2^-11 relative precision per k-step and loses precision when chained (measured,
//          csrc/mma_bench.cu): every block uses fresh accumulators, added in fp32;
//   * online softmax with a lazily updated, warpgroup-uniform reference max per head (and V-scale exponent Eref per
//          group): a block takes the slow path (cross-warp max, rescale, rewrite P) only if some score exceeds the
//          reference by TAU (log2) or some P' would exceed 2^(TAU + SH) (barrier with OR reduction);
//   * order inside a block (n):  wait QK(n) + PV(n-1) -> O += PV(n-1), S(n) -> V fragments(n) -> K fragments(n+1),
//          QK(n+1) -> softmax(n), P(n) -> smem -> barrier -> PV(n).  ptxas treats everything between the first wgmma
//          after a wait and the next wait as one pipeline stage and serializes ALL wgmmas of the kernel if a non-wgmma
//          instruction in it defines wgmma input / accumulator registers (C7511-C7515), so all fragments are
//          produced before the stage starts; only the softmax overlaps QK(n+1).  The ptxas of CUDA 12.4 serializes
//          this kernel anyway; CUDA 12.9's does not (attn_decode_v2.py builds with 12.9).
// Split partials are merged by the last CTA of each (sequence, kv head) as in v1 (variable sequence lengths);
// warp-per-head merge with 16 partial loads in flight, one acq_rel ticket atomic, block-table reads and q not
// waiting for the sequence length, optional programmatic dependent launch (attn2_set_pdl); MODE bit 9 = stream-K
// schedule (sk_prefix; experimental variant 776, not faster than the planned split grid).
// Experiment-only macros: ATTN2_DBG_NOBAR / NOPV / NOQK (wrong results, timing ablations), ATTN2_TIMING (per-phase
// clock64 sums of CTA (0, 0, 0) written to opart, no output), ATTN2_MINB (min CTAs per SM for __launch_bounds__).
#include "attn_decode_v2.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define DEVI __device__ __forceinline__

namespace {

constexpr int G = 4;      // q heads per kv head
constexpr int NWARP = 4;  // one warpgroup
constexpr int D = 128;
constexpr int BLK = 64;   // tokens per block
constexpr unsigned FULL = 0xffffffffu;
constexpr float NEG_INF = -__builtin_huge_valf();
constexpr float TAU = 3.0f;  // lazy softmax threshold (log2): p <= 2^TAU
constexpr int SH = 4;        // V-scale headroom (log2): s' = s / 2^Eref <= 2^SH  ->  P' <= 2^(TAU + SH) = 128 < 448
constexpr int EREF_NONE = -100000;
constexpr int SK_MAXB = 256;    // largest batch of the stream-K mode (sk_pre / sk_len in shared memory)
constexpr int MAX_CHUNK = 8192;  // tokens per split (page ids of a CTA are staged in shared memory)
constexpr float LO_SCALE = 16.f, LO_INV = 1.f / 16.f;  // hi/lo splits: lo stored as e4m3((x - hi) * 16)
// INT4A value zero points: 1 = Z = P'^T z on the tensor cores (default), 0 = per-thread sums of P' z on the CUDA cores
// (1.5 % faster, but the FP8 PV accumulation of the all-positive products P' c is then not offset by the same
// accumulation of P' z: rel-L2 error 3x the symmetric kernel's at 33k tokens)
#ifndef ATTN2_ZVT
#define ATTN2_ZVT 1
#endif

template <int F> struct F2 {
  // INT4 layouts (fp16 scale words); A2_KONLY / A2_KONLYA: the key-only 2x2 codebook caches (attention only)
  // INT2 (vLLM plugin Phase 3a): 2-bit unsigned codes, zero point in bits 1..0 of the scale word (attn_decode_v2.cuh)
  static constexpr bool B2 = F == A2_INT2A || F == A2_KONLYA2;
  static constexpr bool I4 = F == A2_INT4 || F == A2_INT4A || F == A2_KONLY || F == A2_KONLYA || B2;  // (fp16 scale words)
  static constexpr bool ASYM = F == A2_INT4A || F == A2_KONLYA || B2;  // unsigned codes with a per-(token, group) zero point
  static constexpr bool KO = F == A2_KONLY || F == A2_KONLYA || F == A2_KONLYA2;  // keys: per-token I / H32 / whole-head choice
  static constexpr int CBY = B2 ? 32 : 64;                             // code bytes per (token, 128-d row) of K and of V
  static constexpr uint32_t ZM2 = B2 ? 0x00030003u : 0x000F000Fu;      // zero-point bits of two scale words
  static constexpr uint32_t ZM4 = B2 ? 0x03030303u : 0x0F0F0F0Fu;      // ... of four low bytes
  static constexpr int SB = F == A2_MXFP4 ? 4 : 8;   // scale bytes per (token, head)
  static constexpr int NG = F == A2_NVFP4 ? 8 : 4;   // scale groups per 128-d row
  static constexpr int SBE = I4 ? 2 : 1;             // bytes per scale
  static constexpr int NQ = F == A2_NVFP4 ? 16 : 8;  // QK wgmma N
  static constexpr float FV = I4 ? 512.f : 64.f;     // value of an element = e4m3 byte value * FV * scale
};
template <int F, int P> struct Lay {
  static constexpr int CBY = F2<F>::CBY;
  static constexpr int K_OFF = 0, V_OFF = P * CBY, KS_OFF = 2 * P * CBY, VS_OFF = 2 * P * CBY + P * F2<F>::SB;
  static constexpr int CHUNK = P * (2 * CBY + 2 * F2<F>::SB);
  static constexpr int NPB = BLK / P;  // pages per block
  // shared-memory page stride of the ring: INT2 chunks (80 P bytes = 0 mod 128) padded to 64 mod 128, so that the
  // 8-byte V^T entry reads of the two pages of a 32-token half (P = 16, lanes tig 0-1 / 2-3) hit different banks
  // (as attn_codebook_attn LayA::PSTR); INT4: no padding (the INT4 kernels are unchanged)
  static constexpr int PSTR = F2<F>::B2 ? CHUNK + (192 - CHUNK % 128) % 128 : CHUNK;
  static constexpr int STAGE = NPB * PSTR;
  static constexpr int KTB = F2<F>::B2 ? 8 : 16;   // K code bytes of a row per thread (tig)
  static constexpr int VE = F2<F>::B2 ? 8 : 16;    // bytes per V^T entry (8 tokens x 4 / 2 rows of d)
};
// byte offset of token t's K code row within its page (INT2: rows t and t ^ 2 swapped where t & 4, attn_decode_v2.cuh)
template <int F> DEVI int krow_off(int t) { return F2<F>::B2 ? 32 * (t ^ ((t >> 1) & 2)) : 64 * t; }

// ------------------------------------------------------------------------------------------------- PTX helpers
DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
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
DEVI uint32_t lds16(uint32_t a) {  // zero-extended into a 32-bit register (no "=h" outputs, see f32x2_e4m3x2)
  uint32_t u;
  asm volatile("ld.shared.u16 %0, [%1];\n" : "=r"(u) : "r"(a));
  return u;
}
DEVI void sts16(uint32_t a, uint32_t v) { asm volatile("st.shared.u16 [%0], %1;\n" ::"r"(a), "r"(v)); }
DEVI void sts32(uint32_t a, uint32_t v) { asm volatile("st.shared.u32 [%0], %1;\n" ::"r"(a), "r"(v)); }
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
DEVI void cp_mbar_arrive(uint32_t mbar) {  // arrive on mbar when all prior cp.async of this thread have completed
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(mbar) : "memory");
}
DEVI void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
DEVI void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
DEVI void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
DEVI uint64_t sdesc_ls(uint32_t saddr, uint32_t lbo, uint32_t sbo) {  // no swizzle, K-major, LBO / SBO in bytes
  return static_cast<uint64_t>((saddr & 0x3FFFF) >> 4) | (static_cast<uint64_t>(lbo >> 4) << 16) |
         (static_cast<uint64_t>(sbo >> 4) << 32);
}
DEVI uint64_t sdesc(uint32_t saddr) {  // no swizzle, K-major core matrices: LBO (K) = 128 B, SBO (N) = 256 B
  return static_cast<uint64_t>((saddr & 0x3FFFF) >> 4) | (static_cast<uint64_t>(128 >> 4) << 16) |
         (static_cast<uint64_t>(256 >> 4) << 32);
}
// D[64 x N] (+)= A[64 x K] (registers) . B (smem); ACC = 0: D = A . B (fresh accumulator, write-only operands: an
// input constraint would make the compiler define the accumulator registers with moves while another wgmma of the
// warpgroup is in flight, and ptxas then serializes all wgmmas (C7515))
template <int ACC>
DEVI void wg_e4m3_n8(float (&d)[4], const uint32_t (&a)[4], uint64_t desc) {
  if constexpr (ACC)
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, 1, 1, 1;\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  else
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, 0, 1, 1;\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int ACC>
DEVI void wg_e4m3_n16(float (&d)[8], const uint32_t (&a)[4], uint64_t desc) {
  if constexpr (ACC)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, 1, 1, 1;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  else
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, 0, 1, 1;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int ACC>
DEVI void wg_f16_n8(float (&d)[4], const uint32_t (&a)[4], uint64_t desc) {
  if constexpr (ACC)
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, 1, 1, 1, 0;\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  else
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, 0, 1, 1, 0;\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
// A from shared memory (SS mode): adesc = K-major no-swizzle tile (LBO 128 B, SBO 256 B, sdesc)
template <int ACC>
DEVI void wg_e4m3_n16_ss(float (&d)[8], uint64_t adesc, uint64_t bdesc) {
  if constexpr (ACC)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, 1, 1, 1;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "l"(adesc), "l"(bdesc));
  else
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, 0, 1, 1;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
        : "l"(adesc), "l"(bdesc));
}
// m64n8k32 with both operands in shared memory (INT4A value zero points: A = the P' tiles, B = the zero-point tile)
template <int ACC>
DEVI void wg_e4m3_n8_ss(float (&d)[4], uint64_t adesc, uint64_t bdesc) {
  if constexpr (ACC)
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, %4, %5, 1, 1, 1;\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "l"(adesc), "l"(bdesc));
  else
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, %4, %5, 0, 1, 1;\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "l"(adesc), "l"(bdesc));
}
DEVI uint32_t lop_or_and(uint32_t a, uint32_t m, uint32_t o) {  // (a & m) | o in one LOP3
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(r) : "r"(a), "r"(m), "r"(o));
  return r;
}
DEVI uint32_t prmt_b32(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}
template <int N>
DEVI void wg_wait0(float (&d)[N]) {  // also tells the compiler that the accumulators are written here
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(d[i])::"memory");
}
DEVI void wg_wait_all() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
DEVI void wg_wait_1() { asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory"); }
template <int N>
DEVI void pin_regs(uint32_t (&r)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+r"(r[i]));
}

// opaque copies: keep loop-invariant values in registers (the compiler otherwise rematerializes address and wgmma
// descriptor arithmetic inside the main loop)
#ifndef ATTN2_NO_OPAQUE
DEVI uint32_t opaque(uint32_t x) { asm volatile("mov.b32 %0, %0;" : "+r"(x)); return x; }
DEVI uint64_t opaque64(uint64_t x) { asm volatile("mov.b64 %0, %0;" : "+l"(x)); return x; }
#else  // experiment: let the compiler rematerialize (register-capped builds)
DEVI uint32_t opaque(uint32_t x) { return x; }
DEVI uint64_t opaque64(uint64_t x) { return x; }
#endif
DEVI float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
#include "attn_tail.cuh"
DEVI uint32_t e4m3x2_f16x2_lo(uint32_t v) {
  uint32_t r;
  asm("{ .reg .b16 l, h; mov.b32 {l, h}, %1; cvt.rn.f16x2.e4m3x2 %0, l; }" : "=r"(r) : "r"(v));
  return r;
}
DEVI uint32_t e4m3x2_f16x2_hi(uint32_t v) {
  uint32_t r;
  asm("{ .reg .b16 l, h; mov.b32 {l, h}, %1; cvt.rn.f16x2.e4m3x2 %0, h; }" : "=r"(r) : "r"(v));
  return r;
}
// e4m3x2 packs: the 16-bit result is widened inside the asm (a "=h" output reads back as 0 with the CUDA 12.9 compiler)
DEVI uint32_t f32x2_e4m3x2(float lo, float hi) {  // lo -> byte 0
  uint32_t r;
  asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, %1, %2; cvt.u32.u16 %0, t; }\n" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
DEVI uint32_t f16x2_e4m3x2(uint32_t h) {  // low half -> byte 0
  uint32_t r;
  asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, %1; cvt.u32.u16 %0, t; }\n" : "=r"(r) : "r"(h));
  return r;
}
// P' pair (f16x2) -> hi = RN_e4m3(P') at a, lo = RN_e4m3(16 (P' - hi)) at a + 16, all in 16-bit registers (no
// widening / masking instructions between the conversions and the stores)
DEVI void pprime_store(uint32_t P2, uint32_t a) {
  asm volatile(
      "{\n .reg .b16 h, l;\n .reg .b32 hf, d;\n"
      " cvt.rn.satfinite.e4m3x2.f16x2 h, %0;\n"
      " cvt.rn.f16x2.e4m3x2 hf, h;\n"
      " sub.rn.f16x2 d, %0, hf;\n"
      " mul.rn.f16x2 d, d, %2;\n"
      " cvt.rn.satfinite.e4m3x2.f16x2 l, d;\n"
      " st.shared.u16 [%1], h;\n"
      " st.shared.u16 [%1 + 16], l;\n}\n" ::"r"(P2), "r"(a), "r"(0x4C004C00u) : "memory");
}
// pprime_store that also returns the f16x2 of what the tensor core sees, hi + lo / 16 (exact in f16: <= 8 significant
// bits), so that the value zero-point sums use the same P' as PV (INT4A)
DEVI uint32_t pprime_store_t(uint32_t P2, uint32_t a) {
  uint32_t t;
  asm volatile(
      "{\n .reg .b16 h, l;\n .reg .b32 hf, lf, d;\n"
      " cvt.rn.satfinite.e4m3x2.f16x2 h, %1;\n"
      " cvt.rn.f16x2.e4m3x2 hf, h;\n"
      " sub.rn.f16x2 d, %1, hf;\n"
      " mul.rn.f16x2 d, d, %3;\n"
      " cvt.rn.satfinite.e4m3x2.f16x2 l, d;\n"
      " st.shared.u16 [%2], h;\n"
      " st.shared.u16 [%2 + 16], l;\n"
      " cvt.rn.f16x2.e4m3x2 lf, l;\n"
      " fma.rn.f16x2 %0, lf, %4, hf;\n}\n" : "=r"(t) : "r"(P2), "r"(a), "r"(0x4C004C00u), "r"(0x2C002C00u) : "memory");
  return t;
}
DEVI uint32_t hsub2u(uint32_t a, uint32_t b) {
  __half2 r = __hsub2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
DEVI uint32_t hmax2u(uint32_t a, uint32_t b) {
  __half2 r = __hmax2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
DEVI uint32_t clamp_s16x2(uint32_t x, uint32_t lo, uint32_t hi) {  // per 16-bit half: min(max(x, lo), hi)
  uint32_t r;
  asm("max.s16x2 %0, %1, %2;\n" : "=r"(r) : "r"(x), "r"(lo));
  asm("min.s16x2 %0, %1, %2;\n" : "=r"(r) : "r"(r), "r"(hi));
  return r;
}
DEVI float2 h2f2(uint32_t h) { return __half22float2(*reinterpret_cast<const __half2*>(&h)); }
DEVI uint32_t hmul2u(uint32_t a, uint32_t b) {
  __half2 r = __hmul2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

// 8 code elements of a 32-bit word -> e4m3 bytes of the 4 A and the 4 B elements
template <int F>
DEVI void cvt_ab(uint32_t w, uint32_t& a, uint32_t& b) {
  if constexpr (F2<F>::ASYM) {  // unsigned nibbles: byte c is the e4m3 value c * 2^-9 (c <= 15)
    a = w & 0x0F0F0F0Fu;
    b = (w >> 4) & 0x0F0F0F0Fu;
  } else if constexpr (F2<F>::I4) {
    a = w & 0x87878787u;
    b = __funnelshift_l(w, w, 4) & 0x87878787u;
  } else {
    a = w & 0x9C9C9C9Cu;
    b = ((w << 2) & 0x8C8C8C8Cu) | ((w >> 2) & 0x10101010u);
  }
}
// INT2: 16 code elements of a 32-bit word -> e4m3 bytes (c 2^-9, c <= 3) of the A / B words of its two halves
DEVI void cvt2(uint32_t w, uint32_t& a0, uint32_t& b0, uint32_t& a1, uint32_t& b1) {
  a0 = w & 0x03030303u;
  b0 = (w >> 2) & 0x03030303u;
  a1 = (w >> 4) & 0x03030303u;
  b1 = (w >> 6) & 0x03030303u;
}
// K fragments of k-steps j0 .. j0 + 3 from the code rows of a thread (row tau0 at shared address a0, tau1 at a1):
// ka[j][0] / [2] = A / B elements of row tau0, [1] / [3] of row tau1 (INT4: one 16-B load per row; INT2: 8 B)
template <int F, int NJ>
DEVI void k_frags(uint32_t a0, uint32_t a1, uint32_t (&ka)[NJ][4], int j0) {
  if constexpr (F2<F>::B2) {
    const uint2 x0 = lds64(a0), x1 = lds64(a1);
    cvt2(x0.x, ka[j0][0], ka[j0][2], ka[j0 + 1][0], ka[j0 + 1][2]);
    cvt2(x0.y, ka[j0 + 2][0], ka[j0 + 2][2], ka[j0 + 3][0], ka[j0 + 3][2]);
    cvt2(x1.x, ka[j0][1], ka[j0][3], ka[j0 + 1][1], ka[j0 + 1][3]);
    cvt2(x1.y, ka[j0 + 2][1], ka[j0 + 2][3], ka[j0 + 3][1], ka[j0 + 3][3]);
  } else {
    const uint4 x0 = lds128(a0), x1 = lds128(a1);
    const uint32_t w0[4] = {x0.x, x0.y, x0.z, x0.w}, w1[4] = {x1.x, x1.y, x1.z, x1.w};
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      cvt_ab<F>(w0[jj], ka[j0 + jj][0], ka[j0 + jj][2]);
      cvt_ab<F>(w1[jj], ka[j0 + jj][1], ka[j0 + jj][3]);
    }
  }
}
// V^T fragments of one entry (8 tokens of a k-step; rows d (r = 0) and d + 8 of the m tiles 0 / 1): v_m[0] / [2] = A / B
// of row d, [1] / [3] of row d + 8
template <int F>
DEVI void v_frags(uint32_t a, uint32_t (&v0)[4], uint32_t (&v1)[4]) {
  if constexpr (F2<F>::B2) {
    const uint2 x = lds64(a);
    cvt2(x.x, v0[0], v0[2], v0[1], v0[3]);
    cvt2(x.y, v1[0], v1[2], v1[1], v1[3]);
  } else {
    const uint4 x = lds128(a);
    cvt_ab<F>(x.x, v0[0], v0[2]);
    cvt_ab<F>(x.y, v0[1], v0[3]);
    cvt_ab<F>(x.z, v1[0], v1[2]);
    cvt_ab<F>(x.w, v1[1], v1[3]);
  }
}

// ------------------------------------------------------------------------------------------------- FlagRot (v1)
DEVI float rb_rsqrt(int rb) {
  switch (rb) {
    case 2: return 0.70710678118654752f;
    case 4: return 0.5f;
    case 8: return 0.35355339059327373f;
    case 16: return 0.25f;
    case 32: return 0.17677669529663687f;
    case 64: return 0.125f;
    default: return 0.08838834764831845f;
  }
}
// FlagRot of a 128-vector held by a warp (lane l: elements 4l..4l+3), bit-identical to ref.fwht
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

// ------------------------------------------------------------------------------------------------- scales
// K scales of k-step jj from the raw scale bytes of a row (kept raw across the loop: fewer live registers)
// returns (scale of the first / only group, scale of the second 16-d group (NVFP4))
template <int F>
DEVI float2 kscale_step(uint2 raw, int jj) {
  if constexpr (F == A2_NVFP4) {  // groups 2 jj, 2 jj + 1 = bytes 2 jj, 2 jj + 1
    const uint32_t wv = jj < 2 ? raw.x : raw.y;
    return h2f2((jj & 1) ? e4m3x2_f16x2_hi(wv) : e4m3x2_f16x2_lo(wv));
  } else if constexpr (F == A2_MXFP4) {
    const uint32_t e = (raw.x >> (8 * jj)) & 0xFFu;
    return make_float2(__uint_as_float(e ? e << 23 : 0x00400000u), 0.f);
  } else {
    const uint32_t wv = jj < 2 ? raw.x : raw.y;
    const float2 f = h2f2(wv);
    return make_float2((jj & 1) ? f.y : f.x, 0.f);
  }
}

// e8m0 exponent -> fp16 bits of 2^(e-127) * 2^7, clamped to the fp16 normal range (as v1's mx64h)
DEVI uint32_t mx64h(uint32_t e) { return static_cast<uint32_t>(max(min(static_cast<int>(e), 135), 105) - 105) << 10; }

// ------------------------------------------------------------------------------------------------- kernel
template <int F, int P, int ST, int MODE>
struct Cfg2 {
  static constexpr int PVF16 = MODE & 1;
  using L = Lay<F, P>;
  static constexpr int NQ = F2<F>::NQ;
  static constexpr int QT_BYTES = 4 * NQ * 32;                 // Q^T tiles (4 k-steps)
  // FP8 P' tiles [ks][m] (N 16 x K 32): MODE bit 3 = bank-conflict-free layout: K-core matrices 144 B apart (LBO),
  // N-cores 288 B (SBO), tiles 576 B (the 2-way conflict between the two K halves of a warp's P' stores goes away)
  static constexpr int PLBO = (MODE & 8) ? 144 : 128, PSBO = (MODE & 8) ? 288 : 256, PTILE = (MODE & 8) ? 576 : 512;
  static constexpr int PT_BYTES = PVF16 ? 4 * 256 : 4 * PTILE;   // P^T tiles of one block
  static constexpr int RING_OFF = 0;
  static constexpr int QT_OFF = ST * L::STAGE;                 // 16-B aligned (CHUNK % 16 == 0)
  static constexpr int PT_OFF = QT_OFF + QT_BYTES;
  static constexpr int PIDX_OFF = PT_OFF + 2 * PT_BYTES;      // page ids of the CTA's token range
  // MODE bit 2 (SS PV): the decoded e4m3 V^T tiles of a block (4 x [64 d x 32 tok], canonical K-major) for wgmma
  static constexpr int VT_BYTES = (MODE & 4) ? ((MODE & 16) ? 2 : 1) * 4 * 2048 : 0;  // (MODE bit 4: two buffers)
  static constexpr int SRC_OFF = PIDX_OFF + (MAX_CHUNK / P) * 4;  // bulk loader: global address of each page chunk
  static constexpr int VT_OFF = SRC_OFF + ((MODE & 2) ? 0 : (MAX_CHUNK / P) * 8);  // 16-B aligned
  // INT4A: value zero points of a block as the B operand of Z = P' . z (K-major [8 groups (4 used) x 32 tokens] per k-step,
  // k halves 128 B apart: [buf][ks][k half][8 rows][16 B]), double-buffered
  static constexpr int ZB_OFF = VT_OFF + VT_BYTES;
  static constexpr int ZB_BYTES = (F2<F>::ASYM && ATTN2_ZVT) ? 2 * 512 : 0;
  static constexpr int SK_OFF = ZB_OFF + ZB_BYTES;  // MODE bit 9 (stream-K): sk_pre [SK_MAXB + 1] | sk_len [SK_MAXB]
  static constexpr int SMEM = SK_OFF + ((MODE & 512) ? (2 * SK_MAXB + 2) * 4 : 0);
};

#include "attn_merge.cuh"  // PartMap, merge_warp

// split partial / merge (last-arriving CTA of a (sequence, kv head)) and the bf16 store.  x: this CTA's unnormalized
// output of q head w = warp at d = 4 lane .. 4 lane + 3; M / L its softmax max / sum (warp-uniform); wslot: the slot
// this CTA's partial goes to; nparts: all partials of the (sequence, kv head) (body + BF16 window).  All 128 threads;
// returns normally (also when another CTA merges).
template <int NB = 16>  // merge: partial rows in flight per thread (stream-K: 8, register pressure)
DEVI void merge_finalize(const PagedParams& p, float (&x)[4], float M, float L, int b, int kvh, int wslot,
                         const PartMap& pm, int nparts, int* last_flag_p) {
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  int& last_flag = *last_flag_p;
  if (nparts == 1) {
    const float il = L > 0.f ? 1.f / L : 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] *= il;
  } else {
    reinterpret_cast<float4*>(p.opart)[static_cast<size_t>(wslot) * (G * D / 4) + w * (D / 4) + lane] =
        make_float4(x[0], x[1], x[2], x[3]);
    if (lane == 0) reinterpret_cast<float2*>(p.mlpart)[wslot * G + w] = make_float2(M, L);
#ifdef ATTN2_OLD_EPI  // experiment: the earlier ticket (every thread fences)
    __threadfence();
    __syncthreads();
    if (tid == 0) last_flag = atomicAdd(p.counter + b * p.Hkv + kvh, 1) == nparts - 1;
    __syncthreads();
    if (!last_flag) return;
    __threadfence();
#else
    // merge ticket: barrier, then one acq_rel atomic by thread 0 (release: the CTA's partial stores, ordered before it
    // by the barrier; acquire: the other CTAs' partials for the merge after the second barrier) instead of a
    // __threadfence by every thread on both sides
    __syncthreads();
    if (tid == 0) {
      int old;
      asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(p.counter + b * p.Hkv + kvh) : "memory");
      last_flag = old == nparts - 1;
    }
    __syncthreads();
    if (!last_flag) return;
#endif
#ifndef ATTN2_OLD_MERGE
    merge_warp<NB, NB >= 16>(p, pm, nparts, x);
#else  // experiment: the earlier merge loop (one warp computes the weights of the 4 heads one after the
       // other, ~4 loads in flight per thread), on the same data layout
    __shared__ float wgt_old[520 * G];
    __shared__ float rl_old[G];
    const float2* mlp = reinterpret_cast<const float2*>(p.mlpart);
    if (tid < 32) {
#pragma unroll
      for (int h = 0; h < G; ++h) {
        float Mx = NEG_INF;
        for (int s2 = lane; s2 < nparts; s2 += 32) Mx = fmaxf(Mx, __ldcg(mlp + pm.slot(s2) * G + h).x);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) Mx = fmaxf(Mx, __shfl_xor_sync(FULL, Mx, o));
        const float Ms = Mx == NEG_INF ? 0.f : Mx;
        float l = 0.f;
        for (int s2 = lane; s2 < nparts; s2 += 32) {
          const float2 ml = __ldcg(mlp + pm.slot(s2) * G + h);
          const float sc = ml.x == NEG_INF ? 0.f : ex2(ml.x - Ms);
          wgt_old[s2 * G + h] = sc;
          l += sc * ml.y;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(FULL, l, o);
        if (lane == 0) rl_old[h] = l > 0.f ? 1.f / l : 0.f;
      }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float o = 0.f;
#pragma unroll 4
      for (int s2 = 0; s2 < nparts; ++s2)
        o += wgt_old[s2 * G + w] * __ldcg(p.opart + (static_cast<size_t>(pm.slot(s2)) * G + w) * D + 4 * lane + i);
      x[i] = o * rl_old[w];
    }
#endif
    if (tid == 0) p.counter[b * p.Hkv + kvh] = 0;
  }
  // finalize: warp w = q head w (optional FlagRot back-rotation of V), bf16 store
  if (p.vrot) warp_selrot(x, p.vmask + kvh * (D / p.rb), p.rb, lane);
  __nv_bfloat162 o01 = __floats2bfloat162_rn(x[0], x[1]), o23 = __floats2bfloat162_rn(x[2], x[3]);
  uint2 ov;
  ov.x = *reinterpret_cast<uint32_t*>(&o01);
  ov.y = *reinterpret_cast<uint32_t*>(&o23);
  *reinterpret_cast<uint2*>(static_cast<__nv_bfloat16*>(p.out) + b * p.o_bs + (kvh * G + w) * D + lane * 4) = ov;
}

// ------------------------------------------------------------------------------------------------- stream-K schedule
// Stream-K mode (MODE bit 9, PagedParams::sk_grid = number of CTAs, a multiple of Hkv): per kv head, the 64-token
// blocks of all sequences form one flattened sequence (sequences in order, sk_pre[b] = first block of sequence b,
// ceil(len_b / 64) blocks each, T = sk_pre[B] per head).  CTA c works on kv head c % Hkv and owns the blocks
// [floor(j T / G), floor((j + 1) T / G)) of it, j = c / Hkv, G = min(sk_grid / Hkv, T): every CTA gets the same work
// (+-1 block) whatever the batch shape, a range may span several sequences (segments), and CTAs with adjacent ids
// read the same pages of the different kv heads at the same time (as the split grid (Hkv, split, b) does: a
// sequence-major order over (b, kvh) cost 18 % at B = 32, L = 32k through scattered DRAM accesses, measured).  A
// segment that covers a whole sequence writes the output directly; otherwise its partial goes to slot 2 c (first
// segment of CTA c) or 2 c + 1 (last segment), and the last-arriving CTA of the (sequence, kv head) merges (PartMap).
// BF16 window partials: slots 2 sk_grid + (b Hkv + kvh) tail_nslot + j.

// sk_pre [B + 1] (block prefix of one kv head), sk_len [B] (lengths, >= 0).  All threads; ends with a barrier.
DEVI void sk_prefix(const PagedParams& p, int* pre, int* lens) {
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    int carry = 0;
    for (int b0 = 0; b0 < p.B; b0 += 32) {
      const int bb = b0 + lane;
      const int L = bb < p.B ? max(p.seq_lens[bb], 0) : 0;
      int v = (L + BLK - 1) / BLK;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int t = __shfl_up_sync(FULL, v, o);
        if (lane >= o) v += t;
      }
      if (bb < p.B) { pre[bb + 1] = carry + v; lens[bb] = L; }
      carry += __shfl_sync(FULL, v, 31);
    }
    if (lane == 0) pre[0] = 0;
  }
  __syncthreads();
}
// a / d for a quotient < 2^32: 32-bit division when a < 2^32 (the 64-bit division is a ~70-instruction subroutine)
DEVI uint32_t udiv64(uint64_t a, uint32_t d) {
  return (a >> 32) == 0ull ? static_cast<uint32_t>(a) / d : static_cast<uint32_t>(a / d);
}
// CTAs per kv head that get work: min(sk_grid / Hkv, T) (each a non-empty range; the others return at once)
DEVI uint32_t sk_ctas(const PagedParams& p, uint32_t T) { return min(static_cast<uint32_t>(p.sk_grid / p.Hkv), T); }
// sequence of flattened block x (largest b with pre[b] <= x: skips empty sequences)
DEVI int sk_find(const int* pre, int B, int x) {
  int lo = 0, hi = B - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (pre[mid] <= x) lo = mid; else hi = mid - 1;
  }
  return lo;
}
// partial map of (b, kvh): range indices jf .. jl of head kvh hold its body parts (j_of(x) = ((x + 1) G - 1) / T),
// CTA id = j Hkv + kvh
DEVI PartMap sk_pair(const PagedParams& p, const int* pre, int b, int kvh) {
  const uint32_t T = pre[p.B];
  const int t0 = 2 * p.sk_grid + (b * p.Hkv + kvh) * p.tail_nslot;
  const int nbb = pre[b + 1] - pre[b];
  if (nbb == 0) return PartMap{0, 2, 0, 0, t0};
  const uint32_t Gs = sk_ctas(p, T);
  const uint64_t P0 = static_cast<uint32_t>(pre[b]);
  const int jf = static_cast<int>(udiv64((P0 + 1) * Gs - 1, T)), jl = static_cast<int>(udiv64((P0 + nbb) * Gs - 1, T));
  const uint32_t sjf = udiv64(static_cast<uint64_t>(jf) * T, Gs);
  return PartMap{2 * (jf * p.Hkv + kvh), 2 * p.Hkv, sjf < P0 ? 1 : 0, jl - jf + 1, t0};
}
// segment state of a CTA in shared memory (written by thread 0, read by all after a barrier)
enum SkState { SK_JB, SK_R0, SK_R1, SK_KVH, SK_B, SK_C0, SK_C1, SK_PB0, SK_PST, SK_PADJ, SK_PNB, SK_PT0, SK_WS, SK_NSTATE };
// thread 0: the segment starting at local block jbase of the range [r0, r1) of kv head kvh (bhint <= its sequence)
DEVI void sk_setup(const PagedParams& p, const int* pre, const int* lens, int kvh, int jbase, int r0, int r1, int bhint,
                   int* st) {
  const int r = r0 + jbase;
  int b = bhint;
  while (pre[b + 1] <= r) ++b;
  const int n0 = r - pre[b], n1 = min(pre[b + 1] - pre[b], n0 + (r1 - r));
  const PartMap pm = sk_pair(p, pre, b, kvh);
  st[SK_JB] = jbase; st[SK_B] = b; st[SK_C0] = n0 * BLK; st[SK_C1] = min(lens[b], n1 * BLK);
  st[SK_PB0] = pm.b0; st[SK_PST] = pm.stride; st[SK_PADJ] = pm.adj; st[SK_PNB] = pm.nbody; st[SK_PT0] = pm.t0;
  st[SK_WS] = 2 * static_cast<int>(blockIdx.x) + (jbase == 0 ? 0 : 1);
}

// BF16 staging window (attn_tail.cuh): grid (Hkv, tail_nslot, B), CTA j = window tokens [32 j, 32 j + 32); partial
// moved into the V basis of the cache (FlagRot of the values, when the V flags are given), stored as split slot
// ns_b + j (j when the sequence has no cached tokens), then the merge ticket (this CTA may be the last one)
__global__ void __launch_bounds__(128) k_tail_v2(const PagedParams p) {
  extern __shared__ __align__(16) uint8_t smt[];
  __shared__ float tml[2 * G];
  __shared__ int last_flag;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int kvh = blockIdx.x, j = 0, b = blockIdx.z;
  const int tl = min(p.tail_len[b], TAIL_MAX);
  const int len = p.seq_lens[b];
  const int ns_b = len > 0 ? max(1, (len + p.chunk - 1) / p.chunk) : 0;  // (len 0: the body CTAs return at once)
  float* o = reinterpret_cast<float*>(smt);  // [G][D] | scratch
  const __nv_bfloat16* qh = static_cast<const __nv_bfloat16*>(p.q) + b * p.q_bs + kvh * G * D;
  const __nv_bfloat16* tb = static_cast<const __nv_bfloat16*>(p.tail) + (p.tail_idx ? p.tail_idx[b] : b) * p.tail_bs;
  tail_window(qh, p.qscale, tb + kvh * D, tb + (p.Hkv + kvh) * D, p.tail_ts, tl, o + G * D, o, tml);
  if (p.vmask != nullptr && tl > 0) {
    float x[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = o[w * D + lane * 4 + i];
    warp_selrot(x, p.vmask + kvh * (D / p.rb), p.rb, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) o[w * D + lane * 4 + i] = x[i];
    __syncthreads();
  }
  float x[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] = o[w * D + lane * 4 + i];
  if (p.sk_grid > 0) {  // stream-K body (sk_pair): the window parts after the body parts of the pair
    __shared__ int skb[2 * SK_MAXB + 2];
    sk_prefix(p, skb, skb + SK_MAXB + 1);
    const PartMap pm = sk_pair(p, skb, b, kvh);
    merge_finalize(p, x, tml[2 * w], tml[2 * w + 1], b, kvh, pm.t0 + j, pm, pm.nbody + p.tail_nslot, &last_flag);
    return;
  }
  const int base = (b * p.Hkv + kvh) * p.nsplit;
  const PartMap pm{base, 1, 0, ns_b + p.tail_nslot, 0};
  merge_finalize(p, x, tml[2 * w], tml[2 * w + 1], b, kvh, base + ns_b + j, pm, ns_b + p.tail_nslot, &last_flag);
}

#ifndef ATTN2_MINB
#define ATTN2_MINB 1
#endif
#ifndef ATTN2_SK_MINB  // stream-K variants: min CTAs / SM of __launch_bounds__ (1: 128 registers = 4 CTAs / SM anyway;
#define ATTN2_SK_MINB 1  // a bound of 4 changes the ptxas schedule: +13 % cycles per block, measured on variant 264)
#endif
// MODE bit 0: PV path (0 = FP8 PV with P' = p * V scale, 1 = f16 PV); bit 1: page loader (0 = cp.async.bulk issued by
// one lane per warp, 1 = 16-B cp.async spread over all threads, completion via cp.async.mbarrier.arrive)
template <int F, int P, int ST, int MODE>
__global__ void __launch_bounds__(128, (MODE & 512) ? ATTN2_SK_MINB : ATTN2_MINB) k_attn2(const PagedParams p) {
  constexpr int PVF16 = MODE & 1;
  constexpr bool CPA = (MODE & 2) != 0;
  using C = Cfg2<F, P, ST, MODE>;
  using L = Lay<F, P>;
  constexpr int NQ = C::NQ, NG = F2<F>::NG, CHUNK = L::CHUNK, NPB = L::NPB, SBE = F2<F>::SBE;
  constexpr bool FP8PV = !PVF16;
  constexpr bool SSPV = (MODE & 4) != 0;  // PV A operand (V^T) from shared memory: no V fragments held during PV
  // MODE bit 4 (with SS PV): PV(n-1) is not waited for at the top of block n but only right before the softmax barrier
  // of block n (wait_group 1: QK(n+1), committed later, stays in flight); the V^T tiles are double-buffered
  constexpr bool SPLITW = (MODE & 16) != 0;
  static_assert(!SPLITW || SSPV, "split waits need the SS PV path");
  // MODE bit 8 (DEFER): PV(n) runs under the scores / softmax / P' of block n+1, QK(n+1) under the softmax barrier.
  // Order in block n: [PV(n-1) in flight] scores(n), softmax(n), P'(n) -> wait (PV(n-1)), O += PV(n-1) -> V fragments
  // (n), K fragments (n+1) -> QK(n+1) issue -> fence, barrier (slow path) -> wait (QK(n+1)) -> PV(n) issue.  Every wait
  // is a wait_group 0 and every fragment is produced while no wgmma is in flight (ptxas 12.9 serializes all wgmmas if
  // accumulators are read after a wait_group 1, or fragments are defined while a wgmma is in flight).
  constexpr bool DEFER = (MODE & 256) != 0;
  static_assert(!DEFER || (FP8PV && !SSPV && !SPLITW && !CPA), "DEFER: FP8 PV, register fragments, bulk loader");
  // MODE bit 9 (SK): stream-K schedule (sk_prefix ...): grid = p.sk_grid CTAs, each a contiguous, equal range of the
  // flattened (sequence, kv head, block) space, possibly several segments; the bulk-copy ring runs on across segments
  constexpr bool SK = (MODE & 512) != 0;
  static_assert(!SK || DEFER, "SK: on top of the DEFER variant");
  static_assert(!(FP8PV && F == A2_NVFP4), "NVFP4 needs the f16 PV path");
  static_assert(!SSPV || FP8PV, "SS PV: FP8 path only");
  // INT4A (asymmetric).  Keys, on the CUDA cores: score = sum_g s_g (q . c_g - z_g Q_g) per (token, group), Q_g = sum of
  // the QK operand over group g.  Values, on the tensor cores: o_d = sum_t P'_t c_td - Z_g(d), Z = P'^T . z per block with
  // the P' tiles of PV (already in shared memory) as the A operand (rows = (group, head, hi/lo)) and a 4 x 32 zero-point
  // tile per k-step as B (columns = groups), built by 16 lanes of one warp: two m64n8k32 per block, in PV's commit
  // group; the diagonal (row group == column) is accumulated by the lanes holding it and subtracted once per CTA
  constexpr bool ASYM = F2<F>::ASYM;
  constexpr bool ZVT = ASYM && ATTN2_ZVT != 0, ZVS = ASYM && ATTN2_ZVT == 0;  // value zero points: tensor / scalar
  static_assert(!ASYM || (FP8PV && !SSPV && !CPA), "INT4A: FP8 PV, register V fragments, bulk loader");
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ __align__(8) uint64_t mbar[ST];
  __shared__ float red_m[NWARP][G], red_l[NWARP][G], sfac_s[G];
  __shared__ float qs_s[G][4];  // INT4A: 2^15 * sum over group j of the QK operand q_hi + q_lo / 16 of head h
  __shared__ float zred[4][G][2];  // INT4A (tensor V): value zero-point sums [group][head][hi / lo]
  __shared__ float zredw[NWARP][G][4];  // INT4A (scalar V): per-warp sums [warp][head][group]
  __shared__ int red_e[NWARP][4];
  __shared__ int last_flag;
  __shared__ int sks[SK_NSTATE];  // SK: segment state (sk_setup)

  int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, gid = lane >> 2, tig = lane & 3;
  const uint32_t sbase = smem_u32(smem);
  const uint32_t qt = sbase + C::QT_OFF, pt = sbase + C::PT_OFF;
  // the CTA's work: split mode = one segment (sequence b, kv head kvh, tokens [c0, c1)) given by blockIdx; SK = the
  // blocks [r0, r1) of the flattened space (nloc blocks, one or more segments)
  // programmatic dependent launch (attn2_set_pdl): the CTA may be resident before the preceding kernel has finished;
  // nothing produced upstream (q, KV pages, block table, lengths) is read before this point (no-op without PDL)
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  int kvh, split = 0, b, len, c0, c1, ns_b = 0, nloc;
  int* sk_pre = reinterpret_cast<int*>(smem + C::SK_OFF);
  int* sk_len = sk_pre + SK_MAXB + 1;
  int r0 = 0, r1 = 0;
  bool sk_two = true;  // SK: the CTA's range covers at most 2 sequences (unrolled segment path)
  // first copies: the issuing lanes (warp w < NPB, lane 0: page w of blocks 0 .. ST-2) read their page ids straight
  // from the block table, so that the copies start before the CTA's page table is staged; split mode: these reads
  // and q do not wait for the sequence length (only the copy itself does)
  unsigned long long ea[ST - 1];  // chunk addresses of the first copies (0 = no page)
  if constexpr (!SK) {
    kvh = blockIdx.x; split = blockIdx.y; b = blockIdx.z;
    c0 = split * p.chunk;
    len = p.seq_lens[b];
    if (!CPA && w < NPB && lane == 0) {
      int pg[ST - 1];
#pragma unroll
      for (int s = 0; s < ST - 1; ++s) {  // (in-bounds of the block-table row; validity against len below)
        const int pi = s * NPB + w;
        // in-bounds of the block-table row: its length is bt_stride; bt_stride == 0 broadcasts one row to every
        // sequence (row length unknown), so the read is bounded by the sequence length instead
        const bool inb = p.bt_stride > 0 ? c0 / P + pi < p.bt_stride : c0 + pi * P < len;
        pg[s] = (pi * P < p.chunk && inb) ? __ldg(p.block_table + static_cast<size_t>(b) * p.bt_stride + c0 / P + pi) : 0;
      }
      c1 = min(len, c0 + p.chunk);
#pragma unroll
      for (int s = 0; s < ST - 1; ++s)
        ea[s] = c0 + (s * NPB + w) * P < c1
                    ? reinterpret_cast<unsigned long long>(p.pool + (static_cast<uint64_t>(pg[s]) * static_cast<uint32_t>(p.Hkv) + kvh) * CHUNK)
                    : 0ull;
    }
    c1 = min(len, c0 + p.chunk);
    ns_b = max(1, (len + p.chunk - 1) / p.chunk);  // active splits of this sequence
    if (split >= ns_b || len <= 0) return;  // (len 0: the staging-window kernel writes the output)
    nloc = (c1 - c0 + BLK - 1) / BLK;
  } else {
    sk_prefix(p, sk_pre, sk_len);
    const uint32_t T = sk_pre[p.B];
    const int jr = static_cast<int>(blockIdx.x) / p.Hkv;  // range index of the CTA within its kv head
    if (static_cast<uint32_t>(jr) >= T) return;  // (no work for this CTA; also T = 0)
    kvh = static_cast<int>(blockIdx.x) - jr * p.Hkv;
    const uint32_t Ge = sk_ctas(p, T);
    r0 = static_cast<int>(udiv64(static_cast<uint64_t>(jr) * T, Ge));
    r1 = static_cast<int>(udiv64(static_cast<uint64_t>(jr + 1) * T, Ge));
    nloc = r1 - r0;
    b = sk_find(sk_pre, p.B, r0);  // first sequence of the range (q of the first segment, page table)
    sk_two = sk_pre[min(b + 2, p.B)] >= r1;  // the range ends within sequence b or b + 1
    len = c0 = c1 = 0;  // (the segment loop reads the segment from sks)
    if (tid == 0) {
      sks[SK_R0] = r0; sks[SK_R1] = r1; sks[SK_KVH] = kvh;
      sk_setup(p, sk_pre, sk_len, kvh, 0, r0, r1, b, sks);
    }
    if (w < NPB && lane == 0) {
      int bb = b;
#pragma unroll
      for (int s = 0; s < ST - 1; ++s) {
        const int x = r0 + s;
        ea[s] = 0ull;
        if (x < r1) {
          while (sk_pre[bb + 1] <= x) ++bb;
          const int pi = (x - sk_pre[bb]) * NPB + w;
          if (pi * P < sk_len[bb]) {
            const int pg = __ldg(p.block_table + static_cast<size_t>(bb) * p.bt_stride + pi);
            ea[s] = reinterpret_cast<unsigned long long>(p.pool + (static_cast<uint64_t>(pg) * static_cast<uint32_t>(p.Hkv) + kvh) * CHUNK);
          }
        }
      }
    }
  }
  // q of head w (this lane's 4 elements): loaded first, its latency overlaps the prologue
  uint2 qraw = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(p.q) + b * p.q_bs +
                                              (kvh * G + w) * D + lane * 4);

  // ---------------------------------------------------------------- prologue  // @prof:other
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < ST; ++s) mbar_init(smem_u32(&mbar[s]), CPA ? 128 : NPB);  // cp.async: one async arrive per thread
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  // zero Q^T (NVFP4: the masked half-columns) and both P^T buffers (f16 path: the zero columns)
  for (int i = tid; i < (C::QT_BYTES + 2 * C::PT_BYTES) / 16; i += 128)
    reinterpret_cast<uint4*>(smem + C::QT_OFF)[i] = make_uint4(0, 0, 0, 0);
  if constexpr (F2<F>::ASYM && ATTN2_ZVT)  // zero-point tiles: rows 4..7 (unused columns) stay zero
    for (int i = tid; i < C::ZB_BYTES / 16; i += 128) reinterpret_cast<uint4*>(smem + C::ZB_OFF)[i] = make_uint4(0, 0, 0, 0);
  __syncthreads();
  const uint32_t mbar0 = smem_u32(&mbar[0]);
  const uint32_t dst_w = opaque(sbase + C::RING_OFF + w * L::PSTR);
  if constexpr (!CPA) {  // first copies (mbarriers initialized and visible: barrier above), before the page table
    if (w < NPB && lane == 0) {
#pragma unroll
      for (int s = 0; s < ST - 1; ++s) {
        if (s < nloc) {
          const uint32_t mb = mbar0 + 8 * s;
          mbar_expect_tx(mb, ea[s] != 0ull ? CHUNK : 0);
          if (ea[s] != 0ull) bulk_g2s(dst_w + s * L::STAGE, reinterpret_cast<const void*>(ea[s]), CHUNK, mb);
        }
      }
    }
  }
  int ntok = max(c1 - c0, 0), nblk = (ntok + BLK - 1) / BLK;
  const uint8_t* poolh = p.pool + static_cast<size_t>(kvh) * CHUNK;
  {  // page ids (cp.async loader) / page chunk addresses (bulk loader) of this CTA -> smem (no dependent global loads
     // and no address arithmetic in the pipeline); all block-table reads of a thread in flight together
    int* pidx = reinterpret_cast<int*>(smem + C::PIDX_OFF);
    unsigned long long* srcs = reinterpret_cast<unsigned long long*>(smem + C::SRC_OFF);
    if constexpr (SK) {  // entry (local block j, page pp) of the CTA's range: chunk address, 0 = no page
      constexpr int NE = (MAX_CHUNK / BLK) * NPB / 128;  // entries per thread
      int bb = b, pgv[NE], kept[NE];  // (entries ascending: the sequence only moves forward)
#pragma unroll
      for (int k = 0; k < NE; ++k) {
        const int e = tid + 128 * k;
        const int x = r0 + e / NPB, pp = e % NPB;
        kept[k] = 0;
        pgv[k] = 0;
        if (e < nloc * NPB) {
          while (sk_pre[bb + 1] <= x) ++bb;
          const int pi = (x - sk_pre[bb]) * NPB + pp;
          if (pi * P < sk_len[bb]) {
            pgv[k] = __ldg(p.block_table + static_cast<size_t>(bb) * p.bt_stride + pi);
            kept[k] = 1;
          }
        }
      }
#pragma unroll
      for (int k = 0; k < NE; ++k) {
        const int e = tid + 128 * k;
        if (e < nloc * NPB)
          srcs[e] = kept[k] ? reinterpret_cast<unsigned long long>(p.pool + (static_cast<uint64_t>(pgv[k]) * static_cast<uint32_t>(p.Hkv) + kvh) * CHUNK) : 0ull;
      }
    } else {
      const int* bt = p.block_table + static_cast<size_t>(b) * p.bt_stride + c0 / P;
      const int npg = (ntok + P - 1) / P;
      constexpr int NI = MAX_CHUNK / P / 128;  // pages per thread (chunk <= MAX_CHUNK)
      int pgv[NI];
#pragma unroll
      for (int k = 0; k < NI; ++k) pgv[k] = tid + 128 * k < npg ? __ldg(bt + tid + 128 * k) : 0;
#pragma unroll
      for (int k = 0; k < NI; ++k) {
        const int i = tid + 128 * k;
        if (i < npg) {
          if constexpr (CPA) pidx[i] = pgv[k];
          else srcs[i] = reinterpret_cast<unsigned long long>(poolh + static_cast<uint64_t>(pgv[k]) * (static_cast<uint32_t>(p.Hkv) * CHUNK));
        }
      }
    }
  }
  const int* pidx_s = reinterpret_cast<const int*>(smem + C::PIDX_OFF);
  const int npg_all = (ntok + P - 1) / P;
  const uint32_t srcw = opaque(sbase + C::SRC_OFF + 8 * w);  // + 8 NPB nb: chunk address of page w of block nb
  const uint32_t pstride = static_cast<uint32_t>(p.Hkv) * CHUNK;  // bytes per page (all kv heads)
  // warp w < NPB, lane 0: bulk-copy page w of block nb into stage nb % ST (the copies of a block are spread over the
  // warps: a single issuing warp delays the whole warpgroup at the next wgmma by ~300 cycles per block)
  // cp.async loader: warp w copies 16-B units [w Q, (w+1) Q) of the block (Q = NPB * CHUNK / 64): unit w Q + lane + 32 j
  constexpr int UPP = CHUNK / 16, UQ = NPB * UPP / 4, NJ = (UQ + 31) / 32;
  static_assert((NPB * UPP) % 4 == 0, "block units must split over the 4 warps");
  uint32_t cpo[NJ];  // per unit j: (page in block << 16) | byte offset in the page; 0xFFFFFFFF = none
#pragma unroll
  for (int j = 0; j < NJ; ++j) {
    const int ur = lane + 32 * j, u = w * UQ + ur;
    cpo[j] = (CPA && ur < UQ) ? (static_cast<uint32_t>(u / UPP) << 16) | static_cast<uint32_t>((u % UPP) * 16) : 0xFFFFFFFFu;
  }
  auto issue = [&](int nb) {  // @prof:load
    const int s = static_cast<int>(static_cast<uint32_t>(nb) % ST);  // (unsigned: no sign fix-ups)
    const uint32_t mb = mbar0 + 8 * s;
    if constexpr (CPA) {
      const int npv = min(NPB, (ntok - nb * BLK + P - 1) / P);  // pages of this block inside the range
      const uint32_t dst = sbase + C::RING_OFF + s * L::STAGE;
#pragma unroll
      for (int j = 0; j < NJ; ++j) {
        const int pp = static_cast<int>(cpo[j] >> 16);
        if (cpo[j] != 0xFFFFFFFFu && pp < npv) {
          const uint32_t page = static_cast<uint32_t>(pidx_s[nb * NPB + pp]);
          const uint32_t off = cpo[j] & 0xFFFFu;
          cp16(dst + pp * L::PSTR + off, poolh + static_cast<uint64_t>(page) * pstride + off);
        }
      }
      cp_mbar_arrive(mb);
    } else if constexpr (SK) {  // (no-page entries are 0)
      unsigned long long src;
      asm volatile("ld.shared.u64 %0, [%1];\n" : "=l"(src) : "r"(srcw + 8 * NPB * nb));
      mbar_expect_tx(mb, src != 0ull ? CHUNK : 0);
      if (src != 0ull) bulk_g2s(dst_w + s * L::STAGE, reinterpret_cast<const void*>(src), CHUNK, mb);
    } else {
      const bool has = nb * NPB + w < npg_all;
      mbar_expect_tx(mb, has ? CHUNK : 0);
      if (has) {
        unsigned long long src;
        asm volatile("ld.shared.u64 %0, [%1];\n" : "=l"(src) : "r"(srcw + 8 * NPB * nb));
        bulk_g2s(dst_w + s * L::STAGE, reinterpret_cast<const void*>(src), CHUNK, mb);
      }
    }
  };  // @prof:other
  __syncthreads();  // page ids staged
  // ---------------------------------------------------------------- segments (split mode: exactly one)
  // One segment = q prologue, main loop, partial / merge.  Stream-K: a range covering at most 2 sequences runs the
  // segment code twice, unrolled; only longer lists of segments use a loop.  (A loop around the segment code made
  // ptxas keep the wgmma descriptors in regular registers, converted with R2UR every block, and cost 7-9 % per
  // block, measured; the unrolled path has no backward branch around the main loop.)  Returns: another segment.
  int jbase = 0;  // first local block of the segment (ring position)
  auto segment = [&]() -> bool {
  if constexpr (SK) {  // segment state from shared memory only (nothing carried in registers across segments)
    // thread indices redefined inside the segment loop: address arithmetic of the segment prologue / epilogue is then
    // not hoisted out of the loop (it would stay live through the main loop: +10 registers, measured)
    tid = static_cast<int>(opaque(static_cast<uint32_t>(tid)));
    lane = tid & 31; w = tid >> 5; gid = lane >> 2; tig = lane & 3;
    __syncthreads();  // (sks written by thread 0)
    jbase = __shfl_sync(FULL, sks[SK_JB], 0);  // (shuffles: warp-uniform values for the compiler, test)
    b = __shfl_sync(FULL, sks[SK_B], 0); c0 = __shfl_sync(FULL, sks[SK_C0], 0); c1 = __shfl_sync(FULL, sks[SK_C1], 0);
    kvh = __shfl_sync(FULL, sks[SK_KVH], 0);
    nloc = __shfl_sync(FULL, sks[SK_R1] - sks[SK_R0], 0);
    ntok = max(c1 - c0, 0);
    nblk = (ntok + BLK - 1) / BLK;
    if (jbase > 0)  // (the first segment's q was loaded in the prologue)
      qraw = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(p.q) + b * p.q_bs + (kvh * G + w) * D + lane * 4);
  }
  if constexpr (CPA) {
#pragma unroll
    for (int s = 0; s < ST - 1; ++s)
      if (s < nblk) issue(s);
  }
  {  // q of head w: FlagRot, softmax scale, per-head power-of-two prescale, split into e4m3 hi + 16 lo -> Q^T tiles
    const uint2 raw = qraw;
    float x[4] = {__uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xffff0000u),
                  __uint_as_float(raw.y << 16), __uint_as_float(raw.y & 0xffff0000u)};
    if (p.qrot) warp_selrot(x, p.kmask + kvh * (D / p.rb), p.rb, lane);
    float am = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) { x[i] *= p.qscale; am = fmaxf(am, fabsf(x[i])); }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(FULL, am, o));
    // kq = 2^(6 - floor(log2 am)): max |q * kq| in [64, 128); lo is stored x LO_SCALE (stays in the e4m3 normal range)
    const int ea = am > 0.f ? static_cast<int>((__float_as_uint(am) >> 23) & 0xFF) - 127 : 0;
    const float kq = __uint_as_float(static_cast<uint32_t>(127 + 6 - ea) << 23);
    uint32_t hi = 0, lo = 0;
#pragma unroll
    for (int i = 0; i < 4; i += 2) {
      const float a0 = x[i] * kq, a1 = x[i + 1] * kq;
      const uint32_t h = f32x2_e4m3x2(a0, a1);
      const float2 hf = h2f2(e4m3x2_f16x2_lo(h));
      const uint32_t l = f32x2_e4m3x2((a0 - hf.x) * LO_SCALE, (a1 - hf.y) * LO_SCALE);
      hi |= h << (8 * i);
      lo |= l << (8 * i);
    }
    const int d0 = lane * 4, j = d0 / 32, k = d0 % 32;  // k-step j covers d = 32 j + k
    const int half = k / 16;
    const int colh = (NQ == 16 ? 8 * half : 0) + 2 * w;  // hi column of head w (lo = +1)
    const uint32_t a = qt + j * (NQ * 32) + (k / 16) * 128 + (colh / 8) * 256 + (colh % 8) * 16 + (k % 16);
    sts32(a, hi);
    sts32(a + 16, lo);
    if constexpr (ASYM) {  // the effective q of the QK wgmma (hi + lo / 16) summed over the group, times 2^-9 (code
                           // value c 2^-9) and 2^24 (the zero point is read as the FP16 subnormal z 2^-24)
      const float2 h01 = h2f2(e4m3x2_f16x2_lo(hi)), h23 = h2f2(e4m3x2_f16x2_hi(hi));
      const float2 l01 = h2f2(e4m3x2_f16x2_lo(lo)), l23 = h2f2(e4m3x2_f16x2_hi(lo));
      float sq = ((h01.x + h01.y) + (h23.x + h23.y)) + ((l01.x + l01.y) + (l23.x + l23.y)) * LO_INV;
#pragma unroll
      for (int o = 1; o < 8; o <<= 1) sq += __shfl_xor_sync(FULL, sq, o);
      if ((lane & 7) == 0) qs_s[w][j] = sq * 32768.f;
    }
    if (lane == 0) {
      float f = F2<F>::FV / kq;
      if constexpr (F == A2_NVFP4) f /= __ldg(p.gscale);
      sfac_s[w] = f;
    }
  }
  fence_async_smem();
  __syncthreads();
  const float sfac = sfac_s[tig];
  float QS[4] = {0.f, 0.f, 0.f, 0.f};  // INT4A: zero-point weights of head tig per group
  if constexpr (ASYM) {
#if !defined(ATTN2_NO_ZP) && !defined(ATTN2_NO_KZP)  // negative control (wrong results): key zero points ignored
#pragma unroll
    for (int j = 0; j < 4; ++j) QS[j] = qs_s[tig][j];
#endif
  }

  // ---------------------------------------------------------------- per-thread constants (per segment: thread indices opaque in SK mode, so not hoisted)
  // QK rows: row gid -> block token tau0, row gid+8 -> tau1 = tau0 + 2 (adjacent k positions of the PV operand)
  const int bq = gid & 1, ipq = (gid >> 1) & 1, qbq = gid >> 2;
  const int tau0 = 16 * w + 8 * qbq + 4 * ipq + bq, tau1 = tau0 + 2;
  const uint32_t kro0 = opaque((tau0 / P) * L::PSTR + L::K_OFF + krow_off<F>(tau0 % P) + L::KTB * tig);
  const uint32_t kro1 = opaque((tau1 / P) * L::PSTR + L::K_OFF + krow_off<F>(tau1 % P) + L::KTB * tig);
  const uint32_t kso0 = opaque((tau0 / P) * L::PSTR + L::KS_OFF + (tau0 % P) * F2<F>::SB);
  const uint32_t kso1 = opaque((tau1 / P) * L::PSTR + L::KS_OFF + (tau1 % P) * F2<F>::SB);
#ifdef ATTN2_OPAQUE_DESC  // (descriptors pinned in regular registers: measured 0.5 % slower)
  const uint64_t qd0 = opaque64(sdesc(qt));  // k-step jj: + jj * NQ * 32 >> 4
  const uint64_t pdesc0 = opaque64(sdesc_ls(pt, C::PLBO, C::PSBO));  // + (buffer * PT_BYTES + tile offset) >> 4
  const uint64_t vtd0 = SSPV ? opaque64(sdesc(sbase + C::VT_OFF)) : 0ull;  // SS PV: tile (ks, m) at + 2048 (2 ks + m)
#else  // experiment: descriptors left to the compiler (uniform registers)
  const uint64_t qd0 = sdesc(qt);
  const uint64_t pdesc0 = sdesc_ls(pt, C::PLBO, C::PSBO);
  const uint64_t vtd0 = SSPV ? sdesc(sbase + C::VT_OFF) : 0ull;
#endif
  // V fragment entry offsets: FP8 path [ks]: tokens 32 ks + 8 tig..+7; f16 path [u]: tokens 16 tig + 8 u..+7
  uint32_t vo[2];
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int tb = FP8PV ? 32 * u + 8 * tig : 16 * tig + 8 * u;
    vo[u] = opaque((tb / P) * L::PSTR + L::V_OFF + ((w * 8 + gid) * (P / 8) + (tb % P) / 8) * L::VE);
  }
  // FP8 path: V scales of tau0 / tau1 (same 8-chunk and parity -> adjacent), group g at + g * P * SBE
  const int cp0 = (tau0 % P) % 8;
  const uint32_t vso = opaque((tau0 / P) * L::PSTR + L::VS_OFF + (((tau0 % P) / 8) * 8 + ((cp0 & 1) ? 4 + (cp0 >> 1) : (cp0 >> 1))) * SBE);
  // P^T write offset in the buffer (FP8: tile (ks = w >> 1, m = g >> 1), column 8 (g & 1) + 2 tig (+1 lo): + 256 g)
  const int k0p = 16 * bq + 4 * (2 * (w & 1) + qbq) + 2 * ipq;
  const uint32_t pwo = opaque(FP8PV ? ((w >> 1) * 2) * C::PTILE + (k0p / 16) * C::PLBO + (2 * tig) * 16 + (k0p % 16)
                                    : (2 * qbq + bq) * 256 + ipq * 128 + tig * 32 + w * 4);

  // SS PV: register fragment (row 16 w + gid (+8), k 4 tig (+16)) -> canonical tile position (row / 8) * 256 +
  // (k / 16) * 128 + (row % 8) * 16 + k % 16
  const uint32_t vto = opaque(sbase + C::VT_OFF + (2 * w) * 256 + gid * 16 + 4 * tig);
  float O[2][2] = {{0.f, 0.f}, {0.f, 0.f}};  // head tig, d = 64 m + 16 w + gid + 8 r
  float lsum = 0.f, mref = NEG_INF;
  // INT4A: Z accumulator of the block in flight (rows 16 w + gid (+8), columns 2 tig, 2 tig + 1); lanes with w < 2 and
  // tig == w hold the diagonal: zd[0] = Z(group 2 w, head gid >> 1, hi/lo gid & 1), zd[3] = the same for group 2 w + 1
  float zd[4] = {0.f, 0.f, 0.f, 0.f};
  float zs0 = 0.f, zs1 = 0.f;  // their sums over the CTA's blocks (online-softmax rescaled as O)
  float zacc[4] = {0.f, 0.f, 0.f, 0.f};  // INT4A scalar V: sum over this thread's tokens of P' z per group (head tig)
  // zero-point tile builder (warp 3, lanes 0..15): group lane >> 2, k-step (lane >> 1) & 1, k half lane & 1 (even / odd
  // tokens = the A / B halves of the 8-token V scale chunks of the k-step); 4 chunks at + zboff(c)
  const uint32_t zbsrc = opaque(((32 * ((lane >> 1) & 1)) / P) * L::PSTR + L::VS_OFF + (lane >> 2 & 3) * P * 2 +
                                (((32 * ((lane >> 1) & 1)) % P) / 8) * 16 + 8 * (lane & 1));
  const uint32_t zbdst = sbase + C::ZB_OFF + ((lane >> 1) & 1) * 256 + (lane & 1) * 128 + (lane >> 2 & 3) * 16;
  const uint64_t zbdesc0 = sdesc_ls(sbase + C::ZB_OFF, 128, 256);  // + (buf * 512 + ks * 256) >> 4
#ifndef ATTN2_ZBW
#define ATTN2_ZBW 3
#endif
  constexpr int ZBW = ATTN2_ZBW;  // builder warp
  int eref[NG];      // FP8 PV: V scales of group g are applied as s / 2^eref[g] (lazily updated, warpgroup-uniform)
  uint32_t inv2[NG]; // INT4: f16x2 of 2^-eref[g]
#pragma unroll
  for (int g = 0; g < NG; ++g) { eref[g] = EREF_NONE; inv2[g] = 0u; }
  constexpr int NACC = FP8PV ? 8 : 4;
  float pacc[2][NACC];
  float sacc[4][NQ / 2];
  uint32_t ka[4][4];
  uint2 kraw0, kraw1;  // raw K scales of rows tau0, tau1 of the block whose QK is in flight
  // V fragments: read asynchronously by PV(n) until the wait at the top of block n+1 -> declared outside the loop and
  // pinned after that wait, so that the compiler cannot reuse their registers earlier
  uint32_t va[4][2][4];  // FP8 path: [ks (2 used)][m][4];  f16 path: [k16 step][m][4]
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int k = 0; k < 4; ++k) va[i][m][k] = 0u;

  // K fragments + K scales of block nb (stage nb % ST, already waited for) and the QK wgmmas (one fresh accumulator
  // per k-step / scale group)
  auto qk_mma = [&]() {
    wg_fence();
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      if constexpr (NQ == 8) wg_e4m3_n8<0>(*reinterpret_cast<float(*)[4]>(sacc[jj]), ka[jj], qd0 + ((jj * NQ * 32) >> 4));
      else wg_e4m3_n16<0>(*reinterpret_cast<float(*)[8]>(sacc[jj]), ka[jj], qd0 + ((jj * NQ * 32) >> 4));
    }
    wg_commit();
  };
  auto qk_decode = [&](int nb) {  // K fragments + raw K scales of block nb
    const uint32_t sb = sbase + C::RING_OFF + (static_cast<uint32_t>(nb) % ST) * L::STAGE;
    k_frags<F>(sb + kro0, sb + kro1, ka, 0);
    kraw0 = lds64(sb + kso0);
    kraw1 = lds64(sb + kso1);
  };
  auto qk_issue = [&](int nb) {  // @prof:decode
    const uint32_t sb = sbase + C::RING_OFF + (static_cast<uint32_t>(nb) % ST) * L::STAGE;
    k_frags<F>(sb + kro0, sb + kro1, ka, 0);
    if constexpr (F == A2_MXFP4) {
      kraw0 = make_uint2(lds32(sb + kso0), 0u);
      kraw1 = make_uint2(lds32(sb + kso1), 0u);
    } else {
      kraw0 = lds64(sb + kso0);
      kraw1 = lds64(sb + kso1);
    }  // @prof:mma
#ifndef ATTN2_DBG_NOQK
    wg_fence();
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      if constexpr (NQ == 8) wg_e4m3_n8<0>(*reinterpret_cast<float(*)[4]>(sacc[jj]), ka[jj], qd0 + ((jj * NQ * 32) >> 4));
      else wg_e4m3_n16<0>(*reinterpret_cast<float(*)[8]>(sacc[jj]), ka[jj], qd0 + ((jj * NQ * 32) >> 4));
    }
    wg_commit();  // @prof:other
#else
#pragma unroll
    for (int jj = 0; jj < 4; ++jj)
#pragma unroll
      for (int i = 0; i < NQ / 2; ++i) sacc[jj][i] = __uint_as_float(ka[jj][i & 3] & 0x3F000000u);
#endif
  };
  auto pv_accumulate = [&]() {  // @prof:mma
    if constexpr (F2<F>::ASYM && ATTN2_ZVT) { zs0 += zd[0]; zs1 += zd[3]; }
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      if constexpr (FP8PV) {
        const bool g1 = (w >> 1) != 0;  // this warp's d rows belong to column group 1
        O[m][0] += g1 ? fmaf(pacc[m][5], LO_INV, pacc[m][4]) : fmaf(pacc[m][1], LO_INV, pacc[m][0]);
        O[m][1] += g1 ? fmaf(pacc[m][7], LO_INV, pacc[m][6]) : fmaf(pacc[m][3], LO_INV, pacc[m][2]);
      } else {
        O[m][0] += pacc[m][0];
        O[m][1] += pacc[m][2];
      }
    }
  };  // @prof:other

  mbar_wait(smem_u32(&mbar[static_cast<uint32_t>(jbase) % ST]), (static_cast<uint32_t>(jbase) / ST) & 1);  // nblk >= 1
  qk_issue(jbase);
  if constexpr (SPLITW) wg_commit();  // empty group: the first wait_group 1 then covers QK(0)
  if constexpr (DEFER) {  // QK(0) before the loop (the loop expects the scores' accumulators ready)
    wg_wait_all();
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) wg_wait0(sacc[jj]);
#pragma unroll
    for (int k = 0; k < 4; ++k) pin_regs(ka[k]);
  }
#ifdef ATTN2_TIMING  // per-phase clock64 sums of every warp of CTA (0, 0, 0), written to opart[0..] at the end
  long long tph[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  long long tprev = clock64();
#define TPH(i) do { const long long tn = clock64(); tph[i] += tn - tprev; tprev = tn; } while (0)
#else
#define TPH(i) do { } while (0)
#endif
  // loop over the ring position jb (block jb - jbase of the segment; the split mode has jbase = 0)
  const int cb = c0 - jbase * BLK, jend = jbase + nblk;
  if constexpr (SK) {
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int i = 0; i < NACC; ++i) pacc[m][i] = 0.f;
  }
  // SK: par = double-buffer parity of the P' tiles counted from the segment start (a loop counter from a constant,
  // uniform for the compiler: the PV descriptors stay in uniform registers; jb starts at jbase, read from shared memory,
  // which is not); split mode: jbase = 0, the loop is the block loop (parity jb & 1)
  for (int jb = jbase, par = 0; jb < jend; ++jb, par ^= 1) {
    const int s = static_cast<int>(static_cast<uint32_t>(jb) % ST);
    const uint32_t sb = sbase + C::RING_OFF + s * L::STAGE;
    const int t0 = cb + jb * BLK;
#ifndef ATTN2_DBG_NOLOAD
    if ((CPA || (w < NPB && lane == 0)) && jb + ST - 1 < nloc) issue(jb + ST - 1);  // stage (n-1)%ST released by B1(n-1)
#else  // timing experiment: only the first ST - 1 blocks are loaded, later blocks reuse the stale stages (wrong results)
    if ((CPA || (w < NPB && lane == 0)) && jb + ST - 1 < min(nblk, ST - 1)) issue(jb + ST - 1);
#endif

    // ---------------- (a) wait for QK(n) and PV(n-1); O += PV(n-1); scores of rows tau0, tau1 for head tig
    TPH(0);  // TMA issue (warp 0) + loop
    if constexpr (SPLITW) {
      wg_wait_1();  // QK(n) (committed before PV(n-1))
    } else if constexpr (!DEFER) {
      wg_wait_all();  // @prof:mma
    }
    if constexpr (!DEFER) {  // (DEFER: QK(n) was waited for before PV(n-1) was issued)
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) wg_wait0(sacc[jj]);
#pragma unroll
      for (int k = 0; k < 4; ++k) pin_regs(ka[k]);
    }
    if constexpr (!SPLITW && !DEFER) {
#pragma unroll
      for (int m = 0; m < 2; ++m) wg_wait0(pacc[m]);
      if constexpr (ZVT) wg_wait0(zd);
    }
    if constexpr (!SSPV && !DEFER) {
#pragma unroll
      for (int i = 0; i < (FP8PV ? 2 : 4); ++i) { pin_regs(va[i][0]); pin_regs(va[i][1]); }
    }
    if (!SPLITW && !DEFER && jb > jbase) pv_accumulate();
    if constexpr (ZVT) {
      if (w == ZBW && lane < 16) {  // zero-point tile of block jb (bytes z = e4m3 of z 2^-9), published by barrier B1 (block start: the
                                  // shared-memory latency overlaps the scores)
        uint32_t zw[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const uint2 x = lds64(sb + zbsrc + (P == 16 ? (c >> 1) * L::PSTR + (c & 1) * 16 : 16 * c));
          zw[c] = prmt_b32(x.x, x.y, 0x6420u) & F2<F>::ZM4;
        }
        asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(zbdst + (jb & 1) * 512), "r"(zw[0]), "r"(zw[1]),
                     "r"(zw[2]), "r"(zw[3]) : "memory");
      }
    }
    const bool val0 = t0 + tau0 < c1 && t0 + tau0 >= p.kv_start, val1 = t0 + tau1 < c1 && t0 + tau1 >= p.kv_start;  // kv_start: masked leading tokens (sinks)  // @prof:softmax
    float S0 = 0.f, S1 = 0.f, S0b = 0.f, S1b = 0.f;  // two independent chains per row
    float zf0[4], zf1[4];  // INT4A: key zero points of rows tau0 / tau1 (z 2^-24)
    if constexpr (ASYM) {
      const uint32_t a = kraw0.x & F2<F>::ZM2, b = kraw0.y & F2<F>::ZM2;
      const uint32_t c = kraw1.x & F2<F>::ZM2, d = kraw1.y & F2<F>::ZM2;
      zf0[0] = __low2float(*reinterpret_cast<const __half2*>(&a)); zf0[1] = __high2float(*reinterpret_cast<const __half2*>(&a));
      zf0[2] = __low2float(*reinterpret_cast<const __half2*>(&b)); zf0[3] = __high2float(*reinterpret_cast<const __half2*>(&b));
      zf1[0] = __low2float(*reinterpret_cast<const __half2*>(&c)); zf1[1] = __high2float(*reinterpret_cast<const __half2*>(&c));
      zf1[2] = __low2float(*reinterpret_cast<const __half2*>(&d)); zf1[3] = __high2float(*reinterpret_cast<const __half2*>(&d));
    }
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      float& A0 = (jj & 1) ? S0b : S0;
      float& A1 = (jj & 1) ? S1b : S1;
      const float2 k0 = kscale_step<F>(kraw0, jj), k1 = kscale_step<F>(kraw1, jj);
      if constexpr (ASYM) {  // s (acc - z Q): z from the low 4 bits of the scale word, read as the FP16 subnormal z 2^-24
        A0 = fmaf(k0.x, fmaf(-zf0[jj], QS[jj], fmaf(sacc[jj][1], LO_INV, sacc[jj][0])), A0);
        A1 = fmaf(k1.x, fmaf(-zf1[jj], QS[jj], fmaf(sacc[jj][3], LO_INV, sacc[jj][2])), A1);
      } else if constexpr (NQ == 8) {
        A0 = fmaf(k0.x, fmaf(sacc[jj][1], LO_INV, sacc[jj][0]), A0);
        A1 = fmaf(k1.x, fmaf(sacc[jj][3], LO_INV, sacc[jj][2]), A1);
      } else {
        A0 = fmaf(k0.x, fmaf(sacc[jj][1], LO_INV, sacc[jj][0]), A0);
        A0 = fmaf(k0.y, fmaf(sacc[jj][5], LO_INV, sacc[jj][4]), A0);
        A1 = fmaf(k1.x, fmaf(sacc[jj][3], LO_INV, sacc[jj][2]), A1);
        A1 = fmaf(k1.y, fmaf(sacc[jj][7], LO_INV, sacc[jj][6]), A1);
      }
    }
    S0 += S0b;
    S1 += S1b;
    S0 = val0 ? S0 * sfac : NEG_INF;
    S1 = val1 ? S1 * sfac : NEG_INF;  // @prof:other

    TPH(1);  // wait QK(n) + PV(n-1), O += PV, scores
    // ---------------- (c) V^T fragments of block n (+ per-element scale on the f16 path), V scales
    // @prof:decode
    uint32_t svraw[NG];    // FP8 path: raw V scales of (tau0, tau1) per group (MXFP4: 2 bytes, INT4: 2 halves)
    if constexpr (FP8PV) {
#pragma unroll
      for (int ks = 0; ks < (DEFER ? 0 : 2); ++ks) {
        if constexpr (!SSPV) {
          v_frags<F>(sb + vo[ks], va[ks][0], va[ks][1]);
          continue;
        }
        const uint4 x = lds128(sb + vo[ks]);
        const uint32_t wv[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          if constexpr (SSPV) {  // decoded fragment straight into the tile (ks, m)
            uint32_t a0, a1, a2, a3;
            cvt_ab<F>(wv[2 * m], a0, a2);
            cvt_ab<F>(wv[2 * m + 1], a1, a3);
            const uint32_t t = vto + (ks * 2 + m) * 2048 + (SPLITW ? (jb & 1) * 4 * 2048 : 0);
            sts32(t, a0);
            sts32(t + 256, a1);
            sts32(t + 128, a2);
            sts32(t + 384, a3);
            continue;
          }
          cvt_ab<F>(wv[2 * m], va[ks][m][0], va[ks][m][2]);
          cvt_ab<F>(wv[2 * m + 1], va[ks][m][1], va[ks][m][3]);
        }
      }
#pragma unroll
      for (int g = 0; g < NG; ++g) svraw[g] = F2<F>::I4 ? lds32(sb + vso + g * P * 2) : lds16(sb + vso + g * P);
    } else {
      // lane tig: block tokens 16 tig + 8 u + 0..7 (u = 0, 1); k16-step s: u = s >> 1, s even -> A pairs, odd -> B
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int tb = 16 * tig + 8 * u;
        const uint4 x = lds128(sb + vo[u]);
        const uint32_t wv[4] = {x.x, x.y, x.z, x.w};
        const uint32_t vsb = sb + (tb / P) * L::PSTR + L::VS_OFF + (8 * ((tb % P) / 8)) * SBE;
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          // V scale pairs of group g(m) for this chunk: [A0 A1 A2 A3 | B0 B1 B2 B3] -> f16x2 (A0,A1) (A2,A3) (B0,B1) (B2,B3)
          const int g = F == A2_NVFP4 ? 4 * m + w : 2 * m + (w >> 1);
          const uint32_t sa = vsb + g * P * SBE;
          // valid tokens of this 8-chunk: A slots i (token 2i) iff 2i < nv, B slots i (token 2i+1) iff 2i+1 < nv
          const int nv = c1 - (t0 + tb);
          const int na = min(max((nv + 1) >> 1, 0), 4), nb2 = min(max(nv >> 1, 0), 4);
          uint32_t s2[4];
          if constexpr (F == A2_NVFP4) {
            uint2 sv = lds64(sa);
            sv.x &= na >= 4 ? 0xFFFFFFFFu : (1u << (8 * na)) - 1u;  // zero scales of invalid tokens (stale smem
            sv.y &= nb2 >= 4 ? 0xFFFFFFFFu : (1u << (8 * nb2)) - 1u;  // may hold NaN / Inf)
            s2[0] = e4m3x2_f16x2_lo(sv.x); s2[1] = e4m3x2_f16x2_hi(sv.x);
            s2[2] = e4m3x2_f16x2_lo(sv.y); s2[3] = e4m3x2_f16x2_hi(sv.y);
          } else if constexpr (F == A2_MXFP4) {
            const uint2 sv = lds64(sa);
            const uint32_t svv[2] = {sv.x, sv.y};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              const uint32_t wd = svv[k >> 1], sh = 16 * (k & 1);
              s2[k] = mx64h((wd >> sh) & 0xFFu) | (mx64h((wd >> (sh + 8)) & 0xFFu) << 16);
            }
          } else {
            const uint4 sv = lds128(sa);
            const uint64_t ma = na >= 4 ? ~0ull : (1ull << (16 * na)) - 1ull, mb = nb2 >= 4 ? ~0ull : (1ull << (16 * nb2)) - 1ull;
            s2[0] = sv.x & static_cast<uint32_t>(ma); s2[1] = sv.y & static_cast<uint32_t>(ma >> 32);
            s2[2] = sv.z & static_cast<uint32_t>(mb); s2[3] = sv.w & static_cast<uint32_t>(mb >> 32);
          }
          uint32_t a0, b0, a1, b1;
          cvt_ab<F>(wv[2 * m], a0, b0);      // row gid
          cvt_ab<F>(wv[2 * m + 1], a1, b1);  // row gid + 8
          // step 2u (A pairs): a0/a1 = .lo pair (tokens 0,2), a2/a3 = .hi pair (4,6); step 2u+1: B pairs (1,3), (5,7)
          va[2 * u][m][0] = hmul2u(e4m3x2_f16x2_lo(a0), s2[0]);
          va[2 * u][m][1] = hmul2u(e4m3x2_f16x2_lo(a1), s2[0]);
          va[2 * u][m][2] = hmul2u(e4m3x2_f16x2_hi(a0), s2[1]);
          va[2 * u][m][3] = hmul2u(e4m3x2_f16x2_hi(a1), s2[1]);
          va[2 * u + 1][m][0] = hmul2u(e4m3x2_f16x2_lo(b0), s2[2]);
          va[2 * u + 1][m][1] = hmul2u(e4m3x2_f16x2_lo(b1), s2[2]);
          va[2 * u + 1][m][2] = hmul2u(e4m3x2_f16x2_hi(b0), s2[3]);
          va[2 * u + 1][m][3] = hmul2u(e4m3x2_f16x2_hi(b1), s2[3]);
        }
      }
    }

    // ---------------- (b) QK(n+1), overlapping the softmax of block n.  ptxas treats everything from the first
    // wgmma after a wait to the next wait as one pipeline stage and serializes all wgmmas if a non-wgmma instruction
    // inside it defines wgmma input / accumulator registers (C7513 / C7515): so the V fragments of PV(n) (c) and the K
    // fragments of QK(n+1) are produced before this issue, and QK is issued unconditionally (the last block
    // recomputes QK of its own, still valid stage).
    const int jq = min(jb + 1, jend - 1);  // block of the next QK (the last block recomputes its own)
    TPH(2);  // V fragments
#ifndef ATTN2_DBG_NOLOAD
    if (!DEFER && jq != jb) mbar_wait(smem_u32(&mbar[static_cast<uint32_t>(jq) % ST]), (static_cast<uint32_t>(jq) / ST) & 1);  // @prof:load
#else
    if (jq != jb && jq < ST - 1) mbar_wait(smem_u32(&mbar[jq % ST]), (jq / ST) & 1);
#endif
    TPH(3);  // mbarrier wait (TMA data of block n+1)
    if constexpr (!DEFER) qk_issue(jq);
    TPH(4);  // K fragments + QK issue


    // ---------------- (d) softmax (lazy reference), P / P' -> smem
    const uint32_t ptb = pt + (SK ? par : (jb & 1)) * C::PT_BYTES;  // @prof:softmax
    float p0, p1;
    float zb[4] = {0.f, 0.f, 0.f, 0.f};  // INT4A scalar V: this block's P' z per group (rewritten by the slow path's make_p)
    bool pflag = false;  // some P' would exceed 2^(TAU + SH) (V scale grew): take the slow path
    const uint32_t vmask2 = (val0 ? 0xFFFFu : 0u) | (val1 ? 0xFFFF0000u : 0u);
    auto make_p = [&]() {
      p0 = val0 ? ex2(S0 - mref) : 0.f;
      p1 = val1 ? ex2(S1 - mref) : 0.f;
      __half2 ph = __floats2half2_rn(p0, p1);
      const uint32_t p2 = *reinterpret_cast<uint32_t*>(&ph);
      if constexpr (FP8PV) {
        // P' = p * s_v / 2^eref[g] in f16x2, split hi = RN_e4m3(P'), lo = RN_e4m3(16 (P' - hi)); tile (w >> 1, g >> 1)
        uint32_t pm = 0u;
#pragma unroll
        for (int g = 0; g < NG; ++g) {
          uint32_t s2;
          if constexpr (F2<F>::I4) {  // (INT4A: the word is the scale, zero point bits included)
            s2 = hmul2u(svraw[g], inv2[g]);
          } else {  // f16 bits of 2^(e - 127 - eref) = (e - eref - 112) << 10, clamped to [0, 30] (0 -> 0); per
                    // 16-bit half in offset binary (+0x4000) so that the packed add never borrows across halves
            const uint32_t e2 = __byte_perm(svraw[g], 0u, 0x4140);
            const uint32_t c = static_cast<uint32_t>(0x4000 - (min(max(eref[g], -200), 300) + 112));
            s2 = (clamp_s16x2(e2 + (c | (c << 16)), 0x40004000u, 0x401E401Eu) - 0x40004000u) << 10;
          }
          s2 &= vmask2;  // invalid tokens: P' = 0 (their scales may be NaN / Inf)
          const uint32_t P2 = hmul2u(p2, s2);
          pm = hmax2u(pm, P2);
          if constexpr (ZVS) {  // block part of Z_g: P'(tau0) z(tau0) + P'(tau1) z(tau1) with the P' of PV (hi + lo / 16);
                                // z as fp16 integers (1024 + z) - 1024, f16 products and pair sum (P' z <= 128 * 15);
                                // added to zacc after the slow path
            const uint32_t pt = pprime_store_t(P2, ptb + pwo + (g >> 1) * C::PTILE + (g & 1) * C::PSBO);
#if !defined(ATTN2_NO_ZP) && !defined(ATTN2_NO_VZP)  // negative control (wrong results): value zero points ignored
            const uint32_t zh = hsub2u(lop_or_and(svraw[g], F2<F>::ZM2, 0x64006400u), 0x64006400u);
            const uint32_t pzu = hmul2u(pt, zh);
            const __half2 pz = *reinterpret_cast<const __half2*>(&pzu);
            zb[g] = __half2float(__hadd(__low2half(pz), __high2half(pz)));
#else
            (void)pt;
#endif
          }
#ifndef ATTN2_OLD_PPRIME
          if constexpr (!ZVS) pprime_store(P2, ptb + pwo + (g >> 1) * C::PTILE + (g & 1) * C::PSBO);
#else
          const uint32_t hb = f16x2_e4m3x2(P2);
          const uint32_t lb = f16x2_e4m3x2(hmul2u(hsub2u(P2, e4m3x2_f16x2_lo(hb)), 0x4C004C00u));  // x 16
          sts16(ptb + pwo + (g >> 1) * C::PTILE + (g & 1) * C::PSBO, hb);
          sts16(ptb + pwo + (g >> 1) * C::PTILE + (g & 1) * C::PSBO + 16, lb);
#endif
        }
        const float2 pmf = h2f2(pm);
        pflag = !(fmaxf(pmf.x, pmf.y) <= 128.f);  // also catches Inf / NaN
      } else {
        sts32(ptb + pwo, p2);
      }
    };
    bool flag = (S0 > mref + TAU) || (S1 > mref + TAU);
    make_p();
    TPH(5);  // softmax + P / P' to smem
    flag = flag || pflag;
    if constexpr (SPLITW) {  // PV(n-1) (QK(n+1) stays in flight), accumulated before a possible rescale of O
      wg_wait_1();
#pragma unroll
      for (int m = 0; m < 2; ++m) wg_wait0(pacc[m]);
      if constexpr (ZVT) wg_wait0(zd);
      if (jb > jbase) pv_accumulate();
    }
    if constexpr (DEFER) {  // PV(n-1) done: nothing in flight -> fragments of PV(n) and QK(n+1)
      wg_wait_all();
#pragma unroll
      for (int m = 0; m < 2; ++m) wg_wait0(pacc[m]);
      if constexpr (ZVT) wg_wait0(zd);
#pragma unroll
      for (int i = 0; i < 2; ++i) { pin_regs(va[i][0]); pin_regs(va[i][1]); }
#pragma unroll
      for (int k = 0; k < 4; ++k) pin_regs(ka[k]);
      if (SK || jb > jbase) pv_accumulate();  // (SK: pacc zeroed before the first block: jbase not live here)
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) v_frags<F>(sb + vo[ks], va[ks][0], va[ks][1]);
      if (jq != jb) mbar_wait(smem_u32(&mbar[static_cast<uint32_t>(jq) % ST]), (static_cast<uint32_t>(jq) / ST) & 1);
      qk_decode(jq);
      qk_mma();  // QK(n+1) runs under the barrier below
    }
#ifndef ATTN2_DBG_NOBAR
    fence_async_smem();
    const int anyflag = __syncthreads_or(flag);  // B1: P visible; stage n consumed; any warp needs the slow path?
#else
    const int anyflag = 0;
#endif
    TPH(6);  // fence + barrier B1
#ifdef ATTN2_DBG_NOBAR
    if (jb == jbase || anyflag) {  // timing experiment only: references from the first block
#else
    if (anyflag) {
#endif
      // ---------------- slow path: new warpgroup-uniform references, rescale, rewrite P
#ifdef ATTN2_TIMING
      tph[8] += 1000000000000LL;  // slow-path counter (in the high digits of the PV-issue slot)
#endif
      float bm = fmaxf(S0, S1);
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) bm = fmaxf(bm, __shfl_xor_sync(FULL, bm, o));
      if (gid == 0) red_m[w][tig] = bm;
      if constexpr (FP8PV) {
#pragma unroll
        for (int g = 0; g < NG; ++g) {
          int e0, e1;
          if constexpr (F2<F>::I4) {
            e0 = static_cast<int>((svraw[g] >> 10) & 31) - 15;
            e1 = static_cast<int>((svraw[g] >> 26) & 31) - 15;
          } else {
            e0 = static_cast<int>(svraw[g] & 0xFF) - 127;
            e1 = static_cast<int>((svraw[g] >> 8) & 0xFF) - 127;
          }
          int be = max(val0 ? e0 : EREF_NONE, val1 ? e1 : EREF_NONE);
#pragma unroll
          for (int o = 1; o < 32; o <<= 1) be = max(be, __shfl_xor_sync(FULL, be, o));
          if (lane == 0) red_e[w][g] = be;
        }
      }
      __syncthreads();
      float nm = mref;
#pragma unroll
      for (int ww = 0; ww < NWARP; ++ww) nm = fmaxf(nm, red_m[ww][tig]);
      const float alpha = mref == NEG_INF ? 0.f : ex2(mref - nm);
      float beta[2] = {1.f, 1.f};  // per M tile: group of this thread's d rows = 2 m + (w >> 1)
      float bz0 = 1.f, bz1 = 1.f;
      if constexpr (FP8PV) {
#pragma unroll
        for (int g = 0; g < NG; ++g) {
          int ne = eref[g];
#pragma unroll
          for (int ww = 0; ww < NWARP; ++ww) ne = max(ne, red_e[ww][g]);
          if constexpr (F2<F>::I4) ne = (ne == EREF_NONE) ? ne : min(max(ne, -14), 14);  // 2^-eref in f16
          const float bg = (eref[g] == EREF_NONE || eref[g] - ne < -126)
                               ? 0.f : __uint_as_float(static_cast<uint32_t>(127 + eref[g] - ne) << 23);
          if ((g & 1) == (w >> 1)) beta[g >> 1] = bg;
          if constexpr (ZVS) zacc[g] *= alpha * bg;
          if constexpr (ZVT) {  // betas of the groups of zs0 / zs1
            if (g == 2 * (w & 1)) bz0 = bg;
            if (g == 2 * (w & 1) + 1) bz1 = bg;
          }
          eref[g] = ne;
          if constexpr (F2<F>::I4) {
            const uint32_t h = ne == EREF_NONE ? 0u : static_cast<uint32_t>(15 - ne) << 10;
            inv2[g] = h | (h << 16);
          }
        }
      }
      mref = nm;
#pragma unroll
      for (int m = 0; m < 2; ++m) { O[m][0] *= alpha * beta[m]; O[m][1] *= alpha * beta[m]; }
      lsum *= alpha;
      if constexpr (ZVT) {  // zs0 / zs1 belong to head gid >> 1: its alpha from lane gid >> 1 (tig = that head)
        const float ah = __shfl_sync(FULL, alpha, gid >> 1);
        zs0 *= ah * bz0;
        zs1 *= ah * bz1;
      }
      __syncthreads();  // everyone has read red_*; nobody reads P of this block yet
      make_p();
      fence_async_smem();
      __syncthreads();
    }
    TPH(7);  // slow path
    lsum += p0 + p1;  // @prof:mma
    if constexpr (ZVS) {
#pragma unroll
      for (int g = 0; g < 4; ++g) zacc[g] += zb[g];
    }

    // ---------------- (e) PV(n) (fresh accumulators; added to O after the next wait)
    const uint64_t pdb = pdesc0 + (((SK ? par : (jb & 1)) * C::PT_BYTES) >> 4);
    if constexpr (DEFER) {  // QK(n+1) done before PV(n) is issued (scores of n+1 read its accumulators under PV(n))
      wg_wait_all();
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) wg_wait0(sacc[jj]);
#pragma unroll
      for (int k = 0; k < 4; ++k) pin_regs(ka[k]);
    }
#ifndef ATTN2_DBG_NOPV
    wg_fence();
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      if constexpr (SSPV) {
        const uint64_t vtb = vtd0 + (SPLITW ? (((jb & 1) * 4 * 2048) >> 4) : 0);
        wg_e4m3_n16_ss<0>(pacc[m], vtb + (((0 * 2 + m) * 2048) >> 4), pdb + ((0 * 2 + m) * C::PTILE >> 4));
        wg_e4m3_n16_ss<1>(pacc[m], vtb + (((1 * 2 + m) * 2048) >> 4), pdb + ((1 * 2 + m) * C::PTILE >> 4));
      } else if constexpr (FP8PV) {
        wg_e4m3_n16<0>(pacc[m], va[0][m], pdb + ((0 * 2 + m) * C::PTILE >> 4));
        wg_e4m3_n16<1>(pacc[m], va[1][m], pdb + ((1 * 2 + m) * C::PTILE >> 4));
      } else {
        wg_f16_n8<0>(*reinterpret_cast<float(*)[4]>(pacc[m]), va[0][m], pdb + (0 * 256 >> 4));
        wg_f16_n8<1>(*reinterpret_cast<float(*)[4]>(pacc[m]), va[1][m], pdb + (1 * 256 >> 4));
        wg_f16_n8<1>(*reinterpret_cast<float(*)[4]>(pacc[m]), va[2][m], pdb + (2 * 256 >> 4));
        wg_f16_n8<1>(*reinterpret_cast<float(*)[4]>(pacc[m]), va[3][m], pdb + (3 * 256 >> 4));
      }
    }
    if constexpr (ZVT) {  // Z(n) = P'(n)^T . z(n): A = the P' tiles of k-step ks (rows (group, head, hi/lo)), B = z tile
#if !defined(ATTN2_NO_ZP) && !defined(ATTN2_NO_VZP)
      const uint64_t zbd = zbdesc0 + (((jb & 1) * 512) >> 4);
      const uint64_t pa = pdb;  // (same layout: K-cores PLBO apart, 8-row groups PSBO apart; tiles of a k-step adjacent)
      wg_e4m3_n8_ss<0>(zd, pa, zbd);
      wg_e4m3_n8_ss<1>(zd, pa + ((2 * C::PTILE) >> 4), zbd + (256 >> 4));
#endif
    }
    wg_commit();
#endif
    TPH(8);  // PV issue
  }
#ifdef ATTN2_TIMING
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && lane == 0) {
    for (int i = 0; i < 9; ++i) reinterpret_cast<long long*>(p.opart)[w * 16 + i] = tph[i];
    reinterpret_cast<long long*>(p.opart)[w * 16 + 9] = nblk;
  }
  __syncthreads();
  if (p.nsplit > 0) return false;  // timing build: no output
#endif  // @prof:other
  wg_wait_all();
#pragma unroll
  for (int m = 0; m < 2; ++m) wg_wait0(pacc[m]);
  if constexpr (ZVT) wg_wait0(zd);
  pv_accumulate();

  // ---------------------------------------------------------------- CTA totals, split partials / merge
  int skv[SK_NSTATE];  // SK: the segment state, read before the barriers below (thread 0 rewrites sks after the merge)
  if constexpr (SK) {
#pragma unroll
    for (int i = 0; i < SK_NSTATE; ++i) skv[i] = sks[i];
  }
  float ofac[2];  // per M tile: O_acc -> sum_t p_t v_t
  if constexpr (FP8PV) {
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      const int e = (w >> 1) ? eref[2 * m + 1] : eref[2 * m];
      ofac[m] = F2<F>::FV * (e == EREF_NONE ? 0.f : exp2f(static_cast<float>(e)));
    }
  } else if constexpr (F == A2_NVFP4) {
    ofac[0] = ofac[1] = 64.f / __ldg(p.gscale);
  } else if constexpr (F == A2_MXFP4) {
    ofac[0] = ofac[1] = 0.5f;  // e2m1 * 2^-6 * (2^(e-127) * 2^7)
  } else {
    ofac[0] = ofac[1] = 512.f;
  }
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) lsum += __shfl_xor_sync(FULL, lsum, o);
  if constexpr (ZVS) {  // Z_g of head tig: sum over the warp's rows (lanes of equal tig), then over the warps
#pragma unroll
    for (int g = 0; g < 4; ++g)
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) zacc[g] += __shfl_xor_sync(FULL, zacc[g], o);
    if (gid == 0) {
#pragma unroll
      for (int g = 0; g < 4; ++g) zredw[w][tig][g] = zacc[g];
    }
  }
  if constexpr (ZVT) {  // diagonal lanes publish Z(group, head, hi/lo)
    if (w < 2 && tig == w) {
      zred[2 * w][gid >> 1][gid & 1] = zs0;
      zred[2 * w + 1][gid >> 1][gid & 1] = zs1;
    }
  }
  __syncthreads();  // ring no longer used
  if constexpr (ASYM) {  // o_d -= Z_g(d) (units of O: P' x code bytes c 2^-9)
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      const int g = 2 * m + (w >> 1);
      const float z = ZVT ? fmaf(zred[g][tig][1], LO_INV, zred[g][tig][0])
                          : ((zredw[0][tig][g] + zredw[1][tig][g]) + (zredw[2][tig][g] + zredw[3][tig][g])) * (1.f / 512.f);
      O[m][0] -= z;
      O[m][1] -= z;
    }
  }
  float (*fin)[D] = reinterpret_cast<float (*)[D]>(smem + (SK ? C::PT_OFF : 0));  // [G][D] (SK: the ring is busy)
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int r = 0; r < 2; ++r) fin[tig][64 * m + 16 * w + gid + 8 * r] = O[m][r] * ofac[m];
  if (gid == 0) { red_l[w][tig] = lsum; red_m[w][tig] = mref; }
  __syncthreads();
  float x[4];
  {
    const float4 f4 = reinterpret_cast<const float4*>(fin[w])[lane];
    x[0] = f4.x; x[1] = f4.y; x[2] = f4.z; x[3] = f4.w;
  }
  const float Lw = red_l[0][w] + red_l[1][w] + red_l[2][w] + red_l[3][w];
  if constexpr (SK) {
    const PartMap pm{skv[SK_PB0], skv[SK_PST], skv[SK_PADJ], skv[SK_PNB], skv[SK_PT0]};
    b = skv[SK_B]; kvh = skv[SK_KVH];
    merge_finalize<8>(p, x, red_m[0][w], Lw, b, kvh, skv[SK_WS], pm, pm.nbody + p.tail_nslot, &last_flag);
    jbase = skv[SK_JB] + nblk;
    if (jbase >= skv[SK_R1] - skv[SK_R0]) return false;
    if (tid == 0) sk_setup(p, sk_pre, sk_len, kvh, jbase, skv[SK_R0], skv[SK_R1], b, sks);  // next segment
    return true;
  } else {
    const int nparts = ns_b + p.tail_nslot;  // + the staging-window partials
    const int base = (b * p.Hkv + kvh) * p.nsplit;
    const PartMap pm{base, 1, 0, nparts, 0};
    merge_finalize(p, x, red_m[0][w], Lw, b, kvh, base + split, pm, nparts, &last_flag);
    return false;
  }
  };  // segment
  if constexpr (!SK) {
    segment();
  } else if (sk_two) {  // <= 2 segments (thread 0's sk_setup between the two: barrier at the top of segment())
    if (segment()) segment();
  } else {
    while (segment()) {
    }
  }
}

#include "attn_decode_v2g.cuh"  // generalized kernel: any GQA group, head dim 256

// ================================================================================================= paged writer
// One warp per (token, K/V head row): bf16 row -> FlagRot -> quantization identical to kv_selrot_quant (bit-exact
// codes and scales) -> the v2 page layout.  K rows are written whole; V^T nibbles with atomicAnd/atomicOr (tokens of
// the same page may be written concurrently); V scales with byte / half stores.
DEVI float div_rn(float y, float d, float inv) {  // correctly rounded y / d given inv = RN(1/d) (selrot_kernels.cu)
  const float q = __fmul_rn(y, inv);
  const float n = fmaf(q, d, -y);
  return fmaf(-n, inv, q);
}
DEVI uint32_t e2m1_code(float x) {  // identical to selrot_kernels.cu
  const float a = fminf(fabsf(x), 6.0f);
  const uint32_t E = max(__float_as_uint(a) & 0x7F800000u, 0x3F800000u);
  const uint32_t sb = __float_as_uint(__fadd_rn(a, __uint_as_float(E + 0x0B400000u)));
  const uint32_t code = sb - E + (E >> 22) - (0x0B400000u + 254u);
  return code | ((__float_as_uint(x) >> 28) & 8u);
}
template <int GS>  // max over the lanes of a group of GS elements (GS / 4 lanes)
DEVI float group_max(float a) {
#pragma unroll
  for (int o = 1; o < GS / 4; o <<= 1) a = fmaxf(a, __shfl_xor_sync(FULL, a, o));
  return a;
}
template <int GS>
DEVI float group_min(float a) {
#pragma unroll
  for (int o = 1; o < GS / 4; o <<= 1) a = fminf(a, __shfl_xor_sync(FULL, a, o));
  return a;
}
// INT4A scale word: the FP16 bit pattern nearest to RN_fp16(s) (integer distance, ties up) whose low 4 bits are z
// (blockflag.formats.scale_word)
template <int NB = 4>  // NB = 2 (INT2): the 2 low bits are z (blockflag.formats.scale_word(.., bits = 2))
DEVI uint32_t scale_word(float s, uint32_t z) {
  constexpr int M = (1 << NB) - 1;
  const int h = static_cast<int>(__half_as_ushort(__float2half_rn(s)));
  const int x = min(max(h - static_cast<int>(z) + (1 << (NB - 1)), 0) & ~M, 0x7C00 - (1 << NB));
  return static_cast<uint32_t>(x) + z;
}
// INT4A of a 32-group held by 8 lanes (4 elements each): blockflag int4a:32 with the range widened to include 0
// (blockflag int4az / int4ah, identical to int4a whenever min <= 0 <= max): s = max((max' - min') * fp32(1/15), 1e-12),
// z = rne(-min' / s), q = clamp(rne(v / s) + z, 0, 15) (IEEE divisions, as torch on CUDA); returns the scale word
// NB = 2: INT2 (blockflag int2a:32 widened to include 0 = int2az / int2ah: (max' - min') * fp32(1/3), codes 0..3)
template <int NB = 4>
DEVI uint32_t int4a_group(const float (&v)[4], uint32_t (&code)[4]) {
  constexpr float LV = static_cast<float>((1 << NB) - 1);
  float mn = fminf(fminf(v[0], v[1]), fminf(v[2], v[3])), mx = fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3]));
  mn = fminf(group_min<32>(mn), 0.f);
  mx = fmaxf(group_max<32>(mx), 0.f);
  const float s = fmaxf(__fmul_rn(__fsub_rn(mx, mn), 1.0f / LV), 1e-12f);
  const float z = rintf(__fdiv_rn(-mn, s));
#pragma unroll
  for (int i = 0; i < 4; ++i) code[i] = static_cast<uint32_t>(fminf(fmaxf(__fadd_rn(rintf(__fdiv_rn(v[i], s)), z), 0.f), LV));
  return scale_word<NB>(s, static_cast<uint32_t>(z));
}

template <int F, int P>
__global__ void __launch_bounds__(128) k_paged_write(const PagedWriteParams p) {
  using L = Lay<F, P>;
  const int lane = threadIdx.x & 31;
  const long long row = static_cast<long long>(blockIdx.x) * 4 + (threadIdx.x >> 5);
  const int H2 = 2 * p.Hkv;
  if (row >= static_cast<long long>(p.T) * H2) return;
  const int t = static_cast<int>(row / H2), hr = static_cast<int>(row % H2);
  const long long slot = p.slots[t];
  if (slot < 0) return;
  const long long page = slot / P;
  const int pos = static_cast<int>(slot % P);
  const bool isv = hr >= p.Hkv;
  const int head = isv ? hr - p.Hkv : hr;
  uint8_t* chunk = p.pool + (page * p.Hkv + head) * L::CHUNK;

  const __nv_bfloat16* src = static_cast<const __nv_bfloat16*>(p.kv) + t * p.kv_ts + hr * D + lane * 4;
  const uint2 raw = *reinterpret_cast<const uint2*>(src);
  float v[4] = {__uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xffff0000u),
                __uint_as_float(raw.y << 16), __uint_as_float(raw.y & 0xffff0000u)};
  if (p.kmu != nullptr && !isv) {  // static key centering (hd 256: half head hr = the same linear index of kmu)
    const float4 m = *reinterpret_cast<const float4*>(p.kmu + hr * D + lane * 4);
    v[0] = __fsub_rn(v[0], m.x); v[1] = __fsub_rn(v[1], m.y); v[2] = __fsub_rn(v[2], m.z); v[3] = __fsub_rn(v[3], m.w);
  }
  warp_selrot(v, p.mask + hr * (D / p.rb), p.rb, lane);
  float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  uint32_t code[4];  // sign-magnitude 4-bit codes (sign in bit 3)
  uint32_t sraw;     // this lane's group scale (raw bits)
  if constexpr (F == A2_NVFP4) {
    const float gs = __ldg(p.gscale);
    const float a = group_max<16>(am);
    const float sfv = __fmul_rn(gs, __fmul_rn(a, 1.0f / 6.0f));
    const __nv_fp8_storage_t sf = __nv_cvt_float_to_fp8(sfv, __NV_SATFINITE, __NV_E4M3);
    const float sff = __half2float(__half(__nv_cvt_fp8_to_halfraw(sf, __NV_E4M3)));
    const float bs = __fdiv_rn(sff, gs);
    const float inv = bs != 0.f ? __frcp_rn(bs) : 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) code[i] = e2m1_code(bs != 0.f ? div_rn(v[i], bs, inv) : 0.f);
    sraw = sf;
  } else if constexpr (F == A2_MXFP4) {
    const float a = group_max<32>(am);
    int e = -127;
    if (a > 0.f) { int ex; frexpf(a, &ex); e = ex - 1 - 2; }
    e = max(-127, min(127, e));
    const float mul = ldexpf(1.0f, -e);
#pragma unroll
    for (int i = 0; i < 4; ++i) code[i] = e2m1_code(__fmul_rn(v[i], mul));
    sraw = static_cast<uint32_t>(e + 127);
  } else if constexpr (F == A2_INT4A) {
    sraw = int4a_group(v, code);
  } else if constexpr (F == A2_INT2A) {
    sraw = int4a_group<2>(v, code);
  } else {
    const float a = group_max<32>(am);
    const __half sh = __float2half_rn(__fmul_rn(a, 1.0f / 7.0f));
    const float sf = __half2float(sh);
    const float inv = sf != 0.f ? __frcp_rn(sf) : 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float q = sf != 0.f ? rintf(div_rn(v[i], sf, inv)) : 0.f;
      q = fminf(fmaxf(q, -7.f), 7.f);
      const int qi = static_cast<int>(q);
      code[i] = static_cast<uint32_t>(qi < 0 ? (8 | -qi) : qi);
    }
    sraw = __half_as_ushort(sh);
  }
  // bit position of element (A or B, byte i) within a 32-bit word: returns (mask, bits)
  auto place = [&](uint32_t c, bool isB, int i, uint32_t& m, uint32_t& bits) {
    const uint32_t s = (c >> 3) & 1u, mag = c & 7u;
    if constexpr (F == A2_INT4A) {  // unsigned nibble: A low, B high nibble of byte i
      m = (isB ? 0xF0u : 0x0Fu) << (8 * i);
      bits = c << (8 * i + (isB ? 4 : 0));
    } else if constexpr (F == A2_INT4) {
      if (!isB) { m = 0x87u << (8 * i); bits = ((s << 7) | mag) << (8 * i); }
      else {
        const int j = (i + 3) & 3;
        m = (0x08u << (8 * i)) | (0x70u << (8 * j));
        bits = (s << (8 * i + 3)) | (mag << (8 * j + 4));
      }
    } else {
      if (!isB) { m = 0x9Cu << (8 * i); bits = ((s << 7) | (mag << 2)) << (8 * i); }
      else { m = 0x63u << (8 * i); bits = ((s << 5) | ((mag >> 2) << 6) | (mag & 3u)) << (8 * i); }
    }
  };
  if constexpr (F2<F>::B2) {  // INT2 layout (attn_decode_v2.cuh)
    if (!isv) {
      // K row word (tg, jp) at byte 8 tg + 4 jp: groups 2 jp (bits 4 h = 0) and 2 jp + 1 (h = 1), A at + 0 / B at + 2
      // of byte i; lane 8 j + tg builds its group's half word (A: own elements, B: lane + 4), lane 16 jp + tg the word
      uint32_t cb[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) cb[i] = __shfl_down_sync(FULL, code[i], 4);
      uint32_t hw = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) hw |= (code[i] << (8 * i)) | (cb[i] << (8 * i + 2));
      const uint32_t hw1 = __shfl_down_sync(FULL, hw, 8);
      const int tg = lane & 7, j = lane >> 3;
      if (tg < 4 && !(j & 1))
        *reinterpret_cast<uint32_t*>(chunk + L::K_OFF + krow_off<F>(pos) + 8 * tg + 4 * (j >> 1)) = hw | (hw1 << 4);
      if ((lane & 7) == 0) *reinterpret_cast<uint16_t*>(chunk + L::KS_OFF + pos * 8 + 2 * (lane / 8)) = static_cast<uint16_t>(sraw);
    } else {
      const int c8 = pos / 8, r8 = pos % 8, ii = r8 >> 1;
      const int isB = r8 & 1;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int dd = lane * 4 + i;
        const int wq = (dd % 64) / 16, gq = dd % 8, r = (dd % 16) / 8;
        uint32_t* wp = reinterpret_cast<uint32_t*>(chunk + L::V_OFF + ((wq * 8 + gq) * (P / 8) + c8) * 8 + 4 * (dd / 64));
        const int sh = 8 * ii + 4 * r + 2 * isB;
        atomicAnd(wp, ~(3u << sh));
        atomicOr(wp, code[i] << sh);
      }
      if ((lane & 7) == 0) {
        const int slot8 = isB ? 4 + ii : ii;
        *reinterpret_cast<uint16_t*>(chunk + L::VS_OFF + ((lane / 8) * P + 8 * c8 + slot8) * 2) = static_cast<uint16_t>(sraw);
      }
    }
    return;
  }
  if (!isv) {
    // K row: word (tg, j) at byte 16 tg + 4 j, built by lane 8 j + tg (A: own d 4l..4l+3, B: lane + 4)
    uint32_t cb[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) cb[i] = __shfl_down_sync(FULL, code[i], 4);
    const int tg = lane & 7, j = lane >> 3;
    if (tg < 4) {
      uint32_t wd = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        uint32_t m, bits;
        place(code[i], false, i, m, bits); wd |= bits;
        place(cb[i], true, i, m, bits); wd |= bits;
      }
      *reinterpret_cast<uint32_t*>(chunk + L::K_OFF + pos * 64 + 16 * tg + 4 * j) = wd;
    }
    // K scales (standard order)
    if constexpr (F == A2_NVFP4) {
      if ((lane & 3) == 0) chunk[L::KS_OFF + pos * 8 + lane / 4] = static_cast<uint8_t>(sraw);
    } else if constexpr (F == A2_MXFP4) {
      if ((lane & 7) == 0) chunk[L::KS_OFF + pos * 4 + lane / 8] = static_cast<uint8_t>(sraw);
    } else {  // INT4 / INT4A
      if ((lane & 7) == 0) *reinterpret_cast<uint16_t*>(chunk + L::KS_OFF + pos * 8 + 2 * (lane / 8)) = static_cast<uint16_t>(sraw);
    }
  } else {
    const int c8 = pos / 8, r8 = pos % 8, ii = r8 >> 1;
    const bool isB = r8 & 1;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dd = lane * 4 + i;
      const int wq = (dd % 64) / 16, gq = dd % 8, sub = 2 * (dd / 64) + ((dd % 16) / 8);
      uint32_t* wp = reinterpret_cast<uint32_t*>(chunk + L::V_OFF + (((wq * 8 + gq) * (P / 8) + c8) << 4) + 4 * sub);
      uint32_t m, bits;
      place(code[i], isB, ii, m, bits);
      atomicAnd(wp, ~m);
      atomicOr(wp, bits);
    }
    constexpr int GSZ = F == A2_NVFP4 ? 16 : 32;
    if ((lane & (GSZ / 4 - 1)) == 0) {
      const int g = lane * 4 / GSZ;
      const int slot8 = isB ? 4 + ii : ii;
      const int off = L::VS_OFF + (g * P + 8 * c8 + slot8) * F2<F>::SBE;
      if constexpr (F2<F>::I4) *reinterpret_cast<uint16_t*>(chunk + off) = static_cast<uint16_t>(sraw);
      else chunk[off] = static_cast<uint8_t>(sraw);
    }
  }
}

// ================================================================================================= launchers
int g_pdl = 0;  // launch the attention kernel with programmatic stream serialization (attn2_set_pdl)
template <int F, int P, int ST, int MODE>
int launch_k(const PagedParams* p, cudaStream_t st, int* occ) {
  using C = Cfg2<F, P, ST, MODE>;
  if (C::SMEM > 200 * 1024) return -2;
  auto k = k_attn2<F, P, ST, MODE>;
  static int per_sm = -1;
  if (per_sm < 0) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k, 128, C::SMEM);
  }
  if (occ) { *occ = per_sm; return 0; }
  static cudaStream_t side = nullptr;
  static cudaEvent_t e0, e1;
  if (p->tail != nullptr) {  // staging window on a side stream, concurrent with the body splits
    static bool tset = false;
    if (!tset) { cudaFuncSetAttribute(k_tail_v2, cudaFuncAttributeMaxDynamicSharedMemorySize, TAIL_SMEM_FLOATS * 4 + 2048); tset = true; }
    tail_fork(st, side, e0, e1);
    k_tail_v2<<<dim3(p->Hkv, p->tail_nslot, p->B), 128, TAIL_SMEM_FLOATS * 4 + 2048, side>>>(*p);
  }
  const dim3 grid = (MODE & 512) ? dim3(p->sk_grid, 1, 1) : dim3(p->Hkv, p->nsplit - p->tail_nslot, p->B);
  if (g_pdl) {  // programmatic dependent launch: overlaps the launch with the end of the preceding kernel
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(128, 1, 1);
    cfg.dynamicSmemBytes = C::SMEM;
    cfg.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, k, *p);
  } else {
    k<<<grid, 128, C::SMEM, st>>>(*p);
  }
  if (p->tail != nullptr) tail_join(st, side, e1);
  return 0;
}
template <int F, int P, int MODE>
int launch_s(int stages, const PagedParams* p, cudaStream_t st, int* occ) {
  switch (stages) {
    case 3: return launch_k<F, P, 3, MODE>(p, st, occ);
    case 4: return launch_k<F, P, 4, MODE>(p, st, occ);
  }
  return -3;
}
template <int F, int P>
int launch_v(int variant, int stages, const PagedParams* p, cudaStream_t st, int* occ) {
  // variant = MODE: bit 0 PV path (0 FP8 / 1 f16), bit 1 loader (0 bulk copy / 1 cp.async)
  if constexpr (F == A2_INT4A || F == A2_INT2A) {  // asymmetric INT4 / INT2: the default static variant only (264:
                                  // deferred PV wait; 0 / 8 are serialized by ptxas (C7515) with the zero-point accumulators)
    if (variant == 264) return launch_s<F, P, 264>(stages, p, st, occ);
    return -3;
  } else {
    if (variant == 1) return launch_s<F, P, 1>(stages, p, st, occ);
    if (variant == 3) return launch_s<F, P, 3>(stages, p, st, occ);
    if constexpr (F != A2_NVFP4) {
      if (variant == 0) return launch_s<F, P, 0>(stages, p, st, occ);
      if (variant == 2) return launch_s<F, P, 2>(stages, p, st, occ);
      if (variant == 4) return launch_s<F, P, 4>(stages, p, st, occ);  // FP8 PV, A (V^T) from shared memory
      if (variant == 8) return launch_s<F, P, 8>(stages, p, st, occ);  // variant 0 + conflict-free P' tiles
      if (variant == 12) return launch_s<F, P, 12>(stages, p, st, occ);  // variant 4 + conflict-free P' tiles
      if (variant == 28) return launch_s<F, P, 28>(stages, p, st, occ);  // variant 12 + split wgmma waits
      if (variant == 264) return launch_s<F, P, 264>(stages, p, st, occ);  // variant 8 + deferred PV wait / decode
      if (variant == 776) return launch_s<F, P, 776>(stages, p, st, occ);  // variant 264 + stream-K schedule
    }
    return -3;
  }
}
// generalized kernel (attn_decode_v2g.cuh): q tiles of GT heads, head dim 128 NH2
template <int F, int P, int ST, int GT, int NH2>
int launch_g(const PagedParams* p, cudaStream_t st, int* occ) {
  using C = CfgG<F, P, ST, GT, NH2>;
  if (C::SMEM > 227 * 1024) return -2;
  auto k = k_attn2g<F, P, ST, GT, NH2>;
  static int per_sm = -1;
  if (per_sm < 0) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k, 128, C::SMEM);
  }
  if (occ) { *occ = per_sm; return 0; }
  if (p->sk_grid > 0) return -6;  // split mode only
  const int g = p->gqa > 0 ? p->gqa : 4, nt = (g + GT - 1) / GT;
  static cudaStream_t side = nullptr;
  static cudaEvent_t e0, e1;
  if (p->tail != nullptr) {
    constexpr int TS = (GT * C::DD + tail_g_smem_floats<GT, NH2>()) * 4;
    static bool tset = false;
    if (!tset) { cudaFuncSetAttribute(k_tail_g<GT, NH2>, cudaFuncAttributeMaxDynamicSharedMemorySize, TS); tset = true; }
    tail_fork(st, side, e0, e1);
    k_tail_g<GT, NH2><<<dim3(p->Hkv * nt, p->tail_nslot, p->B), 128, TS, side>>>(*p);
  }
  const dim3 grid(p->Hkv * nt, p->nsplit - p->tail_nslot, p->B);
  if (g_pdl) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(128, 1, 1);
    cfg.dynamicSmemBytes = C::SMEM;
    cfg.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, k, *p);
  } else {
    k<<<grid, 128, C::SMEM, st>>>(*p);
  }
  if (p->tail != nullptr) tail_join(st, side, e1);
  return 0;
}
template <int F, int P, int GT, int NH2>
int launch_gs(int stages, const PagedParams* p, cudaStream_t st, int* occ) {
  switch (stages) {
    case 3: return launch_g<F, P, 3, GT, NH2>(p, st, occ);
    case 4: return launch_g<F, P, 4, GT, NH2>(p, st, occ);
  }
  return -3;
}
template <int F, int P>
int launch_gv(int variant, int stages, const PagedParams* p, cudaStream_t st, int* occ) {
  if ((variant & ~(ATTN2_GEN | ATTN2_GT8 | ATTN2_HD256)) != 264) return -3;
  const bool gt8 = variant & ATTN2_GT8, hd256 = variant & ATTN2_HD256;
  if (!gt8 && !hd256) return launch_gs<F, P, 4, 1>(stages, p, st, occ);
  if (gt8 && !hd256) return launch_gs<F, P, 8, 1>(stages, p, st, occ);
  if (!gt8 && hd256) return launch_gs<F, P, 4, 2>(stages, p, st, occ);
  return -3;  // (GT 8 x D 256: register budget, not instantiated)
}
template <int F>
int launch_p(int page, int variant, int stages, const PagedParams* p, cudaStream_t st, int* occ) {
  if (variant & ATTN2_GEN) {
    if constexpr (F2<F>::I4) {
      if (page == 16) return launch_gv<F, 16>(variant, stages, p, st, occ);
      if (page == 32) return launch_gv<F, 32>(variant, stages, p, st, occ);
      return -5;
    }
    return -3;
  }
  if constexpr (F2<F>::KO) {
    return -3;  // key-only codebook caches: generalized kernel only
  } else {
    if (p != nullptr && p->gqa > 0 && p->gqa != 4) return -7;  // k_attn2: 4 q heads per kv head only
    if (page == 16) return launch_v<F, 16>(variant, stages, p, st, occ);
    if (page == 32) return launch_v<F, 32>(variant, stages, p, st, occ);
    return -5;
  }
}
int dispatch(int fmt, int page, int variant, int stages, const PagedParams* p, cudaStream_t st, int* occ) {
#ifdef ATTN2_EXP_G  // compile experiment (ptxas reports): only the generalized INT4A kernel <page 16, 4 stages, GT, NH2>
  (void)fmt; (void)page; (void)variant; (void)stages;
  return launch_g<A2_INT4A, 16, 4, ATTN2_EXP_G / 10, ATTN2_EXP_G % 10>(p, st, occ);
#endif
  switch (fmt) {
    case A2_NVFP4: return launch_p<A2_NVFP4>(page, variant, stages, p, st, occ);
    case A2_MXFP4: return launch_p<A2_MXFP4>(page, variant, stages, p, st, occ);
    case A2_INT4: return launch_p<A2_INT4>(page, variant, stages, p, st, occ);
    case A2_INT4A: return launch_p<A2_INT4A>(page, variant, stages, p, st, occ);
    case A2_KONLY: return launch_p<A2_KONLY>(page, variant, stages, p, st, occ);
    case A2_KONLYA: return launch_p<A2_KONLYA>(page, variant, stages, p, st, occ);
    case A2_INT2A: return launch_p<A2_INT2A>(page, variant, stages, p, st, occ);
    case A2_KONLYA2: return launch_p<A2_KONLYA2>(page, variant, stages, p, st, occ);
  }
  return -4;
}
template <int F>
int write_p(const PagedWriteParams& p0, cudaStream_t st) {
  // head dim 256: the two 128-d halves of a head are written as two half heads (kv [T, 2 Hkv, 256] =
  // [T, 4 Hkv, 128], mask [2 Hkv, 256 / rb] = [4 Hkv, 128 / rb], pool [., Hkv, 2 chunk] = [., 2 Hkv, chunk])
  PagedWriteParams p = p0;
  if (p.hd == 256) p.Hkv *= 2;
  else if (p.hd != 0 && p.hd != 128) return -8;
  const long long rows = static_cast<long long>(p.T) * 2 * p.Hkv;
  const unsigned grid = static_cast<unsigned>((rows + 3) / 4);
  if (grid == 0) return 0;
  if (p.page == 16) k_paged_write<F, 16><<<grid, 128, 0, st>>>(p);
  else if (p.page == 32) k_paged_write<F, 32><<<grid, 128, 0, st>>>(p);
  else return -5;
  return 0;
}

}  // namespace

int attn2_launch(int fmt, int variant, int stages, const PagedParams& p, cudaStream_t st) {
  if (p.B <= 0) return 0;
  return dispatch(fmt, p.page, variant, stages, &p, st, nullptr);
}
void attn2_set_pdl(int on) { g_pdl = on; }
int attn2_occupancy(int fmt, int variant, int stages, int page) {
  int occ = 0;
  const int rc = dispatch(fmt, page, variant, stages, nullptr, nullptr, &occ);
  return rc == 0 ? occ : 0;
}
int attn2_chunk_bytes(int fmt, int page) {  // (INT4A: same chunk as INT4, the zero points live in the scale words)
  const int sb = fmt == A2_MXFP4 ? 4 : 8;
  const int cby = (fmt == A2_INT2A || fmt == A2_KONLYA2) ? 32 : 64;  // INT2: 80 P (2.5 bits per element)
  return page * (2 * cby + 2 * sb);
}
int paged_write_launch(int fmt, const PagedWriteParams& p, cudaStream_t st) {
  switch (fmt) {
    case A2_NVFP4: return write_p<A2_NVFP4>(p, st);
    case A2_MXFP4: return write_p<A2_MXFP4>(p, st);
    case A2_INT4: return write_p<A2_INT4>(p, st);
    case A2_INT4A: return write_p<A2_INT4A>(p, st);
    case A2_INT2A: return write_p<A2_INT2A>(p, st);
  }
  return -4;
}
