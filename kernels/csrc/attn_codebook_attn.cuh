#include <utility>
// Decode attention over the codebook INT4 paged cache (included by attn_codebook.cu).  Based on the v2 INT4 kernel
// (attn_decode_v2.cu, variant 0: wgmma e4m3, q hi/lo, FP8 PV with the V scale folded into P', bulk-copy pipeline).
//
// Inverse transforms (design):
//  * keys, on the query side: every q head is transformed once per CTA into the NVQ variants T_v^T q (reduced: I, H32,
//    H128; full: I, H32, H32 S, the same three on q[PERM], H128, H128 S) -> the QK wgmma has N = 8 NVQ columns per
//    32-d group (variant x head x hi/lo); the score of a token sums, per group, the column of the variant its side
//    info selects (branch-free select), times its K scale.  Exact: q . (T^T c) = (T q) . c.
//  * values, after P V: per (group, class) accumulators, class = the (head mode, vector choice) of a token's 32-vector
//    (reduced: I, H32, F; full: 8 classes).  The class split is done on the wgmma M side: one 64-row tile holds the 32
//    d of a group for two classes (warps 0-1 / 2-3), its A fragment = the V^T codes masked (byte-wise) to the tokens of
//    the warp's class, B = that group's 8 P' columns -> no extra P' columns and no parity waste; the F class of the
//    reduced codebook is token-level and uses the v2 two-group tiles masked by the head bit.  Each class accumulator
//    is transformed back once per CTA (FWHT on 4 x 128 values per class) and the classes are summed.
//  * reduced: everything of a block fits in registers and the v2 software pipeline is kept (QK(n+1) overlaps the
//    softmax of n, PV(n) is waited for at the top of n+1); full: QK in two rounds of two groups (N = 64) and PV in
//    four rounds of four tiles, each waited for (register budget), the ring stage is released after the barrier.

namespace {

constexpr int G = 4;      // q heads per kv head
constexpr int NWARP = 4;
constexpr int BLK = 64;
constexpr float NEG_INF = -__builtin_huge_valf();
constexpr float TAU = 3.0f;
constexpr int SH = 4;
constexpr int EREF_NONE = -100000;
constexpr int MAX_CHUNK = 8192;
constexpr float LO_SCALE = 16.f, LO_INV = 1.f / 16.f;
constexpr float FV = 512.f;  // INT4 e4m3 byte value c * 2^-9

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
DEVI uint32_t lds16(uint32_t a) {
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
DEVI void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
DEVI void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
DEVI void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
DEVI void wg_wait_all() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
DEVI void wg_wait_1() { asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory"); }
// P' tiles [ks][m] (N 16 x K 32), bank-conflict-free layout: K-core matrices 144 B apart (LBO),
// N-cores 288 B (SBO), tiles 576 B
constexpr int PTILE = 576, PLBO = 144, PSBO = 288;
DEVI uint64_t sdesc_p(uint32_t saddr) {
  return static_cast<uint64_t>((saddr & 0x3FFFF) >> 4) | (static_cast<uint64_t>(PLBO >> 4) << 16) |
         (static_cast<uint64_t>(PSBO >> 4) << 32);
}
DEVI uint64_t sdesc(uint32_t saddr) {
  return static_cast<uint64_t>((saddr & 0x3FFFF) >> 4) | (static_cast<uint64_t>(128 >> 4) << 16) |
         (static_cast<uint64_t>(256 >> 4) << 32);
}
// CB_KONLYA: m64n8k32 with both operands in shared memory (Z = P'^T z: A = the P' tiles, B = the zero-point tile)
template <int ACC>
DEVI void wg_n8_ss(float (&d)[4], uint64_t adesc, uint64_t bdesc) {
  if constexpr (ACC)
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, %4, %5, 1, 1, 1;\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "l"(adesc), "l"(bdesc));
  else
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, %4, %5, 0, 1, 1;\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "l"(adesc), "l"(bdesc));
}
DEVI uint64_t sdesc_zb(uint32_t saddr) {  // zero-point tile: K halves 128 B apart (one 8-row N group)
  return static_cast<uint64_t>((saddr & 0x3FFFF) >> 4) | (static_cast<uint64_t>(128 >> 4) << 16) |
         (static_cast<uint64_t>(256 >> 4) << 32);
}
template <int ACC>
DEVI void wg_n8(float (&d)[4], const uint32_t (&a)[4], uint64_t desc) {
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
DEVI void wg_n16(float (&d)[8], const uint32_t (&a)[4], uint64_t desc) {
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
DEVI void wg_n24(float (&d)[12], const uint32_t (&a)[4], uint64_t desc) {  // fresh accumulator
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n24k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11}, "
      "{%12,%13,%14,%15}, %16, 0, 1, 1;\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]),
        "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
DEVI void wg_n64(float (&d)[32], const uint32_t (&a)[4], uint64_t desc) {  // fresh accumulator
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,"
      "%30,%31}, {%32,%33,%34,%35}, %36, 0, 1, 1;\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]),
        "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]),
        "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]),
        "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
DEVI void wg_n120(float (&d)[60], const uint32_t (&a)[4], uint64_t desc) {  // fresh accumulator
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n120k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59}, {%60,%61,%62,%63}, %64, 0, 1, 1;\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]), "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]), "=f"(d[40]), "=f"(d[41]), "=f"(d[42]), "=f"(d[43]), "=f"(d[44]), "=f"(d[45]), "=f"(d[46]), "=f"(d[47]), "=f"(d[48]), "=f"(d[49]), "=f"(d[50]), "=f"(d[51]), "=f"(d[52]), "=f"(d[53]), "=f"(d[54]), "=f"(d[55]), "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
DEVI void wg_n80(float (&d)[40], const uint32_t (&a)[4], uint64_t desc) {  // fresh accumulator
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n80k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39}, {%40,%41,%42,%43}, %44, 0, 1, 1;\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]), "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
DEVI void wg_n144(float (&d)[72], const uint32_t (&a)[4], uint64_t desc) {  // fresh accumulator
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n144k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71}, {%72,%73,%74,%75}, %76, 0, 1, 1;\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]), "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]), "=f"(d[40]), "=f"(d[41]), "=f"(d[42]), "=f"(d[43]), "=f"(d[44]), "=f"(d[45]), "=f"(d[46]), "=f"(d[47]), "=f"(d[48]), "=f"(d[49]), "=f"(d[50]), "=f"(d[51]), "=f"(d[52]), "=f"(d[53]), "=f"(d[54]), "=f"(d[55]), "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]), "=f"(d[60]), "=f"(d[61]), "=f"(d[62]), "=f"(d[63]), "=f"(d[64]), "=f"(d[65]), "=f"(d[66]), "=f"(d[67]), "=f"(d[68]), "=f"(d[69]), "=f"(d[70]), "=f"(d[71])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
DEVI void wg_n56(float (&d)[28], const uint32_t (&a)[4], uint64_t desc) {  // fresh accumulator
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n56k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
      "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27}, {%28,%29,%30,%31}, %32, 0, 1, 1;\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]),
        "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]),
        "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]),
        "=f"(d[25]), "=f"(d[26]), "=f"(d[27])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int ACC>
DEVI void wg_n32a(float (&d)[16], const uint32_t (&a)[4], uint64_t desc) {
  if constexpr (ACC == 0)
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, 0, 1, 1;\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  else 
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, 1, 1, 1;\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int ACC>
DEVI void wg_n64a(float (&d)[32], const uint32_t (&a)[4], uint64_t desc) {
  if constexpr (ACC == 0)
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, 0, 1, 1;\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  else 
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, 1, 1, 1;\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int N, int ACC>
DEVI void wg_kf(float (&d)[N / 2], const uint32_t (&a)[4], uint64_t desc) {
  if constexpr (N == 32) wg_n32a<ACC>(d, a, desc);
  else wg_n64a<ACC>(d, a, desc);
}
template <int N>
DEVI void wg_qk(float (&d)[N / 2], const uint32_t (&a)[4], uint64_t desc) {
  if constexpr (N == 56) wg_n56(d, a, desc);
  else if constexpr (N == 80) wg_n80(d, a, desc);
  else if constexpr (N == 120) wg_n120(d, a, desc);
  else wg_n144(d, a, desc);
}
template <int N>
DEVI void wg_wait0(float (&d)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(d[i])::"memory");
}
template <int N>
DEVI void pin_regs(uint32_t (&r)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+r"(r[i]));
}
DEVI uint32_t opaque(uint32_t x) { asm volatile("mov.b32 %0, %0;" : "+r"(x)); return x; }
DEVI uint64_t opaque64(uint64_t x) { asm volatile("mov.b64 %0, %0;" : "+l"(x)); return x; }
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
DEVI uint32_t f32x2_e4m3x2(float lo, float hi) {
  uint32_t r;
  asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, %1, %2; cvt.u32.u16 %0, t; }\n" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
DEVI uint32_t f16x2_e4m3x2(uint32_t h) {
  uint32_t r;
  asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, %1; cvt.u32.u16 %0, t; }\n" : "=r"(r) : "r"(h));
  return r;
}
DEVI uint32_t hsub2u(uint32_t a, uint32_t b) {
  __half2 r = __hsub2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
DEVI uint32_t hmax2u(uint32_t a, uint32_t b) {
  __half2 r = __hmax2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
DEVI uint32_t hmul2u(uint32_t a, uint32_t b) {
  __half2 r = __hmul2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
DEVI float2 h2f2(uint32_t h) { return __half22float2(*reinterpret_cast<const __half2*>(&h)); }
DEVI uint32_t lop_and3(uint32_t a, uint32_t b, uint32_t c) {  // a & b & c (one LOP3)
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0x80;\n" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}
// INT4 words -> e4m3 bytes of the 4 A (even tokens / first d) and 4 B elements, masked byte-wise
DEVI void cvt_ab_m(uint32_t w, uint32_t ma, uint32_t mb, uint32_t& a, uint32_t& b) {
  a = lop_and3(w, ma, 0x87878787u);
  b = lop_and3(__funnelshift_l(w, w, 4), mb, 0x87878787u);
}
DEVI uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {  // PTX prmt (selector msb: replicate the byte's sign)
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}
DEVI uint32_t sign_bytes(uint32_t x) { return prmt(x, 0u, 0xBA98u); }  // byte i = bit 8i+7 replicated

// compile-time loop (the round index must stay a constant: the class accumulators O[k] would otherwise be
// indexed at run time and live in local memory when the compiler declines to unroll a large body)
template <typename Fn, int... Ks>
DEVI void static_for_impl(Fn&& f, std::integer_sequence<int, Ks...>) { (f(std::integral_constant<int, Ks>{}), ...); }
template <int N, typename Fn>
DEVI void static_for(Fn&& f) { static_for_impl(f, std::make_integer_sequence<int, N>{}); }

// q-side transforms, 4 elements per lane (lane l: d = 4l .. 4l+3), fp32
template <int RB>
DEVI void fwht_q(float (&x)[4], int lane) {
#pragma unroll
  for (int h = 1; h < RB; h <<= 1) {
    if (h < 4) {
#pragma unroll
      for (int i = 0; i < 4; ++i)
        if (!(i & h)) {
          const float a = x[i], c = x[i + h];
          x[i] = a + c;
          x[i + h] = a - c;
        }
    } else {
      const int lm = h >> 2;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float o = __shfl_xor_sync(FULL, x[i], lm);
        x[i] = (lane & lm) ? (o - x[i]) : (x[i] + o);
      }
    }
  }
  const float c = RB == 32 ? C32 : C128;
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] *= c;
}
DEVI void sflip4(float (&x)[4], const unsigned (&sg)[4], int lane) {  // x_d *= S_d, d = 4 lane + i
  const int wi = lane >> 3;  // sign word of this lane (select chain: no local-memory indexing)
  const unsigned word = wi == 0 ? sg[0] : wi == 1 ? sg[1] : wi == 2 ? sg[2] : sg[3];
  const unsigned bits = word >> (4 * (lane & 7));
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] = ((bits >> i) & 1u) ? -x[i] : x[i];
}

// ------------------------------------------------------------------------------------------------- layout / config
template <int CB, int P> struct LayA {
  static constexpr int CBY = cb_b2(CB) ? 32 : 64;   // code bytes per (token, row) of K and V (INT2: attn_decode_v2.cuh)
  static constexpr int K_OFF = 0, V_OFF = P * CBY, KS_OFF = 2 * P * CBY, VS_OFF = 2 * P * CBY + 8 * P;
  static constexpr int SWK_OFF = 144 * P, SWV_OFF = 146 * P, FK_OFF = 144 * P, FV_OFF = 144 * P + 4;
  static constexpr int CHUNK = cb_b2(CB) ? 80 * P : CB == CB_REDUCED ? 144 * P + 16 : cb_k2(CB) ? 144 * P : cb_kfam(CB) ? 146 * P : 148 * P;
  static constexpr int KTB = cb_b2(CB) ? 8 : 16, VE = cb_b2(CB) ? 8 : 16;  // K bytes of a row per thread, V^T entry bytes
  static constexpr uint32_t ZM2 = cb_b2(CB) ? 0x00030003u : 0x000F000Fu, ZM4 = cb_b2(CB) ? 0x03030303u : 0x0F0F0F0Fu;
  static constexpr int NPB = BLK / P;
  // shared-memory page stride of the ring (k_cbk): page offsets = 64 mod 128 B, so the V^T fragment loads of the two
  // pages of a 32-token half (lanes tig 0-1 / 2-3, P = 16) fall into different banks (the cache chunk is unchanged)
  static constexpr int PSTR = CHUNK + (192 - CHUNK % 128) % 128;
  static constexpr int STAGE = NPB * PSTR;
};
#ifndef CB_OSM_ROUNDS
#define CB_OSM_ROUNDS 8
#endif
template <int CB, int P, int ST> struct CfgA {
  using L = LayA<CB, P>;
  // key-only family: hi and lo tiles of NCH = ceil(V / 2) chunks of 8 columns each (variants 2c, 2c+1 x 4 heads)
  static constexpr int NVQ = cb_kfam(CB) ? 2 * ((CbSpec<CB>::V + 1) / 2) : cb_family(CB) ? CbSpec<CB>::V : CB == CB_FULL ? 8 : 3;
  static constexpr int NQ = 8 * NVQ;                           // QK wgmma N per group
  static constexpr int QT_BYTES = 4 * NQ * 32;
  static constexpr int PT_BYTES = 4 * PTILE;
  // V class one-hot table per block (double-buffered): full 1 byte, final 2 bytes (16 classes) per (token, group)
  static constexpr int TBL_BYTES = cb_family(CB) ? 256 * ((NVQ + 7) / 8) : CB == CB_FULL ? 256 : 0;
  static constexpr int RING_OFF = 0;
  static constexpr int QT_OFF = ST * L::STAGE;
  static constexpr int PT_OFF = QT_OFF + QT_BYTES;
  static constexpr int TBL_OFF = PT_OFF + 2 * PT_BYTES;
  static constexpr int PIDX_OFF = TBL_OFF + 2 * TBL_BYTES;
  // family codebooks with more than CB_OSM_ROUNDS PV rounds keep their class accumulators in shared memory
  static constexpr int NRND = cb_family(CB) ? (NVQ + 1) / 2 : 0;
  static constexpr int OSM = NRND > CB_OSM_ROUNDS;
  static constexpr int OSM_OFF = PIDX_OFF + (MAX_CHUNK / P) * 4;
  // CB_KONLYA: value zero points of a block as the B operand of Z = P'^T z ([buf][ks][k half][8 rows][16 B])
  static constexpr int ZB_OFF = OSM_OFF + (OSM ? NRND * 8 * 128 * 4 : 0);
  static constexpr int ZB_BYTES = cb_asym(CB) ? 2 * 512 : 0;
  static constexpr int SMEM = ZB_OFF + ZB_BYTES;
  // finalize buffers in the ring: class accumulators [NCLS][G][D] | out [G][D] | merge weights [512][G] (full: + the
  // P-class scatter buffer at out + 2 G D, dead before the merge)
  static constexpr int NCLS = CB == CB_FULL ? 8 : cb_family(CB) ? 2 : 3;  // family: two classes at a time
  static_assert(cb_b2(CB) ? ST * L::STAGE >= G * D * 4  // (k_cbk: out [G][D] only)
                          : ST * L::STAGE >= (NCLS + 1) * G * D * 4 + 512 * G * 4 && ST * L::STAGE >= (NCLS + 3) * G * D * 4,
                "finalize buffers must fit in the ring");
};

// ------------------------------------------------------------------------------------------------- shared pieces
// q variants of head w (warp w) -> Q^T tiles (column 8 v + 2 w hi, + 1 lo); returns sfac = FV / kq of the head
// qsum (CB_KONLYA): [variant][head][group] = 2^15 * sum over the group of the QK operand (hi + lo, or hi + lo / 16 for
// the scaled lo of H128), i.e. 2^-9 (code value c 2^-9) * 2^24 (zero point read as the FP16 subnormal z 2^-24)
template <int CB, int NQ>
DEVI void q_prologue(const CbAttnParams& p, int b, int kvh, int w, int lane, uint32_t qt, float* qstage, float* sfac_s,
                     float* qsum = nullptr) {
  constexpr int NVQ = cb_kfam(CB) ? CbSpec<CB>::V : NQ / 8;  // variants computed
  const __nv_bfloat16* qp = static_cast<const __nv_bfloat16*>(p.q) + b * p.q_bs + (kvh * G + w) * D + lane * 4;
  const uint2 raw = *reinterpret_cast<const uint2*>(qp);
  float x[4] = {__uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xffff0000u),
                __uint_as_float(raw.y << 16), __uint_as_float(raw.y & 0xffff0000u)};
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] *= p.qscale;
  unsigned sg[4] = {p.sg[0][0], p.sg[0][1], p.sg[0][2], p.sg[0][3]};
  float v[NVQ][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) v[0][i] = x[i];
  if constexpr (cb_family(CB) || cb_kfam(CB)) {  // I, H32 (S_{l-2}) for l = 1..NLH, H128 (S_{m-2}) for m = 1..NHF
    using SP = CbSpec<CB>;
#pragma unroll
    for (int vv = 1; vv < NVQ; ++vv) {
#pragma unroll
      for (int i = 0; i < 4; ++i) v[vv][i] = x[i];
      const int si = vv <= SP::NLH ? vv - 2 : vv - SP::NLH - 2;  // sign table entry (< 0: none)
      if (si >= 0) sflip4(v[vv], p.sg[si >= 0 ? si : 0], lane);
      if (vv <= SP::NLH) fwht_q<32>(v[vv], lane);
      else fwht_q<128>(v[vv], lane);
    }
  } else if constexpr (CB == CB_REDUCED || cb_k2(CB)) {  // I, H32, H128
#pragma unroll
    for (int i = 0; i < 4; ++i) v[1][i] = v[2][i] = x[i];
    fwht_q<32>(v[1], lane);
    fwht_q<128>(v[2], lane);
  } else {  // I, H32, H32 S, P-I, P-H32, P-H32 S, H128, H128 S
    float xp[4];
    {  // x[PERM]: z_{4l+i} = x_{permi(4l+i)} via shared memory
      float* qs = qstage + w * D;
#pragma unroll
      for (int i = 0; i < 4; ++i) qs[4 * lane + i] = x[i];
      __syncwarp();
#pragma unroll
      for (int i = 0; i < 4; ++i) xp[i] = qs[permi(4 * lane + i)];
      __syncwarp();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      v[1][i] = x[i];
      v[2][i] = x[i];
      v[3][i] = xp[i];
      v[4][i] = xp[i];
      v[5][i] = xp[i];
      v[6][i] = x[i];
      v[7][i] = x[i];
    }
    sflip4(v[2], sg, lane);
    sflip4(v[5], sg, lane);
    sflip4(v[7], sg, lane);
    fwht_q<32>(v[1], lane);
    fwht_q<32>(v[2], lane);
    fwht_q<32>(v[4], lane);
    fwht_q<32>(v[5], lane);
    fwht_q<128>(v[6], lane);
    fwht_q<128>(v[7], lane);
  }
  float am = 0.f;
#pragma unroll
  for (int k = 0; k < NVQ; ++k)
#pragma unroll
    for (int i = 0; i < 4; ++i) am = fmaxf(am, fabsf(v[k][i]));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(FULL, am, o));
  const int ea = am > 0.f ? static_cast<int>((__float_as_uint(am) >> 23) & 0xFF) - 127 : 0;
  const float kq = __uint_as_float(static_cast<uint32_t>(127 + 6 - ea) << 23);
  const int d0 = lane * 4, j = d0 / 32, k = d0 % 32;
#pragma unroll
  for (int vv = 0; vv < NVQ; ++vv) {
    // konly (KLO): the I / H32 lo parts are unscaled and chained onto the hi accumulators (see k_cbk)
    const bool klo = (cb_k2(CB) && vv < 2) || cb_kfam(CB);
    uint32_t hi = 0, lo = 0;
#pragma unroll
    for (int i = 0; i < 4; i += 2) {
      const float a0 = v[vv][i] * kq, a1 = v[vv][i + 1] * kq;
      const uint32_t hb = f32x2_e4m3x2(a0, a1);
      const float2 hf = h2f2(e4m3x2_f16x2_lo(hb));
      const float ls = klo ? 1.f : LO_SCALE;
      const uint32_t lb = f32x2_e4m3x2((a0 - hf.x) * ls, (a1 - hf.y) * ls);
      hi |= hb << (8 * i);
      lo |= lb << (8 * i);
    }
    // column of the hi part (lo: next column; konly: I / H32 hi at 2 w + vv, their lo at 16 + 2 w + vv, H128 at 8 + 2 w)
    // (key-only family: hi tile column 8 (v >> 1) + 2 w + (v & 1), the unscaled lo at the same column of the lo tile)
    const int colh = cb_kfam(CB) ? 8 * (vv >> 1) + 2 * w + (vv & 1)
                   : cb_k2(CB) ? (vv < 2 ? 2 * w + vv : 8 + 2 * w) : 8 * vv + 2 * w;
    const int coll = cb_k2(CB) && vv < 2 ? 16 + 2 * w + vv : colh + 1;
    const uint32_t a = qt + j * (NQ * 32) + (k / 16) * 128 + (k % 16);
    sts32(a + (colh / 8) * 256 + (colh % 8) * 16, hi);
    if constexpr (cb_kfam(CB)) sts32(a + NQ * 16 + (colh / 8) * 256 + (colh % 8) * 16, lo);
    else sts32(a + (coll / 8) * 256 + (coll % 8) * 16, lo);
    if constexpr (cb_asym(CB)) {
      const float2 h01 = h2f2(e4m3x2_f16x2_lo(hi)), h23 = h2f2(e4m3x2_f16x2_lo(hi >> 16));
      const float2 l01 = h2f2(e4m3x2_f16x2_lo(lo)), l23 = h2f2(e4m3x2_f16x2_lo(lo >> 16));
      float sq = ((h01.x + h01.y) + (h23.x + h23.y)) + ((l01.x + l01.y) + (l23.x + l23.y)) * (klo ? 1.f : LO_INV);
#pragma unroll
      for (int o = 1; o < 8; o <<= 1) sq += __shfl_xor_sync(FULL, sq, o);
      if ((lane & 7) == 0) qsum[(vv * G + w) * 4 + j] = sq * 32768.f;
    }
  }
  if (lane == 0) sfac_s[w] = FV / kq;
}

// per-thread constants of the P' (V scale) path, shared by both kernels
struct PvConst {
  uint32_t vso, pwo;
};

#include "attn_merge.cuh"  // PartMap, merge_warp (shared with attn_decode_v2.cu)
// split merge + bf16 store; outf: [G][D] final unnormalized per-head output (smem), Lc / Mc of this CTA.  As the static
// v2 kernel: warp h = q head h (d = 4 lane .. +3), one acq_rel ticket atomic, merge_warp
DEVI void merge_store(const CbAttnParams& p, float* outf, float (*red_l)[G], int b, int kvh, int split, int ns_b,
                      const float (&Mc)[G], const float (&Lc)[G], int* last_flag) {
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  (void)red_l;
  float x[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] = outf[w * D + lane * 4 + i];
  const float Mw = Mc[w], Lw = Lc[w];
  __syncthreads();
  if (ns_b == 1) {
    const float il = Lw > 0.f ? 1.f / Lw : 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] *= il;
  } else {
    const int base = (b * p.Hkv + kvh) * p.nsplit;
    reinterpret_cast<float4*>(p.opart)[static_cast<size_t>(base + split) * (G * D / 4) + w * (D / 4) + lane] =
        make_float4(x[0], x[1], x[2], x[3]);
    if (lane == 0) reinterpret_cast<float2*>(p.mlpart)[(base + split) * G + w] = make_float2(Mw, Lw);
    __syncthreads();
    if (tid == 0) {
      int old;
      asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(p.counter + b * p.Hkv + kvh) : "memory");
      *last_flag = old == ns_b - 1;
    }
    __syncthreads();
    if (!*last_flag) return;
    merge_warp<16, true>(p, PartMap{base, 1, 0, ns_b, 0}, ns_b, x);
    if (tid == 0) p.counter[b * p.Hkv + kvh] = 0;
  }
  __nv_bfloat162 o01 = __floats2bfloat162_rn(x[0], x[1]), o23 = __floats2bfloat162_rn(x[2], x[3]);
  uint2 ov;
  ov.x = *reinterpret_cast<uint32_t*>(&o01);
  ov.y = *reinterpret_cast<uint32_t*>(&o23);
  *reinterpret_cast<uint2*>(static_cast<__nv_bfloat16*>(p.out) + b * p.o_bs + (kvh * G + w) * D + lane * 4) = ov;
}

// BF16 staging window (attn_tail.cuh): grid (Hkv, tail_nslot, B), CTA j = window tokens [32 j, 32 j + 32); partial
// in the original basis (as the codebook outputs) as split slot ns_b + j, then the merge ticket
__global__ void __launch_bounds__(128) k_tail_cb(const CbAttnParams p) {
  extern __shared__ __align__(16) uint8_t smt[];
  __shared__ float tml[2 * G];
  __shared__ float red_l[NWARP][G];
  __shared__ int last_flag;
  const int kvh = blockIdx.x, j = 0, b = blockIdx.z;
  const int tl = min(p.tail_len[b], TAIL_MAX);
  const int len = p.seq_lens[b];
  const int ns_b = len > 0 ? max(1, (len + p.chunk - 1) / p.chunk) : 0;
  float* o = reinterpret_cast<float*>(smt);
  const __nv_bfloat16* qh = static_cast<const __nv_bfloat16*>(p.q) + b * p.q_bs + kvh * G * D;
  const __nv_bfloat16* tb = static_cast<const __nv_bfloat16*>(p.tail) + (p.tail_idx ? p.tail_idx[b] : b) * p.tail_bs;
  tail_window(qh, p.qscale, tb + kvh * D, tb + (p.Hkv + kvh) * D, p.tail_ts, tl, o + G * D, o, tml);
  float Mc[G], Lc[G];
#pragma unroll
  for (int h = 0; h < G; ++h) { Mc[h] = tml[2 * h]; Lc[h] = tml[2 * h + 1]; }
  merge_store(p, o, red_l, b, kvh, ns_b + j, ns_b + p.tail_nslot, Mc, Lc, &last_flag);
}

// P' (FP8 PV operand) of block rows tau0 / tau1 for head tig (identical to v2 variant 0, INT4): writes hi/lo bytes of
// every group to the P' buffer; returns pflag (some P' too large: slow path)
DEVI bool make_pprime(float p0, float p1, const uint32_t (&svraw)[4], const uint32_t (&inv2)[4], uint32_t vmask2,
                      uint32_t ptb_pwo) {
  __half2 ph = __floats2half2_rn(p0, p1);
  const uint32_t p2 = *reinterpret_cast<uint32_t*>(&ph);
  uint32_t pm = 0u;
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    const uint32_t s2 = hmul2u(svraw[g], inv2[g]) & vmask2;
    const uint32_t P2 = hmul2u(p2, s2);
    pm = hmax2u(pm, P2);
    // hi = RN_e4m3(P'), lo = RN_e4m3(16 (P' - hi)) converted and stored in 16-bit registers (one asm block)
    asm volatile(
        "{\n .reg .b16 h, l;\n .reg .b32 hf, d;\n"
        " cvt.rn.satfinite.e4m3x2.f16x2 h, %0;\n"
        " cvt.rn.f16x2.e4m3x2 hf, h;\n"
        " sub.rn.f16x2 d, %0, hf;\n"
        " mul.rn.f16x2 d, d, %2;\n"
        " cvt.rn.satfinite.e4m3x2.f16x2 l, d;\n"
        " st.shared.u16 [%1], h;\n"
        " st.shared.u16 [%1 + 16], l;\n}\n" ::"r"(P2), "r"(ptb_pwo + (g >> 1) * PTILE + (g & 1) * PSBO), "r"(0x4C004C00u)
        : "memory");
  }
  const float2 pmf = h2f2(pm);
  return !(fmaxf(pmf.x, pmf.y) <= 128.f);
}

// CB_KONLYA: make_pprime + the block's value zero-point sums zb[g] = P'(tau0) z(tau0) + P'(tau1) z(tau1) (z = bits 3..0
// of the V scale words, as fp16 integers (1024 + z) - 1024; f16 products and pair sum, P' z <= 128 * 15)
DEVI uint32_t lop_or_and(uint32_t a, uint32_t m, uint32_t o) {  // (a & m) | o in one LOP3
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(r) : "r"(a), "r"(m), "r"(o));
  return r;
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
template <uint32_t ZM2 = 0x000F000Fu>
DEVI bool make_pprime_z(float p0, float p1, const uint32_t (&svraw)[4], const uint32_t (&inv2)[4], uint32_t vmask2,
                        uint32_t ptb_pwo, float (&zb)[4]) {
  __half2 ph = __floats2half2_rn(p0, p1);
  const uint32_t p2 = *reinterpret_cast<uint32_t*>(&ph);
  uint32_t pm = 0u;
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    const uint32_t s2 = hmul2u(svraw[g], inv2[g]) & vmask2;
    const uint32_t P2 = hmul2u(p2, s2);
    pm = hmax2u(pm, P2);
    const uint32_t pt = pprime_store_t(P2, ptb_pwo + (g >> 1) * PTILE + (g & 1) * PSBO);
#if !defined(CB_NO_ZP) && !defined(CB_NO_VZP)  // negative control (wrong results): value zero points ignored
    const uint32_t zh = hsub2u(lop_or_and(svraw[g], ZM2, 0x64006400u), 0x64006400u);
    const uint32_t pzu = hmul2u(pt, zh);
    const __half2 pz = *reinterpret_cast<const __half2*>(&pzu);
    zb[g] = __half2float(__hadd(__low2half(pz), __high2half(pz)));
#else
    (void)pt;
#endif
  }
  const float2 pmf = h2f2(pm);
  return !(fmaxf(pmf.x, pmf.y) <= 128.f);
}

// ================================================================================================= reduced kernel
// QK N = 24 (I, H32, H128); PV: (I, H32) M-side tiles per group (N = 8) + F two-group tiles.
// SYNC = 0: v2 software pipeline (QK(n+1) issued before the softmax of n, PV(n) waited for at the top of n+1; all
// fragments of a block live at once, ~250 registers, 2 CTAs / SM); SYNC = 1: QK and PV of a block each waited for
// right after issue (fewer live registers -> more CTAs per SM, the other CTAs hide the wgmma latency).
template <int P, int ST, int SYNC>
__global__ void __launch_bounds__(128, 1) k_cbr(const CbAttnParams p) {
  using C = CfgA<CB_REDUCED, P, ST>;
  using L = LayA<CB_REDUCED, P>;
  constexpr int NQ = C::NQ, CHUNK = L::CHUNK, NPB = L::NPB;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ __align__(8) uint64_t mbar[ST];
  __shared__ float red_m[NWARP][G], red_l[NWARP][G], sfac_s[G];
  __shared__ int red_e[NWARP][4];
  __shared__ int last_flag;

  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, gid = lane >> 2, tig = lane & 3;
  const int kvh = blockIdx.x, split = blockIdx.y, b = blockIdx.z;
  const int len = p.seq_lens[b];
  const int c0 = split * p.chunk, c1 = min(len, c0 + p.chunk);
  const int ns_b = max(1, (len + p.chunk - 1) / p.chunk);
  if (split >= ns_b || len <= 0) return;  // (len 0: the staging-window kernel writes the output)
  const uint32_t sbase = smem_u32(smem);
  const uint32_t qt = sbase + C::QT_OFF, pt = sbase + C::PT_OFF;

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < ST; ++s) mbar_init(smem_u32(&mbar[s]), NPB);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  for (int i = tid; i < (C::QT_BYTES + 2 * C::PT_BYTES) / 16; i += 128)
    reinterpret_cast<uint4*>(smem + C::QT_OFF)[i] = make_uint4(0, 0, 0, 0);
  __syncthreads();
  const int ntok = max(c1 - c0, 0), nblk = (ntok + BLK - 1) / BLK;
  {
    const int* bt = p.block_table + static_cast<size_t>(b) * p.bt_stride + c0 / P;
    int* pidx = reinterpret_cast<int*>(smem + C::PIDX_OFF);
    const int npg = (ntok + P - 1) / P;
    for (int i = tid; i < npg; i += 128) pidx[i] = __ldg(bt + i);
  }
  const int* pidx_s = reinterpret_cast<const int*>(smem + C::PIDX_OFF);
  const uint8_t* poolh = p.pool + static_cast<size_t>(kvh) * CHUNK;
  const uint32_t pstride = static_cast<uint32_t>(p.Hkv) * CHUNK;
  const uint32_t mbar0 = smem_u32(&mbar[0]);
  const uint32_t dst_w = sbase + C::RING_OFF + w * CHUNK;
  auto issue = [&](int nb) {
    const int s = nb % ST;
    const uint32_t mb = mbar0 + 8 * s;
    const bool has = nb * BLK + w * P < ntok;
    mbar_expect_tx(mb, has ? CHUNK : 0);
    if (has) {
      const uint32_t page = static_cast<uint32_t>(pidx_s[nb * NPB + w]);
      bulk_g2s(dst_w + s * L::STAGE, poolh + static_cast<uint64_t>(page) * pstride, CHUNK, mb);
    }
  };
  __syncthreads();
  if (w < NPB && lane == 0) {
#pragma unroll
    for (int s = 0; s < ST - 1; ++s)
      if (s < nblk) issue(s);
  }
  __shared__ float qstage[1];  // (unused by the reduced prologue)
  q_prologue<CB_REDUCED, NQ>(p, b, kvh, w, lane, qt, qstage, sfac_s);
  fence_async_smem();
  __syncthreads();
  const float sfac = sfac_s[tig];

  // ---------------------------------------------------------------- per-thread constants
  const int bq = gid & 1, ipq = (gid >> 1) & 1, qbq = gid >> 2;
  const int tau0 = 16 * w + 8 * qbq + 4 * ipq + bq, tau1 = tau0 + 2;
  const uint32_t kro0 = opaque((tau0 / P) * CHUNK + L::K_OFF + (tau0 % P) * 64 + 16 * tig);
  const uint32_t kro1 = opaque((tau1 / P) * CHUNK + L::K_OFF + (tau1 % P) * 64 + 16 * tig);
  const uint32_t kso0 = opaque((tau0 / P) * CHUNK + L::KS_OFF + (tau0 % P) * 8);
  const uint32_t kso1 = opaque((tau1 / P) * CHUNK + L::KS_OFF + (tau1 % P) * 8);
  const uint32_t kfo0 = opaque((tau0 / P) * CHUNK + L::FK_OFF), kfo1 = opaque((tau1 / P) * CHUNK + L::FK_OFF);
  const int kfb0 = tau0 % P, kfb1 = tau1 % P;
  const uint64_t qd0 = opaque64(sdesc(qt));
  const uint64_t pdesc0 = opaque64(sdesc_p(pt));
  // V^T entries of tokens 32 ks + 8 tig .. +7: wl = h (groups 0, 2) and 2 + h (groups 1, 3), h = w & 1
  const int hh = w & 1;
  uint32_t vo[2][2], vso8[2], vfo[2];
  int vfsh[2];
#pragma unroll
  for (int ks = 0; ks < 2; ++ks) {
    const int tb = 32 * ks + 8 * tig;
    const int c8 = (tb % P) / 8;
#pragma unroll
    for (int e = 0; e < 2; ++e)
      vo[ks][e] = opaque((tb / P) * CHUNK + L::V_OFF + ((((2 * e + hh) * 8 + gid) * (P / 8) + c8) << 4));
    vso8[ks] = opaque((tb / P) * CHUNK + L::VS_OFF + (8 * c8) * 2);  // + g * P * 2: 8 scales of the chunk
    vfo[ks] = opaque((tb / P) * CHUNK + L::FV_OFF);
    vfsh[ks] = 8 * c8;
  }
  const int cp0 = (tau0 % P) % 8;
  const uint32_t vso = opaque((tau0 / P) * CHUNK + L::VS_OFF + (((tau0 % P) / 8) * 8 + ((cp0 & 1) ? 4 + (cp0 >> 1) : (cp0 >> 1))) * 2);
  const int k0p = 16 * bq + 4 * (2 * (w & 1) + qbq) + 2 * ipq;
  const uint32_t pwo = opaque(((w >> 1) * 2) * PTILE + (k0p / 16) * PLBO + (2 * tig) * 16 + (k0p % 16));
  const bool clsH = (w >> 1) != 0;  // (I, H) tiles: warps 0-1 hold class I, warps 2-3 class H

  float OIH[4][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};  // group g: d = 32 g + 16 hh + gid + 8 r
  float OF[2][2] = {{0.f, 0.f}, {0.f, 0.f}};                            // tile m: d = 64 m + 16 w + gid + 8 r
  float lsum = 0.f, mref = NEG_INF;
  int eref[4];
  uint32_t inv2[4];
#pragma unroll
  for (int g = 0; g < 4; ++g) { eref[g] = EREF_NONE; inv2[g] = 0u; }
  float paIH[4][4], paF[2][8];
  float sacc[4][12];
  uint32_t ka[4][4];
  uint2 kraw0, kraw1;
  uint32_t kf0, kf1;
  uint32_t vaIH[4][2][4], vaF[2][2][4];
  if constexpr (!SYNC) {  // pinned at the first wait of the pipeline
#pragma unroll
    for (int g = 0; g < 4; ++g)
#pragma unroll
      for (int ks = 0; ks < 2; ++ks)
#pragma unroll
        for (int k = 0; k < 4; ++k) vaIH[g][ks][k] = 0u;
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int ks = 0; ks < 2; ++ks)
#pragma unroll
        for (int k = 0; k < 4; ++k) vaF[m][ks][k] = 0u;
  }

  auto qk_issue = [&](int nb) {
    const uint32_t sb = sbase + C::RING_OFF + (nb % ST) * L::STAGE;
    const uint4 x0 = lds128(sb + kro0), x1 = lds128(sb + kro1);
    const uint32_t w0[4] = {x0.x, x0.y, x0.z, x0.w}, w1[4] = {x1.x, x1.y, x1.z, x1.w};
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      cvt_ab_m(w0[jj], FULL, FULL, ka[jj][0], ka[jj][2]);
      cvt_ab_m(w1[jj], FULL, FULL, ka[jj][1], ka[jj][3]);
    }
    kraw0 = lds64(sb + kso0);
    kraw1 = lds64(sb + kso1);
    kf0 = lds32(sb + kfo0);
    kf1 = lds32(sb + kfo1);
    wg_fence();
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) wg_n24(sacc[jj], ka[jj], qd0 + ((jj * NQ * 32) >> 4));
    wg_commit();
  };
  auto pv_accumulate = [&]() {
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      OIH[g][0] += fmaf(paIH[g][1], LO_INV, paIH[g][0]);
      OIH[g][1] += fmaf(paIH[g][3], LO_INV, paIH[g][2]);
    }
    const bool g1 = (w >> 1) != 0;
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      OF[m][0] += g1 ? fmaf(paF[m][5], LO_INV, paF[m][4]) : fmaf(paF[m][1], LO_INV, paF[m][0]);
      OF[m][1] += g1 ? fmaf(paF[m][7], LO_INV, paF[m][6]) : fmaf(paF[m][3], LO_INV, paF[m][2]);
    }
  };

  auto pv_retire = [&]() {
#pragma unroll
    for (int g = 0; g < 4; ++g) wg_wait0(paIH[g]);
#pragma unroll
    for (int m = 0; m < 2; ++m) wg_wait0(paF[m]);
#pragma unroll
    for (int g = 0; g < 4; ++g) { pin_regs(vaIH[g][0]); pin_regs(vaIH[g][1]); }
#pragma unroll
    for (int m = 0; m < 2; ++m) { pin_regs(vaF[m][0]); pin_regs(vaF[m][1]); }
  };

  if constexpr (!SYNC) {
    mbar_wait(smem_u32(&mbar[0]), 0);
    qk_issue(0);
  }
  for (int n = 0; n < nblk; ++n) {
    const int s = n % ST;
    const uint32_t sb = sbase + C::RING_OFF + s * L::STAGE;
    const int t0 = c0 + n * BLK;
    if (w < NPB && lane == 0 && n + ST - 1 < nblk) issue(n + ST - 1);
    if constexpr (SYNC) {
      mbar_wait(smem_u32(&mbar[s]), (n / ST) & 1);
      qk_issue(n);
    }

    // ---------------- (a) wait QK(n) (and PV(n-1)); O += PV(n-1); scores
    wg_wait_all();
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) wg_wait0(sacc[jj]);
#pragma unroll
    for (int k = 0; k < 4; ++k) pin_regs(ka[k]);
    if constexpr (!SYNC) {
      pv_retire();
      if (n > 0) pv_accumulate();
    }
    const bool val0 = t0 + tau0 < c1 && t0 + tau0 >= p.kv_start, val1 = t0 + tau1 < c1 && t0 + tau1 >= p.kv_start;  // kv_start: masked leading tokens (sinks)
    float S0 = 0.f, S1 = 0.f, S0b = 0.f, S1b = 0.f;
    {
      const bool F0 = (kf0 >> kfb0) & 1u, F1 = (kf1 >> kfb1) & 1u;
      const uint32_t r0[2] = {kraw0.x, kraw0.y}, r1[2] = {kraw1.x, kraw1.y};
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        float& A0 = (jj & 1) ? S0b : S0;
        float& A1 = (jj & 1) ? S1b : S1;
        const uint32_t h0 = (r0[jj >> 1] >> (16 * (jj & 1))) & 0xFFFFu, h1 = (r1[jj >> 1] >> (16 * (jj & 1))) & 0xFFFFu;
        const float s0 = __half2float(__ushort_as_half(static_cast<unsigned short>(h0 & 0x7FFFu)));
        const float s1 = __half2float(__ushort_as_half(static_cast<unsigned short>(h1 & 0x7FFFu)));
        const float vI0 = fmaf(sacc[jj][1], LO_INV, sacc[jj][0]), vI1 = fmaf(sacc[jj][3], LO_INV, sacc[jj][2]);
        const float vH0 = fmaf(sacc[jj][5], LO_INV, sacc[jj][4]), vH1 = fmaf(sacc[jj][7], LO_INV, sacc[jj][6]);
        const float vF0 = fmaf(sacc[jj][9], LO_INV, sacc[jj][8]), vF1 = fmaf(sacc[jj][11], LO_INV, sacc[jj][10]);
        const float x0 = F0 ? vF0 : ((h0 & 0x8000u) ? vH0 : vI0);
        const float x1 = F1 ? vF1 : ((h1 & 0x8000u) ? vH1 : vI1);
        A0 = fmaf(s0, x0, A0);
        A1 = fmaf(s1, x1, A1);
      }
    }
    S0 += S0b;
    S1 += S1b;
    S0 = val0 ? S0 * sfac : NEG_INF;
    S1 = val1 ? S1 * sfac : NEG_INF;

    // ---------------- (c) V^T fragments of block n, masked per class; V scales of rows tau0 / tau1
    uint32_t svraw[4];
    {
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const uint4 elo = lds128(sb + vo[ks][0]), ehi = lds128(sb + vo[ks][1]);
        const uint32_t fw = lds32(sb + vfo[ks]) >> vfsh[ks];
        const uint32_t Fe = sign_bytes((fw & 0xFu) * 0x10204080u), Fo = sign_bytes(((fw >> 4) & 0xFu) * 0x10204080u);
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          const uint4 sv = lds128(sb + vso8[ks] + g * P * 2);
          const uint32_t He = prmt(sv.x, sv.y, 0xFDB9u), Ho = prmt(sv.z, sv.w, 0xFDB9u);  // fp16 sign bits
          const uint32_t me = clsH ? He : ~(He | Fe), mo = clsH ? Ho : ~(Ho | Fo);
          // (value selects: a reference to one of the two entries would put them in local memory)
          const uint32_t wa = (g & 1) ? ((g >> 1) ? ehi.z : ehi.x) : ((g >> 1) ? elo.z : elo.x);
          const uint32_t wb = (g & 1) ? ((g >> 1) ? ehi.w : ehi.y) : ((g >> 1) ? elo.w : elo.y);
          cvt_ab_m(wa, me, mo, vaIH[g][ks][0], vaIH[g][ks][2]);
          cvt_ab_m(wb, me, mo, vaIH[g][ks][1], vaIH[g][ks][3]);
        }
        // F tiles: rows of warp w: d = 64 m + 16 w + gid + 8 r -> entry wl = w (= hh or 2 + hh), words 2 m + r
        const bool hiw = (w >> 1) != 0;
        cvt_ab_m(hiw ? ehi.x : elo.x, Fe, Fo, vaF[0][ks][0], vaF[0][ks][2]);
        cvt_ab_m(hiw ? ehi.y : elo.y, Fe, Fo, vaF[0][ks][1], vaF[0][ks][3]);
        cvt_ab_m(hiw ? ehi.z : elo.z, Fe, Fo, vaF[1][ks][0], vaF[1][ks][2]);
        cvt_ab_m(hiw ? ehi.w : elo.w, Fe, Fo, vaF[1][ks][1], vaF[1][ks][3]);
      }
#pragma unroll
      for (int g = 0; g < 4; ++g) svraw[g] = lds32(sb + vso + g * P * 2) & 0x7FFF7FFFu;
    }

    // ---------------- (b) QK(n+1)
    if constexpr (!SYNC) {
      const int nq = min(n + 1, nblk - 1);
      if (nq != n) mbar_wait(smem_u32(&mbar[nq % ST]), (nq / ST) & 1);
      qk_issue(nq);
    }

    // ---------------- (d) softmax, P'
    const uint32_t ptb = pt + (n & 1) * C::PT_BYTES;
    float p0, p1;
    bool pflag = false;
    const uint32_t vmask2 = (val0 ? 0xFFFFu : 0u) | (val1 ? 0xFFFF0000u : 0u);
    auto make_p = [&]() {
      p0 = val0 ? ex2(S0 - mref) : 0.f;
      p1 = val1 ? ex2(S1 - mref) : 0.f;
      pflag = make_pprime(p0, p1, svraw, inv2, vmask2, ptb + pwo);
    };
    bool flag = (S0 > mref + TAU) || (S1 > mref + TAU);
    make_p();
    flag = flag || pflag;
    fence_async_smem();
    const int anyflag = __syncthreads_or(flag);
    if (anyflag) {
      float bm = fmaxf(S0, S1);
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) bm = fmaxf(bm, __shfl_xor_sync(FULL, bm, o));
      if (gid == 0) red_m[w][tig] = bm;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const int e0 = static_cast<int>((svraw[g] >> 10) & 31) - 15, e1 = static_cast<int>((svraw[g] >> 26) & 31) - 15;
        int be = max(val0 ? e0 : EREF_NONE, val1 ? e1 : EREF_NONE);
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) be = max(be, __shfl_xor_sync(FULL, be, o));
        if (lane == 0) red_e[w][g] = be;
      }
      __syncthreads();
      float nm = mref;
#pragma unroll
      for (int ww = 0; ww < NWARP; ++ww) nm = fmaxf(nm, red_m[ww][tig]);
      const float alpha = mref == NEG_INF ? 0.f : ex2(mref - nm);
      float bg[4];
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        int ne = eref[g];
#pragma unroll
        for (int ww = 0; ww < NWARP; ++ww) ne = max(ne, red_e[ww][g]);
        ne = (ne == EREF_NONE) ? ne : min(max(ne, -14), 14);
        bg[g] = (eref[g] == EREF_NONE || eref[g] - ne < -126) ? 0.f : __uint_as_float(static_cast<uint32_t>(127 + eref[g] - ne) << 23);
        eref[g] = ne;
        const uint32_t hv = ne == EREF_NONE ? 0u : static_cast<uint32_t>(15 - ne) << 10;
        inv2[g] = hv | (hv << 16);
      }
      mref = nm;
#pragma unroll
      for (int g = 0; g < 4; ++g) { OIH[g][0] *= alpha * bg[g]; OIH[g][1] *= alpha * bg[g]; }
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const float f = alpha * ((w >> 1) ? bg[2 * m + 1] : bg[2 * m]);
        OF[m][0] *= f;
        OF[m][1] *= f;
      }
      lsum *= alpha;
      __syncthreads();
      make_p();
      fence_async_smem();
      __syncthreads();
    }
    lsum += p0 + p1;

    // ---------------- (e) PV(n)
    const uint64_t pdb = pdesc0 + (((n & 1) * C::PT_BYTES) >> 4);
    wg_fence();
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      wg_n8<0>(paIH[g], vaIH[g][0], pdb + (((0 * 2 + (g >> 1)) * PTILE + (g & 1) * PSBO) >> 4));
      wg_n8<1>(paIH[g], vaIH[g][1], pdb + (((1 * 2 + (g >> 1)) * PTILE + (g & 1) * PSBO) >> 4));
    }
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      wg_n16<0>(paF[m], vaF[m][0], pdb + (((0 * 2 + m) * PTILE) >> 4));
      wg_n16<1>(paF[m], vaF[m][1], pdb + (((1 * 2 + m) * PTILE) >> 4));
    }
    wg_commit();
    if constexpr (SYNC) {
      wg_wait_all();
      pv_retire();
      pv_accumulate();
    }
  }
  if constexpr (!SYNC) {
    wg_wait_all();
    pv_retire();
    pv_accumulate();
  }

  // ---------------------------------------------------------------- inverse transforms, CTA totals, merge
  float ofac[4];
#pragma unroll
  for (int g = 0; g < 4; ++g) ofac[g] = FV * (eref[g] == EREF_NONE ? 0.f : exp2f(static_cast<float>(eref[g])));
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) lsum += __shfl_xor_sync(FULL, lsum, o);
  __syncthreads();  // ring no longer used
  float* fin = reinterpret_cast<float*>(smem);  // [3 classes][G][D]
#pragma unroll
  for (int g = 0; g < 4; ++g)
#pragma unroll
    for (int r = 0; r < 2; ++r) fin[((clsH ? 1 : 0) * G + tig) * D + 32 * g + 16 * hh + gid + 8 * r] = OIH[g][r] * ofac[g];
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int r = 0; r < 2; ++r)
      fin[(2 * G + tig) * D + 64 * m + 16 * w + gid + 8 * r] = OF[m][r] * ((w >> 1) ? ofac[2 * m + 1] : ofac[2 * m]);
  if (gid == 0) { red_l[w][tig] = lsum; red_m[w][tig] = mref; }
  __syncthreads();
  float* outf = fin + 3 * G * D;  // [G][D] + merge weights
  {
    float xi[4], xh[4], xf[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      xi[i] = fin[(0 * G + w) * D + 4 * lane + i];
      xh[i] = fin[(1 * G + w) * D + 4 * lane + i];
      xf[i] = fin[(2 * G + w) * D + 4 * lane + i];
    }
    fwht_q<32>(xh, lane);
    fwht_q<128>(xf, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) outf[w * D + 4 * lane + i] = xi[i] + xh[i] + xf[i];
  }
  __syncthreads();
  float Mc[G], Lc[G];
#pragma unroll
  for (int h = 0; h < G; ++h) {
    Mc[h] = red_m[0][h];
    Lc[h] = red_l[0][h] + red_l[1][h] + red_l[2][h] + red_l[3][h];
  }
  merge_store(p, outf, red_l, b, kvh, split, ns_b + p.tail_nslot, Mc, Lc,
              &last_flag);
}

// ================================================================================================= reduced kernel v2
// 2 x 2 (cq+f) on keys and values, kernel variant 2.  Synchronous wgmma rounds (QK(n) and PV(n) each waited before the
// next is issued: the value classes double the PV operands, so overlapping QK(n+1) with PV(n) would cost 2 CTAs / SM),
// but everything that is not a wgmma operand is prefetched while a round is in flight:
//   * during QK(n): the V^T code words, V scale class masks (I / H32 = sign bits, F = page bit plane) and V scales of
//     block n -> after the wait only LOP3 / SHF remain before PV(n);
//   * during PV(n): the K code words, K scales and K head bits of block n+1 (stage n+1 waited for here).
// QK as k_cbk (I / H32 hi N = 16 with H128 hi + 16 lo, I / H32 lo chained N = 8: 8 accumulators per group, select
// x = F ? hi_F + lo_F / 16 : (H ? acc_H : acc_I)); PV as k_cbr (I / H32 M-side tiles per group, F two-group tiles);
// ring pages at a stride of 64 mod 128 B (no bank conflicts on the V^T / V scale loads).
template <int P, int ST>
__global__ void __launch_bounds__(128, 1) k_cbr2(const CbAttnParams p) {
  using C = CfgA<CB_REDUCED, P, ST>;
  using L = LayA<CB_REDUCED, P>;
  constexpr int NQ = C::NQ, CHUNK = L::CHUNK, NPB = L::NPB, PS = L::PSTR;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ __align__(8) uint64_t mbar[ST];
  __shared__ float red_m[NWARP][G], red_l[NWARP][G], sfac_s[G];
  __shared__ int red_e[NWARP][4];
  __shared__ int last_flag;

  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, gid = lane >> 2, tig = lane & 3;
  const int kvh = blockIdx.x, split = blockIdx.y, b = blockIdx.z;
  const int len = p.seq_lens[b];
  const int c0 = split * p.chunk, c1 = min(len, c0 + p.chunk);
  const int ns_b = max(1, (len + p.chunk - 1) / p.chunk);
  if (split >= ns_b || len <= 0) return;  // (len 0: the staging-window kernel writes the output)
  const uint32_t sbase = smem_u32(smem);
  const uint32_t qt = sbase + C::QT_OFF, pt = sbase + C::PT_OFF;

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < ST; ++s) mbar_init(smem_u32(&mbar[s]), NPB);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  for (int i = tid; i < (C::QT_BYTES + 2 * C::PT_BYTES) / 16; i += 128)
    reinterpret_cast<uint4*>(smem + C::QT_OFF)[i] = make_uint4(0, 0, 0, 0);
  __syncthreads();
  const int ntok = max(c1 - c0, 0), nblk = (ntok + BLK - 1) / BLK;
  {
    const int* bt = p.block_table + static_cast<size_t>(b) * p.bt_stride + c0 / P;
    int* pidx = reinterpret_cast<int*>(smem + C::PIDX_OFF);
    const int npg = (ntok + P - 1) / P;
    for (int i = tid; i < npg; i += 128) pidx[i] = __ldg(bt + i);
  }
  const int* pidx_s = reinterpret_cast<const int*>(smem + C::PIDX_OFF);
  const uint8_t* poolh = p.pool + static_cast<size_t>(kvh) * CHUNK;
  const uint32_t pstride = static_cast<uint32_t>(p.Hkv) * CHUNK;
  const uint32_t mbar0 = smem_u32(&mbar[0]);
  const uint32_t dst_w = sbase + C::RING_OFF + w * PS;
  auto issue = [&](int nb) {
    const int s = nb % ST;
    const uint32_t mb = mbar0 + 8 * s;
    const bool has = nb * BLK + w * P < ntok;
    mbar_expect_tx(mb, has ? CHUNK : 0);
    if (has) {
      const uint32_t page = static_cast<uint32_t>(pidx_s[nb * NPB + w]);
      bulk_g2s(dst_w + s * L::STAGE, poolh + static_cast<uint64_t>(page) * pstride, CHUNK, mb);
    }
  };
  __syncthreads();
  if (w < NPB && lane == 0) {
#pragma unroll
    for (int s = 0; s < ST - 1; ++s)
      if (s < nblk) issue(s);
  }
  q_prologue<CB_KONLY, NQ>(p, b, kvh, w, lane, qt, nullptr, sfac_s);  // (konly q layout: chained I / H32 lo)
  fence_async_smem();
  __syncthreads();
  const float sfac = sfac_s[tig];

  // ---------------------------------------------------------------- per-thread constants
  const int bq = gid & 1, ipq = (gid >> 1) & 1, qbq = gid >> 2;
  const int tau0 = 16 * w + 8 * qbq + 4 * ipq + bq, tau1 = tau0 + 2;
  const uint32_t kro0 = opaque((tau0 / P) * PS + L::K_OFF + (tau0 % P) * 64 + 16 * tig);
  const uint32_t kro1 = opaque((tau1 / P) * PS + L::K_OFF + (tau1 % P) * 64 + 16 * tig);
  const uint32_t kso0 = opaque((tau0 / P) * PS + L::KS_OFF + (tau0 % P) * 8);
  const uint32_t kso1 = opaque((tau1 / P) * PS + L::KS_OFF + (tau1 % P) * 8);
  const uint32_t kfo0 = opaque((tau0 / P) * PS + L::FK_OFF), kfo1 = opaque((tau1 / P) * PS + L::FK_OFF);
  const int kfb0 = tau0 % P, kfb1 = tau1 % P;
  const uint64_t qd0 = opaque64(sdesc(qt));
  const uint64_t pdesc0 = opaque64(sdesc_p(pt));
  const int hh = w & 1;
  uint32_t vo[2][2], vso8[2], vfo[2];
  int vfsh[2];
#pragma unroll
  for (int ks = 0; ks < 2; ++ks) {
    const int tb = 32 * ks + 8 * tig;
    const int c8 = (tb % P) / 8;
#pragma unroll
    for (int e = 0; e < 2; ++e)
      vo[ks][e] = opaque((tb / P) * PS + L::V_OFF + ((((2 * e + hh) * 8 + gid) * (P / 8) + c8) << 4));
    vso8[ks] = opaque((tb / P) * PS + L::VS_OFF + (8 * c8) * 2);  // + g * P * 2: the 8 scales of the chunk
    vfo[ks] = opaque((tb / P) * PS + L::FV_OFF);
    vfsh[ks] = 8 * c8;
  }
  const int cp0 = (tau0 % P) % 8;
  const uint32_t vso = opaque((tau0 / P) * PS + L::VS_OFF + (((tau0 % P) / 8) * 8 + ((cp0 & 1) ? 4 + (cp0 >> 1) : (cp0 >> 1))) * 2);
  const int k0p = 16 * bq + 4 * (2 * (w & 1) + qbq) + 2 * ipq;
  const uint32_t pwo = opaque(((w >> 1) * 2) * PTILE + (k0p / 16) * PLBO + (2 * tig) * 16 + (k0p % 16));
  const bool clsH = (w >> 1) != 0;  // (I, H) tiles: warps 0-1 hold class I, warps 2-3 class H

  float OIH[4][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};  // group g: d = 32 g + 16 hh + gid + 8 r
  float OF[2][2] = {{0.f, 0.f}, {0.f, 0.f}};                            // tile m: d = 64 m + 16 w + gid + 8 r
  float lsum = 0.f, mref = NEG_INF;
  int eref[4];
  uint32_t inv2[4];
#pragma unroll
  for (int g = 0; g < 4; ++g) { eref[g] = EREF_NONE; inv2[g] = 0u; }

  // prefetched K data of the next block (plain registers)
  uint4 kx0, kx1;
  uint2 kraw0, kraw1;
  uint32_t kf0, kf1;
  auto k_prefetch = [&](int nb) {
    const uint32_t sb = sbase + C::RING_OFF + (nb % ST) * L::STAGE;
    kx0 = lds128(sb + kro0);
    kx1 = lds128(sb + kro1);
    kraw0 = lds64(sb + kso0);
    kraw1 = lds64(sb + kso1);
    kf0 = lds32(sb + kfo0);
    kf1 = lds32(sb + kfo1);
  };
  mbar_wait(smem_u32(&mbar[0]), 0);
  k_prefetch(0);

  for (int n = 0; n < nblk; ++n) {
    const int s = n % ST;
    const uint32_t sb = sbase + C::RING_OFF + s * L::STAGE;
    const int t0 = c0 + n * BLK;
    if (w < NPB && lane == 0 && n + ST - 1 < nblk) issue(n + ST - 1);

    // ---------------- QK(n): K fragments from the prefetched words, issue
    float sacc[4][8];
    uint32_t ka[4][4];
    {
      const uint32_t w0[4] = {kx0.x, kx0.y, kx0.z, kx0.w}, w1[4] = {kx1.x, kx1.y, kx1.z, kx1.w};
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        ka[jj][0] = w0[jj] & 0x87878787u;
        ka[jj][2] = __funnelshift_l(w0[jj], w0[jj], 4) & 0x87878787u;
        ka[jj][1] = w1[jj] & 0x87878787u;
        ka[jj][3] = __funnelshift_l(w1[jj], w1[jj], 4) & 0x87878787u;
      }
      wg_fence();
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        wg_n16<0>(sacc[jj], ka[jj], qd0 + ((jj * NQ * 32) >> 4));
        wg_n8<1>(*reinterpret_cast<float(*)[4]>(sacc[jj]), ka[jj], qd0 + ((jj * NQ * 32 + 2 * 256) >> 4));
      }
      wg_commit();
    }
    // ---------------- while QK(n) runs: V^T words, class masks, V scales of block n
    uint4 ve[2][2];
    uint32_t mIe[2][4], mIo[2][4], Fe[2], Fo[2];  // (I or H by warp) masks per (ks, group), F masks per ks
    uint32_t svraw[4];
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      ve[ks][0] = lds128(sb + vo[ks][0]);
      ve[ks][1] = lds128(sb + vo[ks][1]);
      const uint32_t fw = lds32(sb + vfo[ks]) >> vfsh[ks];
      Fe[ks] = sign_bytes((fw & 0xFu) * 0x10204080u);
      Fo[ks] = sign_bytes(((fw >> 4) & 0xFu) * 0x10204080u);
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const uint4 sv = lds128(sb + vso8[ks] + g * P * 2);
        const uint32_t He = prmt(sv.x, sv.y, 0xFDB9u), Ho = prmt(sv.z, sv.w, 0xFDB9u);  // fp16 sign bits
        mIe[ks][g] = clsH ? (He & ~Fe[ks]) : ~(He | Fe[ks]);
        mIo[ks][g] = clsH ? (Ho & ~Fo[ks]) : ~(Ho | Fo[ks]);
      }
    }
#pragma unroll
    for (int g = 0; g < 4; ++g) svraw[g] = lds32(sb + vso + g * P * 2) & 0x7FFF7FFFu;

    // ---------------- wait QK(n); scores with the per-(token, group) column select
    wg_wait_all();
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) { wg_wait0(sacc[jj]); pin_regs(ka[jj]); }
    const bool val0 = t0 + tau0 < c1 && t0 + tau0 >= p.kv_start, val1 = t0 + tau1 < c1 && t0 + tau1 >= p.kv_start;  // kv_start: masked leading tokens (sinks)
    float S0 = 0.f, S1 = 0.f, S0b = 0.f, S1b = 0.f;
    {
      const bool F0 = (kf0 >> kfb0) & 1u, F1 = (kf1 >> kfb1) & 1u;
      const uint32_t r0[2] = {kraw0.x, kraw0.y}, r1[2] = {kraw1.x, kraw1.y};
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        float& A0 = (jj & 1) ? S0b : S0;
        float& A1 = (jj & 1) ? S1b : S1;
        const uint32_t hb = 0x8000u << (16 * (jj & 1));
        const bool H0 = (r0[jj >> 1] & hb) != 0, H1 = (r1[jj >> 1] & hb) != 0;
        const float2 m0 = h2f2(r0[jj >> 1] & 0x7FFF7FFFu), m1 = h2f2(r1[jj >> 1] & 0x7FFF7FFFu);
        const float s0 = (jj & 1) ? m0.y : m0.x, s1 = (jj & 1) ? m1.y : m1.x;
        const float x0 = F0 ? fmaf(sacc[jj][5], LO_INV, sacc[jj][4]) : (H0 ? sacc[jj][1] : sacc[jj][0]);
        const float x1 = F1 ? fmaf(sacc[jj][7], LO_INV, sacc[jj][6]) : (H1 ? sacc[jj][3] : sacc[jj][2]);
        A0 = fmaf(s0, x0, A0);
        A1 = fmaf(s1, x1, A1);
      }
    }
    S0 += S0b;
    S1 += S1b;
    S0 = val0 ? S0 * sfac : NEG_INF;
    S1 = val1 ? S1 * sfac : NEG_INF;

    // ---------------- softmax (lazy reference), P' -> smem
    const uint32_t ptb = pt + (n & 1) * C::PT_BYTES;
    float p0, p1;
    bool pflag = false;
    const uint32_t vmask2 = (val0 ? 0xFFFFu : 0u) | (val1 ? 0xFFFF0000u : 0u);
    auto make_p = [&]() {
      p0 = val0 ? ex2(S0 - mref) : 0.f;
      p1 = val1 ? ex2(S1 - mref) : 0.f;
      pflag = make_pprime(p0, p1, svraw, inv2, vmask2, ptb + pwo);
    };
    bool flag = (S0 > mref + TAU) || (S1 > mref + TAU);
    make_p();
    flag = flag || pflag;
    fence_async_smem();
    const int anyflag = __syncthreads_or(flag);
    if (anyflag) {
      float bm = fmaxf(S0, S1);
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) bm = fmaxf(bm, __shfl_xor_sync(FULL, bm, o));
      if (gid == 0) red_m[w][tig] = bm;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const int e0 = static_cast<int>((svraw[g] >> 10) & 31) - 15, e1 = static_cast<int>((svraw[g] >> 26) & 31) - 15;
        int be = max(val0 ? e0 : EREF_NONE, val1 ? e1 : EREF_NONE);
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) be = max(be, __shfl_xor_sync(FULL, be, o));
        if (lane == 0) red_e[w][g] = be;
      }
      __syncthreads();
      float nm = mref;
#pragma unroll
      for (int ww = 0; ww < NWARP; ++ww) nm = fmaxf(nm, red_m[ww][tig]);
      const float alpha = mref == NEG_INF ? 0.f : ex2(mref - nm);
      float bg[4];
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        int ne = eref[g];
#pragma unroll
        for (int ww = 0; ww < NWARP; ++ww) ne = max(ne, red_e[ww][g]);
        ne = (ne == EREF_NONE) ? ne : min(max(ne, -14), 14);
        bg[g] = (eref[g] == EREF_NONE || eref[g] - ne < -126) ? 0.f : __uint_as_float(static_cast<uint32_t>(127 + eref[g] - ne) << 23);
        eref[g] = ne;
        const uint32_t hv = ne == EREF_NONE ? 0u : static_cast<uint32_t>(15 - ne) << 10;
        inv2[g] = hv | (hv << 16);
      }
      mref = nm;
#pragma unroll
      for (int g = 0; g < 4; ++g) { OIH[g][0] *= alpha * bg[g]; OIH[g][1] *= alpha * bg[g]; }
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const float f = alpha * ((w >> 1) ? bg[2 * m + 1] : bg[2 * m]);
        OF[m][0] *= f;
        OF[m][1] *= f;
      }
      lsum *= alpha;
      __syncthreads();
      make_p();
      fence_async_smem();
      __syncthreads();
    }
    lsum += p0 + p1;

    // ---------------- PV(n): masked V^T fragments (LOP3 / SHF only), issue
    uint32_t vaIH[4][2][4], vaF[2][2][4];
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const uint4 elo = ve[ks][0], ehi = ve[ks][1];
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const uint32_t wa = (g & 1) ? ((g >> 1) ? ehi.z : ehi.x) : ((g >> 1) ? elo.z : elo.x);
        const uint32_t wb = (g & 1) ? ((g >> 1) ? ehi.w : ehi.y) : ((g >> 1) ? elo.w : elo.y);
        cvt_ab_m(wa, mIe[ks][g], mIo[ks][g], vaIH[g][ks][0], vaIH[g][ks][2]);
        cvt_ab_m(wb, mIe[ks][g], mIo[ks][g], vaIH[g][ks][1], vaIH[g][ks][3]);
      }
      const bool hiw = (w >> 1) != 0;
      cvt_ab_m(hiw ? ehi.x : elo.x, Fe[ks], Fo[ks], vaF[0][ks][0], vaF[0][ks][2]);
      cvt_ab_m(hiw ? ehi.y : elo.y, Fe[ks], Fo[ks], vaF[0][ks][1], vaF[0][ks][3]);
      cvt_ab_m(hiw ? ehi.z : elo.z, Fe[ks], Fo[ks], vaF[1][ks][0], vaF[1][ks][2]);
      cvt_ab_m(hiw ? ehi.w : elo.w, Fe[ks], Fo[ks], vaF[1][ks][1], vaF[1][ks][3]);
    }
    float paIH[4][4], paF[2][8];
    const uint64_t pdb = pdesc0 + (((n & 1) * C::PT_BYTES) >> 4);
    wg_fence();
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      wg_n8<0>(paIH[g], vaIH[g][0], pdb + (((0 * 2 + (g >> 1)) * PTILE + (g & 1) * PSBO) >> 4));
      wg_n8<1>(paIH[g], vaIH[g][1], pdb + (((1 * 2 + (g >> 1)) * PTILE + (g & 1) * PSBO) >> 4));
    }
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      wg_n16<0>(paF[m], vaF[m][0], pdb + (((0 * 2 + m) * PTILE) >> 4));
      wg_n16<1>(paF[m], vaF[m][1], pdb + (((1 * 2 + m) * PTILE) >> 4));
    }
    wg_commit();
    // ---------------- while PV(n) runs: K data of block n+1
    {
      const int nq = min(n + 1, nblk - 1);
      if (nq != n) mbar_wait(smem_u32(&mbar[nq % ST]), (nq / ST) & 1);
      k_prefetch(nq);
    }
    wg_wait_all();
#pragma unroll
    for (int g = 0; g < 4; ++g) { wg_wait0(paIH[g]); pin_regs(vaIH[g][0]); pin_regs(vaIH[g][1]); }
#pragma unroll
    for (int m = 0; m < 2; ++m) { wg_wait0(paF[m]); pin_regs(vaF[m][0]); pin_regs(vaF[m][1]); }
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      OIH[g][0] += fmaf(paIH[g][1], LO_INV, paIH[g][0]);
      OIH[g][1] += fmaf(paIH[g][3], LO_INV, paIH[g][2]);
    }
    {
      const bool g1 = (w >> 1) != 0;
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        OF[m][0] += g1 ? fmaf(paF[m][5], LO_INV, paF[m][4]) : fmaf(paF[m][1], LO_INV, paF[m][0]);
        OF[m][1] += g1 ? fmaf(paF[m][7], LO_INV, paF[m][6]) : fmaf(paF[m][3], LO_INV, paF[m][2]);
      }
    }
  }

  // ---------------------------------------------------------------- inverse transforms, CTA totals, merge
  float ofac[4];
#pragma unroll
  for (int g = 0; g < 4; ++g) ofac[g] = FV * (eref[g] == EREF_NONE ? 0.f : exp2f(static_cast<float>(eref[g])));
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) lsum += __shfl_xor_sync(FULL, lsum, o);
  __syncthreads();  // ring no longer used
  float* fin = reinterpret_cast<float*>(smem);  // [3 classes][G][D]
#pragma unroll
  for (int g = 0; g < 4; ++g)
#pragma unroll
    for (int r = 0; r < 2; ++r) fin[((clsH ? 1 : 0) * G + tig) * D + 32 * g + 16 * hh + gid + 8 * r] = OIH[g][r] * ofac[g];
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int r = 0; r < 2; ++r)
      fin[(2 * G + tig) * D + 64 * m + 16 * w + gid + 8 * r] = OF[m][r] * ((w >> 1) ? ofac[2 * m + 1] : ofac[2 * m]);
  if (gid == 0) { red_l[w][tig] = lsum; red_m[w][tig] = mref; }
  __syncthreads();
  float* outf = fin + 3 * G * D;  // [G][D] + merge weights
  {
    float xi[4], xh[4], xf[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      xi[i] = fin[(0 * G + w) * D + 4 * lane + i];
      xh[i] = fin[(1 * G + w) * D + 4 * lane + i];
      xf[i] = fin[(2 * G + w) * D + 4 * lane + i];
    }
    fwht_q<32>(xh, lane);
    fwht_q<128>(xf, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) outf[w * D + 4 * lane + i] = xi[i] + xh[i] + xf[i];
  }
  __syncthreads();
  float Mc[G], Lc[G];
#pragma unroll
  for (int h = 0; h < G; ++h) {
    Mc[h] = red_m[0][h];
    Lc[h] = red_l[0][h] + red_l[1][h] + red_l[2][h] + red_l[3][h];
  }
  merge_store(p, outf, red_l, b, kvh, split, ns_b + p.tail_nslot, Mc, Lc,
              &last_flag);
}

// ================================================================================================= full kernel
// QK: two rounds of two groups (wgmma N = 64: 8 variants x 4 heads x hi/lo); PV: four rounds k of four tiles
// (group g, classes 2k / 2k+1 on warps 0-1 / 2-3), class masks from a per-block one-hot table of the V side words.
// PIPE = 1: PV rounds software-pipelined two deep (~210 registers, 2 CTAs / SM); PIPE = 0: each round waited for
// (register-capped to 3 CTAs / SM).
template <int P, int ST, int PIPE>
__global__ void __launch_bounds__(128, PIPE ? 1 : 3) k_cbf(const CbAttnParams p) {
  using C = CfgA<CB_FULL, P, ST>;
  using L = LayA<CB_FULL, P>;
  constexpr int NQ = C::NQ, CHUNK = L::CHUNK, NPB = L::NPB;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ __align__(8) uint64_t mbar[ST];
  __shared__ float red_m[NWARP][G], red_l[NWARP][G], sfac_s[G];
  __shared__ int red_e[NWARP][4];
  __shared__ int last_flag;
  __shared__ float qstage[G * D];

  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, gid = lane >> 2, tig = lane & 3;
  const int kvh = blockIdx.x, split = blockIdx.y, b = blockIdx.z;
  const int len = p.seq_lens[b];
  const int c0 = split * p.chunk, c1 = min(len, c0 + p.chunk);
  const int ns_b = max(1, (len + p.chunk - 1) / p.chunk);
  if (split >= ns_b || len <= 0) return;  // (len 0: the staging-window kernel writes the output)
  const uint32_t sbase = smem_u32(smem);
  const uint32_t qt = sbase + C::QT_OFF, pt = sbase + C::PT_OFF, tbl = sbase + C::TBL_OFF;

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < ST; ++s) mbar_init(smem_u32(&mbar[s]), NPB);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  for (int i = tid; i < (C::QT_BYTES + 2 * C::PT_BYTES) / 16; i += 128)
    reinterpret_cast<uint4*>(smem + C::QT_OFF)[i] = make_uint4(0, 0, 0, 0);
  __syncthreads();
  const int ntok = max(c1 - c0, 0), nblk = (ntok + BLK - 1) / BLK;
  {
    const int* bt = p.block_table + static_cast<size_t>(b) * p.bt_stride + c0 / P;
    int* pidx = reinterpret_cast<int*>(smem + C::PIDX_OFF);
    const int npg = (ntok + P - 1) / P;
    for (int i = tid; i < npg; i += 128) pidx[i] = __ldg(bt + i);
  }
  const int* pidx_s = reinterpret_cast<const int*>(smem + C::PIDX_OFF);
  const uint8_t* poolh = p.pool + static_cast<size_t>(kvh) * CHUNK;
  const uint32_t pstride = static_cast<uint32_t>(p.Hkv) * CHUNK;
  const uint32_t mbar0 = smem_u32(&mbar[0]);
  const uint32_t dst_w = sbase + C::RING_OFF + w * CHUNK;
  auto issue = [&](int nb) {
    const int s = nb % ST;
    const uint32_t mb = mbar0 + 8 * s;
    const bool has = nb * BLK + w * P < ntok;
    mbar_expect_tx(mb, has ? CHUNK : 0);
    if (has) {
      const uint32_t page = static_cast<uint32_t>(pidx_s[nb * NPB + w]);
      bulk_g2s(dst_w + s * L::STAGE, poolh + static_cast<uint64_t>(page) * pstride, CHUNK, mb);
    }
  };
  __syncthreads();
  if (w < NPB && lane == 0) {
#pragma unroll
    for (int s = 0; s < ST - 1; ++s)
      if (s < nblk) issue(s);
  }
  q_prologue<CB_FULL, NQ>(p, b, kvh, w, lane, qt, qstage, sfac_s);
  fence_async_smem();
  __syncthreads();
  const float sfac = sfac_s[tig];

  const int bq = gid & 1, ipq = (gid >> 1) & 1, qbq = gid >> 2;
  const int tau0 = 16 * w + 8 * qbq + 4 * ipq + bq, tau1 = tau0 + 2;
  const uint32_t kro0 = opaque((tau0 / P) * CHUNK + L::K_OFF + (tau0 % P) * 64 + 16 * tig);
  const uint32_t kro1 = opaque((tau1 / P) * CHUNK + L::K_OFF + (tau1 % P) * 64 + 16 * tig);
  const uint32_t kso0 = opaque((tau0 / P) * CHUNK + L::KS_OFF + (tau0 % P) * 8);
  const uint32_t kso1 = opaque((tau1 / P) * CHUNK + L::KS_OFF + (tau1 % P) * 8);
  const uint32_t ksw0 = opaque((tau0 / P) * CHUNK + L::SWK_OFF + (tau0 % P) * 2);
  const uint32_t ksw1 = opaque((tau1 / P) * CHUNK + L::SWK_OFF + (tau1 % P) * 2);
  const uint64_t qd0 = opaque64(sdesc(qt));
  const uint64_t pdesc0 = opaque64(sdesc_p(pt));
  const int hh = w & 1;
  uint32_t vo[2][2];
#pragma unroll
  for (int ks = 0; ks < 2; ++ks) {
    const int tb = 32 * ks + 8 * tig;
    const int c8 = (tb % P) / 8;
#pragma unroll
    for (int e = 0; e < 2; ++e)
      vo[ks][e] = opaque((tb / P) * CHUNK + L::V_OFF + ((((2 * e + hh) * 8 + gid) * (P / 8) + c8) << 4));
  }
  // V side word -> one-hot table conversion: thread t < 64 handles block token t
  const uint32_t vsw_t = opaque(((tid & 63) / P) * CHUNK + L::SWV_OFF + ((tid & 63) % P) * 2);
  const uint32_t tbl_w = opaque((((tid & 63) / 8) * 2 + (tid & 1)) * 4 + ((tid & 7) >> 1));  // + g * 64
  const int cp0 = (tau0 % P) % 8;
  const uint32_t vso = opaque((tau0 / P) * CHUNK + L::VS_OFF + (((tau0 % P) / 8) * 8 + ((cp0 & 1) ? 4 + (cp0 >> 1) : (cp0 >> 1))) * 2);
  const int k0p = 16 * bq + 4 * (2 * (w & 1) + qbq) + 2 * ipq;
  const uint32_t pwo = opaque(((w >> 1) * 2) * PTILE + (k0p / 16) * PLBO + (2 * tig) * 16 + (k0p % 16));
  const int csel = w >> 1;  // class of this warp in pair tile k: 2 k + csel

  float O[4][4][2];  // [round k][group g][r]: class 2k + csel, d = 32 g + 16 hh + gid + 8 r
#pragma unroll
  for (int k = 0; k < 4; ++k)
#pragma unroll
    for (int g = 0; g < 4; ++g) O[k][g][0] = O[k][g][1] = 0.f;
  float lsum = 0.f, mref = NEG_INF;
  int eref[4];
  uint32_t inv2[4];
#pragma unroll
  for (int g = 0; g < 4; ++g) { eref[g] = EREF_NONE; inv2[g] = 0u; }

  for (int n = 0; n < nblk; ++n) {
    const int s = n % ST;
    const uint32_t sb = sbase + C::RING_OFF + s * L::STAGE;
    const int t0 = c0 + n * BLK;
    mbar_wait(smem_u32(&mbar[s]), (n / ST) & 1);
    // ---------------- V class table of this block (published by the barrier below)
    if (tid < 64) {
      const uint32_t sw = lds16(sb + vsw_t);
      const uint32_t m = (sw >> 8) & 3u;
      const uint32_t tb0 = tbl + (n & 1) * C::TBL_BYTES + tbl_w;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const uint32_t l = (sw >> (2 * g)) & 3u;
        const uint32_t cls = m < 2 ? 3 * m + l : 4 + m;
        asm volatile("st.shared.u8 [%0], %1;\n" ::"r"(tb0 + g * 64), "r"(1u << cls));
      }
    }
    // ---------------- QK(n) in two rounds, variant select
    float S0 = 0.f, S1 = 0.f;
    {
      uint32_t ka[4][4];
      const uint4 x0 = lds128(sb + kro0), x1 = lds128(sb + kro1);
      const uint32_t w0[4] = {x0.x, x0.y, x0.z, x0.w}, w1[4] = {x1.x, x1.y, x1.z, x1.w};
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        cvt_ab_m(w0[jj], FULL, FULL, ka[jj][0], ka[jj][2]);
        cvt_ab_m(w1[jj], FULL, FULL, ka[jj][1], ka[jj][3]);
      }
      const uint2 kr0 = lds64(sb + kso0), kr1 = lds64(sb + kso1);
      const uint32_t sw0 = lds16(sb + ksw0), sw1 = lds16(sb + ksw1);
      const uint32_t r0[2] = {kr0.x, kr0.y}, r1[2] = {kr1.x, kr1.y};
      float sacc[2][32];
#pragma unroll
      for (int rd = 0; rd < 2; ++rd) {
        wg_fence();
#pragma unroll
        for (int q2 = 0; q2 < 2; ++q2) {
          const int jj = 2 * rd + q2;
          wg_n64(sacc[q2], ka[jj], qd0 + ((jj * NQ * 32) >> 4));
        }
        wg_commit();
        wg_wait_all();
        wg_wait0(sacc[0]);
        wg_wait0(sacc[1]);
        pin_regs(ka[2 * rd]);
        pin_regs(ka[2 * rd + 1]);
#pragma unroll
        for (int q2 = 0; q2 < 2; ++q2) {
          const int jj = 2 * rd + q2;
          const float2 s01 = h2f2(r0[jj >> 1]), s11 = h2f2(r1[jj >> 1]);
          const float sc0 = (jj & 1) ? s01.y : s01.x, sc1 = (jj & 1) ? s11.y : s11.x;
          const uint32_t m0 = (sw0 >> 8) & 3u, l0 = (sw0 >> (2 * jj)) & 3u, m1 = (sw1 >> 8) & 3u, l1 = (sw1 >> (2 * jj)) & 3u;
          const uint32_t v0 = m0 < 2 ? 3 * m0 + l0 : 4 + m0, v1 = m1 < 2 ? 3 * m1 + l1 : 4 + m1;
          float a0[8], a1[8];
#pragma unroll
          for (int v = 0; v < 8; ++v) {
            a0[v] = fmaf(sacc[q2][4 * v + 1], LO_INV, sacc[q2][4 * v + 0]);
            a1[v] = fmaf(sacc[q2][4 * v + 3], LO_INV, sacc[q2][4 * v + 2]);
          }
          const bool b00 = v0 & 1, b01 = v0 & 2, b02 = v0 & 4, b10 = v1 & 1, b11 = v1 & 2, b12 = v1 & 4;
          const float x0 = b02 ? (b01 ? (b00 ? a0[7] : a0[6]) : (b00 ? a0[5] : a0[4]))
                               : (b01 ? (b00 ? a0[3] : a0[2]) : (b00 ? a0[1] : a0[0]));
          const float x1 = b12 ? (b11 ? (b10 ? a1[7] : a1[6]) : (b10 ? a1[5] : a1[4]))
                               : (b11 ? (b10 ? a1[3] : a1[2]) : (b10 ? a1[1] : a1[0]));
          S0 = fmaf(sc0, x0, S0);
          S1 = fmaf(sc1, x1, S1);
        }
      }
    }
    const bool val0 = t0 + tau0 < c1 && t0 + tau0 >= p.kv_start, val1 = t0 + tau1 < c1 && t0 + tau1 >= p.kv_start;  // kv_start: masked leading tokens (sinks)
    S0 = val0 ? S0 * sfac : NEG_INF;
    S1 = val1 ? S1 * sfac : NEG_INF;
    uint32_t svraw[4];
#pragma unroll
    for (int g = 0; g < 4; ++g) svraw[g] = lds32(sb + vso + g * P * 2);

    // ---------------- softmax, P'
    const uint32_t ptb = pt + (n & 1) * C::PT_BYTES;
    float p0, p1;
    bool pflag = false;
    const uint32_t vmask2 = (val0 ? 0xFFFFu : 0u) | (val1 ? 0xFFFF0000u : 0u);
    auto make_p = [&]() {
      p0 = val0 ? ex2(S0 - mref) : 0.f;
      p1 = val1 ? ex2(S1 - mref) : 0.f;
      pflag = make_pprime(p0, p1, svraw, inv2, vmask2, ptb + pwo);
    };
    bool flag = (S0 > mref + TAU) || (S1 > mref + TAU);
    make_p();
    flag = flag || pflag;
    fence_async_smem();
    const int anyflag = __syncthreads_or(flag);  // P' and the class table visible; iteration n-1 finished by all
    if (anyflag) {
      float bm = fmaxf(S0, S1);
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) bm = fmaxf(bm, __shfl_xor_sync(FULL, bm, o));
      if (gid == 0) red_m[w][tig] = bm;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const int e0 = static_cast<int>((svraw[g] >> 10) & 31) - 15, e1 = static_cast<int>((svraw[g] >> 26) & 31) - 15;
        int be = max(val0 ? e0 : EREF_NONE, val1 ? e1 : EREF_NONE);
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) be = max(be, __shfl_xor_sync(FULL, be, o));
        if (lane == 0) red_e[w][g] = be;
      }
      __syncthreads();
      float nm = mref;
#pragma unroll
      for (int ww = 0; ww < NWARP; ++ww) nm = fmaxf(nm, red_m[ww][tig]);
      const float alpha = mref == NEG_INF ? 0.f : ex2(mref - nm);
      float bg[4];
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        int ne = eref[g];
#pragma unroll
        for (int ww = 0; ww < NWARP; ++ww) ne = max(ne, red_e[ww][g]);
        ne = (ne == EREF_NONE) ? ne : min(max(ne, -14), 14);
        bg[g] = (eref[g] == EREF_NONE || eref[g] - ne < -126) ? 0.f : __uint_as_float(static_cast<uint32_t>(127 + eref[g] - ne) << 23);
        eref[g] = ne;
        const uint32_t hv = ne == EREF_NONE ? 0u : static_cast<uint32_t>(15 - ne) << 10;
        inv2[g] = hv | (hv << 16);
      }
      mref = nm;
#pragma unroll
      for (int k = 0; k < 4; ++k)
#pragma unroll
        for (int g = 0; g < 4; ++g) { O[k][g][0] *= alpha * bg[g]; O[k][g][1] *= alpha * bg[g]; }
      lsum *= alpha;
      __syncthreads();
      make_p();
      fence_async_smem();
      __syncthreads();
    }
    lsum += p0 + p1;
    if (w < NPB && lane == 0 && n + ST - 1 < nblk) issue(n + ST - 1);  // stage (n-1) % ST: free (barrier above)

    // ---------------- PV(n): four rounds of class pairs, software-pipelined two deep (double-buffered A fragments:
    // the masked decode of round k+1 overlaps the wgmmas of round k; wait_group 1 retires the older round)
    const uint64_t pdb = pdesc0 + (((n & 1) * C::PT_BYTES) >> 4);
    const uint32_t tbn = tbl + (n & 1) * C::TBL_BYTES;
    uint32_t va[2][4][2][4];
    float pa[2][4][4];
    auto decode_round = [&](int k, uint32_t (&vr)[4][2][4]) {
      const int cls = 2 * k + csel;
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const uint2 elo = lds64(sb + vo[ks][0] + 0), elo2 = lds64(sb + vo[ks][0] + 8);
        const uint2 ehi = lds64(sb + vo[ks][1] + 0), ehi2 = lds64(sb + vo[ks][1] + 8);
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          const uint2 t2 = lds64(tbn + ((g * 8 + 4 * ks + tig) * 2) * 4);
          const uint32_t me = sign_bytes(t2.x << (7 - cls)), mo = sign_bytes(t2.y << (7 - cls));
          const uint2 e = (g & 1) ? ((g >> 1) ? ehi2 : ehi) : ((g >> 1) ? elo2 : elo);
          cvt_ab_m(e.x, me, mo, vr[g][ks][0], vr[g][ks][2]);
          cvt_ab_m(e.y, me, mo, vr[g][ks][1], vr[g][ks][3]);
        }
      }
    };
    auto issue_round = [&](uint32_t (&vr)[4][2][4], float (&pr)[4][4]) {
      wg_fence();
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        wg_n8<0>(pr[g], vr[g][0], pdb + (((0 * 2 + (g >> 1)) * PTILE + (g & 1) * PSBO) >> 4));
        wg_n8<1>(pr[g], vr[g][1], pdb + (((1 * 2 + (g >> 1)) * PTILE + (g & 1) * PSBO) >> 4));
      }
      wg_commit();
    };
    auto retire_round = [&](int k, uint32_t (&vr)[4][2][4], float (&pr)[4][4]) {
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        wg_wait0(pr[g]);
        pin_regs(vr[g][0]);
        pin_regs(vr[g][1]);
        O[k][g][0] += fmaf(pr[g][1], LO_INV, pr[g][0]);
        O[k][g][1] += fmaf(pr[g][3], LO_INV, pr[g][2]);
      }
    };
    if constexpr (PIPE) {
      decode_round(0, va[0]);
      issue_round(va[0], pa[0]);
      decode_round(1, va[1]);
      issue_round(va[1], pa[1]);
      wg_wait_1();
      retire_round(0, va[0], pa[0]);
      decode_round(2, va[0]);
      issue_round(va[0], pa[0]);
      wg_wait_1();
      retire_round(1, va[1], pa[1]);
      decode_round(3, va[1]);
      issue_round(va[1], pa[1]);
      wg_wait_1();
      retire_round(2, va[0], pa[0]);
      wg_wait_all();
      retire_round(3, va[1], pa[1]);
    } else {
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        decode_round(k, va[0]);
        issue_round(va[0], pa[0]);
        wg_wait_all();
        retire_round(k, va[0], pa[0]);
      }
    }
  }

  // ---------------------------------------------------------------- inverse transforms, CTA totals, merge
  float ofac[4];
#pragma unroll
  for (int g = 0; g < 4; ++g) ofac[g] = FV * (eref[g] == EREF_NONE ? 0.f : exp2f(static_cast<float>(eref[g])));
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) lsum += __shfl_xor_sync(FULL, lsum, o);
  __syncthreads();
  float* fin = reinterpret_cast<float*>(smem);  // [8 classes][G][D]
#pragma unroll
  for (int k = 0; k < 4; ++k)
#pragma unroll
    for (int g = 0; g < 4; ++g)
#pragma unroll
      for (int r = 0; r < 2; ++r) fin[((2 * k + csel) * G + tig) * D + 32 * g + 16 * hh + gid + 8 * r] = O[k][g][r] * ofac[g];
  if (gid == 0) { red_l[w][tig] = lsum; red_m[w][tig] = mref; }
  __syncthreads();
  float* outf = fin + 8 * G * D;
  float* pbuf = outf + 2 * G * D;  // [G][D] scatter buffer of the P classes (beyond the merge weights)
  {
    unsigned sg[4] = {p.sg[0][0], p.sg[0][1], p.sg[0][2], p.sg[0][3]};
    float x[8][4];
#pragma unroll
    for (int c = 0; c < 8; ++c)
#pragma unroll
      for (int i = 0; i < 4; ++i) x[c][i] = fin[(c * G + w) * D + 4 * lane + i];
    fwht_q<32>(x[1], lane);
    fwht_q<32>(x[2], lane);
    sflip4(x[2], sg, lane);
    fwht_q<32>(x[4], lane);
    fwht_q<32>(x[5], lane);
    sflip4(x[5], sg, lane);
    fwht_q<128>(x[6], lane);
    fwht_q<128>(x[7], lane);
    sflip4(x[7], sg, lane);
    // P classes are in x[PERM] coordinates: scatter z_a -> x_permi(a)
#pragma unroll
    for (int i = 0; i < 4; ++i) pbuf[w * D + permi(4 * lane + i)] = x[3][i] + x[4][i] + x[5][i];
    __syncwarp();
#pragma unroll
    for (int i = 0; i < 4; ++i)
      outf[w * D + 4 * lane + i] = x[0][i] + x[1][i] + x[2][i] + pbuf[w * D + 4 * lane + i] + x[6][i] + x[7][i];
  }
  __syncthreads();
  float Mc[G], Lc[G];
#pragma unroll
  for (int h = 0; h < G; ++h) {
    Mc[h] = red_m[0][h];
    Lc[h] = red_l[0][h] + red_l[1][h] + red_l[2][h] + red_l[3][h];
  }
  merge_store(p, outf, red_l, b, kvh, split, ns_b + p.tail_nslot, Mc, Lc,
              &last_flag);
}

// ================================================================================================= final kernel
// The locked codebook (cq+f+S+Q5+FS+R5): 15 query variants / value classes, index v = head mode m ? 7 + m : vector
// choice l (0 I, 1 H32, 2..7 H32 S_k, 8 H128, 9..14 H128 S_k).  QK: four rounds of one group (wgmma N = 120: 15
// variants x 4 heads x hi/lo); PV: eight rounds k of four tiles (group g, classes 2k / 2k+1 on warps 0-1 / 2-3; class
// 15 is empty), class masks from a per-block one-hot table (two byte planes); the class accumulators are inverted two
// at a time at the end.  PIPE as k_cbf.
template <int CB, int P, int ST, int PIPE>
__global__ void __launch_bounds__(128, PIPE ? 1 : 2) k_cbq(const CbAttnParams p) {
  using C = CfgA<CB, P, ST>;
  using L = LayA<CB, P>;
  using SP = CbSpec<CB>;
  constexpr int NV = SP::V;
  constexpr int NR = (NV + 1) / 2;  // PV rounds (class pairs)
  constexpr int NLEV = NV > 16 ? 5 : 4;  // select-tree levels
  constexpr int NQ = C::NQ, CHUNK = L::CHUNK, NPB = L::NPB;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ __align__(8) uint64_t mbar[ST];
  __shared__ float red_m[NWARP][G], red_l[NWARP][G], sfac_s[G];
  __shared__ int red_e[NWARP][4];
  __shared__ int last_flag;
  __shared__ float qstage[G * D];

  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, gid = lane >> 2, tig = lane & 3;
  const int kvh = blockIdx.x, split = blockIdx.y, b = blockIdx.z;
  const int len = p.seq_lens[b];
  const int c0 = split * p.chunk, c1 = min(len, c0 + p.chunk);
  const int ns_b = max(1, (len + p.chunk - 1) / p.chunk);
  if (split >= ns_b || len <= 0) return;  // (len 0: the staging-window kernel writes the output)
  const uint32_t sbase = smem_u32(smem);
  const uint32_t qt = sbase + C::QT_OFF, pt = sbase + C::PT_OFF, tbl = sbase + C::TBL_OFF;

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < ST; ++s) mbar_init(smem_u32(&mbar[s]), NPB);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  for (int i = tid; i < (C::QT_BYTES + 2 * C::PT_BYTES) / 16; i += 128)
    reinterpret_cast<uint4*>(smem + C::QT_OFF)[i] = make_uint4(0, 0, 0, 0);
  __syncthreads();
  const int ntok = max(c1 - c0, 0), nblk = (ntok + BLK - 1) / BLK;
  {
    const int* bt = p.block_table + static_cast<size_t>(b) * p.bt_stride + c0 / P;
    int* pidx = reinterpret_cast<int*>(smem + C::PIDX_OFF);
    const int npg = (ntok + P - 1) / P;
    for (int i = tid; i < npg; i += 128) pidx[i] = __ldg(bt + i);
  }
  const int* pidx_s = reinterpret_cast<const int*>(smem + C::PIDX_OFF);
  const uint8_t* poolh = p.pool + static_cast<size_t>(kvh) * CHUNK;
  const uint32_t pstride = static_cast<uint32_t>(p.Hkv) * CHUNK;
  const uint32_t mbar0 = smem_u32(&mbar[0]);
  const uint32_t dst_w = sbase + C::RING_OFF + w * CHUNK;
  auto issue = [&](int nb) {
    const int s = nb % ST;
    const uint32_t mb = mbar0 + 8 * s;
    const bool has = nb * BLK + w * P < ntok;
    mbar_expect_tx(mb, has ? CHUNK : 0);
    if (has) {
      const uint32_t page = static_cast<uint32_t>(pidx_s[nb * NPB + w]);
      bulk_g2s(dst_w + s * L::STAGE, poolh + static_cast<uint64_t>(page) * pstride, CHUNK, mb);
    }
  };
  __syncthreads();
  if (w < NPB && lane == 0) {
#pragma unroll
    for (int s = 0; s < ST - 1; ++s)
      if (s < nblk) issue(s);
  }
  q_prologue<CB, NQ>(p, b, kvh, w, lane, qt, qstage, sfac_s);
  fence_async_smem();
  __syncthreads();
  const float sfac = sfac_s[tig];

  const int bq = gid & 1, ipq = (gid >> 1) & 1, qbq = gid >> 2;
  const int tau0 = 16 * w + 8 * qbq + 4 * ipq + bq, tau1 = tau0 + 2;
  const uint32_t kro0 = opaque((tau0 / P) * CHUNK + L::K_OFF + (tau0 % P) * 64 + 16 * tig);
  const uint32_t kro1 = opaque((tau1 / P) * CHUNK + L::K_OFF + (tau1 % P) * 64 + 16 * tig);
  const uint32_t kso0 = opaque((tau0 / P) * CHUNK + L::KS_OFF + (tau0 % P) * 8);
  const uint32_t kso1 = opaque((tau1 / P) * CHUNK + L::KS_OFF + (tau1 % P) * 8);
  const uint32_t ksw0 = opaque((tau0 / P) * CHUNK + L::SWK_OFF + (tau0 % P) * 2);
  const uint32_t ksw1 = opaque((tau1 / P) * CHUNK + L::SWK_OFF + (tau1 % P) * 2);
  const uint64_t qd0 = opaque64(sdesc(qt));
  const uint64_t pdesc0 = opaque64(sdesc_p(pt));
  const int hh = w & 1;
  uint32_t vo[2][2];
#pragma unroll
  for (int ks = 0; ks < 2; ++ks) {
    const int tb = 32 * ks + 8 * tig;
    const int c8 = (tb % P) / 8;
#pragma unroll
    for (int e = 0; e < 2; ++e)
      vo[ks][e] = opaque((tb / P) * CHUNK + L::V_OFF + ((((2 * e + hh) * 8 + gid) * (P / 8) + c8) << 4));
  }
  // V side word -> one-hot table conversion: thread t < 64 handles block token t
  const uint32_t vsw_t = opaque(((tid & 63) / P) * CHUNK + L::SWV_OFF + ((tid & 63) % P) * 2);
  const uint32_t tbl_w = opaque((((tid & 63) / 8) * 2 + (tid & 1)) * 4 + ((tid & 7) >> 1));  // + g * 64 + plane * 256
  const int cp0 = (tau0 % P) % 8;
  const uint32_t vso = opaque((tau0 / P) * CHUNK + L::VS_OFF + (((tau0 % P) / 8) * 8 + ((cp0 & 1) ? 4 + (cp0 >> 1) : (cp0 >> 1))) * 2);
  const int k0p = 16 * bq + 4 * (2 * (w & 1) + qbq) + 2 * ipq;
  const uint32_t pwo = opaque(((w >> 1) * 2) * PTILE + (k0p / 16) * PLBO + (2 * tig) * 16 + (k0p % 16));
  const int csel = w >> 1;  // class of this warp in pair tile k: 2 k + csel

  // class accumulators [round k][group g][r]: class 2k + csel, d = 32 g + 16 hh + gid + 8 r (registers, or shared
  // memory [k][g][r][thread] when C::OSM)
  float Oreg[C::OSM ? 1 : NR][4][2];
  float* Osm = reinterpret_cast<float*>(smem + C::OSM_OFF);
  auto Oacc = [&](int k, int g, int r) -> float& {
    if constexpr (C::OSM) return Osm[((k * 4 + g) * 2 + r) * 128 + tid];
    else return Oreg[k][g][r];
  };
#pragma unroll
  for (int k = 0; k < NR; ++k)
#pragma unroll
    for (int g = 0; g < 4; ++g) Oacc(k, g, 0) = Oacc(k, g, 1) = 0.f;
  float lsum = 0.f, mref = NEG_INF;
  int eref[4];
  uint32_t inv2[4];
#pragma unroll
  for (int g = 0; g < 4; ++g) { eref[g] = EREF_NONE; inv2[g] = 0u; }

  for (int n = 0; n < nblk; ++n) {
    const int s = n % ST;
    const uint32_t sb = sbase + C::RING_OFF + s * L::STAGE;
    const int t0 = c0 + n * BLK;
    mbar_wait(smem_u32(&mbar[s]), (n / ST) & 1);
    // ---------------- V class table of this block (published by the barrier below)
    if (tid < 64) {
      const uint32_t sw = lds16(sb + vsw_t);
      const uint32_t m = (sw >> (4 * SP::LB)) & ((1u << SP::HB) - 1u);
      const uint32_t tb0 = tbl + (n & 1) * C::TBL_BYTES + tbl_w;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const uint32_t cls = m ? SP::NLH + m : (sw >> (SP::LB * g)) & ((1u << SP::LB) - 1u);
        const uint32_t bit = 1u << (cls & 7);
#pragma unroll
        for (int pl = 0; pl < C::TBL_BYTES / 256; ++pl)
          asm volatile("st.shared.u8 [%0], %1;\n" ::"r"(tb0 + 256 * pl + g * 64), "r"((cls >> 3) == pl ? bit : 0u));
      }
    }
    // ---------------- QK(n) in four rounds (one group each), variant select
    float S0 = 0.f, S1 = 0.f;
    {
      uint32_t ka[4][4];
      const uint4 x0 = lds128(sb + kro0), x1 = lds128(sb + kro1);
      const uint32_t w0[4] = {x0.x, x0.y, x0.z, x0.w}, w1[4] = {x1.x, x1.y, x1.z, x1.w};
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        cvt_ab_m(w0[jj], FULL, FULL, ka[jj][0], ka[jj][2]);
        cvt_ab_m(w1[jj], FULL, FULL, ka[jj][1], ka[jj][3]);
      }
      const uint2 kr0 = lds64(sb + kso0), kr1 = lds64(sb + kso1);
      const uint32_t sw0 = lds16(sb + ksw0), sw1 = lds16(sb + ksw1);
      const uint32_t r0[2] = {kr0.x, kr0.y}, r1[2] = {kr1.x, kr1.y};
      const uint32_t m0 = (sw0 >> (4 * SP::LB)) & ((1u << SP::HB) - 1u), m1 = (sw1 >> (4 * SP::LB)) & ((1u << SP::HB) - 1u);
      float sacc[NQ / 2];
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        wg_fence();
        wg_qk<NQ>(sacc, ka[jj], qd0 + ((jj * NQ * 32) >> 4));
        wg_commit();
        wg_wait_all();
        wg_wait0(sacc);
        pin_regs(ka[jj]);
        const float2 s01 = h2f2(r0[jj >> 1]), s11 = h2f2(r1[jj >> 1]);
        const float sc0 = (jj & 1) ? s01.y : s01.x, sc1 = (jj & 1) ? s11.y : s11.x;
        const uint32_t lm = (1u << SP::LB) - 1u;
        const uint32_t v0 = m0 ? SP::NLH + m0 : (sw0 >> (SP::LB * jj)) & lm, v1 = m1 ? SP::NLH + m1 : (sw1 >> (SP::LB * jj)) & lm;
        // 4-level select tree on the variant index, on the raw hi / lo accumulators (no temporaries: the 60
        // accumulators and the 64 class accumulators already fill the register file); leaf 15 never selected
        auto pick = [&](uint32_t v, int off) {  // off: 0 / 1 (row gid: hi / lo), 2 / 3 (row gid + 8)
          constexpr int NL0 = 1 << (NLEV - 1);
          float t[NL0];
#pragma unroll
          for (int i = 0; i < NL0; ++i)
            t[i] = (v & 1u) ? sacc[4 * min(2 * i + 1, NV - 1) + off] : sacc[4 * min(2 * i, NV - 1) + off];
          static_for<NLEV - 1>([&](auto lc) {  // level lev halves the candidates (compile-time sizes: no local memory)
            constexpr int lev = decltype(lc)::value + 1;
#pragma unroll
            for (int i = 0; i < (NL0 >> lev); ++i) t[i] = ((v >> lev) & 1u) ? t[2 * i + 1] : t[2 * i];
          });
          return t[0];
        };
        S0 = fmaf(sc0, fmaf(pick(v0, 1), LO_INV, pick(v0, 0)), S0);
        S1 = fmaf(sc1, fmaf(pick(v1, 3), LO_INV, pick(v1, 2)), S1);
      }
    }
    const bool val0 = t0 + tau0 < c1 && t0 + tau0 >= p.kv_start, val1 = t0 + tau1 < c1 && t0 + tau1 >= p.kv_start;  // kv_start: masked leading tokens (sinks)
    S0 = val0 ? S0 * sfac : NEG_INF;
    S1 = val1 ? S1 * sfac : NEG_INF;
    uint32_t svraw[4];
#pragma unroll
    for (int g = 0; g < 4; ++g) svraw[g] = lds32(sb + vso + g * P * 2);

    // ---------------- softmax, P'
    const uint32_t ptb = pt + (n & 1) * C::PT_BYTES;
    float p0, p1;
    bool pflag = false;
    const uint32_t vmask2 = (val0 ? 0xFFFFu : 0u) | (val1 ? 0xFFFF0000u : 0u);
    auto make_p = [&]() {
      p0 = val0 ? ex2(S0 - mref) : 0.f;
      p1 = val1 ? ex2(S1 - mref) : 0.f;
      pflag = make_pprime(p0, p1, svraw, inv2, vmask2, ptb + pwo);
    };
    bool flag = (S0 > mref + TAU) || (S1 > mref + TAU);
    make_p();
    flag = flag || pflag;
    fence_async_smem();
    const int anyflag = __syncthreads_or(flag);  // P' and the class table visible; iteration n-1 finished by all
    if (anyflag) {
      float bm = fmaxf(S0, S1);
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) bm = fmaxf(bm, __shfl_xor_sync(FULL, bm, o));
      if (gid == 0) red_m[w][tig] = bm;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const int e0 = static_cast<int>((svraw[g] >> 10) & 31) - 15, e1 = static_cast<int>((svraw[g] >> 26) & 31) - 15;
        int be = max(val0 ? e0 : EREF_NONE, val1 ? e1 : EREF_NONE);
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) be = max(be, __shfl_xor_sync(FULL, be, o));
        if (lane == 0) red_e[w][g] = be;
      }
      __syncthreads();
      float nm = mref;
#pragma unroll
      for (int ww = 0; ww < NWARP; ++ww) nm = fmaxf(nm, red_m[ww][tig]);
      const float alpha = mref == NEG_INF ? 0.f : ex2(mref - nm);
      float bg[4];
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        int ne = eref[g];
#pragma unroll
        for (int ww = 0; ww < NWARP; ++ww) ne = max(ne, red_e[ww][g]);
        ne = (ne == EREF_NONE) ? ne : min(max(ne, -14), 14);
        bg[g] = (eref[g] == EREF_NONE || eref[g] - ne < -126) ? 0.f : __uint_as_float(static_cast<uint32_t>(127 + eref[g] - ne) << 23);
        eref[g] = ne;
        const uint32_t hv = ne == EREF_NONE ? 0u : static_cast<uint32_t>(15 - ne) << 10;
        inv2[g] = hv | (hv << 16);
      }
      mref = nm;
#pragma unroll
      for (int k = 0; k < NR; ++k)
#pragma unroll
        for (int g = 0; g < 4; ++g) { Oacc(k, g, 0) *= alpha * bg[g]; Oacc(k, g, 1) *= alpha * bg[g]; }
      lsum *= alpha;
      __syncthreads();
      make_p();
      fence_async_smem();
      __syncthreads();
    }
    lsum += p0 + p1;
    if (w < NPB && lane == 0 && n + ST - 1 < nblk) issue(n + ST - 1);  // stage (n-1) % ST: free (barrier above)

    // ---------------- PV(n): eight rounds of class pairs, software-pipelined two deep (double-buffered A fragments:
    // the masked decode of round k+1 overlaps the wgmmas of round k; wait_group 1 retires the older round)
    const uint64_t pdb = pdesc0 + (((n & 1) * C::PT_BYTES) >> 4);
    const uint32_t tbn = tbl + (n & 1) * C::TBL_BYTES;
    uint32_t va[2][4][2][4];
    float pa[2][4][4];
    auto decode_round = [&](int k, uint32_t (&vr)[4][2][4]) {
      const int cls = 2 * k + csel;
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const uint2 elo = lds64(sb + vo[ks][0] + 0), elo2 = lds64(sb + vo[ks][0] + 8);
        const uint2 ehi = lds64(sb + vo[ks][1] + 0), ehi2 = lds64(sb + vo[ks][1] + 8);
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          const uint2 t2 = lds64(tbn + min(cls >> 3, C::TBL_BYTES / 256 - 1) * 256 + ((g * 8 + 4 * ks + tig) * 2) * 4);
          const uint32_t me = sign_bytes(t2.x << (7 - (cls & 7))), mo = sign_bytes(t2.y << (7 - (cls & 7)));
          const uint2 e = (g & 1) ? ((g >> 1) ? ehi2 : ehi) : ((g >> 1) ? elo2 : elo);
          cvt_ab_m(e.x, me, mo, vr[g][ks][0], vr[g][ks][2]);
          cvt_ab_m(e.y, me, mo, vr[g][ks][1], vr[g][ks][3]);
        }
      }
    };
    auto issue_round = [&](uint32_t (&vr)[4][2][4], float (&pr)[4][4]) {
      wg_fence();
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        wg_n8<0>(pr[g], vr[g][0], pdb + (((0 * 2 + (g >> 1)) * PTILE + (g & 1) * PSBO) >> 4));
        wg_n8<1>(pr[g], vr[g][1], pdb + (((1 * 2 + (g >> 1)) * PTILE + (g & 1) * PSBO) >> 4));
      }
      wg_commit();
    };
    auto retire_round = [&](int k, uint32_t (&vr)[4][2][4], float (&pr)[4][4]) {
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        wg_wait0(pr[g]);
        pin_regs(vr[g][0]);
        pin_regs(vr[g][1]);
        Oacc(k, g, 0) += fmaf(pr[g][1], LO_INV, pr[g][0]);
        Oacc(k, g, 1) += fmaf(pr[g][3], LO_INV, pr[g][2]);
      }
    };
    if constexpr (PIPE) {
      decode_round(0, va[0]);
      issue_round(va[0], pa[0]);
      static_for<NR - 1>([&](auto kc) {
        constexpr int k = decltype(kc)::value + 1;
        decode_round(k, va[k & 1]);
        issue_round(va[k & 1], pa[k & 1]);
        wg_wait_1();
        retire_round(k - 1, va[(k - 1) & 1], pa[(k - 1) & 1]);
      });
      wg_wait_all();
      retire_round(NR - 1, va[(NR - 1) & 1], pa[(NR - 1) & 1]);
    } else {
      static_for<NR>([&](auto kc) {
        constexpr int k = decltype(kc)::value;
        decode_round(k, va[0]);
        issue_round(va[0], pa[0]);
        wg_wait_all();
        retire_round(k, va[0], pa[0]);
      });
    }
  }

  // ---------------------------------------------------------------- inverse transforms, CTA totals, merge
  float ofac[4];
#pragma unroll
  for (int g = 0; g < 4; ++g) ofac[g] = FV * (eref[g] == EREF_NONE ? 0.f : exp2f(static_cast<float>(eref[g])));
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) lsum += __shfl_xor_sync(FULL, lsum, o);
  if (gid == 0) { red_l[w][tig] = lsum; red_m[w][tig] = mref; }
  float* fin = reinterpret_cast<float*>(smem);  // [2 classes][G][D] per round
  float* outf = fin + 2 * G * D;                // [G][D] + merge weights
  float o4[4] = {0.f, 0.f, 0.f, 0.f};           // warp w: head w, d = 4 lane .. +3
  static_for<NR>([&](auto kc) {
    constexpr int k = decltype(kc)::value;
    __syncthreads();  // (k = 0: ring no longer used; else: previous classes consumed)
#pragma unroll
    for (int g = 0; g < 4; ++g)
#pragma unroll
      for (int r = 0; r < 2; ++r) fin[(csel * G + tig) * D + 32 * g + 16 * hh + gid + 8 * r] = Oacc(k, g, r) * ofac[g];
    __syncthreads();
#pragma unroll
    for (int cc = 0; cc < 2; ++cc) {
      const int c = 2 * k + cc;  // compile-time after unrolling
      if (c >= NV) continue;
      float x[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) x[i] = fin[(cc * G + w) * D + 4 * lane + i];
      if (c >= 1 && c <= SP::NLH) fwht_q<32>(x, lane);
      if (c > SP::NLH) fwht_q<128>(x, lane);
      const int si = c <= SP::NLH ? c - 2 : c - SP::NLH - 2;  // sign table entry (< 0: none)
      if (si >= 0) sflip4(x, p.sg[si >= 0 ? si : 0], lane);
#pragma unroll
      for (int i = 0; i < 4; ++i) o4[i] += x[i];
    }
  });
#pragma unroll
  for (int i = 0; i < 4; ++i) outf[w * D + 4 * lane + i] = o4[i];
  __syncthreads();
  float Mc[G], Lc[G];
#pragma unroll
  for (int h = 0; h < G; ++h) {
    Mc[h] = red_m[0][h];
    Lc[h] = red_l[0][h] + red_l[1][h] + red_l[2][h] + red_l[3][h];
  }
  merge_store(p, outf, red_l, b, kvh, split, ns_b + p.tail_nslot, Mc, Lc,
              &last_flag);
}


// ================================================================================================= key-only kernel
// cq+f+konly: only the keys carry the per-token choice (vector j: I / H32 = sign bit of its K scale; head: F (H128) =
// sign bit of the token's group-0 V scale), values are plain INT4.  This is the v2 INT4 kernel (variant 0: software
// pipeline, FP8 PV with the V scale in P') with a wider QK: per 32-d group (fresh accumulators as in v2)
//   wgmma m64n16k32: columns [q_hi(I), q_hi(H32)] x 4 heads | [q_hi(H128), 16 q_lo(H128)] x 4 heads, then
//   wgmma m64n8k32 accumulating [q_lo(I), q_lo(H32)] x 4 heads onto the first 8 columns (unscaled lo: its e4m3
//   underflow below 2^-9 is < 2^-16 of the largest |q| element; the chained k-step adds one FP8 accumulation step)
// = the tensor work of N = 24 with 8 instead of 12 accumulator registers per group, and a branch-free select per
// (token, group): x = F ? hi_F + lo_F / 16 : (H ? acc_H : acc_I).  Everything else (V fragments, P', PV, barriers,
// split merge) is v2.  The F bits ride in the V scale word that P' reads anyway (no extra shared-memory load).
#ifndef CBK_MINB
#define CBK_MINB 1
#endif
#ifndef CBK_PREFETCH
#define CBK_PREFETCH 1
#endif
#ifndef CBK_SKIP
#define CBK_SKIP 0
#endif
// CBK_ABL = 1: timing-only ablation (wrong results): no H128 columns in QK (wgmma m64n8k32 hi + chained lo of
// I / H32 only, N = 16 instead of 24) and no F select -- the upper bound of what head-bit segmented pages could save on
// the QK side (they still need the H128 variant for the back segment)
#ifndef CBK_ABL
#define CBK_ABL 0
#endif
// DEF = 1 (variant 1): the deferred PV wait of the static v2 kernel (variant 264): PV(n) runs under
// the scores / softmax / P' of block n+1, QK(n+1) under the softmax barrier, fragments produced with no wgmma in flight
// CB = CB_KONLYA (asymmetric INT4): unsigned codes; keys: x - z Q_v(g) per (token, group) with the Q of the
// selected variant (I / H32 / H128); values: variant 0 = per-thread sums of P' z (P' as the tensor core sees it) on the
// CUDA cores, variant 1 = Z = P'^T z on the tensor cores (as attn_decode_v2.cu INT4A: A = the P' tiles, B = a 4 x 32
// zero-point tile per k-step built by 16 lanes of warp 3, two m64n8k32 per block in PV's commit group); subtracted from
// the group's 32 outputs once per CTA
template <int CB, int P, int ST, int DEF>
__global__ void __launch_bounds__(128, CBK_MINB) k_cbk(const CbAttnParams p) {
  constexpr bool ASYM = cb_asym(CB);
  // value zero points: variant 0 (DEF = 0) per-thread sums of P' z on the CUDA cores, variant 1 (DEF = 1) Z = P'^T z on
  // the tensor cores (in variant 0 the extra wgmmas get serialized by ptxas, C7515)
  constexpr bool ZVT = ASYM && DEF, ZVS = ASYM && !DEF;
  using C = CfgA<CB, P, ST>;
  using L = LayA<CB, P>;
  constexpr int NQ = C::NQ, CHUNK = L::CHUNK, NPB = L::NPB;
  __shared__ float qsum_s[3 * G * 4];       // CB_KONLYA: zero-point weights [variant I / H32 / H128][head][group]
  __shared__ float zred[4][G][2];           // CB_KONLYA variant 1: value zero-point sums [group][head][hi / lo]
  __shared__ float zredw[NWARP][G][4];      // CB_KONLYA variant 0: per-warp value zero-point sums [warp][head][group]
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ __align__(8) uint64_t mbar[ST];
  __shared__ float red_m[NWARP][G], red_l[NWARP][G], sfac_s[G];
  __shared__ int red_e[NWARP][4];
  __shared__ int last_flag;

  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, gid = lane >> 2, tig = lane & 3;
  const int kvh = blockIdx.x, split = blockIdx.y, b = blockIdx.z;
  const int len = p.seq_lens[b];
  const int c0 = split * p.chunk, c1 = min(len, c0 + p.chunk);
  const int ns_b = max(1, (len + p.chunk - 1) / p.chunk);
  if (split >= ns_b || len <= 0) return;  // (len 0: the staging-window kernel writes the output)
  const uint32_t sbase = smem_u32(smem);
  const uint32_t qt = sbase + C::QT_OFF, pt = sbase + C::PT_OFF;

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < ST; ++s) mbar_init(smem_u32(&mbar[s]), NPB);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  for (int i = tid; i < (C::QT_BYTES + 2 * C::PT_BYTES) / 16; i += 128)
    reinterpret_cast<uint4*>(smem + C::QT_OFF)[i] = make_uint4(0, 0, 0, 0);
  if constexpr (ZVT)  // zero-point tiles: rows 4..7 (unused columns) stay zero
    for (int i = tid; i < C::ZB_BYTES / 16; i += 128) reinterpret_cast<uint4*>(smem + C::ZB_OFF)[i] = make_uint4(0, 0, 0, 0);
  __syncthreads();
  const int ntok = max(c1 - c0, 0), nblk = (ntok + BLK - 1) / BLK;
  {
    const int* bt = p.block_table + static_cast<size_t>(b) * p.bt_stride + c0 / P;
    int* pidx = reinterpret_cast<int*>(smem + C::PIDX_OFF);
    const int npg = (ntok + P - 1) / P;
    for (int i = tid; i < npg; i += 128) pidx[i] = __ldg(bt + i);
  }
  const int* pidx_s = reinterpret_cast<const int*>(smem + C::PIDX_OFF);
  const uint8_t* poolh = p.pool + static_cast<size_t>(kvh) * CHUNK;
  const uint32_t pstride = static_cast<uint32_t>(p.Hkv) * CHUNK;
  const uint32_t mbar0 = smem_u32(&mbar[0]);
  const uint32_t dst_w = sbase + C::RING_OFF + w * L::PSTR;
  auto issue = [&](int nb) {
    const int s = nb % ST;
    const uint32_t mb = mbar0 + 8 * s;
    const bool has = nb * BLK + w * P < ntok;
    mbar_expect_tx(mb, has ? CHUNK : 0);
    if (has) {
      const uint32_t page = static_cast<uint32_t>(pidx_s[nb * NPB + w]);
      bulk_g2s(dst_w + s * L::STAGE, poolh + static_cast<uint64_t>(page) * pstride, CHUNK, mb);
    }
  };
  __syncthreads();
  if (w < NPB && lane == 0) {
#pragma unroll
    for (int s = 0; s < ST - 1; ++s)
      if (s < nblk) issue(s);
  }
  q_prologue<CB, NQ>(p, b, kvh, w, lane, qt, nullptr, sfac_s, qsum_s);
  fence_async_smem();
  __syncthreads();
  const float sfac = sfac_s[tig];

  // ---------------------------------------------------------------- per-thread constants (v2 INT4, variant 0)
  const int bq = gid & 1, ipq = (gid >> 1) & 1, qbq = gid >> 2;
  const int tau0 = 16 * w + 8 * qbq + 4 * ipq + bq, tau1 = tau0 + 2;
  constexpr bool B2 = cb_b2(CB);
  auto krow = [](int t) { return B2 ? 32 * (t ^ ((t >> 1) & 2)) : 64 * t; };  // (INT2: swizzled K rows)
  const uint32_t kro0 = opaque((tau0 / P) * L::PSTR + L::K_OFF + krow(tau0 % P) + L::KTB * tig);
  const uint32_t kro1 = opaque((tau1 / P) * L::PSTR + L::K_OFF + krow(tau1 % P) + L::KTB * tig);
  auto ldk = [](uint32_t a) -> uint4 {  // a thread's K code bytes of a row (INT2: 8 bytes in x, y)
    if constexpr (B2) { const uint2 u = lds64(a); return make_uint4(u.x, u.y, 0u, 0u); }
    else return lds128(a);
  };
  const uint32_t kso0 = opaque((tau0 / P) * L::PSTR + L::KS_OFF + (tau0 % P) * 8);
  const uint32_t kso1 = opaque((tau1 / P) * L::PSTR + L::KS_OFF + (tau1 % P) * 8);
  const uint64_t qd0 = opaque64(sdesc(qt));
  const uint64_t pdesc0 = opaque64(sdesc_p(pt));
  uint32_t vo[2];  // V^T entries of tokens 32 ks + 8 tig .. +7, rows d % 64 = 16 w + gid (+ 8)
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int tb = 32 * u + 8 * tig;
    vo[u] = opaque((tb / P) * L::PSTR + L::V_OFF + ((w * 8 + gid) * (P / 8) + (tb % P) / 8) * L::VE);
  }
  const int cp0 = (tau0 % P) % 8;  // V scales of tau0 / tau1 (adjacent halves), group g at + g * P * 2
  const uint32_t vso = opaque((tau0 / P) * L::PSTR + L::VS_OFF + (((tau0 % P) / 8) * 8 + ((cp0 & 1) ? 4 + (cp0 >> 1) : (cp0 >> 1))) * 2);
  const int k0p = 16 * bq + 4 * (2 * (w & 1) + qbq) + 2 * ipq;
  const uint32_t pwo = opaque(((w >> 1) * 2) * PTILE + (k0p / 16) * PLBO + (2 * tig) * 16 + (k0p % 16));

  float O[2][2] = {{0.f, 0.f}, {0.f, 0.f}};  // head tig, d = 64 m + 16 w + gid + 8 r
  float lsum = 0.f, mref = NEG_INF;
  // CB_KONLYA value zero points (as attn_decode_v2.cu INT4A): Z accumulator of the block in flight (lanes w < 2,
  // tig == w hold the diagonal: zd[0] = Z(group 2 w, head gid >> 1, hi/lo gid & 1), zd[3] = group 2 w + 1), their CTA
  // sums zs0 / zs1; zero-point tile builder = warp 3, lanes 0..15 (group lane >> 2, k-step (lane >> 1) & 1, k half lane & 1)
  float zd[4] = {0.f, 0.f, 0.f, 0.f};
  float zs0 = 0.f, zs1 = 0.f;
  float zacc[4] = {0.f, 0.f, 0.f, 0.f};  // (variant 0) sum over this thread's tokens of P' z per group (head tig)
  const uint32_t zbsrc = opaque(((32 * ((lane >> 1) & 1)) / P) * L::PSTR + L::VS_OFF + (lane >> 2 & 3) * P * 2 +
                                (((32 * ((lane >> 1) & 1)) % P) / 8) * 16 + 8 * (lane & 1));
  const uint32_t zbdst = sbase + C::ZB_OFF + ((lane >> 1) & 1) * 256 + (lane & 1) * 128 + (lane >> 2 & 3) * 16;
  const uint64_t zbdesc0 = sdesc_zb(sbase + C::ZB_OFF);
  int eref[4];
  uint32_t inv2[4];
#pragma unroll
  for (int g = 0; g < 4; ++g) { eref[g] = EREF_NONE; inv2[g] = 0u; }
  float pacc[2][8];
  float sacc[4][8];
  uint32_t ka[4][4];
  uint2 kraw0, kraw1;  // raw K scales (sign = H32 bit) of rows tau0 / tau1 of the block whose QK is in flight
  uint32_t vs0;        // raw group-0 V scales of (tau0, tau1) of that block (sign = F bit)
  uint32_t va[2][2][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int k = 0; k < 4; ++k) va[i][m][k] = 0u;

  auto qk_issue = [&](int nb, uint4 x0, uint4 x1) {
    const uint32_t sb = sbase + C::RING_OFF + (nb % ST) * L::STAGE;
    const uint32_t w0[4] = {x0.x, x0.y, x0.z, x0.w}, w1[4] = {x1.x, x1.y, x1.z, x1.w};
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      if constexpr (B2) {  // INT2: word jj >> 1 holds groups 2 (jj >> 1) (+0 / +2 bits) and + 1 (+4 / +6)
        const int sh = 4 * (jj & 1);
        ka[jj][0] = (w0[jj >> 1] >> sh) & 0x03030303u;
        ka[jj][2] = (w0[jj >> 1] >> (sh + 2)) & 0x03030303u;
        ka[jj][1] = (w1[jj >> 1] >> sh) & 0x03030303u;
        ka[jj][3] = (w1[jj >> 1] >> (sh + 2)) & 0x03030303u;
      } else if constexpr (ASYM) {  // unsigned nibbles (v2 int4a layout)
        ka[jj][0] = w0[jj] & 0x0F0F0F0Fu;
        ka[jj][2] = (w0[jj] >> 4) & 0x0F0F0F0Fu;
        ka[jj][1] = w1[jj] & 0x0F0F0F0Fu;
        ka[jj][3] = (w1[jj] >> 4) & 0x0F0F0F0Fu;
      } else {
        ka[jj][0] = w0[jj] & 0x87878787u;
        ka[jj][2] = __funnelshift_l(w0[jj], w0[jj], 4) & 0x87878787u;
        ka[jj][1] = w1[jj] & 0x87878787u;
        ka[jj][3] = __funnelshift_l(w1[jj], w1[jj], 4) & 0x87878787u;
      }
    }
    kraw0 = lds64(sb + kso0);
    kraw1 = lds64(sb + kso1);
    vs0 = lds32(sb + vso);
#if CBK_SKIP
    // block-uniform variant flags from the F bits of all 64 tokens (group-0 V scales, 2 tokens per lane; every warp
    // reads the same words -> warpgroup-uniform): QK of the I / H32 columns only if some token is not F, of the H128
    // columns only if some token is F (tokens sorted by head mode at encode time make most blocks pure)
    const uint32_t fw = lds32(sb + (2 * lane / P) * L::PSTR + L::VS_OFF + (2 * lane % P) * 2) & 0x80008000u;
    const bool hasF = __any_sync(FULL, fw != 0u), hasB = __any_sync(FULL, fw != 0x80008000u);
    wg_fence();
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      if (hasB) {
        wg_n8<0>(*reinterpret_cast<float(*)[4]>(sacc[jj]), ka[jj], qd0 + ((jj * NQ * 32) >> 4));
        wg_n8<1>(*reinterpret_cast<float(*)[4]>(sacc[jj]), ka[jj], qd0 + ((jj * NQ * 32 + 2 * 256) >> 4));
      }
      if (hasF) wg_n8<0>(*reinterpret_cast<float(*)[4]>(sacc[jj] + 4), ka[jj], qd0 + ((jj * NQ * 32 + 256) >> 4));
    }
    wg_commit();
#elif CBK_ABL
    wg_fence();
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      wg_n8<0>(*reinterpret_cast<float(*)[4]>(sacc[jj]), ka[jj], qd0 + ((jj * NQ * 32) >> 4));
      wg_n8<1>(*reinterpret_cast<float(*)[4]>(sacc[jj]), ka[jj], qd0 + ((jj * NQ * 32 + 2 * 256) >> 4));
    }
    wg_commit();
#else
    wg_fence();
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      wg_n16<0>(sacc[jj], ka[jj], qd0 + ((jj * NQ * 32) >> 4));
      wg_n8<1>(*reinterpret_cast<float(*)[4]>(sacc[jj]), ka[jj], qd0 + ((jj * NQ * 32 + 2 * 256) >> 4));
    }
    wg_commit();
#endif
  };
  auto pv_accumulate = [&]() {
    if constexpr (ZVT) {  // (zd is reset here, with no wgmma in flight, and accumulated into by both Z wgmmas: its
                           // registers stay live through the pipeline stage, otherwise ptxas serializes, C7515)
      zs0 += zd[0];
      zs1 += zd[3];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        zd[i] = 0.f;
        asm volatile("" : "+f"(zd[i]));  // (the zeros are produced here, not rematerialized inside the stage)
      }
    }
    const bool g1 = (w >> 1) != 0;  // this warp's d rows belong to column group 1
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      O[m][0] += g1 ? fmaf(pacc[m][5], LO_INV, pacc[m][4]) : fmaf(pacc[m][1], LO_INV, pacc[m][0]);
      O[m][1] += g1 ? fmaf(pacc[m][7], LO_INV, pacc[m][6]) : fmaf(pacc[m][3], LO_INV, pacc[m][2]);
    }
  };

  mbar_wait(smem_u32(&mbar[0]), 0);
  qk_issue(0, ldk(sbase + C::RING_OFF + kro0), ldk(sbase + C::RING_OFF + kro1));
  if constexpr (DEF) {
    wg_wait_all();
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) wg_wait0(sacc[jj]);
#pragma unroll
    for (int k = 0; k < 4; ++k) pin_regs(ka[k]);
  }
  for (int n = 0; n < nblk; ++n) {
    const int s = n % ST;
    const uint32_t sb = sbase + C::RING_OFF + s * L::STAGE;
    const int t0 = c0 + n * BLK;
    if (w < NPB && lane == 0 && n + ST - 1 < nblk) issue(n + ST - 1);

    // raw K words of block n+1 (stage requested two blocks ago): loaded while QK(n) / PV(n-1) are in flight, so the
    // shared-memory latency is off the path wait -> QK(n+1) issue (plain registers, not wgmma operands yet)
    const int nq = min(n + 1, nblk - 1);
#if CBK_PREFETCH
    if (nq != n) mbar_wait(smem_u32(&mbar[nq % ST]), (nq / ST) & 1);
    const uint32_t sbq = sbase + C::RING_OFF + (nq % ST) * L::STAGE;
    const uint4 kx0 = ldk(sbq + kro0), kx1 = ldk(sbq + kro1);
#endif

    // ---------------- (a) wait QK(n) + PV(n-1); O += PV(n-1); scores with the per-(token, group) column select
    if constexpr (!DEF) {
      wg_wait_all();
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) wg_wait0(sacc[jj]);
#pragma unroll
      for (int k = 0; k < 4; ++k) pin_regs(ka[k]);
#pragma unroll
      for (int m = 0; m < 2; ++m) wg_wait0(pacc[m]);
      if constexpr (ZVT) wg_wait0(zd);
#pragma unroll
      for (int i = 0; i < 2; ++i) { pin_regs(va[i][0]); pin_regs(va[i][1]); }
      if (n > 0) pv_accumulate();
    }
    const bool val0 = t0 + tau0 < c1 && t0 + tau0 >= p.kv_start, val1 = t0 + tau1 < c1 && t0 + tau1 >= p.kv_start;  // kv_start: masked leading tokens (sinks)
    float S0 = 0.f, S1 = 0.f, S0b = 0.f, S1b = 0.f;
    {
#if CBK_ABL
      constexpr bool F0 = false, F1 = false;
#else
      const bool F0 = (vs0 & 0x8000u) != 0, F1 = (vs0 & 0x80000000u) != 0;
#endif
      const uint32_t r0[2] = {kraw0.x, kraw0.y}, r1[2] = {kraw1.x, kraw1.y};
      float qI[4], qH[4], qF[4], zf0[4], zf1[4];  // CB_KONLYA: zero-point weights of head tig, key zero points (z 2^-24)
      if constexpr (ASYM) {
#if !defined(CB_NO_ZP) && !defined(CB_NO_KZP)  // negative control (wrong results): key zero points ignored
        const float4 a = *reinterpret_cast<const float4*>(&qsum_s[(0 * G + tig) * 4]);
        const float4 h = *reinterpret_cast<const float4*>(&qsum_s[(1 * G + tig) * 4]);
        const float4 f = *reinterpret_cast<const float4*>(&qsum_s[(2 * G + tig) * 4]);
#else
        const float4 a = make_float4(0.f, 0.f, 0.f, 0.f), h = a, f = a;
#endif
        qI[0] = a.x; qI[1] = a.y; qI[2] = a.z; qI[3] = a.w;
        qH[0] = h.x; qH[1] = h.y; qH[2] = h.z; qH[3] = h.w;
        qF[0] = f.x; qF[1] = f.y; qF[2] = f.z; qF[3] = f.w;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const uint32_t u0 = r0[k] & L::ZM2, u1 = r1[k] & L::ZM2;
          zf0[2 * k] = __low2float(*reinterpret_cast<const __half2*>(&u0));
          zf0[2 * k + 1] = __high2float(*reinterpret_cast<const __half2*>(&u0));
          zf1[2 * k] = __low2float(*reinterpret_cast<const __half2*>(&u1));
          zf1[2 * k + 1] = __high2float(*reinterpret_cast<const __half2*>(&u1));
        }
      }
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        float& A0 = (jj & 1) ? S0b : S0;
        float& A1 = (jj & 1) ? S1b : S1;
        const uint32_t hb = 0x8000u << (16 * (jj & 1));
        const bool H0 = (r0[jj >> 1] & hb) != 0, H1 = (r1[jj >> 1] & hb) != 0;
        const float2 m0 = h2f2(r0[jj >> 1] & 0x7FFF7FFFu), m1 = h2f2(r1[jj >> 1] & 0x7FFF7FFFu);
        const float s0 = (jj & 1) ? m0.y : m0.x, s1 = (jj & 1) ? m1.y : m1.x;
        float x0 = F0 ? fmaf(sacc[jj][5], LO_INV, sacc[jj][4]) : (H0 ? sacc[jj][1] : sacc[jj][0]);
        float x1 = F1 ? fmaf(sacc[jj][7], LO_INV, sacc[jj][6]) : (H1 ? sacc[jj][3] : sacc[jj][2]);
        if constexpr (ASYM) {  // x - z Q (Q of the variant the token's vector uses)
          x0 = fmaf(-zf0[jj], F0 ? qF[jj] : (H0 ? qH[jj] : qI[jj]), x0);
          x1 = fmaf(-zf1[jj], F1 ? qF[jj] : (H1 ? qH[jj] : qI[jj]), x1);
        }
        A0 = fmaf(s0, x0, A0);
        A1 = fmaf(s1, x1, A1);
      }
    }
    S0 += S0b;
    S1 += S1b;
    S0 = val0 ? S0 * sfac : NEG_INF;
    S1 = val1 ? S1 * sfac : NEG_INF;

    // ---------------- (c) V^T fragments of block n (plain INT4), V scales of rows tau0 / tau1
    uint32_t svraw[4];
    auto v_frags = [&]() {
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      if constexpr (B2) {  // INT2 entry: word m = rows d (bits +0 / +2) and d + 8 (+4 / +6) of m tile m
        const uint2 x = lds64(sb + vo[ks]);
        const uint32_t wv[2] = {x.x, x.y};
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          va[ks][m][0] = wv[m] & 0x03030303u;
          va[ks][m][2] = (wv[m] >> 2) & 0x03030303u;
          va[ks][m][1] = (wv[m] >> 4) & 0x03030303u;
          va[ks][m][3] = (wv[m] >> 6) & 0x03030303u;
        }
        continue;
      }
      const uint4 x = lds128(sb + vo[ks]);
      const uint32_t wv[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        if constexpr (ASYM) {
          va[ks][m][0] = wv[2 * m] & 0x0F0F0F0Fu;
          va[ks][m][2] = (wv[2 * m] >> 4) & 0x0F0F0F0Fu;
          va[ks][m][1] = wv[2 * m + 1] & 0x0F0F0F0Fu;
          va[ks][m][3] = (wv[2 * m + 1] >> 4) & 0x0F0F0F0Fu;
        } else {
          va[ks][m][0] = wv[2 * m] & 0x87878787u;
          va[ks][m][2] = __funnelshift_l(wv[2 * m], wv[2 * m], 4) & 0x87878787u;
          va[ks][m][1] = wv[2 * m + 1] & 0x87878787u;
          va[ks][m][3] = __funnelshift_l(wv[2 * m + 1], wv[2 * m + 1], 4) & 0x87878787u;
        }
      }
    }
    };
    if constexpr (!DEF) v_frags();
    svraw[0] = vs0 & 0x7FFF7FFFu;
#pragma unroll
    for (int g = 1; g < 4; ++g) svraw[g] = lds32(sb + vso + g * P * 2);

    // ---------------- (b) QK(n+1) (the last block recomputes its own, still valid stage)
    auto qk_next = [&]() {
#if CBK_PREFETCH
    qk_issue(nq, kx0, kx1);
#else
    if (nq != n) mbar_wait(smem_u32(&mbar[nq % ST]), (nq / ST) & 1);
    {
      const uint32_t sbq = sbase + C::RING_OFF + (nq % ST) * L::STAGE;
      qk_issue(nq, ldk(sbq + kro0), ldk(sbq + kro1));
    }
#endif
    };
    if constexpr (!DEF) qk_next();

    // ---------------- (d) softmax (lazy reference), P' -> smem
    const uint32_t ptb = pt + (n & 1) * C::PT_BYTES;
    float p0, p1;
    bool pflag = false;
    const uint32_t vmask2 = (val0 ? 0xFFFFu : 0u) | (val1 ? 0xFFFF0000u : 0u);
    float zb[4] = {0.f, 0.f, 0.f, 0.f};  // (variant 0) this block's P' z per group (rewritten by the slow path's make_p)
    auto make_p = [&]() {
      p0 = val0 ? ex2(S0 - mref) : 0.f;
      p1 = val1 ? ex2(S1 - mref) : 0.f;
      if constexpr (ZVS) pflag = make_pprime_z<L::ZM2>(p0, p1, svraw, inv2, vmask2, ptb + pwo, zb);
      else pflag = make_pprime(p0, p1, svraw, inv2, vmask2, ptb + pwo);
    };
    bool flag = (S0 > mref + TAU) || (S1 > mref + TAU);
    make_p();
    flag = flag || pflag;
    if constexpr (ZVT) {
      if (w == 3 && lane < 16) {  // zero-point tile of block n (bytes z = e4m3 of z 2^-9), published by the barrier below
        uint32_t zw[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const uint2 x = lds64(sb + zbsrc + (P == 16 ? (c >> 1) * L::PSTR + (c & 1) * 16 : 16 * c));
          zw[c] = prmt(x.x, x.y, 0x6420u) & L::ZM4;
        }
        asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(zbdst + (n & 1) * 512), "r"(zw[0]), "r"(zw[1]),
                     "r"(zw[2]), "r"(zw[3]) : "memory");
      }
    }
    if constexpr (DEF) {  // PV(n-1) done -> nothing in flight: fragments of PV(n), K of n+1, QK(n+1) issue
      wg_wait_all();
#pragma unroll
      for (int m = 0; m < 2; ++m) wg_wait0(pacc[m]);
      if constexpr (ZVT) wg_wait0(zd);
#pragma unroll
      for (int i = 0; i < 2; ++i) { pin_regs(va[i][0]); pin_regs(va[i][1]); }
      if (n > 0) pv_accumulate();
      v_frags();
      qk_next();
    }
    fence_async_smem();
    const int anyflag = __syncthreads_or(flag);
    if (anyflag) {
      float bm = fmaxf(S0, S1);
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) bm = fmaxf(bm, __shfl_xor_sync(FULL, bm, o));
      if (gid == 0) red_m[w][tig] = bm;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const int e0 = static_cast<int>((svraw[g] >> 10) & 31) - 15, e1 = static_cast<int>((svraw[g] >> 26) & 31) - 15;
        int be = max(val0 ? e0 : EREF_NONE, val1 ? e1 : EREF_NONE);
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) be = max(be, __shfl_xor_sync(FULL, be, o));
        if (lane == 0) red_e[w][g] = be;
      }
      __syncthreads();
      float nm = mref;
#pragma unroll
      for (int ww = 0; ww < NWARP; ++ww) nm = fmaxf(nm, red_m[ww][tig]);
      const float alpha = mref == NEG_INF ? 0.f : ex2(mref - nm);
      float beta[2] = {1.f, 1.f};
      float bz0 = 1.f, bz1 = 1.f;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        int ne = eref[g];
#pragma unroll
        for (int ww = 0; ww < NWARP; ++ww) ne = max(ne, red_e[ww][g]);
        ne = (ne == EREF_NONE) ? ne : min(max(ne, -14), 14);
        const float bg = (eref[g] == EREF_NONE || eref[g] - ne < -126) ? 0.f : __uint_as_float(static_cast<uint32_t>(127 + eref[g] - ne) << 23);
        if ((g & 1) == (w >> 1)) beta[g >> 1] = bg;
        if constexpr (ZVS) zacc[g] *= alpha * bg;
        if constexpr (ZVT) {  // betas of the groups of zs0 / zs1
          if (g == 2 * (w & 1)) bz0 = bg;
          if (g == 2 * (w & 1) + 1) bz1 = bg;
        }
        eref[g] = ne;
        const uint32_t hv = ne == EREF_NONE ? 0u : static_cast<uint32_t>(15 - ne) << 10;
        inv2[g] = hv | (hv << 16);
      }
      mref = nm;
#pragma unroll
      for (int m = 0; m < 2; ++m) { O[m][0] *= alpha * beta[m]; O[m][1] *= alpha * beta[m]; }
      lsum *= alpha;
      if constexpr (ZVT) {  // zs0 / zs1 belong to head gid >> 1: its alpha from lane gid >> 1
        const float ah = __shfl_sync(FULL, alpha, gid >> 1);
        zs0 *= ah * bz0;
        zs1 *= ah * bz1;
      }
      __syncthreads();
      make_p();
      fence_async_smem();
      __syncthreads();
    }
    lsum += p0 + p1;
    if constexpr (ZVS) {
#pragma unroll
      for (int g = 0; g < 4; ++g) zacc[g] += zb[g];
    }

    // ---------------- (e) PV(n) (fresh accumulators; added to O after the next wait)
    const uint64_t pdb = pdesc0 + (((n & 1) * C::PT_BYTES) >> 4);
    if constexpr (DEF) {  // QK(n+1) done (block n+1's scores read its accumulators while PV(n) runs)
      wg_wait_all();
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) wg_wait0(sacc[jj]);
#pragma unroll
      for (int k = 0; k < 4; ++k) pin_regs(ka[k]);
    }
    wg_fence();
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      wg_n16<0>(pacc[m], va[0][m], pdb + ((0 * 2 + m) * PTILE >> 4));
      wg_n16<1>(pacc[m], va[1][m], pdb + ((1 * 2 + m) * PTILE >> 4));
    }
    if constexpr (ZVT) {  // Z(n) = P'(n)^T . z(n): A = the P' tiles of k-step ks (rows (group, head, hi/lo)), B = z tile
#if !defined(CB_NO_ZP) && !defined(CB_NO_VZP)  // negative control (wrong results): value zero points ignored
      const uint64_t zbd = zbdesc0 + (((n & 1) * 512) >> 4);
      wg_n8_ss<1>(zd, pdb, zbd);
      wg_n8_ss<1>(zd, pdb + ((2 * PTILE) >> 4), zbd + (256 >> 4));
#endif
    }
    wg_commit();
  }
  wg_wait_all();
#pragma unroll
  for (int m = 0; m < 2; ++m) wg_wait0(pacc[m]);
  if constexpr (ZVT) wg_wait0(zd);
  pv_accumulate();

  // ---------------------------------------------------------------- CTA totals, merge
  float ofac[2];
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    const int e = (w >> 1) ? eref[2 * m + 1] : eref[2 * m];
    ofac[m] = FV * (e == EREF_NONE ? 0.f : exp2f(static_cast<float>(e)));
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
  if constexpr (ASYM) {  // o_d -= Z_g(d) (units of O)
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      const int g = 2 * m + (w >> 1);
      const float z = ZVT ? fmaf(zred[g][tig][1], LO_INV, zred[g][tig][0])
                          : ((zredw[0][tig][g] + zredw[1][tig][g]) + (zredw[2][tig][g] + zredw[3][tig][g])) * (1.f / 512.f);
      O[m][0] -= z;
      O[m][1] -= z;
    }
  }
  float* outf = reinterpret_cast<float*>(smem);  // [G][D] + merge weights
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int r = 0; r < 2; ++r) outf[tig * D + 64 * m + 16 * w + gid + 8 * r] = O[m][r] * ofac[m];
  if (gid == 0) { red_l[w][tig] = lsum; red_m[w][tig] = mref; }
  __syncthreads();
  float Mc[G], Lc[G];
#pragma unroll
  for (int h = 0; h < G; ++h) {
    Mc[h] = red_m[0][h];
    Lc[h] = red_l[0][h] + red_l[1][h] + red_l[2][h] + red_l[3][h];
  }
  merge_store(p, outf, red_l, b, kvh, split, ns_b + p.tail_nslot, Mc, Lc,
              &last_flag);
}


// ================================================================================================= key-only family
// '...+konly' of the sign-flipped-Hadamard codebooks (4 x 4, locked 8 x 8): the keys carry the per-token choice
// (variant v = head mode m ? NLH + m : vector choice l, from the K side word), values are plain INT4 (static v2 PV).
// QK per 32-d group: hi tile wgmma m64nNk32 (N = 8 ceil(V / 2): column 8 (v >> 1) + 2 h + (v & 1)) and the unscaled lo
// tile chained onto the same accumulators (N / 2 per thread instead of N), in rounds of RG groups; per (token, group)
// a select tree over the variants (2 ceil(V / 2) - 1 FSEL).  Synchronous rounds as k_cbr2, with the K words / side
// words of block n+1 and the V words of block n prefetched while wgmmas run.
template <int CB, int P, int ST>
__global__ void __launch_bounds__(128, 3) k_cbkf(const CbAttnParams p) {
  using C = CfgA<CB, P, ST>;
  using L = LayA<CB, P>;
  using SP = CbSpec<CB>;
  constexpr int NV = SP::V, NCH = (NV + 1) / 2, NH = 8 * NCH;  // variants, 8-column chunks, hi tile N
  constexpr int RG = NCH > 4 ? 2 : 4;                           // groups per QK round (accumulator budget)
  constexpr int NQ = C::NQ, CHUNK = L::CHUNK, NPB = L::NPB, PS = L::PSTR;
  constexpr int NLEV = NCH > 4 ? 3 : 2;                         // tree levels over the chunks
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ __align__(8) uint64_t mbar[ST];
  __shared__ float red_m[NWARP][G], red_l[NWARP][G], sfac_s[G];
  __shared__ int red_e[NWARP][4];
  __shared__ int last_flag;
  __shared__ float qstage[G * D];

  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, gid = lane >> 2, tig = lane & 3;
  const int kvh = blockIdx.x, split = blockIdx.y, b = blockIdx.z;
  const int len = p.seq_lens[b];
  const int c0 = split * p.chunk, c1 = min(len, c0 + p.chunk);
  const int ns_b = max(1, (len + p.chunk - 1) / p.chunk);
  if (split >= ns_b || len <= 0) return;  // (len 0: the staging-window kernel writes the output)
  const uint32_t sbase = smem_u32(smem);
  const uint32_t qt = sbase + C::QT_OFF, pt = sbase + C::PT_OFF;

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < ST; ++s) mbar_init(smem_u32(&mbar[s]), NPB);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  for (int i = tid; i < (C::QT_BYTES + 2 * C::PT_BYTES) / 16; i += 128)
    reinterpret_cast<uint4*>(smem + C::QT_OFF)[i] = make_uint4(0, 0, 0, 0);
  __syncthreads();
  const int ntok = max(c1 - c0, 0), nblk = (ntok + BLK - 1) / BLK;
  {
    const int* bt = p.block_table + static_cast<size_t>(b) * p.bt_stride + c0 / P;
    int* pidx = reinterpret_cast<int*>(smem + C::PIDX_OFF);
    const int npg = (ntok + P - 1) / P;
    for (int i = tid; i < npg; i += 128) pidx[i] = __ldg(bt + i);
  }
  const int* pidx_s = reinterpret_cast<const int*>(smem + C::PIDX_OFF);
  const uint8_t* poolh = p.pool + static_cast<size_t>(kvh) * CHUNK;
  const uint32_t pstride = static_cast<uint32_t>(p.Hkv) * CHUNK;
  const uint32_t mbar0 = smem_u32(&mbar[0]);
  const uint32_t dst_w = sbase + C::RING_OFF + w * PS;
  auto issue = [&](int nb) {
    const int s = nb % ST;
    const uint32_t mb = mbar0 + 8 * s;
    const bool has = nb * BLK + w * P < ntok;
    mbar_expect_tx(mb, has ? CHUNK : 0);
    if (has) {
      const uint32_t page = static_cast<uint32_t>(pidx_s[nb * NPB + w]);
      bulk_g2s(dst_w + s * L::STAGE, poolh + static_cast<uint64_t>(page) * pstride, CHUNK, mb);
    }
  };
  __syncthreads();
  if (w < NPB && lane == 0) {
#pragma unroll
    for (int s = 0; s < ST - 1; ++s)
      if (s < nblk) issue(s);
  }
  q_prologue<CB, NQ>(p, b, kvh, w, lane, qt, qstage, sfac_s);
  fence_async_smem();
  __syncthreads();
  const float sfac = sfac_s[tig];

  const int bq = gid & 1, ipq = (gid >> 1) & 1, qbq = gid >> 2;
  const int tau0 = 16 * w + 8 * qbq + 4 * ipq + bq, tau1 = tau0 + 2;
  const uint32_t kro0 = opaque((tau0 / P) * PS + L::K_OFF + (tau0 % P) * 64 + 16 * tig);
  const uint32_t kro1 = opaque((tau1 / P) * PS + L::K_OFF + (tau1 % P) * 64 + 16 * tig);
  const uint32_t kso0 = opaque((tau0 / P) * PS + L::KS_OFF + (tau0 % P) * 8);
  const uint32_t kso1 = opaque((tau1 / P) * PS + L::KS_OFF + (tau1 % P) * 8);
  const uint32_t ksw0 = opaque((tau0 / P) * PS + L::SWK_OFF + (tau0 % P) * 2);
  const uint32_t ksw1 = opaque((tau1 / P) * PS + L::SWK_OFF + (tau1 % P) * 2);
  const uint64_t qd0 = opaque64(sdesc(qt));
  const uint64_t pdesc0 = opaque64(sdesc_p(pt));
  uint32_t vo[2];
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int tb = 32 * u + 8 * tig;
    vo[u] = opaque((tb / P) * PS + L::V_OFF + (((w * 8 + gid) * (P / 8) + (tb % P) / 8) << 4));
  }
  const int cp0 = (tau0 % P) % 8;
  const uint32_t vso = opaque((tau0 / P) * PS + L::VS_OFF + (((tau0 % P) / 8) * 8 + ((cp0 & 1) ? 4 + (cp0 >> 1) : (cp0 >> 1))) * 2);
  const int k0p = 16 * bq + 4 * (2 * (w & 1) + qbq) + 2 * ipq;
  const uint32_t pwo = opaque(((w >> 1) * 2) * PTILE + (k0p / 16) * PLBO + (2 * tig) * 16 + (k0p % 16));

  float O[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
  float lsum = 0.f, mref = NEG_INF;
  int eref[4];
  uint32_t inv2[4];
#pragma unroll
  for (int g = 0; g < 4; ++g) { eref[g] = EREF_NONE; inv2[g] = 0u; }

  uint4 kx0, kx1;
  uint2 kraw0, kraw1;
  uint32_t sw0, sw1;
  auto k_prefetch = [&](int nb) {
    const uint32_t sb = sbase + C::RING_OFF + (nb % ST) * L::STAGE;
    kx0 = lds128(sb + kro0);
    kx1 = lds128(sb + kro1);
    kraw0 = lds64(sb + kso0);
    kraw1 = lds64(sb + kso1);
    sw0 = lds16(sb + ksw0);
    sw1 = lds16(sb + ksw1);
  };
  mbar_wait(smem_u32(&mbar[0]), 0);
  k_prefetch(0);

  for (int n = 0; n < nblk; ++n) {
    const int s = n % ST;
    const uint32_t sb = sbase + C::RING_OFF + s * L::STAGE;
    const int t0 = c0 + n * BLK;
    if (w < NPB && lane == 0 && n + ST - 1 < nblk) issue(n + ST - 1);

    // ---------------- QK(n) in 4 / RG rounds, select per (token, group)
    float S0 = 0.f, S1 = 0.f;
    uint4 ve[2];
    uint32_t svraw[4];
    {
      const uint32_t w0[4] = {kx0.x, kx0.y, kx0.z, kx0.w}, w1[4] = {kx1.x, kx1.y, kx1.z, kx1.w};
      const uint32_t r0[2] = {kraw0.x, kraw0.y}, r1[2] = {kraw1.x, kraw1.y};
      const uint32_t m0 = (sw0 >> (4 * SP::LB)) & ((1u << SP::HB) - 1u), m1 = (sw1 >> (4 * SP::LB)) & ((1u << SP::HB) - 1u);
      static_for<4 / RG>([&](auto rc) {
        constexpr int r = decltype(rc)::value;
        uint32_t ka[RG][4];
        float sacc[RG][NH / 2];
#pragma unroll
        for (int i = 0; i < RG; ++i) {
          const int jj = r * RG + i;
          ka[i][0] = w0[jj] & 0x87878787u;
          ka[i][2] = __funnelshift_l(w0[jj], w0[jj], 4) & 0x87878787u;
          ka[i][1] = w1[jj] & 0x87878787u;
          ka[i][3] = __funnelshift_l(w1[jj], w1[jj], 4) & 0x87878787u;
        }
        wg_fence();
#pragma unroll
        for (int i = 0; i < RG; ++i) {
          const int jj = r * RG + i;
          wg_kf<NH, 0>(sacc[i], ka[i], qd0 + ((jj * NQ * 32) >> 4));
          wg_kf<NH, 1>(sacc[i], ka[i], qd0 + ((jj * NQ * 32 + NQ * 16) >> 4));
        }
        wg_commit();
        if constexpr (r == 0) {  // V words / V scales of block n while the first round runs
          ve[0] = lds128(sb + vo[0]);
          ve[1] = lds128(sb + vo[1]);
#pragma unroll
          for (int g = 0; g < 4; ++g) svraw[g] = lds32(sb + vso + g * P * 2);
        }
        wg_wait_all();
#pragma unroll
        for (int i = 0; i < RG; ++i) { wg_wait0(sacc[i]); pin_regs(ka[i]); }
#pragma unroll
        for (int i = 0; i < RG; ++i) {
          const int jj = r * RG + i;
          const float2 s01 = h2f2(r0[jj >> 1]), s11 = h2f2(r1[jj >> 1]);
          const float sc0 = (jj & 1) ? s01.y : s01.x, sc1 = (jj & 1) ? s11.y : s11.x;
          const uint32_t lm = (1u << SP::LB) - 1u;
          const uint32_t v0 = m0 ? SP::NLH + m0 : (sw0 >> (SP::LB * jj)) & lm, v1 = m1 ? SP::NLH + m1 : (sw1 >> (SP::LB * jj)) & lm;
          auto pick = [&](uint32_t v, int row) {  // accumulator 4 c + 2 row + (v & 1) of chunk c = v >> 1
            float t[NCH];
#pragma unroll
            for (int c = 0; c < NCH; ++c) t[c] = (v & 1u) ? sacc[i][4 * c + 2 * row + 1] : sacc[i][4 * c + 2 * row];
            static_for<NLEV>([&](auto lc) {
              constexpr int lev = decltype(lc)::value;
#pragma unroll
              for (int c = 0; c < (NCH >> (lev + 1)); ++c) t[c] = ((v >> (lev + 1)) & 1u) ? t[2 * c + 1] : t[2 * c];
            });
            return t[0];
          };
          S0 = fmaf(sc0, pick(v0, 0), S0);
          S1 = fmaf(sc1, pick(v1, 1), S1);
        }
      });
    }
    const bool val0 = t0 + tau0 < c1 && t0 + tau0 >= p.kv_start, val1 = t0 + tau1 < c1 && t0 + tau1 >= p.kv_start;  // kv_start: masked leading tokens (sinks)
    S0 = val0 ? S0 * sfac : NEG_INF;
    S1 = val1 ? S1 * sfac : NEG_INF;

    // ---------------- softmax, P'
    const uint32_t ptb = pt + (n & 1) * C::PT_BYTES;
    float p0, p1;
    bool pflag = false;
    const uint32_t vmask2 = (val0 ? 0xFFFFu : 0u) | (val1 ? 0xFFFF0000u : 0u);
    auto make_p = [&]() {
      p0 = val0 ? ex2(S0 - mref) : 0.f;
      p1 = val1 ? ex2(S1 - mref) : 0.f;
      pflag = make_pprime(p0, p1, svraw, inv2, vmask2, ptb + pwo);
    };
    bool flag = (S0 > mref + TAU) || (S1 > mref + TAU);
    make_p();
    flag = flag || pflag;
    fence_async_smem();
    const int anyflag = __syncthreads_or(flag);
    if (anyflag) {
      float bm = fmaxf(S0, S1);
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) bm = fmaxf(bm, __shfl_xor_sync(FULL, bm, o));
      if (gid == 0) red_m[w][tig] = bm;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const int e0 = static_cast<int>((svraw[g] >> 10) & 31) - 15, e1 = static_cast<int>((svraw[g] >> 26) & 31) - 15;
        int be = max(val0 ? e0 : EREF_NONE, val1 ? e1 : EREF_NONE);
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) be = max(be, __shfl_xor_sync(FULL, be, o));
        if (lane == 0) red_e[w][g] = be;
      }
      __syncthreads();
      float nm = mref;
#pragma unroll
      for (int ww = 0; ww < NWARP; ++ww) nm = fmaxf(nm, red_m[ww][tig]);
      const float alpha = mref == NEG_INF ? 0.f : ex2(mref - nm);
      float beta[2] = {1.f, 1.f};
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        int ne = eref[g];
#pragma unroll
        for (int ww = 0; ww < NWARP; ++ww) ne = max(ne, red_e[ww][g]);
        ne = (ne == EREF_NONE) ? ne : min(max(ne, -14), 14);
        const float bg = (eref[g] == EREF_NONE || eref[g] - ne < -126) ? 0.f : __uint_as_float(static_cast<uint32_t>(127 + eref[g] - ne) << 23);
        if ((g & 1) == (w >> 1)) beta[g >> 1] = bg;
        eref[g] = ne;
        const uint32_t hv = ne == EREF_NONE ? 0u : static_cast<uint32_t>(15 - ne) << 10;
        inv2[g] = hv | (hv << 16);
      }
      mref = nm;
#pragma unroll
      for (int m = 0; m < 2; ++m) { O[m][0] *= alpha * beta[m]; O[m][1] *= alpha * beta[m]; }
      lsum *= alpha;
      __syncthreads();
      make_p();
      fence_async_smem();
      __syncthreads();
    }
    lsum += p0 + p1;

    // ---------------- PV(n) (static v2 path), K data of block n+1 prefetched while it runs
    uint32_t va[2][2][4];
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const uint32_t wv[4] = {ve[ks].x, ve[ks].y, ve[ks].z, ve[ks].w};
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        va[ks][m][0] = wv[2 * m] & 0x87878787u;
        va[ks][m][2] = __funnelshift_l(wv[2 * m], wv[2 * m], 4) & 0x87878787u;
        va[ks][m][1] = wv[2 * m + 1] & 0x87878787u;
        va[ks][m][3] = __funnelshift_l(wv[2 * m + 1], wv[2 * m + 1], 4) & 0x87878787u;
      }
    }
    float pacc[2][8];
    const uint64_t pdb = pdesc0 + (((n & 1) * C::PT_BYTES) >> 4);
    wg_fence();
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      wg_n16<0>(pacc[m], va[0][m], pdb + ((0 * 2 + m) * PTILE >> 4));
      wg_n16<1>(pacc[m], va[1][m], pdb + ((1 * 2 + m) * PTILE >> 4));
    }
    wg_commit();
    {
      const int nq = min(n + 1, nblk - 1);
      if (nq != n) mbar_wait(smem_u32(&mbar[nq % ST]), (nq / ST) & 1);
      k_prefetch(nq);
    }
    wg_wait_all();
#pragma unroll
    for (int m = 0; m < 2; ++m) { wg_wait0(pacc[m]); pin_regs(va[0][m]); pin_regs(va[1][m]); }
    {
      const bool g1 = (w >> 1) != 0;
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        O[m][0] += g1 ? fmaf(pacc[m][5], LO_INV, pacc[m][4]) : fmaf(pacc[m][1], LO_INV, pacc[m][0]);
        O[m][1] += g1 ? fmaf(pacc[m][7], LO_INV, pacc[m][6]) : fmaf(pacc[m][3], LO_INV, pacc[m][2]);
      }
    }
  }

  float ofac[2];
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    const int e = (w >> 1) ? eref[2 * m + 1] : eref[2 * m];
    ofac[m] = FV * (e == EREF_NONE ? 0.f : exp2f(static_cast<float>(e)));
  }
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) lsum += __shfl_xor_sync(FULL, lsum, o);
  __syncthreads();
  float* outf = reinterpret_cast<float*>(smem);
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int r = 0; r < 2; ++r) outf[tig * D + 64 * m + 16 * w + gid + 8 * r] = O[m][r] * ofac[m];
  if (gid == 0) { red_l[w][tig] = lsum; red_m[w][tig] = mref; }
  __syncthreads();
  float Mc[G], Lc[G];
#pragma unroll
  for (int h = 0; h < G; ++h) {
    Mc[h] = red_m[0][h];
    Lc[h] = red_l[0][h] + red_l[1][h] + red_l[2][h] + red_l[3][h];
  }
  merge_store(p, outf, red_l, b, kvh, split, ns_b + p.tail_nslot, Mc, Lc,
              &last_flag);
}


// ================================================================================================= launchers
template <int KIND, int P, int ST, int VAR>
int launch_a(const CbAttnParams* p, cudaStream_t st, int* occ) {
  using C = CfgA<KIND, P, ST>;
  if (C::SMEM > 200 * 1024) return -2;
  auto k = cb_family(KIND) ? k_cbq<cb_family(KIND) ? KIND : CB_FINAL, P, ST, 1 - VAR>
            : cb_kfam(KIND) ? k_cbkf<cb_kfam(KIND) ? KIND : CB_Q4K, P, ST>
            : KIND == CB_FULL ? k_cbf<P, ST, 1 - VAR> : cb_k2(KIND) ? k_cbk<cb_k2(KIND) ? KIND : CB_KONLY, P, ST, VAR == 1>
            : VAR == 2 ? k_cbr2<P, ST> : k_cbr<P, ST, VAR == 2 ? 1 : VAR>;
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
    if (!tset) { cudaFuncSetAttribute(k_tail_cb, cudaFuncAttributeMaxDynamicSharedMemorySize, TAIL_SMEM_FLOATS * 4 + 2048); tset = true; }
    tail_fork(st, side, e0, e1);
    k_tail_cb<<<dim3(p->Hkv, p->tail_nslot, p->B), 128, TAIL_SMEM_FLOATS * 4 + 2048, side>>>(*p);
  }
  dim3 grid(p->Hkv, p->nsplit - p->tail_nslot, p->B);
  k<<<grid, 128, C::SMEM, st>>>(*p);
  if (p->tail != nullptr) tail_join(st, side, e1);
  return 0;
}
template <int KIND, int P, int VAR>
int launch_as(int stages, const CbAttnParams* p, cudaStream_t st, int* occ) {
  switch (stages) {
    case 3: return launch_a<KIND, P, 3, VAR>(p, st, occ);
    case 4: return launch_a<KIND, P, 4, VAR>(p, st, occ);
  }
  return -3;
}
template <int KIND, int VAR>
int launch_ap(int page, int stages, const CbAttnParams* p, cudaStream_t st, int* occ) {
  if (page == 16) return launch_as<KIND, 16, VAR>(stages, p, st, occ);
  if (page == 32) return launch_as<KIND, 32, VAR>(stages, p, st, occ);
  return -5;
}
// variant 0 = pipelined (reduced: v2 pipeline; full: PV rounds two deep), 1 = synchronous (fewer registers)
int dispatch_a(int kind, int variant, int page, int stages, const CbAttnParams* p, cudaStream_t st, int* occ) {
  if (kind == CB_FINAL) {
    if (variant == 0) return launch_ap<CB_FINAL, 0>(page, stages, p, st, occ);
    if (variant == 1) return launch_ap<CB_FINAL, 1>(page, stages, p, st, occ);
    return -3;
  }
  if (kind == CB_Q4) {
    if (variant == 0) return launch_ap<CB_Q4, 0>(page, stages, p, st, occ);
    if (variant == 1) return launch_ap<CB_Q4, 1>(page, stages, p, st, occ);
    return -3;
  }
  if (kind == CB_HEAD8) {
    if (variant == 0) return launch_ap<CB_HEAD8, 0>(page, stages, p, st, occ);
    if (variant == 1) return launch_ap<CB_HEAD8, 1>(page, stages, p, st, occ);
    return -3;
  }
  if (kind == CB_HEAD16) {
    if (variant == 0) return launch_ap<CB_HEAD16, 0>(page, stages, p, st, occ);
    if (variant == 1) return launch_ap<CB_HEAD16, 1>(page, stages, p, st, occ);
    return -3;
  }
  if (kind == CB_FULL) {
    if (variant == 0) return launch_ap<CB_FULL, 0>(page, stages, p, st, occ);
    if (variant == 1) return launch_ap<CB_FULL, 1>(page, stages, p, st, occ);
    return -3;
  }
  if (kind == CB_KONLY) {
    if (variant == 0) return launch_ap<CB_KONLY, 0>(page, stages, p, st, occ);
    if (variant == 1) return launch_ap<CB_KONLY, 1>(page, stages, p, st, occ);
    return -3;
  }
  if (kind == CB_KONLYA) {
    if (variant == 0) return launch_ap<CB_KONLYA, 0>(page, stages, p, st, occ);
    if (variant == 1) return launch_ap<CB_KONLYA, 1>(page, stages, p, st, occ);
    return -3;
  }
  if (kind == CB_KONLYA2) {
    if (variant == 0) return launch_ap<CB_KONLYA2, 0>(page, stages, p, st, occ);
    if (variant == 1) return launch_ap<CB_KONLYA2, 1>(page, stages, p, st, occ);
    return -3;
  }
  if (kind == CB_Q4K) return variant == 0 ? launch_ap<CB_Q4K, 0>(page, stages, p, st, occ) : -3;
  if (kind == CB_FINALK) return variant == 0 ? launch_ap<CB_FINALK, 0>(page, stages, p, st, occ) : -3;
  if (kind == CB_REDUCED) {
    if (variant == 2) return launch_ap<CB_REDUCED, 2>(page, stages, p, st, occ);
    if (variant == 0) return launch_ap<CB_REDUCED, 0>(page, stages, p, st, occ);
    if (variant == 1) return launch_ap<CB_REDUCED, 1>(page, stages, p, st, occ);
    return -3;
  }
  return -4;
}

}  // namespace

int cb_attn_launch(int kind, int variant, int stages, const CbAttnParams& p, cudaStream_t st) {
  if (p.B <= 0) return 0;
  return dispatch_a(kind, variant, p.page, stages, &p, st, nullptr);
}
int cb_attn_occupancy(int kind, int variant, int stages, int page) {
  int occ = 0;
  const int rc = dispatch_a(kind, variant, page, stages, nullptr, nullptr, &occ);
  return rc == 0 ? occ : 0;
}
