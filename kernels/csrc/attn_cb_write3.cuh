// Codebook cache writer v3: tensor-core candidate evaluation with an exact fallback.  Included by
// attn_cb_write3.cu inside its anonymous namespace (uses the helpers of attn_cb_common.cuh).
//
// Same output as k_cb_write / codebook_ref.py (codes, fp16 scales, choices bit for bit) for the codebooks without the
// RoPE-pair candidate P (reduced 2x2, konly, q4x4 (+k), final 8x8 (+k), head8, head16).  Two kernels per call, both
// persistent over tiles of tokens of one kv head (x of the next tile prefetched by cp.async):
//
// k_cbw3_v (values, 8-token tiles): every candidate is generated in registers by one thread (local candidate of a
//   32-vector: 32 elements, FWHT32 without shuffles) or by 4 adjacent lanes (whole-head candidate: one quarter each,
//   the two cross-quarter butterfly stages by shuffles), and its error under the diagonal weighting c_o is evaluated
//   exactly in the reference order (halving trees over registers / quarter lanes).  No approximation.  V^T codes of
//   an aligned 8-slot group are stored as whole words (no atomics).
//
// k_cbw3_k (keys, tiles of TT tokens = 32 local + 32 head candidate rows): the C_q quadratic forms run on the tensor
// cores.
//   A  candidate generation as above; d' = 2^sigma_t d (d = rec - x, the reference's fp32 values; sigma_t from max|x|,
//      exact) is stored as fp16 hi = fp16(d') and lo = fp16(d' - hi) in swizzled rows (ldmatrix operands).
//   B  local forms d_j^T C_jj d_j for every (token, vector, candidate) and head forms d^T C d for every (token, head
//      candidate) by mma.sync m16n8k16 (fp16, fp32 accumulation): A = hi rows, B = C1 = fp16(2^tau (C + C^T) / 2)
//      (host, swizzled in smem).  Epilogue: e~ = sum_i (hi_i + 2 lo_i) (C1 hi)_i (first-order correction of the
//      rounding of d', C1 symmetric).  The rounding of C, E = 2^tau C_sym - C1, is corrected on the 8 columns D where
//      it is largest: one extra n-tile F (rows D of E, off-D entries doubled) adds sum_n hi_{D_n} (F hi)_n = hi^T E hi
//      minus the rest block E_RR.  Remaining error per candidate: |e~ - e_ref| <= beta ||d'||^2 with beta =
//      ||E_RR||_2 + ||F - fp16(F)||_2 + 2^-10 ||E||_2 + kappa rho(|C|) (host, fp64, per head and per diagonal 32-block;
//      kappa covers the tensor-core accumulation, the epilogue, the fp16 lo term, the second-order term and the
//      reference's own fp32 rounding).
//   C  selection with the bound: candidate c can be the reference's first-argmin only if e~_c - b_c <= min (e~ + b).
//      One possible candidate: it is the reference's choice.  Several (near-ties): exact tasks, one warp per (item,
//      candidate): the candidate is regenerated lane-parallel in the reference's fp32 order and its error evaluated in
//      the reference order (C from L1); then the first index among the exact minima.  So the choice equals
//      codebook_ref.encode exactly (no tie-level flips).
//   D  winner's codes / scales (kept from A) into the page, side words as k_cb_write.
// Shared-memory layouts are padded / swizzled so that the per-thread 32-element accesses are bank-conflict free.
#pragma once

namespace w3 {
#ifndef W3_VMINB
#define W3_VMINB 3
#endif
#ifndef W3_PROF
#define W3_PROF 0
#endif
// per-phase cycle accounting of CTA (0, y) (W3_PROF builds only)
#define W3_PROF_MARK(k) if (W3_PROF) { const unsigned long long nw = clock64(); prof[pph] += nw - pt; pt = nw; pph = (k); }
#ifndef W3_HNQ
#define W3_HNQ 2
#endif
#ifndef W3_NOFB
#define W3_NOFB 0
#endif
#ifndef W3_NOTC
#define W3_NOTC 0
#endif

DEVI void cp_async16(void* s, const void* g) {
  const unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(s));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(g));
}
DEVI void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
DEVI void cp_async_wait0() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
DEVI void ldsm_x4(uint32_t (&r)[4], const void* s) {
  const unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(s));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sa));
}
DEVI void ldsm_x2(uint32_t (&r)[2], const void* s) {
  const unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(s));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(sa));
}
DEVI void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI uint32_t pack_h2(float lo, float hi) {
  const __half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}
DEVI float2 unpack_h2(uint32_t u) { return __half22float2(*reinterpret_cast<const __half2*>(&u)); }
constexpr int al16(int x) { return (x + 15) / 16 * 16; }

// ---- padded fp32 rows: element i of a row at (i / 32) * CPF + i % 32 (32-vector chunks 144 B apart), rows XP floats
// apart: a thread's 16-byte accesses to its own 32-vector hit distinct banks across lanes
constexpr int CPF = 36, XP = 4 * CPF;
DEVI int xo(int i) { return (i >> 5) * CPF + (i & 31); }
constexpr int CBP = 144;  // int8 code rows: 144 B apart
// sign tables: 32-sign chunks SCP floats apart, rows SGP apart: 8 rows x one chunk and 2 rows x 4 chunks both hit 8
// distinct 16-byte bank slots
#ifndef W3_SCP
#define W3_SCP 36
#endif
constexpr int SCP = W3_SCP, SGP = 4 * SCP + 4;
DEVI int so(int i) { return (i >> 5) * SCP + (i & 31); }
// swizzled fp16 operand rows (256 B = 16 chunks of 16 B): conflict-free for ldmatrix (8 rows, one chunk) and for the
// generators' stores (8 rows x one 32-vector, or 2 rows x 4 quarters)
DEVI int swz(int c, int r) { return c ^ (r & 7) ^ ((c >> 3) << 1); }

// in-register Sylvester FWHT (stages h = 1, 2, .., N/2 in order: the reference's association) without the final scale
// (compile-time recursion: every index constant, the arrays stay in registers)
template <int N, int H = 1> DEVI void fwht_reg(float* v) {
  if constexpr (H < N) {
#pragma unroll
    for (int i = 0; i < N; ++i)
      if (!(i & H)) {
        const float a = v[i], b = v[i + H];
        v[i] = __fadd_rn(a, b);
        v[i + H] = __fsub_rn(a, b);
      }
    fwht_reg<N, 2 * H>(v);
  }
}
template <int N> DEVI void halve_reg(float* v) {  // halving tree v[:N] + v[N:] down to v[0]
  if constexpr (N >= 1) {
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = __fadd_rn(v[i], v[i + N]);
    halve_reg<N / 2>(v);
  }
}
template <int N> DEVI void max_reg(float* v) {  // halving max tree down to v[0] (compile-time indices)
  if constexpr (N >= 1) {
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = fmaxf(v[i], v[i + N]);
    max_reg<N / 2>(v);
  }
}
// INT4 quantization of a 32-group held in registers (= quant_val per element, shared scale), cheaper form:
//  - rint(y / s) via the magic constant 1.5 * 2^23 (FADD rounds to the nearest even integer, |y / s| < 2^22), whose
//    bit pattern also holds the code in its low byte (two's complement): no FRND / F2I (quarter-rate pipes);
//  - no clamp for a normal fp16 scale: s >= (amax / 7)(1 - 2^-11 - 2^-23) gives |y / s| <= 7.004, so rint <= 7; a
//    subnormal scale takes the clamped path (as the reference).  s = 0: inv = 0 gives q = 0 (amax = 0, all y = 0).
//  (rintf gives -0 where this gives +0: codes, scales and errors compare equal.)
DEVI void quant32(const float* y, float* deq, uint32_t* qw, unsigned& sbits) {  // qw: 8 words of 4 int8 codes
  constexpr float MAGIC = 12582912.f;
  float m16[16];  // max |y| as a tree (exact in any order; a 32-deep chain would expose the latency)
#pragma unroll
  for (int a = 0; a < 16; ++a) m16[a] = fmaxf(fabsf(y[a]), fabsf(y[a + 16]));
  max_reg<8>(m16);
  const __half sh = __float2half_rn(__fmul_rn(m16[0], 1.0f / 7.0f));
  const float sf = __half2float(sh);
  const float inv = sf != 0.f ? __frcp_rn(sf) : 0.f;
  uint32_t bits[4];
  if (sf >= 6.103515625e-05f) {
#pragma unroll
    for (int a = 0; a < 32; ++a) {
      const float tt = __fadd_rn(div_rn(y[a], sf, inv), MAGIC);
      deq[a] = __fmul_rn(__fsub_rn(tt, MAGIC), sf);
      bits[a & 3] = __float_as_uint(tt);
      if ((a & 3) == 3)
        qw[a >> 2] = __byte_perm(__byte_perm(bits[0], bits[1], 0x0040), __byte_perm(bits[2], bits[3], 0x0040), 0x5410);
    }
  } else {
#pragma unroll
    for (int a = 0; a < 32; ++a) {
      const float q = fminf(fmaxf(__fsub_rn(__fadd_rn(div_rn(y[a], sf, inv), MAGIC), MAGIC), -7.f), 7.f);
      deq[a] = __fmul_rn(q, sf);
      bits[a & 3] = __float_as_uint(__fadd_rn(q, MAGIC));
      if ((a & 3) == 3)
        qw[a >> 2] = __byte_perm(__byte_perm(bits[0], bits[1], 0x0040), __byte_perm(bits[2], bits[3], 0x0040), 0x5410);
    }
  }
  sbits = __half_as_ushort(sh);
}
template <int N> DEVI void min_reg(float* v) {  // halving min tree down to v[0] (compile-time indices)
  if constexpr (N >= 1) {
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = fminf(v[i], v[i + N]);
    min_reg<N / 2>(v);
  }
}
// rint(v) for |v| < 2^22 by the magic constant (FADD rounds to the nearest even integer); the bit pattern of
// v + MAGIC holds the integer in its low bits
DEVI float rint_magic(float v) { return __fsub_rn(__fadd_rn(v, 12582912.f), 12582912.f); }
// Asymmetric INT4 (blockflag int4a:32) of a 32-group held in registers, = quant_val_a per element:
//   error form (deq):  s = max((mx - mn) * fp32(1/15), 1e-12), z = rne(-mn / s) (unbounded), qa = clamp(rne(y / s) + z,
//                      0, 15), deq = (qa - z) s
//   stored form:       the same with the range widened to include 0 (identical whenever mn <= 0 <= mx): unsigned codes
//                      0..15 (qw, one byte each), scale word = the FP16 nearest to s whose low 4 bits are z (sbits)
//  - divisions: y / s = div_rn(y, s, RN(1/s)) (correctly rounded: Markstein; a result in the subnormal range is < 1 and
//    rounds to 0 either way), rne via the magic constant while |y| / s < 2^21 for every element of the group (always
//    unless the group is (nearly) constant: s = 1e-12), else rintf.
// NB = 2: asymmetric INT2 (int2a:32 / int2ah:32, vLLM plugin Phase 3a): (mx - mn) * fp32(1/3), codes 0..3, 2-bit z
template <int NB = 4>
DEVI void quant32a_core(const float* y, float mn, float mx, bool err, float* deq, uint32_t* qw, unsigned* sbits) {
  constexpr float LV = static_cast<float>((1 << NB) - 1);
  const float s = fmaxf(__fmul_rn(__fsub_rn(mx, mn), 1.0f / LV), 1e-12f);
  const float inv = __frcp_rn(s);
  const bool fast = fmaxf(fabsf(mn), fabsf(mx)) < __fmul_rn(s, 2097152.f);
  const float zq = div_rn(-mn, s, inv);
  const float z = fast ? rint_magic(zq) : rintf(zq);
  uint32_t bits[4];
#pragma unroll
  for (int a = 0; a < 32; ++a) {
    const float v = div_rn(y[a], s, inv);
    const float q = fminf(fmaxf(__fadd_rn(fast ? rint_magic(v) : rintf(v), z), 0.f), LV);
    if (err) deq[a] = __fmul_rn(__fsub_rn(q, z), s);
    if (qw) {
      bits[a & 3] = __float_as_uint(__fadd_rn(q, 12582912.f));
      if ((a & 3) == 3)
        qw[a >> 2] = __byte_perm(__byte_perm(bits[0], bits[1], 0x0040), __byte_perm(bits[2], bits[3], 0x0040), 0x5410);
    }
  }
  if (sbits) *sbits = scale_word_a<NB>(s, static_cast<unsigned>(z));
}
template <int NB = 4>
DEVI void quant32a(const float* y, float* deq, uint32_t* qw, unsigned& sbits) {  // deq may alias y
  float t[16];
#pragma unroll
  for (int a = 0; a < 16; ++a) t[a] = fminf(y[a], y[a + 16]);
  min_reg<8>(t);
  const float mn = t[0];
#pragma unroll
  for (int a = 0; a < 16; ++a) t[a] = fmaxf(y[a], y[a + 16]);
  max_reg<8>(t);
  const float mx = t[0];
  if (mn > 0.f || mx < 0.f) {  // rare: the stored form differs (codes / word from the widened range, first)
    quant32a_core<NB>(y, fminf(mn, 0.f), fmaxf(mx, 0.f), false, nullptr, qw, &sbits);
    quant32a_core<NB>(y, mn, mx, true, deq, nullptr, nullptr);
  } else {
    quant32a_core<NB>(y, mn, mx, true, deq, qw, &sbits);
  }
}
// quantizer mode A (= cb_qm): 0 INT4 (fp16 amax / 7), 1 int4a:32 / int4ah:32, 2 int2a:32 / int2ah:32
template <int A> DEVI void quant32x(const float* y, float* deq, uint32_t* qw, unsigned& sbits) {
  if constexpr (A) quant32a<A == 2 ? 2 : 4>(y, deq, qw, sbits);
  else quant32(y, deq, qw, sbits);
}
DEVI void store_codes32(int8_t* dst, const uint32_t* qw) {  // 32 packed codes, 16-byte aligned destination
  *reinterpret_cast<uint4*>(dst) = make_uint4(qw[0], qw[1], qw[2], qw[3]);
  *reinterpret_cast<uint4*>(dst + 16) = make_uint4(qw[4], qw[5], qw[6], qw[7]);
}
DEVI float tree_reg32(float* v) {  // halving tree over 32 registers (destroys v)
  halve_reg<16>(v);
  return v[0];
}

// Sign flips by a float table in smem (rows 0 .. CB_NSG-1 = S_k in the order of p.sg as +-1.0f, row SGF_ONE = +1):
// y = x * s (exact) before the transform, d = fma(rec, s, -x) = RN(s rec - x) after it (the reference's s * rec - x).
constexpr int SGF_ONE = CB_NSG;

// one local candidate of a 32-vector x (smem chunk, fp32): fw = apply H32 (all but I), sg = its 32 signs; codes ->
// cdst (32 B), scale bits -> *sdst, y <- d = rec - z (exact)
template <int A = 0>
DEVI void local_cand(const float* x, const float* sg, bool fw, float* y, int8_t* cdst, uint16_t* sdst) {
#pragma unroll
  for (int v = 0; v < 8; ++v) {
    const float4 f = *reinterpret_cast<const float4*>(x + 4 * v);
    const float4 g = *reinterpret_cast<const float4*>(sg + 4 * v);
    y[4 * v] = f.x * g.x; y[4 * v + 1] = f.y * g.y; y[4 * v + 2] = f.z * g.z; y[4 * v + 3] = f.w * g.w;
  }
  if (fw) {
    fwht_reg<32>(y);
#pragma unroll
    for (int a = 0; a < 32; ++a) y[a] = __fmul_rn(y[a], C32);
  }
  {
    uint32_t q[8];
    unsigned sb;
    quant32x<A>(y, y, q, sb);  // y <- deq
    store_codes32(cdst, q);
    *sdst = static_cast<uint16_t>(sb);
  }
  if (fw) {
    fwht_reg<32>(y);
#pragma unroll
    for (int a = 0; a < 32; ++a) y[a] = __fmul_rn(y[a], C32);
  }
#pragma unroll
  for (int v = 0; v < 8; ++v) {
    const float4 f = *reinterpret_cast<const float4*>(x + 4 * v);
    const float4 g = *reinterpret_cast<const float4*>(sg + 4 * v);
    y[4 * v] = fmaf(y[4 * v], g.x, -f.x); y[4 * v + 1] = fmaf(y[4 * v + 1], g.y, -f.y);
    y[4 * v + 2] = fmaf(y[4 * v + 2], g.z, -f.z); y[4 * v + 3] = fmaf(y[4 * v + 3], g.w, -f.w);
  }
}

// FWHT128 stages h = 32, 64 across the 4 quarter lanes of a row (lane & 3 = quarter b: elements 32 b .. 32 b + 31)
DEVI void cross_stages(float* v, int b) {
  const float s1 = (b & 1) ? -1.f : 1.f, s2 = (b & 2) ? -1.f : 1.f;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const float o = __shfl_xor_sync(FULL, v[i], 1);
    v[i] = fmaf(v[i], s1, o);  // (b & 1) ? o - v : v + o, one rounding (sign product exact)
  }
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const float o = __shfl_xor_sync(FULL, v[i], 2);
    v[i] = __fmul_rn(fmaf(v[i], s2, o), C128);
  }
}
// quarter b (= lane & 3) of the whole-head candidate H128 S (x, sg: padded rows; sg = +1 for H128): codes of group b
// -> cdst (32 B), scale bits -> *sdst (if act), v <- d = rec - x (exact).  All 32 lanes of the warp call it (quarters
// of a row on 4 adjacent lanes).
template <int A = 0>
DEVI void head_cand_q(const float* x, const float* sg, int b, float* v, int8_t* cdst, uint16_t* sdst, bool act) {
  x += CPF * b;
  sg += SCP * b;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float4 f = *reinterpret_cast<const float4*>(x + 4 * i);
    const float4 g = *reinterpret_cast<const float4*>(sg + 4 * i);
    v[4 * i] = f.x * g.x; v[4 * i + 1] = f.y * g.y; v[4 * i + 2] = f.z * g.z; v[4 * i + 3] = f.w * g.w;
  }
  fwht_reg<32>(v);
  cross_stages(v, b);
  {
    uint32_t q[8];
    unsigned sb;
    quant32x<A>(v, v, q, sb);
    if (act) {
      store_codes32(cdst, q);
      *sdst = static_cast<uint16_t>(sb);
    }
  }
  fwht_reg<32>(v);
  cross_stages(v, b);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float4 f = *reinterpret_cast<const float4*>(x + 4 * i);
    const float4 g = *reinterpret_cast<const float4*>(sg + 4 * i);
    v[4 * i] = fmaf(v[4 * i], g.x, -f.x); v[4 * i + 1] = fmaf(v[4 * i + 1], g.y, -f.y);
    v[4 * i + 2] = fmaf(v[4 * i + 2], g.z, -f.z); v[4 * i + 3] = fmaf(v[4 * i + 3], g.w, -f.w);
  }
}
// exact value error of a row held by 4 quarter lanes: tree128((e e) c) in the reference order (v[:64] + v[64:] first);
// valid in the quarter-0 lane.  co: padded row (smem).  All lanes call it.
DEVI float vhead_err_q(float* v, const float* co, int b) {
#pragma unroll
  for (int i = 0; i < 32; ++i) v[i] = __fmul_rn(__fmul_rn(v[i], v[i]), co[CPF * b + i]);
#pragma unroll
  for (int i = 0; i < 32; ++i) v[i] = __fadd_rn(v[i], __shfl_xor_sync(FULL, v[i], 2));
#pragma unroll
  for (int i = 0; i < 32; ++i) v[i] = __fadd_rn(v[i], __shfl_xor_sync(FULL, v[i], 1));
  halve_reg<16>(v);
  return v[0];
}

template <int CB> struct W3 {
  static constexpr bool RED = CB == CB_REDUCED || cb_k2(CB);       // 2 x 2 keys, side info in sign bits
  static constexpr bool ASYM = cb_asym(CB);                         // asymmetric INT4 (konlya, int4a:32 / int4ah:32)
  static constexpr int QM = cb_qm(CB);                              // quantizer mode (2: konlya2, INT2 page layout)
  static constexpr int NL = RED ? 2 : 1 + CbSpec<CB>::NLH;         // local candidates (I, H, HS_k)
  static constexpr int NHF = RED ? 1 : CbSpec<CB>::NHF;            // whole-head transforms (F, FS_k)
  static constexpr int NH = 1 + NHF;                                // head candidates incl. base
  static constexpr int MX = NL > NH ? NL : NH;
  static constexpr int TT = MX >= 32 ? 1 : 32 / MX;                 // key tokens per tile (32 local + 32 head rows)
  static constexpr bool VID = cb_konly(CB);                         // values: identity only
  static constexpr int VNL = VID ? 1 : NL, VNHF = VID ? 0 : NHF, VNH = 1 + VNHF;
  static constexpr int TTV = 8;                                     // value tokens per tile (one 8-token V^T group)
};
// host constants per kv head (aux = beta, fp32 [Hkv][AUX]): 0 head bound, 1..4 diagonal-block bounds, 5 absolute
// slack; 8..15 the correction columns D of the head, 16 + 8 j .. the correction columns D_j of block j (as floats)
constexpr int AUX = 48;
constexpr int CROWS = 168;  // C1 [128 rows] + F head [8] + F blocks [4 x 8] (fp16, 256 B per row)

// ------------------------------------------------------------------------------------------------ smem layouts
template <int CB> struct KSm {
  using W = W3<CB>;
  static constexpr int TT = W::TT, NL = W::NL, NH = W::NH, NHF = W::NHF;
  static constexpr int RLV = TT * NL, RHV = TT * NH, RL = al16(RLV), RH = al16(RHV), R = RL + RH;
  static constexpr int C1_OFF = 0;
  static constexpr int HI_OFF = CROWS * 256;                      // d' hi rows [R][256 B] (swizzled)
  static constexpr int LO_OFF = HI_OFF + R * 256;                 // d' lo rows
  static constexpr int X_OFF = LO_OFF + R * 256;                  // x [TT][XP]
  static constexpr int CL_OFF = X_OFF + TT * XP * 4;              // local codes [TT * NL][CBP]
  static constexpr int CH_OFF = CL_OFF + TT * NL * CBP;           // head codes [TT * NHF][CBP]
  static constexpr int SL_OFF = CH_OFF + TT * NHF * CBP;
  static constexpr int SH_OFF = al16(SL_OFF + TT * NL * 8);
  static constexpr int EL_OFF = al16(SH_OFF + TT * NHF * 8);      // errL, nrmL [RL][4]
  static constexpr int RLP = RL + 1;                              // errL, nrmL [4][RLP] (padded)
  static constexpr int EH_OFF = al16(EL_OFF + 4 * RLP * 4 * 2);   // errH, nrmH partials [RH][4]
  static constexpr int SG_OFF = EH_OFF + RH * 4 * 4 * 2;          // per-token scale 2^sigma
  static constexpr int LS_OFF = SG_OFF + al16(TT * 4);            // local choices [TT][4] (u8)
  static constexpr int HS_OFF = LS_OFF + al16(TT * 4);            // head choice [TT] (int)
  static constexpr int DI_OFF = HS_OFF + al16(TT * 4);            // correction columns (int [40])
  static constexpr int FC_OFF = DI_OFF + 160;                     // counters: local items, tasks, head items, tasks
  static constexpr int NLT = TT * 4 * NL, NHT = TT * NH;          // max exact tasks
  static constexpr int FI_OFF = FC_OFF + 16;                      // local items (it, first task, n) [TT*4][3]
  static constexpr int FT_OFF = FI_OFF + TT * 4 * 12;             // local tasks (it | c << 8) [NLT]
  static constexpr int FE_OFF = FT_OFF + NLT * 4;                 // local task errors [NLT]
  static constexpr int GI_OFF = FE_OFF + NLT * 4;                 // head items [TT][3]
  static constexpr int GT_OFF = GI_OFF + TT * 12;                 // head tasks (t | m << 8) [NHT]
  static constexpr int GE_OFF = GT_OFF + NHT * 4;                 // head task errors [NHT]
  static constexpr int SGF_OFF = al16(GE_OFF + NHT * 4);          // sign table [16][SGP] fp32
  static constexpr int SCR_OFF = SGF_OFF + 16 * SGP * 4;          // per-warp scratch rows [8][128] fp32 (exact tasks)
  static constexpr int DB_OFF = SCR_OFF + 8 * 512;                // 2 regenerated candidate rows [2][128] fp32
  static constexpr int XR_OFF = DB_OFF + 2 * 512;                 // next tile's x, raw bf16 (cp.async prefetch)
  static constexpr int TOTAL = XR_OFF + TT * 256;
};
template <int CB> struct VSm {
  using W = W3<CB>;
  static constexpr int TT = W::TTV, NL = W::VNL, NH = W::VNH, NHF = W::VNHF;
  static constexpr int X_OFF = 0;                                  // x [TT][XP]
  static constexpr int D_OFF = TT * XP * 4;                        // local candidates' d [TT * NL][XP]
  static constexpr int CL_OFF = D_OFF + TT * NL * XP * 4;
  static constexpr int CH_OFF = CL_OFF + TT * NL * CBP;
  static constexpr int SL_OFF = CH_OFF + TT * NHF * CBP;
  static constexpr int SH_OFF = al16(SL_OFF + TT * NL * 8);
  static constexpr int EL_OFF = al16(SH_OFF + TT * NHF * 8 + 8);  // errL [TT][NL][4]
  static constexpr int RLP = TT * NL + 1;                         // errL [4][RLP] (padded)
  static constexpr int EH_OFF = al16(EL_OFF + 4 * RLP * 4);        // errH [TT][NH]
  static constexpr int LS_OFF = al16(EH_OFF + TT * NH * 4);
  static constexpr int HS_OFF = LS_OFF + al16(TT * 4);
  static constexpr int SGF_OFF = HS_OFF + al16(TT * 4);            // sign table [16][SGP] fp32
  static constexpr int GF_OFF = SGF_OFF + 16 * SGP * 4;
  static constexpr int XR_OFF = GF_OFF + 16;                       // next tile's x, raw bf16 (cp.async prefetch)
  static constexpr int CO_OFF = XR_OFF + TT * 256;                 // c_o of the head (padded row)
  static constexpr int TOTAL = CO_OFF + XP * 4;
};

// v2 INT4 element layout of a 4-bit code inside a 32-bit word (k_cb_write's `place`)
template <int A = 0>
DEVI void place_bits(uint32_t cc, bool isB, int i, uint32_t& m, uint32_t& bits) {
  const uint32_t s = (cc >> 3) & 1u, mag = cc & 7u;
  if (A) {  // int4a: unsigned nibble, A low / B high nibble of byte i
    m = (isB ? 0xF0u : 0x0Fu) << (8 * i);
    bits = cc << (8 * i + (isB ? 4 : 0));
  } else if (!isB) { m = 0x87u << (8 * i); bits = ((s << 7) | mag) << (8 * i); }
  else {
    const int jj = (i + 3) & 3;
    m = (0x08u << (8 * i)) | (0x70u << (8 * jj));
    bits = (s << (8 * i + 3)) | (mag << (8 * jj + 4));
  }
}
template <int A = 0>
DEVI uint32_t code4(int qi) { return A ? static_cast<uint32_t>(qi) & 15u : static_cast<uint32_t>(qi < 0 ? (8 | -qi) : qi); }
DEVI uint32_t vt_word_off(int dd, int c8, int P) {  // byte offset of element dd's V^T word for 8-token group c8
  const int wq = (dd % 64) / 16, gq = dd % 8, sub = 2 * (dd / 64) + ((dd % 16) / 8);
  return (((wq * 8 + gq) * (P / 8) + c8) << 4) + 4 * sub;
}
DEVI long long slot_of(const CbWriteParams& p, int t, int h) {
  return p.slots[static_cast<long long>(t) * (p.slot_ph ? p.Hkv : 1) + (p.slot_ph ? h : 0)];
}
// x rows of a tile: raw bf16 prefetch (cp.async, 256 B per row) and conversion to padded fp32 rows (zero past the end)
DEVI void issue_x(uint8_t* xr, const CbWriteParams& p, int t0, int ntok, int TT, int hr, int tid) {
  for (int c = tid; c < TT * 16; c += 256) {
    const int t = c >> 4, k = c & 15;
    if (t < ntok) cp_async16(xr + t * 256 + k * 16, static_cast<const __nv_bfloat16*>(p.kv) + (long long)(t0 + t) * p.kv_ts + hr * D + k * 8);
  }
  cp_async_commit();
}
DEVI void convert_x(float* xs, const uint8_t* xr, int ntok, int TT, int tid, const float* mu = nullptr) {
  for (int i = tid; i < TT * 32; i += 256) {
    const int t = i >> 5, d4 = (i & 31) * 4;
    float4 f = make_float4(0.f, 0.f, 0.f, 0.f);
    if (t < ntok) {
      const uint2 u = *reinterpret_cast<const uint2*>(xr + t * 256 + d4 * 2);
      const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
      const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
      f = make_float4(a.x, a.y, b.x, b.y);
      if (mu != nullptr) {  // static key centering (fp32, RN), mu = kmu row of this kv head
        const float4 m = *reinterpret_cast<const float4*>(mu + d4);
        f = make_float4(__fsub_rn(f.x, m.x), __fsub_rn(f.y, m.y), __fsub_rn(f.z, m.z), __fsub_rn(f.w, m.w));
      }
    }
    *reinterpret_cast<float4*>(xs + t * XP + xo(d4)) = f;
  }
}
DEVI void build_sign_table(float* sgf, int pitch, const CbWriteParams& p, int tid) {  // p.sg index warp-uniform
  for (int i = tid; i < 16 * 128; i += 256) {
    const int k = i >> 7, d = i & 127;
    sgf[k * pitch + so(d)] = (k < CB_NSG && ((p.sg[k][d >> 5] >> (d & 31)) & 1u)) ? -1.f : 1.f;
  }
}

// ================================================================================================== values
template <int CB, int P>
__global__ void __launch_bounds__(256, W3_VMINB) k_cbw3_v(const CbWriteParams p) {
  using W = W3<CB>;
  using S = VSm<CB>;
  using L = CbLay<CB, P>;
  constexpr int TT = S::TT, NL = S::NL, NH = S::NH, NHF = S::NHF;
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int h = blockIdx.x;
  const int ntiles = (p.T + TT - 1) / TT;
  if (static_cast<int>(blockIdx.y) >= ntiles) return;
  float* xs = reinterpret_cast<float*>(sm + S::X_OFF);
  float* dl = reinterpret_cast<float*>(sm + S::D_OFF);
  int8_t* cl = reinterpret_cast<int8_t*>(sm + S::CL_OFF);
  int8_t* ch = reinterpret_cast<int8_t*>(sm + S::CH_OFF);
  uint16_t* sl = reinterpret_cast<uint16_t*>(sm + S::SL_OFF);
  uint16_t* sh = reinterpret_cast<uint16_t*>(sm + S::SH_OFF);
  float* el = reinterpret_cast<float*>(sm + S::EL_OFF);
  float* eh = reinterpret_cast<float*>(sm + S::EH_OFF);
  uint8_t* ls = reinterpret_cast<uint8_t*>(sm + S::LS_OFF);
  int* hs = reinterpret_cast<int*>(sm + S::HS_OFF);
  float* sgf = reinterpret_cast<float*>(sm + S::SGF_OFF);
  int* gflag = reinterpret_cast<int*>(sm + S::GF_OFF);
  uint8_t* xr = sm + S::XR_OFF;
  float* co = reinterpret_cast<float*>(sm + S::CO_OFF);
  if (tid < D) co[xo(tid)] = p.co[h * D + tid];
  build_sign_table(sgf, SGP, p, tid);
  issue_x(xr, p, blockIdx.y * TT, min(TT, p.T - static_cast<int>(blockIdx.y) * TT), TT, p.Hkv + h, tid);
  for (int tile = blockIdx.y; tile < ntiles; tile += gridDim.y) {
    const int t0 = tile * TT;
    const int ntok = min(TT, p.T - t0);
    cp_async_wait0();
    __syncthreads();  // x landed; the previous tile is done with the smem
    convert_x(xs, xr, ntok, TT, tid);
    if (w == 7) {  // whole aligned 8-slot group (consecutive slots of one page)?
      const long long sb = slot_of(p, t0, h);
      const long long st = lane < ntok ? slot_of(p, t0 + lane, h) : -1;
      const bool ok = __all_sync(FULL, lane >= TT || st == sb + lane);
      if (lane == 0) gflag[0] = ok && ntok == TT && sb >= 0 && (sb % 8) == 0;
    }
    __syncthreads();
    if (tile + static_cast<int>(gridDim.y) < ntiles) {  // prefetch the next tile while this one is encoded
      const int n0 = (tile + gridDim.y) * TT;
      issue_x(xr, p, n0, min(TT, p.T - n0), TT, p.Hkv + h, tid);
    }
    // ---- A: candidates with their exact errors.  local: thread per (row r = t NL + c, vector j), lanes = 2 rows x 4
    // vectors (conflict-free padded rows); whole-head: 4 quarter lanes per (token, transform)
    constexpr int NLI = TT * 4 * NL, NLP = (NLI + 31) / 32 * 32, NHI = TT * NHF * 4, NHP = (NHI + 31) / 32 * 32;
    for (int i = tid; i < NLP + NHP; i += 256) {
      if (i < NLP) {
        if (i < NLI) {
          const int r = i >> 2, j = i & 3, t = r / NL, c = r % NL;
          float d[32];
          local_cand<W::QM>(xs + t * XP + CPF * j, sgf + (c >= 2 ? c - 2 : SGF_ONE) * SGP + SCP * j, c > 0, d,
                     cl + r * CBP + 32 * j, sl + r * 4 + j);
          if constexpr (NL > 1) {
            float* dst = dl + r * XP + CPF * j;
#pragma unroll
            for (int v = 0; v < 8; ++v)
              *reinterpret_cast<float4*>(dst + 4 * v) = make_float4(d[4 * v], d[4 * v + 1], d[4 * v + 2], d[4 * v + 3]);
#pragma unroll
            for (int a = 0; a < 32; ++a) d[a] = __fmul_rn(__fmul_rn(d[a], d[a]), co[CPF * j + a]);
            el[j * (S::RLP) + r] = tree_reg32(d);
          }
        }
      } else if constexpr (NHF > 0) {
        const int hq = i - NLP;
        const bool act = hq < NHI;
        const int row = act ? hq >> 2 : 0, b = hq & 3;
        const int m = 1 + row % NHF, t = row / NHF;
        float v[32];
        head_cand_q<W::QM>(xs + t * XP, sgf + (m >= 2 ? m - 2 : SGF_ONE) * SGP, b, v, ch + (t * NHF + m - 1) * CBP + 32 * b,
                    sh + (t * NHF + m - 1) * 4 + b, act);
        const float e = vhead_err_q(v, co, b);
        if (act && b == 0) eh[t * NH + m] = e;
      }
    }
    __syncthreads();
    // ---- local choice (exact errors: strict first argmin)
    if (tid < TT * 4) {
      const int t = tid >> 2, j = tid & 3;
      int best = 0;
      if constexpr (NL > 1) {
        float be = el[j * (S::RLP) + t * NL];
#pragma unroll
        for (int c = 1; c < NL; ++c) {
          const float e = el[j * (S::RLP) + t * NL + c];
          if (e < be) { be = e; best = c; }
        }
      }
      ls[tid] = static_cast<uint8_t>(best);
    }
    __syncthreads();
    if constexpr (NHF > 0) {  // base head error (4 lanes per token), then the head choice
      if (w == 0) {
        const int t = lane >> 2, b = lane & 3;  // TT = 8 tokens x 4 quarters = 32 lanes
        const float* src = dl + (t * NL + ls[t * 4 + b]) * XP + CPF * b;
        float v[32];
#pragma unroll
        for (int a = 0; a < 32; a += 4) {
          const float4 f = *reinterpret_cast<const float4*>(src + a);
          v[a] = f.x; v[a + 1] = f.y; v[a + 2] = f.z; v[a + 3] = f.w;
        }
        const float e = vhead_err_q(v, co, b);
        if (b == 0) eh[t * NH] = e;
      }
      __syncthreads();
      if (tid < TT) {
        int best = 0;
        float be = eh[tid * NH];
#pragma unroll
        for (int m = 1; m < NH; ++m)
          if (eh[tid * NH + m] < be) { be = eh[tid * NH + m]; best = m; }
        hs[tid] = best;
      }
    } else {
      if (tid < TT) hs[tid] = 0;
    }
    __syncthreads();
    // ---- D: page writes.  V^T codes: an aligned run of 8 consecutive slots (one 8-token group of a page) is written
    // as whole words; otherwise per token with atomic nibble updates
    auto code_of = [&](int t, int dd) -> int {
      const int best = hs[t];
      return best == 0 ? cl[(t * NL + ls[t * 4 + (dd >> 5)]) * CBP + dd] : ch[(t * NHF + best - 1) * CBP + dd];
    };
    const bool group = gflag[0] != 0;
    if constexpr (W::QM == 2) {  // INT2: the word of (dd, dd + 8) built by the lanes dd (r = 0) and dd + 8 (r = 1)
      if (group && tid < D) {
        const long long s0 = slot_of(p, t0, h);
        const int dd = tid;
        uint32_t wd = 0;
#pragma unroll
        for (int t = 0; t < TT; ++t) wd |= static_cast<uint32_t>(code_of(t, dd)) << i2_vshift(dd & ~8, t);
        const uint32_t w8 = __shfl_down_sync(FULL, wd, 8);
        if (!(dd & 8)) {
          const long long page = s0 / P;
          *reinterpret_cast<uint32_t*>(p.pool + (page * p.Hkv + h) * L::CHUNK + L::V_OFF + i2_vword(dd, static_cast<int>(s0 % P), P)) = wd | (w8 << 4);
        }
      }
    } else if (group && tid < D) {
      const long long s0 = slot_of(p, t0, h);
      const int dd = tid;
      uint32_t wd = 0;
#pragma unroll
      for (int t = 0; t < TT; ++t) {
        uint32_t m, bits;
        place_bits<W::QM>(code4<W::QM>(code_of(t, dd)), t & 1, t >> 1, m, bits);
        wd |= bits;
      }
      const long long page = s0 / P;
      const int c8 = static_cast<int>(s0 % P) / 8;
      *reinterpret_cast<uint32_t*>(p.pool + (page * p.Hkv + h) * L::CHUNK + L::V_OFF + vt_word_off(dd, c8, P)) = wd;
    }
    for (int t = w; t < ntok; t += 8) {
      const long long slot = slot_of(p, t0 + t, h);
      if (slot < 0) continue;
      const int best = hs[t];
      const int g = lane >> 3;
      const int lc = best == 0 ? ls[t * 4 + g] : 0;
      uint32_t sraw = best == 0 ? sl[(t * NL + lc) * 4 + g] : sh[(t * NHF + best - 1) * 4 + g];
      const int mode = W::RED ? (best ? 2 : 0) : best;
      if (CB == CB_REDUCED && mode == 0 && lc == 1) sraw |= 0x8000u;
      const long long page = slot / P;
      const int pos = static_cast<int>(slot % P);
      uint8_t* chunk = p.pool + (page * p.Hkv + h) * L::CHUNK;
      const int c8 = pos / 8, r8 = pos % 8, ii = r8 >> 1;
      const bool isB = r8 & 1;
      if (!group && W::QM == 2) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int dd = lane * 4 + i;
          uint32_t* wp = reinterpret_cast<uint32_t*>(chunk + L::V_OFF + i2_vword(dd, pos, P));
          const int sh = i2_vshift(dd, pos);
          atomicAnd(wp, ~(3u << sh));
          atomicOr(wp, static_cast<uint32_t>(code_of(t, dd)) << sh);
        }
      } else if (!group) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int dd = lane * 4 + i;
          uint32_t* wp = reinterpret_cast<uint32_t*>(chunk + L::V_OFF + vt_word_off(dd, c8, P));
          uint32_t m, bits;
          place_bits<W::QM>(code4<W::QM>(code_of(t, dd)), isB, ii, m, bits);
          atomicAnd(wp, ~m);
          atomicOr(wp, bits);
        }
      }
      if ((lane & 7) == 0) {
        const int slot8 = isB ? 4 + ii : ii;
        const size_t off = L::VS_OFF + (g * P + 8 * c8 + slot8) * 2;
        if (cb_k2(CB) && g == 0) {  // key-only 2x2: bit 15 of the group-0 V scale belongs to the key kernel (F bit), so
          // the two kernels may run concurrently: bits 0..14 by atomics on the 32-bit word (the other half = another token)
          uint32_t* wp = reinterpret_cast<uint32_t*>(chunk + (off & ~size_t(3)));
          const int sh = 8 * static_cast<int>(off & 2);
          atomicAnd(wp, ~(0x7FFFu << sh));
          atomicOr(wp, (sraw & 0x7FFFu) << sh);
        } else {
          *reinterpret_cast<uint16_t*>(chunk + off) = static_cast<uint16_t>(sraw);
        }
      }
      if (lane == 0) {
        if constexpr (CB == CB_REDUCED) {
          uint32_t* fw = reinterpret_cast<uint32_t*>(chunk + L::FV_OFF);
          const int bit = 8 * (pos / 8) + ((pos & 1) ? 4 : 0) + ((pos % 8) >> 1);
          if (mode == 2) atomicOr(fw, 1u << bit);
          else atomicAnd(fw, ~(1u << bit));
        } else if constexpr (!W::VID) {
          constexpr int LB = CbSpec<CB>::LB;
          uint32_t sw = static_cast<uint32_t>(mode) << (4 * LB);
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) sw |= static_cast<uint32_t>(best == 0 ? ls[t * 4 + jj] : 0) << (LB * jj);
          *reinterpret_cast<uint16_t*>(chunk + L::SWV_OFF + 2 * pos) = static_cast<uint16_t>(sw);
        }
      }
    }
  }  // tiles
}

// ================================================================================================== keys
// exact regeneration of a candidate in the reference's fp32 order (the k_cb_write building blocks, lane-parallel)
// local candidate c of 32-vector j of padded row x: lane a -> d_a
template <int A = 0>
DEVI float regen_local(const float* x, const float* sgf, int sgp, int j, int c, int lane) {
  const float z = x[CPF * j + lane];
  const float s = sgf[(c >= 2 ? c - 2 : SGF_ONE) * sgp + SCP * j + lane];
  float y = z * s;
  if (c > 0) y = fwht32_lane(y, lane);
  float deq, q;
  unsigned sb;
  if constexpr (A) quant_val_a<A == 2 ? 2 : 4>(y, min32(y), max32(y), deq, q, sb);
  else quant_val(y, max32(fabsf(y)), deq, q, sb);
  const float rec = c > 0 ? fwht32_lane(deq, lane) : deq;
  return fmaf(rec, s, -z);
}
// whole-head candidate m >= 1 (H128 S_{m-2}): lane l -> d[4 l .. 4 l + 3]
template <int A = 0>
DEVI void regen_head(const float* x, const float* sgf, int sgp, int m, int lane, float (&d)[4]) {
  const float* sg = sgf + (m >= 2 ? m - 2 : SGF_ONE) * sgp;
  float z[4], s[4], y[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    z[i] = x[xo(4 * lane + i)];
    s[i] = sg[so(4 * lane + i)];
    y[i] = z[i] * s[i];
  }
  fwht128_row(y, lane);
  float am = fmaxf(fmaxf(fabsf(y[0]), fabsf(y[1])), fmaxf(fabsf(y[2]), fabsf(y[3])));
  float mn = fminf(fminf(y[0], y[1]), fminf(y[2], y[3])), mx = fmaxf(fmaxf(y[0], y[1]), fmaxf(y[2], y[3]));
#pragma unroll
  for (int o = 1; o < 8; o <<= 1) {
    am = fmaxf(am, __shfl_xor_sync(FULL, am, o));
    mn = fminf(mn, __shfl_xor_sync(FULL, mn, o));
    mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, o));
  }
  float deq[4], q;
  unsigned sb;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if constexpr (A) quant_val_a<A == 2 ? 2 : 4>(y[i], mn, mx, deq[i], q, sb);
    else quant_val(y[i], am, deq[i], q, sb);
  }
  fwht128_row(deq, lane);
#pragma unroll
  for (int i = 0; i < 4; ++i) d[i] = fmaf(deq[i], s[i], -z[i]);
}
// local (32-vector j, lane a holds d_a): u_a = seq_k C[32j + a][32j + k] d_k
DEVI float exact_local_err(const float* CT, float da, int j, int lane) {
  float cv[32];  // all loads in flight before the sequential chain (C is mostly an L2 hit: L1 holds little next to
                 // the shared memory)
#pragma unroll
  for (int kk = 0; kk < 32; ++kk) cv[kk] = __ldg(CT + (32 * j + kk) * D + 32 * j + lane);
  float u = 0.f;
#pragma unroll
  for (int kk = 0; kk < 32; ++kk) {
    const float pr = __fmul_rn(cv[kk], __shfl_sync(FULL, da, kk));
    u = kk == 0 ? pr : __fadd_rn(u, pr);
  }
  return tree32(__fmul_rn(da, u));
}
// one block b of the exact head error: U_b,i = seq_{k in block b} C[i][k] e_k for i = 4 lane .. 4 lane + 3 (loads in
// two batches of 16 rows ahead of the chain) -> out[4 lane ..]
DEVI void exact_head_U(const float* CT, const float* e, int b, int lane, float* out) {
  float U[4];
#pragma unroll
  for (int hf = 0; hf < 2; ++hf) {
    float4 cv[16];
#pragma unroll
    for (int kk = 0; kk < 16; ++kk) cv[kk] = __ldg(reinterpret_cast<const float4*>(CT + (32 * b + 16 * hf + kk) * D) + lane);
#pragma unroll
    for (int kk = 0; kk < 16; ++kk) {
      const float ek = e[32 * b + 16 * hf + kk];
      const float c4[4] = {cv[kk].x, cv[kk].y, cv[kk].z, cv[kk].w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float pr = __fmul_rn(c4[i], ek);
        U[i] = (hf == 0 && kk == 0) ? pr : __fadd_rn(U[i], pr);
      }
    }
  }
  *reinterpret_cast<float4*>(out + 4 * lane) = make_float4(U[0], U[1], U[2], U[3]);
}
// d' = 2^sigma d as fp16 hi / lo into the swizzled operand rows (chunks c0 .. c0 + 3 of row r)
DEVI void store_hilo(uint8_t* hi, uint8_t* lo, int r, int c0, const float* d, float sg) {
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    uint32_t wh[4], wl[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float a = d[8 * q + 2 * k] * sg, b = d[8 * q + 2 * k + 1] * sg;
      wh[k] = pack_h2(a, b);
      const float2 hv = unpack_h2(wh[k]);
      wl[k] = pack_h2(a - hv.x, b - hv.y);  // exact differences
    }
    const int off = r * 256 + swz(c0 + q, r) * 16;
    *reinterpret_cast<uint4*>(hi + off) = make_uint4(wh[0], wh[1], wh[2], wh[3]);
    *reinterpret_cast<uint4*>(lo + off) = make_uint4(wl[0], wl[1], wl[2], wl[3]);
  }
}
DEVI void bar_local() { asm volatile("bar.sync 1, 128;\n" ::: "memory"); }  // warps 0-3
DEVI float hi_at(const uint8_t* hi, int r, int col) {  // fp16 hi element (row r, column col)
  return __half2float(*reinterpret_cast<const __half*>(hi + r * 256 + swz(col >> 3, r) * 16 + (col & 7) * 2));
}

template <int CB, int P>
__global__ void __launch_bounds__(256, 2) k_cbw3_k(const CbWriteParams p) {
  using W = W3<CB>;
  using S = KSm<CB>;
  using L = CbLay<CB, P>;
  constexpr int TT = S::TT, NL = S::NL, NH = S::NH, NHF = S::NHF, RL = S::RL;
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int g = lane >> 2, tq = lane & 3;
  const int h = blockIdx.x;
  const int ntiles = (p.T + TT - 1) / TT;
  if (static_cast<int>(blockIdx.y) >= ntiles) return;
  uint8_t* c1s = sm + S::C1_OFF;
  uint8_t* hi = sm + S::HI_OFF;
  uint8_t* lo = sm + S::LO_OFF;
  float* xs = reinterpret_cast<float*>(sm + S::X_OFF);
  int8_t* cl = reinterpret_cast<int8_t*>(sm + S::CL_OFF);
  int8_t* chd = reinterpret_cast<int8_t*>(sm + S::CH_OFF);
  uint16_t* sl = reinterpret_cast<uint16_t*>(sm + S::SL_OFF);
  uint16_t* sh = reinterpret_cast<uint16_t*>(sm + S::SH_OFF);
  float* el = reinterpret_cast<float*>(sm + S::EL_OFF);
  float* nl = el + 4 * S::RLP;
  float* eh = reinterpret_cast<float*>(sm + S::EH_OFF);
  float* nh = eh + S::RH * 4;
  float* sig = reinterpret_cast<float*>(sm + S::SG_OFF);
  uint8_t* ls = reinterpret_cast<uint8_t*>(sm + S::LS_OFF);
  int* hs = reinterpret_cast<int*>(sm + S::HS_OFF);
  int* di = reinterpret_cast<int*>(sm + S::DI_OFF);  // [0..7] head D, [8 + 8 j ..] block j D_j
  int* fc = reinterpret_cast<int*>(sm + S::FC_OFF);
  int* fi = reinterpret_cast<int*>(sm + S::FI_OFF);
  int* ft = reinterpret_cast<int*>(sm + S::FT_OFF);
  float* fe = reinterpret_cast<float*>(sm + S::FE_OFF);
  int* gi = reinterpret_cast<int*>(sm + S::GI_OFF);
  int* gt = reinterpret_cast<int*>(sm + S::GT_OFF);
  float* ge = reinterpret_cast<float*>(sm + S::GE_OFF);
  float* sgf = reinterpret_cast<float*>(sm + S::SGF_OFF);
  float* scr = reinterpret_cast<float*>(sm + S::SCR_OFF) + w * 128;  // this warp's scratch row
  uint8_t* xr = sm + S::XR_OFF;
  float* dbuf = reinterpret_cast<float*>(sm + S::DB_OFF);
  const float* aux = p.beta + h * AUX;
  {  // C1 | F_head | F_blocks -> smem: row n (256 B) as 16 chunks of 16 B, chunk c at c ^ (n & 7)
    const __half* C1g = static_cast<const __half*>(p.C1) + static_cast<size_t>(h) * CROWS * D;
    for (int c = tid; c < CROWS * 16; c += 256) {
      const int n = c >> 4, k = c & 15;
      cp_async16(c1s + n * 256 + ((k ^ (n & 7)) << 4), C1g + n * 128 + k * 8);
    }
    cp_async_commit();
    if (tid < 40) di[tid] = static_cast<int>(aux[8 + tid]);
    build_sign_table(sgf, SGP, p, tid);
    issue_x(xr, p, blockIdx.y * TT, min(TT, p.T - static_cast<int>(blockIdx.y) * TT), TT, h, tid);
  }
  const float* CT = p.CT + static_cast<size_t>(h) * D * D;
  const float bH = aux[0], slack = aux[5];
  int stat_l = 0, stat_lf = 0, stat_h = 0, stat_hf = 0;
  unsigned long long prof[8] = {0, 0, 0, 0, 0, 0, 0, 0}, pt = clock64();
  int pph = 0;

  for (int tile = blockIdx.y; tile < ntiles; tile += gridDim.y) {
    const int t0 = tile * TT;
    const int ntok = min(TT, p.T - t0);
    W3_PROF_MARK(0);
    cp_async_wait0();
    __syncthreads();  // x (and C1) landed; the previous tile is done with the smem
    const float* kmu = p.kmu != nullptr ? p.kmu + h * D : nullptr;
    convert_x(xs, xr, ntok, TT, tid, kmu);
    if (tid < 4) fc[tid] = 0;
    for (int t = w; t < TT; t += 8) {  // 2^sigma from the raw row: max|d'| <= ||x||_2 2^sigma < 2^15.5
      float am = 0.f;
      if (t < ntok) {
        const uint2 u = *reinterpret_cast<const uint2*>(xr + t * 256 + lane * 8);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        float2 a2 = a;
        if (kmu != nullptr) {  // the centered row (= xs)
          const float4 m = *reinterpret_cast<const float4*>(kmu + lane * 4);
          a2 = make_float2(__fsub_rn(a.x, m.x), __fsub_rn(a.y, m.y));
          b = make_float2(__fsub_rn(b.x, m.z), __fsub_rn(b.y, m.w));
        }
        am = fmaxf(fmaxf(fabsf(a2.x), fabsf(a2.y)), fmaxf(fabsf(b.x), fabsf(b.y)));
      }
      am = max32(am);
      int ex = 0;
      frexpf(am, &ex);
      ex = max(min(11 - ex, 100), -100);
      if (lane == 0) sig[t] = am > 0.f ? ldexpf(1.f, ex) : 1.f;
    }
    __syncthreads();
    if (tile + static_cast<int>(gridDim.y) < ntiles) {  // prefetch the next tile
      const int n0 = (tile + gridDim.y) * TT;
      issue_x(xr, p, n0, min(TT, p.T - n0), TT, h, tid);
    }
    W3_PROF_MARK(1);
    // ---- A..C1, two warp groups: warps 0-3 generate the local candidates, evaluate their forms on the tensor cores,
    // choose (with the exact fallback) and assemble the base rows, while warps 4-7 generate the whole-head candidates
    constexpr int NLI = S::RLV * 4, NLP = (NLI + 31) / 32 * 32, NHI = TT * NHF * 4, NHP = (NHI + 31) / 32 * 32;
    if (w < 4) {
      for (int i = tid; i < NLP; i += 128) {  // local: thread per (row r = t NL + c, vector j), lanes = 8 rows x 1 vector
        const int r = (i >> 5) * 8 + (i & 7), j = (i >> 3) & 3;
        if (r < S::RLV) {
          const int t = r / NL, c = r % NL;
          float d[32];
          local_cand<W::QM>(xs + t * XP + CPF * j, sgf + (c >= 2 ? c - 2 : SGF_ONE) * SGP + SCP * j, c > 0, d,
                     cl + r * CBP + 32 * j, sl + r * 4 + j);
          store_hilo(hi, lo, r, 4 * j, d, sig[t]);
        }
      }
      bar_local();
      // ---- B1: local forms on the tensor cores (task = (16-row tile, 32-vector j)): C1 block (j, j) + correction tile
      for (int task = w; task < (W3_NOTC ? 0 : (RL / 16) * 4); task += 4) {
        const int mt = task >> 2, j = task & 3;
        const int r0 = mt * 16 + g, r1 = r0 + 8;
        float acc[5][4];
  #pragma unroll
        for (int a = 0; a < 5; ++a) acc[a][0] = acc[a][1] = acc[a][2] = acc[a][3] = 0.f;
        uint32_t ah[2][4], al[2][4];
  #pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          const int k0 = 32 * j + 16 * kk;
          const int ar = mt * 16 + (lane & 15), ac = (k0 >> 3) + (lane >> 4);
          ldsm_x4(ah[kk], hi + ar * 256 + swz(ac, ar) * 16);
          ldsm_x4(al[kk], lo + ar * 256 + swz(ac, ar) * 16);
  #pragma unroll
          for (int np = 0; np < 2; ++np) {
            const int n = 32 * j + 16 * np + ((lane >> 4) << 3) + (lane & 7);
            const int kc = (k0 >> 3) + ((lane >> 3) & 1);
            uint32_t b[4];
            ldsm_x4(b, c1s + n * 256 + ((kc ^ (n & 7)) << 4));
            mma16816(acc[2 * np], ah[kk], b[0], b[1]);
            mma16816(acc[2 * np + 1], ah[kk], b[2], b[3]);
          }
        }
        {  // correction tile: rows F_j (C1 rows 136 + 8 j ..), k chunks 4 j .. 4 j + 3
          const int n = 136 + 8 * j + (lane & 7), kc = 4 * j + (lane >> 3);
          uint32_t b[4];
          ldsm_x4(b, c1s + n * 256 + ((kc ^ (n & 7)) << 4));
          mma16816(acc[4], ah[0], b[0], b[1]);
          mma16816(acc[4], ah[1], b[2], b[3]);
        }
        float e0 = 0.f, e1 = 0.f, q0 = 0.f, q1 = 0.f;
  #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int kk = nt >> 1, hf = nt & 1;
          const float2 h0 = unpack_h2(ah[kk][2 * hf]), h1 = unpack_h2(ah[kk][2 * hf + 1]);
          const float2 l0 = unpack_h2(al[kk][2 * hf]), l1 = unpack_h2(al[kk][2 * hf + 1]);
          e0 = fmaf(fmaf(2.f, l0.x, h0.x), acc[nt][0], e0);
          e0 = fmaf(fmaf(2.f, l0.y, h0.y), acc[nt][1], e0);
          e1 = fmaf(fmaf(2.f, l1.x, h1.x), acc[nt][2], e1);
          e1 = fmaf(fmaf(2.f, l1.y, h1.y), acc[nt][3], e1);
          const float a0 = h0.x + l0.x, a1 = h0.y + l0.y, b0 = h1.x + l1.x, b1 = h1.y + l1.y;
          q0 = fmaf(a0, a0, fmaf(a1, a1, q0));
          q1 = fmaf(b0, b0, fmaf(b1, b1, q1));
        }
        {
          const int c0 = di[8 + 8 * j + 2 * tq], c1 = di[8 + 8 * j + 2 * tq + 1];
          e0 = fmaf(hi_at(hi, r0, c0), acc[4][0], fmaf(hi_at(hi, r0, c1), acc[4][1], e0));
          e1 = fmaf(hi_at(hi, r1, c0), acc[4][2], fmaf(hi_at(hi, r1, c1), acc[4][3], e1));
        }
  #pragma unroll
        for (int o = 1; o < 4; o <<= 1) {
          e0 += __shfl_xor_sync(FULL, e0, o); e1 += __shfl_xor_sync(FULL, e1, o);
          q0 += __shfl_xor_sync(FULL, q0, o); q1 += __shfl_xor_sync(FULL, q1, o);
        }
        if (tq == 0) {
          el[j * S::RLP + r0] = e0; nl[j * S::RLP + r0] = q0;
          el[j * S::RLP + r1] = e1; nl[j * S::RLP + r1] = q1;
        }
      }
      bar_local();
      // ---- C1: local choice with the bound; near-ties -> exact tasks (item, candidate)
      if (tid < TT * 4) {
        const int t = tid >> 2, j = tid & 3;
        int choice = 0;
        if (t < ntok) {
          const float bl = aux[1 + j];
          float lo_[NL], mh = INFINITY, ea = INFINITY;
  #pragma unroll
          for (int c = 0; c < NL; ++c) {
            const float e = el[j * S::RLP + t * NL + c], b = fmaf(bl, nl[j * S::RLP + t * NL + c], slack);
            lo_[c] = e - b;
            if (e < ea) { ea = e; choice = c; }  // tensor-core argmin (kept unless the exact tasks overrule it)
            mh = fminf(mh, e + b);
          }
          unsigned mask = 0;
  #pragma unroll
          for (int c = 0; c < NL; ++c) mask |= (lo_[c] <= mh ? 1u : 0u) << c;
          if (__popc(mask) == 1) choice = __ffs(mask) - 1;
          else {
            if (mask == 0) mask = (1u << NL) - 1;  // non-finite values: evaluate all
            const int n = __popc(mask);
            const int k = atomicAdd(&fc[0], 1), f0 = atomicAdd(&fc[1], n);
            fi[3 * k] = tid; fi[3 * k + 1] = f0; fi[3 * k + 2] = n;
            int q = f0;
            for (int c = 0; c < NL; ++c)
              if ((mask >> c) & 1u) ft[q++] = tid | (c << 8);
          }
        }
        ls[tid] = static_cast<uint8_t>(choice);
      }
      bar_local();
      if (tid == 0) { stat_l += ntok * 4; stat_lf += fc[0]; }
      if (!W3_NOFB && fc[0] > 0) {  // (uniform) exact local errors, candidate regenerated in the reference order
        const int nt = fc[1];
        for (int k = w; k < nt; k += 4) {
          const int it = ft[k] & 255, c = ft[k] >> 8, t = it >> 2, j = it & 3;
          const float da = regen_local<W::QM>(xs + t * XP, sgf, SGP, j, c, lane);
          const float err = exact_local_err(CT, da, j, lane);
          if (lane == 0) fe[k] = err;
        }
        bar_local();
        if (tid < fc[0]) {  // first index among the exact minima
          const int it = fi[3 * tid], f0 = fi[3 * tid + 1], n = fi[3 * tid + 2];
          int bi = ft[f0] >> 8;
          float be = fe[f0];
          for (int q = 1; q < n; ++q)
            if (fe[f0 + q] < be) { be = fe[f0 + q]; bi = ft[f0 + q] >> 8; }
          ls[it] = static_cast<uint8_t>(bi);
        }
        bar_local();
      }
      for (int i = tid; i < TT * 32; i += 128) {  // base candidate rows = the local winners (hi and lo chunks)
        const int t = i >> 5, j = (i >> 3) & 3, q = (i >> 1) & 3, part = i & 1;
        const int rs = t * NL + ls[t * 4 + j], rd = RL + t * NH, c = 4 * j + q;
        uint8_t* base = part ? lo : hi;
        *reinterpret_cast<uint4*>(base + rd * 256 + swz(c, rd) * 16) =
            *reinterpret_cast<const uint4*>(base + rs * 256 + swz(c, rs) * 16);
      }
    } else {
      for (int hq = tid - 128; hq < NHP; hq += 128) {  // whole-head: 4 quarter lanes per (token, transform)
        const bool act = hq < NHI;
        const int row = act ? hq >> 2 : 0, b = hq & 3;
        const int m = 1 + row % NHF, t = row / NHF;
        float v[32];
        head_cand_q<W::QM>(xs + t * XP, sgf + (m >= 2 ? m - 2 : SGF_ONE) * SGP, b, v, chd + (t * NHF + m - 1) * CBP + 32 * b,
                    sh + (t * NHF + m - 1) * 4 + b, act);
        if (act) store_hilo(hi, lo, RL + t * NH + m, 4 * b, v, sig[t]);
      }
    }
    __syncthreads();
    W3_PROF_MARK(4);
    // ---- B2: head forms (task = (16-row tile, column slice nq of CW = 128 / NQ columns)); slice 0 also takes the
    // correction tile
    {
      constexpr int NQ = W3_HNQ, CW = 128 / NQ, NTL = CW / 8, KTE = CW / 16;
      for (int task = w; task < (W3_NOTC ? 0 : (S::RH / 16) * NQ); task += 8) {
        const int mt = task / NQ, nq = task % NQ;
        const int r0 = RL + mt * 16 + g, r1 = r0 + 8;
        float acc[NTL + 1][4];
#pragma unroll
        for (int a = 0; a <= NTL; ++a) acc[a][0] = acc[a][1] = acc[a][2] = acc[a][3] = 0.f;
        const int ar = RL + mt * 16 + (lane & 15);
        const uint8_t* arow = hi + ar * 256;
        const int ax = ar & 7, hb = lane >> 4;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
          const int k0 = 16 * kk;
          uint32_t af[4];
          ldsm_x4(af, arow + (((2 * kk + hb) ^ ax ^ (kk >= 4 ? 2 : 0)) << 4));
#pragma unroll
          for (int np = 0; np < NTL / 2; ++np) {
            const int n = CW * nq + 16 * np + ((lane >> 4) << 3) + (lane & 7);
            const int kc = (k0 >> 3) + ((lane >> 3) & 1);
            uint32_t b[4];
            ldsm_x4(b, c1s + n * 256 + ((kc ^ (lane & 7)) << 4));
            mma16816(acc[2 * np], af, b[0], b[1]);
            mma16816(acc[2 * np + 1], af, b[2], b[3]);
          }
          if (nq == 0) {
            const int n = 128 + (lane & 7), kc = (k0 >> 3) + ((lane >> 3) & 1);
            uint32_t b[2];
            ldsm_x2(b, c1s + n * 256 + ((kc ^ (lane & 7)) << 4));
            mma16816(acc[NTL], af, b[0], b[1]);
          }
        }
        float e0 = 0.f, e1 = 0.f, q0 = 0.f, q1 = 0.f;
#pragma unroll
        for (int kq = 0; kq < KTE; ++kq) {  // epilogue: hi / lo fragments of the slice's own k-tiles
          const int c = 2 * (KTE * nq + kq) + hb;
          uint32_t fh[4], fl[4];
          ldsm_x4(fh, arow + swz(c, ar) * 16);
          ldsm_x4(fl, lo + ar * 256 + swz(c, ar) * 16);
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            const int nt = 2 * kq + hf;
            const float2 h0 = unpack_h2(fh[2 * hf]), h1 = unpack_h2(fh[2 * hf + 1]);
            const float2 l0 = unpack_h2(fl[2 * hf]), l1 = unpack_h2(fl[2 * hf + 1]);
            e0 = fmaf(fmaf(2.f, l0.x, h0.x), acc[nt][0], e0);
            e0 = fmaf(fmaf(2.f, l0.y, h0.y), acc[nt][1], e0);
            e1 = fmaf(fmaf(2.f, l1.x, h1.x), acc[nt][2], e1);
            e1 = fmaf(fmaf(2.f, l1.y, h1.y), acc[nt][3], e1);
            const float a0 = h0.x + l0.x, a1 = h0.y + l0.y, b0 = h1.x + l1.x, b1 = h1.y + l1.y;
            q0 = fmaf(a0, a0, fmaf(a1, a1, q0));
            q1 = fmaf(b0, b0, fmaf(b1, b1, q1));
          }
        }
        if (nq == 0) {
          const int c0 = di[2 * tq], c1 = di[2 * tq + 1];
          e0 = fmaf(hi_at(hi, r0, c0), acc[NTL][0], fmaf(hi_at(hi, r0, c1), acc[NTL][1], e0));
          e1 = fmaf(hi_at(hi, r1, c0), acc[NTL][2], fmaf(hi_at(hi, r1, c1), acc[NTL][3], e1));
        }
#pragma unroll
        for (int o = 1; o < 4; o <<= 1) {
          e0 += __shfl_xor_sync(FULL, e0, o); e1 += __shfl_xor_sync(FULL, e1, o);
          q0 += __shfl_xor_sync(FULL, q0, o); q1 += __shfl_xor_sync(FULL, q1, o);
        }
        if (tq == 0) {
#pragma unroll
          for (int z = 0; z < 4 / NQ; ++z) {  // partial sums in slots [4]
            const int sl_ = (4 / NQ) * nq + z;
            eh[(r0 - RL) * 4 + sl_] = z ? 0.f : e0; nh[(r0 - RL) * 4 + sl_] = z ? 0.f : q0;
            eh[(r1 - RL) * 4 + sl_] = z ? 0.f : e1; nh[(r1 - RL) * 4 + sl_] = z ? 0.f : q1;
          }
        }
      }
    }
    __syncthreads();
    W3_PROF_MARK(5);
    // ---- C2: head choice with the bound; near-ties -> exact tasks (token, candidate)
    if (tid < TT) {
      const int t = tid;
      int choice = 0;
      if (t < ntok) {
        float lo_[NH], mh = INFINITY, ea = INFINITY;
#pragma unroll
        for (int m = 0; m < NH; ++m) {
          const int r = t * NH + m;
          const float e = (eh[r * 4] + eh[r * 4 + 1]) + (eh[r * 4 + 2] + eh[r * 4 + 3]);
          const float q = (nh[r * 4] + nh[r * 4 + 1]) + (nh[r * 4 + 2] + nh[r * 4 + 3]);
          const float b = fmaf(bH, q, slack);
          lo_[m] = e - b;
          if (e < ea) { ea = e; choice = m; }
          mh = fminf(mh, e + b);
        }
        unsigned mask = 0;
#pragma unroll
        for (int m = 0; m < NH; ++m) mask |= (lo_[m] <= mh ? 1u : 0u) << m;
        if (__popc(mask) == 1) choice = __ffs(mask) - 1;
        else {
          if (mask == 0) mask = (1u << NH) - 1;
          const int n = __popc(mask);
          const int k = atomicAdd(&fc[2], 1), f0 = atomicAdd(&fc[3], n);
          gi[3 * k] = t; gi[3 * k + 1] = f0; gi[3 * k + 2] = n;
          int q = f0;
          for (int m = 0; m < NH; ++m)
            if ((mask >> m) & 1u) gt[q++] = t | (m << 8);
        }
      }
      hs[t] = choice;
    }
    __syncthreads();
    if (tid == 0) { stat_h += ntok; stat_hf += fc[2]; }
    if (!W3_NOFB && fc[2] > 0) {  // (uniform) exact head errors, rounds of 2 tasks: regenerate the candidate rows (warps
      const int nt = fc[3];     // 0, 1), U_b per (task, block) on all 8 warps, combine in the reference order
      for (int r0 = 0; r0 < nt; r0 += 2) {
        if (w < 2 && r0 + w < nt) {
          const int t = gt[r0 + w] & 255, m = gt[r0 + w] >> 8;
          float* dd = dbuf + w * 128;
          if (m == 0) {  // base: the 4 local winners
#pragma unroll 1
            for (int j = 0; j < 4; ++j) dd[32 * j + lane] = regen_local<W::QM>(xs + t * XP, sgf, SGP, j, ls[t * 4 + j], lane);
          } else {
            float d4[4];
            regen_head<W::QM>(xs + t * XP, sgf, SGP, m, lane, d4);
            *reinterpret_cast<float4*>(dd + 4 * lane) = make_float4(d4[0], d4[1], d4[2], d4[3]);
          }
        }
        __syncthreads();
        if (r0 + (w >> 2) < nt) exact_head_U(CT, dbuf + (w >> 2) * 128, w & 3, lane, scr);
        __syncthreads();
        if (w < 2 && r0 + w < nt) {
          const float* dd = dbuf + w * 128;
          const float* U = reinterpret_cast<const float*>(sm + S::SCR_OFF) + 4 * w * 128;
          float pr[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int ii = 4 * lane + i;
            pr[i] = __fmul_rn(dd[ii], __fadd_rn(__fadd_rn(__fadd_rn(U[ii], U[128 + ii]), U[256 + ii]), U[384 + ii]));
          }
          const float err = tree128_row(pr);
          if (lane == 0) ge[r0 + w] = err;
        }
        __syncthreads();
      }
      if (tid < fc[2]) {
        const int t = gi[3 * tid], f0 = gi[3 * tid + 1], n = gi[3 * tid + 2];
        int bi = gt[f0] >> 8;
        float be = ge[f0];
        for (int q = 1; q < n; ++q)
          if (ge[f0 + q] < be) { be = ge[f0 + q]; bi = gt[f0 + q] >> 8; }
        hs[t] = bi;
      }
      __syncthreads();
    }
    W3_PROF_MARK(6);
    // ---- D: page writes (warp per token)
    for (int t = w; t < ntok; t += 8) {
      const long long slot = slot_of(p, t0 + t, h);
      if (slot < 0) continue;
      const int best = hs[t];
      const int gg = lane >> 3;
      const int lc = best == 0 ? ls[t * 4 + gg] : 0;
      const int8_t* src = best == 0 ? cl + (t * NL + lc) * CBP : chd + (t * NHF + best - 1) * CBP;
      uint32_t sraw = best == 0 ? sl[(t * NL + lc) * 4 + gg] : sh[(t * NHF + best - 1) * 4 + gg];
      const int mode = W::RED ? (best ? 2 : 0) : best;
      if (W::RED && mode == 0 && lc == 1) sraw |= 0x8000u;  // vector gg uses H32
      const char4 c4 = *reinterpret_cast<const char4*>(src + 4 * lane);
      const uint32_t code[4] = {code4<W::QM>(c4.x), code4<W::QM>(c4.y), code4<W::QM>(c4.z), code4<W::QM>(c4.w)};
      const long long page = slot / P;
      const int pos = static_cast<int>(slot % P);
      uint8_t* chunk = p.pool + (page * p.Hkv + h) * L::CHUNK;
      uint32_t cb[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) cb[i] = __shfl_down_sync(FULL, code[i], 4);
      const int tg = lane & 7, j = lane >> 3;
      if constexpr (W::QM == 2) {  // INT2 K row (attn_decode_v2.cuh)
        const uint32_t hw = i2_halfword(code, cb), hw1 = __shfl_down_sync(FULL, hw, 8);
        if (tg < 4 && !(j & 1))
          *reinterpret_cast<uint32_t*>(chunk + L::K_OFF + i2_krow(pos) + 8 * tg + 4 * (j >> 1)) = hw | (hw1 << 4);
      } else if (tg < 4) {
        uint32_t wd = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          uint32_t m, bits;
          place_bits<W::QM>(code[i], false, i, m, bits); wd |= bits;
          place_bits<W::QM>(cb[i], true, i, m, bits); wd |= bits;
        }
        *reinterpret_cast<uint32_t*>(chunk + L::K_OFF + pos * 64 + 16 * tg + 4 * j) = wd;
      }
      if ((lane & 7) == 0) *reinterpret_cast<uint16_t*>(chunk + L::KS_OFF + pos * 8 + 2 * gg) = static_cast<uint16_t>(sraw);
      if (lane == 0) {
        if constexpr (cb_k2(CB)) {  // the key's F bit = sign bit of the token's group-0 V scale, set or cleared
          // atomically (the value kernel owns bits 0..14 of that word: the two kernels may run in either order)
          const int c8 = pos / 8, r8 = pos % 8, ii = r8 >> 1, slot8 = (r8 & 1) ? 4 + ii : ii;
          const size_t off = L::VS_OFF + (8 * c8 + slot8) * 2;
          uint32_t* wp = reinterpret_cast<uint32_t*>(chunk + (off & ~size_t(3)));
          if (mode) atomicOr(wp, 0x8000u << (8 * (off & 2)));
          else atomicAnd(wp, ~(0x8000u << (8 * (off & 2))));
        } else if constexpr (CB == CB_REDUCED) {
          uint32_t* fw = reinterpret_cast<uint32_t*>(chunk + L::FK_OFF);
          if (mode == 2) atomicOr(fw, 1u << pos);
          else atomicAnd(fw, ~(1u << pos));
        } else {
          constexpr int LB = CbSpec<CB>::LB;
          uint32_t sw = static_cast<uint32_t>(mode) << (4 * LB);
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) sw |= static_cast<uint32_t>(best == 0 ? ls[t * 4 + jj] : 0) << (LB * jj);
          *reinterpret_cast<uint16_t*>(chunk + L::SWK_OFF + 2 * pos) = static_cast<uint16_t>(sw);
        }
      }
    }
  }
  W3_PROF_MARK(7);
  if (W3_PROF && p.stats && tid == 0 && blockIdx.x == 0) for (int q = 0; q < 8; ++q) atomicAdd(reinterpret_cast<unsigned long long*>(p.stats) + 4 + q, prof[q]);
  if (p.stats && tid == 0) {
    atomicAdd(p.stats + 0, stat_l); atomicAdd(p.stats + 1, stat_lf);
    atomicAdd(p.stats + 2, stat_h); atomicAdd(p.stats + 3, stat_hf);
  }
}

template <int CB, int P>
int write3_kp(const CbWriteParams& p, cudaStream_t st) {
  using W = W3<CB>;
  constexpr int ks = KSm<CB>::TOTAL, vs = VSm<CB>::TOTAL;
  static_assert(ks <= 113 * 1024, "key kernel: 2 CTAs per SM need <= 113 KB of dynamic smem each");
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(k_cbw3_k<CB, P>, cudaFuncAttributeMaxDynamicSharedMemorySize, ks);
    cudaFuncSetAttribute(k_cbw3_v<CB, P>, cudaFuncAttributeMaxDynamicSharedMemorySize, vs);
    attr = true;
  }
  // (key-only 2x2: keys and values own disjoint bits of the group-0 V scale words, any order / concurrent)
  const int ntv = (p.T + W::TTV - 1) / W::TTV;
  // (nchunk_v / nchunk < 0: skip that kernel -- the caller may run the two on concurrent streams)
  if (p.nchunk_v >= 0) k_cbw3_v<CB, P><<<dim3(p.Hkv, p.nchunk_v > 0 ? min(p.nchunk_v, ntv) : ntv), 256, vs, st>>>(p);
  const int ntiles = (p.T + W::TT - 1) / W::TT;
  const int nch = p.nchunk > 0 ? min(p.nchunk, ntiles) : ntiles;
  if (p.nchunk >= 0) k_cbw3_k<CB, P><<<dim3(p.Hkv, nch), 256, ks, st>>>(p);
  return 0;
}
template <int CB>
int write3_k(const CbWriteParams& p, cudaStream_t st) {
  if (p.T <= 0) return 0;
  if (p.page == 16) return write3_kp<CB, 16>(p, st);
  if (p.page == 32) return write3_kp<CB, 32>(p, st);
  return -5;
}

// ================================================================================================== static writer
// k_w3s: static FlagRot INT4 (A = false) / INT4A (A = true) cache writer, CTA = (kv head, tile of 32
// tokens), warp w: rows of kind w / 4 (K, V) for the tokens 8 (w % 4) .. + 7 of the tile, lane = 4 token + quarter b
// (32-vector b of the row, in registers; no shared memory for x).  Per row, in the order of the plugin's v1 path:
//  1. whole-head H128 if hfull[row] (fr_head_fwht: unnormalized Sylvester butterflies h = 1 .. 64 in fp32, the two
//     cross-quarter stages by shuffles, then bf16(v / 16));
//  2. rb-32 block flag mask[row][b]: H32 (k_paged_write's warp_selrot: butterflies, then * fp32(1/sqrt(32)));
//  3. INT4 (quant32: fp16 scale amax / 7) or INT4A (int4a_group: range widened to include 0, scale word with z);
//  4. K: 4 code words + scale per (token, vector); V: codes staged in shared memory, an aligned run of 8 consecutive
//     slots (one V^T word column) written as whole words by the warp that owns those 8 tokens, else atomic nibbles.
// The warps are independent (no block barrier).
template <int A, int P>
__global__ void __launch_bounds__(256) k_w3s(const W3StaticParams p) {
  constexpr int TT = 32;
  constexpr int CBY = A == 2 ? 32 : 64;  // A = 2: INT2 page layout (attn_decode_v2.cuh), 80 P bytes
  constexpr int V_OFF = CBY * P, KS_OFF = 2 * CBY * P, VS_OFF = KS_OFF + 8 * P, CHUNK = KS_OFF + 16 * P;
  __shared__ __align__(16) int8_t vc[TT][128];  // V codes of the tile (token, element)
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int h = blockIdx.x;
  const int kind = w >> 2, g8 = w & 3;
  const int tl = 8 * g8 + (lane >> 2), b = lane & 3;
  const int t = blockIdx.y * TT + tl;
  const bool act = t < p.T;
  const int hr = kind * p.Hkv + h;
  float v[32];
  if (act) {
    const uint4* src = reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(p.kv) + t * p.kv_ts + hr * D + 32 * b);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const uint4 u = src[k];
      const uint32_t uu[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        v[8 * k + 2 * i] = __uint_as_float(uu[i] << 16);
        v[8 * k + 2 * i + 1] = __uint_as_float(uu[i] & 0xffff0000u);
      }
    }
  } else {
#pragma unroll
    for (int a = 0; a < 32; ++a) v[a] = 0.f;
  }
  if (p.hfull != nullptr && __ldg(p.hfull + hr)) {  // warp-uniform
    fwht_reg<32>(v);
    const float s1 = (b & 1) ? -1.f : 1.f, s2 = (b & 2) ? -1.f : 1.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) v[i] = fmaf(v[i], s1, __shfl_xor_sync(FULL, v[i], 1));
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      v[i] = fmaf(v[i], s2, __shfl_xor_sync(FULL, v[i], 2));
      v[i] = __bfloat162float(__float2bfloat16_rn(__fmul_rn(v[i], 0.0625f)));
    }
  }
  if (kind == 0 && act && p.kmu != nullptr) {  // static key centering in the stored basis
    const float4* m4 = reinterpret_cast<const float4*>(p.kmu + h * D + 32 * b);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float4 m = __ldg(m4 + k);
      v[4 * k] = __fsub_rn(v[4 * k], m.x); v[4 * k + 1] = __fsub_rn(v[4 * k + 1], m.y);
      v[4 * k + 2] = __fsub_rn(v[4 * k + 2], m.z); v[4 * k + 3] = __fsub_rn(v[4 * k + 3], m.w);
    }
  }
  if (p.mask != nullptr && __ldg(p.mask + hr * 4 + b)) {
    fwht_reg<32>(v);
#pragma unroll
    for (int a = 0; a < 32; ++a) v[a] = __fmul_rn(v[a], C32);
  }
  uint32_t qw[8];
  unsigned sb;
  if constexpr (A) {
    float t16[16];
#pragma unroll
    for (int a = 0; a < 16; ++a) t16[a] = fminf(v[a], v[a + 16]);
    min_reg<8>(t16);
    const float mn = fminf(t16[0], 0.f);
#pragma unroll
    for (int a = 0; a < 16; ++a) t16[a] = fmaxf(v[a], v[a + 16]);
    max_reg<8>(t16);
    quant32a_core<A == 2 ? 2 : 4>(v, mn, fmaxf(t16[0], 0.f), false, nullptr, qw, &sb);
  } else {
    quant32(v, v, qw, sb);
  }
  const long long slot = act ? p.slots[t] : -1;
  const long long page = slot / P;
  const int pos = static_cast<int>(slot % P);
  uint8_t* chunk = p.pool + (page * p.Hkv + h) * CHUNK;
  auto byte_of = [&](int e) -> int {  // code of element e of this vector (int8, sign extended)
    return static_cast<int>(static_cast<int8_t>((qw[e >> 2] >> (8 * (e & 3))) & 0xFFu));
  };
  if constexpr (A == 2) {
    if (kind == 0) {  // INT2 K row words (tg, jp): groups b = 2 jp (this lane) and 2 jp + 1 (lane + 1)
      uint32_t hw[4];
#pragma unroll
      for (int tg = 0; tg < 4; ++tg) {
        uint32_t ca[4], cbv[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { ca[i] = byte_of(4 * tg + i) & 3; cbv[i] = byte_of(16 + 4 * tg + i) & 3; }
        hw[tg] = i2_halfword(ca, cbv);
      }
      uint32_t h1[4];
#pragma unroll
      for (int tg = 0; tg < 4; ++tg) h1[tg] = __shfl_down_sync(FULL, hw[tg], 1);
      if (slot >= 0) {
        if (!(b & 1)) {
#pragma unroll
          for (int tg = 0; tg < 4; ++tg)
            *reinterpret_cast<uint32_t*>(chunk + i2_krow(pos) + 8 * tg + 4 * (b >> 1)) = hw[tg] | (h1[tg] << 4);
        }
        *reinterpret_cast<uint16_t*>(chunk + KS_OFF + pos * 8 + 2 * b) = static_cast<uint16_t>(sb);
      }
      return;
    }
  }
  if (kind == 0) {
    if (slot >= 0) {
#pragma unroll
      for (int tg = 0; tg < 4; ++tg) {  // word (tg, j = b): A = elements 4 tg + i, B = 16 + 4 tg + i
        uint32_t wd = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          uint32_t m, bits;
          place_bits<A>(code4<A>(byte_of(4 * tg + i)), false, i, m, bits); wd |= bits;
          place_bits<A>(code4<A>(byte_of(16 + 4 * tg + i)), true, i, m, bits); wd |= bits;
        }
        *reinterpret_cast<uint32_t*>(chunk + pos * 64 + 16 * tg + 4 * b) = wd;
      }
      *reinterpret_cast<uint16_t*>(chunk + KS_OFF + pos * 8 + 2 * b) = static_cast<uint16_t>(sb);
    }
    return;
  }
  // ---- V
  *reinterpret_cast<uint4*>(&vc[tl][32 * b]) = make_uint4(qw[0], qw[1], qw[2], qw[3]);
  *reinterpret_cast<uint4*>(&vc[tl][32 * b + 16]) = make_uint4(qw[4], qw[5], qw[6], qw[7]);
  const int c8 = pos / 8, r8 = pos % 8, ii = r8 >> 1;
  const bool isB = r8 & 1;
  if (slot >= 0) {
    const int slot8 = isB ? 4 + ii : ii;
    *reinterpret_cast<uint16_t*>(chunk + VS_OFF + (b * P + 8 * c8 + slot8) * 2) = static_cast<uint16_t>(sb);
  }
  const long long s0 = __shfl_sync(FULL, slot, 0);
  const bool run = __all_sync(FULL, slot == s0 + (lane >> 2)) && s0 >= 0 && (s0 % 8) == 0;  // 8 consecutive slots
  __syncwarp();
  if (A == 2 && run) {  // INT2 whole words: lane (dd = lane + 32 k, r = 0) + lane + 8 (r = 1)
    const long long pg0 = s0 / P;
    uint8_t* ch0 = p.pool + (pg0 * p.Hkv + h) * CHUNK + V_OFF;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int dd = lane + 32 * k;
      uint32_t wd = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) wd |= static_cast<uint32_t>(vc[8 * g8 + i][dd] & 3) << i2_vshift(dd & ~8, i);
      const uint32_t w8 = __shfl_down_sync(FULL, wd, 8);
      if (!(dd & 8)) *reinterpret_cast<uint32_t*>(ch0 + i2_vword(dd, static_cast<int>(s0 % P), P)) = wd | (w8 << 4);
    }
  } else if (A == 2 && slot >= 0) {
#pragma unroll 4
    for (int a = 0; a < 32; ++a) {
      const int dd = 32 * b + a;
      uint32_t* wp = reinterpret_cast<uint32_t*>(chunk + V_OFF + i2_vword(dd, pos, P));
      const int sh = i2_vshift(dd, pos);
      atomicAnd(wp, ~(3u << sh));
      atomicOr(wp, static_cast<uint32_t>(byte_of(a) & 3) << sh);
    }
  } else if (run) {  // whole V^T words: lane -> elements dd = lane + 32 k of the 8 tokens
    const long long pg0 = s0 / P;
    const int cc8 = static_cast<int>(s0 % P) / 8;
    uint8_t* ch0 = p.pool + (pg0 * p.Hkv + h) * CHUNK + V_OFF;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int dd = lane + 32 * k;
      uint32_t wd = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        uint32_t m, bits;
        place_bits<A>(code4<A>(vc[8 * g8 + i][dd]), i & 1, i >> 1, m, bits);
        wd |= bits;
      }
      *reinterpret_cast<uint32_t*>(ch0 + vt_word_off(dd, cc8, P)) = wd;
    }
  } else if (slot >= 0) {
#pragma unroll 4
    for (int a = 0; a < 32; ++a) {
      const int dd = 32 * b + a;
      uint32_t* wp = reinterpret_cast<uint32_t*>(chunk + V_OFF + vt_word_off(dd, c8, P));
      uint32_t m, bits;
      place_bits<A>(code4<A>(byte_of(a)), isB, ii, m, bits);
      atomicAnd(wp, ~m);
      atomicOr(wp, bits);
    }
  }
}
// k_w3s_dec: small writes (decode steps: a few tokens at scattered slots).  One warp per (token, K/V row), lane l holds
// elements 4 l .. 4 l + 3 (the v2 writer's mapping: 8 lanes per 32-group, more threads per token than k_w3s), same
// operations in the same order as fr_head_fwht (+ bf16) and k_paged_write (warp_selrot rb 32, quantizer, layout).
template <int A, int P>
__global__ void __launch_bounds__(128) k_w3s_dec(const W3StaticParams p) {
  constexpr int CBY = A == 2 ? 32 : 64;  // A = 2: INT2 page layout
  constexpr int V_OFF = CBY * P, KS_OFF = 2 * CBY * P, VS_OFF = KS_OFF + 8 * P, CHUNK = KS_OFF + 16 * P;
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
  uint8_t* chunk = p.pool + (page * p.Hkv + head) * CHUNK;
  const uint2 raw = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(p.kv) + t * p.kv_ts + hr * D + lane * 4);
  float v[4] = {__uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xffff0000u), __uint_as_float(raw.y << 16),
                __uint_as_float(raw.y & 0xffff0000u)};
  if (p.hfull != nullptr && __ldg(p.hfull + hr)) {  // fr_head_fwht
    const float s0 = v[0] + v[1], d0 = v[0] - v[1], s1 = v[2] + v[3], d1 = v[2] - v[3];
    v[0] = s0 + s1; v[1] = d0 + d1; v[2] = s0 - s1; v[3] = d0 - d1;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float o = __shfl_xor_sync(FULL, v[i], m);
        v[i] = (lane & m) ? __fsub_rn(o, v[i]) : __fadd_rn(v[i], o);
      }
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = __bfloat162float(__float2bfloat16_rn(__fmul_rn(v[i], 0.0625f)));
  }
  if (!isv && p.kmu != nullptr) {  // static key centering in the stored basis (= k_paged_write)
    const float4 m = *reinterpret_cast<const float4*>(p.kmu + head * D + lane * 4);
    v[0] = __fsub_rn(v[0], m.x); v[1] = __fsub_rn(v[1], m.y); v[2] = __fsub_rn(v[2], m.z); v[3] = __fsub_rn(v[3], m.w);
  }
  if (p.mask != nullptr) {  // warp_selrot, rb 32
    const bool f = __ldg(p.mask + hr * 4 + lane / 8) != 0;
    if (__any_sync(FULL, f)) {
#pragma unroll
      for (int h = 1; h < 32; h <<= 1) {
        if (h < 4) {
#pragma unroll
          for (int i = 0; i < 4; ++i)
            if (!(i & h)) {
              const float a = v[i], c = v[i + h];
              if (f) { v[i] = __fadd_rn(a, c); v[i + h] = __fsub_rn(a, c); }
            }
        } else {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const float o = __shfl_xor_sync(FULL, v[i], h >> 2);
            if (f) v[i] = (lane & (h >> 2)) ? __fsub_rn(o, v[i]) : __fadd_rn(v[i], o);
          }
        }
      }
      if (f) {
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = __fmul_rn(v[i], C32);
      }
    }
  }
  uint32_t code[4], sraw;
  if constexpr (A) {
    float mn = fminf(fminf(v[0], v[1]), fminf(v[2], v[3])), mx = fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3]));
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) {
      mn = fminf(mn, __shfl_xor_sync(FULL, mn, o));
      mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, o));
    }
    mn = fminf(mn, 0.f);
    mx = fmaxf(mx, 0.f);
    constexpr float LV = A == 2 ? 3.f : 15.f;
    const float s = fmaxf(__fmul_rn(__fsub_rn(mx, mn), 1.0f / LV), 1e-12f);
    const float z = rintf(__fdiv_rn(-mn, s));
#pragma unroll
    for (int i = 0; i < 4; ++i) code[i] = static_cast<uint32_t>(fminf(fmaxf(__fadd_rn(rintf(__fdiv_rn(v[i], s)), z), 0.f), LV));
    sraw = scale_word_a<A == 2 ? 2 : 4>(s, static_cast<unsigned>(z));
  } else {
    float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) am = fmaxf(am, __shfl_xor_sync(FULL, am, o));
    float deq, q;
    unsigned sb = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      quant_val(v[i], am, deq, q, sb);
      code[i] = code4<0>(static_cast<int>(q));
    }
    sraw = sb;
  }
  if (A == 2 && !isv) {  // INT2 K row (attn_decode_v2.cuh)
    uint32_t cb[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) cb[i] = __shfl_down_sync(FULL, code[i], 4);
    const uint32_t hw = i2_halfword(code, cb), hw1 = __shfl_down_sync(FULL, hw, 8);
    const int tg = lane & 7, j = lane >> 3;
    if (tg < 4 && !(j & 1)) *reinterpret_cast<uint32_t*>(chunk + i2_krow(pos) + 8 * tg + 4 * (j >> 1)) = hw | (hw1 << 4);
    if ((lane & 7) == 0) *reinterpret_cast<uint16_t*>(chunk + KS_OFF + pos * 8 + 2 * (lane / 8)) = static_cast<uint16_t>(sraw);
  } else if (!isv) {
    uint32_t cb[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) cb[i] = __shfl_down_sync(FULL, code[i], 4);
    const int tg = lane & 7, j = lane >> 3;
    if (tg < 4) {
      uint32_t wd = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        uint32_t m, bits;
        place_bits<A>(code[i], false, i, m, bits); wd |= bits;
        place_bits<A>(cb[i], true, i, m, bits); wd |= bits;
      }
      *reinterpret_cast<uint32_t*>(chunk + pos * 64 + 16 * tg + 4 * j) = wd;
    }
    if ((lane & 7) == 0) *reinterpret_cast<uint16_t*>(chunk + KS_OFF + pos * 8 + 2 * (lane / 8)) = static_cast<uint16_t>(sraw);
  } else {
    const int c8 = pos / 8, r8 = pos % 8, ii = r8 >> 1;
    const bool isB = r8 & 1;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if constexpr (A == 2) {
        const int dd = lane * 4 + i, sh = i2_vshift(dd, pos);
        uint32_t* wp = reinterpret_cast<uint32_t*>(chunk + V_OFF + i2_vword(dd, pos, P));
        atomicAnd(wp, ~(3u << sh));
        atomicOr(wp, code[i] << sh);
        continue;
      }
      uint32_t* wp = reinterpret_cast<uint32_t*>(chunk + V_OFF + vt_word_off(lane * 4 + i, c8, P));
      uint32_t m, bits;
      place_bits<A>(code[i], isB, ii, m, bits);
      atomicAnd(wp, ~m);
      atomicOr(wp, bits);
    }
    if ((lane & 7) == 0) {
      const int slot8 = isB ? 4 + ii : ii;
      *reinterpret_cast<uint16_t*>(chunk + VS_OFF + ((lane / 8) * P + 8 * c8 + slot8) * 2) = static_cast<uint16_t>(sraw);
    }
  }
}
#ifndef W3S_DEC_MAXT
#define W3S_DEC_MAXT 128   // T <= this: k_w3s_dec (decode steps at scattered slots)
#endif
template <int A>
int w3s_k(const W3StaticParams& p, cudaStream_t st) {
  if (p.T <= 0) return 0;
  if (p.path == 2 || (p.path == 0 && p.T <= W3S_DEC_MAXT)) {
    const unsigned grid = static_cast<unsigned>((static_cast<long long>(p.T) * 2 * p.Hkv + 3) / 4);
    if (p.page == 16) k_w3s_dec<A, 16><<<grid, 128, 0, st>>>(p);
    else if (p.page == 32) k_w3s_dec<A, 32><<<grid, 128, 0, st>>>(p);
    else return -5;
    return 0;
  }
  const dim3 grid(p.Hkv, (p.T + 31) / 32);
  if (p.page == 16) k_w3s<A, 16><<<grid, 256, 0, st>>>(p);
  else if (p.page == 32) k_w3s<A, 32><<<grid, 256, 0, st>>>(p);
  else return -5;
  return 0;
}

}  // namespace w3
