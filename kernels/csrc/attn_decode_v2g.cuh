// Generalized static v2 INT4 / INT4A paged decode attention, included by attn_decode_v2.cu inside its
// anonymous namespace after k_attn2.  The kernel is variant 264 of k_attn2 (FP8 PV with the V scale folded into P',
// conflict-free P' tiles, deferred PV wait, bulk-copy page ring, split-K; INT4A: key zero points on the CUDA cores,
// value zero points on the tensor cores) with two generalizations:
//   * GQA groups of any size gqa = PagedParams::gqa: the q heads of a kv head are served in q tiles of GT = 4 or 8
//     heads (template), grid.x = Hkv * ceil(gqa / GT) "virtual heads" vh (adjacent vh of one kv head read the same
//     pages at the same time: L2 hits); the last tile of a kv head may be partial: its padded heads get q = 0 (finite
//     scores, nothing stored).  GT = 8 widens the tensor-core tiles instead of adding CTAs: QK N = 2 GT = 16
//     (head h: column 2 h hi, 2 h + 1 lo), PV N = 4 GT = 32 (2 groups x 8 heads x hi / lo); a thread then holds
//     the scores / outputs of the two heads tig and tig + 4 (hh = 0, 1).
//   * head_dim DD = 128 NH2 (NH2 = 2: 256): the chunk of a (page, kv head) is two 128-d half chunks, each exactly the
//     v2 INT4 chunk (K codes | V^T codes | K scales | V scales of d in [128 s, 128 s + 128)), so the paged writer is
//     the 128-d writer over 2 Hkv half heads (paged_write_launch, PagedWriteParams::hd).  QK runs NKS = 4 NH2 k-steps
//     (scale groups), PV NMT = 2 NH2 m tiles; the INT4A value zero-point product Z = P'^T z uses all 64 rows of its A
//     operand when GT NH2 = 8 (GT = 4, NH2 = 1: rows 32..63 are unused, as in k_attn2).
//   * key-only 2x2 codebook caches (F = A2_KONLY / A2_KONLYA = attn_codebook CB_KONLY / CB_KONLYA, 'cq+f+konly'): the
//     keys of a token carry a per-token choice (per 32-d group I / H32: sign bit of the group's K scale; whole head
//     H_D: sign bit of the token's group-0 V scale), the values are plain INT4 / INT4A (the static PV path above).
//     The inverse key transforms are applied on the query side as in k_cbk: each q head is transformed into its three
//     variants I, H32 (per group), H_D (D = 128 / 256, Sylvester), QK per k-step = one m64n(4 GT)k32 over
//     [hi(I) hi(H32) | hi(H_D) 16 lo(H_D)] x GT heads + one chained m64n(2 GT)k32 adding the unscaled lo(I) lo(H32)
//     to the first 2 GT columns; per (token, group) the column of the token's variant is selected (FSEL); INT4A: the
//     key zero-point term uses the q sum of that variant.  No FlagRot masks (qrot / vrot off).
// Partials: slot = (b * Hkv * NT + vh) * nsplit + split, [GT][DD] floats + [GT] (max, sum); merge ticket per (b, vh).
// Constraints: INT4 layouts, split mode only (no stream-K), GT * NH2 <= 8.

template <int F, int P, int ST, int GT, int NH2>
struct CfgG {
  static_assert(F2<F>::I4, "generalized kernel: INT4 formats");
  static_assert(!F2<F>::ASYM || ATTN2_ZVT, "generalized kernel: INT4A value zero points on the tensor cores only");
  static_assert((GT == 4 || GT == 8) && (NH2 == 1 || NH2 == 2) && GT * NH2 <= 8, "q tile / head dim");
  static constexpr bool KO = F2<F>::KO;  // key-only 2x2 codebook
  static constexpr int NHT = GT / 4;     // q heads per thread (tig + 4 hh)
  static constexpr int NQ = KO ? 6 * GT : 2 * GT;  // Q^T tile columns per k-step
  static constexpr int NQA = KO ? 4 * GT : 2 * GT; // QK accumulator columns (first wgmma N)
  static constexpr int NKS = 4 * NH2;    // QK k-steps = 32-d scale groups
  static constexpr int NG = NKS;
  static constexpr int NMT = 2 * NH2;    // PV m tiles (64 d)
  static constexpr int DD = 128 * NH2;
  static constexpr int NPV = 4 * GT;     // PV wgmma N
  static constexpr int NCORE = NPV / 8;  // 8-column core groups per P' tile
  using L = Lay<F, P>;                   // one 128-d half chunk
  static constexpr int HCH = L::CHUNK;
  static constexpr int CHUNK = NH2 * HCH;
  static constexpr int NPB = L::NPB;
  // shared-memory page stride (INT2: padded to 64 mod 128 B, as Lay::PSTR; INT4: the chunk)
  static constexpr int PSTR = F2<F>::B2 ? CHUNK + (192 - CHUNK % 128) % 128 : CHUNK;
  static constexpr int STAGE = NPB * PSTR;
  static constexpr int QT_BYTES = NKS * NQ * 32;
  static constexpr int PLBO = 144, PSBO = 288, PTILE = NCORE * PSBO;  // (k_attn2 MODE bit 3 layout)
  static constexpr int PT_BYTES = 2 * NMT * PTILE;                    // tiles [ks][m] of one block
  static constexpr int QT_OFF = ST * STAGE;
  static constexpr int PT_OFF = QT_OFF + QT_BYTES;
  static constexpr int SRC_OFF = PT_OFF + 2 * PT_BYTES;
  static constexpr int ZB_OFF = SRC_OFF + (MAX_CHUNK / P) * 8;
  static constexpr int ZB_BYTES = F2<F>::ASYM ? 2 * 512 : 0;
  static constexpr int SMEM = ZB_OFF + ZB_BYTES;
  static constexpr int FIN_BYTES = GT * DD * 4;
  static_assert(FIN_BYTES <= ST * STAGE, "finalize buffer in the ring");
};

// q tile of a CTA: vh = blockIdx.x -> kv head kvh, first q head qh0 = kvh * gqa + GT * qt, nv valid heads
struct QTile {
  int vh, kvh, qh0, nv, HV;
};
template <int GT>
DEVI QTile q_tile(int gqa, int Hkv) {
  const int g = gqa > 0 ? gqa : 4;
  const int nt = (g + GT - 1) / GT;
  QTile t;
  t.vh = blockIdx.x;
  t.kvh = t.vh / nt;
  const int qt = t.vh - t.kvh * nt;
  t.qh0 = t.kvh * g + GT * qt;
  t.nv = min(GT, g - GT * qt);
  t.HV = Hkv * nt;
  return t;
}

template <int ACC>
DEVI void wg_e4m3_n32(float (&d)[16], const uint32_t (&a)[4], uint64_t desc) {
  if constexpr (ACC)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, "
        "{%16,%17,%18,%19}, %20, 1, 1, 1;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  else
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, "
        "{%16,%17,%18,%19}, %20, 0, 1, 1;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]),
          "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int N, int ACC>
DEVI void wg_e4m3_nn(float (&d)[N / 2], const uint32_t (&a)[4], uint64_t desc) {
  if constexpr (N == 8) wg_e4m3_n8<ACC>(d, a, desc);
  else if constexpr (N == 16) wg_e4m3_n16<ACC>(d, a, desc);
  else wg_e4m3_n32<ACC>(d, a, desc);
}
// "+f" accumulators with an immediate scale-d SD (SD = 0: fresh result, the old register values are ignored but stay
// live): every accumulator register is defined before the pipeline stage (initialized once before the main loop), so
// ptxas never inserts definitions of accumulator registers inside a stage (C7515 serialized all wgmmas of the GT 8 /
// head 256 instantiations with write-only "=f" accumulators, measured)
template <int SD>
DEVI void wgk_n8(float (&d)[4], const uint32_t (&a)[4], uint64_t desc) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, %9, 1, 1;\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "n"(SD));
}
template <int SD>
DEVI void wgk_n16(float (&d)[8], const uint32_t (&a)[4], uint64_t desc) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, %13, 1, 1;\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "n"(SD));
}
template <int SD>
DEVI void wgk_n32(float (&d)[16], const uint32_t (&a)[4], uint64_t desc) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, "
      "{%16,%17,%18,%19}, %20, %21, 1, 1;\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "n"(SD));
}
template <int N, int SD>
DEVI void wgk(float (&d)[N / 2], const uint32_t (&a)[4], uint64_t desc) {
  if constexpr (N == 8) wgk_n8<SD>(d, a, desc);
  else if constexpr (N == 16) wgk_n16<SD>(d, a, desc);
  else wgk_n32<SD>(d, a, desc);
}
template <int SD>
DEVI void wgk_n8_ss(float (&d)[4], uint64_t adesc, uint64_t bdesc) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, %4, %5, %6, 1, 1;\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "l"(adesc), "l"(bdesc), "n"(SD));
}
#ifndef ATTN2G_KEEP  // "+f" accumulators (wgk*) instead of a write-only first wgmma: bit 0 QK, 1 PV, 2 Z; bit 3: only
#define ATTN2G_KEEP 0  // for the head-dim-256 INT4A instantiation (compile experiments)
#endif

// split partial / merge / bf16 store for the generalized kernels: x[hh][s] = this CTA's unnormalized output of q head
// w + 4 hh of the tile at d = 128 s + 4 lane .. + 3, M / L its softmax max / sum; returns normally.
template <int GT, int NH2>
DEVI void merge_finalize_g(const PagedParams& p, float (&x)[GT / 4][NH2][4], const float (&M)[GT / 4],
                           const float (&L)[GT / 4], int b, const QTile& qt, int wslot, const PartMap& pm, int nparts,
                           int* last_flag_p) {
  constexpr int NHT = GT / 4, DD = 128 * NH2;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  int& last_flag = *last_flag_p;
  if (nparts == 1) {
#pragma unroll
    for (int hh = 0; hh < NHT; ++hh) {
      const float il = L[hh] > 0.f ? 1.f / L[hh] : 0.f;
#pragma unroll
      for (int s = 0; s < NH2; ++s)
#pragma unroll
        for (int i = 0; i < 4; ++i) x[hh][s][i] *= il;
    }
  } else {
#pragma unroll
    for (int hh = 0; hh < NHT; ++hh) {
      const int h = w + 4 * hh;
#pragma unroll
      for (int s = 0; s < NH2; ++s)
        reinterpret_cast<float4*>(p.opart)[(static_cast<size_t>(wslot) * GT + h) * (DD / 4) + 32 * s + lane] =
            make_float4(x[hh][s][0], x[hh][s][1], x[hh][s][2], x[hh][s][3]);
      if (lane == 0) reinterpret_cast<float2*>(p.mlpart)[wslot * GT + h] = make_float2(M[hh], L[hh]);
    }
    __syncthreads();
    if (tid == 0) {
      int old;
      asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(p.counter + b * qt.HV + qt.vh) : "memory");
      last_flag = old == nparts - 1;
    }
    __syncthreads();
    if (!last_flag) return;
#pragma unroll
    for (int hh = 0; hh < NHT; ++hh) merge_warp_g<16 / NH2, GT, NH2>(p, pm, nparts, w + 4 * hh, x[hh]);
    if (tid == 0) p.counter[b * qt.HV + qt.vh] = 0;
  }
#pragma unroll
  for (int hh = 0; hh < NHT; ++hh) {
    const int h = w + 4 * hh;
    if (h >= qt.nv) continue;  // (warp-uniform) padded head of a partial tile
#pragma unroll
    for (int s = 0; s < NH2; ++s) {
      if (p.vrot) warp_selrot(x[hh][s], p.vmask + qt.kvh * (DD / p.rb) + s * (128 / p.rb), p.rb, lane);
      __nv_bfloat162 o01 = __floats2bfloat162_rn(x[hh][s][0], x[hh][s][1]), o23 = __floats2bfloat162_rn(x[hh][s][2], x[hh][s][3]);
      uint2 ov;
      ov.x = *reinterpret_cast<uint32_t*>(&o01);
      ov.y = *reinterpret_cast<uint32_t*>(&o23);
      *reinterpret_cast<uint2*>(static_cast<__nv_bfloat16*>(p.out) + b * p.o_bs + (qt.qh0 + h) * DD + 128 * s + lane * 4) = ov;
    }
  }
}

// BF16 staging window of the generalized kernels: grid (Hkv * NT, tail_nslot, B), as k_tail_v2
template <int GT, int NH2>
__global__ void __launch_bounds__(128) k_tail_g(const PagedParams p) {
  constexpr int NHT = GT / 4, DD = 128 * NH2;
  extern __shared__ __align__(16) uint8_t smt[];
  __shared__ float tml[2 * GT];
  __shared__ int last_flag;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const QTile qt = q_tile<GT>(p.gqa, p.Hkv);
  const int j = 0, b = blockIdx.z;
  const int tl = min(p.tail_len[b], TAIL_MAX);
  const int len = p.seq_lens[b];
  const int ns_b = len > 0 ? max(1, (len + p.chunk - 1) / p.chunk) : 0;
  float* o = reinterpret_cast<float*>(smt);  // [GT][DD] | scratch
  const __nv_bfloat16* qh = static_cast<const __nv_bfloat16*>(p.q) + b * p.q_bs + qt.qh0 * DD;
  const __nv_bfloat16* tb = static_cast<const __nv_bfloat16*>(p.tail) + (p.tail_idx ? p.tail_idx[b] : b) * p.tail_bs;
  tail_window_g<GT, NH2>(qh, qt.nv, p.qscale, tb + qt.kvh * DD, tb + (p.Hkv + qt.kvh) * DD, p.tail_ts, tl, o + GT * DD, o, tml);
  float x[NHT][NH2][4], M[NHT], L[NHT];
#pragma unroll
  for (int hh = 0; hh < NHT; ++hh) {
    const int h = w + 4 * hh;
    M[hh] = tml[2 * h];
    L[hh] = tml[2 * h + 1];
#pragma unroll
    for (int s = 0; s < NH2; ++s) {
#pragma unroll
      for (int i = 0; i < 4; ++i) x[hh][s][i] = o[h * DD + 128 * s + lane * 4 + i];
      if (p.vmask != nullptr && tl > 0) warp_selrot(x[hh][s], p.vmask + qt.kvh * (DD / p.rb) + s * (128 / p.rb), p.rb, lane);
    }
  }
  const int base = (b * qt.HV + qt.vh) * p.nsplit;
  const PartMap pm{base, 1, 0, ns_b + p.tail_nslot, 0};
  merge_finalize_g<GT, NH2>(p, x, M, L, b, qt, base + ns_b + j, pm, ns_b + p.tail_nslot, &last_flag);
}

// unnormalized Sylvester butterflies of a 128-vector held by a warp (lane l: elements 4 l .. 4 l + 3) over blocks of n
// (n = 32: per 32-d group, n = 128: the whole half)
DEVI void ko_butterflies(float (&x)[4], int lane, int n) {
#pragma unroll
  for (int h = 1; h < 128; h <<= 1) {
    if (h >= n) break;
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
}
#ifndef ATTN2G_X  // compile experiments only (wrong results): bit 0 no slow path, 1 no zero-point tile, 2 no Z sums
#define ATTN2G_X 0
#endif
// min CTAs per SM of __launch_bounds__: GT 8 static (INT4 / INT4A): 3 (168 registers, no spills, measured: 180 -> 166
// registers, 2 -> 3 CTAs / SM); everything else 1 (GT 4: 125-128 registers = 4 CTAs / SM anyway; key-only GT 8 and
// head 256 spill under a 168-register cap)
#ifndef ATTN2G_MINB
#define ATTN2G_MINB 0
#endif
template <int F, int GT, int NH2>
constexpr int g_minb() { return ATTN2G_MINB ? ATTN2G_MINB : (GT == 8 && NH2 == 1 && !F2<F>::KO) ? 3 : 1; }
#ifdef ATTN2G_E1
constexpr bool defined_E1 = true;
#else
constexpr bool defined_E1 = false;
#endif
template <int F, int P, int ST, int GT, int NH2>
__global__ void __launch_bounds__(128, g_minb<F, GT, NH2>()) k_attn2g(const PagedParams p) {
  using C = CfgG<F, P, ST, GT, NH2>;
  using L = Lay<F, P>;
  constexpr int NHT = C::NHT, NQ = C::NQ, NQA = C::NQA, NKS = C::NKS, NG = C::NG, NMT = C::NMT, DD = C::DD, NCORE = C::NCORE;
  constexpr bool KO = C::KO;
  constexpr int KEEPM = (ATTN2G_KEEP & 8) ? ((NH2 == 2 && F2<F>::ASYM) ? ATTN2G_KEEP : 0) : ATTN2G_KEEP;
  constexpr bool KEEP_QK = KEEPM & 1, KEEP_PV = KEEPM & 2, KEEP_Z = KEEPM & 4;
  constexpr int HCH = C::HCH, CHUNK = C::CHUNK, NPB = C::NPB;
  constexpr bool ASYM = F2<F>::ASYM;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ __align__(8) uint64_t mbar[ST];
  __shared__ float red_m[NWARP][GT], red_l[NWARP][GT], sfac_s[GT];
  __shared__ float qs_s[KO ? 3 : 1][GT][NG];  // INT4A: 2^15 * sum over group j of the QK operand of head h (KO: per variant)
  __shared__ float zred[NG][GT][2];   // INT4A: value zero-point sums [group][head][hi / lo]
  __shared__ int red_e[NWARP][NG];
  __shared__ int last_flag;

  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, gid = lane >> 2, tig = lane & 3;
  const uint32_t sbase = smem_u32(smem);
  const uint32_t qt_s = sbase + C::QT_OFF, pt = sbase + C::PT_OFF;
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  const QTile qtl = q_tile<GT>(p.gqa, p.Hkv);
  const int kvh = qtl.kvh, split = blockIdx.y, b = blockIdx.z;
  const int c0 = split * p.chunk;
  const int len = p.seq_lens[b];
  unsigned long long ea[ST - 1];
  if (w < NPB && lane == 0) {
    int pg[ST - 1];
#pragma unroll
    for (int s = 0; s < ST - 1; ++s) {
      const int pi = s * NPB + w;
      const bool inb = p.bt_stride > 0 ? c0 / P + pi < p.bt_stride : c0 + pi * P < len;
      pg[s] = (pi * P < p.chunk && inb) ? __ldg(p.block_table + static_cast<size_t>(b) * p.bt_stride + c0 / P + pi) : 0;
    }
    const int c1e = min(len, c0 + p.chunk);
#pragma unroll
    for (int s = 0; s < ST - 1; ++s)
      ea[s] = c0 + (s * NPB + w) * P < c1e
                  ? reinterpret_cast<unsigned long long>(p.pool + (static_cast<uint64_t>(pg[s]) * static_cast<uint32_t>(p.Hkv) + kvh) * CHUNK)
                  : 0ull;
  }
  const int c1 = min(len, c0 + p.chunk);
  const int ns_b = max(1, (len + p.chunk - 1) / p.chunk);
  if (split >= ns_b || len <= 0) return;
  const int nloc = (c1 - c0 + BLK - 1) / BLK;
  // q of heads w + 4 hh of the tile (this lane's 4 elements per 128-d half), zero for padded heads
  uint2 qraw[NHT][NH2];
#pragma unroll
  for (int hh = 0; hh < NHT; ++hh)
#pragma unroll
    for (int s = 0; s < NH2; ++s)
      qraw[hh][s] = (w + 4 * hh < qtl.nv)
                        ? *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(p.q) + b * p.q_bs +
                                                          (qtl.qh0 + w + 4 * hh) * DD + 128 * s + lane * 4)
                        : make_uint2(0u, 0u);

  // ---------------------------------------------------------------- prologue
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < ST; ++s) mbar_init(smem_u32(&mbar[s]), NPB);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  for (int i = tid; i < (C::QT_BYTES + 2 * C::PT_BYTES) / 16; i += 128)
    reinterpret_cast<uint4*>(smem + C::QT_OFF)[i] = make_uint4(0, 0, 0, 0);
  if constexpr (ASYM)
    for (int i = tid; i < C::ZB_BYTES / 16; i += 128) reinterpret_cast<uint4*>(smem + C::ZB_OFF)[i] = make_uint4(0, 0, 0, 0);
  __syncthreads();
  const uint32_t mbar0 = smem_u32(&mbar[0]);
  const uint32_t dst_w = opaque(sbase + w * C::PSTR);
  if (w < NPB && lane == 0) {
#pragma unroll
    for (int s = 0; s < ST - 1; ++s) {
      if (s < nloc) {
        const uint32_t mb = mbar0 + 8 * s;
        mbar_expect_tx(mb, ea[s] != 0ull ? CHUNK : 0);
        if (ea[s] != 0ull) bulk_g2s(dst_w + s * C::STAGE, reinterpret_cast<const void*>(ea[s]), CHUNK, mb);
      }
    }
  }
  const int ntok = max(c1 - c0, 0), nblk = (ntok + BLK - 1) / BLK;
  const uint8_t* poolh = p.pool + static_cast<size_t>(kvh) * CHUNK;
  {
    unsigned long long* srcs = reinterpret_cast<unsigned long long*>(smem + C::SRC_OFF);
    const int* bt = p.block_table + static_cast<size_t>(b) * p.bt_stride + c0 / P;
    const int npg = (ntok + P - 1) / P;
    constexpr int NI = MAX_CHUNK / P / 128;
    int pgv[NI];
#pragma unroll
    for (int k = 0; k < NI; ++k) pgv[k] = tid + 128 * k < npg ? __ldg(bt + tid + 128 * k) : 0;
#pragma unroll
    for (int k = 0; k < NI; ++k) {
      const int i = tid + 128 * k;
      if (i < npg) srcs[i] = reinterpret_cast<unsigned long long>(poolh + static_cast<uint64_t>(pgv[k]) * (static_cast<uint32_t>(p.Hkv) * CHUNK));
    }
  }
  const int npg_all = (ntok + P - 1) / P;
  const uint32_t srcw = opaque(sbase + C::SRC_OFF + 8 * w);
  auto issue = [&](int nb) {
    const int s = static_cast<int>(static_cast<uint32_t>(nb) % ST);
    const uint32_t mb = mbar0 + 8 * s;
    const bool has = nb * NPB + w < npg_all;
    mbar_expect_tx(mb, has ? CHUNK : 0);
    if (has) {
      unsigned long long src;
      asm volatile("ld.shared.u64 %0, [%1];\n" : "=l"(src) : "r"(srcw + 8 * NPB * nb));
      bulk_g2s(dst_w + s * C::STAGE, reinterpret_cast<const void*>(src), CHUNK, mb);
    }
  };
  __syncthreads();  // page ids staged

  // q prologue: heads w + 4 hh: FlagRot (per 128-d half), softmax scale, per-head power-of-two prescale, hi / lo
  // (KO: the three variants I, H32, H_D of every head, one prescale over all three)
#pragma unroll
  for (int hh = 0; hh < NHT; ++hh) {
    const int h = w + 4 * hh;
    constexpr int NV = KO ? 3 : 1;
    float x[NV][NH2][4];
#pragma unroll
    for (int s = 0; s < NH2; ++s) {
      const uint2 raw = qraw[hh][s];
      x[0][s][0] = __uint_as_float(raw.x << 16); x[0][s][1] = __uint_as_float(raw.x & 0xffff0000u);
      x[0][s][2] = __uint_as_float(raw.y << 16); x[0][s][3] = __uint_as_float(raw.y & 0xffff0000u);
      if (!KO && p.qrot) warp_selrot(x[0][s], p.kmask + kvh * (DD / p.rb) + s * (128 / p.rb), p.rb, lane);
#pragma unroll
      for (int i = 0; i < 4; ++i) x[0][s][i] *= p.qscale;
    }
    if constexpr (KO) {
#pragma unroll
      for (int s = 0; s < NH2; ++s)
#pragma unroll
        for (int i = 0; i < 4; ++i) x[1][s][i] = x[2][s][i] = x[0][s][i];
      // H32 per 32-d group (8 lanes) and H_D: Sylvester butterflies, then the orthonormal scale
#pragma unroll
      for (int s = 0; s < NH2; ++s) {
        ko_butterflies(x[1][s], lane, 32);
        ko_butterflies(x[2][s], lane, 128);
#pragma unroll
        for (int i = 0; i < 4; ++i) x[1][s][i] *= 0.17677669529663687f;  // 1 / sqrt(32)
      }
      if constexpr (NH2 == 2) {  // H_256 = H_2 (x) H_128: (a + b, a - b) of the two halves
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float a = x[2][0][i], b2 = x[2][1][i];
          x[2][0][i] = a + b2;
          x[2][1][i] = a - b2;
        }
      }
#pragma unroll
      for (int s = 0; s < NH2; ++s)
#pragma unroll
        for (int i = 0; i < 4; ++i) x[2][s][i] *= NH2 == 2 ? 0.0625f : 0.08838834764831845f;  // 1 / sqrt(D)
    }
    float am = 0.f;
#pragma unroll
    for (int vv = 0; vv < NV; ++vv)
#pragma unroll
      for (int s = 0; s < NH2; ++s)
#pragma unroll
        for (int i = 0; i < 4; ++i) am = fmaxf(am, fabsf(x[vv][s][i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(FULL, am, o));
    const int ea2 = am > 0.f ? static_cast<int>((__float_as_uint(am) >> 23) & 0xFF) - 127 : 0;
    const float kq = __uint_as_float(static_cast<uint32_t>(127 + 6 - ea2) << 23);
#pragma unroll
    for (int vv = 0; vv < NV; ++vv)
#pragma unroll
    for (int s = 0; s < NH2; ++s) {
      // KO: I / H32 lo unscaled (chained onto the hi accumulators), H_D lo x 16 (as k_cbk)
      const bool klo = KO && vv < 2;
      uint32_t hi = 0, lo = 0;
#pragma unroll
      for (int i = 0; i < 4; i += 2) {
        const float a0 = x[vv][s][i] * kq, a1 = x[vv][s][i + 1] * kq;
        const uint32_t hb = f32x2_e4m3x2(a0, a1);
        const float2 hf = h2f2(e4m3x2_f16x2_lo(hb));
        const float ls = klo ? 1.f : LO_SCALE;
        const uint32_t lb = f32x2_e4m3x2((a0 - hf.x) * ls, (a1 - hf.y) * ls);
        hi |= hb << (8 * i);
        lo |= lb << (8 * i);
      }
      const int d0 = lane * 4, j = 4 * s + d0 / 32, k = d0 % 32;
      // columns: plain 2 h hi, + 1 lo; KO: I / H32 hi at 2 h + vv, their lo at 4 GT + 2 h + vv, H_D 2 GT + 2 h (+ 1)
      const int colh = !KO ? 2 * h : vv < 2 ? 2 * h + vv : 2 * GT + 2 * h;
      const int coll = !KO ? colh + 1 : vv < 2 ? 4 * GT + 2 * h + vv : colh + 1;
      const uint32_t a = qt_s + j * (NQ * 32) + (k / 16) * 128 + (k % 16);
      sts32(a + (colh / 8) * 256 + (colh % 8) * 16, hi);
      sts32(a + (coll / 8) * 256 + (coll % 8) * 16, lo);
      if constexpr (ASYM) {
        const float2 h01 = h2f2(e4m3x2_f16x2_lo(hi)), h23 = h2f2(e4m3x2_f16x2_hi(hi));
        const float2 l01 = h2f2(e4m3x2_f16x2_lo(lo)), l23 = h2f2(e4m3x2_f16x2_hi(lo));
        float sq = ((h01.x + h01.y) + (h23.x + h23.y)) + ((l01.x + l01.y) + (l23.x + l23.y)) * (klo ? 1.f : LO_INV);
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) sq += __shfl_xor_sync(FULL, sq, o);
        if ((lane & 7) == 0) qs_s[vv][h][j] = sq * 32768.f;
      }
    }
    if (lane == 0) sfac_s[h] = F2<F>::FV / kq;
  }
  fence_async_smem();
  __syncthreads();
  float sfac[NHT];
  float QS[NHT][NG];  // (KO: the zero-point weights are read from shared memory per block: 3 variants)
#pragma unroll
  for (int hh = 0; hh < NHT; ++hh) {
    sfac[hh] = sfac_s[tig + 4 * hh];
#pragma unroll
    for (int j = 0; j < NG; ++j) QS[hh][j] = 0.f;
    if constexpr (ASYM && !KO) {
#if !defined(ATTN2_NO_ZP) && !defined(ATTN2_NO_KZP)
#pragma unroll
      for (int j = 0; j < NG; ++j) QS[hh][j] = qs_s[0][tig + 4 * hh][j];
#endif
    }
  }

  // ---------------------------------------------------------------- per-thread constants (offsets in half 0; half s at + s HCH)
  const int bq = gid & 1, ipq = (gid >> 1) & 1, qbq = gid >> 2;
  const int tau0 = 16 * w + 8 * qbq + 4 * ipq + bq, tau1 = tau0 + 2;
  constexpr int PS = C::PSTR;
  const uint32_t kro0 = opaque((tau0 / P) * PS + L::K_OFF + krow_off<F>(tau0 % P) + L::KTB * tig);
  const uint32_t kro1 = opaque((tau1 / P) * PS + L::K_OFF + krow_off<F>(tau1 % P) + L::KTB * tig);
  const uint32_t kso0 = opaque((tau0 / P) * PS + L::KS_OFF + (tau0 % P) * 8);
  const uint32_t kso1 = opaque((tau1 / P) * PS + L::KS_OFF + (tau1 % P) * 8);
  const uint64_t qd0 = sdesc(qt_s);
  const uint64_t pdesc0 = sdesc_ls(pt, C::PLBO, C::PSBO);
  uint32_t vo[2];
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int tb = 32 * u + 8 * tig;
    vo[u] = opaque((tb / P) * PS + L::V_OFF + ((w * 8 + gid) * (P / 8) + (tb % P) / 8) * L::VE);
  }
  const int cp0 = (tau0 % P) % 8;
  const uint32_t vso = opaque((tau0 / P) * PS + L::VS_OFF + (((tau0 % P) / 8) * 8 + ((cp0 & 1) ? 4 + (cp0 >> 1) : (cp0 >> 1))) * 2);
  const int k0p = 16 * bq + 4 * (2 * (w & 1) + qbq) + 2 * ipq;
  const uint32_t pwo = opaque((w >> 1) * NMT * C::PTILE + (k0p / 16) * C::PLBO + (2 * tig) * 16 + (k0p % 16));
  float O[NMT][NHT][2];
#pragma unroll
  for (int m = 0; m < NMT; ++m)
#pragma unroll
    for (int hh = 0; hh < NHT; ++hh) { O[m][hh][0] = 0.f; O[m][hh][1] = 0.f; }
  float lsum[NHT], mref[NHT];
#pragma unroll
  for (int hh = 0; hh < NHT; ++hh) { lsum[hh] = 0.f; mref[hh] = NEG_INF; }
  // INT4A value zero points: Z of the block in flight; row half rh of warp w = N core i = 2 w + rh of the k-step's P'
  // tiles -> group zg[rh], heads zh[rh] + (gid >> 1), hi / lo gid & 1; the lanes with tig == zg >> 1 hold its column
  float zd[4] = {0.f, 0.f, 0.f, 0.f};
  // their sums over the CTA's blocks (online-softmax rescaled as O) are kept per thread in shared memory: a register
  // accumulator made ptxas serialize all wgmmas for GT 8 / NH2 2 (C7515, measured)
  float zs4[4] = {0.f, 0.f, 0.f, 0.f};  // (all four entries summed: no lane-dependent select on the accumulators)
  int zg[2], zh[2];
  bool zon[2];
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    const int i = 2 * w + rh, m = i / NCORE, col = 8 * (i % NCORE);
    zg[rh] = 2 * m + col / (2 * GT);
    zh[rh] = (col % (2 * GT)) / 2;
    zon[rh] = i < NH2 * GT;  // useful row groups: 2 NH2 tiles x GT / 2 cores (GT = 4, NH2 = 1: warps 0, 1 only)
  }
  // zero-point tile builder: warp 3, lanes < 4 NG: group lane >> 2, k-step (lane >> 1) & 1, k half lane & 1
  const int zgl = lane >> 2;
  const uint32_t zbsrc = opaque(((32 * ((lane >> 1) & 1)) / P) * PS + (zgl >> 2) * HCH + L::VS_OFF + (zgl & 3) * P * 2 +
                                (((32 * ((lane >> 1) & 1)) % P) / 8) * 16 + 8 * (lane & 1));
  const uint32_t zbdst = sbase + C::ZB_OFF + ((lane >> 1) & 1) * 256 + (lane & 1) * 128 + (zgl & 7) * 16;
  const uint64_t zbdesc0 = sdesc_ls(sbase + C::ZB_OFF, 128, 256);
  int eref[NG];
  uint32_t inv2[NG];
#pragma unroll
  for (int g = 0; g < NG; ++g) { eref[g] = EREF_NONE; inv2[g] = 0u; }
  float pacc[NMT][2 * GT];
  float sacc[NKS][NQA / 2];
  if constexpr (KEEP_PV) {  // ("+f" accumulators: defined once before the loop)
#pragma unroll
    for (int m = 0; m < NMT; ++m)
#pragma unroll
      for (int i = 0; i < 2 * GT; ++i) pacc[m][i] = 0.f;
  }
  if constexpr (KEEP_QK) {
#pragma unroll
    for (int jj = 0; jj < NKS; ++jj)
#pragma unroll
      for (int i = 0; i < NQA / 2; ++i) sacc[jj][i] = 0.f;
  }
  uint32_t ka[NKS][4];
  uint2 kraw0[NH2], kraw1[NH2];
  uint32_t vs0 = 0u;  // KO: raw group-0 V scales of (tau0, tau1) of the block whose QK is in flight (sign = whole-head bit)
  uint32_t va[2][NMT][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int m = 0; m < NMT; ++m)
#pragma unroll
      for (int k = 0; k < 4; ++k) va[i][m][k] = 0u;

  auto qk_mma = [&]() {
    wg_fence();
#pragma unroll
    for (int jj = 0; jj < NKS; ++jj) {
      if constexpr (KEEP_QK) wgk<NQA, 0>(sacc[jj], ka[jj], qd0 + ((jj * NQ * 32) >> 4));
      else wg_e4m3_nn<NQA, 0>(sacc[jj], ka[jj], qd0 + ((jj * NQ * 32) >> 4));
      if constexpr (KO)  // unscaled lo(I) lo(H32) chained onto the first 2 GT columns
        wg_e4m3_nn<2 * GT, 1>(*reinterpret_cast<float(*)[GT]>(sacc[jj]), ka[jj], qd0 + ((jj * NQ * 32 + (GT / 2) * 256) >> 4));
    }
    wg_commit();
  };
  auto qk_decode = [&](int nb) {
    const uint32_t sb = sbase + (static_cast<uint32_t>(nb) % ST) * C::STAGE;
#pragma unroll
    for (int s = 0; s < NH2; ++s) {
      k_frags<F>(sb + kro0 + s * HCH, sb + kro1 + s * HCH, ka, 4 * s);
      kraw0[s] = lds64(sb + kso0 + s * HCH);
      kraw1[s] = lds64(sb + kso1 + s * HCH);
    }
    if constexpr (KO) vs0 = lds32(sb + vso);
  };
  auto pv_accumulate = [&]() {
    if constexpr (ASYM && ATTN2_ZVT) {
#pragma unroll
#pragma unroll
      for (int i = 0; i < 4; ++i) zs4[i] += zd[i];
    }
    const bool g1 = (w >> 1) != 0;  // column group (within the tile) of this warp's d rows
#pragma unroll
    for (int m = 0; m < NMT; ++m)
#pragma unroll
      for (int hh = 0; hh < NHT; ++hh) {
        constexpr int K1 = 4 * NHT;  // core of group 1 = NHT + hh (x 4 accumulators)
        O[m][hh][0] += g1 ? fmaf(pacc[m][K1 + 4 * hh + 1], LO_INV, pacc[m][K1 + 4 * hh]) : fmaf(pacc[m][4 * hh + 1], LO_INV, pacc[m][4 * hh]);
        O[m][hh][1] += g1 ? fmaf(pacc[m][K1 + 4 * hh + 3], LO_INV, pacc[m][K1 + 4 * hh + 2]) : fmaf(pacc[m][4 * hh + 3], LO_INV, pacc[m][4 * hh + 2]);
      }
  };

  mbar_wait(mbar0, 0);
  qk_decode(0);
  qk_mma();
  wg_wait_all();
#pragma unroll
  for (int jj = 0; jj < NKS; ++jj) wg_wait0(sacc[jj]);
#pragma unroll
  for (int k = 0; k < NKS; ++k) pin_regs(ka[k]);
  const int jend = nblk;
  for (int jb = 0; jb < jend; ++jb) {
    const int s = static_cast<int>(static_cast<uint32_t>(jb) % ST);
    const uint32_t sb = sbase + s * C::STAGE;
    const int t0 = c0 + jb * BLK;
    if (w < NPB && lane == 0 && jb + ST - 1 < nloc) issue(jb + ST - 1);
    if constexpr (ASYM && ATTN2_ZVT) {
      if (w == 3 && lane < 4 * NG && !(ATTN2G_X & 2)) {  // zero-point tile of block jb (bytes = e4m3 of z 2^-9), published by barrier B1
        uint32_t zw[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const uint2 x = lds64(sb + zbsrc + (P == 16 ? (c >> 1) * PS + (c & 1) * 16 : 16 * c));
          zw[c] = prmt_b32(x.x, x.y, 0x6420u) & F2<F>::ZM4;
        }
        asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(zbdst + (jb & 1) * 512), "r"(zw[0]), "r"(zw[1]),
                     "r"(zw[2]), "r"(zw[3]) : "memory");
      }
    }
    const bool val0 = t0 + tau0 < c1 && t0 + tau0 >= p.kv_start, val1 = t0 + tau1 < c1 && t0 + tau1 >= p.kv_start;  // kv_start: masked leading tokens (sinks)
    float S0[NHT], S1[NHT];
    {
      float S0b[NHT], S1b[NHT];
#pragma unroll
      for (int hh = 0; hh < NHT; ++hh) { S0[hh] = S1[hh] = S0b[hh] = S1b[hh] = 0.f; }
      float zf0[NG], zf1[NG];
      if constexpr (ASYM) {
#pragma unroll
        for (int s2 = 0; s2 < NH2; ++s2) {
          const uint32_t a = kraw0[s2].x & F2<F>::ZM2, bb = kraw0[s2].y & F2<F>::ZM2;
          const uint32_t c = kraw1[s2].x & F2<F>::ZM2, d = kraw1[s2].y & F2<F>::ZM2;
          zf0[4 * s2 + 0] = __low2float(*reinterpret_cast<const __half2*>(&a)); zf0[4 * s2 + 1] = __high2float(*reinterpret_cast<const __half2*>(&a));
          zf0[4 * s2 + 2] = __low2float(*reinterpret_cast<const __half2*>(&bb)); zf0[4 * s2 + 3] = __high2float(*reinterpret_cast<const __half2*>(&bb));
          zf1[4 * s2 + 0] = __low2float(*reinterpret_cast<const __half2*>(&c)); zf1[4 * s2 + 1] = __high2float(*reinterpret_cast<const __half2*>(&c));
          zf1[4 * s2 + 2] = __low2float(*reinterpret_cast<const __half2*>(&d)); zf1[4 * s2 + 3] = __high2float(*reinterpret_cast<const __half2*>(&d));
        }
      }
      if constexpr (!KO) {
#pragma unroll
      for (int jj = 0; jj < NKS; ++jj) {
        const float2 k0 = kscale_step<F>(kraw0[jj >> 2], jj & 3), k1 = kscale_step<F>(kraw1[jj >> 2], jj & 3);
#pragma unroll
        for (int hh = 0; hh < NHT; ++hh) {
          float& A0 = (jj & 1) ? S0b[hh] : S0[hh];
          float& A1 = (jj & 1) ? S1b[hh] : S1[hh];
          const float x0 = fmaf(sacc[jj][4 * hh + 1], LO_INV, sacc[jj][4 * hh]);
          const float x1 = fmaf(sacc[jj][4 * hh + 3], LO_INV, sacc[jj][4 * hh + 2]);
          if constexpr (ASYM) {
            A0 = fmaf(k0.x, fmaf(-zf0[jj], QS[hh][jj], x0), A0);
            A1 = fmaf(k1.x, fmaf(-zf1[jj], QS[hh][jj], x1), A1);
          } else {
            A0 = fmaf(k0.x, x0, A0);
            A1 = fmaf(k1.x, x1, A1);
          }
        }
      }
      } else {  // key-only 2x2: per (token, group) the column of the token's variant (I / H32 / H_D)
        const bool F0 = (vs0 & 0x8000u) != 0, F1 = (vs0 & 0x80000000u) != 0;
#pragma unroll
        for (int jj = 0; jj < NKS; ++jj) {
          const uint32_t r0 = (jj & 2) ? kraw0[jj >> 2].y : kraw0[jj >> 2].x, r1 = (jj & 2) ? kraw1[jj >> 2].y : kraw1[jj >> 2].x;
          const uint32_t hb = 0x8000u << (16 * (jj & 1));
          const bool H0 = (r0 & hb) != 0, H1 = (r1 & hb) != 0;
          const float2 m0 = h2f2(r0 & 0x7FFF7FFFu), m1 = h2f2(r1 & 0x7FFF7FFFu);
          const float s0 = (jj & 1) ? m0.y : m0.x, s1 = (jj & 1) ? m1.y : m1.x;
#pragma unroll
          for (int hh = 0; hh < NHT; ++hh) {
            float& A0 = (jj & 1) ? S0b[hh] : S0[hh];
            float& A1 = (jj & 1) ? S1b[hh] : S1[hh];
            const float* sa = sacc[jj];
            constexpr int FC = 4 * (GT / 4);  // first float of the H_D columns (core GT / 4 + hh)
            float x0 = F0 ? fmaf(sa[FC + 4 * hh + 1], LO_INV, sa[FC + 4 * hh]) : (H0 ? sa[4 * hh + 1] : sa[4 * hh]);
            float x1 = F1 ? fmaf(sa[FC + 4 * hh + 3], LO_INV, sa[FC + 4 * hh + 2]) : (H1 ? sa[4 * hh + 3] : sa[4 * hh + 2]);
            if constexpr (ASYM) {
#if !defined(ATTN2_NO_ZP) && !defined(ATTN2_NO_KZP)
              const int hq = tig + 4 * hh;
              const float qI = qs_s[0][hq][jj], qH = qs_s[1][hq][jj], qF = qs_s[2][hq][jj];
              x0 = fmaf(-zf0[jj], F0 ? qF : (H0 ? qH : qI), x0);
              x1 = fmaf(-zf1[jj], F1 ? qF : (H1 ? qH : qI), x1);
#endif
            }
            A0 = fmaf(s0, x0, A0);
            A1 = fmaf(s1, x1, A1);
          }
        }
      }
#pragma unroll
      for (int hh = 0; hh < NHT; ++hh) {
        S0[hh] = val0 ? (S0[hh] + S0b[hh]) * sfac[hh] : NEG_INF;
        S1[hh] = val1 ? (S1[hh] + S1b[hh]) * sfac[hh] : NEG_INF;
      }
    }
    uint32_t svraw[NG];
#pragma unroll
    for (int g = 0; g < NG; ++g) svraw[g] = lds32(sb + vso + (g >> 2) * HCH + (g & 3) * P * 2);
    if constexpr (KO) svraw[0] &= 0x7FFF7FFFu;  // (the whole-head key bit)
    const int jq = min(jb + 1, jend - 1);

    // softmax (lazy reference), P' -> smem
    const uint32_t ptb = pt + (jb & 1) * C::PT_BYTES;
    float p0[NHT], p1[NHT];
    bool pflag = false;
    const uint32_t vmask2 = (val0 ? 0xFFFFu : 0u) | (val1 ? 0xFFFF0000u : 0u);
    auto make_p = [&]() {
      uint32_t p2[NHT];
#pragma unroll
      for (int hh = 0; hh < NHT; ++hh) {
        p0[hh] = val0 ? ex2(S0[hh] - mref[hh]) : 0.f;
        p1[hh] = val1 ? ex2(S1[hh] - mref[hh]) : 0.f;
        __half2 ph = __floats2half2_rn(p0[hh], p1[hh]);
        p2[hh] = *reinterpret_cast<uint32_t*>(&ph);
      }
      uint32_t pm = 0u;
#pragma unroll
      for (int g = 0; g < NG; ++g) {
        const uint32_t s2 = hmul2u(svraw[g], inv2[g]) & vmask2;
#pragma unroll
        for (int hh = 0; hh < NHT; ++hh) {
          const uint32_t P2 = hmul2u(p2[hh], s2);
          pm = hmax2u(pm, P2);
          pprime_store(P2, ptb + pwo + (g >> 1) * C::PTILE + ((g & 1) * NHT + hh) * C::PSBO);
        }
      }
      const float2 pmf = h2f2(pm);
      pflag = !(fmaxf(pmf.x, pmf.y) <= 128.f);
    };
    bool flag = false;
#pragma unroll
    for (int hh = 0; hh < NHT; ++hh) flag = flag || (S0[hh] > mref[hh] + TAU) || (S1[hh] > mref[hh] + TAU);
    make_p();
    flag = flag || pflag;
    // PV(n-1) done: nothing in flight -> fragments of PV(n) and QK(n+1)
    wg_wait_all();
#pragma unroll
    for (int m = 0; m < NMT; ++m) wg_wait0(pacc[m]);
    if constexpr (ASYM && ATTN2_ZVT) wg_wait0(zd);
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int m = 0; m < NMT; ++m) pin_regs(va[i][m]);
#pragma unroll
    for (int k = 0; k < NKS; ++k) pin_regs(ka[k]);
    if (KEEP_PV || jb > 0) pv_accumulate();  // (KEEP_PV: pacc / zd start at 0, unconditional = no divergent path)
#pragma unroll
    for (int ks = 0; ks < 2; ++ks)
#pragma unroll
      for (int s2 = 0; s2 < NH2; ++s2) {
        v_frags<F>(sb + vo[ks] + s2 * HCH, va[ks][2 * s2], va[ks][2 * s2 + 1]);
      }
    if (jq != jb) mbar_wait(smem_u32(&mbar[static_cast<uint32_t>(jq) % ST]), (static_cast<uint32_t>(jq) / ST) & 1);
    qk_decode(jq);
    qk_mma();  // QK(n+1) runs under the barrier below
    fence_async_smem();
    const int anyflag = __syncthreads_or(flag);  // B1: P' visible; stage n consumed; slow path?
    if (anyflag && !(ATTN2G_X & 1)) {
      float bm[NHT];
#pragma unroll
      for (int hh = 0; hh < NHT; ++hh) {
        bm[hh] = fmaxf(S0[hh], S1[hh]);
#pragma unroll
        for (int o = 4; o < 32; o <<= 1) bm[hh] = fmaxf(bm[hh], __shfl_xor_sync(FULL, bm[hh], o));
        if (gid == 0) red_m[w][tig + 4 * hh] = bm[hh];
      }
#pragma unroll
      for (int g = 0; g < NG; ++g) {
        const int e0 = static_cast<int>((svraw[g] >> 10) & 31) - 15, e1 = static_cast<int>((svraw[g] >> 26) & 31) - 15;
        int be = max(val0 ? e0 : EREF_NONE, val1 ? e1 : EREF_NONE);
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) be = max(be, __shfl_xor_sync(FULL, be, o));
        if (lane == 0) red_e[w][g] = be;
      }
      __syncthreads();
      float alpha[NHT];
#pragma unroll
      for (int hh = 0; hh < NHT; ++hh) {
        float nm = mref[hh];
#pragma unroll
        for (int ww = 0; ww < NWARP; ++ww) nm = fmaxf(nm, red_m[ww][tig + 4 * hh]);
        alpha[hh] = mref[hh] == NEG_INF ? 0.f : ex2(mref[hh] - nm);
        mref[hh] = nm;
      }
      float beta[NMT];
#pragma unroll
      for (int m = 0; m < NMT; ++m) beta[m] = 1.f;
      float bz[2] = {1.f, 1.f};
#pragma unroll
      for (int g = 0; g < NG; ++g) {
        int ne = eref[g];
#pragma unroll
        for (int ww = 0; ww < NWARP; ++ww) ne = max(ne, red_e[ww][g]);
        ne = (ne == EREF_NONE) ? ne : min(max(ne, -14), 14);
        const float bg = (eref[g] == EREF_NONE || eref[g] - ne < -126) ? 0.f : __uint_as_float(static_cast<uint32_t>(127 + eref[g] - ne) << 23);
        if ((g & 1) == (w >> 1)) beta[g >> 1] = bg;
#pragma unroll
        for (int rh = 0; rh < 2; ++rh)
          if (g == zg[rh]) bz[rh] = bg;
        eref[g] = ne;
        const uint32_t hv = ne == EREF_NONE ? 0u : static_cast<uint32_t>(15 - ne) << 10;
        inv2[g] = hv | (hv << 16);
      }
#pragma unroll
      for (int m = 0; m < NMT; ++m)
#pragma unroll
        for (int hh = 0; hh < NHT; ++hh) { O[m][hh][0] *= alpha[hh] * beta[m]; O[m][hh][1] *= alpha[hh] * beta[m]; }
#pragma unroll
      for (int hh = 0; hh < NHT; ++hh) lsum[hh] *= alpha[hh];
      if constexpr (ASYM && ATTN2_ZVT) {  // zs[rh] belongs to head zh[rh] + (gid >> 1): its alpha from lane gid >> 1
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          const float ah = __shfl_sync(FULL, (NHT > 1 && zh[rh] >= 4) ? alpha[NHT - 1] : alpha[0], gid >> 1);
          zs4[2 * rh] *= ah * bz[rh];
          zs4[2 * rh + 1] *= ah * bz[rh];
        }
      }
      __syncthreads();
      make_p();
      fence_async_smem();
      __syncthreads();
    }
#pragma unroll
    for (int hh = 0; hh < NHT; ++hh) lsum[hh] += p0[hh] + p1[hh];

    // PV(n) (fresh accumulators; added to O after the next wait); QK(n+1) waited for first
    const uint64_t pdb = pdesc0 + (((jb & 1) * C::PT_BYTES) >> 4);
    wg_wait_all();
#pragma unroll
    for (int jj = 0; jj < NKS; ++jj) wg_wait0(sacc[jj]);
#pragma unroll
    for (int k = 0; k < NKS; ++k) pin_regs(ka[k]);
    const uint64_t zbd = zbdesc0 + (((jb & 1) * 512) >> 4);
    wg_fence();
    if constexpr (ASYM && ATTN2_ZVT) {  // Z(n) = P'(n)^T . z(n): A = the P' tiles of k-step ks, B = the zero-point tile
#if !defined(ATTN2_NO_ZP) && !defined(ATTN2_NO_VZP)
      if constexpr (KEEP_Z) wgk_n8_ss<0>(zd, pdb, zbd);
      else wg_e4m3_n8_ss<0>(zd, pdb, zbd);
      wg_e4m3_n8_ss<1>(zd, pdb + ((NMT * C::PTILE) >> 4), zbd + (256 >> 4));
#endif
    }
#pragma unroll
    for (int m = 0; m < NMT; ++m) {
      if constexpr (KEEP_PV) wgk<4 * GT, 0>(pacc[m], va[0][m], pdb + ((0 * NMT + m) * C::PTILE >> 4));
      else wg_e4m3_nn<4 * GT, 0>(pacc[m], va[0][m], pdb + ((0 * NMT + m) * C::PTILE >> 4));
      wg_e4m3_nn<4 * GT, 1>(pacc[m], va[1][m], pdb + ((1 * NMT + m) * C::PTILE >> 4));
    }
    wg_commit();
  }
  wg_wait_all();
#pragma unroll
  for (int m = 0; m < NMT; ++m) wg_wait0(pacc[m]);
  if constexpr (ASYM && ATTN2_ZVT) wg_wait0(zd);
  pv_accumulate();

  // ---------------------------------------------------------------- CTA totals, split partials / merge
  float ofac[NMT];
#pragma unroll
  for (int m = 0; m < NMT; ++m) {
    const int e = (w >> 1) ? eref[2 * m + 1] : eref[2 * m];
    ofac[m] = F2<F>::FV * (e == EREF_NONE ? 0.f : exp2f(static_cast<float>(e)));
  }
#pragma unroll
  for (int hh = 0; hh < NHT; ++hh)
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) lsum[hh] += __shfl_xor_sync(FULL, lsum[hh], o);
  if constexpr (ASYM && ATTN2_ZVT) {  // diagonal lanes publish Z(group, head, hi / lo)
#pragma unroll
    for (int rh = 0; rh < 2; ++rh)
      if (zon[rh] && tig == (zg[rh] >> 1) && !(ATTN2G_X & 4)) zred[zg[rh]][zh[rh] + (gid >> 1)][gid & 1] = (zg[rh] & 1) ? zs4[2 * rh + 1] : zs4[2 * rh];
  }
  __syncthreads();  // ring no longer used
  if constexpr (ASYM) {
#pragma unroll
    for (int m = 0; m < NMT; ++m) {
      const int g = 2 * m + (w >> 1);
#pragma unroll
      for (int hh = 0; hh < NHT; ++hh) {
        const float z = fmaf(zred[g][tig + 4 * hh][1], LO_INV, zred[g][tig + 4 * hh][0]);
        O[m][hh][0] -= z;
        O[m][hh][1] -= z;
      }
    }
  }
  float* fin = reinterpret_cast<float*>(smem);  // [GT][DD]
#pragma unroll
  for (int m = 0; m < NMT; ++m)
#pragma unroll
    for (int hh = 0; hh < NHT; ++hh)
#pragma unroll
      for (int r = 0; r < 2; ++r) fin[(tig + 4 * hh) * DD + 64 * m + 16 * w + gid + 8 * r] = O[m][hh][r] * ofac[m];
  if (gid == 0) {
#pragma unroll
    for (int hh = 0; hh < NHT; ++hh) { red_l[w][tig + 4 * hh] = lsum[hh]; red_m[w][tig + 4 * hh] = mref[hh]; }
  }
  __syncthreads();
  float x[NHT][NH2][4], Mw[NHT], Lw[NHT];
#pragma unroll
  for (int hh = 0; hh < NHT; ++hh) {
    const int h = w + 4 * hh;
#pragma unroll
    for (int s2 = 0; s2 < NH2; ++s2) {
      const float4 f4 = reinterpret_cast<const float4*>(fin + h * DD + 128 * s2)[lane];
      x[hh][s2][0] = f4.x; x[hh][s2][1] = f4.y; x[hh][s2][2] = f4.z; x[hh][s2][3] = f4.w;
    }
    Mw[hh] = red_m[0][h];
    Lw[hh] = red_l[0][h] + red_l[1][h] + red_l[2][h] + red_l[3][h];
  }
  const int nparts = ns_b + p.tail_nslot;
  const int base = (b * qtl.HV + qtl.vh) * p.nsplit;
  const PartMap pm{base, 1, 0, nparts, 0};
  merge_finalize_g<GT, NH2>(p, x, Mw, Lw, b, qtl, base + split, pm, nparts, &last_flag);
}
