// Per-token transform codebook for the INT4 paged KV cache (H20, sm_90a): writer + decode attention.
// Layout and side word: attn_codebook.cuh.  Reference semantics: kernels/codebook_ref.py (= blockflag dyn_quant, rule
// 'cq', rb 32, INT4 g32 with the cache's fp16 scales).
//
// WRITER (k_cb_write): CTA = (kv head, group of tg tokens), 4 warps; K rows, then V rows of the CTA's tokens:
//   A  local candidates: warp j owns 32-vector j (one element per lane): I / H32 / H32 S (and the same on x[PERM] for
//      P), INT4 quantization (group max over the warp), reconstruction, error under the block-diagonal weighting,
//      first-argmin; F / FS on whole rows (4 elements per lane, (token, candidate) items over the warps)
//   B  keys: partial products U_w = C[:, 32w:32w+32] e[32w:32w+32] of every head candidate (warp w, register-blocked
//      over 4 tokens x the candidates, C streamed from L1/L2)
//   C  head errors (one warp per (token, candidate)): u = ((U0 + U1) + U2) + U3, halving-tree sum of e * u
//   D  first-argmin over the head candidates, codes / scales / side info into the page (v2 INT4 element layout)
//   Every product and sum is a separately rounded fp32 operation (__fmul_rn / __fadd_rn: no FMA contraction) in the
//   order of codebook_ref.py, so codes, scales and choices are bit-identical to the PyTorch reference.
#include "attn_codebook.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "attn_cb_common.cuh"

namespace {

// ------------------------------------------------------------------------------------------------- writer smem
template <int CB> struct WSmem {
  static constexpr int NH = CbW<CB>::NH;
  int tg;
  DEVI WSmem(int tg_) : tg(tg_) {}
  // [tg][128] x | [tg][NH][128] e | [4][tg][NH][128] U | [tg][NH][128] codes (int8) | [tg][NH][4] scale bits (u16)
  // | [tg][NH][4] local choice (u8) | [tg][NH] head err
  DEVI size_t xs() const { return 0; }
  DEVI size_t ec() const { return xs() + (size_t)tg * D * 4; }
  DEVI size_t up() const { return ec() + (size_t)tg * NH * D * 4; }
  DEVI size_t cq() const { return up() + (size_t)4 * tg * NH * D * 4; }
  DEVI size_t sb() const { return cq() + (size_t)tg * NH * D; }
  DEVI size_t lc() const { return sb() + (size_t)tg * NH * 4 * 2; }
  DEVI size_t he() const { return lc() + (size_t)tg * NH * 4; }
  DEVI size_t total() const { return he() + (size_t)tg * NH * 4; }
};
template <int CB> size_t wsmem_bytes(int tg) {
  constexpr int NH = CbW<CB>::NH;
  return (size_t)tg * D * 4 + (size_t)tg * NH * D * 4 * 5 + (size_t)tg * NH * D + (size_t)tg * NH * 8 +
         (size_t)tg * NH * 4 + (size_t)tg * NH * 4;
}

template <int CB, int P>
__global__ void __launch_bounds__(128) k_cb_write(const CbWriteParams p) {
  using W = CbW<CB>;
  using L = CbLay<CB, P>;
  constexpr int NH = W::NH, NZ = W::NZ, NL = W::NL;
  extern __shared__ __align__(16) uint8_t sm[];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int h = blockIdx.x;
  const int t0 = blockIdx.y * p.tg;
  const int ntok = min(p.tg, p.T - t0);
  if (ntok <= 0) return;
  const WSmem<CB> S(p.tg);
  float* xs = reinterpret_cast<float*>(sm + S.xs());
  float* ec = reinterpret_cast<float*>(sm + S.ec());
  float* up = reinterpret_cast<float*>(sm + S.up());
  int8_t* cq = reinterpret_cast<int8_t*>(sm + S.cq());
  uint16_t* sb = reinterpret_cast<uint16_t*>(sm + S.sb());
  uint8_t* lc = reinterpret_cast<uint8_t*>(sm + S.lc());
  float* he = reinterpret_cast<float*>(sm + S.he());

  const float* CT = p.CT + static_cast<size_t>(h) * D * D;
  const float* co = p.co + h * D;

  uint32_t kF = 0;  // konly: bit t / 4 = the key of this warp's token t (phase D mapping) uses F
  for (int kind = 0; kind < 2; ++kind) {  // 0: K row (head h), 1: V row (head h)
    const int hr = kind * p.Hkv + h;
    const bool ident = cb_konly(CB) && kind == 1;  // konly values: identity only (no candidate search)
    __syncthreads();  // smem of the previous pass consumed
    for (int i = threadIdx.x; i < ntok * D; i += 128) {
      const int t = i / D, d = i % D;
      const __nv_bfloat16* src = static_cast<const __nv_bfloat16*>(p.kv) + (t0 + t) * p.kv_ts + hr * D + d;
      xs[i] = __bfloat162float(*src);
      if (kind == 0 && p.kmu != nullptr) xs[i] = __fsub_rn(xs[i], __ldg(p.kmu + h * D + d));  // key centering
    }
    __syncthreads();
    // ---------------------------------------------------------------- A1: local candidates (warp w = vector j)
    {
      const int j = w, a = lane;
      for (int t = 0; t < ntok; ++t) {
#pragma unroll
        for (int zm = 0; zm < NZ; ++zm) {
          const int zi = zm ? permi(32 * j + a) : 32 * j + a;  // original coordinate of this lane's element
          const float z = xs[t * D + zi];
          float e[NL], qv[NL];
          unsigned sbits[NL];
#pragma unroll
          for (int c = 0; c < NL; ++c) {
            // S_{c-2} in z coordinates (the 32-slice of the 128-vector for this lane's position)
            const bool ng = c >= 2 ? ((p.sg[c - 2][j] >> a) & 1u) : false;
            const float y = c == 0 ? z : fwht32_lane(sflip(z, ng), lane);
            float deq;
            if constexpr (cb_asym(CB)) quant_val_a<CbQ<CB>::NB>(y, min32(y), max32(y), deq, qv[c], sbits[c]);
            else quant_val(y, max32(fabsf(y)), deq, qv[c], sbits[c]);
            float rec = c == 0 ? deq : fwht32_lane(deq, lane);
            rec = sflip(rec, ng);
            e[c] = __fsub_rn(rec, z);
          }
          float err[NL];
          if (kind == 0) {
            float u[NL];
#pragma unroll 4
            for (int k = 0; k < 32; ++k) {
              const int row = zm ? permi(32 * j + k) : 32 * j + k;
              const float cv = __ldg(CT + row * D + zi);  // C[z_a][z_k]
#pragma unroll
              for (int c = 0; c < NL; ++c) {
                const float pr = __fmul_rn(cv, __shfl_sync(FULL, e[c], k));
                u[c] = k == 0 ? pr : __fadd_rn(u[c], pr);
              }
            }
#pragma unroll
            for (int c = 0; c < NL; ++c) err[c] = tree32(__fmul_rn(e[c], u[c]));
          } else {
            const float cw = __ldg(co + zi);
#pragma unroll
            for (int c = 0; c < NL; ++c) err[c] = tree32(__fmul_rn(__fmul_rn(e[c], e[c]), cw));
          }
          int best = 0;
          float be = err[0];
#pragma unroll
          for (int c = 1; c < NL; ++c)
            if (err[c] < be) { be = err[c]; best = c; }
          if (ident) best = 0;
          float eb = e[0], qb = qv[0];
          unsigned sbb = sbits[0];
#pragma unroll
          for (int c = 1; c < NL; ++c)
            if (best == c) { eb = e[c]; qb = qv[c]; sbb = sbits[c]; }
          // head candidate zm (base / P): e in original coordinates, codes in stored coordinates
          ec[(t * NH + zm) * D + zi] = eb;
          cq[(t * NH + zm) * D + 32 * j + a] = static_cast<int8_t>(qb);
          if (lane == 0) {
            sb[(t * NH + zm) * 4 + j] = static_cast<uint16_t>(sbb);
            lc[(t * NH + zm) * 4 + j] = static_cast<uint8_t>(best);
          }
        }
      }
    }
    // ---------------------------------------------------------------- A2: whole-head candidates F, FS
    {
      constexpr int NF = NH - NZ;  // full: F, FS; reduced: F; final: F, FS_k (6)
      for (int it = ident ? ntok * NF : w; it < ntok * NF; it += 4) {
        const int t = it / NF, c = NZ + it % NF;
        const int hf = c - NZ;  // 0: F, k >= 1: H128 after sign vector k - 1
        // this lane's 4 sign bits of S_{hf-1} (word lane / 8, bits 4 (lane % 8) ..)
        const unsigned sgw = hf >= 1 ? (p.sg[hf - 1][lane >> 3] >> (4 * (lane & 7))) & 15u : 0u;
        float x[4], y[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          x[i] = xs[t * D + 4 * lane + i];
          y[i] = sflip(x[i], (sgw >> i) & 1u);
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
        float deq[4], qv[4];
        unsigned sbits = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          if constexpr (cb_asym(CB)) quant_val_a<CbQ<CB>::NB>(y[i], mn, mx, deq[i], qv[i], sbits);
          else quant_val(y[i], am, deq[i], qv[i], sbits);
        }
        fwht128_row(deq, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float rec = sflip(deq[i], (sgw >> i) & 1u);
          ec[(t * NH + c) * D + 4 * lane + i] = __fsub_rn(rec, x[i]);
          cq[(t * NH + c) * D + 4 * lane + i] = static_cast<int8_t>(qv[i]);
        }
        if ((lane & 7) == 0) {
          sb[(t * NH + c) * 4 + lane / 8] = static_cast<uint16_t>(sbits);
          lc[(t * NH + c) * 4 + lane / 8] = 0;
        }
      }
    }
    __syncthreads();
    // ---------------------------------------------------------------- B: keys, U_w = C[:, 32w:32w+32] e[32w:32w+32]
    if (kind == 0) {  // (konly values: kind 1, no B pass)
      constexpr int TB = W::TB;
      for (int tb = 0; tb < ntok; tb += TB) {
        float u[TB][NH][4];
        const int nt = min(TB, ntok - tb);
#pragma unroll 2
        for (int kk = 0; kk < 32; ++kk) {
          const int k = 32 * w + kk;
          const float4 cv = __ldg(reinterpret_cast<const float4*>(CT + k * D) + lane);  // C[4l .. 4l+3][k]
          const float cc[4] = {cv.x, cv.y, cv.z, cv.w};
#pragma unroll
          for (int tt = 0; tt < TB; ++tt) {
            if (tt < nt) {
#pragma unroll
              for (int c = 0; c < NH; ++c) {
                const float ek = ec[((tb + tt) * NH + c) * D + k];
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                  const float pr = __fmul_rn(cc[m], ek);
                  u[tt][c][m] = kk == 0 ? pr : __fadd_rn(u[tt][c][m], pr);
                }
              }
            }
          }
        }
#pragma unroll
        for (int tt = 0; tt < TB; ++tt)
          if (tt < nt)
#pragma unroll
            for (int c = 0; c < NH; ++c)
              *reinterpret_cast<float4*>(up + ((static_cast<size_t>(w) * p.tg + tb + tt) * NH + c) * D + 4 * lane) =
                  make_float4(u[tt][c][0], u[tt][c][1], u[tt][c][2], u[tt][c][3]);
      }
      __syncthreads();
    }
    // ---------------------------------------------------------------- C: head errors
    for (int it = ident ? ntok * NH : w; it < ntok * NH; it += 4) {
      const int t = it / NH, c = it % NH;
      float pr[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const int i = 4 * lane + m;
        const float e = ec[(t * NH + c) * D + i];
        if (kind == 0) {
          const size_t st = static_cast<size_t>(p.tg) * NH * D, o = (t * NH + c) * D + i;
          const float uu = __fadd_rn(__fadd_rn(__fadd_rn(up[o], up[st + o]), up[2 * st + o]), up[3 * st + o]);
          pr[m] = __fmul_rn(e, uu);
        } else {
          pr[m] = __fmul_rn(__fmul_rn(e, e), __ldg(co + i));
        }
      }
      const float err = tree128_row(pr);
      if (lane == 0) he[t * NH + c] = err;
    }
    __syncthreads();
    // ---------------------------------------------------------------- D: head choice, write the page
    for (int t = w; t < ntok; t += 4) {
      const long long slot = p.slots[static_cast<long long>(t0 + t) * (p.slot_ph ? p.Hkv : 1) + (p.slot_ph ? h : 0)];
      if (slot < 0) continue;
      int best = 0;
      if (!ident) {
        float be = he[t * NH];
#pragma unroll
        for (int c = 1; c < NH; ++c)
          if (he[t * NH + c] < be) { be = he[t * NH + c]; best = c; }
      }
      const int mode = (CB == CB_REDUCED || cb_k2(CB)) ? (best ? 2 : 0) : best;
      if (cb_k2(CB) && kind == 0 && mode) kF |= 1u << (t >> 2);  // full: 0 base 1 P 2 F 3 FS; final: 0 base 1 F 2.. FS_k
      const long long page = slot / P;
      const int pos = static_cast<int>(slot % P);
      uint8_t* chunk = p.pool + (page * p.Hkv + h) * L::CHUNK;
      uint32_t code[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int qi = cq[(t * NH + best) * D + 4 * lane + i];
        code[i] = cb_asym(CB) ? static_cast<uint32_t>(qi) : static_cast<uint32_t>(qi < 0 ? (8 | -qi) : qi);
      }
      const int g = lane / 8;  // this lane's scale group
      uint32_t sraw = sb[(t * NH + best) * 4 + g];
      const int lch = lc[(t * NH + best) * 4 + g];
      if ((CB == CB_REDUCED || cb_k2(CB)) && mode == 0 && lch == 1) sraw |= 0x8000u;  // vector g uses H32
      if (ident && g == 0 && ((kF >> (t >> 2)) & 1u)) sraw |= 0x8000u;  // konly: the token's key uses F
      auto place = [&](uint32_t cc, bool isB, int i, uint32_t& m, uint32_t& bits) {  // v2 INT4 element layout
        const uint32_t s = (cc >> 3) & 1u, mag = cc & 7u;
        if (cb_asym(CB)) {  // v2 int4a: unsigned nibble, A low / B high nibble of byte i
          m = (isB ? 0xF0u : 0x0Fu) << (8 * i);
          bits = cc << (8 * i + (isB ? 4 : 0));
        } else if (!isB) { m = 0x87u << (8 * i); bits = ((s << 7) | mag) << (8 * i); }
        else {
          const int jj = (i + 3) & 3;
          m = (0x08u << (8 * i)) | (0x70u << (8 * jj));
          bits = (s << (8 * i + 3)) | (mag << (8 * jj + 4));
        }
      };
      if constexpr (cb_b2(CB)) {  // INT2 page layout (attn_decode_v2.cuh)
        if (kind == 0) {
          uint32_t cb[4];
      #pragma unroll
          for (int i = 0; i < 4; ++i) cb[i] = __shfl_down_sync(FULL, code[i], 4);
          const uint32_t hw = i2_halfword(code, cb), hw1 = __shfl_down_sync(FULL, hw, 8);
          const int tq = lane & 7, j = lane >> 3;
          if (tq < 4 && !(j & 1))
            *reinterpret_cast<uint32_t*>(chunk + L::K_OFF + i2_krow(pos) + 8 * tq + 4 * (j >> 1)) = hw | (hw1 << 4);
          if ((lane & 7) == 0) *reinterpret_cast<uint16_t*>(chunk + L::KS_OFF + pos * 8 + 2 * g) = static_cast<uint16_t>(sraw);
        } else {
      #pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int dd = lane * 4 + i;
            uint32_t* wp = reinterpret_cast<uint32_t*>(chunk + L::V_OFF + i2_vword(dd, pos, P));
            const int sh = i2_vshift(dd, pos);
            atomicAnd(wp, ~(3u << sh));
            atomicOr(wp, code[i] << sh);
          }
          if ((lane & 7) == 0) {
            const int r8 = pos % 8, slot8 = (r8 & 1) ? 4 + (r8 >> 1) : (r8 >> 1);
            *reinterpret_cast<uint16_t*>(chunk + L::VS_OFF + (g * P + 8 * (pos / 8) + slot8) * 2) = static_cast<uint16_t>(sraw);
          }
        }
      } else if (kind == 0) {
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
        if ((lane & 7) == 0) *reinterpret_cast<uint16_t*>(chunk + L::KS_OFF + pos * 8 + 2 * g) = static_cast<uint16_t>(sraw);
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
        if ((lane & 7) == 0) {
          const int slot8 = isB ? 4 + ii : ii;
          *reinterpret_cast<uint16_t*>(chunk + L::VS_OFF + (g * P + 8 * c8 + slot8) * 2) = static_cast<uint16_t>(sraw);
        }
      }
      if (lane == 0) {
        if constexpr (cb_k2(CB)) {
          // no side word / trailer (sign bits only)
        } else if (cb_kfam(CB) && kind == 1) {
          // key-only family: no value side word
        } else if constexpr (CB != CB_REDUCED) {
          constexpr int LB = W::SF ? CbSpec<CB>::LB : 2;  // bits per vector choice; head mode at bit 4 LB
          uint32_t sw = static_cast<uint32_t>(mode) << (4 * LB);
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) sw |= static_cast<uint32_t>(lc[(t * NH + best) * 4 + jj]) << (LB * jj);
          *reinterpret_cast<uint16_t*>(chunk + (kind ? L::SWV_OFF : L::SWK_OFF) + 2 * pos) = static_cast<uint16_t>(sw);
        } else {
          uint32_t* fw = reinterpret_cast<uint32_t*>(chunk + (kind ? L::FV_OFF : L::FK_OFF));
          const int bit = kind ? 8 * (pos / 8) + ((pos & 1) ? 4 : 0) + ((pos % 8) >> 1) : pos;
          if (mode == 2) atomicOr(fw, 1u << bit);
          else atomicAnd(fw, ~(1u << bit));
        }
      }
    }
  }
}

template <int CB>
int write_k(const CbWriteParams& p, cudaStream_t st) {
  if (p.T <= 0) return 0;
  const size_t smem = wsmem_bytes<CB>(p.tg);
  if (smem > 200 * 1024) return -2;
  dim3 grid(p.Hkv, (p.T + p.tg - 1) / p.tg);
  static size_t set16 = 0, set32 = 0;  // dynamic smem limit raised once per size (not on every launch)
  if (p.page == 16) {
    auto k = k_cb_write<CB, 16>;
    if (smem > set16) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)); set16 = smem; }
    k<<<grid, 128, smem, st>>>(p);
  } else if (p.page == 32) {
    auto k = k_cb_write<CB, 32>;
    if (smem > set32) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)); set32 = smem; }
    k<<<grid, 128, smem, st>>>(p);
  } else {
    return -5;
  }
  return 0;
}


// ------------------------------------------------------------------------------------------------- head dim 256
// Writer of the key-only 2x2 kinds (CB_KONLY / CB_KONLYA) for head dim 256 (Qwen3.5 / 3.8 full-attention
// layers): the 8 key 32-vectors choose I / H32 (block-diagonal C_q weighted error), the head chooses between that base
// and the whole-head H256 (C_q weighted error, the 8 column blocks of C summed left to right), the values are
// quantized without transform.  Page chunk = two 128-d half chunks of the 144 P layout (attn_decode_v2.cuh), half s =
// dims [128 s, 128 s + 128): H32 bit of key group g (0..7) = sign bit of its K scale word, H256 bit = sign bit of the
// token's group-0 V scale word (half 0).  Same fp32 operation order as codebook_ref.encode (bit-exact codes, scale
// words and choices).  CTA = (kv head, tg tokens), 4 warps (vectors w, w + 4; tokens / items over the warps).
constexpr int D2 = 256;
DEVI void fwht256_row(float (&x)[2][4], int lane) {  // Sylvester H256 / 16, lane: dims 4 lane + i of each half
#pragma unroll
  for (int s = 0; s < 2; ++s) {
#pragma unroll
    for (int h = 1; h < 128; h <<= 1) {
      if (h < 4) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
          if (!(i & h)) {
            const float a = x[s][i], c = x[s][i + h];
            x[s][i] = __fadd_rn(a, c);
            x[s][i + h] = __fsub_rn(a, c);
          }
      } else {
        const int lm = h >> 2;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float o = __shfl_xor_sync(FULL, x[s][i], lm);
          x[s][i] = (lane & lm) ? __fsub_rn(o, x[s][i]) : __fadd_rn(x[s][i], o);
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {  // last stage h = 128: the two halves
    const float a = x[0][i], c = x[1][i];
    x[0][i] = __fmul_rn(__fadd_rn(a, c), 0.0625f);
    x[1][i] = __fmul_rn(__fsub_rn(a, c), 0.0625f);
  }
}
size_t wsmem256_bytes(int tg) {
  // xs [tg][256] f32 | ec [tg][2][256] f32 | up [8][tg][2][256] f32 | cq [tg][2][256] i8 | sb [tg][2][8] u16 |
  // lc [tg][2][8] u8 | he [tg][2] f32
  return (size_t)tg * D2 * 4 + (size_t)tg * 2 * D2 * 4 + (size_t)8 * tg * 2 * D2 * 4 + (size_t)tg * 2 * D2 +
         (size_t)tg * 2 * 8 * 2 + (size_t)tg * 2 * 8 + (size_t)tg * 2 * 4;
}

template <int CB, int P>
__global__ void __launch_bounds__(128) k_cb_write256(const CbWriteParams p) {
  static_assert(cb_k2(CB), "head dim 256: key-only 2x2 kinds");
  using L = CbLay<CB, P>;  // one 128-d half chunk
  constexpr int HCH = L::CHUNK, NH = 2, NB = 8;
  extern __shared__ __align__(16) uint8_t sm[];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int h = blockIdx.x;
  const int tg = p.tg;
  const int t0 = blockIdx.y * tg;
  const int ntok = min(tg, p.T - t0);
  if (ntok <= 0) return;
  float* xs = reinterpret_cast<float*>(sm);
  float* ec = xs + tg * D2;
  float* up = ec + tg * NH * D2;
  int8_t* cq = reinterpret_cast<int8_t*>(up + NB * tg * NH * D2);
  uint16_t* sb = reinterpret_cast<uint16_t*>(cq + tg * NH * D2);
  uint8_t* lc = reinterpret_cast<uint8_t*>(sb + tg * NH * NB);
  float* he = reinterpret_cast<float*>(lc + tg * NH * NB);
  const float* CT = p.CT + static_cast<size_t>(h) * D2 * D2;
  uint32_t kF = 0;  // bit t / 4: the key of this warp's token t (phase D mapping) uses H256
  for (int kind = 0; kind < 2; ++kind) {  // 0: K row, 1: V row (no transform)
    const int hr = kind * p.Hkv + h;
    const bool ident = kind == 1;
    __syncthreads();
    for (int i = threadIdx.x; i < ntok * D2; i += 128) {
      const int t = i / D2, d = i % D2;
      xs[i] = __bfloat162float(static_cast<const __nv_bfloat16*>(p.kv)[(t0 + t) * p.kv_ts + hr * D2 + d]);
      if (kind == 0 && p.kmu != nullptr) xs[i] = __fsub_rn(xs[i], __ldg(p.kmu + h * D2 + d));  // key centering
    }
    __syncthreads();
    // ---------------------------------------------------------------- A1: local candidates, vectors w, w + 4
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const int j = w + 4 * jj, zi = 32 * j + lane;
      for (int t = 0; t < ntok; ++t) {
        const float z = xs[t * D2 + zi];
        float e[2], qv[2], deq;
        unsigned sbits[2];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          if (ident && c == 1) { e[1] = e[0]; qv[1] = qv[0]; sbits[1] = sbits[0]; continue; }
          const float y = c == 0 ? z : fwht32_lane(z, lane);
          if constexpr (cb_asym(CB)) quant_val_a<CbQ<CB>::NB>(y, min32(y), max32(y), deq, qv[c], sbits[c]);
          else quant_val(y, max32(fabsf(y)), deq, qv[c], sbits[c]);
          const float rec = c == 0 ? deq : fwht32_lane(deq, lane);
          e[c] = __fsub_rn(rec, z);
        }
        int best = 0;
        if (!ident) {
          float u[2];
#pragma unroll 4
          for (int k = 0; k < 32; ++k) {
            const float cv = __ldg(CT + (32 * j + k) * D2 + zi);  // C[z_a][z_k]
#pragma unroll
            for (int c = 0; c < 2; ++c) {
              const float pr = __fmul_rn(cv, __shfl_sync(FULL, e[c], k));
              u[c] = k == 0 ? pr : __fadd_rn(u[c], pr);
            }
          }
          const float e0 = tree32(__fmul_rn(e[0], u[0])), e1 = tree32(__fmul_rn(e[1], u[1]));
          best = e1 < e0 ? 1 : 0;
        }
        ec[(t * NH + 0) * D2 + zi] = best ? e[1] : e[0];
        cq[(t * NH + 0) * D2 + zi] = static_cast<int8_t>(best ? qv[1] : qv[0]);
        if (lane == 0) {
          sb[(t * NH + 0) * NB + j] = static_cast<uint16_t>(best ? sbits[1] : sbits[0]);
          lc[(t * NH + 0) * NB + j] = static_cast<uint8_t>(best);
        }
      }
    }
    // ---------------------------------------------------------------- A2: whole-head candidate H256 (keys)
    for (int t = ident ? ntok : w; t < ntok; t += 4) {
      float x[2][4], y[2][4];
#pragma unroll
      for (int s = 0; s < 2; ++s)
#pragma unroll
        for (int i = 0; i < 4; ++i) x[s][i] = y[s][i] = xs[t * D2 + 128 * s + 4 * lane + i];
      fwht256_row(y, lane);
      float deq[2][4], qv[2][4];
      unsigned sbits[2] = {0u, 0u};
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        float am = fmaxf(fmaxf(fabsf(y[s][0]), fabsf(y[s][1])), fmaxf(fabsf(y[s][2]), fabsf(y[s][3])));
        float mn = fminf(fminf(y[s][0], y[s][1]), fminf(y[s][2], y[s][3])), mx = fmaxf(fmaxf(y[s][0], y[s][1]), fmaxf(y[s][2], y[s][3]));
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) {
          am = fmaxf(am, __shfl_xor_sync(FULL, am, o));
          mn = fminf(mn, __shfl_xor_sync(FULL, mn, o));
          mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, o));
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          if constexpr (cb_asym(CB)) quant_val_a<CbQ<CB>::NB>(y[s][i], mn, mx, deq[s][i], qv[s][i], sbits[s]);
          else quant_val(y[s][i], am, deq[s][i], qv[s][i], sbits[s]);
        }
      }
      fwht256_row(deq, lane);
#pragma unroll
      for (int s = 0; s < 2; ++s) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          ec[(t * NH + 1) * D2 + 128 * s + 4 * lane + i] = __fsub_rn(deq[s][i], x[s][i]);
          cq[(t * NH + 1) * D2 + 128 * s + 4 * lane + i] = static_cast<int8_t>(qv[s][i]);
        }
        if ((lane & 7) == 0) {
          sb[(t * NH + 1) * NB + 4 * s + lane / 8] = static_cast<uint16_t>(sbits[s]);
          lc[(t * NH + 1) * NB + 4 * s + lane / 8] = 0;
        }
      }
    }
    __syncthreads();
    if (!ident) {
      // ---------------------------------------------------------------- B: U_b = C[:, 32b:32b+32] e[32b:32b+32]
      constexpr int TB = 2;
      for (int tb = 0; tb < ntok; tb += TB) {
        const int nt = min(TB, ntok - tb);
#pragma unroll 1
        for (int bb = 0; bb < 2; ++bb) {
          const int blk = w + 4 * bb;
          float u[TB][NH][8];
#pragma unroll 2
          for (int kk = 0; kk < 32; ++kk) {
            const int k = 32 * blk + kk;
            const float4 c0 = __ldg(reinterpret_cast<const float4*>(CT + k * D2) + lane);        // C[4l .. 4l+3][k]
            const float4 c1 = __ldg(reinterpret_cast<const float4*>(CT + k * D2 + 128) + lane);  // C[128 + 4l ..][k]
            const float cc[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
            for (int tt = 0; tt < TB; ++tt) {
              if (tt < nt) {
#pragma unroll
                for (int c = 0; c < NH; ++c) {
                  const float ek = ec[((tb + tt) * NH + c) * D2 + k];
#pragma unroll
                  for (int m = 0; m < 8; ++m) {
                    const float pr = __fmul_rn(cc[m], ek);
                    u[tt][c][m] = kk == 0 ? pr : __fadd_rn(u[tt][c][m], pr);
                  }
                }
              }
            }
          }
#pragma unroll
          for (int tt = 0; tt < TB; ++tt)
            if (tt < nt)
#pragma unroll
              for (int c = 0; c < NH; ++c) {
                float* dst = up + ((static_cast<size_t>(blk) * tg + tb + tt) * NH + c) * D2 + 4 * lane;
                *reinterpret_cast<float4*>(dst) = make_float4(u[tt][c][0], u[tt][c][1], u[tt][c][2], u[tt][c][3]);
                *reinterpret_cast<float4*>(dst + 128) = make_float4(u[tt][c][4], u[tt][c][5], u[tt][c][6], u[tt][c][7]);
              }
        }
      }
      __syncthreads();
      // ---------------------------------------------------------------- C: head errors (keys)
      for (int it = w; it < ntok * NH; it += 4) {
        const int t = it / NH, c = it % NH;
        float pr[4];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          float ph[2];
#pragma unroll
          for (int s = 0; s < 2; ++s) {
            const int i = 128 * s + 4 * lane + m;
            const size_t st = static_cast<size_t>(tg) * NH * D2, o = (t * NH + c) * D2 + i;
            float uu = up[o];
#pragma unroll
            for (int bq = 1; bq < NB; ++bq) uu = __fadd_rn(uu, up[bq * st + o]);
            ph[s] = __fmul_rn(ec[o], uu);
          }
          pr[m] = __fadd_rn(ph[0], ph[1]);  // halving tree: first (i, i + 128)
        }
        const float err = tree128_row(pr);
        if (lane == 0) he[t * NH + c] = err;
      }
      __syncthreads();
    }
    // ---------------------------------------------------------------- D: head choice, write the page (two halves)
    for (int t = w; t < ntok; t += 4) {
      const long long slot = p.slots[static_cast<long long>(t0 + t) * (p.slot_ph ? p.Hkv : 1) + (p.slot_ph ? h : 0)];
      if (slot < 0) continue;
      const int best = ident ? 0 : (he[t * NH + 1] < he[t * NH] ? 1 : 0);
      if (kind == 0 && best) kF |= 1u << (t >> 2);
      const long long page = slot / P;
      const int pos = static_cast<int>(slot % P);
      uint8_t* chunk0 = p.pool + (page * p.Hkv + h) * (2 * HCH);
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        uint8_t* chunk = chunk0 + s * HCH;
        uint32_t code[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int qi = cq[(t * NH + best) * D2 + 128 * s + 4 * lane + i];
          code[i] = cb_asym(CB) ? static_cast<uint32_t>(qi) : static_cast<uint32_t>(qi < 0 ? (8 | -qi) : qi);
        }
        const int g = lane / 8, gg = 4 * s + g;
        uint32_t sraw = sb[(t * NH + best) * NB + gg];
        if (!best && lc[(t * NH + best) * NB + gg] == 1) sraw |= 0x8000u;  // key vector gg uses H32
        if (ident && gg == 0 && ((kF >> (t >> 2)) & 1u)) sraw |= 0x8000u;    // the token's key uses H256
        auto place = [&](uint32_t cc, bool isB, int i, uint32_t& m, uint32_t& bits) {  // v2 INT4 element layout
          const uint32_t sg = (cc >> 3) & 1u, mag = cc & 7u;
          if (cb_asym(CB)) {
            m = (isB ? 0xF0u : 0x0Fu) << (8 * i);
            bits = cc << (8 * i + (isB ? 4 : 0));
          } else if (!isB) { m = 0x87u << (8 * i); bits = ((sg << 7) | mag) << (8 * i); }
          else {
            const int j2 = (i + 3) & 3;
            m = (0x08u << (8 * i)) | (0x70u << (8 * j2));
            bits = (sg << (8 * i + 3)) | (mag << (8 * j2 + 4));
          }
        };
        if constexpr (cb_b2(CB)) {  // INT2 page layout (attn_decode_v2.cuh)
          if (kind == 0) {
            uint32_t cb[4];
        #pragma unroll
            for (int i = 0; i < 4; ++i) cb[i] = __shfl_down_sync(FULL, code[i], 4);
            const uint32_t hw = i2_halfword(code, cb), hw1 = __shfl_down_sync(FULL, hw, 8);
            const int tq = lane & 7, j = lane >> 3;
            if (tq < 4 && !(j & 1))
              *reinterpret_cast<uint32_t*>(chunk + L::K_OFF + i2_krow(pos) + 8 * tq + 4 * (j >> 1)) = hw | (hw1 << 4);
            if ((lane & 7) == 0) *reinterpret_cast<uint16_t*>(chunk + L::KS_OFF + pos * 8 + 2 * g) = static_cast<uint16_t>(sraw);
          } else {
        #pragma unroll
            for (int i = 0; i < 4; ++i) {
              const int dd = lane * 4 + i;
              uint32_t* wp = reinterpret_cast<uint32_t*>(chunk + L::V_OFF + i2_vword(dd, pos, P));
              const int sh = i2_vshift(dd, pos);
              atomicAnd(wp, ~(3u << sh));
              atomicOr(wp, code[i] << sh);
            }
            if ((lane & 7) == 0) {
              const int r8 = pos % 8, slot8 = (r8 & 1) ? 4 + (r8 >> 1) : (r8 >> 1);
              *reinterpret_cast<uint16_t*>(chunk + L::VS_OFF + (g * P + 8 * (pos / 8) + slot8) * 2) = static_cast<uint16_t>(sraw);
            }
          }
        } else if (kind == 0) {
          uint32_t cb[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) cb[i] = __shfl_down_sync(FULL, code[i], 4);
          const int tq = lane & 7, j = lane >> 3;
          if (tq < 4) {
            uint32_t wd = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              uint32_t m, bits;
              place(code[i], false, i, m, bits); wd |= bits;
              place(cb[i], true, i, m, bits); wd |= bits;
            }
            *reinterpret_cast<uint32_t*>(chunk + L::K_OFF + pos * 64 + 16 * tq + 4 * j) = wd;
          }
          if ((lane & 7) == 0) *reinterpret_cast<uint16_t*>(chunk + L::KS_OFF + pos * 8 + 2 * g) = static_cast<uint16_t>(sraw);
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
          if ((lane & 7) == 0) {
            const int slot8 = isB ? 4 + ii : ii;
            *reinterpret_cast<uint16_t*>(chunk + L::VS_OFF + (g * P + 8 * c8 + slot8) * 2) = static_cast<uint16_t>(sraw);
          }
        }
      }
    }
  }
}
template <int CB>
int write_k256(const CbWriteParams& p, cudaStream_t st) {
  if (p.T <= 0) return 0;
  const size_t smem = wsmem256_bytes(p.tg);
  if (smem > 200 * 1024) return -2;
  dim3 grid(p.Hkv, (p.T + p.tg - 1) / p.tg);
  static size_t set16 = 0, set32 = 0;
  if (p.page == 16) {
    auto k = k_cb_write256<CB, 16>;
    if (smem > set16) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)); set16 = smem; }
    k<<<grid, 128, smem, st>>>(p);
  } else if (p.page == 32) {
    auto k = k_cb_write256<CB, 32>;
    if (smem > set32) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)); set32 = smem; }
    k<<<grid, 128, smem, st>>>(p);
  } else {
    return -5;
  }
  return 0;
}

}  // namespace

#include "attn_codebook_attn.cuh"

int cb_chunk_bytes(int kind, int page) {
  if (cb_b2(kind)) return 80 * page;
  return kind == CB_REDUCED ? 144 * page + 16 : cb_k2(kind) ? 144 * page : cb_kfam(kind) ? 146 * page : 148 * page;
}
int cb_write_smem_hd(int kind, int tg, int hd) {
  if (hd == 256) return cb_k2(kind) ? static_cast<int>(wsmem256_bytes(tg)) : -1;
  return cb_write_smem(kind, tg);
}
int cb_write_smem(int kind, int tg) {
  switch (kind) {
    case CB_FULL: return static_cast<int>(wsmem_bytes<CB_FULL>(tg));
    case CB_FINAL: return static_cast<int>(wsmem_bytes<CB_FINAL>(tg));
    case CB_HEAD8: return static_cast<int>(wsmem_bytes<CB_HEAD8>(tg));
    case CB_HEAD16: return static_cast<int>(wsmem_bytes<CB_HEAD16>(tg));
    case CB_Q4: return static_cast<int>(wsmem_bytes<CB_Q4>(tg));
    case CB_Q4K: return static_cast<int>(wsmem_bytes<CB_Q4K>(tg));
    case CB_FINALK: return static_cast<int>(wsmem_bytes<CB_FINALK>(tg));
  }
  return static_cast<int>(wsmem_bytes<CB_REDUCED>(tg));
}
int cb_write_launch(int kind, const CbWriteParams& p, cudaStream_t st) {
  if (p.hd == 256) {  // (key-only 2x2 kinds only)
    if (kind == CB_KONLY) return write_k256<CB_KONLY>(p, st);
    if (kind == CB_KONLYA) return write_k256<CB_KONLYA>(p, st);
    if (kind == CB_KONLYA2) return write_k256<CB_KONLYA2>(p, st);
    return -4;
  }
  if (p.hd != 0 && p.hd != 128) return -8;
  if (kind == CB_FINAL) return write_k<CB_FINAL>(p, st);
  if (kind == CB_HEAD8) return write_k<CB_HEAD8>(p, st);
  if (kind == CB_HEAD16) return write_k<CB_HEAD16>(p, st);
  if (kind == CB_Q4) return write_k<CB_Q4>(p, st);
  if (kind == CB_Q4K) return write_k<CB_Q4K>(p, st);
  if (kind == CB_FINALK) return write_k<CB_FINALK>(p, st);
  if (kind == CB_FULL) return write_k<CB_FULL>(p, st);
  if (kind == CB_REDUCED) return write_k<CB_REDUCED>(p, st);
  if (kind == CB_KONLY) return write_k<CB_KONLY>(p, st);
  if (kind == CB_KONLYA) return write_k<CB_KONLYA>(p, st);
  if (kind == CB_KONLYA2) return write_k<CB_KONLYA2>(p, st);
  return -4;
}
