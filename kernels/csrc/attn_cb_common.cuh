// Shared device helpers of the codebook INT4 cache (writer v1 / v3, attention): constants, the INT4 quantizer, FWHT
// building blocks in the reference's fp32 order, page layout.  Included by attn_codebook.cu and attn_cb_write3.cu.
#pragma once
#include "attn_codebook.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define DEVI __device__ __forceinline__

namespace {

constexpr int D = 128;
constexpr unsigned FULL = 0xffffffffu;
constexpr float C32 = 0.17677669529663687f;   // fp32(1 / sqrt(32))
constexpr float C128 = 0.08838834764831845f;  // fp32(1 / sqrt(128))

template <int CB> struct CbW {
  static constexpr bool SF = cb_family(CB) || cb_kfam(CB);  // sign-flipped-Hadamard family (CbSpec)
  // head candidates: full base, P, F, FS / reduced base, F / family base, F, FS_k
  static constexpr int NH = CB == CB_FULL ? 4 : SF ? 1 + CbSpec<CB>::NHF : 2;
  static constexpr int NZ = CB == CB_FULL ? 2 : 1;  // local coordinate systems (x, x[PERM])
  // local candidates I, H (, HS) / family I, H, HS_k; candidate c >= 2 uses sign vector c - 2
  static constexpr int NL = CB == CB_FULL ? 3 : SF ? 1 + CbSpec<CB>::NLH : 2;
  static constexpr int TB = NH > 8 ? 1 : NH > 4 ? 2 : 4;  // tokens per register block in phase B
};

DEVI int permi(int a) { return (a >> 1) + ((a & 1) << 6); }  // z = x[PERM]: z_a = x_permi(a)
DEVI bool sgneg(const unsigned (&sg)[4], int d) { return (sg[d >> 5] >> (d & 31)) & 1u; }
DEVI float sflip(float v, bool neg) { return neg ? -v : v; }   // == v * (+-1) exactly

DEVI float div_rn(float y, float d, float inv) {  // correctly rounded y / d given inv = RN(1/d) (selrot_kernels.cu)
  const float q = __fmul_rn(y, inv);
  const float n = fmaf(q, d, -y);
  return fmaf(-n, inv, q);
}
// INT4 quantization of one value with group max amax (ref.int4_quant): s = fp16(amax * fp32(1/7)), q = clamp(rne(y/s))
DEVI void quant_val(float y, float amax, float& deq, float& q, unsigned& sbits) {
  const __half sh = __float2half_rn(__fmul_rn(amax, 1.0f / 7.0f));
  const float sf = __half2float(sh);
  const float inv = sf != 0.f ? __frcp_rn(sf) : 0.f;
  q = sf != 0.f ? rintf(div_rn(y, sf, inv)) : 0.f;
  q = fminf(fmaxf(q, -7.f), 7.f);
  deq = __fmul_rn(q, sf);
  sbits = __half_as_ushort(sh);
}
// NB = 4: int4a:32 / int4ah:32; NB = 2: int2a:32 / int2ah:32 (scale word with z in its 2 low bits)
template <int NB = 4>
DEVI unsigned scale_word_a(float s, unsigned z) {
  constexpr int M = (1 << NB) - 1;
  const int h = static_cast<int>(__half_as_ushort(__float2half_rn(s)));
  const int x = min(max(h - static_cast<int>(z) + (1 << (NB - 1)), 0) & ~M, 0x7C00 - (1 << NB));
  return static_cast<unsigned>(x) + z;
}
template <int NB = 4>
DEVI void asym_params(float mn, float mx, float& s, float& z) {
  s = fmaxf(__fmul_rn(__fsub_rn(mx, mn), 1.0f / static_cast<float>((1 << NB) - 1)), 1e-12f);
  z = rintf(__fdiv_rn(-mn, s));
}
template <int NB = 4>
DEVI float asym_code(float y, float s, float z) {
  return fminf(fmaxf(__fadd_rn(rintf(__fdiv_rn(y, s)), z), 0.f), static_cast<float>((1 << NB) - 1));
}
// one value of a group with min mn / max mx: deq (int4a:32 / int2a:32), stored code q, stored scale word sbits
template <int NB = 4>
DEVI void quant_val_a(float y, float mn, float mx, float& deq, float& q, unsigned& sbits) {
  float s, z;
  asym_params<NB>(mn, mx, s, z);
  const float qa = asym_code<NB>(y, s, z);
  deq = __fmul_rn(__fsub_rn(qa, z), s);
  if (mn > 0.f || mx < 0.f) asym_params<NB>(fminf(mn, 0.f), fmaxf(mx, 0.f), s, z);  // stored form: range includes 0
  q = asym_code<NB>(y, s, z);
  sbits = scale_word_a<NB>(s, static_cast<unsigned>(z));
}
// quantizer of a codebook kind (cb_qm): 2 -> INT2 (NB 2), else INT4
template <int CB> struct CbQ { static constexpr int NB = cb_b2(CB) ? 2 : 4; };
// one element per lane (a 32-vector per warp)
// one element per lane (a 32-vector per warp)
DEVI float fwht32_lane(float v, int lane) {
#pragma unroll
  for (int h = 1; h < 32; h <<= 1) {
    const float o = __shfl_xor_sync(FULL, v, h);
    v = (lane & h) ? __fsub_rn(o, v) : __fadd_rn(v, o);
  }
  return __fmul_rn(v, C32);
}
DEVI float max32(float a) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(FULL, a, o));
  return a;
}
DEVI float min32(float a) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) a = fminf(a, __shfl_xor_sync(FULL, a, o));
  return a;
}
DEVI float tree32(float v) {  // halving tree over the lanes
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(FULL, v, o));
  return v;
}
// four elements per lane (a 128-vector per warp, lane l: 4l .. 4l+3); Sylvester FWHT over the whole row
DEVI void fwht128_row(float (&x)[4], int lane) {
#pragma unroll
  for (int h = 1; h < 128; h <<= 1) {
    if (h < 4) {
#pragma unroll
      for (int i = 0; i < 4; ++i)
        if (!(i & h)) {
          const float a = x[i], c = x[i + h];
          x[i] = __fadd_rn(a, c);
          x[i + h] = __fsub_rn(a, c);
        }
    } else {
      const int lm = h >> 2;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float o = __shfl_xor_sync(FULL, x[i], lm);
        x[i] = (lane & lm) ? __fsub_rn(o, x[i]) : __fadd_rn(x[i], o);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] = __fmul_rn(x[i], C128);
}
DEVI float tree128_row(const float (&p)[4]) {  // halving tree of a 128-vector held as 4 elements per lane
  float v[4] = {p[0], p[1], p[2], p[3]};
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = __fadd_rn(v[i], __shfl_xor_sync(FULL, v[i], o));
  return __fadd_rn(__fadd_rn(v[0], v[2]), __fadd_rn(v[1], v[3]));
}

template <int CB, int P> struct CbLay {
  static constexpr int CBY = cb_b2(CB) ? 32 : 64;                // code bytes per (token, row) of K and of V
  static constexpr int K_OFF = 0, V_OFF = CBY * P, KS_OFF = 2 * CBY * P, VS_OFF = 2 * CBY * P + 8 * P;
  static constexpr int SWK_OFF = 144 * P, SWV_OFF = 146 * P;     // full
  static constexpr int FK_OFF = 144 * P, FV_OFF = 144 * P + 4;   // reduced
  static constexpr int CHUNK = cb_b2(CB) ? 80 * P : CB == CB_REDUCED ? 144 * P + 16 : cb_k2(CB) ? 144 * P : cb_kfam(CB) ? 146 * P : 148 * P;
};
// INT2 page layout helpers (attn_decode_v2.cuh): K row of token pos (swizzled), V^T word of element d / token pos and
// the bit shift of its 2-bit code
DEVI int i2_krow(int pos) { return 32 * (pos ^ ((pos >> 1) & 2)); }
DEVI int i2_vword(int d, int pos, int P) {
  return (((d % 64) / 16 * 8 + d % 8) * (P / 8) + pos / 8) * 8 + 4 * (d / 64);
}
DEVI int i2_vshift(int d, int pos) { return 8 * ((pos % 8) >> 1) + 4 * ((d % 16) / 8) + 2 * (pos & 1); }
// K row word (tg, jp) of a token: 4 lanes' (A, B) code pairs -> one 32-bit word; codes of group j = 2 jp + h:
// element i of A (d = 32 j + 4 tg + i) at bits 8 i + 4 h, of B (d = 32 j + 16 + 4 tg + i) at 8 i + 4 h + 2
DEVI uint32_t i2_halfword(const uint32_t (&a)[4], const uint32_t (&b)[4]) {  // h = 0 position
  uint32_t w = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) w |= (a[i] << (8 * i)) | (b[i] << (8 * i + 2));
  return w;
}

}  // namespace
