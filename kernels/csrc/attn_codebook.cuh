// Per-token transform codebook ("FlagRot, dynamic codebook") for the INT4 (g32, fp16 scale) paged KV cache on H20
// (sm_90a): cache writer with the codebook choice (bit-exact vs kernels/codebook_ref.py) and paged split-K GQA
// decode attention that applies the inverse transforms.  No torch headers.  Design notes: attn_codebook.cu,
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// codebook kinds
//   CB_FULL    'cq+S+FS+P+F': per 32-vector {I, H32, H32 S} x per head {base, P (RoPE-pair interleaved), F (H128),
//              FS (H128 S)}; 16-bit side word per (token, head, K/V)
//   CB_REDUCED 'cq+f': per 32-vector {I, H32} x per head {base, F}; vector bit = sign bit of its fp16 scale, head bit
//              in a per-page bit plane
//   CB_FINAL   'cq+f+S+Q5+FS+R5' (the locked method): per 32-vector {I, H32, H32 S_k (k = 0, 2..6)} x per head {base,
//              F (H128), H128 S_k (k = 0, 2..6)}: 8 x 8 candidates, 15-bit side word (no P)
//   CB_HEAD8 / CB_HEAD16  'cq+f+FS+R6' / 'cq+f+FS+R14': per 32-vector {I, H32} x per head {base, H128, H128 S_k} with
//              8 / 16 whole-head transforms (S_k, k = 0, 2..7 / 0, 2..15)
//   CB_KONLY   'cq+f+konly': keys as CB_REDUCED, values quantized without transform (the static v2 INT4 PV path);
//              key vector bit = sign bit of its fp16 K scale, key head bit (F) = sign bit of the token's group-0 V scale
//   CB_Q4      'cq+f+S+Q1+FS+R1' (4 x 4): per 32-vector {I, H32, H32 S_0, H32 S_2} x per head {base, H128, H128 S_0,
//              H128 S_2}; 10-bit side word (2 bits per vector, head mode at bits 8..9)
//   CB_Q4K / CB_FINALK  '...+konly' of 4 x 4 / locked 8 x 8: the keys as CB_Q4 / CB_FINAL (K side word), values
//              plain INT4 (no V side word: chunk 146 P), decoded by the key-only family kernel (static PV)
//   CB_KONLYA  CB_KONLY with asymmetric INT4 (blockflag int4a:32): unsigned codes 0..15 (attn_decode_v2.cuh
//              int4a element layout), zero point z in bits 3..0 of every 16-bit scale word (the word without its sign bit
//              read as FP16 = the scale); flags in the sign bits as CB_KONLY.  The choice uses the int4a:32 error (FP32
//              scale, unbounded zero point = dyn_quant); the chosen vector is stored with the range widened to include 0
//   CB_KONLYA2 CB_KONLYA with asymmetric INT2 (int2a:32 choice, int2ah:32 storage; vLLM plugin Phase 3a): the INT2 page
//              layout of attn_decode_v2.cuh (80 P bytes per (page, kv head)), zero point in bits 1..0 of the scale words
enum CbKind : int { CB_REDUCED = 1, CB_FULL = 2, CB_FINAL = 3, CB_HEAD8 = 4, CB_HEAD16 = 5, CB_KONLY = 6, CB_Q4 = 7,
                    CB_Q4K = 8, CB_FINALK = 9, CB_KONLYA = 10, CB_KONLYA2 = 11 };
constexpr bool cb_family(int cb) { return (cb >= CB_FINAL && cb <= CB_HEAD16) || cb == CB_Q4; }
constexpr int cb_base(int cb) { return cb == CB_Q4K ? CB_Q4 : cb == CB_FINALK ? CB_FINAL : cb; }  // key codebook
constexpr bool cb_kfam(int cb) { return cb == CB_Q4K || cb == CB_FINALK; }   // key-only family
constexpr bool cb_k2(int cb) { return cb == CB_KONLY || cb == CB_KONLYA || cb == CB_KONLYA2; }  // key-only 2x2 (k_cbk)
constexpr bool cb_asym(int cb) { return cb == CB_KONLYA || cb == CB_KONLYA2; }  // asymmetric (int4a:32 / int2a:32)
constexpr bool cb_b2(int cb) { return cb == CB_KONLYA2; }                        // 2-bit codes (INT2 page layout)
constexpr int cb_qm(int cb) { return cb_b2(cb) ? 2 : cb_asym(cb) ? 1 : 0; }      // quantizer: 0 int4, 1 int4a, 2 int2a
constexpr bool cb_konly(int cb) { return cb_k2(cb) || cb_kfam(cb); }   // values without transform
constexpr int CB_NSG = 15;  // sign table: entry 0 = seed 0, entry i >= 1 = seed i + 1 (blockflag _signs(128, seed))

// Sign-flipped-Hadamard codebooks (final, head8, head16): vector choice l in [0, NLH] (0 I, 1 H32, l >= 2 H32 S_{l-2}),
// head mode m in [0, NHF] (0 base, 1 H128, m >= 2 H128 S_{m-2}); variant / value-class index v = m ? NLH + m : l;
// side word = sum_j l_j << (LB j) | m << (4 LB)
template <int CB0> struct CbSpec {
  static constexpr int CB = cb_base(CB0);
  static constexpr int NLH = CB == CB_FINAL ? 7 : CB == CB_Q4 ? 3 : 1;
  static constexpr int NHF = CB == CB_FINAL ? 7 : CB == CB_Q4 ? 3 : CB == CB_HEAD8 ? 8 : 16;
  static constexpr int LB = CB == CB_FINAL ? 3 : CB == CB_Q4 ? 2 : 1;
  static constexpr int HB = CB == CB_FINAL ? 3 : CB == CB_Q4 ? 2 : CB == CB_HEAD8 ? 4 : 5;
  static constexpr int V = 1 + NLH + NHF;  // final 15, q4 7, head8 10, head16 18
};

// Page layout (bytes), one contiguous chunk per (page, kv head) = the v2 INT4 layout (attn_decode_v2.cuh) + side info:
//   [0, 64P)        K codes          [64P, 128P)   V^T codes
//   [128P, 136P)    K scales (4 x fp16 per token)            [136P, 144P)  V scales (transposed, as v2)
//   full / final: [144P, 146P) K side words (uint16, token order)  [146P, 148P) V side words     CHUNK = 148 P
//            full side word:  bits 2j..2j+1 = choice of 32-vector j (0 I, 1 H32, 2 H32 S; 0 for F / FS),
//                             bits 8..9 = head mode (0 base, 1 P, 2 F, 3 FS)
//            final side word: bits 3j..3j+2 = choice of 32-vector j (0 I, 1 H32, 2..7 H32 S_k, k = 0, 2..6; 0 unless
//                             base), bits 12..14 = head mode (0 base, 1 F, 2..7 H128 S_k)
//            head8 / head16:  bit j = vector j uses H32, bits 4.. = head mode (0 base, 1 F, 2.. H128 S_k)
//   reduced: scale sign bit of group j = vector j uses H32 (scale magnitude = |fp16|);
//            [144P, +4) K head bits (bit t = token t uses F), [144P + 4, +4) V head bits in V^T chunk order
//            (byte c8: bit i = token 8 c8 + 2 i, bit 4 + i = token 8 c8 + 2 i + 1), [144P + 8, +8) pad   CHUNK = 144P + 16
//   konly:   K scale sign bit of group j = key vector j uses H32; V scale sign bit of group 0 = the token's key uses F;
//            V codes / scales untransformed; no trailer                                                CHUNK = 144P
// Stored codes / scales are those of the chosen transformed vector y (coordinates of the chosen candidate: x[PERM]
// for P, PERM = (0, 64, 1, 65, ...)).  The fixed sign vectors S_k are passed as CB_NSG x 4 x 32 bits (bit d: S_d = -1;
// full / reduced use S_0 only).

struct CbWriteParams {
  const void* kv;              // [T, 2*Hkv, 128] bf16 (head rows contiguous, token stride kv_ts elements)
  long long kv_ts;
  const long long* slots;      // [T] (or [T, Hkv] if slot_ph) slot = page * P + offset; < 0: skip
  int slot_ph;                 // 1: per-(token, kv head) slots (encoder-side reordering of a window per kv head)
  int T, Hkv, page, tg;        // tg: tokens per CTA
  const float* CT;             // [Hkv, 128 (k), 128 (i)] = C_q[h][i][k] (keys' weighting, fp32)
  const float* co;             // [Hkv, 128] o_proj column energy (values' weighting, fp32)
  unsigned sg[CB_NSG][4];
  uint8_t* pool;
  // writer v3 (attn_cb_write3.cuh, attn_codebook.w3_calib): C1 fp16 [Hkv, 168, 128] (C rows + correction rows), beta =
  // aux fp32 [Hkv, 48] (bounds, correction columns); nchunk = persistent key CTAs per kv head
  // (0: one per tile); stats (optional, int [4] += local items, local fallbacks, head items, head fallbacks)
  const void* C1;
  const float* beta;
  int nchunk, nchunk_v;        // persistent CTAs per kv head (keys / values)
  int* stats;
  int hd;                      // head dim (0 = 128; 256: key-only 2x2 kinds, kv [T, 2*Hkv, 256], CT [Hkv, 256, 256],
                               // co [Hkv, 256], pool chunk = two 128-d half chunks; writer v1 only)
  const float* kmu;            // optional [Hkv, hd] fp32: static key centering, K rows -> x - kmu (fp32, RN) at the load,
                               // before any candidate transform (writers v1 and v3); null = off
};

struct CbAttnParams {
  const void* q;               // [B, 4*Hkv, 128] bf16
  long long q_bs;
  const uint8_t* pool;         // [num_pages, Hkv, CHUNK]
  const int* block_table;      // [B, bt_stride]
  int bt_stride;
  const int* seq_lens;         // [B]
  int B, Hkv, page, nsplit, chunk;
  float qscale;                // softmax scale * log2(e)
  unsigned sg[CB_NSG][4];
  float* opart;                // [B, Hkv, nsplit, 4, 128]
  float* mlpart;               // [B, Hkv, nsplit, 4, 2]
  int* counter;                // [B * Hkv]
  void* out;                   // [B, 4*Hkv*128] bf16
  long long o_bs;
  // BF16 staging window (optional, codebook family kernels): the newest tail_len[b] <= 32 tokens of sequence b are
  // not encoded yet; tail [B, 32, 2*Hkv, 128] bf16 (K rows then V rows; token stride tail_ts elements); attended in
  // addition to the seq_lens[b] tokens of the paged body (positions are irrelevant: RoPE is applied before caching)
  const void* tail;
  const int* tail_len;
  long long tail_bs, tail_ts;
  int tail_nslot;              // merge slots of the window (ceil(W / 32), attn_tail.cuh); 0 without a window
  const int* tail_idx;         // optional [B]: window row of sequence b = tail_idx[b] (vLLM plugin: per-request
                               // staging rings indexed by the request slot); nullptr: row b
  int kv_start;                // tokens [0, kv_start) masked in the paged body (Phase 2k sinks); 0 = none
};

int cb_chunk_bytes(int kind, int page);
int cb_write_launch(int kind, const CbWriteParams& p, cudaStream_t st);
int cb_write_smem(int kind, int tg);
int cb_write_smem_hd(int kind, int tg, int hd);  // (hd 256: key-only 2x2 kinds)
int cb_write3_launch(int kind, const CbWriteParams& p, cudaStream_t st);  // -4: kind not supported by v3

// Static FlagRot INT4 / INT4A paged writer in the v3 style (attn_cb_write3.cuh): one thread per (token,
// K/V row, 32-vector), the whole row in registers, V^T codes of aligned 8-slot runs as whole words.  Output = the v2
// writer (attn_decode_v2.cu k_paged_write, rb 32) applied after the plugin's whole-head rotation (fr_head_fwht: rows of
// heads with hfull set -> bf16(H128 x / 16), unnormalized +-1 Sylvester butterflies in fp32), byte for byte.
struct W3StaticParams {
  const void* kv;              // [T, 2*Hkv, 128] bf16 (head rows contiguous, token stride kv_ts elements, 16-B aligned rows)
  long long kv_ts;
  const long long* slots;      // [T] slot = page * P + offset; < 0: skip
  int T, Hkv, page;
  const uint8_t* mask;         // [2*Hkv, 4] rb-32 block flags (K rows first) or null
  const uint8_t* hfull;        // [2*Hkv] whole-head flags (K rows first) or null
  int asym;                    // 0: A2_INT4 (symmetric, fp16 scale), 1: A2_INT4A (int4a:32 stored as int4ah:32),
                               // 2: A2_INT2A (int2a:32 stored as int2ah:32, INT2 page layout, 80 * page bytes)
  uint8_t* pool;               // [num_pages, Hkv, 144 * page]
  int path;                    // 0: by T (T <= W3S_DEC_MAXT: k_w3s_dec), 1: k_w3s (tiles of 32 tokens), 2: k_w3s_dec
  const float* kmu;            // optional [Hkv, 128] fp32: static key centering in the STORED basis (after the whole-head
                               // H128 + bf16 rounding of hfull heads, = fr_head_fwht + fr_v2_write with kmu), before the
                               // block flags; null = off
};
int w3_static_write_launch(const W3StaticParams& p, cudaStream_t st);
// variant: 0 = software-pipelined (reduced: v2 pipeline; full: PV rounds two deep), 1 = synchronous wgmma rounds
// (fewer registers, more CTAs per SM)
int cb_attn_launch(int kind, int variant, int stages, const CbAttnParams& p, cudaStream_t st);
int cb_attn_occupancy(int kind, int variant, int stages, int page);
