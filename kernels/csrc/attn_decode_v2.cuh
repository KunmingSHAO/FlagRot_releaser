// Paged split-K GQA decode attention over 4-bit KV caches (NVFP4 / MXFP4 / INT4), H20 (sm_90a), v2:
// warpgroup FP8 tensor cores (wgmma e4m3), q split hi/lo, bulk-copy (TMA) page pipeline.  No torch headers.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

enum AttnFmt2 : int { A2_NVFP4 = 2, A2_MXFP4 = 3, A2_INT4 = 4, A2_INT4A = 5, A2_KONLY = 6, A2_KONLYA = 7,
                      A2_INT2A = 8, A2_KONLYA2 = 9 };
// 2..4: same ids as AttnFmt (attn_decode.cuh).  A2_KONLY / A2_KONLYA: the key-only 2x2 codebook caches of
// attn_codebook.cuh (CB_KONLY / CB_KONLYA: v2 INT4 / INT4A layout, H32 flag of key group j = sign bit of its K scale,
// whole-head flag = sign bit of the token's group-0 V scale), decoded by the generalized kernel only (attn_decode_v2g.cuh,
// any GQA group / head dim 256); written by cb_write_launch

// Page layout (bytes), one contiguous chunk per (page, kv head):  pool[(page * Hkv + head) * CHUNK + ...]
//   [0, 64P)          K codes, token row t at 64 t; row byte 16 tg + 4 j + i holds the elements A = d(32 j + 4 tg + i)
//                     and B = d(32 j + 16 + 4 tg + i) (bit layout below)
//   [64P, 128P)       V^T codes: 16-B entries e = (w * 8 + gid) * (P / 8) + c8 (w, gid < 8: d % 64 = 16 w + gid + 8 r,
//                     c8 = 8-token chunk of the page); entry word s = 2 (d / 64) + r holds the 8 tokens 8 c8 + 0..7 of d:
//                     A_i = token 8 c8 + 2 i, B_i = token 8 c8 + 2 i + 1
//   [128P, 128P+SB P) K scales, row t: the SB bytes of kv_selrot_quant (group order)
//   [.., +SB P)       V scales, transposed: group g, page chunk c8: [A0 A1 A2 A3 B0 B1 B2 B3] (token order as V^T)
// Element bit layout inside a 32-bit word of 4 A + 4 B elements (byte i holds A_i):
//   e2m1 (NVFP4/MXFP4): A_i: sign -> bit 8i+7, |code| -> bits 8i+4..2; B_i: sign -> 8i+5, |code| bit 2 -> 8i+6,
//                       bits 1..0 -> 8i+1..0.  Then  A = w & 0x9C9C9C9C,  B = ((w << 2) & 0x8C8C8C8C) | ((w >> 2) &
//                       0x10101010)  are e4m3 bytes of e2m1 * 2^-6 (the e2m1 subnormal 0.5 is the e4m3 subnormal).
//   int4 (sign-magnitude of q = code - 8): A_i: sign -> 8i+7, |q| -> 8i+2..0; B_i: sign -> 8i+3, |q| -> bits 4..6 of
//                       byte (i+3)%4.  Then A = w & 0x87878787, B = rotl(w, 4) & 0x87878787 are e4m3 bytes of q * 2^-9
//                       (bytes 0..15 of e4m3 are exactly c * 2^-9).
//   int4a (asymmetric, unsigned codes c in [0, 15], value (c - z) * s): A_i -> bits 8i+3..8i, B_i -> bits 8i+7..8i+4.
//                       Then A = w & 0x0F0F0F0F, B = (w >> 4) & 0x0F0F0F0F are e4m3 bytes of c * 2^-9.
//                       Scale word (16 bits, K and V scales as int4): bits 3..0 = zero point z, the whole word read as
//                       FP16 = the scale s (the writer picks the FP16 nearest to s whose 4 low bits are z, so no masking).
// INT2 layouts (A2_INT2A = static int2a:32 stored as blockflag int2ah:32; A2_KONLYA2 = the key-only 2x2 codebook cache
// attn_codebook CB_KONLYA2 with the same codes; vLLM plugin Phase 3a): unsigned 2-bit codes c in [0, 3] (value (c - z) s),
// the zero point z in bits 1..0 of the FP16 scale word (the FP16 nearest to s whose 2 low bits are z), flags in the
// sign bits as for INT4.  Chunk per (page, kv head, 128-d half) = P * 80 bytes (2.5 bits per element):
//   [0, 32P)          K codes: token t's 32-byte row at physical row t ^ ((t >> 1) & 2) (rows t and t ^ 2 swapped
//                     where t & 4: the 8-byte row reads of a half warp, rows {0, 1, 4, 5} (+ 2, + 8), hit 32 distinct
//                     banks); row byte 8 tg + 4 jp + i, bits 4 h + 2 b + 1 .. 4 h + 2 b hold the element of group
//                     j = 2 jp + h, b = 0: A = d(32 j + 4 tg + i), b = 1: B = d(32 j + 16 + 4 tg + i)
//   [32P, 64P)        V^T codes: 8-byte entries e = (w * 8 + gid) * (P / 8) + c8 (as INT4), entry word m (4 B at 4 m)
//                     holds d = 64 m + 16 w + gid + 8 r (r = 0, 1): byte i bits 4 r + 2 b + 1 .. 4 r + 2 b = token
//                     8 c8 + 2 i + b
//   [64P, 72P)        K scales, [72P, 80P) V scales: exactly the INT4 scale layout (scale words with z in bits 1..0)
//                     One 32-bit code word w (16 elements) -> e4m3 bytes of c 2^-9: a0 = w & 0x03030303, b0 = (w >> 2)
//                     & .., a1 = (w >> 4) & .., b1 = (w >> 6) & .. (the A / B words of the two INT4 words it replaces).
struct PagedParams {
  const void* q;               // [B, 4*Hkv, 128] bf16 (head rows contiguous)
  long long q_bs;
  const uint8_t* pool;         // [num_pages, Hkv, CHUNK]
  const int* block_table;      // [B, bt_stride] page ids
  int bt_stride;
  const int* seq_lens;         // [B] tokens attended (<= bt_stride * page)
  int B, Hkv, page, nsplit, chunk;  // split s of sequence b covers tokens [s*chunk, min(len_b, (s+1)*chunk)); chunk % 64 == 0
  float qscale;                // softmax scale * log2(e)
  const float* gscale;         // nvfp4 global scale (device scalar): value = e2m1 * e4m3 / gs
  const uint8_t* kmask;        // [Hkv, D/rb] FlagRot flags of K (q rotated in the prologue if qrot)
  const uint8_t* vmask;        // [Hkv, D/rb] flags of V (output back-rotated if vrot)
  int rb, qrot, vrot;
  float* opart;                // [B, Hkv, nsplit, 4, 128]
  float* mlpart;               // [B, Hkv, nsplit, 4, 2]
  int* counter;                // [B * Hkv] zero-initialized, self-resetting
  void* out;                   // [B, 4*Hkv*128] bf16
  long long o_bs;
  // optional BF16 staging window (attn_tail.cuh): the newest tail_len[b] <= 256 tokens of sequence b, not encoded,
  // tail [B, W, 2*Hkv, 128] bf16 (original basis; K rows then V rows, token stride tail_ts), attended by one extra
  // split CTA per (sequence, kv head) next to the seq_lens[b] cached tokens; nsplit then counts that CTA
  const void* tail;
  const int* tail_len;
  long long tail_bs, tail_ts;
  int tail_nslot;              // merge slots of the window (ceil(W / 32)); 0 without a window
  const int* tail_idx;         // optional [B]: window row of sequence b = tail_idx[b] (vLLM plugin: per-request
                               // staging rings indexed by the request slot); nullptr: row b
  // stream-K schedule (kernel variants with MODE bit 9): number of CTAs; 0 = split mode (grid (Hkv, nsplit, B)).
  // opart / mlpart then hold 2 sk_grid + B Hkv tail_nslot slots (attn_decode_v2.cu, sk_prefix)
  int sk_grid;
  // generalized kernel (variant bit ATTN2_GEN, attn_decode_v2g.cuh): q heads per kv head (0 = 4); q is
  // [B, gqa * Hkv, D], out [B, gqa * Hkv * D] with D = 256 under ATTN2_HD256 (pool chunk = two 128-d half chunks).
  // Partials: B * Hkv * ceil(gqa / GT) * nsplit slots of [GT][D] floats (+ [GT] (max, sum)), counter B * Hkv * ceil(gqa / GT)
  int gqa;
  // tokens [0, kv_start) of every sequence are masked in the paged body (vLLM plugin Phase 2k: the BF16 sink tokens are
  // stored in the pages at their real positions but attended from the BF16 tail rows); 0 = none
  int kv_start;
};

struct PagedWriteParams {
  const void* kv;              // [T, 2*Hkv, 128] bf16 (head rows contiguous, token stride kv_ts elements)
  long long kv_ts;
  const long long* slots;      // [T] slot = page * P + offset; < 0: skip
  int T, Hkv, page;
  const uint8_t* mask;         // [2*Hkv, 128/rb] FlagRot flags (K rows first)
  int rb;
  const float* gscale;
  uint8_t* pool;
  int hd;                      // head dim (0 = 128; 256: kv [T, 2*Hkv, 256], pool chunk = two 128-d half chunks of
                               // the v2 layout, written as 2 * Hkv half heads)
  const float* kmu;            // optional [Hkv, hd] fp32: static key centering (vLLM plugin option "center"), the K rows
                               // become x - kmu (fp32, RN) right after the bf16 load, before the FlagRot rotation; null = off
};

// variant bits of the generalized static kernel (attn_decode_v2g.cuh, INT4 / INT4A, = variant 264 + these bits):
// ATTN2_GEN (required whenever gqa != 4 or D = 256), ATTN2_GT8 (q tiles of 8 heads, D = 128), ATTN2_HD256 (D = 256)
constexpr int ATTN2_GEN = 1024, ATTN2_GT8 = 2048, ATTN2_HD256 = 4096;
// variant: 0 = FP8 PV (P' = p * V-scale split hi/lo, MXFP4 / INT4 only), 1 = f16 PV (per-element V scale); stages
int attn2_launch(int fmt, int variant, int stages, const PagedParams& p, cudaStream_t st);
int attn2_occupancy(int fmt, int variant, int stages, int page);
void attn2_set_pdl(int on);  // programmatic dependent launch of the attention kernel (graph-capturable), default off
int attn2_chunk_bytes(int fmt, int page);  // (128-d head; D = 256: twice)
int paged_write_launch(int fmt, const PagedWriteParams& p, cudaStream_t st);
