// Split-K ("flash-decoding") GQA decode attention over a quantized KV cache (H20 / sm_90a).  No torch headers.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// KV-cache formats = exactly what kv_selrot_quant writes (per token: [2*Hkv, row] codes, [2*Hkv, sb] scales)
enum AttnFmt : int {
  AF_BF16 = 0,   // bf16 [D]                                            (row 256 B, no scale)
  AF_FP8 = 1,    // e4m3 [D] + fp32 scale per (token, head)             (row 128 B, scale 4 B)
  AF_NVFP4 = 2,  // e2m1 x2 [D/2] + e4m3 per 16 (x 1/global_scale)      (row 64 B, scale 8 B)
  AF_MXFP4 = 3,  // e2m1 x2 [D/2] + e8m0 per 32                         (row 64 B, scale 4 B)
  AF_INT4 = 4    // (q + 8) x2 [D/2] + fp16 per 32                      (row 64 B, scale 8 B)
};

struct AttnParams {
  const void* q;               // q [B, Hq, D] bf16 (head stride D); Hq = 4 * Hkv, D = 128
  long long q_bs;              // q batch stride (elements)
  const uint8_t* k;            // K codes of (batch 0, token 0, kv head 0)
  const uint8_t* v;            // V codes of (batch 0, token 0, kv head 0)
  const uint8_t* ks;           // K scales (unused for bf16)
  const uint8_t* vs;           // V scales
  long long tok_stride, bat_stride;    // bytes, codes (head stride = row bytes)
  long long stok_stride, sbat_stride;  // bytes, scales (head stride = scale bytes)
  int B, L, Hkv, nsplit, chunk;        // tokens [0, L) attended; split s covers [s*chunk, min(L, (s+1)*chunk))
  float qscale;                // softmax scale * log2(e) (folded into q)
  const float* gscale;         // nvfp4: global scale (device scalar); dequant = e2m1 * e4m3 / gscale
  const uint8_t* kmask;        // [Hkv, D/rb] FlagRot flags of the keys (q rotated in the prologue if QROT)
  const uint8_t* vmask;        // [Hkv, D/rb] flags of the values (output back-rotated if vrot)
  int rb;                      // FlagRot block size (power of two, 4..128)
  int vrot;                    // 1: rotate the output back (else: rotation folded into W_o offline)
  float* opart;                // [B, Hkv, nsplit, 4, D] fp32 split partials (unnormalized)
  float* mlpart;               // [B, Hkv, nsplit, 4, 2] (running max (log2 domain), sum)
  int* counter;                // [B * Hkv] zero-initialized; reset by the last CTA of each (b, kv head)
  void* out;                   // out [B, Hq * D] bf16
  long long o_bs;              // out batch stride (elements)
};

// variant: (tokens per warp iteration NT x 16, cp.async pipeline stages ST) = 0:(1,3) 1:(1,4) 2:(2,3) 3:(2,4) 4:(1,6)
// 5:(2,6) 6:(2,2) 7:(4,2); warpgroup (wgmma) kernel, 64-token blocks: 8:(ST 2) 9:(ST 3) 10:(ST 4) 11:(ST 6);
// hybrid (QK wgmma, PV mma.sync), 64-token blocks: 12:(ST 2) 13:(ST 3) 14:(ST 4) 15:(ST 6)
int attn_decode_launch(int fmt, bool qrot, int variant, const AttnParams& p, cudaStream_t st);
int attn_occupancy(int fmt, bool qrot, int variant);   // resident CTAs / SM (0 = variant not available)
// stand-alone q pre-rotation (FlagRot), q [B, 4*Hkv, D] bf16 -> out (bf16, contiguous)
void qrot_launch(const void* q, long long q_bs, void* out, int B, int Hkv, const uint8_t* kmask, int rb,
                 cudaStream_t st);
