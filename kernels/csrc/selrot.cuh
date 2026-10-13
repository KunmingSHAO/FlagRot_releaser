// Shared declarations for the selective-FWHT + quantization kernels (no torch headers here -> fast nvcc builds).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// prologue applied to the bf16 input before the selective FWHT
enum SelrotPro : int { PRO_NONE = 0, PRO_SWIGLU = 1, PRO_RMS = 2 };
// output format
enum SelrotOut : int {
  OUT_FP8_MN = 0,   // fp8 e4m3 codes [T,K], fp32 scales [K/g, T] (MN-major, DeepGEMM layout)
  OUT_FP8_ROW = 1,  // fp8 e4m3 codes [T,K], fp32 scales [T, K/g] (row-major; KV cache per-(token,head) when g = D)
  OUT_NVFP4 = 2,    // e2m1 packed 2/byte [T,K/2], e4m3 scales per 16 [T,K/16], times global fp32 scale
  OUT_MXFP4 = 3,    // e2m1 packed 2/byte [T,K/2], e8m0 scales per 32 [T,K/32]
  OUT_INT4 = 4      // symmetric int4 (code = q + 8, 2/byte, low nibble = even element) [T,K/2], fp16 scale per 32
};

struct SelrotParams {
  const void* x;        // PRO_NONE / PRO_RMS: x [T, K] bf16; PRO_SWIGLU: gate
  const void* x2;       // PRO_SWIGLU: up; PRO_RMS: rms weight [K]
  const uint8_t* mask;  // [K / rb], element k rotated iff mask[k / rb] != 0
  void* q;              // codes
  void* s;              // scales
  const float* gscale;  // OUT_NVFP4: global scale (device scalar)
  long long T;          // rows (tokens)
  long long sld;        // OUT_FP8_MN: leading dim of the [K/g, sld] scale buffer (sld >= T; DeepGEMM wants sld % 4 == 0)
  int K;                // row length (output)
  long long ld1, ld2;   // row strides in elements of x / x2 (PRO_SWIGLU, PRO_RMS for x)
  float eps;            // PRO_RMS
  int hf_round;         // 1: HF-style bf16 roundings in the prologue (bit-matches the eager unfused path)
  const int* perm;      // PRO_RMS only, optional: [K/8] source 8-element chunk of each output chunk (gather on load)
  // PRO_NONE with OUT_FP8_ROW / NVFP4 / MXFP4 / INT4 only (KV-cache writes), optional (0 = contiguous); needs
  // K % 2048 == 0 so that a 32-slot slab never straddles rows:
  //   ld1: input row stride (elements), ldq: output code row stride (bytes), lds: scale row stride (scale words:
  //   fp32 for FP8_ROW, uint32 (4 scales) for NVFP4 / INT4 (2 fp16 scales), uint16 (2 e8m0) for MXFP4)
  long long ldq, lds;
};

// returns 0 on success, <0 on unsupported configuration
int selrot_launch(int pro, int out, int rb, int g, const SelrotParams& p, int num_sms, cudaStream_t st);
// standalone bf16 helpers (unfused baselines)
void rmsnorm_bf16_launch(const void* x, const void* w, void* y, long long T, int K, float eps, cudaStream_t st);
void silu_mul_bf16_launch(const void* g, const void* u, void* y, long long T, int I, long long ldg, long long ldu,
                          cudaStream_t st);
void cast_bf16_fp8_launch(const void* x, void* q, long long n, int num_sms, cudaStream_t st);
