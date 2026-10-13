// Flat C ABI over the FlagRot KV-cache kernels of kernels/csrc (no torch headers): the vLLM plugin loads the shared
// library built from this file + kernels/csrc/{attn_decode_v2,attn_codebook,attn_cb_write3}.cu with ctypes
// (vllm_flagrot/kernels.py).  Why a plain library: the kernels are built with the CUDA 12.9 toolkit (its ptxas
// pipelines the wgmma of these kernels, 12.4 serializes them), while the vLLM env runs
// torch 2.13 + CUDA 13.0; a torch extension would have to be compiled against torch 2.13 with a CUDA 13 toolkit.
// The library links the CUDA 12.9 runtime statically, needs only the driver (580 / CUDA 13.0 here) and takes raw
// device pointers and the caller's cudaStream_t (torch's current stream), so it works under CUDA graph capture.
//
// Stable entry points: the launchers of kernels/csrc/{attn_decode_v2,attn_codebook}.cuh (attn2_launch,
// paged_write_launch, cb_attn_launch, cb_write_launch, occupancy, chunk sizes) and, since ABI 6, the fused writers v3
// (cb_write3_launch, w3_static_write_launch.  Improved kernel versions with the same
// launchers and parameter structs drop in by rebuilding this library (fr_abi_version() bumps on any change here).
#include <cstdint>
#include <cstring>

#include "attn_codebook.cuh"
#include "attn_decode_v2.cuh"

extern "C" {

int fr_abi_version() { return 9; }  // 9 (int2c): optional static key centering kmu [Hkv, hd] fp32 (last argument of the four writers)
// was: 8  // 8 (Phase 3a): INT2 formats A2_INT2A (8) / A2_KONLYA2 (9), CB_KONLYA2 (11), w3 asym 2  // 5: gqa / head dim; 6: + writer v3 (fr_cb_write3, fr_w3_static_write);
                                    // 7: kv_start (masked leading tokens) of fr_v2_attn / fr_cb_attn (vLLM plugin Phase 2k)

// ------------------------------------------------------------------ static FlagRot INT4 (v2 kernel)
// fmt = A2_INT4 (4: symmetric, sign-magnitude codes) or A2_INT4A (5: asymmetric int4a:32, unsigned codes, zero point in
// the 4 low bits of the FP16 scale word = blockflag int4ah:32.  Same chunk layout / size.
// ABI 8 (Phase 3a): A2_INT2A (8: int2a:32 stored as int2ah:32, the INT2 page layout of attn_decode_v2.cuh, 80 B per
// (token, 128-d half)) and A2_KONLYA2 (9: the key-only 2x2 codebook cache CB_KONLYA2, generalized kernel).
// q [B, 4*Hkv, 128] bf16 (row stride q_bs elements); pool [num_pages, Hkv, chunk]; block_table int32 [B, bt_stride]
// (bt_stride may be 0: all rows share row 0); seq_lens int32 [B]; out [B, 4*Hkv*128] bf16 (row stride o_bs);
// kmask / vmask [Hkv, 128/rb] uint8 or null; tail (BF16 window [rows, W, 2*Hkv, 128], strides tail_bs / tail_ts
// elements) + tail_len int32 [B] (+ tail_idx int32 [B]: window row of sequence b) or null.
// nsplit counts the body splits only (the window adds one merge slot); opart / mlpart / counter: split-K workspace
// sized for B * Hkv * (nsplit + 1) partials (counter: B * Hkv ints, zero-initialized once, self-resetting).
// ABI 5): gqa = q heads per kv head; gqa != 4 or head dim 256 need the generalized kernel (variant
// 264 | ATTN2_GEN [| ATTN2_GT8] [| ATTN2_HD256]); q [B, gqa*Hkv, D], out [B, gqa*Hkv*D]; partial slots are then
// per q tile ("virtual head": Hkv * ceil(gqa / GT) of them) and hold GT x D floats.  fmt 6 / 7 = the key-only 2x2
// codebook caches (CB_KONLY / CB_KONLYA) read by the generalized kernel (no masks).
// ABI 7 (vLLM plugin Phase 2k): kv_start = tokens [0, kv_start) of every sequence masked in the paged body (the BF16
// sink tokens are stored at their real positions and attended from the tail rows); 0 = none.  Same for fr_cb_attn.
int fr_v2_attn(int fmt, const void* q, long long q_bs, const uint8_t* pool, const int* block_table, int bt_stride,
               const int* seq_lens, int B, int Hkv, int page, int nsplit, int chunk, float qscale,
               const uint8_t* kmask, const uint8_t* vmask, int rb, int qrot, int vrot, int variant, int stages,
               float* opart, float* mlpart, int* counter, void* out, long long o_bs, const void* tail,
               const int* tail_len, const int* tail_idx, long long tail_bs, long long tail_ts, void* stream, int gqa,
               int kv_start) {
  PagedParams p{};
  p.gqa = gqa;
  p.kv_start = kv_start;
  p.q = q; p.q_bs = q_bs; p.pool = pool; p.block_table = block_table; p.bt_stride = bt_stride; p.seq_lens = seq_lens;
  p.B = B; p.Hkv = Hkv; p.page = page; p.nsplit = nsplit; p.chunk = chunk; p.qscale = qscale; p.gscale = nullptr;
  p.kmask = qrot ? kmask : nullptr;
  p.rb = rb; p.qrot = qrot; p.vrot = vrot;
  p.vmask = (vrot || tail != nullptr) ? vmask : nullptr;
  if (tail != nullptr) {
    p.tail = tail; p.tail_len = tail_len; p.tail_idx = tail_idx; p.tail_bs = tail_bs; p.tail_ts = tail_ts;
    p.tail_nslot = 1;
    p.nsplit = nsplit + 1;
  }
  p.opart = opart; p.mlpart = mlpart; p.counter = counter;
  p.out = out; p.o_bs = o_bs;
  return attn2_launch(fmt, variant, stages, p, static_cast<cudaStream_t>(stream));
}

// kv [T, 2*Hkv, hd] bf16 (token stride kv_ts elements) -> pool at slots int64 [T] (< 0: skipped); mask [2*Hkv, hd/rb]
// (hd 256: two 128-d half chunks per (page, kv head))
int fr_v2_write(int fmt, const void* kv, long long kv_ts, const long long* slots, int T, int Hkv, int page,
                const uint8_t* mask, int rb, uint8_t* pool, void* stream, int hd, const float* kmu) {
  if (T <= 0) return 0;
  PagedWriteParams p{};
  p.hd = hd;
  p.kmu = kmu;
  p.kv = kv; p.kv_ts = kv_ts; p.slots = slots; p.T = T; p.Hkv = Hkv; p.page = page; p.mask = mask; p.rb = rb;
  p.gscale = nullptr; p.pool = pool;
  return paged_write_launch(fmt, p, static_cast<cudaStream_t>(stream));
}

int fr_v2_occupancy(int fmt, int variant, int stages, int page) { return attn2_occupancy(fmt, variant, stages, page); }
int fr_v2_chunk_bytes(int fmt, int page) { return attn2_chunk_bytes(fmt, page); }

// ------------------------------------------------------------------ per-token codebook (kind = CbKind)
// sg: CB_NSG x 4 sign words (bit d of word d/32: S_d = -1)
int fr_cb_nsg() { return CB_NSG; }

int fr_cb_write(const void* kv, long long kv_ts, const long long* slots, int slot_ph, int T, int Hkv, int page,
                int tg, const float* CT, const float* co, const unsigned* sg, int kind, uint8_t* pool, void* stream,
                int hd, const float* kmu) {
  if (T <= 0) return 0;
  CbWriteParams p{};
  p.hd = hd;
  p.kmu = kmu;  // (256: key-only 2x2 kinds, CT [Hkv, 256, 256], co [Hkv, 256])
  p.kv = kv; p.kv_ts = kv_ts; p.slots = slots; p.slot_ph = slot_ph; p.T = T; p.Hkv = Hkv; p.page = page; p.tg = tg;
  p.CT = CT; p.co = co;
  std::memcpy(p.sg, sg, sizeof(p.sg));
  p.pool = pool;
  return cb_write_launch(kind, p, static_cast<cudaStream_t>(stream));
}

// writer v3 (kernels/csrc/attn_cb_write3.cuh): byte-identical pages to fr_cb_write.  C1 fp16
// [Hkv, 168, 128] + beta fp32 [Hkv, 48] = attn_codebook.w3_calib(C_q) (kernels.w3_calib); nchunk / nchunk_v = resident
// key / value CTAs per kv head, < 0: skip that kernel (decode steps run K and V on two streams: the key-only 2x2 kinds
// split the group-0 V scale word bitwise between the two kernels).  kv rows 16-byte aligned, kv_ts % 8 == 0.
int fr_cb_write3(const void* kv, long long kv_ts, const long long* slots, int slot_ph, int T, int Hkv, int page,
                 const float* CT, const float* co, const void* C1, const float* beta, const unsigned* sg, int kind,
                 uint8_t* pool, int nchunk, int nchunk_v, void* stream, const float* kmu) {
  if (T <= 0) return 0;
  CbWriteParams p{};
  p.kmu = kmu;
  p.kv = kv; p.kv_ts = kv_ts; p.slots = slots; p.slot_ph = slot_ph; p.T = T; p.Hkv = Hkv; p.page = page; p.tg = 1;
  p.CT = CT; p.co = co; p.C1 = C1; p.beta = beta; p.nchunk = nchunk; p.nchunk_v = nchunk_v; p.stats = nullptr;
  std::memcpy(p.sg, sg, sizeof(p.sg));
  p.pool = pool;
  return cb_write3_launch(kind, p, static_cast<cudaStream_t>(stream));
}

// static INT4 / INT4A writer v3: = fr_head_fwht (hfull heads -> bf16(H128 x / 16)) + fr_v2_write (rb 32,
// mask [2*Hkv, 4]) fused, byte for byte.  hfull uint8 [2*Hkv] or null (K rows first); asym 0: int4, 1: int4a, 2: int2a;
// path 0: by T (T <= 128: k_w3s_dec), 1: k_w3s (prefill, tiles of 32 tokens), 2: k_w3s_dec (decode, scattered slots).
int fr_w3_static_write(const void* kv, long long kv_ts, const long long* slots, int T, int Hkv, int page,
                       const uint8_t* mask, const uint8_t* hfull, int asym, uint8_t* pool, int path, void* stream,
                       const float* kmu) {
  if (T <= 0) return 0;
  W3StaticParams p{};
  p.kmu = kmu;
  p.kv = kv; p.kv_ts = kv_ts; p.slots = slots; p.T = T; p.Hkv = Hkv; p.page = page; p.mask = mask; p.hfull = hfull;
  p.asym = asym; p.pool = pool; p.path = path;
  return w3_static_write_launch(p, static_cast<cudaStream_t>(stream));
}

int fr_cb_attn(const void* q, long long q_bs, const uint8_t* pool, const int* block_table, int bt_stride,
               const int* seq_lens, int B, int Hkv, int page, int nsplit, int chunk, float qscale, const unsigned* sg,
               int kind, int variant, int stages, float* opart, float* mlpart, int* counter, void* out, long long o_bs,
               const void* tail, const int* tail_len, const int* tail_idx, long long tail_bs, long long tail_ts,
               void* stream, int kv_start) {
  CbAttnParams p{};
  p.kv_start = kv_start;
  p.q = q; p.q_bs = q_bs; p.pool = pool; p.block_table = block_table; p.bt_stride = bt_stride; p.seq_lens = seq_lens;
  p.B = B; p.Hkv = Hkv; p.page = page; p.nsplit = nsplit; p.chunk = chunk; p.qscale = qscale;
  std::memcpy(p.sg, sg, sizeof(p.sg));
  if (tail != nullptr) {
    p.tail = tail; p.tail_len = tail_len; p.tail_idx = tail_idx; p.tail_bs = tail_bs; p.tail_ts = tail_ts;
    p.tail_nslot = 1;
    p.nsplit = nsplit + 1;
  }
  p.opart = opart; p.mlpart = mlpart; p.counter = counter;
  p.out = out; p.o_bs = o_bs;
  return cb_attn_launch(kind, variant, stages, p, static_cast<cudaStream_t>(stream));
}

int fr_cb_occupancy(int kind, int variant, int stages, int page) { return cb_attn_occupancy(kind, variant, stages, page); }
int fr_cb_chunk_bytes(int kind, int page) { return cb_chunk_bytes(kind, page); }
int fr_cb_write_smem(int kind, int tg, int hd) { return cb_write_smem_hd(kind, tg, hd); }

const char* fr_last_error() { return cudaGetErrorString(cudaGetLastError()); }

}  // extern "C"

// ------------------------------------------------------------------ whole-head Hadamard of selected heads (static mode)
// out[t, h, :] = flags[h] ? scale * H128 x[t, h, :] : x[t, h, :]  with the unnormalized Sylvester H128 (+-1) in fp32
// (exact butterflies; with scale a power of two the only rounding is the final bf16 store).  One warp per (t, h),
// 4 elements per lane; x and out may alias.  Used for k, v -> H x / 16 before the writer, q -> H q / 8 before the
// attention and o -> H o / 8 after it (flagrot_int4_static heads whose static choice is the whole-head transform).
#include <cuda_bf16.h>

namespace {
__global__ void k_head_fwht(const __nv_bfloat16* __restrict__ x, long long x_ts, __nv_bfloat16* out, long long o_ts,
                            int T, int H, const uint8_t* __restrict__ flags, float scale) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (warp >= T * H) return;
  const int t = warp / H, h = warp % H;
  const __nv_bfloat16* xr = x + t * x_ts + h * 128;
  __nv_bfloat16* orow = out + t * o_ts + h * 128;
  const uint2 raw = *reinterpret_cast<const uint2*>(xr + 4 * lane);
  if (!flags[h]) {
    *reinterpret_cast<uint2*>(orow + 4 * lane) = raw;
    return;
  }
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
  float v[4];
  const float2 a = __bfloat1622float2(p[0]), b = __bfloat1622float2(p[1]);
  v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
  // within the lane: index bits 0, 1
  float s0 = v[0] + v[1], d0 = v[0] - v[1], s1 = v[2] + v[3], d1 = v[2] - v[3];
  v[0] = s0 + s1; v[1] = d0 + d1; v[2] = s0 - s1; v[3] = d0 - d1;
  // across lanes: index bits 2..6 = lane bits 0..4
#pragma unroll
  for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float o = __shfl_xor_sync(0xffffffffu, v[i], m);
      v[i] = (lane & m) ? (o - v[i]) : (v[i] + o);
    }
  }
  __nv_bfloat162 r[2] = {__floats2bfloat162_rn(v[0] * scale, v[1] * scale), __floats2bfloat162_rn(v[2] * scale, v[3] * scale)};
  *reinterpret_cast<uint2*>(orow + 4 * lane) = *reinterpret_cast<uint2*>(r);
}
}  // namespace

// D = 256 (unnormalized Sylvester H256, H H = 256 I): one warp per (t, h), 8 elements per lane (d = 4 lane + i and
// 128 + 4 lane + i): H128 butterflies on each half, then (a + b, a - b) across the halves
namespace {
__global__ void k_head_fwht256(const __nv_bfloat16* __restrict__ x, long long x_ts, __nv_bfloat16* out, long long o_ts,
                               int T, int H, const uint8_t* __restrict__ flags, float scale) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (warp >= T * H) return;
  const int t = warp / H, h = warp % H;
  const __nv_bfloat16* xr = x + t * x_ts + h * 256;
  __nv_bfloat16* orow = out + t * o_ts + h * 256;
  uint2 raw[2];
  raw[0] = *reinterpret_cast<const uint2*>(xr + 4 * lane);
  raw[1] = *reinterpret_cast<const uint2*>(xr + 128 + 4 * lane);
  if (!flags[h]) {
    *reinterpret_cast<uint2*>(orow + 4 * lane) = raw[0];
    *reinterpret_cast<uint2*>(orow + 128 + 4 * lane) = raw[1];
    return;
  }
  float v[2][4];
#pragma unroll
  for (int s = 0; s < 2; ++s) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw[s]);
    const float2 a = __bfloat1622float2(p[0]), b = __bfloat1622float2(p[1]);
    v[s][0] = a.x; v[s][1] = a.y; v[s][2] = b.x; v[s][3] = b.y;
    float s0 = v[s][0] + v[s][1], d0 = v[s][0] - v[s][1], s1 = v[s][2] + v[s][3], d1 = v[s][2] - v[s][3];
    v[s][0] = s0 + s1; v[s][1] = d0 + d1; v[s][2] = s0 - s1; v[s][3] = d0 - d1;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float o = __shfl_xor_sync(0xffffffffu, v[s][i], m);
        v[s][i] = (lane & m) ? (o - v[s][i]) : (v[s][i] + o);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float a = v[0][i], b = v[1][i];
    v[0][i] = a + b;
    v[1][i] = a - b;
  }
#pragma unroll
  for (int s = 0; s < 2; ++s) {
    __nv_bfloat162 r[2] = {__floats2bfloat162_rn(v[s][0] * scale, v[s][1] * scale), __floats2bfloat162_rn(v[s][2] * scale, v[s][3] * scale)};
    *reinterpret_cast<uint2*>(orow + 128 * s + 4 * lane) = *reinterpret_cast<uint2*>(r);
  }
}
}  // namespace

// D = 128 | 256 (ABI 5)
extern "C" int fr_head_fwht(const void* x, long long x_ts, void* out, long long o_ts, int T, int H,
                            const uint8_t* flags, float scale, void* stream, int D) {
  if (T <= 0) return 0;
  const long long warps = static_cast<long long>(T) * H;
  const int threads = 256;
  const unsigned blocks = static_cast<unsigned>((warps * 32 + threads - 1) / threads);
  if (D == 256)
    k_head_fwht256<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(x), x_ts, static_cast<__nv_bfloat16*>(out), o_ts, T, H, flags, scale);
  else if (D == 128)
    k_head_fwht<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(x), x_ts, static_cast<__nv_bfloat16*>(out), o_ts, T, H, flags, scale);
  else
    return -8;
  return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}
