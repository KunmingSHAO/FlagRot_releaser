// torch bindings for the v2 paged decode attention (csrc/attn_decode_v2.cu) and the paged KV writer.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "attn_decode_v2.cuh"

namespace {

void check_u8(const torch::Tensor& t, const char* n) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kUInt8, n, " must be a CUDA uint8 tensor");
}

// q [B, Hq, 128] bf16; pool uint8 [num_pages, Hkv, CHUNK]; block_table int32 [B, max_pages]; seq_lens int32 [B]
void attn2(torch::Tensor q, torch::Tensor pool, torch::Tensor block_table, torch::Tensor seq_lens, int64_t fmt,
           int64_t page, int64_t nsplit, int64_t chunk, double qscale, c10::optional<torch::Tensor> gscale,
           c10::optional<torch::Tensor> kmask, c10::optional<torch::Tensor> vmask, int64_t rb, bool qrot, bool vrot,
           int64_t variant, int64_t stages, torch::Tensor opart, torch::Tensor mlpart, torch::Tensor counter,
           torch::Tensor out, c10::optional<torch::Tensor> tail, c10::optional<torch::Tensor> tail_len,
           int64_t sk_grid) {
  const int64_t D = q.size(2);
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.dim() == 3 && (D == 128 || D == 256) &&
                  q.stride(2) == 1 && q.stride(1) == D, "q must be bf16 [B, Hq, 128 | 256] with contiguous heads");
  check_u8(pool, "pool");
  const int64_t B = q.size(0), Hq = q.size(1), Hkv = pool.size(1), gqa = Hq / Hkv;
  TORCH_CHECK(Hq == gqa * Hkv && gqa >= 1, "q heads must be a multiple of the kv heads");
  TORCH_CHECK((gqa == 4 && D == 128) || (variant & ATTN2_GEN), "GQA group != 4 or head dim 256: generalized variant (ATTN2_GEN)");
  TORCH_CHECK(((variant & ATTN2_HD256) != 0) == (D == 256), "head dim 256 <-> variant bit ATTN2_HD256");
  TORCH_CHECK(pool.dim() == 3 && pool.size(2) == (D / 128) * attn2_chunk_bytes(fmt, page) && pool.is_contiguous(),
              "pool must be contiguous uint8 [num_pages, Hkv, chunk bytes]");
  const int64_t GT = (variant & ATTN2_GEN) ? ((variant & ATTN2_GT8) ? 8 : 4) : 4;
  const int64_t HV = Hkv * ((gqa + GT - 1) / GT);  // kernel "virtual heads" (q tiles)
  TORCH_CHECK(block_table.is_cuda() && block_table.scalar_type() == torch::kInt32 && block_table.dim() == 2 &&
                  block_table.size(0) == B && block_table.stride(1) == 1, "block_table must be int32 [B, max_pages]");
  TORCH_CHECK(seq_lens.is_cuda() && seq_lens.scalar_type() == torch::kInt32 && seq_lens.numel() == B &&
                  seq_lens.is_contiguous(), "seq_lens must be int32 [B]");
  TORCH_CHECK(chunk % 64 == 0 && chunk % page == 0 && chunk <= 8192 && nsplit >= 1 && nsplit <= 512, "bad split (chunk must be a multiple of 64 and of the page size, <= 8192)");
  PagedParams p{};
  p.q = q.data_ptr(); p.q_bs = q.stride(0);
  p.pool = pool.data_ptr<uint8_t>();
  p.block_table = block_table.data_ptr<int>(); p.bt_stride = static_cast<int>(block_table.stride(0));
  p.seq_lens = seq_lens.data_ptr<int>();
  p.B = static_cast<int>(B); p.Hkv = static_cast<int>(Hkv); p.page = static_cast<int>(page); p.gqa = static_cast<int>(gqa);
  p.nsplit = static_cast<int>(nsplit); p.chunk = static_cast<int>(chunk);
  p.qscale = static_cast<float>(qscale);
  if (fmt == A2_NVFP4) {
    TORCH_CHECK(gscale.has_value() && gscale->is_cuda() && gscale->scalar_type() == torch::kFloat32, "nvfp4 needs gscale");
    p.gscale = gscale->data_ptr<float>();
  }
  p.rb = static_cast<int>(rb); p.qrot = qrot; p.vrot = vrot;
  if (qrot || vrot) TORCH_CHECK(rb >= 4 && rb <= 128 && (rb & (rb - 1)) == 0, "rb must be a power of two in [4, 128]");
  if (qrot) {
    TORCH_CHECK(kmask.has_value() && kmask->is_cuda() && kmask->scalar_type() == torch::kUInt8 && kmask->is_contiguous() &&
                    kmask->numel() == Hkv * D / rb, "kmask must be uint8 [Hkv, D/rb]");
    p.kmask = kmask->data_ptr<uint8_t>();
  }
  if (vrot || (tail.has_value() && vmask.has_value())) {  // (the tail CTA rotates its partial into the V basis)
    TORCH_CHECK(vmask.has_value() && vmask->is_cuda() && vmask->scalar_type() == torch::kUInt8 && vmask->is_contiguous() &&
                    vmask->numel() == Hkv * D / rb, "vmask must be uint8 [Hkv, D/rb]");
    p.vmask = vmask->data_ptr<uint8_t>();
  }
  if (tail.has_value()) {  // BF16 staging window [B, W <= 256, 2*Hkv, 128] + lengths [B]: one more split slot
    TORCH_CHECK(tail->is_cuda() && tail->scalar_type() == torch::kBFloat16 && tail->dim() == 4 && tail->size(0) == B &&
                    tail->size(1) <= 256 && tail->size(2) == 2 * Hkv && tail->size(3) == D && tail->stride(3) == 1 &&
                    tail->stride(2) == D, "tail must be bf16 [B, W <= 256, 2*Hkv, D] with contiguous head rows");
    TORCH_CHECK(tail_len.has_value() && tail_len->is_cuda() && tail_len->scalar_type() == torch::kInt32 &&
                    tail_len->numel() == B && tail_len->is_contiguous(), "tail_len must be int32 [B]");
    p.tail = tail->data_ptr(); p.tail_len = tail_len->data_ptr<int>();
    p.tail_bs = tail->stride(0); p.tail_ts = tail->stride(1);
    p.tail_nslot = 1;  // one CTA per (sequence, kv head) for the whole window (attn_tail.cuh)
    nsplit += p.tail_nslot;
    p.nsplit = static_cast<int>(nsplit);
    TORCH_CHECK(p.rb >= 1, "rb");
  }
  // stream-K schedule (variants with MODE bit 9): sk_grid CTAs, partial slots 2 sk_grid + B Hkv tail_nslot
  TORCH_CHECK(sk_grid >= 0 && sk_grid <= 65535, "sk_grid");
  TORCH_CHECK(((variant & 512) != 0) == (sk_grid > 0), "sk_grid > 0 exactly for the stream-K variants");
  if (sk_grid > 0) TORCH_CHECK(B <= 256 && sk_grid % Hkv == 0, "stream-K: B <= 256, sk_grid a multiple of Hkv");
  p.sk_grid = static_cast<int>(sk_grid);
  const int64_t nslots = sk_grid > 0 ? 2 * sk_grid + B * Hkv * p.tail_nslot : B * HV * nsplit;
  if (nsplit > 1 || sk_grid > 0) {
    TORCH_CHECK(opart.is_cuda() && opart.scalar_type() == torch::kFloat32 && opart.numel() >= nslots * GT * D, "opart too small");
    TORCH_CHECK(mlpart.is_cuda() && mlpart.scalar_type() == torch::kFloat32 && mlpart.numel() >= nslots * GT * 2, "mlpart too small");
    TORCH_CHECK(counter.is_cuda() && counter.scalar_type() == torch::kInt32 && counter.numel() >= B * HV, "counter too small");
    p.opart = opart.data_ptr<float>(); p.mlpart = mlpart.data_ptr<float>(); p.counter = counter.data_ptr<int>();
  }
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16 && out.dim() == 2 && out.size(0) == B &&
                  out.size(1) == Hq * D && out.stride(1) == 1, "out must be bf16 [B, Hq*D]");
  p.out = out.data_ptr(); p.o_bs = out.stride(0);
  const at::cuda::CUDAGuard guard(q.device());
  const int rc = attn2_launch(static_cast<int>(fmt), static_cast<int>(variant), static_cast<int>(stages), p,
                              at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(rc == 0, "attn2_launch: unsupported configuration (code ", rc, ")");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// kv [T, 2*Hkv, 128] bf16 (row stride arbitrary, head rows contiguous) -> pool at slots [T] (int64; < 0: skip)
void paged_write(torch::Tensor kv, torch::Tensor slots, torch::Tensor mask, int64_t rb, int64_t fmt,
                 c10::optional<torch::Tensor> gscale, torch::Tensor pool, int64_t page) {
  const int64_t D = kv.size(2);
  TORCH_CHECK(kv.is_cuda() && kv.scalar_type() == torch::kBFloat16 && kv.dim() == 3 && (D == 128 || D == 256) &&
                  kv.stride(2) == 1 && kv.stride(1) == D, "kv must be bf16 [T, 2*Hkv, 128 | 256] with contiguous head rows");
  const int64_t T = kv.size(0), H2 = kv.size(1), Hkv = H2 / 2;
  TORCH_CHECK(slots.is_cuda() && slots.scalar_type() == torch::kInt64 && slots.numel() == T && slots.is_contiguous(), "slots");
  TORCH_CHECK(mask.is_cuda() && mask.scalar_type() == torch::kUInt8 && mask.is_contiguous() && mask.numel() == H2 * D / rb, "mask");
  TORCH_CHECK(rb >= 4 && rb <= 128 && (rb & (rb - 1)) == 0, "rb");
  check_u8(pool, "pool");
  TORCH_CHECK(pool.dim() == 3 && pool.size(1) == Hkv && pool.size(2) == (D / 128) * attn2_chunk_bytes(fmt, page) && pool.is_contiguous(), "pool");
  TORCH_CHECK(D == 128 || rb <= 128, "head dim 256: rb <= 128 (block flags inside each 128-d half)");
  PagedWriteParams p{};
  p.hd = static_cast<int>(D);
  p.kv = kv.data_ptr(); p.kv_ts = kv.stride(0); p.slots = reinterpret_cast<const long long*>(slots.data_ptr<int64_t>());
  p.T = static_cast<int>(T); p.Hkv = static_cast<int>(Hkv); p.page = static_cast<int>(page);
  p.mask = mask.data_ptr<uint8_t>(); p.rb = static_cast<int>(rb); p.pool = pool.data_ptr<uint8_t>();
  if (fmt == A2_NVFP4) {
    TORCH_CHECK(gscale.has_value() && gscale->is_cuda() && gscale->scalar_type() == torch::kFloat32, "nvfp4 needs gscale");
    p.gscale = gscale->data_ptr<float>();
  }
  const at::cuda::CUDAGuard guard(kv.device());
  const int rc = paged_write_launch(static_cast<int>(fmt), p, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(rc == 0, "paged_write_launch: unsupported configuration (code ", rc, ")");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t occupancy(int64_t fmt, int64_t variant, int64_t stages, int64_t page) {
  return attn2_occupancy(static_cast<int>(fmt), static_cast<int>(variant), static_cast<int>(stages), static_cast<int>(page));
}
int64_t chunk_bytes(int64_t fmt, int64_t page) { return attn2_chunk_bytes(static_cast<int>(fmt), static_cast<int>(page)); }

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn2", &attn2);
  m.def("paged_write", &paged_write);
  m.def("occupancy", &occupancy);
  m.def("set_pdl", [](bool on) { attn2_set_pdl(on ? 1 : 0); });
  m.def("chunk_bytes", &chunk_bytes);
}
