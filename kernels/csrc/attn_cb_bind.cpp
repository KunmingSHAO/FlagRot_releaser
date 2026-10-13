// torch bindings for the codebook INT4 KV cache (csrc/attn_codebook.cu): writer and decode attention.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "attn_codebook.cuh"

namespace {

void check_u8(const torch::Tensor& t, const char* n) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kUInt8, n, " must be a CUDA uint8 tensor");
}
void set_sg(unsigned (&dst)[CB_NSG][4], const std::vector<int64_t>& sg) {
  TORCH_CHECK(sg.size() == 4 * CB_NSG, "sg must hold 4 words per sign vector");
  for (int k = 0; k < CB_NSG; ++k)
    for (int i = 0; i < 4; ++i) dst[k][i] = static_cast<unsigned>(sg[4 * k + i]);
}

// kv [T, 2*Hkv, 128] bf16 -> pool at slots [T]; CT [Hkv, 128, 128] fp32 (CT[h][k][i] = C_q[h][i][k]); co [Hkv, 128]
void cb_write(torch::Tensor kv, torch::Tensor slots, torch::Tensor CT, torch::Tensor co, std::vector<int64_t> sg,
              int64_t kind, torch::Tensor pool, int64_t page, int64_t tg) {
  const int64_t D = kv.size(2);
  TORCH_CHECK(kv.is_cuda() && kv.scalar_type() == torch::kBFloat16 && kv.dim() == 3 && (D == 128 || D == 256) &&
                  kv.stride(2) == 1 && kv.stride(1) == D, "kv must be bf16 [T, 2*Hkv, 128 | 256] with contiguous head rows");
  const int64_t T = kv.size(0), Hkv = kv.size(1) / 2;
  TORCH_CHECK(slots.is_cuda() && slots.scalar_type() == torch::kInt64 && slots.is_contiguous() &&
                  (slots.numel() == T || slots.numel() == T * Hkv), "slots must be int64 [T] or [T, Hkv]");
  TORCH_CHECK(CT.is_cuda() && CT.scalar_type() == torch::kFloat32 && CT.is_contiguous() && CT.numel() == Hkv * D * D, "CT");
  TORCH_CHECK(co.is_cuda() && co.scalar_type() == torch::kFloat32 && co.is_contiguous() && co.numel() == Hkv * D, "co");
  check_u8(pool, "pool");
  TORCH_CHECK(pool.dim() == 3 && pool.size(1) == Hkv && pool.size(2) == (D / 128) * cb_chunk_bytes(kind, page) && pool.is_contiguous(), "pool");
  TORCH_CHECK(tg >= 1 && tg <= 64, "tg");
  CbWriteParams p{};
  p.hd = static_cast<int>(D);
  p.kv = kv.data_ptr(); p.kv_ts = kv.stride(0);
  p.slots = reinterpret_cast<const long long*>(slots.data_ptr<int64_t>());
  p.slot_ph = slots.numel() == T * Hkv && Hkv > 1 ? 1 : 0;
  p.T = static_cast<int>(T); p.Hkv = static_cast<int>(Hkv); p.page = static_cast<int>(page); p.tg = static_cast<int>(tg);
  p.CT = CT.data_ptr<float>(); p.co = co.data_ptr<float>();
  set_sg(p.sg, sg);
  p.pool = pool.data_ptr<uint8_t>();
  const at::cuda::CUDAGuard guard(kv.device());
  const int rc = cb_write_launch(static_cast<int>(kind), p, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(rc == 0, "cb_write_launch: unsupported configuration (code ", rc, ")");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// writer v3: + C1 fp16 [Hkv, 168, 128], aux fp32 [Hkv, 48] (attn_codebook.w3_calib), nchunk, optional stats int32 [4]
void cb_write3(torch::Tensor kv, torch::Tensor slots, torch::Tensor CT, torch::Tensor co, torch::Tensor C1,
               torch::Tensor beta, std::vector<int64_t> sg, int64_t kind, torch::Tensor pool, int64_t page,
               int64_t nchunk, int64_t nchunk_v, c10::optional<torch::Tensor> stats) {
  TORCH_CHECK(kv.is_cuda() && kv.scalar_type() == torch::kBFloat16 && kv.dim() == 3 && kv.size(2) == 128 &&
                  kv.stride(2) == 1 && kv.stride(1) == 128 && kv.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(kv.data_ptr()) % 16 == 0,
              "kv must be bf16 [T, 2*Hkv, 128] with contiguous head rows, 16-byte aligned rows (token stride % 8 == 0)");
  const int64_t T = kv.size(0), Hkv = kv.size(1) / 2;
  TORCH_CHECK(slots.is_cuda() && slots.scalar_type() == torch::kInt64 && slots.is_contiguous() &&
                  (slots.numel() == T || slots.numel() == T * Hkv), "slots must be int64 [T] or [T, Hkv]");
  TORCH_CHECK(CT.is_cuda() && CT.scalar_type() == torch::kFloat32 && CT.is_contiguous() && CT.numel() == Hkv * 128 * 128, "CT");
  TORCH_CHECK(co.is_cuda() && co.scalar_type() == torch::kFloat32 && co.is_contiguous() && co.numel() == Hkv * 128, "co");
  TORCH_CHECK(C1.is_cuda() && C1.scalar_type() == torch::kFloat16 && C1.is_contiguous() && C1.numel() == Hkv * 168 * 128, "C1 (fp16 [Hkv, 168, 128])");
  TORCH_CHECK(beta.is_cuda() && beta.scalar_type() == torch::kFloat32 && beta.is_contiguous() && beta.numel() == Hkv * 48, "beta (aux fp32 [Hkv, 48])");
  check_u8(pool, "pool");
  TORCH_CHECK(pool.dim() == 3 && pool.size(1) == Hkv && pool.size(2) == cb_chunk_bytes(kind, page) && pool.is_contiguous(), "pool");
  CbWriteParams p{};
  p.kv = kv.data_ptr(); p.kv_ts = kv.stride(0);
  p.slots = reinterpret_cast<const long long*>(slots.data_ptr<int64_t>());
  p.slot_ph = slots.numel() == T * Hkv && Hkv > 1 ? 1 : 0;
  p.T = static_cast<int>(T); p.Hkv = static_cast<int>(Hkv); p.page = static_cast<int>(page); p.tg = 1;
  p.CT = CT.data_ptr<float>(); p.co = co.data_ptr<float>();
  p.C1 = C1.data_ptr(); p.beta = beta.data_ptr<float>(); p.nchunk = static_cast<int>(nchunk);
  p.nchunk_v = static_cast<int>(nchunk_v);
  if (stats.has_value()) {
    TORCH_CHECK(stats->is_cuda() && stats->scalar_type() == torch::kInt32 && stats->numel() >= 4, "stats");
    p.stats = stats->data_ptr<int>();
  }
  set_sg(p.sg, sg);
  p.pool = pool.data_ptr<uint8_t>();
  const at::cuda::CUDAGuard guard(kv.device());
  const int rc = cb_write3_launch(static_cast<int>(kind), p, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(rc == 0, "cb_write3_launch: unsupported configuration (code ", rc, ")");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// static FlagRot INT4 / INT4A writer in the v3 style: kv [T, 2*Hkv, 128] bf16, slots int64 [T], mask uint8
// [2*Hkv, 4] or None, hfull uint8 [2*Hkv] or None, pool uint8 [num_pages, Hkv, 144 * page]
void w3_static_write(torch::Tensor kv, torch::Tensor slots, c10::optional<torch::Tensor> mask,
                     c10::optional<torch::Tensor> hfull, int64_t asym, torch::Tensor pool, int64_t page,
                     int64_t path) {
  TORCH_CHECK(kv.is_cuda() && kv.scalar_type() == torch::kBFloat16 && kv.dim() == 3 && kv.size(2) == 128 &&
                  kv.stride(2) == 1 && kv.stride(1) == 128 && kv.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(kv.data_ptr()) % 16 == 0, "kv must be bf16 [T, 2*Hkv, 128], 16-byte aligned rows");
  const int64_t T = kv.size(0), Hkv = kv.size(1) / 2;
  TORCH_CHECK(slots.is_cuda() && slots.scalar_type() == torch::kInt64 && slots.is_contiguous() && slots.numel() == T, "slots");
  check_u8(pool, "pool");
  TORCH_CHECK(pool.dim() == 3 && pool.size(1) == Hkv && pool.size(2) == 144 * page && pool.is_contiguous(), "pool");
  W3StaticParams p{};
  p.kv = kv.data_ptr(); p.kv_ts = kv.stride(0);
  p.slots = reinterpret_cast<const long long*>(slots.data_ptr<int64_t>());
  p.T = static_cast<int>(T); p.Hkv = static_cast<int>(Hkv); p.page = static_cast<int>(page);
  if (mask.has_value()) {
    check_u8(*mask, "mask");
    TORCH_CHECK(mask->is_contiguous() && mask->numel() == 2 * Hkv * 4, "mask [2*Hkv, 4]");
    p.mask = mask->data_ptr<uint8_t>();
  }
  if (hfull.has_value()) {
    check_u8(*hfull, "hfull");
    TORCH_CHECK(hfull->is_contiguous() && hfull->numel() == 2 * Hkv, "hfull [2*Hkv]");
    p.hfull = hfull->data_ptr<uint8_t>();
  }
  p.asym = static_cast<int>(asym);
  p.path = static_cast<int>(path);
  p.pool = pool.data_ptr<uint8_t>();
  const at::cuda::CUDAGuard guard(kv.device());
  const int rc = w3_static_write_launch(p, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(rc == 0, "w3_static_write_launch: unsupported configuration (code ", rc, ")");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void cb_attn(torch::Tensor q, torch::Tensor pool, torch::Tensor block_table, torch::Tensor seq_lens, int64_t kind,
             int64_t page, int64_t nsplit, int64_t chunk, double qscale, std::vector<int64_t> sg, int64_t variant,
             int64_t stages,
             torch::Tensor opart, torch::Tensor mlpart, torch::Tensor counter, torch::Tensor out,
             c10::optional<torch::Tensor> tail, c10::optional<torch::Tensor> tail_len) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.dim() == 3 && q.size(2) == 128 &&
                  q.stride(2) == 1 && q.stride(1) == 128, "q must be bf16 [B, Hq, 128] with contiguous heads");
  const int64_t B = q.size(0), Hq = q.size(1), Hkv = Hq / 4;
  TORCH_CHECK(Hq == 4 * Hkv, "GQA group must be 4");
  check_u8(pool, "pool");
  TORCH_CHECK(pool.dim() == 3 && pool.size(1) == Hkv && pool.size(2) == cb_chunk_bytes(kind, page) && pool.is_contiguous(),
              "pool must be contiguous uint8 [num_pages, Hkv, chunk bytes]");
  TORCH_CHECK(block_table.is_cuda() && block_table.scalar_type() == torch::kInt32 && block_table.dim() == 2 &&
                  block_table.size(0) == B && block_table.stride(1) == 1, "block_table must be int32 [B, max_pages]");
  TORCH_CHECK(seq_lens.is_cuda() && seq_lens.scalar_type() == torch::kInt32 && seq_lens.numel() == B &&
                  seq_lens.is_contiguous(), "seq_lens must be int32 [B]");
  TORCH_CHECK(chunk % 64 == 0 && chunk % page == 0 && chunk <= 8192 && nsplit >= 1 && nsplit <= 512, "bad split");
  CbAttnParams p{};
  p.q = q.data_ptr(); p.q_bs = q.stride(0);
  p.pool = pool.data_ptr<uint8_t>();
  p.block_table = block_table.data_ptr<int>(); p.bt_stride = static_cast<int>(block_table.stride(0));
  p.seq_lens = seq_lens.data_ptr<int>();
  p.B = static_cast<int>(B); p.Hkv = static_cast<int>(Hkv); p.page = static_cast<int>(page);
  p.nsplit = static_cast<int>(nsplit); p.chunk = static_cast<int>(chunk);
  p.qscale = static_cast<float>(qscale);
  set_sg(p.sg, sg);
  if (tail.has_value()) {  // BF16 staging window [B, W <= 256, 2*Hkv, 128] + lengths [B]: one extra split CTA
    TORCH_CHECK(tail->is_cuda() && tail->scalar_type() == torch::kBFloat16 && tail->dim() == 4 && tail->size(0) == B &&
                    tail->size(1) <= 256 && tail->size(2) == 2 * Hkv && tail->size(3) == 128 && tail->stride(3) == 1 &&
                    tail->stride(2) == 128, "tail must be bf16 [B, W <= 256, 2*Hkv, 128] with contiguous head rows");
    TORCH_CHECK(tail_len.has_value() && tail_len->is_cuda() && tail_len->scalar_type() == torch::kInt32 &&
                    tail_len->numel() == B && tail_len->is_contiguous(), "tail_len must be int32 [B]");
    p.tail = tail->data_ptr(); p.tail_len = tail_len->data_ptr<int>();
    p.tail_bs = tail->stride(0); p.tail_ts = tail->stride(1);
    p.tail_nslot = 1;  // one CTA per (sequence, kv head) for the whole window (attn_tail.cuh)
    nsplit += p.tail_nslot;
    p.nsplit = static_cast<int>(nsplit);
  }
  if (nsplit > 1) {
    TORCH_CHECK(opart.is_cuda() && opart.scalar_type() == torch::kFloat32 && opart.numel() >= B * Hkv * nsplit * 4 * 128, "opart too small");
    TORCH_CHECK(mlpart.is_cuda() && mlpart.scalar_type() == torch::kFloat32 && mlpart.numel() >= B * Hkv * nsplit * 4 * 2, "mlpart too small");
    TORCH_CHECK(counter.is_cuda() && counter.scalar_type() == torch::kInt32 && counter.numel() >= B * Hkv, "counter too small");
    p.opart = opart.data_ptr<float>(); p.mlpart = mlpart.data_ptr<float>(); p.counter = counter.data_ptr<int>();
  }
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16 && out.dim() == 2 && out.size(0) == B &&
                  out.size(1) == Hq * 128 && out.stride(1) == 1, "out must be bf16 [B, Hq*D]");
  p.out = out.data_ptr(); p.o_bs = out.stride(0);
  const at::cuda::CUDAGuard guard(q.device());
  const int rc = cb_attn_launch(static_cast<int>(kind), static_cast<int>(variant), static_cast<int>(stages), p,
                                at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(rc == 0, "cb_attn_launch: unsupported configuration (code ", rc, ")");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t occupancy(int64_t kind, int64_t variant, int64_t stages, int64_t page) {
  return cb_attn_occupancy(static_cast<int>(kind), static_cast<int>(variant), static_cast<int>(stages), static_cast<int>(page));
}
int64_t chunk_bytes(int64_t kind, int64_t page) { return cb_chunk_bytes(static_cast<int>(kind), static_cast<int>(page)); }
int64_t write_smem(int64_t kind, int64_t tg, int64_t hd) {
  return cb_write_smem_hd(static_cast<int>(kind), static_cast<int>(tg), static_cast<int>(hd));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cb_write", &cb_write);
  m.def("cb_write3", &cb_write3);
  m.def("w3_static_write", &w3_static_write);
  m.def("cb_attn", &cb_attn);
  m.def("occupancy", &occupancy);
  m.def("chunk_bytes", &chunk_bytes);
  m.def("write_smem", &write_smem, "", py::arg("kind"), py::arg("tg"), py::arg("hd") = 128);
}
