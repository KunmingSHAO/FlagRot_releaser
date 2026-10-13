// torch bindings for the quantized-KV decode attention kernel.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "attn_decode.cuh"

namespace {

void check_cache(const torch::Tensor& t, const char* n, int64_t B, int64_t Hkv, int64_t row) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kUInt8 && t.dim() == 4, n, " must be a CUDA uint8 [B, Lmax, Hkv, bytes] view");
  TORCH_CHECK(t.size(0) == B && t.size(2) == Hkv && t.size(3) == row && t.stride(3) == 1 && t.stride(2) == row, n,
              ": expected [B, Lmax, Hkv, ", row, "] with contiguous head rows");
}

// q [B, Hq, D] bf16 (head rows contiguous); kc / vc: uint8 views [B, Lmax, Hkv, row bytes] of the KV cache codes;
// ks / vs: uint8 views [B, Lmax, Hkv, scale bytes]; out [B, Hq * D] bf16 (row stride arbitrary).
void attn_decode(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, c10::optional<torch::Tensor> ks,
                 c10::optional<torch::Tensor> vs, int64_t L, int64_t fmt, int64_t nsplit, int64_t chunk, double qscale,
                 c10::optional<torch::Tensor> gscale, c10::optional<torch::Tensor> kmask, c10::optional<torch::Tensor> vmask,
                 int64_t rb, bool qrot, bool vrot, int64_t variant, torch::Tensor opart, torch::Tensor mlpart,
                 torch::Tensor counter, torch::Tensor out) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.dim() == 3 && q.size(2) == 128 &&
                  q.stride(2) == 1 && q.stride(1) == 128, "q must be bf16 [B, Hq, 128] with contiguous heads");
  const int64_t B = q.size(0), Hq = q.size(1), Hkv = Hq / 4;
  TORCH_CHECK(Hq == 4 * Hkv, "GQA group must be 4");
  static const int RB[5] = {256, 128, 64, 64, 64}, SB[5] = {0, 4, 8, 4, 8};
  TORCH_CHECK(fmt >= 0 && fmt <= 4, "fmt");
  check_cache(kc, "kc", B, Hkv, RB[fmt]);
  check_cache(vc, "vc", B, Hkv, RB[fmt]);
  TORCH_CHECK(kc.stride(0) == vc.stride(0) && kc.stride(1) == vc.stride(1), "K and V caches must share strides");
  TORCH_CHECK(L >= 1 && L <= kc.size(1), "L out of range");
  AttnParams p{};
  p.q = q.data_ptr(); p.q_bs = q.stride(0);
  p.k = kc.data_ptr<uint8_t>(); p.v = vc.data_ptr<uint8_t>();
  p.tok_stride = kc.stride(1); p.bat_stride = kc.stride(0);
  if (SB[fmt]) {
    TORCH_CHECK(ks.has_value() && vs.has_value(), "scales required");
    check_cache(*ks, "ks", B, Hkv, SB[fmt]);
    check_cache(*vs, "vs", B, Hkv, SB[fmt]);
    TORCH_CHECK(ks->stride(0) == vs->stride(0) && ks->stride(1) == vs->stride(1), "K and V scales must share strides");
    p.ks = ks->data_ptr<uint8_t>(); p.vs = vs->data_ptr<uint8_t>();
    p.stok_stride = ks->stride(1); p.sbat_stride = ks->stride(0);
  }
  p.B = static_cast<int>(B); p.L = static_cast<int>(L); p.Hkv = static_cast<int>(Hkv);
  p.nsplit = static_cast<int>(nsplit); p.chunk = static_cast<int>(chunk);
  TORCH_CHECK(nsplit >= 1 && nsplit <= 512 && chunk >= 1 && nsplit * chunk >= L && (nsplit - 1) * chunk < L, "bad split");
  p.qscale = static_cast<float>(qscale);
  if (fmt == AF_NVFP4) {
    TORCH_CHECK(gscale.has_value() && gscale->is_cuda() && gscale->scalar_type() == torch::kFloat32, "nvfp4 needs gscale");
    p.gscale = gscale->data_ptr<float>();
  }
  p.rb = static_cast<int>(rb);
  if (qrot || vrot) TORCH_CHECK(rb >= 4 && rb <= 128 && (rb & (rb - 1)) == 0, "rb must be a power of two in [4, 128]");
  if (qrot) {
    TORCH_CHECK(kmask.has_value() && kmask->is_cuda() && kmask->scalar_type() == torch::kUInt8 && kmask->is_contiguous() &&
                    kmask->numel() == Hkv * 128 / rb, "kmask must be uint8 [Hkv, D/rb]");
    p.kmask = kmask->data_ptr<uint8_t>();
  }
  if (vrot) {
    TORCH_CHECK(vmask.has_value() && vmask->is_cuda() && vmask->scalar_type() == torch::kUInt8 && vmask->is_contiguous() &&
                    vmask->numel() == Hkv * 128 / rb, "vmask must be uint8 [Hkv, D/rb]");
    p.vmask = vmask->data_ptr<uint8_t>();
  }
  p.vrot = vrot;
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
  const int rc = attn_decode_launch(static_cast<int>(fmt), qrot, static_cast<int>(variant), p, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(rc == 0, "attn_decode_launch: unsupported configuration (code ", rc, ")");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor qrot(torch::Tensor q, torch::Tensor kmask, int64_t rb) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.dim() == 3 && q.size(2) == 128 &&
                  q.stride(2) == 1 && q.stride(1) == 128, "q must be bf16 [B, Hq, 128]");
  const int64_t B = q.size(0), Hkv = q.size(1) / 4;
  TORCH_CHECK(kmask.is_cuda() && kmask.scalar_type() == torch::kUInt8 && kmask.numel() == Hkv * 128 / rb, "kmask");
  auto out = torch::empty({B, q.size(1), 128}, q.options());
  const at::cuda::CUDAGuard guard(q.device());
  qrot_launch(q.data_ptr(), q.stride(0), out.data_ptr(), static_cast<int>(B), static_cast<int>(Hkv),
              kmask.data_ptr<uint8_t>(), static_cast<int>(rb), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

int64_t occupancy(int64_t fmt, bool qrot, int64_t variant) {
  return attn_occupancy(static_cast<int>(fmt), qrot, static_cast<int>(variant));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_decode", &attn_decode);
  m.def("qrot", &qrot);
  m.def("occupancy", &occupancy);
}
