// torch bindings for the selective-FWHT kernels.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "selrot.cuh"

namespace {

void check_x(const torch::Tensor& x, const char* n) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16, n, " must be a CUDA bf16 tensor");
}
int num_sms(const torch::Tensor& x) {
  return at::cuda::getDeviceProperties(x.get_device())->multiProcessorCount;
}
void check_mask(const torch::Tensor& mask, int64_t n) {
  TORCH_CHECK(mask.is_cuda() && mask.scalar_type() == torch::kUInt8 && mask.is_contiguous(),
              "mask must be a contiguous CUDA uint8 tensor");
  TORCH_CHECK(mask.numel() == n, "mask must have K/rb = ", n, " entries, got ", mask.numel());
}
void check_cfg(int64_t K, int64_t rb, int64_t g) {
  TORCH_CHECK(rb == 8 || rb == 16 || rb == 32 || rb == 64, "rb must be 8/16/32/64");
  TORCH_CHECK(g == 64 || g == 128, "g must be 64/128");
  TORCH_CHECK(g % rb == 0 && K % g == 0, "need rb | g | K");
}
void run(int pro, int out, int rb, int g, const SelrotParams& p, const torch::Tensor& ref) {
  const at::cuda::CUDAGuard guard(ref.device());
  const int rc = selrot_launch(pro, out, rb, g, p, num_sms(ref), at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(rc == 0, "selrot_launch: unsupported configuration (code ", rc, ")");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// MN-major fp32 scales [KG, T] stored with a leading dim padded to a multiple of 4 (16 B), i.e. exactly DeepGEMM's
// TMA-aligned SFA layout: s.t() is [T, KG] with strides (1, align(T, 4)) and is consumed without a copy.
torch::Tensor mn_scales(int64_t KG, int64_t T, const torch::TensorOptions& o) {
  const int64_t Tp = (T + 3) / 4 * 4;
  return torch::empty({KG, Tp}, o.dtype(torch::kFloat32)).narrow(1, 0, T);
}

std::vector<torch::Tensor> selrot_quant_fp8(torch::Tensor x, torch::Tensor mask, int64_t rb, int64_t g) {
  check_x(x, "x");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous(), "x must be contiguous [T, K]");
  const int64_t T = x.size(0), K = x.size(1);
  check_cfg(K, rb, g);
  check_mask(mask, K / rb);
  auto q = torch::empty({T, K}, x.options().dtype(torch::kFloat8_e4m3fn));
  auto s = mn_scales(K / g, T, x.options());
  SelrotParams p{};
  p.x = x.data_ptr(); p.mask = mask.data_ptr<uint8_t>(); p.q = q.data_ptr(); p.s = s.data_ptr();
  p.T = T; p.K = static_cast<int>(K); p.sld = s.stride(0);
  run(PRO_NONE, OUT_FP8_MN, rb, g, p, x);
  return {q, s};
}

std::vector<torch::Tensor> rmsnorm_selrot_quant_fp8(torch::Tensor x, torch::Tensor w, double eps, torch::Tensor mask,
                                                    int64_t rb, int64_t g, bool hf_round,
                                                    c10::optional<torch::Tensor> perm) {
  check_x(x, "x"); check_x(w, "rms_weight");
  TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1 && x.stride(0) % 8 == 0, "x must be row-contiguous [T, K]");
  const int64_t T = x.size(0), K = x.size(1);
  TORCH_CHECK(w.is_contiguous() && w.numel() == K, "rms_weight must be contiguous [K]");
  TORCH_CHECK(K <= 16384, "K <= 16384 for the fused RMSNorm kernel");
  check_cfg(K, rb, g);
  check_mask(mask, K / rb);
  auto q = torch::empty({T, K}, x.options().dtype(torch::kFloat8_e4m3fn));
  auto s = mn_scales(K / g, T, x.options());
  SelrotParams p{};
  p.x = x.data_ptr(); p.x2 = w.data_ptr(); p.mask = mask.data_ptr<uint8_t>(); p.q = q.data_ptr(); p.s = s.data_ptr();
  p.T = T; p.K = static_cast<int>(K); p.sld = s.stride(0); p.ld1 = x.stride(0); p.eps = static_cast<float>(eps); p.hf_round = hf_round;
  if (perm.has_value()) {  // output chunk c (8 elements) reads input chunk perm[c]; rms_weight is given in OUTPUT order
    TORCH_CHECK(perm->is_cuda() && perm->scalar_type() == torch::kInt32 && perm->is_contiguous() &&
                    perm->numel() == K / 8, "perm must be a contiguous CUDA int32 [K/8] chunk table");
    p.perm = perm->data_ptr<int>();
  }
  run(PRO_RMS, OUT_FP8_MN, rb, g, p, x);
  return {q, s};
}

std::vector<torch::Tensor> swiglu_selrot_quant_fp8(torch::Tensor gate, torch::Tensor up, torch::Tensor mask,
                                                   int64_t rb, int64_t g, bool hf_round) {
  check_x(gate, "gate"); check_x(up, "up");
  TORCH_CHECK(gate.dim() == 2 && gate.sizes() == up.sizes() && gate.stride(1) == 1 && up.stride(1) == 1,
              "gate/up must be [T, I] with unit column stride");
  TORCH_CHECK(gate.stride(0) % 8 == 0 && up.stride(0) % 8 == 0, "row strides must be multiples of 8");
  const int64_t T = gate.size(0), I = gate.size(1);
  check_cfg(I, rb, g);
  check_mask(mask, I / rb);
  auto q = torch::empty({T, I}, gate.options().dtype(torch::kFloat8_e4m3fn));
  auto s = mn_scales(I / g, T, gate.options());
  SelrotParams p{};
  p.x = gate.data_ptr(); p.x2 = up.data_ptr(); p.mask = mask.data_ptr<uint8_t>(); p.q = q.data_ptr();
  p.s = s.data_ptr(); p.T = T; p.sld = s.stride(0); p.K = static_cast<int>(I); p.ld1 = gate.stride(0); p.ld2 = up.stride(0);
  p.hf_round = hf_round;
  run(PRO_SWIGLU, OUT_FP8_MN, rb, g, p, gate);
  return {q, s};
}

// kv [T, H, D] (rows may be strided, e.g. the k,v slice of the qkv GEMM output); mask [H, D/rb];
// fmt 0 = fp8 (scale per (token, head)), 2 = nvfp4, 3 = mxfp4, 4 = int4 (symmetric, fp16 scale per 32).
// out_q / out_s (optional): destination views [T, H, *] with contiguous rows and any row stride (e.g. cache[:, pos]);
// strided input / output needs H * D % 2048 == 0.
std::vector<torch::Tensor> kv_selrot_quant(torch::Tensor kv, torch::Tensor mask, int64_t rb, int64_t fmt,
                                           c10::optional<torch::Tensor> global_scale,
                                           c10::optional<torch::Tensor> out_q, c10::optional<torch::Tensor> out_s) {
  check_x(kv, "kv");
  TORCH_CHECK(kv.dim() == 3, "kv must be [T, H, D]");
  const int64_t T = kv.size(0), H = kv.size(1), D = kv.size(2), K = H * D;
  TORCH_CHECK(kv.stride(2) == 1 && kv.stride(1) == D && kv.stride(0) % 8 == 0, "kv rows must be contiguous [H, D]");
  TORCH_CHECK(D == 64 || D == 128, "D must be 64 or 128");
  check_cfg(K, rb, D);
  check_mask(mask, K / rb);
  TORCH_CHECK(fmt == 0 || fmt == 2 || fmt == 3 || fmt == 4, "fmt must be 0 (fp8), 2 (nvfp4), 3 (mxfp4) or 4 (int4)");
  SelrotParams p{};
  p.x = kv.data_ptr(); p.mask = mask.data_ptr<uint8_t>(); p.T = T; p.K = static_cast<int>(K);
  if (!kv.is_contiguous() && T > 1) p.ld1 = kv.stride(0);
  const int64_t Dq = fmt == 0 ? D : D / 2;
  const int64_t ns = fmt == 0 ? 1 : D / (fmt == 2 ? 16 : 32);
  const auto qdt = fmt == 0 ? torch::kFloat8_e4m3fn : torch::kUInt8;
  const auto sdt = fmt == 0 ? torch::kFloat32 : (fmt == 2 ? torch::kFloat8_e4m3fn : (fmt == 3 ? torch::kUInt8 : torch::kFloat16));
  torch::Tensor q, s;
  if (out_q.has_value()) {
    q = *out_q;
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == qdt && q.dim() == 3 && q.size(0) == T && q.size(1) == H &&
                    q.size(2) == Dq && q.stride(2) == 1 && q.stride(1) == Dq, "out_q: bad dtype / shape / layout");
    if (T > 1 && q.stride(0) != H * Dq) p.ldq = q.stride(0) * q.element_size();
  } else {
    q = torch::empty({T, H, Dq}, kv.options().dtype(qdt));
  }
  if (out_s.has_value()) {
    s = *out_s;
    TORCH_CHECK(s.is_cuda() && s.scalar_type() == sdt && s.size(0) == T && s.size(1) == H &&
                    (fmt == 0 ? (s.dim() == 2 && s.stride(1) == 1)
                              : (s.dim() == 3 && s.size(2) == ns && s.stride(2) == 1 && s.stride(1) == ns)),
                "out_s: bad dtype / shape / layout");
    const int64_t word = fmt == 0 ? 4 : (fmt == 3 ? 2 : 4);  // bytes per scale store word (see SelrotParams::lds)
    const int64_t sb = s.stride(0) * s.element_size();
    TORCH_CHECK(sb % word == 0, "out_s row stride must be a multiple of ", word, " bytes");
    if (T > 1 && s.stride(0) != H * ns) p.lds = sb / word;
  } else {
    s = fmt == 0 ? torch::empty({T, H}, kv.options().dtype(torch::kFloat32))
                 : torch::empty({T, H, ns}, kv.options().dtype(sdt));
  }
  TORCH_CHECK(!(p.ld1 || p.ldq || p.lds) || K % 2048 == 0, "strided kv / out needs H * D % 2048 == 0");
  p.q = q.data_ptr(); p.s = s.data_ptr();
  if (fmt == 2) {
    TORCH_CHECK(global_scale.has_value() && global_scale->is_cuda() &&
                    global_scale->scalar_type() == torch::kFloat32 && global_scale->numel() == 1,
                "nvfp4 needs a CUDA fp32 global_scale scalar");
    p.gscale = global_scale->data_ptr<float>();
  }
  const int out = fmt == 0 ? OUT_FP8_ROW : (fmt == 2 ? OUT_NVFP4 : (fmt == 3 ? OUT_MXFP4 : OUT_INT4));
  run(PRO_NONE, out, rb, fmt == 0 ? D : 128, p, kv);
  return {q, s};
}

torch::Tensor rmsnorm_bf16(torch::Tensor x, torch::Tensor w, double eps) {
  check_x(x, "x"); check_x(w, "w");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous() && x.size(1) % 8 == 0, "x must be contiguous [T, K], K % 8 == 0");
  auto y = torch::empty_like(x);
  const at::cuda::CUDAGuard guard(x.device());
  rmsnorm_bf16_launch(x.data_ptr(), w.data_ptr(), y.data_ptr(), x.size(0), static_cast<int>(x.size(1)),
                      static_cast<float>(eps), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor silu_mul_bf16(torch::Tensor gate, torch::Tensor up) {
  check_x(gate, "gate"); check_x(up, "up");
  TORCH_CHECK(gate.dim() == 2 && gate.stride(1) == 1 && up.stride(1) == 1 && gate.size(1) % 8 == 0 &&
                  gate.stride(0) % 8 == 0 && up.stride(0) % 8 == 0, "bad gate/up layout");
  auto y = torch::empty(gate.sizes(), gate.options());
  const at::cuda::CUDAGuard guard(gate.device());
  silu_mul_bf16_launch(gate.data_ptr(), up.data_ptr(), y.data_ptr(), gate.size(0), static_cast<int>(gate.size(1)),
                       gate.stride(0), up.stride(0), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor cast_bf16_fp8(torch::Tensor x) {
  check_x(x, "x");
  TORCH_CHECK(x.is_contiguous() && x.numel() % 16 == 0, "x must be contiguous with numel % 16 == 0");
  auto q = torch::empty(x.sizes(), x.options().dtype(torch::kFloat8_e4m3fn));
  const at::cuda::CUDAGuard guard(x.device());
  cast_bf16_fp8_launch(x.data_ptr(), q.data_ptr(), x.numel(), num_sms(x), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return q;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("selrot_quant_fp8", &selrot_quant_fp8);
  m.def("rmsnorm_selrot_quant_fp8", &rmsnorm_selrot_quant_fp8);
  m.def("swiglu_selrot_quant_fp8", &swiglu_selrot_quant_fp8);
  m.def("kv_selrot_quant", &kv_selrot_quant);
  m.def("rmsnorm_bf16", &rmsnorm_bf16);
  m.def("silu_mul_bf16", &silu_mul_bf16);
  m.def("cast_bf16_fp8", &cast_bf16_fp8);
}
