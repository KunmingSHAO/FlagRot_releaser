// Codebook cache writer v3 (tensor-core candidate evaluation + exact fallback): its own translation unit so that
// writer changes do not recompile the attention kernels.  Kernels and design notes: attn_cb_write3.cuh,
#include "attn_cb_common.cuh"

namespace {
#include "attn_cb_write3.cuh"
}  // namespace

int cb_write3_launch(int kind, const CbWriteParams& p, cudaStream_t st) {
  if (kind == CB_FINAL) return w3::write3_k<CB_FINAL>(p, st);
  if (kind == CB_REDUCED) return w3::write3_k<CB_REDUCED>(p, st);
  if (kind == CB_Q4) return w3::write3_k<CB_Q4>(p, st);
  if (kind == CB_KONLY) return w3::write3_k<CB_KONLY>(p, st);
  if (kind == CB_KONLYA) return w3::write3_k<CB_KONLYA>(p, st);
  if (kind == CB_KONLYA2) return w3::write3_k<CB_KONLYA2>(p, st);
  if (kind == CB_Q4K) return w3::write3_k<CB_Q4K>(p, st);
  if (kind == CB_FINALK) return w3::write3_k<CB_FINALK>(p, st);
  if (kind == CB_HEAD8) return w3::write3_k<CB_HEAD8>(p, st);
  if (kind == CB_HEAD16) return w3::write3_k<CB_HEAD16>(p, st);
  return -4;
}
int w3_static_write_launch(const W3StaticParams& p, cudaStream_t st) {
  if (p.asym == 2) return w3::w3s_k<2>(p, st);  // INT2 (int2a:32 / int2ah:32)
  return p.asym ? w3::w3s_k<1>(p, st) : w3::w3s_k<0>(p, st);
}
