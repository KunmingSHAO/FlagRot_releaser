#!/usr/bin/env bash
# Build libflagrot_prefill.so (Phase 2f INT4 prefill kernel, csrc/int4_prefill.cu): CUDA 12.9 toolkit, sm_90a, static
# CUDA runtime (as build.sh).   integrations/vllm_flagrot/csrc/build_prefill.sh [OUT_DIR]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/../build}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
CUDA_HOME="${FLAGROT_CUDA_HOME:-/usr/local/cuda-12.9}"
TMP="$OUT/pobj.$$"
mkdir -p "$TMP"
"$CUDA_HOME/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a --expt-relaxed-constexpr -lineinfo \
  -Xcompiler -fPIC -Xptxas -v ${FLAGROT_EXTRA_FLAGS:-} -shared -cudart static -o "$TMP/libflagrot_prefill.so" \
  "$HERE/int4_prefill.cu" 2> "$TMP/ptxas.log" || { cat "$TMP/ptxas.log"; exit 1; }
grep -E "registers|spill|C75" "$TMP/ptxas.log" >&2 || true
mv -f "$TMP/libflagrot_prefill.so" "$OUT/libflagrot_prefill.so"
rm -rf "$TMP"
echo "$OUT/libflagrot_prefill.so"
