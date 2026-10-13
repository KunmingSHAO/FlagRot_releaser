#!/usr/bin/env bash
# Build libflagrot_kivi.so (kivi2: FlagRot-KIVI INT2 pages, csrc/kivi.cu), CUDA 12.9 toolkit, sm_90a, static cudart.
#   integrations/vllm_flagrot/csrc/build_kivi.sh [OUT_DIR]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/../build}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
CUDA_HOME="${FLAGROT_CUDA_HOME:-/usr/local/cuda-12.9}"
NVCC="$CUDA_HOME/bin/nvcc"
TMP="$OUT/kobj.$$"
mkdir -p "$TMP"
"$NVCC" -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a --expt-relaxed-constexpr -lineinfo -Xcompiler -fPIC \
  ${FLAGROT_EXTRA_FLAGS:-} -c "$HERE/kivi.cu" -o "$TMP/kivi.o"
"$NVCC" -shared -gencode arch=compute_90a,code=sm_90a -cudart static -o "$TMP/libflagrot_kivi.so" "$TMP/kivi.o"
mv -f "$TMP/libflagrot_kivi.so" "$OUT/libflagrot_kivi.so"
rm -rf "$TMP"
echo "$OUT/libflagrot_kivi.so"
